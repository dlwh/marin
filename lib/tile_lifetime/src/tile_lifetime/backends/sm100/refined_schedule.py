# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-convex SM100 schedule attachment for source-granular task plans."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from tile_lifetime.event_dataflow import TaskAxis, TaskDependence, TaskFamily, TaskRelation
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    ExecutablePhysicalRequest,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalOrderKind,
    PhysicalPhase,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskOrder,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskOperationKind,
    TaskResourceEstimate,
    TiledTask,
    install_physical_schedule_cover,
    physical_schedule_is_eligible,
)
from tile_lifetime.task_plan_refinement import CanonicalTaskPlanRefinement, TaskRefinementBinding
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    FoldPrimitive,
    MapPrimitive,
    RelationPrimitive,
    SelectionPrimitive,
    deserialize_tensor_program,
)

REFINED_SELECTION_SCHEDULE_FAMILY = "sm100_dependency_convex_contract_fold_selection_tcgen05"
REFINED_GATHERED_NORMALIZED_EXP_SCHEDULE_FAMILY = "sm100_dependency_convex_gathered_normalized_exp_contract"
_SUPPORTED_K = (4, 8, 16, 32)
_WMMA_EXTENT = 16
_TCGEN05_SELECTION_ROWS = 64
_GATHERED_NORMALIZED_EXP_TENSOR_WARPS = 8
_GATHERED_NORMALIZED_EXP_THREADS = 256
_GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE = _GATHERED_NORMALIZED_EXP_TENSOR_WARPS * _WMMA_EXTENT
_MAX_GATHERED_NORMALIZED_EXP_SHARED_BYTES = 160 * 1024
REFINED_SELECTION_NUMERICAL_REJECTION = (
    "refined SM100 ordered Selection requires a certified tcgen05-to-canonical Contract error interval, "
    "including scalar-Map interval propagation; FAST FP32 accumulation does not provide that certificate"
)


@dataclass(frozen=True)
class Sm100DependencyConvexSelectionRegion:
    """One source-granular region proven to quotient to one physical component."""

    source_request_fingerprint: str
    task_ids: tuple[int, ...]
    contract_task_id: int
    fold_task_id: int
    selection_task_id: int
    refinement: CanonicalTaskPlanRefinement
    physical_schedule: PhysicalSchedule


@dataclass(frozen=True)
class Sm100RefinedSelectionAttachment:
    """Replacement request plus its exact source-to-coarse refinement proof."""

    source_request_fingerprint: str
    source_request: ExecutablePhysicalRequest
    scheduled_request: ExecutablePhysicalRequest
    region: Sm100DependencyConvexSelectionRegion


@dataclass(frozen=True)
class Sm100DependencyConvexGatheredNormalizedExpRegion:
    """A gathered normalized-exponential interval with an exact quotient proof."""

    source_request_fingerprint: str
    task_ids: tuple[int, ...]
    gather_task_ids: tuple[int, ...]
    producer_contract_task_id: int
    terminal_contract_task_id: int
    fold_task_ids: tuple[int, ...]
    refinement: CanonicalTaskPlanRefinement
    physical_schedule: PhysicalSchedule


@dataclass(frozen=True)
class Sm100RefinedGatheredNormalizedExpAttachment:
    """Replacement request plus the gathered-region refinement proof."""

    source_request_fingerprint: str
    source_request: ExecutablePhysicalRequest
    scheduled_request: ExecutablePhysicalRequest
    region: Sm100DependencyConvexGatheredNormalizedExpRegion


def select_sm100_dependency_convex_selection_regions(
    request: ExecutablePhysicalRequest,
) -> tuple[Sm100DependencyConvexSelectionRegion, ...]:
    """Find Contract/Fold/Selection intervals from task capabilities and dependencies."""
    if request.target is not PhysicalTarget.SM100 or request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        return ()
    if request.task_plan.bounded_buffers:
        return ()
    successors, predecessors = _task_graph(request.task_plan)
    selections = tuple(
        task.id for task in request.task_plan.tasks if task.structure.principal_kind is TaskOperationKind.SELECTION
    )
    regions = []
    for selection_id in selections:
        ancestors = _reachable((selection_id,), predecessors)
        for contract in request.task_plan.tasks:
            if contract.id not in ancestors or contract.structure.principal_kind is not TaskOperationKind.CONTRACT:
                continue
            task_ids = _extend_with_closed_side_producers(
                request,
                tuple(sorted(_reachable((contract.id,), successors) & ancestors)),
                successors,
            )
            tasks = tuple(request.task_plan.tasks[task_id] for task_id in task_ids)
            kinds = tuple(task.structure.principal_kind for task in tasks)
            folds = tuple(task.id for task in tasks if task.structure.principal_kind is TaskOperationKind.FOLD)
            if kinds.count(TaskOperationKind.CONTRACT) != 1 or len(folds) != 1:
                continue
            if kinds.count(TaskOperationKind.SELECTION) != 1 or any(
                kind
                not in {
                    TaskOperationKind.CONTRACT,
                    TaskOperationKind.FOLD,
                    TaskOperationKind.MAP,
                    TaskOperationKind.RELATION,
                    TaskOperationKind.SELECTION,
                }
                for kind in kinds
            ):
                continue
            if not _selection_physical_boundary_is_supported(request, contract.id, selection_id):
                continue
            refinement = _coarsen_region(request, task_ids)
            schedule = _selection_schedule(request, task_ids, refinement)
            regions.append(
                Sm100DependencyConvexSelectionRegion(
                    request.fingerprint,
                    task_ids,
                    contract.id,
                    folds[0],
                    selection_id,
                    refinement,
                    schedule,
                )
            )
    return tuple(regions)


def attach_sm100_dependency_convex_selection_schedule(
    request: ExecutablePhysicalRequest,
    region: Sm100DependencyConvexSelectionRegion,
) -> Sm100RefinedSelectionAttachment:
    """Replace the source-granular interval by its one-component SM100 schedule."""
    if region.source_request_fingerprint != request.fingerprint:
        raise ValueError("SM100 refined Selection region does not derive from the supplied request")
    selected = set(region.task_ids)
    retained = []
    removed: set[int] = set()
    for schedule in request.schedules:
        overlap = selected & set(schedule.task_ids)
        if not overlap:
            retained.append(schedule)
            continue
        if overlap != set(schedule.task_ids):
            raise ValueError("SM100 refined Selection schedule partially overlaps an existing schedule")
        removed.update(overlap)
    if removed != selected:
        raise ValueError("SM100 refined Selection schedule does not replace an exact existing task cover")
    schedules = tuple(sorted((*retained, region.physical_schedule), key=lambda schedule: min(schedule.task_ids)))
    scheduled = install_physical_schedule_cover(
        request,
        schedules,
        source_request_fingerprint=request.fingerprint,
    )
    scheduled = replace(
        scheduled,
        provenance={
            **scheduled.provenance,
            "sm100_dependency_convex_refinement": {
                "source_request_fingerprint": request.fingerprint,
                "quotient_fingerprint": region.refinement.quotient_fingerprint,
                "proof": region.refinement.proof_record,
                "schedule_fingerprint": region.physical_schedule.fingerprint,
            },
        },
    )
    return Sm100RefinedSelectionAttachment(request.fingerprint, request, scheduled, region)


def select_sm100_dependency_convex_gathered_normalized_exp_regions(
    request: ExecutablePhysicalRequest,
) -> tuple[Sm100DependencyConvexGatheredNormalizedExpRegion, ...]:
    """Find maximal Selection-descendant gathered normalized-exp intervals."""
    if request.target is not PhysicalTarget.SM100 or request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        return ()
    if request.task_plan.bounded_buffers:
        return ()
    successors, predecessors = _task_graph(request.task_plan)
    selections = tuple(
        task.id for task in request.task_plan.tasks if task.structure.principal_kind is TaskOperationKind.SELECTION
    )
    sinks = tuple(task.id for task in request.task_plan.tasks if not successors[task.id])
    candidates: list[Sm100DependencyConvexGatheredNormalizedExpRegion] = []
    for selection_id in selections:
        descendants = _reachable((selection_id,), successors) - {selection_id}
        for sink_id in sinks:
            if sink_id not in descendants:
                continue
            original_task_ids = tuple(sorted(descendants & _reachable((sink_id,), predecessors)))
            task_ids = _extend_with_closed_side_producers(
                request,
                original_task_ids,
                successors,
            )
            tasks = tuple(request.task_plan.tasks[task_id] for task_id in task_ids)
            gather_ids = tuple(
                task.id
                for task in tasks
                if task.structure.principal_kind is TaskOperationKind.RELATION
                and any(access.relation_class is RelationAccessClass.BOUNDED_DYNAMIC for access in task.accesses)
            )
            contracts = tuple(task.id for task in tasks if task.structure.principal_kind is TaskOperationKind.CONTRACT)
            folds = tuple(task.id for task in tasks if task.structure.principal_kind is TaskOperationKind.FOLD)
            if len(gather_ids) < 2 or len(contracts) != 2 or len(folds) != 2:
                continue
            if any(
                task.structure.principal_kind
                not in {
                    TaskOperationKind.CONTRACT,
                    TaskOperationKind.FOLD,
                    TaskOperationKind.MAP,
                    TaskOperationKind.RELATION,
                }
                for task in tasks
            ):
                continue
            contract_order = tuple(task_id for task_id in task_ids if task_id in contracts)
            if not _gathered_normalized_exp_boundary_is_supported(request, task_ids, contract_order, folds):
                continue
            refinement = _coarsen_region(
                request,
                task_ids,
                family_name="sm100.dependency_convex.gathered_normalized_exp_contract",
            )
            schedule = _gathered_normalized_exp_schedule(request, task_ids, refinement)
            candidates.append(
                Sm100DependencyConvexGatheredNormalizedExpRegion(
                    request.fingerprint,
                    task_ids,
                    gather_ids,
                    contract_order[0],
                    contract_order[1],
                    folds,
                    refinement,
                    schedule,
                )
            )
    maximal = []
    for candidate in sorted(candidates, key=lambda item: (-len(item.task_ids), item.task_ids)):
        if any(set(candidate.task_ids) < set(other.task_ids) for other in candidates):
            continue
        maximal.append(candidate)
    return tuple(maximal)


def _extend_with_closed_side_producers(
    request: ExecutablePhysicalRequest,
    task_ids: tuple[int, ...],
    successors: Mapping[int, set[int]],
) -> tuple[int, ...]:
    """Include pure side-input expressions whose every consumer is in the region."""
    selected = set(task_ids)
    original_frontier = min(task_ids)
    event_sources = {
        dependence.source_task_id
        for dependence in request.task_plan.dependence_programs
        if dependence.target_task_id in selected
    }
    event_frontier = max(event_sources, default=original_frontier - 1)
    program = deserialize_tensor_program(json.dumps(request.algebra_record, sort_keys=True, separators=(",", ":")))
    changed = True
    while changed:
        changed = False
        for task in reversed(request.task_plan.tasks):
            if (
                task.id < original_frontier
                or task.id <= event_frontier
                or task.id in selected
                or task.id in event_sources
                or task.structure.principal_kind
                not in {
                    TaskOperationKind.MAP,
                    TaskOperationKind.RELATION,
                }
            ):
                continue
            if any(
                not isinstance(program.operations[ordinal], (MapPrimitive, RelationPrimitive))
                for ordinal in task.operation_ordinals
            ):
                continue
            consumers = successors[task.id]
            if consumers and consumers <= selected:
                selected.add(task.id)
                changed = True
    return tuple(sorted(selected))


def attach_sm100_dependency_convex_gathered_normalized_exp_schedule(
    request: ExecutablePhysicalRequest,
    region: Sm100DependencyConvexGatheredNormalizedExpRegion,
) -> Sm100RefinedGatheredNormalizedExpAttachment:
    """Replace one exact gathered normalized-exp interval by one schedule."""
    if region.source_request_fingerprint != request.fingerprint:
        raise ValueError("SM100 gathered normalized-exp region does not derive from the supplied request")
    selected = set(region.task_ids)
    retained = []
    removed: set[int] = set()
    for schedule in request.schedules:
        overlap = selected & set(schedule.task_ids)
        if not overlap:
            retained.append(schedule)
            continue
        if overlap != set(schedule.task_ids):
            raise ValueError("SM100 gathered normalized-exp schedule partially overlaps an existing schedule")
        removed.update(overlap)
    if removed != selected:
        raise ValueError("SM100 gathered normalized-exp schedule does not replace an exact existing task cover")
    schedules = tuple(sorted((*retained, region.physical_schedule), key=lambda schedule: min(schedule.task_ids)))
    scheduled = install_physical_schedule_cover(
        request,
        schedules,
        source_request_fingerprint=request.fingerprint,
    )
    scheduled = replace(
        scheduled,
        provenance={
            **scheduled.provenance,
            "sm100_dependency_convex_gathered_normalized_exp_refinement": {
                "source_request_fingerprint": request.fingerprint,
                "quotient_fingerprint": region.refinement.quotient_fingerprint,
                "proof": region.refinement.proof_record,
                "schedule_fingerprint": region.physical_schedule.fingerprint,
            },
        },
    )
    return Sm100RefinedGatheredNormalizedExpAttachment(request.fingerprint, request, scheduled, region)


def _selection_physical_boundary_is_supported(
    request: ExecutablePhysicalRequest,
    contract_task_id: int,
    selection_task_id: int,
) -> bool:
    program = deserialize_tensor_program(json.dumps(request.algebra_record, sort_keys=True, separators=(",", ":")))
    contract_task = request.task_plan.tasks[contract_task_id]
    selection_task = request.task_plan.tasks[selection_task_id]
    contract_ordinals = tuple(
        ordinal
        for ordinal in contract_task.operation_ordinals
        if isinstance(program.operations[ordinal], ContractPrimitive)
    )
    selection_ordinals = tuple(
        ordinal
        for ordinal in selection_task.operation_ordinals
        if isinstance(program.operations[ordinal], SelectionPrimitive)
    )
    if len(contract_ordinals) != 1 or len(selection_ordinals) != 1:
        return False
    contract = program.operations[contract_ordinals[0]]
    selection = program.operations[selection_ordinals[0]]
    assert isinstance(contract, ContractPrimitive) and isinstance(selection, SelectionPrimitive)
    structurally_supported = (
        len(contract.reduction_axes) == 1
        and contract.reduction_axes[0].extent == 128
        and selection.count in _SUPPORTED_K
        and selection.dimension.extent >= selection.count
    )
    if not structurally_supported:
        return False

    # The selected values have a source-index-stable total order, so an
    # approximate Contract may prune a candidate only with a proof that its
    # canonical score cannot cross the kth boundary.  The request currently
    # declares FAST/order-free FP32 accumulation, but carries neither a
    # tcgen05 rounding model nor a scalar-Map interval certificate.  A fixed
    # over-selection count is not a proof, particularly when the cutoff is a
    # tie.  Keep the dependency-convex region on the exact SIMT fallback until
    # that certificate is represented at the physical boundary.
    return False


def sm100_dependency_convex_selection_numerical_rejection() -> str:
    """Explain why the current tcgen05 ordered-Selection schedule is ineligible."""
    return REFINED_SELECTION_NUMERICAL_REJECTION


def _gathered_normalized_exp_boundary_is_supported(
    request: ExecutablePhysicalRequest,
    task_ids: tuple[int, ...],
    contract_task_ids: tuple[int, ...],
    fold_task_ids: tuple[int, ...],
) -> bool:
    program = deserialize_tensor_program(json.dumps(request.algebra_record, sort_keys=True, separators=(",", ":")))
    operations = tuple(
        program.operations[ordinal]
        for task_id in task_ids
        for ordinal in request.task_plan.tasks[task_id].operation_ordinals
    )
    contracts = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))
    folds = tuple(operation for operation in operations if isinstance(operation, FoldPrimitive))
    if len(contracts) != 2 or len(folds) != 2:
        return False
    if (
        tuple(
            task_id
            for task_id in contract_task_ids
            for ordinal in request.task_plan.tasks[task_id].operation_ordinals
            if isinstance(program.operations[ordinal], ContractPrimitive)
        )
        != contract_task_ids
    ):
        return False
    if (
        tuple(
            task_id
            for task_id in fold_task_ids
            for ordinal in request.task_plan.tasks[task_id].operation_ordinals
            if isinstance(program.operations[ordinal], FoldPrimitive)
        )
        != fold_task_ids
    ):
        return False
    producer, terminal = contracts
    maximum = tuple(fold for fold in folds if fold.reducer.value == "maximum")
    summation = tuple(fold for fold in folds if fold.reducer.value == "sum")
    value_operands = tuple(
        operand
        for operand_index, operand in enumerate(terminal.inputs)
        if terminal.output.axes[-1] in {mapping.domain_axis for mapping in terminal.index_maps_for_input(operand_index)}
    )
    if len(value_operands) != 1:
        return False
    value_operand_index = terminal.inputs.index(value_operands[0])
    value_domain_axes = {
        *value_operands[0].axes,
        *(m.domain_axis for m in terminal.index_maps_for_input(value_operand_index)),
    }
    owner_group_axes = tuple(axis for axis in terminal.output.axes[:-1] if axis not in value_domain_axes)
    if len(owner_group_axes) != 1:
        return False
    owner_rows = owner_group_axes[0].extent
    normalized_extent = terminal.reduction_axes[0].extent
    shared_bytes = (
        owner_rows * _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE * 4
        + _TCGEN05_SELECTION_ROWS * 128 * 2
        + 128 * 128 * 2
        + 128
        + 16
    )
    return (
        len(maximum) == 1
        and len(summation) == 1
        and len(producer.reduction_axes) == 1
        and producer.reduction_axes[0].extent == 128
        and len(terminal.reduction_axes) == 1
        and terminal.reduction_axes[0].extent == maximum[0].reduction_axes[0].extent
        and terminal.reduction_axes[0].extent == summation[0].reduction_axes[0].extent
        and terminal.output.shape[-1] == 128
        and 0 < owner_rows <= _WMMA_EXTENT
        and normalized_extent % _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE == 0
        and shared_bytes <= _MAX_GATHERED_NORMALIZED_EXP_SHARED_BYTES
        and all(value.dtype is DType.BF16 for contract in contracts for value in contract.inputs)
        and producer.accumulation_dtype.value == "fp32"
        and terminal.accumulation_dtype.value == "fp32"
    )


def _gathered_normalized_exp_schedule(
    request: ExecutablePhysicalRequest,
    task_ids: tuple[int, ...],
    refinement: CanonicalTaskPlanRefinement,
) -> PhysicalSchedule:
    tasks = tuple(request.task_plan.tasks[task_id] for task_id in task_ids)
    program = deserialize_tensor_program(json.dumps(request.algebra_record, sort_keys=True, separators=(",", ":")))
    operations = tuple(
        program.operations[ordinal]
        for task_id in task_ids
        for ordinal in request.task_plan.tasks[task_id].operation_ordinals
    )
    terminal = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))[1]
    value_operand_index = next(
        operand_index
        for operand_index in range(len(terminal.inputs))
        if terminal.output.axes[-1] in {mapping.domain_axis for mapping in terminal.index_maps_for_input(operand_index)}
    )
    value_operand = terminal.inputs[value_operand_index]
    value_domain_axes = {
        *value_operand.axes,
        *(m.domain_axis for m in terminal.index_maps_for_input(value_operand_index)),
    }
    owner_group_axis = next(axis for axis in terminal.output.axes[:-1] if axis not in value_domain_axes)
    normalized_extent = terminal.reduction_axes[0].extent
    owner_rows = owner_group_axis.extent
    shared_memory_bytes = (
        owner_rows * _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE * 4
        + _TCGEN05_SELECTION_ROWS * 128 * 2
        + 128 * 128 * 2
        + 128
        + 16
    )
    family = PhysicalScheduleFamily(
        REFINED_GATHERED_NORMALIZED_EXP_SCHEDULE_FAMILY,
        PhysicalTarget.SM100,
        tuple(_exact_structural_constraint(task) for task in tasks),
    )
    selected = set(task_ids)
    family_to_id = {task.family: task.id for task in request.task_plan.tasks}
    orders = tuple(
        _program_order(family_to_id[dependence.relation.source], family_to_id[dependence.relation.target], dependence)
        for dependence in request.task_plan.dependences
        if family_to_id[dependence.relation.source] in selected and family_to_id[dependence.relation.target] in selected
    )
    schedule = PhysicalSchedule(
        family,
        task_ids,
        {
            "component_realization": "dependency_convex_refinement",
            "matrix_instruction": "tcgen05_m64n128k16_bf16_fp32_pair",
            "threads": _GATHERED_NORMALIZED_EXP_THREADS,
            "owner_rows": owner_rows,
            "score_feature_tile": 128,
            "normalized_tile": normalized_extent,
            "value_feature_tile": 128,
            "normalized_wave": _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE,
            "normalized_passes": 3,
            "tensor_warps": _GATHERED_NORMALIZED_EXP_TENSOR_WARPS,
            "resident_warps_per_cta": _GATHERED_NORMALIZED_EXP_THREADS // 32,
            "active_tensor_warps": 1,
            "cta_barriers_per_component": 5 + 11 * normalized_extent // _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE,
            "tensor_core_arrivals_per_component": 4 * normalized_extent // _GATHERED_NORMALIZED_EXP_NORMALIZED_WAVE,
            "shared_memory_bytes": shared_memory_bytes,
            "resident_ctas_by_shared_memory": _MAX_GATHERED_NORMALIZED_EXP_SHARED_BYTES // shared_memory_bytes,
            "pipeline_stages": 1,
            "tensor_memory_columns": 256,
            "tensor_core_warps": 1,
            "refinement_quotient_fingerprint": refinement.quotient_fingerprint,
        },
        orders,
    )
    if not physical_schedule_is_eligible(schedule, request.task_plan):
        raise ValueError("SM100 gathered normalized-exp schedule violates its structural family")
    return schedule


def _selection_schedule(
    request: ExecutablePhysicalRequest,
    task_ids: tuple[int, ...],
    refinement: CanonicalTaskPlanRefinement,
) -> PhysicalSchedule:
    tasks = tuple(request.task_plan.tasks[task_id] for task_id in task_ids)
    family = PhysicalScheduleFamily(
        REFINED_SELECTION_SCHEDULE_FAMILY,
        PhysicalTarget.SM100,
        tuple(_exact_structural_constraint(task) for task in tasks),
    )
    selected = set(task_ids)
    family_to_id = {task.family: task.id for task in request.task_plan.tasks}
    orders = tuple(
        _program_order(family_to_id[dependence.relation.source], family_to_id[dependence.relation.target], dependence)
        for dependence in request.task_plan.dependences
        if family_to_id[dependence.relation.source] in selected and family_to_id[dependence.relation.target] in selected
    )
    schedule = PhysicalSchedule(
        family,
        task_ids,
        {
            "matrix_instruction": "tcgen05_m64n128k16_bf16_fp32",
            "component_realization": "dependency_convex_refinement",
            "threads": 256,
            "score_tile_rows": _TCGEN05_SELECTION_ROWS,
            "score_tile_columns": 128,
            "contraction_tile": 128,
            "pipeline_stages": 1,
            "tensor_memory_columns": 128,
            "tensor_core_warps": 1,
            "selection_warps": 2,
            "selection_threads_per_row": 1,
            "refinement_quotient_fingerprint": refinement.quotient_fingerprint,
        },
        orders,
    )
    if not physical_schedule_is_eligible(schedule, request.task_plan):
        raise ValueError("SM100 dependency-convex Selection schedule violates its structural family")
    return schedule


def _program_order(source_id: int, target_id: int, dependence: TaskDependence) -> PhysicalTaskOrder:
    return PhysicalTaskOrder(
        PhysicalOrderKind.PROGRAM_ORDER,
        source_id,
        target_id,
        dependence.relation,
        {"kind": "dependency_convex_refined_phase_order"},
    )


def _exact_structural_constraint(task: TiledTask) -> PhysicalTaskConstraint:
    capabilities = tuple(dict.fromkeys(phase.capability for phase in task.structure.phases))
    relations = tuple(
        dict.fromkeys(
            (
                *(edge.relation_class for edge in task.structure.phase_edges),
                *(access.relation_class for access in task.accesses),
                RelationAccessClass.IDENTITY,
            )
        )
    )
    required_edges = tuple(
        PhaseEdgeConstraint(
            task.structure.phases[edge.source_phase].capability,
            task.structure.phases[edge.target_phase].capability,
            edge.relation_class,
        )
        for edge in task.structure.phase_edges
    )
    return PhysicalTaskConstraint(
        task.structure.principal_kind,
        capabilities,
        capabilities,
        required_edges,
        len(task.structure.owner_axes),
        len(task.structure.streamed_axes),
        len(task.structure.reduction_axes),
        len(task.structure.contraction_axes),
        relations,
        task.structure.element_types,
        task.structure.layout_classes,
        task.structure.resources,
    )


def _coarsen_region(
    request: ExecutablePhysicalRequest,
    region_task_ids: tuple[int, ...],
    *,
    family_name: str = "sm100.dependency_convex.contract_fold_selection",
) -> CanonicalTaskPlanRefinement:
    refined = request.task_plan
    region_ids = set(region_task_ids)
    if any(len(task.family.coordinates) != 1 for task in refined.tasks if task.id in region_ids):
        raise ValueError("SM100 dependency-convex refinement currently requires one coordinate per source task")
    incoming_programs = tuple(
        dependence
        for dependence in refined.dependence_programs
        if dependence.source_task_id not in region_ids and dependence.target_task_id in region_ids
    )
    incoming_ranks = {len(dependence.target_coordinates) for dependence in incoming_programs}
    if len(incoming_ranks) > 1:
        raise ValueError("SM100 dependency-convex refinement has incompatible runtime Event target ranks")
    coarse_family_axes = refined.tasks[incoming_programs[0].target_task_id].family.axes if incoming_programs else ()
    groups: list[tuple[int, ...]] = []
    for task in refined.tasks:
        if task.id in region_ids:
            if task.id == region_task_ids[0]:
                groups.append(region_task_ids)
            continue
        groups.append((task.id,))
    coarse_tasks = []
    refined_to_coarse = {}
    for coarse_id, group in enumerate(groups):
        for refined_id in group:
            refined_to_coarse[refined_id] = coarse_id
        if len(group) == 1:
            coarse_tasks.append(replace(refined.tasks[group[0]], id=coarse_id))
        else:
            coarse_tasks.append(
                _coarse_region_task(
                    refined,
                    coarse_id,
                    group,
                    family_name=family_name,
                    family_axes=coarse_family_axes,
                )
            )
    bindings = tuple(
        _refinement_binding(refined, tuple(coarse_tasks), refined_id, refined_to_coarse[refined_id])
        for refined_id in range(len(refined.tasks))
    )
    coordinate_maps = {
        binding.refined_task_id: {pair.source: pair.target for pair in binding.coordinate_relation.pairs}
        for binding in bindings
    }
    family_to_id = {task.family: task.id for task in refined.tasks}
    projected: dict[tuple[int, int, object], set[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(set)
    for dependence in refined.dependences:
        source_refined = family_to_id[dependence.relation.source]
        target_refined = family_to_id[dependence.relation.target]
        source = refined_to_coarse[source_refined]
        target = refined_to_coarse[target_refined]
        if source == target:
            continue
        projected[(source, target, dependence.visibility)].update(
            (coordinate_maps[source_refined][pair.source], coordinate_maps[target_refined][pair.target])
            for pair in dependence.relation.pairs
        )
    coarse_dependences = tuple(
        TaskDependence(
            TaskRelation.from_pairs(
                coarse_tasks[source].family,
                coarse_tasks[target].family,
                tuple(sorted(pairs)),
            ),
            visibility,
        )
        for (source, target, visibility), pairs in projected.items()
    )
    coarse_programs = tuple(
        replace(
            dependence,
            source_task_id=refined_to_coarse[dependence.source_task_id],
            target_task_id=refined_to_coarse[dependence.target_task_id],
        )
        for dependence in refined.dependence_programs
        if refined_to_coarse[dependence.source_task_id] != refined_to_coarse[dependence.target_task_id]
    )
    coarse = CanonicalTaskPlan(tuple(coarse_tasks), coarse_dependences, dependence_programs=coarse_programs)
    retained_program_indices = tuple(
        index
        for index, dependence in enumerate(refined.dependence_programs)
        if refined_to_coarse[dependence.source_task_id] != refined_to_coarse[dependence.target_task_id]
    )
    coarse_event_programs = tuple(
        replace(request.event_programs[index], dependence_program_index=coarse_index)
        for coarse_index, index in enumerate(retained_program_indices)
    )
    return CanonicalTaskPlanRefinement(
        request.algebra_fingerprint,
        request.target,
        request.numerical_policy,
        coarse,
        refined,
        bindings,
        coarse_event_programs=coarse_event_programs,
        refined_event_programs=request.event_programs,
    )


def _coarse_region_task(
    plan: CanonicalTaskPlan,
    coarse_id: int,
    group: tuple[int, ...],
    *,
    family_name: str,
    family_axes: tuple[TaskAxis, ...],
) -> TiledTask:
    members = tuple(plan.tasks[task_id] for task_id in group)
    operations = tuple(
        (ordinal, kind, phase.capability, phase.domain_shape, phase.reduction_dimensions)
        for task in members
        for ordinal, kind in zip(task.operation_ordinals, task.operation_kinds, strict=True)
        for phase in task.structure.phases
        if ordinal in phase.operation_ordinals
    )
    phases = tuple(
        PhysicalPhase(
            index,
            capability,
            (ordinal,),
            domain_shape,
            reduction_dimensions,
        )
        for index, (ordinal, _, capability, domain_shape, reduction_dimensions) in enumerate(operations)
    )
    anchor = members[0].structure
    structure = PhysicalTaskStructure(
        TaskOperationKind.CONTRACT,
        phases,
        (),
        anchor.owner_axes,
        anchor.streamed_axes,
        anchor.reduction_axes,
        anchor.contraction_axes,
        anchor.traversal,
        tuple(dict.fromkeys(value for task in members for value in task.structure.element_types)),
        tuple(dict.fromkeys(value for task in members for value in task.structure.layout_classes)),
        TaskResourceEstimate(
            max(task.structure.resources.registers_per_thread for task in members),
            max(task.structure.resources.shared_memory_bytes for task in members),
            max(task.structure.resources.barrier_count for task in members),
        ),
    )
    return TiledTask(
        coarse_id,
        TaskFamily(family_name, family_axes),
        tuple(ordinal for ordinal, _, _, _, _ in operations),
        tuple(kind for _, kind, _, _, _ in operations),
        structure,
        (),
    )


def _refinement_binding(
    refined: CanonicalTaskPlan,
    coarse_tasks: tuple[TiledTask, ...],
    refined_id: int,
    coarse_id: int,
) -> TaskRefinementBinding:
    refined_family = refined.tasks[refined_id].family
    coarse_family = coarse_tasks[coarse_id].family
    if refined_family == coarse_family:
        pairs = tuple(zip(refined_family.coordinates, coarse_family.coordinates, strict=True))
    else:
        if len(coarse_family.coordinates) != 1:
            raise ValueError("SM100 dependency-convex quotient currently requires one coarse coordinate")
        pairs = tuple((coordinate, coarse_family.coordinates[0]) for coordinate in refined_family.coordinates)
    return TaskRefinementBinding(
        refined_id,
        coarse_id,
        TaskRelation.from_pairs(refined_family, coarse_family, pairs),
    )


def _task_graph(plan: CanonicalTaskPlan) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    successors = {task.id: set() for task in plan.tasks}
    predecessors = {task.id: set() for task in plan.tasks}
    family_to_id = {task.family: task.id for task in plan.tasks}
    for dependence in plan.dependences:
        source = family_to_id[dependence.relation.source]
        target = family_to_id[dependence.relation.target]
        successors[source].add(target)
        predecessors[target].add(source)
    for dependence in plan.dependence_programs:
        successors[dependence.source_task_id].add(dependence.target_task_id)
        predecessors[dependence.target_task_id].add(dependence.source_task_id)
    return successors, predecessors


def _reachable(starts: tuple[int, ...], edges: dict[int, set[int]]) -> set[int]:
    reached = set(starts)
    pending = list(starts)
    while pending:
        source = pending.pop()
        for target in edges[source]:
            if target in reached:
                continue
            reached.add(target)
            pending.append(target)
    return reached
