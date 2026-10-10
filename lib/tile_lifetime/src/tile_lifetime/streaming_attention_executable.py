# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable physical-request boundary for normalized-exponential contractions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum

from tile_lifetime.dense_flow_planner import derive_composed_task_plan, eventize_composed_task_plan
from tile_lifetime.event_buffering import (
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    EventMemoryScope,
    EventStorageBinding,
    TaskFamily,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90BackwardAccumulatorSchedule,
    Sm90ForwardPipelineSchedule,
    select_sm90_contract_schedule,
)
from tile_lifetime.index_domain_canonicalization import (
    NormalizedExpDomainCanonicalizationError,
    canonicalize_normalized_exp_index_domain,
)
from tile_lifetime.layout_relation_canonicalization import canonicalize_contract_layout_relations
from tile_lifetime.normalized_exp_contract_region import (
    NormalizedExpContractProgram,
    NormalizedExpContractPullback,
    NormalizedExpContractPullbackCapability,
    NormalizedExpContractPullbackProgram,
    NormalizedExpContractRegionCapability,
    derive_reverse_ownership_plan,
    normalized_exp_contract_pullback_tensor_program,
    normalized_exp_contract_region,
    normalized_exp_contract_region_capability,
    normalized_exp_pullback_capability,
    normalized_exp_region_capability,
)
from tile_lifetime.physical_task_contract import (
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalOrderKind,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskOrder,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskResourceEstimate,
    TaskValueAccessKind,
    TiledTask,
    canonical_fingerprint,
    install_physical_schedule_cover,
)
from tile_lifetime.streaming_pullback_event_schedule import (
    NormalizedExpPullbackTaskDataflow,
    derive_normalized_exp_pullback_task_dataflow,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    RelationPrimitive,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    deserialize_tensor_program,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)
from tile_lifetime.tensor_program_rewrites import (
    compose_scalar_map_chains,
    fold_widening_converts_into_contracts,
    sink_fold_invariant_division_through_contract,
)

_SM90_MAXIMUM_RESOURCES = TaskResourceEstimate(255, 228 * 1024, 32)
_SM90_SUPPORTED_RELATIONS = (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE)
_SM90_SUPPORTED_ELEMENT_TYPES = ("bf16", "bool", "fp32", "int32")
_SM90_SUPPORTED_LAYOUTS = ("logical_row_major",)
_SM90_CONTRACT_FOLD_CAPABILITIES = (
    TaskCapability.MATRIX_CONTRACT,
    TaskCapability.ASSOCIATIVE_FOLD,
    TaskCapability.SCALAR_MAP,
    TaskCapability.INDEX_RELATION,
)


class ReverseAlgebraAcceptance(StrEnum):
    """Whether JAX-owned AD has established a canonical reverse import."""

    CANDIDATE_AWAITING_JAXPR_IMPORT = "candidate_awaiting_jaxpr_import"


class Sm90ContractFoldDirection(StrEnum):
    """Schedule direction derived from terminal computation capabilities."""

    FORWARD = "forward"
    PULLBACK = "pullback"


@dataclass(frozen=True)
class NormalizedExpExecutableRequests:
    """Forward request and an explicitly unaccepted reverse candidate."""

    forward: ExecutablePhysicalRequest
    reverse_candidate: ExecutablePhysicalRequest
    reverse_task_dataflow: NormalizedExpPullbackTaskDataflow
    reverse_acceptance: ReverseAlgebraAcceptance


@dataclass(frozen=True)
class Sm90ContractFoldScheduleAttachment:
    """One fused SM90 schedule installed on a common executable request."""

    request: ExecutablePhysicalRequest
    fused_task_ids: tuple[int, ...]
    forward_core_task_ids: tuple[int, ...]

    @property
    def physical_schedule(self) -> PhysicalSchedule:
        return next(
            (
                schedule
                for schedule in self.request.schedules
                if schedule.parameters.get("direction") == Sm90ContractFoldDirection.PULLBACK.value
            ),
            self.request.schedules[0],
        )


@dataclass(frozen=True)
class _Sm90ContractFoldCore:
    producer_task: int
    maximum_fold_task: int
    normalizer_fold_task: int
    consumer_task: int
    normalized_axis_id: int
    producer_feature_extent: int
    consumer_feature_extent: int
    owner_extent: int
    streamed_extent: int
    finalizer_tasks: tuple[int, ...] = ()

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.producer_task,
                    self.maximum_fold_task,
                    self.normalizer_fold_task,
                    self.consumer_task,
                    *self.finalizer_tasks,
                )
            )
        )


def attach_sm90_contract_fold_schedule(
    request: ExecutablePhysicalRequest,
) -> Sm90ContractFoldScheduleAttachment:
    """Install the FA4-derived schedule from generic task and Event structure.

    Candidate discovery uses only phase capabilities, exact task relations,
    axis roles, operand index relations, dtypes, layouts, and resource bounds.
    Scalar AST contents remain exclusively in the request's algebra record and
    therefore cannot select a schedule family.
    """
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the fused Contract/Fold schedule requires an SM90 request")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError(
            "the SM90 normalized-exponential Contract/Fold schedule does not implement "
            f"{request.numerical_policy.name} Fold evaluation; select FAST explicitly"
        )
    core = _sm90_contract_fold_core(request)
    forward_task_ids = _sm90_owned_task_subset(request, core)
    declared_pullback_tasks = _declared_pullback_task_ids(request)
    split_saved_state = _pullback_recomputes_normalized_weight(request)
    if split_saved_state:
        declared_pullback_tasks = tuple(
            task_id for task_id in declared_pullback_tasks if task_id not in set(forward_task_ids)
        )
        declared_pullback_tasks = _include_pullback_recompute_tasks(
            request,
            declared_pullback_tasks,
            forward_task_ids,
        )
        declared_pullback_tasks = _absorb_terminal_relation_tasks(request, declared_pullback_tasks)
    if declared_pullback_tasks and split_saved_state:
        successors, _ = _task_graph(request)
        if not _task_subset_is_dependency_convex(set(declared_pullback_tasks), successors):
            raise ValueError("planner-declared normalized-exp pullback operations do not form a dependency-convex cut")
        forward_schedule = _sm90_region_schedule(
            request,
            core,
            forward_task_ids,
            direction=Sm90ContractFoldDirection.FORWARD,
        )
        pullback_schedule = _sm90_region_schedule(
            request,
            core,
            declared_pullback_tasks,
            direction=Sm90ContractFoldDirection.PULLBACK,
        )
        component = set(forward_task_ids) | set(declared_pullback_tasks)
        retained_schedules = tuple(schedule for schedule in request.schedules if not set(schedule.task_ids) & component)
        overlapping = tuple(
            schedule
            for schedule in request.schedules
            if set(schedule.task_ids) & component and not set(schedule.task_ids) <= component
        )
        if overlapping:
            raise ValueError("the split SM90 components partially overlap an existing physical schedule")
        attached = install_physical_schedule_cover(request, (forward_schedule, pullback_schedule, *retained_schedules))
        attached = replace(
            attached,
            provenance={
                **attached.provenance,
                "sm90_schedule_attachment": {
                    "family": (forward_schedule.family.name, pullback_schedule.family.name),
                    "selection": "phase_axes_relations_dtype_layout_resources",
                    "implemented_capabilities": tuple(
                        capability.value for capability in _SM90_CONTRACT_FOLD_CAPABILITIES
                    ),
                    "scalar_maps": "canonical_tensor_program_ast",
                    "internal_events": "schedule_owned_program_order",
                    "saved_state_boundary": "typed_value_dependences",
                },
            },
        )
        return Sm90ContractFoldScheduleAttachment(
            attached,
            tuple(sorted(component)),
            core.task_ids,
        )

    fused_task_ids = forward_task_ids
    if declared_pullback_tasks:
        owned = set(fused_task_ids) | set(declared_pullback_tasks)
        successors, _ = _task_graph(request)
        if not _task_subset_is_dependency_convex(owned, successors):
            raise ValueError("planner-declared normalized-exp pullback operations do not form a dependency-convex cut")
        fused_task_ids = tuple(sorted(owned))
    component = set(fused_task_ids)
    direction = Sm90ContractFoldDirection.PULLBACK if declared_pullback_tasks else Sm90ContractFoldDirection.FORWARD
    fused_schedule = _sm90_region_schedule(request, core, fused_task_ids, direction=direction)
    retained_schedules = tuple(schedule for schedule in request.schedules if not set(schedule.task_ids) & component)
    overlapping = tuple(
        schedule
        for schedule in request.schedules
        if set(schedule.task_ids) & component and not set(schedule.task_ids) <= component
    )
    if overlapping:
        raise ValueError("the fused SM90 component partially overlaps an existing physical schedule")
    schedules = (fused_schedule, *retained_schedules)
    attached = install_physical_schedule_cover(request, schedules)
    attached = replace(
        attached,
        provenance={
            **attached.provenance,
            "sm90_schedule_attachment": {
                "family": fused_schedule.family.name,
                "selection": "phase_axes_relations_dtype_layout_resources",
                "implemented_capabilities": tuple(capability.value for capability in _SM90_CONTRACT_FOLD_CAPABILITIES),
                "scalar_maps": "canonical_tensor_program_ast",
                "internal_events": "schedule_owned_program_order",
            },
        },
    )
    return Sm90ContractFoldScheduleAttachment(attached, fused_task_ids, core.task_ids)


def _sm90_region_schedule(
    request: ExecutablePhysicalRequest,
    core: _Sm90ContractFoldCore,
    task_ids: tuple[int, ...],
    *,
    direction: Sm90ContractFoldDirection,
) -> PhysicalSchedule:
    component = set(task_ids)
    return PhysicalSchedule(
        family=PhysicalScheduleFamily(
            (
                "sm90_contract_fold_pullback_pipeline_v1"
                if direction is Sm90ContractFoldDirection.PULLBACK
                else "sm90_contract_fold_map_pipeline_v1"
            ),
            PhysicalTarget.SM90,
            tuple(
                _sm90_task_constraint(
                    request.task_plan.tasks[task_id],
                    is_contract_fold_spine=direction is Sm90ContractFoldDirection.FORWARD
                    and task_id == core.producer_task,
                )
                for task_id in task_ids
            ),
        ),
        task_ids=task_ids,
        parameters=_sm90_schedule_parameters(request, core, task_ids, direction=direction),
        enforced_orders=tuple(
            PhysicalTaskOrder(
                PhysicalOrderKind.PROGRAM_ORDER,
                _task_id(request, plan.required_dependence.relation.source),
                _task_id(request, plan.required_dependence.relation.target),
                plan.scheduled_dependence,
                {"mechanism": "fused_schedule_phase_order"},
            )
            for plan in request.event_plans
            if {
                _task_id(request, plan.required_dependence.relation.source),
                _task_id(request, plan.required_dependence.relation.target),
            }
            <= component
        ),
    )


def _declared_pullback_task_ids(request: ExecutablePhysicalRequest) -> tuple[int, ...]:
    if not request.normalized_exp_contract_pullbacks:
        return ()
    if len(request.normalized_exp_contract_pullbacks) != 1:
        raise ValueError("select one planner-declared normalized-exp pullback capability explicitly")
    operation_to_task = {
        operation_ordinal: task.id for task in request.task_plan.tasks for operation_ordinal in task.operation_ordinals
    }
    capability = request.normalized_exp_contract_pullbacks[0]
    try:
        return tuple(sorted({operation_to_task[ordinal] for ordinal in capability.operation_ordinals}))
    except KeyError as error:
        raise ValueError("planner-declared normalized-exp pullback operation ordinal is stale") from error


def _pullback_recomputes_normalized_weight(request: ExecutablePhysicalRequest) -> bool:
    """Whether the pullback owns a distinct normalized-weight producer.

    Equality means the reverse consumes the forward exponential materialization.
    A distinct operation is the typed saved-state boundary: reverse recomputes
    normalized weights after the intervening graph.
    """
    if not request.normalized_exp_contract_pullbacks:
        return False
    if len(request.normalized_exp_contract_regions) != 1 or len(request.normalized_exp_contract_pullbacks) != 1:
        raise ValueError("split normalized-exp scheduling requires one forward region and one pullback")
    region = request.normalized_exp_contract_regions[0]
    pullback = request.normalized_exp_contract_pullbacks[0]
    if pullback.normalized_weight_operation == region.normalized_weight_operation:
        return False

    algebra_record = request.algebra_record
    encoded_program = (
        algebra_record.get("program")
        if algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1"
        else algebra_record
    )
    if not isinstance(encoded_program, dict):
        raise ValueError("normalized-exp request must contain a canonical TensorProgram")
    program = deserialize_tensor_program(json.dumps(encoded_program, sort_keys=True, separators=(",", ":")))
    producer_by_value = {
        output: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    pending = list(primitive_inputs(program.operations[pullback.normalized_weight_operation]))
    visited = set(pending)
    while pending:
        value = pending.pop()
        producer_ordinal = producer_by_value.get(value)
        if producer_ordinal is None:
            continue
        producer = program.operations[producer_ordinal]
        if isinstance(producer, ContractPrimitive):
            if producer_ordinal != region.producer_operation:
                return True
            continue
        for operand in primitive_inputs(producer):
            if operand not in visited:
                visited.add(operand)
                pending.append(operand)
    return False


def _absorb_terminal_relation_tasks(
    request: ExecutablePhysicalRequest,
    pullback_task_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Extend a reverse refinement through dependency-proven layout tails."""
    algebra_record = request.algebra_record
    encoded_program = (
        algebra_record.get("program")
        if algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1"
        else algebra_record
    )
    if not isinstance(encoded_program, dict):
        raise ValueError("normalized-exp request must contain a canonical TensorProgram")
    program = deserialize_tensor_program(json.dumps(encoded_program, sort_keys=True, separators=(",", ":")))
    successors, predecessors = _task_graph(request)
    tasks = {task.id: task for task in request.task_plan.tasks}
    owned = set(pullback_task_ids)
    frontier = {successor for task_id in owned for successor in successors[task_id]}
    while frontier:
        task_id = min(frontier)
        frontier.remove(task_id)
        task = tasks[task_id]
        if not predecessors[task_id] <= owned:
            continue
        if not all(
            _is_layout_only_terminal_operation(program.operations[ordinal]) for ordinal in task.operation_ordinals
        ):
            continue
        owned.add(task_id)
        frontier.update(successors[task_id] - owned)
    return tuple(sorted(owned))


def _is_layout_only_terminal_operation(operation: TensorPrimitive) -> bool:
    if isinstance(operation, RelationPrimitive):
        return True
    if not isinstance(operation, FoldPrimitive):
        return False
    input_elements = math.prod(operation.input.shape)
    output_elements = math.prod(operation.output.shape)
    return input_elements == output_elements and operation.input.dtype is operation.output.dtype


def _include_pullback_recompute_tasks(
    request: ExecutablePhysicalRequest,
    pullback_task_ids: tuple[int, ...],
    forward_task_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Close the reverse normalized-weight producer over its recompute slice."""
    algebra_record = request.algebra_record
    encoded_program = (
        algebra_record.get("program")
        if algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1"
        else algebra_record
    )
    if not isinstance(encoded_program, dict):
        raise ValueError("normalized-exp request must contain a canonical TensorProgram")
    program = deserialize_tensor_program(json.dumps(encoded_program, sort_keys=True, separators=(",", ":")))
    capability = request.normalized_exp_contract_pullbacks[0]
    operation_to_task = {ordinal: task.id for task in request.task_plan.tasks for ordinal in task.operation_ordinals}
    producer_by_value = {
        output: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    forward_tasks = set(forward_task_ids)
    owned = set(pullback_task_ids)
    pending = list(primitive_inputs(program.operations[capability.normalized_weight_operation]))
    visited = set(pending)
    while pending:
        value = pending.pop()
        producer_ordinal = producer_by_value.get(value)
        if producer_ordinal is None:
            continue
        task_id = operation_to_task[producer_ordinal]
        if task_id in forward_tasks:
            continue
        owned.add(task_id)
        # The first Contract on the normalized-weight ancestry is the
        # recomputed producer owned by this physical region.  Its operands are
        # component inputs, even when a containing program produced them with
        # other Contracts.  Walking through that boundary would incorrectly
        # absorb unrelated projection work and the original forward region.
        if isinstance(program.operations[producer_ordinal], ContractPrimitive):
            continue
        for operand in primitive_inputs(program.operations[producer_ordinal]):
            if operand not in visited:
                visited.add(operand)
                pending.append(operand)
    return tuple(sorted(owned))


def _sm90_contract_fold_core(request: ExecutablePhysicalRequest) -> _Sm90ContractFoldCore:
    tasks = request.task_plan.tasks
    task_successors, _ = _task_graph(request)
    fused_candidates = tuple(
        task
        for task in tasks
        if _task_contains_contract_fold_region(task)
        and len(task.structure.streamed_axes) == 1
        and len(task.structure.contraction_axes) == 1
    )
    if fused_candidates:
        if len(fused_candidates) != 1:
            raise ValueError(f"expected one fused Contract/Fold phase DAG, found {len(fused_candidates)}")
        task = fused_candidates[0]
        normalized_axis = task.structure.streamed_axes[0]
        producer_feature = task.structure.contraction_axes[0]
        return _Sm90ContractFoldCore(
            task.id,
            task.id,
            task.id,
            task.id,
            normalized_axis.axis_id,
            producer_feature.extent,
            _consumer_feature_extent(task, normalized_axis.axis_id, producer_feature.axis_id),
            _owner_extent(task, normalized_axis.axis_id, producer_feature.axis_id),
            normalized_axis.extent,
        )

    split_candidates = _split_contract_fold_cores(request)
    if split_candidates:
        if len(split_candidates) != 1:
            raise ValueError(f"expected one split Contract/Fold phase region, found {len(split_candidates)}")
        return split_candidates[0]

    contract_tasks = tuple(task for task in tasks if _phase_count(task, TaskCapability.MATRIX_CONTRACT))
    fold_tasks = tuple(task for task in tasks if _phase_count(task, TaskCapability.ASSOCIATIVE_FOLD))
    task_successors, _ = _task_graph(request)
    candidates: set[_Sm90ContractFoldCore] = set()
    for consumer in contract_tasks:
        if len(consumer.structure.contraction_axes) != 1:
            continue
        normalized_axis = consumer.structure.contraction_axes[0]
        matching_folds = tuple(
            task
            for task in fold_tasks
            if normalized_axis.axis_id
            in {*_axis_ids(task.structure.reduction_axes), *_axis_ids(task.structure.streamed_axes)}
        )
        for maximum in matching_folds:
            for normalizer in matching_folds:
                if maximum.id == normalizer.id or not _task_reaches(request, maximum.id, normalizer.id):
                    continue
                if not _task_reaches(request, normalizer.id, consumer.id):
                    continue
                for producer in contract_tasks:
                    if producer.id == consumer.id or len(producer.structure.contraction_axes) != 1:
                        continue
                    if producer.id == maximum.id:
                        continue
                    producer_feature = producer.structure.contraction_axes[0]
                    if producer_feature.axis_id == normalized_axis.axis_id:
                        continue
                    if not _task_reaches(request, producer.id, maximum.id):
                        continue
                    try:
                        candidate = _Sm90ContractFoldCore(
                            producer.id,
                            maximum.id,
                            normalizer.id,
                            consumer.id,
                            normalized_axis.axis_id,
                            producer_feature.extent,
                            _consumer_feature_extent(consumer, normalized_axis.axis_id, producer_feature.axis_id),
                            _owner_extent(maximum, normalized_axis.axis_id, producer_feature.axis_id),
                            normalized_axis.extent,
                        )
                    except ValueError:
                        continue
                    if not _task_subset_is_dependency_convex(set(candidate.task_ids), task_successors):
                        continue
                    candidates.add(candidate)
    if len(candidates) != 1:
        raise ValueError(
            "expected one emitter-covered Contract/Fold/Contract phase region containing only "
            "Map, Fold, and Contract phases; Relation phases must first be canonicalized into "
            f"typed operand relations or domain predicates, found {len(candidates)}"
        )
    return next(iter(candidates))


def _sm90_schedule_parameters(
    request: ExecutablePhysicalRequest,
    core: _Sm90ContractFoldCore,
    fused_task_ids: tuple[int, ...],
    *,
    direction: Sm90ContractFoldDirection,
) -> dict[str, object]:
    tasks = request.task_plan.tasks
    producer_feature_extent = core.producer_feature_extent
    consumer_feature_extent = core.consumer_feature_extent
    owner_extent = core.owner_extent
    streamed_extent = core.streamed_extent
    if producer_feature_extent != 128 or consumer_feature_extent != 128:
        raise ValueError("the current SM90 Contract/Fold schedule requires explicit producer and consumer D=128")
    for task_id in fused_task_ids:
        task = tasks[task_id]
        if any(element_type not in _SM90_SUPPORTED_ELEMENT_TYPES for element_type in task.structure.element_types):
            raise ValueError(
                "the current SM90 Contract/Fold schedule supports BF16 data with scalar predicate/index values"
            )
        if any(layout not in _SM90_SUPPORTED_LAYOUTS for layout in task.structure.layout_classes):
            raise ValueError("the current SM90 Contract/Fold schedule requires logical row-major layouts")
        if any(access.relation_class not in _SM90_SUPPORTED_RELATIONS for access in task.accesses):
            raise ValueError("the current SM90 Contract/Fold schedule supports identity and affine accesses")
        if not _resources_fit(task.structure.resources, _SM90_MAXIMUM_RESOURCES):
            raise ValueError("generic task resource estimates exceed the current SM90 schedule family")
    is_forward = direction is Sm90ContractFoldDirection.FORWARD
    if is_forward:
        contract = select_sm90_contract_schedule(
            owner_extent=owner_extent,
            streamed_extent=streamed_extent,
            producer_feature_extent=producer_feature_extent,
            consumer_feature_extent=consumer_feature_extent,
        )
        schedule = Sm90ForwardPipelineSchedule(contract=contract, stages=2)
        specialization = {
            "execution_graph": "forward_contract_fold_pipeline",
            "stages": schedule.stages,
            "resident_left_in_registers": schedule.resident_left_in_registers,
            "intra_warpgroup_overlap": schedule.intra_warpgroup_overlap,
            "consumer_register_source": schedule.consumer_register_source,
        }
    else:
        contract = select_sm90_contract_schedule(
            owner_extent=owner_extent,
            streamed_extent=streamed_extent,
            producer_feature_extent=producer_feature_extent,
            consumer_feature_extent=consumer_feature_extent,
            threads=384,
            owner_tile=64,
            streamed_tile=128,
        )
        schedule = Sm90BackwardAccumulatorSchedule(contract=contract)
        specialization = {
            "execution_graph": "contract_fold_pullback_pipeline",
            "resident_left_stages": schedule.resident_left_stages,
            "consumer_output_cotangent_stages": schedule.consumer_output_cotangent_stages,
            "score_cotangent_stages": schedule.score_cotangent_stages,
            "deterministic_accumulation": schedule.deterministic_accumulation,
            "epilogue_tile_m": schedule.epilogue_tile_m,
            "epilogue_threads": schedule.epilogue_threads,
        }
    return {
        "direction": direction.value,
        "contract": {
            "tile_m": contract.tile_m,
            "tile_n": contract.tile_n,
            "threads": contract.threads,
            "wgmma_instruction": contract.wgmma_instruction,
            "accumulator_dtype": contract.accumulator_dtype.value,
            "producer_feature_extent": producer_feature_extent,
            "consumer_feature_extent": consumer_feature_extent,
            "owner_extent": owner_extent,
            "streamed_extent": streamed_extent,
        },
        "owned_phase_regions": ("contract_fold",) if is_forward else ("contract_fold", "map_contract"),
        "scalar_maps": "compiled_from_canonical_ast",
        "domain_predicates": _request_domain_predicate_records(request),
        "event_realization": "schedule_owned_program_order",
        **specialization,
    }


def _request_domain_predicate_records(request: ExecutablePhysicalRequest) -> tuple[object, ...]:
    restrictions = request.algebra_record.get("domain_restrictions", ())
    if not isinstance(restrictions, (tuple, list)):
        raise ValueError("canonical domain restrictions must be an ordered sequence")
    predicates: list[object] = []
    for restriction in restrictions:
        if not isinstance(restriction, dict) or "predicate" not in restriction:
            raise ValueError("canonical domain restriction must carry a typed predicate")
        predicates.append(restriction["predicate"])
    return tuple(predicates)


def _sm90_task_constraint(
    task: TiledTask,
    *,
    is_contract_fold_spine: bool,
) -> PhysicalTaskConstraint:
    structure = task.structure
    return PhysicalTaskConstraint(
        principal_kind=structure.principal_kind,
        required_capabilities=(
            (
                TaskCapability.MATRIX_CONTRACT,
                TaskCapability.MATRIX_CONTRACT,
                TaskCapability.ASSOCIATIVE_FOLD,
                TaskCapability.SCALAR_MAP,
            )
            if is_contract_fold_spine
            else (_principal_capability(task),)
        ),
        supported_capabilities=(
            _SM90_CONTRACT_FOLD_CAPABILITIES
            if is_contract_fold_spine
            else tuple(dict.fromkeys(phase.capability for phase in structure.phases))
        ),
        required_edges=(),
        owner_rank=len(structure.owner_axes),
        streamed_rank=len(structure.streamed_axes),
        reduction_rank=len(structure.reduction_axes),
        contraction_rank=len(structure.contraction_axes),
        supported_relation_classes=_SM90_SUPPORTED_RELATIONS,
        supported_element_types=_SM90_SUPPORTED_ELEMENT_TYPES,
        supported_layout_classes=_SM90_SUPPORTED_LAYOUTS,
        maximum_resources=_SM90_MAXIMUM_RESOURCES,
    )


def _sm90_owned_task_subset(
    request: ExecutablePhysicalRequest,
    core: _Sm90ContractFoldCore,
) -> tuple[int, ...]:
    """Return the dependency-convex tasks implemented by one schedule.

    The forward region may already be one fused task. Planner-declared
    pullback tasks are added by the caller; unrelated descendants must never
    be inferred from reachability alone. Pure layout tails are retained here
    because the emitter implements them as its physical output layout.
    """
    core_ids = set(core.task_ids)
    successors, predecessors = _task_graph(request)
    owned = core_ids
    declared_tail_operations = {
        ordinal for region in request.normalized_exp_contract_regions for ordinal in region.finalizer_tail_operations
    }
    # Layout-only Relation tails are legalizations of the component result,
    # not separate computation. Absorb a dependency-closed chain so the
    # physical kernel can expose its native contiguous result directly.
    while True:
        relation_tails = {
            task_id
            for task_id in _reachable(successors, owned) - owned
            if predecessors[task_id] <= owned
            and set(request.task_plan.tasks[task_id].operation_ordinals) <= declared_tail_operations
            and all(
                phase.capability is TaskCapability.INDEX_RELATION
                for phase in request.task_plan.tasks[task_id].structure.phases
            )
        }
        if not relation_tails:
            break
        owned |= relation_tails
    unsupported = tuple(
        task_id
        for task_id in sorted(owned)
        if not _task_supported_by_sm90_contract_fold(request.task_plan.tasks[task_id])
    )
    if unsupported:
        raise ValueError(f"SM90 Contract/Fold paths contain unsupported physical tasks: {unsupported}")
    if not _task_subset_is_dependency_convex(owned, successors):
        raise ValueError("the selected SM90 Contract/Fold task subset is not dependency-convex")
    return tuple(sorted(owned))


def _split_contract_fold_cores(request: ExecutablePhysicalRequest) -> tuple[_Sm90ContractFoldCore, ...]:
    """Find a fork/join Fold finalization split across canonical tasks."""
    successors, _ = _task_graph(request)
    candidates: set[_Sm90ContractFoldCore] = set()
    for spine in request.task_plan.tasks:
        if (
            _phase_count(spine, TaskCapability.MATRIX_CONTRACT) < 2
            or _phase_count(spine, TaskCapability.ASSOCIATIVE_FOLD) < 1
            or _phase_count(spine, TaskCapability.SCALAR_MAP) < 1
            or len(spine.structure.streamed_axes) != 1
            or len(spine.structure.contraction_axes) != 1
            or not _task_supported_by_sm90_contract_fold(spine)
        ):
            continue
        normalized_axis = spine.structure.streamed_axes[0]
        producer_feature = spine.structure.contraction_axes[0]
        descendants = _reachable(successors, {spine.id}) - {spine.id}
        folds = tuple(
            task
            for task in request.task_plan.tasks
            if task.id in descendants
            and _phase_count(task, TaskCapability.ASSOCIATIVE_FOLD)
            and normalized_axis.axis_id
            in {*_axis_ids(task.structure.reduction_axes), *_axis_ids(task.structure.streamed_axes)}
            and _task_supported_by_sm90_contract_fold(task)
        )
        for fold in folds:
            fold_descendants = _reachable(successors, {fold.id}) - {fold.id}
            finalizers = tuple(
                task
                for task in request.task_plan.tasks
                if task.id in fold_descendants
                and task.id in descendants
                and _phase_count(task, TaskCapability.SCALAR_MAP)
                and _task_supported_by_sm90_contract_fold(task)
            )
            for finalizer in finalizers:
                core_ids = {spine.id, fold.id, finalizer.id}
                if not _task_subset_is_dependency_convex(core_ids, successors):
                    continue
                try:
                    candidate = _Sm90ContractFoldCore(
                        spine.id,
                        spine.id,
                        fold.id,
                        spine.id,
                        normalized_axis.axis_id,
                        producer_feature.extent,
                        _consumer_feature_extent(
                            spine,
                            normalized_axis.axis_id,
                            producer_feature.axis_id,
                        ),
                        _owner_extent(spine, normalized_axis.axis_id, producer_feature.axis_id),
                        normalized_axis.extent,
                        (finalizer.id,),
                    )
                except ValueError:
                    continue
                candidates.add(candidate)
    return tuple(candidates)


def _task_contains_contract_fold_region(task: TiledTask) -> bool:
    """Recognize a reusable Contract/Fold + Map/Contract phase region."""
    structure = task.structure
    if not _task_supported_by_sm90_contract_fold(task):
        return False
    successors = {phase.id: set() for phase in structure.phases}
    predecessors = {phase.id: set() for phase in structure.phases}
    for edge in structure.phase_edges:
        successors[edge.source_phase].add(edge.target_phase)
        predecessors[edge.target_phase].add(edge.source_phase)
    contracts = tuple(phase.id for phase in structure.phases if phase.capability is TaskCapability.MATRIX_CONTRACT)
    folds = tuple(phase.id for phase in structure.phases if phase.capability is TaskCapability.ASSOCIATIVE_FOLD)
    finalizers = tuple(phase.id for phase in structure.phases if phase.capability is TaskCapability.SCALAR_MAP)
    if len(contracts) < 2 or len(folds) < 2 or not finalizers:
        return False
    for producer in contracts:
        producer_descendants = _reachable(successors, {producer})
        downstream_folds = set(folds) & producer_descendants
        if len(downstream_folds) < 2:
            continue
        for consumer in set(contracts) & producer_descendants - {producer}:
            for finalizer in finalizers:
                ancestors = _reachable(predecessors, {finalizer})
                if consumer not in ancestors or not downstream_folds <= ancestors:
                    continue
                direct_predecessors = predecessors[finalizer]
                if consumer not in direct_predecessors or not downstream_folds & direct_predecessors:
                    continue
                undirected = {phase_id: successors[phase_id] | predecessors[phase_id] for phase_id in successors}
                # A fused task may have several result branches (for example,
                # an output and Fold state needed by a pullback). They remain
                # one reusable region as long as every phase is connected to
                # the Contract/Fold spine and has a supported capability.
                if _reachable(undirected, {producer}) == set(successors):
                    return True
    return False


def _task_supported_by_sm90_contract_fold(task: TiledTask) -> bool:
    structure = task.structure
    return (
        all(phase.capability in _SM90_CONTRACT_FOLD_CAPABILITIES for phase in structure.phases)
        and all(access.relation_class in _SM90_SUPPORTED_RELATIONS for access in task.accesses)
        and all(element_type in _SM90_SUPPORTED_ELEMENT_TYPES for element_type in structure.element_types)
        and all(layout in _SM90_SUPPORTED_LAYOUTS for layout in structure.layout_classes)
        and _resources_fit(structure.resources, _SM90_MAXIMUM_RESOURCES)
    )


def _task_graph(request: ExecutablePhysicalRequest) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    successors = {task.id: set() for task in request.task_plan.tasks}
    predecessors = {task.id: set() for task in request.task_plan.tasks}
    family_ids = {task.family: task.id for task in request.task_plan.tasks}
    for dependence in request.task_plan.dependences:
        source_family = dependence.relation.source
        target_family = dependence.relation.target
        if not isinstance(source_family, TaskFamily) or not isinstance(target_family, TaskFamily):
            raise TypeError("SM90 schedule attachment requires tiled task-family dependences")
        source = family_ids[source_family]
        target = family_ids[target_family]
        successors[source].add(target)
        predecessors[target].add(source)
    return successors, predecessors


def _reachable(edges: dict[int, set[int]], starts: set[int]) -> set[int]:
    reached: set[int] = set()
    frontier = list(starts)
    while frontier:
        task_id = frontier.pop()
        if task_id in reached:
            continue
        reached.add(task_id)
        frontier.extend(edges[task_id] - reached)
    return reached


def _task_subset_is_dependency_convex(owned: set[int], successors: dict[int, set[int]]) -> bool:
    for source in owned:
        frontier = list(successors[source])
        visited: set[int] = set()
        while frontier:
            task_id = frontier.pop()
            if task_id in visited:
                continue
            visited.add(task_id)
            if task_id in owned:
                continue
            if _reachable(successors, {task_id}) & owned:
                return False
            frontier.extend(successors[task_id] - visited)
    return True


def _has_task_dependence(request: ExecutablePhysicalRequest, source: int, target: int) -> bool:
    source_family = request.task_plan.tasks[source].family
    target_family = request.task_plan.tasks[target].family
    return any(
        dependence.relation.source == source_family and dependence.relation.target == target_family
        for dependence in request.task_plan.dependences
    )


def _task_reaches(request: ExecutablePhysicalRequest, source: int, target: int) -> bool:
    successors = {task.id: set() for task in request.task_plan.tasks}
    family_ids = {task.family: task.id for task in request.task_plan.tasks}
    for dependence in request.task_plan.dependences:
        source_family = dependence.relation.source
        target_family = dependence.relation.target
        if not isinstance(source_family, TaskFamily) or not isinstance(target_family, TaskFamily):
            raise TypeError("SM90 schedule attachment requires tiled task-family dependences")
        successors[family_ids[source_family]].add(family_ids[target_family])
    frontier = [source]
    reached: set[int] = set()
    while frontier:
        task_id = frontier.pop()
        if task_id == target:
            return True
        if task_id in reached:
            continue
        reached.add(task_id)
        frontier.extend(successors[task_id] - reached)
    return False


def _phase_count(task: TiledTask, capability: TaskCapability) -> int:
    return sum(phase.capability is capability for phase in task.structure.phases)


def _principal_capability(task: TiledTask) -> TaskCapability:
    matches = {
        phase.capability
        for phase in task.structure.phases
        if phase.capability.value.startswith(task.structure.principal_kind.value)
    }
    if len(matches) == 1:
        return next(iter(matches))
    capabilities = tuple(dict.fromkeys(phase.capability for phase in task.structure.phases))
    if len(capabilities) == 1:
        return capabilities[0]
    preferred = {
        "contract": TaskCapability.MATRIX_CONTRACT,
        "fold": TaskCapability.ASSOCIATIVE_FOLD,
        "map": TaskCapability.SCALAR_MAP,
        "relation": TaskCapability.INDEX_RELATION,
    }.get(task.structure.principal_kind.value)
    if preferred is None or preferred not in capabilities:
        raise ValueError("SM90 task has no schedule-visible principal capability")
    return preferred


def _consumer_feature_extent(task: TiledTask, normalized_axis_id: int, producer_feature_axis_id: int) -> int:
    candidates = _consumer_feature_axes(task, normalized_axis_id, producer_feature_axis_id)
    extents = {extent for _, extent in candidates}
    if len(extents) != 1:
        raise ValueError("the current SM90 consumer Contract requires one explicit result feature dimension")
    return next(iter(extents))


def _consumer_feature_axes(
    task: TiledTask,
    normalized_axis_id: int,
    producer_feature_axis_id: int,
) -> set[tuple[int, int]]:
    successors = {phase.id: set() for phase in task.structure.phases}
    for edge in task.structure.phase_edges:
        successors[edge.source_phase].add(edge.target_phase)
    fold_phases = tuple(
        phase.id for phase in task.structure.phases if phase.capability is TaskCapability.ASSOCIATIVE_FOLD
    )
    consumer_phases = tuple(
        phase.id
        for phase in task.structure.phases
        if phase.capability is TaskCapability.MATRIX_CONTRACT
        and any(phase.id in _reachable(successors, {fold_phase}) for fold_phase in fold_phases)
    )
    if len(consumer_phases) != 1:
        raise ValueError("the current SM90 schedule requires one Fold-dependent consumer Contract phase")
    consumer_phase = consumer_phases[0]

    def output_domains(phase_ids: tuple[int, ...]) -> set[tuple[tuple[int, int], ...]]:
        domains: set[tuple[tuple[int, int], ...]] = set()
        for access in task.accesses:
            if access.phase_id not in phase_ids or access.kind is TaskValueAccessKind.LIVE_IN:
                continue
            axes = tuple(int(axis) for axis in access.relation_record.get("source_axes", ()))
            shape = tuple(int(extent) for extent in access.relation_record.get("source_shape", ()))
            target_axes = tuple(int(axis) for axis in access.relation_record.get("target_axes", ()))
            target_shape = tuple(int(extent) for extent in access.relation_record.get("target_shape", ()))
            if len(axes) == len(shape) and axes == target_axes and shape == target_shape:
                domains.add(tuple(zip(axes, shape, strict=True)))
        return domains

    consumer_domains = output_domains((consumer_phase,))
    if len(consumer_domains) != 1:
        raise ValueError("the Fold-dependent consumer Contract must have one explicit output domain")
    owner_domains: set[tuple[tuple[int, int], ...]] = set()
    for fold_phase in fold_phases:
        if consumer_phase in _reachable(successors, {fold_phase}):
            owner_domains.update(output_domains((fold_phase,)))
    owner_axis_sets = {frozenset(domain) for domain in owner_domains}
    if len(owner_axis_sets) != 1:
        raise ValueError("normalized Fold outputs must agree on one owner domain")
    consumer_domain = next(iter(consumer_domains))
    owner_axes = next(iter(owner_axis_sets))
    return {
        (axis, extent)
        for axis, extent in consumer_domain
        if (axis, extent) not in owner_axes and axis not in (normalized_axis_id, producer_feature_axis_id)
    }


def _owner_extent(task: TiledTask, normalized_axis_id: int, producer_feature_axis_id: int) -> int:
    consumer_feature_axis_ids = {
        axis_id for axis_id, _ in _consumer_feature_axes(task, normalized_axis_id, producer_feature_axis_id)
    }
    candidates = tuple(
        axis.extent
        for axis in task.structure.owner_axes
        if axis.axis_id not in (normalized_axis_id, producer_feature_axis_id)
        and axis.axis_id not in consumer_feature_axis_ids
        and axis.extent > 1
    )
    if not candidates:
        raise ValueError("the current SM90 schedule cannot identify a relation-independent owner dimension")
    return max(candidates)


def _task_id(request: ExecutablePhysicalRequest, family: TaskFamily) -> int:
    matches = tuple(task.id for task in request.task_plan.tasks if task.family == family)
    if len(matches) != 1:
        raise ValueError("Event relation endpoint does not identify one canonical task")
    return matches[0]


def _axis_ids(axes: tuple[TaskAxisTile, ...]) -> tuple[int, ...]:
    return tuple(axis.axis_id for axis in axes)


def _resources_fit(actual: TaskResourceEstimate, maximum: TaskResourceEstimate) -> bool:
    return (
        actual.registers_per_thread <= maximum.registers_per_thread
        and actual.shared_memory_bytes <= maximum.shared_memory_bytes
        and actual.barrier_count <= maximum.barrier_count
    )


def jax_tensor_program_executable_request(
    program: TensorProgram,
    *,
    tile_size: int,
    target: PhysicalTarget,
    normalized_exp_pullback: NormalizedExpContractPullback | None = None,
) -> ExecutablePhysicalRequest:
    """Lower an imported JAX TensorProgram through generic algebra rewrites to the common ABI."""
    if tile_size <= 0:
        raise ValueError("executable tile size must be positive")
    source_fingerprint = program.semantic_fingerprint
    widened = fold_widening_converts_into_contracts(program)
    canonicalized = sink_fold_invariant_division_through_contract(widened.program)
    layout = canonicalize_contract_layout_relations(canonicalized.program)
    composed = compose_scalar_map_chains(layout.program)
    try:
        domain = canonicalize_normalized_exp_index_domain(composed.program)
    except NormalizedExpDomainCanonicalizationError as error:
        task_program = composed.program
        algebra_record = task_program.canonical_record
        domain_record: dict[str, object] | None = None
        domain_status = {"status": "not_applicable", "reason": str(error)}
        try:
            normalized_exp_region = normalized_exp_contract_region(task_program)
        except ValueError:
            normalized_exp_regions: tuple[NormalizedExpContractRegionCapability, ...] = ()
        else:
            normalized_exp_regions = (normalized_exp_region_capability(task_program, normalized_exp_region),)
    else:
        task_program = domain.task_program
        algebra_record = domain.canonical_record
        domain_record = domain.restriction.canonical_record
        domain_status = {"status": "applied", "restriction": domain_record}
        normalized_exp_regions = (normalized_exp_region_capability(task_program, domain.region),)
    normalized_exp_pullbacks: tuple[NormalizedExpContractPullbackCapability, ...] = ()
    if normalized_exp_pullback is not None:
        if not normalized_exp_regions:
            normalized_exp_regions = (normalized_exp_region_capability(task_program, normalized_exp_pullback.region),)
        if len(normalized_exp_regions) != 1:
            raise ValueError("normalized-exp pullback requires one declared forward region capability")
        normalized_exp_pullbacks = (normalized_exp_pullback_capability(task_program, normalized_exp_pullback),)
    return _generic_request(
        task_program,
        tile_extents={axis.id: min(axis.extent, tile_size) for axis in _program_axes(task_program)},
        target=target,
        algebra_record=algebra_record,
        normalized_exp_contract_regions=normalized_exp_regions,
        normalized_exp_contract_pullbacks=normalized_exp_pullbacks,
        provenance={
            "algebra_source": "jax_owned_jaxpr",
            "differentiation_owner": "jax",
            "source_algebra_fingerprint": source_fingerprint,
            "transformed_algebra_fingerprint": canonical_fingerprint(algebra_record),
            "algebra_rewrites": ",".join(
                rewrite.name for rewrite in (*widened.rewrites, *canonicalized.rewrites, *composed.rewrites)
            ),
            "algebra_rewrite_proofs": tuple(
                {
                    "name": rewrite.name,
                    "semantic_properties": rewrite.semantic_properties,
                    "legality_checks": rewrite.legality_checks,
                    "numerical_equivalence": rewrite.numerical_equivalence.value,
                    "numerical_effect": rewrite.numerical_effect,
                }
                for rewrite in (*widened.rewrites, *canonicalized.rewrites, *composed.rewrites)
            ),
            "layout_canonicalization": layout.canonical_record,
            "index_domain_canonicalization": domain_status,
        },
    )


def normalized_exp_executable_requests(
    forward: NormalizedExpContractProgram[object],
    reverse: NormalizedExpContractPullbackProgram[object, object],
    *,
    owner_tile_size: int,
    streamed_tile_size: int,
    target: PhysicalTarget,
) -> NormalizedExpExecutableRequests:
    """Lower forward algebra and its local reverse candidate to the common ABI.

    The reverse request is deliberately marked as a candidate. Numerical
    agreement with JAX does not prove that the local scalar-VJP construction
    is the canonical import of JAX-owned AD.
    """
    if reverse.forward != forward:
        raise ValueError("reverse candidate must derive from the supplied forward program")
    if min(owner_tile_size, streamed_tile_size) <= 0:
        raise ValueError("normalized-exp executable tile sizes must be positive")

    forward_tiles = _tile_extents(
        forward.source,
        owner_axes=forward.region.owner_axes,
        streamed_axes=forward.region.dimensions.normalized_axes,
        owner_tile_size=owner_tile_size,
        streamed_tile_size=streamed_tile_size,
    )
    forward_request = _generic_request(
        forward.source,
        tile_extents=forward_tiles,
        target=target,
        provenance={"algebra_source": "generic_tensor_program"},
        normalized_exp_contract_regions=(normalized_exp_contract_region_capability(forward),),
    )

    reverse_program = replace(
        normalized_exp_contract_pullback_tensor_program(reverse.pullback),
        numerical_policy=forward.source.numerical_policy,
    )
    reverse_tiles = _tile_extents(
        reverse_program,
        owner_axes=forward.region.owner_axes,
        streamed_axes=forward.region.dimensions.normalized_axes,
        owner_tile_size=owner_tile_size,
        streamed_tile_size=streamed_tile_size,
    )
    reverse_request = _generic_request(
        reverse_program,
        tile_extents=reverse_tiles,
        target=target,
        provenance={
            "algebra_source": "local_normalized_exp_pullback_candidate",
            "acceptance": ReverseAlgebraAcceptance.CANDIDATE_AWAITING_JAXPR_IMPORT.value,
        },
    )
    ownership = derive_reverse_ownership_plan(
        reverse.pullback,
        traversal=reverse.ownership.traversal,
    )
    reverse_dataflow = derive_normalized_exp_pullback_task_dataflow(
        reverse.pullback,
        ownership,
        owner_tile_size=owner_tile_size,
        streamed_tile_size=streamed_tile_size,
        visibility_scope=EventMemoryScope.DEVICE,
    )
    return NormalizedExpExecutableRequests(
        forward_request,
        reverse_request,
        reverse_dataflow,
        ReverseAlgebraAcceptance.CANDIDATE_AWAITING_JAXPR_IMPORT,
    )


def _generic_request(
    program: TensorProgram,
    *,
    tile_extents: dict[int, int],
    target: PhysicalTarget,
    provenance: dict[str, object],
    algebra_record: dict[str, object] | None = None,
    normalized_exp_contract_regions: tuple[NormalizedExpContractRegionCapability, ...] = (),
    normalized_exp_contract_pullbacks: tuple[NormalizedExpContractPullbackCapability, ...] = (),
) -> ExecutablePhysicalRequest:
    numerical_policy = PhysicalNumericalPolicy(program.numerical_policy.value)
    values_by_ordinal = {ordinal: value for value, ordinal in tensor_program_value_ordinals(program).items()}
    forward_finalizer_boundaries = tuple(
        output
        for capability in normalized_exp_contract_regions
        for finalizer_value in (values_by_ordinal[capability.finalizer_value],)
        for operation in program.operations
        if operation is not program.operations[capability.finalizer_operation]
        and finalizer_value in primitive_inputs(operation)
        for output in primitive_outputs(operation)
    )
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extents,
        numerical_policy=numerical_policy,
        fusion_boundary_values=tuple(
            (
                *(
                    value
                    for capability in normalized_exp_contract_regions
                    for ordinal in capability.finalizer_tail_operations
                    for value in primitive_outputs(program.operations[ordinal])
                ),
                *forward_finalizer_boundaries,
                *(
                    value
                    for capability in normalized_exp_contract_pullbacks
                    for value in primitive_inputs(program.operations[capability.post_normalization_vjp_operation])
                ),
            )
        ),
    )
    event_program = eventize_composed_task_plan(task_plan)
    schedules = tuple(_schedule_for_task(task, target=target) for task in task_plan.tasks)
    allocations = tuple(
        EventTensorAllocation(
            EventStorageBinding(
                plan.domain,
                tuple(range(len(plan.domain.coordinates))),
                (0,) * len(plan.domain.coordinates),
            ),
            capacity=len(plan.domain.coordinates),
            generation_count=1,
        )
        for plan in event_program.event_plans
    )
    audit = verify_event_realizations(
        event_program,
        tuple(
            physical_event_realization(
                plan,
                mechanism="generation_counted_event",
                reason="physical schedule selection has not yet provided an exact covering order",
            )
            for plan in event_program.event_plans
        ),
    )
    canonical_algebra = program.canonical_record if algebra_record is None else algebra_record
    return ExecutablePhysicalRequest(
        algebra_record=canonical_algebra,
        algebra_fingerprint=canonical_fingerprint(canonical_algebra),
        task_plan=task_plan,
        event_plans=event_program.event_plans,
        event_allocations=allocations,
        target=target,
        numerical_policy=numerical_policy,
        schedules=schedules,
        event_realizations=EventRealizationProof(audit, ()),
        provenance=provenance,
        normalized_exp_contract_regions=normalized_exp_contract_regions,
        normalized_exp_contract_pullbacks=normalized_exp_contract_pullbacks,
    )


def _schedule_for_task(task: TiledTask, *, target: PhysicalTarget) -> PhysicalSchedule:
    structure = task.structure
    constraint = PhysicalTaskConstraint(
        principal_kind=structure.principal_kind,
        required_capabilities=tuple(phase.capability for phase in structure.phases),
        supported_capabilities=tuple(dict.fromkeys(phase.capability for phase in structure.phases)),
        required_edges=(),
        owner_rank=len(structure.owner_axes),
        streamed_rank=len(structure.streamed_axes),
        reduction_rank=len(structure.reduction_axes),
        contraction_rank=len(structure.contraction_axes),
        supported_relation_classes=tuple(dict.fromkeys(access.relation_class for access in task.accesses))
        or (RelationAccessClass.IDENTITY,),
        supported_element_types=structure.element_types,
        supported_layout_classes=structure.layout_classes,
        maximum_resources=TaskResourceEstimate(255, 228 * 1024, 32),
    )
    return PhysicalSchedule(
        PhysicalScheduleFamily(f"{structure.principal_kind.value}_task", target, (constraint,)),
        (task.id,),
        {"tile_geometry": tuple(axis.tile_extent for axis in (*structure.owner_axes, *structure.contraction_axes))},
    )


def _program_axes(program: TensorProgram) -> set[TensorAxis]:
    return {
        axis
        for value in (
            *program.inputs,
            *program.outputs,
            *(value for operation in program.operations for value in primitive_outputs(operation)),
        )
        for axis in value.axes
    }


def _tile_extents(
    program: TensorProgram,
    *,
    owner_axes: tuple[TensorAxis, ...],
    streamed_axes: tuple[TensorAxis, ...],
    owner_tile_size: int,
    streamed_tile_size: int,
) -> dict[int, int]:
    owner = set(owner_axes)
    streamed = set(streamed_axes)
    axes = {axis for value in (*program.inputs, *program.outputs) for axis in value.axes}
    for operation in program.operations:
        for output in primitive_outputs(operation):
            axes.update(output.axes)
    return {
        axis.id: min(
            axis.extent,
            owner_tile_size if axis in owner else streamed_tile_size if axis in streamed else axis.extent,
        )
        for axis in axes
    }
