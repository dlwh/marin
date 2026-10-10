# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose scheduled tasks into an ordered, whole-request kernel boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineMap,
    CuteEventCoverEndpoint,
    CuteEventCoverPiece,
    CuteEventRealization,
    CuteEventRealizationKind,
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineSlot,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
    CutePipelineTensorMapArgument,
    affine_constant,
    bind_physical_event_provenance,
    combine_physical_event_schedules,
    launch_dependency_event_schedule,
    verify_materialized_cut_launch_dependencies,
)
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.event_dataflow import CoordinateDomain, TaskFamily, TaskRelation
from tile_lifetime.physical_task_contract import (
    ExecutableEventLowering,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalOrderKind,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTaskOrder,
    TaskValueAccessKind,
    canonical_fingerprint,
    install_physical_schedule_cover,
    lower_executable_event_operations,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    TensorPrimitive,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)

_GENERIC_COMPONENT_STORAGE_DTYPES = frozenset({"bf16", "fp32", "int32", "bool"})
_PRIVATE_MAP_FOLD_MAXIMUM_OPERATIONS = 8
_PRIVATE_MAP_FOLD_MAXIMUM_INPUTS = 4
_PRIVATE_MAP_FOLD_MAXIMUM_OUTPUTS = 1
_PRIVATE_MAP_FOLD_DTYPES = frozenset({"bf16", "fp32"})


class FusedComponentEvaluation(StrEnum):
    RECURSIVE = "recursive"
    SAME_COORDINATE_SSA = "same_coordinate_ssa"


def supports_same_coordinate_producer_reuse(
    operations: tuple[TensorPrimitive, ...],
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
) -> bool:
    """Prove shared Contract results can stay in per-coordinate scalar SSA."""
    if len(live_outputs) < 2 or not operations:
        return False
    axes = live_outputs[0].axes
    if any(value.axes != axes for value in live_outputs):
        return False
    available = set(live_inputs)
    contract_ancestors: dict[ProgramValue, frozenset[ProgramValue]] = {}
    for operation in operations:
        if not isinstance(operation, (ContractPrimitive, MapPrimitive)):
            return False
        if operation.output.axes != axes or not set(primitive_inputs(operation)) <= available:
            return False
        if isinstance(operation, ContractPrimitive):
            # Reduction coordinates must never read per-output scalar bindings.
            if not set(operation.inputs) <= set(live_inputs):
                return False
            contract_ancestors[operation.output] = frozenset({operation.output})
        else:
            contract_ancestors[operation.output] = frozenset().union(
                *(contract_ancestors.get(value, frozenset()) for value in operation.inputs)
            )
        available.add(operation.output)
    if not set(live_outputs) <= set(contract_ancestors):
        return False
    seen: set[ProgramValue] = set()
    for value in live_outputs:
        ancestors = contract_ancestors[value]
        if seen & ancestors:
            return True
        seen.update(ancestors)
    return False


@dataclass(frozen=True)
class ConcreteKernelEventBoundary:
    """One exact Event whose endpoints belong to distinct kernel components."""

    event_id: int
    producer_component: int
    consumer_component: int
    relation_dimensions: int
    capacity: int
    generation_count: int


@dataclass(frozen=True)
class CanonicalKernelComponent:
    """One selected schedule and the algebra/value boundary its emitter owns."""

    id: int
    schedule_index: int
    schedule_fingerprint: str
    task_ids: tuple[int, ...]
    operation_ordinals: tuple[int, ...]
    operations: tuple[dict[str, Any], ...]
    live_input_value_ids: tuple[int, ...]
    live_output_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class DependencyConvexReductionPhase:
    """One launch group over existing canonical physical phases."""

    id: int
    physical_phases: tuple[PhysicalPhase, ...]
    component: CanonicalKernelComponent
    resource_record: Mapping[str, Any]


class DependencyConvexPhaseCut(StrEnum):
    REDUCTION_STATE = "dependency_convex_reduction_state_cut"
    CONTRACT_OPERAND = "dependency_convex_contract_operand_cut"
    CONTRACT_RESULT = "dependency_convex_contract_result_cut"


@dataclass(frozen=True)
class DependencyConvexReductionPhaseSchedule:
    """Ordered launch groups with explicit single-generation cut storage."""

    source_component_id: int
    phases: tuple[DependencyConvexReductionPhase, ...]
    crossing_phase_edges: tuple[PhysicalPhaseEdge, ...]
    cut_value_ids: tuple[int, ...]
    event_capacity: int
    generation_count: int
    kind: DependencyConvexPhaseCut = DependencyConvexPhaseCut.REDUCTION_STATE

    @property
    def cut_value_edges(self) -> tuple[tuple[int, int, int], ...]:
        """Return each materialized value's producer and every later reader group."""
        return tuple(
            (value_id, producer.id, consumer.id)
            for producer in self.phases
            for consumer in self.phases[producer.id + 1 :]
            for value_id in sorted(
                set(producer.component.live_output_value_ids) & set(consumer.component.live_input_value_ids)
            )
        )


@dataclass(frozen=True)
class CanonicalKernelComposition:
    """A launch-ordered task cover with exact cross-component Event boundaries."""

    source_request_fingerprint: str
    scheduled_request: ExecutablePhysicalRequest
    components: tuple[CanonicalKernelComponent, ...]
    concrete_events: tuple[ConcreteKernelEventBoundary, ...]
    prelegalization_event_lowerings: tuple[ExecutableEventLowering, ...]
    legalized_event_lowerings: tuple[ExecutableEventLowering, ...]

    def __post_init__(self) -> None:
        if tuple(component.id for component in self.components) != tuple(range(len(self.components))):
            raise ValueError("kernel component ids must be contiguous in launch order")
        covered = tuple(task for component in self.components for task in component.task_ids)
        expected = tuple(range(len(self.scheduled_request.task_plan.tasks)))
        if tuple(sorted(covered)) != expected or len(covered) != len(set(covered)):
            raise ValueError("kernel components must cover every task exactly once")
        if self.scheduled_request.event_realizations.audit.physical:
            raise ValueError("launch-ordered kernel composition must realize every concrete Event")
        expected_events = tuple(plan.name for plan in self.scheduled_request.event_plans)
        if tuple(lowering.plan_name for lowering in self.prelegalization_event_lowerings) != expected_events:
            raise ValueError("prelegalization Event lowerings must cover every Event exactly once")
        if tuple(lowering.plan_name for lowering in self.legalized_event_lowerings) != expected_events:
            raise ValueError("legalized Event lowerings must cover every Event exactly once")
        if any(lowering.retained_operations for lowering in self.legalized_event_lowerings):
            raise ValueError("launch-ordered composition must discharge every imperative Event operation")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "source_request_fingerprint": self.source_request_fingerprint,
                "scheduled_request_fingerprint": self.scheduled_request.fingerprint,
                "components": tuple(
                    {
                        "id": component.id,
                        "schedule_index": component.schedule_index,
                        "schedule_fingerprint": component.schedule_fingerprint,
                        "task_ids": component.task_ids,
                        "operation_ordinals": component.operation_ordinals,
                        "operations": component.operations,
                        "live_input_value_ids": component.live_input_value_ids,
                        "live_output_value_ids": component.live_output_value_ids,
                    }
                    for component in self.components
                ),
                "concrete_events": tuple(
                    {
                        "event_id": event.event_id,
                        "producer_component": event.producer_component,
                        "consumer_component": event.consumer_component,
                        "relation_dimensions": event.relation_dimensions,
                        "capacity": event.capacity,
                        "generation_count": event.generation_count,
                    }
                    for event in self.concrete_events
                ),
                "prelegalization_event_lowerings": tuple(
                    lowering.canonical_record for lowering in self.prelegalization_event_lowerings
                ),
                "legalized_event_lowerings": tuple(
                    lowering.canonical_record for lowering in self.legalized_event_lowerings
                ),
            }
        )


@dataclass(frozen=True)
class CuteKernelComponentFragment:
    """Emitter output for one schedule before global value-slot binding."""

    component_id: int
    schedule_fingerprint: str
    slots_by_value_id: Mapping[int, CutePipelineSlot]
    launches: tuple[CuteKernelLaunch, ...]
    private_slots: tuple[CutePipelineSlot, ...] = ()
    physical_event_schedules: tuple[Mapping[str, Any], ...] = ()
    materialized_cut_value_ids: tuple[int, ...] = ()


def coalesce_dependency_convex_elementwise_components(
    composition: CanonicalKernelComposition,
    *,
    maximum_operations: int = 8,
    maximum_live_inputs: int = 4,
    maximum_live_outputs: int = 2,
) -> CanonicalKernelComposition:
    """Co-launch bounded contiguous Map/Relation components with exact Event proof.

    Contiguous ranges in a topological component order are dependency-convex:
    every intermediate on a path between two selected components must occur
    between them in that order. Admission additionally uses only the generic
    scalar/Relation operation records and the exact live value boundary.
    """
    if maximum_operations < 2 or maximum_live_inputs < 1 or maximum_live_outputs < 1:
        raise ValueError("elementwise co-launch bounds must be positive and admit at least two operations")
    components = composition.components
    output_ids = {_value_ordinal(identifier) for identifier in _program_record(composition.scheduled_request)["outputs"]}
    groups: list[tuple[CanonicalKernelComponent, ...]] = []
    cursor = 0
    while cursor < len(components):
        group = (components[cursor],)
        if _elementwise_component(components[cursor]):
            for candidate in components[cursor + 1 :]:
                extended = (*group, candidate)
                if not _elementwise_component(candidate) or not _co_launch_boundary_fits(
                    extended,
                    components,
                    output_ids=output_ids,
                    maximum_operations=maximum_operations,
                    maximum_live_inputs=maximum_live_inputs,
                    maximum_live_outputs=maximum_live_outputs,
                ):
                    break
                group = extended
        groups.append(group)
        cursor += len(group)
    return _coalesce_component_groups(
        composition,
        groups,
        family_name="generic_dependency_convex_elementwise_colaunch_v1",
        parameters={
            "maximum_operations": maximum_operations,
            "maximum_live_inputs": maximum_live_inputs,
            "maximum_live_outputs": maximum_live_outputs,
            "selection": "contiguous_topological_dependency_convex_map_relation_boundary",
        },
    )


def coalesce_private_map_fold_components(
    program: TensorProgram,
    composition: CanonicalKernelComposition,
) -> CanonicalKernelComposition:
    """Co-launch private same-domain Map chains with one non-scalar FAST Fold.

    Each intermediate has exactly one consumer and the Fold visits every Map
    coordinate once. No producer is recomputed for another output or through a
    broadcast. Typed scalar expressions and casts remain unchanged; FAST permits
    the physical emitter to contract FP32 multiply/add at the removed boundary.
    """
    request = composition.scheduled_request
    if program.canonical_record != _program_record(request):
        raise ValueError("Map/Fold co-launch requires the composition's canonical algebra")
    if (
        program.numerical_policy is not NumericalPolicy.FAST
        or request.numerical_policy is not PhysicalNumericalPolicy.FAST
    ):
        return composition
    components = composition.components
    groups: list[tuple[CanonicalKernelComponent, ...]] = []
    cursor = 0
    while cursor < len(components):
        group = (components[cursor],)
        candidate = group
        for following in components[cursor + 1 :]:
            if not all(
                isinstance(program.operations[ordinal], MapPrimitive)
                for member in candidate
                for ordinal in member.operation_ordinals
            ):
                break
            candidate = (*candidate, following)
            if _private_map_fold_group(program, candidate):
                group = candidate
                break
            if sum(len(member.operation_ordinals) for member in candidate) >= _PRIVATE_MAP_FOLD_MAXIMUM_OPERATIONS:
                break
        groups.append(group)
        cursor += len(group)
    return _coalesce_component_groups(
        composition,
        groups,
        family_name="generic_private_map_fold_colaunch_v1",
        internal_order_mechanism="fold_owner_evaluates_private_map_coordinates_before_accumulation",
        execution_owner_position=-1,
        parameters={
            "maximum_operations": _PRIVATE_MAP_FOLD_MAXIMUM_OPERATIONS,
            "maximum_live_inputs": _PRIVATE_MAP_FOLD_MAXIMUM_INPUTS,
            "maximum_live_outputs": _PRIVATE_MAP_FOLD_MAXIMUM_OUTPUTS,
            "selection": "private_same_domain_map_chain_fold",
            "finite_precision_effect": "fast_fp32_multiply_add_contraction_permitted_typed_casts_preserved",
        },
    )


def _private_map_fold_group(
    program: TensorProgram,
    group: tuple[CanonicalKernelComponent, ...],
) -> bool:
    ordinals = tuple(ordinal for component in group for ordinal in component.operation_ordinals)
    operations = tuple(program.operations[ordinal] for ordinal in ordinals)
    fold = operations[-1]
    maps = tuple(operation for operation in operations[:-1] if isinstance(operation, MapPrimitive))
    if (
        not isinstance(fold, FoldPrimitive)
        or not maps
        or len(maps) != len(operations) - 1
        or len(operations) > _PRIVATE_MAP_FOLD_MAXIMUM_OPERATIONS
    ):
        return False
    if (
        not fold.output.shape
        or fold.order is not EvaluationOrder.ORDER_FREE
        or any(
            dtype.value not in _PRIVATE_MAP_FOLD_DTYPES
            for dtype in (fold.accumulation_dtype, fold.input.dtype, fold.output.dtype)
        )
    ):
        return False
    if not all(operation.output.axes == fold.input.axes for operation in maps):
        return False
    produced = {value for operation in operations for value in primitive_outputs(operation)}
    inputs = {value for operation in operations for value in primitive_inputs(operation) if value not in produced}
    if len(inputs) > _PRIVATE_MAP_FOLD_MAXIMUM_INPUTS or any(
        not value.shape or value.dtype.value not in _GENERIC_COMPONENT_STORAGE_DTYPES for value in (*produced, *inputs)
    ):
        return False
    group_ordinals = set(ordinals)
    for operation in maps:
        value = operation.output
        if value in program.outputs:
            return False
        uses = tuple(
            ordinal
            for ordinal, consumer in enumerate(program.operations)
            for operand in primitive_inputs(consumer)
            if operand == value
        )
        if len(uses) != 1 or uses[0] not in group_ordinals:
            return False
    return True


def _coalesce_component_groups(
    composition: CanonicalKernelComposition,
    groups: list[tuple[CanonicalKernelComponent, ...]],
    *,
    family_name: str,
    parameters: Mapping[str, Any],
    internal_order_mechanism: str = "single_kernel_dependency_convex_program_order",
    execution_owner_position: int | None = None,
) -> CanonicalKernelComposition:
    if all(len(group) == 1 for group in groups):
        return composition

    request = composition.scheduled_request
    family_to_task = {task.family: task.id for task in request.task_plan.tasks}
    schedules: list[PhysicalSchedule] = []
    for group in groups:
        member_schedules = tuple(request.schedules[component.schedule_index] for component in group)
        if len(group) == 1:
            schedules.append(member_schedules[0])
            continue
        task_ids = tuple(task_id for schedule in member_schedules for task_id in schedule.task_ids)
        task_id_set = set(task_ids)
        external_orders = tuple(
            order
            for schedule in member_schedules
            for order in schedule.enforced_orders
            if not {order.source_task_id, order.target_task_id} <= task_id_set
        )
        internal_relations: dict[tuple[int, int], list[TaskRelation]] = {}
        for plan in request.event_plans:
            relation = plan.required_dependence.relation
            source = _task_id(relation.source, family_to_task)
            target = _task_id(relation.target, family_to_task)
            if source in task_id_set and target in task_id_set:
                internal_relations.setdefault((source, target), []).append(plan.scheduled_dependence)
        internal_orders = tuple(
            PhysicalTaskOrder(
                PhysicalOrderKind.PROGRAM_ORDER,
                source,
                target,
                TaskRelation(
                    relations[0].source,
                    relations[0].target,
                    tuple(sorted({pair for relation in relations for pair in relation.pairs})),
                ),
                {
                    "mechanism": internal_order_mechanism,
                    "source_components": tuple(component.id for component in group),
                    **(
                        {"execution_owner_task": task_ids[execution_owner_position]}
                        if execution_owner_position is not None
                        else {}
                    ),
                },
            )
            for (source, target), relations in internal_relations.items()
        )
        operation_ordinals = tuple(ordinal for component in group for ordinal in component.operation_ordinals)
        schedules.append(
            PhysicalSchedule(
                PhysicalScheduleFamily(
                    family_name,
                    request.target,
                    tuple(constraint for schedule in member_schedules for constraint in schedule.family.tasks),
                ),
                task_ids,
                {
                    "operation_ordinals": operation_ordinals,
                    **parameters,
                    **(
                        {
                            "execution_owner_task": task_ids[execution_owner_position],
                            "internal_order_mechanism": internal_order_mechanism,
                        }
                        if execution_owner_position is not None
                        else {}
                    ),
                },
                (*external_orders, *internal_orders),
            )
        )
    installed = install_physical_schedule_cover(
        request,
        tuple(schedules),
        source_request_fingerprint=composition.source_request_fingerprint,
    )
    return lower_canonical_kernel_composition(installed)


def _elementwise_component(component: CanonicalKernelComponent) -> bool:
    return bool(component.operations) and all(
        operation.get("kind") in {"map", "relation"} for operation in component.operations
    )


def _co_launch_boundary_fits(
    group: tuple[CanonicalKernelComponent, ...],
    components: tuple[CanonicalKernelComponent, ...],
    *,
    output_ids: set[int],
    maximum_operations: int,
    maximum_live_inputs: int,
    maximum_live_outputs: int,
) -> bool:
    if sum(len(component.operation_ordinals) for component in group) > maximum_operations:
        return False
    produced = {value_id for component in group for value_id in component.live_output_value_ids}
    live_inputs = {
        value_id for component in group for value_id in component.live_input_value_ids if value_id not in produced
    }
    group_ids = {component.id for component in group}
    outside_inputs = {
        value_id
        for component in components
        if component.id not in group_ids
        for value_id in component.live_input_value_ids
    }
    live_outputs = produced & (outside_inputs | output_ids)
    return (
        len(live_inputs) <= maximum_live_inputs
        and 1 <= len(live_outputs) <= maximum_live_outputs
        and all(
            result.get("axes")
            for component in group
            for operation in component.operations
            for result in operation.get("results", ())
            if _value_ordinal(result["id"]) in live_outputs
        )
    )


def dependency_convex_component_resource_record(
    program: TensorProgram,
    component: CanonicalKernelComponent,
) -> dict[str, Any]:
    """Return the exact algebra boundary and reduction depth of one component."""
    operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
    ordinals_by_value = tensor_program_value_ordinals(program)
    values_by_ordinal = {ordinal: value for value, ordinal in ordinals_by_value.items()}
    live_inputs = tuple(values_by_ordinal[ordinal] for ordinal in component.live_input_value_ids)
    live_outputs = tuple(values_by_ordinal[ordinal] for ordinal in component.live_output_value_ids)
    producer_by_value = {value: operation for operation in operations for value in primitive_outputs(operation)}

    def reduction_depth(value: Any) -> int:
        operation = producer_by_value.get(value)
        if operation is None:
            return 0
        return int(isinstance(operation, (ContractPrimitive, FoldPrimitive))) + max(
            (reduction_depth(operand) for operand in primitive_inputs(operation)),
            default=0,
        )

    def value_record(value: Any) -> dict[str, Any]:
        return {
            "value_ordinal": ordinals_by_value[value],
            "shape": value.shape,
            "dtype": value.dtype.value,
            "axes": tuple(axis.id for axis in value.axes),
            "layout": "canonical_contiguous",
        }

    return {
        "operation_count": len(operations),
        "operation_kinds": tuple(type(operation).__name__ for operation in operations),
        "live_inputs": tuple(value_record(value) for value in live_inputs),
        "live_outputs": tuple(value_record(value) for value in live_outputs),
        "maximum_reduction_depth": max((reduction_depth(value) for value in live_outputs), default=0),
    }


def dependency_convex_component_fits_resource_envelope(
    resource_record: Mapping[str, Any],
    *,
    maximum_operations: int,
    maximum_live_inputs: int,
    maximum_live_outputs: int,
    maximum_reduction_depth: int,
) -> bool:
    """Test generic component legality from its exact boundary and resource record."""
    live_inputs = tuple(resource_record["live_inputs"])
    live_outputs = tuple(resource_record["live_outputs"])

    def legal_value(value: Mapping[str, Any]) -> bool:
        shape = tuple(value["shape"])
        axes = tuple(value["axes"])
        return (
            bool(shape)
            and len(shape) == len(axes)
            and len(set(axes)) == len(axes)
            and all(extent > 0 for extent in shape)
            and value["dtype"] in _GENERIC_COMPONENT_STORAGE_DTYPES
            and value["layout"] == "canonical_contiguous"
        )

    return (
        2 <= resource_record["operation_count"] <= maximum_operations
        and len(live_inputs) <= maximum_live_inputs
        and 1 <= len(live_outputs) <= maximum_live_outputs
        and resource_record["maximum_reduction_depth"] <= maximum_reduction_depth
        and all(legal_value(value) for value in (*live_inputs, *live_outputs))
    )


def dependency_convex_reduction_phase_schedule(
    program: TensorProgram,
    composition: CanonicalKernelComposition,
    component: CanonicalKernelComponent,
) -> DependencyConvexReductionPhaseSchedule | None:
    """Split one depth-two component at its canonical reduction-state cut."""
    operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
    producer_by_value = {value: operation for operation in operations for value in primitive_outputs(operation)}

    def value_depth(value: Any) -> int:
        operation = producer_by_value.get(value)
        if operation is None:
            return 0
        return int(isinstance(operation, (ContractPrimitive, FoldPrimitive))) + max(
            (value_depth(operand) for operand in primitive_inputs(operation)),
            default=0,
        )

    depth_by_ordinal = {
        ordinal: max(value_depth(value) for value in primitive_outputs(operation))
        for ordinal, operation in zip(component.operation_ordinals, operations, strict=True)
    }
    if max(depth_by_ordinal.values(), default=0) != 2:
        return None
    operation_groups = (
        tuple(ordinal for ordinal in component.operation_ordinals if depth_by_ordinal[ordinal] < 2),
        tuple(ordinal for ordinal in component.operation_ordinals if depth_by_ordinal[ordinal] == 2),
    )
    if any(not group for group in operation_groups):
        return None

    return _dependency_convex_phase_schedule(
        program, composition, component, operation_groups, DependencyConvexPhaseCut.REDUCTION_STATE
    )


def dependency_convex_contract_phase_schedule(
    program: TensorProgram,
    composition: CanonicalKernelComposition,
    component: CanonicalKernelComponent,
) -> DependencyConvexReductionPhaseSchedule | None:
    """Propose source-ordered Contract cuts without splitting PhysicalPhases.

    Existing reduction groups survive except where a Contract is isolated.
    Intervening operations retain their order and typed storage boundaries. The
    compiler selects this proposal only after every isolated Contract admits.
    """
    operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
    if len(operations) < 2 or len(component.task_ids) != 1:
        return None
    if not any(isinstance(operation, ContractPrimitive) for operation in operations):
        return None
    reduction_schedule = dependency_convex_reduction_phase_schedule(program, composition, component)
    original_groups = (
        tuple(phase.component.operation_ordinals for phase in reduction_schedule.phases)
        if reduction_schedule is not None
        else (component.operation_ordinals,)
    )
    groups_list = []
    for original in original_groups:
        residue = []
        for ordinal in original:
            if isinstance(program.operations[ordinal], ContractPrimitive):
                if residue:
                    groups_list.append(tuple(residue))
                    residue = []
                groups_list.append((ordinal,))
            else:
                residue.append(ordinal)
        if residue:
            groups_list.append(tuple(residue))
    groups = tuple(groups_list)
    if len(groups) < 2 or groups == original_groups:
        return None
    kind = (
        DependencyConvexPhaseCut.CONTRACT_OPERAND
        if isinstance(operations[-1], ContractPrimitive)
        and all(isinstance(operation, MapPrimitive) for operation in operations[:-1])
        else DependencyConvexPhaseCut.CONTRACT_RESULT
    )
    if tuple(ordinal for group in groups for ordinal in group) != component.operation_ordinals:
        return None
    producing_group = {
        value: group_id
        for group_id, group in enumerate(groups)
        for ordinal in group
        for value in primitive_outputs(program.operations[ordinal])
    }
    if not any(
        producing_group.get(value, group_id) < group_id
        for group_id, group in enumerate(groups)
        for ordinal in group
        for value in primitive_inputs(program.operations[ordinal])
    ):
        return None
    task = composition.scheduled_request.task_plan.tasks[component.task_ids[0]]
    if task.structure is None:
        return None
    group_by_operation = {ordinal: group_id for group_id, group in enumerate(groups) for ordinal in group}
    if any(
        len({group_by_operation[ordinal] for ordinal in phase.operation_ordinals}) != 1
        for phase in task.structure.phases
    ):
        return None
    return _dependency_convex_phase_schedule(program, composition, component, groups, kind)


def _dependency_convex_phase_schedule(
    program: TensorProgram,
    composition: CanonicalKernelComposition,
    component: CanonicalKernelComponent,
    operation_groups: tuple[tuple[int, ...], ...],
    kind: DependencyConvexPhaseCut,
) -> DependencyConvexReductionPhaseSchedule:
    if len(component.task_ids) != 1:
        raise ValueError("a component phase split currently requires one canonical source task")
    task = composition.scheduled_request.task_plan.tasks[component.task_ids[0]]
    if task.structure is None:
        raise ValueError("a component phase split requires canonical PhysicalPhase provenance")
    phase_by_operation = {ordinal: phase for phase in task.structure.phases for ordinal in phase.operation_ordinals}
    if (
        len(operation_groups) < 2
        or any(not group for group in operation_groups)
        or len(tuple(ordinal for group in operation_groups for ordinal in group)) != len(component.operation_ordinals)
        or {ordinal for group in operation_groups for ordinal in group} != set(component.operation_ordinals)
    ):
        raise ValueError("component phase groups must partition the source operations")
    if set(phase_by_operation) != set(component.operation_ordinals):
        raise ValueError("canonical PhysicalPhases must cover the component operations exactly")
    group_by_operation = {ordinal: group_id for group_id, group in enumerate(operation_groups) for ordinal in group}
    if any(
        len({group_by_operation[ordinal] for ordinal in phase.operation_ordinals}) != 1
        for phase in task.structure.phases
    ):
        raise ValueError("one canonical PhysicalPhase cannot straddle a component cut")

    value_ordinals = tensor_program_value_ordinals(program)
    original_outputs = set(component.live_output_value_ids)
    operation_records = dict(zip(component.operation_ordinals, component.operations, strict=True))
    scheduled_phases = []
    for group_id, group in enumerate(operation_groups):
        group_operations = tuple(program.operations[ordinal] for ordinal in group)
        produced = {value for operation in group_operations for value in primitive_outputs(operation)}
        inputs = {
            value for operation in group_operations for value in primitive_inputs(operation) if value not in produced
        }
        outside_inputs = {
            value
            for ordinal in component.operation_ordinals
            if ordinal not in group
            for value in primitive_inputs(program.operations[ordinal])
        }
        outputs = {value for value in produced if value in outside_inputs or value_ordinals[value] in original_outputs}
        phase_component = replace(
            component,
            id=group_id,
            operation_ordinals=group,
            operations=tuple(operation_records[ordinal] for ordinal in group),
            live_input_value_ids=tuple(sorted(value_ordinals[value] for value in inputs)),
            live_output_value_ids=tuple(sorted(value_ordinals[value] for value in outputs)),
        )
        scheduled_phases.append(
            DependencyConvexReductionPhase(
                group_id,
                tuple(phase for phase in task.structure.phases if phase.operation_ordinals[0] in group),
                phase_component,
                dependency_convex_component_resource_record(program, phase_component),
            )
        )

    cut_value_ids = tuple(
        sorted(
            {
                value_id
                for producer in scheduled_phases
                for consumer in scheduled_phases[producer.id + 1 :]
                for value_id in set(producer.component.live_output_value_ids) & set(
                    consumer.component.live_input_value_ids
                )
            }
        )
    )
    crossing_edges = tuple(
        edge
        for edge in task.structure.phase_edges
        if group_by_operation[task.structure.phases[edge.source_phase].operation_ordinals[0]]
        != group_by_operation[task.structure.phases[edge.target_phase].operation_ordinals[0]]
    )
    if not cut_value_ids or not crossing_edges:
        raise ValueError("a component phase split requires value storage and a canonical phase edge")
    return DependencyConvexReductionPhaseSchedule(
        component.id,
        tuple(scheduled_phases),
        crossing_edges,
        cut_value_ids,
        event_capacity=1,
        generation_count=1,
        kind=kind,
    )


def compose_cute_pipeline_launch_recipe(
    composition: CanonicalKernelComposition,
    fragments: tuple[CuteKernelComponentFragment, ...],
) -> CutePipelineLaunchRecipe:
    """Bind component launches to canonical values and the exact Event cover."""
    fragment_by_component = {fragment.component_id: fragment for fragment in fragments}
    if len(fragment_by_component) != len(fragments) or set(fragment_by_component) != set(
        range(len(composition.components))
    ):
        raise ValueError("CuTe fragments must cover every kernel component exactly once")
    program = _program_record(composition.scheduled_request)
    values = _value_records(program)
    input_ids = tuple(_value_ordinal(value["id"]) for value in program["inputs"])
    output_ids = tuple(_value_ordinal(identifier) for identifier in program["outputs"])

    slot_specs: dict[int, CutePipelineSlot] = {}
    private_slots: list[CutePipelineSlot] = []
    private_reuse_groups: dict[tuple[int, int], int] = {}
    private_slot_names_by_component: dict[int, dict[str, str]] = {}
    for component in composition.components:
        fragment = fragment_by_component[component.id]
        if fragment.schedule_fingerprint != component.schedule_fingerprint:
            raise ValueError(f"CuTe fragment {component.id} does not match its selected schedule")
        required = set(component.live_input_value_ids) | set(component.live_output_value_ids)
        if not required <= set(fragment.slots_by_value_id):
            raise ValueError(
                f"CuTe fragment {component.id} omits live values {sorted(required - set(fragment.slots_by_value_id))}"
            )
        if not set(fragment.materialized_cut_value_ids) <= set(fragment.slots_by_value_id):
            raise ValueError(f"CuTe fragment {component.id} omits a materialized phase-cut binding")
        for value_id, slot in fragment.slots_by_value_id.items():
            previous = slot_specs.setdefault(value_id, slot)
            if (previous.element_type, previous.shape) != (slot.element_type, slot.shape):
                raise ValueError(f"CuTe fragments disagree on value {value_id} storage")
        local_private_names: dict[str, str] = {}
        for slot in fragment.private_slots:
            if (
                slot.kind is not CutePipelineSlotKind.SCRATCH
                or slot.input_index is not None
                or slot.output_index is not None
            ):
                raise ValueError("component-private CuTe slots must be scratch storage")
            if slot.name in local_private_names:
                raise ValueError(f"CuTe fragment {component.id} repeats private slot {slot.name!r}")
            global_name = f"component_{component.id}_private_{slot.name}"
            local_private_names[slot.name] = global_name
            reuse_group = None
            if slot.reuse_group is not None:
                identity = (component.id, slot.reuse_group)
                reuse_group = private_reuse_groups.setdefault(identity, len(private_reuse_groups))
            private_slots.append(replace(slot, name=global_name, reuse_group=reuse_group))
        private_slot_names_by_component[component.id] = local_private_names

    slots = (
        *tuple(
            CutePipelineSlot(
                name=f"value_{value_id}",
                element_type=slot_specs[value_id].element_type,
                shape=tuple(values[value_id]["shape"]),
                kind=_slot_kind(value_id, input_ids=input_ids, output_ids=output_ids),
                alignment=max(
                    slot.alignment
                    for fragment in fragments
                    for item, slot in fragment.slots_by_value_id.items()
                    if item == value_id
                ),
                input_index=input_ids.index(value_id) if value_id in input_ids else None,
                output_index=output_ids.index(value_id) if value_id in output_ids else None,
            )
            for value_id in sorted(slot_specs)
        ),
        *private_slots,
    )
    for value_id, slot in slot_specs.items():
        record = values[value_id]
        expected_shape = tuple(record["shape"])
        physical_shape = expected_shape or (1,)
        if slot.shape != physical_shape or slot.element_type is not _element_type(record["dtype"]):
            raise ValueError(f"CuTe slot for value {value_id} disagrees with canonical tensor storage")

    launches: list[CuteKernelLaunch] = []
    launch_range_by_component: dict[int, tuple[int, int]] = {}
    launch_indices_by_task: dict[int, list[int]] = {}
    for component in composition.components:
        fragment = fragment_by_component[component.id]
        begin = len(launches)
        covered_tasks: set[int] = set()
        for local_ordinal, launch in enumerate(fragment.launches):
            local_to_global = {
                **{slot.name: f"value_{value_id}" for value_id, slot in fragment.slots_by_value_id.items()},
                **private_slot_names_by_component[component.id],
            }
            source_tasks = launch.source_tasks
            if not source_tasks and len(fragment.launches) == 1:
                source_tasks = component.task_ids
            if not set(source_tasks) <= set(component.task_ids):
                raise ValueError(f"CuTe fragment {component.id} claims tasks outside its schedule")
            covered_tasks.update(source_tasks)
            launch_index = len(launches)
            for task_id in source_tasks:
                launch_indices_by_task.setdefault(task_id, []).append(launch_index)
            launches.append(
                replace(
                    launch,
                    name=f"component_{component.id}_launch_{local_ordinal}",
                    arguments=tuple(
                        bind_cute_pipeline_argument(argument, local_to_global) for argument in launch.arguments
                    ),
                    source_task_roles=(),
                    source_tasks=source_tasks,
                )
            )
        if covered_tasks != set(component.task_ids):
            raise ValueError(f"CuTe fragment {component.id} does not cover every scheduled task")
        launch_range_by_component[component.id] = (begin, len(launches) - 1)

    family_to_task = {task.family: task.id for task in composition.scheduled_request.task_plan.tasks}
    task_to_component = {task_id: component.id for component in composition.components for task_id in component.task_ids}
    event_realizations = tuple(
        _event_realization(
            event_id,
            plan,
            composition.scheduled_request.event_realizations.audit.entries[event_id].kind,
            family_to_task=family_to_task,
            task_to_component=task_to_component,
            launch_range_by_component=launch_range_by_component,
            launch_range_by_task={
                task_id: (min(indices), max(indices)) for task_id, indices in launch_indices_by_task.items()
            },
        )
        for event_id, plan in enumerate(composition.scheduled_request.event_plans)
    )
    host_arguments = tuple(
        f"value_{value_id}"
        for value_id in (
            *input_ids,
            *output_ids,
            *(value for value in sorted(slot_specs) if value not in input_ids + output_ids),
        )
    )
    host_arguments = (*host_arguments, *(slot.name for slot in private_slots))
    recipe = CutePipelineLaunchRecipe(
        schema_version=2,
        slots=slots,
        launches=tuple(launches),
        event_realizations=event_realizations,
        host_call_arguments=host_arguments,
    )
    launch_schedule = launch_dependency_event_schedule(recipe.launches, slots=recipe.slots)
    verify_materialized_cut_launch_dependencies(
        recipe.launches,
        slots=recipe.slots,
        cut_slots=tuple(
            f"value_{value_id}" for fragment in fragments for value_id in fragment.materialized_cut_value_ids
        ),
        launch_schedule=launch_schedule,
    )
    return bind_physical_event_provenance(
        recipe,
        combine_physical_event_schedules(
            launch_schedule,
            *(schedule for fragment in fragments for schedule in fragment.physical_event_schedules),
        ),
    )


def install_composed_cute_artifact(
    device_artifact: CutePipelineArtifact,
    composition: CanonicalKernelComposition,
    fragments: tuple[CuteKernelComponentFragment, ...],
) -> CutePipelineArtifact:
    """Seal one jointly compiled device module over the whole request recipe."""
    recipe = compose_cute_pipeline_launch_recipe(composition, fragments)
    required_symbols = {launch.kernel_symbol for launch in recipe.launches}
    missing = required_symbols - set(device_artifact.kernel_names)
    if missing:
        raise ValueError(f"joint CuTe device module omits composed kernels {sorted(missing)}")
    compile_request = composed_cute_compile_request(composition, recipe)
    return replace(device_artifact, request=compile_request, launch_recipe=recipe)


def composed_cute_compile_request(
    composition: CanonicalKernelComposition,
    recipe: CutePipelineLaunchRecipe,
) -> CutePipelineCompileRequest:
    """Bind whole-request algebra and schedule identity to a composed recipe."""
    program = _program_record(composition.scheduled_request)
    return CutePipelineCompileRequest(
        schema_version=1,
        target=composition.scheduled_request.target.value,
        numerical_policy=composition.scheduled_request.numerical_policy.value,
        physical_template="canonical_schedule_component_pipeline_v1",
        tensors=tuple(slot.canonical_record for slot in recipe.slots),
        scalar_programs={
            f"operation_{ordinal}": operation["expression"]
            for ordinal, operation in enumerate(program["operations"])
            if operation.get("kind") == "map"
        },
        index_relations=tuple(
            operation
            for operation in program["operations"]
            if operation.get("kind") in {"contract", "relation"}
            or (operation.get("kind") == "map" and operation.get("predicate") is not None)
        ),
        contraction_dimensions={},
        schedule={
            "composition_fingerprint": composition.fingerprint,
            "physical_schedules": tuple(
                schedule.canonical_record for schedule in composition.scheduled_request.schedules
            ),
            "components": tuple(
                {
                    "id": component.id,
                    "schedule_fingerprint": component.schedule_fingerprint,
                    "task_ids": component.task_ids,
                }
                for component in composition.components
            ),
        },
        algebra_identity={
            "algebra_fingerprint": composition.scheduled_request.algebra_fingerprint,
            "task_plan_fingerprint": composition.scheduled_request.task_plan.fingerprint,
            "source_request_fingerprint": composition.source_request_fingerprint,
        },
    )


def lower_canonical_kernel_composition(request: ExecutablePhysicalRequest) -> CanonicalKernelComposition:
    """Derive launch order and exact Event realization from schedule/task records."""
    source_fingerprint = request.fingerprint
    prelegalization_event_lowerings = lower_executable_event_operations(request)
    schedule_order = _topological_schedule_order(request)
    component_by_schedule = {schedule_index: component for component, schedule_index in enumerate(schedule_order)}
    task_to_schedule = {
        task_id: schedule_index
        for schedule_index, schedule in enumerate(request.schedules)
        for task_id in schedule.task_ids
    }
    task_to_component = {task_id: component_by_schedule[index] for task_id, index in task_to_schedule.items()}
    family_to_task = {task.family: task.id for task in request.task_plan.tasks}

    concrete_events = tuple(
        ConcreteKernelEventBoundary(
            event_id=event_id,
            producer_component=task_to_component[_task_id(plan.required_dependence.relation.source, family_to_task)],
            consumer_component=task_to_component[_task_id(plan.required_dependence.relation.target, family_to_task)],
            relation_dimensions=len(plan.domain.axes),
            capacity=request.event_allocations[event_id].capacity,
            generation_count=request.event_allocations[event_id].generation_count,
        )
        for event_id, plan in enumerate(request.event_plans)
        if task_to_component[_task_id(plan.required_dependence.relation.source, family_to_task)]
        != task_to_component[_task_id(plan.required_dependence.relation.target, family_to_task)]
    )

    schedules = [_coalesce_orders(schedule) for schedule in request.schedules]
    cross_relations: dict[tuple[int, int], list[TaskRelation]] = {}
    for event in concrete_events:
        relation = request.event_plans[event.event_id].scheduled_dependence
        producer_task = _task_id(relation.source, family_to_task)
        consumer_task = _task_id(relation.target, family_to_task)
        cross_relations.setdefault((producer_task, consumer_task), []).append(relation)
    for (producer_task, consumer_task), relations in cross_relations.items():
        producer_schedule = task_to_schedule[producer_task]
        relation = TaskRelation(
            relations[0].source,
            relations[0].target,
            tuple(sorted({pair for item in relations for pair in item.pairs})),
        )
        if any(
            order.relation.source == relation.source
            and order.relation.target == relation.target
            and set(relation.pairs) <= set(order.relation.pairs)
            for schedule in schedules
            for order in schedule.enforced_orders
        ):
            continue
        schedules[producer_schedule] = replace(
            schedules[producer_schedule],
            enforced_orders=(
                *schedules[producer_schedule].enforced_orders,
                PhysicalTaskOrder(
                    PhysicalOrderKind.LAUNCH_ORDER,
                    producer_task,
                    consumer_task,
                    relation,
                    {
                        "mechanism": "single_stream_launch_order",
                        "producer_component": task_to_component[producer_task],
                        "consumer_component": task_to_component[consumer_task],
                    },
                ),
            ),
        )
    scheduled = install_physical_schedule_cover(request, tuple(schedules))
    operations = _program_record(scheduled).get("operations")
    if not isinstance(operations, (tuple, list)) or not all(isinstance(operation, dict) for operation in operations):
        raise ValueError("canonical algebra record does not contain mapping-valued operations")
    components = tuple(
        _component(
            scheduled,
            component_id=component_id,
            schedule_index=schedule_index,
            operations=operations,
        )
        for component_id, schedule_index in enumerate(schedule_order)
    )
    return CanonicalKernelComposition(
        source_fingerprint,
        scheduled,
        components,
        concrete_events,
        prelegalization_event_lowerings,
        lower_executable_event_operations(scheduled),
    )


def _topological_schedule_order(request: ExecutablePhysicalRequest) -> tuple[int, ...]:
    schedule_by_task = {
        task_id: schedule_index
        for schedule_index, schedule in enumerate(request.schedules)
        for task_id in schedule.task_ids
    }
    family_to_task = {task.family: task.id for task in request.task_plan.tasks}
    successors = {index: set() for index in range(len(request.schedules))}
    predecessors = {index: set() for index in range(len(request.schedules))}
    for dependence in request.task_plan.dependences:
        source = schedule_by_task[_task_id(dependence.relation.source, family_to_task)]
        target = schedule_by_task[_task_id(dependence.relation.target, family_to_task)]
        if source == target:
            continue
        successors[source].add(target)
        predecessors[target].add(source)
    for dependence in request.task_plan.dependence_programs:
        source = schedule_by_task[dependence.source_task_id]
        target = schedule_by_task[dependence.target_task_id]
        if source == target:
            continue
        successors[source].add(target)
        predecessors[target].add(source)
    ready = sorted(index for index, incoming in predecessors.items() if not incoming)
    ordered: list[int] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for target in sorted(successors[current]):
            predecessors[target].remove(current)
            if not predecessors[target]:
                ready.append(target)
                ready.sort()
    if len(ordered) != len(request.schedules):
        raise ValueError("physical schedule component graph contains a cycle")
    return tuple(ordered)


def _coalesce_orders(schedule: PhysicalSchedule) -> PhysicalSchedule:
    grouped: dict[tuple[PhysicalOrderKind, int, int], list[PhysicalTaskOrder]] = {}
    for order in schedule.enforced_orders:
        grouped.setdefault((order.kind, order.source_task_id, order.target_task_id), []).append(order)
    orders = []
    for (kind, source_task, target_task), members in grouped.items():
        relation = TaskRelation(
            members[0].relation.source,
            members[0].relation.target,
            tuple(sorted({pair for member in members for pair in member.relation.pairs})),
        )
        orders.append(
            PhysicalTaskOrder(
                kind,
                source_task,
                target_task,
                relation,
                {
                    "mechanism": "coalesced_schedule_order",
                    "source_order_count": len(members),
                },
            )
        )
    return replace(schedule, enforced_orders=tuple(orders))


def _component(
    request: ExecutablePhysicalRequest,
    *,
    component_id: int,
    schedule_index: int,
    operations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> CanonicalKernelComponent:
    schedule = request.schedules[schedule_index]
    tasks = tuple(request.task_plan.tasks[task_id] for task_id in schedule.task_ids)
    ordinals = tuple(sorted(ordinal for task in tasks for ordinal in task.operation_ordinals))
    produced = {
        access.value_id for task in tasks for access in task.accesses if access.kind is TaskValueAccessKind.LIVE_OUT
    }
    consumed = {
        access.value_id for task in tasks for access in task.accesses if access.kind is TaskValueAccessKind.LIVE_IN
    }
    consumed_outside = {
        access.value_id
        for task in request.task_plan.tasks
        if task.id not in schedule.task_ids
        for access in task.accesses
        if access.kind is TaskValueAccessKind.LIVE_IN
    }
    public_outputs = {_value_ordinal(output) for output in _program_record(request)["outputs"]}
    return CanonicalKernelComponent(
        id=component_id,
        schedule_index=schedule_index,
        schedule_fingerprint=schedule.fingerprint,
        task_ids=schedule.task_ids,
        operation_ordinals=ordinals,
        operations=tuple(dict(operations[ordinal]) for ordinal in ordinals),
        live_input_value_ids=tuple(sorted(consumed - produced)),
        live_output_value_ids=tuple(
            sorted((produced - consumed) | (produced & consumed_outside) | (produced & public_outputs))
        ),
    )


def _program_record(request: ExecutablePhysicalRequest) -> dict[str, Any]:
    record = request.algebra_record
    if record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        record = record.get("program")
    if not isinstance(record, dict) or record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("kernel composition requires a canonical TensorProgram record")
    return record


def _value_records(program: dict[str, Any]) -> dict[int, dict[str, Any]]:
    axes = program.get("axes")
    if not isinstance(axes, (tuple, list)):
        raise ValueError("canonical TensorProgram axes must be a sequence")
    extent_by_axis = {
        axis["id"]: axis["extent"]
        for axis in axes
        if isinstance(axis, dict) and isinstance(axis.get("id"), str) and type(axis.get("extent")) is int
    }
    definitions = [*program["inputs"]]
    for operation in program["operations"]:
        definitions.extend(operation["results"])
    records: dict[int, dict[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("canonical TensorProgram value definitions must be mappings")
        axis_ids = definition.get("axes")
        if not isinstance(axis_ids, (tuple, list)) or not all(axis in extent_by_axis for axis in axis_ids):
            raise ValueError("canonical TensorProgram value references unavailable axes")
        records[_value_ordinal(definition.get("id"))] = {
            "dtype": definition.get("dtype"),
            "shape": tuple(extent_by_axis[axis] for axis in axis_ids),
        }
    return records


def _value_ordinal(identifier: object) -> int:
    if not isinstance(identifier, str) or not identifier.startswith("v") or not identifier[1:].isdigit():
        raise ValueError("canonical TensorProgram values require dense vN identifiers")
    return int(identifier[1:])


def _element_type(dtype: object) -> CutePipelineElementType:
    element_types: dict[object, CutePipelineElementType] = {
        "bf16": CutePipelineElementType.BF16,
        "fp32": CutePipelineElementType.F32,
        "int32": CutePipelineElementType.I32,
        "bool": CutePipelineElementType.PRED,
    }
    try:
        return element_types[dtype]
    except KeyError as error:
        raise ValueError(f"CuTe composition does not support tensor storage dtype {dtype!r}") from error


def _slot_kind(
    value_id: int,
    *,
    input_ids: tuple[int, ...],
    output_ids: tuple[int, ...],
) -> CutePipelineSlotKind:
    if value_id in input_ids and value_id in output_ids:
        return CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT
    if value_id in input_ids:
        return CutePipelineSlotKind.PUBLIC_INPUT
    if value_id in output_ids:
        return CutePipelineSlotKind.PUBLIC_OUTPUT
    return CutePipelineSlotKind.SCRATCH


def bind_cute_pipeline_argument(argument: object, local_to_global: dict[str, str]):
    """Rebind value addresses in pointer, tensor-map, and structured arguments."""
    if isinstance(argument, CutePipelineStructuredArgument):
        return replace(
            argument,
            fields=tuple(
                (
                    replace(field, slot=local_to_global[field.slot])
                    if isinstance(field, CutePipelineStructuredSlotAddressField)
                    else field
                )
                for field in argument.fields
            ),
        )
    if isinstance(argument, (CutePipelineSlotArgument, CutePipelineTensorMapArgument)):
        try:
            slot = local_to_global[argument.slot]
        except KeyError as error:
            raise ValueError(f"CuTe component argument references unbound slot {argument.slot!r}") from error
        return replace(argument, slot=slot)
    return argument


def _event_realization(
    event_id: int,
    plan: Any,
    source_kind: EventRealizationKind,
    *,
    family_to_task: dict[TaskFamily, int],
    task_to_component: dict[int, int],
    launch_range_by_component: dict[int, tuple[int, int]],
    launch_range_by_task: dict[int, tuple[int, int]],
) -> CuteEventRealization:
    source_task = _task_id(plan.required_dependence.relation.source, family_to_task)
    target_task = _task_id(plan.required_dependence.relation.target, family_to_task)
    source_component = task_to_component[source_task]
    target_component = task_to_component[target_task]
    source_launches = launch_range_by_task.get(source_task, launch_range_by_component[source_component])
    target_launches = launch_range_by_task.get(target_task, launch_range_by_component[target_component])
    dimensions = len(plan.domain.axes)
    owner = CuteAffineMap(dimensions, (affine_constant(0), affine_constant(0), affine_constant(0)))
    if source_component != target_component:
        kind = CuteEventRealizationKind.STREAM_ORDER
        producer_launch = source_launches[1]
        consumer_launch = target_launches[0]
        producer_phase = consumer_phase = 0
    elif source_launches[0] != source_launches[1]:
        kind = CuteEventRealizationKind.STREAM_ORDER
        producer_launch = source_launches[0]
        consumer_launch = source_launches[1]
        producer_phase = consumer_phase = 0
    else:
        if source_kind not in {
            EventRealizationKind.ERASED_PROGRAM_ORDER,
            EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
        }:
            raise ValueError("one-launch Event refinement lacks a same-launch source proof")
        kind = CuteEventRealizationKind.ALGEBRAIC_REFINEMENT
        producer_launch = consumer_launch = source_launches[0]
        producer_phase, consumer_phase = 0, 1
    producer_time = CuteAffineMap(dimensions, (affine_constant(producer_phase),))
    consumer_time = CuteAffineMap(dimensions, (affine_constant(consumer_phase),))
    piece = CuteEventCoverPiece(
        None,
        CuteEventCoverEndpoint(producer_launch, owner, producer_time),
        CuteEventCoverEndpoint(consumer_launch, owner, consumer_time),
    )
    return CuteEventRealization(
        event_id,
        kind,
        (producer_launch, producer_phase, consumer_launch, consumer_phase),
        (piece,),
    )


def _task_id(domain: CoordinateDomain, family_to_task: dict[TaskFamily, int]) -> int:
    if not isinstance(domain, TaskFamily):
        raise TypeError("kernel composition requires task-family dependence endpoints")
    return family_to_task[domain]
