# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable reference lowering for row-statistic Fold/Map/Contract graphs.

The public helpers in this module are fixtures for two familiar mathematical
programs.  The lowering itself consumes only generic tensor algebra and emits
the shared tiled-task, EventTensor, bounded-buffer, and physical-request ABI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tile_lifetime.event_buffering import (
    BoundedBufferPlan,
    EventRealizationAudit,
    derive_bounded_buffer_plan,
    physical_event_realization,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    ImperativeEventOperation,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    lower_event_tensor_plan,
    phased_event_storage_binding,
)
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccess,
    TaskValueAccessKind,
    TiledTask,
    physical_phase_iteration_space,
)
from tile_lifetime.tensor_program import (
    FAST_FP32_SUM_ERROR_CONTRACT,
    ContractPrimitive,
    FoldPrimitive,
    IndexExpressionKind,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    RelationPrimitive,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    canonical_tensor_program_record,
    primitive_inputs,
    primitive_outputs,
    tensor_program_fingerprint,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class RowStatisticReferencePipeline:
    """Canonical forward algebra, tasks, Events, and physical request."""

    forward: TensorProgram
    forward_request: ExecutablePhysicalRequest
    forward_event_program: EventDataflowProgram
    forward_imperative_events: tuple[tuple[ImperativeEventOperation, ...], ...]


def compile_row_statistic_reference_pipeline(
    source: TensorProgram,
    *,
    row_tile: int = 32,
    feature_tile: int = 64,
    target: PhysicalTarget = PhysicalTarget.SM90,
) -> RowStatisticReferencePipeline:
    """Compile generic Fold-dependent Map/Contract algebra without semantic dispatch."""
    if row_tile <= 0 or feature_tile <= 0:
        raise ValueError("task tile extents must be positive")
    if source.numerical_policy is not NumericalPolicy.FAST:
        raise ValueError("the fused two-stage physical request requires the FAST numerical policy")
    forward_plan, forward_events = _fold_map_contract_plan(source, row_tile=row_tile, feature_tile=feature_tile)
    schedule = _fold_map_contract_schedule(forward_plan, target=target, row_tile=row_tile, feature_tile=feature_tile)
    event_allocations = _event_allocations(forward_events)
    realizations = EventRealizationAudit(
        tuple(
            physical_event_realization(
                plan,
                mechanism="generation_counted_wait_notify",
                reason="the two task families may execute concurrently and reuse bounded row-state storage",
            )
            for plan in forward_events.event_plans
        )
    )
    forward_request = ExecutablePhysicalRequest(
        algebra_record=canonical_tensor_program_record(source),
        algebra_fingerprint=tensor_program_fingerprint(source),
        task_plan=forward_plan,
        event_plans=forward_events.event_plans,
        event_allocations=event_allocations,
        target=target,
        numerical_policy=PhysicalNumericalPolicy.FAST,
        schedules=(schedule,),
        event_realizations=EventRealizationProof(realizations, ()),
        provenance={"algebra_origin": "executable_reference_builder"},
    )
    return RowStatisticReferencePipeline(
        forward=source,
        forward_request=forward_request,
        forward_event_program=forward_events,
        forward_imperative_events=_imperative_events(forward_events, event_allocations),
    )


def _fold_map_contract_plan(
    program: TensorProgram,
    *,
    row_tile: int,
    feature_tile: int,
) -> tuple[CanonicalTaskPlan, EventDataflowProgram]:
    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    contract = _unique_output_contract(program, producer_by_value)
    contract_ancestors = _operation_ancestors(contract, producer_by_value)
    folds = tuple(operation for operation in contract_ancestors if isinstance(operation, FoldPrimitive))
    if not folds:
        raise ValueError("the generic two-stage cut requires Fold-produced state reaching a Contract")
    boundary_folds = tuple(
        fold
        for fold in folds
        if not any(fold is not other and fold in _operation_ancestors(other, producer_by_value) for other in folds)
    )
    owner_axes = boundary_folds[0].output.axes
    if any(fold.output.axes != owner_axes for fold in boundary_folds[1:]):
        raise ValueError("Fold states reaching one Contract require compatible owner axes")
    producer_set = {
        operation for fold in boundary_folds for operation in (*_operation_ancestors(fold, producer_by_value), fold)
    }
    output_slice = {operation for output in program.outputs for operation in _value_ancestors(output, producer_by_value)}
    consumer_set = output_slice - producer_set
    if contract not in consumer_set:
        raise ValueError("the Fold/Contract cut did not retain the output Contract")
    producer_ordinals = tuple(index for index, operation in enumerate(program.operations) if operation in producer_set)
    consumer_ordinals = tuple(index for index, operation in enumerate(program.operations) if operation in consumer_set)
    producer_operations = tuple(program.operations[index] for index in producer_ordinals)
    consumer_operations = tuple(program.operations[index] for index in consumer_ordinals)
    crossing_values = _crossing_values_between(program, producer_set, consumer_set)

    tile_for_axis = _axis_tile_function(program, row_tile=row_tile, feature_tile=feature_tile)
    producer = _fused_task(
        task_id=0,
        family_name="task_0",
        operations=producer_operations,
        ordinals=producer_ordinals,
        owner_axes=owner_axes,
        reduction_axes=_ordered_unique_axes(
            axis
            for operation in producer_operations
            if isinstance(operation, FoldPrimitive)
            for axis in operation.reduction_axes
        ),
        contraction_axes=(),
        principal_kind=TaskOperationKind.FOLD,
        program=program,
        tile_for_axis=tile_for_axis,
        live_out_values=crossing_values,
        recompute_sources=(),
    )
    consumer = _fused_task(
        task_id=1,
        family_name="task_1",
        operations=consumer_operations,
        ordinals=consumer_ordinals,
        owner_axes=contract.output.axes,
        reduction_axes=(),
        contraction_axes=contract.reduction_axes,
        principal_kind=TaskOperationKind.CONTRACT,
        program=program,
        tile_for_axis=tile_for_axis,
        live_out_values=tuple(
            value
            for value in program.outputs
            if value in {output for operation in consumer_set for output in primitive_outputs(operation)}
        ),
        recompute_sources=crossing_values,
    )
    data_relation = _overlap_relation(
        producer.family,
        producer.structure.owner_axes,
        consumer.family,
        consumer.structure.owner_axes,
        target_to_source_axes=_contract_owner_axis_relations(contract, owner_axes),
    )
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    data_dependence = TaskDependence(data_relation, visibility)
    traversal_dependence = _consumer_traversal_dependence(consumer, contract.output.axes)
    initial_dependences = (data_dependence,) if traversal_dependence is None else (data_dependence, traversal_dependence)
    initial_events = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event_{index}",
            generation_policy=EventGenerationPolicy.PHASED,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        for index, dependence in enumerate(initial_dependences)
    )
    initial_program = EventDataflowProgram((producer.family, consumer.family), initial_dependences, initial_events)
    buffer = _row_state_buffer(
        initial_program,
        producer=producer.family,
        use=data_relation,
        capacity=min(2, len(producer.family.coordinates)),
        visibility=visibility,
    )
    dependences = (*initial_dependences, *buffer.reuse_dependences)
    events = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event_{index}",
            generation_policy=EventGenerationPolicy.PHASED,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        for index, dependence in enumerate(dependences)
    )
    task_plan = CanonicalTaskPlan((producer, consumer), dependences, (buffer,))
    event_program = EventDataflowProgram(task_plan.task_families, dependences, events)
    # Keep the exact crossing cut observable through task accesses. This also
    # proves that the uncentered graph carries its original input and Fold state.
    value_ordinals = tensor_program_value_ordinals(program)
    crossing_ids = {value_ordinals[value] for value in crossing_values}
    live_out_ids = {access.value_id for access in producer.accesses if access.kind is TaskValueAccessKind.LIVE_OUT}
    if crossing_ids != live_out_ids:
        raise ValueError("the fused producer task does not expose the complete algebraic cut")
    return task_plan, event_program


def _fused_task(
    *,
    task_id: int,
    family_name: str,
    operations: tuple[TensorPrimitive, ...],
    ordinals: tuple[int, ...],
    owner_axes: tuple[TensorAxis, ...],
    reduction_axes: tuple[TensorAxis, ...],
    contraction_axes: tuple[TensorAxis, ...],
    principal_kind: TaskOperationKind,
    program: TensorProgram,
    tile_for_axis,
    live_out_values: tuple[ProgramValue, ...],
    recompute_sources: tuple[ProgramValue, ...],
) -> TiledTask:
    phases = tuple(
        PhysicalPhase(
            index,
            _operation_capability(operation),
            (ordinal,),
            *physical_phase_iteration_space(operation),
        )
        for index, (ordinal, operation) in enumerate(zip(ordinals, operations, strict=True))
    )
    operation_index = {
        output.name: index for index, operation in enumerate(operations) for output in primitive_outputs(operation)
    }
    edges = {
        PhysicalPhaseEdge(operation_index[value.name], target, RelationAccessClass.IDENTITY)
        for target, operation in enumerate(operations)
        for value in primitive_inputs(operation)
        if value.name in operation_index
    }
    axis_tiles = tuple(TaskAxisTile(axis.id, axis.extent, tile_for_axis(axis)) for axis in owner_axes)
    family = TaskFamily(
        family_name,
        tuple(
            TaskAxis(axis.label or f"axis_{axis.id}", math.ceil(axis.extent / tile_for_axis(axis)))
            for axis in owner_axes
        ),
        placement="gpu",
    )
    value_ordinals = tensor_program_value_ordinals(program)
    task_outputs = {output.name for operation in operations for output in primitive_outputs(operation)}
    external_inputs = {
        value.name: value
        for operation in operations
        for value in primitive_inputs(operation)
        if value.name not in task_outputs
    }
    live_out_names = {value.name for value in live_out_values}
    recompute_source_names = {value.name for value in recompute_sources}
    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    boundary_accesses: dict[tuple[int, TaskValueAccessKind], TaskValueAccess] = {}
    for value in external_inputs.values():
        phase = min(index for index, operation in enumerate(operations) if value in primitive_inputs(operation))
        cut_sources = _cut_boundary_leaves(value, producer_by_value)
        if value not in cut_sources and {source.name for source in cut_sources} <= recompute_source_names:
            for source in cut_sources:
                key = (value_ordinals[source], TaskValueAccessKind.LIVE_IN)
                access = _value_access(value_ordinals[source], TaskValueAccessKind.LIVE_IN, phase, source)
                previous = boundary_accesses.get(key)
                if previous is None or phase < previous.phase:
                    boundary_accesses[key] = access
            boundary_accesses[(value_ordinals[value], TaskValueAccessKind.RECOMPUTED)] = _value_access(
                value_ordinals[value], TaskValueAccessKind.RECOMPUTED, phase, value
            )
        else:
            key = (value_ordinals[value], TaskValueAccessKind.LIVE_IN)
            boundary_accesses[key] = _value_access(value_ordinals[value], TaskValueAccessKind.LIVE_IN, phase, value)
    accesses = list(boundary_accesses.values())
    for phase, operation in enumerate(operations):
        for output in primitive_outputs(operation):
            kind = TaskValueAccessKind.LIVE_OUT if output.name in live_out_names else TaskValueAccessKind.TEMPORARY
            accesses.append(_value_access(value_ordinals[output], kind, phase, output))
    for value in external_inputs.values():
        if value.name in live_out_names:
            # A source operand used on both sides of the cut is a genuine
            # live-through task result, not a hidden reload in the consumer.
            accesses.append(_value_access(value_ordinals[value], TaskValueAccessKind.LIVE_OUT, len(phases) - 1, value))
    accesses.sort(key=lambda access: (access.phase_id, access.value_id, access.kind.value))
    structure = PhysicalTaskStructure(
        principal_kind=principal_kind,
        phases=phases,
        phase_edges=tuple(sorted(edges)),
        owner_axes=axis_tiles,
        streamed_axes=(),
        reduction_axes=tuple(TaskAxisTile(axis.id, axis.extent, axis.extent) for axis in reduction_axes),
        contraction_axes=tuple(TaskAxisTile(axis.id, axis.extent, min(axis.extent, 64)) for axis in contraction_axes),
        traversal=TaskTraversal.FORWARD,
        element_types=tuple(
            dict.fromkeys(
                value.dtype.value
                for operation in operations
                for value in (*primitive_inputs(operation), *primitive_outputs(operation))
            )
        ),
        layout_classes=("strided_tensor",),
        resources=TaskResourceEstimate(
            registers_per_thread=32 + 4 * len(phases),
            shared_memory_bytes=0,
            barrier_count=max(0, len(phases) - 1),
        ),
    )
    return TiledTask(
        task_id,
        family,
        ordinals,
        tuple(_operation_kind(operation) for operation in operations),
        structure,
        tuple(accesses),
    )


def _fold_map_contract_schedule(
    plan: CanonicalTaskPlan,
    *,
    target: PhysicalTarget,
    row_tile: int,
    feature_tile: int,
) -> PhysicalSchedule:
    maximum = TaskResourceEstimate(255, 228 * 1024, 32)
    relation_classes = (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE)
    element_types = ("fp32", "bf16", "f32")
    layouts = ("strided_tensor",)
    family = PhysicalScheduleFamily(
        name="fold_then_map_contract",
        target=target,
        tasks=(
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.FOLD,
                required_capabilities=(TaskCapability.SCALAR_MAP, TaskCapability.ASSOCIATIVE_FOLD),
                supported_capabilities=(
                    TaskCapability.SCALAR_MAP,
                    TaskCapability.ASSOCIATIVE_FOLD,
                    TaskCapability.INDEX_RELATION,
                ),
                required_edges=(
                    PhaseEdgeConstraint(
                        TaskCapability.SCALAR_MAP,
                        TaskCapability.ASSOCIATIVE_FOLD,
                        RelationAccessClass.IDENTITY,
                    ),
                ),
                owner_rank=1,
                streamed_rank=0,
                reduction_rank=1,
                contraction_rank=0,
                supported_relation_classes=relation_classes,
                supported_element_types=element_types,
                supported_layout_classes=layouts,
                maximum_resources=maximum,
            ),
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.CONTRACT,
                required_capabilities=(TaskCapability.SCALAR_MAP, TaskCapability.MATRIX_CONTRACT),
                supported_capabilities=(
                    TaskCapability.SCALAR_MAP,
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.INDEX_RELATION,
                ),
                required_edges=(
                    PhaseEdgeConstraint(
                        TaskCapability.SCALAR_MAP,
                        TaskCapability.MATRIX_CONTRACT,
                        RelationAccessClass.IDENTITY,
                    ),
                ),
                owner_rank=2,
                streamed_rank=0,
                reduction_rank=0,
                contraction_rank=1,
                supported_relation_classes=relation_classes,
                supported_element_types=element_types,
                supported_layout_classes=layouts,
                maximum_resources=maximum,
            ),
        ),
    )
    return PhysicalSchedule(
        family=family,
        task_ids=tuple(task.id for task in plan.tasks),
        parameters={
            "row_tile": row_tile,
            "feature_tile": feature_tile,
            "fold_accumulation": "fp32",
            "fold_error_contract": FAST_FP32_SUM_ERROR_CONTRACT.canonical_record,
            "buffer_capacity": plan.bounded_buffers[0].capacity,
        },
    )


def _unique_output_contract(
    program: TensorProgram,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> ContractPrimitive:
    output_slice = {operation for output in program.outputs for operation in _value_ancestors(output, producer_by_value)}
    contracts = tuple(
        operation
        for operation in program.operations
        if operation in output_slice and isinstance(operation, ContractPrimitive)
    )
    if len(contracts) != 1:
        raise ValueError(f"the generic two-stage cut requires one output-slice Contract, found {len(contracts)}")
    return contracts[0]


def _value_ancestors(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> frozenset[TensorPrimitive]:
    producer = producer_by_value.get(value)
    if producer is None:
        return frozenset()
    return frozenset(
        {
            producer,
            *(
                ancestor
                for operand in primitive_inputs(producer)
                for ancestor in _value_ancestors(operand, producer_by_value)
            ),
        }
    )


def _operation_ancestors(
    operation: TensorPrimitive,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> frozenset[TensorPrimitive]:
    return frozenset(
        ancestor for operand in primitive_inputs(operation) for ancestor in _value_ancestors(operand, producer_by_value)
    )


def _crossing_values_between(
    program: TensorProgram,
    producer_operations: set[TensorPrimitive],
    consumer_operations: set[TensorPrimitive],
) -> tuple[ProgramValue, ...]:
    producer_inputs = {value for operation in producer_operations for value in primitive_inputs(operation)}
    consumer_inputs = {value for operation in consumer_operations for value in primitive_inputs(operation)}
    produced = {output for operation in producer_operations for output in primitive_outputs(operation)}
    live_through_inputs = set(program.inputs) & producer_inputs
    available = set(program.inputs) | produced
    crossing = available & consumer_inputs & (produced | live_through_inputs)
    producer_by_value = {
        output: operation for operation in producer_operations for output in primitive_outputs(operation)
    }
    rewritten_crossing = {
        boundary_value for value in crossing for boundary_value in _cut_boundary_leaves(value, producer_by_value)
    }
    value_ordinals = tensor_program_value_ordinals(program)
    return tuple(sorted(rewritten_crossing, key=value_ordinals.__getitem__))


def _cut_boundary_leaves(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> frozenset[ProgramValue]:
    producer = producer_by_value.get(value)
    if isinstance(producer, MapPrimitive):
        return frozenset(
            leaf for operand in producer.inputs for leaf in _cut_boundary_leaves(operand, producer_by_value)
        )
    if isinstance(producer, RelationPrimitive):
        return _cut_boundary_leaves(_affine_view_source(producer, value), producer_by_value)
    return frozenset((value,))


def _affine_view_source(operation: RelationPrimitive, output: ProgramValue) -> ProgramValue:
    if operation.predicate is not None:
        raise ValueError("the generic Fold/Contract cut cannot recompute a predicated Relation")
    output_index = operation.outputs.index(output)
    if operation.resolved_result_expressions[output_index] is not None:
        raise ValueError("the generic Fold/Contract cut cannot recompute a value-producing Relation")
    source = operation.resolved_result_sources[output_index]
    if source is None:
        raise ValueError("the generic Fold/Contract cut requires an indexed Relation source")
    indexing = next((item for item in operation.input_indexing if item.operand == source), None)
    if indexing is None or indexing.predicate is not None:
        raise ValueError("the generic Fold/Contract cut requires an unpredicated affine Relation")
    for axis, expression in zip(source.axes, indexing.indices, strict=True):
        if expression.kind is IndexExpressionKind.AXIS and expression.axis == axis:
            continue
        if expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and axis.extent == 1:
            continue
        raise ValueError("the generic Fold/Contract cut supports only identity and singleton-broadcast Relations")
    return source


def _row_state_buffer(
    program: EventDataflowProgram,
    *,
    producer: TaskFamily,
    use: TaskRelation,
    capacity: int,
    visibility: MemoryVisibility,
) -> BoundedBufferPlan:
    coordinates = producer.coordinates
    ordinal = {coordinate: index for index, coordinate in enumerate(coordinates)}
    return derive_bounded_buffer_plan(
        name="fold_state",
        program=program,
        producer=producer,
        uses=(use,),
        capacity=capacity,
        allocation_owner_for={coordinate: () for coordinate in coordinates},
        slot_for={coordinate: ordinal[coordinate] % capacity for coordinate in coordinates},
        generation_for={coordinate: ordinal[coordinate] // capacity for coordinate in coordinates},
        visibility=visibility,
    )


def _event_allocations(program: EventDataflowProgram) -> tuple[EventTensorAllocation, ...]:
    allocations = []
    for plan in program.event_plans:
        event_count = len(plan.domain.coordinates)
        capacity = min(2, event_count)
        coordinate_ordinal = {coordinate: index for index, coordinate in enumerate(plan.domain.coordinates)}
        storage = phased_event_storage_binding(
            plan,
            slot=lambda coordinate, ordinals=coordinate_ordinal, modulus=capacity: ordinals[coordinate] % modulus,
            generation=lambda coordinate, ordinals=coordinate_ordinal, modulus=capacity: ordinals[coordinate] // modulus,
        )
        allocations.append(
            EventTensorAllocation(
                storage=storage,
                capacity=capacity,
                generation_count=math.ceil(event_count / capacity),
            )
        )
    return tuple(allocations)


def _imperative_events(
    program: EventDataflowProgram,
    allocations: tuple[EventTensorAllocation, ...],
) -> tuple[tuple[ImperativeEventOperation, ...], ...]:
    return tuple(
        lower_event_tensor_plan(
            plan,
            scheduling_mode=EventSchedulingMode.STATIC,
            storage_binding=allocation.storage,
        )
        for plan, allocation in zip(program.event_plans, allocations, strict=True)
    )


def _consumer_traversal_dependence(
    consumer: TiledTask,
    output_axes: tuple[TensorAxis, ...],
) -> TaskDependence | None:
    if len(output_axes) < 2 or len(consumer.family.axes) < 2 or consumer.family.axes[-1].extent <= 1:
        return None
    pairs = []
    for coordinate in consumer.family.coordinates:
        if coordinate[-1] + 1 < consumer.family.axes[-1].extent:
            pairs.append((coordinate, (*coordinate[:-1], coordinate[-1] + 1)))
    return TaskDependence(
        TaskRelation.from_pairs(consumer.family, consumer.family, tuple(pairs)),
        MemoryVisibility(EventMemoryScope.CTA),
    )


def _overlap_relation(
    source: TaskFamily,
    source_axes: tuple[TaskAxisTile, ...],
    target: TaskFamily,
    target_axes: tuple[TaskAxisTile, ...],
    *,
    value_axes: tuple[TensorAxis, ...] | None = None,
    target_to_source_axes: dict[int, int] | None = None,
) -> TaskRelation:
    source_position = {axis.axis_id: index for index, axis in enumerate(source_axes)}
    target_to_source_axes = target_to_source_axes or {}
    target_position = {
        target_to_source_axes.get(axis.axis_id, axis.axis_id): index for index, axis in enumerate(target_axes)
    }
    allowed = None if value_axes is None else {axis.id for axis in value_axes}
    common = set(source_position) & set(target_position)
    if allowed is not None:
        common &= allowed
    pairs = tuple(
        (source_coordinate, target_coordinate)
        for source_coordinate in source.coordinates
        for target_coordinate in target.coordinates
        if all(source_coordinate[source_position[axis]] == target_coordinate[target_position[axis]] for axis in common)
    )
    return TaskRelation.from_pairs(source, target, pairs)


def _contract_owner_axis_relations(
    contract: ContractPrimitive,
    owner_axes: tuple[TensorAxis, ...],
) -> dict[int, int]:
    relations = {}
    for owner_axis in owner_axes:
        if owner_axis in contract.output.axes:
            relations[owner_axis.id] = owner_axis.id
            continue
        matches = tuple(
            mapping.domain_axis
            for input_index in range(len(contract.inputs))
            for mapping in contract.index_maps_for_input(input_index)
            if mapping.operand_axis == owner_axis
        )
        if len(matches) != 1:
            raise ValueError("Contract output must have one affine relation for each Fold owner axis")
        relations[matches[0].id] = owner_axis.id
    return relations


def _value_access(value_id: int, kind: TaskValueAccessKind, phase: int, value: ProgramValue) -> TaskValueAccess:
    return TaskValueAccess(
        value_id=value_id,
        kind=kind,
        phase_id=phase,
        relation_class=RelationAccessClass.IDENTITY,
        relation_record={"kind": "identity", "axis_ids": tuple(axis.id for axis in value.axes)},
    )


def _axis_tile_function(program: TensorProgram, *, row_tile: int, feature_tile: int):
    output_axes = program.outputs[0].axes
    row_axis = output_axes[0] if output_axes else None
    feature_axis = output_axes[-1] if len(output_axes) > 1 else None

    def tile(axis: TensorAxis) -> int:
        if axis == row_axis:
            return min(axis.extent, row_tile)
        if axis == feature_axis:
            return min(axis.extent, feature_tile)
        return axis.extent

    return tile


def _operation_kind(operation: TensorPrimitive) -> TaskOperationKind:
    if isinstance(operation, MapPrimitive):
        return TaskOperationKind.MAP
    if isinstance(operation, FoldPrimitive):
        return TaskOperationKind.FOLD
    if isinstance(operation, ContractPrimitive):
        return TaskOperationKind.CONTRACT
    if isinstance(operation, RelationPrimitive):
        return TaskOperationKind.RELATION
    raise ValueError(f"the generic two-stage task template cannot schedule {type(operation).__name__}")


def _operation_capability(operation: TensorPrimitive) -> TaskCapability:
    if isinstance(operation, MapPrimitive):
        return TaskCapability.SCALAR_MAP
    if isinstance(operation, FoldPrimitive):
        return TaskCapability.ASSOCIATIVE_FOLD
    if isinstance(operation, ContractPrimitive):
        return TaskCapability.MATRIX_CONTRACT
    if isinstance(operation, RelationPrimitive):
        return TaskCapability.INDEX_RELATION
    raise ValueError(f"the generic two-stage task template cannot schedule {type(operation).__name__}")


def _ordered_unique_axes(axes) -> tuple[TensorAxis, ...]:
    return tuple(dict.fromkeys(axes))
