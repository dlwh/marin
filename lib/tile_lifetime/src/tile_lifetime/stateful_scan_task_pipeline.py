# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ownership, exact dependencies, and Events for generic chunked scans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise

from tile_lifetime.dense_flow_planner import physical_request_for_composed_tasks
from tile_lifetime.event_buffering import (
    EventRealizationKind,
    derive_bounded_buffer_plan,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    EventTensorPlan,
    ImperativeEventOperation,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    lower_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.layout_relation_canonicalization import canonicalize_contract_layout_relations
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    EventErasureWitness,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalOrderKind,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskOrder,
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
)
from tile_lifetime.stateful_scan_reference import NaturalAffineScanConfig, natural_affine_scan_program
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    ProgramValue,
    ScanDirection,
    ScanPrimitive,
    TensorProgram,
    canonical_operand_indexing_record,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class ChunkedScanTiling:
    """Workload-independent tile geometry for one ordered state operation."""

    batch_extent: int
    ordered_extent: int
    parallel_extent: int
    state_row_extent: int
    state_column_extent: int
    additional_contractions: tuple[tuple[int, int], ...]
    chunk_extent: int
    state_column_tile: int
    traversal: TaskTraversal = TaskTraversal.FORWARD
    batch_axis_id: int = 0
    ordered_axis_id: int = 1
    parallel_axis_id: int = 2
    state_row_axis_id: int = 3
    state_column_axis_id: int = 4

    def __post_init__(self) -> None:
        dimensions = (
            self.batch_extent,
            self.ordered_extent,
            self.parallel_extent,
            self.state_row_extent,
            self.state_column_extent,
            self.chunk_extent,
            self.state_column_tile,
        )
        if min(dimensions) <= 0:
            raise ValueError("chunked scan dimensions and tile extents must be positive")
        if self.traversal not in (TaskTraversal.FORWARD, TaskTraversal.REVERSE):
            raise ValueError("chunked scans require explicit forward or reverse traversal")
        axis_ids = tuple(
            axis_id
            for axis_id in (
                self.batch_axis_id,
                self.ordered_axis_id,
                self.parallel_axis_id,
                self.state_row_axis_id,
                self.state_column_axis_id,
                *(axis_id for axis_id, _ in self.additional_contractions),
            )
            if axis_id is not None
        )
        if min(axis_ids) < 0 or len(set(axis_ids)) != len(axis_ids):
            raise ValueError("chunked scan logical axis ids must be non-negative and distinct")
        if any(extent <= 0 for _, extent in self.additional_contractions):
            raise ValueError("additional scan contraction extents must be positive")

    @property
    def chunk_count(self) -> int:
        return (self.ordered_extent + self.chunk_extent - 1) // self.chunk_extent

    @property
    def state_column_tile_count(self) -> int:
        return (self.state_column_extent + self.state_column_tile - 1) // self.state_column_tile


@dataclass(frozen=True)
class ChunkedScanReferencePlan:
    """Canonical tasks, exact Events, and imperative synchronization requirements."""

    task_plan: CanonicalTaskPlan
    event_plans: tuple[EventTensorPlan, ...]
    imperative_events: tuple[tuple[ImperativeEventOperation, ...], ...]

    @property
    def event_program(self) -> EventDataflowProgram:
        return EventDataflowProgram(
            self.task_plan.task_families,
            self.task_plan.dependences,
            self.event_plans,
        )


@dataclass(frozen=True)
class ExecutableScanReferencePipeline:
    """Executable DSL boundaries from generic Scan algebra to a physical request."""

    algebra: TensorProgram
    plan: ChunkedScanReferencePlan
    physical_request: ExecutablePhysicalRequest


@dataclass(frozen=True)
class WholeProgramScanRefinement:
    """A whole-program task plan with every generic Scan expanded into chunks."""

    task_plan: CanonicalTaskPlan
    event_plans: tuple[EventTensorPlan, ...]
    imperative_events: tuple[tuple[ImperativeEventOperation, ...], ...]

    @property
    def event_program(self) -> EventDataflowProgram:
        return EventDataflowProgram(
            self.task_plan.task_families,
            self.task_plan.dependences,
            self.event_plans,
        )


def extract_scan_state_region(program: TensorProgram, operation_ordinal: int) -> TensorProgram:
    """Extract one Scan's carried-state path and canonicalize exact layout views.

    Terminal body operations used only by emitted results are removed by the
    ordinary body DCE performed during layout canonicalization. This is a graph
    slice over generic Scan bindings, not a recurrence-pattern recognizer.
    """
    operation = program.operations[operation_ordinal]
    if not isinstance(operation, ScanPrimitive):
        raise ValueError("the requested operation is not a Scan")
    state_outputs = tuple(binding.body_result for binding in operation.state)
    body_slice = TensorProgram(
        operation.body.inputs,
        operation.body.operations,
        state_outputs,
        operation.body.numerical_policy,
    )
    canonical_body = canonicalize_contract_layout_relations(body_slice).program
    scan = replace(operation, body=canonical_body, emitted_outputs=())
    return TensorProgram(
        primitive_inputs(scan),
        (scan,),
        tuple(binding.final for binding in scan.state),
        program.numerical_policy,
    )


def compile_natural_affine_scan_reference_pipeline(
    config: NaturalAffineScanConfig,
    *,
    chunk_extent: int,
    state_column_tile: int,
    target: PhysicalTarget = PhysicalTarget.SM90,
) -> ExecutableScanReferencePipeline:
    """Compile an ordinary JAX affine recurrence through the executable DSL."""
    algebra = natural_affine_scan_program(config)
    scan = algebra.operations[0]
    assert isinstance(scan, ScanPrimitive)
    plan = derive_chunked_scan_reference_plan(
        algebra,
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
    )
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=algebra.canonical_record,
        algebra_fingerprint=algebra.semantic_fingerprint,
        target=target,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={
            "source_boundary": "ordinary_jax",
            "automatic_differentiation": "jax",
            "source_operation": "lax.scan",
        },
    )
    return ExecutableScanReferencePipeline(algebra, plan, request)


def derive_chunked_scan_tiling(
    scan: ScanPrimitive,
    *,
    chunk_extent: int,
    state_column_tile: int,
) -> ChunkedScanTiling:
    """Derive the initial matrix-state tiling from generic Scan metadata."""
    if len(scan.state) != 1 or len(scan.state[0].initial.axes) != 4:
        raise ValueError("the initial chunk schedule requires one rank-four carried state")
    batch, parallel, state_row, state_column = scan.state[0].initial.axes
    contraction_axes = {
        axis
        for operation in scan.body.operations
        if isinstance(operation, ContractPrimitive)
        for axis in operation.reduction_axes
    }
    state_body_axis_ids = {axis.id for axis in scan.state[0].body_argument.axes}
    update_rank_axes = tuple(
        sorted(
            (axis for axis in contraction_axes if axis.id not in state_body_axis_ids),
            key=lambda axis: (axis.id, axis.extent),
        )
    )
    traversal = TaskTraversal.FORWARD if scan.direction is ScanDirection.FORWARD else TaskTraversal.REVERSE
    return ChunkedScanTiling(
        batch_extent=batch.extent,
        ordered_extent=scan.axis.extent,
        parallel_extent=parallel.extent,
        state_row_extent=state_row.extent,
        state_column_extent=state_column.extent,
        additional_contractions=tuple((axis.id, axis.extent) for axis in update_rank_axes),
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
        traversal=traversal,
        batch_axis_id=batch.id,
        ordered_axis_id=scan.axis.id,
        parallel_axis_id=parallel.id,
        state_row_axis_id=state_row.id,
        state_column_axis_id=state_column.id,
    )


def derive_chunked_scan_reference_plan(
    program: TensorProgram,
    *,
    chunk_extent: int,
    state_column_tile: int,
) -> ChunkedScanReferencePlan:
    """Derive ownership, tiled tasks, exact state edges, and Event Tensors."""
    scans = tuple(
        (ordinal, operation)
        for ordinal, operation in enumerate(program.operations)
        if isinstance(operation, ScanPrimitive)
    )
    if len(scans) != 1:
        raise ValueError("the initial chunk scheduler requires one generic Scan operation")
    operation_ordinal, scan = scans[0]
    tiling = derive_chunked_scan_tiling(
        scan,
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
    )
    chunk_family = TaskFamily(
        "ordered_state_chunks",
        (
            TaskAxis("batch", tiling.batch_extent),
            TaskAxis("parallel", tiling.parallel_extent),
            TaskAxis("chunk", tiling.chunk_count),
            TaskAxis("state_column_tile", tiling.state_column_tile_count),
        ),
        placement="device",
    )
    commit_family = TaskFamily(
        "state_commit",
        (
            TaskAxis("batch", tiling.batch_extent),
            TaskAxis("parallel", tiling.parallel_extent),
            TaskAxis("state_column_tile", tiling.state_column_tile_count),
        ),
        placement="device",
    )
    traversal = tiling.traversal
    chunk_task, final_state_value_id = _chunk_task(
        program,
        scan,
        operation_ordinal,
        tiling,
        chunk_family,
        traversal,
    )
    commit_task = _commit_task(
        operation_ordinal,
        final_state_value_id,
        tiling,
        commit_family,
        traversal,
    )

    carry_pairs = []
    for batch in range(tiling.batch_extent):
        for parallel in range(tiling.parallel_extent):
            for state_column_tile in range(tiling.state_column_tile_count):
                chunk_order = (
                    range(tiling.chunk_count)
                    if traversal is TaskTraversal.FORWARD
                    else range(tiling.chunk_count - 1, -1, -1)
                )
                ordered_chunks = tuple(chunk_order)
                carry_pairs.extend(
                    (
                        (batch, parallel, source, state_column_tile),
                        (batch, parallel, target, state_column_tile),
                    )
                    for source, target in pairwise(ordered_chunks)
                )
    carry_relation = TaskRelation.from_pairs(chunk_family, chunk_family, carry_pairs)
    terminal_chunk = tiling.chunk_count - 1 if traversal is TaskTraversal.FORWARD else 0
    commit_relation = TaskRelation.from_pairs(
        chunk_family,
        commit_family,
        tuple(
            (
                (batch, parallel, terminal_chunk, state_column_tile),
                (batch, parallel, state_column_tile),
            )
            for batch in range(tiling.batch_extent)
            for parallel in range(tiling.parallel_extent)
            for state_column_tile in range(tiling.state_column_tile_count)
        ),
    )
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    dependences = (
        TaskDependence(carry_relation, visibility),
        TaskDependence(commit_relation, visibility),
    )
    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=name,
            generation_policy=EventGenerationPolicy.PER_INVOCATION,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        for dependence, name in zip(dependences, ("state_carry", "state_commit"), strict=True)
    )
    event_program = EventDataflowProgram((chunk_family, commit_family), dependences, event_plans)
    verify_event_dataflow_program(event_program)

    coordinates = chunk_family.coordinates
    traversal_index = {
        coordinate: coordinate[2] if traversal is TaskTraversal.FORWARD else tiling.chunk_count - 1 - coordinate[2]
        for coordinate in coordinates
    }
    buffer_capacity = min(2, tiling.chunk_count)
    bounded_state = derive_bounded_buffer_plan(
        name="carried_state",
        program=event_program,
        producer=chunk_family,
        uses=(carry_relation, commit_relation),
        capacity=buffer_capacity,
        allocation_owner_for={coordinate: (coordinate[0], coordinate[1], coordinate[3]) for coordinate in coordinates},
        slot_for={coordinate: traversal_index[coordinate] % buffer_capacity for coordinate in coordinates},
        generation_for={coordinate: traversal_index[coordinate] // buffer_capacity for coordinate in coordinates},
        visibility=visibility,
    )
    task_plan = CanonicalTaskPlan((chunk_task, commit_task), dependences, (bounded_state,))
    imperative_events = tuple(
        lower_event_tensor_plan(plan, scheduling_mode=EventSchedulingMode.STATIC) for plan in event_plans
    )
    return ChunkedScanReferencePlan(task_plan, event_plans, imperative_events)


@dataclass(frozen=True)
class _TaskCoordinateAxes:
    axes: Mapping[int, tuple[int, int, int]]
    unique_axis_id: int | None = None


def refine_whole_program_scan_tasks(
    program: TensorProgram,
    coarse_plan: CanonicalTaskPlan,
    *,
    chunk_extent: int,
    state_column_tile: int,
) -> WholeProgramScanRefinement:
    """Replace singleton Scan tasks with exact chunk and terminal-commit tasks.

    The input plan supplies the grouping of all non-Scan algebra. Scan boundary
    edges are rederived from the values crossing those groups, while each
    recurrence contributes its directional carry edge and bounded state buffer.
    """
    scan_by_ordinal = {
        ordinal: operation
        for ordinal, operation in enumerate(program.operations)
        if isinstance(operation, ScanPrimitive)
    }
    if not scan_by_ordinal:
        raise ValueError("whole-program Scan refinement requires at least one generic Scan")
    task_for_operation = {ordinal: task for task in coarse_plan.tasks for ordinal in task.operation_ordinals}
    if set(task_for_operation) != set(range(len(program.operations))):
        raise ValueError("coarse task plan must cover every whole-program operation exactly once")
    for ordinal in scan_by_ordinal:
        task = task_for_operation[ordinal]
        if task.operation_ordinals != (ordinal,):
            raise ValueError("whole-program Scan refinement requires each Scan to be a singleton coarse task")

    tasks: list[TiledTask] = []
    operation_task: dict[int, int] = {}
    output_task: dict[ProgramValue, int] = {}
    coordinate_axes: dict[int, _TaskCoordinateAxes] = {}
    old_to_new: dict[int, int] = {}
    internal_dependences: list[tuple[str, TaskDependence]] = []
    scan_buffers = []

    for coarse_task in coarse_plan.tasks:
        ordinal = coarse_task.operation_ordinals[0] if len(coarse_task.operation_ordinals) == 1 else None
        scan = scan_by_ordinal.get(ordinal) if ordinal is not None else None
        if scan is None:
            task_id = len(tasks)
            task = replace(coarse_task, id=task_id)
            tasks.append(task)
            old_to_new[coarse_task.id] = task_id
            for operation_ordinal in task.operation_ordinals:
                operation_task[operation_ordinal] = task_id
                for output in primitive_outputs(program.operations[operation_ordinal]):
                    output_task[output] = task_id
            coordinate_axes[task_id] = _ordinary_task_coordinate_axes(task)
            continue

        assert ordinal is not None
        tiling = derive_chunked_scan_tiling(
            scan,
            chunk_extent=chunk_extent,
            state_column_tile=state_column_tile,
        )
        chunk_family = TaskFamily(
            f"scan.{ordinal}.chunks",
            (
                TaskAxis("batch", tiling.batch_extent),
                TaskAxis("parallel", tiling.parallel_extent),
                TaskAxis("chunk", tiling.chunk_count),
                TaskAxis("state_column_tile", tiling.state_column_tile_count),
            ),
            placement="device",
        )
        commit_family = TaskFamily(
            f"scan.{ordinal}.commit",
            (
                TaskAxis("batch", tiling.batch_extent),
                TaskAxis("parallel", tiling.parallel_extent),
                TaskAxis("state_column_tile", tiling.state_column_tile_count),
            ),
            placement="device",
        )
        chunk_id = len(tasks)
        chunk_task, final_state_value_id = _chunk_task(
            program,
            scan,
            ordinal,
            tiling,
            chunk_family,
            tiling.traversal,
        )
        chunk_task = replace(chunk_task, id=chunk_id)
        tasks.append(chunk_task)
        commit_id = len(tasks)
        commit_task = replace(
            _commit_task(ordinal, final_state_value_id, tiling, commit_family, tiling.traversal),
            id=commit_id,
        )
        tasks.append(commit_task)
        operation_task[ordinal] = chunk_id
        old_to_new[coarse_task.id] = chunk_id
        state = scan.state[0]
        output_task[state.final] = commit_id
        for binding in scan.emitted_outputs:
            output_task[binding.destination] = chunk_id
        coordinate_axes[chunk_id] = _scan_chunk_coordinate_axes(tiling)
        coordinate_axes[commit_id] = _scan_commit_coordinate_axes(tiling)

        carry_relation, commit_relation = _scan_internal_relations(tiling, chunk_family, commit_family)
        visibility = MemoryVisibility(EventMemoryScope.DEVICE)
        carry = TaskDependence(carry_relation, visibility)
        commit = TaskDependence(commit_relation, visibility)
        internal_dependences.extend(((f"scan.{ordinal}.state_carry", carry), (f"scan.{ordinal}.state_commit", commit)))
        scan_event_program = EventDataflowProgram(
            (chunk_family, commit_family),
            (carry, commit),
            tuple(
                derive_event_tensor_plan(
                    dependence,
                    name=name,
                    generation_policy=EventGenerationPolicy.PER_INVOCATION,
                    scheduling_mode=EventSchedulingMode.STATIC,
                )
                for name, dependence in internal_dependences[-2:]
            ),
        )
        traversal_index = {
            coordinate: (
                coordinate[2] if tiling.traversal is TaskTraversal.FORWARD else tiling.chunk_count - 1 - coordinate[2]
            )
            for coordinate in chunk_family.coordinates
        }
        capacity = min(2, tiling.chunk_count)
        scan_buffers.append(
            derive_bounded_buffer_plan(
                name=f"scan.{ordinal}.carried_state",
                program=scan_event_program,
                producer=chunk_family,
                uses=(carry_relation, commit_relation),
                capacity=capacity,
                allocation_owner_for={
                    coordinate: (coordinate[0], coordinate[1], coordinate[3]) for coordinate in chunk_family.coordinates
                },
                slot_for={coordinate: traversal_index[coordinate] % capacity for coordinate in chunk_family.coordinates},
                generation_for={
                    coordinate: traversal_index[coordinate] // capacity for coordinate in chunk_family.coordinates
                },
                visibility=visibility,
            )
        )

    scan_coarse_ids = {task_for_operation[ordinal].id for ordinal in scan_by_ordinal}
    preserved_dependences = tuple(
        dependence
        for dependence in coarse_plan.dependences
        if all(
            task.id not in scan_coarse_ids
            for task in coarse_plan.tasks
            if task.family in (dependence.relation.source, dependence.relation.target)
        )
    )
    external_pairs: dict[tuple[int, int], set[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(set)
    producer_operation = {
        output: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    for consumer_ordinal, operation in enumerate(program.operations):
        target_id = operation_task[consumer_ordinal]
        for value in primitive_inputs(operation):
            source_ordinal = producer_operation.get(value)
            if source_ordinal is None:
                continue
            source_coarse_id = task_for_operation[source_ordinal].id
            target_coarse_id = task_for_operation[consumer_ordinal].id
            if source_coarse_id not in scan_coarse_ids and target_coarse_id not in scan_coarse_ids:
                continue
            source_id = output_task.get(value, operation_task[source_ordinal])
            if source_id == target_id:
                continue
            external_pairs[(source_id, target_id)].update(
                _identity_value_task_pairs(
                    tasks[source_id],
                    tasks[target_id],
                    value,
                    coordinate_axes[source_id],
                    coordinate_axes[target_id],
                )
            )

    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    external_dependences = tuple(
        TaskDependence(
            TaskRelation.from_pairs(tasks[source].family, tasks[target].family, tuple(sorted(pairs))),
            visibility,
        )
        for (source, target), pairs in sorted(external_pairs.items())
    )
    dependences = (
        tuple(dependence for _, dependence in internal_dependences) + preserved_dependences + external_dependences
    )

    dependence_programs = []
    for dependence in coarse_plan.dependence_programs:
        if dependence.source_task_id in scan_coarse_ids or dependence.target_task_id in scan_coarse_ids:
            raise ValueError("runtime-parametric dependencies crossing a refined Scan are not yet supported")
        dependence_programs.append(
            replace(
                dependence,
                source_task_id=old_to_new[dependence.source_task_id],
                target_task_id=old_to_new[dependence.target_task_id],
            )
        )
    scan_families = {task_for_operation[ordinal].family for ordinal in scan_by_ordinal}
    if any(
        buffer.producer in scan_families
        or any(use.source in scan_families or use.target in scan_families for use in buffer.uses)
        for buffer in coarse_plan.bounded_buffers
    ):
        raise ValueError("coarse bounded buffers crossing a refined Scan are not yet supported")
    task_plan = CanonicalTaskPlan(
        tuple(tasks),
        dependences,
        (*coarse_plan.bounded_buffers, *scan_buffers),
        tuple(dependence_programs),
    )
    event_names = (
        tuple(name for name, _ in internal_dependences)
        + tuple(f"preserved.{index}" for index in range(len(preserved_dependences)))
        + tuple(f"edge.{index}" for index in range(len(external_dependences)))
    )
    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=name,
            generation_policy=EventGenerationPolicy.PER_INVOCATION,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        for name, dependence in zip(event_names, dependences, strict=True)
    )
    event_program = EventDataflowProgram(task_plan.task_families, dependences, event_plans)
    verify_event_dataflow_program(event_program)
    imperative_events = tuple(
        lower_event_tensor_plan(event, scheduling_mode=EventSchedulingMode.STATIC) for event in event_plans
    )
    return WholeProgramScanRefinement(task_plan, event_plans, imperative_events)


def _ordinary_task_coordinate_axes(task: TiledTask) -> _TaskCoordinateAxes:
    if len(task.family.axes) != len(task.structure.owner_axes):
        raise ValueError("non-Scan task coordinates must correspond to its owner axes")
    return _TaskCoordinateAxes(
        {
            axis.axis_id: (position, axis.tile_extent, axis.extent)
            for position, axis in enumerate(task.structure.owner_axes)
        }
    )


def _scan_chunk_coordinate_axes(tiling: ChunkedScanTiling) -> _TaskCoordinateAxes:
    return _TaskCoordinateAxes(
        {
            tiling.batch_axis_id: (0, 1, tiling.batch_extent),
            tiling.parallel_axis_id: (1, 1, tiling.parallel_extent),
            tiling.ordered_axis_id: (2, tiling.chunk_extent, tiling.ordered_extent),
            tiling.state_column_axis_id: (3, tiling.state_column_tile, tiling.state_column_extent),
        },
        unique_axis_id=tiling.state_column_axis_id,
    )


def _scan_commit_coordinate_axes(tiling: ChunkedScanTiling) -> _TaskCoordinateAxes:
    return _TaskCoordinateAxes(
        {
            tiling.batch_axis_id: (0, 1, tiling.batch_extent),
            tiling.parallel_axis_id: (1, 1, tiling.parallel_extent),
            tiling.state_column_axis_id: (2, tiling.state_column_tile, tiling.state_column_extent),
        }
    )


def _identity_value_task_pairs(
    source: TiledTask,
    target: TiledTask,
    value: ProgramValue,
    source_axes: _TaskCoordinateAxes,
    target_axes: _TaskCoordinateAxes,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    value_axis_ids = {axis.id for axis in value.axes}
    common = value_axis_ids & set(source_axes.axes) & set(target_axes.axes)

    def interval(coordinate: tuple[int, ...], description: tuple[int, int, int]) -> tuple[int, int]:
        position, tile, extent = description
        start = coordinate[position] * tile
        return start, min(start + tile, extent)

    pairs = []
    for source_coordinate in source.family.coordinates:
        if source_axes.unique_axis_id is not None and source_axes.unique_axis_id not in value_axis_ids:
            position = source_axes.axes[source_axes.unique_axis_id][0]
            if source_coordinate[position] != 0:
                continue
        for target_coordinate in target.family.coordinates:
            if all(
                (lambda left, right: left[0] < right[1] and right[0] < left[1])(
                    interval(source_coordinate, source_axes.axes[axis]),
                    interval(target_coordinate, target_axes.axes[axis]),
                )
                for axis in common
            ):
                pairs.append((source_coordinate, target_coordinate))
    return tuple(pairs)


def _scan_internal_relations(
    tiling: ChunkedScanTiling,
    chunk_family: TaskFamily,
    commit_family: TaskFamily,
) -> tuple[TaskRelation, TaskRelation]:
    carry_pairs = []
    for batch in range(tiling.batch_extent):
        for parallel in range(tiling.parallel_extent):
            for state_column_tile in range(tiling.state_column_tile_count):
                ordered_chunks = tuple(
                    range(tiling.chunk_count)
                    if tiling.traversal is TaskTraversal.FORWARD
                    else range(tiling.chunk_count - 1, -1, -1)
                )
                carry_pairs.extend(
                    (
                        (batch, parallel, source, state_column_tile),
                        (batch, parallel, target, state_column_tile),
                    )
                    for source, target in pairwise(ordered_chunks)
                )
    terminal_chunk = tiling.chunk_count - 1 if tiling.traversal is TaskTraversal.FORWARD else 0
    commit_pairs = tuple(
        (
            (batch, parallel, terminal_chunk, state_column_tile),
            (batch, parallel, state_column_tile),
        )
        for batch in range(tiling.batch_extent)
        for parallel in range(tiling.parallel_extent)
        for state_column_tile in range(tiling.state_column_tile_count)
    )
    return (
        TaskRelation.from_pairs(chunk_family, chunk_family, carry_pairs),
        TaskRelation.from_pairs(chunk_family, commit_family, commit_pairs),
    )


def attach_chunked_scan_physical_request(
    plan: ChunkedScanReferencePlan,
    *,
    algebra_record: dict[str, object],
    algebra_fingerprint: str,
    target: PhysicalTarget,
    numerical_policy: PhysicalNumericalPolicy,
    provenance: dict[str, object],
) -> ExecutablePhysicalRequest:
    """Attach a persistent-owner schedule and prove its ordered Events erase."""
    chunk_structure = plan.task_plan.tasks[0].structure
    state_row_extent = chunk_structure.contraction_axes[0].extent
    state_column_tile = chunk_structure.owner_axes[-1].tile_extent
    ordered_chunk_extent = chunk_structure.streamed_axes[0].tile_extent
    orders = tuple(
        PhysicalTaskOrder(
            PhysicalOrderKind.PROGRAM_ORDER,
            source_task_id,
            target_task_id,
            event.scheduled_dependence,
            {
                "kind": "persistent_owner_loop",
                "traversal": plan.task_plan.tasks[0].structure.traversal.value,
            },
        )
        for event, (source_task_id, target_task_id) in zip(
            plan.event_plans,
            ((0, 0), (0, 1)),
            strict=True,
        )
    )
    schedule = PhysicalSchedule(
        family=_chunked_scan_schedule_family(
            target,
            contraction_rank=len(plan.task_plan.tasks[0].structure.contraction_axes),
        ),
        task_ids=(0, 1),
        parameters={
            "logical_state_buffer_capacity": plan.task_plan.bounded_buffers[0].capacity,
            "physical_state_buffer_capacity": 1,
            "event_mechanism": "persistent_owner_program_order",
            "body_evaluation": "topological_per_owner_memoized",
            "ordered_chunk_extent": ordered_chunk_extent,
            "state_row_extent": state_row_extent,
            "state_column_tile": state_column_tile,
            "state_tile_elements": state_row_extent * state_column_tile,
            "resident_state_bytes": 4 * state_row_extent * state_column_tile,
        },
        enforced_orders=orders,
    )
    realizations = tuple(
        erased_event_realization(
            event,
            kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
            mechanism="persistent owner loop",
            reason="one GPU program owns a state-column tile and executes its chunks and terminal store in order",
            ordering=order.relation,
        )
        for event, order in zip(plan.event_plans, orders, strict=True)
    )
    audit = verify_event_realizations(plan.event_program, realizations)
    allocations = tuple(
        EventTensorAllocation(
            storage=EventStorageBinding(
                event.domain,
                tuple(range(len(event.domain.coordinates))),
                (0,) * len(event.domain.coordinates),
            ),
            capacity=len(event.domain.coordinates),
            generation_count=1,
        )
        for event in plan.event_plans
    )
    return ExecutablePhysicalRequest(
        algebra_record=algebra_record,
        algebra_fingerprint=algebra_fingerprint,
        task_plan=plan.task_plan,
        event_plans=plan.event_plans,
        event_allocations=allocations,
        target=target,
        numerical_policy=numerical_policy,
        schedules=(schedule,),
        event_realizations=EventRealizationProof(
            audit,
            tuple(EventErasureWitness(event.name, 0, index) for index, event in enumerate(plan.event_plans)),
        ),
        provenance=provenance,
    )


def attach_whole_program_scan_physical_request(
    program: TensorProgram,
    refinement: WholeProgramScanRefinement,
    *,
    target: PhysicalTarget,
    numerical_policy: PhysicalNumericalPolicy,
    provenance: Mapping[str, object],
) -> ExecutablePhysicalRequest:
    """Attach one persistent schedule component to each refined Scan pair."""
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=refinement.task_plan,
        event_program=refinement.event_program,
        target=target,
        numerical_policy=numerical_policy,
    )
    scan_ordinals = {
        ordinal for ordinal, operation in enumerate(program.operations) if isinstance(operation, ScanPrimitive)
    }
    paired_task_ids: set[int] = set()
    scan_schedules = []
    for ordinal in sorted(scan_ordinals):
        members = tuple(
            task
            for task in refinement.task_plan.tasks
            if task.operation_ordinals == (ordinal,) and task.structure.principal_kind is TaskOperationKind.SCAN
        )
        chunk = tuple(task for task in members if task.structure.streamed_axes)
        commit = tuple(task for task in members if not task.structure.streamed_axes)
        if len(chunk) != 1 or len(commit) != 1:
            raise ValueError("each refined Scan requires exactly one streamed chunk task and one commit task")
        chunk_task, commit_task = chunk[0], commit[0]
        paired_task_ids.update((chunk_task.id, commit_task.id))
        orders = tuple(
            PhysicalTaskOrder(
                PhysicalOrderKind.PROGRAM_ORDER,
                chunk_task.id if dependence.relation.source is chunk_task.family else commit_task.id,
                chunk_task.id if dependence.relation.target is chunk_task.family else commit_task.id,
                dependence.relation,
                {
                    "kind": "persistent_owner_loop",
                    "traversal": chunk_task.structure.traversal.value,
                },
            )
            for dependence in refinement.task_plan.dependences
            if dependence.relation.source in {chunk_task.family, commit_task.family}
            and dependence.relation.target in {chunk_task.family, commit_task.family}
        )
        buffer = next(
            item for item in refinement.task_plan.bounded_buffers if item.name == f"scan.{ordinal}.carried_state"
        )
        scan_schedules.append(
            PhysicalSchedule(
                _chunked_scan_schedule_family(
                    target,
                    contraction_rank=len(chunk_task.structure.contraction_axes),
                ),
                (chunk_task.id, commit_task.id),
                {
                    "logical_state_buffer_capacity": buffer.capacity,
                    "physical_state_buffer_capacity": 1,
                    "event_mechanism": "persistent_owner_program_order",
                    "body_evaluation": "topological_per_owner_memoized",
                    "ordered_chunk_extent": chunk_task.structure.streamed_axes[0].tile_extent,
                    "state_row_extent": chunk_task.structure.contraction_axes[0].extent,
                    "state_column_tile": chunk_task.structure.owner_axes[-1].tile_extent,
                    "state_tile_elements": (
                        chunk_task.structure.contraction_axes[0].extent * chunk_task.structure.owner_axes[-1].tile_extent
                    ),
                    "resident_state_bytes": (
                        4
                        * chunk_task.structure.contraction_axes[0].extent
                        * chunk_task.structure.owner_axes[-1].tile_extent
                    ),
                },
                orders,
            )
        )
    schedules = tuple(
        schedule
        for schedule in request.schedules
        if not any(task_id in paired_task_ids for task_id in schedule.task_ids)
    ) + tuple(scan_schedules)
    realizations = []
    witnesses = []
    for event in refinement.event_plans:
        matching_orders = tuple(
            (schedule_index, order_index, order)
            for schedule_index, schedule in enumerate(schedules)
            for order_index, order in enumerate(schedule.enforced_orders)
            if order.relation == event.required_dependence.relation
        )
        if len(matching_orders) == 1:
            schedule_index, order_index, order = matching_orders[0]
            realizations.append(
                erased_event_realization(
                    event,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="persistent owner loop",
                    reason="one GPU program executes Scan chunks and terminal state commit in traversal order",
                    ordering=order.relation,
                )
            )
            witnesses.append(EventErasureWitness(event.name, schedule_index, order_index))
            continue
        realizations.append(
            physical_event_realization(
                event,
                mechanism="bounded_counter",
                reason="the dependence crosses independently scheduled physical components",
            )
        )
    audit = verify_event_realizations(refinement.event_program, tuple(realizations))
    return replace(
        request,
        schedules=schedules,
        event_realizations=EventRealizationProof(audit, tuple(witnesses)),
        provenance=dict(provenance),
    )


def _chunk_task(
    program: TensorProgram,
    scan: ScanPrimitive,
    operation_ordinal: int,
    tiling: ChunkedScanTiling,
    family: TaskFamily,
    traversal: TaskTraversal,
) -> tuple[TiledTask, int]:
    chunk_domain = (
        tiling.batch_extent,
        tiling.parallel_extent,
        tiling.state_column_extent,
        tiling.ordered_extent,
    )
    phases = (
        PhysicalPhase(0, TaskCapability.ORDERED_STATE, (operation_ordinal,), chunk_domain, ()),
        PhysicalPhase(1, TaskCapability.ORDERED_STATE, (operation_ordinal,), chunk_domain, ()),
        PhysicalPhase(2, TaskCapability.MATRIX_CONTRACT, (operation_ordinal,), chunk_domain, (3,)),
    )
    structure = PhysicalTaskStructure(
        principal_kind=TaskOperationKind.SCAN,
        phases=phases,
        phase_edges=(
            PhysicalPhaseEdge(0, 1, RelationAccessClass.FINITE),
            PhysicalPhaseEdge(0, 2, RelationAccessClass.FINITE),
            PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
        ),
        owner_axes=(
            TaskAxisTile(tiling.batch_axis_id, tiling.batch_extent, 1),
            TaskAxisTile(tiling.parallel_axis_id, tiling.parallel_extent, 1),
            TaskAxisTile(tiling.state_column_axis_id, tiling.state_column_extent, tiling.state_column_tile),
        ),
        streamed_axes=(TaskAxisTile(tiling.ordered_axis_id, tiling.ordered_extent, tiling.chunk_extent),),
        reduction_axes=(),
        contraction_axes=(
            TaskAxisTile(tiling.state_row_axis_id, tiling.state_row_extent, tiling.state_row_extent),
            *(TaskAxisTile(axis_id, extent, extent) for axis_id, extent in tiling.additional_contractions),
        ),
        traversal=traversal,
        element_types=("bf16", "f32"),
        layout_classes=("logical_contiguous",),
        resources=TaskResourceEstimate(
            registers_per_thread=96,
            shared_memory_bytes=4 * tiling.state_row_extent * min(tiling.state_column_tile, tiling.state_column_extent),
            barrier_count=2,
        ),
    )
    ordinals = tensor_program_value_ordinals(program)
    accesses: list[TaskValueAccess] = []
    for binding in scan.invariants:
        uses = tuple(
            body_ordinal
            for body_ordinal, operation in enumerate(scan.body.operations)
            if binding.body_argument in primitive_inputs(operation)
        )
        phase = 2 if uses and min(uses) == len(scan.body.operations) - 1 else 0
        accesses.append(
            TaskValueAccess(
                ordinals[binding.source],
                TaskValueAccessKind.LIVE_IN,
                phase,
                RelationAccessClass.IDENTITY,
                {
                    "kind": "scan_invariant",
                    "source_axes": tuple(axis.id for axis in binding.source.axes),
                    "body_axes": tuple(axis.id for axis in binding.body_argument.axes),
                },
            )
        )
    for binding in scan.scanned_inputs:
        uses = tuple(
            body_ordinal
            for body_ordinal, operation in enumerate(scan.body.operations)
            if binding.body_argument in primitive_inputs(operation)
        )
        phase = 2 if uses and min(uses) == len(scan.body.operations) - 1 else 0
        accesses.append(
            TaskValueAccess(
                ordinals[binding.source],
                TaskValueAccessKind.LIVE_IN,
                phase,
                RelationAccessClass.AFFINE,
                canonical_operand_indexing_record(program, binding.indexing),
            )
        )
    state_binding = scan.state[0]
    final_state_value_id = ordinals[state_binding.final]
    accesses.extend(
        (
            TaskValueAccess(
                ordinals[state_binding.initial],
                TaskValueAccessKind.LIVE_IN,
                1,
                RelationAccessClass.IDENTITY,
                {"kind": "carried_state"},
            ),
            TaskValueAccess(
                final_state_value_id,
                TaskValueAccessKind.TEMPORARY,
                1,
                RelationAccessClass.IDENTITY,
                {"kind": "carried_state"},
            ),
        )
    )
    body_producer = {
        result: body_ordinal
        for body_ordinal, operation in enumerate(scan.body.operations)
        for result in primitive_outputs(operation)
    }
    for binding in scan.emitted_outputs:
        producer = body_producer.get(binding.body_result)
        phase = 2 if producer is not None and isinstance(scan.body.operations[producer], ContractPrimitive) else 1
        accesses.append(
            TaskValueAccess(
                ordinals[binding.destination],
                TaskValueAccessKind.LIVE_OUT,
                phase,
                RelationAccessClass.AFFINE,
                canonical_operand_indexing_record(program, binding.indexing),
            )
        )
    task = TiledTask(
        0,
        family,
        (operation_ordinal,),
        (TaskOperationKind.SCAN,),
        structure,
        tuple(accesses),
    )
    return task, final_state_value_id


def _commit_task(
    operation_ordinal: int,
    final_state_value_id: int,
    tiling: ChunkedScanTiling,
    family: TaskFamily,
    traversal: TaskTraversal,
) -> TiledTask:
    structure = PhysicalTaskStructure(
        principal_kind=TaskOperationKind.SCAN,
        phases=(
            PhysicalPhase(
                0,
                TaskCapability.ORDERED_STATE,
                (operation_ordinal,),
                (tiling.batch_extent, tiling.parallel_extent, tiling.state_column_extent),
                (),
            ),
        ),
        phase_edges=(),
        owner_axes=(
            TaskAxisTile(tiling.batch_axis_id, tiling.batch_extent, 1),
            TaskAxisTile(tiling.parallel_axis_id, tiling.parallel_extent, 1),
            TaskAxisTile(tiling.state_column_axis_id, tiling.state_column_extent, tiling.state_column_tile),
        ),
        streamed_axes=(),
        reduction_axes=(),
        contraction_axes=(),
        traversal=traversal,
        element_types=("f32",),
        layout_classes=("logical_contiguous",),
        resources=TaskResourceEstimate(16, 0, 0),
    )
    accesses = (
        TaskValueAccess(
            final_state_value_id,
            TaskValueAccessKind.LIVE_IN,
            0,
            RelationAccessClass.IDENTITY,
            {"kind": "terminal_state"},
        ),
        TaskValueAccess(
            final_state_value_id,
            TaskValueAccessKind.LIVE_OUT,
            0,
            RelationAccessClass.IDENTITY,
            {"kind": "terminal_state"},
        ),
    )
    return TiledTask(
        1,
        family,
        (operation_ordinal,),
        (TaskOperationKind.SCAN,),
        structure,
        accesses,
    )


def _chunked_scan_schedule_family(target: PhysicalTarget, *, contraction_rank: int) -> PhysicalScheduleFamily:
    maximum = TaskResourceEstimate(255, 228_000, 16)
    supported_relations = (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE, RelationAccessClass.FINITE)
    return PhysicalScheduleFamily(
        name="ordered_state_chunk_pipeline",
        target=target,
        tasks=(
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.SCAN,
                required_capabilities=(
                    TaskCapability.ORDERED_STATE,
                    TaskCapability.ORDERED_STATE,
                    TaskCapability.MATRIX_CONTRACT,
                ),
                supported_capabilities=(TaskCapability.ORDERED_STATE, TaskCapability.MATRIX_CONTRACT),
                required_edges=(
                    PhaseEdgeConstraint(
                        TaskCapability.ORDERED_STATE, TaskCapability.ORDERED_STATE, RelationAccessClass.FINITE
                    ),
                    PhaseEdgeConstraint(
                        TaskCapability.ORDERED_STATE, TaskCapability.MATRIX_CONTRACT, RelationAccessClass.FINITE
                    ),
                    PhaseEdgeConstraint(
                        TaskCapability.ORDERED_STATE, TaskCapability.MATRIX_CONTRACT, RelationAccessClass.IDENTITY
                    ),
                ),
                owner_rank=3,
                streamed_rank=1,
                reduction_rank=0,
                contraction_rank=contraction_rank,
                supported_relation_classes=supported_relations,
                supported_element_types=("bf16", "f32"),
                supported_layout_classes=("logical_contiguous",),
                maximum_resources=maximum,
            ),
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.SCAN,
                required_capabilities=(TaskCapability.ORDERED_STATE,),
                supported_capabilities=(TaskCapability.ORDERED_STATE,),
                required_edges=(),
                owner_rank=3,
                streamed_rank=0,
                reduction_rank=0,
                contraction_rank=0,
                supported_relation_classes=supported_relations,
                supported_element_types=("f32",),
                supported_layout_classes=("logical_contiguous",),
                maximum_resources=maximum,
            ),
        ),
    )
