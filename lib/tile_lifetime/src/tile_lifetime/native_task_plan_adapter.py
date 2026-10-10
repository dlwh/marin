# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical task/Event view of a structured native Shuttle request."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise, product
from math import prod
from typing import Any

from shuttle.ir import DType
from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineExpression,
    CuteAffineExpressionKind,
    CuteAffineMap,
    CuteIntegerSet,
)
from tile_lifetime.dense_flow_planner import (
    RuntimeRelationDependenceSite,
    audit_runtime_relation_dependences,
    materialize_runtime_relation_dependences,
    physical_request_for_composed_tasks,
    task_value_accesses,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventTensorPlan,
    IndexPair,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_plan_record,
    lower_event_tensor_plan,
)
from tile_lifetime.physical_artifact_compiler import (
    GpuPhysicalArtifactRequest,
    ShuttleEventRealizationRecord,
    ShuttleExactEventRequirement,
    ShuttleOperationRecord,
)
from tile_lifetime.physical_normalized_exp_contract import (
    scalar_expression_from_source_map,
    source_bound_schedule_tasks,
    source_bound_tasks,
    source_relation_semantics,
)
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    ExecutablePhysicalRequest,
    ParametricEventDerivation,
    ParametricEventRealization,
    ParametricEventTensorPlan,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalTarget,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccessKind,
    TiledTask,
    canonical_fingerprint,
    physical_phase_iteration_space,
)
from tile_lifetime.stateful_scan_task_pipeline import (
    WholeProgramScanRefinement,
    attach_whole_program_scan_physical_request,
    refine_whole_program_scan_tasks,
)
from tile_lifetime.task_plan_refinement import canonical_task_event_boundary_fingerprint
from tile_lifetime.tensor_program import (
    ContractOperandPrecision,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    ScanPrimitive,
    SelectionPrimitive,
    TensorProgram,
    index_axis,
    index_compare,
    index_constant,
    index_tensor_element,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_select,
    tensor_program_value_ordinals,
)
from tile_lifetime.tensor_program_rewrites import remove_dead_operations


@dataclass(frozen=True)
class NativeImplementationHelperRefinement:
    """Exact provenance and coordinate covers for one physical-only task."""

    helper_task_id: int
    operation_owners: tuple[tuple[int, int], ...]
    coordinate_covers: tuple[tuple[int, TaskRelation], ...]

    def __post_init__(self) -> None:
        if self.helper_task_id < 0 or not self.operation_owners or not self.coordinate_covers:
            raise ValueError("native helper refinement requires a helper, operations, and coordinate covers")
        operation_ordinals = tuple(ordinal for ordinal, _ in self.operation_owners)
        if min(operation_ordinals) < 0 or len(set(operation_ordinals)) != len(operation_ordinals):
            raise ValueError("native helper refinement operation ordinals must be unique and non-negative")
        owner_ids = {owner for _, owner in self.operation_owners}
        if self.helper_task_id in owner_ids or min(owner_ids) < 0:
            raise ValueError("native helper refinement must target distinct canonical owner tasks")
        if {owner for owner, _ in self.coordinate_covers} != owner_ids:
            raise ValueError("native helper refinement requires one coordinate cover per owner task")
        helper_families = {relation.source for _, relation in self.coordinate_covers}
        if len(helper_families) != 1:
            raise ValueError("native helper coordinate covers must share one helper task family")
        for _, relation in self.coordinate_covers:
            if {pair.source for pair in relation.pairs} != set(relation.source.coordinates):
                raise ValueError("native helper coordinate cover must cover every helper coordinate")
            if {pair.target for pair in relation.pairs} != set(relation.target.coordinates):
                raise ValueError("native helper coordinate cover must cover every owner coordinate")

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the source-name-free helper refinement proof."""
        return {
            "helper_task": self.helper_task_id,
            "operation_owners": tuple(
                {"operation": operation, "owner_task": owner} for operation, owner in self.operation_owners
            ),
            "coordinate_covers": tuple(
                {
                    "owner_task": owner,
                    "pairs": tuple({"helper": pair.source, "owner": pair.target} for pair in relation.pairs),
                }
                for owner, relation in self.coordinate_covers
            ),
        }


@dataclass(frozen=True)
class NativeCanonicalTaskEventPlan:
    """Source-derived canonical semantics retained by a native request."""

    algebra_fingerprint: str
    task_plan: CanonicalTaskPlan
    events: tuple[EventTensorPlan, ...]
    event_programs: tuple[ParametricEventTensorPlan, ...]
    physical_order_witnesses: tuple[ShuttleEventRealizationRecord, ...]
    implementation_helper_refinements: tuple[NativeImplementationHelperRefinement, ...] = ()
    runtime_relation_dependence_sites: tuple[RuntimeRelationDependenceSite, ...] = ()

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the lossless native task/Event conformance record."""
        return {
            "schema": "shuttle.native-canonical-task-event-plan.v2",
            "algebra_fingerprint": self.algebra_fingerprint,
            "task_plan": self.task_plan.canonical_record,
            "events": tuple(event_tensor_plan_record(event) for event in self.events),
            "event_programs": tuple(
                {
                    "dependence_program": event.dependence_program_index,
                    "derivation": event.derivation.value,
                    "capacity": event.capacity,
                    "slots": event.slots,
                    "generations": event.generations,
                    "generation_count": event.generation_count,
                    "memory_scope": event.memory_scope.value,
                    "generation_policy": event.generation_policy.value,
                    "scheduling_mode": event.scheduling_mode.value,
                    "realization": event.realization.value,
                }
                for event in self.event_programs
            ),
            "implementation_helper_refinements": tuple(
                refinement.canonical_record for refinement in self.implementation_helper_refinements
            ),
            "runtime_relation_dependence_sites": tuple(
                site.canonical_record for site in self.runtime_relation_dependence_sites
            ),
        }

    @property
    def fingerprint(self) -> str:
        """Return the identity of the complete native conformance record."""
        return canonical_fingerprint(self.canonical_record)

    @property
    def event_fingerprint(self) -> str:
        """Return the shared logical task/Event boundary identity."""
        return canonical_task_event_boundary_fingerprint(self.task_plan, self.events, self.event_programs)


_OPERATION_KINDS = {
    "shuttle.map": (TaskOperationKind.MAP, TaskCapability.SCALAR_MAP),
    "shuttle.fold": (TaskOperationKind.FOLD, TaskCapability.ASSOCIATIVE_FOLD),
    "shuttle.contract": (TaskOperationKind.CONTRACT, TaskCapability.MATRIX_CONTRACT),
    "shuttle.selection": (TaskOperationKind.SELECTION, TaskCapability.BOUNDED_SELECTION),
    "shuttle.scan": (TaskOperationKind.SCAN, TaskCapability.ORDERED_STATE),
    "shuttle.relation": (TaskOperationKind.RELATION, TaskCapability.INDEX_RELATION),
    "shuttle.transport": (TaskOperationKind.TRANSPORT, TaskCapability.DATA_TRANSPORT),
}


def _native_physical_phase(
    phase_id: int,
    phase,
    *,
    canonical_ordinal_by_native: dict[int, int],
    algebra_program: TensorProgram,
) -> PhysicalPhase:
    ordinal = canonical_ordinal_by_native.get(phase.source_operation)
    if ordinal is None:
        raise ValueError("native composition phase does not bind canonical algebra")
    operation = algebra_program.operations[ordinal]
    kind = _tensor_operation_kind(operation)
    capability = _OPERATION_KINDS[f"shuttle.{kind.value}"][1]
    expected_kind = f"#shuttle.materialization_task_kind<{kind.value}>"
    if phase.kind != expected_kind:
        raise ValueError("native composition phase kind disagrees with canonical algebra")
    expected_domain, expected_reductions = physical_phase_iteration_space(operation)
    if phase.domain_shape != expected_domain or phase.reduction_dimensions != expected_reductions:
        raise ValueError("native composition phase domain disagrees with canonical algebra")
    return PhysicalPhase(
        phase_id,
        capability,
        (ordinal,),
        phase.domain_shape,
        phase.reduction_dimensions,
    )


def _native_physical_phase_edges(
    source_task,
    *,
    phases: tuple[PhysicalPhase, ...],
    algebra_program: TensorProgram,
) -> tuple[PhysicalPhaseEdge, ...]:
    edges: list[PhysicalPhaseEdge] = []
    for edge in source_task.phase_edges:
        if edge.source_phase >= len(phases) or edge.target_phase >= len(phases):
            raise ValueError("native composition edge references an unavailable canonical phase")
        source_operation = algebra_program.operations[phases[edge.source_phase].operation_ordinals[0]]
        target_operation = algebra_program.operations[phases[edge.target_phase].operation_ordinals[0]]
        source_outputs = primitive_outputs(source_operation)
        target_inputs = primitive_inputs(target_operation)
        if edge.source_result >= len(source_outputs) or edge.target_operand >= len(target_inputs):
            raise ValueError("native composition edge references an unavailable algebra value")
        source_value = source_outputs[edge.source_result]
        if target_inputs[edge.target_operand] != source_value:
            raise ValueError("native composition edge disagrees with canonical SSA use-def")
        if edge.access_map.dimensions != len(phases[edge.target_phase].domain_shape):
            raise ValueError("native composition edge access rank disagrees with its target phase")
        if len(edge.access_map.results) != len(source_value.axes):
            raise ValueError("native composition edge access rank disagrees with its source value")
        edges.append(
            PhysicalPhaseEdge(
                edge.source_phase,
                edge.target_phase,
                _affine_relation_class(edge.access_map),
            )
        )
    if len(edges) != len(set(edges)):
        raise ValueError("native composition contains duplicate canonical phase edges")
    return tuple(sorted(edges))


def _affine_relation_class(access_map: CuteAffineMap) -> RelationAccessClass:
    identity = access_map.dimensions == len(access_map.results) and all(
        expression.kind is CuteAffineExpressionKind.DIMENSION
        and expression.payload == position
        and not expression.operands
        for position, expression in enumerate(access_map.results)
    )
    return RelationAccessClass.IDENTITY if identity else RelationAccessClass.AFFINE


def _verify_native_result_bindings(
    source_task,
    *,
    phases: tuple[PhysicalPhase, ...],
    accesses,
    algebra_program: TensorProgram,
) -> None:
    value_ordinals = tensor_program_value_ordinals(algebra_program)
    expected = set()
    for binding in source_task.result_bindings:
        if binding.phase >= len(phases):
            raise ValueError("native result binding references an unavailable canonical phase")
        operation = algebra_program.operations[phases[binding.phase].operation_ordinals[0]]
        outputs = primitive_outputs(operation)
        if binding.result >= len(outputs):
            raise ValueError("native result binding references an unavailable algebra result")
        expected.add((binding.phase, value_ordinals[outputs[binding.result]]))
    actual = {(access.phase_id, access.value_id) for access in accesses if access.kind is TaskValueAccessKind.LIVE_OUT}
    if expected != actual or len(expected) != len(source_task.result_bindings):
        raise ValueError("native result bindings disagree with the canonical live value boundary")


def canonical_algebra_fingerprint_from_native_request(
    request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
) -> str:
    """Prove native source SSA/use-def semantics against canonical algebra."""
    algebra_program = remove_dead_operations(algebra_program)
    operations, _ = _source_operations(request.region)
    _verify_tensor_program_binding(request.region, operations, algebra_program)
    return algebra_program.semantic_fingerprint


def canonical_task_event_plan_from_native_request(
    request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
) -> NativeCanonicalTaskEventPlan:
    """Adapt one typed native materialization/schedule/Event request.

    The logical quotient retains exact Events even when a native physical-order
    proof erases their imperative realization. Physical launch/phase witnesses
    are returned separately as proof evidence rather than folded into algebra.
    """
    algebra_program = remove_dead_operations(algebra_program)
    source_tasks = source_bound_tasks(request)
    schedules = source_bound_schedule_tasks(request)
    operations, operation_by_source = _source_operations(request.region)
    inline_literal_ordinals = {
        ordinal for ordinal, operation in enumerate(operations) if _inlineable_nullary_literal(operation)
    }
    operation_groups = _verify_tensor_program_binding(request.region, operations, algebra_program)
    materialization_operations = tuple(
        operation
        for operation in request.materialization.body().operations
        if operation.name == "shuttle.materialization_task"
    )
    if len(materialization_operations) != len(source_tasks):
        raise ValueError("native task adapter requires one structured operation per materialization task")
    operation_task_owners = _native_operation_task_owners(
        operations,
        operation_by_source,
        materialization_operations,
        inline_literal_ordinals,
    )
    explicit_task_owners: dict[int, set[int]] = {}
    for task_ordinal, materialization in enumerate(materialization_operations):
        for source_key in _materialization_source_keys(materialization):
            operation_entry = operation_by_source.get(source_key)
            if operation_entry is not None:
                explicit_task_owners.setdefault(operation_entry[0], set()).add(task_ordinal)
    canonical_owner_by_ordinal: dict[int, int] = {}
    for canonical_ordinal, group in enumerate(operation_groups):
        terminal_ordinal = next(
            operations.index(operation)
            for operation in reversed(group)
            if operations.index(operation) not in inline_literal_ordinals
        )
        owners = explicit_task_owners.get(terminal_ordinal, set(operation_task_owners[terminal_ordinal]))
        canonical_owner_by_ordinal[canonical_ordinal] = min(owners)
    canonical_ordinal_by_native = {
        operations.index(operation): canonical_ordinal
        for canonical_ordinal, group in enumerate(operation_groups)
        for operation in group
    }
    task_components = tuple((task_ordinal,) for task_ordinal in range(len(source_tasks)))

    tiled_tasks: list[TiledTask] = []
    native_task_id_by_source_ordinal: dict[int, int] = {}
    coordinate_cover_by_source_ordinal: dict[int, dict[tuple[int, ...], tuple[tuple[int, ...], ...]]] = {}
    for component in task_components:
        representative = max(component)
        source_task = source_tasks[representative]
        schedule = schedules[representative]
        materialization_operation = materialization_operations[representative]
        explicit_contraction_dimensions = set(source_task.producer_contraction_dimensions) | set(
            source_task.consumer_contraction_dimensions
        )
        compact_tile = len(schedule.tile_shape) == 1 and len(source_task.domain_shape) != 1
        if not compact_tile and len(schedule.tile_shape) != len(source_task.domain_shape):
            raise ValueError("native task adapter requires full-rank or flattened task tiles")
        component_operations = {
            operation_ordinal for operation_ordinal, owners in operation_task_owners.items() if owners & set(component)
        }
        canonical_ordinals = tuple(
            ordinal
            for ordinal, group in enumerate(operation_groups)
            if any(operations.index(operation) in component_operations for operation in group)
        )
        task_operations = tuple(
            (
                ordinal,
                _tensor_operation_kind(algebra_program.operations[ordinal]),
                _native_group_capability(operation_groups[ordinal], kind),
            )
            for ordinal in canonical_ordinals
            if canonical_owner_by_ordinal[ordinal] == representative
            for kind in (_tensor_operation_kind(algebra_program.operations[ordinal]),)
        )
        is_helper = not task_operations
        helper_operations = tuple(
            operation
            for operation_ordinal, operation in enumerate(operations)
            if representative
            in (operation_task_owners.get(operation_ordinal, ()) or explicit_task_owners.get(operation_ordinal, ()))
        )
        if is_helper and not helper_operations:
            raise ValueError("native implementation helper has no source operation")
        if is_helper:
            phases = (
                PhysicalPhase(
                    0,
                    TaskCapability.IMPLEMENTATION_HELPER,
                    (),
                    source_task.domain_shape,
                    source_task.reduction_dimensions,
                ),
            )
        elif source_task.phases:
            phases = tuple(
                _native_physical_phase(
                    phase_id,
                    phase,
                    canonical_ordinal_by_native=canonical_ordinal_by_native,
                    algebra_program=algebra_program,
                )
                for phase_id, phase in enumerate(source_task.phases)
            )
            if tuple(ordinal for phase in phases for ordinal in phase.operation_ordinals) != tuple(
                ordinal for ordinal, _, _ in task_operations
            ):
                raise ValueError("native composition phases disagree with canonical task operations")
        else:
            phases = tuple(
                PhysicalPhase(
                    phase_id,
                    capability,
                    (ordinal,),
                    *physical_phase_iteration_space(algebra_program.operations[ordinal]),
                )
                for phase_id, (ordinal, _, capability) in enumerate(task_operations)
            )
        result_phase = {
            result.name: phase_id
            for phase_id, (ordinal, _, _) in enumerate(task_operations)
            for result in primitive_outputs(algebra_program.operations[ordinal])
        }
        phase_edges = (
            set(
                _native_physical_phase_edges(
                    source_task,
                    phases=phases,
                    algebra_program=algebra_program,
                )
            )
            if source_task.phases
            else {
                PhysicalPhaseEdge(source_phase, target_phase, RelationAccessClass.AFFINE)
                for target_phase, (ordinal, _, _) in enumerate(task_operations)
                for operand in primitive_inputs(algebra_program.operations[ordinal])
                if (source_phase := result_phase.get(operand.name)) is not None and source_phase != target_phase
            }
        )
        element_types = tuple(
            sorted(
                {
                    result.dtype.value
                    for ordinal, _, _ in task_operations
                    for result in primitive_outputs(algebra_program.operations[ordinal])
                }
            )
        )
        if is_helper:
            element_types = tuple(
                sorted(
                    {
                        _native_value_type(result.type)[0]
                        for operation in helper_operations
                        for result in operation.results
                    }
                )
            )
        if not element_types:
            raise ValueError("native task adapter requires typed algebra results")
        operation_kinds = (
            (TaskOperationKind.IMPLEMENTATION_HELPER,) if is_helper else tuple(kind for _, kind, _ in task_operations)
        )
        principal_kind = (
            TaskOperationKind.IMPLEMENTATION_HELPER
            if is_helper
            else next(
                (kind for kind in operation_kinds if kind is TaskOperationKind.CONTRACT),
                operation_kinds[0],
            )
        )
        task_domain_axis_ids = tuple(range(len(source_task.domain_shape)))
        relation_operations = tuple(
            operation
            for ordinal in canonical_ordinals
            if isinstance(operation := algebra_program.operations[ordinal], RelationPrimitive)
        )
        matched_relation_domain = False
        for relation_operation in reversed(relation_operations):
            prefix_axes = relation_operation.domain_axes[: len(source_task.domain_shape)]
            if tuple(axis.extent for axis in prefix_axes) == source_task.domain_shape:
                task_domain_axis_ids = tuple(axis.id for axis in prefix_axes)
                matched_relation_domain = True
                break
        if len(canonical_ordinals) == 1 and relation_operations and not matched_relation_domain:
            raise ValueError("native Relation task domain disagrees with canonical algebra axes")

        if compact_tile:
            reduction_dimensions = set(source_task.reduction_dimensions)
            owner_extent = prod(
                extent for axis, extent in enumerate(source_task.domain_shape) if axis not in reduction_dimensions
            )
            owner_axes = (TaskAxisTile(0, owner_extent, min(owner_extent, schedule.tile_shape[0])),)
        else:
            vector_owner_dimensions = tuple(
                axis
                for axis in source_task.vector_dimensions
                if axis not in explicit_contraction_dimensions and axis not in source_task.reduction_dimensions
            )
            owner_axes = tuple(
                TaskAxisTile(task_domain_axis_ids[axis], source_task.domain_shape[axis], schedule.tile_shape[axis])
                for axis in (*source_task.owner_dimensions, *vector_owner_dimensions)
            )
        if materialization_operation.attributes.get("composition_kind") == "indexed_accumulation":
            logical_grid_shape = tuple((axis.extent + axis.tile_extent - 1) // axis.tile_extent for axis in owner_axes)
            if prod(logical_grid_shape) != prod(schedule.grid_shape):
                raise ValueError("native schedule grid volume disagrees with its algebraic owner axes")
            family_axes = tuple(
                TaskAxis(f"owner.{axis.axis_id}", extent)
                for axis, extent in zip(owner_axes, logical_grid_shape, strict=True)
            )
        else:
            family_axes = tuple(TaskAxis(f"grid.{axis}", extent) for axis, extent in enumerate(schedule.grid_shape))
        family = TaskFamily(f"native.task.{source_task.ordinal}", family_axes, placement="device")
        for native_ordinal in component:
            native_schedule = schedules[native_ordinal]
            native_coordinates = tuple(product(*(range(extent) for extent in native_schedule.grid_shape)))
            if len(component) == 1:
                if len(native_coordinates) != len(family.coordinates):
                    raise ValueError("native schedule coordinates disagree with its canonical task")
                coordinate_cover_by_source_ordinal[native_ordinal] = {
                    source: (target,) for source, target in zip(native_coordinates, family.coordinates, strict=True)
                }
            elif native_ordinal == representative:
                if len(native_coordinates) != len(family.coordinates):
                    raise ValueError("native representative schedule coordinates disagree with its canonical task")
                coordinate_cover_by_source_ordinal[native_ordinal] = {
                    source: (target,) for source, target in zip(native_coordinates, family.coordinates, strict=True)
                }
            elif len(native_coordinates) == 1:
                coordinate_cover_by_source_ordinal[native_ordinal] = {native_coordinates[0]: family.coordinates}
            elif len(native_coordinates) == len(family.coordinates):
                coordinate_cover_by_source_ordinal[native_ordinal] = {
                    source: (target,) for source, target in zip(native_coordinates, family.coordinates, strict=True)
                }
            else:
                raise ValueError(
                    "native helper-task coordinate cover requires singleton broadcast or equal schedule grids"
                )
        fold_dimension_set = (
            set(source_task.reduction_dimensions) if TaskOperationKind.FOLD in operation_kinds else set()
        )
        contract_dimension_set = (
            explicit_contraction_dimensions
            if explicit_contraction_dimensions
            else set(source_task.reduction_dimensions) if TaskOperationKind.CONTRACT in operation_kinds else set()
        )
        shared_stream_dimensions = fold_dimension_set & contract_dimension_set
        streamed_dimensions = tuple(
            sorted(
                shared_stream_dimensions
                | (set(source_task.stream_dimensions) - fold_dimension_set - contract_dimension_set)
            )
        )
        fold_dimensions = tuple(sorted(fold_dimension_set - shared_stream_dimensions))
        contraction_dimensions = tuple(sorted(contract_dimension_set - shared_stream_dimensions))
        task_id = len(tiled_tasks)
        native_task_id_by_source_ordinal.update((native_ordinal, task_id) for native_ordinal in component)
        operation_ordinals = tuple(ordinal for ordinal, _, _ in task_operations)
        accesses = () if is_helper else task_value_accesses(algebra_program, operation_ordinals)
        if source_task.result_bindings:
            _verify_native_result_bindings(
                source_task,
                phases=phases,
                accesses=accesses,
                algebra_program=algebra_program,
            )
        tiled_tasks.append(
            TiledTask(
                task_id,
                family,
                operation_ordinals,
                operation_kinds,
                PhysicalTaskStructure(
                    principal_kind=principal_kind,
                    phases=phases,
                    phase_edges=tuple(sorted(phase_edges)),
                    owner_axes=owner_axes,
                    streamed_axes=tuple(
                        TaskAxisTile(
                            task_domain_axis_ids[axis],
                            source_task.domain_shape[axis],
                            source_task.domain_shape[axis] if compact_tile else schedule.tile_shape[axis],
                        )
                        for axis in streamed_dimensions
                    ),
                    reduction_axes=tuple(
                        TaskAxisTile(
                            task_domain_axis_ids[axis],
                            source_task.domain_shape[axis],
                            source_task.domain_shape[axis] if compact_tile else schedule.tile_shape[axis],
                        )
                        for axis in fold_dimensions
                    ),
                    contraction_axes=tuple(
                        TaskAxisTile(
                            task_domain_axis_ids[axis],
                            source_task.domain_shape[axis],
                            source_task.domain_shape[axis] if compact_tile else schedule.tile_shape[axis],
                        )
                        for axis in contraction_dimensions
                    ),
                    traversal=(
                        TaskTraversal.SOURCE_ORDERED
                        if source_task.stream_dimensions
                        or algebra_program.numerical_policy is NumericalPolicy.SOURCE_ORDERED
                        else TaskTraversal.UNORDERED
                    ),
                    element_types=element_types,
                    layout_classes=("native_structured",),
                    resources=TaskResourceEstimate(0, schedule.scratch_bytes, 0),
                ),
                accesses,
            )
        )

    assigned = [ordinal for task in tiled_tasks for ordinal in task.operation_ordinals]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(range(len(algebra_program.operations))):
        duplicates = tuple(sorted(ordinal for ordinal in set(assigned) if assigned.count(ordinal) > 1))
        missing = tuple(sorted(set(range(len(algebra_program.operations))) - set(assigned)))
        raise ValueError(
            "native materialization tasks must partition every source-bound algebra operation; "
            f"duplicates={duplicates}, missing={missing}"
        )
    native_task_ids = {
        operation_ordinal: {native_task_id_by_source_ordinal[task_ordinal] for task_ordinal in physical_task_ordinals}
        for operation_ordinal, physical_task_ordinals in operation_task_owners.items()
    }
    runtime_operation_ordinals = set(range(len(operations))) - inline_literal_ordinals
    if set(native_task_ids) != runtime_operation_ordinals or any(not task_ids for task_ids in native_task_ids.values()):
        raise ValueError("native materialization tasks must cover every native source operation")
    task_by_id = {task.id: task for task in tiled_tasks}
    helper_refinements: list[NativeImplementationHelperRefinement] = []
    for helper in tiled_tasks:
        if helper.operation_kinds != (TaskOperationKind.IMPLEMENTATION_HELPER,):
            continue
        helper_native_ordinals = {
            operation_ordinal
            for operation_ordinal in range(len(operations))
            if helper.id
            in (
                set(operation_task_owners.get(operation_ordinal, ()))
                | explicit_task_owners.get(operation_ordinal, set())
            )
        }
        operation_owners = tuple(
            (canonical_ordinal, canonical_owner_by_ordinal[canonical_ordinal])
            for canonical_ordinal, group in enumerate(operation_groups)
            if any(operations.index(operation) in helper_native_ordinals for operation in group)
        )
        coordinate_covers: list[tuple[int, TaskRelation]] = []
        for owner_id in sorted({owner for _, owner in operation_owners}):
            owner = task_by_id[owner_id]
            if len(helper.family.coordinates) == 1:
                pairs = tuple(
                    IndexPair(helper.family.coordinates[0], coordinate) for coordinate in owner.family.coordinates
                )
            elif len(helper.family.coordinates) == len(owner.family.coordinates):
                pairs = tuple(
                    IndexPair(source, target)
                    for source, target in zip(helper.family.coordinates, owner.family.coordinates, strict=True)
                )
            else:
                raise ValueError(
                    "native implementation helper coordinates require singleton broadcast or equal owner coverage"
                )
            coordinate_covers.append((owner_id, TaskRelation(helper.family, owner.family, pairs)))
        helper_refinements.append(
            NativeImplementationHelperRefinement(helper.id, operation_owners, tuple(coordinate_covers))
        )
    dependences: list[TaskDependence] = []
    events: list[EventTensorPlan] = []
    dependence_index_by_edge: dict[tuple[int, int], int] = {}
    event_relations = request.event_relations or request.event_requirements
    realization_by_event = {realization.ordinal: realization for realization in request.event_realizations}
    if request.event_realizations and set(realization_by_event) != set(range(len(event_relations))):
        raise ValueError("native Event realizations must cover every logical schedule EventTensor")
    imperative_by_event = {requirement.ordinal: requirement for requirement in request.event_requirements}
    for requirement in event_relations:
        producer_id = native_task_id_by_source_ordinal.get(requirement.producer_task)
        consumer_id = native_task_id_by_source_ordinal.get(requirement.consumer_task)
        if producer_id is None or consumer_id is None:
            continue
        if producer_id == consumer_id:
            continue
        producer = task_by_id.get(producer_id)
        consumer = task_by_id.get(consumer_id)
        if producer is None or consumer is None:
            raise ValueError("native Event references an unavailable materialization task")
        key = (producer.id, consumer.id)
        visibility = MemoryVisibility(
            _event_scope(requirement.scope),
            release_on_notify=requirement.release,
            acquire_before_consumer=requirement.acquire,
        )
        relation_pairs: list[IndexPair] = []
        producer_cover = coordinate_cover_by_source_ordinal[requirement.producer_task]
        consumer_cover = coordinate_cover_by_source_ordinal[requirement.consumer_task]
        for coordinate in product(*(range(extent) for extent in requirement.relation_domain_shape)):
            if not _predicate_holds(requirement.relation_predicate, coordinate):
                continue
            producer_coordinate = _evaluate_map(requirement.producer_indices, coordinate)
            consumer_coordinate = _evaluate_map(requirement.consumer_indices, coordinate)
            try:
                covered_producers = producer_cover[producer_coordinate]
                covered_consumers = consumer_cover[consumer_coordinate]
            except KeyError as error:
                raise ValueError(
                    f"native Event {requirement.ordinal} coordinate is outside its helper-task physical cover: "
                    f"producer task {requirement.producer_task} coordinate {producer_coordinate}, "
                    f"consumer task {requirement.consumer_task} coordinate {consumer_coordinate}"
                ) from error
            relation_pairs.extend(
                IndexPair(covered_producer, covered_consumer)
                for covered_producer in covered_producers
                for covered_consumer in covered_consumers
            )
        relation = TaskRelation(
            producer.family,
            consumer.family,
            tuple(dict.fromkeys(relation_pairs)),
        )
        dependence = TaskDependence(relation, visibility)
        realization = realization_by_event.get(requirement.ordinal)
        if realization is not None and realization.kind == "imperative_wait_notify":
            if imperative_by_event.get(requirement.ordinal) != requirement:
                raise ValueError("native imperative Event metadata disagrees with its logical EventTensor")
        event = derive_event_tensor_plan(
            dependence,
            name=f"native.event.{requirement.ordinal}",
            memory_scope=visibility.scope,
            generation_policy=(
                EventGenerationPolicy.PHASED if requirement.generations > 1 else EventGenerationPolicy.PER_INVOCATION
            ),
            scheduling_mode=(
                EventSchedulingMode.DYNAMIC
                if realization is None or realization.kind == "imperative_wait_notify"
                else EventSchedulingMode.STATIC
            ),
        )
        counts = {count.value for count in event.initial_count.counts}
        if counts != {requirement.notify_count} and not _is_unit_count_edge_factorization(requirement):
            raise ValueError("native Event notify count disagrees with its exact relation")
        previous_index = dependence_index_by_edge.get(key)
        if previous_index is None:
            dependence_index_by_edge[key] = len(dependences)
            dependences.append(dependence)
            events.append(event)
        else:
            previous = dependences[previous_index]
            if previous.visibility != dependence.visibility:
                raise ValueError("native Events on one coalesced task edge disagree on visibility")
            merged = TaskDependence(
                TaskRelation(
                    producer.family,
                    consumer.family,
                    tuple(dict.fromkeys((*previous.relation.pairs, *dependence.relation.pairs))),
                ),
                visibility,
            )
            dependences[previous_index] = merged
            events[previous_index] = derive_event_tensor_plan(
                merged,
                name=f"native.event.edge.{producer.id}.{consumer.id}",
                memory_scope=visibility.scope,
                generation_policy=EventGenerationPolicy.PER_INVOCATION,
                scheduling_mode=EventSchedulingMode.DYNAMIC,
            )

    declared_dependencies = {
        (native_task_id_by_source_ordinal[dependency], native_task_id_by_source_ordinal[task.ordinal])
        for task in source_tasks
        if task.ordinal in native_task_id_by_source_ordinal
        for dependency in task.dependencies
        if dependency in native_task_id_by_source_ordinal
        if native_task_id_by_source_ordinal[dependency] != native_task_id_by_source_ordinal[task.ordinal]
    }
    if set(dependence_index_by_edge) != declared_dependencies:
        raise ValueError(
            "native task adapter requires every materialization dependency to retain one imperative exact Event"
        )

    task_plan = CanonicalTaskPlan(tuple(tiled_tasks), tuple(dependences))
    tile_extent_by_axis = {
        axis.id: 1
        for value in (
            *algebra_program.inputs,
            *(output for operation in algebra_program.operations for output in primitive_outputs(operation)),
        )
        for axis in value.axes
    }
    for task in task_plan.tasks:
        if not any(
            isinstance(algebra_program.operations[ordinal], RelationPrimitive) for ordinal in task.operation_ordinals
        ):
            continue
        for axis in (
            *task.structure.owner_axes,
            *task.structure.streamed_axes,
            *task.structure.reduction_axes,
            *task.structure.contraction_axes,
        ):
            tile_extent_by_axis[axis.axis_id] = axis.tile_extent
    task_plan = materialize_runtime_relation_dependences(
        algebra_program,
        task_plan,
        tile_extent_by_axis=tile_extent_by_axis,
    )
    retained_dependences = set(task_plan.dependences)
    events = [event for event in events if event.required_dependence in retained_dependences]
    event_programs = tuple(
        ParametricEventTensorPlan(
            dependence_program_index=index,
            derivation=ParametricEventDerivation.EXACT_CONSUMER,
            capacity=len(task_plan.tasks[dependence.target_task_id].family.coordinates),
            slots=tuple(range(len(task_plan.tasks[dependence.target_task_id].family.coordinates))),
            generations=(0,) * len(task_plan.tasks[dependence.target_task_id].family.coordinates),
            generation_count=1,
            memory_scope=EventMemoryScope.DEVICE,
            generation_policy=EventGenerationPolicy.PER_INVOCATION,
            scheduling_mode=EventSchedulingMode.DYNAMIC,
            realization=ParametricEventRealization.IMPERATIVE_WAIT_NOTIFY,
        )
        for index, dependence in enumerate(task_plan.dependence_programs)
    )
    runtime_relation_dependence_sites = audit_runtime_relation_dependences(algebra_program, task_plan)

    return NativeCanonicalTaskEventPlan(
        algebra_program.semantic_fingerprint,
        task_plan,
        tuple(events),
        event_programs,
        request.event_realizations,
        tuple(helper_refinements),
        runtime_relation_dependence_sites,
    )


def executable_physical_request_from_native_request(
    request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
) -> ExecutablePhysicalRequest:
    """Adapt a native request to the shared executable physical schema.

    This boundary preserves the native schedule geometry as physical
    parameters while reusing the canonical task/Event adapter. It currently
    accepts retained bounded-counter Events; native Event erasure needs an
    explicit schedule-order witness in the shared schema.
    """
    native = canonical_task_event_plan_from_native_request(request, algebra_program=algebra_program)
    if any(realization.kind != "imperative_wait_notify" for realization in request.event_realizations):
        raise ValueError("native physical-request adaptation does not yet encode Event erasure witnesses")
    target = {
        "sm_90": PhysicalTarget.SM90,
        "sm_100": PhysicalTarget.SM100,
    }[request.target]
    numerical_policy = {
        NumericalPolicy.FAST: PhysicalNumericalPolicy.FAST,
        NumericalPolicy.SOURCE_ORDERED: PhysicalNumericalPolicy.SOURCE_ORDERED,
    }[algebra_program.numerical_policy]
    event_program = EventDataflowProgram(native.task_plan.task_families, native.task_plan.dependences, native.events)
    physical = physical_request_for_composed_tasks(
        algebra_record=algebra_program.canonical_record,
        algebra_fingerprint=algebra_program.semantic_fingerprint,
        task_plan=native.task_plan,
        event_program=event_program,
        target=target,
        numerical_policy=numerical_policy,
    )
    native_schedules = source_bound_schedule_tasks(request)
    source_tasks = source_bound_tasks(request)
    if len(native_schedules) > len(physical.schedules):
        raise ValueError("native physical schedule cannot be aligned with canonical task ownership")
    schedules = (
        *(
            (
                schedule
                if source_task.phases
                else replace(schedule, parameters=_source_schedule_parameters(source_schedule))
            )
            for schedule, source_schedule, source_task in zip(
                physical.schedules, native_schedules, source_tasks, strict=False
            )
        ),
        *physical.schedules[len(native_schedules) :],
    )
    return replace(
        physical,
        schedules=schedules,
        provenance={
            "source_boundary": "native_mlir",
            "provider_request_fingerprint": request.fingerprint,
        },
    )


def _source_schedule_parameters(source: Any) -> dict[str, object]:
    return {
        "kind": source.kind,
        "domain_shape": source.domain_shape,
        "grid_shape": source.grid_shape,
        "tile_shape": source.tile_shape,
        "serial_tiles": source.serial_tiles,
        "workgroup_threads": source.workgroup_threads,
        "subgroup_size": source.subgroup_size,
        "scratch_bytes": source.scratch_bytes,
        "reduction_axis": source.reduction_axis,
        "owner_dimensions": source.owner_dimensions,
        "vector_dimensions": source.vector_dimensions,
        "stream_dimensions": source.stream_dimensions,
        "producer_contraction_dimensions": source.producer_contraction_dimensions,
        "consumer_contraction_dimensions": source.consumer_contraction_dimensions,
        "online_mma": source.online_mma,
    }


def refine_native_scan_task_event_plan(
    request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
    chunk_extent: int,
    state_column_tile: int,
) -> NativeCanonicalTaskEventPlan:
    """Refine typed native Scan tasks through the canonical exact-Event model.

    Native extraction proves the typed algebra boundary first. The ordinary
    generic Scan refinement then replaces each singleton coarse Scan task with
    chunk and terminal-commit tasks, derives exact directional dependencies,
    and allocates bounded carried-state generations. This intentionally shares
    the same refinement used by the executable TensorProgram path.
    """
    coarse = canonical_task_event_plan_from_native_request(request, algebra_program=algebra_program)

    refined = refine_whole_program_scan_tasks(
        remove_dead_operations(algebra_program),
        coarse.task_plan,
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
    )
    return NativeCanonicalTaskEventPlan(
        algebra_fingerprint=coarse.algebra_fingerprint,
        task_plan=refined.task_plan,
        events=refined.event_plans,
        event_programs=(),
        physical_order_witnesses=(),
    )


def executable_scan_physical_request_from_native_request(
    request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
    chunk_extent: int,
    state_column_tile: int,
) -> ExecutablePhysicalRequest:
    """Refine a native Scan request into the shared executable boundary."""
    native = refine_native_scan_task_event_plan(
        request,
        algebra_program=algebra_program,
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
    )
    refinement = WholeProgramScanRefinement(
        native.task_plan,
        native.events,
        tuple(lower_event_tensor_plan(event, scheduling_mode=EventSchedulingMode.STATIC) for event in native.events),
    )
    target = {
        "sm_90": PhysicalTarget.SM90,
        "sm_100": PhysicalTarget.SM100,
    }[request.target]
    numerical_policy = {
        NumericalPolicy.FAST: PhysicalNumericalPolicy.FAST,
        NumericalPolicy.SOURCE_ORDERED: PhysicalNumericalPolicy.SOURCE_ORDERED,
    }[algebra_program.numerical_policy]
    return attach_whole_program_scan_physical_request(
        algebra_program,
        refinement,
        target=target,
        numerical_policy=numerical_policy,
        provenance={
            "source_boundary": "native_mlir",
            "provider_request_fingerprint": request.fingerprint,
            "native_task_event_fingerprint": native.event_fingerprint,
        },
    )


def _is_unit_count_edge_factorization(requirement: ShuttleExactEventRequirement) -> bool:
    """Recognize a native per-edge Event refinement of consumer-counted readiness."""
    if requirement.notify_count != 1:
        return False
    storage_generations: list[tuple[int, int]] = []
    for coordinate in product(*(range(extent) for extent in requirement.relation_domain_shape)):
        if not _predicate_holds(requirement.relation_predicate, coordinate):
            continue
        storage = _evaluate_map(requirement.storage_slot, coordinate)
        generation = _evaluate_map(requirement.generation_index, coordinate)
        if len(storage) != 1 or len(generation) != 1:
            return False
        if not 0 <= storage[0] < requirement.storage_capacity:
            return False
        if not 0 <= generation[0] < requirement.generations:
            return False
        storage_generations.append((storage[0], generation[0]))
    return len(storage_generations) == len(set(storage_generations))


def _source_operations(
    region: ShuttleOperationRecord,
) -> tuple[tuple[ShuttleOperationRecord, ...], dict[str, tuple[int, ShuttleOperationRecord]]]:
    operations: list[ShuttleOperationRecord] = []
    by_source: dict[str, tuple[int, ShuttleOperationRecord]] = {}
    for operation in region.body().operations:
        if operation.name not in _OPERATION_KINDS:
            continue
        source = operation.attributes.get("source")
        sources = operation.attributes.get("sources")
        if source is not None and sources is not None:
            raise ValueError(f"native algebra operation {operation.name} has singular and plural source provenance")
        if sources is None:
            sources = () if source is None else (source,)
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError(f"native algebra operation {operation.name} has no source provenance")
        ordinal = len(operations)
        operations.append(operation)
        for source in sources:
            key = _expanded_source_key(source, operation.attributes.get("source_group"))
            if key in by_source:
                raise ValueError("native algebra source provenance is not unique")
            by_source[key] = (ordinal, operation)
    if not operations:
        raise ValueError("native task adapter found no source-bound algebra operations")
    return tuple(operations), by_source


def _operation_kind(operation: ShuttleOperationRecord) -> tuple[TaskOperationKind, TaskCapability]:
    semantics = operation.attributes.get("semantics")
    if operation.name == "shuttle.map" and semantics in (
        {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<transpose>"},
        {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<broadcast_in_dim>"},
        {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<reshape>"},
    ):
        return TaskOperationKind.RELATION, TaskCapability.INDEX_RELATION
    if operation.name == "shuttle.relation" and semantics in (
        {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<compare>"},
        {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<select>"},
        {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<convert>"},
    ):
        return TaskOperationKind.MAP, TaskCapability.SCALAR_MAP
    try:
        return _OPERATION_KINDS[operation.name]
    except KeyError as error:
        raise ValueError(f"unsupported source-bound algebra operation: {operation.name}") from error


def _native_relation_semantics_and_parameters(operation: ShuttleOperationRecord) -> tuple[str, tuple[object, ...]]:
    encoded = operation.attributes.get("semantics")
    if not isinstance(encoded, dict) or encoded.get("kind") != "mlir_attribute":
        raise ValueError("native Relation requires typed semantics")
    value = encoded.get("value")
    prefix = "#shuttle.relation_semantics<"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(">"):
        raise ValueError("native Relation semantics have an invalid spelling")
    parameters = operation.attributes.get("parameters", ())
    if not isinstance(parameters, (list, tuple)):
        raise ValueError("native Relation parameters must be a structured sequence")
    return value[len(prefix) : -1], tuple(parameters)


def _attribute_key(value: object) -> str:
    return canonical_fingerprint({"attribute": value})


def _materialization_sources(operation: ShuttleOperationRecord) -> list[object]:
    sources = operation.attributes.get("composition_sources")
    if sources is None:
        sources = operation.attributes.get("fold_region_sources")
    source = operation.attributes.get("source")
    if sources is None:
        return [] if source is None else [source]
    if not isinstance(sources, list):
        raise ValueError("native materialization task source group must be an array")
    if not sources:
        raise ValueError("native materialization task source group must be nonempty")
    if source is not None and source not in sources:
        raise ValueError("native materialization principal source is outside its composition")
    return sources


def _expanded_source_key(source: object, expansion: object | None) -> str:
    return _attribute_key(source if expansion is None else [source, expansion])


def _materialization_source_keys(operation: ShuttleOperationRecord) -> tuple[str, ...]:
    sources = _materialization_sources(operation)
    expansion = operation.attributes.get("source_expansion")
    expansion_group = operation.attributes.get("source_expansion_group")
    if expansion is not None and expansion_group is not None:
        raise ValueError("native source expansion cannot be both singular and grouped")
    if expansion_group is not None:
        if not isinstance(expansion_group, list) or len(expansion_group) != len(sources):
            raise ValueError("native source expansion group must align with composition sources")
        return tuple(
            _expanded_source_key(source, member) for source, member in zip(sources, expansion_group, strict=True)
        )
    if expansion is not None and len(sources) != 1:
        raise ValueError("native source expansion requires one principal algebra source")
    return tuple(_expanded_source_key(source, expansion) for source in sources)


def _native_operation_task_owners(
    operations: tuple[ShuttleOperationRecord, ...],
    operation_by_source: dict[str, tuple[int, ShuttleOperationRecord]],
    materialization_operations: tuple[ShuttleOperationRecord, ...],
    inline_literal_ordinals: set[int],
) -> dict[int, frozenset[int]]:
    """Derive physical task owners for absorbed operations from exact uses."""
    owners: dict[int, frozenset[int]] = {}
    for task_ordinal, materialization in enumerate(materialization_operations):
        for source_key in _materialization_source_keys(materialization):
            operation_entry = operation_by_source.get(source_key)
            if operation_entry is None:
                raise ValueError("native materialization task references an unknown algebra source")
            operation_ordinal = operation_entry[0]
            if operation_ordinal not in inline_literal_ordinals:
                owners[operation_ordinal] = frozenset((task_ordinal,))

    producer_by_value = {
        result.id: ordinal for ordinal, operation in enumerate(operations) for result in operation.results
    }
    consumers: dict[int, set[int]] = {ordinal: set() for ordinal in range(len(operations))}
    for consumer_ordinal, operation in enumerate(operations):
        for operand in operation.operands:
            producer = producer_by_value.get(operand)
            if producer is not None:
                consumers[producer].add(consumer_ordinal)
    for operation_ordinal in reversed(range(len(operations))):
        if operation_ordinal in inline_literal_ordinals or operation_ordinal in owners:
            continue
        downstream = frozenset(
            owner for consumer in consumers[operation_ordinal] for owner in owners.get(consumer, frozenset())
        )
        if downstream:
            owners[operation_ordinal] = downstream
    missing = tuple(
        ordinal for ordinal in range(len(operations)) if ordinal not in inline_literal_ordinals and ordinal not in owners
    )
    if missing:
        raise ValueError(f"native use-def provenance does not reach a materialization task for operations {missing}")
    return owners


def _coalesced_native_task_components(
    task_count: int,
    operation_groups: tuple[tuple[ShuttleOperationRecord, ...], ...],
    operations: tuple[ShuttleOperationRecord, ...],
    operation_task_owners: dict[int, frozenset[int]],
    inline_literal_ordinals: set[int],
) -> tuple[tuple[int, ...], ...]:
    """Quotient physical helper tasks that implement one canonical operation."""
    parent = list(range(task_count))

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for group in operation_groups:
        task_ordinals = sorted(
            {
                task_ordinal
                for operation in group
                if (operation_ordinal := operations.index(operation)) not in inline_literal_ordinals
                for task_ordinal in operation_task_owners[operation_ordinal]
            }
        )
        for task_ordinal in task_ordinals[1:]:
            union(task_ordinals[0], task_ordinal)
    components: dict[int, list[int]] = {}
    for task_ordinal in range(task_count):
        components.setdefault(root(task_ordinal), []).append(task_ordinal)
    return tuple(tuple(members) for _, members in sorted(components.items(), key=lambda item: min(item[1])))


def _evaluate_map(map_: CuteAffineMap, coordinate: tuple[int, ...]) -> tuple[int, ...]:
    if len(coordinate) != map_.dimensions:
        raise ValueError("native Event affine map rank disagrees with its relation domain")
    return tuple(_evaluate_expression(expression, coordinate) for expression in map_.results)


def _predicate_holds(predicate: CuteIntegerSet | None, coordinate: tuple[int, ...]) -> bool:
    if predicate is None:
        return True
    if len(coordinate) != predicate.dimensions:
        raise ValueError("native Event predicate rank disagrees with its relation domain")
    return all(
        value == 0 if equality else value >= 0
        for value, equality in zip(
            (_evaluate_expression(expression, coordinate) for expression in predicate.constraints),
            predicate.equalities,
            strict=True,
        )
    )


def _evaluate_expression(expression: CuteAffineExpression, coordinate: tuple[int, ...]) -> int:
    if expression.kind is CuteAffineExpressionKind.DIMENSION:
        return coordinate[expression.payload]
    if expression.kind is CuteAffineExpressionKind.CONSTANT:
        return expression.payload
    left = _evaluate_expression(expression.operands[0], coordinate)
    right = _evaluate_expression(expression.operands[1], coordinate)
    if expression.kind is CuteAffineExpressionKind.ADD:
        return left + right
    if expression.kind is CuteAffineExpressionKind.MULTIPLY:
        return left * right
    if right <= 0:
        raise ValueError("native Event affine division requires a positive divisor")
    if expression.kind is CuteAffineExpressionKind.MODULO:
        return left % right
    if expression.kind is CuteAffineExpressionKind.FLOOR_DIVIDE:
        return left // right
    if expression.kind is CuteAffineExpressionKind.CEIL_DIVIDE:
        return -(-left // right)
    raise ValueError(f"unsupported native Event affine operation: {expression.kind.value}")


def _event_scope(scope: str) -> EventMemoryScope:
    scopes = {
        "thread": EventMemoryScope.CTA,
        "workgroup": EventMemoryScope.CTA,
        "device": EventMemoryScope.DEVICE,
        "system": EventMemoryScope.SYSTEM,
    }
    return scopes[scope]


def _verify_tensor_program_binding(
    region: ShuttleOperationRecord,
    native_operations: tuple[ShuttleOperationRecord, ...],
    program: TensorProgram,
) -> tuple[tuple[ShuttleOperationRecord, ...], ...]:
    """Verify the represented native subset against canonical TensorProgram semantics."""
    pending = list(program.operations)
    while pending:
        operation = pending.pop()
        if isinstance(operation, ScanPrimitive):
            pending.extend(operation.body.operations)
        elif isinstance(operation, ContractPrimitive) and any(
            precision is not ContractOperandPrecision.DEFAULT for precision in operation.operand_precision
        ):
            raise ValueError("native Contract conformance cannot preserve explicit HIGH/HIGHEST operand precision")
    operation_groups = _native_operation_groups(region, native_operations, program)
    type_by_value = {argument.id: argument.type for argument in region.body().arguments}
    for native_operation in native_operations:
        type_by_value.update({result.id: result.type for result in native_operation.results})
    terminal_uses = {
        operand
        for native_operation in region.body().operations
        if native_operation.name not in _OPERATION_KINDS
        for operand in native_operation.operands
    }
    inline_literal_values = frozenset(
        operation.results[0].id for operation in native_operations if _inlineable_nullary_literal(operation)
    )
    if len(region.body().arguments) != len(program.inputs):
        raise ValueError("native and canonical algebra input counts disagree")
    native_values: dict[int, int] = {}
    for ordinal, (argument, value) in enumerate(zip(region.body().arguments, program.inputs, strict=True)):
        if _native_value_type(argument.type) != (value.dtype.value, value.shape):
            raise ValueError("native and canonical algebra input types disagree")
        native_values[argument.id] = ordinal
    canonical_value_ordinals = {value.name: ordinal for ordinal, value in enumerate(program.inputs)}
    next_value = len(program.inputs)
    for group_index, (native_group, operation) in enumerate(zip(operation_groups, program.operations, strict=True)):
        canonical_kind = _tensor_operation_kind(operation)
        expected_operands = tuple(canonical_value_ordinals[value.name] for value in primitive_inputs(operation))
        suffix = tuple(source for group in operation_groups[group_index + 1 :] for source in group)
        boundary_inputs, boundary_outputs = _native_group_boundary(
            native_group,
            suffix,
            terminal_uses,
            ignored_outputs=inline_literal_values,
        )
        try:
            actual_operands = tuple(native_values[operand] for operand in boundary_inputs)
        except KeyError as error:
            raise ValueError("native algebra operand does not follow canonical SSA provenance") from error
        reordered_map_operands = actual_operands != expected_operands
        if reordered_map_operands and not (
            isinstance(operation, MapPrimitive) and sorted(actual_operands) == sorted(expected_operands)
        ):
            raise ValueError(
                f"native and canonical algebra SSA operands disagree at operation {group_index}: "
                f"native={actual_operands}, canonical={expected_operands}"
            )
        outputs = primitive_outputs(operation)
        if len(boundary_outputs) != len(outputs):
            raise ValueError(
                f"native and canonical algebra result counts disagree at canonical operation {group_index} "
                f"({canonical_kind.value}): native={len(boundary_outputs)}, canonical={len(outputs)}"
            )
        for native_result_id, output in zip(boundary_outputs, outputs, strict=True):
            native_result_type = type_by_value[native_result_id]
            if _native_value_type(native_result_type) != (output.dtype.value, output.shape):
                raise ValueError("native and canonical algebra result types disagree")
            native_values[native_result_id] = next_value
            canonical_value_ordinals[output.name] = next_value
            next_value += 1
        try:
            if (
                len(native_group) == 1
                and _operation_kind(native_group[0])[0] is canonical_kind
                and not (isinstance(operation, MapPrimitive) and native_group[0].name == "shuttle.relation")
                and not reordered_map_operands
            ):
                _verify_native_operation_semantics(native_group[0], operation)
            elif isinstance(operation, MapPrimitive):
                canonical_input_by_ordinal = {canonical_value_ordinals[value.name]: value for value in operation.inputs}
                _verify_composed_native_map_semantics(
                    native_group,
                    operation,
                    boundary_inputs,
                    tuple(canonical_input_by_ordinal[native_values[value]] for value in boundary_inputs),
                    boundary_outputs,
                )
            elif isinstance(operation, FoldPrimitive):
                _verify_composed_native_fold_semantics(native_group, operation)
            elif isinstance(operation, ContractPrimitive):
                canonical_input_by_ordinal = {canonical_value_ordinals[value.name]: value for value in operation.inputs}
                _verify_composed_native_contract_semantics(
                    native_group,
                    operation,
                    boundary_inputs,
                    tuple(canonical_input_by_ordinal[native_values[value]] for value in boundary_inputs),
                    boundary_outputs,
                )
            elif isinstance(operation, RelationPrimitive):
                _verify_composed_native_relation_semantics(native_group, operation, type_by_value)
            else:
                raise ValueError("native source group has no canonical algebra composition rule")
        except ValueError as error:
            raise ValueError(f"canonical operation {group_index} ({canonical_kind.value}): {error}") from error
    return operation_groups


def _native_operation_groups(
    region: ShuttleOperationRecord,
    native_operations: tuple[ShuttleOperationRecord, ...],
    program: TensorProgram,
) -> tuple[tuple[ShuttleOperationRecord, ...], ...]:
    """Group native source operations by exact canonical SSA/use-def boundaries.

    Native conversion may hoist a pure scalar literal and share it among many
    later operations.  TensorProgram embeds that same literal in each scalar or
    index AST.  Such values are selected through use-def provenance and may
    therefore participate in more than one semantic group; non-literal source
    operations still form an exact, ordered partition.
    """
    type_by_value = {argument.id: argument.type for argument in region.body().arguments}
    for operation in native_operations:
        type_by_value.update({result.id: result.type for result in operation.results})
    terminal_uses = {
        operand
        for operation in region.body().operations
        if operation.name not in _OPERATION_KINDS
        for operand in operation.operands
    }
    inline_literals = tuple(operation for operation in native_operations if _inlineable_nullary_literal(operation))
    inline_value_to_operation = {operation.results[0].id: operation for operation in inline_literals}
    primary_operations = tuple(operation for operation in native_operations if operation not in inline_literals)
    inline_values = frozenset(inline_value_to_operation)
    memo: dict[tuple[int, int], tuple[tuple[ShuttleOperationRecord, ...], ...] | None] = {}
    furthest = (0, 0)

    def groups_from(
        canonical_ordinal: int,
        cursor: int,
    ) -> tuple[tuple[ShuttleOperationRecord, ...], ...] | None:
        nonlocal furthest
        if canonical_ordinal > furthest[0]:
            furthest = (canonical_ordinal, cursor)
        key = (canonical_ordinal, cursor)
        if key in memo:
            return memo[key]
        if canonical_ordinal == len(program.operations):
            result = () if cursor == len(primary_operations) else None
            memo[key] = result
            return result
        if cursor >= len(primary_operations):
            memo[key] = None
            return None
        canonical = program.operations[canonical_ordinal]
        expected_kind = _tensor_operation_kind(canonical)
        for end in range(cursor + 1, len(primary_operations) + 1):
            primary_group = primary_operations[cursor:end]
            literal_ancestors = tuple(
                operation
                for operation in inline_literals
                if operation.results[0].id in {operand for candidate in primary_group for operand in candidate.operands}
            )
            group = (*literal_ancestors, *primary_group)
            if not _native_group_supports_kind(group, expected_kind):
                continue
            inputs, outputs = _native_group_boundary(
                group,
                primary_operations[end:],
                terminal_uses,
                ignored_outputs=inline_values,
            )
            if _boundary_types(inputs, type_by_value) != tuple(
                (value.dtype.value, value.shape) for value in primitive_inputs(canonical)
            ):
                continue
            if _boundary_types(outputs, type_by_value) != tuple(
                (value.dtype.value, value.shape) for value in primitive_outputs(canonical)
            ):
                continue
            suffix_groups = groups_from(canonical_ordinal + 1, end)
            if suffix_groups is not None:
                result = (group, *suffix_groups)
                memo[key] = result
                return result
        memo[key] = None
        return None

    groups = groups_from(0, 0)
    if groups is None:
        canonical_ordinal, cursor = furthest
        expected = (
            _tensor_operation_kind(program.operations[canonical_ordinal]).value
            if canonical_ordinal < len(program.operations)
            else "end"
        )
        actual = _operation_kind(primary_operations[cursor])[0].value if cursor < len(primary_operations) else "end"
        raise ValueError(
            "native source use-def graph has no exact grouping onto the canonical TensorProgram "
            f"after canonical operation {canonical_ordinal} ({expected}) and native runtime operation "
            f"{cursor} ({actual}); {len(primary_operations)} native runtime operations and "
            f"{len(program.operations)} canonical operations"
        )
    return groups


def _inlineable_nullary_literal(operation: ShuttleOperationRecord) -> bool:
    """Return whether a native scalar source is safely duplicated into ASTs."""
    if operation.operands or len(operation.results) != 1 or operation.results[0].type.shape:
        return False
    if operation.name == "shuttle.map":
        try:
            expression = scalar_expression_from_source_map(operation, ())
        except ValueError:
            return False
        return not _scalar_expression_input_names(expression)
    if operation.name != "shuttle.relation":
        return False
    try:
        semantics, parameters = source_relation_semantics(operation)
    except ValueError:
        return False
    return semantics == "constant" and len(parameters) == 1


def _scalar_expression_input_names(expression: ScalarExpression) -> frozenset[str]:
    if expression.kind is ScalarExpressionKind.INPUT:
        return frozenset((expression.input_name,))
    return frozenset().union(*(_scalar_expression_input_names(operand) for operand in expression.operands))


def _native_group_supports_kind(group: tuple[ShuttleOperationRecord, ...], expected_kind: TaskOperationKind) -> bool:
    kinds = tuple(_operation_kind(operation)[0] for operation in group)
    if len(group) == 1:
        return kinds == (expected_kind,)
    if expected_kind is TaskOperationKind.RELATION:
        return all(kind is TaskOperationKind.RELATION for kind in kinds)
    if expected_kind is TaskOperationKind.MAP:
        return all(kind in (TaskOperationKind.MAP, TaskOperationKind.RELATION) for kind in kinds)
    if expected_kind is TaskOperationKind.FOLD:
        return (
            kinds[-1] is TaskOperationKind.FOLD
            and all(kind is TaskOperationKind.MAP for kind in kinds[:-1])
            and all(not operation.operands for operation in group[:-1])
        )
    if expected_kind is TaskOperationKind.CONTRACT:
        return (
            kinds[-1] is TaskOperationKind.CONTRACT and all(kind is TaskOperationKind.MAP for kind in kinds[:-1])
        ) or all(kind is TaskOperationKind.MAP for kind in kinds)
    return False


def _native_principal_operation(
    group: tuple[ShuttleOperationRecord, ...], expected_kind: TaskOperationKind
) -> ShuttleOperationRecord:
    candidates = tuple(operation for operation in group if _operation_kind(operation)[0] is expected_kind)
    if not candidates:
        raise ValueError("native source group has no operation carrying its canonical capability")
    return candidates[-1]


def _native_group_capability(
    group: tuple[ShuttleOperationRecord, ...], expected_kind: TaskOperationKind
) -> TaskCapability:
    candidates = tuple(operation for operation in group if _operation_kind(operation)[0] is expected_kind)
    if candidates:
        return _operation_kind(candidates[-1])[1]
    if expected_kind is TaskOperationKind.CONTRACT and all(
        _operation_kind(operation)[0] is TaskOperationKind.MAP for operation in group
    ):
        return TaskCapability.MATRIX_CONTRACT
    if expected_kind is TaskOperationKind.MAP and all(
        _operation_kind(operation)[0] in (TaskOperationKind.MAP, TaskOperationKind.RELATION) for operation in group
    ):
        return TaskCapability.SCALAR_MAP
    raise ValueError(
        f"native source group has no operation carrying its canonical {expected_kind.value} capability; "
        f"native kinds={tuple(_operation_kind(operation)[0].value for operation in group)}"
    )


def _native_group_boundary(
    group: tuple[ShuttleOperationRecord, ...],
    suffix: tuple[ShuttleOperationRecord, ...],
    terminal_uses: set[int],
    *,
    ignored_outputs: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    defined = {result.id for operation in group for result in operation.results}
    inputs = tuple(
        dict.fromkeys(operand for operation in group for operand in operation.operands if operand not in defined)
    )
    future_uses = terminal_uses | {operand for operation in suffix for operand in operation.operands}
    outputs = tuple(
        result.id
        for operation in group
        for result in operation.results
        if result.id in future_uses and result.id not in ignored_outputs
    )
    return inputs, outputs


def _boundary_types(values: tuple[int, ...], type_by_value: dict[int, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    try:
        return tuple(_native_value_type(type_by_value[value]) for value in values)
    except KeyError as error:
        raise ValueError("native SSA boundary references a value without a structured type") from error


def _tensor_operation_kind(operation: object) -> TaskOperationKind:
    if isinstance(operation, ContractPrimitive):
        return TaskOperationKind.CONTRACT
    if isinstance(operation, FoldPrimitive):
        return TaskOperationKind.FOLD
    if isinstance(operation, MapPrimitive):
        return TaskOperationKind.MAP
    if isinstance(operation, SelectionPrimitive):
        return TaskOperationKind.SELECTION
    if isinstance(operation, ScanPrimitive):
        return TaskOperationKind.SCAN
    if isinstance(operation, RelationPrimitive):
        return TaskOperationKind.RELATION
    raise ValueError(f"native canonical algebra binding does not yet support {type(operation).__name__}")


def _verify_composed_native_map_semantics(
    native_group: tuple[ShuttleOperationRecord, ...],
    operation: MapPrimitive,
    boundary_inputs: tuple[int, ...],
    boundary_input_values: tuple[Any, ...],
    boundary_outputs: tuple[int, ...],
) -> None:
    if len(boundary_inputs) != len(operation.inputs) or len(boundary_outputs) != 1:
        raise ValueError("composed native Map must retain its exact canonical boundary")
    expression_by_value = {
        value_id: scalar_input(value.name)
        for value_id, value in zip(boundary_inputs, boundary_input_values, strict=True)
    }
    for native in native_group:
        operands = tuple(expression_by_value[operand] for operand in native.operands)
        if native.name == "shuttle.map":
            placeholders = tuple(f"native.operand.{operand}" for operand in native.operands)
            decoded = scalar_expression_from_source_map(native, placeholders)
            expression = _substitute_scalar_inputs(
                decoded,
                dict(zip(placeholders, operands, strict=True)),
            )
        else:
            semantics, parameters = _native_relation_semantics_and_parameters(native)
            if semantics in ("broadcast_in_dim", "reshape") and len(operands) == 1:
                expression = operands[0]
            elif semantics == "compare" and len(parameters) == 2 and len(operands) == 2:
                try:
                    comparison = {
                        "LT": ScalarExpressionKind.LESS,
                        "LE": ScalarExpressionKind.LESS_EQUAL,
                        "GT": ScalarExpressionKind.GREATER,
                        "GE": ScalarExpressionKind.GREATER_EQUAL,
                        "EQ": ScalarExpressionKind.EQUAL,
                        "NE": ScalarExpressionKind.NOT_EQUAL,
                    }[parameters[1]]
                except KeyError as error:
                    raise ValueError(f"native compare predicate {parameters[1]!r} has no scalar Map rule") from error
                expression = scalar_binary(comparison, *operands)
            elif semantics == "select" and not parameters and len(operands) == 3:
                expression = scalar_select(*operands)
            elif semantics == "convert" and not parameters and len(operands) == 1:
                result_dtype = DType(_native_element_type(native.results[0].type.element_type))
                expression = scalar_convert(
                    operands[0],
                    result_dtype=result_dtype,
                    semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
                )
            elif semantics == "constant" and len(parameters) == 1 and not operands:
                expression = scalar_constant(parameters[0])
            elif semantics in ("add", "subtract", "multiply") and not parameters and len(operands) == 2:
                expression = scalar_binary(
                    {
                        "add": ScalarExpressionKind.ADD,
                        "subtract": ScalarExpressionKind.SUBTRACT,
                        "multiply": ScalarExpressionKind.MULTIPLY,
                    }[semantics],
                    *operands,
                )
            else:
                raise ValueError(f"native {semantics!r} Relation has no scalar Map composition rule")
        if len(native.results) != 1:
            raise ValueError("composed native Map sources must produce one result")
        expression_by_value[native.results[0].id] = expression
    decoded = expression_by_value.get(boundary_outputs[0])
    input_dtypes = {value.name: value.dtype for value in operation.inputs}
    if decoded is None or not _scalar_expressions_equivalent(decoded, operation.expression, input_dtypes):
        raise ValueError("native and canonical composed scalar Map ASTs disagree")


def _verify_composed_native_contract_semantics(
    native_group: tuple[ShuttleOperationRecord, ...],
    operation: ContractPrimitive,
    boundary_inputs: tuple[int, ...],
    boundary_input_values: tuple[Any, ...],
    boundary_outputs: tuple[int, ...],
) -> None:
    """Prove native exact conversions plus Contract, or an indexed outer product."""
    if not operation.reduction_axes:
        _verify_composed_native_zero_reduction_contract(
            native_group,
            operation,
            boundary_inputs,
            boundary_input_values,
            boundary_outputs,
        )
        return
    if len(boundary_inputs) != 2 or len(boundary_outputs) != 1:
        raise ValueError("composed native Contract must retain its binary canonical boundary")
    contracts = tuple(native for native in native_group if native.name == "shuttle.contract")
    if len(contracts) != 1 or native_group[-1] is not contracts[0]:
        raise ValueError("composed native Contract requires one terminal Contract operation")
    contract = contracts[0]
    if boundary_outputs != tuple(result.id for result in contract.results):
        raise ValueError("composed native Contract must expose the terminal Contract result")
    _verify_native_operation_semantics(contract, operation)

    expression_by_value = {
        value_id: scalar_input(value.name)
        for value_id, value in zip(boundary_inputs, boundary_input_values, strict=True)
    }
    producer_by_result: dict[int, ShuttleOperationRecord] = {}
    for native in native_group[:-1]:
        if native.name != "shuttle.map" or len(native.results) != 1:
            raise ValueError("composed native Contract prelude requires single-result scalar Maps")
        operands = tuple(expression_by_value[operand] for operand in native.operands)
        placeholders = tuple(f"native.operand.{operand}" for operand in native.operands)
        decoded = scalar_expression_from_source_map(native, placeholders)
        expression_by_value[native.results[0].id] = _substitute_scalar_inputs(
            decoded, dict(zip(placeholders, operands, strict=True))
        )
        producer_by_result[native.results[0].id] = native
    for operand_id, boundary_input, value in zip(contract.operands, boundary_inputs, boundary_input_values, strict=True):
        expected = scalar_convert(
            scalar_input(value.name),
            result_dtype=operation.accumulation_dtype,
            semantics=ScalarConversionSemantics.EXACT,
        )
        decoded = expression_by_value.get(operand_id)
        if decoded is None or not _scalar_expressions_equivalent(decoded, expected, {value.name: value.dtype}):
            raise ValueError("native Contract operand conversion disagrees with canonical accumulation semantics")
        for coordinate in product(*(range(extent) for extent in value.shape)):
            if _trace_native_map_coordinate(operand_id, coordinate, boundary_input, producer_by_result) != coordinate:
                raise ValueError("native Contract operand conversion changes canonical indexing")


def _verify_composed_native_zero_reduction_contract(
    native_group: tuple[ShuttleOperationRecord, ...],
    operation: ContractPrimitive,
    boundary_inputs: tuple[int, ...],
    boundary_input_values: tuple[Any, ...],
    boundary_outputs: tuple[int, ...],
) -> None:
    """Prove a native indexed-Map DAG is one reduction-free Contract."""
    if len(boundary_inputs) != 2 or len(boundary_outputs) != 1:
        raise ValueError("native Map composition can represent only a binary reduction-free Contract")
    expression_by_value = {
        value_id: scalar_input(value.name)
        for value_id, value in zip(boundary_inputs, boundary_input_values, strict=True)
    }
    for native in native_group:
        if native.name != "shuttle.map":
            raise ValueError("reduction-free Contract composition requires indexed scalar Maps")
        operands = tuple(expression_by_value[operand] for operand in native.operands)
        placeholders = tuple(f"native.operand.{operand}" for operand in native.operands)
        decoded = scalar_expression_from_source_map(native, placeholders)
        expression = _substitute_scalar_inputs(decoded, dict(zip(placeholders, operands, strict=True)))
        if len(native.results) != 1:
            raise ValueError("reduction-free Contract scalar Maps must produce one result")
        expression_by_value[native.results[0].id] = expression

    expected_operands = tuple(
        scalar_convert(
            scalar_input(value.name),
            result_dtype=operation.accumulation_dtype,
            semantics=ScalarConversionSemantics.EXACT,
        )
        for value in operation.inputs
    )
    expected = scalar_binary(ScalarExpressionKind.MULTIPLY, *expected_operands)
    decoded = expression_by_value.get(boundary_outputs[0])
    input_dtypes = {value.name: value.dtype for value in operation.inputs}
    if decoded is None or not _scalar_expressions_equivalent(decoded, expected, input_dtypes):
        raise ValueError("native indexed-Map DAG disagrees with reduction-free Contract scalar semantics")

    producer_by_result = {result.id: native for native in native_group for result in native.results}
    output_shape = operation.output.shape
    if prod(output_shape) > 1_000_000:
        raise ValueError("reduction-free Contract exceeds bounded exact indexing verification")
    resolved_maps = []
    for input_index, value in enumerate(operation.inputs):
        aliases = {
            mapping.operand_axis: mapping.domain_axis
            for mapping in operation.index_maps_for_input(input_index)
            if mapping.divisor == 1 and mapping.offset == 0 and mapping.modulus is None
        }
        if len(aliases) != len(operation.index_maps_for_input(input_index)):
            raise ValueError("reduction-free Contract requires projected affine canonical indexing")
        resolved_maps.append(tuple(operation.output.axes.index(aliases.get(axis, axis)) for axis in value.axes))
    for output_coordinate in product(*(range(extent) for extent in output_shape)):
        for boundary_input, expected_positions in zip(boundary_inputs, resolved_maps, strict=True):
            native_coordinate = _trace_native_map_coordinate(
                boundary_outputs[0],
                output_coordinate,
                boundary_input,
                producer_by_result,
            )
            expected_coordinate = tuple(output_coordinate[position] for position in expected_positions)
            if native_coordinate != expected_coordinate:
                raise ValueError("native indexed-Map DAG disagrees with reduction-free Contract indexing")


def _trace_native_map_coordinate(
    value: int,
    coordinate: tuple[int, ...],
    target: int,
    producer_by_result: dict[int, ShuttleOperationRecord],
) -> tuple[int, ...]:
    if value == target:
        return coordinate
    producer = producer_by_result.get(value)
    if producer is None:
        raise ValueError("native indexed-Map DAG does not reach the requested Contract operand")
    maps = producer.attributes.get("indexing_maps")
    if not isinstance(maps, list) or len(maps) != len(producer.operands) + 1:
        raise ValueError("native indexed Map requires one affine map per operand and result")
    result_map = maps[-1]
    if not _is_identity_result_map(result_map, len(coordinate)):
        raise ValueError("native indexed-Map Contract composition requires identity result indexing")
    candidates: list[tuple[int, ...]] = []
    for operand, encoded_map in zip(producer.operands, maps[:-1], strict=True):
        if target != operand and target not in _native_value_ancestors(operand, producer_by_result):
            continue
        candidates.append(
            _trace_native_map_coordinate(
                operand,
                _evaluate_encoded_affine_map(encoded_map, coordinate, len(coordinate)),
                target,
                producer_by_result,
            )
        )
    if len(candidates) != 1:
        raise ValueError("native indexed-Map DAG must have one provenance path per Contract operand")
    return candidates[0]


def _native_value_ancestors(value: int, producer_by_result: dict[int, ShuttleOperationRecord]) -> frozenset[int]:
    producer = producer_by_result.get(value)
    if producer is None:
        return frozenset()
    return frozenset(producer.operands).union(
        *(_native_value_ancestors(operand, producer_by_result) for operand in producer.operands)
    )


def _is_identity_result_map(encoded: object, rank: int) -> bool:
    return isinstance(encoded, dict) and encoded == {
        "kind": "affine_map",
        "dimensions": rank,
        "symbols": 0,
        "results": [{"kind": "dimension", "position": position} for position in range(rank)],
    }


def _substitute_scalar_inputs(
    expression: ScalarExpression,
    replacements: dict[str, ScalarExpression],
) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        try:
            return replacements[expression.input_name]
        except KeyError as error:
            raise ValueError("native scalar Map references an unavailable composed operand") from error
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_substitute_scalar_inputs(operand, replacements) for operand in expression.operands),
        input_name=expression.input_name,
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def _verify_composed_native_fold_semantics(
    native_group: tuple[ShuttleOperationRecord, ...], operation: FoldPrimitive
) -> None:
    fold = native_group[-1]
    if fold.name != "shuttle.fold" or len(fold.operands) != 2:
        raise ValueError("composed native Fold must end in one data-plus-initializer Fold")
    initializer_group = native_group[:-1]
    if len(initializer_group) != 1 or len(initializer_group[0].results) != 1:
        raise ValueError("composed native Fold requires one scalar initializer source")
    initializer = scalar_expression_from_source_map(initializer_group[0], ())
    expected = scalar_constant(float("-inf") if operation.reducer is FoldReducer.MAXIMUM else 0.0)
    if operation.output.dtype is DType.BF16:
        expected = scalar_convert(
            expected,
            result_dtype=DType.BF16,
            semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        )
    if (
        not _scalar_expressions_equivalent(initializer, expected, {})
        or fold.operands[1] != initializer_group[0].results[0].id
    ):
        raise ValueError("native Fold initializer disagrees with its canonical reducer identity")
    _verify_native_operation_semantics(fold, operation)


def _verify_native_operation_semantics(native: ShuttleOperationRecord, operation: object) -> None:
    if isinstance(operation, ScanPrimitive):
        _verify_native_scan_semantics(native, operation)
        return
    if isinstance(operation, ContractPrimitive):
        encoded_maps = native.attributes.get("indexing_maps")
        if not isinstance(encoded_maps, list) or len(encoded_maps) != len(operation.inputs) + 1:
            raise ValueError("native Contract requires one affine indexing map per operand and result")
        resolved_input_axes = []
        for input_index, value in enumerate(operation.inputs):
            aliases = {
                mapping.operand_axis: mapping.domain_axis
                for mapping in operation.index_maps_for_input(input_index)
                if mapping.divisor == 1 and mapping.offset == 0 and mapping.modulus is None
            }
            if len(aliases) != len(operation.index_maps_for_input(input_index)):
                raise ValueError("native canonical Contract binding requires projected affine axis aliases")
            resolved_input_axes.append(tuple(aliases.get(axis, axis) for axis in value.axes))
        values_and_axes = (
            *((value, axes) for value, axes in zip(operation.inputs, resolved_input_axes, strict=True)),
            (operation.output, operation.output.axes),
        )
        axis_by_dimension: dict[int, object] = {}
        dimensions: int | None = None
        for encoded_map, (value, resolved_axes) in zip(encoded_maps, values_and_axes, strict=True):
            if (
                not isinstance(encoded_map, dict)
                or encoded_map.get("kind") != "affine_map"
                or encoded_map.get("symbols") != 0
                or not isinstance(encoded_map.get("dimensions"), int)
                or not isinstance(encoded_map.get("results"), list)
            ):
                raise ValueError("native Contract indexing must use structured symbol-free affine maps")
            if dimensions is None:
                dimensions = encoded_map["dimensions"]
            elif dimensions != encoded_map["dimensions"]:
                raise ValueError("native Contract indexing maps must share one iteration domain")
            results = encoded_map["results"]
            if len(results) != len(value.axes):
                raise ValueError("native Contract indexing-map rank disagrees with canonical operand axes")
            for result, axis in zip(results, resolved_axes, strict=True):
                if not isinstance(result, dict) or set(result) != {"kind", "position"} or result["kind"] != "dimension":
                    raise ValueError("native canonical algebra binding supports projected Contract dimensions only")
                position = result["position"]
                if not isinstance(position, int) or isinstance(position, bool):
                    raise ValueError("native Contract dimension positions must be integers")
                previous = axis_by_dimension.setdefault(position, axis)
                if previous != axis:
                    raise ValueError("native Contract dimension aliases disagree with canonical axes")
        if set(axis_by_dimension.values()) != set(operation.output.axes) | set(operation.reduction_axes):
            raise ValueError("native Contract iteration dimensions disagree with canonical reduction axes")
        accumulator_types = native.attributes.get("accumulator_types")
        if accumulator_types is not None:
            if not isinstance(accumulator_types, list) or len(accumulator_types) != 1:
                raise ValueError("native Contract requires one accumulator type")
            accumulator_type = accumulator_types[0]
            if (
                not isinstance(accumulator_type, dict)
                or accumulator_type.get("kind") != "scalar"
                or _native_element_type(accumulator_type.get("element_type")) != operation.accumulation_dtype.value
            ):
                raise ValueError("native and canonical Contract accumulation dtypes disagree")
        return
    if isinstance(operation, FoldPrimitive):
        input_axes = operation.input.axes
        expected_dimensions = tuple(input_axes.index(axis) for axis in operation.reduction_axes)
        dimensions = native.attributes.get("reduction_dimensions")
        if dimensions != list(expected_dimensions):
            raise ValueError("native and canonical Fold reduction dimensions disagree")
        reducer_operations = native.body().operations
        reducer_names = tuple(candidate.name for candidate in reducer_operations)
        expected_reducer = {
            FoldReducer.SUM: "arith.addf",
            FoldReducer.MAXIMUM: "arith.maximumf",
        }[operation.reducer]
        if expected_reducer not in reducer_names:
            raise ValueError("native and canonical Fold reducer semantics disagree")
        return
    if isinstance(operation, MapPrimitive):
        decoded = scalar_expression_from_source_map(native, tuple(value.name for value in operation.inputs))
        input_dtypes = {value.name: value.dtype for value in operation.inputs}
        if not _scalar_expressions_equivalent(decoded, operation.expression, input_dtypes):
            raise ValueError("native and canonical scalar Map ASTs disagree")
        return
    if isinstance(operation, SelectionPrimitive):
        dimension = native.attributes.get("dimension")
        expected_dimension = operation.score.axes.index(operation.dimension)
        if dimension != expected_dimension or native.attributes.get("count") != operation.count:
            raise ValueError("native and canonical Selection dimension/count disagree")
        if native.attributes.get("stable") != operation.stable:
            raise ValueError("native and canonical Selection stability disagree")
        return
    if isinstance(operation, RelationPrimitive):
        if native.name == "shuttle.map":
            _verify_native_indexing_map_relation(native, operation)
            return
        semantics, raw_parameters = _native_relation_semantics_and_parameters(native)
        if semantics == "gather":
            _verify_native_gather_relation(raw_parameters, operation)
            return
        if semantics == "indexed_contributions":
            _verify_native_indexed_contributions_relation(raw_parameters, operation)
            return
        semantics, parameters = source_relation_semantics(native)
        if semantics == "iota":
            if (
                len(operation.inputs) != 0
                or len(operation.outputs) != 1
                or len(operation.resolved_result_expressions) != 1
                or operation.resolved_result_sources != (None,)
                or operation.predicate is not None
                or len(parameters) != 1
            ):
                raise ValueError("native iota Relation cannot bind this canonical Relation")
            dimension = parameters[0]
            if dimension < 0 or dimension >= len(operation.domain_axes):
                raise ValueError("native iota dimension is outside its result domain")
            expression = operation.resolved_result_expressions[0]
            assert expression is not None
            for coordinate in product(*(range(axis.extent) for axis in operation.domain_axes)):
                bindings = dict(zip(operation.domain_axes, coordinate, strict=True))
                if _evaluate_index_expression(expression, bindings) != coordinate[dimension]:
                    raise ValueError("native iota dimension disagrees with the canonical result expression")
            return
        if (
            len(operation.inputs) != 1
            or len(operation.outputs) != 1
            or operation.predicate is not None
            or operation.resolved_result_expressions != (None,)
            or operation.resolved_result_sources != (operation.inputs[0],)
        ):
            raise ValueError(
                "native Relation semantic binding currently requires one unpredicated indexed-source result"
            )
        operand = operation.inputs[0]
        if semantics == "broadcast_in_dim":
            if len(parameters) != len(operand.axes) or any(
                dimension < 0 or dimension >= len(operation.domain_axes) for dimension in parameters
            ):
                raise ValueError("native broadcast Relation has invalid result-dimension parameters")
            expected_indices = tuple(
                index_constant(0) if axis.extent == 1 else index_axis(operation.domain_axes[dimension])
                for axis, dimension in zip(operand.axes, parameters, strict=True)
            )
        elif semantics == "transpose":
            if sorted(parameters) != list(range(len(operand.axes))) or len(operation.domain_axes) != len(operand.axes):
                raise ValueError("native transpose Relation requires one exact permutation")
            expected_indices = tuple(
                index_axis(operation.domain_axes[parameters.index(operand_dimension)])
                for operand_dimension in range(len(operand.axes))
            )
        elif semantics == "reshape":
            if parameters:
                raise ValueError("native reshape Relation does not accept integer parameters")
            if len(operation.input_indexing) != 1 or not _reshape_indices_match(
                operation.input_indexing[0].indices,
                operation.domain_axes,
                operand.shape,
            ):
                raise ValueError("native reshape Relation disagrees with canonical row-major IndexExpressions")
            return
        else:
            raise ValueError(
                f"native Relation semantics {semantics!r} do not yet have a canonical IndexExpression translation"
            )
        if len(operation.input_indexing) != 1 or operation.input_indexing[0].indices != expected_indices:
            raise ValueError("native Relation parameters disagree with canonical operand IndexExpressions")
        return
    raise ValueError(f"unsupported native canonical algebra semantics: {type(operation).__name__}")


def _verify_native_scan_semantics(native: ShuttleOperationRecord, operation: ScanPrimitive) -> None:
    """Verify one typed native ordered recurrence against canonical Scan algebra."""

    def integer(name: str) -> int:
        value = native.attributes.get(name)
        if type(value) is not int:
            raise ValueError(f"native Scan {name} must be an integer")
        return value

    def integers(name: str) -> tuple[int, ...]:
        value = native.attributes.get(name)
        if not isinstance(value, list) or any(type(item) is not int for item in value):
            raise ValueError(f"native Scan {name} must be an integer array")
        return tuple(value)

    direction = native.attributes.get("direction")
    expected_direction = {
        "forward": "#shuttle.scan_direction<forward>",
        "reverse": "#shuttle.scan_direction<reverse>",
    }[operation.direction.value]
    if direction != {"kind": "mlir_attribute", "value": expected_direction}:
        raise ValueError("native and canonical Scan traversal directions disagree")
    if integer("extent") != operation.axis.extent:
        raise ValueError("native and canonical Scan ordered extents disagree")

    canonical_inputs = primitive_inputs(operation)
    canonical_outputs = primitive_outputs(operation)
    input_position = {value: index for index, value in enumerate(canonical_inputs)}
    output_position = {value: index for index, value in enumerate(canonical_outputs)}
    body_input_position = {value: index for index, value in enumerate(operation.body.inputs)}
    body_output_position = {value: index for index, value in enumerate(operation.body.outputs)}
    expected = {
        "state_operands": tuple(input_position[binding.initial] for binding in operation.state),
        "state_body_arguments": tuple(body_input_position[binding.body_argument] for binding in operation.state),
        "state_body_results": tuple(body_output_position[binding.body_result] for binding in operation.state),
        "state_results": tuple(output_position[binding.final] for binding in operation.state),
        "invariant_operands": tuple(input_position[binding.source] for binding in operation.invariants),
        "invariant_body_arguments": tuple(
            body_input_position[binding.body_argument] for binding in operation.invariants
        ),
        "scanned_operands": tuple(input_position[binding.source] for binding in operation.scanned_inputs),
        "scanned_body_arguments": tuple(
            body_input_position[binding.body_argument] for binding in operation.scanned_inputs
        ),
        "scanned_dimensions": tuple(
            binding.indexing.domain_axes.index(operation.axis) for binding in operation.scanned_inputs
        ),
        "emitted_body_results": tuple(
            body_output_position[binding.body_result] for binding in operation.emitted_outputs
        ),
        "emitted_results": tuple(output_position[binding.destination] for binding in operation.emitted_outputs),
        "emitted_dimensions": tuple(
            binding.indexing.domain_axes.index(operation.axis) for binding in operation.emitted_outputs
        ),
    }
    for name, expected_values in expected.items():
        if integers(name) != expected_values:
            raise ValueError(f"native and canonical Scan {name} disagree")

    native_body, _ = _source_operations(native)
    _verify_tensor_program_binding(native, native_body, operation.body)


def _verify_native_gather_relation(parameters: tuple[object, ...], operation: RelationPrimitive) -> None:
    """Verify the supported full-slice Gather as typed dynamic indexing."""
    if len(parameters) != 1 or not isinstance(parameters[0], dict):
        raise ValueError("native gather Relation requires one structured metadata record")
    metadata = parameters[0]
    expected_fields = {
        "offset_dims",
        "collapsed_slice_dims",
        "operand_batching_dims",
        "start_indices_batching_dims",
        "start_index_map",
        "index_vector_dim",
        "slice_sizes",
        "indices_are_sorted",
    }
    if set(metadata) != expected_fields:
        raise ValueError("native gather Relation metadata fields are incomplete")
    if (
        len(operation.inputs) != 2
        or len(operation.outputs) != 1
        or operation.predicate is not None
        or operation.resolved_result_expressions != (None,)
        or operation.resolved_result_sources != (operation.inputs[0],)
    ):
        raise ValueError("native gather Relation cannot bind this canonical Relation")
    operand, indices = operation.inputs
    if indices.dtype is not DType.INT32:
        raise ValueError("native gather Relation requires INT32 indices")

    def dimensions(name: str) -> tuple[int, ...]:
        encoded = metadata[name]
        if not isinstance(encoded, list) or any(type(value) is not int for value in encoded):
            raise ValueError(f"native gather {name} must be an integer array")
        return tuple(encoded)

    offset_dimensions = dimensions("offset_dims")
    collapsed_dimensions = dimensions("collapsed_slice_dims")
    operand_batching_dimensions = dimensions("operand_batching_dims")
    indices_batching_dimensions = dimensions("start_indices_batching_dims")
    indexed_dimensions = dimensions("start_index_map")
    slice_sizes = dimensions("slice_sizes")
    index_vector_dimension = metadata["index_vector_dim"]
    if type(index_vector_dimension) is not int:
        raise ValueError("native gather index_vector_dim must be an integer")
    if (
        len(collapsed_dimensions) != 1
        or collapsed_dimensions != indexed_dimensions
        or len(operand_batching_dimensions) != len(indices_batching_dimensions)
        or len(slice_sizes) != len(operand.axes)
        or not 0 <= index_vector_dimension < len(indices.axes)
        or indices.shape[index_vector_dimension] != 1
    ):
        raise ValueError("native gather Relation is outside the supported full-slice form")
    collapsed_dimension = collapsed_dimensions[0]
    if collapsed_dimension in operand_batching_dimensions:
        raise ValueError("native gather indexed and batched operand dimensions must differ")
    if any(
        not 0 <= dimension < len(operand.axes)
        for dimension in (*collapsed_dimensions, *operand_batching_dimensions, *indexed_dimensions)
    ) or any(not 0 <= dimension < len(indices.axes) for dimension in indices_batching_dimensions):
        raise ValueError("native gather Relation dimension is outside its operand")
    if any(
        operand.shape[operand_dimension] != indices.shape[index_dimension]
        for operand_dimension, index_dimension in zip(
            operand_batching_dimensions, indices_batching_dimensions, strict=True
        )
    ):
        raise ValueError("native gather Relation batching extents disagree")
    if any(
        slice_size != (1 if dimension == collapsed_dimension or dimension in operand_batching_dimensions else extent)
        for dimension, (slice_size, extent) in enumerate(zip(slice_sizes, operand.shape, strict=True))
    ):
        raise ValueError("native gather Relation does not take full non-indexed slices")

    output = operation.outputs[0]
    if any(not 0 <= dimension < len(output.axes) for dimension in offset_dimensions):
        raise ValueError("native gather offset dimension is outside its output")
    batch_output_dimensions = tuple(
        dimension for dimension in range(len(output.axes)) if dimension not in offset_dimensions
    )
    index_batch_dimensions = tuple(
        dimension for dimension in range(len(indices.axes)) if dimension != index_vector_dimension
    )
    if len(offset_dimensions) != len(operand.axes) - 1 - len(operand_batching_dimensions) or tuple(
        output.shape[dimension] for dimension in batch_output_dimensions
    ) != tuple(indices.shape[dimension] for dimension in index_batch_dimensions):
        raise ValueError("native gather Relation output shape disagrees with its indices")

    batch_indices = tuple(index_axis(output.axes[dimension]) for dimension in batch_output_dimensions)
    dynamic_index_coordinates = list(batch_indices)
    dynamic_index_coordinates.insert(index_vector_dimension, index_constant(0))
    dynamic_index = index_tensor_element(indices, *dynamic_index_coordinates)
    operand_batch_by_indices = dict(zip(operand_batching_dimensions, indices_batching_dimensions, strict=True))
    output_batch_by_indices = dict(zip(index_batch_dimensions, batch_output_dimensions, strict=True))
    offset_by_operand = dict(
        zip(
            (
                dimension
                for dimension in range(len(operand.axes))
                if dimension != collapsed_dimension and dimension not in operand_batching_dimensions
            ),
            offset_dimensions,
            strict=True,
        )
    )
    operand_indices = tuple(
        (
            dynamic_index
            if dimension == collapsed_dimension
            else (
                index_axis(output.axes[output_batch_by_indices[operand_batch_by_indices[dimension]]])
                if dimension in operand_batch_by_indices
                else index_axis(output.axes[offset_by_operand[dimension]])
            )
        )
        for dimension in range(len(operand.axes))
    )
    expected_indexing = (
        OperandIndexing(operand, output.axes, operand_indices),
        OperandIndexing(indices, output.axes, tuple(dynamic_index_coordinates)),
    )
    if operation.input_indexing != expected_indexing:
        raise ValueError("native gather metadata disagrees with canonical dynamic IndexExpressions")


def _verify_native_indexed_contributions_relation(parameters: tuple[object, ...], operation: RelationPrimitive) -> None:
    """Verify a source-ordered indexed-contribution Relation without naming its workload."""
    if len(parameters) != 1 or not isinstance(parameters[0], dict):
        raise ValueError("native indexed-contributions Relation requires one structured metadata record")
    metadata = parameters[0]
    expected_fields = {
        "output_shape",
        "update_window_dims",
        "inserted_window_dims",
        "input_batching_dims",
        "scatter_indices_batching_dims",
        "scatter_dims_to_operand_dims",
        "index_vector_dim",
        "indices_are_sorted",
        "unique_indices",
        "conflict_resolution",
    }
    if set(metadata) != expected_fields:
        raise ValueError("native indexed-contributions Relation metadata fields are incomplete")
    if (
        len(operation.inputs) != 2
        or len(operation.outputs) != 1
        or operation.resolved_result_expressions != (None,)
        or operation.resolved_result_sources != (operation.inputs[0],)
        or operation.predicate is None
        or operation.order.value != "source_ordered"
    ):
        raise ValueError("native indexed-contributions Relation cannot bind this canonical Relation")
    updates, indices = operation.inputs
    output = operation.outputs[0]
    if indices.dtype is not DType.INT32:
        raise ValueError("native indexed-contributions Relation requires INT32 indices")

    def dimensions(name: str) -> tuple[int, ...]:
        encoded = metadata[name]
        if not isinstance(encoded, list) or any(type(value) is not int for value in encoded):
            raise ValueError(f"native indexed-contributions {name} must be an integer array")
        return tuple(encoded)

    output_shape = dimensions("output_shape")
    window_dimensions = dimensions("update_window_dims")
    inserted_dimensions = dimensions("inserted_window_dims")
    operand_batching_dimensions = dimensions("input_batching_dims")
    indices_batching_dimensions = dimensions("scatter_indices_batching_dims")
    scattered_dimensions = dimensions("scatter_dims_to_operand_dims")
    index_vector_dimension = metadata["index_vector_dim"]
    conflict_resolution = metadata["conflict_resolution"]
    if type(index_vector_dimension) is not int or conflict_resolution != "source_ordered_add":
        raise ValueError("native indexed-contributions Relation requires source-ordered additive conflicts")
    if (
        len(inserted_dimensions) != 1
        or inserted_dimensions != scattered_dimensions
        or len(operand_batching_dimensions) != len(indices_batching_dimensions)
        or not 0 <= index_vector_dimension < len(indices.axes)
        or indices.shape[index_vector_dimension] != 1
        or output_shape != output.shape[: len(output_shape)]
    ):
        raise ValueError("native indexed-contributions Relation is outside the supported full-window form")
    route_dimensions = tuple(dimension for dimension in range(len(updates.axes)) if dimension not in window_dimensions)
    index_dimensions = tuple(dimension for dimension in range(len(indices.axes)) if dimension != index_vector_dimension)
    if len(route_dimensions) != len(index_dimensions) or tuple(
        updates.shape[dimension] for dimension in route_dimensions
    ) != tuple(indices.shape[dimension] for dimension in index_dimensions):
        raise ValueError("native indexed-contributions route shape disagrees with its indices")
    batch_update_to_operand = tuple(
        (route_dimensions[index_dimension], operand_dimension)
        for operand_dimension, index_dimension in zip(
            operand_batching_dimensions, indices_batching_dimensions, strict=True
        )
    )
    reduction_dimensions = tuple(
        dimension for dimension in route_dimensions if dimension not in dict(batch_update_to_operand)
    )
    if output.axes != (
        *operation.domain_axes[: len(output_shape)],
        *tuple(updates.axes[dimension] for dimension in reduction_dimensions),
    ):
        raise ValueError("native indexed-contributions output axes disagree with its route domain")
    scatter_dimension = inserted_dimensions[0]
    window_operand_dimensions = tuple(
        dimension
        for dimension in range(len(output_shape))
        if dimension != scatter_dimension and dimension not in operand_batching_dimensions
    )
    if len(window_dimensions) != len(window_operand_dimensions) or tuple(
        updates.shape[dimension] for dimension in window_dimensions
    ) != tuple(output_shape[dimension] for dimension in window_operand_dimensions):
        raise ValueError("native indexed-contributions updates do not cover full destination windows")
    output_axis_by_update = {
        update_dimension: operation.domain_axes[operand_dimension]
        for update_dimension, operand_dimension in (
            *zip(window_dimensions, window_operand_dimensions, strict=True),
            *batch_update_to_operand,
        )
    }
    route_axis_by_update = {
        dimension: operation.domain_axes[len(output_shape) + ordinal]
        for ordinal, dimension in enumerate(reduction_dimensions)
    }
    update_indices = tuple(
        (
            index_axis(output_axis_by_update[dimension])
            if dimension in output_axis_by_update
            else index_axis(route_axis_by_update[dimension])
        )
        for dimension in range(len(updates.axes))
    )
    operand_batch_by_index = {
        index_dimension: operand_dimension
        for operand_dimension, index_dimension in zip(
            operand_batching_dimensions, indices_batching_dimensions, strict=True
        )
    }
    scatter_index_expressions = tuple(
        (
            index_axis(operation.domain_axes[operand_batch_by_index[index_dimension]])
            if index_dimension in operand_batch_by_index
            else index_axis(route_axis_by_update[route_dimensions[index_dimension]])
        )
        for index_dimension in range(len(index_dimensions))
    )
    index_coordinates = list(scatter_index_expressions)
    index_coordinates.insert(index_vector_dimension, index_constant(0))
    dynamic_destination = index_tensor_element(indices, *index_coordinates)
    expected_indexing = (
        OperandIndexing(updates, operation.domain_axes, update_indices),
        OperandIndexing(indices, operation.domain_axes, tuple(index_coordinates)),
    )
    expected_predicate = index_compare(
        IndexPredicateKind.EQUAL,
        index_axis(operation.domain_axes[scatter_dimension]),
        dynamic_destination,
    )
    if operation.input_indexing != expected_indexing or operation.predicate != expected_predicate:
        raise ValueError("native indexed-contributions metadata disagrees with canonical IndexExpressions")


def _scalar_expressions_equivalent(
    left: ScalarExpression,
    right: ScalarExpression,
    input_dtypes: dict[str, DType],
) -> bool:
    """Compare scalar ASTs modulo observationally exact widening semantics."""
    left = _normalize_scalar_identity(left)
    right = _normalize_scalar_identity(right)
    if (
        left.kind != right.kind
        or left.input_name != right.input_name
        or left.constant != right.constant
        or left.result_dtype != right.result_dtype
        or len(left.operands) != len(right.operands)
    ):
        return False
    if left.conversion_semantics != right.conversion_semantics and not (
        {left.conversion_semantics, right.conversion_semantics}
        == {ScalarConversionSemantics.EXACT, ScalarConversionSemantics.ROUND_NEAREST_EVEN}
        and left.result_dtype is DType.FP32
        and len(left.operands) == 1
        and left.operands[0].input_name is not None
        and input_dtypes.get(left.operands[0].input_name) is DType.BF16
    ):
        return False
    return all(
        _scalar_expressions_equivalent(left_operand, right_operand, input_dtypes)
        for left_operand, right_operand in zip(left.operands, right.operands, strict=True)
    )


def _normalize_scalar_identity(expression: ScalarExpression) -> ScalarExpression:
    operands = tuple(_normalize_scalar_identity(operand) for operand in expression.operands)
    if (
        expression.kind is ScalarExpressionKind.CONVERT
        and len(operands) == 1
        and operands[0].kind is ScalarExpressionKind.CONSTANT
        and operands[0].constant in (0.0, float("inf"), float("-inf"))
    ):
        # Reducer identities are exactly representable in every supported
        # floating-point type. Native MLIR may spell the result-type-implied
        # conversion explicitly while the TensorProgram Fold stores the
        # mathematical identity as an untyped scalar literal.
        return operands[0]
    if expression.kind is ScalarExpressionKind.MULTIPLY and len(operands) == 2:
        if _is_scalar_one(operands[0]):
            return operands[1]
        if _is_scalar_one(operands[1]):
            return operands[0]
    return ScalarExpression(
        kind=expression.kind,
        operands=operands,
        input_name=expression.input_name,
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def _is_scalar_one(expression: ScalarExpression) -> bool:
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return expression.constant == 1.0
    return (
        expression.kind is ScalarExpressionKind.CONVERT
        and len(expression.operands) == 1
        and _is_scalar_one(expression.operands[0])
    )


def _verify_native_indexing_map_relation(native: ShuttleOperationRecord, operation: RelationPrimitive) -> None:
    """Canonicalize a native indexed Map as an algebraic Relation."""
    if (
        native.attributes.get("semantics")
        not in (
            {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<transpose>"},
            {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<broadcast_in_dim>"},
            {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<reshape>"},
        )
        or len(operation.inputs) != 1
        or len(operation.outputs) != 1
        or operation.predicate is not None
        or operation.resolved_result_expressions != (None,)
        or operation.resolved_result_sources != (operation.inputs[0],)
        or len(operation.input_indexing) != 1
    ):
        raise ValueError("native layout Map cannot be canonicalized as this Relation")
    encoded_maps = native.attributes.get("indexing_maps")
    if not isinstance(encoded_maps, list) or len(encoded_maps) != 2:
        raise ValueError("native layout Map requires operand and result affine indexing maps")
    operand_map, result_map = encoded_maps
    domain_rank = len(operation.domain_axes)
    domain_shape = tuple(axis.extent for axis in operation.domain_axes)
    if prod(domain_shape) > 1_000_000:
        raise ValueError("native indexed Map exceeds the bounded exact IndexExpression verifier")
    for coordinate in product(*(range(extent) for extent in domain_shape)):
        if _evaluate_encoded_affine_map(result_map, coordinate, domain_rank) != coordinate:
            raise ValueError("native indexed Map result indexing is not the canonical domain identity")
        expected = tuple(
            _evaluate_index_expression(index, dict(zip(operation.domain_axes, coordinate, strict=True)))
            for index in operation.input_indexing[0].indices
        )
        if _evaluate_encoded_affine_map(operand_map, coordinate, domain_rank) != expected:
            raise ValueError("native indexed Map operand indexing disagrees with canonical Relation")


def _evaluate_encoded_affine_map(encoded_map: object, coordinate: tuple[int, ...], domain_rank: int) -> tuple[int, ...]:
    if (
        not isinstance(encoded_map, dict)
        or encoded_map.get("kind") != "affine_map"
        or encoded_map.get("symbols") != 0
        or encoded_map.get("dimensions") != domain_rank
        or not isinstance(encoded_map.get("results"), list)
    ):
        raise ValueError("native indexing must use a structured symbol-free affine map")
    return tuple(_evaluate_encoded_affine_expression(result, coordinate) for result in encoded_map["results"])


def _evaluate_encoded_affine_expression(encoded: object, coordinate: tuple[int, ...]) -> int:
    if not isinstance(encoded, dict):
        raise ValueError("native affine expression must be a structured record")
    kind = encoded.get("kind")
    if kind == "dimension":
        position = encoded.get("position")
        if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < len(coordinate):
            raise ValueError("native affine dimension is outside its map domain")
        return coordinate[position]
    if kind == "constant":
        value = encoded.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("native affine constant must be an integer")
        return value
    left = _evaluate_encoded_affine_expression(encoded.get("left"), coordinate)
    right = _evaluate_encoded_affine_expression(encoded.get("right"), coordinate)
    if kind == "add":
        return left + right
    if kind == "multiply":
        return left * right
    if right <= 0 and kind in ("floor_divide", "ceil_divide", "modulo"):
        raise ValueError("native affine division requires a positive divisor")
    if kind == "floor_divide":
        return left // right
    if kind == "ceil_divide":
        return -(-left // right)
    if kind == "modulo":
        return left % right
    raise ValueError(f"unsupported native affine expression {kind!r}")


def _verify_composed_native_relation_semantics(
    native_group: tuple[ShuttleOperationRecord, ...],
    operation: RelationPrimitive,
    type_by_value: dict[int, Any],
) -> None:
    """Prove a unary native Relation chain equals one canonical index relation."""
    if (
        len(operation.inputs) != 1
        or len(operation.outputs) != 1
        or operation.predicate is not None
        or operation.resolved_result_expressions != (None,)
        or operation.resolved_result_sources != (operation.inputs[0],)
        or len(operation.input_indexing) != 1
    ):
        raise ValueError("composed native Relation binding currently requires one unpredicated indexed-source result")
    if any(len(native.operands) != 1 or len(native.results) != 1 for native in native_group):
        raise ValueError("composed native Relation binding currently requires a unary SSA chain")
    for left, right in pairwise(native_group):
        if right.operands != (left.results[0].id,):
            raise ValueError("composed native Relation sources must form one exact SSA chain")

    domain_shape = tuple(axis.extent for axis in operation.domain_axes)
    element_count = 1
    for extent in domain_shape:
        element_count *= extent
    if element_count > 1_000_000:
        raise ValueError("composed native Relation exceeds the bounded exact IndexExpression verifier")

    for output_coordinate in product(*(range(extent) for extent in domain_shape)):
        native_coordinate = output_coordinate
        for native in reversed(native_group):
            semantics, parameters = source_relation_semantics(native)
            operand_shape = _native_value_type(type_by_value[native.operands[0]])[1]
            result_shape = _native_value_type(native.results[0].type)[1]
            if tuple(result_shape) != tuple(_native_value_type(type_by_value[native.results[0].id])[1]):
                raise ValueError("composed native Relation result type is inconsistent")
            if semantics == "broadcast_in_dim":
                if len(parameters) != len(operand_shape) or any(
                    dimension < 0 or dimension >= len(result_shape) for dimension in parameters
                ):
                    raise ValueError("native broadcast Relation has invalid result-dimension parameters")
                native_coordinate = tuple(native_coordinate[dimension] for dimension in parameters)
            elif semantics == "transpose":
                if len(parameters) != len(result_shape) or sorted(parameters) != list(range(len(result_shape))):
                    raise ValueError("native transpose Relation requires one exact permutation")
                native_coordinate = tuple(
                    native_coordinate[parameters.index(operand_dimension)]
                    for operand_dimension in range(len(operand_shape))
                )
            elif semantics == "reshape":
                if parameters:
                    raise ValueError("native reshape Relation does not accept integer parameters")
                native_coordinate = _row_major_reshape_coordinate(native_coordinate, result_shape, operand_shape)
            else:
                raise ValueError(
                    f"composed native Relation semantics {semantics!r} do not yet have an "
                    "IndexExpression composition rule"
                )
        bindings = dict(zip(operation.domain_axes, output_coordinate, strict=True))
        canonical_coordinate = tuple(
            _evaluate_index_expression(index, bindings) for index in operation.input_indexing[0].indices
        )
        if native_coordinate != canonical_coordinate:
            raise ValueError("composed native Relation chain disagrees with canonical operand IndexExpressions")


def _row_major_reshape_coordinate(
    result_coordinate: tuple[int, ...],
    result_shape: tuple[int, ...],
    operand_shape: tuple[int, ...],
) -> tuple[int, ...]:
    result_count = 1
    for extent in result_shape:
        result_count *= extent
    operand_count = 1
    for extent in operand_shape:
        operand_count *= extent
    if result_count != operand_count:
        raise ValueError("native reshape Relation must preserve its static element count")
    linear = 0
    for value, extent in zip(result_coordinate, result_shape, strict=True):
        linear = linear * extent + value
    operand_coordinate: list[int] = []
    for extent in reversed(operand_shape):
        linear, coordinate = divmod(linear, extent)
        operand_coordinate.append(coordinate)
    operand_coordinate.reverse()
    return tuple(operand_coordinate)


def _native_value_type(type_: Any) -> tuple[str, tuple[int, ...]]:
    return _native_element_type(type_.element_type), type_.shape


def _native_element_type(element_type: object) -> object:
    dtypes = {"f32": "fp32", "f64": "fp64", "i32": "int32", "i1": "bool"}
    return dtypes.get(element_type, element_type)


def _reshape_indices_match(
    indices: tuple[IndexExpression, ...],
    domain_axes: tuple[Any, ...],
    operand_shape: tuple[int, ...],
) -> bool:
    domain_shape = tuple(axis.extent for axis in domain_axes)
    element_count = 1
    for extent in domain_shape:
        element_count *= extent
    operand_count = 1
    for extent in operand_shape:
        operand_count *= extent
    if element_count != operand_count:
        raise ValueError("native reshape Relation must preserve its static element count")
    if element_count > 1_000_000:
        raise ValueError("native reshape Relation exceeds the bounded exact IndexExpression verifier")
    for coordinate in product(*(range(extent) for extent in domain_shape)):
        linear = 0
        for value, extent in zip(coordinate, domain_shape, strict=True):
            linear = linear * extent + value
        expected: list[int] = []
        residual = linear
        for extent in reversed(operand_shape):
            residual, index = divmod(residual, extent)
            expected.append(index)
        expected.reverse()
        bindings = dict(zip(domain_axes, coordinate, strict=True))
        if tuple(_evaluate_index_expression(index, bindings) for index in indices) != tuple(expected):
            return False
    return True


def _evaluate_index_expression(expression: IndexExpression, bindings: dict[Any, int]) -> int:
    if expression.kind is IndexExpressionKind.AXIS:
        return bindings[expression.axis]
    if expression.kind is IndexExpressionKind.CONSTANT:
        assert isinstance(expression.constant, int)
        return expression.constant
    if expression.kind is IndexExpressionKind.TENSOR_ELEMENT:
        raise ValueError("native static Relation verification cannot evaluate tensor-dependent IndexExpressions")
    if expression.kind is IndexExpressionKind.SELECT:
        raise ValueError("native static Relation verification cannot evaluate predicate-dependent IndexExpressions")
    left = _evaluate_index_expression(expression.operands[0], bindings)
    right = _evaluate_index_expression(expression.operands[1], bindings)
    operations = {
        IndexExpressionKind.ADD: lambda: left + right,
        IndexExpressionKind.SUBTRACT: lambda: left - right,
        IndexExpressionKind.MULTIPLY: lambda: left * right,
        IndexExpressionKind.FLOOR_DIVIDE: lambda: left // right,
        IndexExpressionKind.MODULO: lambda: left % right,
        IndexExpressionKind.MINIMUM: lambda: min(left, right),
        IndexExpressionKind.MAXIMUM: lambda: max(left, right),
    }
    try:
        return operations[expression.kind]()
    except KeyError as error:
        raise ValueError(f"unsupported canonical IndexExpression {expression.kind.value!r}") from error
