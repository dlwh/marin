# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical conformance of equivalent coarse and refined task/Event plans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from tile_lifetime.event_dataflow import (
    Coordinate,
    CoordinateDomain,
    EventTensorPlan,
    MemoryVisibility,
    TaskDependence,
    TaskFamily,
    TaskRelation,
)
from tile_lifetime.physical_task_contract import (
    BoundedTaskCoordinate,
    CanonicalTaskPlan,
    ParametricEventTensorPlan,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalTarget,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskCapability,
    TaskDependenceProgram,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TiledTask,
    canonical_fingerprint,
    physical_phase_iteration_space,
)
from tile_lifetime.tensor_program import (
    NumericalPolicy,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class TaskRefinementBinding:
    """Map every coordinate of one refined task onto a coarse task."""

    refined_task_id: int
    coarse_task_id: int
    coordinate_relation: TaskRelation

    def __post_init__(self) -> None:
        if self.refined_task_id < 0 or self.coarse_task_id < 0:
            raise ValueError("task refinement ids must be non-negative")


@dataclass(frozen=True)
class CanonicalTaskPlanRefinement:
    """Proof that a task/Event plan is an exact refinement of another plan.

    Algebra operation ordinals identify shared semantics. Coordinate relations
    identify how refined task instances quotient onto coarse task instances.
    Dependencies inside one coarse task are internalized; every remaining exact
    dependency and Event relation must project onto the coarse plan exactly.
    """

    algebra_fingerprint: str
    target: PhysicalTarget
    numerical_policy: PhysicalNumericalPolicy
    coarse_plan: CanonicalTaskPlan
    refined_plan: CanonicalTaskPlan
    bindings: tuple[TaskRefinementBinding, ...]
    coarse_events: tuple[EventTensorPlan, ...] = ()
    refined_events: tuple[EventTensorPlan, ...] = ()
    coarse_event_programs: tuple[ParametricEventTensorPlan, ...] = ()
    refined_event_programs: tuple[ParametricEventTensorPlan, ...] = ()

    def __post_init__(self) -> None:
        if len(self.algebra_fingerprint) != 64:
            raise ValueError("task refinement requires a canonical algebra fingerprint")
        if self.coarse_plan.bounded_buffers or self.refined_plan.bounded_buffers:
            raise ValueError("task refinement currently requires plans without bounded buffers")
        binding_by_refined = _binding_by_refined_task(self)
        coordinate_maps = _coordinate_maps(self, binding_by_refined)
        _verify_operation_partition(self, binding_by_refined)
        _verify_dependency_projection(self, binding_by_refined, coordinate_maps)
        _verify_parametric_dependency_projection(self, binding_by_refined)
        _verify_event_projection(self, binding_by_refined, coordinate_maps)
        _verify_parametric_event_projection(self, binding_by_refined)

    @property
    def quotient_record(self) -> dict[str, object]:
        """Return the shared algebra/task/Event identity after refinement."""
        return {
            "schema_version": 1,
            "boundary": "algebra_task_event_refinement",
            "algebra_fingerprint": self.algebra_fingerprint,
            "target": self.target.value,
            "numerical_policy": self.numerical_policy.value,
            "task_plan": self.coarse_plan.canonical_record,
            "events": _canonical_event_records(self.coarse_plan, self.coarse_events),
            "event_programs": tuple(_parametric_event_record(event) for event in self.coarse_event_programs),
        }

    @property
    def quotient_fingerprint(self) -> str:
        """Return the frontend-independent conformance fingerprint."""
        return canonical_fingerprint(self.quotient_record)

    @property
    def proof_record(self) -> dict[str, object]:
        """Return source-specific refinement evidence excluded from identity."""
        coarse_ids = {task.family: task.id for task in self.coarse_plan.tasks}
        refined_ids = {task.family: task.id for task in self.refined_plan.tasks}
        return {
            "schema_version": 1,
            "quotient_fingerprint": self.quotient_fingerprint,
            "coarse_task_plan_fingerprint": self.coarse_plan.fingerprint,
            "refined_task_plan_fingerprint": self.refined_plan.fingerprint,
            "bindings": tuple(
                {
                    "refined_task": binding.refined_task_id,
                    "coarse_task": binding.coarse_task_id,
                    "coordinate_pairs": tuple(
                        {"refined": pair.source, "coarse": pair.target} for pair in binding.coordinate_relation.pairs
                    ),
                }
                for binding in sorted(self.bindings, key=lambda binding: binding.refined_task_id)
            ),
            "internalized_dependencies": tuple(
                _dependence_record(dependence, refined_ids)
                for dependence in self.refined_plan.dependences
                if _mapped_task_ids(dependence, refined_ids, self.bindings)[0]
                == _mapped_task_ids(dependence, refined_ids, self.bindings)[1]
            ),
            "coarse_dependencies": tuple(
                _dependence_record(dependence, coarse_ids) for dependence in self.coarse_plan.dependences
            ),
            "internalized_parametric_events": tuple(
                event.dependence_program_index
                for event in self.refined_event_programs
                if _parametric_event_is_internal(event, self.refined_plan, self.bindings)
            ),
        }


def canonical_task_event_boundary_record(
    plan: CanonicalTaskPlan,
    events: tuple[EventTensorPlan, ...],
    event_programs: tuple[ParametricEventTensorPlan, ...] = (),
) -> dict[str, object]:
    """Return the source-name-free logical task/Event boundary.

    Concrete Event names and task-family display names are diagnostic source
    metadata. Task ids, exact coordinate relations, visibility, bounded
    generations, and runtime-parametric dependence programs remain semantic.
    """
    if len(event_programs) != len(plan.dependence_programs):
        raise ValueError("parametric Event plans must cover every task-dependence program exactly once")
    if tuple(sorted(event.dependence_program_index for event in event_programs)) != tuple(
        range(len(plan.dependence_programs))
    ):
        raise ValueError("parametric Event plans must reference every dependence program exactly once")
    _verify_events_cover_dependences(plan, events)
    return {
        "schema_version": 1,
        "boundary": "canonical_task_event",
        "task_plan_fingerprint": plan.fingerprint,
        "dependence_programs": plan.canonical_record["dependence_programs"],
        "events": _canonical_event_records(plan, events),
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
            for event in sorted(event_programs, key=lambda event: event.dependence_program_index)
        ),
    }


def canonical_task_event_boundary_fingerprint(
    plan: CanonicalTaskPlan,
    events: tuple[EventTensorPlan, ...],
    event_programs: tuple[ParametricEventTensorPlan, ...] = (),
) -> str:
    """Fingerprint the frontend-independent logical task/Event boundary."""
    return canonical_fingerprint(canonical_task_event_boundary_record(plan, events, event_programs))


def identity_task_refinement_binding(
    refined_task_id: int,
    coarse_task_id: int,
    *,
    refined_family: TaskFamily,
    coarse_family: TaskFamily,
) -> TaskRefinementBinding:
    """Construct a coordinate identity after checking rectangular domains."""
    refined_extents = tuple(axis.extent for axis in refined_family.axes)
    coarse_extents = tuple(axis.extent for axis in coarse_family.axes)
    if refined_extents != coarse_extents:
        raise ValueError("identity task refinement requires equal coordinate extents")
    return TaskRefinementBinding(
        refined_task_id,
        coarse_task_id,
        TaskRelation.from_pairs(
            refined_family,
            coarse_family,
            tuple(zip(refined_family.coordinates, coarse_family.coordinates, strict=True)),
        ),
    )


_CAPABILITY_BY_OPERATION_KIND = {
    TaskOperationKind.MAP: TaskCapability.SCALAR_MAP,
    TaskOperationKind.FOLD: TaskCapability.ASSOCIATIVE_FOLD,
    TaskOperationKind.CONTRACT: TaskCapability.MATRIX_CONTRACT,
    TaskOperationKind.SELECTION: TaskCapability.BOUNDED_SELECTION,
    TaskOperationKind.SCAN: TaskCapability.ORDERED_STATE,
    TaskOperationKind.RELATION: TaskCapability.INDEX_RELATION,
    TaskOperationKind.TRANSPORT: TaskCapability.DATA_TRANSPORT,
}


def whole_program_task_event_refinement(
    program: TensorProgram,
    refined_plan: CanonicalTaskPlan,
    *,
    target: PhysicalTarget,
    numerical_policy: PhysicalNumericalPolicy,
    refined_events: tuple[EventTensorPlan, ...],
    refined_event_programs: tuple[ParametricEventTensorPlan, ...],
) -> CanonicalTaskPlanRefinement:
    """Prove a task/Event plan refines one logical whole-program owner.

    This is a logical quotient only. It internalizes all task dependencies and
    Event allocations, but makes no claim that the resulting task is a legal or
    efficient physical schedule.
    """
    operation_map = _operation_map(refined_plan)
    expected_ordinals = tuple(range(len(program.operations)))
    if tuple(sorted(operation_map)) != expected_ordinals:
        raise ValueError("whole-program refinement requires every algebra operation exactly once")
    operation_kinds = tuple(operation_map[ordinal] for ordinal in expected_ordinals)
    phases = tuple(
        PhysicalPhase(
            ordinal,
            _CAPABILITY_BY_OPERATION_KIND[kind],
            (ordinal,),
            *physical_phase_iteration_space(program.operations[ordinal]),
        )
        for ordinal, kind in enumerate(operation_kinds)
    )
    producer_phase = {
        output.name: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    phase_edges = tuple(
        sorted(
            {
                PhysicalPhaseEdge(source, target, RelationAccessClass.AFFINE)
                for target, operation in enumerate(program.operations)
                for operand in primitive_inputs(operation)
                if (source := producer_phase.get(operand.name)) is not None
            }
        )
    )
    element_types = tuple(
        sorted({output.dtype.value for operation in program.operations for output in primitive_outputs(operation)})
    )
    coarse_family = TaskFamily("logical.whole_program", (), placement="device")
    coarse_task = TiledTask(
        0,
        coarse_family,
        expected_ordinals,
        operation_kinds,
        PhysicalTaskStructure(
            principal_kind=operation_kinds[0],
            phases=phases,
            phase_edges=phase_edges,
            owner_axes=(),
            streamed_axes=(),
            reduction_axes=(),
            contraction_axes=(),
            traversal=(
                TaskTraversal.SOURCE_ORDERED
                if program.numerical_policy is NumericalPolicy.SOURCE_ORDERED
                else TaskTraversal.UNORDERED
            ),
            element_types=element_types,
            layout_classes=("logical_quotient",),
            resources=TaskResourceEstimate(0, 0, 0),
        ),
        (),
    )
    coarse_plan = CanonicalTaskPlan((coarse_task,), ())
    bindings = tuple(
        TaskRefinementBinding(
            task.id,
            coarse_task.id,
            TaskRelation.from_pairs(
                task.family,
                coarse_family,
                tuple((coordinate, ()) for coordinate in task.family.coordinates),
            ),
        )
        for task in refined_plan.tasks
    )
    return CanonicalTaskPlanRefinement(
        program.semantic_fingerprint,
        target,
        numerical_policy,
        coarse_plan,
        refined_plan,
        bindings,
        coarse_events=(),
        refined_events=refined_events,
        coarse_event_programs=(),
        refined_event_programs=refined_event_programs,
    )


def _binding_by_refined_task(
    refinement: CanonicalTaskPlanRefinement,
) -> dict[int, TaskRefinementBinding]:
    bindings = {binding.refined_task_id: binding for binding in refinement.bindings}
    if len(bindings) != len(refinement.bindings):
        raise ValueError("each refined task requires exactly one refinement binding")
    expected = set(range(len(refinement.refined_plan.tasks)))
    if set(bindings) != expected:
        raise ValueError("task refinement bindings must cover every refined task")
    coarse_ids = set(range(len(refinement.coarse_plan.tasks)))
    if any(binding.coarse_task_id not in coarse_ids for binding in refinement.bindings):
        raise ValueError("task refinement binding references an unavailable coarse task")
    for refined_id, binding in bindings.items():
        refined_family = refinement.refined_plan.tasks[refined_id].family
        coarse_family = refinement.coarse_plan.tasks[binding.coarse_task_id].family
        if binding.coordinate_relation.source != refined_family:
            raise ValueError("task refinement coordinate source does not match its refined task")
        if binding.coordinate_relation.target != coarse_family:
            raise ValueError("task refinement coordinate target does not match its coarse task")
    return bindings


def _coordinate_maps(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
) -> dict[int, dict[Coordinate, Coordinate]]:
    result: dict[int, dict[Coordinate, Coordinate]] = {}
    covered_by_coarse: dict[int, set[Coordinate]] = defaultdict(set)
    for refined_id, binding in bindings.items():
        coordinate_map: dict[Coordinate, Coordinate] = {}
        for pair in binding.coordinate_relation.pairs:
            if pair.source in coordinate_map:
                raise ValueError("task refinement coordinate relation must be single-valued")
            coordinate_map[pair.source] = pair.target
            covered_by_coarse[binding.coarse_task_id].add(pair.target)
        refined_coordinates = set(refinement.refined_plan.tasks[refined_id].family.coordinates)
        if set(coordinate_map) != refined_coordinates:
            raise ValueError("task refinement coordinate relation must be total")
        result[refined_id] = coordinate_map
    for coarse_task in refinement.coarse_plan.tasks:
        if covered_by_coarse[coarse_task.id] != set(coarse_task.family.coordinates):
            raise ValueError("task refinement bindings must cover every coarse task coordinate")
    return result


def _operation_map(plan: CanonicalTaskPlan) -> dict[int, TaskOperationKind]:
    result: dict[int, TaskOperationKind] = {}
    for task in plan.tasks:
        if not task.operation_ordinals:
            continue
        for ordinal, kind in zip(task.operation_ordinals, task.operation_kinds, strict=True):
            if ordinal in result:
                raise ValueError("a task plan assigns one algebra operation to multiple tasks")
            result[ordinal] = kind
    return result


def _verify_operation_partition(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
) -> None:
    coarse_operations = _operation_map(refinement.coarse_plan)
    refined_operations = _operation_map(refinement.refined_plan)
    if refined_operations != coarse_operations:
        raise ValueError("coarse and refined task plans must cover the same typed algebra operations")
    covered_by_coarse: dict[int, set[int]] = defaultdict(set)
    for refined_task in refinement.refined_plan.tasks:
        coarse_id = bindings[refined_task.id].coarse_task_id
        coarse_ordinals = set(refinement.coarse_plan.tasks[coarse_id].operation_ordinals)
        if not set(refined_task.operation_ordinals) <= coarse_ordinals:
            raise ValueError("a refined task contains an operation outside its coarse task")
        covered_by_coarse[coarse_id].update(refined_task.operation_ordinals)
    for coarse_task in refinement.coarse_plan.tasks:
        if covered_by_coarse[coarse_task.id] != set(coarse_task.operation_ordinals):
            raise ValueError("refined tasks must partition every coarse task's operations")


def _projected_dependencies(
    plan: CanonicalTaskPlan,
    bindings: Mapping[int, TaskRefinementBinding],
    coordinate_maps: Mapping[int, Mapping[Coordinate, Coordinate]],
) -> dict[tuple[int, int, MemoryVisibility], set[tuple[Coordinate, Coordinate]]]:
    task_ids = {task.family: task.id for task in plan.tasks}
    projected: dict[tuple[int, int, MemoryVisibility], set[tuple[Coordinate, Coordinate]]] = defaultdict(set)
    for dependence in plan.dependences:
        source_refined = task_ids[_task_family(dependence.relation.source)]
        target_refined = task_ids[_task_family(dependence.relation.target)]
        source_coarse = bindings[source_refined].coarse_task_id
        target_coarse = bindings[target_refined].coarse_task_id
        if source_coarse == target_coarse:
            continue
        projected[(source_coarse, target_coarse, dependence.visibility)].update(
            (coordinate_maps[source_refined][pair.source], coordinate_maps[target_refined][pair.target])
            for pair in dependence.relation.pairs
        )
    return projected


def _coarse_dependencies(
    plan: CanonicalTaskPlan,
) -> dict[tuple[int, int, MemoryVisibility], set[tuple[Coordinate, Coordinate]]]:
    task_ids = {task.family: task.id for task in plan.tasks}
    result: dict[tuple[int, int, MemoryVisibility], set[tuple[Coordinate, Coordinate]]] = defaultdict(set)
    for dependence in plan.dependences:
        key = (
            task_ids[_task_family(dependence.relation.source)],
            task_ids[_task_family(dependence.relation.target)],
            dependence.visibility,
        )
        result[key].update((pair.source, pair.target) for pair in dependence.relation.pairs)
    return result


def _verify_dependency_projection(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
    coordinate_maps: Mapping[int, Mapping[Coordinate, Coordinate]],
) -> None:
    if _projected_dependencies(refinement.refined_plan, bindings, coordinate_maps) != _coarse_dependencies(
        refinement.coarse_plan
    ):
        raise ValueError("refined exact dependencies do not quotient to the coarse task plan")


def _verify_parametric_dependency_projection(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
) -> None:
    projected: list[dict[str, object]] = []
    for dependence in refinement.refined_plan.dependence_programs:
        source = bindings[dependence.source_task_id]
        target = bindings[dependence.target_task_id]
        if source.coarse_task_id == target.coarse_task_id:
            continue
        if not _is_coordinate_identity(source.coordinate_relation) or not _is_coordinate_identity(
            target.coordinate_relation
        ):
            raise ValueError("parametric task refinement currently requires identity endpoint coordinates")
        projected.append(
            _parametric_dependence_record(
                dependence,
                source_task_id=source.coarse_task_id,
                target_task_id=target.coarse_task_id,
            )
        )
    coarse = [
        _parametric_dependence_record(
            dependence,
            source_task_id=dependence.source_task_id,
            target_task_id=dependence.target_task_id,
        )
        for dependence in refinement.coarse_plan.dependence_programs
    ]
    if sorted(projected, key=canonical_fingerprint) != sorted(coarse, key=canonical_fingerprint):
        raise ValueError("refined parametric dependencies do not quotient to the coarse task plan")


def _verify_parametric_event_projection(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
) -> None:
    _verify_parametric_event_coverage(refinement.coarse_plan, refinement.coarse_event_programs)
    _verify_parametric_event_coverage(refinement.refined_plan, refinement.refined_event_programs)
    projected: list[dict[str, object]] = []
    for event in refinement.refined_event_programs:
        dependence = refinement.refined_plan.dependence_programs[event.dependence_program_index]
        source = bindings[dependence.source_task_id]
        target = bindings[dependence.target_task_id]
        if source.coarse_task_id == target.coarse_task_id:
            continue
        if not _is_coordinate_identity(source.coordinate_relation) or not _is_coordinate_identity(
            target.coordinate_relation
        ):
            raise ValueError("parametric Event refinement currently requires identity endpoint coordinates")
        projected.append(
            {
                "dependence": _parametric_dependence_record(
                    dependence,
                    source_task_id=source.coarse_task_id,
                    target_task_id=target.coarse_task_id,
                ),
                "event": _parametric_event_record(event, dependence_program_index=0),
            }
        )
    coarse = [
        {
            "dependence": _parametric_dependence_record(
                refinement.coarse_plan.dependence_programs[event.dependence_program_index],
                source_task_id=refinement.coarse_plan.dependence_programs[event.dependence_program_index].source_task_id,
                target_task_id=refinement.coarse_plan.dependence_programs[event.dependence_program_index].target_task_id,
            ),
            "event": _parametric_event_record(event, dependence_program_index=0),
        }
        for event in refinement.coarse_event_programs
    ]
    if sorted(projected, key=canonical_fingerprint) != sorted(coarse, key=canonical_fingerprint):
        raise ValueError("refined parametric Events do not quotient to the coarse Event plan")


def _verify_parametric_event_coverage(
    plan: CanonicalTaskPlan,
    events: tuple[ParametricEventTensorPlan, ...],
) -> None:
    if len(events) != len(plan.dependence_programs):
        raise ValueError("parametric Events must cover every dependence program exactly once")
    if tuple(sorted(event.dependence_program_index for event in events)) != tuple(range(len(plan.dependence_programs))):
        raise ValueError("parametric Events must cover every dependence program exactly once")


def _parametric_event_record(
    event: ParametricEventTensorPlan,
    *,
    dependence_program_index: int | None = None,
) -> dict[str, object]:
    return {
        "dependence_program": (
            event.dependence_program_index if dependence_program_index is None else dependence_program_index
        ),
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


def _parametric_event_is_internal(
    event: ParametricEventTensorPlan,
    plan: CanonicalTaskPlan,
    bindings: tuple[TaskRefinementBinding, ...],
) -> bool:
    dependence = plan.dependence_programs[event.dependence_program_index]
    coarse_by_refined = {binding.refined_task_id: binding.coarse_task_id for binding in bindings}
    return coarse_by_refined[dependence.source_task_id] == coarse_by_refined[dependence.target_task_id]


def _is_coordinate_identity(relation: TaskRelation) -> bool:
    return all(pair.source == pair.target for pair in relation.pairs)


def _parametric_dependence_record(
    dependence: TaskDependenceProgram,
    *,
    source_task_id: int,
    target_task_id: int,
) -> dict[str, object]:
    ordinals = tensor_program_value_ordinals(dependence.relation_program)

    def coordinate_record(coordinate: BoundedTaskCoordinate) -> dict[str, int]:
        return {
            "value": ordinals[coordinate.value],
            "lower_bound": coordinate.lower_bound,
            "upper_bound": coordinate.upper_bound,
        }

    return {
        "source_task": source_task_id,
        "target_task": target_task_id,
        "relation_program": dependence.relation_program.canonical_record,
        "runtime_bindings": tuple(
            {"input": ordinals[binding.input], "algebra_value": binding.algebra_value_ordinal}
            for binding in dependence.runtime_bindings
        ),
        "source_coordinates": tuple(coordinate_record(value) for value in dependence.source_coordinates),
        "target_coordinates": tuple(coordinate_record(value) for value in dependence.target_coordinates),
        "validity": None if dependence.validity is None else ordinals[dependence.validity],
        "visibility": {
            "scope": dependence.visibility.scope.value,
            "release_on_notify": dependence.visibility.release_on_notify,
            "acquire_before_consumer": dependence.visibility.acquire_before_consumer,
        },
    }


def _event_signature(plan: EventTensorPlan) -> tuple[object, ...]:
    return (
        plan.memory_scope,
        plan.generation_policy,
        plan.scheduling_mode,
        plan.visibility,
    )


def _verify_events_cover_dependences(plan: CanonicalTaskPlan, events: tuple[EventTensorPlan, ...]) -> None:
    if not events:
        return
    required = list(plan.dependences)
    for event in events:
        try:
            required.remove(event.required_dependence)
        except ValueError as error:
            raise ValueError("Event plans must cover declared task dependencies exactly once") from error
    if required:
        raise ValueError("Event plans must cover declared task dependencies exactly once")


def _projected_events(
    plan: CanonicalTaskPlan,
    events: tuple[EventTensorPlan, ...],
    bindings: Mapping[int, TaskRefinementBinding],
    coordinate_maps: Mapping[int, Mapping[Coordinate, Coordinate]],
) -> dict[tuple[int, int, tuple[object, ...]], set[tuple[Coordinate, Coordinate]]]:
    task_ids = {task.family: task.id for task in plan.tasks}
    projected: dict[tuple[int, int, tuple[object, ...]], set[tuple[Coordinate, Coordinate]]] = defaultdict(set)
    for event in events:
        relation = event.scheduled_dependence
        source_refined = task_ids[_task_family(relation.source)]
        target_refined = task_ids[_task_family(relation.target)]
        source_coarse = bindings[source_refined].coarse_task_id
        target_coarse = bindings[target_refined].coarse_task_id
        if source_coarse == target_coarse:
            continue
        projected[(source_coarse, target_coarse, _event_signature(event))].update(
            (coordinate_maps[source_refined][pair.source], coordinate_maps[target_refined][pair.target])
            for pair in relation.pairs
        )
    return projected


def _coarse_events(
    plan: CanonicalTaskPlan,
    events: tuple[EventTensorPlan, ...],
) -> dict[tuple[int, int, tuple[object, ...]], set[tuple[Coordinate, Coordinate]]]:
    task_ids = {task.family: task.id for task in plan.tasks}
    result: dict[tuple[int, int, tuple[object, ...]], set[tuple[Coordinate, Coordinate]]] = defaultdict(set)
    for event in events:
        relation = event.scheduled_dependence
        key = (
            task_ids[_task_family(relation.source)],
            task_ids[_task_family(relation.target)],
            _event_signature(event),
        )
        result[key].update((pair.source, pair.target) for pair in relation.pairs)
    return result


def _verify_event_projection(
    refinement: CanonicalTaskPlanRefinement,
    bindings: Mapping[int, TaskRefinementBinding],
    coordinate_maps: Mapping[int, Mapping[Coordinate, Coordinate]],
) -> None:
    if not refinement.coarse_events and not refinement.refined_events:
        return
    _verify_events_cover_dependences(refinement.coarse_plan, refinement.coarse_events)
    _verify_events_cover_dependences(refinement.refined_plan, refinement.refined_events)
    if _projected_events(
        refinement.refined_plan,
        refinement.refined_events,
        bindings,
        coordinate_maps,
    ) != _coarse_events(refinement.coarse_plan, refinement.coarse_events):
        raise ValueError("refined Event relations do not quotient to the coarse Event plan")


def _canonical_event_records(
    plan: CanonicalTaskPlan,
    events: tuple[EventTensorPlan, ...],
) -> tuple[dict[str, object], ...]:
    task_ids = {task.family: task.id for task in plan.tasks}
    records: list[dict[str, object]] = []
    for event in events:
        relation = event.scheduled_dependence
        records.append(
            {
                "source_task": task_ids[_task_family(relation.source)],
                "target_task": task_ids[_task_family(relation.target)],
                "pairs": tuple({"source": pair.source, "target": pair.target} for pair in relation.pairs),
                "memory_scope": event.memory_scope.value,
                "generation_policy": event.generation_policy.value,
                "scheduling_mode": None if event.scheduling_mode is None else event.scheduling_mode.value,
                "visibility": {
                    "scope": event.visibility.scope.value,
                    "release_on_notify": event.visibility.release_on_notify,
                    "acquire_before_consumer": event.visibility.acquire_before_consumer,
                },
            }
        )
    return tuple(records)


def _dependence_record(dependence: TaskDependence, task_ids: Mapping[TaskFamily, int]) -> dict[str, object]:
    return {
        "source_task": task_ids[_task_family(dependence.relation.source)],
        "target_task": task_ids[_task_family(dependence.relation.target)],
        "pairs": tuple({"source": pair.source, "target": pair.target} for pair in dependence.relation.pairs),
    }


def _mapped_task_ids(
    dependence: TaskDependence,
    refined_ids: Mapping[TaskFamily, int],
    bindings: tuple[TaskRefinementBinding, ...],
) -> tuple[int, int]:
    coarse_by_refined = {binding.refined_task_id: binding.coarse_task_id for binding in bindings}
    return (
        coarse_by_refined[refined_ids[_task_family(dependence.relation.source)]],
        coarse_by_refined[refined_ids[_task_family(dependence.relation.target)]],
    )


def _task_family(domain: CoordinateDomain) -> TaskFamily:
    if not isinstance(domain, TaskFamily):
        raise TypeError("task refinement relations must connect task families")
    return domain
