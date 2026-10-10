# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical executable-DSL boundary from tiled tasks to physical schedules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from shuttle.ir import DType
from tile_lifetime.event_buffering import (
    BoundedBufferPlan,
    EventRealization,
    EventRealizationAudit,
    EventRealizationKind,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    CoordinateDomain,
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    EventTensorPlan,
    ImperativeEventOperation,
    MemoryVisibility,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    lower_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.normalized_exp_contract_region import (
    NormalizedExpContractPullbackCapability,
    NormalizedExpContractRegionCapability,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScanPrimitive,
    SelectionPrimitive,
    TensorProgram,
    TransportPrimitive,
    execute_tensor_program,
    primitive_outputs,
    tensor_program_value_ordinals,
)

type CanonicalValue = None | bool | int | float | str | tuple[CanonicalValue, ...] | dict[str, CanonicalValue]


class TaskOperationKind(StrEnum):
    """Generic algebra family principally implemented by one tiled task."""

    MAP = "map"
    FOLD = "fold"
    CONTRACT = "contract"
    SELECTION = "selection"
    SCAN = "scan"
    RELATION = "relation"
    TRANSPORT = "transport"
    INPUT_STAGE = "input_stage"
    IMPLEMENTATION_HELPER = "implementation_helper"


class TaskCapability(StrEnum):
    """Generic physical phase capability, independent of source workload."""

    SCALAR_MAP = "scalar_map"
    ASSOCIATIVE_FOLD = "associative_fold"
    MATRIX_CONTRACT = "matrix_contract"
    BOUNDED_SELECTION = "bounded_selection"
    ORDERED_STATE = "ordered_state"
    INDEX_RELATION = "index_relation"
    DATA_TRANSPORT = "data_transport"
    INPUT_STAGING = "input_staging"
    IMPLEMENTATION_HELPER = "implementation_helper"


class TaskValueAccessKind(StrEnum):
    """Lifetime role of one value at a tiled task boundary."""

    LIVE_IN = "live_in"
    LIVE_OUT = "live_out"
    TEMPORARY = "temporary"
    RECOMPUTED = "recomputed"


class RelationAccessClass(StrEnum):
    """Physical access class derived from an algebraic index relation."""

    IDENTITY = "identity"
    AFFINE = "affine"
    GATHER = "gather"
    BOUNDED_DYNAMIC = "bounded_dynamic"
    FINITE = "finite"


class TaskTraversal(StrEnum):
    """Ordering constraint visible to physical scheduling."""

    UNORDERED = "unordered"
    FORWARD = "forward"
    REVERSE = "reverse"
    SOURCE_ORDERED = "source_ordered"


class PhysicalTarget(StrEnum):
    """Target architecture at the executable physical boundary."""

    SM90 = "sm_90"
    SM100 = "sm_100"


class PhysicalNumericalPolicy(StrEnum):
    """Finite-precision contract selected at the physical boundary."""

    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


class PhysicalOrderKind(StrEnum):
    """Physical mechanism that orders two scheduled task families."""

    PROGRAM_ORDER = "program_order"
    LAUNCH_ORDER = "launch_order"
    STREAM_ORDER = "stream_order"
    ALGEBRAIC_REFINEMENT = "algebraic_refinement"


class ParametricEventDerivation(StrEnum):
    """Canonical Event factorization derived from a runtime dependence."""

    EXACT_CONSUMER = "exact_consumer"


class ParametricEventRealization(StrEnum):
    """Initial legal realization for a runtime-instantiated Event relation."""

    IMPERATIVE_WAIT_NOTIFY = "imperative_wait_notify"


@dataclass(frozen=True)
class TaskAxisTile:
    """One logical task axis and its physical tile extent."""

    axis_id: int
    extent: int
    tile_extent: int

    def __post_init__(self) -> None:
        if self.axis_id < 0:
            raise ValueError("task axis ids must be non-negative")
        if self.extent <= 0 or self.tile_extent <= 0:
            raise ValueError("task axis and tile extents must be positive")


@dataclass(frozen=True)
class TaskResourceEstimate:
    """Conservative resources required by one physicalizable task."""

    registers_per_thread: int
    shared_memory_bytes: int
    barrier_count: int

    def __post_init__(self) -> None:
        if min(self.registers_per_thread, self.shared_memory_bytes, self.barrier_count) < 0:
            raise ValueError("task resource estimates must be non-negative")


@dataclass(frozen=True)
class PhysicalPhase:
    """One generic phase in a fused task's intra-task dependence graph."""

    id: int
    capability: TaskCapability
    operation_ordinals: tuple[int, ...]
    domain_shape: tuple[int, ...]
    reduction_dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.id < 0 or any(ordinal < 0 for ordinal in self.operation_ordinals):
            raise ValueError("physical phase and operation ordinals must be non-negative")
        if not self.operation_ordinals and self.capability not in (
            TaskCapability.INPUT_STAGING,
            TaskCapability.IMPLEMENTATION_HELPER,
        ):
            raise ValueError("a physical phase must retain at least one algebra operation")
        if any(extent <= 0 for extent in self.domain_shape):
            raise ValueError("a physical phase requires a static domain")
        if any(dimension < 0 or dimension >= len(self.domain_shape) for dimension in self.reduction_dimensions) or len(
            set(self.reduction_dimensions)
        ) != len(self.reduction_dimensions):
            raise ValueError("physical phase reductions must uniquely index its domain")


def physical_phase_iteration_space(operation) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the operation's explicit phase-local domain and reductions."""
    if isinstance(operation, FoldPrimitive):
        axes = operation.input.axes
        reductions = operation.reduction_axes
    elif isinstance(operation, ContractPrimitive):
        axes = (*operation.output.axes, *operation.reduction_axes)
        reductions = operation.reduction_axes
    elif isinstance(operation, RelationPrimitive | TransportPrimitive):
        axes = operation.domain_axes
        reductions = ()
    elif isinstance(operation, SelectionPrimitive):
        axes = operation.score.axes
        reductions = (operation.dimension,)
    elif isinstance(operation, ScanPrimitive):
        emitted_axes = tuple(
            dict.fromkeys(axis for binding in operation.emitted_outputs for axis in binding.destination.axes)
        )
        state_axes = tuple(dict.fromkeys(axis for binding in operation.state for axis in binding.final.axes))
        axes = emitted_axes or (operation.axis, *state_axes)
        reductions = ()
    else:
        outputs = primitive_outputs(operation)
        if not outputs:
            raise ValueError("an algebra phase requires at least one output domain")
        axes = outputs[0].axes
        if any(output.axes != axes for output in outputs):
            raise ValueError("one algebra phase cannot have multiple iteration domains")
        reductions = ()
    return (
        tuple(axis.extent for axis in axes),
        tuple(axes.index(axis) for axis in reductions),
    )


@dataclass(frozen=True, order=True)
class PhysicalPhaseEdge:
    """Required intra-task value flow between two generic phases."""

    source_phase: int
    target_phase: int
    relation_class: RelationAccessClass

    def __post_init__(self) -> None:
        if self.source_phase < 0 or self.target_phase < 0:
            raise ValueError("physical phase edge ids must be non-negative")
        if self.source_phase == self.target_phase:
            raise ValueError("a physical phase cannot depend on itself")


@dataclass(frozen=True)
class TaskValueAccess:
    """One task-boundary value access with its complete index semantics."""

    value_id: int
    kind: TaskValueAccessKind
    phase_id: int
    relation_class: RelationAccessClass
    relation_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.value_id < 0 or self.phase_id < 0:
            raise ValueError("task value and phase ids must be non-negative")
        _canonicalize(self.relation_record)


@dataclass(frozen=True)
class PhysicalTaskStructure:
    """Schedule-visible structure of one fused physicalizable task.

    Algebra operation ordinals and exact scalar/index ASTs deliberately do not
    occur here. The phase graph retains only the dataflow a physical template
    must implement, while exact semantics remain on :class:`TiledTask` and in
    the canonical algebra record.
    """

    principal_kind: TaskOperationKind
    phases: tuple[PhysicalPhase, ...]
    phase_edges: tuple[PhysicalPhaseEdge, ...]
    owner_axes: tuple[TaskAxisTile, ...]
    streamed_axes: tuple[TaskAxisTile, ...]
    reduction_axes: tuple[TaskAxisTile, ...]
    contraction_axes: tuple[TaskAxisTile, ...]
    traversal: TaskTraversal
    element_types: tuple[str, ...]
    layout_classes: tuple[str, ...]
    resources: TaskResourceEstimate

    def __post_init__(self) -> None:
        phase_by_id = {phase.id: phase for phase in self.phases}
        if len(phase_by_id) != len(self.phases) or set(phase_by_id) != set(range(len(self.phases))):
            raise ValueError("physical phase ids must be unique and contiguous from zero")
        if not self.phases:
            raise ValueError("a physical task structure requires at least one phase")
        for edge in self.phase_edges:
            if edge.source_phase not in phase_by_id or edge.target_phase not in phase_by_id:
                raise ValueError("physical phase edges must reference declared phases")
        if len(self.phase_edges) != len(set(self.phase_edges)):
            raise ValueError("physical task structure contains duplicate phase edges")
        _verify_phase_dag(self.phases, self.phase_edges)
        axes = (*self.owner_axes, *self.streamed_axes, *self.reduction_axes, *self.contraction_axes)
        axis_ids = tuple(axis.axis_id for axis in axes)
        if len(axis_ids) != len(set(axis_ids)):
            raise ValueError("physical axis roles must be disjoint")
        if not self.element_types or any(not value for value in self.element_types):
            raise ValueError("physical task element types must be non-empty")
        if any(not layout for layout in self.layout_classes):
            raise ValueError("physical layout classes must be non-empty strings")

    @property
    def schedule_record(self) -> dict[str, Any]:
        """Return the source-name-free structure used for schedule matching."""

        def axes_record(axes: tuple[TaskAxisTile, ...]) -> tuple[CanonicalValue, ...]:
            return tuple({"extent": axis.extent, "tile_extent": axis.tile_extent} for axis in axes)

        return {
            "principal_kind": self.principal_kind.value,
            "phases": tuple(
                {
                    "id": phase.id,
                    "capability": phase.capability.value,
                    "domain_shape": phase.domain_shape,
                    "reduction_dimensions": phase.reduction_dimensions,
                }
                for phase in self.phases
            ),
            "phase_edges": tuple(
                {
                    "source_phase": edge.source_phase,
                    "target_phase": edge.target_phase,
                    "relation_class": edge.relation_class.value,
                }
                for edge in sorted(self.phase_edges)
            ),
            "owner_axes": axes_record(self.owner_axes),
            "streamed_axes": axes_record(self.streamed_axes),
            "reduction_axes": axes_record(self.reduction_axes),
            "contraction_axes": axes_record(self.contraction_axes),
            "traversal": self.traversal.value,
            "element_types": self.element_types,
            "layout_classes": self.layout_classes,
            "resources": {
                "registers_per_thread": self.resources.registers_per_thread,
                "shared_memory_bytes": self.resources.shared_memory_bytes,
                "barrier_count": self.resources.barrier_count,
            },
        }

    @property
    def schedule_fingerprint(self) -> str:
        """Fingerprint only the physical structure relevant to scheduling."""
        return canonical_fingerprint(self.schedule_record)


@dataclass(frozen=True)
class TiledTask:
    """A physicalizable fused task derived from generic algebra and ownership."""

    id: int
    family: TaskFamily
    operation_ordinals: tuple[int, ...]
    operation_kinds: tuple[TaskOperationKind, ...]
    structure: PhysicalTaskStructure
    accesses: tuple[TaskValueAccess, ...]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("a tiled task requires a non-negative id")
        is_synthetic = (
            not self.operation_ordinals
            and self.operation_kinds in ((TaskOperationKind.INPUT_STAGE,), (TaskOperationKind.IMPLEMENTATION_HELPER,))
            and self.structure.principal_kind is self.operation_kinds[0]
        )
        if not self.operation_ordinals and not is_synthetic:
            raise ValueError("a synthetic tiled task must declare its physical-only role")
        if not is_synthetic and len(self.operation_ordinals) != len(self.operation_kinds):
            raise ValueError("task operation ordinals and kinds must have equal length")
        if len(set(self.operation_ordinals)) != len(self.operation_ordinals):
            raise ValueError("a tiled task cannot contain an algebra operation twice")
        phase_ids = {phase.id for phase in self.structure.phases}
        if any(access.phase_id not in phase_ids for access in self.accesses):
            raise ValueError("task accesses must attach to declared physical phases")
        retained = {ordinal for phase in self.structure.phases for ordinal in phase.operation_ordinals}
        if retained != set(self.operation_ordinals):
            raise ValueError("physical phases must retain exactly the task's algebra operations")


@dataclass(frozen=True)
class BoundedTaskCoordinate:
    """One INT32 coordinate result and its proven half-open bounds."""

    value: ProgramValue
    lower_bound: int
    upper_bound: int

    def __post_init__(self) -> None:
        if self.value.dtype is not DType.INT32:
            raise ValueError("task-dependence coordinates must be INT32")
        if self.lower_bound < 0 or self.upper_bound <= self.lower_bound:
            raise ValueError("task-coordinate bounds must be a nonempty non-negative interval")


@dataclass(frozen=True)
class DependenceRuntimeBinding:
    """Bind one dependence-program input to a canonical algebra value ordinal."""

    input: ProgramValue
    algebra_value_ordinal: int

    def __post_init__(self) -> None:
        if self.algebra_value_ordinal < 0:
            raise ValueError("algebra value ordinals must be non-negative")


@dataclass(frozen=True)
class TaskDependenceProgram:
    """Parametric exact task relation computed by generic tensor algebra."""

    source_task_id: int
    target_task_id: int
    relation_program: TensorProgram
    runtime_bindings: tuple[DependenceRuntimeBinding, ...]
    source_coordinates: tuple[BoundedTaskCoordinate, ...]
    target_coordinates: tuple[BoundedTaskCoordinate, ...]
    validity: ProgramValue | None
    visibility: MemoryVisibility

    def __post_init__(self) -> None:
        if self.source_task_id < 0 or self.target_task_id < 0:
            raise ValueError("task-dependence program task ids must be non-negative")
        if tuple(binding.input for binding in self.runtime_bindings) != self.relation_program.inputs:
            raise ValueError("runtime bindings must cover dependence-program inputs in order")
        ordinals = tuple(binding.algebra_value_ordinal for binding in self.runtime_bindings)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("a dependence program cannot bind one algebra value more than once")
        if not self.source_coordinates or not self.target_coordinates:
            raise ValueError("a task-dependence program requires source and target coordinates")
        coordinate_values = tuple(
            coordinate.value for coordinate in (*self.source_coordinates, *self.target_coordinates)
        )
        outputs = set(self.relation_program.outputs)
        if not set(coordinate_values) <= outputs:
            raise ValueError("task coordinates must be dependence-program outputs")
        domain_axes = coordinate_values[0].axes
        if any(value.axes != domain_axes for value in coordinate_values):
            raise ValueError("task-dependence coordinate outputs must share one relation domain")
        if self.validity is not None:
            if self.validity not in outputs or self.validity.dtype is not DType.BOOL:
                raise ValueError("task-dependence validity must be a Boolean program output")
            if self.validity.axes != domain_axes:
                raise ValueError("task-dependence validity must use the relation domain")

    @property
    def relation_domain_shape(self) -> tuple[int, ...]:
        return self.source_coordinates[0].value.shape


@dataclass(frozen=True)
class CanonicalTaskPlan:
    """Ownership-derived tiled tasks and their exact dependences."""

    tasks: tuple[TiledTask, ...]
    dependences: tuple[TaskDependence, ...]
    bounded_buffers: tuple[BoundedBufferPlan, ...] = ()
    dependence_programs: tuple[TaskDependenceProgram, ...] = ()

    def __post_init__(self) -> None:
        if tuple(task.id for task in self.tasks) != tuple(range(len(self.tasks))):
            raise ValueError("tiled task ids must be unique and contiguous from zero")
        families = tuple(task.family for task in self.tasks)
        if len(families) != len(set(families)):
            raise ValueError("every tiled task must own one distinct task family")
        family_names = tuple(family.name for family in families)
        if len(family_names) != len(set(family_names)):
            raise ValueError("tiled task family display names must be unique within a task plan")
        family_set = set(families)
        for dependence in self.dependences:
            if dependence.relation.source not in family_set or dependence.relation.target not in family_set:
                raise ValueError("exact dependence endpoints must be task-plan families")
        for buffer in self.bounded_buffers:
            if buffer.producer not in family_set:
                raise ValueError("bounded-buffer producers must be task-plan families")
        task_by_id = {task.id: task for task in self.tasks}
        for dependence in self.dependence_programs:
            if dependence.source_task_id not in task_by_id or dependence.target_task_id not in task_by_id:
                raise ValueError("task-dependence programs must reference declared tiled tasks")
            source = task_by_id[dependence.source_task_id].family
            target = task_by_id[dependence.target_task_id].family
            _verify_coordinate_bounds(dependence.source_coordinates, source)
            _verify_coordinate_bounds(dependence.target_coordinates, target)

    @property
    def task_families(self) -> tuple[TaskFamily, ...]:
        return tuple(task.family for task in self.tasks)

    @property
    def canonical_record(self) -> dict[str, CanonicalValue]:
        return canonical_task_plan_record(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class PhaseEdgeConstraint:
    """One required capability-to-capability edge in a schedule family."""

    source: TaskCapability
    target: TaskCapability
    relation_class: RelationAccessClass


@dataclass(frozen=True)
class PhysicalTaskConstraint:
    """Declarative substructure and resource contract for a schedule family."""

    principal_kind: TaskOperationKind
    required_capabilities: tuple[TaskCapability, ...]
    supported_capabilities: tuple[TaskCapability, ...]
    required_edges: tuple[PhaseEdgeConstraint, ...]
    owner_rank: int
    streamed_rank: int
    reduction_rank: int
    contraction_rank: int
    supported_relation_classes: tuple[RelationAccessClass, ...]
    supported_element_types: tuple[str, ...]
    supported_layout_classes: tuple[str, ...]
    maximum_resources: TaskResourceEstimate

    def __post_init__(self) -> None:
        ranks = (self.owner_rank, self.streamed_rank, self.reduction_rank, self.contraction_rank)
        if min(ranks) < 0:
            raise ValueError("schedule-family axis ranks must be non-negative")
        if not self.required_capabilities:
            raise ValueError("a schedule family must require physical capabilities")
        if not self.supported_capabilities:
            raise ValueError("a schedule family must declare supported physical capabilities")
        if len(self.supported_capabilities) != len(set(self.supported_capabilities)):
            raise ValueError("supported physical capabilities must form a unique set")
        if not set(self.required_capabilities) <= set(self.supported_capabilities):
            raise ValueError("required capabilities must be included in supported capabilities")
        edge_capabilities = {capability for edge in self.required_edges for capability in (edge.source, edge.target)}
        if not edge_capabilities <= set(self.supported_capabilities):
            raise ValueError("required phase edges must use supported capabilities")
        if not self.supported_relation_classes or not self.supported_element_types:
            raise ValueError("a schedule family must declare supported accesses and element types")


@dataclass(frozen=True)
class PhysicalScheduleFamily:
    """Workload-neutral declarative eligibility contract for physical tasks."""

    name: str
    target: PhysicalTarget
    tasks: tuple[PhysicalTaskConstraint, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.tasks:
            raise ValueError("a physical schedule family needs a name and task constraints")

    @property
    def canonical_record(self) -> dict[str, CanonicalValue]:
        return {
            "name": self.name,
            "target": self.target.value,
            "tasks": tuple(_task_constraint_record(task) for task in self.tasks),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class PhysicalTaskOrder:
    """Exact task relation enforced by a selected physical schedule."""

    kind: PhysicalOrderKind
    source_task_id: int
    target_task_id: int
    relation: TaskRelation
    mechanism_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source_task_id < 0 or self.target_task_id < 0:
            raise ValueError("physical order task ids must be non-negative")
        mechanism = _canonicalize(self.mechanism_record)
        if not mechanism:
            raise ValueError("a physical task order requires a structured mechanism")


@dataclass(frozen=True)
class PhysicalSchedule:
    """One schedule-family attachment and its physical parameters."""

    family: PhysicalScheduleFamily
    task_ids: tuple[int, ...]
    parameters: Mapping[str, Any]
    enforced_orders: tuple[PhysicalTaskOrder, ...] = ()

    def __post_init__(self) -> None:
        if len(self.task_ids) != len(self.family.tasks) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("schedule task ids must match family constraints one-for-one")
        if any(task_id < 0 for task_id in self.task_ids):
            raise ValueError("schedule task ids must be non-negative")
        _canonicalize(self.parameters)

    @property
    def canonical_record(self) -> dict[str, CanonicalValue]:
        return {
            "family": self.family.canonical_record,
            "task_ids": self.task_ids,
            "parameters": _canonicalize(self.parameters),
            "enforced_orders": tuple(_physical_task_order_record(order) for order in self.enforced_orders),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class EventErasureWitness:
    """Reference to a selected schedule order that erases one logical Event."""

    plan_name: str
    schedule_index: int
    order_index: int

    def __post_init__(self) -> None:
        if not self.plan_name:
            raise ValueError("an Event erasure witness requires a plan name")
        if self.schedule_index < 0 or self.order_index < 0:
            raise ValueError("Event erasure schedule and order indices must be non-negative")


@dataclass(frozen=True)
class EventRealizationProof:
    """Verified physical realization or erasure proof for every Event Tensor."""

    audit: EventRealizationAudit
    erasure_witnesses: tuple[EventErasureWitness, ...]

    def verify(
        self,
        program: EventDataflowProgram,
        *,
        schedules: tuple[PhysicalSchedule, ...],
        task_plan: CanonicalTaskPlan,
    ) -> None:
        """Verify logical coverage and the physical order claimed for erasure."""
        verified = verify_event_realizations(program, self.audit.entries)
        if verified != self.audit:
            raise ValueError("event realization audit is not in canonical event-plan order")
        witnesses = {witness.plan_name: witness for witness in self.erasure_witnesses}
        if len(witnesses) != len(self.erasure_witnesses):
            raise ValueError("Event erasure witnesses must have unique plan names")
        erased = {entry.plan_name: entry for entry in self.audit.erased}
        if set(witnesses) != set(erased):
            raise ValueError("Event erasure witnesses must cover every erased Event exactly")
        for plan_name, entry in erased.items():
            witness = witnesses[plan_name]
            if witness.schedule_index >= len(schedules):
                raise ValueError(f"Event erasure witness references an unavailable schedule for {plan_name}")
            schedule = schedules[witness.schedule_index]
            if witness.order_index >= len(schedule.enforced_orders):
                raise ValueError(f"Event erasure witness references an unavailable physical order for {plan_name}")
            order = schedule.enforced_orders[witness.order_index]
            expected_order_kinds = {
                EventRealizationKind.ERASED_PROGRAM_ORDER: {PhysicalOrderKind.PROGRAM_ORDER},
                EventRealizationKind.ERASED_STREAM_ORDER: {
                    PhysicalOrderKind.LAUNCH_ORDER,
                    PhysicalOrderKind.STREAM_ORDER,
                },
                EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT: {PhysicalOrderKind.ALGEBRAIC_REFINEMENT},
            }[entry.kind]
            if order.kind not in expected_order_kinds:
                raise ValueError(f"Event erasure kind does not match its scheduled physical order for {plan_name}")
            tasks = {task.id: task for task in task_plan.tasks}
            if order.source_task_id not in tasks or order.target_task_id not in tasks:
                raise ValueError(f"Event erasure order references unavailable tasks for {plan_name}")
            if order.relation.source != tasks[order.source_task_id].family:
                raise ValueError(f"Event erasure order source task does not match its relation for {plan_name}")
            if order.relation.target != tasks[order.target_task_id].family:
                raise ValueError(f"Event erasure order target task does not match its relation for {plan_name}")
            if order.kind in (PhysicalOrderKind.PROGRAM_ORDER, PhysicalOrderKind.ALGEBRAIC_REFINEMENT) and not {
                order.source_task_id,
                order.target_task_id,
            } <= set(schedule.task_ids):
                raise ValueError(f"same-launch Event erasure tasks are not both owned by the schedule for {plan_name}")
            if order.kind in (PhysicalOrderKind.LAUNCH_ORDER, PhysicalOrderKind.STREAM_ORDER) and not {
                order.source_task_id,
                order.target_task_id,
            } & set(schedule.task_ids):
                raise ValueError(f"cross-launch Event erasure is not anchored by its declaring schedule for {plan_name}")
            required = entry.ordering
            assert required is not None
            enforced = order.relation
            if enforced.source != required.source or enforced.target != required.target:
                raise ValueError(f"Event erasure witness has incompatible task domains for {plan_name}")
            missing = set(required.pairs) - set(enforced.pairs)
            if missing:
                raise ValueError(
                    f"Event erasure witness omits required physical order for {plan_name}: {sorted(missing)}"
                )


@dataclass(frozen=True)
class ExecutableEventLowering:
    """Storage-bound imperative Event operations and their physical disposition."""

    plan_name: str
    allocation: EventTensorAllocation
    imperative_operations: tuple[ImperativeEventOperation, ...]
    retained_operations: tuple[ImperativeEventOperation, ...]
    realization: EventRealization
    erasure_proof: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.plan_name != self.realization.plan_name:
            raise ValueError("imperative Event lowering and realization names must match")
        erased = self.realization.kind is not EventRealizationKind.PHYSICAL
        if erased != (self.erasure_proof is not None):
            raise ValueError("erased imperative Events require exactly one physical-order witness")
        expected_retained = () if erased else self.imperative_operations
        if self.retained_operations != expected_retained:
            raise ValueError("imperative Event operations may erase only after a verified physical-order proof")
        if self.erasure_proof is not None:
            _canonicalize(self.erasure_proof)

    @property
    def canonical_record(self) -> dict[str, CanonicalValue]:
        return {
            "plan": self.plan_name,
            "allocation": {
                "capacity": self.allocation.capacity,
                "generation_count": self.allocation.generation_count,
                "slots": self.allocation.storage.slots,
                "generations": self.allocation.storage.generations,
            },
            "imperative_operations": tuple(
                _imperative_event_operation_record(operation) for operation in self.imperative_operations
            ),
            "retained_operations": tuple(
                _imperative_event_operation_record(operation) for operation in self.retained_operations
            ),
            "realization": self.realization.kind.value,
            "erasure_proof": None if self.erasure_proof is None else _canonicalize(self.erasure_proof),
        }


@dataclass(frozen=True)
class EventTensorAllocation:
    """Bounded physical slots and generations for one Event Tensor."""

    storage: EventStorageBinding
    capacity: int
    generation_count: int

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.generation_count <= 0:
            raise ValueError("Event storage capacity and generation count must be positive")
        if any(slot < 0 or slot >= self.capacity for slot in self.storage.slots):
            raise ValueError("Event storage slot is outside the declared capacity")
        if any(generation < 0 or generation >= self.generation_count for generation in self.storage.generations):
            raise ValueError("Event generation is outside the declared generation count")


@dataclass(frozen=True)
class ParametricEventTensorPlan:
    """Compile-time Event/buffer rule for a runtime-instantiated dependence."""

    dependence_program_index: int
    derivation: ParametricEventDerivation
    capacity: int
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    generation_count: int
    memory_scope: EventMemoryScope
    generation_policy: EventGenerationPolicy
    scheduling_mode: EventSchedulingMode
    realization: ParametricEventRealization = ParametricEventRealization.IMPERATIVE_WAIT_NOTIFY

    def __post_init__(self) -> None:
        if self.dependence_program_index < 0:
            raise ValueError("parametric Event dependence indices must be non-negative")
        if self.capacity <= 0 or self.generation_count <= 0:
            raise ValueError("parametric Event capacity and generation count must be positive")
        if len(self.slots) != len(self.generations):
            raise ValueError("parametric Event slots and generations must have equal length")
        if any(slot < 0 or slot >= self.capacity for slot in self.slots):
            raise ValueError("parametric Event slot is outside its capacity")
        if any(generation < 0 or generation >= self.generation_count for generation in self.generations):
            raise ValueError("parametric Event generation is outside its generation count")


@dataclass(frozen=True)
class ExecutablePhysicalRequest:
    """Canonical generic tasks, Events, and schedule consumed by a backend."""

    algebra_record: Mapping[str, Any]
    algebra_fingerprint: str
    task_plan: CanonicalTaskPlan
    event_plans: tuple[EventTensorPlan, ...]
    event_allocations: tuple[EventTensorAllocation, ...]
    target: PhysicalTarget
    numerical_policy: PhysicalNumericalPolicy
    schedules: tuple[PhysicalSchedule, ...]
    event_realizations: EventRealizationProof
    provenance: Mapping[str, Any]
    event_programs: tuple[ParametricEventTensorPlan, ...] = ()
    normalized_exp_contract_regions: tuple[NormalizedExpContractRegionCapability, ...] = ()
    normalized_exp_contract_pullbacks: tuple[NormalizedExpContractPullbackCapability, ...] = ()

    def __post_init__(self) -> None:
        algebra_record = _canonicalize(self.algebra_record)
        if canonical_fingerprint(algebra_record) != self.algebra_fingerprint:
            raise ValueError("algebra fingerprint does not match its canonical record")
        if len(self.event_allocations) != len(self.event_plans):
            raise ValueError("every Event Tensor requires one bounded storage allocation")
        for plan, allocation in zip(self.event_plans, self.event_allocations, strict=True):
            if allocation.storage.domain != plan.domain:
                raise ValueError("Event storage allocation domain does not match its Event Tensor")
            event_tensor_runtime_inputs(plan, storage_binding=allocation.storage)
        expected_programs = tuple(range(len(self.task_plan.dependence_programs)))
        if tuple(sorted(event.dependence_program_index for event in self.event_programs)) != expected_programs:
            raise ValueError("parametric Event plans must cover every task-dependence program exactly once")
        for event in self.event_programs:
            dependence = self.task_plan.dependence_programs[event.dependence_program_index]
            target = self.task_plan.tasks[dependence.target_task_id].family
            event_count = int(np.prod([axis.extent for axis in target.axes], dtype=np.int64))
            if len(event.slots) != event_count:
                raise ValueError("exact-consumer parametric Event storage must cover every consumer task")
        if self.numerical_policy is PhysicalNumericalPolicy.SOURCE_ORDERED:
            unordered = tuple(
                task.id for task in self.task_plan.tasks if task.structure.traversal is TaskTraversal.UNORDERED
            )
            if unordered:
                rendered = ", ".join(str(task_id) for task_id in unordered)
                raise ValueError(f"SOURCE_ORDERED request cannot contain unordered task {rendered}")
        scheduled_task_ids = tuple(task_id for schedule in self.schedules for task_id in schedule.task_ids)
        if tuple(sorted(scheduled_task_ids)) != tuple(range(len(self.task_plan.tasks))):
            raise ValueError("physical schedules must cover every tiled task exactly once")
        for schedule in self.schedules:
            if schedule.family.target is not self.target:
                raise ValueError("physical schedule target does not match the request target")
            if not physical_schedule_is_eligible(schedule, self.task_plan):
                raise ValueError("tiled task structure is not eligible for the selected schedule family")
            _verify_schedule_orders(schedule, self.task_plan)
        if tuple(region.ordinal for region in self.normalized_exp_contract_regions) != tuple(
            range(len(self.normalized_exp_contract_regions))
        ):
            raise ValueError("normalized-exp Contract region ordinals must be dense and source ordered")
        if tuple(pullback.ordinal for pullback in self.normalized_exp_contract_pullbacks) != tuple(
            range(len(self.normalized_exp_contract_pullbacks))
        ):
            raise ValueError("normalized-exp Contract pullback ordinals must be dense and source ordered")
        if any(
            pullback.forward_region_ordinal >= len(self.normalized_exp_contract_regions)
            for pullback in self.normalized_exp_contract_pullbacks
        ):
            raise ValueError("normalized-exp pullback references an absent forward region capability")
        program = EventDataflowProgram(self.task_plan.task_families, self.task_plan.dependences, self.event_plans)
        verify_event_dataflow_program(program)
        self.event_realizations.verify(program, schedules=self.schedules, task_plan=self.task_plan)
        _canonicalize(self.provenance)

    @property
    def canonical_record(self) -> dict[str, CanonicalValue]:
        family_ids = {task.family: task.id for task in self.task_plan.tasks}
        return {
            "schema_version": 1,
            "algebra": _canonicalize(self.algebra_record),
            "algebra_fingerprint": self.algebra_fingerprint,
            "task_plan": self.task_plan.canonical_record,
            "task_plan_fingerprint": self.task_plan.fingerprint,
            "event_plans": tuple(
                _event_plan_record(plan, family_ids=family_ids, event_id=event_id)
                for event_id, plan in enumerate(self.event_plans)
            ),
            "event_allocations": tuple(
                _event_allocation_record(plan, allocation, event_id=event_id)
                for event_id, (plan, allocation) in enumerate(zip(self.event_plans, self.event_allocations, strict=True))
            ),
            "event_programs": tuple(_parametric_event_record(event) for event in self.event_programs),
            "normalized_exp_contract_regions": tuple(
                region.canonical_record for region in self.normalized_exp_contract_regions
            ),
            "normalized_exp_contract_pullbacks": tuple(
                pullback.canonical_record for pullback in self.normalized_exp_contract_pullbacks
            ),
            "target": self.target.value,
            "numerical_policy": self.numerical_policy.value,
            "schedules": tuple(schedule.canonical_record for schedule in self.schedules),
            "schedule_fingerprints": tuple(schedule.fingerprint for schedule in self.schedules),
            "event_realizations": {
                "audit": tuple(
                    _event_realization_record(entry, event_id=event_id, family_ids=family_ids)
                    for event_id, entry in enumerate(self.event_realizations.audit.entries)
                ),
                "erasure_witnesses": tuple(
                    _event_erasure_witness_record(
                        witness,
                        event_id=_event_id_by_name(self.event_plans, witness.plan_name),
                    )
                    for witness in self.event_realizations.erasure_witnesses
                ),
            },
        }

    @property
    def fingerprint(self) -> str:
        """Return the provenance-independent physical-request fingerprint."""
        return canonical_fingerprint(self.canonical_record)

    @property
    def semantic_fingerprint(self) -> str:
        """Return the canonical DSL/native conformance identity."""
        return self.fingerprint

    @property
    def artifact_record(self) -> dict[str, CanonicalValue]:
        """Return the semantic request plus complete source/tool provenance."""
        return {
            "request": self.canonical_record,
            "diagnostics": {
                "task_families": tuple(
                    {
                        "task": task.id,
                        "name": task.family.name,
                        "axis_names": tuple(axis.name for axis in task.family.axes),
                    }
                    for task in self.task_plan.tasks
                ),
                "events": tuple(
                    {
                        "event": event_id,
                        "name": plan.name,
                        "axis_names": tuple(axis.name for axis in plan.domain.axes),
                    }
                    for event_id, plan in enumerate(self.event_plans)
                ),
                "buffers": tuple(buffer.name for buffer in self.task_plan.bounded_buffers),
                "realization_reasons": tuple(entry.reason for entry in self.event_realizations.audit.entries),
            },
            "provenance": _canonicalize(self.provenance),
        }

    @property
    def artifact_fingerprint(self) -> str:
        """Return the identity of the request together with its provenance."""
        return canonical_fingerprint(self.artifact_record)

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.canonical_record)

    @property
    def artifact_json(self) -> str:
        return canonical_json(self.artifact_record)


def lower_executable_event_operations(
    request: ExecutablePhysicalRequest,
) -> tuple[ExecutableEventLowering, ...]:
    """Lower concrete Events through bounded storage, then apply verified realizations."""
    program = EventDataflowProgram(request.task_plan.task_families, request.task_plan.dependences, request.event_plans)
    request.event_realizations.verify(program, schedules=request.schedules, task_plan=request.task_plan)
    realizations = {entry.plan_name: entry for entry in request.event_realizations.audit.entries}
    witnesses = {entry.plan_name: entry for entry in request.event_realizations.erasure_witnesses}
    lowerings = []
    for plan, allocation in zip(request.event_plans, request.event_allocations, strict=True):
        scheduling_mode = plan.scheduling_mode or EventSchedulingMode.STATIC
        operations = lower_event_tensor_plan(
            plan,
            scheduling_mode=scheduling_mode,
            storage_binding=allocation.storage,
        )
        realization = realizations[plan.name]
        witness = witnesses.get(plan.name)
        erasure_proof = None
        if witness is not None:
            order = request.schedules[witness.schedule_index].enforced_orders[witness.order_index]
            erasure_proof = {
                "kind": "selected_physical_order",
                "schedule_index": witness.schedule_index,
                "order_index": witness.order_index,
                "order_kind": order.kind.value,
                "mechanism": order.mechanism_record,
            }
        lowerings.append(
            ExecutableEventLowering(
                plan.name,
                allocation,
                operations,
                operations if realization.kind is EventRealizationKind.PHYSICAL else (),
                realization,
                erasure_proof,
            )
        )
    return tuple(lowerings)


def erase_executable_event_operations(
    lowering: ExecutableEventLowering,
    plan: EventTensorPlan,
    *,
    kind: EventRealizationKind,
    mechanism: str,
    reason: str,
    proof: Mapping[str, Any],
) -> ExecutableEventLowering:
    """Erase imperative operations after an exact additional physical covering proof."""
    if lowering.plan_name != plan.name:
        raise ValueError("imperative Event lowering and erasure plan names must match")
    if kind is EventRealizationKind.PHYSICAL:
        raise ValueError("imperative Event erasure requires a non-physical realization kind")
    if lowering.realization.kind is not EventRealizationKind.PHYSICAL:
        raise ValueError("imperative Event operations have already been erased")
    canonical_proof = _canonicalize(proof)
    if not canonical_proof:
        raise ValueError("imperative Event erasure requires a structured covering proof")
    realization = erased_event_realization(
        plan,
        kind=kind,
        mechanism=mechanism,
        reason=reason,
        ordering=plan.scheduled_dependence,
    )
    return ExecutableEventLowering(
        lowering.plan_name,
        lowering.allocation,
        lowering.imperative_operations,
        (),
        realization,
        canonical_proof,
    )


def physical_schedule_is_eligible(schedule: PhysicalSchedule, plan: CanonicalTaskPlan) -> bool:
    """Return whether selected tasks satisfy the family's declarative constraints."""
    task_by_id = {task.id: task for task in plan.tasks}
    if any(task_id not in task_by_id for task_id in schedule.task_ids):
        return False
    return all(
        _task_satisfies_constraint(task_by_id[task_id], constraint)
        for task_id, constraint in zip(schedule.task_ids, schedule.family.tasks, strict=True)
    )


def install_physical_schedule_cover(
    request: ExecutablePhysicalRequest,
    schedules: Sequence[PhysicalSchedule],
    *,
    source_request_fingerprint: str | None = None,
) -> ExecutablePhysicalRequest:
    """Install an exact task cover and derive concrete Event realizations.

    Schedule-family discovery and cost selection happen before this boundary.
    This function performs the workload-independent composition step: it
    verifies that the chosen schedules partition every tiled task, erases an
    Event only when one selected physical order covers its complete relation,
    and otherwise retains the Event as an imperative physical requirement.
    Runtime-parametric Event programs are deliberately left physical.
    """
    if source_request_fingerprint is None:
        source_request_fingerprint = request.fingerprint
    elif len(source_request_fingerprint) != 64:
        raise ValueError("source request fingerprint must be a canonical SHA-256 digest")
    installed = tuple(schedules)
    covered_task_ids = tuple(task_id for schedule in installed for task_id in schedule.task_ids)
    expected_task_ids = tuple(range(len(request.task_plan.tasks)))
    if tuple(sorted(covered_task_ids)) != expected_task_ids:
        raise ValueError("physical schedules must partition every tiled task exactly once")
    if any(schedule.family.target is not request.target for schedule in installed):
        raise ValueError("physical schedule target does not match the request target")
    if any(not physical_schedule_is_eligible(schedule, request.task_plan) for schedule in installed):
        raise ValueError("a selected physical schedule is not eligible for its tiled tasks")

    realizations: list[EventRealization] = []
    witnesses: list[EventErasureWitness] = []
    for plan in request.event_plans:
        covers = _schedule_order_covers(plan.scheduled_dependence, installed)
        if len(covers) > 1:
            raise ValueError(f"Event {plan.name!r} has multiple selected physical-order covers")
        if not covers:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="generation_counted_event",
                    reason="no selected physical order covers the complete Event relation",
                )
            )
            continue
        schedule_index, order_index = covers[0]
        order = installed[schedule_index].enforced_orders[order_index]
        kind = {
            PhysicalOrderKind.PROGRAM_ORDER: EventRealizationKind.ERASED_PROGRAM_ORDER,
            PhysicalOrderKind.LAUNCH_ORDER: EventRealizationKind.ERASED_STREAM_ORDER,
            PhysicalOrderKind.STREAM_ORDER: EventRealizationKind.ERASED_STREAM_ORDER,
            PhysicalOrderKind.ALGEBRAIC_REFINEMENT: EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
        }[order.kind]
        realizations.append(
            erased_event_realization(
                plan,
                kind=kind,
                mechanism=f"selected_schedule_{order.kind.value}",
                reason="one selected physical order covers the complete Event relation",
                ordering=order.relation,
            )
        )
        witnesses.append(EventErasureWitness(plan.name, schedule_index, order_index))

    program = EventDataflowProgram(request.task_plan.task_families, request.task_plan.dependences, request.event_plans)
    audit = verify_event_realizations(program, tuple(realizations))
    return replace(
        request,
        schedules=installed,
        event_realizations=EventRealizationProof(audit, tuple(witnesses)),
        provenance={
            **request.provenance,
            "physical_schedule_installation": {
                "source_request_fingerprint": source_request_fingerprint,
                "schedule_fingerprints": tuple(schedule.fingerprint for schedule in installed),
                "runtime_parametric_events": tuple(event.dependence_program_index for event in request.event_programs),
            },
        },
    )


def _schedule_order_covers(
    required: TaskRelation,
    schedules: tuple[PhysicalSchedule, ...],
) -> tuple[tuple[int, int], ...]:
    required_pairs = set(required.pairs)
    covers = []
    for schedule_index, schedule in enumerate(schedules):
        for order_index, order in enumerate(schedule.enforced_orders):
            if order.relation.source != required.source or order.relation.target != required.target:
                continue
            if required_pairs <= set(order.relation.pairs):
                covers.append((schedule_index, order_index))
    return tuple(covers)


def _verify_schedule_orders(schedule: PhysicalSchedule, plan: CanonicalTaskPlan) -> None:
    task_by_id = {task.id: task for task in plan.tasks}
    for order in schedule.enforced_orders:
        if order.source_task_id not in task_by_id or order.target_task_id not in task_by_id:
            raise ValueError("physical schedule order references an unavailable tiled task")
        if order.relation.source != task_by_id[order.source_task_id].family:
            raise ValueError("physical schedule order source family does not match its task id")
        if order.relation.target != task_by_id[order.target_task_id].family:
            raise ValueError("physical schedule order target family does not match its task id")
        owned = set(schedule.task_ids)
        endpoints = {order.source_task_id, order.target_task_id}
        if order.kind in (PhysicalOrderKind.PROGRAM_ORDER, PhysicalOrderKind.ALGEBRAIC_REFINEMENT):
            if not endpoints <= owned:
                raise ValueError("program-order and algebraic orders require one schedule to own both tasks")
        elif not endpoints & owned:
            raise ValueError("launch-order and stream-order records must be anchored by their declaring schedule")


def canonical_task_plan_record(plan: CanonicalTaskPlan) -> dict[str, CanonicalValue]:
    """Return the lossless canonical record for ownership-derived tiled tasks."""
    family_ids = {task.family: task.id for task in plan.tasks}
    return {
        "tasks": tuple(_tiled_task_record(task) for task in plan.tasks),
        "dependences": tuple(_dependence_record(dependence, family_ids) for dependence in plan.dependences),
        "bounded_buffers": tuple(
            _bounded_buffer_record(buffer, buffer_id=buffer_id, family_ids=family_ids)
            for buffer_id, buffer in enumerate(plan.bounded_buffers)
        ),
        "dependence_programs": tuple(
            _task_dependence_program_record(dependence) for dependence in plan.dependence_programs
        ),
    }


def instantiate_task_dependence(
    dependence: TaskDependenceProgram,
    *,
    task_plan: CanonicalTaskPlan,
    algebra_values: Mapping[int, Any],
) -> TaskDependence:
    """Evaluate one parametric relation for an invocation's algebra values."""
    missing = tuple(
        binding.algebra_value_ordinal
        for binding in dependence.runtime_bindings
        if binding.algebra_value_ordinal not in algebra_values
    )
    if missing:
        raise ValueError(f"missing runtime algebra values for task dependence: {missing}")
    inputs = {
        binding.input.name: algebra_values[binding.algebra_value_ordinal] for binding in dependence.runtime_bindings
    }
    results = execute_tensor_program(dependence.relation_program, inputs)
    source_arrays = tuple(np.asarray(results[coordinate.value.name]) for coordinate in dependence.source_coordinates)
    target_arrays = tuple(np.asarray(results[coordinate.value.name]) for coordinate in dependence.target_coordinates)
    validity = (
        np.ones(dependence.relation_domain_shape, dtype=np.bool_)
        if dependence.validity is None
        else np.asarray(results[dependence.validity.name], dtype=np.bool_)
    )
    if validity.shape != dependence.relation_domain_shape:
        raise ValueError("instantiated task-dependence validity has the wrong relation-domain shape")
    task_by_id = {task.id: task for task in task_plan.tasks}
    source_family = task_by_id[dependence.source_task_id].family
    target_family = task_by_id[dependence.target_task_id].family
    _verify_instantiated_coordinates(source_arrays, dependence.source_coordinates, source_family, validity)
    _verify_instantiated_coordinates(target_arrays, dependence.target_coordinates, target_family, validity)
    pairs = {
        (
            tuple(int(array[index]) for array in source_arrays),
            tuple(int(array[index]) for array in target_arrays),
        )
        for index in np.ndindex(dependence.relation_domain_shape)
        if validity[index]
    }
    return TaskDependence(
        TaskRelation.from_pairs(source_family, target_family, tuple(sorted(pairs))), dependence.visibility
    )


def instantiate_parametric_event_tensor(
    event: ParametricEventTensorPlan,
    *,
    task_plan: CanonicalTaskPlan,
    algebra_values: Mapping[int, Any],
    name: str,
) -> tuple[EventTensorPlan, EventTensorAllocation]:
    """Instantiate runtime pairs/counts while retaining compile-time storage bounds."""
    dependence = instantiate_task_dependence(
        task_plan.dependence_programs[event.dependence_program_index],
        task_plan=task_plan,
        algebra_values=algebra_values,
    )
    if event.derivation is not ParametricEventDerivation.EXACT_CONSUMER:
        raise ValueError(f"unsupported parametric Event derivation {event.derivation.value}")
    plan = derive_event_tensor_plan(
        dependence,
        name=name,
        memory_scope=event.memory_scope,
        generation_policy=event.generation_policy,
        scheduling_mode=event.scheduling_mode,
    )
    allocation = EventTensorAllocation(
        EventStorageBinding(plan.domain, event.slots, event.generations),
        capacity=event.capacity,
        generation_count=event.generation_count,
    )
    event_tensor_runtime_inputs(plan, storage_binding=allocation.storage)
    return plan, allocation


def canonical_json(value: object) -> str:
    """Encode a JSON-compatible compiler record deterministically."""
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_fingerprint(value: object) -> str:
    """Fingerprint a canonical compiler boundary record."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _task_satisfies_constraint(task: TiledTask, constraint: PhysicalTaskConstraint) -> bool:
    structure = task.structure
    if structure.principal_kind is not constraint.principal_kind:
        return False
    if (
        len(structure.owner_axes),
        len(structure.streamed_axes),
        len(structure.reduction_axes),
        len(structure.contraction_axes),
    ) != (
        constraint.owner_rank,
        constraint.streamed_rank,
        constraint.reduction_rank,
        constraint.contraction_rank,
    ):
        return False
    capabilities = tuple(phase.capability for phase in structure.phases)
    if any(capability not in constraint.supported_capabilities for capability in capabilities):
        return False
    if any(
        capabilities.count(capability) < constraint.required_capabilities.count(capability)
        for capability in set(constraint.required_capabilities)
    ):
        return False
    actual_edges = tuple(
        PhaseEdgeConstraint(
            structure.phases[edge.source_phase].capability,
            structure.phases[edge.target_phase].capability,
            edge.relation_class,
        )
        for edge in structure.phase_edges
    )
    if any(actual_edges.count(edge) < constraint.required_edges.count(edge) for edge in set(constraint.required_edges)):
        return False
    supported_relations = set(constraint.supported_relation_classes)
    if any(access.relation_class not in supported_relations for access in task.accesses):
        return False
    if any(element_type not in constraint.supported_element_types for element_type in structure.element_types):
        return False
    if any(layout not in constraint.supported_layout_classes for layout in structure.layout_classes):
        return False
    maximum = constraint.maximum_resources
    actual = structure.resources
    return (
        actual.registers_per_thread <= maximum.registers_per_thread
        and actual.shared_memory_bytes <= maximum.shared_memory_bytes
        and actual.barrier_count <= maximum.barrier_count
    )


def _tiled_task_record(task: TiledTask) -> dict[str, CanonicalValue]:
    return {
        "id": task.id,
        "family": {
            "axis_extents": tuple(axis.extent for axis in task.family.axes),
            "placement": task.family.placement,
        },
        "operation_ordinals": task.operation_ordinals,
        "operation_kinds": tuple(kind.value for kind in task.operation_kinds),
        "structure": {
            **task.structure.schedule_record,
            "phase_operation_ordinals": tuple(
                {"phase": phase.id, "operation_ordinals": phase.operation_ordinals} for phase in task.structure.phases
            ),
            "axis_ids": {
                "owner": tuple(axis.axis_id for axis in task.structure.owner_axes),
                "streamed": tuple(axis.axis_id for axis in task.structure.streamed_axes),
                "reduction": tuple(axis.axis_id for axis in task.structure.reduction_axes),
                "contraction": tuple(axis.axis_id for axis in task.structure.contraction_axes),
            },
        },
        "accesses": tuple(
            {
                "value_id": access.value_id,
                "kind": access.kind.value,
                "phase_id": access.phase_id,
                "relation_class": access.relation_class.value,
                "relation": _canonicalize(access.relation_record),
            }
            for access in task.accesses
        ),
    }


def _task_constraint_record(constraint: PhysicalTaskConstraint) -> dict[str, Any]:
    return {
        "principal_kind": constraint.principal_kind.value,
        "required_capabilities": tuple(capability.value for capability in constraint.required_capabilities),
        "supported_capabilities": tuple(capability.value for capability in constraint.supported_capabilities),
        "required_edges": tuple(
            {"source": edge.source.value, "target": edge.target.value, "relation_class": edge.relation_class.value}
            for edge in constraint.required_edges
        ),
        "axis_ranks": {
            "owner": constraint.owner_rank,
            "streamed": constraint.streamed_rank,
            "reduction": constraint.reduction_rank,
            "contraction": constraint.contraction_rank,
        },
        "supported_relation_classes": tuple(value.value for value in constraint.supported_relation_classes),
        "supported_element_types": constraint.supported_element_types,
        "supported_layout_classes": constraint.supported_layout_classes,
        "maximum_resources": {
            "registers_per_thread": constraint.maximum_resources.registers_per_thread,
            "shared_memory_bytes": constraint.maximum_resources.shared_memory_bytes,
            "barrier_count": constraint.maximum_resources.barrier_count,
        },
    }


def _dependence_record(
    dependence: TaskDependence,
    family_ids: Mapping[TaskFamily, int],
) -> dict[str, Any]:
    return {
        "relation": _task_relation_record(dependence.relation, family_ids),
        "visibility": _visibility_record(dependence.visibility),
    }


def _task_relation_record(
    relation: TaskRelation,
    family_ids: Mapping[TaskFamily, int],
) -> dict[str, Any]:
    return {
        "source_task": family_ids[_task_family(relation.source)],
        "target_task": family_ids[_task_family(relation.target)],
        "pairs": tuple({"source": pair.source, "target": pair.target} for pair in relation.pairs),
    }


def _task_family(domain: CoordinateDomain) -> TaskFamily:
    if not isinstance(domain, TaskFamily):
        raise TypeError("canonical task relations must connect task families")
    return domain


def _visibility_record(visibility: MemoryVisibility) -> dict[str, CanonicalValue]:
    return {
        "scope": visibility.scope.value,
        "release_on_notify": visibility.release_on_notify,
        "acquire_before_consumer": visibility.acquire_before_consumer,
    }


def _bounded_buffer_record(
    buffer: BoundedBufferPlan,
    *,
    buffer_id: int,
    family_ids: Mapping[TaskFamily, int],
) -> dict[str, Any]:
    family_id_by_name = {family.name: task_id for family, task_id in family_ids.items()}
    return {
        "id": buffer_id,
        "producer_task": family_ids[buffer.producer],
        "uses": tuple(_task_relation_record(relation, family_ids) for relation in buffer.uses),
        "capacity": buffer.capacity,
        "allocation_owners": buffer.allocation_owners,
        "slots": buffer.slots,
        "generations": buffer.generations,
        "last_consumers": tuple(
            {
                "item": item,
                "task": {"task": family_id_by_name[consumer.family], "coordinate": consumer.coordinate},
            }
            for item, consumer in buffer.last_consumers
        ),
        "reuse_dependences": tuple(_dependence_record(value, family_ids) for value in buffer.reuse_dependences),
    }


def _task_dependence_program_record(dependence: TaskDependenceProgram) -> dict[str, Any]:
    ordinals = tensor_program_value_ordinals(dependence.relation_program)

    def coordinate_record(coordinate: BoundedTaskCoordinate) -> dict[str, int]:
        return {
            "value": ordinals[coordinate.value],
            "lower_bound": coordinate.lower_bound,
            "upper_bound": coordinate.upper_bound,
        }

    return {
        "source_task": dependence.source_task_id,
        "target_task": dependence.target_task_id,
        "relation_program": dependence.relation_program.canonical_record,
        "runtime_bindings": tuple(
            {
                "input": ordinals[binding.input],
                "algebra_value": binding.algebra_value_ordinal,
            }
            for binding in dependence.runtime_bindings
        ),
        "source_coordinates": tuple(coordinate_record(value) for value in dependence.source_coordinates),
        "target_coordinates": tuple(coordinate_record(value) for value in dependence.target_coordinates),
        "validity": None if dependence.validity is None else ordinals[dependence.validity],
        "visibility": _visibility_record(dependence.visibility),
    }


def _verify_coordinate_bounds(coordinates: tuple[BoundedTaskCoordinate, ...], family: TaskFamily) -> None:
    if len(coordinates) != len(family.axes):
        raise ValueError("task-dependence coordinate rank does not match its task family")
    for coordinate, axis in zip(coordinates, family.axes, strict=True):
        if coordinate.upper_bound > axis.extent:
            raise ValueError("task-dependence coordinate bound exceeds its task-family extent")


def _verify_instantiated_coordinates(
    arrays: tuple[np.ndarray, ...],
    coordinates: tuple[BoundedTaskCoordinate, ...],
    family: TaskFamily,
    validity: np.ndarray,
) -> None:
    _verify_coordinate_bounds(coordinates, family)
    for array, coordinate in zip(arrays, coordinates, strict=True):
        if array.shape != coordinate.value.shape:
            raise ValueError("instantiated task coordinate has the wrong relation-domain shape")
        selected = array[validity]
        if selected.size and (np.any(selected < coordinate.lower_bound) or np.any(selected >= coordinate.upper_bound)):
            raise ValueError("instantiated task coordinate violates its proven compile-time bounds")


def _event_plan_record(
    plan: EventTensorPlan,
    *,
    family_ids: Mapping[TaskFamily, int],
    event_id: int,
) -> dict[str, Any]:
    producer = _task_family(plan.notify_relation.source)
    consumer = _task_family(plan.trigger_relation.target)
    return {
        "event": event_id,
        "required_dependence": _dependence_record(plan.required_dependence, family_ids),
        "domain_extents": tuple(axis.extent for axis in plan.domain.axes),
        "notify_relation": {
            "producer_task": family_ids[producer],
            "pairs": tuple({"producer": pair.source, "event": pair.target} for pair in plan.notify_relation.pairs),
        },
        "trigger_relation": {
            "consumer_task": family_ids[consumer],
            "pairs": tuple({"event": pair.source, "consumer": pair.target} for pair in plan.trigger_relation.pairs),
        },
        "initial_count": tuple(
            {"event": count.coordinate, "count": count.value} for count in sorted(plan.initial_count.counts)
        ),
        "memory_scope": plan.memory_scope.value,
        "generation_policy": plan.generation_policy.value,
        "visibility": _visibility_record(plan.visibility),
        "scheduling_mode": None if plan.scheduling_mode is None else plan.scheduling_mode.value,
    }


def _event_allocation_record(
    plan: EventTensorPlan,
    allocation: EventTensorAllocation,
    *,
    event_id: int,
) -> dict[str, CanonicalValue]:
    return {
        "event": event_id,
        "capacity": allocation.capacity,
        "generation_count": allocation.generation_count,
        "slots": allocation.storage.slots,
        "generations": allocation.storage.generations,
    }


def _parametric_event_record(event: ParametricEventTensorPlan) -> dict[str, Any]:
    return {
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


def _imperative_event_operation_record(operation: ImperativeEventOperation) -> dict[str, CanonicalValue]:
    return {
        "kind": operation.kind.value,
        "event_coordinate": operation.event,
        "task_coordinate": operation.task,
        "count": operation.count,
        "scope": operation.memory_scope.value,
        "release_on_notify": operation.visibility.release_on_notify,
        "acquire_before_consumer": operation.visibility.acquire_before_consumer,
        "storage_slot": operation.storage_slot,
        "generation": operation.generation,
    }


def _event_id_by_name(plans: tuple[EventTensorPlan, ...], name: str) -> int:
    matches = tuple(index for index, plan in enumerate(plans) if plan.name == name)
    if len(matches) != 1:
        raise ValueError(f"expected one Event Tensor named {name!r}, found {len(matches)}")
    return matches[0]


def _event_realization_record(
    realization: EventRealization,
    *,
    event_id: int,
    family_ids: Mapping[TaskFamily, int],
) -> dict[str, CanonicalValue]:
    record: dict[str, CanonicalValue] = {
        "event": event_id,
        "kind": realization.kind.value,
        "mechanism": realization.mechanism,
    }
    if realization.kind is not EventRealizationKind.PHYSICAL:
        assert realization.ordering is not None
        record["covering_order"] = _task_relation_record(realization.ordering, family_ids)
    return record


def _event_erasure_witness_record(
    witness: EventErasureWitness,
    *,
    event_id: int,
) -> dict[str, CanonicalValue]:
    return {
        "event": event_id,
        "schedule": witness.schedule_index,
        "order": witness.order_index,
    }


def _physical_task_order_record(order: PhysicalTaskOrder) -> dict[str, Any]:
    return {
        "kind": order.kind.value,
        "source_task": order.source_task_id,
        "target_task": order.target_task_id,
        "pairs": tuple({"source": pair.source, "target": pair.target} for pair in order.relation.pairs),
        "mechanism": _canonicalize(order.mechanism_record),
    }


def _verify_phase_dag(phases: tuple[PhysicalPhase, ...], edges: tuple[PhysicalPhaseEdge, ...]) -> None:
    successors = {phase.id: [] for phase in phases}
    indegree = {phase.id: 0 for phase in phases}
    for edge in edges:
        successors[edge.source_phase].append(edge.target_phase)
        indegree[edge.target_phase] += 1
    frontier = [phase for phase, count in indegree.items() if count == 0]
    visited = 0
    while frontier:
        source = frontier.pop()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    if visited != len(phases):
        raise ValueError("physical phase dependencies must be acyclic")


def _canonicalize(value: object) -> CanonicalValue:
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical compiler records cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical compiler record keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_canonicalize(item) for item in value)
    raise TypeError(f"unsupported canonical compiler record value: {type(value).__name__}")
