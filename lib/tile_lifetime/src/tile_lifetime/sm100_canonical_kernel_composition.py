# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose generic SM100 task schedules into one executable boundary.

This module starts after algebra, ownership, task, and Event Tensor lowering.
It does not recover a source workload.  Each kernel component is selected from
one or more generic task structures, and every cross-component Event remains an
explicit concrete or runtime-parametric boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from tile_lifetime.event_dataflow import CoordinateDomain, TaskFamily
from tile_lifetime.physical_task_contract import (
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    RelationAccessClass,
    TaskCapability,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskValueAccessKind,
    TiledTask,
    canonical_fingerprint,
    install_physical_schedule_cover,
    physical_schedule_is_eligible,
)
from tile_lifetime.sm100_projected_routed_lowering import (
    SM100ComposedPhysicalLowering,
    SM100RuntimeRelationABI,
)

_PROJECTION_CAPABILITIES = (
    TaskCapability.MATRIX_CONTRACT,
    TaskCapability.INDEX_RELATION,
    TaskCapability.SCALAR_MAP,
)
_PROJECTION_RELATIONS = (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE)
_PROJECTION_ELEMENT_TYPES = ("bf16", "fp32")
_PROJECTION_LAYOUTS = ("row_major",)
_PROJECTION_RESOURCES = TaskResourceEstimate(255, 232_448, 32)


@dataclass(frozen=True)
class SM100ProjectionContractLowering:
    """One generic projection-shaped Contract schedule attachment."""

    source_request_fingerprint: str
    task_id: int
    physical_schedule: PhysicalSchedule
    matrix_phase_id: int
    contraction_extent: int

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_request_fingerprint": self.source_request_fingerprint,
            "task_id": self.task_id,
            "schedule": self.physical_schedule.canonical_record,
            "matrix_phase_id": self.matrix_phase_id,
            "contraction_extent": self.contraction_extent,
        }


@dataclass(frozen=True)
class SM100ConcreteEventBoundary:
    """One storage-bound concrete Event crossing kernel components."""

    event_id: int
    producer_component: int
    consumer_component: int
    capacity: int
    generation_count: int
    event_count: int
    notify_operation_count: int
    wait_operation_count: int
    memory_scope: str
    release_on_notify: bool
    acquire_before_consumer: bool
    realization: str = "imperative_wait_notify"

    def __post_init__(self) -> None:
        counts = (
            self.capacity,
            self.generation_count,
            self.event_count,
            self.notify_operation_count,
            self.wait_operation_count,
        )
        if self.event_id < 0 or min(counts) <= 0:
            raise ValueError("a concrete SM100 Event boundary requires positive storage and operations")
        if self.realization != "imperative_wait_notify":
            raise ValueError("the current cross-kernel SM100 Event boundary requires imperative wait/notify")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "producer_component": self.producer_component,
            "consumer_component": self.consumer_component,
            "capacity": self.capacity,
            "generation_count": self.generation_count,
            "event_count": self.event_count,
            "notify_operation_count": self.notify_operation_count,
            "wait_operation_count": self.wait_operation_count,
            "memory_scope": self.memory_scope,
            "release_on_notify": self.release_on_notify,
            "acquire_before_consumer": self.acquire_before_consumer,
            "realization": self.realization,
        }


@dataclass(frozen=True)
class SM100ParametricEventBoundary:
    """One runtime-instantiated exact relation and bounded Event allocation."""

    dependence_program_id: int
    producer_component: int
    consumer_component: int
    relation: SM100RuntimeRelationABI

    def __post_init__(self) -> None:
        if self.dependence_program_id < 0:
            raise ValueError("parametric SM100 Event ids must be non-negative")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "dependence_program_id": self.dependence_program_id,
            "producer_component": self.producer_component,
            "consumer_component": self.consumer_component,
            "relation": self.relation.canonical_record,
        }


@dataclass(frozen=True)
class SM100KernelComponent:
    """One launchable schedule component and its exact algebra payload."""

    id: int
    schedule_index: int
    schedule_family_fingerprint: str
    schedule_fingerprint: str
    task_ids: tuple[int, ...]
    operation_ordinals: tuple[int, ...]
    operations: tuple[dict[str, Any], ...]
    live_input_value_ids: tuple[int, ...]
    live_output_value_ids: tuple[int, ...]
    concrete_wait_event_ids: tuple[int, ...]
    concrete_notify_event_ids: tuple[int, ...]
    parametric_wait_event_ids: tuple[int, ...]
    parametric_notify_event_ids: tuple[int, ...]

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schedule_index": self.schedule_index,
            "schedule_family_fingerprint": self.schedule_family_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "task_ids": self.task_ids,
            "operation_ordinals": self.operation_ordinals,
            "operations": self.operations,
            "live_input_value_ids": self.live_input_value_ids,
            "live_output_value_ids": self.live_output_value_ids,
            "concrete_wait_event_ids": self.concrete_wait_event_ids,
            "concrete_notify_event_ids": self.concrete_notify_event_ids,
            "parametric_wait_event_ids": self.parametric_wait_event_ids,
            "parametric_notify_event_ids": self.parametric_notify_event_ids,
        }


@dataclass(frozen=True)
class SM100CanonicalKernelComposition:
    """Complete schedule and Event ABI awaiting only device-code emitters."""

    source_request_fingerprint: str
    scheduled_request: ExecutablePhysicalRequest
    projection_lowerings: tuple[SM100ProjectionContractLowering, ...]
    components: tuple[SM100KernelComponent, ...]
    concrete_events: tuple[SM100ConcreteEventBoundary, ...]
    parametric_events: tuple[SM100ParametricEventBoundary, ...]

    def __post_init__(self) -> None:
        if tuple(component.id for component in self.components) != tuple(range(len(self.components))):
            raise ValueError("SM100 kernel component ids must be contiguous from zero")
        task_ids = tuple(task_id for component in self.components for task_id in component.task_ids)
        expected = tuple(range(len(self.scheduled_request.task_plan.tasks)))
        if tuple(sorted(task_ids)) != expected or len(task_ids) != len(set(task_ids)):
            raise ValueError("SM100 kernel components must cover every task exactly once")
        component_ids = {component.id for component in self.components}
        for event in (*self.concrete_events, *self.parametric_events):
            if event.producer_component not in component_ids or event.consumer_component not in component_ids:
                raise ValueError("SM100 Event boundary references an unavailable kernel component")
            if event.producer_component == event.consumer_component:
                raise ValueError("SM100 cross-component Event cannot have one component at both endpoints")
        retained_names = {entry.plan_name for entry in self.scheduled_request.event_realizations.audit.physical}
        expected_concrete_ids = tuple(
            event_id for event_id, plan in enumerate(self.scheduled_request.event_plans) if plan.name in retained_names
        )
        if tuple(event.event_id for event in self.concrete_events) != expected_concrete_ids:
            raise ValueError("SM100 concrete Event boundaries must cover every retained Event exactly")
        expected_parametric_ids = tuple(range(len(self.scheduled_request.event_programs)))
        if tuple(event.dependence_program_id for event in self.parametric_events) != expected_parametric_ids:
            raise ValueError("SM100 parametric Event boundaries must cover every runtime dependence exactly")

    @property
    def complete_task_and_event_coverage(self) -> bool:
        """Return whether all scheduled work and retained Events have an ABI owner."""
        return True

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_request_fingerprint": self.source_request_fingerprint,
            "scheduled_request_fingerprint": self.scheduled_request.fingerprint,
            "algebra_fingerprint": self.scheduled_request.algebra_fingerprint,
            "task_plan_fingerprint": self.scheduled_request.task_plan.fingerprint,
            "projection_lowerings": tuple(item.canonical_record for item in self.projection_lowerings),
            "components": tuple(component.canonical_record for component in self.components),
            "concrete_events": tuple(event.canonical_record for event in self.concrete_events),
            "parametric_events": tuple(event.canonical_record for event in self.parametric_events),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


def select_sm100_projection_contract_lowerings(
    request: ExecutablePhysicalRequest,
    *,
    task_ids: tuple[int, ...],
    source_request_fingerprint: str | None = None,
) -> tuple[SM100ProjectionContractLowering, ...]:
    """Select single-kernel Contract schedules from generic task structure."""
    if request.target is not PhysicalTarget.SM100 or request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        return ()
    if source_request_fingerprint is None:
        source_request_fingerprint = request.fingerprint
    elif len(source_request_fingerprint) != 64:
        raise ValueError("source request fingerprint must be a canonical SHA-256 digest")
    selected = []
    for task_id in task_ids:
        task = request.task_plan.tasks[task_id]
        if not _projection_task_is_supported(request, task):
            continue
        matrix_phase = next(
            phase for phase in task.structure.phases if phase.capability is TaskCapability.MATRIX_CONTRACT
        )
        family = _projection_schedule_family(task)
        schedule = PhysicalSchedule(
            family=family,
            task_ids=(task_id,),
            parameters={
                "matrix_instruction": "tcgen05_mma",
                "output_tile_rows": 128,
                "output_tile_columns": 128,
                "contraction_tile": 128,
                "pipeline_stages": 4,
                "cluster_ctas": 2,
                "threads": 256,
                "matrix_phase": matrix_phase.id,
                "phase_count": len(task.structure.phases),
            },
        )
        if not physical_schedule_is_eligible(schedule, request.task_plan):
            raise ValueError("selected SM100 projection schedule does not satisfy its declared family")
        selected.append(
            SM100ProjectionContractLowering(
                source_request_fingerprint=source_request_fingerprint,
                task_id=task_id,
                physical_schedule=schedule,
                matrix_phase_id=matrix_phase.id,
                contraction_extent=task.structure.contraction_axes[0].extent,
            )
        )
    return tuple(selected)


def select_sm100_projection_simt_lowerings(
    request: ExecutablePhysicalRequest,
    *,
    task_ids: tuple[int, ...],
    source_request_fingerprint: str | None = None,
) -> tuple[SM100ProjectionContractLowering, ...]:
    """Select the executable SIMT fallback for generic projection components."""
    if request.target is not PhysicalTarget.SM100 or request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        return ()
    if source_request_fingerprint is None:
        source_request_fingerprint = request.fingerprint
    elif len(source_request_fingerprint) != 64:
        raise ValueError("source request fingerprint must be a canonical SHA-256 digest")
    selected = []
    for task_id in task_ids:
        task = request.task_plan.tasks[task_id]
        if not _projection_task_is_supported(request, task):
            continue
        matrix_phase = next(
            phase for phase in task.structure.phases if phase.capability is TaskCapability.MATRIX_CONTRACT
        )
        schedule = PhysicalSchedule(
            family=_projection_schedule_family(task, name="sm100_simt_contract_relations_maps"),
            task_ids=(task_id,),
            parameters={
                "matrix_instruction": "cuda_core_fma",
                "operation_realization": "ordered_multi_launch",
                "threads": 256,
                "matrix_phase": matrix_phase.id,
                "phase_count": len(task.structure.phases),
            },
        )
        if not physical_schedule_is_eligible(schedule, request.task_plan):
            raise ValueError("selected SM100 SIMT projection schedule does not satisfy its declared family")
        selected.append(
            SM100ProjectionContractLowering(
                source_request_fingerprint=source_request_fingerprint,
                task_id=task_id,
                physical_schedule=schedule,
                matrix_phase_id=matrix_phase.id,
                contraction_extent=task.structure.contraction_axes[0].extent,
            )
        )
    return tuple(selected)


def lower_sm100_canonical_kernel_composition(
    lowering: SM100ComposedPhysicalLowering,
) -> SM100CanonicalKernelComposition:
    """Cover residual Contracts with the plan-only tensor-core schedule."""
    request = lowering.request
    projection_lowerings = select_sm100_projection_contract_lowerings(
        request,
        task_ids=lowering.emitter_abi.residual_task_ids,
        source_request_fingerprint=lowering.emitter_abi.scheduled_request_fingerprint,
    )
    return _lower_sm100_canonical_kernel_composition(lowering, projection_lowerings)


def lower_sm100_canonical_simt_kernel_composition(
    lowering: SM100ComposedPhysicalLowering,
) -> SM100CanonicalKernelComposition:
    """Cover residual Contracts with the executable multi-launch SIMT schedule."""
    request = lowering.request
    projection_lowerings = select_sm100_projection_simt_lowerings(
        request,
        task_ids=lowering.emitter_abi.residual_task_ids,
        source_request_fingerprint=lowering.emitter_abi.scheduled_request_fingerprint,
    )
    return _lower_sm100_canonical_kernel_composition(
        lowering,
        projection_lowerings,
        remaining_schedule=_simt_fallback_schedule,
    )


def _lower_sm100_canonical_kernel_composition(
    lowering: SM100ComposedPhysicalLowering,
    projection_lowerings: tuple[SM100ProjectionContractLowering, ...],
    *,
    remaining_schedule: Callable[[PhysicalSchedule], PhysicalSchedule] | None = None,
) -> SM100CanonicalKernelComposition:
    request = lowering.request
    if request.target is not PhysicalTarget.SM100:
        raise ValueError("SM100 kernel composition requires an SM100 request")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("the current SM100 projection composition requires FAST numerical policy")

    source_request_fingerprint = lowering.emitter_abi.scheduled_request_fingerprint
    residual_task_ids = lowering.emitter_abi.residual_task_ids
    covered_residuals = tuple(item.task_id for item in projection_lowerings)
    if covered_residuals != residual_task_ids:
        missing = tuple(task_id for task_id in residual_task_ids if task_id not in set(covered_residuals))
        raise ValueError(f"SM100 projection schedule does not cover residual tasks {missing}")

    projection_by_task = {item.task_id: item for item in projection_lowerings}
    schedules = tuple(
        (
            projection_by_task[schedule.task_ids[0]].physical_schedule
            if len(schedule.task_ids) == 1 and schedule.task_ids[0] in projection_by_task
            else remaining_schedule(schedule) if remaining_schedule is not None else schedule
        )
        for schedule in request.schedules
    )
    installed = install_physical_schedule_cover(
        request,
        schedules,
        source_request_fingerprint=source_request_fingerprint,
    )
    scheduled = replace(
        installed,
        provenance={
            **installed.provenance,
            "kernel_composition": {
                "source_request_fingerprint": source_request_fingerprint,
                "projection_schedule_fingerprints": tuple(
                    item.physical_schedule.fingerprint for item in projection_lowerings
                ),
                "concrete_event_realization": "imperative_wait_notify",
                "parametric_event_realization": "runtime_exact_relation_wait_notify",
            },
        },
    )

    task_to_component = {
        task_id: component_id
        for component_id, schedule in enumerate(scheduled.schedules)
        for task_id in schedule.task_ids
    }
    family_to_task = {task.family: task.id for task in scheduled.task_plan.tasks}
    physical_names = {entry.plan_name for entry in scheduled.event_realizations.audit.physical}
    concrete_events = tuple(
        SM100ConcreteEventBoundary(
            event_id=event_id,
            producer_component=_component_for_family(
                plan.scheduled_dependence.source, family_to_task, task_to_component
            ),
            consumer_component=_component_for_family(
                plan.scheduled_dependence.target, family_to_task, task_to_component
            ),
            capacity=scheduled.event_allocations[event_id].capacity,
            generation_count=scheduled.event_allocations[event_id].generation_count,
            event_count=len(plan.domain.coordinates),
            notify_operation_count=len(plan.notify_relation.pairs),
            wait_operation_count=len(plan.trigger_relation.pairs),
            memory_scope=plan.memory_scope.value,
            release_on_notify=plan.visibility.release_on_notify,
            acquire_before_consumer=plan.visibility.acquire_before_consumer,
        )
        for event_id, plan in enumerate(scheduled.event_plans)
        if plan.name in physical_names
    )
    parametric_events = tuple(
        SM100ParametricEventBoundary(
            dependence_program_id=dependence_program_id,
            producer_component=task_to_component[relation.source_task_id],
            consumer_component=task_to_component[relation.target_task_id],
            relation=relation,
        )
        for dependence_program_id, relation in enumerate(lowering.emitter_abi.runtime_relations)
    )

    operations = scheduled.algebra_record.get("operations")
    if not isinstance(operations, (list, tuple)) or not all(isinstance(operation, dict) for operation in operations):
        raise ValueError("canonical algebra record does not contain mapping-valued operations")
    components = tuple(
        _kernel_component(
            scheduled,
            schedule_index=schedule_index,
            task_to_component=task_to_component,
            operations=operations,
            concrete_events=concrete_events,
            parametric_events=parametric_events,
        )
        for schedule_index in range(len(scheduled.schedules))
    )
    return SM100CanonicalKernelComposition(
        source_request_fingerprint=source_request_fingerprint,
        scheduled_request=scheduled,
        projection_lowerings=projection_lowerings,
        components=components,
        concrete_events=concrete_events,
        parametric_events=parametric_events,
    )


def _component_for_family(
    domain: CoordinateDomain,
    family_to_task: dict[TaskFamily, int],
    task_to_component: dict[int, int],
) -> int:
    if not isinstance(domain, TaskFamily):
        raise TypeError("SM100 kernel composition requires task-family Event endpoints")
    return task_to_component[family_to_task[domain]]


def _projection_task_is_supported(request: ExecutablePhysicalRequest, task: TiledTask) -> bool:
    structure = task.structure
    capabilities = tuple(phase.capability for phase in structure.phases)
    matrix_phases = tuple(phase for phase in structure.phases if phase.capability is TaskCapability.MATRIX_CONTRACT)
    if len(matrix_phases) != 1 or len(matrix_phases[0].operation_ordinals) != 1:
        return False
    operations = request.algebra_record.get("operations")
    axes = request.algebra_record.get("axes")
    if not isinstance(operations, (list, tuple)) or not isinstance(axes, (list, tuple)):
        return False
    contract = operations[matrix_phases[0].operation_ordinals[0]]
    if not isinstance(contract, dict) or contract.get("kind") != "contract":
        return False
    results = contract.get("results")
    if not isinstance(results, (list, tuple)) or len(results) != 1 or not isinstance(results[0], dict):
        return False
    extent_by_axis = {
        axis.get("id"): axis.get("extent")
        for axis in axes
        if isinstance(axis, dict) and isinstance(axis.get("id"), str) and isinstance(axis.get("extent"), int)
    }
    result_axis_ids = results[0].get("axes")
    if not isinstance(result_axis_ids, (list, tuple)):
        return False
    return (
        structure.principal_kind is TaskOperationKind.CONTRACT
        and len(structure.owner_axes) in (1, 2)
        and not structure.streamed_axes
        and not structure.reduction_axes
        and len(structure.contraction_axes) == 1
        and structure.contraction_axes[0].extent == 128
        and bool(result_axis_ids)
        and extent_by_axis.get(result_axis_ids[-1]) == 128
        and capabilities.count(TaskCapability.MATRIX_CONTRACT) == 1
        and set(capabilities) <= set(_PROJECTION_CAPABILITIES)
        and all(
            edge.source_phase + 1 == edge.target_phase and edge.relation_class is RelationAccessClass.IDENTITY
            for edge in structure.phase_edges
        )
        and all(access.relation_class in _PROJECTION_RELATIONS for access in task.accesses)
        and set(structure.element_types) <= set(_PROJECTION_ELEMENT_TYPES)
        and set(structure.layout_classes) <= set(_PROJECTION_LAYOUTS)
        and structure.resources.registers_per_thread <= _PROJECTION_RESOURCES.registers_per_thread
        and structure.resources.shared_memory_bytes <= _PROJECTION_RESOURCES.shared_memory_bytes
        and structure.resources.barrier_count <= _PROJECTION_RESOURCES.barrier_count
    )


def _projection_schedule_family(
    task: TiledTask,
    *,
    name: str = "sm100_contract_relations_maps_128",
) -> PhysicalScheduleFamily:
    structure = task.structure
    return PhysicalScheduleFamily(
        name=name,
        target=PhysicalTarget.SM100,
        tasks=(
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.CONTRACT,
                required_capabilities=(TaskCapability.MATRIX_CONTRACT,),
                supported_capabilities=_PROJECTION_CAPABILITIES,
                required_edges=(),
                owner_rank=len(structure.owner_axes),
                streamed_rank=0,
                reduction_rank=0,
                contraction_rank=1,
                supported_relation_classes=_PROJECTION_RELATIONS,
                supported_element_types=_PROJECTION_ELEMENT_TYPES,
                supported_layout_classes=_PROJECTION_LAYOUTS,
                maximum_resources=_PROJECTION_RESOURCES,
            ),
        ),
    )


def _simt_fallback_schedule(schedule: PhysicalSchedule) -> PhysicalSchedule:
    return PhysicalSchedule(
        family=replace(schedule.family, name="sm100_simt_canonical_multi_launch"),
        task_ids=schedule.task_ids,
        parameters={
            "matrix_instruction": "cuda_core_fma",
            "operation_realization": "ordered_multi_launch",
            "threads": 256,
        },
        enforced_orders=schedule.enforced_orders,
    )


def _kernel_component(
    request: ExecutablePhysicalRequest,
    *,
    schedule_index: int,
    task_to_component: dict[int, int],
    operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    concrete_events: tuple[SM100ConcreteEventBoundary, ...],
    parametric_events: tuple[SM100ParametricEventBoundary, ...],
) -> SM100KernelComponent:
    schedule = request.schedules[schedule_index]
    tasks = tuple(request.task_plan.tasks[task_id] for task_id in schedule.task_ids)
    ordinals = tuple(sorted(ordinal for task in tasks for ordinal in task.operation_ordinals))
    live_inputs = tuple(
        sorted(
            {access.value_id for task in tasks for access in task.accesses if access.kind is TaskValueAccessKind.LIVE_IN}
        )
    )
    live_outputs = tuple(
        sorted(
            {
                access.value_id
                for task in tasks
                for access in task.accesses
                if access.kind is TaskValueAccessKind.LIVE_OUT
            }
        )
    )
    component_id = task_to_component[schedule.task_ids[0]]
    return SM100KernelComponent(
        id=component_id,
        schedule_index=schedule_index,
        schedule_family_fingerprint=schedule.family.fingerprint,
        schedule_fingerprint=schedule.fingerprint,
        task_ids=schedule.task_ids,
        operation_ordinals=ordinals,
        operations=tuple(dict(operations[ordinal]) for ordinal in ordinals),
        live_input_value_ids=live_inputs,
        live_output_value_ids=live_outputs,
        concrete_wait_event_ids=tuple(
            event.event_id for event in concrete_events if event.consumer_component == component_id
        ),
        concrete_notify_event_ids=tuple(
            event.event_id for event in concrete_events if event.producer_component == component_id
        ),
        parametric_wait_event_ids=tuple(
            event.dependence_program_id for event in parametric_events if event.consumer_component == component_id
        ),
        parametric_notify_event_ids=tuple(
            event.dependence_program_id for event in parametric_events if event.producer_component == component_id
        ),
    )
