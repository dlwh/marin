# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical bounded allocations for refined internal EventTensor plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tile_lifetime.event_buffering import BoundedBufferPlan, EventRealizationAudit, EventRealizationKind
from tile_lifetime.event_dataflow import Coordinate, EventMemoryScope, EventTensorPlan, event_tensor_plan_record


class InternalEventMechanismKind(StrEnum):
    """Machine primitive required by one bounded internal Event allocation."""

    CTA_MBARRIER = "cta_mbarrier"
    CTA_TRANSACTION_MBARRIER = "cta_transaction_mbarrier"
    CTA_ASYNC_COMPLETION_QUEUE = "cta_async_completion_queue"
    DEVICE_SEMAPHORE_QUEUE = "device_semaphore_queue"


class InternalEventDeviceOperationKind(StrEnum):
    """Imperative operation emitted for one allocation relation."""

    NOTIFY = "notify"
    WAIT = "wait"


@dataclass(frozen=True, order=True)
class InternalEventNotify:
    """One task notification into a physical mechanism coordinate."""

    task_family: str
    task_coordinate: Coordinate
    mechanism_coordinate: Coordinate


@dataclass(frozen=True, order=True)
class InternalEventWait:
    """One task wait on a physical mechanism coordinate."""

    mechanism_coordinate: Coordinate
    task_family: str
    task_coordinate: Coordinate


@dataclass(frozen=True)
class InternalEventAllocation:
    """One finite physical allocation covering exact internal Event plans."""

    allocation_id: str
    covered_plans: tuple[EventTensorPlan, ...]
    mechanism_kind: InternalEventMechanismKind
    scope: EventMemoryScope
    release_on_notify: bool
    acquire_before_consumer: bool
    capacity: int
    allocation_owners: tuple[Coordinate, ...]
    mechanism_coordinates: tuple[Coordinate, ...]
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    generation_count: int
    notify_relations: tuple[InternalEventNotify, ...]
    wait_relations: tuple[InternalEventWait, ...]
    initial_counts: tuple[int, ...]
    transaction_bytes_by_coordinate: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.allocation_id or not self.covered_plans:
            raise ValueError("internal Event allocations require an identity and covered plans")
        plan_names = tuple(plan.name for plan in self.covered_plans)
        if len(set(plan_names)) != len(plan_names):
            raise ValueError("an internal Event allocation may cover each plan only once")
        item_count = len(self.mechanism_coordinates)
        if self.capacity <= 0 or self.generation_count <= 0 or item_count == 0:
            raise ValueError("internal Event allocation extents must be positive")
        if not (len(self.allocation_owners) == len(self.slots) == len(self.generations) == item_count):
            raise ValueError("internal Event allocation mappings must cover every logical coordinate")
        if any(slot < 0 or slot >= self.capacity for slot in self.slots):
            raise ValueError("internal Event allocation slot is outside its capacity")
        if any(generation < 0 or generation >= self.generation_count for generation in self.generations):
            raise ValueError("internal Event allocation generation is outside its generation count")
        if len(set(zip(self.allocation_owners, self.slots, self.generations, strict=True))) != item_count:
            raise ValueError("one internal Event owner/slot/generation tuple may identify only one logical item")
        if self.transaction_bytes_by_coordinate is not None:
            if len(self.transaction_bytes_by_coordinate) != item_count or any(
                value <= 0 for value in self.transaction_bytes_by_coordinate
            ):
                raise ValueError("internal Event transaction bytes must be positive for every logical coordinate")
        if any(plan.memory_scope is not self.scope for plan in self.covered_plans):
            raise ValueError("internal Event allocation scope must match every covered plan")
        if any(plan.visibility.release_on_notify != self.release_on_notify for plan in self.covered_plans):
            raise ValueError("internal Event allocation release visibility must match every covered plan")
        if any(plan.visibility.acquire_before_consumer != self.acquire_before_consumer for plan in self.covered_plans):
            raise ValueError("internal Event allocation acquire visibility must match every covered plan")
        if self.mechanism_kind is InternalEventMechanismKind.CTA_TRANSACTION_MBARRIER:
            if self.scope is not EventMemoryScope.CTA or self.transaction_bytes_by_coordinate is None:
                raise ValueError("transaction mbarriers require CTA scope and transaction bytes")
        elif self.mechanism_kind in (
            InternalEventMechanismKind.CTA_MBARRIER,
            InternalEventMechanismKind.CTA_ASYNC_COMPLETION_QUEUE,
        ):
            if self.scope is not EventMemoryScope.CTA or self.transaction_bytes_by_coordinate is not None:
                raise ValueError("CTA barriers and async queues do not carry transaction bytes")
        elif self.scope is not EventMemoryScope.DEVICE or self.transaction_bytes_by_coordinate is not None:
            raise ValueError("device semaphore queues require device scope without transaction bytes")
        mechanism_set = set(self.mechanism_coordinates)
        if len(mechanism_set) != item_count:
            raise ValueError("internal Event mechanism coordinates must be unique")
        if any(notify.mechanism_coordinate not in mechanism_set for notify in self.notify_relations):
            raise ValueError("internal Event notification names an unknown mechanism coordinate")
        if any(wait.mechanism_coordinate not in mechanism_set for wait in self.wait_relations):
            raise ValueError("internal Event wait names an unknown mechanism coordinate")
        if len(self.initial_counts) != item_count or any(count < 0 for count in self.initial_counts):
            raise ValueError("internal Event initial counts must cover every mechanism coordinate")
        notify_counts = {coordinate: 0 for coordinate in self.mechanism_coordinates}
        for notify in set(self.notify_relations):
            notify_counts[notify.mechanism_coordinate] += 1
        if self.initial_counts != tuple(notify_counts[coordinate] for coordinate in self.mechanism_coordinates):
            raise ValueError("internal Event initial counts must equal deduplicated notify indegrees")
        _verify_mechanism_coverage(self)


@dataclass(frozen=True)
class InternalEventDeviceOperation:
    """One compiler-owned imperative operation derived from an allocation."""

    ordinal: int
    kind: InternalEventDeviceOperationKind
    relation_ordinal: int
    task_family: str
    task_coordinate: Coordinate
    mechanism_coordinate: Coordinate
    mechanism_kind: InternalEventMechanismKind
    scope: EventMemoryScope
    allocation_owner: Coordinate
    slot: int
    generation: int
    generation_count: int
    operation_count: int
    initial_count: int
    release_acquire: bool
    transaction_bytes: int | None


@dataclass(frozen=True)
class InternalEventDeviceLegalization:
    """Exact imperative operations emitted for one internal allocation."""

    allocation_id: str
    operations: tuple[InternalEventDeviceOperation, ...]


@dataclass(frozen=True)
class InternalEventDeviceLegalizationRegion:
    """Compiler-owned device legalizations for one physical Event region."""

    region_ordinal: int
    legalizations: tuple[InternalEventDeviceLegalization, ...]

    def __post_init__(self) -> None:
        if self.region_ordinal < 0:
            raise ValueError("internal Event legalization region ordinal must be nonnegative")
        allocation_ids = tuple(legalization.allocation_id for legalization in self.legalizations)
        if len(set(allocation_ids)) != len(allocation_ids):
            raise ValueError("internal Event legalization allocation identities must be unique within a region")


def internal_event_device_legalization(allocation: InternalEventAllocation) -> InternalEventDeviceLegalization:
    """Expand one verified allocation into its exact imperative operations."""
    coordinate_ordinals = {coordinate: ordinal for ordinal, coordinate in enumerate(allocation.mechanism_coordinates)}
    operations: list[InternalEventDeviceOperation] = []
    for relation_ordinal, relation in enumerate(allocation.notify_relations):
        mechanism_ordinal = coordinate_ordinals[relation.mechanism_coordinate]
        operations.append(
            InternalEventDeviceOperation(
                ordinal=len(operations),
                kind=InternalEventDeviceOperationKind.NOTIFY,
                relation_ordinal=relation_ordinal,
                task_family=relation.task_family,
                task_coordinate=relation.task_coordinate,
                mechanism_coordinate=relation.mechanism_coordinate,
                mechanism_kind=allocation.mechanism_kind,
                scope=allocation.scope,
                allocation_owner=allocation.allocation_owners[mechanism_ordinal],
                slot=allocation.slots[mechanism_ordinal],
                generation=allocation.generations[mechanism_ordinal],
                generation_count=allocation.generation_count,
                operation_count=1,
                initial_count=allocation.initial_counts[mechanism_ordinal],
                release_acquire=allocation.release_on_notify,
                transaction_bytes=(
                    None
                    if allocation.transaction_bytes_by_coordinate is None
                    else allocation.transaction_bytes_by_coordinate[mechanism_ordinal]
                ),
            )
        )
    for relation_ordinal, relation in enumerate(allocation.wait_relations):
        mechanism_ordinal = coordinate_ordinals[relation.mechanism_coordinate]
        initial_count = allocation.initial_counts[mechanism_ordinal]
        operations.append(
            InternalEventDeviceOperation(
                ordinal=len(operations),
                kind=InternalEventDeviceOperationKind.WAIT,
                relation_ordinal=relation_ordinal,
                task_family=relation.task_family,
                task_coordinate=relation.task_coordinate,
                mechanism_coordinate=relation.mechanism_coordinate,
                mechanism_kind=allocation.mechanism_kind,
                scope=allocation.scope,
                allocation_owner=allocation.allocation_owners[mechanism_ordinal],
                slot=allocation.slots[mechanism_ordinal],
                generation=allocation.generations[mechanism_ordinal],
                generation_count=allocation.generation_count,
                operation_count=initial_count,
                initial_count=initial_count,
                release_acquire=allocation.acquire_before_consumer,
                transaction_bytes=None,
            )
        )
    return InternalEventDeviceLegalization(allocation.allocation_id, tuple(operations))


def internal_event_device_legalizations(
    allocations: tuple[InternalEventAllocation, ...],
) -> tuple[InternalEventDeviceLegalization, ...]:
    """Expand verified allocations into deterministic device legalizations."""
    allocation_ids = tuple(allocation.allocation_id for allocation in allocations)
    if len(set(allocation_ids)) != len(allocation_ids):
        raise ValueError("internal Event allocation identities must be unique")
    return tuple(internal_event_device_legalization(allocation) for allocation in allocations)


def internal_event_device_legalization_region(
    region_ordinal: int,
    allocations: tuple[InternalEventAllocation, ...],
) -> InternalEventDeviceLegalizationRegion:
    """Bind exact allocation operations to their physical provenance region."""
    return InternalEventDeviceLegalizationRegion(region_ordinal, internal_event_device_legalizations(allocations))


def internal_event_device_legalization_region_record(
    region: InternalEventDeviceLegalizationRegion,
) -> tuple[dict[str, Any], ...]:
    """Encode the in-process native-provider records for one region."""
    return tuple(
        {
            "region_ordinal": region.region_ordinal,
            "allocation_id": legalization.allocation_id,
            "operations": tuple(
                {
                    "ordinal": operation.ordinal,
                    "kind": operation.kind.value,
                    "relation_ordinal": operation.relation_ordinal,
                    "task_family": operation.task_family,
                    "task_coordinate": operation.task_coordinate,
                    "mechanism_coordinate": operation.mechanism_coordinate,
                    "mechanism_kind": operation.mechanism_kind.value,
                    "scope": operation.scope.value,
                    "allocation_owner": operation.allocation_owner,
                    "slot": operation.slot,
                    "generation": operation.generation,
                    "generation_count": operation.generation_count,
                    "operation_count": operation.operation_count,
                    "initial_count": operation.initial_count,
                    "release_acquire": operation.release_acquire,
                    "transaction_bytes": operation.transaction_bytes,
                }
                for operation in legalization.operations
            ),
        }
        for legalization in region.legalizations
    )


def internal_event_allocation_record(allocation: InternalEventAllocation) -> dict[str, Any]:
    """Serialize one allocation without naming a backend synchronization mechanism."""

    return {
        "allocation_id": allocation.allocation_id,
        "covered_plans": tuple(plan.name for plan in allocation.covered_plans),
        "mechanism_kind": allocation.mechanism_kind.value,
        "scope": allocation.scope.value,
        "release_on_notify": allocation.release_on_notify,
        "acquire_before_consumer": allocation.acquire_before_consumer,
        "capacity": allocation.capacity,
        "allocation_owners": allocation.allocation_owners,
        "mechanism_coordinates": allocation.mechanism_coordinates,
        "slots": allocation.slots,
        "generations": allocation.generations,
        "generation_count": allocation.generation_count,
        "notify_relations": tuple(
            {
                "task_family": notify.task_family,
                "task_coordinate": notify.task_coordinate,
                "mechanism_coordinate": notify.mechanism_coordinate,
            }
            for notify in allocation.notify_relations
        ),
        "wait_relations": tuple(
            {
                "mechanism_coordinate": wait.mechanism_coordinate,
                "task_family": wait.task_family,
                "task_coordinate": wait.task_coordinate,
            }
            for wait in allocation.wait_relations
        ),
        "initial_counts": allocation.initial_counts,
        "transaction_bytes_by_coordinate": allocation.transaction_bytes_by_coordinate,
    }


def _verify_mechanism_coverage(allocation: InternalEventAllocation) -> None:
    notify_coordinates: dict[tuple[str, Coordinate], set[Coordinate]] = {}
    for notify in allocation.notify_relations:
        notify_coordinates.setdefault((notify.task_family, notify.task_coordinate), set()).add(
            notify.mechanism_coordinate
        )
    waits_by_mechanism: dict[Coordinate, set[tuple[str, Coordinate]]] = {}
    for wait in allocation.wait_relations:
        waits_by_mechanism.setdefault(wait.mechanism_coordinate, set()).add((wait.task_family, wait.task_coordinate))
    scheduled = {
        (source_family, source_coordinate, target_family, target_coordinate)
        for (source_family, source_coordinate), mechanisms in notify_coordinates.items()
        for mechanism in mechanisms
        for target_family, target_coordinate in waits_by_mechanism.get(mechanism, set())
    }
    for plan in allocation.covered_plans:
        relation = plan.required_dependence.relation
        required = {(relation.source.name, pair.source, relation.target.name, pair.target) for pair in relation.pairs}
        missing = required - scheduled
        if missing:
            raise ValueError(
                f"internal Event mechanism {allocation.allocation_id} omits {plan.name} edges {sorted(missing)}"
            )


def verify_internal_event_allocations(
    audit: EventRealizationAudit,
    allocations: tuple[InternalEventAllocation, ...],
) -> None:
    """Require physical realizations to have allocations and erased ones to have none."""
    allocation_ids = tuple(allocation.allocation_id for allocation in allocations)
    if len(set(allocation_ids)) != len(allocation_ids):
        raise ValueError("internal Event allocation identities must be unique")
    coverage: dict[str, list[str]] = {}
    for allocation in allocations:
        for plan in allocation.covered_plans:
            coverage.setdefault(plan.name, []).append(allocation.allocation_id)
    for realization in audit.entries:
        covering = coverage.get(realization.plan_name, [])
        if realization.kind is EventRealizationKind.PHYSICAL and not covering:
            raise ValueError(f"physical internal Event plan {realization.plan_name} has no mechanism allocation")
        if realization.kind is not EventRealizationKind.PHYSICAL and covering:
            raise ValueError(f"erased internal Event plan {realization.plan_name} has mechanism allocations {covering}")
    unknown = set(coverage) - {entry.plan_name for entry in audit.entries}
    if unknown:
        raise ValueError(f"internal Event allocations cover plans outside the realization audit: {sorted(unknown)}")


def internal_event_realization_records(
    audit: EventRealizationAudit,
    plans_by_name: dict[str, EventTensorPlan],
    allocations: tuple[InternalEventAllocation, ...],
) -> tuple[dict[str, Any], ...]:
    """Serialize realization decisions with their verified allocation join."""
    verify_internal_event_allocations(audit, allocations)
    return tuple(
        {
            "plan": entry.plan_name,
            "event": event_tensor_plan_record(plans_by_name[entry.plan_name]),
            "kind": entry.kind.value,
            "mechanism": entry.mechanism,
            "allocation_ids": tuple(
                allocation.allocation_id
                for allocation in allocations
                if entry.plan_name in {plan.name for plan in allocation.covered_plans}
            ),
        }
        for entry in audit.entries
    )


def phased_internal_event_allocation(
    *,
    allocation_id: str,
    plans: tuple[EventTensorPlan, ...],
    capacity: int,
    allocation_owners: tuple[Coordinate, ...],
    slots: tuple[int, ...],
    generations: tuple[int, ...],
    mechanism_kind: InternalEventMechanismKind,
) -> InternalEventAllocation:
    """Bind congruent Event domains to an explicit owner-relative phased allocation."""
    coordinates = plans[0].domain.coordinates
    if any(plan.domain.coordinates != coordinates for plan in plans):
        raise ValueError("phased internal Event allocation requires congruent Event domains")
    notifies = tuple(
        sorted(
            {
                InternalEventNotify(plan.notify_relation.source.name, pair.source, pair.target)
                for plan in plans
                for pair in plan.notify_relation.pairs
            }
        )
    )
    waits = tuple(
        sorted(
            {
                InternalEventWait(pair.source, plan.trigger_relation.target.name, pair.target)
                for plan in plans
                for pair in plan.trigger_relation.pairs
            }
        )
    )
    counts = {coordinate: 0 for coordinate in coordinates}
    for notify in notifies:
        counts[notify.mechanism_coordinate] += 1
    return InternalEventAllocation(
        allocation_id=allocation_id,
        covered_plans=plans,
        mechanism_kind=mechanism_kind,
        scope=plans[0].memory_scope,
        release_on_notify=plans[0].visibility.release_on_notify,
        acquire_before_consumer=plans[0].visibility.acquire_before_consumer,
        capacity=capacity,
        allocation_owners=allocation_owners,
        mechanism_coordinates=coordinates,
        slots=slots,
        generations=generations,
        generation_count=max(generations, default=0) + 1,
        notify_relations=notifies,
        wait_relations=waits,
        initial_counts=tuple(counts[coordinate] for coordinate in coordinates),
    )


def bounded_buffer_internal_event_allocation(
    *,
    allocation_id: str,
    buffer: BoundedBufferPlan,
    plans: tuple[EventTensorPlan, ...],
    transaction_bytes_by_coordinate: tuple[int, ...] | None = None,
) -> InternalEventAllocation:
    """Bind exact Event plans to an existing bounded-buffer assignment."""
    mechanism_coordinates = buffer.producer.coordinates
    notifies: set[InternalEventNotify] = set()
    waits: set[InternalEventWait] = set()
    for plan in plans:
        relation = plan.required_dependence.relation
        if plan.required_dependence.relation.source == buffer.producer:
            for pair in relation.pairs:
                notifies.add(InternalEventNotify(relation.source.name, pair.source, pair.source))
                waits.add(InternalEventWait(pair.source, relation.target.name, pair.target))
        elif plan.required_dependence.relation.target == buffer.producer:
            for pair in relation.pairs:
                notifies.add(InternalEventNotify(relation.source.name, pair.source, pair.target))
                waits.add(InternalEventWait(pair.target, relation.target.name, pair.target))
        else:
            raise ValueError(f"internal Event plan {plan.name} does not access bounded buffer {buffer.name}")
    sorted_notifies = tuple(sorted(notifies))
    sorted_waits = tuple(sorted(waits))
    counts = {coordinate: 0 for coordinate in mechanism_coordinates}
    for notify in sorted_notifies:
        counts[notify.mechanism_coordinate] += 1

    return InternalEventAllocation(
        allocation_id=allocation_id,
        covered_plans=plans,
        mechanism_kind=(
            InternalEventMechanismKind.CTA_TRANSACTION_MBARRIER
            if transaction_bytes_by_coordinate is not None
            else InternalEventMechanismKind.CTA_MBARRIER
        ),
        scope=plans[0].memory_scope,
        release_on_notify=plans[0].visibility.release_on_notify,
        acquire_before_consumer=plans[0].visibility.acquire_before_consumer,
        capacity=buffer.capacity,
        allocation_owners=buffer.allocation_owners,
        mechanism_coordinates=mechanism_coordinates,
        slots=buffer.slots,
        generations=buffer.generations,
        generation_count=max(buffer.generations, default=0) + 1,
        notify_relations=sorted_notifies,
        wait_relations=sorted_waits,
        initial_counts=tuple(counts[coordinate] for coordinate in mechanism_coordinates),
        transaction_bytes_by_coordinate=transaction_bytes_by_coordinate,
    )
