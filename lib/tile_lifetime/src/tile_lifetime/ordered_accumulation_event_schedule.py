# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact Event planning for deterministic ordered accumulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from tile_lifetime.event_buffering import (
    EventRealizationAudit,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    Coordinate,
    EventDataflowError,
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventTensorPlan,
    MemoryVisibility,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventAllocation,
    InternalEventMechanismKind,
    InternalEventNotify,
    InternalEventWait,
    internal_event_allocation_record,
    internal_event_realization_records,
    verify_internal_event_allocations,
)


@dataclass(frozen=True)
class OwnedPhasedEventAllocation:
    """One monotonically increasing counter per exact accumulation owner."""

    plan: EventTensorPlan
    active_coordinates: tuple[Coordinate, ...]
    allocation_owners: tuple[Coordinate, ...]
    generations: tuple[int, ...]

    def __post_init__(self) -> None:
        item_count = len(self.active_coordinates)
        if len(self.allocation_owners) != item_count or len(self.generations) != item_count:
            raise ValueError("owned phased Event allocation must cover every active contribution")
        if len(set(self.active_coordinates)) != item_count:
            raise ValueError("owned phased Event coordinates must be unique")
        if len(set(zip(self.allocation_owners, self.generations, strict=True))) != item_count:
            raise ValueError("one owner/generation pair may identify only one active contribution")
        generations_by_owner: dict[Coordinate, list[int]] = defaultdict(list)
        for owner, generation in zip(self.allocation_owners, self.generations, strict=True):
            generations_by_owner[owner].append(generation)
        for owner, generations in generations_by_owner.items():
            if sorted(generations) != list(range(len(generations))):
                raise ValueError(f"ordered accumulator owner {owner} generations must start at zero and be contiguous")


@dataclass(frozen=True)
class OrderedAccumulationEventSchedule:
    """Exact contributor chain and its device-scope queue realization."""

    name: str
    program: EventDataflowProgram
    contribution_task: TaskFamily
    allocation: OwnedPhasedEventAllocation
    audit: EventRealizationAudit


def derive_ordered_accumulation_event_schedule(
    *,
    name: str,
    contribution_task: TaskFamily,
    active_coordinates: tuple[Coordinate, ...],
    allocation_owner_for: dict[Coordinate, Coordinate],
    order_key_for: dict[Coordinate, tuple[int, ...]],
) -> OrderedAccumulationEventSchedule:
    """Derive adjacent contributor Events for one deterministic reduction.

    The caller supplies the exact active task subset and two generic index
    relations: a projection to the accumulator owner and a total order within
    each owner. Coordinates outside ``active_coordinates`` are predicated
    no-op task instances and create no accumulator dependency.
    """
    if not name:
        raise ValueError("ordered accumulation Event schedule name must be non-empty")
    active = set(active_coordinates)
    if len(active) != len(active_coordinates):
        raise EventDataflowError(("ordered accumulation active coordinates must be unique",))
    unknown = active - set(contribution_task.coordinates)
    if unknown:
        raise EventDataflowError((f"ordered accumulation coordinates are outside the task domain: {sorted(unknown)}",))
    if set(allocation_owner_for) != active or set(order_key_for) != active:
        raise EventDataflowError(("ordered accumulation owner and order maps must cover exactly the active domain",))

    by_owner: dict[Coordinate, list[Coordinate]] = defaultdict(list)
    for coordinate in active_coordinates:
        by_owner[allocation_owner_for[coordinate]].append(coordinate)

    predecessor_pairs: list[tuple[Coordinate, Coordinate]] = []
    generation_for: dict[Coordinate, int] = {}
    for owner, coordinates in sorted(by_owner.items()):
        ordered = sorted(coordinates, key=order_key_for.__getitem__)
        keys = tuple(order_key_for[coordinate] for coordinate in ordered)
        if len(set(keys)) != len(keys):
            raise EventDataflowError((f"ordered accumulator owner {owner} has duplicate order keys",))
        generation_for.update((coordinate, generation) for generation, coordinate in enumerate(ordered))
        predecessor_pairs.extend(pairwise(ordered))

    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    dependence = TaskDependence(
        TaskRelation.from_pairs(contribution_task, contribution_task, tuple(predecessor_pairs)),
        visibility,
    )
    plan = derive_event_tensor_plan(
        dependence,
        name=f"{name}.ordered_contribution",
        generation_policy=EventGenerationPolicy.PHASED,
        memory_scope=EventMemoryScope.DEVICE,
    )
    program = EventDataflowProgram((contribution_task,), (dependence,), (plan,))
    verify_event_dataflow_program(program)
    allocation = OwnedPhasedEventAllocation(
        plan,
        active_coordinates,
        tuple(allocation_owner_for[coordinate] for coordinate in active_coordinates),
        tuple(generation_for[coordinate] for coordinate in active_coordinates),
    )
    audit = verify_event_realizations(
        program,
        (
            physical_event_realization(
                plan,
                mechanism="device semaphore queue with owner-relative phased generations",
                reason="contributors to one accumulator must commit in the selected deterministic order",
            ),
        ),
    )
    return OrderedAccumulationEventSchedule(name, program, contribution_task, allocation, audit)


def ordered_accumulation_event_schedule_record(schedule: OrderedAccumulationEventSchedule) -> dict[str, Any]:
    """Serialize the exact contributor relation and owner-relative queue."""
    plan = schedule.allocation.plan
    allocation = owned_phased_internal_event_allocation(schedule)
    verify_internal_event_allocations(schedule.audit, (allocation,))
    return {
        "schema_version": 7,
        "name": schedule.name,
        "task_axes": tuple({"name": axis.name, "extent": axis.extent} for axis in schedule.contribution_task.axes),
        "predecessor_pairs": tuple((pair.source, pair.target) for pair in plan.required_dependence.relation.pairs),
        "realizations": internal_event_realization_records(
            schedule.audit,
            {plan.name: plan},
            (allocation,),
        ),
        "allocations": (internal_event_allocation_record(allocation),),
    }


def owned_phased_internal_event_allocation(
    schedule: OrderedAccumulationEventSchedule,
) -> InternalEventAllocation:
    """Return the normalized owner-relative allocation for an ordered queue."""
    allocation = schedule.allocation
    plan = allocation.plan
    notifies = tuple(
        InternalEventNotify(plan.notify_relation.source.name, pair.source, pair.target)
        for pair in plan.required_dependence.relation.pairs
    )
    waits = tuple(
        InternalEventWait(pair.target, plan.trigger_relation.target.name, pair.target)
        for pair in plan.required_dependence.relation.pairs
    )
    counts = {coordinate: 0 for coordinate in allocation.active_coordinates}
    for notify in notifies:
        counts[notify.mechanism_coordinate] += 1
    return InternalEventAllocation(
        allocation_id=f"{schedule.name}.allocation",
        covered_plans=(allocation.plan,),
        mechanism_kind=InternalEventMechanismKind.DEVICE_SEMAPHORE_QUEUE,
        scope=plan.memory_scope,
        release_on_notify=plan.visibility.release_on_notify,
        acquire_before_consumer=plan.visibility.acquire_before_consumer,
        capacity=1,
        allocation_owners=allocation.allocation_owners,
        mechanism_coordinates=allocation.active_coordinates,
        slots=(0,) * len(allocation.active_coordinates),
        generations=allocation.generations,
        generation_count=max(allocation.generations, default=0) + 1,
        notify_relations=notifies,
        wait_relations=waits,
        initial_counts=tuple(counts[coordinate] for coordinate in allocation.active_coordinates),
    )
