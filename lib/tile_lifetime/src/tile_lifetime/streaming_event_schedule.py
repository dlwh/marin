# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor attachment for a generic streaming Contract/Fold schedule.

This module is below tensor semantics and above CUDA synchronization.  It
derives the logical producer/consumer and buffer-reuse edges implemented by a
resident-left streaming contraction skeleton. Backend-specific barrier IDs and
pipeline classes remain physical allocation choices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil
from typing import Any

from tile_lifetime.event_buffering import (
    BoundedBufferPlan,
    EventRealizationAudit,
    EventRealizationKind,
    derive_bounded_buffer_plan,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    Coordinate,
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventTensorPlan,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventAllocation,
    InternalEventMechanismKind,
    bounded_buffer_internal_event_allocation,
    internal_event_allocation_record,
    internal_event_realization_records,
    phased_internal_event_allocation,
    verify_internal_event_allocations,
)


@dataclass(frozen=True)
class StreamingContractFoldDescriptor:
    """Target-independent facts for one resident-left streaming skeleton.

    This descriptor is produced after tensor semantics have been decomposed
    into two Contracts around an ordered Fold. It deliberately carries no
    tensor-operation or model identity. Those semantics determine the task
    graph before this schedule boundary; Event Tensor construction only needs
    task names, extents, payload widths, and the selected bounded pipeline
    shape.
    """

    first_contract_name: str
    fold_update_name: str
    second_contract_name: str
    finalize_name: str
    fold_extent: int
    resident_tile_size: int
    streamed_tile_size: int
    pipeline_depth: int
    resident_reduction_dimension: int
    streamed_reduction_dimension: int
    output_dimension: int
    element_bytes: int

    def __post_init__(self) -> None:
        names = (
            self.first_contract_name,
            self.fold_update_name,
            self.second_contract_name,
            self.finalize_name,
        )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("streaming task-family names must be non-empty and distinct")
        dimensions = (
            self.fold_extent,
            self.resident_tile_size,
            self.streamed_tile_size,
            self.pipeline_depth,
            self.resident_reduction_dimension,
            self.streamed_reduction_dimension,
            self.output_dimension,
            self.element_bytes,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("streaming extents, tile sizes, dimensions, and element width must be positive")


@dataclass(frozen=True)
class StreamingWorkerAssignment:
    """Finite worker decomposition selected for one physical tile skeleton."""

    transfer_warps: int
    transfer_warpgroup_threads: int
    matrix_warpgroups: int
    threads_per_warpgroup: int

    @property
    def matrix_warps(self) -> int:
        """Return the number of warps participating in matrix consumption."""
        return self.matrix_warpgroups * (self.threads_per_warpgroup // 32)

    @property
    def cta_threads(self) -> int:
        """Return the complete CTA size, including the transfer warpgroup."""
        return (self.matrix_warpgroups + 1) * self.threads_per_warpgroup

    @property
    def scheduler_arrival_threads(self) -> int:
        """Return pairwise matrix-warpgroup handshake participants."""
        return 2 * self.threads_per_warpgroup if self.matrix_warpgroups > 1 else 0


@dataclass(frozen=True)
class StreamingPipelineDataflow:
    """Kernel-local task graph and bounded buffers for two tile generations."""

    program: EventDataflowProgram
    resident_input_stage: TaskFamily
    first_streamed_input_stage: TaskFamily
    second_streamed_input_stage: TaskFamily
    first_contract: TaskFamily
    fold_update: TaskFamily
    second_contract: TaskFamily
    finalize: TaskFamily
    partition_count: int
    generation_count: int
    resident_input_buffer: BoundedBufferPlan
    first_streamed_input_buffer: BoundedBufferPlan
    second_streamed_input_buffer: BoundedBufferPlan


@dataclass(frozen=True)
class StreamingBoundedPipelineAllocation:
    """One same-CTA full/empty pipeline derived from exact internal events."""

    name: str
    full_event: EventTensorPlan
    empty_event: EventTensorPlan | None
    coordinates: tuple[Coordinate, ...]
    capacity: int
    allocation_owners: tuple[Coordinate, ...]
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    generation_count: int
    notify_count: int
    transaction_bytes: int
    extra_transaction_bytes: tuple[int, ...]
    scope: EventMemoryScope = EventMemoryScope.CTA
    physical_barriers_per_stage: int = 2

    def __post_init__(self) -> None:
        if self.transaction_bytes <= 0:
            raise ValueError("streaming pipeline base transaction bytes must be positive")
        if len(self.extra_transaction_bytes) != len(self.coordinates):
            raise ValueError("streaming pipeline extra transaction bytes must cover every producer coordinate")
        if any(extra < 0 for extra in self.extra_transaction_bytes):
            raise ValueError("streaming pipeline extra transaction bytes must be nonnegative")

    @property
    def transaction_bytes_by_coordinate(self) -> tuple[int, ...]:
        """Return the exact expected transaction bytes for every full Event."""
        return tuple(self.transaction_bytes + extra for extra in self.extra_transaction_bytes)

    @property
    def physical_barrier_slots(self) -> int:
        """Return barrier storage reserved by the selected physical template."""
        return self.capacity * self.physical_barriers_per_stage

    @property
    def required_barriers_per_stage(self) -> int:
        """Return phases required by the exact full/optional-empty Events."""
        return 1 + int(self.empty_event is not None)

    @property
    def required_barrier_slots(self) -> int:
        """Return the minimum storage extent required by the exact Events."""
        return self.capacity * self.required_barriers_per_stage


@dataclass(frozen=True)
class StreamingPhysicalEventSchedule:
    """Verified readiness configuration consumed by a physical GPU skeleton."""

    dataflow: StreamingPipelineDataflow
    audit: EventRealizationAudit
    workers: StreamingWorkerAssignment
    resident_input_pipeline: StreamingBoundedPipelineAllocation
    first_streamed_input_pipeline: StreamingBoundedPipelineAllocation
    second_streamed_input_pipeline: StreamingBoundedPipelineAllocation
    fingerprint: str

    @property
    def allocations(self) -> tuple[InternalEventAllocation, ...]:
        """Return the exact bounded allocations consumed by the backend."""
        return _streaming_internal_event_allocations(self)

    @property
    def resident_input_stages(self) -> int:
        """Return the resident-input allocation capacity."""
        return self.resident_input_pipeline.capacity

    @property
    def first_streamed_input_stages(self) -> int:
        """Return the first streamed-input allocation capacity."""
        return self.first_streamed_input_pipeline.capacity

    @property
    def second_streamed_input_stages(self) -> int:
        """Return the second streamed-input allocation capacity."""
        return self.second_streamed_input_pipeline.capacity

    @property
    def physical_barriers_per_stage(self) -> int:
        """Return the common barrier count reserved per physical pipeline stage."""
        counts = {
            self.resident_input_pipeline.physical_barriers_per_stage,
            self.first_streamed_input_pipeline.physical_barriers_per_stage,
            self.second_streamed_input_pipeline.physical_barriers_per_stage,
        }
        if len(counts) != 1:
            raise ValueError("streaming input pipelines disagree on barrier phases")
        return next(iter(counts))

    @property
    def resident_input_transaction_bytes(self) -> int:
        """Return bytes associated with one resident-input full notification."""
        return self.resident_input_pipeline.transaction_bytes

    @property
    def first_streamed_input_transaction_bytes(self) -> int:
        """Return bytes associated with one first streamed-input notification."""
        return self.first_streamed_input_pipeline.transaction_bytes

    @property
    def second_streamed_input_transaction_bytes(self) -> int:
        """Return bytes associated with one second streamed-input notification."""
        return self.second_streamed_input_pipeline.transaction_bytes

    @property
    def resident_input_physical_barrier_slots(self) -> int:
        """Return physical barrier storage reserved for the resident input."""
        return self.resident_input_pipeline.physical_barrier_slots

    @property
    def first_streamed_input_physical_barrier_slots(self) -> int:
        """Return physical barrier storage reserved for the first streamed input."""
        return self.first_streamed_input_pipeline.physical_barrier_slots

    @property
    def second_streamed_input_physical_barrier_slots(self) -> int:
        """Return physical barrier storage reserved for the second streamed input."""
        return self.second_streamed_input_pipeline.physical_barrier_slots


def streaming_physical_event_schedule_record(schedule: StreamingPhysicalEventSchedule) -> dict[str, Any]:
    """Serialize exact internal Events and their selected bounded allocations."""

    plans_by_name = {plan.name: plan for plan in schedule.dataflow.program.event_plans}
    allocations = schedule.allocations
    verify_internal_event_allocations(schedule.audit, allocations)

    return {
        "schema_version": 5,
        "fingerprint": schedule.fingerprint,
        "realizations": internal_event_realization_records(schedule.audit, plans_by_name, allocations),
        "allocations": tuple(internal_event_allocation_record(allocation) for allocation in allocations),
    }


def canonical_streaming_physical_event_schedule_record(schedule: StreamingPhysicalEventSchedule) -> dict[str, Any]:
    """Serialize exact internal tasks and Events with source names alpha-renamed by role."""
    record = streaming_physical_event_schedule_record(schedule)
    dataflow = schedule.dataflow
    family_names = {
        dataflow.resident_input_stage.name: "resident_input_stage",
        dataflow.first_streamed_input_stage.name: "first_streamed_input_stage",
        dataflow.second_streamed_input_stage.name: "second_streamed_input_stage",
        dataflow.first_contract.name: "first_contract",
        dataflow.fold_update.name: "streaming_fold_update",
        dataflow.second_contract.name: "second_contract",
        dataflow.finalize.name: "finalize",
    }
    plan_names: dict[str, str] = {}
    event_domain_names: dict[str, str] = {}
    for ordinal, plan in enumerate(dataflow.program.event_plans):
        relation = plan.required_dependence.relation
        canonical_name = f"event_{ordinal}_{family_names[relation.source.name]}_to_{family_names[relation.target.name]}"
        plan_names[plan.name] = canonical_name
        event_domain_names[f"{plan.name}.events"] = f"{canonical_name}.events"
    allocation_names = {
        f"{plan_name}.allocation": f"{canonical_name}.allocation" for plan_name, canonical_name in plan_names.items()
    }
    names = {**family_names, **plan_names, **event_domain_names, **allocation_names}

    def alpha_rename(value: Any) -> Any:
        if isinstance(value, dict):
            renamed = {key: alpha_rename(item) for key, item in value.items() if key != "fingerprint"}
            for relation_key in ("notify_relations", "wait_relations"):
                if relation_key in renamed:
                    renamed[relation_key] = tuple(
                        sorted(
                            renamed[relation_key],
                            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
                        )
                    )
            return renamed
        if isinstance(value, tuple):
            return tuple(alpha_rename(item) for item in value)
        if isinstance(value, list):
            return [alpha_rename(item) for item in value]
        if isinstance(value, str):
            return names.get(value, value)
        return value

    canonical = alpha_rename(record)
    canonical["fingerprint"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return canonical


def _streaming_internal_event_allocations(
    schedule: StreamingPhysicalEventSchedule,
) -> tuple[InternalEventAllocation, ...]:
    pipeline_allocations = tuple(
        bounded_buffer_internal_event_allocation(
            allocation_id=allocation.name,
            buffer=buffer,
            plans=(
                (allocation.full_event,)
                if allocation.empty_event is None
                else (allocation.full_event, allocation.empty_event)
            ),
            transaction_bytes_by_coordinate=allocation.transaction_bytes_by_coordinate,
        )
        for allocation, buffer in (
            (schedule.resident_input_pipeline, schedule.dataflow.resident_input_buffer),
            (schedule.first_streamed_input_pipeline, schedule.dataflow.first_streamed_input_buffer),
            (schedule.second_streamed_input_pipeline, schedule.dataflow.second_streamed_input_buffer),
        )
    )
    covered = {plan.name for allocation in pipeline_allocations for plan in allocation.covered_plans}
    plans_by_name = {plan.name: plan for plan in schedule.dataflow.program.event_plans}
    residual = tuple(
        phased_internal_event_allocation(
            allocation_id=f"{entry.plan_name}.allocation",
            plans=(plans_by_name[entry.plan_name],),
            capacity=1,
            allocation_owners=((),) * len(plans_by_name[entry.plan_name].domain.coordinates),
            slots=(0,) * len(plans_by_name[entry.plan_name].domain.coordinates),
            generations=tuple(range(len(plans_by_name[entry.plan_name].domain.coordinates))),
            mechanism_kind=InternalEventMechanismKind.CTA_MBARRIER,
        )
        for entry in schedule.audit.physical
        if entry.plan_name not in covered
    )
    return (*pipeline_allocations, *residual)


def verify_streaming_event_backend_parameters(
    schedule: StreamingPhysicalEventSchedule,
    *,
    resident_input_stages: int,
    first_streamed_input_stages: int,
    second_streamed_input_stages: int,
    barriers_per_stage: int,
    transfer_warps: int,
    matrix_warpgroups: int,
    scheduler_arrival_threads: int,
) -> None:
    """Reject a physical emitter whose synchronization constants drifted.

    This is the source-audit boundary: the backend supplies the values it will
    emit, and the schedule-level Event Tensor derivation remains the source of
    truth. Concrete barrier identifiers are intentionally outside this check.
    """
    actual = {
        "resident_input_stages": resident_input_stages,
        "first_streamed_input_stages": first_streamed_input_stages,
        "second_streamed_input_stages": second_streamed_input_stages,
        "barriers_per_stage": barriers_per_stage,
        "transfer_warps": transfer_warps,
        "matrix_warpgroups": matrix_warpgroups,
        "scheduler_arrival_threads": scheduler_arrival_threads,
    }
    expected = {
        "resident_input_stages": schedule.resident_input_stages,
        "first_streamed_input_stages": schedule.first_streamed_input_stages,
        "second_streamed_input_stages": schedule.second_streamed_input_stages,
        "barriers_per_stage": schedule.physical_barriers_per_stage,
        "transfer_warps": schedule.workers.transfer_warps,
        "matrix_warpgroups": schedule.workers.matrix_warpgroups,
        "scheduler_arrival_threads": schedule.workers.scheduler_arrival_threads,
    }
    mismatches = tuple(
        f"{name}: expected {expected[name]}, found {actual[name]}" for name in expected if actual[name] != expected[name]
    )
    if mismatches:
        raise ValueError("streaming Event Tensor/backend mismatch: " + "; ".join(mismatches))


def derive_streaming_physical_event_schedule(
    descriptor: StreamingContractFoldDescriptor,
) -> StreamingPhysicalEventSchedule:
    """Derive synchronization for a resident-left streaming skeleton.

    The task graph is one kernel-local resident invocation. Every streamed
    partition is represented, so slot reuse arises only when that invocation's
    streamed loop wraps the selected bounded pipeline.
    """
    resident_tile = descriptor.resident_tile_size
    streamed_tile = descriptor.streamed_tile_size
    partition_count = ceil(descriptor.fold_extent / streamed_tile)
    pipeline_depth = descriptor.pipeline_depth

    dataflow = _derive_pipeline_dataflow(
        descriptor,
        partition_count=partition_count,
        pipeline_depth=pipeline_depth,
        generation_count=1,
    )
    matrix_warpgroups = resident_tile // 64
    if matrix_warpgroups <= 0 or resident_tile % 64:
        raise ValueError("the physical worker decomposition requires resident tiles divisible by 64")
    workers = StreamingWorkerAssignment(
        transfer_warps=1,
        transfer_warpgroup_threads=128,
        matrix_warpgroups=matrix_warpgroups,
        threads_per_warpgroup=128,
    )
    audit = _realization_audit(dataflow, workers)
    resident_input_transaction_bytes = resident_tile * descriptor.resident_reduction_dimension * descriptor.element_bytes
    first_streamed_input_transaction_bytes = (
        streamed_tile * descriptor.streamed_reduction_dimension * descriptor.element_bytes
    )
    second_streamed_input_transaction_bytes = streamed_tile * descriptor.output_dimension * descriptor.element_bytes
    resident_input_pipeline = derive_streaming_bounded_pipeline_allocation(
        name="resident_input_pipeline",
        program=dataflow.program,
        buffer=dataflow.resident_input_buffer,
        consumer=dataflow.first_contract,
        transaction_bytes=resident_input_transaction_bytes,
    )
    first_streamed_input_pipeline = derive_streaming_bounded_pipeline_allocation(
        name="first_streamed_input_pipeline",
        program=dataflow.program,
        buffer=dataflow.first_streamed_input_buffer,
        consumer=dataflow.first_contract,
        transaction_bytes=first_streamed_input_transaction_bytes,
    )
    second_streamed_input_pipeline = derive_streaming_bounded_pipeline_allocation(
        name="second_streamed_input_pipeline",
        program=dataflow.program,
        buffer=dataflow.second_streamed_input_buffer,
        consumer=dataflow.second_contract,
        transaction_bytes=second_streamed_input_transaction_bytes,
    )
    payload = {
        "partition_count": partition_count,
        "pipeline_depth": pipeline_depth,
        "resident_tile": resident_tile,
        "streamed_tile": streamed_tile,
        "workers": {
            "transfer_warps": workers.transfer_warps,
            "matrix_warpgroups": workers.matrix_warpgroups,
            "cta_threads": workers.cta_threads,
        },
        "events": tuple((entry.plan_name, entry.kind.value, entry.mechanism) for entry in audit.entries),
        "buffers": {
            "resident_input": _pipeline_fingerprint_record(resident_input_pipeline),
            "first_streamed_input": _pipeline_fingerprint_record(first_streamed_input_pipeline),
            "second_streamed_input": _pipeline_fingerprint_record(second_streamed_input_pipeline),
        },
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StreamingPhysicalEventSchedule(
        dataflow=dataflow,
        audit=audit,
        workers=workers,
        resident_input_pipeline=resident_input_pipeline,
        first_streamed_input_pipeline=first_streamed_input_pipeline,
        second_streamed_input_pipeline=second_streamed_input_pipeline,
        fingerprint=fingerprint,
    )


def derive_streaming_bounded_pipeline_allocation(
    *,
    name: str,
    program: EventDataflowProgram,
    buffer: BoundedBufferPlan,
    consumer: TaskFamily,
    transaction_bytes: int,
    extra_transaction_bytes: dict[Coordinate, int] | None = None,
) -> StreamingBoundedPipelineAllocation:
    """Legalize one exact full/reuse Event pair to a same-CTA pipeline."""
    if not name:
        raise ValueError("streaming pipeline allocation name must be non-empty")
    if transaction_bytes <= 0:
        raise ValueError("streaming pipeline transaction bytes must be positive")
    if extra_transaction_bytes is None:
        extra_transaction_bytes = {coordinate: 0 for coordinate in buffer.producer.coordinates}
    if set(extra_transaction_bytes) != set(buffer.producer.coordinates):
        raise ValueError("streaming pipeline extra transaction bytes must cover the producer domain")
    if any(extra < 0 for extra in extra_transaction_bytes.values()):
        raise ValueError("streaming pipeline extra transaction bytes must be nonnegative")
    matching_uses = tuple(use for use in buffer.uses if use.target == consumer)
    if len(matching_uses) != 1:
        raise ValueError("streaming pipeline allocation requires one exact readiness relation to its first consumer")
    if len(buffer.reuse_dependences) > 1:
        raise ValueError("streaming pipeline allocation supports at most one exact slot-reuse dependence")

    full_event = _unique_event_plan(program, matching_uses[0], "full readiness")
    empty_event = (
        None
        if not buffer.reuse_dependences
        else _unique_event_plan(program, buffer.reuse_dependences[0].relation, "empty reuse")
    )
    full_relation = full_event.required_dependence.relation
    if full_relation.source != buffer.producer or full_relation.target != consumer:
        raise ValueError("streaming pipeline full Event does not connect the buffer producer to its consumer")
    if full_event.memory_scope is not EventMemoryScope.CTA:
        raise ValueError("streaming pipeline Events must have CTA scope")
    if full_event.generation_policy is not EventGenerationPolicy.PER_INVOCATION:
        raise ValueError("streaming pipeline full Event must use per-invocation generations")
    for event in (full_event,) if empty_event is None else (full_event, empty_event):
        visibility = event.visibility
        if not visibility.release_on_notify or not visibility.acquire_before_consumer:
            raise ValueError("streaming pipeline Events require release/acquire visibility")

    full_counts = {count.value for count in full_event.initial_count.counts}
    if len(full_counts) != 1 or 0 in full_counts:
        raise ValueError("streaming pipeline full Event must have one uniform positive notify count")
    notify_count = next(iter(full_counts))
    if empty_event is not None:
        empty_relation = empty_event.required_dependence.relation
        if empty_relation.target != buffer.producer:
            raise ValueError("streaming pipeline empty Event does not release the next buffer producer")
        if empty_event.memory_scope is not EventMemoryScope.CTA:
            raise ValueError("streaming pipeline Events must have CTA scope")
        if empty_event.generation_policy is not EventGenerationPolicy.PHASED:
            raise ValueError("streaming pipeline empty Event must use phased generations")
        empty_counts = {count.value for count in empty_event.initial_count.counts}
        if not empty_counts <= {0, notify_count} or notify_count not in empty_counts:
            raise ValueError("streaming pipeline empty Event counts do not match full readiness")

    coordinates = buffer.producer.coordinates
    if (
        len(buffer.allocation_owners) != len(coordinates)
        or len(buffer.slots) != len(coordinates)
        or len(buffer.generations) != len(coordinates)
    ):
        raise ValueError("streaming pipeline owner/slot/generation mapping does not cover its producer domain")
    initial_slots = {slot for slot, generation in zip(buffer.slots, buffer.generations, strict=True) if generation == 0}
    if initial_slots != set(range(min(buffer.capacity, len(coordinates)))):
        raise ValueError("streaming pipeline initial generation does not initialize every active slot")
    generation_count = max(buffer.generations, default=0) + 1
    return StreamingBoundedPipelineAllocation(
        name=name,
        full_event=full_event,
        empty_event=empty_event,
        coordinates=coordinates,
        capacity=buffer.capacity,
        allocation_owners=buffer.allocation_owners,
        slots=buffer.slots,
        generations=buffer.generations,
        generation_count=generation_count,
        notify_count=notify_count,
        transaction_bytes=transaction_bytes,
        extra_transaction_bytes=tuple(extra_transaction_bytes[coordinate] for coordinate in coordinates),
    )


def _unique_event_plan(program: EventDataflowProgram, relation: TaskRelation, label: str) -> EventTensorPlan:
    matches = tuple(plan for plan in program.event_plans if plan.required_dependence.relation == relation)
    if len(matches) != 1:
        raise ValueError(f"streaming pipeline requires one exact {label} Event, found {len(matches)}")
    return matches[0]


def _pipeline_fingerprint_record(allocation: StreamingBoundedPipelineAllocation) -> tuple[object, ...]:
    return (
        allocation.full_event.name,
        None if allocation.empty_event is None else allocation.empty_event.name,
        allocation.capacity,
        allocation.allocation_owners,
        allocation.slots,
        allocation.generations,
        allocation.generation_count,
        allocation.notify_count,
        allocation.transaction_bytes,
        allocation.extra_transaction_bytes,
        allocation.scope.value,
        allocation.physical_barriers_per_stage,
    )


def _derive_pipeline_dataflow(
    descriptor: StreamingContractFoldDescriptor,
    *,
    partition_count: int,
    pipeline_depth: int,
    generation_count: int,
) -> StreamingPipelineDataflow:
    generation_axis = TaskAxis("generation", generation_count)
    partition_axis = TaskAxis("fold_partition", partition_count)
    generation_domain = (generation_axis,)
    partition_domain = (generation_axis, partition_axis)
    resident_input_stage = TaskFamily("resident_input_stage", generation_domain, placement="transfer_workers")
    first_streamed_input_stage = TaskFamily("first_streamed_input_stage", partition_domain, placement="transfer_workers")
    second_streamed_input_stage = TaskFamily(
        "second_streamed_input_stage", partition_domain, placement="transfer_workers"
    )
    first_contract = TaskFamily(descriptor.first_contract_name, partition_domain, placement="matrix_workers")
    fold_update = TaskFamily(descriptor.fold_update_name, partition_domain, placement="matrix_workers")
    second_contract = TaskFamily(descriptor.second_contract_name, partition_domain, placement="matrix_workers")
    finalize = TaskFamily(descriptor.finalize_name, generation_domain, placement="matrix_workers")
    visibility = MemoryVisibility(EventMemoryScope.CTA)

    pointwise = tuple(
        ((generation, partition), (generation, partition))
        for generation in range(generation_count)
        for partition in range(partition_count)
    )
    resident_input_to_first_contract = TaskDependence(
        TaskRelation.from_pairs(
            resident_input_stage,
            first_contract,
            tuple(
                ((generation,), (generation, partition))
                for generation in range(generation_count)
                for partition in range(partition_count)
            ),
        ),
        visibility,
    )
    first_streamed_input_to_first_contract = TaskDependence(
        TaskRelation.from_pairs(first_streamed_input_stage, first_contract, pointwise), visibility
    )
    first_contract_to_fold = TaskDependence(TaskRelation.from_pairs(first_contract, fold_update, pointwise), visibility)
    fold_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(fold_update, second_contract, pointwise), visibility
    )
    second_streamed_input_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(second_streamed_input_stage, second_contract, pointwise), visibility
    )
    second_contract_to_next_first_contract = TaskDependence(
        TaskRelation.from_pairs(
            second_contract,
            first_contract,
            tuple(
                ((generation, partition), (generation, partition + 1))
                for generation in range(generation_count)
                for partition in range(partition_count - 1)
            ),
        ),
        visibility,
    )
    second_contract_to_finalize = TaskDependence(
        TaskRelation.from_pairs(
            second_contract,
            finalize,
            tuple(((generation, partition_count - 1), (generation,)) for generation in range(generation_count)),
        ),
        visibility,
    )
    initial_dependences = (
        resident_input_to_first_contract,
        first_streamed_input_to_first_contract,
        first_contract_to_fold,
        fold_to_second_contract,
        second_streamed_input_to_second_contract,
        second_contract_to_next_first_contract,
        second_contract_to_finalize,
    )
    families = (
        resident_input_stage,
        first_streamed_input_stage,
        second_streamed_input_stage,
        first_contract,
        fold_update,
        second_contract,
        finalize,
    )
    initial_program = _event_program(families, initial_dependences)

    resident_input_buffer = derive_bounded_buffer_plan(
        name="resident_input_pipeline",
        program=initial_program,
        producer=resident_input_stage,
        uses=(resident_input_to_first_contract.relation,),
        capacity=1,
        allocation_owner_for={coordinate: () for coordinate in resident_input_stage.coordinates},
        slot_for={(generation,): 0 for generation in range(generation_count)},
        generation_for={(generation,): generation for generation in range(generation_count)},
        visibility=visibility,
    )

    def linear_index(coordinate: tuple[int, ...]) -> int:
        generation, partition = coordinate
        return generation * partition_count + partition

    pipeline_coordinates = first_streamed_input_stage.coordinates
    slots = {coordinate: linear_index(coordinate) % pipeline_depth for coordinate in pipeline_coordinates}
    generations = {coordinate: linear_index(coordinate) // pipeline_depth for coordinate in pipeline_coordinates}
    first_streamed_input_buffer = derive_bounded_buffer_plan(
        name="first_streamed_input_pipeline",
        program=initial_program,
        producer=first_streamed_input_stage,
        uses=(first_streamed_input_to_first_contract.relation,),
        capacity=pipeline_depth,
        allocation_owner_for={coordinate: () for coordinate in pipeline_coordinates},
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    second_streamed_input_buffer = derive_bounded_buffer_plan(
        name="second_streamed_input_pipeline",
        program=initial_program,
        producer=second_streamed_input_stage,
        uses=(second_streamed_input_to_second_contract.relation,),
        capacity=pipeline_depth,
        allocation_owner_for={coordinate: () for coordinate in pipeline_coordinates},
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    dependences = (
        *initial_dependences,
        *resident_input_buffer.reuse_dependences,
        *first_streamed_input_buffer.reuse_dependences,
        *second_streamed_input_buffer.reuse_dependences,
    )
    complete_program = _event_program(families, dependences)
    verify_event_dataflow_program(complete_program)
    return StreamingPipelineDataflow(
        program=complete_program,
        resident_input_stage=resident_input_stage,
        first_streamed_input_stage=first_streamed_input_stage,
        second_streamed_input_stage=second_streamed_input_stage,
        first_contract=first_contract,
        fold_update=fold_update,
        second_contract=second_contract,
        finalize=finalize,
        partition_count=partition_count,
        generation_count=generation_count,
        resident_input_buffer=resident_input_buffer,
        first_streamed_input_buffer=first_streamed_input_buffer,
        second_streamed_input_buffer=second_streamed_input_buffer,
    )


def _event_program(
    families: tuple[TaskFamily, ...],
    dependences: tuple[TaskDependence, ...],
) -> EventDataflowProgram:
    plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event_{index}_{dependence.relation.source.name}_to_{dependence.relation.target.name}",
            generation_policy=(
                EventGenerationPolicy.PHASED
                if dependence.relation.target.name.endswith("_stage")
                else EventGenerationPolicy.PER_INVOCATION
            ),
        )
        for index, dependence in enumerate(dependences)
    )
    return EventDataflowProgram(families, dependences, plans)


def _realization_audit(
    dataflow: StreamingPipelineDataflow,
    workers: StreamingWorkerAssignment,
) -> EventRealizationAudit:
    reuse_relations = {
        dependence.relation
        for buffer in (
            dataflow.resident_input_buffer,
            dataflow.first_streamed_input_buffer,
            dataflow.second_streamed_input_buffer,
        )
        for dependence in buffer.reuse_dependences
    }
    realizations = []
    for plan in dataflow.program.event_plans:
        relation = plan.required_dependence.relation
        endpoints = (relation.source, relation.target)
        if relation in reuse_relations:
            buffer_name = {
                dataflow.resident_input_stage.name: "resident input",
                dataflow.first_streamed_input_stage.name: "first streamed input",
                dataflow.second_streamed_input_stage.name: "second streamed input",
            }[relation.target.name]
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism=f"{buffer_name} pipeline empty barrier with phased slot generation",
                    reason="the derived last consumer must release a bounded slot before reuse",
                )
            )
        elif relation.source in (
            dataflow.resident_input_stage,
            dataflow.first_streamed_input_stage,
            dataflow.second_streamed_input_stage,
        ):
            payload = {
                dataflow.resident_input_stage.name: "resident input",
                dataflow.first_streamed_input_stage.name: "first streamed input",
                dataflow.second_streamed_input_stage.name: "second streamed input",
            }[relation.source.name]
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism=f"{payload} pipeline full barrier",
                    reason="matrix consumers require acquire visibility after asynchronous tile movement",
                )
            )
        elif endpoints == (dataflow.second_contract, dataflow.first_contract) and workers.matrix_warpgroups > 1:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="pairwise matrix-warpgroup scheduler barrier",
                    reason="the selected overlap schedule hands ordered Fold state between matrix warpgroups",
                )
            )
        else:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="matrix-owner program order",
                    reason="the selected skeleton executes this dependence in one ordered tile body",
                )
            )
    return verify_event_realizations(dataflow.program, tuple(realizations))
