# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor planning for streamed inputs to a Contract-region pullback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
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
    EventStorageBinding,
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
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractPullback,
    ReverseOwnershipPlan,
    derive_reverse_ownership_plan,
)
from tile_lifetime.ordered_accumulation_event_schedule import (
    OrderedAccumulationEventSchedule,
    derive_ordered_accumulation_event_schedule,
    owned_phased_internal_event_allocation,
)
from tile_lifetime.sm90_backward_accumulation import Sm90BackwardAccumulationPlan
from tile_lifetime.streaming_event_schedule import (
    StreamingBoundedPipelineAllocation,
    derive_streaming_bounded_pipeline_allocation,
)


class StreamingIntermediateStorage(StrEnum):
    """Selected storage for a pullback value consumed by a terminal Contract."""

    REGISTERS = "registers"
    SHARED = "shared"


@dataclass(frozen=True)
class NormalizedExpPullbackTaskDataflow:
    """Pre-physical tiled tasks and exact Events for a generic pullback.

    Every Event uses finite, non-reused generation-zero storage at this
    boundary. A physical schedule may subsequently fuse tasks, erase Events
    with a structural ordering proof, or select bounded reusable storage.
    """

    program: EventDataflowProgram
    storage_bindings: tuple[EventStorageBinding, ...]
    score_tile_coordinates: tuple[Coordinate, ...]
    owner_preparation: TaskFamily
    score_contract: TaskFamily
    normalized_fold: TaskFamily
    normalized_weight: TaskFamily
    output_cotangent_contract: TaskFamily
    normalized_exp_pullback: TaskFamily
    score_vjp: TaskFamily
    terminal_gradient_contributions: tuple[TaskFamily, ...]
    terminal_gradient_finalizers: tuple[TaskFamily, ...]


def derive_normalized_exp_pullback_task_dataflow(
    pullback: NormalizedExpContractPullback,
    ownership: ReverseOwnershipPlan,
    *,
    owner_tile_size: int,
    streamed_tile_size: int,
    visibility_scope: EventMemoryScope = EventMemoryScope.CTA,
) -> NormalizedExpPullbackTaskDataflow:
    """Tile a normalized-exp pullback before choosing a GPU implementation."""
    if owner_tile_size <= 0 or streamed_tile_size <= 0:
        raise ValueError("normalized-exp pullback tile sizes must be positive")
    region = pullback.region
    expected_owner_axes = region.owner_axes
    expected_ownership = derive_reverse_ownership_plan(pullback, traversal=ownership.traversal)
    if ownership != expected_ownership:
        raise ValueError("pullback task ownership must derive from the normalized-exp pullback")
    if len(region.dimensions.normalized_axes) != 1:
        raise ValueError("pullback task decomposition currently requires one normalized axis")

    producer_left, producer_right = region.producer.inputs
    projected_right_axes = set(producer_right.axes)
    for relation in region.producer.index_maps_for_input(1):
        projected_right_axes.remove(relation.operand_axis)
        projected_right_axes.add(relation.domain_axis)
    resident_axes = tuple(
        axis for axis in expected_owner_axes if axis in producer_left.axes and axis not in projected_right_axes
    )
    if len(resident_axes) != 1:
        raise ValueError("pullback task decomposition requires one resident owner axis")
    resident_axis = resident_axes[0]
    streamed_axis = region.dimensions.normalized_axes[0]

    def task_axis(axis, *, tile_size: int | None = None) -> TaskAxis:
        extent = axis.extent if tile_size is None else ceil(axis.extent / tile_size)
        suffix = "_tile" if tile_size is not None else ""
        return TaskAxis(f"{axis.label or f'axis_{axis.id}'}{suffix}", extent)

    owner_task_axes = tuple(
        task_axis(axis, tile_size=owner_tile_size if axis == resident_axis else None) for axis in expected_owner_axes
    )
    streamed_task_axes = (task_axis(streamed_axis, tile_size=streamed_tile_size),)
    logical_score_task_axes = (*owner_task_axes, *streamed_task_axes)
    owner_rank = len(owner_task_axes)

    logical_score_coordinates = tuple(
        coordinate
        for coordinate in TaskFamily("logical_score_tiles", logical_score_task_axes).coordinates
        if ownership.traversal is not ContractDomainTraversal.LOWER_TRIANGULAR
        or coordinate[-1] * streamed_tile_size
        < min((coordinate[expected_owner_axes.index(resident_axis)] + 1) * owner_tile_size, resident_axis.extent)
    )
    score_task_axes = (TaskAxis("score_tile", len(logical_score_coordinates)),)

    owner_preparation = TaskFamily("normalized_exp_pullback.owner_preparation", owner_task_axes)
    score_contract = TaskFamily(f"recompute.{region.producer.name}", score_task_axes)
    normalized_fold = TaskFamily("normalized_exp_pullback.normalized_fold", owner_task_axes)
    normalized_weight = TaskFamily(pullback.normalized_weight_map.name, score_task_axes)
    output_cotangent_contract = TaskFamily(pullback.consumer_weight_cotangent.name, score_task_axes)
    normalized_exp_pullback = TaskFamily(pullback.normalized_exp_pullback.name, score_task_axes)
    score_vjp = TaskFamily(pullback.score_map_vjp.name, score_task_axes)
    terminal_gradient_contributions = tuple(
        TaskFamily(gradient.contract.name, score_task_axes) for gradient in pullback.terminal_gradients
    )

    mapped_operand_axes = {relation.operand_axis: relation for relation in region.producer.index_maps_for_input(1)}
    axis_position = {axis: index for index, axis in enumerate((*expected_owner_axes, streamed_axis))}

    def destination_family(name: str, axes: tuple, feature_axes: tuple) -> TaskFamily:
        return TaskFamily(
            name,
            tuple(
                task_axis(
                    axis,
                    tile_size=(
                        streamed_tile_size
                        if axis == streamed_axis
                        else owner_tile_size if axis == resident_axis else None
                    ),
                )
                for axis in axes
                if axis not in set(feature_axes)
            ),
        )

    left_finalize = TaskFamily(f"finalize.{pullback.terminal_gradients[0].destination.name}", owner_task_axes)
    producer_right_finalize = destination_family(
        f"finalize.{pullback.terminal_gradients[1].destination.name}",
        producer_right.axes,
        region.dimensions.producer_features,
    )
    consumer_operand = region.consumer.inputs[1]
    consumer_right_finalize = destination_family(
        f"finalize.{pullback.terminal_gradients[2].destination.name}",
        consumer_operand.axes,
        region.dimensions.consumer_features,
    )
    terminal_gradient_finalizers = (left_finalize, producer_right_finalize, consumer_right_finalize)

    score_coordinates = score_contract.coordinates
    pointwise = tuple((coordinate, coordinate) for coordinate in score_coordinates)
    owner_to_score = tuple(
        (logical[:owner_rank], score)
        for score, logical in zip(score_coordinates, logical_score_coordinates, strict=True)
    )
    score_to_owner = tuple(
        (score, logical[:owner_rank])
        for score, logical in zip(score_coordinates, logical_score_coordinates, strict=True)
    )

    def destination_coordinate(coordinate: Coordinate, operand_axes: tuple, feature_axes: tuple) -> Coordinate:
        logical_coordinate = {axis: coordinate[position] for axis, position in axis_position.items()}
        result = []
        for axis in operand_axes:
            if axis in feature_axes:
                continue
            relation = mapped_operand_axes.get(axis)
            if relation is not None:
                domain_coordinate = logical_coordinate[relation.domain_axis]
                if relation.domain_axis == resident_axis or relation.domain_axis == streamed_axis:
                    raise ValueError("a tiled pullback axis cannot feed a nontrivial operand index relation")
                mapped = domain_coordinate // relation.divisor + relation.offset
                result.append(mapped if relation.modulus is None else mapped % relation.modulus)
                continue
            result.append(logical_coordinate[axis])
        return tuple(result)

    right_contributors = tuple(
        (score, destination_coordinate(logical, producer_right.axes, region.dimensions.producer_features))
        for score, logical in zip(score_coordinates, logical_score_coordinates, strict=True)
    )
    consumer_right_contributors = tuple(
        (
            score,
            destination_coordinate(logical, consumer_operand.axes, region.dimensions.consumer_features),
        )
        for score, logical in zip(score_coordinates, logical_score_coordinates, strict=True)
    )

    visibility = MemoryVisibility(visibility_scope)

    def dependence(source: TaskFamily, target: TaskFamily, pairs) -> TaskDependence:
        return TaskDependence(TaskRelation.from_pairs(source, target, pairs), visibility)

    left_contribution, producer_right_contribution, consumer_right_contribution = terminal_gradient_contributions
    score_vjp_primal_dependences = (
        (dependence(score_contract, score_vjp, pointwise),)
        if region.producer.output in pullback.score_map_vjp.inputs
        else ()
    )
    dependences = (
        dependence(score_contract, normalized_fold, score_to_owner),
        dependence(score_contract, normalized_weight, pointwise),
        dependence(normalized_fold, normalized_weight, owner_to_score),
        dependence(owner_preparation, output_cotangent_contract, owner_to_score),
        dependence(normalized_weight, normalized_exp_pullback, pointwise),
        dependence(output_cotangent_contract, normalized_exp_pullback, pointwise),
        dependence(owner_preparation, normalized_exp_pullback, owner_to_score),
        dependence(normalized_exp_pullback, score_vjp, pointwise),
        *score_vjp_primal_dependences,
        dependence(score_vjp, left_contribution, pointwise),
        dependence(score_vjp, producer_right_contribution, pointwise),
        dependence(normalized_weight, consumer_right_contribution, pointwise),
        dependence(owner_preparation, consumer_right_contribution, owner_to_score),
        dependence(left_contribution, left_finalize, score_to_owner),
        dependence(producer_right_contribution, producer_right_finalize, right_contributors),
        dependence(consumer_right_contribution, consumer_right_finalize, consumer_right_contributors),
    )
    families = (
        owner_preparation,
        score_contract,
        normalized_fold,
        normalized_weight,
        output_cotangent_contract,
        normalized_exp_pullback,
        score_vjp,
        *terminal_gradient_contributions,
        *terminal_gradient_finalizers,
    )
    event_plans = tuple(
        derive_event_tensor_plan(
            dependency,
            name=f"event_{index}_{dependency.relation.source.name}_to_{dependency.relation.target.name}",
        )
        for index, dependency in enumerate(dependences)
    )
    program = EventDataflowProgram(families, dependences, event_plans)
    verify_event_dataflow_program(program)
    storage_bindings = tuple(
        EventStorageBinding(plan.domain, tuple(range(len(plan.domain.coordinates))), (0,) * len(plan.domain.coordinates))
        for plan in event_plans
    )
    return NormalizedExpPullbackTaskDataflow(
        program=program,
        storage_bindings=storage_bindings,
        score_tile_coordinates=logical_score_coordinates,
        owner_preparation=owner_preparation,
        score_contract=score_contract,
        normalized_fold=normalized_fold,
        normalized_weight=normalized_weight,
        output_cotangent_contract=output_cotangent_contract,
        normalized_exp_pullback=normalized_exp_pullback,
        score_vjp=score_vjp,
        terminal_gradient_contributions=terminal_gradient_contributions,
        terminal_gradient_finalizers=terminal_gradient_finalizers,
    )


@dataclass(frozen=True)
class StreamingPullbackInputDescriptor:
    """Algebraic task names and physical extents for two streamed pullback inputs."""

    producer_lhs_stage_name: str
    output_cotangent_stage_name: str
    score_contract_name: str
    output_cotangent_contract_name: str
    normalized_weight_name: str
    pullback_map_name: str
    score_vjp_name: str
    producer_lhs_gradient_name: str
    lhs_gradient_handoff_name: str
    producer_rhs_gradient_name: str
    consumer_rhs_gradient_name: str
    producer_rhs_epilogue_stage_name: str
    producer_rhs_epilogue_transfer_name: str
    consumer_rhs_epilogue_stage_name: str
    consumer_rhs_epilogue_transfer_name: str
    owner_extent: int
    owner_tile_size: int
    streamed_extent: int
    streamed_tile_size: int
    score_tile_coordinates: tuple[Coordinate, ...]
    relation_group_extent: int
    deterministic_accumulation: bool
    producer_lhs_pipeline_depth: int
    output_cotangent_pipeline_depth: int
    score_workspace_depth: int
    normalized_weight_storage: StreamingIntermediateStorage
    matrix_warpgroups: int
    lhs_gradient_warpgroups: int
    producer_lhs_transaction_bytes: int
    output_cotangent_transaction_bytes: int
    producer_rhs_resident_transaction_bytes: int
    consumer_rhs_resident_transaction_bytes: int
    accumulation_plan: Sm90BackwardAccumulationPlan | None = None

    def __post_init__(self) -> None:
        names = (
            self.producer_lhs_stage_name,
            self.output_cotangent_stage_name,
            self.score_contract_name,
            self.output_cotangent_contract_name,
            self.normalized_weight_name,
            self.pullback_map_name,
            self.score_vjp_name,
            self.producer_lhs_gradient_name,
            self.lhs_gradient_handoff_name,
            self.producer_rhs_gradient_name,
            self.consumer_rhs_gradient_name,
            self.producer_rhs_epilogue_stage_name,
            self.producer_rhs_epilogue_transfer_name,
            self.consumer_rhs_epilogue_stage_name,
            self.consumer_rhs_epilogue_transfer_name,
        )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("streaming pullback task-family names must be non-empty and distinct")
        extents = (
            self.owner_extent,
            self.owner_tile_size,
            self.streamed_extent,
            self.streamed_tile_size,
            self.relation_group_extent,
            self.producer_lhs_pipeline_depth,
            self.output_cotangent_pipeline_depth,
            self.score_workspace_depth,
            self.matrix_warpgroups,
            self.lhs_gradient_warpgroups,
            self.producer_lhs_transaction_bytes,
            self.output_cotangent_transaction_bytes,
            self.producer_rhs_resident_transaction_bytes,
            self.consumer_rhs_resident_transaction_bytes,
        )
        if any(value <= 0 for value in extents):
            raise ValueError("streaming pullback extents, depths, and transaction bytes must be positive")
        if not isinstance(self.normalized_weight_storage, StreamingIntermediateStorage):
            raise ValueError("streaming pullback normalized-weight storage must be explicit")
        if self.lhs_gradient_warpgroups > self.matrix_warpgroups:
            raise ValueError("lhs-gradient workers must be a subset of the matrix warpgroups")
        if self.accumulation_plan is not None:
            plan = self.accumulation_plan
            if not self.deterministic_accumulation or self.relation_group_extent != 1:
                raise ValueError("ordered SM90 plan requires one deterministic dQ queue")
            if (plan.owner_extent, plan.streamed_extent, plan.owner_tile, plan.streamed_tile) != (
                self.owner_extent,
                self.streamed_extent,
                self.owner_tile_size,
                self.streamed_tile_size,
            ) or set(plan.contributor_ranks) != set(self.score_tile_coordinates):
                raise ValueError("ordered SM90 plan must match the exact Event contributor domain")
        owner_partitions = ceil(self.owner_extent / self.owner_tile_size)
        streamed_partitions = ceil(self.streamed_extent / self.streamed_tile_size)
        score_domain = {
            (owner_partition, streamed_partition)
            for owner_partition in range(owner_partitions)
            for streamed_partition in range(streamed_partitions)
        }
        if len(set(self.score_tile_coordinates)) != len(self.score_tile_coordinates):
            raise ValueError("streaming pullback score-tile coordinates must be unique")
        if not set(self.score_tile_coordinates) <= score_domain:
            raise ValueError("streaming pullback score-tile coordinates must lie in the tiled Contract domain")


@dataclass(frozen=True)
class StreamingPullbackInputDataflow:
    """Exact task graph and bounded input buffers for one pullback mainloop."""

    program: EventDataflowProgram
    producer_lhs_stage: TaskFamily
    output_cotangent_stage: TaskFamily
    score_contract: TaskFamily
    output_cotangent_contract: TaskFamily
    normalized_weight: TaskFamily
    pullback_map: TaskFamily
    score_vjp: TaskFamily
    producer_lhs_gradient: TaskFamily
    lhs_gradient_handoff: TaskFamily
    producer_rhs_gradient: TaskFamily
    consumer_rhs_gradient: TaskFamily
    producer_rhs_epilogue_stage: TaskFamily
    producer_rhs_epilogue_transfer: TaskFamily
    consumer_rhs_epilogue_stage: TaskFamily
    consumer_rhs_epilogue_transfer: TaskFamily
    partition_count: int
    producer_lhs_buffer: BoundedBufferPlan
    output_cotangent_buffer: BoundedBufferPlan
    normalized_weight_buffer: BoundedBufferPlan | None
    score_cotangent_buffer: BoundedBufferPlan
    lhs_gradient_handoff_buffer: BoundedBufferPlan


@dataclass(frozen=True)
class StreamingPullbackInputEventSchedule:
    """Verified TMA full/empty pipelines for streamed pullback operands."""

    dataflow: StreamingPullbackInputDataflow
    audit: EventRealizationAudit
    producer_lhs_pipeline: StreamingBoundedPipelineAllocation
    output_cotangent_pipeline: StreamingBoundedPipelineAllocation
    normalized_weight_storage: StreamingIntermediateStorage
    device_accumulation_queues: tuple[OrderedAccumulationEventSchedule, ...]
    fingerprint: str

    @property
    def allocations(self) -> tuple[InternalEventAllocation, ...]:
        """Return the exact bounded allocations consumed by the backend."""
        return _pullback_internal_event_allocations(self)


def derive_streaming_pullback_input_event_schedule(
    descriptor: StreamingPullbackInputDescriptor,
) -> StreamingPullbackInputEventSchedule:
    """Derive multi-consumer input readiness and reuse for a pullback mainloop."""
    partition_count = ceil(descriptor.owner_extent / descriptor.owner_tile_size)
    dataflow = _derive_pullback_input_dataflow(descriptor, partition_count)
    first_coordinate = dataflow.producer_lhs_stage.coordinates[0]
    producer_lhs_pipeline = derive_streaming_bounded_pipeline_allocation(
        name="producer_lhs_state_pipeline",
        program=dataflow.program,
        buffer=dataflow.producer_lhs_buffer,
        consumer=dataflow.score_contract,
        transaction_bytes=descriptor.producer_lhs_transaction_bytes,
        extra_transaction_bytes={
            coordinate: descriptor.producer_rhs_resident_transaction_bytes if coordinate == first_coordinate else 0
            for coordinate in dataflow.producer_lhs_stage.coordinates
        },
    )
    output_cotangent_pipeline = derive_streaming_bounded_pipeline_allocation(
        name="output_cotangent_state_pipeline",
        program=dataflow.program,
        buffer=dataflow.output_cotangent_buffer,
        consumer=dataflow.output_cotangent_contract,
        transaction_bytes=descriptor.output_cotangent_transaction_bytes,
        extra_transaction_bytes={
            coordinate: descriptor.consumer_rhs_resident_transaction_bytes if coordinate == first_coordinate else 0
            for coordinate in dataflow.output_cotangent_stage.coordinates
        },
    )
    audit = _pullback_input_realization_audit(dataflow)
    device_accumulation_queues = _derive_device_accumulation_queues(descriptor, partition_count)
    payload = streaming_pullback_input_event_schedule_record(
        StreamingPullbackInputEventSchedule(
            dataflow,
            audit,
            producer_lhs_pipeline,
            output_cotangent_pipeline,
            descriptor.normalized_weight_storage,
            device_accumulation_queues,
            fingerprint="",
        )
    )
    payload.pop("fingerprint")
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StreamingPullbackInputEventSchedule(
        dataflow,
        audit,
        producer_lhs_pipeline,
        output_cotangent_pipeline,
        descriptor.normalized_weight_storage,
        device_accumulation_queues,
        fingerprint,
    )


def streaming_pullback_input_event_schedule_record(
    schedule: StreamingPullbackInputEventSchedule,
) -> dict[str, Any]:
    """Serialize exact pullback input Events and their bounded mechanisms."""

    plans_by_name = {plan.name: plan for plan in schedule.dataflow.program.event_plans}
    queue_plans = {queue.allocation.plan.name: queue.allocation.plan for queue in schedule.device_accumulation_queues}
    all_plans = plans_by_name | queue_plans
    combined_audit = EventRealizationAudit(
        (
            *schedule.audit.entries,
            *(entry for queue in schedule.device_accumulation_queues for entry in queue.audit.entries),
        )
    )
    allocations = schedule.allocations
    verify_internal_event_allocations(combined_audit, allocations)
    return {
        "schema_version": 6,
        "fingerprint": schedule.fingerprint,
        "partition_count": schedule.dataflow.partition_count,
        "realizations": internal_event_realization_records(combined_audit, all_plans, allocations),
        "allocations": tuple(internal_event_allocation_record(allocation) for allocation in allocations),
    }


def _pullback_internal_event_allocations(
    schedule: StreamingPullbackInputEventSchedule,
) -> tuple[InternalEventAllocation, ...]:
    dataflow = schedule.dataflow

    def pipeline_allocation(
        allocation: StreamingBoundedPipelineAllocation,
        buffer: BoundedBufferPlan,
    ) -> InternalEventAllocation:
        plans = (
            (allocation.full_event,)
            if allocation.empty_event is None
            else (allocation.full_event, allocation.empty_event)
        )
        return bounded_buffer_internal_event_allocation(
            allocation_id=allocation.name,
            buffer=buffer,
            plans=plans,
            transaction_bytes_by_coordinate=allocation.transaction_bytes_by_coordinate,
        )

    allocations: list[InternalEventAllocation] = [
        pipeline_allocation(schedule.producer_lhs_pipeline, dataflow.producer_lhs_buffer),
        pipeline_allocation(schedule.output_cotangent_pipeline, dataflow.output_cotangent_buffer),
    ]
    physical_names = {entry.plan_name for entry in schedule.audit.physical}
    covered = {plan.name for allocation in allocations for plan in allocation.covered_plans}
    buffer_candidates = (
        *((dataflow.normalized_weight_buffer,) if dataflow.normalized_weight_buffer is not None else ()),
        dataflow.score_cotangent_buffer,
        dataflow.lhs_gradient_handoff_buffer,
    )
    for buffer in buffer_candidates:
        buffer_relations = {*buffer.uses, *(dependence.relation for dependence in buffer.reuse_dependences)}
        plans = tuple(
            plan
            for plan in dataflow.program.event_plans
            if plan.name in physical_names
            and plan.name not in covered
            and plan.required_dependence.relation in buffer_relations
        )
        if not plans:
            continue
        allocation = bounded_buffer_internal_event_allocation(
            allocation_id=buffer.name,
            buffer=buffer,
            plans=plans,
        )
        allocations.append(allocation)
        covered.update(plan.name for plan in plans)

    residual = tuple(
        plan for plan in dataflow.program.event_plans if plan.name in physical_names and plan.name not in covered
    )
    if residual:
        mechanism_coordinates = residual[0].domain.coordinates
        phased_parameters = {
            "capacity": 1,
            "allocation_owners": ((),) * len(mechanism_coordinates),
            "slots": (0,) * len(mechanism_coordinates),
            "generations": tuple(range(len(mechanism_coordinates))),
        }
        allocations.append(
            phased_internal_event_allocation(
                allocation_id="rhs_gradient_epilogue_scratch",
                plans=residual,
                mechanism_kind=InternalEventMechanismKind.CTA_MBARRIER,
                **phased_parameters,
            )
        )
        scratch_reuse = tuple(
            plan
            for plan in residual
            if plan.required_dependence.relation.source == dataflow.producer_rhs_epilogue_transfer
            and plan.required_dependence.relation.target == dataflow.consumer_rhs_epilogue_stage
        )
        if scratch_reuse:
            allocations.append(
                phased_internal_event_allocation(
                    allocation_id="rhs_gradient_epilogue_async_transfer_queue",
                    plans=scratch_reuse,
                    mechanism_kind=InternalEventMechanismKind.CTA_ASYNC_COMPLETION_QUEUE,
                    **phased_parameters,
                )
            )
    allocations.extend(owned_phased_internal_event_allocation(queue) for queue in schedule.device_accumulation_queues)
    return tuple(allocations)


def _derive_device_accumulation_queues(
    descriptor: StreamingPullbackInputDescriptor,
    owner_partition_count: int,
) -> tuple[OrderedAccumulationEventSchedule, ...]:
    if not descriptor.deterministic_accumulation:
        return ()
    streamed_partition_count = ceil(descriptor.streamed_extent / descriptor.streamed_tile_size)
    owner_axis = TaskAxis("owner_partition", owner_partition_count)
    streamed_axis = TaskAxis("streamed_partition", streamed_partition_count)
    lhs_contribution = TaskFamily(
        f"{descriptor.producer_lhs_gradient_name} global accumulation contribution",
        (owner_axis, streamed_axis),
        placement="device_accumulator_workers",
    )
    score_coordinates = descriptor.score_tile_coordinates
    ranks = None if descriptor.accumulation_plan is None else descriptor.accumulation_plan.contributor_ranks
    queues = [
        derive_ordered_accumulation_event_schedule(
            name="producer_lhs_gradient_accumulation",
            contribution_task=lhs_contribution,
            active_coordinates=score_coordinates,
            allocation_owner_for={coordinate: (coordinate[0],) for coordinate in score_coordinates},
            order_key_for={
                coordinate: (ranks[coordinate],) if ranks is not None else (coordinate[1],)
                for coordinate in score_coordinates
            },
        )
    ]
    if descriptor.relation_group_extent == 1:
        return tuple(queues)

    relation_group_axis = TaskAxis("relation_group", descriptor.relation_group_extent)
    rhs_coordinates = tuple(
        (streamed_partition, relation_group)
        for streamed_partition in range(streamed_partition_count)
        for relation_group in range(descriptor.relation_group_extent)
    )
    for name, task_name in (
        ("producer_rhs_gradient_accumulation", descriptor.producer_rhs_gradient_name),
        ("consumer_rhs_gradient_accumulation", descriptor.consumer_rhs_gradient_name),
    ):
        contribution = TaskFamily(
            f"{task_name} global accumulation contribution",
            (streamed_axis, relation_group_axis),
            placement="device_accumulator_workers",
        )
        queues.append(
            derive_ordered_accumulation_event_schedule(
                name=name,
                contribution_task=contribution,
                active_coordinates=rhs_coordinates,
                allocation_owner_for={coordinate: (coordinate[0],) for coordinate in rhs_coordinates},
                order_key_for={coordinate: (coordinate[1],) for coordinate in rhs_coordinates},
            )
        )
    return tuple(queues)


def _derive_pullback_input_dataflow(
    descriptor: StreamingPullbackInputDescriptor,
    partition_count: int,
) -> StreamingPullbackInputDataflow:
    generation_axis = TaskAxis("generation", 1)
    partition_axis = TaskAxis("owner_partition", partition_count)
    domain = (generation_axis, partition_axis)
    lhs_gradient_warpgroup_axis = TaskAxis("matrix_warpgroup", descriptor.lhs_gradient_warpgroups)
    lhs_gradient_domain = (*domain, lhs_gradient_warpgroup_axis)
    producer_lhs_stage = TaskFamily(descriptor.producer_lhs_stage_name, domain, placement="transfer_workers")
    output_cotangent_stage = TaskFamily(
        descriptor.output_cotangent_stage_name,
        domain,
        placement="transfer_workers",
    )
    score_contract = TaskFamily(descriptor.score_contract_name, domain, placement="matrix_workers")
    output_cotangent_contract = TaskFamily(
        descriptor.output_cotangent_contract_name,
        domain,
        placement="matrix_workers",
    )
    normalized_weight = TaskFamily(descriptor.normalized_weight_name, domain, placement="matrix_workers")
    pullback_map = TaskFamily(descriptor.pullback_map_name, domain, placement="matrix_workers")
    score_vjp = TaskFamily(descriptor.score_vjp_name, domain, placement="matrix_workers")
    producer_lhs_gradient = TaskFamily(
        descriptor.producer_lhs_gradient_name,
        lhs_gradient_domain,
        placement="matrix_workers",
    )
    lhs_gradient_handoff = TaskFamily(
        descriptor.lhs_gradient_handoff_name,
        lhs_gradient_domain,
        placement="transfer_workers",
    )
    producer_rhs_gradient = TaskFamily(
        descriptor.producer_rhs_gradient_name,
        domain,
        placement="matrix_workers",
    )
    consumer_rhs_gradient = TaskFamily(
        descriptor.consumer_rhs_gradient_name,
        domain,
        placement="matrix_workers",
    )
    epilogue_domain = (generation_axis,)
    producer_rhs_epilogue_stage = TaskFamily(
        descriptor.producer_rhs_epilogue_stage_name,
        epilogue_domain,
        placement="matrix_workers",
    )
    producer_rhs_epilogue_transfer = TaskFamily(
        descriptor.producer_rhs_epilogue_transfer_name,
        epilogue_domain,
        placement="transfer_workers",
    )
    consumer_rhs_epilogue_stage = TaskFamily(
        descriptor.consumer_rhs_epilogue_stage_name,
        epilogue_domain,
        placement="matrix_workers",
    )
    consumer_rhs_epilogue_transfer = TaskFamily(
        descriptor.consumer_rhs_epilogue_transfer_name,
        epilogue_domain,
        placement="transfer_workers",
    )
    pointwise = tuple((coordinate, coordinate) for coordinate in producer_lhs_stage.coordinates)
    visibility = MemoryVisibility(EventMemoryScope.CTA)

    def dependence(source: TaskFamily, target: TaskFamily) -> TaskDependence:
        return TaskDependence(TaskRelation.from_pairs(source, target, pointwise), visibility)

    producer_lhs_to_score = dependence(producer_lhs_stage, score_contract)
    producer_lhs_to_rhs_gradient = dependence(producer_lhs_stage, producer_rhs_gradient)
    output_cotangent_to_contract = dependence(output_cotangent_stage, output_cotangent_contract)
    output_cotangent_to_rhs_gradient = dependence(output_cotangent_stage, consumer_rhs_gradient)
    normalized_weight_to_consumer_gradient = dependence(normalized_weight, consumer_rhs_gradient)
    score_cotangent_to_lhs_gradient = TaskDependence(
        TaskRelation.from_pairs(
            score_vjp,
            producer_lhs_gradient,
            tuple(
                (coordinate, (*coordinate, warpgroup))
                for coordinate in score_vjp.coordinates
                for warpgroup in range(descriptor.lhs_gradient_warpgroups)
            ),
        ),
        visibility,
    )
    score_cotangent_to_rhs_gradient = dependence(score_vjp, producer_rhs_gradient)
    lhs_gradient_to_rhs_gradient = TaskDependence(
        TaskRelation.from_pairs(
            producer_lhs_gradient,
            producer_rhs_gradient,
            tuple(
                ((*coordinate, warpgroup), coordinate)
                for coordinate in producer_rhs_gradient.coordinates
                for warpgroup in range(descriptor.lhs_gradient_warpgroups)
            ),
        ),
        visibility,
    )
    lhs_gradient_to_handoff = TaskDependence(
        TaskRelation.from_pairs(
            producer_lhs_gradient,
            lhs_gradient_handoff,
            tuple((coordinate, coordinate) for coordinate in producer_lhs_gradient.coordinates),
        ),
        visibility,
    )
    producer_rhs_to_epilogue = TaskDependence(
        TaskRelation.from_pairs(
            producer_rhs_gradient,
            producer_rhs_epilogue_stage,
            tuple((coordinate, (coordinate[0],)) for coordinate in producer_rhs_gradient.coordinates),
        ),
        visibility,
    )
    consumer_rhs_to_epilogue = TaskDependence(
        TaskRelation.from_pairs(
            consumer_rhs_gradient,
            consumer_rhs_epilogue_stage,
            tuple((coordinate, (coordinate[0],)) for coordinate in consumer_rhs_gradient.coordinates),
        ),
        visibility,
    )
    producer_epilogue_stage_to_transfer = TaskDependence(
        TaskRelation.from_pairs(
            producer_rhs_epilogue_stage,
            producer_rhs_epilogue_transfer,
            tuple((coordinate, coordinate) for coordinate in producer_rhs_epilogue_stage.coordinates),
        ),
        visibility,
    )
    consumer_epilogue_stage_to_transfer = TaskDependence(
        TaskRelation.from_pairs(
            consumer_rhs_epilogue_stage,
            consumer_rhs_epilogue_transfer,
            tuple((coordinate, coordinate) for coordinate in consumer_rhs_epilogue_stage.coordinates),
        ),
        visibility,
    )
    epilogue_scratch_reuse = TaskDependence(
        TaskRelation.from_pairs(
            producer_rhs_epilogue_transfer,
            consumer_rhs_epilogue_stage,
            tuple((coordinate, coordinate) for coordinate in producer_rhs_epilogue_transfer.coordinates),
        ),
        visibility,
    )
    initial_dependences = (
        producer_lhs_to_score,
        producer_lhs_to_rhs_gradient,
        output_cotangent_to_contract,
        output_cotangent_to_rhs_gradient,
        dependence(score_contract, normalized_weight),
        dependence(output_cotangent_contract, pullback_map),
        dependence(normalized_weight, pullback_map),
        dependence(pullback_map, score_vjp),
        score_cotangent_to_lhs_gradient,
        score_cotangent_to_rhs_gradient,
        normalized_weight_to_consumer_gradient,
        dependence(pullback_map, consumer_rhs_gradient),
        lhs_gradient_to_rhs_gradient,
        lhs_gradient_to_handoff,
        producer_rhs_to_epilogue,
        consumer_rhs_to_epilogue,
        producer_epilogue_stage_to_transfer,
        epilogue_scratch_reuse,
        consumer_epilogue_stage_to_transfer,
    )
    families = (
        producer_lhs_stage,
        output_cotangent_stage,
        score_contract,
        output_cotangent_contract,
        normalized_weight,
        pullback_map,
        score_vjp,
        producer_lhs_gradient,
        lhs_gradient_handoff,
        producer_rhs_gradient,
        consumer_rhs_gradient,
        producer_rhs_epilogue_stage,
        producer_rhs_epilogue_transfer,
        consumer_rhs_epilogue_stage,
        consumer_rhs_epilogue_transfer,
    )
    initial_program = _event_program(families, initial_dependences)

    coordinates = producer_lhs_stage.coordinates
    producer_lhs_buffer = derive_bounded_buffer_plan(
        name="producer_lhs_state_pipeline",
        program=initial_program,
        producer=producer_lhs_stage,
        uses=(producer_lhs_to_score.relation, producer_lhs_to_rhs_gradient.relation),
        capacity=descriptor.producer_lhs_pipeline_depth,
        allocation_owner_for={coordinate: () for coordinate in coordinates},
        slot_for={
            coordinate: index % descriptor.producer_lhs_pipeline_depth for index, coordinate in enumerate(coordinates)
        },
        generation_for={
            coordinate: index // descriptor.producer_lhs_pipeline_depth for index, coordinate in enumerate(coordinates)
        },
        visibility=visibility,
    )
    output_cotangent_buffer = derive_bounded_buffer_plan(
        name="output_cotangent_state_pipeline",
        program=initial_program,
        producer=output_cotangent_stage,
        uses=(output_cotangent_to_contract.relation, output_cotangent_to_rhs_gradient.relation),
        capacity=descriptor.output_cotangent_pipeline_depth,
        allocation_owner_for={coordinate: () for coordinate in coordinates},
        slot_for={
            coordinate: index % descriptor.output_cotangent_pipeline_depth
            for index, coordinate in enumerate(coordinates)
        },
        generation_for={
            coordinate: index // descriptor.output_cotangent_pipeline_depth
            for index, coordinate in enumerate(coordinates)
        },
        visibility=visibility,
    )
    normalized_weight_buffer = (
        None
        if descriptor.normalized_weight_storage is StreamingIntermediateStorage.REGISTERS
        else derive_bounded_buffer_plan(
            name="normalized_weight_workspace",
            program=initial_program,
            producer=normalized_weight,
            uses=(normalized_weight_to_consumer_gradient.relation,),
            capacity=descriptor.score_workspace_depth,
            allocation_owner_for={coordinate: () for coordinate in coordinates},
            slot_for={
                coordinate: index % descriptor.score_workspace_depth for index, coordinate in enumerate(coordinates)
            },
            generation_for={
                coordinate: index // descriptor.score_workspace_depth for index, coordinate in enumerate(coordinates)
            },
            visibility=visibility,
        )
    )
    score_cotangent_buffer = derive_bounded_buffer_plan(
        name="score_cotangent_workspace",
        program=initial_program,
        producer=score_vjp,
        uses=(score_cotangent_to_lhs_gradient.relation, score_cotangent_to_rhs_gradient.relation),
        capacity=descriptor.score_workspace_depth,
        allocation_owner_for={coordinate: () for coordinate in coordinates},
        slot_for={coordinate: index % descriptor.score_workspace_depth for index, coordinate in enumerate(coordinates)},
        generation_for={
            coordinate: index // descriptor.score_workspace_depth for index, coordinate in enumerate(coordinates)
        },
        visibility=visibility,
    )
    lhs_gradient_handoff_buffer = derive_bounded_buffer_plan(
        name="lhs_gradient_handoff",
        program=initial_program,
        producer=producer_lhs_gradient,
        uses=(lhs_gradient_to_handoff.relation,),
        capacity=descriptor.lhs_gradient_warpgroups,
        allocation_owner_for={coordinate: () for coordinate in producer_lhs_gradient.coordinates},
        slot_for={coordinate: coordinate[-1] for coordinate in producer_lhs_gradient.coordinates},
        generation_for={coordinate: coordinate[1] for coordinate in producer_lhs_gradient.coordinates},
        visibility=visibility,
    )
    workspace_reuse_dependences = (
        *(() if normalized_weight_buffer is None else normalized_weight_buffer.reuse_dependences),
        *score_cotangent_buffer.reuse_dependences,
        *lhs_gradient_handoff_buffer.reuse_dependences,
    )
    complete_program = _event_program(
        families,
        (
            *initial_dependences,
            *producer_lhs_buffer.reuse_dependences,
            *output_cotangent_buffer.reuse_dependences,
            *workspace_reuse_dependences,
        ),
    )
    verify_event_dataflow_program(complete_program)
    return StreamingPullbackInputDataflow(
        complete_program,
        producer_lhs_stage,
        output_cotangent_stage,
        score_contract,
        output_cotangent_contract,
        normalized_weight,
        pullback_map,
        score_vjp,
        producer_lhs_gradient,
        lhs_gradient_handoff,
        producer_rhs_gradient,
        consumer_rhs_gradient,
        producer_rhs_epilogue_stage,
        producer_rhs_epilogue_transfer,
        consumer_rhs_epilogue_stage,
        consumer_rhs_epilogue_transfer,
        partition_count,
        producer_lhs_buffer,
        output_cotangent_buffer,
        normalized_weight_buffer,
        score_cotangent_buffer,
        lhs_gradient_handoff_buffer,
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
                if dependence.relation.target.placement == "transfer_workers"
                else EventGenerationPolicy.PER_INVOCATION
            ),
        )
        for index, dependence in enumerate(dependences)
    )
    return EventDataflowProgram(families, dependences, plans)


def _pullback_input_realization_audit(
    dataflow: StreamingPullbackInputDataflow,
) -> EventRealizationAudit:
    input_reuse_relations = {
        dependence.relation
        for buffer in (dataflow.producer_lhs_buffer, dataflow.output_cotangent_buffer)
        for dependence in buffer.reuse_dependences
    }
    workspace_buffers = (
        *((dataflow.normalized_weight_buffer,) if dataflow.normalized_weight_buffer is not None else ()),
        dataflow.score_cotangent_buffer,
    )
    workspace_reuse_relations = {
        dependence.relation for buffer in workspace_buffers for dependence in buffer.reuse_dependences
    }
    lhs_gradient_handoff_reuse_relations = {
        dependence.relation for dependence in dataflow.lhs_gradient_handoff_buffer.reuse_dependences
    }
    lhs_gradient_handoff_full_relation = next(
        dependence.relation
        for dependence in dataflow.program.dependences
        if dependence.relation.source == dataflow.producer_lhs_gradient
        and dependence.relation.target == dataflow.lhs_gradient_handoff
    )
    normalized_weight_full_relation = next(
        relation
        for relation in dataflow.program.dependences
        if relation.relation.source == dataflow.normalized_weight
        and relation.relation.target == dataflow.consumer_rhs_gradient
    ).relation
    score_cotangent_full_relations = {
        dependence.relation
        for dependence in dataflow.program.dependences
        if dependence.relation.source == dataflow.score_vjp
        and dependence.relation.target in (dataflow.producer_lhs_gradient, dataflow.producer_rhs_gradient)
    }
    terminal_to_epilogue_relations = {
        dependence.relation
        for dependence in dataflow.program.dependences
        if (
            dependence.relation.source == dataflow.producer_rhs_gradient
            and dependence.relation.target == dataflow.producer_rhs_epilogue_stage
        )
        or (
            dependence.relation.source == dataflow.consumer_rhs_gradient
            and dependence.relation.target == dataflow.consumer_rhs_epilogue_stage
        )
    }
    epilogue_full_relations = {
        dependence.relation
        for dependence in dataflow.program.dependences
        if (
            dependence.relation.source == dataflow.producer_rhs_epilogue_stage
            and dependence.relation.target == dataflow.producer_rhs_epilogue_transfer
        )
        or (
            dependence.relation.source == dataflow.consumer_rhs_epilogue_stage
            and dependence.relation.target == dataflow.consumer_rhs_epilogue_transfer
        )
    }
    epilogue_scratch_reuse_relation = next(
        dependence.relation
        for dependence in dataflow.program.dependences
        if dependence.relation.source == dataflow.producer_rhs_epilogue_transfer
        and dependence.relation.target == dataflow.consumer_rhs_epilogue_stage
    )
    first_consumers = {dataflow.score_contract, dataflow.output_cotangent_contract}
    later_buffer_consumers = {dataflow.producer_rhs_gradient, dataflow.consumer_rhs_gradient}
    realizations = []
    for plan in dataflow.program.event_plans:
        relation = plan.required_dependence.relation
        if relation in input_reuse_relations:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="TMA pipeline empty barrier with phased slot generation",
                    reason="the true last matrix consumer must release a bounded input slot before reuse",
                )
            )
        elif relation in lhs_gradient_handoff_reuse_relations:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="lhs-gradient handoff empty barrier",
                    reason=(
                        "the transfer worker must release each warpgroup slot before the next gradient tile writes it"
                    ),
                )
            )
        elif relation == lhs_gradient_handoff_full_relation:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="lhs-gradient handoff full barrier",
                    reason="the transfer worker consumes a shared gradient tile produced by a matrix warpgroup",
                )
            )
        elif relation in workspace_reuse_relations:
            buffer = next(
                buffer
                for buffer in workspace_buffers
                if relation in {dependence.relation for dependence in buffer.reuse_dependences}
            )
            if buffer.capacity == 1:
                realizations.append(
                    physical_event_realization(
                        plan,
                        mechanism="cross-warpgroup score-workspace reuse barrier",
                        reason="the prior matrix consumer must finish before one shared score slot is overwritten",
                    )
                )
            else:
                realizations.append(
                    erased_event_realization(
                        plan,
                        kind=EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
                        mechanism="bounded generations plus ordered WGMMA completion",
                        reason="distinct workspace slots and loop completion order cover reuse after refinement",
                    )
                )
        elif relation == normalized_weight_full_relation:
            if dataflow.normalized_weight_buffer is None:
                realizations.append(
                    erased_event_realization(
                        plan,
                        kind=EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
                        mechanism="register operand source",
                        reason="the normalized weight stays in registers for its terminal Contract",
                    )
                )
            else:
                realizations.append(
                    physical_event_realization(
                        plan,
                        mechanism="cross-warpgroup normalized-weight full barrier",
                        reason="the terminal Contract consumes a shared-memory normalized-weight tile",
                    )
                )
        elif relation in score_cotangent_full_relations:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="cross-warpgroup score-cotangent full barrier",
                    reason="terminal gradient Contracts consume the shared score-cotangent tile",
                )
            )
        elif relation in terminal_to_epilogue_relations:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
                    mechanism="matrix accumulator and ordered WGMMA completion",
                    reason="the terminal Contract accumulator is complete before its fused epilogue stage",
                )
            )
        elif relation in epilogue_full_relations:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="rhs-gradient epilogue full barrier and asynchronous transfer queue",
                    reason="the transfer worker consumes a collectively staged shared-memory gradient tile",
                )
            )
        elif relation == epilogue_scratch_reuse_relation:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="overlapping epilogue scratch reuse barrier and asynchronous wait",
                    reason="the first shared epilogue view must finish transfer before the overlapping view is written",
                )
            )
        elif relation.source in (dataflow.producer_lhs_stage, dataflow.output_cotangent_stage) and (
            relation.target in first_consumers
        ):
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="TMA pipeline full barrier",
                    reason="the first matrix consumer requires acquire visibility after asynchronous tile movement",
                )
            )
        elif relation.source in (dataflow.producer_lhs_stage, dataflow.output_cotangent_stage) and (
            relation.target in later_buffer_consumers
        ):
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
                    mechanism="first-consumer readiness plus matrix-owner refinement",
                    reason="the selected tile body reaches the later buffer consumer only after the full barrier wait",
                )
            )
        else:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="matrix-owner program order",
                    reason="the selected tile body executes this register dependence in order",
                )
            )
    return verify_event_realizations(dataflow.program, tuple(realizations))
