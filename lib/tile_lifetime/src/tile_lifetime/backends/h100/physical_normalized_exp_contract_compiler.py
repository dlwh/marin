# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile typed Shuttle normalized-exponential regions into SM90 artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from threading import Lock
from typing import Any

import torch

from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineExpression,
    CuteEventRealizationKind,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CuteSourceTaskRole,
    CuteStructuralEventCover,
    bind_event_realizations,
    bind_launch_source_tasks,
    bind_physical_event_provenance,
    combine_physical_event_schedules,
)
from tile_lifetime.h100_streaming_lowering import (
    H100StreamingSchedule,
    Sm90BackwardAccumulatorSchedule,
    Sm90ForwardPipelineSchedule,
    Sm90RelationPackedLayout,
    select_sm90_contract_schedule,
    sm90_relation_packed_layout,
)
from tile_lifetime.internal_event_allocation import internal_event_device_legalization_region
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractProgram,
    normalized_exp_contract_pullback_program,
)
from tile_lifetime.physical_artifact_compiler import (
    GpuPhysicalArtifactRequest,
    ShuttleOperationRecord,
    ShuttleTypeRecord,
    native_gpu_physical_artifact_provider,
)
from tile_lifetime.physical_event_cover import (
    affine_expression_from_record,
    derive_lpt_algebraic_refinement_covers,
    derive_single_tile_algebraic_refinement_covers,
)
from tile_lifetime.physical_normalized_exp_contract import (
    NormalizedExpContractGeometry,
    PhysicalNormalizedExpContractPullback,
    PhysicalNormalizedExpContractRegion,
    PhysicalNormalizedExpForwardMatch,
    PhysicalNormalizedExpPullbackMatch,
    SourceBoundBuffer,
    contract_indexing_dimensions,
    normalized_exp_consumer_dimension_roles,
    normalized_exp_contract_program,
    normalized_exp_sufficient_state_cut_types,
    physical_normalized_exp_contract_match,
    source_argument_buffers,
    source_bound_schedule_tasks,
    source_result_buffers,
)
from tile_lifetime.streaming_event_schedule import streaming_physical_event_schedule_record
from tile_lifetime.streaming_pullback_event_schedule import streaming_pullback_input_event_schedule_record
from tile_lifetime.tensor_program import serialize_scalar_expression

from .cute_streaming_backward_emitter import (
    compile_h100_normalized_exponential_value_and_grad,
    export_h100_normalized_exponential_forward_artifact,
    export_h100_normalized_exponential_value_and_grad_artifact,
)
from .cute_streaming_emitter import compile_h100_streaming_program

_ARTIFACT_CACHE: dict[str, CutePipelineArtifact] = {}
_ARTIFACT_CACHE_LOCK = Lock()
_SM90_LOGICAL_SUBGROUP_SIZE = 32
_SM90_FORWARD_LOGICAL_THREADS = 128
_SM90_FORWARD_LOGICAL_SCRATCH_BYTES = 64 * 1024
_SM90_REVERSE_THREADS = 384
_SM90_REVERSE_SCRATCH_BYTES = 194 * 1024
_ROW_FOLD_TASK = "#shuttle.schedule_task_kind<row_fold>"
_CONTRACT_TASK = "#shuttle.schedule_task_kind<contract>"


def compile_h100_normalized_exp_contract_request(
    request: GpuPhysicalArtifactRequest,
) -> CutePipelineArtifact | None:
    """Compile one generic FAST Contract/Fold region or decline an unrelated request."""
    if request.target != "sm_90":
        return None
    with _ARTIFACT_CACHE_LOCK:
        cached = _ARTIFACT_CACHE.get(request.fingerprint)
    if cached is not None:
        return cached
    try:
        match = physical_normalized_exp_contract_match(request)
    except ValueError as error:
        raise ValueError(f"SM90 normalized-exponential physical extraction failed: {error}") from error
    if match is None:
        return None
    if isinstance(match, PhysicalNormalizedExpPullbackMatch):
        artifact = _compile_pullback_request(request, match.pullback)
        with _ARTIFACT_CACHE_LOCK:
            incumbent = _ARTIFACT_CACHE.setdefault(request.fingerprint, artifact)
        return incumbent
    assert isinstance(match, PhysicalNormalizedExpForwardMatch)
    region = match.region

    producer_left_buffer, producer_right_buffer, consumer_buffer, output_buffer = _source_boundary(request, region)
    program = _streaming_program(region, schedule=_sm90_forward_schedule(request, region))
    query_storage = _empty_bf16(producer_left_buffer.type.shape)
    key_storage = _empty_bf16(producer_right_buffer.type.shape)
    value_storage = _empty_bf16(consumer_buffer.type.shape)
    output_storage = _empty_bf16(output_buffer.type.shape)
    query, key, value, output = _relation_packed_views(
        region,
        program.schedule.contract.relation_layout,
        query_storage=query_storage,
        key_storage=key_storage,
        value_storage=value_storage,
        output_storage=output_storage,
    )
    compiled = compile_h100_streaming_program(
        program,
        query=query,
        key=key,
        value=value,
        output=output,
        log_sum_exp=None,
        domain_traversal=_domain_traversal(region),
    )
    physical_event_provenance = streaming_physical_event_schedule_record(compiled.event_schedule)
    artifact = export_h100_normalized_exponential_forward_artifact(
        compiled,
        compile_request=_compile_request(
            request,
            region,
            compiled.schedule,
            physical_event_schedule=physical_event_provenance,
        ),
        function_prefix=f"shuttle_physical_{request.fingerprint[:16]}",
        source_slot_shapes={
            "query": producer_left_buffer.type.shape,
            "key": producer_right_buffer.type.shape,
            "value": consumer_buffer.type.shape,
            "output": output_buffer.type.shape,
        },
        source_slot_bindings={
            "query": _source_abi_binding(request, producer_left_buffer.ordinal),
            "key": _source_abi_binding(request, producer_right_buffer.ordinal),
            "value": _source_abi_binding(request, consumer_buffer.ordinal),
            "output": _source_abi_binding(request, output_buffer.ordinal),
        },
        source_slot_ordinals={
            "query": producer_left_buffer.ordinal,
            "key": producer_right_buffer.ordinal,
            "value": consumer_buffer.ordinal,
            "output": output_buffer.ordinal,
        },
    )
    artifact = _bind_structural_source_tasks(
        artifact,
        request,
        region=region,
        forward_schedule=compiled.schedule,
    )
    artifact = _bind_internal_event_provenance(artifact, physical_event_provenance)
    artifact = replace(
        artifact,
        internal_event_legalizations=(
            internal_event_device_legalization_region(0, compiled.event_schedule.allocations),
        ),
    )
    with _ARTIFACT_CACHE_LOCK:
        incumbent = _ARTIFACT_CACHE.setdefault(request.fingerprint, artifact)
    return incumbent


def _compile_pullback_request(
    request: GpuPhysicalArtifactRequest,
    pullback: PhysicalNormalizedExpContractPullback,
) -> CutePipelineArtifact:
    region = pullback.forward
    query_buffer, key_buffer, value_buffer, output_cotangent_buffer = _pullback_source_boundary(request, pullback)
    program = _streaming_program(region, schedule=_sm90_contract_fold_forward_schedule(region))
    backward = normalized_exp_contract_pullback_program(
        program,
        schedule=_sm90_reverse_schedule(request, region),
        traversal=_domain_traversal(region),
    )
    if backward.pullback.score_map_vjp.expression != pullback.score_map_vjp_expression:
        raise ValueError("source-bound score-Map VJP disagrees with generic scalar differentiation")
    backward = replace(
        backward,
        pullback=replace(
            backward.pullback,
            score_map_vjp=replace(
                backward.pullback.score_map_vjp,
                expression=pullback.score_map_vjp_expression,
            ),
            normalized_exp_pullback=replace(
                backward.pullback.normalized_exp_pullback,
                expression=pullback.normalized_exp_pullback_expression,
            ),
        ),
    )

    query_storage = _empty_bf16(query_buffer.type.shape)
    key_storage = _empty_bf16(key_buffer.type.shape)
    value_storage = _empty_bf16(value_buffer.type.shape)
    output_storage = _empty_bf16(region.buffers[pullback.forward_output_buffer].type.shape)
    output_cotangent_storage = _empty_bf16(output_cotangent_buffer.type.shape)
    query, key, value, output = _relation_packed_views(
        region,
        program.schedule.contract.relation_layout,
        query_storage=query_storage,
        key_storage=key_storage,
        value_storage=value_storage,
        output_storage=output_storage,
    )
    output_cotangent = output_cotangent_storage.as_strided(output.shape, output.stride())
    compiled = compile_h100_normalized_exponential_value_and_grad(
        backward,
        query=query,
        key=key,
        value=value,
        output_cotangent=output_cotangent,
        domain_traversal=_domain_traversal(region),
    )
    buffers = region.buffers
    artifact = export_h100_normalized_exponential_value_and_grad_artifact(
        compiled,
        function_prefix=f"shuttle_physical_{request.fingerprint[:16]}",
        source_slot_shapes={
            "query": query_buffer.type.shape,
            "key": key_buffer.type.shape,
            "value": value_buffer.type.shape,
            "output_cotangent": output_cotangent_buffer.type.shape,
            "output": buffers[pullback.forward_output_buffer].type.shape,
            "key_cotangent": buffers[pullback.key_cotangent_buffer].type.shape,
            "value_cotangent": buffers[pullback.value_cotangent_buffer].type.shape,
            "query_cotangent": buffers[pullback.query_cotangent_buffer].type.shape,
            "log_sum_exp": buffers[pullback.maximum_state_buffer].type.shape,
        },
        source_slot_bindings={
            "query": _source_abi_binding(request, query_buffer.ordinal),
            "key": _source_abi_binding(request, key_buffer.ordinal),
            "value": _source_abi_binding(request, value_buffer.ordinal),
            "output_cotangent": _source_abi_binding(request, output_cotangent_buffer.ordinal),
            "output": _source_abi_binding(request, pullback.forward_output_buffer),
            "key_cotangent": _source_abi_binding(request, pullback.key_cotangent_buffer),
            "value_cotangent": _source_abi_binding(request, pullback.value_cotangent_buffer),
            "query_cotangent": _source_abi_binding(request, pullback.query_cotangent_buffer),
        },
        source_slot_ordinals={
            "query": query_buffer.ordinal,
            "key": key_buffer.ordinal,
            "value": value_buffer.ordinal,
            "output_cotangent": output_cotangent_buffer.ordinal,
            "output": pullback.forward_output_buffer,
            "key_cotangent": pullback.key_cotangent_buffer,
            "value_cotangent": pullback.value_cotangent_buffer,
            "query_cotangent": pullback.query_cotangent_buffer,
            "log_sum_exp": pullback.maximum_state_buffer,
            "log_sum_exp_log2": pullback.normalizer_state_buffer,
            "output_dot": pullback.output_dependent_state_buffer,
        },
        source_slot_reuse_groups={
            "log_sum_exp": _source_reuse_group(request, pullback.maximum_state_buffer),
            "log_sum_exp_log2": _source_reuse_group(request, pullback.normalizer_state_buffer),
            "output_dot": _source_reuse_group(request, pullback.output_dependent_state_buffer),
        },
    )
    artifact = _bind_structural_source_tasks(
        artifact,
        request,
        region=region,
        forward_schedule=compiled.forward_template.schedule,
    )
    physical_event_provenance = _combined_internal_event_provenance(
        streaming_physical_event_schedule_record(compiled.forward_template.event_schedule),
        streaming_pullback_input_event_schedule_record(compiled.backward_template.lowering.input_event_schedule),
    )
    artifact = _bind_internal_event_provenance(artifact, physical_event_provenance)
    artifact = replace(
        artifact,
        internal_event_legalizations=(
            internal_event_device_legalization_region(0, compiled.forward_template.event_schedule.allocations),
            internal_event_device_legalization_region(
                1, compiled.backward_template.lowering.input_event_schedule.allocations
            ),
        ),
    )
    return replace(
        artifact,
        request=_compile_request(
            request,
            region,
            compiled.backward_template.physical_schedule,
            physical_event_schedule=physical_event_provenance,
            lowered_scalar_programs=compiled.backward_template.lowering.compile_request.scalar_programs,
        ),
    )


def _pullback_source_boundary(
    request: GpuPhysicalArtifactRequest,
    pullback: PhysicalNormalizedExpContractPullback,
) -> tuple[SourceBoundBuffer, SourceBoundBuffer, SourceBoundBuffer, SourceBoundBuffer]:
    region = pullback.forward
    geometry = region.geometry
    inputs = source_argument_buffers(request, region.buffers)
    try:
        query = inputs[region.producer_left_argument]
        key = inputs[region.producer_right_argument]
        value = inputs[region.consumer_argument]
        output_cotangent = inputs[pullback.output_cotangent_argument]
    except KeyError as error:
        raise ValueError("stateful Contract/Fold source bindings exceed the public input boundary") from error
    expected = (
        (
            *geometry.batch_shape,
            geometry.shared_relation_extent,
            *geometry.relation_group_shape,
            geometry.owner_extent,
            geometry.producer_feature_extent,
        ),
        (
            *geometry.batch_shape,
            geometry.shared_relation_extent,
            geometry.streamed_extent,
            geometry.producer_feature_extent,
        ),
        (
            *geometry.batch_shape,
            geometry.shared_relation_extent,
            geometry.streamed_extent,
            geometry.consumer_feature_extent,
        ),
        (
            *geometry.batch_shape,
            geometry.shared_relation_extent,
            *geometry.relation_group_shape,
            geometry.owner_extent,
            geometry.consumer_feature_extent,
        ),
    )
    for name, buffer, shape in zip(
        ("producer_left", "producer_right", "consumer_operand", "output_cotangent"),
        (query, key, value, output_cotangent),
        expected,
        strict=True,
    ):
        if buffer.type.element_type != "bf16" or buffer.type.shape != shape:
            raise ValueError(f"source {name} must be BF16 with relation-derived shape {shape}")
    if geometry.producer_feature_extent != 128 or geometry.consumer_feature_extent != 128:
        raise ValueError("the current SM90 value-and-pullback artifact requires producer and consumer D=128")
    if geometry.relation_group_size & (geometry.relation_group_size - 1):
        raise ValueError("the current SM90 packed relation requires a power-of-two relation group")
    return query, key, value, output_cotangent


def _source_abi_binding(request: GpuPhysicalArtifactRequest, ordinal: int) -> int:
    if ordinal < 0 or ordinal >= len(request.binding_indices):
        raise ValueError(f"source buffer {ordinal} has no public ABI binding record")
    binding = request.binding_indices[ordinal]
    if binding is None:
        raise ValueError(f"source buffer {ordinal} is not an external public ABI binding")
    return binding


def _source_reuse_group(request: GpuPhysicalArtifactRequest, ordinal: int) -> int:
    if ordinal < 0 or ordinal >= len(request.reuse_groups):
        raise ValueError(f"source buffer {ordinal} has no physical reuse record")
    return request.reuse_groups[ordinal]


def _event_index_expression(event: ShuttleOperationRecord, name: str) -> tuple[int, CuteAffineExpression]:
    mapping = event.attributes.get(name)
    if not isinstance(mapping, dict):
        raise ValueError(f"source EventTensor has no structured {name} map")
    dimensions = mapping.get("dimensions")
    results = mapping.get("results")
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or not isinstance(results, list)
        or len(results) != 1
    ):
        raise ValueError("the current SM90 refinement requires one flattened Event endpoint index")
    return dimensions, affine_expression_from_record(results[0])


def _algebraic_refinement_covers(
    cover: CuteStructuralEventCover,
    event: ShuttleOperationRecord,
    geometry: NormalizedExpContractGeometry,
    schedule: H100StreamingSchedule,
    traversal: ContractDomainTraversal,
) -> tuple[CuteStructuralEventCover, ...]:
    """Derive the exact common-CTA map for an online Contract/Fold refinement."""
    dimensions, producer_row = _event_index_expression(event, "producer_indices")
    layout = schedule.contract.relation_layout
    if layout is None:
        raise ValueError("SM90 algebraic Event refinement requires an explicit relation layout")
    batch = layout.batch_extent
    shared_relation = layout.shared_relation_axis.extent
    relation_group = layout.relation_group_size
    owner_extent = layout.owner_extent
    if (
        batch != geometry.batch_size
        or shared_relation != geometry.shared_relation_extent
        or relation_group != geometry.relation_group_size
        or owner_extent != geometry.owner_extent
    ):
        raise ValueError("SM90 Event owner map disagrees with the algebraic relation geometry")

    arguments = {
        "producer_row": producer_row,
        "relation_dimensions": dimensions,
        "batch": batch,
        "shared_relation": shared_relation,
        "relation_group": relation_group,
        "owner_extent": owner_extent,
        "owner_tile": schedule.tile_m,
    }
    if traversal is ContractDomainTraversal.FULL:
        return derive_single_tile_algebraic_refinement_covers(cover, **arguments)
    return derive_lpt_algebraic_refinement_covers(
        cover,
        **arguments,
        producer_feature_extent=geometry.producer_feature_extent,
        consumer_feature_extent=geometry.consumer_feature_extent,
    )


def _bind_structural_source_tasks(
    artifact: CutePipelineArtifact,
    request: GpuPhysicalArtifactRequest,
    *,
    region: PhysicalNormalizedExpContractRegion | None = None,
    forward_schedule: H100StreamingSchedule | None = None,
) -> CutePipelineArtifact:
    schedules = source_bound_schedule_tasks(request)
    recipe = artifact.launch_recipe
    if recipe is None:
        raise ValueError("physical source-task projection requires a CuTe launch recipe")
    tasks_by_role: dict[CuteSourceTaskRole, list[int]] = {}
    task_roles: dict[int, CuteSourceTaskRole] = {}
    stateless_cut = len(schedules) == 2 and all(
        not schedule.state_inputs and not schedule.state_outputs for schedule in schedules
    )
    for schedule in schedules:
        if stateless_cut:
            role = CuteSourceTaskRole.STATE_ROOT if not schedule.dependencies else CuteSourceTaskRole.STATE_TERMINAL
        elif not schedule.state_inputs:
            role = CuteSourceTaskRole.STATE_ROOT
        elif schedule.state_outputs:
            role = CuteSourceTaskRole.STATE_CARRY
        else:
            role = CuteSourceTaskRole.STATE_TERMINAL
        tasks_by_role.setdefault(role, []).append(schedule.ordinal)
        task_roles[schedule.ordinal] = role
    bound_recipe = bind_launch_source_tasks(
        recipe,
        {role: tuple(source_tasks) for role, source_tasks in tasks_by_role.items()},
    )
    event_operations = tuple(
        operation for operation in request.schedule.body().operations if operation.name == "shuttle.event_tensor"
    )
    structural_covers: list[CuteStructuralEventCover] = []
    for cover in bound_recipe.structural_event_covers:
        if cover.kind is not CuteEventRealizationKind.ALGEBRAIC_REFINEMENT:
            structural_covers.append(cover)
            continue
        if region is None or forward_schedule is None:
            raise ValueError("algebraic Event refinement requires the selected physical schedule")
        matching_events = tuple(
            operation
            for operation in event_operations
            if task_roles[int(operation.attributes["producer_task"])] is cover.producer_role
            and task_roles[int(operation.attributes["consumer_task"])] is cover.consumer_role
        )
        if len(matching_events) != 1:
            raise ValueError("algebraic Event refinement requires one exact source Event relation")
        structural_covers.extend(
            _algebraic_refinement_covers(
                cover,
                matching_events[0],
                region.geometry,
                forward_schedule,
                _domain_traversal(region),
            )
        )
    bound_recipe = replace(bound_recipe, structural_event_covers=tuple(structural_covers))
    source_events = tuple(
        (
            int(operation.attributes["ordinal"]),
            int(operation.attributes["producer_task"]),
            int(operation.attributes["consumer_task"]),
            len(operation.attributes["relation_domain_shape"]),
            operation.attributes.get("producer_phase"),
            operation.attributes.get("consumer_phase"),
        )
        for operation in event_operations
    )
    bound_recipe = bind_event_realizations(
        bound_recipe,
        source_events=source_events,
        source_task_roles=task_roles,
    )
    return replace(artifact, launch_recipe=bound_recipe)


def _source_boundary(
    request: GpuPhysicalArtifactRequest,
    region: PhysicalNormalizedExpContractRegion,
) -> tuple[SourceBoundBuffer, SourceBoundBuffer, SourceBoundBuffer, SourceBoundBuffer]:
    inputs = source_argument_buffers(request, region.buffers)
    outputs = source_result_buffers(request, region.buffers)
    if len(inputs) != 3 or len(outputs) != 1:
        raise ValueError("the current SM90 forward boundary requires three public inputs and one public output")
    try:
        producer_left = inputs[region.producer_left_argument]
        producer_right = inputs[region.producer_right_argument]
        consumer_operand = inputs[region.consumer_argument]
        output = outputs[region.output_result]
    except (IndexError, KeyError) as error:
        raise ValueError("normalized-exponential source bindings exceed the materialization boundary") from error
    geometry = region.geometry
    for name, buffer in (
        ("producer_left", producer_left),
        ("producer_right", producer_right),
        ("consumer_operand", consumer_operand),
        ("output", output),
    ):
        if buffer.type.element_type != "bf16":
            raise ValueError(f"the current SM90 forward boundary requires BF16 {name} storage")
    if geometry.relation_group_size & (geometry.relation_group_size - 1):
        raise ValueError("the current SM90 packed relation requires a power-of-two relation group")
    return producer_left, producer_right, consumer_operand, output


def _streaming_program(
    region: PhysicalNormalizedExpContractRegion,
    *,
    schedule: Sm90ForwardPipelineSchedule,
) -> NormalizedExpContractProgram[Sm90ForwardPipelineSchedule]:
    return normalized_exp_contract_program(
        region,
        schedule=schedule,
    )


def _sm90_contract_fold_forward_schedule(
    region: PhysicalNormalizedExpContractRegion,
) -> Sm90ForwardPipelineSchedule:
    """Select the shared SM90 family from Contract/Fold resource dimensions."""
    geometry = region.geometry
    return Sm90ForwardPipelineSchedule(
        contract=select_sm90_contract_schedule(
            owner_extent=geometry.owner_extent,
            streamed_extent=geometry.streamed_extent,
            producer_feature_extent=geometry.producer_feature_extent,
            consumer_feature_extent=geometry.consumer_feature_extent,
            relation_layout=_sm90_relation_layout(region),
        ),
        stages=2,
    )


def _sm90_forward_schedule(
    request: GpuPhysicalArtifactRequest,
    region: PhysicalNormalizedExpContractRegion,
) -> Sm90ForwardPipelineSchedule:
    schedules = source_bound_schedule_tasks(request)
    candidates = tuple(schedule for schedule in schedules if not schedule.state_inputs and not schedule.state_outputs)
    if len(candidates) != 2:
        raise ValueError(
            "SM90 forward Contract/Fold lowering requires one algebraically refinable Contract/Fold cut, "
            f"found {len(candidates)} stateless tasks"
        )
    _verify_online_refinement_cut(request, region, candidates)
    return _sm90_contract_fold_forward_schedule(region)


def _verify_online_refinement_cut(
    request: GpuPhysicalArtifactRequest,
    region: PhysicalNormalizedExpContractRegion,
    schedules: tuple[Any, ...],
) -> None:
    """Bind the generic online sufficient-state rewrite to its exact source cut."""
    producer, consumer = schedules
    if producer.kind != _ROW_FOLD_TASK or consumer.kind != _CONTRACT_TASK:
        raise ValueError("online SM90 refinement requires row Fold followed by weighted Contract")
    if producer.dependencies or consumer.dependencies != (producer.ordinal,):
        raise ValueError("online SM90 refinement requires one exact Fold-to-Contract dependency")
    if producer.subgroup_size != 32 or consumer.subgroup_size != 32:
        raise ValueError("online SM90 refinement requires warp-sized logical subgroups")

    inputs = source_argument_buffers(request, region.buffers)
    outputs = source_result_buffers(request, region.buffers)
    producer_left = inputs[region.producer_left_argument].ordinal
    producer_right = inputs[region.producer_right_argument].ordinal
    consumer_operand = inputs[region.consumer_argument].ordinal
    output = outputs[region.output_result].ordinal
    if (
        len(producer.input_buffers) != 2
        or set(producer.input_buffers) != {producer_left, producer_right}
        or len(producer.output_buffers) != 3
    ):
        raise ValueError("online SM90 refinement requires Contract/Fold to return score and two sufficient states")
    if len(consumer.input_buffers) != 4 or set(consumer.input_buffers) != {
        *producer.output_buffers,
        consumer_operand,
    }:
        raise ValueError("online SM90 refinement requires Map/Contract to consume the complete sufficient-state cut")
    if consumer.output_buffers != (output,):
        raise ValueError("online SM90 refinement requires the weighted Contract to produce the public result")

    buffers = {buffer.ordinal: buffer for buffer in region.buffers}
    cut = tuple(buffers[ordinal] for ordinal in producer.output_buffers)
    if any(
        buffer.live_in
        or buffer.live_out
        or buffer.producer != producer.ordinal
        or buffer.consumers != (consumer.ordinal,)
        for buffer in cut
    ):
        raise ValueError("online SM90 refinement requires a private single-consumer sufficient-state cut")
    expected_cut_types = normalized_exp_sufficient_state_cut_types(region)
    actual_cut_types = tuple(buffer.type for buffer in cut)
    if sorted(actual_cut_types, key=_source_type_sort_key) != sorted(expected_cut_types, key=_source_type_sort_key):
        raise ValueError("online SM90 refinement requires one score and two compact Fold-state buffers")

    events = tuple(
        operation for operation in request.schedule.body().operations if operation.name == "shuttle.event_tensor"
    )
    if len(events) != 1:
        raise ValueError("online SM90 refinement requires one exact source EventTensor")
    event = events[0].attributes
    if event.get("producer_task") != producer.ordinal or event.get("consumer_task") != consumer.ordinal:
        raise ValueError("online SM90 refinement EventTensor does not bind the sufficient-state cut")


def _source_type_sort_key(type_record: ShuttleTypeRecord) -> tuple[str, str, tuple[int, ...]]:
    return type_record.kind, type_record.element_type, type_record.shape


def _sm90_reverse_schedule(
    request: GpuPhysicalArtifactRequest,
    region: PhysicalNormalizedExpContractRegion,
) -> Sm90BackwardAccumulatorSchedule:
    schedules = source_bound_schedule_tasks(request)
    owner = tuple(schedule for schedule in schedules if schedule.state_inputs and schedule.state_outputs)
    streamed = tuple(schedule for schedule in schedules if schedule.state_inputs and not schedule.state_outputs)
    if len(owner) != 1 or len(streamed) != 1:
        raise ValueError("SM90 reverse Contract/Fold lowering requires one owner and one streamed adjoint schedule")
    owner_schedule = owner[0]
    streamed_schedule = streamed[0]
    if (
        owner_schedule.workgroup_threads != _SM90_REVERSE_THREADS
        or streamed_schedule.workgroup_threads != _SM90_REVERSE_THREADS
        or owner_schedule.subgroup_size != _SM90_LOGICAL_SUBGROUP_SIZE
        or streamed_schedule.subgroup_size != _SM90_LOGICAL_SUBGROUP_SIZE
        or owner_schedule.scratch_bytes != _SM90_REVERSE_SCRATCH_BYTES
        or streamed_schedule.scratch_bytes != _SM90_REVERSE_SCRATCH_BYTES
    ):
        raise ValueError("SM90 reverse Contract/Fold logical resources do not select the shared schedule family")
    logical_owner_tile = owner_schedule.tile_shape[owner_schedule.owner_dimensions[-1]]
    logical_streamed_tile = owner_schedule.tile_shape[owner_schedule.stream_dimensions[-1]]
    if (
        streamed_schedule.tile_shape[streamed_schedule.owner_dimensions[-1]] != logical_streamed_tile
        or streamed_schedule.tile_shape[streamed_schedule.stream_dimensions[-1]] != logical_owner_tile
    ):
        raise ValueError("SM90 reverse owner and streamed schedules disagree on transposed Contract tiles")
    return Sm90BackwardAccumulatorSchedule(
        contract=select_sm90_contract_schedule(
            owner_extent=region.geometry.owner_extent,
            streamed_extent=region.geometry.streamed_extent,
            producer_feature_extent=region.geometry.producer_feature_extent,
            consumer_feature_extent=region.geometry.consumer_feature_extent,
            threads=_SM90_REVERSE_THREADS,
            relation_layout=_sm90_relation_layout(region),
        )
    )


def _domain_traversal(region: PhysicalNormalizedExpContractRegion) -> ContractDomainTraversal:
    # Exact element masking is compiled from the affine predicate. Until tile
    # pruning accepts an affine envelope, traverse the full relation so a
    # positive streamed offset cannot omit selected tiles.
    return ContractDomainTraversal.FULL


def _sm90_relation_layout(region: PhysicalNormalizedExpContractRegion) -> Sm90RelationPackedLayout:
    geometry = region.geometry
    return sm90_relation_packed_layout(
        batch_shape=geometry.batch_shape,
        shared_relation_extent=geometry.shared_relation_extent,
        relation_group_shape=geometry.relation_group_shape,
        owner_extent=geometry.owner_extent,
        streamed_extent=geometry.streamed_extent,
        producer_feature_extent=geometry.producer_feature_extent,
        consumer_feature_extent=geometry.consumer_feature_extent,
    )


def _relation_packed_views(
    region: PhysicalNormalizedExpContractRegion,
    layout: Sm90RelationPackedLayout | None,
    *,
    query_storage: torch.Tensor,
    key_storage: torch.Tensor,
    value_storage: torch.Tensor,
    output_storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout is None:
        raise ValueError("SM90 physical lowering requires a selected relation-packed layout")
    producer_maps = contract_indexing_dimensions(region.producer)
    consumer_maps = contract_indexing_dimensions(region.consumer)
    if len(producer_maps) != 3 or len(consumer_maps) != 3:
        raise ValueError("SM90 relation views require two-input Contract relations")
    geometry = region.geometry
    owner_candidates = tuple(
        index
        for index, dimensions in enumerate(producer_maps[:2])
        if geometry.owner_dimension in dimensions and geometry.streamed_dimension not in dimensions
    )
    stream_candidates = tuple(
        index
        for index, dimensions in enumerate(producer_maps[:2])
        if geometry.streamed_dimension in dimensions and geometry.owner_dimension not in dimensions
    )
    if len(owner_candidates) != 1 or len(stream_candidates) != 1:
        raise ValueError("SM90 relation views cannot identify unique producer owner and stream operands")
    value_candidates = tuple(
        index
        for index, dimensions in enumerate(consumer_maps[:2])
        if geometry.consumer_feature_dimension in dimensions and geometry.normalized_dimension in dimensions
    )
    if len(value_candidates) != 1:
        raise ValueError("SM90 relation views cannot identify the weighted Contract value operand")
    value_operand = value_candidates[0]
    (
        consumer_batch_dimensions,
        consumer_shared_relation_dimension,
        consumer_relation_group_dimensions,
        consumer_owner_dimension,
    ) = normalized_exp_consumer_dimension_roles(region)
    query = _owner_relation_view(
        query_storage,
        producer_maps[owner_candidates[0]],
        geometry,
        geometry.producer_feature_dimension,
        geometry.producer_feature_extent,
        "producer owner",
        batch_dimensions=geometry.batch_dimensions,
        shared_relation_dimension=geometry.shared_relation_dimension,
        relation_group_dimensions=geometry.relation_group_dimensions,
        owner_dimension=geometry.owner_dimension,
    )
    key = _stream_relation_view(
        key_storage,
        producer_maps[stream_candidates[0]],
        geometry,
        geometry.producer_feature_dimension,
        geometry.producer_feature_extent,
        "producer stream",
        batch_dimensions=geometry.batch_dimensions,
        shared_relation_dimension=geometry.shared_relation_dimension,
        stream_dimension=geometry.streamed_dimension,
    )
    value = _stream_relation_view(
        value_storage,
        consumer_maps[value_operand],
        geometry,
        geometry.consumer_feature_dimension,
        geometry.consumer_feature_extent,
        "weighted Contract value",
        batch_dimensions=consumer_batch_dimensions,
        shared_relation_dimension=consumer_shared_relation_dimension,
        stream_dimension=geometry.normalized_dimension,
    )
    output = _owner_relation_view(
        output_storage,
        region.public_output_dimensions,
        geometry,
        geometry.consumer_feature_dimension,
        geometry.consumer_feature_extent,
        "weighted Contract output",
        batch_dimensions=consumer_batch_dimensions,
        shared_relation_dimension=consumer_shared_relation_dimension,
        relation_group_dimensions=consumer_relation_group_dimensions,
        owner_dimension=consumer_owner_dimension,
    )
    expected = (
        layout.owner_physical_shape(layout.producer_feature_extent),
        layout.streamed_physical_shape(layout.producer_feature_extent),
        layout.streamed_physical_shape(layout.consumer_feature_extent),
        layout.owner_physical_shape(layout.consumer_feature_extent),
    )
    if tuple(tuple(tensor.shape) for tensor in (query, key, value, output)) != expected:
        raise ValueError("relation-derived source views disagree with the selected SM90 layout")
    return query, key, value, output


def _collapsed_relation_axis(
    storage: torch.Tensor,
    dimensions: tuple[int, ...],
    selected: tuple[int, ...],
    label: str,
) -> tuple[int, int]:
    if not selected:
        return 1, storage.numel()
    positions = tuple(dimensions.index(dimension) for dimension in selected if dimensions.count(dimension) == 1)
    if len(positions) != len(selected) or positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ValueError(f"{label} dimensions are not a contiguous public storage projection")
    shape = tuple(int(storage.shape[position]) for position in positions)
    strides = tuple(int(storage.stride()[position]) for position in positions)
    if any(strides[index] != strides[index + 1] * shape[index + 1] for index in range(len(shape) - 1)):
        raise ValueError(f"{label} dimensions are not compact in public storage")
    return math.prod(shape), strides[-1]


def _projected_axis(
    storage: torch.Tensor,
    dimensions: tuple[int, ...],
    dimension: int,
    label: str,
) -> tuple[int, int]:
    positions = tuple(index for index, candidate in enumerate(dimensions) if candidate == dimension)
    if len(positions) != 1:
        raise ValueError(f"{label} is not projected exactly once in public storage")
    position = positions[0]
    return int(storage.shape[position]), int(storage.stride()[position])


def _owner_relation_view(
    storage: torch.Tensor,
    dimensions: tuple[int, ...],
    geometry: NormalizedExpContractGeometry,
    feature_dimension: int,
    feature_extent_expected: int,
    label: str,
    *,
    batch_dimensions: tuple[int, ...],
    shared_relation_dimension: int,
    relation_group_dimensions: tuple[int, ...],
    owner_dimension: int,
) -> torch.Tensor:
    expected_dimensions = {
        *batch_dimensions,
        shared_relation_dimension,
        *relation_group_dimensions,
        owner_dimension,
        feature_dimension,
    }
    if set(dimensions) != expected_dimensions or len(dimensions) != len(expected_dimensions):
        raise ValueError(f"{label} relation has unsupported projected dimensions")
    batch_extent, batch_stride = _collapsed_relation_axis(storage, dimensions, batch_dimensions, f"{label} batch")
    relation_extent, relation_stride = _collapsed_relation_axis(
        storage,
        dimensions,
        (shared_relation_dimension, *relation_group_dimensions),
        f"{label} packed relation",
    )
    owner_extent, owner_stride = _projected_axis(storage, dimensions, owner_dimension, f"{label} owner dimension")
    feature_extent, feature_stride = _projected_axis(storage, dimensions, feature_dimension, f"{label} feature")
    expected = (
        geometry.batch_size,
        geometry.owner_extent,
        geometry.expanded_relation_extent,
        feature_extent_expected,
    )
    actual = (batch_extent, owner_extent, relation_extent, feature_extent)
    if actual != expected or feature_stride != 1:
        raise ValueError(f"{label} source shape/strides {actual}/{storage.stride()} disagree with {expected}")
    return storage.as_strided(actual, (batch_stride, owner_stride, relation_stride, feature_stride))


def _stream_relation_view(
    storage: torch.Tensor,
    dimensions: tuple[int, ...],
    geometry: NormalizedExpContractGeometry,
    feature_dimension: int,
    feature_extent_expected: int,
    label: str,
    *,
    batch_dimensions: tuple[int, ...],
    shared_relation_dimension: int,
    stream_dimension: int,
) -> torch.Tensor:
    expected_dimensions = {
        *batch_dimensions,
        shared_relation_dimension,
        stream_dimension,
        feature_dimension,
    }
    if set(dimensions) != expected_dimensions or len(dimensions) != len(expected_dimensions):
        raise ValueError(f"{label} relation has unsupported projected dimensions")
    batch_extent, batch_stride = _collapsed_relation_axis(storage, dimensions, batch_dimensions, f"{label} batch")
    stream_extent, stream_stride = _projected_axis(storage, dimensions, stream_dimension, f"{label} stream dimension")
    shared_extent, shared_stride = _projected_axis(
        storage, dimensions, shared_relation_dimension, f"{label} shared relation"
    )
    feature_extent, feature_stride = _projected_axis(storage, dimensions, feature_dimension, f"{label} feature")
    expected = (
        geometry.batch_size,
        geometry.streamed_extent,
        geometry.shared_relation_extent,
        feature_extent_expected,
    )
    actual = (batch_extent, stream_extent, shared_extent, feature_extent)
    if actual != expected or feature_stride != 1:
        raise ValueError(f"{label} source shape/strides {actual}/{storage.stride()} disagree with {expected}")
    return storage.as_strided(actual, (batch_stride, stream_stride, shared_stride, feature_stride))


def _empty_bf16(shape: tuple[int, ...]) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("SM90 artifact compilation requires a CUDA compiler device")
    return torch.empty(shape, dtype=torch.bfloat16, device=torch.device("cuda", torch.cuda.current_device()))


def _compile_request(
    request: GpuPhysicalArtifactRequest,
    region: PhysicalNormalizedExpContractRegion,
    schedule: Any,
    *,
    physical_event_schedule: Mapping[str, Any] | None = None,
    lowered_scalar_programs: Mapping[str, Any] | None = None,
) -> CutePipelineCompileRequest:
    geometry = region.geometry
    source_score_map = json.loads(serialize_scalar_expression(region.score_expression))
    source_finalizer = json.loads(serialize_scalar_expression(region.finalizer_expression))
    scalar_programs = {
        "source_score_map": source_score_map,
        "source_finalizer": source_finalizer,
        **(
            {"score_map": source_score_map, "finalizer": source_finalizer}
            if lowered_scalar_programs is None
            else dict(lowered_scalar_programs)
        ),
    }
    return CutePipelineCompileRequest(
        schema_version=3,
        target=request.target,
        numerical_policy="fast",
        physical_template="sm90_contract_fold_contract",
        tensors=tuple(
            {
                "ordinal": buffer.ordinal,
                "shape": buffer.type.shape,
                "element_type": buffer.type.element_type,
                "live_in": buffer.live_in,
                "live_out": buffer.live_out,
            }
            for buffer in region.buffers
        ),
        scalar_programs=scalar_programs,
        index_relations=(
            {"contract": "producer", "maps": region.producer.attributes["indexing_maps"]},
            {"contract": "consumer", "maps": region.consumer.attributes["indexing_maps"]},
            {
                "contract": "normalized_score",
                "producer_to_consumer_dimensions": region.producer_to_consumer_dimensions,
            },
            {
                "contract": "domain",
                "predicate": (
                    None
                    if region.domain_predicate is None
                    else {
                        "expression": region.domain_predicate.predicate.canonical_record,
                        "source_operations": tuple(
                            _operation_identity(operation) for operation in region.domain_predicate.source_operations
                        ),
                        "select": _operation_identity(region.domain_predicate.relation),
                    }
                ),
            },
        ),
        contraction_dimensions={
            "batch_shape": geometry.batch_shape,
            "shared_relation_extent": geometry.shared_relation_extent,
            "relation_group_shape": geometry.relation_group_shape,
            "owner_extent": geometry.owner_extent,
            "streamed_extent": geometry.streamed_extent,
            "producer_feature_extent": geometry.producer_feature_extent,
            "consumer_feature_extent": geometry.consumer_feature_extent,
            "producer_feature_dimension": geometry.producer_feature_dimension,
            "normalized_dimension": geometry.normalized_dimension,
            "consumer_feature_dimension": geometry.consumer_feature_dimension,
        },
        schedule={
            **asdict(schedule),
            "family": schedule.contract.family,
            "source_event_schedule": _operation_identity(request.schedule),
            "physical_event_schedule": None if physical_event_schedule is None else dict(physical_event_schedule),
        },
    )


def _combined_internal_event_provenance(*regions: Mapping[str, Any]) -> dict[str, Any]:
    return combine_physical_event_schedules(*regions)


def _bind_internal_event_provenance(
    artifact: CutePipelineArtifact,
    provenance: Mapping[str, Any],
) -> CutePipelineArtifact:
    if artifact.launch_recipe is None:
        raise ValueError("physical Event provenance requires a launch recipe")
    return replace(
        artifact,
        launch_recipe=bind_physical_event_provenance(artifact.launch_recipe, provenance),
    )


def _operation_identity(operation: ShuttleOperationRecord) -> dict[str, Any]:
    return {
        "name": operation.name,
        "attributes": operation.attributes,
        "operands": operation.operands,
        "results": tuple(
            {"id": result.id, "shape": result.type.shape, "element_type": result.type.element_type}
            for result in operation.results
        ),
        "regions": tuple(
            tuple(
                {
                    "arguments": tuple(argument.id for argument in block.arguments),
                    "operations": tuple(_operation_identity(nested) for nested in block.operations),
                }
                for block in blocks
            )
            for blocks in operation.regions
        ),
    }


compile_h100_normalized_exp_contract_artifact = native_gpu_physical_artifact_provider(
    compile_h100_normalized_exp_contract_request
)
