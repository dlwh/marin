# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

import pytest

from shuttle.ir import DType
from tile_lifetime.cute_pipeline_artifact import CutePipelineArtifact, CutePipelineCompileRequest
from tile_lifetime.event_buffering import EventRealizationKind, derive_bounded_buffer_plan
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    TaskDependence,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    lower_event_tensor_plan,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventDeviceOperationKind,
    InternalEventMechanismKind,
    internal_event_device_legalizations,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    derive_normalized_exp_contract_pullback,
    derive_reverse_ownership_plan,
    normalized_exp_contract_pullback_program,
)
from tile_lifetime.ordered_accumulation_event_schedule import ordered_accumulation_event_schedule_record
from tile_lifetime.physical_task_contract import PhysicalTarget, TaskOperationKind
from tile_lifetime.streaming_attention import (
    ScoreMapSpec,
    StreamingTileSchedule,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.streaming_attention_executable import (
    ReverseAlgebraAcceptance,
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
    normalized_exp_executable_requests,
)
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    canonical_streaming_physical_event_schedule_record,
    derive_streaming_bounded_pipeline_allocation,
    derive_streaming_physical_event_schedule,
    streaming_physical_event_schedule_record,
    verify_streaming_event_backend_parameters,
)
from tile_lifetime.streaming_pullback_event_schedule import (
    StreamingIntermediateStorage,
    StreamingPullbackInputDescriptor,
    derive_normalized_exp_pullback_task_dataflow,
    derive_streaming_pullback_input_event_schedule,
    streaming_pullback_input_event_schedule_record,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
)


def _descriptor(*, resident_tile: int = 128, streamed_tile: int = 64, fold_extent: int = 512, pipeline_depth: int = 3):
    return StreamingContractFoldDescriptor(
        first_contract_name="first_contract",
        fold_update_name="state_update",
        second_contract_name="second_contract",
        finalize_name="finalize",
        fold_extent=fold_extent,
        resident_tile_size=resident_tile,
        streamed_tile_size=streamed_tile,
        pipeline_depth=pipeline_depth,
        resident_reduction_dimension=128,
        streamed_reduction_dimension=128,
        output_dimension=128,
        element_bytes=2,
    )


def _pullback_descriptor(
    *,
    owner_extent: int = 512,
    lhs_depth: int = 2,
    cotangent_depth: int = 2,
    score_depth: int = 2,
    weight_storage: StreamingIntermediateStorage = StreamingIntermediateStorage.REGISTERS,
    lhs_gradient_warpgroups: int = 2,
    deterministic_accumulation: bool = False,
    score_tile_coordinates: tuple[tuple[int, int], ...] | None = None,
):
    owner_tile_size = 64
    streamed_extent = owner_extent
    streamed_tile_size = 128
    owner_partitions = (owner_extent + owner_tile_size - 1) // owner_tile_size
    streamed_partitions = (streamed_extent + streamed_tile_size - 1) // streamed_tile_size
    if score_tile_coordinates is None:
        score_tile_coordinates = tuple(
            (owner_partition, streamed_partition)
            for owner_partition in range(owner_partitions)
            for streamed_partition in range(streamed_partitions)
        )
    return StreamingPullbackInputDescriptor(
        producer_lhs_stage_name="producer_lhs_state_stage",
        output_cotangent_stage_name="output_cotangent_state_stage",
        score_contract_name="score_contract",
        output_cotangent_contract_name="output_cotangent_contract",
        normalized_weight_name="normalized_weight",
        pullback_map_name="normalized_exp_pullback",
        score_vjp_name="score_map_vjp",
        producer_lhs_gradient_name="producer_lhs_gradient_contract",
        lhs_gradient_handoff_name="producer_lhs_gradient_handoff",
        producer_rhs_gradient_name="producer_rhs_gradient_contract",
        consumer_rhs_gradient_name="consumer_rhs_gradient_contract",
        producer_rhs_epilogue_stage_name="producer_rhs_epilogue_stage",
        producer_rhs_epilogue_transfer_name="producer_rhs_epilogue_transfer",
        consumer_rhs_epilogue_stage_name="consumer_rhs_epilogue_stage",
        consumer_rhs_epilogue_transfer_name="consumer_rhs_epilogue_transfer",
        owner_extent=owner_extent,
        owner_tile_size=owner_tile_size,
        streamed_extent=streamed_extent,
        streamed_tile_size=streamed_tile_size,
        score_tile_coordinates=score_tile_coordinates,
        relation_group_extent=4,
        deterministic_accumulation=deterministic_accumulation,
        producer_lhs_pipeline_depth=lhs_depth,
        output_cotangent_pipeline_depth=cotangent_depth,
        score_workspace_depth=score_depth,
        normalized_weight_storage=weight_storage,
        matrix_warpgroups=2,
        lhs_gradient_warpgroups=lhs_gradient_warpgroups,
        producer_lhs_transaction_bytes=64 * 128 * 2 + 64 * 4,
        output_cotangent_transaction_bytes=64 * 128 * 2 + 64 * 4,
        producer_rhs_resident_transaction_bytes=128 * 128 * 2,
        consumer_rhs_resident_transaction_bytes=128 * 128 * 2,
    )


def _logical_pullback_dataflow(
    score_map: ScoreMapSpec,
    *,
    traversal: ContractDomainTraversal = ContractDomainTraversal.FULL,
):
    forward = derive_streaming_attention(
        build_attention_tensor_program(
            batch_size=2,
            query_length=7,
            key_length=9,
            query_heads=4,
            key_value_heads=2,
            key_dimension=8,
            value_dimension=8,
            score_map=score_map,
            input_dtype=DType.BF16,
        ),
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )
    pullback = derive_normalized_exp_contract_pullback(forward.region)
    ownership = derive_reverse_ownership_plan(pullback, traversal=traversal)
    return (
        pullback,
        ownership,
        derive_normalized_exp_pullback_task_dataflow(
            pullback,
            ownership,
            owner_tile_size=3,
            streamed_tile_size=4,
        ),
    )


def test_generic_score_vjps_share_one_prephysical_reverse_task_and_event_graph() -> None:
    identity_map = scaled_score_map(0.5)
    softcap_map = apply_tanh_softcap(identity_map, 1.7)
    affine_map = ScoreMapSpec(
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                identity_map.expression,
                scalar_constant(1.25),
            ),
            scalar_constant(0.25),
        )
    )
    variants = tuple(_logical_pullback_dataflow(score_map) for score_map in (identity_map, softcap_map, affine_map))
    pullbacks = tuple(variant[0] for variant in variants)
    ownerships = tuple(variant[1] for variant in variants)
    dataflows = tuple(variant[2] for variant in variants)

    assert ownerships[0] == ownerships[1] == ownerships[2]
    assert len({pullback.score_map_vjp.expression for pullback in pullbacks}) == 3
    assert dataflows[0].program.task_families == dataflows[1].program.task_families == dataflows[2].program.task_families
    assert len(dataflows[1].program.event_plans) == len(dataflows[0].program.event_plans) + 1
    assert len(dataflows[2].program.event_plans) == len(dataflows[0].program.event_plans)
    assert any(
        plan.scheduled_dependence.source == dataflows[1].score_contract
        and plan.scheduled_dependence.target == dataflows[1].score_vjp
        for plan in dataflows[1].program.event_plans
    )

    dataflow = dataflows[0]
    producer_right_finalize = dataflow.terminal_gradient_finalizers[1]
    producer_right_event = next(
        plan for plan in dataflow.program.event_plans if plan.trigger_relation.target == producer_right_finalize
    )
    assert producer_right_finalize.axes[-1].name == "key_value_head"
    assert set(producer_right_event.initial_count.as_mapping().values()) == {6}
    for plan, binding in zip(dataflow.program.event_plans, dataflow.storage_bindings, strict=True):
        runtime = event_tensor_runtime_inputs(plan, storage_binding=binding)
        assert runtime.event_generations == (0,) * len(plan.domain.coordinates)
        imperative = lower_event_tensor_plan(plan, scheduling_mode=EventSchedulingMode.STATIC)
        assert {operation.kind.value for operation in imperative} >= {"initialize", "wait", "notify"}


def test_prephysical_reverse_tasks_apply_domain_traversal_before_grouped_gradient_relations() -> None:
    _, _, full = _logical_pullback_dataflow(scaled_score_map(0.5))
    _, _, triangular = _logical_pullback_dataflow(
        scaled_score_map(0.5),
        traversal=ContractDomainTraversal.LOWER_TRIANGULAR,
    )

    assert len(full.score_tile_coordinates) == 72
    assert len(triangular.score_tile_coordinates) == 40
    assert all(
        key_tile * 4 < min((query_tile + 1) * 3, 7) for _, _, query_tile, key_tile in triangular.score_tile_coordinates
    )
    producer_right_event = next(
        plan
        for plan in triangular.program.event_plans
        if plan.trigger_relation.target == triangular.terminal_gradient_finalizers[1]
    )
    assert set(producer_right_event.initial_count.as_mapping().values()) == {0, 4, 6}


def test_generic_scalar_maps_share_tasks_and_schedules_but_change_executable_identity() -> None:
    identity = scaled_score_map(0.5)
    softcap = apply_tanh_softcap(identity, 1.7)
    affine = ScoreMapSpec(
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(ScalarExpressionKind.MULTIPLY, identity.expression, scalar_constant(1.25)),
            scalar_constant(0.25),
        )
    )
    requests = []
    for score_map in (identity, softcap, affine):
        forward = derive_streaming_attention(
            replace(
                build_attention_tensor_program(
                    batch_size=1,
                    query_length=7,
                    key_length=9,
                    query_heads=4,
                    key_value_heads=2,
                    key_dimension=128,
                    value_dimension=128,
                    score_map=score_map,
                    input_dtype=DType.BF16,
                ),
                numerical_policy=NumericalPolicy.FAST,
            ),
            schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
        )
        reverse = normalized_exp_contract_pullback_program(
            forward,
            schedule=object(),
            traversal=ContractDomainTraversal.FULL,
        )
        requests.append(
            normalized_exp_executable_requests(
                forward,
                reverse,
                owner_tile_size=3,
                streamed_tile_size=4,
                target=PhysicalTarget.SM90,
            )
        )

    assert len({item.forward.fingerprint for item in requests}) == 3
    assert len({item.reverse_candidate.fingerprint for item in requests}) == 3
    assert len({item.forward.task_plan.fingerprint for item in requests}) == 1
    assert (
        len({tuple(schedule.family.fingerprint for schedule in item.reverse_candidate.schedules) for item in requests})
        == 1
    )
    assert all(item.reverse_acceptance is ReverseAlgebraAcceptance.CANDIDATE_AWAITING_JAXPR_IMPORT for item in requests)
    attachments = tuple(attach_sm90_contract_fold_schedule(item.forward) for item in requests)
    assert len({attachment.physical_schedule.family.fingerprint for attachment in attachments}) == 1
    assert all(
        attachment.physical_schedule.parameters == attachments[0].physical_schedule.parameters
        for attachment in attachments[1:]
    )
    reverse_attachments = tuple(attach_sm90_contract_fold_schedule(item.reverse_candidate) for item in requests)
    assert len({attachment.physical_schedule.family.fingerprint for attachment in reverse_attachments}) == 1
    for attachment in reverse_attachments:
        scheduled = tuple(task_id for schedule in attachment.request.schedules for task_id in schedule.task_ids)
        assert tuple(sorted(scheduled)) == tuple(range(len(attachment.request.task_plan.tasks)))
        family_ids = {task.family: task.id for task in attachment.request.task_plan.tasks}
        internal_names = {
            plan.name
            for plan in attachment.request.event_plans
            if {
                family_ids[plan.scheduled_dependence.source],
                family_ids[plan.scheduled_dependence.target],
            }
            <= set(attachment.fused_task_ids)
        }
        assert {entry.plan_name for entry in attachment.request.event_realizations.audit.erased} == internal_names
        assert {entry.plan_name for entry in attachment.request.event_realizations.audit.physical} == {
            plan.name for plan in attachment.request.event_plans
        } - internal_names
    for item in requests:
        proof = item.reverse_candidate.event_realizations
        assert not proof.erasure_witnesses
        assert not proof.audit.erased
        assert len(proof.audit.physical) == len(item.reverse_candidate.event_plans)
        assert all(
            plan.required_dependence.visibility.scope is EventMemoryScope.DEVICE
            for plan in item.reverse_candidate.event_plans
        )
        assert all(
            dependence.visibility.scope is EventMemoryScope.DEVICE
            for dependence in item.reverse_task_dataflow.program.dependences
        )


def test_sm90_attachment_schedules_an_extra_finalizer_as_a_generic_map_task() -> None:
    program = build_attention_tensor_program(
        batch_size=1,
        query_length=7,
        key_length=9,
        query_heads=4,
        key_value_heads=2,
        key_dimension=128,
        value_dimension=128,
        score_map=scaled_score_map(0.5),
        input_dtype=DType.BF16,
    )
    output = program.outputs[0]
    affine_output = ProgramValue("attention.affine_output", output.axes, output.dtype)
    affine_finalizer = MapPrimitive(
        "affine output Map",
        (output,),
        affine_output,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(output.name),
            scalar_constant(0.25),
        ),
    )
    perturbed = TensorProgram(
        program.inputs,
        (*program.operations, affine_finalizer),
        (affine_output,),
        numerical_policy=NumericalPolicy.FAST,
    )
    perturbed_request = jax_tensor_program_executable_request(perturbed, tile_size=4, target=PhysicalTarget.SM90)

    attachment = attach_sm90_contract_fold_schedule(perturbed_request)

    assert attachment.fused_task_ids == (0,)
    assert len(attachment.request.task_plan.tasks) == 2
    assert attachment.request.task_plan.tasks[1].operation_kinds == (TaskOperationKind.MAP,)
    assert [schedule.family.name for schedule in attachment.request.schedules] == [
        "sm90_contract_fold_map_pipeline_v1",
        "map_task",
    ]
    assert len(attachment.request.event_plans) == 1


def test_sm90_attachment_does_not_absorb_an_arbitrary_downstream_contract() -> None:
    program = build_attention_tensor_program(
        batch_size=1,
        query_length=7,
        key_length=9,
        query_heads=4,
        key_value_heads=2,
        key_dimension=128,
        value_dimension=128,
        score_map=scaled_score_map(0.5),
        input_dtype=DType.BF16,
    )
    attention_output = program.outputs[0]
    value_feature = attention_output.axes[-1]
    projected_feature = TensorAxis(7, 64, "projected_feature")
    projection = ProgramValue("projection", (value_feature, projected_feature), DType.BF16)
    projected = ProgramValue(
        "projected",
        (*attention_output.axes[:-1], projected_feature),
        DType.BF16,
    )
    downstream = ContractPrimitive(
        "downstream Contract",
        (attention_output, projection),
        projected,
        (value_feature,),
        DType.FP32,
    )
    composed = TensorProgram(
        (*program.inputs, projection),
        (*program.operations, downstream),
        (projected,),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(composed, tile_size=4, target=PhysicalTarget.SM90)

    attachment = attach_sm90_contract_fold_schedule(request)

    assert attachment.fused_task_ids == (0,)
    assert tuple(task_id for schedule in attachment.request.schedules for task_id in schedule.task_ids) == (0, 1)
    assert not attachment.request.event_realizations.audit.erased
    assert tuple(entry.plan_name for entry in attachment.request.event_realizations.audit.physical) == ("event.0",)


def test_canonical_streaming_event_record_alpha_renames_source_task_names() -> None:
    first = derive_streaming_physical_event_schedule(_descriptor())
    renamed = derive_streaming_physical_event_schedule(
        replace(
            _descriptor(),
            first_contract_name="frontend.contract.17",
            fold_update_name="frontend.map.23",
            second_contract_name="frontend.contract.31",
            finalize_name="frontend.map.37",
        )
    )

    assert canonical_streaming_physical_event_schedule_record(first) == (
        canonical_streaming_physical_event_schedule_record(renamed)
    )


def test_streaming_event_schedule_derives_distinct_input_lifetimes() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor())

    assert schedule.resident_input_pipeline.required_barrier_slots == 1
    assert schedule.resident_input_physical_barrier_slots == 2
    assert schedule.first_streamed_input_physical_barrier_slots == 6
    assert schedule.second_streamed_input_physical_barrier_slots == 6
    assert schedule.dataflow.partition_count == 8
    assert schedule.workers.matrix_warpgroups == 2
    assert schedule.workers.matrix_warps == 8
    assert schedule.workers.cta_threads == 384
    assert schedule.workers.scheduler_arrival_threads == 256

    resident_last_consumers = dict(schedule.dataflow.resident_input_buffer.last_consumers)
    first_streamed_last_consumers = dict(schedule.dataflow.first_streamed_input_buffer.last_consumers)
    second_streamed_last_consumers = dict(schedule.dataflow.second_streamed_input_buffer.last_consumers)
    assert {consumer.family for consumer in resident_last_consumers.values()} == {schedule.dataflow.first_contract.name}
    assert {consumer.family for consumer in first_streamed_last_consumers.values()} == {
        schedule.dataflow.first_contract.name
    }
    assert {consumer.family for consumer in second_streamed_last_consumers.values()} == {
        schedule.dataflow.second_contract.name
    }
    assert resident_last_consumers[(0,)].coordinate == (0, 7)

    physical_mechanisms = {entry.mechanism for entry in schedule.audit.physical}
    assert physical_mechanisms == {
        "resident input pipeline full barrier",
        "first streamed input pipeline full barrier",
        "second streamed input pipeline full barrier",
        "first streamed input pipeline empty barrier with phased slot generation",
        "second streamed input pipeline empty barrier with phased slot generation",
        "pairwise matrix-warpgroup scheduler barrier",
    }


def test_pipeline_depth_mutation_regenerates_buffer_events() -> None:
    depth_two = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=2))
    depth_three = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=3))

    assert depth_two.first_streamed_input_stages == depth_two.second_streamed_input_stages == 2
    assert depth_three.first_streamed_input_stages == depth_three.second_streamed_input_stages == 3
    assert depth_two.first_streamed_input_physical_barrier_slots == 4
    assert depth_three.first_streamed_input_physical_barrier_slots == 6
    assert depth_two.dataflow.first_streamed_input_buffer.slots != depth_three.dataflow.first_streamed_input_buffer.slots
    assert (
        depth_two.dataflow.second_streamed_input_buffer.generations
        != depth_three.dataflow.second_streamed_input_buffer.generations
    )
    assert depth_two.fingerprint != depth_three.fingerprint

    depth_two_pipeline = depth_two.first_streamed_input_pipeline
    depth_three_pipeline = depth_three.first_streamed_input_pipeline
    assert depth_two_pipeline.capacity == 2
    assert depth_three_pipeline.capacity == 3
    assert depth_two_pipeline.slots[:6] == (0, 1, 0, 1, 0, 1)
    assert depth_three_pipeline.slots[:6] == (0, 1, 2, 0, 1, 2)
    assert depth_two_pipeline.generations[:6] == (0, 0, 1, 1, 2, 2)
    assert depth_three_pipeline.generations[:6] == (0, 0, 0, 1, 1, 1)
    assert depth_two_pipeline.physical_barrier_slots == 4
    assert depth_three_pipeline.physical_barrier_slots == 6


def test_physical_event_provenance_preserves_exact_events_and_allocation_identity() -> None:
    depth_two = derive_streaming_physical_event_schedule(
        _descriptor(resident_tile=64, streamed_tile=128, fold_extent=512, pipeline_depth=2)
    )
    depth_three = derive_streaming_physical_event_schedule(
        _descriptor(resident_tile=64, streamed_tile=128, fold_extent=512, pipeline_depth=3)
    )
    depth_two_record = streaming_physical_event_schedule_record(depth_two)
    depth_three_record = streaming_physical_event_schedule_record(depth_three)
    assert depth_two_record["schema_version"] == depth_three_record["schema_version"] == 5
    depth_two_allocations = {allocation["allocation_id"]: allocation for allocation in depth_two_record["allocations"]}
    assert tuple(realization["event"]["name"] for realization in depth_two_record["realizations"]) == tuple(
        realization["plan"] for realization in depth_two_record["realizations"]
    )
    assert len(depth_two_record["realizations"]) == len(depth_two.dataflow.program.event_plans)

    resident = depth_two_allocations["resident_input_pipeline"]
    assert len(resident["covered_plans"]) == 1
    assert resident["mechanism_coordinates"] == ((0,),)
    assert resident["initial_counts"] == (1,)
    assert len(resident["notify_relations"]) == 1
    assert len(resident["wait_relations"]) == 4

    streamed = depth_two_allocations["first_streamed_input_pipeline"]
    assert streamed["scope"] == "cta"
    assert streamed["capacity"] == 2
    assert streamed["slots"] == (0, 1, 0, 1)
    assert streamed["generations"] == (0, 0, 1, 1)
    assert streamed["generation_count"] == 2
    assert streamed["mechanism_coordinates"] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert streamed["transaction_bytes_by_coordinate"] == (128 * 128 * 2,) * 4
    assert len(streamed["covered_plans"]) == 2
    full_event = next(
        realization["event"]
        for realization in depth_two_record["realizations"]
        if realization["plan"] == streamed["covered_plans"][0]
    )
    empty_event = next(
        realization["event"]
        for realization in depth_two_record["realizations"]
        if realization["plan"] == streamed["covered_plans"][1]
    )
    assert full_event["required_dependence"]["relation"]["pairs"] == (
        {"source": (0, 0), "target": (0, 0)},
        {"source": (0, 1), "target": (0, 1)},
        {"source": (0, 2), "target": (0, 2)},
        {"source": (0, 3), "target": (0, 3)},
    )
    assert empty_event["required_dependence"]["relation"]["pairs"] == (
        {"source": (0, 0), "target": (0, 2)},
        {"source": (0, 1), "target": (0, 3)},
    )
    assert full_event["visibility"] == {
        "scope": "cta",
        "release_on_notify": True,
        "acquire_before_consumer": True,
    }

    source_events = {
        "realizations": (
            {
                "kind": "algebraic_refinement",
                "mechanism": None,
            },
        )
    }

    def request(physical_event_schedule: dict[str, Any]) -> CutePipelineCompileRequest:
        return CutePipelineCompileRequest(
            schema_version=3,
            target="sm_90a",
            numerical_policy="fast",
            physical_template="contract_fold_contract",
            tensors=(),
            scalar_programs={},
            index_relations=(),
            contraction_dimensions={},
            schedule={
                "source_event_schedule": source_events,
                "physical_event_schedule": physical_event_schedule,
            },
        )

    depth_two_request = request(depth_two_record)
    depth_three_request = request(depth_three_record)
    assert depth_two_request.schedule["source_event_schedule"] == depth_three_request.schedule["source_event_schedule"]
    assert depth_two_request.digest != depth_three_request.digest

    artifact = CutePipelineArtifact(
        request=depth_two_request,
        host_object=b"host-object",
        device_image=b"device-image",
        kernel_names=("streaming_contract_fold",),
        host_entry="streaming_contract_fold_entry",
        compiler_ir="module {}",
        toolchain={"compiler": "test"},
        source_digests={"emitter": "0" * 64},
    )
    manifest_schedule = artifact.manifest["request"]["schedule"]
    assert manifest_schedule["source_event_schedule"]["realizations"] == [
        {"kind": "algebraic_refinement", "mechanism": None}
    ]
    manifest_streamed = manifest_schedule["physical_event_schedule"]["allocations"][1]
    assert manifest_streamed["allocation_id"] == "first_streamed_input_pipeline"
    assert manifest_streamed["slots"] == [0, 1, 0, 1]


def test_streaming_allocations_expand_to_exact_imperative_device_operations() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=2))
    legalizations = internal_event_device_legalizations(schedule.allocations)
    by_allocation = {legalization.allocation_id: legalization for legalization in legalizations}

    streamed = by_allocation["first_streamed_input_pipeline"]
    notifies = tuple(
        operation for operation in streamed.operations if operation.kind is InternalEventDeviceOperationKind.NOTIFY
    )
    waits = tuple(
        operation for operation in streamed.operations if operation.kind is InternalEventDeviceOperationKind.WAIT
    )

    assert all(operation.mechanism_kind is InternalEventMechanismKind.CTA_TRANSACTION_MBARRIER for operation in notifies)
    assert all(operation.transaction_bytes == schedule.first_streamed_input_transaction_bytes for operation in notifies)
    assert all(operation.release_acquire for operation in (*notifies, *waits))
    assert tuple(operation.ordinal for operation in streamed.operations) == tuple(range(len(streamed.operations)))
    assert all(operation.transaction_bytes is None for operation in waits)
    assert all(operation.operation_count == operation.initial_count for operation in waits)


def test_streaming_pipeline_allocation_allows_full_only_without_slot_reuse() -> None:
    schedule = derive_streaming_physical_event_schedule(
        _descriptor(resident_tile=64, streamed_tile=128, fold_extent=256, pipeline_depth=2)
    )

    assert schedule.first_streamed_input_pipeline.empty_event is None
    assert schedule.second_streamed_input_pipeline.empty_event is None
    assert schedule.first_streamed_input_pipeline.capacity == 2
    assert schedule.first_streamed_input_pipeline.slots == (0, 1)
    assert schedule.first_streamed_input_pipeline.generations == (0, 0)


def test_streaming_pipeline_allocation_pairs_real_wrap_reuse() -> None:
    schedule = derive_streaming_physical_event_schedule(
        _descriptor(resident_tile=64, streamed_tile=128, fold_extent=512, pipeline_depth=2)
    )

    allocation = schedule.first_streamed_input_pipeline
    assert allocation.empty_event is not None
    assert allocation.capacity == 2
    assert allocation.slots == (0, 1, 0, 1)
    assert allocation.generations == (0, 0, 1, 1)


def test_streaming_pipeline_reuse_waits_for_last_of_multiple_consumers() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=2))
    dataflow = schedule.dataflow
    first_use = dataflow.first_streamed_input_buffer.uses[0]
    later_use = TaskRelation.from_pairs(
        dataflow.first_streamed_input_stage,
        dataflow.fold_update,
        tuple((coordinate, coordinate) for coordinate in dataflow.first_streamed_input_stage.coordinates),
    )
    visibility = next(
        dependence.visibility for dependence in dataflow.program.dependences if dependence.relation == first_use
    )
    later_dependence = TaskDependence(later_use, visibility)
    later_event = derive_event_tensor_plan(
        later_dependence,
        name="event_streamed_input_to_later_consumer",
        generation_policy=EventGenerationPolicy.PER_INVOCATION,
    )
    program_without_reuse = EventDataflowProgram(
        dataflow.program.task_families,
        (*dataflow.program.dependences, later_dependence),
        (*dataflow.program.event_plans, later_event),
    )
    original = dataflow.first_streamed_input_buffer
    slot_for = dict(zip(original.producer.coordinates, original.slots, strict=True))
    generation_for = dict(zip(original.producer.coordinates, original.generations, strict=True))
    buffer = derive_bounded_buffer_plan(
        name=original.name,
        program=program_without_reuse,
        producer=original.producer,
        uses=(first_use, later_use),
        capacity=original.capacity,
        allocation_owner_for=dict(zip(original.producer.coordinates, original.allocation_owners, strict=True)),
        slot_for=slot_for,
        generation_for=generation_for,
        visibility=later_dependence.visibility,
    )
    reuse_events = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event_later_consumer_releases_{index}",
            generation_policy=EventGenerationPolicy.PHASED,
        )
        for index, dependence in enumerate(buffer.reuse_dependences)
    )
    program = EventDataflowProgram(
        program_without_reuse.task_families,
        (*program_without_reuse.dependences, *buffer.reuse_dependences),
        (*program_without_reuse.event_plans, *reuse_events),
    )

    allocation = derive_streaming_bounded_pipeline_allocation(
        name=original.name,
        program=program,
        buffer=buffer,
        consumer=dataflow.first_contract,
        transaction_bytes=schedule.first_streamed_input_transaction_bytes,
    )

    assert allocation.full_event.required_dependence.relation == first_use
    assert allocation.empty_event is not None
    assert allocation.empty_event.required_dependence.relation.source == dataflow.fold_update


def test_streaming_pipeline_records_first_generation_resident_operand_transaction() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=2))
    dataflow = schedule.dataflow
    original = dataflow.first_streamed_input_buffer
    resident_bytes = 128 * 128 * 2
    extra = {
        coordinate: resident_bytes if index == 0 else 0 for index, coordinate in enumerate(original.producer.coordinates)
    }

    allocation = derive_streaming_bounded_pipeline_allocation(
        name=original.name,
        program=dataflow.program,
        buffer=original,
        consumer=dataflow.first_contract,
        transaction_bytes=schedule.first_streamed_input_transaction_bytes,
        extra_transaction_bytes=extra,
    )

    assert allocation.extra_transaction_bytes == (resident_bytes, 0, 0, 0, 0, 0, 0, 0)
    assert allocation.transaction_bytes_by_coordinate == (
        schedule.first_streamed_input_transaction_bytes + resident_bytes,
        *(schedule.first_streamed_input_transaction_bytes for _ in range(7)),
    )


def test_streaming_pullback_inputs_derive_last_consumer_reuse_and_first_tile_transactions() -> None:
    schedule = derive_streaming_pullback_input_event_schedule(_pullback_descriptor())
    dataflow = schedule.dataflow
    record = streaming_pullback_input_event_schedule_record(schedule)
    assert record["schema_version"] == 6
    allocations = {allocation["allocation_id"]: allocation for allocation in record["allocations"]}

    assert dataflow.partition_count == 8
    assert {consumer.family for _, consumer in dataflow.producer_lhs_buffer.last_consumers} == {
        dataflow.producer_rhs_gradient.name
    }
    assert {consumer.family for _, consumer in dataflow.output_cotangent_buffer.last_consumers} == {
        dataflow.consumer_rhs_gradient.name
    }
    assert schedule.producer_lhs_pipeline.transaction_bytes_by_coordinate == (
        64 * 128 * 2 + 64 * 4 + 128 * 128 * 2,
        *(64 * 128 * 2 + 64 * 4 for _ in range(7)),
    )
    assert schedule.output_cotangent_pipeline.transaction_bytes_by_coordinate == (
        64 * 128 * 2 + 64 * 4 + 128 * 128 * 2,
        *(64 * 128 * 2 + 64 * 4 for _ in range(7)),
    )
    assert allocations["producer_lhs_state_pipeline"]["mechanism_kind"] == "cta_transaction_mbarrier"
    assert allocations["score_cotangent_workspace"]["mechanism_kind"] == "cta_mbarrier"
    scratch = allocations["rhs_gradient_epilogue_scratch"]
    async_queue = allocations["rhs_gradient_epilogue_async_transfer_queue"]
    assert scratch["mechanism_kind"] == "cta_mbarrier"
    assert async_queue["mechanism_kind"] == "cta_async_completion_queue"
    assert scratch["scope"] == async_queue["scope"] == "cta"
    assert scratch["capacity"] == async_queue["capacity"] == 1
    assert scratch["transaction_bytes_by_coordinate"] is async_queue["transaction_bytes_by_coordinate"] is None
    physical_mechanisms = tuple(entry.mechanism for entry in schedule.audit.physical)
    assert physical_mechanisms.count("TMA pipeline full barrier") == 2
    assert physical_mechanisms.count("TMA pipeline empty barrier with phased slot generation") == 2
    assert physical_mechanisms.count("cross-warpgroup score-cotangent full barrier") == 2
    assert any(
        entry.mechanism == "register operand source" and entry.kind is EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT
        for entry in schedule.audit.entries
    )


def test_streaming_pullback_pipeline_depth_changes_slots_without_changing_algebraic_tasks() -> None:
    depth_two = derive_streaming_pullback_input_event_schedule(_pullback_descriptor(lhs_depth=2, cotangent_depth=2))
    depth_three = derive_streaming_pullback_input_event_schedule(_pullback_descriptor(lhs_depth=3, cotangent_depth=3))

    assert depth_two.dataflow.program.task_families == depth_three.dataflow.program.task_families
    assert depth_two.producer_lhs_pipeline.slots != depth_three.producer_lhs_pipeline.slots
    assert depth_two.output_cotangent_pipeline.generations != depth_three.output_cotangent_pipeline.generations
    assert depth_two.fingerprint != depth_three.fingerprint


def test_streaming_pullback_score_workspace_erases_register_and_double_buffered_events() -> None:
    register_schedule = derive_streaming_pullback_input_event_schedule(_pullback_descriptor())
    shared_single_stage = derive_streaming_pullback_input_event_schedule(
        _pullback_descriptor(score_depth=1, weight_storage=StreamingIntermediateStorage.SHARED)
    )

    assert register_schedule.dataflow.normalized_weight_buffer is None
    assert any(
        entry.mechanism == "register operand source" and entry.kind is EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT
        for entry in register_schedule.audit.entries
    )
    assert any(
        entry.mechanism == "bounded generations plus ordered WGMMA completion"
        and entry.kind is EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT
        for entry in register_schedule.audit.entries
    )
    assert shared_single_stage.dataflow.normalized_weight_buffer is not None
    assert (
        sum(
            entry.mechanism == "cross-warpgroup score-workspace reuse barrier"
            for entry in shared_single_stage.audit.physical
        )
        == 2
    )
    assert any(
        entry.mechanism == "cross-warpgroup normalized-weight full barrier"
        for entry in shared_single_stage.audit.physical
    )


def test_streaming_pullback_lhs_gradient_handoff_tracks_each_participating_warpgroup() -> None:
    two_warpgroups = derive_streaming_pullback_input_event_schedule(_pullback_descriptor())
    one_warpgroup = derive_streaming_pullback_input_event_schedule(_pullback_descriptor(lhs_gradient_warpgroups=1))

    two_buffer = two_warpgroups.dataflow.lhs_gradient_handoff_buffer
    one_buffer = one_warpgroup.dataflow.lhs_gradient_handoff_buffer
    assert two_buffer.capacity == 2
    assert two_buffer.slots[:6] == (0, 1, 0, 1, 0, 1)
    assert two_buffer.generations[:6] == (0, 0, 1, 1, 2, 2)
    assert one_buffer.capacity == 1
    assert one_buffer.slots[:3] == (0, 0, 0)
    assert one_buffer.generations[:3] == (0, 1, 2)
    for schedule in (two_warpgroups, one_warpgroup):
        mechanisms = tuple(entry.mechanism for entry in schedule.audit.physical)
        assert mechanisms.count("lhs-gradient handoff full barrier") == 1
        assert mechanisms.count("lhs-gradient handoff empty barrier") == 1


def test_streaming_pullback_rhs_epilogue_factors_partition_accumulators_before_shared_transfer() -> None:
    schedule = derive_streaming_pullback_input_event_schedule(_pullback_descriptor())
    dataflow = schedule.dataflow
    terminal_epilogue_plans = tuple(
        plan
        for plan in dataflow.program.event_plans
        if (
            plan.required_dependence.relation.source == dataflow.producer_rhs_gradient
            and plan.required_dependence.relation.target == dataflow.producer_rhs_epilogue_stage
        )
        or (
            plan.required_dependence.relation.source == dataflow.consumer_rhs_gradient
            and plan.required_dependence.relation.target == dataflow.consumer_rhs_epilogue_stage
        )
    )

    assert len(terminal_epilogue_plans) == 2
    assert all(
        {count.value for count in plan.initial_count.counts} == {dataflow.partition_count}
        for plan in terminal_epilogue_plans
    )
    physical_mechanisms = tuple(entry.mechanism for entry in schedule.audit.physical)
    assert physical_mechanisms.count("rhs-gradient epilogue full barrier and asynchronous transfer queue") == 2
    assert physical_mechanisms.count("overlapping epilogue scratch reuse barrier and asynchronous wait") == 1
    assert (
        sum(
            entry.mechanism == "matrix accumulator and ordered WGMMA completion"
            and entry.kind is EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT
            for entry in schedule.audit.entries
        )
        == 2
    )

    record = streaming_pullback_input_event_schedule_record(schedule)
    realizations = {realization["plan"]: realization for realization in record["realizations"]}
    scratch_plan = next(
        plan.name
        for plan in dataflow.program.event_plans
        if plan.required_dependence.relation.source == dataflow.producer_rhs_epilogue_transfer
        and plan.required_dependence.relation.target == dataflow.consumer_rhs_epilogue_stage
    )
    assert realizations[scratch_plan]["allocation_ids"] == (
        "rhs_gradient_epilogue_scratch",
        "rhs_gradient_epilogue_async_transfer_queue",
    )
    assert all(
        realization["allocation_ids"] if realization["kind"] == "physical" else not realization["allocation_ids"]
        for realization in realizations.values()
    )


def test_streaming_pullback_deterministic_accumulators_use_owner_relative_device_queues() -> None:
    schedule = derive_streaming_pullback_input_event_schedule(_pullback_descriptor(deterministic_accumulation=True))

    lhs, producer_rhs, consumer_rhs = schedule.device_accumulation_queues
    assert lhs.allocation.allocation_owners[:4] == ((0,), (0,), (0,), (0,))
    assert lhs.allocation.generations[:4] == (0, 1, 2, 3)
    assert len(lhs.allocation.plan.required_dependence.relation.pairs) == 24
    for queue in (producer_rhs, consumer_rhs):
        assert queue.allocation.allocation_owners[:4] == ((0,), (0,), (0,), (0,))
        assert queue.allocation.generations[:4] == (0, 1, 2, 3)
        assert len(queue.allocation.plan.required_dependence.relation.pairs) == 12
        assert queue.audit.physical[0].mechanism.startswith("device semaphore queue")

    lhs_record = ordered_accumulation_event_schedule_record(lhs)
    assert lhs_record["schema_version"] == 7
    assert lhs_record["realizations"][0]["allocation_ids"] == ("producer_lhs_gradient_accumulation.allocation",)
    assert lhs_record["allocations"][0]["covered_plans"] == (lhs_record["realizations"][0]["plan"],)


def test_streaming_pullback_fast_unordered_accumulation_allocates_no_semaphore_queue() -> None:
    schedule = derive_streaming_pullback_input_event_schedule(_pullback_descriptor())

    assert schedule.device_accumulation_queues == ()


def test_streaming_pipeline_allocation_rejects_mismatched_exact_event_pair() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor())
    allocation = schedule.first_streamed_input_pipeline
    malformed_buffer = replace(
        schedule.dataflow.first_streamed_input_buffer,
        reuse_dependences=schedule.dataflow.second_streamed_input_buffer.reuse_dependences,
    )

    with pytest.raises(ValueError, match="empty Event does not release the next buffer producer"):
        derive_streaming_bounded_pipeline_allocation(
            name=allocation.name,
            program=schedule.dataflow.program,
            buffer=malformed_buffer,
            consumer=schedule.dataflow.first_contract,
            transaction_bytes=allocation.transaction_bytes,
        )


def test_worker_mutation_changes_scheduler_event_realization() -> None:
    one_warpgroup = derive_streaming_physical_event_schedule(_descriptor(resident_tile=64))
    two_warpgroups = derive_streaming_physical_event_schedule(_descriptor(resident_tile=128))

    assert one_warpgroup.workers.matrix_warpgroups == 1
    assert one_warpgroup.workers.cta_threads == 256
    assert one_warpgroup.workers.scheduler_arrival_threads == 0
    assert two_warpgroups.workers.matrix_warpgroups == 2
    assert two_warpgroups.workers.cta_threads == 384
    assert two_warpgroups.workers.scheduler_arrival_threads == 256

    def scheduler_edges(schedule):
        plan_names = {
            plan.name
            for plan in schedule.dataflow.program.event_plans
            if plan.notify_relation.source == schedule.dataflow.second_contract
            and plan.trigger_relation.target == schedule.dataflow.first_contract
        }
        return tuple(entry for entry in schedule.audit.entries if entry.plan_name in plan_names)

    one_scheduler_edges = scheduler_edges(one_warpgroup)
    two_scheduler_edges = scheduler_edges(two_warpgroups)
    assert len(one_scheduler_edges) == len(two_scheduler_edges) == 1
    assert one_scheduler_edges[0].kind is EventRealizationKind.ERASED_PROGRAM_ORDER
    assert two_scheduler_edges[0].kind is EventRealizationKind.PHYSICAL


def test_backend_source_audit_rejects_stale_synchronization_constants() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=3))

    verify_streaming_event_backend_parameters(
        schedule,
        resident_input_stages=1,
        first_streamed_input_stages=3,
        second_streamed_input_stages=3,
        barriers_per_stage=2,
        transfer_warps=1,
        matrix_warpgroups=2,
        scheduler_arrival_threads=256,
    )
    with pytest.raises(ValueError, match="first_streamed_input_stages: expected 3, found 2"):
        verify_streaming_event_backend_parameters(
            schedule,
            resident_input_stages=1,
            first_streamed_input_stages=2,
            second_streamed_input_stages=3,
            barriers_per_stage=2,
            transfer_warps=1,
            matrix_warpgroups=2,
            scheduler_arrival_threads=256,
        )
