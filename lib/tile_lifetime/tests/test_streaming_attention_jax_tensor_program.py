# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.dense_flow_planner import derive_composed_task_plan
from tile_lifetime.event_dataflow import EventMemoryScope, ImperativeEventOpKind
from tile_lifetime.event_dataflow_adapters import streaming_contract_fold_event_descriptor
from tile_lifetime.executable_normalized_exp_contract import prepare_sm90_normalized_exp_contract
from tile_lifetime.h100_streaming_lowering import lower_h100_streaming_program
from tile_lifetime.index_domain_canonicalization import canonicalize_normalized_exp_index_domain
from tile_lifetime.internal_event_allocation import InternalEventDeviceOperationKind
from tile_lifetime.jax_normalized_exp_contract_pullback import (
    prepare_sm90_jax_owned_normalized_exp_pullback,
    trace_jax_owned_pullback,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.layout_relation_canonicalization import canonicalize_contract_layout_relations
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    derive_normalized_exp_contract_pullback,
    normalized_exp_contract_pullback_tensor_program,
    normalized_exp_contract_region,
)
from tile_lifetime.physical_task_contract import (
    EventRealizationKind,
    PhysicalNumericalPolicy,
    PhysicalTarget,
    TaskCapability,
    lower_executable_event_operations,
)
from tile_lifetime.streaming_attention_backward_reference import (
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention,
    causal_gqa_attention_training,
    causal_gqa_attention_vjp,
)
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.streaming_event_schedule import (
    canonical_streaming_physical_event_schedule_record,
    derive_streaming_physical_event_schedule,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    NumericalPolicy,
    ProgramValue,
    TensorAxis,
    TensorProgram,
    execute_tensor_program,
)
from tile_lifetime.tensor_program_rewrites import (
    compose_scalar_map_chains,
    fold_widening_converts_into_contracts,
    sink_fold_invariant_division_through_contract,
)

# Produced by the native MLIR benchmark fixture after projection into the shared region record.
_NATIVE_S2048_D128_NORMALIZED_EXP_REGION_FINGERPRINT = "0b3445ed3c0d92fbba0361bde6a1c3968a1eb309adab3532d62af2326ff3c67d"
_NATIVE_S2048_D128_STREAMING_EVENT_FINGERPRINT = "0bedfba633fa924e86b6e38c5deab686ec33c4aae7f0acd4bbbddff6d6c6e7bc"


@pytest.mark.parametrize("score_transform", ("identity", "tanh", "affine"))
def test_ordinary_jax_relation_rich_logical_pullback_matches_jax_vjp(score_transform: str) -> None:
    batch, query_length, key_length = 1, 3, 3
    key_value_heads, relation_group, key_dimension, value_dimension = 2, 2, 3, 4
    query_heads = key_value_heads * relation_group

    def function(query, key, value):
        grouped_query = query.reshape(
            batch,
            query_length,
            key_value_heads,
            relation_group,
            key_dimension,
        )
        score = jnp.einsum("bqhgd,bkhd->bhgqk", grouped_query, key) * 0.7
        if score_transform == "tanh":
            score = 1.6 * jnp.tanh(score / 1.6)
        elif score_transform == "affine":
            score = 1.25 * score + 0.2
        row_maximum = jnp.max(score, axis=-1, keepdims=True)
        exponential = jnp.exp(score - row_maximum)
        probability = exponential / jnp.sum(exponential, axis=-1, keepdims=True)
        return jnp.einsum("bhgqk,bkhv->bhvgq", probability, value)

    rng = np.random.default_rng(72)
    query = rng.normal(size=(batch, query_length, query_heads, key_dimension)).astype(np.float32)
    key = rng.normal(size=(batch, key_length, key_value_heads, key_dimension)).astype(np.float32)
    value = rng.normal(size=(batch, key_length, key_value_heads, value_dimension)).astype(np.float32)
    output_cotangent = rng.normal(size=(batch, key_value_heads, value_dimension, relation_group, query_length)).astype(
        np.float32
    )

    imported = replace(
        jax_function_to_tensor_program(function, query, key, value),
        numerical_policy=NumericalPolicy.FAST,
    )
    widened = fold_widening_converts_into_contracts(imported).program
    normalized = sink_fold_invariant_division_through_contract(widened).program
    relation_rich = canonicalize_contract_layout_relations(normalized).program
    composed = compose_scalar_map_chains(relation_rich).program
    region = normalized_exp_contract_region(composed)
    pullback = derive_normalized_exp_contract_pullback(region)
    logical = normalized_exp_contract_pullback_tensor_program(pullback)

    grouped_query = query.reshape(
        batch,
        query_length,
        key_value_heads,
        relation_group,
        key_dimension,
    )
    actual = execute_tensor_program(
        logical,
        {
            region.producer.inputs[0].name: grouped_query,
            region.producer.inputs[1].name: key,
            region.consumer.inputs[1].name: value,
            pullback.output_cotangent.name: output_cotangent,
        },
    )
    _, jax_pullback = jax.vjp(function, query, key, value)
    expected = jax_pullback(output_cotangent)
    actual_cotangents = (
        actual[pullback.terminal_gradients[0].destination.name].reshape(query.shape),
        actual[pullback.terminal_gradients[1].destination.name],
        actual[pullback.terminal_gradients[2].destination.name],
    )
    for candidate, reference in zip(actual_cotangents, expected, strict=True):
        np.testing.assert_allclose(candidate, np.asarray(reference), rtol=2e-5, atol=2e-5)


def test_affine_domain_restriction_preserves_logical_forward_and_pullback() -> None:
    def function(query, key, value):
        score = jnp.einsum("bqd,bkd->bqk", query, key) * 0.7
        query_index = jnp.arange(score.shape[-2])[:, None]
        key_index = jnp.arange(score.shape[-1])[None, :]
        score = jnp.where(key_index <= query_index + 1, score, -jnp.inf)
        row_maximum = jnp.max(score, axis=-1, keepdims=True)
        exponential = jnp.exp(score - row_maximum)
        probability = exponential / jnp.sum(exponential, axis=-1, keepdims=True)
        return jnp.einsum("bqk,bkv->bqv", probability, value)

    rng = np.random.default_rng(171)
    query = rng.normal(size=(1, 3, 2)).astype(np.float32)
    key = rng.normal(size=(1, 4, 2)).astype(np.float32)
    value = rng.normal(size=(1, 4, 3)).astype(np.float32)
    output_cotangent = rng.normal(size=(1, 3, 3)).astype(np.float32)

    imported = replace(
        jax_function_to_tensor_program(function, query, key, value),
        numerical_policy=NumericalPolicy.FAST,
    )
    widened = fold_widening_converts_into_contracts(imported).program
    normalized = sink_fold_invariant_division_through_contract(widened).program
    relation_rich = canonicalize_contract_layout_relations(normalized).program
    composed = compose_scalar_map_chains(relation_rich).program
    domain = canonicalize_normalized_exp_index_domain(composed)

    actual_forward = execute_tensor_program(
        domain.task_program,
        {value.name: argument for value, argument in zip(domain.task_program.inputs, (query, key, value), strict=True)},
    )
    np.testing.assert_allclose(
        actual_forward[domain.task_program.outputs[0].name],
        np.asarray(function(query, key, value)),
        rtol=2e-5,
        atol=2e-5,
    )

    pullback = derive_normalized_exp_contract_pullback(domain.region)
    logical_pullback = normalized_exp_contract_pullback_tensor_program(pullback)
    actual_pullback = execute_tensor_program(
        logical_pullback,
        {
            domain.region.producer.inputs[0].name: query,
            domain.region.producer.inputs[1].name: key,
            domain.region.consumer.inputs[1].name: value,
            pullback.output_cotangent.name: output_cotangent,
        },
    )
    expected_pullback = jax.vjp(function, query, key, value)[1](output_cotangent)
    for gradient, expected in zip(pullback.terminal_gradients, expected_pullback, strict=True):
        np.testing.assert_allclose(
            actual_pullback[gradient.destination.name], np.asarray(expected), rtol=2e-5, atol=2e-5
        )


def test_jax_owned_attention_programs_execute_and_reach_physical_request() -> None:
    config = StreamingAttentionBackwardDebugConfig()
    rng = np.random.default_rng(23)
    query = rng.normal(size=(config.batch, config.query_length, config.query_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    key = rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    value = rng.normal(size=key.shape).astype(jnp.bfloat16)
    output_cotangent = rng.normal(size=query.shape).astype(jnp.bfloat16)
    cases = (
        (causal_gqa_attention(config), (query, key, value)),
        (causal_gqa_attention_vjp(config), (query, key, value, output_cotangent)),
        (causal_gqa_attention_training(config), (query, key, value, output_cotangent)),
    )

    requests = []
    for function, arguments in cases:
        program = replace(
            jax_function_to_tensor_program(function, *arguments),
            numerical_policy=NumericalPolicy.FAST,
        )
        actual = execute_tensor_program(
            program,
            {f"argument.{index}": np.asarray(argument) for index, argument in enumerate(arguments)},
        )
        expected = function(*arguments)
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        for output, expected_output in zip(program.outputs, expected_outputs, strict=True):
            np.testing.assert_allclose(actual[output.name], np.asarray(expected_output), rtol=2e-5, atol=2e-5)

        request = jax_tensor_program_executable_request(
            program,
            tile_size=4,
            target=PhysicalTarget.SM90,
        )
        assert request.provenance["source_algebra_fingerprint"] == program.semantic_fingerprint
        assert request.provenance["transformed_algebra_fingerprint"] == request.algebra_fingerprint
        assert request.provenance["differentiation_owner"] == "jax"
        assert request.provenance["algebra_rewrites"].endswith("move_fold_invariant_division_through_contract")
        assert all(plan.required_dependence.visibility.scope is EventMemoryScope.DEVICE for plan in request.event_plans)
        assert not request.event_realizations.erasure_witnesses
        assert {entry.kind for entry in request.event_realizations.audit.physical} == {EventRealizationKind.PHYSICAL}
        event_lowerings = lower_executable_event_operations(request)
        assert all(lowering.retained_operations for lowering in event_lowerings)
        assert {operation.kind for lowering in event_lowerings for operation in lowering.retained_operations} == {
            ImperativeEventOpKind.INITIALIZE,
            ImperativeEventOpKind.WAIT,
            ImperativeEventOpKind.NOTIFY,
        }
        requests.append(request)

    assert len({request.fingerprint for request in requests}) == 3


def test_sm90_attachment_uses_canonical_domain_and_internalizes_legal_relation_views() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    program = replace(
        jax_function_to_tensor_program(causal_gqa_attention(config), query, key, key),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(program, tile_size=4, target=PhysicalTarget.SM90)

    attached = attach_sm90_contract_fold_schedule(request)
    restrictions = request.algebra_record["domain_restrictions"]
    assert restrictions == (
        {
            "domain_extents": (1, 2, 2, 4, 4),
            "streamed_dimensions": (4,),
            "coordinate_basis": "domain_position",
            "predicate": {
                "comparison": "signed_less_equal",
                "left": {"coefficients": ((4, 1),), "constant": 0},
                "right": {"coefficients": ((3, 1),), "constant": 0},
            },
        },
    )
    assert attached.physical_schedule.parameters["domain_predicates"] == tuple(
        restriction["predicate"] for restriction in restrictions
    )
    relation_tasks = {
        task.id
        for task in request.task_plan.tasks
        if any(phase.capability is TaskCapability.INDEX_RELATION for phase in task.structure.phases)
    }
    internal_relations = relation_tasks & set(attached.fused_task_ids)
    external_relations = relation_tasks - set(attached.fused_task_ids)
    assert internal_relations
    assert external_relations
    assert attached.physical_schedule.parameters["direction"] == "forward"
    assert all(
        all(
            phase.capability is TaskCapability.INDEX_RELATION
            for phase in request.task_plan.tasks[task_id].structure.phases
        )
        for task_id in internal_relations
    )
    family_to_task = {task.family: task.id for task in request.task_plan.tasks}
    fused = set(attached.fused_task_ids)
    for event, realization in zip(
        request.event_plans,
        attached.request.event_realizations.audit.entries,
        strict=True,
    ):
        endpoints = {
            family_to_task[event.scheduled_dependence.source],
            family_to_task[event.scheduled_dependence.target],
        }
        expected = "erased_program_order" if endpoints <= fused else "physical"
        assert realization.kind.value == expected


def test_executable_request_preserves_numerical_policy_until_physical_selection() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    source_program = jax_function_to_tensor_program(causal_gqa_attention(config), query, key, key)
    fast_program = replace(source_program, numerical_policy=NumericalPolicy.FAST)

    source_request = jax_tensor_program_executable_request(source_program, tile_size=4, target=PhysicalTarget.SM90)
    fast_request = jax_tensor_program_executable_request(fast_program, tile_size=4, target=PhysicalTarget.SM90)

    assert source_request.numerical_policy is PhysicalNumericalPolicy.SOURCE_ORDERED
    assert fast_request.numerical_policy is PhysicalNumericalPolicy.FAST
    assert source_request.algebra_fingerprint != fast_request.algebra_fingerprint
    assert source_request.task_plan.fingerprint != fast_request.task_plan.fingerprint
    assert source_request.fingerprint != fast_request.fingerprint
    with pytest.raises(
        ValueError,
        match="SM90 normalized-exponential Contract/Fold schedule does not implement SOURCE_ORDERED Fold evaluation",
    ):
        attach_sm90_contract_fold_schedule(source_request)
    attach_sm90_contract_fold_schedule(fast_request)


def test_ordinary_jax_forward_preserves_crossing_events_around_generic_cute_region() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    imported = replace(
        jax_function_to_tensor_program(causal_gqa_attention(config), query, key, key),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(imported, tile_size=4, target=PhysicalTarget.SM90)
    attached = attach_sm90_contract_fold_schedule(request)

    plan = prepare_sm90_normalized_exp_contract(attached.request)
    lowering = lower_h100_streaming_program(plan.program)

    assert request.provenance["algebra_rewrites"] == (
        "fold_exact_widening_convert_into_contract,move_fold_invariant_division_through_contract"
    )
    assert request.provenance["algebra_rewrite_proofs"][0]["numerical_equivalence"] == "bitwise_exact"
    assert plan.owned_task_ids == attached.fused_task_ids
    assert plan.residual_task_ids == (0, 5)
    assert plan.boundary_event_names == ("event.0", "event.4")
    assert plan.finalizer_fusion is not None
    assert plan.finalizer_fusion["task_id"] == 5
    algebra = request.algebra_record["program"]
    assert plan.finalizer_fusion["operation_ordinal"] == len(algebra["operations"]) - 1
    capability = request.normalized_exp_contract_regions[0]
    assert plan.finalizer_fusion["commuted_relation_ordinals"] == capability.finalizer_tail_operations[:-1]
    assert plan.finalizer_fusion["conversion"] == {
        "source_dtype": "fp32",
        "result_dtype": "bf16",
        "semantics": "round_nearest_even",
    }
    assert plan.finalizer_fusion["event_witness"]["event"] == "event.4"
    assert all(proof["zero_copy"] and proof["abi_compatible"] for proof in plan.relation_view_proofs)
    assert tuple(witness["event"] for witness in plan.event_erasure_witnesses) == (
        "event.1",
        "event.2",
        "event.3",
    )
    assert tuple(lowering.plan_name for lowering in plan.event_lowerings) == tuple(
        event.name for event in attached.request.event_plans
    )
    assert all(lowering.imperative_operations for lowering in plan.event_lowerings)
    retained = {lowering.plan_name for lowering in plan.event_lowerings if lowering.retained_operations}
    assert retained == {"event.0", "event.4"}
    assert {lowering.plan_name for lowering in plan.event_lowerings if lowering.erasure_proof is not None} == {
        "event.1",
        "event.2",
        "event.3",
    }
    assert {operation.kind for lowering in plan.event_lowerings for operation in lowering.imperative_operations} == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }
    assert plan.compile_request.physical_template == "sm90_contract_fold_contract"
    source_event_lowering = plan.compile_request.schedule["source_event_lowering"]
    assert tuple(record["plan"] for record in source_event_lowering) == tuple(
        event.name for event in attached.request.event_plans
    )
    assert all(record["imperative_operations"] for record in source_event_lowering)
    assert {record["plan"] for record in source_event_lowering if record["retained_operations"]} == {
        "event.0",
        "event.4",
    }
    refinement = plan.schedule_local_refinement
    assert tuple(sorted(ordinal for binding in refinement.bindings for ordinal in binding.operation_ordinals)) == (
        refinement.coarse_operation_ordinals
    )
    with pytest.raises(ValueError, match="cover every coarse algebra operation exactly once"):
        replace(
            refinement,
            coarse_operation_ordinals=(
                *refinement.coarse_operation_ordinals,
                max(refinement.coarse_operation_ordinals) + 1,
            ),
        )
    device_operations = tuple(
        operation
        for legalization in plan.internal_event_legalization.legalizations
        for operation in legalization.operations
    )
    assert {operation.kind for operation in device_operations} == {
        InternalEventDeviceOperationKind.WAIT,
        InternalEventDeviceOperationKind.NOTIFY,
    }
    assert not plan.whole_request_runtime_supported
    with pytest.raises(ValueError, match=r"boundary Events \[event.0, event.4\]"):
        plan.require_whole_request_runtime()
    assert lowering.score_map.domain_traversal is ContractDomainTraversal.FULL
    assert lowering.relation_layout.public_owner_shape == (1, 2, 2, 4, 128)
    assert lowering.relation_layout.public_streamed_producer_shape == (1, 2, 4, 128)
    assert all(value.dtype is DType.BF16 for value in plan.program.source.inputs)
    assert plan.program.region.producer.order is EvaluationOrder.ORDER_FREE
    assert plan.program.region.fold_state.maximum.order is EvaluationOrder.ORDER_FREE
    assert plan.program.region.fold_state.normalizer.order is EvaluationOrder.ORDER_FREE
    assert plan.program.region.consumer.order is EvaluationOrder.ORDER_FREE
    assert plan.compile_request.scalar_programs["score_map"] == {
        "kind": "multiply",
        "operands": [
            {"input": plan.program.region.producer.output.name, "kind": "input"},
            {"constant": 0.5, "kind": "constant"},
        ],
    }
    assert plan.compile_request.scalar_programs["finalizer"]["kind"] == "convert"
    assert plan.compile_request.schedule["finalizer_storage_conversion"] == {
        "destination_dtype": "bf16",
        "semantics": "round_nearest_even",
        "realization": "native_destination_store",
    }
    assert plan.compile_request.index_relations[-2]["predicate"] == plan.program.region.domain_predicate.canonical_record
    public_abi = plan.compile_request.index_relations[-1]["mapping"]
    assert public_abi["owner"] == {
        "shape": (1, 4, 4, 128),
        "required_strides": (2048, 512, 128, 1),
        "dimensions": ("batch", "owner", "flat_relation", "producer_feature"),
    }
    assert public_abi["output"]["shape"] == (1, 4, 4, 128)
    assert plan.compile_request.algebra_identity == {
        "request_algebra_fingerprint": attached.request.algebra_fingerprint,
        "source_algebra_fingerprint": imported.semantic_fingerprint,
        "transformed_algebra_fingerprint": attached.request.algebra_fingerprint,
    }


def test_ordinary_jax_region_matches_native_s2048_d128_semantic_fingerprint() -> None:
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=2048,
        key_length=2048,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        scale=128**-0.5,
    )
    query = np.empty((1, 2048, 32, 128), dtype=jnp.bfloat16)
    key = np.empty((1, 2048, 8, 128), dtype=jnp.bfloat16)
    imported = replace(
        jax_function_to_tensor_program(causal_gqa_attention(config), query, key, key),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(imported, tile_size=128, target=PhysicalTarget.SM90)

    program = prepare_sm90_normalized_exp_contract(attach_sm90_contract_fold_schedule(request).request).program

    assert program.canonical_region_record["algebra"] == program.source.canonical_record
    assert program.canonical_region_record["domain_predicate"] == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": (("a5", 1),), "constant": 0},
        "right": {"coefficients": (("a3", 1),), "constant": 0},
    }
    assert program.region_fingerprint == _NATIVE_S2048_D128_NORMALIZED_EXP_REGION_FINGERPRINT
    event_schedule = derive_streaming_physical_event_schedule(streaming_contract_fold_event_descriptor(program))
    event_record = canonical_streaming_physical_event_schedule_record(event_schedule)
    assert event_record["fingerprint"] == _NATIVE_S2048_D128_STREAMING_EVENT_FINGERPRINT
    assert tuple(realization["event"]["name"] for realization in event_record["realizations"]) == (
        "event_0_resident_input_stage_to_first_contract",
        "event_1_first_streamed_input_stage_to_first_contract",
        "event_2_first_contract_to_streaming_fold_update",
        "event_3_streaming_fold_update_to_second_contract",
        "event_4_second_streamed_input_stage_to_second_contract",
        "event_5_second_contract_to_first_contract",
        "event_6_second_contract_to_finalize",
        "event_7_first_contract_to_first_streamed_input_stage",
        "event_8_second_contract_to_second_streamed_input_stage",
    )


def test_jax_owned_pullback_replaces_every_composed_task_and_event() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    rng = np.random.default_rng(41)
    query = rng.normal(size=(config.batch, config.query_length, config.query_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    key = rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    value = rng.normal(size=key.shape).astype(jnp.bfloat16)
    output_cotangent = rng.normal(size=query.shape).astype(jnp.bfloat16)
    trace = trace_jax_owned_pullback(
        causal_gqa_attention(config),
        query,
        key,
        value,
        output_cotangent=output_cotangent,
    )
    trace = replace(
        trace,
        forward=replace(trace.forward, numerical_policy=NumericalPolicy.FAST),
        reverse=replace(trace.reverse, numerical_policy=NumericalPolicy.FAST),
        training=replace(trace.training, numerical_policy=NumericalPolicy.FAST),
    )
    request = jax_tensor_program_executable_request(
        trace.training,
        tile_size=128,
        target=PhysicalTarget.SM90,
    )

    plan = prepare_sm90_jax_owned_normalized_exp_pullback(request, trace, tile_size=128)

    assert request.provenance["index_domain_canonicalization"]["status"] == "applied"
    assert trace.canonical_record["differentiation_owner"] == "jax"
    assert plan.whole_request_runtime_supported
    assert plan.replacement.covered_task_ids == tuple(task.id for task in request.task_plan.tasks)
    assert plan.replacement.covered_operation_ordinals == tuple(
        range(sum(len(task.operation_ordinals) for task in request.task_plan.tasks))
    )
    assert tuple(witness["event"] for witness in plan.replacement.event_witnesses) == tuple(
        event.name for event in request.event_plans
    )
    assert all(lowering.retained_operations for lowering in plan.source_event_lowerings)
    assert {
        operation.kind for lowering in plan.source_event_lowerings for operation in lowering.retained_operations
    } == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }
    assert all(
        not lowering.retained_operations and lowering.erasure_proof is not None
        for lowering in plan.replacement_event_lowerings
    )
    assert all(lowering.retained_operations for lowering in plan.pullback_preselection_event_lowerings)
    pullback_task_ids = tuple(task.id for task in plan.pullback_request.task_plan.tasks)
    composed_task_ids = tuple(
        sorted(task_id for component in plan.pullback_composition.components for task_id in component.task_ids)
    )
    assert composed_task_ids == pullback_task_ids
    backward_device_operations = tuple(
        operation
        for legalization in plan.backward_internal_event_legalization.legalizations
        for operation in legalization.operations
    )
    assert {operation.kind for operation in backward_device_operations} == {
        InternalEventDeviceOperationKind.WAIT,
        InternalEventDeviceOperationKind.NOTIFY,
    }
    assert plan.lowering.compile_request.physical_template == "contract_fold_dependent_map_contract_sm90_v1"
    assert plan.lowering.compile_request.scalar_programs["score_map"] == {
        "kind": "multiply",
        "operands": [
            {"input": plan.program.forward.region.producer.output.name, "kind": "input"},
            {"constant": config.scale, "kind": "constant"},
        ],
    }
    assert plan.lowering.score_map_vjp_expression == plan.program.pullback.score_map_vjp.expression
    plan.require_whole_request_runtime()


def test_jax_owned_score_map_chains_share_schedule_but_retain_scalar_programs() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.empty((config.batch, config.query_length, config.query_heads, config.head_dimension), dtype=jnp.bfloat16)
    key = np.empty((config.batch, config.key_length, config.key_value_heads, config.head_dimension), dtype=jnp.bfloat16)
    output_cotangent = np.empty_like(query)

    def plan(score_transform):
        trace = trace_jax_owned_pullback(
            causal_gqa_attention(config, score_transform),
            query,
            key,
            key,
            output_cotangent=output_cotangent,
        )
        trace = replace(
            trace,
            forward=replace(trace.forward, numerical_policy=NumericalPolicy.FAST),
            reverse=replace(trace.reverse, numerical_policy=NumericalPolicy.FAST),
            training=replace(trace.training, numerical_policy=NumericalPolicy.FAST),
        )
        request = jax_tensor_program_executable_request(
            trace.training,
            tile_size=128,
            target=PhysicalTarget.SM90,
        )
        return prepare_sm90_jax_owned_normalized_exp_pullback(request, trace, tile_size=128)

    identity = plan(None)
    bounded = plan(lambda score: 10.0 * jnp.tanh(score / 10.0))
    affine = plan(lambda score: 1.25 * score + 0.2)

    lowerings = (identity.lowering, bounded.lowering, affine.lowering)
    assert all(candidate.physical_schedule == identity.lowering.physical_schedule for candidate in lowerings)
    assert all(candidate.relation_layout == identity.lowering.relation_layout for candidate in lowerings)
    assert len({str(candidate.compile_request.scalar_programs["score_map"]) for candidate in lowerings}) == 3
    assert len({str(candidate.compile_request.scalar_programs["score_map_vjp"]) for candidate in lowerings}) == 3
    assert len({candidate.compile_request.digest for candidate in lowerings}) == 3
    assert "tanh" in str(bounded.lowering.compile_request.scalar_programs["score_map"])
    assert "tanh" in str(bounded.lowering.compile_request.scalar_programs["score_map_vjp"])


def test_finite_fallback_mask_remains_explicit_and_cannot_attach() -> None:
    def finite_fallback(scores: jnp.ndarray) -> jnp.ndarray:
        query = jnp.arange(scores.shape[-2])[:, None]
        key = jnp.arange(scores.shape[-1])[None, :]
        return jnp.where(key <= query, scores, jnp.zeros_like(scores))

    scores = np.ones((3, 4), dtype=np.float32)
    program = jax_function_to_tensor_program(finite_fallback, scores)
    request = jax_tensor_program_executable_request(program, tile_size=4, target=PhysicalTarget.SM90)

    assert request.algebra_record["format"] == "shuttle_tensor_program_v1"
    assert request.provenance["index_domain_canonicalization"]["status"] == "not_applicable"
    assert any(
        phase.capability is TaskCapability.INDEX_RELATION
        for task in request.task_plan.tasks
        for phase in task.structure.phases
    )
    with pytest.raises(ValueError, match="Contract/Fold"):
        attach_sm90_contract_fold_schedule(request)


@pytest.mark.parametrize("result_view", [lambda value: value[..., ::2], lambda value: jnp.swapaxes(value, 1, 2)])
def test_public_output_views_outside_the_compiled_affine_abi_remain_physical(result_view) -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    attention = causal_gqa_attention(config)

    def attention_with_result_view(query, key, value):
        return result_view(attention(query, key, value))

    query = np.ones((config.batch, config.query_length, config.query_heads, config.head_dimension), dtype=jnp.bfloat16)
    key = np.ones((config.batch, config.key_length, config.key_value_heads, config.head_dimension), dtype=jnp.bfloat16)
    imported = replace(
        jax_function_to_tensor_program(attention_with_result_view, query, key, key),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(imported, tile_size=4, target=PhysicalTarget.SM90)
    attached = attach_sm90_contract_fold_schedule(request)

    plan = prepare_sm90_normalized_exp_contract(attached.request)

    assert plan.residual_task_ids
    assert plan.boundary_event_names
    assert any(lowering.retained_operations for lowering in plan.event_lowerings)
    assert any(not proof["abi_compatible"] for proof in plan.relation_view_proofs)
    with pytest.raises(ValueError, match=r"lacks a covering witness|runtime composition"):
        plan.require_whole_request_runtime()


def test_jax_attention_relation_boundaries_preserve_exact_events() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    program = jax_function_to_tensor_program(causal_gqa_attention(config), query, key, key)
    request = jax_tensor_program_executable_request(program, tile_size=4, target=PhysicalTarget.SM90)

    relation_tasks = tuple(
        task
        for task in request.task_plan.tasks
        if any(phase.capability is TaskCapability.INDEX_RELATION for phase in task.structure.phases)
    )
    assert relation_tasks
    assert all(dependence.relation.pairs for dependence in request.task_plan.dependences)
    assert len(request.event_plans) == len(request.task_plan.dependences)


def test_gqa_contract_dependence_uses_exact_head_relation() -> None:
    query_head = TensorAxis(0, 4, "query_head")
    key_value_head = TensorAxis(1, 2, "key_value_head")
    feature = TensorAxis(2, 2, "feature")
    folded_axis = TensorAxis(3, 2, "folded")
    source = ProgramValue("source", (key_value_head, folded_axis, feature), DType.FP32)
    key = ProgramValue("key", (key_value_head, feature), DType.FP32)
    query = ProgramValue("query", (query_head, feature), DType.FP32)
    score = ProgramValue("score", (query_head,), DType.FP32)
    fold = FoldPrimitive("produce key", source, key, (folded_axis,), FoldReducer.SUM, DType.FP32)
    contract = ContractPrimitive(
        "grouped score Contract",
        (query, key),
        score,
        (feature,),
        DType.FP32,
        input_index_maps=((), (AxisIndexMap(query_head, key_value_head, divisor=2),)),
    )
    program = TensorProgram((source, query), (fold, contract), (score,))

    plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: 1 for axis in (query_head, key_value_head, feature, folded_axis)},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    assert len(plan.tasks) == 2
    dependence = plan.dependences[0].relation
    assert tuple((pair.source, pair.target) for pair in dependence.pairs) == (
        ((0, 0), (0,)),
        ((0, 0), (1,)),
        ((0, 1), (0,)),
        ((0, 1), (1,)),
        ((1, 0), (2,)),
        ((1, 0), (3,)),
        ((1, 1), (2,)),
        ((1, 1), (3,)),
    )
    contract_task = plan.tasks[1]
    mapped_access = next(access for access in contract_task.accesses if "index_maps" in access.relation_record)
    assert mapped_access.relation_record == {
        "source_axes": (key_value_head.id, feature.id),
        "target_axes": (query_head.id,),
        "source_shape": (key_value_head.extent, feature.extent),
        "target_shape": (query_head.extent,),
        "index_maps": (
            {
                "domain_axis": query_head.id,
                "operand_axis": key_value_head.id,
                "divisor": 2,
                "offset": 0,
                "modulus": None,
            },
        ),
    }


def test_contract_dependence_uses_exact_wrapped_affine_tile_relation() -> None:
    domain = TensorAxis(0, 9, "domain")
    operand = TensorAxis(1, 5, "operand")
    feature = TensorAxis(2, 2, "feature")
    folded = TensorAxis(3, 2, "folded")
    source = ProgramValue("source", (operand, folded, feature), DType.FP32)
    projected = ProgramValue("projected", (operand, feature), DType.FP32)
    resident = ProgramValue("resident", (domain, feature), DType.FP32)
    result = ProgramValue("result", (domain,), DType.FP32)
    fold = FoldPrimitive("project", source, projected, (folded,), FoldReducer.SUM, DType.FP32)
    contract = ContractPrimitive(
        "consume projection",
        (resident, projected),
        result,
        (feature,),
        DType.FP32,
        input_index_maps=((), (AxisIndexMap(domain, operand, divisor=2, offset=-1, modulus=3),)),
    )
    program = TensorProgram((source, resident), (fold, contract), (result,))

    plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={domain.id: 3, operand.id: 2, feature.id: 2, folded.id: 2},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    dependence = plan.dependences[0].relation
    assert tuple((pair.source, pair.target) for pair in dependence.pairs) == (
        ((0, 0), (0,)),
        ((0, 0), (1,)),
        ((0, 0), (2,)),
        ((1, 0), (0,)),
        ((1, 0), (2,)),
    )
