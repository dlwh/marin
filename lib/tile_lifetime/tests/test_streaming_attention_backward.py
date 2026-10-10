# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime import (
    FoldAttachmentSite,
    FoldPrimitive,
    FoldReducer,
    FoldResultDisposition,
    ForwardOwnershipPlan,
    MapPrimitive,
    OwnerComputeTraversal,
    OwnerTileAvailability,
    ProgramValue,
    SharedReverseFusionDisposition,
    Sm90BackwardAccumulatorSchedule,
    Sm90ContractSchedule,
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardFoldOrder,
    StreamingAttentionBackwardProvenance,
    StreamingTileSchedule,
    TensorAxis,
    apply_causal_score_mask,
    apply_tanh_softcap,
    attach_fold_to_owner_preparation,
    build_attention_tensor_program,
    derive_forward_ownership_plan,
    derive_streaming_attention,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_fusion_plan,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    execute_streaming_attention_backward,
    execute_streaming_attention_with_state,
    lower_h100_normalized_exp_backward,
    lower_normalized_exp_score_map,
    normalized_exp_contract_pullback_program,
    relation_group_size,
    scaled_score_map,
    verify_normalized_exp_finalizer_vjp,
    verify_owner_preparation_fold_attachment,
    verify_streaming_attention_backward_score_map_vjp,
)
from tile_lifetime.h100_scalar_ast import factor_uniform_multiplicative_constant
from tile_lifetime.normalized_exp_contract_region import normalized_exp_contract_pullback_tensor_program
from tile_lifetime.normalized_exp_regional_pullback import split_normalized_exp_saved_state
from tile_lifetime.tensor_program import (
    NumericalPolicy,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    evaluate_scalar_expression,
    execute_tensor_program,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_expression_inputs,
    scalar_input,
    scalar_select,
    scalar_unary,
    serialize_scalar_expression,
)


def _program(
    *,
    causal: bool,
    softcap: float | None = None,
    output_scale: float = 1.0,
    query_heads: int = 4,
    key_value_heads: int = 2,
    query_length: int = 5,
    key_length: int = 5,
    head_dimension: int = 3,
):
    score_map = scaled_score_map(0.7)
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    if causal:
        score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=query_length,
        key_length=key_length,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=head_dimension,
        value_dimension=head_dimension,
        score_map=score_map,
        input_dtype=DType.FP32,
    )
    forward = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=2, key_value_tile_size=3, pipeline_depth=2),
    )
    if output_scale != 1.0:
        finalize = replace(
            forward.region.finalizer,
            expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_constant(output_scale),
                forward.region.finalizer.expression,
            ),
        )
        forward = replace(forward, region=replace(forward.region, finalizer=finalize))
    return forward


def _jax_reference(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    causal: bool,
    softcap: float | None,
    output_scale: float,
) -> jax.Array:
    expanded_key = jnp.repeat(key, 2, axis=2)
    expanded_value = jnp.repeat(value, 2, axis=2)
    score = jnp.einsum("bqhd,bkhd->bhqk", query, expanded_key) * 0.7
    if softcap is not None:
        score = softcap * jnp.tanh(score / softcap)
    if causal:
        query_position = jnp.arange(query.shape[1])
        key_position = jnp.arange(key.shape[1])
        score = jnp.where(key_position[None, None, None, :] <= query_position[None, None, :, None], score, -jnp.inf)
    probability = jax.nn.softmax(score, axis=-1)
    return output_scale * jnp.einsum("bhqk,bkhv->bqhv", probability, expanded_value)


def _with_nonlinear_post_normalization(forward):
    normalized = forward.region.finalizer.expression
    finalizer = replace(
        forward.region.finalizer,
        expression=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_unary(ScalarExpressionKind.TANH, normalized),
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(0.25), normalized),
        ),
    )
    return replace(forward, region=replace(forward.region, finalizer=finalizer))


@pytest.mark.parametrize("nonlinear_finalizer", (False, True))
def test_saved_state_cut_recomputes_reverse_algebra_across_finalizer_boundary(
    nonlinear_finalizer: bool,
) -> None:
    forward = _program(causal=True, query_heads=2, key_value_heads=2)
    if nonlinear_finalizer:
        forward = _with_nonlinear_post_normalization(forward)
    pullback = derive_streaming_attention_backward(forward).pullback
    split = split_normalized_exp_saved_state(forward.region, pullback)
    generator = np.random.default_rng(91)
    primal_inputs = {value.name: generator.normal(size=value.shape).astype(np.float32) for value in split.forward.inputs}
    forward_values = execute_tensor_program(split.forward, primal_inputs)
    output_cotangent = generator.normal(size=forward.region.finalizer.output.shape).astype(np.float32)
    reverse_inputs = {
        value.name: (
            output_cotangent
            if value.name == pullback.output_cotangent.name
            else forward_values[value.name] if value.name in forward_values else primal_inputs[value.name]
        )
        for value in split.reverse.inputs
    }
    split_values = execute_tensor_program(split.reverse, reverse_inputs)
    closed = normalized_exp_contract_pullback_tensor_program(pullback)
    closed_inputs = {
        value.name: output_cotangent if value.name == pullback.output_cotangent.name else primal_inputs[value.name]
        for value in closed.inputs
    }
    closed_values = execute_tensor_program(closed, closed_inputs)

    for split_output, closed_output in zip(split.reverse_outputs, closed.outputs[:3], strict=True):
        np.testing.assert_allclose(
            split_values[split_output.name],
            closed_values[closed_output.name],
            rtol=3e-6,
            atol=3e-6,
        )
    assert split.saved_values[0].shape == forward.region.fold_state.row_sum_exp.shape
    assert len(split.saved_values) == (2 if nonlinear_finalizer else 1)
    assert split.recomputed_values[0].shape == forward.region.producer.output.shape
    assert not (
        {value for operation in split.forward.operations for value in primitive_outputs(operation)}
        & {value for operation in split.reverse.operations for value in primitive_outputs(operation)}
    )


def test_backward_derivation_looks_through_terminal_storage_conversion() -> None:
    forward = _program(causal=True)
    finalizer = replace(
        forward.region.finalizer,
        expression=scalar_convert(
            forward.region.finalizer.expression,
            result_dtype=DType.FP32,
            semantics=ScalarConversionSemantics.EXACT,
        ),
    )

    backward = derive_streaming_attention_backward(replace(forward, region=replace(forward.region, finalizer=finalizer)))

    assert backward.output_scale == 1.0


def test_linear_post_normalization_scale_is_generic_pullback_algebra() -> None:
    backward = derive_streaming_attention_backward(_program(causal=False, output_scale=0.75))

    assert backward.output_scale == 0.75
    assert backward.pullback.consumer_weight_cotangent.inputs[0] == backward.pullback.post_normalization_vjp.output
    assert scalar_expression_inputs(backward.pullback.post_normalization_vjp.expression) == {
        backward.output_cotangent.name
    }


def test_pullback_closes_into_generic_algebra_with_relation_folds() -> None:
    identity = _program(causal=False)
    softcap = _program(causal=False, softcap=1.6)
    affine = _program(causal=False)
    affine_score = replace(
        affine.region.score_map,
        expression=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                affine.region.score_map.expression,
                scalar_constant(1.25),
            ),
            scalar_constant(0.25),
        ),
    )
    affine = replace(affine, region=replace(affine.region, score_map=affine_score))

    programs = tuple(
        normalized_exp_contract_pullback_tensor_program(derive_streaming_attention_backward(forward).pullback)
        for forward in (identity, softcap, affine)
    )

    assert len({program.semantic_fingerprint for program in programs}) == 3
    assert (
        tuple(type(operation) for operation in programs[0].operations)
        == tuple(type(operation) for operation in programs[1].operations)
        == tuple(type(operation) for operation in programs[2].operations)
    )
    assert sum(isinstance(operation, RelationPrimitive) for operation in programs[0].operations) == 2
    relation_folds = tuple(
        operation
        for operation in programs[0].operations
        if isinstance(operation, FoldPrimitive) and "relation Fold" in operation.name
    )
    assert len(relation_folds) == 2
    assert tuple(fold.output for fold in relation_folds) == programs[0].outputs[1:3]


@pytest.mark.parametrize(
    ("causal", "softcap", "output_scale"),
    ((True, None, 1.0), (False, 1.6, 0.75)),
)
def test_streaming_attention_backward_matches_jax_autodiff(
    causal: bool,
    softcap: float | None,
    output_scale: float,
) -> None:
    forward = _program(causal=causal, softcap=softcap, output_scale=output_scale)
    backward = derive_streaming_attention_backward(forward)
    rng = np.random.default_rng(91)
    query = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    key = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    value = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    output_cotangent = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    inputs = {"query": query, "key": key, "value": value}
    if causal:
        inputs.update(
            {
                "query.position": np.arange(5, dtype=np.int32),
                "key.position": np.arange(5, dtype=np.int32),
            }
        )
    execution = execute_streaming_attention_with_state(forward, inputs)
    actual = execute_streaming_attention_backward(backward, inputs, execution, output_cotangent)

    def loss(q, k, v):
        output = _jax_reference(
            q,
            k,
            v,
            causal=causal,
            softcap=softcap,
            output_scale=output_scale,
        )
        return jnp.sum(output * output_cotangent)

    expected_query, expected_key, expected_value = jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

    np.testing.assert_allclose(actual.query_cotangent, expected_query, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.key_cotangent, expected_key, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.value_cotangent, expected_value, rtol=2e-5, atol=2e-5)


def test_post_normalization_scalar_vjp_matches_jax_autodiff() -> None:
    forward = _with_nonlinear_post_normalization(_program(causal=False))
    backward = derive_streaming_attention_backward(forward)
    rng = np.random.default_rng(13)
    query = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    key = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    value = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    output_cotangent = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    inputs = {"query": query, "key": key, "value": value}
    execution = execute_streaming_attention_with_state(forward, inputs)

    actual = execute_streaming_attention_backward(backward, inputs, execution, output_cotangent)

    def loss(q, k, v):
        normalized = _jax_reference(q, k, v, causal=False, softcap=None, output_scale=1.0)
        output = jnp.tanh(normalized) + 0.25 * normalized
        return jnp.sum(output * output_cotangent)

    expected_query, expected_key, expected_value = jax.grad(loss, argnums=(0, 1, 2))(query, key, value)
    np.testing.assert_allclose(actual.query_cotangent, expected_query, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.key_cotangent, expected_key, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.value_cotangent, expected_value, rtol=2e-5, atol=2e-5)
    assert backward.output_dot_map.inputs == (
        backward.pullback.normalized_output.output,
        backward.pullback.post_normalization_vjp.output,
    )
    assert backward.pullback.consumer_weight_cotangent.inputs[0] == backward.pullback.post_normalization_vjp.output


def test_post_normalization_scalar_vjp_retains_auxiliary_map_inputs() -> None:
    forward = _program(causal=False)
    auxiliary = ProgramValue("output.gate", forward.region.finalizer.output.axes, DType.FP32)
    finalizer = replace(
        forward.region.finalizer,
        inputs=(*forward.region.finalizer.inputs, auxiliary),
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            forward.region.finalizer.expression,
            scalar_unary(ScalarExpressionKind.TANH, scalar_input(auxiliary.name)),
        ),
    )

    backward = derive_streaming_attention_backward(replace(forward, region=replace(forward.region, finalizer=finalizer)))

    assert backward.pullback.post_normalization.inputs == (
        backward.pullback.normalized_output.output,
        auxiliary,
    )
    assert auxiliary in backward.pullback.post_normalization_vjp.inputs
    assert auxiliary.name in scalar_expression_inputs(backward.pullback.post_normalization_vjp.expression)


def test_pullback_rejects_finalizer_that_uses_unnormalized_fold_state() -> None:
    forward = _program(causal=False)
    finalizer = replace(
        forward.region.finalizer,
        expression=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(forward.region.consumer.output.name),
            scalar_input(forward.region.fold_state.row_sum_exp.name),
        ),
    )

    with pytest.raises(ValueError, match="factor through their quotient"):
        derive_streaming_attention_backward(replace(forward, region=replace(forward.region, finalizer=finalizer)))


def test_finalizer_vjp_verifier_rejects_a_fixed_backend_formula() -> None:
    backward = derive_streaming_attention_backward(_with_nonlinear_post_normalization(_program(causal=False)))
    forged_vjp = replace(
        backward.pullback.post_normalization_vjp,
        expression=scalar_input(backward.output_cotangent.name),
        inputs=(backward.output_cotangent,),
    )
    program = normalized_exp_contract_pullback_program(
        backward.forward,
        schedule=Sm90BackwardAccumulatorSchedule(contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384)),
        traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )

    with pytest.raises(ValueError, match="finalizer Map VJP is not the derivative"):
        verify_normalized_exp_finalizer_vjp(
            replace(program, pullback=replace(program.pullback, post_normalization_vjp=forged_vjp))
        )


def test_score_map_mutation_reuses_one_backward_stage_family() -> None:
    baseline = derive_streaming_attention_backward(_program(causal=True))
    mutated = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))

    assert mutated.stages == baseline.stages
    assert mutated.materialized_values == baseline.materialized_values
    assert mutated.score_map_vjp.expression != baseline.score_map_vjp.expression
    assert tuple(value.name for value in mutated.materialized_values) == (
        "cotangent.query",
        "cotangent.key",
        "cotangent.value",
        "normalized_exp_contract.output_dot_cotangent",
    )
    verify_streaming_attention_backward_score_map_vjp(baseline)
    verify_streaming_attention_backward_score_map_vjp(mutated)


def test_shared_region_pullback_contains_fold_dependent_maps_and_terminal_contracts() -> None:
    forward = _program(causal=False, head_dimension=3)
    backward = derive_streaming_attention_backward(forward)
    pullback = backward.pullback

    assert pullback.region == forward.region
    assert pullback.output_dependent_fold.reduction_axes == forward.region.dimensions.consumer_features
    assert pullback.consumer_weight_cotangent.reduction_axes == forward.region.dimensions.consumer_features
    assert pullback.normalized_weight_map.output == pullback.normalized_weight
    assert pullback.normalized_weight_map.inputs == (
        forward.region.fold_state.exponential.output,
        forward.region.fold_state.row_sum_exp,
    )
    assert pullback.normalized_weight_map.expression.kind is ScalarExpressionKind.DIVIDE
    assert pullback.normalized_exp_pullback.inputs[-1] == pullback.output_dependent_fold.output
    assert pullback.score_map_vjp.inputs[0] == pullback.normalized_exp_pullback.output
    assert tuple(gradient.operand.name for gradient in pullback.terminal_gradients) == ("query", "key", "value")
    assert tuple(gradient.destination.shape for gradient in pullback.terminal_gradients) == (
        forward.region.producer.inputs[0].shape,
        forward.region.producer.inputs[1].shape,
        forward.region.consumer.inputs[1].shape,
    )
    assert tuple(gradient.contract.output.shape for gradient in pullback.terminal_gradients) == (
        (1, 5, 4, 3),
        (1, 5, 4, 3),
        (1, 5, 4, 3),
    )


@pytest.mark.parametrize(
    ("query_heads", "key_value_heads", "expected_indices"),
    (
        (4, 4, (0, 1, 2, 3)),
        (4, 2, (0, 0, 1, 1)),
        (4, 1, (0, 0, 0, 0)),
    ),
)
def test_head_sharing_is_only_an_operand_index_relation(
    query_heads: int,
    key_value_heads: int,
    expected_indices: tuple[int, ...],
) -> None:
    forward = _program(
        causal=False,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
    )
    relation = forward.region.producer.index_maps_for_input(1)

    assert relation[0].indices() == expected_indices
    assert relation_group_size(relation) == query_heads // key_value_heads
    assert forward.region.consumer.index_maps_for_input(1) == relation
    assert not hasattr(forward.region, "head_mode")


def test_scheduled_normalized_exp_programs_require_region_derived_ownership() -> None:
    forward = _program(causal=False)
    with pytest.raises(ValueError, match="forward ownership must be derived"):
        replace(
            forward,
            ownership=replace(forward.ownership, streamed_axes=forward.ownership.owner_axes),
        )

    pullback = normalized_exp_contract_pullback_program(
        forward,
        schedule=Sm90BackwardAccumulatorSchedule(contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384)),
        traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )
    with pytest.raises(ValueError, match="reverse ownership must be derived"):
        replace(
            pullback,
            ownership=replace(pullback.ownership, left_streamed_axes=pullback.ownership.left_owner_axes),
        )


def _h100_reverse_lowering(
    *,
    causal: bool = False,
    softcap: float | None = None,
    affine_bias: float | None = None,
    key_dimension: int = 128,
    value_dimension: int = 128,
    physical_schedule: Sm90BackwardAccumulatorSchedule | None = None,
    finalizer_unary: ScalarExpressionKind | None = None,
    output_scale: float = 1.0,
    key_value_heads: int = 8,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
):
    score_map = scaled_score_map(0.125)
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    if affine_bias is not None:
        score_map = replace(
            score_map,
            expression=scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_binary(
                    ScalarExpressionKind.MULTIPLY,
                    score_map.expression,
                    scalar_constant(1.25),
                ),
                scalar_constant(affine_bias),
            ),
        )
    if causal:
        score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=2048,
        key_length=2048,
        query_heads=32,
        key_value_heads=key_value_heads,
        key_dimension=key_dimension,
        value_dimension=value_dimension,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    source = replace(source, numerical_policy=numerical_policy)
    forward = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=2),
    )
    if output_scale != 1.0:
        finalizer = replace(
            forward.region.finalizer,
            expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_constant(output_scale),
                forward.region.finalizer.expression,
            ),
        )
        forward = replace(forward, region=replace(forward.region, finalizer=finalizer))
    if finalizer_unary is not None:
        finalizer = replace(
            forward.region.finalizer,
            expression=scalar_unary(finalizer_unary, forward.region.finalizer.expression),
        )
        forward = replace(forward, region=replace(forward.region, finalizer=finalizer))
    backward = normalized_exp_contract_pullback_program(
        forward,
        schedule=(
            Sm90BackwardAccumulatorSchedule(contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384))
            if physical_schedule is None
            else physical_schedule
        ),
        traversal=(
            StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR
            if causal
            else StreamingAttentionBackwardDomainTraversal.FULL
        ),
    )
    return lower_h100_normalized_exp_backward(backward)


def test_sm90_reverse_lowering_preserves_selected_accumulator_and_epilogue_schedule() -> None:
    schedule = Sm90BackwardAccumulatorSchedule(
        contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384),
        deterministic_accumulation=False,
        resident_consumer_operand_in_registers=True,
        left_gradient_single_warpgroup=True,
        owner_subtile_factor=2,
        epilogue_cluster_size=2,
    )

    lowering = _h100_reverse_lowering(physical_schedule=schedule)

    assert (
        replace(
            lowering.physical_schedule,
            contract=replace(lowering.physical_schedule.contract, relation_layout=None),
        )
        == schedule
    )
    assert lowering.physical_schedule.contract.relation_layout is lowering.relation_layout
    assert lowering.compile_request.schedule["deterministic_accumulation"] is False
    assert lowering.compile_request.schedule["resident_consumer_operand_in_registers"] is True
    assert lowering.compile_request.schedule["left_gradient_single_warpgroup"] is True
    assert lowering.compile_request.schedule["owner_subtile_factor"] == 2
    assert lowering.compile_request.schedule["epilogue_cluster_size"] == 2
    event_schedule = lowering.compile_request.schedule["physical_event_schedule"]
    queues = tuple(allocation for allocation in event_schedule["allocations"] if allocation["scope"] == "device")
    assert queues == ()


def test_sm90_reverse_lower_triangular_domain_derives_exact_deterministic_contributor_relation() -> None:
    schedule = Sm90BackwardAccumulatorSchedule(
        contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384),
        deterministic_accumulation=True,
    )

    lowering = _h100_reverse_lowering(
        causal=True, physical_schedule=schedule, key_value_heads=32, numerical_policy=NumericalPolicy.FAST
    )
    event_schedule = lowering.compile_request.schedule["physical_event_schedule"]
    lhs_queue = next(
        allocation
        for allocation in event_schedule["allocations"]
        if allocation["allocation_id"] == "producer_lhs_gradient_accumulation.allocation"
    )
    lhs_realization = next(
        realization
        for realization in event_schedule["realizations"]
        if lhs_queue["allocation_id"] in realization["allocation_ids"]
    )
    predecessor_pairs = tuple(
        (pair["source"], pair["target"]) for pair in lhs_realization["event"]["required_dependence"]["relation"]["pairs"]
    )

    assert len(lhs_queue["mechanism_coordinates"]) == 272
    assert len(predecessor_pairs) == 240
    assert ((0, 0), (0, 1)) not in predecessor_pairs
    assert ((2, 1), (2, 0)) in predecessor_pairs
    assert ((2, 0), (2, 1)) not in predecessor_pairs


def test_identity_softcap_and_affine_change_scalar_ast_but_share_sm90_schedule_family() -> None:
    identity = _h100_reverse_lowering()
    softcap = _h100_reverse_lowering(softcap=10.0)
    affine = _h100_reverse_lowering(affine_bias=0.25)

    assert identity.physical_schedule == softcap.physical_schedule == affine.physical_schedule
    assert identity.physical_schedule.contract.family == "sm90_contract_fold_v1"
    assert identity.compile_request.schedule == softcap.compile_request.schedule == affine.compile_request.schedule
    assert (
        identity.compile_request.index_relations
        == softcap.compile_request.index_relations
        == affine.compile_request.index_relations
    )
    assert (
        identity.compile_request.contraction_dimensions
        == softcap.compile_request.contraction_dimensions
        == affine.compile_request.contraction_dimensions
    )
    assert (
        len({str(lowering.compile_request.scalar_programs["score_map"]) for lowering in (identity, softcap, affine)})
        == 3
    )
    assert (
        len({str(lowering.compile_request.scalar_programs["score_map_vjp"]) for lowering in (identity, softcap, affine)})
        == 3
    )
    assert identity.compile_request.digest != softcap.compile_request.digest
    assert identity.compile_request.schedule["relation_layout"]["relation_group_axes"][0]["extent"] == 4
    assert identity.compile_request.schedule["right_owner_fold_order"] == (
        "streamed_axes_major_mapped_relation_minor_tree"
    )
    assert "key_value_fold_order" not in identity.compile_request.schedule
    assert tuple(tensor["role"] for tensor in identity.compile_request.tensors) == (
        "producer_left",
        "producer_right",
        "consumer_operand",
        "final_output",
    )
    assert set(identity.compile_request.scalar_programs) >= {
        "post_normalization_finalizer",
        "post_normalization_finalizer_vjp",
    }


def test_post_normalization_maps_share_sm90_family_but_change_required_cotangent_pipeline() -> None:
    identity = _h100_reverse_lowering()
    linear = _h100_reverse_lowering(output_scale=0.75)
    nonlinear = _h100_reverse_lowering(finalizer_unary=ScalarExpressionKind.TANH)

    assert identity.physical_schedule == linear.physical_schedule == nonlinear.physical_schedule
    structural_schedules = tuple(
        {name: value for name, value in lowering.compile_request.schedule.items() if name != "physical_event_schedule"}
        for lowering in (identity, linear, nonlinear)
    )
    assert structural_schedules[0] == structural_schedules[1] == structural_schedules[2]
    identity_cotangent_pipeline = identity.input_event_schedule.output_cotangent_pipeline
    linear_cotangent_pipeline = linear.input_event_schedule.output_cotangent_pipeline
    nonlinear_cotangent_pipeline = nonlinear.input_event_schedule.output_cotangent_pipeline
    assert identity_cotangent_pipeline.capacity == 2
    assert linear_cotangent_pipeline.capacity == nonlinear_cotangent_pipeline.capacity == 1
    assert linear_cotangent_pipeline.transaction_bytes - identity_cotangent_pipeline.transaction_bytes == 64 * 128 * 2
    assert linear_cotangent_pipeline.transaction_bytes == nonlinear_cotangent_pipeline.transaction_bytes
    finalizer_vjps = {
        str(lowering.compile_request.scalar_programs["post_normalization_finalizer_vjp"])
        for lowering in (identity, linear, nonlinear)
    }
    assert len(finalizer_vjps) == 3
    assert len({lowering.compile_request.digest for lowering in (identity, linear, nonlinear)}) == 3
    assert identity.compile_request.schedule["post_normalization_pullback"] == {
        "normalized_output_storage": "fp32",
        "normalized_output_cotangent_storage": "bf16",
        "normalized_output_cotangent_expansion": "iterated_bf16_residual",
        "normalized_output_cotangent_terms": 2,
        "normalized_output_cotangent_stages_per_term": 1,
        "expanded_cotangent_shared_memory_bytes": 32768,
        "feature_fold": "normalized_output_times_cotangent_sum",
    }


def test_two_term_bf16_cotangent_expansion_reduces_early_rounding_error() -> None:
    generator = np.random.default_rng(17)
    cotangent = jnp.asarray(generator.standard_normal(4096), dtype=jnp.float32)

    primary = cotangent.astype(jnp.bfloat16).astype(jnp.float32)
    residual = (cotangent - primary).astype(jnp.bfloat16).astype(jnp.float32)
    reconstructed = primary + residual

    one_term_error = np.asarray(jnp.abs(cotangent - primary))
    two_term_error = np.asarray(jnp.abs(cotangent - reconstructed))
    assert two_term_error.max() <= 3.1e-5
    assert two_term_error.max() < one_term_error.max() / 100


def test_two_term_cotangent_schedule_preserves_the_existing_shared_memory_budget() -> None:
    lowering = _h100_reverse_lowering(finalizer_unary=ScalarExpressionKind.TANH)
    schedule = lowering.compile_request.schedule
    pullback = schedule["post_normalization_pullback"]

    assert schedule["post_normalization_cotangent_expansion_terms"] == 2
    assert pullback["normalized_output_cotangent_stages_per_term"] == 1
    assert pullback["expanded_cotangent_shared_memory_bytes"] == 2 * 64 * 128 * 2
    assert schedule["mainloop_dynamic_shared_memory_bytes"] == 198_656


@pytest.mark.parametrize(("softcap", "expects_primal_score"), ((None, False), (10.0, True)))
def test_causal_score_vjp_is_restricted_by_the_certified_domain(
    softcap: float | None,
    expects_primal_score: bool,
) -> None:
    lowering = _h100_reverse_lowering(causal=True, softcap=softcap)

    assert lowering.score_map.domain_traversal.value == "lower_triangular"
    assert {"query.position", "key.position"}.isdisjoint(scalar_expression_inputs(lowering.score_map_vjp_expression))
    assert (lowering.score_map.score_input in scalar_expression_inputs(lowering.score_map_vjp_expression)) is (
        expects_primal_score
    )


def test_uniform_vjp_factor_peephole_preserves_identity_and_scaled_expressions() -> None:
    cotangent = scalar_input("cotangent")
    identity, identity_scale = factor_uniform_multiplicative_constant(cotangent)
    scaled, scaled_scale = factor_uniform_multiplicative_constant(
        scalar_binary(ScalarExpressionKind.MULTIPLY, cotangent, scalar_constant(0.125))
    )

    assert identity == cotangent
    assert identity_scale == 1.0
    assert scaled == cotangent
    assert scaled_scale == 0.125
    zero_expression = scalar_binary(ScalarExpressionKind.MULTIPLY, cotangent, scalar_constant(0.0))
    zero_residual, zero_scale = factor_uniform_multiplicative_constant(zero_expression)
    assert zero_residual == zero_expression
    assert zero_scale == 1.0


@pytest.mark.parametrize(
    ("mutation", "expected_scale", "residual_kind"),
    (({}, 0.125, "input"), ({"softcap": 1.3}, 0.125, "tanh"), ({"affine_bias": 0.2}, 0.15625, "input")),
)
def test_uniform_vjp_factor_peephole_preserves_full_ast_identity_and_numerics(
    mutation: dict[str, float],
    expected_scale: float,
    residual_kind: str,
) -> None:
    lowering = _h100_reverse_lowering(**mutation)
    full_expression = lowering.score_map_vjp_expression
    residual_expression = lowering.score_map_vjp_codegen_expression
    inputs = {name: 0.3 for name in scalar_expression_inputs(full_expression)}

    full = float(evaluate_scalar_expression(full_expression, inputs))
    residual = float(evaluate_scalar_expression(residual_expression, inputs))
    assert lowering.terminal_contract_epilogue_scale == expected_scale
    assert residual * lowering.terminal_contract_epilogue_scale == pytest.approx(full)
    assert lowering.compile_request.scalar_programs["score_map_vjp"] == json.loads(
        serialize_scalar_expression(full_expression)
    )
    if residual_kind == "input":
        assert residual_expression.kind is ScalarExpressionKind.INPUT
    else:
        pending = [residual_expression]
        kinds = set()
        while pending:
            current = pending.pop()
            kinds.add(current.kind)
            pending.extend(current.operands)
        assert ScalarExpressionKind.TANH in kinds


def test_sm90_triangular_score_lowering_uses_region_axes_instead_of_workload_names() -> None:
    forward = _program(causal=True)
    region = forward.region
    streamed_axis = region.dimensions.normalized_axes[0]
    streamed_position = next(value for value in region.score_map.inputs if value.axes == (streamed_axis,))
    owner_position = next(
        value for value in region.score_map.inputs if value != region.producer.output and value != streamed_position
    )
    renamed_streamed = replace(streamed_position, name="streamed.coordinate")
    renamed_owner = replace(owner_position, name="owner.coordinate")
    selected = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input(region.producer.output.name),
        scalar_constant(0.7),
    )
    renamed_score_map = replace(
        region.score_map,
        inputs=(region.producer.output, renamed_streamed, renamed_owner),
        expression=scalar_select(
            scalar_binary(
                ScalarExpressionKind.LESS_EQUAL,
                scalar_input(renamed_streamed.name),
                scalar_input(renamed_owner.name),
            ),
            selected,
            scalar_constant(float("-inf")),
        ),
    )

    lowering = lower_normalized_exp_score_map(replace(region, score_map=renamed_score_map))

    assert lowering.domain_traversal.value == "lower_triangular"
    assert lowering.expression == selected


@pytest.mark.parametrize(
    ("key_dimension", "value_dimension", "message"),
    (
        (64, 128, "producer feature dimension of 128"),
        (128, 64, "consumer feature dimension of 128"),
    ),
)
def test_sm90_reverse_boundary_distinguishes_producer_and_consumer_features(
    key_dimension: int,
    value_dimension: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _h100_reverse_lowering(key_dimension=key_dimension, value_dimension=value_dimension)


def test_score_map_vjp_verifier_rejects_a_fixed_backend_formula() -> None:
    backward = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))
    forged_vjp = replace(
        backward.score_map_vjp,
        expression=scalar_constant(0.7),
        inputs=(),
    )

    with pytest.raises(ValueError, match="not the derivative"):
        verify_streaming_attention_backward_score_map_vjp(
            replace(backward, pullback=replace(backward.pullback, score_map_vjp=forged_vjp))
        )


def test_grouped_key_value_schedule_is_derived_from_contract_index_relation() -> None:
    backward = derive_streaming_attention_backward(_program(causal=True))
    recovered = replace(backward, provenance=StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY)
    schedule = derive_streaming_attention_backward_tile_schedule(
        recovered,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    work = estimate_streaming_attention_backward_work(recovered, schedule)
    forward_ownership = derive_forward_ownership_plan(recovered.forward.region)

    assert isinstance(forward_ownership, ForwardOwnershipPlan)
    assert recovered.forward.ownership == forward_ownership
    assert schedule.ownership.producer_index_relations == forward_ownership.producer_index_relations
    assert schedule.ownership.consumer_index_relations == forward_ownership.consumer_index_relations
    assert schedule.ownership.left_streamed_axes == recovered.forward.region.dimensions.normalized_axes
    assert schedule.query_heads_per_key_value_tile == 2
    assert (
        schedule.key_value_fold_order
        is StreamingAttentionBackwardFoldOrder.STREAMED_AXES_MAJOR_MAPPED_RELATION_MINOR_TREE
    )
    assert len(schedule.query_owner_attachments) == 1
    attachment = schedule.query_owner_attachments[0]
    assert attachment.producer == recovered.output_dot_map
    assert attachment.fold == recovered.output_dot_fold
    assert attachment.site is FoldAttachmentSite.OWNER_PREPARATION
    assert attachment.result_disposition is FoldResultDisposition.MATERIALIZE_FOR_CONSUMERS
    assert work.logical_query_key_tile_pairs == 60
    assert work.fully_restricted_tile_pairs == 40
    assert work.query_gradient_contract_invocations == 90
    assert work.scalar_head_query_gradient_contract_invocations == 180
    assert work.full_domain_query_gradient_contract_invocations == 150
    assert work.full_domain_scalar_head_query_gradient_contract_invocations == 300
    assert work.query_gradient_contract_invocation_reduction == 2.0
    assert work.query_gradient_contract_invocation_reduction_from_full_scalar == 10 / 3
    assert work.key_value_gradient_contract_invocations == 120
    assert work.scalar_head_key_value_contract_invocations == 240
    assert work.full_domain_scalar_head_key_value_contract_invocations == 400
    assert work.key_value_contract_invocation_reduction == 2.0
    assert work.key_value_contract_invocation_reduction_from_full_scalar == 10 / 3
    assert work.packed_query_rows == 2
    assert work.peak_score_tile_elements == 2
    assert work.peak_query_tile_elements == 6
    assert work.key_value_gradient_accumulator_elements == 6


def test_owner_fold_attachment_uses_only_generic_axes_and_values() -> None:
    row = TensorAxis(100, 4, "arbitrary_row")
    feature = TensorAxis(101, 8, "arbitrary_feature")
    left = ProgramValue("left_value", (row, feature), DType.FP32)
    right = ProgramValue("right_value", (row, feature), DType.FP32)
    product = ProgramValue("product_value", (row, feature), DType.FP32)
    reduced = ProgramValue("reduced_value", (row,), DType.FP32)
    producer = MapPrimitive(
        "arbitrary pointwise producer",
        (left, right),
        product,
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(left.name),
            scalar_input(right.name),
        ),
    )
    fold = FoldPrimitive(
        "arbitrary row Fold",
        product,
        reduced,
        (feature,),
        FoldReducer.SUM,
        DType.FP32,
    )

    attachment = attach_fold_to_owner_preparation(
        producer,
        fold,
        owner_axes=(row,),
        input_availability=(
            OwnerTileAvailability(left, (feature,)),
            OwnerTileAvailability(right, (feature,)),
        ),
        result_disposition=FoldResultDisposition.MATERIALIZE_FOR_CONSUMERS,
    )

    assert attachment.fold.output == reduced
    assert attachment.owner_axes == (row,)
    assert attachment.site is FoldAttachmentSite.OWNER_PREPARATION


def test_owner_fold_attachment_rejects_partial_reduction_axis() -> None:
    row = TensorAxis(100, 4, "arbitrary_row")
    feature = TensorAxis(101, 8, "arbitrary_feature")
    left = ProgramValue("left_value", (row, feature), DType.FP32)
    right = ProgramValue("right_value", (row, feature), DType.FP32)
    product = ProgramValue("product_value", (row, feature), DType.FP32)
    reduced = ProgramValue("reduced_value", (row,), DType.FP32)
    producer = MapPrimitive(
        "arbitrary pointwise producer",
        (left, right),
        product,
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(left.name),
            scalar_input(right.name),
        ),
    )
    fold = FoldPrimitive(
        "arbitrary row Fold",
        product,
        reduced,
        (feature,),
        FoldReducer.SUM,
        DType.FP32,
    )

    with pytest.raises(ValueError, match="complete along the Fold reduction axes"):
        attach_fold_to_owner_preparation(
            producer,
            fold,
            owner_axes=(row,),
            input_availability=(
                OwnerTileAvailability(left, (feature,)),
                OwnerTileAvailability(right, ()),
            ),
            result_disposition=FoldResultDisposition.MATERIALIZE_FOR_CONSUMERS,
        )

    valid = attach_fold_to_owner_preparation(
        producer,
        fold,
        owner_axes=(row,),
        input_availability=(
            OwnerTileAvailability(left, (feature,)),
            OwnerTileAvailability(right, (feature,)),
        ),
        result_disposition=FoldResultDisposition.MATERIALIZE_FOR_CONSUMERS,
    )
    forged = replace(
        valid,
        input_availability=(
            OwnerTileAvailability(left, (feature,)),
            OwnerTileAvailability(right, ()),
        ),
    )
    with pytest.raises(ValueError, match="complete along the Fold reduction axes"):
        verify_owner_preparation_fold_attachment(forged)


@pytest.mark.parametrize(
    ("query_heads", "key_value_heads", "expected_group_size"),
    ((4, 4, 1), (4, 2, 2), (8, 2, 4)),
)
def test_query_gradient_packing_tracks_contract_head_index_relation(
    query_heads: int,
    key_value_heads: int,
    expected_group_size: int,
) -> None:
    backward = derive_streaming_attention_backward(
        _program(
            causal=False,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
        )
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        backward,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )
    work = estimate_streaming_attention_backward_work(backward, schedule)

    assert schedule.query_heads_per_key_value_tile == expected_group_size
    assert work.query_gradient_contract_invocation_reduction == expected_group_size
    assert work.key_value_contract_invocation_reduction == expected_group_size


def test_score_map_mutation_changes_domain_work_without_changing_grouped_schedule() -> None:
    causal = derive_streaming_attention_backward(_program(causal=True))
    softcap = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))
    causal_schedule = derive_streaming_attention_backward_tile_schedule(
        causal,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    softcap_schedule = derive_streaming_attention_backward_tile_schedule(
        softcap,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )

    causal_work = estimate_streaming_attention_backward_work(causal, causal_schedule)
    softcap_work = estimate_streaming_attention_backward_work(softcap, softcap_schedule)
    assert causal_schedule.query_heads_per_key_value_tile == softcap_schedule.query_heads_per_key_value_tile
    assert causal_work.logical_query_key_tile_pairs == 60
    assert softcap_work.logical_query_key_tile_pairs == 100
    assert softcap_work.fully_restricted_tile_pairs == 0
    assert causal.provenance is StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP
    assert softcap.provenance is StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP


def test_fused_reverse_relation_tracks_causal_domain_restriction() -> None:
    causal = derive_streaming_attention_backward(_program(causal=True))
    unrestricted = derive_streaming_attention_backward(_program(causal=False))
    causal_schedule = derive_streaming_attention_backward_tile_schedule(
        causal,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    unrestricted_schedule = derive_streaming_attention_backward_tile_schedule(
        unrestricted,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )

    causal_fusion = derive_streaming_attention_backward_fusion_plan(
        causal,
        causal_schedule,
        local_capacity_bytes=176,
    )
    unrestricted_fusion = derive_streaming_attention_backward_fusion_plan(
        unrestricted,
        unrestricted_schedule,
        local_capacity_bytes=176,
    )

    assert causal_fusion.disposition is SharedReverseFusionDisposition.FUSED_LOCAL
    assert unrestricted_fusion.disposition is SharedReverseFusionDisposition.FUSED_LOCAL
    assert len(causal_fusion.relation.pairs) == 30
    assert len(unrestricted_fusion.relation.pairs) == 50
    assert causal_fusion.baseline_contract_invocations == 210
    assert causal_fusion.fused_contract_invocations == 150
    assert unrestricted_fusion.baseline_contract_invocations == 350
    assert unrestricted_fusion.fused_contract_invocations == 250


def test_fused_reverse_ownership_is_independent_of_score_map_vjp() -> None:
    baseline = derive_streaming_attention_backward(_program(causal=True))
    mutated = derive_streaming_attention_backward(_program(causal=True, softcap=1.3))
    schedule = derive_streaming_attention_backward_tile_schedule(
        baseline,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    mutated_schedule = derive_streaming_attention_backward_tile_schedule(
        mutated,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )

    baseline_fusion = derive_streaming_attention_backward_fusion_plan(
        baseline,
        schedule,
        local_capacity_bytes=176,
    )
    mutated_fusion = derive_streaming_attention_backward_fusion_plan(
        mutated,
        mutated_schedule,
        local_capacity_bytes=176,
    )

    assert baseline.score_map_vjp.expression != mutated.score_map_vjp.expression
    assert baseline_fusion.components == mutated_fusion.components
    assert baseline_fusion.baseline_contract_invocations == mutated_fusion.baseline_contract_invocations
    assert baseline_fusion.fused_contract_invocations == mutated_fusion.fused_contract_invocations


def test_primary_fused_reverse_rejects_large_owner_compute_frontier() -> None:
    backward = derive_streaming_attention_backward(
        _program(
            causal=True,
            query_heads=32,
            key_value_heads=8,
            query_length=2048,
            key_length=2048,
            head_dimension=128,
        )
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        backward,
        query_tile_size=32,
        key_value_tile_size=32,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    fusion = derive_streaming_attention_backward_fusion_plan(
        backward,
        schedule,
        local_capacity_bytes=227 * 1024,
    )

    assert fusion.disposition is SharedReverseFusionDisposition.REJECTED_LOCAL_CAPACITY
    assert len(fusion.components) == 8
    assert all(component.selected_traversal is OwnerComputeTraversal.SOURCE_MAJOR for component in fusion.components)
    assert fusion.required_local_bytes == 2_195_456
    assert fusion.baseline_contract_invocations == 116_480
    assert fusion.fused_contract_invocations == 83_200
    assert fusion.physical_contract_reduction == 1.4
    assert fusion.source_accumulator_elements == 16_384
    assert fusion.target_accumulator_elements == 8_192
    assert fusion.transient_edge_elements == 16_384
