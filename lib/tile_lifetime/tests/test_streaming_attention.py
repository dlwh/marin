# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime import (
    AttentionScoreAxis,
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    MapPrimitive,
    NormalizedExpContractProgram,
    ProgramValue,
    ScalarExpressionKind,
    StreamingAttentionStage,
    StreamingTileSchedule,
    TensorProgram,
    add_score_bias,
    apply_arbitrary_score_mask,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    execute_streaming_attention,
    execute_tensor_program,
    scaled_score_map,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90ContractSchedule,
    Sm90ForwardPipelineSchedule,
    Sm90ProducerOperandOrientation,
    lower_h100_streaming_program,
    prove_sm90_packed_row_state_writes,
    prove_sm90_producer_consumer_handoff,
    select_sm90_contract_schedule,
    sm90_relation_packed_layout,
)
from tile_lifetime.tensor_program import TensorAxis, scalar_binary, scalar_constant, scalar_input


class ScoreMutation(StrEnum):
    """Semantically distinct score programs exercised through one generator."""

    CAUSAL = "causal"
    BIAS_AND_MASK = "bias_and_mask"
    TANH_SOFTCAP = "tanh_softcap"


def test_sm90_swapped_producer_consumer_handoff_is_bijective_and_explicit() -> None:
    relation_layout = sm90_relation_packed_layout(
        batch_shape=(1,),
        shared_relation_extent=1,
        relation_group_shape=(1,),
        owner_extent=128,
        streamed_extent=128,
        producer_feature_extent=128,
        consumer_feature_extent=128,
    )
    contract = Sm90ContractSchedule(128, 128, relation_layout=relation_layout)
    proof = prove_sm90_producer_consumer_handoff(
        contract,
        Sm90ProducerOperandOrientation.STREAMED_OWNER,
    )

    assert proof.logical_element_count == 128 * 128
    assert proof.emitted_element_count == proof.logical_element_count
    assert proof.unique_element_count == proof.logical_element_count
    assert proof.shared_memory_bytes == 128 * 128 * 2
    assert proof.requires_register_to_shared_transpose
    assert proof.producer_fold_rows_per_thread == 32
    assert proof.consumer_fold_rows_per_thread == 2
    assert proof.required_schedule_local_handoffs == 3
    assert proof.producer_fragment_axes == ("streamed", "owner")
    assert proof.fold_fragment_axes == ("owner", "streamed")
    assert proof.barrier_generations == ("empty_before_reuse", "full_after_store")
    assert proof.barrier_participants == 256
    assert proof.estimated_shared_storage_bytes == 197_632

    with pytest.raises(ValueError, match="register-to-shared transpose"):
        Sm90ForwardPipelineSchedule(
            contract,
            stages=2,
            producer_operand_orientation=Sm90ProducerOperandOrientation.STREAMED_OWNER,
        )
    schedule = Sm90ForwardPipelineSchedule(
        contract,
        stages=2,
        consumer_register_source=False,
        producer_operand_orientation=Sm90ProducerOperandOrientation.STREAMED_OWNER,
    )
    assert schedule.producer_operand_orientation is Sm90ProducerOperandOrientation.STREAMED_OWNER


def test_sm90_relation_layout_preserves_public_axes_and_selects_fa4_flattening() -> None:
    layout = sm90_relation_packed_layout(
        batch_shape=(2, 3),
        shared_relation_extent=5,
        relation_group_shape=(2, 4),
        owner_extent=7,
        streamed_extent=11,
        producer_feature_extent=13,
        consumer_feature_extent=17,
    )

    assert layout.public_owner_shape == (2, 3, 5, 2, 4, 7, 13)
    assert layout.public_streamed_producer_shape == (2, 3, 5, 11, 13)
    assert layout.public_streamed_consumer_shape == (2, 3, 5, 11, 17)
    assert layout.owner_physical_shape(13) == (6, 7, 40, 13)
    assert layout.streamed_physical_shape(13) == (6, 11, 5, 13)
    assert layout.owner_coordinate_shape(13) == (7, 13, 40, 6)
    assert layout.streamed_coordinate_shape(13) == (11, 13, 5, 6)
    assert layout.streamed_coordinate_shape(17) == (11, 17, 5, 6)
    assert layout.owner_coordinate_shape(17) == (7, 17, 40, 6)
    assert layout.shared_relation_map.indices() == tuple(index // 8 for index in range(40))
    assert layout.group_relation_maps[0].indices() == tuple((index // 4) % 2 for index in range(40))
    assert layout.group_relation_maps[1].indices() == tuple(index % 4 for index in range(40))

    proof = prove_sm90_packed_row_state_writes(layout, owner_tile=64)
    assert proof.logical_row_count == 6 * 40 * 7
    assert proof.emitted_write_count == proof.logical_row_count
    assert proof.unique_write_count == proof.logical_row_count

    batch_flat, shared, group_0, group_1, owner, feature = 5, 3, 1, 2, 6, 12
    flattened_relation = shared * 8 + group_0 * 4 + group_1
    physical_offset = sum(
        index * stride
        for index, stride in zip(
            (batch_flat, owner, flattened_relation, feature),
            layout.owner_physical_strides(13),
            strict=True,
        )
    )
    public_offset = ((((batch_flat * 5 + shared) * 2 + group_0) * 4 + group_1) * 7 + owner) * 13 + feature
    assert physical_offset == public_offset


def test_sm90_contract_schedule_is_selected_from_contraction_dimensions() -> None:
    schedule = select_sm90_contract_schedule(
        owner_extent=2048,
        streamed_extent=2048,
        producer_feature_extent=128,
        consumer_feature_extent=128,
    )

    assert schedule.family == "sm90_contract_fold_v1"
    assert (schedule.tile_m, schedule.tile_n, schedule.threads) == (128, 128, 384)

    short_domain_schedule = select_sm90_contract_schedule(
        owner_extent=19,
        streamed_extent=23,
        producer_feature_extent=128,
        consumer_feature_extent=128,
    )
    assert (short_domain_schedule.tile_m, short_domain_schedule.tile_n, short_domain_schedule.threads) == (
        64,
        128,
        256,
    )

    with pytest.raises(ValueError, match="one producer warpgroup"):
        Sm90ForwardPipelineSchedule(
            contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384),
            stages=2,
        )

    with pytest.raises(ValueError, match="consumer features 64 or 128"):
        select_sm90_contract_schedule(
            owner_extent=2048,
            streamed_extent=2048,
            producer_feature_extent=128,
            consumer_feature_extent=80,
        )


def _program_and_inputs(
    mutation: ScoreMutation,
    *,
    query_heads: int = 4,
    key_value_heads: int = 2,
    score_scale: float = 0.5,
) -> tuple[TensorProgram, dict[str, np.ndarray]]:
    batch, query_length, key_length = 2, 7, 9
    rng = np.random.default_rng(17)
    score_map = scaled_score_map(score_scale)
    inputs: dict[str, np.ndarray] = {
        "query": rng.normal(size=(batch, query_length, query_heads, 4)).astype(np.float32),
        "key": rng.normal(size=(batch, key_length, key_value_heads, 4)).astype(np.float32),
        "value": rng.normal(size=(batch, key_length, key_value_heads, 3)).astype(np.float32),
    }
    if mutation is ScoreMutation.CAUSAL:
        score_map = apply_causal_score_mask(score_map)
        inputs["query.position"] = np.arange(query_length, dtype=np.int32)
        inputs["key.position"] = np.arange(key_length, dtype=np.int32)
    else:
        if mutation is ScoreMutation.TANH_SOFTCAP:
            score_map = apply_tanh_softcap(score_map, cap=0.35)
        else:
            score_map = add_score_bias(
                score_map,
                axes=(
                    AttentionScoreAxis.HEAD,
                    AttentionScoreAxis.QUERY,
                    AttentionScoreAxis.KEY,
                ),
            )
            inputs["score.bias"] = rng.normal(size=(query_heads, query_length, key_length)).astype(np.float32) * 0.2
        score_map = apply_arbitrary_score_mask(score_map)
        mask = rng.random(size=(batch, query_heads, query_length, key_length)) > 0.45
        mask[..., 0] = True
        inputs["score.mask"] = mask

    source = build_attention_tensor_program(
        batch_size=batch,
        query_length=query_length,
        key_length=key_length,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=4,
        value_dimension=3,
        score_map=score_map,
        input_dtype=DType.FP32,
    )
    return source, inputs


def test_attention_source_is_only_contract_map_fold_and_derives_bounded_state() -> None:
    source, _ = _program_and_inputs(ScoreMutation.CAUSAL)
    schedule = StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2)

    generated = derive_streaming_attention(source, schedule=schedule)

    assert [type(operation) for operation in source.operations] == [
        ContractPrimitive,
        MapPrimitive,
        FoldPrimitive,
        MapPrimitive,
        MapPrimitive,
        FoldPrimitive,
        ContractPrimitive,
        MapPrimitive,
    ]
    assert isinstance(generated, NormalizedExpContractProgram)
    assert generated.region.fold_state.row_max.shape == (2, 4, 7)
    assert generated.region.fold_state.row_sum_exp.shape == (2, 4, 7)
    assert generated.region.consumer.output.shape == (2, 7, 4, 3)
    assert generated.materialized_values == source.outputs
    assert generated.schedule.stages == tuple(StreamingAttentionStage)
    assert not hasattr(generated, "backend")


def test_region_extraction_follows_dataflow_with_unrelated_interleaved_operations() -> None:
    source, _ = _program_and_inputs(ScoreMutation.CAUSAL)
    side_axis = source.inputs[0].axes[0]
    side_input = ProgramValue("side.input", (side_axis,), DType.FP32)
    side_output = ProgramValue("side.output", (side_axis,), DType.FP32)
    side_map = MapPrimitive(
        "unrelated side Map",
        (side_input,),
        side_output,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(side_input.name),
            scalar_constant(1.0),
        ),
    )
    mixed = TensorProgram(
        inputs=(*source.inputs, side_input),
        operations=(source.operations[0], side_map, *source.operations[1:]),
        outputs=(side_output, *source.outputs),
    )

    generated = derive_streaming_attention(
        mixed,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    assert generated.region.producer == source.operations[0]
    assert generated.region.finalizer == source.operations[-1]
    assert side_map not in (
        generated.region.producer,
        generated.region.score_map,
        generated.region.fold_state.maximum,
        generated.region.fold_state.center,
        generated.region.fold_state.exponential,
        generated.region.fold_state.normalizer,
        generated.region.consumer,
        generated.region.finalizer,
    )


@pytest.mark.parametrize("mutation", tuple(ScoreMutation))
def test_score_semantic_mutations_use_same_generator_and_match_materialized_program(
    mutation: ScoreMutation,
) -> None:
    source, inputs = _program_and_inputs(mutation)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    materialized = execute_tensor_program(source, inputs)["attention.output"]
    streamed = execute_streaming_attention(generated, inputs)
    difference = np.abs(streamed - materialized)

    assert float(np.max(difference)) < 2e-6
    assert float(np.mean(difference)) < 2e-7


def test_changed_score_map_changes_results_without_changing_streaming_structure() -> None:
    _, inputs = _program_and_inputs(ScoreMutation.BIAS_AND_MASK)
    softcap_source, softcap_inputs = _program_and_inputs(ScoreMutation.TANH_SOFTCAP)
    shared_inputs = {
        "query": inputs["query"],
        "key": inputs["key"],
        "value": inputs["value"],
        "score.mask": inputs["score.mask"],
    }
    softcap_inputs.update(shared_inputs)
    plain_score_map = apply_arbitrary_score_mask(scaled_score_map(0.5))
    plain_source = build_attention_tensor_program(
        batch_size=2,
        query_length=7,
        key_length=9,
        query_heads=4,
        key_value_heads=2,
        key_dimension=4,
        value_dimension=3,
        score_map=plain_score_map,
        input_dtype=DType.FP32,
    )
    schedule = StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2)

    plain = derive_streaming_attention(plain_source, schedule=schedule)
    softcap = derive_streaming_attention(softcap_source, schedule=schedule)
    plain_output = execute_streaming_attention(plain, shared_inputs)
    softcap_output = execute_streaming_attention(softcap, softcap_inputs)

    assert plain.schedule == softcap.schedule
    assert plain.region.fold_state == softcap.region.fold_state
    assert not np.allclose(plain_output, softcap_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("score_scale", [0.125, 0.5, 1.25])
def test_score_scale_mutation_uses_same_generator_and_matches_materialized_program(
    score_scale: float,
) -> None:
    source, inputs = _program_and_inputs(ScoreMutation.CAUSAL, score_scale=score_scale)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    materialized = execute_tensor_program(source, inputs)["attention.output"]
    streamed = execute_streaming_attention(generated, inputs)

    assert float(np.max(np.abs(streamed - materialized))) < 2e-6


def test_finalization_mutation_reuses_the_derived_fold_schedule() -> None:
    source, inputs = _program_and_inputs(ScoreMutation.CAUSAL)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )
    output_scale = 0.375
    scaled_finalize = replace(
        generated.region.finalizer,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            generated.region.finalizer.expression,
            scalar_constant(output_scale),
        ),
    )
    scaled_source = replace(source, operations=(*source.operations[:-1], scaled_finalize))
    scaled_generated = replace(
        generated,
        source=scaled_source,
        region=replace(generated.region, finalizer=scaled_finalize),
    )

    materialized = execute_tensor_program(scaled_source, inputs)["attention.output"]
    streamed = execute_streaming_attention(scaled_generated, inputs)
    baseline = execute_streaming_attention(generated, inputs)

    assert scaled_generated.region.producer == generated.region.producer
    assert scaled_generated.region.score_map == generated.region.score_map
    assert scaled_generated.region.consumer == generated.region.consumer
    assert scaled_generated.region.fold_state == generated.region.fold_state
    assert scaled_generated.schedule == generated.schedule
    assert float(np.max(np.abs(streamed - materialized))) < 2e-6
    assert float(np.max(np.abs(streamed - baseline * output_scale))) < 2e-6


@pytest.mark.parametrize("query_heads,key_value_heads", [(8, 2), (8, 4)])
def test_grouped_head_index_map_is_explicit_and_matches_expanded_reference(
    query_heads: int,
    key_value_heads: int,
) -> None:
    source, inputs = _program_and_inputs(
        ScoreMutation.CAUSAL,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
    )
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    key_mapping = generated.region.producer.index_maps_for_input(1)
    value_mapping = generated.region.consumer.index_maps_for_input(1)
    expected_mapping = AxisIndexMap(
        domain_axis=generated.region.producer.output.axes[1],
        operand_axis=generated.region.producer.inputs[1].axes[2],
        divisor=query_heads // key_value_heads,
    )
    assert key_mapping == (expected_mapping,)
    assert value_mapping[0].divisor == query_heads // key_value_heads

    ratio = query_heads // key_value_heads
    expanded_inputs = dict(inputs)
    expanded_inputs["key"] = np.repeat(inputs["key"], ratio, axis=2)
    expanded_inputs["value"] = np.repeat(inputs["value"], ratio, axis=2)
    query = inputs["query"].astype(np.float32)
    key = expanded_inputs["key"].astype(np.float32)
    value = expanded_inputs["value"].astype(np.float32)
    scores = np.einsum("bqhd,bkhd->bhqk", query, key) * 0.5
    query_positions = np.arange(query.shape[1])[:, None]
    key_positions = np.arange(key.shape[1])[None, :]
    scores = np.where(key_positions <= query_positions, scores, -np.inf)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    reference = np.einsum("bhqk,bkhd->bqhd", probabilities, value)

    streamed = execute_streaming_attention(generated, inputs)
    assert float(np.max(np.abs(streamed - reference))) < 2e-6


def test_head_sharing_variants_are_only_axis_index_relations() -> None:
    query_head = TensorAxis("query_head", 8)
    relations = (
        AxisIndexMap(query_head, TensorAxis("identity_head", 8)),
        AxisIndexMap(query_head, TensorAxis("grouped_head", 2), divisor=4),
        AxisIndexMap(query_head, TensorAxis("constant_head", 1), divisor=8),
    )

    assert tuple(relation.indices() for relation in relations) == (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )


def _h100_program(score_map):
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=2048,
        key_length=2048,
        query_heads=32,
        key_value_heads=8,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    logical = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )
    return replace(
        logical,
        schedule=Sm90ForwardPipelineSchedule(
            contract=Sm90ContractSchedule(tile_m=128, tile_n=128),
            stages=2,
        ),
    )


def test_h100_lowering_recovers_causal_scale_schedule_and_gqa_without_cuda() -> None:
    program = _h100_program(apply_causal_score_mask(scaled_score_map(0.125)))

    lowering = lower_h100_streaming_program(program)

    assert lowering.score_map.domain_traversal.value == "lower_triangular"
    assert lowering.score_map.expression == scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input(program.region.producer.output.name),
        scalar_constant(0.125),
    )
    assert lowering.relation_layout.batch_shape == (1,)
    assert lowering.relation_layout.relation_group_shape == (4,)
    assert lowering.relation_layout.relation_group_size == 4
    assert (
        lowering.schedule.tile_m,
        lowering.schedule.tile_n,
        lowering.schedule.stages,
    ) == (128, 128, 2)


def test_h100_lowering_preserves_softcap_ast_without_named_attention_dispatch() -> None:
    score_map = apply_tanh_softcap(apply_causal_score_mask(scaled_score_map(0.0625)), cap=16.0)
    lowering = lower_h100_streaming_program(_h100_program(score_map))

    assert lowering.score_map.domain_traversal.value == "lower_triangular"
    assert lowering.score_map.expression.kind is ScalarExpressionKind.MULTIPLY
    assert any(operand.kind is ScalarExpressionKind.TANH for operand in lowering.score_map.expression.operands)


def test_h100_lowering_preserves_finalizer_ast_without_changing_schedule() -> None:
    program = _h100_program(apply_causal_score_mask(scaled_score_map(0.125)))
    output_scale = 0.375
    scaled_finalize = replace(
        program.region.finalizer,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            program.region.finalizer.expression,
            scalar_constant(output_scale),
        ),
    )
    scaled_program = replace(program, region=replace(program.region, finalizer=scaled_finalize))

    baseline = lower_h100_streaming_program(program)
    scaled = lower_h100_streaming_program(scaled_program)

    assert baseline.finalizer_expression == program.region.finalizer.expression
    assert scaled.finalizer_expression == scaled_finalize.expression
    assert scaled.score_map == baseline.score_map
    assert scaled.schedule == baseline.schedule
    assert scaled.relation_layout == baseline.relation_layout


def test_h100_lowering_reports_tensor_mask_as_missing_auxiliary_emitter() -> None:
    program = _h100_program(apply_arbitrary_score_mask(scaled_score_map(0.125)))

    with pytest.raises(ValueError, match="lower-triangular comparison"):
        lower_h100_streaming_program(program)


def test_sm90_streaming_skeleton_owns_fold_map_and_domain_semantics() -> None:
    backend = Path(__file__).parents[1] / "src" / "tile_lifetime" / "backends" / "h100"
    physical_sources = tuple(
        (backend / filename).read_text() for filename in ("cute_streaming_base.py", "cute_streaming_sm90.py")
    )
    forbidden_dependencies = (
        "flash_attn.cute.softmax",
        "flash_attn.cute.mask",
        "apply_score_mod_inner",
        "AttentionMask",
        "Softmax.create",
    )

    for source in physical_sources:
        assert not any(dependency in source for dependency in forbidden_dependencies)
        assert "NormalizedExpFoldState" in source
        assert "DomainRestriction" in source
        assert "apply_score_map_inner" in source
        assert "finalize(output_scale=self.output_scale" in source


def test_sm90_normalized_exp_kernel_abi_keeps_score_scaling_in_scalar_ast() -> None:
    backend = Path(__file__).parents[1] / "src" / "tile_lifetime" / "backends" / "h100"

    normalized_exp = ast.parse((backend / "cute_normalized_exp.py").read_text())
    fold_state = next(
        node for node in normalized_exp.body if isinstance(node, ast.ClassDef) and node.name == "NormalizedExpFoldState"
    )
    fold_fields = {node.target.id for node in fold_state.body if isinstance(node, ast.AnnAssign)}
    score_map_helper = next(
        node
        for node in normalized_exp.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_score_map_inner"
    )
    score_map_vjp_helper = next(
        node
        for node in normalized_exp.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_score_map_vjp_inner"
    )
    assert "score_scale" not in fold_fields
    assert "score_scale" not in {argument.arg for argument in score_map_helper.args.args}
    assert "score_scale" not in {argument.arg for argument in score_map_vjp_helper.args.args}

    for filename, class_name in (
        ("cute_streaming_sm90.py", "ShuttleNormalizedExponentialContractSm90"),
        ("cute_normalized_exp_backward_sm90.py", "ShuttleNormalizedExponentialBackwardSm90"),
    ):
        module = ast.parse((backend / filename).read_text())
        kernel_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        call = next(node for node in kernel_class.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        assert "softmax_scale" not in {argument.arg for argument in call.args.args}

    emitter = ast.parse((backend / "cute_streaming_backward_emitter.py").read_text())
    forward_launch = next(
        node for node in emitter.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_launch"
    )
    scalar_arguments = {
        call.args[0].value
        for call in ast.walk(forward_launch)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_scalar_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    assert scalar_arguments == {"scale_log2"}


def test_normalized_exp_finalize_binds_register_state_before_child_region() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "tile_lifetime" / "backends" / "h100" / "cute_normalized_exp.py"
    ).read_text()

    assert "row_sum = self.row_sum" in source
    assert "row_max = self.row_max" in source
    assert "row_sum.store(utils.warp_reduce(row_sum.load()" in source
    assert "self.row_sum.store(utils.warp_reduce(self.row_sum.load()" not in source
    assert "init_val=row_sum[row] * prior_accumulator_scale[row]" in source
    assert "self.row_sum[row] * prior_accumulator_scale[row]" not in source
