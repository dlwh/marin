# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from shuttle.ir import DType
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.streaming_attention_backward_reference import (
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    execute_tensor_program,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
)
from tile_lifetime.tensor_program_rewrites import (
    fold_invariant_division_boundaries,
    fold_widening_converts_into_contracts,
    remove_dead_operations,
    sink_fold_invariant_division_through_contract,
)


def test_dead_full_shape_identity_does_not_change_fold_algebra() -> None:
    row = TensorAxis(0, 2, "row")
    feature = TensorAxis(1, 4, "feature")
    source = ProgramValue("source", (row, feature), DType.FP32)
    dead_identity = ProgramValue("dead.identity", source.axes, DType.FP32)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    identity = MapPrimitive("dead reducer identity", (), dead_identity, scalar_constant(0.0))
    fold = FoldPrimitive("sum", source, reduced, (feature,), FoldReducer.SUM, DType.FP32)
    program = TensorProgram((source,), (identity, fold), (reduced,), NumericalPolicy.FAST)

    normalized = remove_dead_operations(program)

    assert normalized.operations == (fold,)


def test_live_full_shape_constant_is_not_erased_as_a_fold_identity() -> None:
    row = TensorAxis(0, 2, "row")
    feature = TensorAxis(1, 4, "feature")
    source = ProgramValue("source", (row, feature), DType.FP32)
    live_constant = ProgramValue("live.constant", source.axes, DType.FP32)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    constant = MapPrimitive("live constant", (), live_constant, scalar_constant(0.0))
    fold = FoldPrimitive("sum", source, reduced, (feature,), FoldReducer.SUM, DType.FP32)
    program = TensorProgram((source,), (constant, fold), (reduced, live_constant), NumericalPolicy.FAST)

    assert remove_dead_operations(program) == program


def _normalized_contract_program(*, policy: NumericalPolicy, affine_offset: float) -> TensorProgram:
    row = TensorAxis(0, 3, "row")
    reduction = TensorAxis(1, 5, "reduction")
    feature = TensorAxis(2, 4, "feature")
    raw = ProgramValue("raw", (row, reduction), DType.FP32)
    values = ProgramValue("values", (reduction, feature), DType.FP32)
    shifted = ProgramValue("shifted", raw.axes, DType.FP32)
    normalizer = ProgramValue("normalizer", (row,), DType.FP32)
    normalized = ProgramValue("normalized", raw.axes, DType.FP32)
    output = ProgramValue("output", (row, feature), DType.FP32)
    shift = MapPrimitive(
        "affine numerator",
        (raw,),
        shifted,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(raw.name),
            scalar_constant(affine_offset),
        ),
    )
    fold = FoldPrimitive("row Fold", shifted, normalizer, (reduction,), FoldReducer.SUM, DType.FP32)
    divide = MapPrimitive(
        "Fold-dependent division",
        (shifted, normalizer),
        normalized,
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input(shifted.name),
            scalar_input(normalizer.name),
        ),
    )
    contract = ContractPrimitive("linear consumer", (normalized, values), output, (reduction,), DType.FP32)
    return TensorProgram((raw, values), (shift, fold, divide, contract), (output,), policy)


def test_fast_fold_invariant_division_sinks_after_linear_contract() -> None:
    program = _normalized_contract_program(policy=NumericalPolicy.FAST, affine_offset=0.25)
    boundaries = fold_invariant_division_boundaries(program)
    rewrite = sink_fold_invariant_division_through_contract(program)
    producer_by_value = {
        output.name: operation for operation in rewrite.program.operations for output in primitive_outputs(operation)
    }
    finalizer = producer_by_value[rewrite.program.outputs[0].name]

    assert len(rewrite.rewrites) == 1
    assert len(boundaries) == 1
    assert boundaries[0].contract_operation == 3
    assert boundaries[0].division_operation == 2
    assert boundaries[0].normalized_operand == 0
    assert boundaries[0].numerator.name == "shifted"
    assert boundaries[0].denominator.name == "normalizer"
    assert boundaries[0].fold_state.name == "normalizer"
    assert boundaries[0].denominator_path == ()
    assert rewrite.source_fingerprint == program.semantic_fingerprint
    assert rewrite.program.semantic_fingerprint != program.semantic_fingerprint
    assert isinstance(finalizer, MapPrimitive)
    assert finalizer.expression.kind is ScalarExpressionKind.DIVIDE
    assert any(isinstance(producer_by_value[value.name], ContractPrimitive) for value in finalizer.inputs)
    assert any(isinstance(producer_by_value[value.name], FoldPrimitive) for value in finalizer.inputs)

    rng = np.random.default_rng(7)
    inputs = {
        "raw": rng.uniform(0.5, 1.5, size=program.inputs[0].shape).astype(np.float32),
        "values": rng.normal(size=program.inputs[1].shape).astype(np.float32),
    }
    expected = execute_tensor_program(program, inputs)["output"]
    actual = execute_tensor_program(rewrite.program, inputs)["output"]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_source_ordered_fold_invariant_division_is_not_reassociated() -> None:
    program = _normalized_contract_program(policy=NumericalPolicy.SOURCE_ORDERED, affine_offset=-0.125)
    rewrite = sink_fold_invariant_division_through_contract(program)

    assert rewrite.program is program
    assert not rewrite.rewrites


def test_scalar_numerator_perturbation_keeps_generic_rewrite_structure() -> None:
    first = sink_fold_invariant_division_through_contract(
        _normalized_contract_program(policy=NumericalPolicy.FAST, affine_offset=0.0)
    )
    second = sink_fold_invariant_division_through_contract(
        _normalized_contract_program(policy=NumericalPolicy.FAST, affine_offset=0.375)
    )

    assert first.program.semantic_fingerprint != second.program.semantic_fingerprint
    assert tuple(type(operation) for operation in first.program.operations) == tuple(
        type(operation) for operation in second.program.operations
    )
    assert tuple(rewrite.name for rewrite in first.rewrites) == tuple(rewrite.name for rewrite in second.rewrites)


def test_imported_jax_normalization_sinking_preserves_values_and_relations() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=4)
    rng = np.random.default_rng(19)
    query = rng.normal(size=(config.batch, config.query_length, config.query_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    key = rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    value = rng.normal(size=key.shape).astype(jnp.bfloat16)
    program = jax_function_to_tensor_program(
        causal_gqa_attention(config),
        query,
        key,
        value,
        numerical_policy=NumericalPolicy.FAST,
    )
    boundaries = fold_invariant_division_boundaries(program)
    rewrite = sink_fold_invariant_division_through_contract(program)
    inputs = {"argument.0": query, "argument.1": key, "argument.2": value}
    producer_by_value = {
        output.name: operation for operation in rewrite.program.operations for output in primitive_outputs(operation)
    }

    assert len(rewrite.rewrites) == 1
    assert len(boundaries) == 1
    assert boundaries[0].numerator.dtype is boundaries[0].denominator.dtype
    assert boundaries[0].fold_state.dtype is DType.FP32
    assert all(kind in {"convert", "relation"} for _, kind, _, _ in boundaries[0].denominator_path)
    assert any(
        isinstance(operation, ContractPrimitive)
        and any(
            isinstance(producer_by_value.get(input_value.name), MapPrimitive)
            and producer_by_value[input_value.name].expression.kind is ScalarExpressionKind.EXP
            for input_value in operation.inputs
        )
        for operation in rewrite.program.operations
    )
    expected = execute_tensor_program(program, inputs)[program.outputs[0].name]
    actual = execute_tensor_program(rewrite.program, inputs)[rewrite.program.outputs[0].name]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_private_widening_converts_fold_into_contract_accumulation_exactly() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=4)
    rng = np.random.default_rng(29)
    query = rng.normal(size=(config.batch, config.query_length, config.query_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    key = rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
        jnp.bfloat16
    )
    value = rng.normal(size=key.shape).astype(jnp.bfloat16)
    program = jax_function_to_tensor_program(
        causal_gqa_attention(config),
        query,
        key,
        value,
        numerical_policy=NumericalPolicy.FAST,
    )

    rewrite = fold_widening_converts_into_contracts(program)

    assert tuple(explanation.name for explanation in rewrite.rewrites) == ("fold_exact_widening_convert_into_contract",)
    producers = {
        output: operation for operation in rewrite.program.operations for output in primitive_outputs(operation)
    }
    assert not any(
        isinstance(producers.get(operand), MapPrimitive)
        and producers[operand].expression.kind is ScalarExpressionKind.CONVERT
        for operation in rewrite.program.operations
        if isinstance(operation, ContractPrimitive)
        for operand in operation.inputs
    )
    inputs = {"argument.0": query, "argument.1": key, "argument.2": value}
    expected = execute_tensor_program(program, inputs)[program.outputs[0].name]
    actual = execute_tensor_program(rewrite.program, inputs)[rewrite.program.outputs[0].name]
    np.testing.assert_array_equal(actual, expected)


def test_shared_widening_convert_remains_explicit() -> None:
    row = TensorAxis(0, 2, "row")
    reduction = TensorAxis(1, 3, "reduction")
    feature = TensorAxis(2, 4, "feature")
    narrow = ProgramValue("narrow", (row, reduction), DType.BF16)
    right = ProgramValue("right", (reduction, feature), DType.FP32)
    wide = ProgramValue("wide", narrow.axes, DType.FP32)
    first = ProgramValue("first", (row, feature), DType.FP32)
    second = ProgramValue("second", (row, feature), DType.FP32)
    conversion = MapPrimitive(
        "shared widen",
        (narrow,),
        wide,
        scalar_convert(
            scalar_input(narrow.name),
            result_dtype=DType.FP32,
            semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        ),
    )
    program = TensorProgram(
        (narrow, right),
        (
            conversion,
            ContractPrimitive("first Contract", (wide, right), first, (reduction,), DType.FP32),
            ContractPrimitive("second Contract", (wide, right), second, (reduction,), DType.FP32),
        ),
        (first, second),
        NumericalPolicy.FAST,
    )

    rewrite = fold_widening_converts_into_contracts(program)

    assert rewrite.program is program
    assert not rewrite.rewrites
