# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from shuttle.ir import DType
from tile_lifetime.h100_scalar_ast import (
    finalizer_storage_conversion,
    normalized_finalizer_expression,
    score_input_roles,
)
from tile_lifetime.tensor_program import (
    ScalarConversionSemantics,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_expression_inputs,
    scalar_input,
    scalar_unary,
)


def test_finalizer_ast_factors_through_the_online_normalized_output() -> None:
    quotient = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input("weighted"),
        scalar_input("normalizer"),
    )
    expression = scalar_binary(ScalarExpressionKind.ADD, quotient, scalar_constant(0.25))

    lowered = normalized_finalizer_expression(
        expression,
        accumulator_input="weighted",
        normalizer_input="normalizer",
    )

    assert lowered.kind is ScalarExpressionKind.ADD
    assert scalar_expression_inputs(lowered) == {"normalized_output"}


def test_finalizer_ast_preserves_terminal_typed_conversion() -> None:
    quotient = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input("weighted"),
        scalar_input("normalizer"),
    )
    expression = scalar_convert(
        quotient,
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )

    lowered = normalized_finalizer_expression(
        expression,
        accumulator_input="weighted",
        normalizer_input="normalizer",
    )

    assert lowered.kind is ScalarExpressionKind.CONVERT
    assert lowered.result_dtype is DType.BF16
    assert lowered.conversion_semantics is ScalarConversionSemantics.ROUND_NEAREST_EVEN
    assert scalar_expression_inputs(lowered) == {"normalized_output"}


def test_terminal_bf16_conversion_is_realized_by_destination_storage() -> None:
    residual = scalar_binary(ScalarExpressionKind.ADD, scalar_input("normalized_output"), scalar_constant(0.25))
    expression = scalar_convert(
        residual,
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )

    legalization = finalizer_storage_conversion(expression, destination_dtype=DType.BF16)

    assert legalization is not None
    assert legalization.residual_expression == residual
    assert legalization.canonical_record == {
        "destination_dtype": "bf16",
        "semantics": "round_nearest_even",
        "realization": "native_destination_store",
    }
    assert finalizer_storage_conversion(expression, destination_dtype=DType.FP32) is None


def test_finalizer_ast_reuses_the_same_normalized_output_in_a_branching_expression() -> None:
    quotient = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input("weighted"),
        scalar_input("normalizer"),
    )
    expression = scalar_binary(
        ScalarExpressionKind.ADD,
        scalar_unary(ScalarExpressionKind.TANH, quotient),
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(0.25), quotient),
    )

    lowered = normalized_finalizer_expression(
        expression,
        accumulator_input="weighted",
        normalizer_input="normalizer",
    )

    assert lowered.kind is ScalarExpressionKind.ADD
    assert scalar_expression_inputs(lowered) == {"normalized_output"}


def test_finalizer_ast_binds_an_already_normalized_contract_result() -> None:
    expression = scalar_convert(
        scalar_input("weighted"),
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )

    lowered = normalized_finalizer_expression(
        expression,
        accumulator_input="weighted",
        normalizer_input="normalizer",
    )

    assert lowered.kind is ScalarExpressionKind.CONVERT
    assert lowered.operands == (scalar_input("normalized_output"),)


def test_finalizer_ast_accepts_generic_exponential_composition() -> None:
    expression = scalar_unary(
        ScalarExpressionKind.EXP,
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input("weighted"),
            scalar_input("normalizer"),
        ),
    )

    lowered = normalized_finalizer_expression(
        expression,
        accumulator_input="weighted",
        normalizer_input="normalizer",
    )

    assert lowered.kind is ScalarExpressionKind.EXP
    assert scalar_expression_inputs(lowered) == {"normalized_output"}


def test_vjp_role_binding_does_not_require_an_unused_primal_score() -> None:
    expression = scalar_input("mapped_score_cotangent")

    roles = score_input_roles(
        expression,
        score_input="raw_score",
        cotangent_input="mapped_score_cotangent",
    )

    assert roles == {"mapped_score_cotangent": "cotangent"}
