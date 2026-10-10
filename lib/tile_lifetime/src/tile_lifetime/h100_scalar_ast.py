# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free validation and physical binding of SM90 scalar ASTs."""

import math
from dataclasses import dataclass, replace

from shuttle.ir import DType
from tile_lifetime.tensor_program import (
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    scalar_constant,
    scalar_expression_inputs,
)

_SCORE = "score"
_COTANGENT = "cotangent"
_OWNER_INDEX = "owner_index"
_STREAMED_INDEX = "streamed_index"
_NORMALIZED_OUTPUT = "normalized_output"


@dataclass(frozen=True)
class FinalizerStorageConversion:
    """An outer scalar conversion exactly implemented by the destination store."""

    residual_expression: ScalarExpression
    destination_dtype: DType
    semantics: ScalarConversionSemantics

    @property
    def canonical_record(self) -> dict[str, str]:
        return {
            "destination_dtype": self.destination_dtype.value,
            "semantics": self.semantics.value,
            "realization": "native_destination_store",
        }


def finalizer_storage_conversion(
    expression: ScalarExpression,
    *,
    destination_dtype: DType,
) -> FinalizerStorageConversion | None:
    """Split an outer conversion when the physical store has identical semantics."""
    if (
        expression.kind is not ScalarExpressionKind.CONVERT
        or expression.result_dtype is not destination_dtype
        or expression.conversion_semantics is not ScalarConversionSemantics.ROUND_NEAREST_EVEN
        or destination_dtype is not DType.BF16
    ):
        return None
    return FinalizerStorageConversion(expression.operands[0], destination_dtype, expression.conversion_semantics)


def factor_uniform_multiplicative_constant(expression: ScalarExpression) -> tuple[ScalarExpression, float]:
    """Factor a safe uniform constant for a terminal floating-point epilogue."""
    if _has_unsafe_factor_constant(expression):
        return expression, 1.0
    factor, residual = _uniform_multiplicative_factor(expression)
    if not math.isfinite(factor) or factor in (0.0, 1.0):
        return expression, 1.0
    return residual, factor


def _uniform_multiplicative_factor(expression: ScalarExpression) -> tuple[float, ScalarExpression]:
    if expression.kind is ScalarExpressionKind.CONSTANT:
        value = expression.constant
        if type(value) is not float:
            return 1.0, expression
        return value, scalar_constant(1.0)
    if expression.kind is ScalarExpressionKind.NEGATE:
        factor, residual = _uniform_multiplicative_factor(expression.operands[0])
        return -factor, residual
    if expression.kind not in (ScalarExpressionKind.MULTIPLY, ScalarExpressionKind.DIVIDE):
        return 1.0, expression

    lhs_factor, lhs = _uniform_multiplicative_factor(expression.operands[0])
    rhs_factor, rhs = _uniform_multiplicative_factor(expression.operands[1])
    if expression.kind is ScalarExpressionKind.DIVIDE and rhs_factor == 0.0:
        return 1.0, expression
    factor = lhs_factor * rhs_factor if expression.kind is ScalarExpressionKind.MULTIPLY else lhs_factor / rhs_factor
    if not math.isfinite(factor):
        return 1.0, expression
    if _is_one(lhs):
        if expression.kind is ScalarExpressionKind.MULTIPLY:
            return factor, rhs
        return factor, replace(expression, operands=(scalar_constant(1.0), rhs))
    if _is_one(rhs):
        return factor, lhs
    return factor, replace(expression, operands=(lhs, rhs))


def _is_one(expression: ScalarExpression) -> bool:
    return expression.kind is ScalarExpressionKind.CONSTANT and expression.constant == 1.0


def _has_unsafe_factor_constant(expression: ScalarExpression) -> bool:
    if expression.kind is ScalarExpressionKind.CONSTANT:
        value = expression.constant
        return type(value) is not float or value == 0.0 or not math.isfinite(value)
    return any(_has_unsafe_factor_constant(operand) for operand in expression.operands)


def score_input_roles(
    expression: ScalarExpression,
    *,
    score_input: str,
    cotangent_input: str | None = None,
    owner_inputs: tuple[str, ...] = (),
    streamed_inputs: tuple[str, ...] = (),
) -> dict[str, str]:
    """Bind graph scalar inputs to generic score-hook coordinates."""
    if len(owner_inputs) > 1 or len(streamed_inputs) > 1:
        raise ValueError("the current CuTe scalar hook supports at most one owner and one streamed coordinate input")
    inputs = scalar_expression_inputs(expression)
    roles = {score_input: _SCORE} if score_input in inputs else {}
    if cotangent_input is not None and cotangent_input in inputs:
        roles[cotangent_input] = _COTANGENT
    roles.update({name: _OWNER_INDEX for name in owner_inputs if name in inputs})
    roles.update({name: _STREAMED_INDEX for name in streamed_inputs if name in inputs})
    missing = inputs - set(roles)
    if missing:
        raise ValueError(f"CuTe scalar AST has no physical binding for inputs {sorted(missing)}")
    return roles


def normalized_finalizer_expression(
    expression: ScalarExpression,
    *,
    accumulator_input: str,
    normalizer_input: str,
) -> ScalarExpression:
    """Bind a source finalizer to the normalized value produced by the online Fold."""
    source_inputs = scalar_expression_inputs(expression)
    if source_inputs == {accumulator_input}:

        def bind_accumulator(node: ScalarExpression) -> ScalarExpression:
            if node.kind is ScalarExpressionKind.INPUT:
                assert node.input_name == accumulator_input
                return ScalarExpression(kind=ScalarExpressionKind.INPUT, input_name=_NORMALIZED_OUTPUT)
            return ScalarExpression(
                kind=node.kind,
                operands=tuple(bind_accumulator(operand) for operand in node.operands),
                input_name=node.input_name,
                constant=node.constant,
                result_dtype=node.result_dtype,
                conversion_semantics=node.conversion_semantics,
            )

        rewritten = bind_accumulator(expression)
        validate_cute_scalar_expression(rewritten)
        return rewritten

    if source_inputs != {accumulator_input, normalizer_input}:
        raise ValueError(
            "the current CuTe finalizer hook requires the weighted Contract, with optional normalized-exp state"
        )

    quotient_count = 0

    def rewrite(node: ScalarExpression) -> ScalarExpression:
        nonlocal quotient_count
        if node.kind is ScalarExpressionKind.DIVIDE:
            numerator, denominator = node.operands
            if (
                numerator.kind is ScalarExpressionKind.INPUT
                and numerator.input_name == accumulator_input
                and denominator.kind is ScalarExpressionKind.INPUT
                and denominator.input_name == normalizer_input
            ):
                quotient_count += 1
                return ScalarExpression(kind=ScalarExpressionKind.INPUT, input_name=_NORMALIZED_OUTPUT)
        if node.kind is ScalarExpressionKind.INPUT and node.input_name in (accumulator_input, normalizer_input):
            raise ValueError(
                "the current CuTe finalizer hook requires accumulator/normalizer uses to factor through their quotient"
            )
        return ScalarExpression(
            kind=node.kind,
            operands=tuple(rewrite(operand) for operand in node.operands),
            input_name=node.input_name,
            constant=node.constant,
            result_dtype=node.result_dtype,
            conversion_semantics=node.conversion_semantics,
        )

    rewritten = rewrite(expression)
    if quotient_count < 1 or scalar_expression_inputs(rewritten) != {_NORMALIZED_OUTPUT}:
        raise ValueError("the current CuTe finalizer hook requires normalized accumulator quotient inputs")
    validate_cute_scalar_expression(rewritten)
    return rewritten


def validate_cute_scalar_expression(expression: ScalarExpression) -> None:
    """Reject scalar operations outside the closed CuTe codegen set."""
    supported = {
        ScalarExpressionKind.INPUT,
        ScalarExpressionKind.CONSTANT,
        ScalarExpressionKind.CONVERT,
        ScalarExpressionKind.ADD,
        ScalarExpressionKind.SUBTRACT,
        ScalarExpressionKind.MULTIPLY,
        ScalarExpressionKind.DIVIDE,
        ScalarExpressionKind.NEGATE,
        ScalarExpressionKind.EXP,
        ScalarExpressionKind.LOG,
        ScalarExpressionKind.RSQRT,
        ScalarExpressionKind.TANH,
        ScalarExpressionKind.MAXIMUM,
        ScalarExpressionKind.MINIMUM,
        ScalarExpressionKind.LESS,
        ScalarExpressionKind.LESS_EQUAL,
        ScalarExpressionKind.GREATER,
        ScalarExpressionKind.GREATER_EQUAL,
        ScalarExpressionKind.EQUAL,
        ScalarExpressionKind.NOT_EQUAL,
        ScalarExpressionKind.SELECT,
    }
    if expression.kind not in supported:
        raise ValueError(f"unsupported CuTe scalar operation {expression.kind.value!r}")
    for operand in expression.operands:
        validate_cute_scalar_expression(operand)
