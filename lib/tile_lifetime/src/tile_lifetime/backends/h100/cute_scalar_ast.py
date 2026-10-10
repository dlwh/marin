# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower closed tensor-program scalar ASTs to CuTe compile-time callables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cutlass
import cutlass.cute as cute

from shuttle.ir import DType
from tile_lifetime.h100_scalar_ast import validate_cute_scalar_expression
from tile_lifetime.tensor_program import (
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    scalar_expression_inputs,
)

_SCORE = "score"
_COTANGENT = "cotangent"
_BATCH_INDEX = "batch_index"
_RELATION_INDEX = "relation_index"
_OWNER_INDEX = "owner_index"
_STREAMED_INDEX = "streamed_index"
_NORMALIZED_OUTPUT = "normalized_output"


def cute_score_map(
    expression: ScalarExpression,
    input_roles: Mapping[str, str],
):
    """Create a CuTe score callable from one closed scalar expression."""
    _validate_roles(expression, input_roles, {_SCORE, _BATCH_INDEX, _RELATION_INDEX, _OWNER_INDEX, _STREAMED_INDEX})

    @cute.jit
    def generated_score_map(scores, batch_idx, relation_idx, owner_idx, streamed_idx, seqlen_info, aux_tensors):
        del seqlen_info, aux_tensors
        values = {
            _SCORE: scores,
            _BATCH_INDEX: batch_idx,
            _RELATION_INDEX: relation_idx,
            _OWNER_INDEX: owner_idx,
            _STREAMED_INDEX: streamed_idx,
        }
        return _evaluate(expression, {name: values[role] for name, role in input_roles.items()}, cute)

    return generated_score_map


def cute_score_map_vjp(
    expression: ScalarExpression,
    input_roles: Mapping[str, str],
):
    """Create the reverse score callable from the autodiff-produced VJP AST."""
    _validate_roles(
        expression,
        input_roles,
        {_SCORE, _COTANGENT, _BATCH_INDEX, _RELATION_INDEX, _OWNER_INDEX, _STREAMED_INDEX},
    )

    @cute.jit
    def generated_score_map_vjp(
        gradient,
        scores,
        batch_idx,
        relation_idx,
        owner_idx,
        streamed_idx,
        seqlen_info,
        aux_tensors,
    ):
        del seqlen_info, aux_tensors
        values = {
            _SCORE: scores,
            _COTANGENT: gradient,
            _BATCH_INDEX: batch_idx,
            _RELATION_INDEX: relation_idx,
            _OWNER_INDEX: owner_idx,
            _STREAMED_INDEX: streamed_idx,
        }
        return _evaluate(expression, {name: values[role] for name, role in input_roles.items()}, cute)

    return generated_score_map_vjp


def cute_keyword_score_map_adapter(score_map):
    """Adapt the generic positional score hook to the imported CuTe helper ABI."""

    @cute.jit
    def keyword_score_map(scores, batch_idx, relation_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return score_map(scores, batch_idx, relation_idx, q_idx, kv_idx, seqlen_info, aux_tensors)

    return keyword_score_map


def cute_keyword_score_map_vjp_adapter(score_map_vjp):
    """Adapt the generic positional pullback hook to the imported CuTe helper ABI."""

    @cute.jit
    def keyword_score_map_vjp(
        gradient,
        scores,
        batch_idx,
        relation_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors,
    ):
        return score_map_vjp(
            gradient,
            scores,
            batch_idx,
            relation_idx,
            q_idx,
            kv_idx,
            seqlen_info,
            aux_tensors,
        )

    return keyword_score_map_vjp


def cute_finalizer_map(expression: ScalarExpression):
    """Create an elementwise post-normalization finalizer callable."""
    _validate_roles(expression, {_NORMALIZED_OUTPUT: _NORMALIZED_OUTPUT}, {_NORMALIZED_OUTPUT})

    @cute.jit
    def generated_finalizer(value):
        return _evaluate(expression, {_NORMALIZED_OUTPUT: value}, cute)

    return generated_finalizer


def cute_unary_scalar_map(expression: ScalarExpression, *, input_name: str):
    """Create a unary callable for a generic tensor-program ``Map`` AST."""
    _validate_roles(expression, {input_name: "value"}, {"value"})

    @cute.jit
    def generated_unary_map(value):
        return _evaluate(expression, {input_name: value}, cute)

    return generated_unary_map


def evaluate_cute_scalar_expression(expression: ScalarExpression, bindings: Mapping[str, Any]):
    """Evaluate a validated scalar AST while generating CuTe device code."""
    validate_cute_scalar_expression(expression)
    return _evaluate(expression, bindings, cute)


def cute_post_normalization_vjp(
    expression: ScalarExpression,
    *,
    normalized_output_input: str,
    cotangent_input: str,
):
    """Create the elementwise VJP callable for a post-normalization Map."""
    expression_inputs = scalar_expression_inputs(expression)
    input_roles = {
        name: role
        for name, role in (
            (normalized_output_input, _NORMALIZED_OUTPUT),
            (cotangent_input, _COTANGENT),
        )
        if name in expression_inputs
    }
    _validate_roles(expression, input_roles, {_NORMALIZED_OUTPUT, _COTANGENT})

    @cute.jit
    def generated_post_normalization_vjp(cotangent, normalized_output):
        values = {
            _NORMALIZED_OUTPUT: normalized_output,
            _COTANGENT: cotangent,
        }
        return _evaluate(expression, {name: values[role] for name, role in input_roles.items()}, cute)

    return generated_post_normalization_vjp


def _validate_roles(expression: ScalarExpression, input_roles: Mapping[str, str], supported_roles: set[str]) -> None:
    if scalar_expression_inputs(expression) != set(input_roles):
        raise ValueError("CuTe scalar input bindings must exactly cover the scalar AST")
    unsupported_roles = set(input_roles.values()) - supported_roles
    if unsupported_roles:
        raise ValueError(f"unsupported CuTe scalar input roles {sorted(unsupported_roles)}")
    validate_cute_scalar_expression(expression)


def _evaluate(expression: ScalarExpression, bindings: Mapping[str, Any], cute: Any):
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return bindings[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    operands = tuple(_evaluate(operand, bindings, cute) for operand in expression.operands)
    if kind is ScalarExpressionKind.CONVERT:
        assert expression.result_dtype is not None and expression.conversion_semantics is not None
        if expression.result_dtype is DType.FP32 and expression.conversion_semantics in (
            ScalarConversionSemantics.EXACT,
            ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        ):
            return _convert_value(operands[0], cutlass.Float32, cute)
        if (
            expression.result_dtype is DType.BF16
            and expression.conversion_semantics is ScalarConversionSemantics.ROUND_NEAREST_EVEN
        ):
            return _convert_value(operands[0], cutlass.BFloat16, cute)
        if expression.result_dtype is DType.INT32 and expression.conversion_semantics is ScalarConversionSemantics.EXACT:
            return _convert_value(operands[0], cutlass.Int32, cute)
        if expression.result_dtype is DType.BOOL and expression.conversion_semantics is ScalarConversionSemantics.EXACT:
            return _convert_value(operands[0], cutlass.Boolean, cute)
        raise ValueError(
            "unsupported CuTe scalar conversion "
            f"{expression.conversion_semantics.value} to {expression.result_dtype.value}"
        )
    if kind is ScalarExpressionKind.ADD:
        return operands[0] + operands[1]
    if kind is ScalarExpressionKind.SUBTRACT:
        return operands[0] - operands[1]
    if kind is ScalarExpressionKind.MULTIPLY:
        return operands[0] * operands[1]
    if kind is ScalarExpressionKind.DIVIDE:
        return operands[0] / operands[1]
    if kind is ScalarExpressionKind.NEGATE:
        return -operands[0]
    if kind is ScalarExpressionKind.EXP:
        return cute.math.exp(operands[0], fastmath=True)
    if kind is ScalarExpressionKind.LOG:
        return cute.math.log(operands[0], fastmath=True)
    if kind is ScalarExpressionKind.RSQRT:
        return cute.math.rsqrt(operands[0], fastmath=True)
    if kind is ScalarExpressionKind.TANH:
        # CuTe's transcendental overload does not widen BF16 inputs.  JAX's
        # scalar semantics compute tanh in FP32 before the typed Map stores its
        # result, so make that promotion explicit in generated device code.
        return cute.math.tanh(_convert_value(operands[0], cutlass.Float32, cute), fastmath=True)
    if kind is ScalarExpressionKind.MAXIMUM:
        return cute.where(operands[0] >= operands[1], operands[0], operands[1])
    if kind is ScalarExpressionKind.MINIMUM:
        return cute.where(operands[0] <= operands[1], operands[0], operands[1])
    if kind is ScalarExpressionKind.LESS:
        return operands[0] < operands[1]
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return operands[0] <= operands[1]
    if kind is ScalarExpressionKind.GREATER:
        return operands[0] > operands[1]
    if kind is ScalarExpressionKind.GREATER_EQUAL:
        return operands[0] >= operands[1]
    if kind is ScalarExpressionKind.EQUAL:
        return operands[0] == operands[1]
    if kind is ScalarExpressionKind.NOT_EQUAL:
        return operands[0] != operands[1]
    if kind is ScalarExpressionKind.SELECT:
        return cute.where(operands[0], operands[1], operands[2])
    raise ValueError(f"unsupported CuTe scalar operation {kind.value!r}")


def _convert_value(value: Any, dtype: Any, cute: Any):
    """Preserve vector SSA shape while applying one scalar element conversion."""
    if isinstance(value, cute.TensorSSA):
        return value.to(dtype)
    return dtype(value)
