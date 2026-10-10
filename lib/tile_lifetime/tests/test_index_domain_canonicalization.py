# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.autodiff import differentiate_tensor_program
from tile_lifetime.index_domain_canonicalization import canonicalize_negative_infinity_domain
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.tensor_program import (
    FoldPrimitive,
    FoldReducer,
    IndexExpressionKind,
    IndexPredicateKind,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    scalar_input,
    serialize_tensor_program,
)


def _masked_scores(scores: jax.Array) -> jax.Array:
    query = jnp.arange(scores.shape[-2])[:, None]
    key = jnp.arange(scores.shape[-1])[None, :]
    valid = jnp.broadcast_to(key <= query, scores.shape)
    return jax.lax.select(valid, scores, jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype))


def _masked_scores_with_views(scores: jax.Array) -> jax.Array:
    query = jnp.arange(scores.shape[-2])[:, None]
    key = jnp.arange(scores.shape[-1])[None, :]
    valid = jax.lax.stop_gradient((key <= query).T).T
    valid = jnp.broadcast_to(valid.reshape((1, scores.shape[-2], scores.shape[-1])), scores.shape)
    return jax.lax.select(valid, scores, jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype))


def _selected_value(program: TensorProgram) -> ProgramValue:
    select = next(
        operation
        for operation in program.operations
        if isinstance(operation, MapPrimitive) and operation.expression.kind is ScalarExpressionKind.SELECT
    )
    return select.output


def test_jax_index_mask_canonicalization_preserves_exact_values_and_domain() -> None:
    scores = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    program = jax_function_to_tensor_program(_masked_scores, scores)
    masked = _selected_value(program)
    restriction = canonicalize_negative_infinity_domain(program, masked, streamed_axes=(masked.axes[-1],))

    observable = TensorProgram(program.inputs, program.operations, (masked, restriction.selected_value))
    actual = execute_tensor_program(observable, {program.inputs[0].name: scores})
    predicate_value = ProgramValue("predicate", masked.axes, DType.BOOL)
    predicate_program = TensorProgram(
        (),
        (
            RelationPrimitive(
                "materialize domain",
                (),
                (predicate_value,),
                masked.axes,
                (),
                result_expressions=(restriction.predicate,),
            ),
        ),
        (predicate_value,),
    )
    valid = execute_tensor_program(predicate_program, {})[predicate_value.name]
    reconstructed = np.where(valid, actual[restriction.selected_value.name], -np.inf)

    np.testing.assert_array_equal(reconstructed, actual[masked.name])
    np.testing.assert_array_equal(valid[0], np.tril(np.ones((4, 4), dtype=np.bool_)))
    assert restriction.affine_predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((2, 1),), "constant": 0},
        "right": {"coefficients": ((1, 1),), "constant": 0},
    }
    assert restriction.source_fingerprint == program.semantic_fingerprint
    assert restriction.predicate_operation_ordinals


def test_index_mask_canonicalization_tolerates_relation_only_views() -> None:
    scores = np.zeros((2, 4, 4), dtype=np.float32)
    plain = jax_function_to_tensor_program(_masked_scores, scores)
    viewed = jax_function_to_tensor_program(_masked_scores_with_views, scores)
    plain_masked = _selected_value(plain)
    viewed_masked = _selected_value(viewed)

    plain_domain = canonicalize_negative_infinity_domain(plain, plain_masked, streamed_axes=(plain_masked.axes[-1],))
    viewed_domain = canonicalize_negative_infinity_domain(viewed, viewed_masked, streamed_axes=(viewed_masked.axes[-1],))

    assert viewed_domain.affine_predicate.canonical_record == plain_domain.affine_predicate.canonical_record
    assert viewed_domain.source_fingerprint != plain_domain.source_fingerprint


def test_index_mask_canonicalization_rejects_non_infinite_fallback() -> None:
    def finite_fallback(scores: jax.Array) -> jax.Array:
        query = jnp.arange(scores.shape[-2])[:, None]
        key = jnp.arange(scores.shape[-1])[None, :]
        valid = jnp.broadcast_to(key <= query, scores.shape)
        return jax.lax.select(valid, scores, jnp.zeros_like(scores))

    scores = np.zeros((3, 4), dtype=np.float32)
    program = jax_function_to_tensor_program(finite_fallback, scores)
    masked = _selected_value(program)

    with pytest.raises(ValueError, match="exactly one negative-infinity rejected branch"):
        canonicalize_negative_infinity_domain(program, masked, streamed_axes=(masked.axes[-1],))


def test_predicated_map_roundtrip_executes_affine_domain_and_fill() -> None:
    row = TensorAxis(0, 3, "row")
    column = TensorAxis(1, 4, "column")
    source = ProgramValue("source", (row, column), DType.FP32)
    restricted = ProgramValue("restricted", source.axes, source.dtype)
    predicate = index_compare(
        IndexPredicateKind.LESS_EQUAL,
        index_axis(column),
        index_binary(IndexExpressionKind.ADD, index_axis(row), index_constant(1)),
    )
    operation = MapPrimitive(
        "affine half-space",
        (source,),
        restricted,
        scalar_input(source.name),
        predicate=predicate,
        rejected_value=-np.inf,
    )
    program = TensorProgram((source,), (operation,), (restricted,))
    decoded = deserialize_tensor_program(serialize_tensor_program(program))

    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    actual = execute_tensor_program(decoded, {decoded.inputs[0].name: values})[decoded.outputs[0].name]
    expected = np.where(np.arange(4)[None, :] <= np.arange(3)[:, None] + 1, values, -np.inf)

    np.testing.assert_array_equal(actual, expected)
    assert decoded.semantic_fingerprint == program.semantic_fingerprint
    assert program.canonical_record["operations"][0]["kind"] == "map"
    assert (
        program.canonical_record["operations"][0]["predicate"] == decoded.canonical_record["operations"][0]["predicate"]
    )
    assert "domain_restriction" not in serialize_tensor_program(program)
    zero_filled = TensorProgram(
        (source,),
        (
            MapPrimitive(
                "affine half-space",
                (source,),
                restricted,
                scalar_input(source.name),
                predicate=predicate,
                rejected_value=0.0,
            ),
        ),
        (restricted,),
    )
    assert zero_filled.semantic_fingerprint != program.semantic_fingerprint


def test_predicated_map_pullback_zeros_rejected_cotangents() -> None:
    row = TensorAxis(0, 3, "row")
    column = TensorAxis(1, 4, "column")
    source = ProgramValue("source", (row, column), DType.FP32)
    restricted = ProgramValue("restricted", source.axes, source.dtype)
    total = ProgramValue("total", (), DType.FP32)
    predicate = index_compare(
        IndexPredicateKind.LESS_EQUAL,
        index_axis(column),
        index_binary(IndexExpressionKind.ADD, index_axis(row), index_constant(1)),
    )
    program = TensorProgram(
        (source,),
        (
            MapPrimitive(
                "affine half-space",
                (source,),
                restricted,
                scalar_input(source.name),
                predicate=predicate,
                rejected_value=0.0,
            ),
            FoldPrimitive("sum restricted values", restricted, total, source.axes, FoldReducer.SUM, DType.FP32),
        ),
        (total,),
    )
    differentiated = differentiate_tensor_program(program, with_respect_to=(source.name,))

    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    actual = execute_tensor_program(
        differentiated.program,
        {source.name: values, differentiated.output_cotangents[0].name: np.asarray(1.0, dtype=np.float32)},
    )[differentiated.input_gradients[0].name]
    expected = np.asarray(np.arange(4)[None, :] <= np.arange(3)[:, None] + 1, dtype=np.float32)

    np.testing.assert_array_equal(actual, expected)
