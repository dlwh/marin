# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.msa_frontend import (
    MSA_INPUT_NAMES,
    MSADebugConfig,
    export_debug_msa_training_step_text,
    export_debug_msa_value_and_grad_text,
    msa_preprojected_main_region,
    msa_projected_region,
    msa_region,
    msa_training_step,
    msa_value_and_grad,
)


def _inputs(config: MSADebugConfig, seed: int = 0):
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(MSA_INPUT_NAMES) + 1)
    shapes = (
        (config.query_length, config.hidden_dimension),
        (config.key_value_length, config.hidden_dimension),
        (config.hidden_dimension, config.query_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.index_dimension),
        (config.hidden_dimension, config.index_dimension),
    )
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    inputs = tuple(
        jax.random.normal(input_key, shape, dtype=dtype)
        for input_key, shape, dtype in zip(keys[:-1], shapes, dtypes, strict=True)
    )
    output_cotangent = jax.random.normal(
        keys[-1],
        (config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    return inputs, output_cotangent


def test_msa_value_and_grad_matches_composed_training_step() -> None:
    config = MSADebugConfig()
    inputs, output_cotangent = _inputs(config)

    objective, *objective_gradients = jax.jit(msa_value_and_grad(config))(*inputs, output_cotangent)
    output, *training_gradients = jax.jit(msa_training_step(config))(*inputs, output_cotangent)

    np.testing.assert_array_equal(output, msa_region(config)(*inputs))
    np.testing.assert_array_equal(
        objective,
        jnp.vdot(output.astype(jnp.float32), output_cotangent.astype(jnp.float32)),
    )
    assert [gradient.shape for gradient in training_gradients] == [value.shape for value in inputs]
    assert [gradient.dtype for gradient in training_gradients] == [value.dtype for value in inputs]
    for objective_gradient, training_gradient in zip(objective_gradients, training_gradients, strict=True):
        np.testing.assert_array_equal(objective_gradient, training_gradient)
    for gradient in training_gradients[:5]:
        assert np.any(np.asarray(gradient))
    np.testing.assert_array_equal(training_gradients[5], jnp.zeros_like(inputs[5]))
    np.testing.assert_array_equal(training_gradients[6], jnp.zeros_like(inputs[6]))


def test_msa_reverse_exports_are_ordinary_jax_stablehlo() -> None:
    value_and_grad = export_debug_msa_value_and_grad_text()
    training_step = export_debug_msa_training_step_text()

    for module in (value_and_grad, training_step):
        assert "stablehlo.composite" in module
        assert "stablehlo.gather" in module
        assert "stablehlo.scatter" in module
        assert "stablehlo.dot_general" in module
        assert "custom_call" not in module
        assert "custom_vjp" not in module


def test_msa_projected_source_boundaries_preserve_forward_math() -> None:
    config = MSADebugConfig()
    inputs, _ = _inputs(config)
    query_hidden, key_value_hidden, query_weight, key_weight, value_weight, left_weight, right_weight = inputs
    query = (
        jnp.matmul(query_hidden, query_weight, preferred_element_type=jnp.float32)
        .astype(jnp.bfloat16)
        .reshape(config.query_length, config.query_heads, config.head_dimension)
    )
    key = (
        jnp.matmul(key_value_hidden, key_weight, preferred_element_type=jnp.float32)
        .astype(jnp.bfloat16)
        .reshape(config.key_value_length, config.key_value_heads, config.head_dimension)
    )
    value = (
        jnp.matmul(key_value_hidden, value_weight, preferred_element_type=jnp.float32)
        .astype(jnp.bfloat16)
        .reshape(config.key_value_length, config.key_value_heads, config.head_dimension)
    )
    left_index = (
        jnp.matmul(query_hidden.astype(jnp.float32), left_weight, preferred_element_type=jnp.float32)
        .astype(jnp.bfloat16)
        .reshape(config.query_length, config.key_value_heads, config.index_dimension)
    )
    right_index = jnp.matmul(
        key_value_hidden.astype(jnp.float32), right_weight, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)

    expected = msa_region(config)(*inputs)
    preprojected_main = msa_preprojected_main_region(config)(
        query_hidden,
        key_value_hidden,
        left_weight,
        right_weight,
        query,
        key,
        value,
    )
    fully_projected = msa_projected_region(config)(query, key, value, left_index, right_index)

    np.testing.assert_array_equal(preprojected_main, expected)
    np.testing.assert_array_equal(fully_projected, expected)
