# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX row-statistic projections and JAX-owned reverse boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RowProjectionConfig:
    """Static dimensions for a row-statistic Map/Fold/Contract fixture."""

    rows: int = 4
    hidden: int = 8
    features: int = 6
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if min(self.rows, self.hidden, self.features) <= 0:
            raise ValueError("row projection dimensions must be positive")
        if self.epsilon < 0:
            raise ValueError("row projection epsilon must be non-negative")


RowProjection = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def _round_to_source_dtype(value: jax.Array, source: jax.Array) -> jax.Array:
    if source.dtype == jnp.dtype(jnp.bfloat16):
        # A convert-to-BF16 immediately widened back to FP32 may be removed by
        # XLA under FAST math. reduce_precision is the value-level rounding
        # operation and remains observable even when surrounding converts fold.
        value = jax.lax.reduce_precision(value, exponent_bits=8, mantissa_bits=7)
    return value.astype(source.dtype)


def rms_projection(config: RowProjectionConfig) -> RowProjection:
    """Return an ordinary JAX uncentered row-statistic projection."""

    def project(x: jax.Array, feature_scale: jax.Array, weight: jax.Array) -> jax.Array:
        x_fp32 = x.astype(jnp.float32)
        second_moment = jnp.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
        normalized = _round_to_source_dtype(x_fp32 * jax.lax.rsqrt(second_moment + config.epsilon), x)
        scaled = _round_to_source_dtype(
            normalized.astype(jnp.float32) * feature_scale.astype(jnp.float32),
            x,
        )
        contracted = jnp.matmul(scaled, weight, preferred_element_type=jnp.float32)
        return _round_to_source_dtype(contracted, x)

    return project


def layer_projection(config: RowProjectionConfig) -> RowProjection:
    """Return an ordinary JAX centered row-statistic projection."""

    def project(x: jax.Array, feature_scale: jax.Array, weight: jax.Array) -> jax.Array:
        x_fp32 = x.astype(jnp.float32)
        centered = _round_to_source_dtype(x_fp32 - jnp.mean(x_fp32, axis=-1, keepdims=True), x)
        centered_fp32 = centered.astype(jnp.float32)
        second_moment = jnp.mean(centered_fp32 * centered_fp32, axis=-1, keepdims=True)
        normalized = _round_to_source_dtype(
            centered_fp32 * jax.lax.rsqrt(second_moment + config.epsilon),
            x,
        )
        scaled = _round_to_source_dtype(
            normalized.astype(jnp.float32) * feature_scale.astype(jnp.float32),
            x,
        )
        contracted = jnp.matmul(scaled, weight, preferred_element_type=jnp.float32)
        return _round_to_source_dtype(contracted, x)

    return project


def projection_value_and_grad(project: RowProjection):
    """Build a scalar-cotangent objective and reverse entirely through JAX AD."""

    def objective(
        x: jax.Array,
        feature_scale: jax.Array,
        weight: jax.Array,
        output_cotangent: jax.Array,
    ) -> jax.Array:
        output = project(x, feature_scale, weight)
        return jnp.sum(output * output_cotangent.astype(output.dtype))

    return jax.value_and_grad(objective, argnums=(0, 1, 2))


def projection_training_step(project: RowProjection):
    """Build the composed forward and JAX-owned pullback boundary."""

    def training_step(
        x: jax.Array,
        feature_scale: jax.Array,
        weight: jax.Array,
        output_cotangent: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        output, pullback = jax.vjp(project, x, feature_scale, weight)
        return output, pullback(output_cotangent.astype(output.dtype))

    return training_step


def export_projection_forward(project: RowProjection, config: RowProjectionConfig) -> bytes:
    """Export the ordinary forward boundary as StableHLO bytecode."""
    exported = jax.export.export(jax.jit(project))(*_forward_specifications(config))
    return exported.mlir_module_serialized


def export_projection_value_and_grad(project: RowProjection, config: RowProjectionConfig) -> bytes:
    """Export the JAX-owned scalar-objective reverse as StableHLO bytecode."""
    exported = jax.export.export(jax.jit(projection_value_and_grad(project)))(*_training_specifications(config))
    return exported.mlir_module_serialized


def export_projection_training_step(project: RowProjection, config: RowProjectionConfig) -> bytes:
    """Export a composed ordinary-JAX training boundary as StableHLO bytecode."""
    exported = jax.export.export(jax.jit(projection_training_step(project)))(*_training_specifications(config))
    return exported.mlir_module_serialized


def _forward_specifications(config: RowProjectionConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        jax.ShapeDtypeStruct((config.rows, config.hidden), jnp.float32),
        jax.ShapeDtypeStruct((config.hidden,), jnp.float32),
        jax.ShapeDtypeStruct((config.hidden, config.features), jnp.float32),
    )


def _training_specifications(config: RowProjectionConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        *_forward_specifications(config),
        jax.ShapeDtypeStruct((config.rows, config.features), jnp.float32),
    )
