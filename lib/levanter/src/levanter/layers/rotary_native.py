# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-native default and linear-scaled rotary embeddings."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class RotaryConfig:
    theta: float = 10000.0
    factor: float = 1.0


def apply_rotary(
    x: Float[Array, "B S H D"], position_ids: Int[Array, "B S"], config: RotaryConfig
) -> Float[Array, "B S H D"]:
    """Apply split-half RoPE using caller-supplied token positions."""
    dim = x.shape[-1]
    inv_freq = 1.0 / (config.theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    frequencies = position_ids[..., None] * (inv_freq / config.factor)
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)[..., None, :]
    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)
    first, second = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return x * cos + rotated * sin
