# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX projected-relation sparse attention and StableHLO export."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

MSA_INPUT_NAMES = (
    "query_hidden",
    "key_value_hidden",
    "query_weight",
    "key_weight",
    "value_weight",
    "left_index_weight",
    "right_index_weight",
)
MSA_BACKWARD_INPUT_NAMES = (*MSA_INPUT_NAMES, "output_cotangent")
MSA_PROJECTED_INPUT_NAMES = ("query", "key", "value", "left_index", "right_index")
MSA_PREPROJECTED_MAIN_INPUT_NAMES = (
    "query_hidden",
    "key_value_hidden",
    "left_index_weight",
    "right_index_weight",
    "query",
    "key",
    "value",
)


@dataclass(frozen=True)
class MSADebugConfig:
    """Small static shape for the natural projected-relation fixture."""

    query_length: int = 8
    key_value_length: int = 16
    hidden_dimension: int = 12
    query_heads: int = 4
    key_value_heads: int = 2
    head_dimension: int = 8
    index_dimension: int = 4
    block_size: int = 4
    selected_blocks: int = 2

    def __post_init__(self) -> None:
        dimensions = (
            self.query_length,
            self.key_value_length,
            self.hidden_dimension,
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
            self.index_dimension,
            self.block_size,
            self.selected_blocks,
        )
        if min(dimensions) <= 0:
            raise ValueError("projected sparse-attention dimensions must be positive")
        if self.key_value_length % self.block_size:
            raise ValueError("key/value length must be divisible by block size")
        if self.query_length > self.key_value_length:
            raise ValueError("bottom-right causal prefill requires query length no larger than key/value length")
        if self.query_heads % self.key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.selected_blocks > self.block_count:
            raise ValueError("selected block count cannot exceed block count")

    @property
    def block_count(self) -> int:
        """Number of key/value blocks."""
        return self.key_value_length // self.block_size

    @property
    def query_position_offset(self) -> int:
        """Absolute position of the first query in bottom-right causal alignment."""
        return self.key_value_length - self.query_length

    @property
    def attention_scale(self) -> float:
        """Source-level main-attention score scale."""
        return self.head_dimension**-0.5

    @property
    def index_scale(self) -> float:
        """Source-level index score scale."""
        return self.index_dimension**-0.5


def export_debug_msa(config: MSADebugConfig = MSADebugConfig()) -> bytes:
    """Export projections, relation selection, and selected exact attention."""
    exported = jax.export.export(jax.jit(msa_region(config)))(*_msa_input_specifications(config))
    return exported.mlir_module_serialized


def export_debug_msa_value_and_grad(config: MSADebugConfig = MSADebugConfig()) -> bytes:
    """Export a scalar objective and its JAX-owned reverse as StableHLO."""
    return _export_debug_msa_value_and_grad(config).mlir_module_serialized


def export_debug_msa_value_and_grad_text(config: MSADebugConfig = MSADebugConfig()) -> str:
    """Render the JAX-owned MSA value-and-grad boundary for provenance audits."""
    return _export_debug_msa_value_and_grad(config).mlir_module()


def export_debug_msa_training_step(config: MSADebugConfig = MSADebugConfig()) -> bytes:
    """Export the composed MSA forward-and-reverse training boundary."""
    return _export_debug_msa_training_step(config).mlir_module_serialized


def export_debug_msa_training_step_text(config: MSADebugConfig = MSADebugConfig()) -> str:
    """Render the composed MSA training boundary for provenance audits."""
    return _export_debug_msa_training_step(config).mlir_module()


def _msa_input_specifications(config: MSADebugConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    bf16 = jnp.bfloat16
    return (
        jax.ShapeDtypeStruct((config.query_length, config.hidden_dimension), bf16),
        jax.ShapeDtypeStruct((config.key_value_length, config.hidden_dimension), bf16),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.query_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.head_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.hidden_dimension, config.key_value_heads * config.index_dimension),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct((config.hidden_dimension, config.index_dimension), jnp.float32),
    )


def _msa_backward_specifications(config: MSADebugConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    output_cotangent = jax.ShapeDtypeStruct(
        (config.query_length, config.query_heads, config.head_dimension),
        jnp.bfloat16,
    )
    return (*_msa_input_specifications(config), output_cotangent)


def _export_debug_msa_value_and_grad(config: MSADebugConfig):
    return jax.export.export(jax.jit(msa_value_and_grad(config)))(*_msa_backward_specifications(config))


def _export_debug_msa_training_step(config: MSADebugConfig):
    return jax.export.export(jax.jit(msa_training_step(config)))(*_msa_backward_specifications(config))


def msa_region(config: MSADebugConfig):
    """Return natural projected-routing and exact selected-attention math."""

    def region(
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_index_weight,
        right_index_weight,
    ):
        query = _project_main(query_hidden, query_weight).reshape(
            config.query_length,
            config.query_heads,
            config.head_dimension,
        )
        key = _project_main(key_value_hidden, key_weight).reshape(
            config.key_value_length,
            config.key_value_heads,
            config.head_dimension,
        )
        value = _project_main(key_value_hidden, value_weight).reshape(
            config.key_value_length,
            config.key_value_heads,
            config.head_dimension,
        )

        _, selected, selected_valid = _select_blocks(
            config,
            query_hidden,
            key_value_hidden,
            left_index_weight,
            right_index_weight,
        )
        return _selected_attention(config, query, key, value, selected, selected_valid)

    return region


def msa_projected_region(config: MSADebugConfig):
    """Return the same sparse-attention math from already-projected public inputs."""

    def region(query, key, value, left_index, right_index):
        _, selected, selected_valid = _select_projected_blocks(config, left_index, right_index)
        return _selected_attention(config, query, key, value, selected, selected_valid)

    return region


def msa_preprojected_main_region(config: MSADebugConfig):
    """Return sparse attention with main Q/K/V supplied and index projections retained."""

    def region(query_hidden, key_value_hidden, left_index_weight, right_index_weight, query, key, value):
        _, selected, selected_valid = _select_blocks(
            config,
            query_hidden,
            key_value_hidden,
            left_index_weight,
            right_index_weight,
        )
        return _selected_attention(config, query, key, value, selected, selected_valid)

    return region


def _selected_attention(config, query, key, value, selected, selected_valid):
    safe_selected = jax.lax.select(selected_valid, selected, jnp.zeros_like(selected))

    blocked_key = key.reshape(
        config.block_count,
        config.block_size,
        config.key_value_heads,
        config.head_dimension,
    ).transpose(2, 0, 1, 3)
    blocked_value = value.reshape(
        config.block_count,
        config.block_size,
        config.key_value_heads,
        config.head_dimension,
    ).transpose(2, 0, 1, 3)
    group_indices = safe_selected.transpose(1, 0, 2)
    selected_key = jax.vmap(lambda blocks, indices: blocks[indices])(blocked_key, group_indices).transpose(1, 0, 2, 3, 4)
    selected_value = jax.vmap(lambda blocks, indices: blocks[indices])(
        blocked_value,
        group_indices,
    ).transpose(1, 0, 2, 3, 4)

    heads_per_group = config.query_heads // config.key_value_heads
    grouped_query = query.reshape(
        config.query_length,
        config.key_value_heads,
        heads_per_group,
        config.head_dimension,
    )
    scores = jnp.einsum(
        "qhgd,qhkbd->qhgkb",
        grouped_query,
        selected_key,
        preferred_element_type=jnp.float32,
    )
    scores *= config.attention_scale
    selected_key_position = (
        safe_selected[..., None] * config.block_size + jnp.arange(config.block_size)[None, None, None, :]
    )
    query_position = jnp.arange(config.query_length) + config.query_position_offset
    score_valid = selected_key_position <= query_position[:, None, None, None]
    scores = jax.lax.select(
        jnp.broadcast_to(score_valid[:, :, None, :, :], scores.shape),
        scores,
        jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype),
    )
    scores = jax.lax.select(
        jnp.broadcast_to(selected_valid[:, :, None, :, None], scores.shape),
        scores,
        jnp.full(scores.shape, -jnp.inf, dtype=scores.dtype),
    )
    flattened_scores = scores.reshape(
        config.query_length,
        config.key_value_heads,
        heads_per_group,
        config.selected_blocks * config.block_size,
    )
    probability = jax.nn.softmax(flattened_scores, axis=-1).astype(jnp.bfloat16)
    flattened_value = selected_value.reshape(
        config.query_length,
        config.key_value_heads,
        config.selected_blocks * config.block_size,
        config.head_dimension,
    )
    output = jnp.einsum(
        "qhgk,qhkd->qhgd",
        probability,
        flattened_value,
        preferred_element_type=jnp.float32,
    )
    return output.reshape(config.query_length, config.query_heads, config.head_dimension).astype(jnp.bfloat16)


def msa_selection(config: MSADebugConfig):
    """Return ordinary-JAX ranked scores, selected blocks, and slot validity.

    This diagnostic boundary makes route identity and exact cutoff ties
    observable without inserting a compiler marker into the computation.
    """

    def selection(query_hidden, key_value_hidden, left_index_weight, right_index_weight):
        return _select_blocks(
            config,
            query_hidden,
            key_value_hidden,
            left_index_weight,
            right_index_weight,
        )

    return selection


def msa_value_and_grad(config: MSADebugConfig):
    """Return a scalar objective and reverse whose differentiation is owned by JAX."""
    region = msa_region(config)

    def objective(
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_index_weight,
        right_index_weight,
        output_cotangent,
    ):
        output = region(
            query_hidden,
            key_value_hidden,
            query_weight,
            key_weight,
            value_weight,
            left_index_weight,
            right_index_weight,
        )
        return jnp.vdot(output.astype(jnp.float32), output_cotangent.astype(jnp.float32))

    differentiated = jax.value_and_grad(objective, argnums=tuple(range(len(MSA_INPUT_NAMES))))

    def value_and_grad(*arguments):
        value, gradients = differentiated(*arguments)
        return value, *gradients

    return value_and_grad


def msa_training_step(config: MSADebugConfig):
    """Return the ordinary forward composed with its JAX-owned pullback."""
    region = msa_region(config)

    def training_step(
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_index_weight,
        right_index_weight,
        output_cotangent,
    ):
        inputs = (
            query_hidden,
            key_value_hidden,
            query_weight,
            key_weight,
            value_weight,
            left_index_weight,
            right_index_weight,
        )
        output, pullback = jax.vjp(region, *inputs)
        return output, *pullback(output_cotangent)

    return training_step


def _project_main(hidden, weight):
    return jnp.matmul(hidden, weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _select_blocks(config, query_hidden, key_value_hidden, left_index_weight, right_index_weight):
    detached_query = jax.lax.stop_gradient(query_hidden).astype(jnp.float32)
    detached_key_value = jax.lax.stop_gradient(key_value_hidden).astype(jnp.float32)
    left_index = jnp.matmul(
        detached_query,
        left_index_weight,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    left_index = left_index.reshape(config.query_length, config.key_value_heads, config.index_dimension)
    right_index = jnp.matmul(
        detached_key_value,
        right_index_weight,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    return _select_projected_blocks(config, left_index, right_index)


def _select_projected_blocks(config, left_index, right_index):
    token_scores = jnp.einsum(
        "qhd,kd->qhk",
        left_index,
        right_index,
        preferred_element_type=jnp.float32,
    )
    token_scores *= config.index_scale
    query_position = (jnp.arange(config.query_length) + config.query_position_offset)[:, None]
    key_position = jnp.arange(config.key_value_length)[None, :]
    token_scores = jax.lax.select(
        jnp.broadcast_to((key_position <= query_position)[:, None, :], token_scores.shape),
        token_scores,
        jnp.full(token_scores.shape, -jnp.inf, dtype=token_scores.dtype),
    )
    block_scores = jnp.max(
        token_scores.reshape(
            config.query_length,
            config.key_value_heads,
            config.block_count,
            config.block_size,
        ),
        axis=-1,
    )
    all_local_blocks = jnp.repeat(jnp.arange(config.block_count), config.block_size)
    local_block = jax.lax.slice_in_dim(
        all_local_blocks,
        config.query_position_offset,
        config.query_position_offset + config.query_length,
    )
    block_identity = jnp.arange(config.block_count)
    forced_local = block_identity[None, None, :] == local_block[:, None, None]
    ranked_scores = jax.lax.select(
        jnp.broadcast_to(forced_local, block_scores.shape),
        jnp.full(block_scores.shape, jnp.inf, dtype=block_scores.dtype),
        block_scores,
    )
    _, selected = jax.lax.top_k(ranked_scores, config.selected_blocks)
    selected_valid = selected <= local_block[:, None, None]
    return ranked_scores, selected, selected_valid
