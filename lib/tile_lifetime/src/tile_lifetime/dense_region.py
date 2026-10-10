# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary dense composition plus a quarantined named-graph reference path."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from shuttle.ir import DType
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_algebra import dense_composition_tensor_program
from tile_lifetime.dense_flow import erase_dense_transformer_semantics
from tile_lifetime.dense_flow_planner import (
    DenseCompositionExecutable,
    compile_reference_erased_dense_transformer_region,
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.ir import TensorGraph
from tile_lifetime.physical_task_contract import PhysicalNumericalPolicy, PhysicalTarget
from tile_lifetime.plan import NumericalPolicy, ReferenceRegionPlan
from tile_lifetime.qkv_rope import project_qkv
from tile_lifetime.swiglu import silu_product
from tile_lifetime.tensor_program import NumericalPolicy as TensorNumericalPolicy


@dataclass(frozen=True)
class DenseCompositionShape:
    """Static dimensions of the ordinary mathematical program."""

    heads: int
    query_key_dimension: int
    value_dimension: int
    epsilon: float = 1e-6
    causal: bool = True

    def __post_init__(self) -> None:
        if min(self.heads, self.query_key_dimension, self.value_dimension) <= 0:
            raise ValueError("dense composition dimensions must be positive")
        if self.query_key_dimension % 2:
            raise ValueError("adjacent-pair rotation requires an even query/key dimension")
        if self.epsilon <= 0:
            raise ValueError("normalization epsilon must be positive")


class DenseCompositionWeights(NamedTuple):
    """Tensor operands for the ordinary JAX composition."""

    input_scale: Array
    query: Array
    key: Array
    value: Array
    output: Array
    mlp_scale: Array
    gate: Array
    up: Array
    down: Array


def rms_scale(source: Array, scale: Array, *, epsilon: float) -> Array:
    """Apply row normalization using only ordinary JAX operations."""
    inverse_root_mean_square = jax.lax.rsqrt(jnp.mean(source * source, axis=-1, keepdims=True) + epsilon)
    return source * inverse_root_mean_square * scale


def dense_composition_forward(
    source: Array,
    weights: DenseCompositionWeights,
    shape: DenseCompositionShape,
) -> Array:
    """Evaluate a transformer-like composition as ordinary JAX."""
    normalized = rms_scale(source, weights.input_scale, epsilon=shape.epsilon)
    query, key, value = project_qkv(normalized, weights.query, weights.key, weights.value)

    if key.shape[2] != value.shape[2] or shape.heads % key.shape[2]:
        raise ValueError("query heads must be a multiple of matching key/value heads")
    if key.shape[2] != shape.heads:
        key = jnp.repeat(key, shape.heads // key.shape[2], axis=2)
        value = jnp.repeat(value, shape.heads // value.shape[2], axis=2)

    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key)
    scores = scores * jnp.asarray(shape.query_key_dimension**-0.5, dtype=scores.dtype)
    if shape.causal:
        query_index = jnp.arange(scores.shape[-2])[:, None]
        key_index = jnp.arange(scores.shape[-1])[None, :]
        scores = jnp.where(key_index <= query_index, scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhqk,bkhv->bqhv", probabilities, value)
    projected = jnp.einsum("bshv,hvm->bsm", attended, weights.output)
    residual = source + projected

    mlp_input = rms_scale(residual, weights.mlp_scale, epsilon=shape.epsilon)
    gate = jnp.einsum("bsm,mf->bsf", mlp_input, weights.gate)
    up = jnp.einsum("bsm,mf->bsf", mlp_input, weights.up)
    hidden = silu_product(gate, up)
    return residual + jnp.einsum("bsf,fm->bsm", hidden, weights.down)


def dense_composition_backward(
    source: Array,
    weights: DenseCompositionWeights,
    output_cotangent: Array,
    shape: DenseCompositionShape,
) -> tuple[Array, DenseCompositionWeights]:
    """Differentiate the ordinary forward program with JAX's pullback."""
    _, pullback = jax.vjp(lambda x, parameters: dense_composition_forward(x, parameters, shape), source, weights)
    source_gradient, weight_gradients = pullback(output_cotangent)
    return source_gradient, weight_gradients


def dense_composition_loss(
    source: Array,
    weights: DenseCompositionWeights,
    shape: DenseCompositionShape,
) -> Array:
    """Return the ordinary scalar quadratic objective for the dense graph."""
    output = dense_composition_forward(source, weights, shape)
    return _quadratic_mean(output)


def dense_composition_value_and_grad(
    source: Array,
    weights: DenseCompositionWeights,
    shape: DenseCompositionShape,
) -> tuple[Array, tuple[Array, DenseCompositionWeights]]:
    """Differentiate the scalar objective with JAX-owned automatic differentiation."""
    differentiated = jax.value_and_grad(
        lambda x, parameters: dense_composition_loss(x, parameters, shape),
        argnums=(0, 1),
    )
    return differentiated(source, weights)


def dense_composition_training_step(
    source: Array,
    weights: DenseCompositionWeights,
    shape: DenseCompositionShape,
) -> tuple[Array, Array, tuple[Array, DenseCompositionWeights]]:
    """Compose the forward program and JAX-owned reverse pass at one boundary."""

    def objective(x: Array, parameters: DenseCompositionWeights) -> tuple[Array, Array]:
        output = dense_composition_forward(x, parameters, shape)
        return _quadratic_mean(output), output

    (loss, output), gradients = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(source, weights)
    return loss, output, gradients


def _quadratic_mean(output: Array) -> Array:
    return jnp.mean(output * output)


def compile_dense_composition_reference(
    source: Array,
    weights: DenseCompositionWeights,
    shape: DenseCompositionShape,
    *,
    tile_extent_by_axis: dict[int, int],
    target: PhysicalTarget,
    numerical_policy: PhysicalNumericalPolicy,
) -> DenseCompositionExecutable:
    """Compile the ordinary program through every executable DSL boundary."""
    if source.ndim != 3:
        raise ValueError("dense composition source must have batch, token, and model dimensions")
    if weights.key.shape[1] != shape.heads or weights.value.shape[1] != shape.heads:
        raise ValueError(
            "the hand-constructed dense algebra requires equal heads; import ordinary JAX for grouped heads"
        )
    dtype = DType.BF16 if source.dtype == jnp.bfloat16 else DType.FP32
    batch, sequence, model = source.shape
    program = dense_composition_tensor_program(
        batch_size=batch,
        sequence_length=sequence,
        model_dimension=model,
        heads=shape.heads,
        query_key_dimension=shape.query_key_dimension,
        value_dimension=shape.value_dimension,
        feed_forward_dimension=weights.gate.shape[-1],
        epsilon=shape.epsilon,
        causal=shape.causal,
        dtype=dtype,
        accumulation_dtype=DType.FP32,
        numerical_policy=TensorNumericalPolicy(numerical_policy.value),
    )
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extent_by_axis,
        numerical_policy=numerical_policy,
    )
    event_program = eventize_composed_task_plan(task_plan)
    physical_request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=event_program,
        target=target,
        numerical_policy=numerical_policy,
    )
    return DenseCompositionExecutable(program, task_plan, event_program, physical_request)


def compile_reference_dense_transformer_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_PROLOGUE,
) -> ReferenceRegionPlan:
    """Compile the historical named dense graph for reference replay only."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("the dense structural prototype requires the rounding-reorder numerical policy")
    erased = erase_dense_transformer_semantics(graph)
    return compile_reference_erased_dense_transformer_region(
        erased,
        row_scale_placement=rms_scale_placement,
    )
