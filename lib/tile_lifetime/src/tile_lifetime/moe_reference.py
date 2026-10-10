# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX reference/export for the first shared-plus-routed MoE region."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

MOE_REGION_INPUT_NAMES = (
    "x",
    "router_weight",
    "shared_gate_weight",
    "shared_up_weight",
    "shared_down_weight",
    "routed_gate_weight",
    "routed_up_weight",
    "routed_down_weight",
)
MOE_BACKWARD_INPUT_NAMES = (*MOE_REGION_INPUT_NAMES, "output_cotangent")

# JAX emits top-k as a public StableHLO composite and expert indexing as gather.
# Its private top-k decomposition contains stablehlo.sort; the importer deliberately
# ignores that private implementation and validates the versioned composite boundary.
MOE_PUBLIC_STABLEHLO_EXTENSIONS = ("stablehlo.composite[chlo.top_k]", "stablehlo.gather")
MOE_UNIMPORTED_PRIVATE_OPERATIONS = ("stablehlo.sort",)


@dataclass(frozen=True)
class MoEDebugConfig:
    """Static dimensions for a small single-device semantic MoE fixture."""

    tokens: int = 8
    hidden: int = 16
    intermediate: int = 32
    experts: int = 4
    top_k: int = 2

    def __post_init__(self) -> None:
        if min(self.tokens, self.hidden, self.intermediate, self.experts) <= 0:
            raise ValueError("MoE dimensions must be positive")
        if not 0 < self.top_k <= self.experts:
            raise ValueError("top-k must be positive and no larger than the expert count")


def export_debug_moe_region(config: MoEDebugConfig = MoEDebugConfig()) -> bytes:
    """Export the ordinary shared-plus-routed JAX region as portable StableHLO."""
    exported = jax.export.export(jax.jit(moe_region(config)))(*_moe_input_specifications(config))
    return exported.mlir_module_serialized


def export_debug_moe_value_and_grad(config: MoEDebugConfig = MoEDebugConfig()) -> bytes:
    """Export a scalar objective and its JAX-owned reverse as StableHLO."""
    return _export_debug_moe_value_and_grad(config).mlir_module_serialized


def export_debug_moe_value_and_grad_text(config: MoEDebugConfig = MoEDebugConfig()) -> str:
    """Render the JAX-owned reverse boundary for source-lineage audits."""
    return _export_debug_moe_value_and_grad(config).mlir_module()


def export_debug_moe_training_step(config: MoEDebugConfig = MoEDebugConfig()) -> bytes:
    """Export the ordinary forward composed with its JAX-owned pullback."""
    return _export_debug_moe_training_step(config).mlir_module_serialized


def export_debug_moe_training_step_text(config: MoEDebugConfig = MoEDebugConfig()) -> str:
    """Render the composed forward-and-reverse boundary for lineage audits."""
    return _export_debug_moe_training_step(config).mlir_module()


def _moe_input_specifications(config: MoEDebugConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    bf16 = jnp.bfloat16
    return (
        jax.ShapeDtypeStruct((config.tokens, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.experts), bf16),
        jax.ShapeDtypeStruct((config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.intermediate), bf16),
        jax.ShapeDtypeStruct((config.experts, config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.experts, config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.experts, config.hidden, config.intermediate), bf16),
    )


def _moe_backward_specifications(config: MoEDebugConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    output_cotangent = jax.ShapeDtypeStruct((config.tokens, config.hidden), jnp.bfloat16)
    return (*_moe_input_specifications(config), output_cotangent)


def _export_debug_moe_value_and_grad(config: MoEDebugConfig):
    return jax.export.export(jax.jit(moe_value_and_grad(config)))(*_moe_backward_specifications(config))


def _export_debug_moe_training_step(config: MoEDebugConfig):
    return jax.export.export(jax.jit(moe_training_step(config)))(*_moe_backward_specifications(config))


def moe_region(config: MoEDebugConfig):
    """Return the ordinary JAX function used to create the semantic fixture."""

    def region(
        x,
        router_weight,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        routed_gate_weight,
        routed_up_weight,
        routed_down_weight,
    ):
        router_logits = _linear(x, router_weight, "th,he->te")
        top_values, expert_indices = jax.lax.top_k(router_logits, config.top_k)
        route_weights = _softmax(top_values.astype(jnp.float32))

        shared_gate = _linear(x, shared_gate_weight, "th,ih->ti")
        shared_up = _linear(x, shared_up_weight, "th,ih->ti")
        shared_hidden = _silu(shared_gate) * shared_up
        shared_output = _linear(shared_hidden, shared_down_weight, "ti,hi->th")

        selected_gate_weight = routed_gate_weight[expert_indices]
        selected_up_weight = routed_up_weight[expert_indices]
        selected_down_weight = routed_down_weight[expert_indices]
        routed_gate = _linear(x, selected_gate_weight, "th,tkih->tki")
        routed_up = _linear(x, selected_up_weight, "th,tkih->tki")
        routed_hidden = _silu(routed_gate) * routed_up
        routed_output = _linear(routed_hidden, selected_down_weight, "tki,tkhi->tkh")

        weighted = routed_output.astype(jnp.float32) * route_weights[..., None]
        routed_sum = jnp.sum(weighted, axis=1).astype(jnp.bfloat16)
        return shared_output + routed_sum, expert_indices, route_weights

    return region


def moe_value_and_grad(config: MoEDebugConfig):
    """Return a scalar objective whose differentiation is owned entirely by JAX."""
    region = moe_region(config)

    def objective(*arguments):
        *inputs, output_cotangent = arguments
        output, expert_indices, route_weights = region(*inputs)
        loss = jnp.vdot(output.astype(jnp.float32), output_cotangent.astype(jnp.float32))
        return loss, (output, expert_indices, route_weights)

    differentiated = jax.value_and_grad(
        objective,
        argnums=tuple(range(len(MOE_REGION_INPUT_NAMES))),
        has_aux=True,
    )

    def value_and_grad(*arguments):
        (value, auxiliary), gradients = differentiated(*arguments)
        output, expert_indices, route_weights = auxiliary
        return value, output, expert_indices, route_weights, *gradients

    return value_and_grad


def moe_training_step(config: MoEDebugConfig):
    """Return the ordinary forward composed directly with its JAX-owned pullback."""
    region = moe_region(config)

    def differentiable_output(*inputs):
        output, expert_indices, route_weights = region(*inputs)
        return output, (expert_indices, route_weights)

    def training_step(*arguments):
        *inputs, output_cotangent = arguments
        output_value, pullback, auxiliary = jax.vjp(
            differentiable_output,
            *inputs,
            has_aux=True,
        )
        expert_indices, route_weights = auxiliary
        return output_value, expert_indices, route_weights, *pullback(output_cotangent)

    return training_step


def _linear(left, right, specification: str):
    return jnp.einsum(specification, left, right, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _softmax(value):
    shifted = value - jnp.max(value, axis=-1, keepdims=True)
    exponential = jnp.exp(shifted)
    return exponential / jnp.sum(exponential, axis=-1, keepdims=True)


def _silu(value):
    one = jnp.asarray(1, dtype=value.dtype)
    return value / (one + jnp.exp(-value))
