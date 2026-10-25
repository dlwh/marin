# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach shared quantization operations to named and array-native projections."""

import dataclasses
from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
from jax.tree_util import GetAttrKey

from haliax import NamedArray
from haliax.quantization import (
    CustomGradientAccumulation,
    Fp8DotGeneralOp,
    QuantizationConfig,
    partition_for_grad_overwrite,
)
from haliax.quantization import quantize_linear_layers as quantize_named_linear_layers

from levanter.layers.linear import Linear
from levanter.layers.stacked import ArrayStacked

T = TypeVar("T")


def has_array_linears(tree) -> bool:
    return any(
        isinstance(leaf, Linear) for leaf in jax.tree.leaves(tree, is_leaf=lambda leaf: isinstance(leaf, Linear))
    )


def quantize_linear_layers(tree: T, config: QuantizationConfig) -> T:
    """Convert projections, preserving the shared named-layer quantization contract.

    Array-native layers support FP8 only. Their stacked state is initialized
    independently for each layer, matching the leading axis of their parameters.
    """
    if not has_array_linears(tree):
        return quantize_named_linear_layers(tree, config)
    if not config.fp8 or config.int8:
        raise ValueError("Array-native quantization supports FP8 only")

    tree = quantize_named_linear_layers(tree, config)

    def is_projection_or_stack(module):
        return isinstance(module, (Linear, ArrayStacked))

    def convert(subtree, prefix):
        def convert_module(path, module):
            path = prefix + path
            if isinstance(module, ArrayStacked):
                stacked = eqx.filter_vmap(
                    lambda layer: convert(layer, path + (GetAttrKey("stacked"),)), axis_size=module.num_layers
                )(module.stacked)
                return dataclasses.replace(module, stacked=stacked)
            if isinstance(module, Linear) and config.matches_target(path):
                operation = Fp8DotGeneralOp.init(config.amax_history_length, config.compute_dtype)
                return dataclasses.replace(module, dot_general=operation)
            return module

        return jax.tree_util.tree_map_with_path(convert_module, subtree, is_leaf=is_projection_or_stack)

    return convert(tree, ())


def cast_model_preserving_quantization_state(model: T, cast_fn: Callable[[T], T]) -> T:
    """Cast ordinary model arrays while leaving custom gradient state unchanged."""
    state, parameters = partition_for_grad_overwrite(model)
    return eqx.combine(
        state, cast_fn(parameters), is_leaf=lambda node: isinstance(node, (NamedArray, CustomGradientAccumulation))
    )
