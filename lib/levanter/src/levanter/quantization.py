# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Mixed precision at the model and gradient-carried state boundary."""

from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
from haliax import NamedArray
from haliax.quantization import CustomGradientAccumulation, partition_for_grad_overwrite

T = TypeVar("T")


def cast_model_preserving_quantization_state(model: T, cast_fn: Callable[[T], T]) -> T:
    """Cast ordinary model arrays while leaving custom gradient state unchanged."""
    state, parameters = partition_for_grad_overwrite(model)
    return eqx.combine(
        state, cast_fn(parameters), is_leaf=lambda node: isinstance(node, (NamedArray, CustomGradientAccumulation))
    )
