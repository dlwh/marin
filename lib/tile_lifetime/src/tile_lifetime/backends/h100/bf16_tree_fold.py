# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed admission and launch geometry for bounded BF16 sum trees."""

import math
from dataclasses import dataclass

from tile_lifetime.tensor_program import (
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    NumericalPolicy,
    TensorPrimitive,
)

TREE_THREADS = 256
TREE_SHARED_BYTES = 32 * 1024
_BF16_BYTES = 2
_MAXIMUM_LINEAR_INDEX = (1 << 31) - 1


def requires_bf16_tree_fold(operation: TensorPrimitive, policy: NumericalPolicy) -> bool:
    """Select long order-free BF16 sums without changing short or ordered Folds."""
    return (
        isinstance(operation, FoldPrimitive)
        and policy is NumericalPolicy.FAST
        and operation.order is EvaluationOrder.ORDER_FREE
        and operation.reducer is FoldReducer.SUM
        and operation.input.dtype is DType.BF16
        and operation.output.dtype is DType.BF16
        and operation.accumulation_dtype is DType.BF16
        and math.prod(axis.extent for axis in operation.reduction_axes) > TREE_THREADS
    )


def require_bf16_tree_boundary(operations: tuple[TensorPrimitive, ...], policy: NumericalPolicy) -> None:
    """Reject a new materialization unless the canonical plan already isolates the Fold."""
    if len(operations) > 1 and any(requires_bf16_tree_fold(operation, policy) for operation in operations):
        raise ValueError(
            "BF16 tree Fold requires an isolated canonical phase; a fused residue would add an unproved cut"
        )


@dataclass(frozen=True)
class Bf16TreeFoldPlan:
    """One CTA owns each output tile and rounds every tree node to BF16."""

    operation: FoldPrimitive
    numerical_policy: NumericalPolicy

    def __post_init__(self) -> None:
        if not requires_bf16_tree_fold(self.operation, self.numerical_policy):
            raise ValueError("BF16 tree requires a long FAST/order-free same-dtype BF16 SUM")
        if self.leaf_capacity * _BF16_BYTES > TREE_SHARED_BYTES:
            raise ValueError("BF16 tree reduction exceeds the 32 KiB shared-memory implementation bound")
        if math.prod(self.operation.input.shape) > _MAXIMUM_LINEAR_INDEX:
            raise ValueError("BF16 tree source exceeds the signed 32-bit indexing bound")

    @property
    def reduction_elements(self) -> int:
        return math.prod(axis.extent for axis in self.operation.reduction_axes)

    @property
    def output_elements(self) -> int:
        return math.prod(self.operation.output.shape)

    @property
    def depth(self) -> int:
        return (self.reduction_elements - 1).bit_length()

    @property
    def leaf_capacity(self) -> int:
        return 1 << self.depth

    @property
    def outputs_per_cta(self) -> int:
        # Interleave adjacent outputs only when that follows contiguous source storage.
        if self.operation.input.axes[-1] not in self.operation.output.axes:
            return 1
        return min(8, TREE_SHARED_BYTES // (_BF16_BYTES * self.leaf_capacity))

    @property
    def grid(self) -> tuple[int, int, int]:
        return ((self.output_elements + self.outputs_per_cta - 1) // self.outputs_per_cta, 1, 1)

    @property
    def block(self) -> tuple[int, int, int]:
        return (TREE_THREADS, 1, 1)

    @property
    def shared_elements(self) -> int:
        return self.leaf_capacity * self.outputs_per_cta

    @property
    def shared_bytes(self) -> int:
        return _BF16_BYTES * self.shared_elements

    @property
    def source_output_strides(self) -> tuple[int, ...]:
        axes = self.operation.input.axes
        return tuple(
            math.prod(self.operation.input.shape[axes.index(axis) + 1 :]) for axis in self.operation.output.axes
        )

    @property
    def source_reduction_strides(self) -> tuple[int, ...]:
        axes = self.operation.input.axes
        return tuple(
            math.prod(self.operation.input.shape[axes.index(axis) + 1 :]) for axis in self.operation.reduction_axes
        )

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "bf16_adjacent_tree_sum",
            "input_shape": self.operation.input.shape,
            "output_shape": self.operation.output.shape,
            "reduction_axes": tuple(self.operation.input.axes.index(axis) for axis in self.operation.reduction_axes),
            "source_output_strides": self.source_output_strides,
            "source_reduction_strides": self.source_reduction_strides,
            "grid": self.grid,
            "block": self.block,
            "dynamic_shared_memory_bytes": self.shared_bytes,
            "outputs_per_cta": self.outputs_per_cta,
            "reduction_elements": self.reduction_elements,
            "maximum_rounding_depth": self.depth,
            "accumulation_dtype": DType.BF16.value,
            "rounding": "round_to_nearest_even_at_every_addition",
            "odd_tail": "carry_without_addition",
            "finite_precision_effect": "reassociates_order_free_sum_into_adjacent_pairs",
            "conditional_error_bound": "gamma(depth) * sum(abs(realized_terms)); u=2^-8",
            "error_bound_assumptions": "finite terms, round-to-nearest, no overflow or underflow",
        }
