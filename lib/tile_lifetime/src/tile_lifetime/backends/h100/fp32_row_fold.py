# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU admission and launch geometry for bounded FP32 row sums."""

import math
from dataclasses import dataclass

from tile_lifetime.h100_scalar_ast import validate_cute_scalar_expression
from tile_lifetime.tensor_program import (
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorPrimitive,
    primitive_inputs,
    primitive_outputs,
)

ROW_THREADS = 256
ROW_MAXIMUM_ITEMS_PER_THREAD = 256
ROW_MAXIMUM_LIVE_INPUTS = 8
_MAXIMUM_LINEAR_INDEX = (1 << 31) - 1
_FLOAT_DTYPES = (DType.BF16, DType.FP32)


def _supported_conversions(expression: ScalarExpression) -> bool:
    if expression.kind is ScalarExpressionKind.CONVERT:
        if expression.result_dtype is DType.FP32:
            if expression.conversion_semantics not in (
                ScalarConversionSemantics.EXACT,
                ScalarConversionSemantics.ROUND_NEAREST_EVEN,
            ):
                return False
        elif expression.result_dtype is DType.BF16:
            if expression.conversion_semantics is not ScalarConversionSemantics.ROUND_NEAREST_EVEN:
                return False
        else:
            return False
    return all(_supported_conversions(operand) for operand in expression.operands)


def _rejection_reason(
    operations: tuple[TensorPrimitive, ...],
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
    policy: NumericalPolicy,
) -> str | None:
    if not operations or not isinstance(operations[-1], FoldPrimitive):
        return "requires one terminal Fold"
    fold = operations[-1]
    if (
        policy is not NumericalPolicy.FAST
        or fold.order is not EvaluationOrder.ORDER_FREE
        or fold.reducer is not FoldReducer.SUM
        or fold.accumulation_dtype is not DType.FP32
        or fold.output.dtype is not DType.FP32
        or fold.input.dtype not in _FLOAT_DTYPES
    ):
        return "requires FAST/order-free SUM with FP32 accumulation and output"
    if not fold.output.axes:
        return "all-axis scalar reductions retain their existing hierarchy"
    if fold.input.axes != (*fold.output.axes, *fold.reduction_axes):
        return "reduction axes must be the unique contiguous trailing source axes in order"
    count = math.prod(axis.extent for axis in fold.reduction_axes)
    if not ROW_THREADS < count <= ROW_THREADS * ROW_MAXIMUM_ITEMS_PER_THREAD:
        return "requires a long row within the bounded per-thread iteration count"
    if live_outputs != (fold.output,):
        return "the terminal Fold must be the only live output"
    if len(live_inputs) > ROW_MAXIMUM_LIVE_INPUTS:
        return "prefix boundary exceeds the existing eight-input padded ABI"
    if len(set(live_inputs)) != len(live_inputs):
        return "live inputs must be unique"
    produced: set[ProgramValue] = set()
    external: set[ProgramValue] = set()
    needed = {fold.input}
    for operation in reversed(operations[:-1]):
        if not isinstance(operation, MapPrimitive) or operation.predicate is not None:
            return "prefix operations must be unpredicated Maps"
        if operation.output not in needed:
            return "every prefix Map must contribute to the terminal Fold"
        if operation.output.axes != fold.input.axes:
            return "prefix Maps must cover the complete Fold input domain"
        needed.remove(operation.output)
        needed.update(operation.inputs)
        # Validate the same expression dialect consumed by the GPU emitter.
        validate_cute_scalar_expression(operation.expression)
        if not _supported_conversions(operation.expression):
            return "prefix conversion is outside the BF16/FP32 physical implementation"
    for operation in operations:
        for value in (*primitive_inputs(operation), *primitive_outputs(operation)):
            if value.dtype not in _FLOAT_DTYPES:
                return "row inputs and prefix values must be BF16 or FP32"
            if math.prod(value.shape) > _MAXIMUM_LINEAR_INDEX:
                return "row indexing exceeds the signed 32-bit source bound"
            if value.axes != tuple(axis for axis in fold.input.axes if axis in value.axes):
                return "every operand must be an ordered projection of the row domain"
        for value in primitive_inputs(operation):
            if value not in produced:
                external.add(value)
        for value in primitive_outputs(operation):
            if value in produced or value in live_inputs:
                return "each result requires one source-order producer"
            produced.add(value)
    if external != set(live_inputs) or set(live_inputs) & produced:
        return "live inputs must exactly cover the source-order external dependencies"
    return None


@dataclass(frozen=True)
class Fp32RowFoldPlan:
    """One CTA reduces a row after preserving each typed prefix Map result."""

    operations: tuple[TensorPrimitive, ...]
    live_inputs: tuple[ProgramValue, ...]
    live_outputs: tuple[ProgramValue, ...]
    numerical_policy: NumericalPolicy

    def __post_init__(self) -> None:
        reason = _rejection_reason(self.operations, self.live_inputs, self.live_outputs, self.numerical_policy)
        if reason is not None:
            raise ValueError(f"FP32 row Fold: {reason}")

    @property
    def fold(self) -> FoldPrimitive:
        operation = self.operations[-1]
        assert isinstance(operation, FoldPrimitive)
        return operation

    @property
    def reduction_elements(self) -> int:
        return math.prod(axis.extent for axis in self.fold.reduction_axes)

    @property
    def output_elements(self) -> int:
        return math.prod(self.fold.output.shape)

    @property
    def items_per_thread(self) -> int:
        return (self.reduction_elements + ROW_THREADS - 1) // ROW_THREADS

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.output_elements, 1, 1)

    @property
    def block(self) -> tuple[int, int, int]:
        return (ROW_THREADS, 1, 1)

    @property
    def shared_bytes(self) -> int:
        return (ROW_THREADS // 32) * 4

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "fp32_row_block_sum",
            "input_shape": self.fold.input.shape,
            "output_shape": self.fold.output.shape,
            "reduction_axes": tuple(self.fold.input.axes.index(axis) for axis in self.fold.reduction_axes),
            "grid": self.grid,
            "block": self.block,
            "dynamic_shared_memory_bytes": self.shared_bytes,
            "reduction_elements": self.reduction_elements,
            "items_per_thread": self.items_per_thread,
            "accumulation_dtype": DType.FP32.value,
            "prefix_result_dtypes": tuple(
                primitive_outputs(operation)[0].dtype.value for operation in self.operations[:-1]
            ),
            "prefix_rounding": "preserve_each_source_Map_conversion",
            "finite_precision_effect": "reassociate_order_free_FP32_sum",
            "accumulation_order": "strided_thread_partials_then_warp_tree_then_sequential_warp_totals",
            "maximum_additions_on_path": self.items_per_thread + 5 + ROW_THREADS // 32,
            "conditional_error_bound": "gamma(maximum_additions_on_path) * sum(abs(realized_terms)); u=2^-24",
            "error_bound_assumptions": "finite terms, round-to-nearest, no overflow or underflow",
        }


def plan_fp32_row_fold(
    operations: tuple[TensorPrimitive, ...],
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
    policy: NumericalPolicy,
) -> Fp32RowFoldPlan | None:
    """Admit a closed Map-prefix row sum, retaining other generic lowering paths."""
    if _rejection_reason(operations, live_inputs, live_outputs, policy) is not None:
        return None
    return Fp32RowFoldPlan(operations, live_inputs, live_outputs, policy)
