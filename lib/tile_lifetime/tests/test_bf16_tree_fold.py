# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tile_lifetime.backends.h100.bf16_tree_fold import (
    Bf16TreeFoldPlan,
    require_bf16_tree_boundary,
    requires_bf16_tree_fold,
)
from tile_lifetime.tensor_program import (
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    NumericalPolicy,
    ProgramValue,
    TensorAxis,
    deserialize_tensor_program,
)


def _fold(shape, reductions):
    axes = tuple(TensorAxis(index, size, f"axis{index}") for index, size in enumerate(shape))
    source = ProgramValue("source", axes, DType.BF16)
    result = ProgramValue(
        "result", tuple(axis for index, axis in enumerate(axes) if index not in reductions), DType.BF16
    )
    return FoldPrimitive("sum", source, result, tuple(axes[index] for index in reductions), FoldReducer.SUM, DType.BF16)


def _recursive_reference(values):
    if len(values) == 1:
        return values[0]
    split = 1 << ((len(values) - 1).bit_length() - 1)
    return ml_dtypes.bfloat16(
        np.float32(_recursive_reference(values[:split])) + np.float32(_recursive_reference(values[split:]))
    )


def _execute_shared_plan(plan, values):
    """Interpret the typed geometry on CPU; this does not execute the CuTe kernel."""
    result = np.empty(plan.output_elements, dtype=ml_dtypes.bfloat16)
    writes = np.zeros(plan.output_elements, dtype=np.int32)
    source = values.reshape(-1)
    input_reads = []
    for block in range(plan.grid[0]):
        tile = plan.outputs_per_cta
        shared = np.full((plan.leaf_capacity, tile), np.nan, dtype=ml_dtypes.bfloat16)
        for local in range(tile):
            output = block * tile + local
            if output >= plan.output_elements:
                continue
            coordinates = np.unravel_index(output, plan.operation.output.shape)
            base = sum(x * stride for x, stride in zip(coordinates, plan.source_output_strides, strict=True))
            for reduction in range(plan.reduction_elements):
                coordinates = np.unravel_index(reduction, tuple(axis.extent for axis in plan.operation.reduction_axes))
                offset = base + sum(
                    x * stride for x, stride in zip(coordinates, plan.source_reduction_strides, strict=True)
                )
                input_reads.append(offset)
                shared[reduction, local] = source[offset]
        for level in range(plan.depth):
            distance = 1 << level
            for left in range(0, plan.reduction_elements, 2 * distance):
                if left + distance < plan.reduction_elements:
                    shared[left] = (shared[left].astype(np.float32) + shared[left + distance].astype(np.float32)).astype(
                        ml_dtypes.bfloat16
                    )
        for local in range(tile):
            output = block * tile + local
            if output < plan.output_elements:
                result[output] = shared[0, local]
                writes[output] += 1
    return result.reshape(plan.operation.output.shape), writes, input_reads


@pytest.mark.parametrize(
    "shape,reductions",
    [
        ((257,), (0,)),
        ((3, 513), (1,)),
        ((2, 257, 9), (1,)),
        ((2, 257, 3), (0, 1)),
        ((2, 257, 3), (1, 0)),
        ((1, 257, 1, 7), (0, 1, 2)),
        ((4096, 3), (0,)),
        ((1, 16384), (1,)),
    ],
)
def test_plan_preserves_storage_tree_rounding_and_exact_output_ownership(shape, reductions):
    operation = _fold(shape, reductions)
    plan = Bf16TreeFoldPlan(operation, NumericalPolicy.FAST)
    values = np.random.default_rng(91).uniform(-1, 1, shape).astype(ml_dtypes.bfloat16)
    actual, writes, reads = _execute_shared_plan(plan, values)
    kept = tuple(index for index in range(len(shape)) if index not in reductions)
    ordered = values.transpose((*kept, *reductions)).reshape(plan.output_elements, plan.reduction_elements)
    expected = np.array([_recursive_reference(row) for row in ordered], dtype=ml_dtypes.bfloat16)
    assert actual.tobytes() == expected.tobytes()
    assert np.all(writes == 1)
    assert sorted(reads) == list(range(values.size))
    assert plan.shared_bytes <= 32768
    exact = ordered.astype(np.float64).sum(axis=1)
    gamma = plan.depth * 2**-8 / (1 - plan.depth * 2**-8)
    bound = gamma * np.abs(ordered.astype(np.float64)).sum(axis=1)
    assert np.all(np.abs(actual.astype(np.float64).ravel() - exact) <= bound)


def test_tree_fixes_serial_bf16_absorption_without_promoting_accumulation():
    plan = Bf16TreeFoldPlan(_fold((2048,), (0,)), NumericalPolicy.FAST)
    ones = np.ones(2048, dtype=ml_dtypes.bfloat16)
    actual, _, _ = _execute_shared_plan(plan, ones)
    serial = ml_dtypes.bfloat16(0)
    for value in ones:
        serial = ml_dtypes.bfloat16(np.float32(serial) + np.float32(value))
    assert actual == 2048 and serial == 256
    values = np.zeros(2048, dtype=ml_dtypes.bfloat16)
    values[:3] = [1, 2**-8, 2**-8]
    rounded, _, _ = _execute_shared_plan(plan, values)
    promoted = values.astype(np.float32).sum().astype(ml_dtypes.bfloat16)
    assert rounded == 1 and promoted != rounded


@pytest.mark.parametrize(
    "change",
    [
        "source_ordered_program",
        "source_ordered_fold",
        "fp32_accumulator",
        "fp32_input",
        "fp32_output",
        "maximum",
        "short",
    ],
)
def test_ineligible_folds_keep_their_existing_semantics(change):
    operation = _fold((257,), (0,))
    policy = NumericalPolicy.FAST
    if change == "source_ordered_program":
        policy = NumericalPolicy.SOURCE_ORDERED
    elif change == "source_ordered_fold":
        operation = replace(operation, order=EvaluationOrder.SOURCE_ORDERED)
    elif change == "fp32_accumulator":
        operation = replace(operation, accumulation_dtype=DType.FP32)
    elif change == "fp32_input":
        operation = replace(operation, input=replace(operation.input, dtype=DType.FP32))
    elif change == "fp32_output":
        operation = replace(operation, output=replace(operation.output, dtype=DType.FP32))
    elif change == "maximum":
        operation = replace(operation, reducer=FoldReducer.MAXIMUM)
    else:
        operation = _fold((4,), (0,))
    assert not requires_bf16_tree_fold(operation, policy)
    with pytest.raises(ValueError):
        Bf16TreeFoldPlan(operation, policy)


@pytest.mark.parametrize("shape,reductions", [((16385,), (0,)), ((257, 1 << 24), (0,))])
def test_tree_rejects_unproved_resource_and_index_ranges(shape, reductions):
    operation = _fold(shape, reductions)
    assert requires_bf16_tree_fold(operation, NumericalPolicy.FAST)
    with pytest.raises(ValueError):
        Bf16TreeFoldPlan(operation, NumericalPolicy.FAST)


def test_long_fold_requires_existing_materialized_boundary_between_reductions():
    first = _fold((257, 3), (0,))
    result = ProgramValue("total", (), DType.BF16)
    second = FoldPrimitive("total", first.output, result, first.output.axes, FoldReducer.SUM, DType.BF16)
    with pytest.raises(ValueError):
        require_bf16_tree_boundary((first, second), NumericalPolicy.FAST)
    require_bf16_tree_boundary((first,), NumericalPolicy.FAST)
    require_bf16_tree_boundary((second,), NumericalPolicy.FAST)
    require_bf16_tree_boundary((first, second), NumericalPolicy.SOURCE_ORDERED)
    assert Bf16TreeFoldPlan(first, NumericalPolicy.FAST).operation.output is second.input


def test_exact_dense_program_selects_only_long_generic_bf16_sums():
    program = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    plans = {
        ordinal: Bf16TreeFoldPlan(operation, program.numerical_policy)
        for ordinal, operation in enumerate(program.operations)
        if requires_bf16_tree_fold(operation, program.numerical_policy)
    }
    assert set(plans) == {70, 75, 125, 130}
    assert [(plans[i].grid, plans[i].shared_bytes) for i in (70, 75, 125)] == [
        ((512, 1, 1), 32768),
        ((2048, 1, 1), 8192),
        ((512, 1, 1), 32768),
    ]
    assert all(plan.operation is program.operations[ordinal] for ordinal, plan in plans.items())
    for ordinal in (94, 111, 114):
        assert not requires_bf16_tree_fold(program.operations[ordinal], program.numerical_policy)
