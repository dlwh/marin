# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tile_lifetime.backends.h100.fp32_row_fold import Fp32RowFoldPlan, plan_fp32_row_fold
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import (
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    index_axis,
    index_compare,
    index_constant,
    scalar_binary,
    scalar_convert,
    scalar_input,
)


def _fold(shape=(3, 513), reduction_count=1, dtype=DType.FP32):
    axes = tuple(TensorAxis(i, extent) for i, extent in enumerate(shape))
    source = ProgramValue("source", axes, dtype)
    output = ProgramValue("sum", axes[:-reduction_count], DType.FP32)
    return FoldPrimitive("row sum", source, output, axes[-reduction_count:], FoldReducer.SUM, DType.FP32)


def _plan(operation, policy=NumericalPolicy.FAST):
    return plan_fp32_row_fold((operation,), (operation.input,), (operation.output,), policy)


def _interpret_row_geometry(plan, values):
    """CPU schedule interpretation; neither CuTe execution nor a GPU parity claim."""
    threads = plan.block[0]
    source = values.astype(np.float32).ravel()
    outputs = np.empty(plan.output_elements, np.float32)
    reads = []
    for row in range(plan.grid[0]):
        partials = np.zeros(threads, np.float32)
        for step in range(plan.items_per_thread):
            for thread in range(threads):
                reduction = thread + step * threads
                if reduction < plan.reduction_elements:
                    offset = row * plan.reduction_elements + reduction
                    reads.append(offset)
                    partials[thread] += source[offset]
        warps = partials.reshape(-1, 32)
        for distance in (16, 8, 4, 2, 1):
            warps = (warps[:, :distance] + warps[:, distance : 2 * distance]).astype(np.float32)
        total = np.float32(0)
        for subtotal in warps[:, 0]:
            total = np.float32(total + subtotal)
        outputs[row] = total
    return outputs.reshape(plan.fold.output.shape), reads


@pytest.mark.parametrize(
    "shape,reductions",
    [((3, 257), 1), ((2, 3, 513), 1), ((2, 3, 7, 129), 2), ((1, 65536), 1), ((5, 1, 1024), 2)],
)
@pytest.mark.parametrize("dtype", [DType.FP32, DType.BF16])
def test_row_plan_covers_storage_once_and_obeys_fp32_error_bound(shape, reductions, dtype):
    operation = _fold(shape, reductions, dtype)
    plan = _plan(operation)
    assert plan is not None
    storage_dtype = np.float32 if dtype is DType.FP32 else ml_dtypes.bfloat16
    values = np.random.default_rng(91).uniform(-1, 1, shape).astype(storage_dtype)
    actual, reads = _interpret_row_geometry(plan, values)
    exact = values.astype(np.float64).sum(axis=tuple(range(len(shape) - reductions, len(shape))))
    absolute_terms = np.abs(values.astype(np.float64)).sum(axis=tuple(range(len(shape) - reductions, len(shape))))
    depth = plan.canonical_record()["maximum_additions_on_path"]
    gamma = depth * 2**-24 / (1 - depth * 2**-24)
    assert np.all(np.abs(actual.astype(np.float64) - exact) <= gamma * absolute_terms)
    assert sorted(reads) == list(range(values.size))
    assert plan.grid[0] == exact.size
    assert plan.shared_bytes == 4 * plan.block[0] // 32
    # Independent multidimensional indexing fixes the row-to-source correspondence.
    for output_index in np.ndindex(operation.output.shape):
        row = np.ravel_multi_index(output_index, operation.output.shape)
        for reduction_index in (
            tuple(0 for _ in operation.reduction_axes),
            tuple(a.extent - 1 for a in operation.reduction_axes),
        ):
            reduced = np.ravel_multi_index(reduction_index, tuple(a.extent for a in operation.reduction_axes))
            assert row * plan.reduction_elements + reduced == np.ravel_multi_index(
                (*output_index, *reduction_index), shape
            )


def _mixed_prefix_program():
    fold = _fold((2, 3, 513))
    source = replace(fold.input, name="source", dtype=DType.BF16)
    offset = ProgramValue("offset", (source.axes[-1],), DType.FP32)
    scaled = replace(source, name="rounded", dtype=DType.BF16)
    squared = replace(source, name="squared", dtype=DType.FP32)
    maps = (
        MapPrimitive(
            "bias with rounding",
            (source, offset),
            scaled,
            scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_convert(
                    scalar_input(source.name), result_dtype=DType.FP32, semantics=ScalarConversionSemantics.EXACT
                ),
                scalar_input(offset.name),
            ),
        ),
        MapPrimitive(
            "square widened rounded value",
            (scaled,),
            squared,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_convert(
                    scalar_input(scaled.name), result_dtype=DType.FP32, semantics=ScalarConversionSemantics.EXACT
                ),
                scalar_convert(
                    scalar_input(scaled.name), result_dtype=DType.FP32, semantics=ScalarConversionSemantics.EXACT
                ),
            ),
        ),
    )
    fold = replace(fold, input=squared)
    return TensorProgram((source, offset), (*maps, fold), (fold.output,), NumericalPolicy.FAST)


def test_generic_broadcast_prefix_preserves_intermediate_bf16_rounding_and_fp32_result():
    program = _mixed_prefix_program()
    plan = plan_fp32_row_fold(program.operations, program.inputs, program.outputs, program.numerical_policy)
    assert plan is not None
    rng = np.random.default_rng(22)
    source = rng.uniform(-1, 1, program.inputs[0].shape).astype(ml_dtypes.bfloat16)
    offset = rng.uniform(-0.03, 0.03, program.inputs[1].shape).astype(np.float32)
    prefix_program = replace(program, operations=program.operations[:-1], outputs=(plan.fold.input,))
    realized = execute_tensor_program(prefix_program, {"source": source, "offset": offset})[plan.fold.input.name]
    rounded = (source.astype(np.float32) + offset).astype(ml_dtypes.bfloat16).astype(np.float32)
    independent_terms = rounded * rounded
    np.testing.assert_array_equal(realized, independent_terms)
    actual, _ = _interpret_row_geometry(plan, realized)
    expected = independent_terms.astype(np.float64).sum(axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
    unrounded = ((source.astype(np.float32) + offset) ** 2).sum(axis=-1)
    assert np.max(np.abs(actual - unrounded)) > 1e-3
    record = plan.canonical_record()
    assert record["prefix_result_dtypes"] == ("bf16", "fp32")
    assert record["accumulation_dtype"] == "fp32"


@pytest.mark.parametrize("ordinals", [(0, 1), (31,)])
def test_exact_dense_boundary_has_one_row_cta_without_source_or_event_rewrite(ordinals):
    full = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    operations = tuple(full.operations[i] for i in ordinals)
    inputs = (operations[0].inputs[0],) if isinstance(operations[0], MapPrimitive) else (operations[0].input,)
    program = TensorProgram(inputs, operations, (operations[-1].output,), full.numerical_policy)
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    composition = lower_canonical_kernel_composition(request)
    before = program.canonical_record
    plan = plan_fp32_row_fold(operations, inputs, program.outputs, program.numerical_policy)
    assert plan is not None
    assert plan.grid == (2048, 1, 1) and plan.block == (256, 1, 1) and plan.shared_bytes == 32
    assert plan.items_per_thread == 16
    assert plan.fold.input.dtype is DType.BF16 and plan.fold.accumulation_dtype is DType.FP32
    assert tuple(c.operation_ordinals for c in composition.components) == (tuple(range(len(operations))),)
    assert tuple(plan.operations) == tuple(full.operations[i] for i in ordinals)
    assert program.canonical_record == before
    assert (
        request.event_programs
        == jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90).event_programs
    )


@pytest.mark.parametrize(
    "change",
    [
        "ordered_program",
        "ordered_fold",
        "bf16_acc",
        "bf16_result",
        "maximum",
        "short",
        "too_long",
        "scalar",
        "strided",
        "permuted_k",
        "overflow",
    ],
)
def test_ineligible_folds_do_not_select_parallel_row_reassociation(change):
    operation = _fold()
    policy = NumericalPolicy.FAST
    if change == "ordered_program":
        policy = NumericalPolicy.SOURCE_ORDERED
    elif change == "ordered_fold":
        operation = replace(operation, order=EvaluationOrder.SOURCE_ORDERED)
    elif change == "bf16_acc":
        operation = replace(operation, accumulation_dtype=DType.BF16)
    elif change == "bf16_result":
        operation = replace(operation, output=replace(operation.output, dtype=DType.BF16))
    elif change == "maximum":
        operation = replace(operation, reducer=FoldReducer.MAXIMUM)
    elif change == "short":
        operation = _fold((3, 256))
    elif change == "too_long":
        operation = _fold((3, 65537))
    elif change == "scalar":
        operation = _fold((513,), 1)
    elif change == "strided":
        operation = _fold((513, 3))
        operation = replace(
            operation,
            output=replace(operation.output, axes=(operation.input.axes[1],)),
            reduction_axes=(operation.input.axes[0],),
        )
    elif change == "permuted_k":
        operation = _fold((3, 17, 31), 2)
        operation = replace(operation, reduction_axes=operation.reduction_axes[::-1])
    else:
        operation = _fold((1 << 23, 513))
    assert _plan(operation, policy) is None
    with pytest.raises(ValueError):
        Fp32RowFoldPlan((operation,), (operation.input,), (operation.output,), policy)


@pytest.mark.parametrize(
    "change",
    [
        "side_output",
        "extra_input",
        "missing_input",
        "duplicate_input",
        "predicate",
        "relation",
        "permuted_operand",
        "disconnected_map",
        "duplicate_producer",
        "future_value",
    ],
)
def test_prefix_admission_requires_exact_source_order_domain_and_live_boundary(change):
    program = _mixed_prefix_program()
    operations, inputs, outputs = program.operations, program.inputs, program.outputs
    if change == "side_output":
        outputs = (*outputs, operations[0].output)
    elif change == "extra_input":
        inputs = (*inputs, replace(inputs[0], name="unused"))
    elif change == "missing_input":
        inputs = inputs[:1]
    elif change == "duplicate_input":
        inputs = (*inputs, inputs[0])
    elif change == "predicate":
        pred = index_compare(IndexPredicateKind.LESS, index_axis(inputs[0].axes[-1]), index_constant(400))
        operations = (replace(operations[0], predicate=pred, rejected_value=0.0), *operations[1:])
    elif change == "relation":
        first = operations[0]
        relation = RelationPrimitive(
            "view",
            (inputs[0],),
            (first.output,),
            first.output.axes,
            (OperandIndexing(inputs[0], first.output.axes, tuple(index_axis(axis) for axis in inputs[0].axes)),),
        )
        operations = (relation, *operations[1:])
        inputs = inputs[:1]
    elif change == "permuted_operand":
        changed = replace(inputs[0], axes=inputs[0].axes[::-1])
        operations = (replace(operations[0], inputs=(changed, inputs[1])), *operations[1:])
        inputs = (changed, inputs[1])
    elif change == "disconnected_map":
        operations = (replace(operations[0], output=replace(operations[0].output, name="dead")), *operations)
    elif change == "duplicate_producer":
        operations = (operations[0], *operations)
    else:
        operations = (operations[1], operations[0], operations[2])
    assert plan_fp32_row_fold(operations, inputs, outputs, program.numerical_policy) is None


@pytest.mark.parametrize(
    "dtype,conversion",
    [
        (DType.FP64, ScalarConversionSemantics.EXACT),
        (DType.INT32, ScalarConversionSemantics.EXACT),
        (DType.BF16, ScalarConversionSemantics.EXACT),
    ],
)
def test_row_prefix_rejects_conversions_without_a_matching_physical_implementation(dtype, conversion):
    operation = _fold(dtype=DType.BF16)
    mapped = replace(operation.input, name="mapped")
    expression = scalar_convert(scalar_input(operation.input.name), result_dtype=dtype, semantics=conversion)
    expression = scalar_convert(expression, result_dtype=DType.FP32, semantics=ScalarConversionSemantics.EXACT)
    prefix = MapPrimitive("unsupported conversion", (operation.input,), mapped, expression)
    fold = replace(operation, input=mapped)
    assert plan_fp32_row_fold((prefix, fold), (operation.input,), (fold.output,), NumericalPolicy.FAST) is None


@pytest.mark.parametrize("input_count", [8, 9])
def test_row_prefix_retains_padded_abi_input_bound(input_count):
    fold = _fold()
    inputs = tuple(replace(fold.input, name=f"input{i}") for i in range(input_count))
    current = inputs[0]
    maps = []
    for index, value in enumerate(inputs[1:]):
        output = replace(current, name=f"added{index}")
        maps.append(
            MapPrimitive(
                "add",
                (current, value),
                output,
                scalar_binary(ScalarExpressionKind.ADD, scalar_input(current.name), scalar_input(value.name)),
            )
        )
        current = output
    operations = (*maps, replace(fold, input=current))
    plan = plan_fp32_row_fold(operations, inputs, (fold.output,), NumericalPolicy.FAST)
    assert (plan is not None) == (input_count == 8)
