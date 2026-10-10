# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    launch_dependency_event_schedule,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.layout_relation_canonicalization import bijective_layout_relation
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    FoldPrimitive,
    FoldReducer,
    IndexExpressionKind,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    scalar_input,
    scalar_unary,
    tensor_program_value_ordinals,
)
from tile_lifetime.terminal_gradient_layout import bind_dense_output_view, terminal_gradient_layout


def _view_program(shape, transform):
    relation_program = jax_function_to_tensor_program(
        transform, np.zeros(shape, dtype=np.float32), numerical_policy=NumericalPolicy.FAST
    )
    source = relation_program.inputs[0]
    reduction = TensorAxis(1000, 2)
    left = ProgramValue("left", (*source.axes, reduction), source.dtype)
    right = ProgramValue("right", (reduction,), source.dtype)
    terminal = ContractPrimitive("terminal", (left, right), source, (reduction,), source.dtype)
    program = TensorProgram(
        (left, right), (terminal, *relation_program.operations), relation_program.outputs, NumericalPolicy.FAST
    )
    return program, source


def _binding(program, source):
    return terminal_gradient_layout(program, tuple(range(len(program.operations))), 0, source, program.outputs)


def _physical_view(binding, source, physical, physical_axes=None):
    return binding.bind_storage(
        source.axes if physical_axes is None else physical_axes,
        physical.shape,
        tuple(stride // physical.itemsize for stride in physical.strides),
    )


def _read_view(view, storage):
    return np.ndarray(
        view.output.shape,
        dtype=storage.dtype,
        buffer=storage,
        strides=tuple(stride * storage.itemsize for stride in view.result_strides),
    )


@pytest.mark.parametrize(
    "shape,transform",
    [
        ((2, 12, 5), lambda x: x.reshape(2, 3, 4, 5)),
        ((2, 3, 4, 5), lambda x: x.reshape(2, 12, 5)),
        ((2, 3, 4), lambda x: x.transpose(2, 0, 1)),
        ((2, 12, 5), lambda x: x.reshape(2, 3, 4, 5).transpose(1, 0, 3, 2)),
        ((1, 2, 1, 12), lambda x: x.reshape(2, 3, 4)),
    ],
)
def test_gradient_views_match_independent_indexing_and_explicit_transport(shape, transform):
    program, source = _view_program(shape, transform)
    binding = _binding(program, source)
    storage = np.arange(np.prod(shape), dtype=np.float32)
    physical = storage.reshape(shape)
    view = _physical_view(binding, source, physical)
    actual = _read_view(view, storage)
    # The NumPy operation is independent of the imported symbolic Relation and
    # bound strides; every source element and destination coordinate is checked.
    expected = transform(physical)
    np.testing.assert_array_equal(actual, expected)
    assert np.shares_memory(actual, physical)
    assert actual.ctypes.data == physical.ctypes.data
    assert view.output.axes == program.outputs[0].axes
    transported = execute_tensor_program(view.transport_program(), {"physical_storage": storage})
    np.testing.assert_array_equal(transported[view.output.name], expected)
    assert view.contiguous == expected.flags.c_contiguous


def test_split_of_strided_axis_preserves_storage_and_explicit_copy():
    program, source = _view_program((3, 8, 5), lambda x: x.reshape(3, 2, 4, 5))
    storage = np.arange(120, dtype=np.float32)
    physical = storage.reshape(8, 3, 5).transpose(1, 0, 2)
    binding = _binding(program, source)
    view = _physical_view(binding, source, physical)
    expected = physical.reshape(3, 2, 4, 5)
    np.testing.assert_array_equal(_read_view(view, storage), expected)
    assert view.result_strides == (5, 60, 15, 1)
    assert not view.contiguous
    copied = execute_tensor_program(view.transport_program(), {"physical_storage": storage})
    np.testing.assert_array_equal(copied[view.output.name], expected)


def test_reshape_rejects_crossing_a_stride_discontinuity():
    program, source = _view_program((3, 8, 5), lambda x: x.reshape(24, 5))
    physical = np.arange(120, dtype=np.float32).reshape(8, 3, 5).transpose(1, 0, 2)
    with pytest.raises(ValueError):
        _physical_view(_binding(program, source), source, physical)


def test_packed_physical_axes_expand_without_losing_unit_or_feature_axes():
    batch, row, group, local, feature = tuple(TensorAxis(i, n) for i, n in enumerate((1, 3, 2, 4, 5)))
    output = ProgramValue("output", (batch, group, local, feature, row), DType.FP32)
    storage = np.arange(120, dtype=np.float32)
    physical = storage.reshape(1, 3, 8, 5)
    view = bind_dense_output_view(output, (batch, row, group, local, feature), physical.shape, (120, 40, 5, 1))
    np.testing.assert_array_equal(_read_view(view, storage), storage.reshape(1, 3, 2, 4, 5).transpose(0, 2, 3, 4, 1))
    assert view.output.axes == (batch, group, local, feature, row)
    with pytest.raises(ValueError):
        bind_dense_output_view(output, (batch, row, group, feature), physical.shape, (120, 40, 5, 1))


@pytest.mark.parametrize("change", ["shape", "stride", "offset", "dtype"])
def test_bound_view_rejects_changed_runtime_storage(change):
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    view = _physical_view(_binding(program, source), source, np.zeros((2, 6), np.float32))
    shape, strides, offset, dtype = (2, 6), (6, 1), 0, DType.FP32
    view.validate_storage(shape, strides, offset, dtype)
    if change == "shape":
        shape, strides = (3, 4), (4, 1)
    elif change == "stride":
        strides = (1, 2)
    elif change == "offset":
        offset = 1
    else:
        dtype = DType.BF16
    with pytest.raises(ValueError):
        view.validate_storage(shape, strides, offset, dtype)


@pytest.mark.parametrize("strides", [(1, 1), (12, 2), (-6, 1), (6, 0)])
def test_view_rejects_overlapping_padded_or_reversed_storage(strides):
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    with pytest.raises(ValueError):
        _binding(program, source).bind_storage(source.axes, (2, 6), strides)


@pytest.mark.parametrize("change", ["relation_predicate", "input_predicate", "alias", "offset", "arithmetic"])
def test_terminal_relation_rejects_unproved_indexing(change):
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    relation = program.operations[1]
    indexing = relation.input_indexing[0]
    predicate = index_compare(IndexPredicateKind.LESS, index_axis(relation.domain_axes[0]), index_constant(1))
    if change == "relation_predicate":
        relation = replace(relation, predicate=predicate)
    elif change == "input_predicate":
        relation = replace(relation, input_indexing=(replace(indexing, predicate=predicate),))
    elif change == "alias":
        relation = replace(
            relation, input_indexing=(replace(indexing, indices=(index_constant(0), indexing.indices[1])),)
        )
    elif change == "offset":
        relation = replace(
            relation,
            input_indexing=(
                replace(
                    indexing,
                    indices=(
                        indexing.indices[0],
                        index_binary(IndexExpressionKind.ADD, indexing.indices[1], index_constant(1)),
                    ),
                ),
            ),
        )
    else:
        # Index-expression results have their own integer semantics; equal shapes
        # do not make arithmetic a storage view.
        integer_source = replace(source, dtype=DType.INT32)
        integer_result = replace(relation.outputs[0], dtype=DType.INT32)
        relation = replace(
            relation,
            inputs=(integer_source,),
            outputs=(integer_result,),
            input_indexing=(replace(indexing, operand=integer_source),),
            result_sources=(None,),
            result_expressions=(index_constant(7),),
        )
    with pytest.raises(ValueError):
        bijective_layout_relation(relation)


def test_terminal_path_rejects_ambiguous_live_outputs_and_pointwise_arithmetic():
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    extra_output = replace(source, name="extra")
    extra = MapPrimitive(
        "negate", (source,), extra_output, scalar_unary(ScalarExpressionKind.NEGATE, scalar_input(source.name))
    )
    branched = replace(program, operations=(*program.operations, extra), outputs=(*program.outputs, extra_output))
    with pytest.raises(ValueError):
        _binding(branched, source)
    arithmetic = replace(program, operations=(program.operations[0], extra), outputs=(extra_output,))
    with pytest.raises(ValueError):
        _binding(arithmetic, source)


def test_nonunit_sum_remains_after_the_live_view_cut():
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    viewed = program.outputs[0]
    group = viewed.axes[1]
    result = ProgramValue("summed", (viewed.axes[0], viewed.axes[2]), viewed.dtype)
    fold = FoldPrimitive("group_sum", viewed, result, (group,), FoldReducer.SUM, viewed.dtype)
    extended = replace(program, operations=(*program.operations, fold), outputs=(result,))
    with pytest.raises(ValueError):
        _binding(extended, source)
    binding = terminal_gradient_layout(extended, (0, 1), 0, source, (viewed,))
    physical = np.arange(12, dtype=np.float32).reshape(2, 6)
    view = _physical_view(binding, source, physical)
    tail = TensorProgram((viewed,), (fold,), (result,), NumericalPolicy.FAST)
    actual = execute_tensor_program(tail, {viewed.name: _read_view(view, physical)})
    np.testing.assert_array_equal(actual[result.name], physical.reshape(2, 2, 3).sum(axis=1))


def test_captured_terminal_gradient_views_keep_all_axes_source_folds_and_live_cuts():
    path = Path(__file__).parent / "fixtures/dense_compile_boundary_program.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b7d8571ce93af918f632b9495e3a1af761f7694a13084465878d73cf66f2b5ea"
    )
    program = deserialize_tensor_program(path.read_text())
    fingerprint = program.semantic_fingerprint
    values = {ordinal: value for value, ordinal in tensor_program_value_ordinals(program).items()}
    owned = (*range(89, 111), 113)
    live_outputs = tuple(values[i] for i in (117, 120, 123))
    for terminal, primal, output, expected_path in (
        (98, 20, 117, (99, 100, 107)),
        (101, 24, 123, (102, 103, 108, 113)),
        (104, 26, 120, (105, 106, 109, 110)),
    ):
        binding = terminal_gradient_layout(program, owned, terminal, values[primal], live_outputs)
        view = binding.bind_storage(values[primal].axes, (1, 2048, 32, 128), (8388608, 4096, 128, 1))
        assert binding.output == values[output]
        assert binding.operation_ordinals == expected_path
        assert view.output.axes == values[output].axes
        assert view.output.shape == values[output].shape
        assert view.contiguous
        # Offset comparisons cover every output coordinate in bounded blocks,
        # independent of the source symbolic simplifier and view composition.
        if output != 117:
            assert view.result_strides[1:] == (4096, 512, 128, 1)
            for row in (0, 1, 1023, 2047):
                h, g, d = np.indices((8, 4, 128))
                canonical_offsets = (
                    row * view.result_strides[1]
                    + h * view.result_strides[2]
                    + g * view.result_strides[3]
                    + d * view.result_strides[4]
                )
                np.testing.assert_array_equal(canonical_offsets, (row * 32 + 4 * h + g) * 128 + d)
        assert set(binding.operation_ordinals).isdisjoint((111, 114))
    for ordinal, source_id, group_id in ((111, 120, 46), (114, 123, 48)):
        fold = program.operations[ordinal]
        assert fold.input == values[source_id]
        assert fold.reducer is FoldReducer.SUM
        assert fold.accumulation_dtype is fold.input.dtype is fold.output.dtype is DType.BF16
        assert [(axis.id, axis.extent) for axis in fold.reduction_axes] == [(group_id - 1, 1), (group_id, 4)]
    assert (
        program.semantic_fingerprint == fingerprint == "7f45e343c02efd500660e6f55b9ad6cd7abad195b71ddd415d6581e661bba933"
    )


@pytest.mark.parametrize("change", ("second_source", "second_result"))
def test_layout_relation_rejects_multiple_storage_sources_or_results(change):
    program, source = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    relation = program.operations[1]
    if change == "second_source":
        other = replace(source, name="other")
        relation = replace(
            relation,
            inputs=(source, other),
            input_indexing=(*relation.input_indexing, replace(relation.input_indexing[0], operand=other)),
            result_sources=(source,),
        )
    else:
        other = replace(relation.outputs[0], name="other")
        relation = replace(relation, outputs=(*relation.outputs, other), result_sources=(source, source))
    with pytest.raises(ValueError):
        bijective_layout_relation(relation)


@pytest.mark.parametrize("change", ("result_domain", "input_domain", "dtype"))
def test_serialized_terminal_relation_rejects_incomplete_domain_or_changed_dtype(change):
    program, _ = _view_program((2, 6), lambda x: x.reshape(2, 2, 3))
    record = program.canonical_record
    relation = record["operations"][1]
    if change == "result_domain":
        relation["domain_axes"] = relation["domain_axes"][:-1]
        relation["input_indexing"][0]["domain_axes"] = relation["domain_axes"]
    elif change == "input_domain":
        relation["input_indexing"][0]["domain_axes"] = relation["domain_axes"][:-1]
    else:
        relation["results"][0]["dtype"] = "bf16"
    # Invalid domain metadata is rejected at the real IR ingestion boundary,
    # before the stricter view proof can consume it.
    with pytest.raises(ValueError):
        deserialize_tensor_program(json.dumps(record))


def test_axis_rename_and_unit_adjoint_fold_preserve_bf16_storage():
    program, source = _view_program((2, 3), lambda x: x)
    source = replace(source, dtype=DType.BF16)
    terminal = replace(program.operations[0], output=source, accumulation_dtype=DType.BF16)
    a, b, unit = TensorAxis(20, 2), TensorAxis(21, 3), TensorAxis(22, 1)
    expanded = ProgramValue("expanded", (b, unit, a), DType.BF16)
    result = ProgramValue("renamed", (b, a), DType.BF16)
    relation = RelationPrimitive(
        "rename_expand",
        (source,),
        (expanded,),
        expanded.axes,
        (
            OperandIndexing(
                source,
                expanded.axes,
                (index_binary(IndexExpressionKind.ADD, index_axis(a), index_axis(unit)), index_axis(b)),
            ),
        ),
    )
    fold = FoldPrimitive("unit_sum", expanded, result, (unit,), FoldReducer.SUM, DType.BF16)
    program = TensorProgram(program.inputs, (terminal, relation, fold), (result,), NumericalPolicy.FAST)
    physical = np.array([[1.0078125, -0.25, 256.0], [-128.0, 0.001953125, 1.5]], dtype=jnp.bfloat16)
    view = _physical_view(_binding(program, source), source, physical)
    actual = _read_view(view, physical)
    np.testing.assert_array_equal(actual, physical.T)
    tail = TensorProgram((source,), (relation, fold), (result,), NumericalPolicy.FAST)
    executed = execute_tensor_program(tail, {source.name: physical})
    np.testing.assert_array_equal(actual, executed[result.name])
    assert view.output.dtype is DType.BF16
    assert np.shares_memory(actual, physical)
    assert actual.ctypes.data == physical.ctypes.data
    with pytest.raises(ValueError):
        _binding(replace(program, operations=(terminal, relation, replace(fold, accumulation_dtype=DType.FP32))), source)


def test_captured_live_gradient_storage_keeps_both_typed_event_dependencies():
    program = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    values = {ordinal: value for value, ordinal in tensor_program_value_ordinals(program).items()}
    live = tuple(values[i] for i in (117, 120, 123))
    bindings = tuple(
        terminal_gradient_layout(program, (*range(89, 111), 113), ordinal, values[primal], live)
        for ordinal, primal in ((98, 20), (101, 24), (104, 26))
    )
    views = {
        binding.output: binding.bind_storage(binding.source.axes, (1, 2048, 32, 128), (8388608, 4096, 128, 1))
        for binding in bindings
    }
    # Structural launches retain the actual captured value dependencies. These
    # are CPU Event proofs, not compiled kernels or a captured runtime recipe.
    gradient = CuteKernelLaunch(
        "gradient",
        "gradient",
        (1, 1, 1),
        (128, 1, 1),
        0,
        (),
        tuple(CutePipelineSlotArgument(v.name, v.name, CutePipelineSlotAccess.WRITE) for v in live),
    )
    folds = tuple(
        CuteKernelLaunch(
            f"fold_{ordinal}",
            f"fold_{ordinal}",
            (1, 1, 1),
            (128, 1, 1),
            0,
            (),
            (
                CutePipelineSlotArgument("source", program.operations[ordinal].input.name, CutePipelineSlotAccess.READ),
                CutePipelineSlotArgument(
                    "result", program.operations[ordinal].output.name, CutePipelineSlotAccess.WRITE
                ),
            ),
        )
        for ordinal in (111, 114)
    )
    used = (*live, program.operations[111].output, program.operations[114].output)
    canonical_slots = tuple(
        CutePipelineSlot(v.name, CutePipelineElementType.BF16, v.shape, CutePipelineSlotKind.SCRATCH, 16) for v in used
    )
    physical_slots = tuple(
        replace(slot, shape=views[v].source_shape) if v in views else slot
        for slot, v in zip(canonical_slots, used, strict=True)
    )
    schedule = launch_dependency_event_schedule((gradient, *folds), slots=canonical_slots)
    assert schedule == launch_dependency_event_schedule((gradient, *folds), slots=physical_slots)
    assert set(schedule["predecessor_pairs"]) == {((0,), (1,)), ((0,), (2,))}
    required = schedule["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
    assert {(tuple(edge["source"]), tuple(edge["target"])) for edge in required} == {((0,), (1,)), ((0,), (2,))}
    assert {v.name: np.prod(v.shape) for v in used} == {slot.name: np.prod(slot.shape) for slot in physical_slots}


def test_binding_uses_actual_unit_axis_strides_without_changing_element_offsets():
    program, source = _view_program((1, 2, 6), lambda x: x.reshape(1, 2, 2, 3))
    storage = np.arange(12, dtype=np.float32)
    physical = np.ndarray((1, 2, 6), dtype=np.float32, buffer=storage, strides=(4096, 24, 4))
    view = _physical_view(_binding(program, source), source, physical)
    view.validate_storage((1, 2, 6), (1024, 6, 1), 0, DType.FP32)
    np.testing.assert_array_equal(_read_view(view, storage), storage.reshape(1, 2, 2, 3))
    assert view.canonical_record["source_strides"] == (1024, 6, 1)
    assert view.canonical_record["storage_offset"] == 0
    with pytest.raises(ValueError):
        view.validate_storage((1, 2, 6), (12, 6, 1), 0, DType.FP32)
