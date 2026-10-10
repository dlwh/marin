# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.tensor_program import (
    EvaluationOrder,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    ScanDirection,
    ScanInputBinding,
    ScanOutputBinding,
    ScanPrimitive,
    ScanStateBinding,
    SelectionPrimitive,
    TensorAxis,
    TensorProgram,
    TransportPrimitive,
    deserialize_tensor_program,
    evaluate_scalar_expression,
    execute_tensor_program,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    index_select,
    index_tensor_element,
    infer_scalar_expression_dtype,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_select,
    scalar_unary,
    serialize_tensor_program,
    tensor_program_value_ordinals,
)


def test_scalar_ast_evaluator_preserves_composed_math_and_rounding() -> None:
    x = np.array([-3.0, 0.25, 4.0], dtype=np.float32)
    bounded = scalar_binary(
        ScalarExpressionKind.MINIMUM,
        scalar_binary(
            ScalarExpressionKind.MAXIMUM,
            scalar_unary(ScalarExpressionKind.NEGATE, scalar_input("x")),
            scalar_constant(-1.0),
        ),
        scalar_constant(2.0),
    )
    expression = scalar_convert(
        scalar_select(
            scalar_binary(ScalarExpressionKind.GREATER_EQUAL, scalar_input("x"), scalar_constant(0.0)),
            bounded,
            scalar_constant(0.0),
        ),
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )

    actual = evaluate_scalar_expression(expression, {"x": x})
    expected = np.where(x >= 0, np.minimum(np.maximum(-x, -1.0), 2.0), 0.0).astype(actual.dtype)
    np.testing.assert_array_equal(actual, expected)
    assert infer_scalar_expression_dtype(expression, {"x": DType.FP32}) is DType.BF16


def test_selection_dynamic_relation_round_trip_and_execute() -> None:
    row = TensorAxis(7, 2, "row")
    item = TensorAxis(11, 4, "item")
    slot = TensorAxis(13, 2, "slot")
    feature = TensorAxis(17, 3, "feature")
    score = ProgramValue("routing.score", (row, item), DType.FP32)
    valid = ProgramValue("routing.valid", (row, item), DType.BOOL)
    data = ProgramValue("payload", (item, feature), DType.FP32)
    decoy = ProgramValue("decoy", (item, feature), DType.FP32)
    selected = ProgramValue("selected.score", (row, slot), DType.FP32)
    indices = ProgramValue("selected.index", (row, slot), DType.INT32)
    selected_valid = ProgramValue("selected.valid", (row, slot), DType.BOOL)
    gathered = ProgramValue("gathered", (row, slot, feature), DType.FP32)
    selection = SelectionPrimitive(
        "bounded selection",
        score,
        selected,
        indices,
        item,
        2,
        valid,
        selected_valid,
        invalid_index=-99,
    )
    domain = (row, slot, feature)
    validity_predicate = IndexPredicate(
        IndexPredicateKind.IS_TRUE,
        expressions=(index_tensor_element(selected_valid, index_axis(row), index_axis(slot)),),
    )
    relation = RelationPrimitive(
        name="dynamic gather",
        inputs=(data, decoy, indices, selected_valid),
        outputs=(gathered,),
        domain_axes=domain,
        input_indexing=(
            OperandIndexing(
                data,
                domain,
                (index_tensor_element(indices, index_axis(row), index_axis(slot)), index_axis(feature)),
                validity_predicate,
            ),
            OperandIndexing(
                decoy,
                domain,
                (index_tensor_element(indices, index_axis(row), index_axis(slot)), index_axis(feature)),
                validity_predicate,
            ),
            OperandIndexing(indices, domain, (index_axis(row), index_axis(slot))),
            OperandIndexing(selected_valid, domain, (index_axis(row), index_axis(slot))),
        ),
        result_sources=(data,),
    )
    program = TensorProgram(
        (score, valid, data, decoy),
        (selection, relation),
        (selected, indices, gathered),
        NumericalPolicy.FAST,
    )

    score_array = np.array([[1, 4, 3, 2], [8, 7, 6, 9]], dtype=np.float32)
    valid_array = np.array([[False, True, False, False], [True, False, True, True]])
    data_array = np.arange(12, dtype=np.float32).reshape(4, 3)
    actual = execute_tensor_program(
        program,
        {
            "routing.score": score_array,
            "routing.valid": valid_array,
            "payload": data_array,
            "decoy": data_array + 100,
        },
    )
    np.testing.assert_array_equal(actual["selected.index"], [[1, -99], [3, 0]])
    expected_gather = np.array([[[3, 4, 5], [0, 0, 0]], [[9, 10, 11], [0, 1, 2]]], dtype=np.float32)
    np.testing.assert_array_equal(actual["gathered"], expected_gather)

    restored = deserialize_tensor_program(serialize_tensor_program(program))
    assert serialize_tensor_program(restored) == serialize_tensor_program(program)
    assert restored.semantic_fingerprint == program.semantic_fingerprint
    assert (
        replace(program, numerical_policy=NumericalPolicy.SOURCE_ORDERED).semantic_fingerprint
        != program.semantic_fingerprint
    )
    assert tuple(tensor_program_value_ordinals(program).values()) == tuple(range(8))
    assert primitive_outputs(selection) == (selected, indices, selected_valid)


def test_source_free_relation_materializes_axis_predicate_and_select() -> None:
    token = TensorAxis(3, 5, "token")
    slot = TensorAxis(5, 3, "slot")
    block = ProgramValue("block", (token, slot), DType.INT32)
    valid = ProgramValue("valid", (token, slot), DType.BOOL)
    block_index = index_binary(IndexExpressionKind.FLOOR_DIVIDE, index_axis(token), index_constant(2))
    is_valid = index_compare(IndexPredicateKind.LESS_EQUAL, index_axis(slot), block_index)
    clamped = index_select(is_valid, block_index, index_constant(-1))
    relation = RelationPrimitive(
        name="generated integer relation",
        inputs=(),
        outputs=(block, valid),
        domain_axes=(token, slot),
        input_indexing=(),
        result_expressions=(clamped, is_valid),
        order=EvaluationOrder.SOURCE_ORDERED,
    )
    program = TensorProgram((), (relation,), (block, valid))

    actual = execute_tensor_program(program, {})
    expected_valid = np.arange(3)[None, :] <= (np.arange(5)[:, None] // 2)
    np.testing.assert_array_equal(actual["valid"], expected_valid)
    np.testing.assert_array_equal(actual["block"], np.where(expected_valid, np.arange(5)[:, None] // 2, -1))
    assert serialize_tensor_program(
        deserialize_tensor_program(serialize_tensor_program(program))
    ) == serialize_tensor_program(program)


def test_scan_and_transport_execute_without_workload_semantics() -> None:
    step = TensorAxis(19, 3, "step")
    lane = TensorAxis(23, 2, "lane")
    replica = TensorAxis(29, 2, "replica")
    initial = ProgramValue("initial", (lane,), DType.FP32)
    source = ProgramValue("source", (step, lane), DType.FP32)
    body_state = ProgramValue("body.state", (lane,), DType.FP32)
    body_input = ProgramValue("body.input", (lane,), DType.FP32)
    body_next = ProgramValue("body.next", (lane,), DType.FP32)
    body = TensorProgram(
        (body_state, body_input),
        (
            MapPrimitive(
                "state update",
                (body_state, body_input),
                body_next,
                scalar_binary(
                    ScalarExpressionKind.ADD,
                    scalar_input(body_state.name),
                    scalar_input(body_input.name),
                ),
            ),
        ),
        (body_next,),
    )
    final = ProgramValue("final", (lane,), DType.FP32)
    scanned = ProgramValue("scanned", (step, lane), DType.FP32)
    scan_domain = (step, lane)
    scan = ScanPrimitive(
        name="prefix state",
        axis=step,
        body=body,
        state=(ScanStateBinding(initial, body_state, body_next, final),),
        scanned_inputs=(
            ScanInputBinding(
                source,
                body_input,
                OperandIndexing(source, scan_domain, (index_axis(step), index_axis(lane))),
            ),
        ),
        emitted_outputs=(
            ScanOutputBinding(
                body_next,
                scanned,
                OperandIndexing(scanned, scan_domain, (index_axis(step), index_axis(lane))),
            ),
        ),
        direction=ScanDirection.FORWARD,
    )
    replicated = ProgramValue("replicated", (step, replica, lane), DType.FP32)
    transport_domain = (step, replica, lane)
    transport = TransportPrimitive(
        name="replicate placement",
        source=scanned,
        destination=replicated,
        routing_inputs=(),
        domain_axes=transport_domain,
        source_indexing=OperandIndexing(scanned, transport_domain, (index_axis(step), index_axis(lane))),
        destination_indexing=OperandIndexing(
            replicated,
            transport_domain,
            (index_axis(step), index_axis(replica), index_axis(lane)),
        ),
    )
    program = TensorProgram((initial, source), (scan, transport), (final, replicated))
    source_array = np.arange(6, dtype=np.float32).reshape(3, 2)
    actual = execute_tensor_program(program, {"initial": np.ones(2, np.float32), "source": source_array})
    expected = np.cumsum(source_array, axis=0) + 1
    np.testing.assert_array_equal(actual["final"], expected[-1])
    np.testing.assert_array_equal(actual["replicated"], np.repeat(expected[:, None, :], 2, axis=1))
    restored = deserialize_tensor_program(serialize_tensor_program(program))
    assert restored.semantic_fingerprint == program.semantic_fingerprint

    reversed_source_index = index_binary(IndexExpressionKind.SUBTRACT, index_constant(step.extent - 1), index_axis(step))
    reversed_scan = replace(
        scan,
        scanned_inputs=(
            replace(
                scan.scanned_inputs[0],
                indexing=OperandIndexing(
                    source,
                    scan_domain,
                    (reversed_source_index, index_axis(lane)),
                ),
            ),
        ),
    )
    reversed_program = replace(program, operations=(reversed_scan, transport))
    reversed_actual = execute_tensor_program(
        reversed_program, {"initial": np.ones(2, np.float32), "source": source_array}
    )
    reversed_expected = np.cumsum(source_array[::-1], axis=0) + 1
    np.testing.assert_array_equal(reversed_actual["final"], reversed_expected[-1])
    np.testing.assert_array_equal(reversed_actual["replicated"], np.repeat(reversed_expected[:, None, :], 2, axis=1))
    assert reversed_program.semantic_fingerprint != program.semantic_fingerprint


def test_canonical_identity_ignores_names_but_preserves_scalar_semantics() -> None:
    axis = TensorAxis(41, 4, "named axis")
    source = ProgramValue("source", (axis,), DType.FP32)
    result = ProgramValue("result", (axis,), DType.FP32)
    maximum = MapPrimitive(
        "named operation",
        (source,),
        result,
        scalar_binary(ScalarExpressionKind.MAXIMUM, scalar_input(source.name), scalar_constant(0.0)),
    )
    program = TensorProgram((source,), (maximum,), (result,), NumericalPolicy.FAST)
    renamed_axis = TensorAxis(99, 4, "unrelated label")
    renamed_source = ProgramValue("alpha", (renamed_axis,), DType.FP32)
    renamed_result = ProgramValue("omega", (renamed_axis,), DType.FP32)
    renamed = TensorProgram(
        (renamed_source,),
        (
            MapPrimitive(
                "different operation name",
                (renamed_source,),
                renamed_result,
                scalar_binary(
                    ScalarExpressionKind.MAXIMUM,
                    scalar_input(renamed_source.name),
                    scalar_constant(0.0),
                ),
            ),
        ),
        (renamed_result,),
        NumericalPolicy.FAST,
    )
    assert renamed.semantic_fingerprint == program.semantic_fingerprint

    minimum = replace(
        program,
        operations=(
            replace(
                maximum,
                expression=scalar_binary(
                    ScalarExpressionKind.MINIMUM,
                    scalar_input(source.name),
                    scalar_constant(0.0),
                ),
            ),
        ),
    )
    assert minimum.semantic_fingerprint != program.semantic_fingerprint


def test_transport_rejects_implicit_many_to_one_reduction() -> None:
    source_axis = TensorAxis(47, 2)
    destination_axis = TensorAxis(53, 1)
    source = ProgramValue("source", (source_axis,), DType.FP32)
    destination = ProgramValue("destination", (destination_axis,), DType.FP32)
    transport = TransportPrimitive(
        "collision",
        source,
        destination,
        (),
        (source_axis,),
        OperandIndexing(source, (source_axis,), (index_axis(source_axis),)),
        OperandIndexing(destination, (source_axis,), (index_constant(0),)),
    )
    program = TensorProgram((source,), (transport,), (destination,))

    with pytest.raises(ValueError, match="explicit Fold"):
        execute_tensor_program(program, {"source": np.array([1, 2], dtype=np.float32)})
