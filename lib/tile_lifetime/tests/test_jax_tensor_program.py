# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.jax_tensor_program import JaxprConversionError, jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.row_normalization_jax import (
    RowProjectionConfig,
    layer_projection,
    projection_training_step,
    projection_value_and_grad,
    rms_projection,
)
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import (
    ContractOperandPrecision,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    MapPrimitive,
    NumericalPolicy,
    RelationPrimitive,
    ScalarExpressionKind,
    ScanDirection,
    ScanPrimitive,
    deserialize_tensor_program,
    execute_tensor_program,
    serialize_tensor_program,
    tensor_program_fingerprint,
)


def _arguments(config: RowProjectionConfig) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(113)
    return (
        jnp.asarray(rng.normal(size=(config.rows, config.hidden)).astype(np.float32)),
        jnp.asarray(rng.normal(size=(config.hidden,)).astype(np.float32)),
        jnp.asarray(rng.normal(size=(config.hidden, config.features)).astype(np.float32)),
        jnp.asarray(rng.normal(size=(config.rows, config.features)).astype(np.float32)),
    )


def _execute(function, arguments: tuple[jax.Array, ...]):
    program = jax_function_to_tensor_program(function, *arguments)
    inputs = {value.name: np.asarray(argument) for value, argument in zip(program.inputs, arguments, strict=True)}
    return program, tuple(execute_tensor_program(program, inputs).values())


def _affine_scan(initial, sequence, scale):
    def step(state, value):
        return state * scale + value, state * value

    return jax.lax.scan(step, initial, sequence)


def test_lax_scan_import_preserves_invariants_direction_and_canonical_identity() -> None:
    initial = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    sequence = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 7
    scale = jnp.asarray([0.75, 0.5], dtype=jnp.float32)

    def reverse_scan(state, values, multiplier):
        def step(carry, value):
            return carry * multiplier + value, carry * value

        return jax.lax.scan(step, state, values, reverse=True)

    program, actual = _execute(reverse_scan, (initial, sequence, scale))
    expected = jax.tree.leaves(reverse_scan(initial, sequence, scale))
    scan = program.operations[0]

    assert isinstance(scan, ScanPrimitive)
    assert scan.direction is ScanDirection.REVERSE
    assert len(scan.invariants) == 1
    assert scan.invariants[0].source == program.inputs[2]
    assert scan.axis.extent == sequence.shape[0]
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    decoded = deserialize_tensor_program(serialize_tensor_program(program))
    assert tensor_program_fingerprint(decoded) == tensor_program_fingerprint(program)


def test_jax_owned_scan_pullback_imports_as_forward_and_reverse_generic_scans() -> None:
    initial = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    sequence = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 7
    scale = jnp.asarray([0.75, 0.5], dtype=jnp.float32)
    final_cotangent = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    output_cotangent = jnp.ones_like(sequence)

    def forward_and_pullback(state, values, multiplier, final_ct, output_ct):
        result, pullback = jax.vjp(_affine_scan, state, values, multiplier)
        return (*result, *pullback((final_ct, output_ct)))

    arguments = (initial, sequence, scale, final_cotangent, output_cotangent)
    program, actual = _execute(forward_and_pullback, arguments)
    expected = jax.tree.leaves(forward_and_pullback(*arguments))
    scans = tuple(operation for operation in program.operations if isinstance(operation, ScanPrimitive))

    assert tuple(scan.direction for scan in scans) == (ScanDirection.FORWARD, ScanDirection.REVERSE)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-6)


def test_unsupported_scan_body_primitive_reports_outer_and_body_equations() -> None:
    def function(initial, sequence):
        def step(state, value):
            return state + value, jnp.sin(value)

        return jax.lax.scan(step, initial, sequence)

    with pytest.raises(JaxprConversionError) as raised:
        jax_function_to_tensor_program(
            function,
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.ones((3, 1), dtype=jnp.float32),
        )

    assert raised.value.primitive == "sin"
    assert raised.value.equation == 0
    assert raised.value.equation_path == (0, 1)


@pytest.mark.parametrize("factory", (rms_projection, layer_projection))
def test_ordinary_jax_forward_converts_to_generic_algebra_and_matches_jax(factory) -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=3e-4)
    arguments = _arguments(config)[:3]
    function = factory(config)

    program, actual = _execute(function, arguments)

    np.testing.assert_allclose(actual[0], function(*arguments), rtol=3e-6, atol=3e-6)
    assert {type(operation) for operation in program.operations} == {
        ContractPrimitive,
        FoldPrimitive,
        MapPrimitive,
        RelationPrimitive,
    }
    serialized = serialize_tensor_program(program).lower()
    assert all(marker not in serialized for marker in ("rms", "layernorm", "normalization"))


@pytest.mark.parametrize("factory", (rms_projection, layer_projection))
@pytest.mark.parametrize("ad_boundary", (projection_training_step, projection_value_and_grad))
def test_jax_owned_pullback_converts_without_custom_derivative(factory, ad_boundary) -> None:
    config = RowProjectionConfig(rows=2, hidden=5, features=3, epsilon=2e-4)
    arguments = _arguments(config)
    function = ad_boundary(factory(config))
    closed_jaxpr = jax.make_jaxpr(function)(*arguments)

    program, actual = _execute(function, arguments)
    expected = tuple(np.asarray(value) for value in jax.tree.leaves(function(*arguments)))

    assert "custom" not in {equation.primitive.name for equation in closed_jaxpr.jaxpr.eqns}
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)
    assert any(isinstance(operation, ContractPrimitive) for operation in program.operations)
    assert any(isinstance(operation, FoldPrimitive) for operation in program.operations)


def test_unsupported_jax_primitive_reports_the_exact_equation() -> None:
    argument = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(JaxprConversionError) as raised:
        jax_function_to_tensor_program(jnp.sin, argument)

    assert raised.value.primitive == "sin"
    assert raised.value.equation == 0
    assert raised.value.reason == "no generic Shuttle algebra conversion is registered"


def test_stop_gradient_converts_as_a_generic_identity_relation() -> None:
    argument = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    program, actual = _execute(jax.lax.stop_gradient, (argument,))

    np.testing.assert_array_equal(actual[0], argument)
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], RelationPrimitive)


def test_static_slice_converts_as_a_generic_affine_relation() -> None:
    argument = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)

    def function(value):
        return jax.lax.slice(value, (1, 0), (4, 5), (2, 1))

    program, actual = _execute(function, (argument,))

    np.testing.assert_array_equal(actual[0], function(argument))
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], RelationPrimitive)


def test_nested_call_jaxprs_inline_and_bind_multiple_results() -> None:
    @jax.jit
    def nested(left, right):
        positive = left > 0
        return jnp.where(positive, left, right), jnp.where(positive, right, left), left + right

    left = jnp.asarray([-2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([5.0, 7.0], dtype=jnp.float32)

    program, actual = _execute(nested, (left, right))
    expected = nested(left, right)

    assert len(actual) == 3
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert all(operation.name.split(".")[-1] != "jit" for operation in program.operations)


def test_static_integer_power_lowers_to_scalar_arithmetic() -> None:
    argument = jnp.asarray([[2.0, 4.0]], dtype=jnp.float32)

    def function(value):
        return jax.lax.integer_pow(value, -2)

    program, actual = _execute(function, (argument,))

    np.testing.assert_array_equal(actual[0], function(argument))
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], MapPrimitive)


def test_bf16_integer_power_retains_typed_identity_constants() -> None:
    argument = jnp.asarray([[2.0, 4.0]], dtype=jnp.bfloat16)

    def function(value):
        return jax.lax.integer_pow(value, -2)

    program, actual = _execute(function, (argument,))
    operation = program.operations[0]

    np.testing.assert_array_equal(actual[0], function(argument))
    assert isinstance(operation, MapPrimitive)
    numerator, denominator = operation.expression.operands
    assert numerator.kind is ScalarExpressionKind.CONVERT
    assert numerator.result_dtype is DType.BF16
    assert denominator.operands[0].operands[0] == numerator


def test_bf16_reduce_precision_lowers_to_explicit_round_and_widen() -> None:
    argument = jnp.asarray([[1.003, -2.007]], dtype=jnp.float32)

    def function(value):
        return jax.lax.reduce_precision(value, exponent_bits=8, mantissa_bits=7)

    program, actual = _execute(function, (argument,))
    operation = program.operations[0]

    np.testing.assert_array_equal(actual[0], function(argument))
    assert isinstance(operation, MapPrimitive)
    widen = operation.expression
    assert widen.kind is ScalarExpressionKind.CONVERT
    assert widen.result_dtype is DType.FP32
    assert widen.operands[0].kind is ScalarExpressionKind.CONVERT
    assert widen.operands[0].result_dtype is DType.BF16


def test_reduce_precision_rejects_an_unrepresentable_scalar_format() -> None:
    argument = jnp.asarray([[1.0]], dtype=jnp.float32)

    with pytest.raises(JaxprConversionError, match="only the e8m7 format corresponding to BF16"):
        jax_function_to_tensor_program(
            lambda value: jax.lax.reduce_precision(value, exponent_bits=5, mantissa_bits=10),
            argument,
        )


def test_logistic_lowers_to_composed_scalar_arithmetic() -> None:
    argument = jnp.asarray([[-2.0, 0.0, 3.0]], dtype=jnp.float32)

    program, actual = _execute(jax.lax.logistic, (argument,))

    np.testing.assert_allclose(actual[0], jax.lax.logistic(argument), rtol=2e-7, atol=2e-7)
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], MapPrimitive)


def test_nested_unsupported_primitive_reports_outer_and_inner_equations() -> None:
    @jax.jit
    def nested(value):
        return jnp.sin(value)

    with pytest.raises(JaxprConversionError) as raised:
        jax_function_to_tensor_program(nested, jnp.ones((2,), dtype=jnp.float32))

    assert raised.value.primitive == "sin"
    assert raised.value.equation == 0
    assert raised.value.equation_path == (0, 0)


def test_contract_clones_independent_free_axis_occurrences() -> None:
    def projected_pairwise_contract(source, left_weight, right_weight):
        left = source @ left_weight
        right = source @ right_weight
        return jnp.einsum("qd,kd->qk", left, right)

    source = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7
    left_weight = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 11
    right_weight = jnp.flip(left_weight, axis=1)

    program, actual = _execute(projected_pairwise_contract, (source, left_weight, right_weight))

    np.testing.assert_allclose(
        actual[0],
        projected_pairwise_contract(source, left_weight, right_weight),
        rtol=2e-6,
        atol=2e-6,
    )
    output = program.outputs[0]
    assert output.axes[0] != output.axes[1]
    assert output.axes[0].extent == output.axes[1].extent == source.shape[0]
    terminal_contract = program.operations[-1]
    assert isinstance(terminal_contract, ContractPrimitive)
    assert all(terminal_contract.input_index_maps)


def test_full_slice_gather_lowers_to_exact_relation() -> None:
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )

    def gather(operand, indices):
        return jax.lax.gather(
            operand,
            indices[..., None],
            dimensions,
            slice_sizes=(1, 6, 4),
            mode="promise_in_bounds",
        )

    operand = jnp.arange(3 * 6 * 4, dtype=jnp.bfloat16).reshape(3, 6, 4)
    indices = jnp.asarray([[2, 0], [1, 2], [0, 0], [2, 1]], dtype=jnp.int32)

    program, actual = _execute(gather, (operand, indices))

    np.testing.assert_array_equal(actual[0], gather(operand, indices))
    relation = program.operations[-1]
    assert isinstance(relation, RelationPrimitive)
    assert relation.result_sources == (relation.inputs[0],)
    assert actual[0].shape == (4, 2, 6, 4)


def test_batched_full_slice_gather_isolates_each_operand_batch() -> None:
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )

    def gather(operand, indices):
        return jax.lax.gather(
            operand,
            indices,
            dimensions,
            slice_sizes=(1, 1, 4),
            mode="promise_in_bounds",
        )

    operand = jnp.arange(2 * 3 * 4, dtype=jnp.bfloat16).reshape(2, 3, 4)
    indices = jnp.asarray([[[0], [2]], [[1], [0]]], dtype=jnp.int32)

    program, actual = _execute(gather, (operand, indices))

    np.testing.assert_array_equal(actual[0], gather(operand, indices))
    np.testing.assert_array_equal(
        actual[0],
        np.stack((np.asarray(operand[0])[[0, 2]], np.asarray(operand[1])[[1, 0]])),
    )
    assert isinstance(program.operations[-1], RelationPrimitive)


@pytest.mark.parametrize("mode", ("clip", "fill"))
def test_full_slice_gather_rejects_non_promised_out_of_bounds_policies(mode: str) -> None:
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )

    def gather(operand, indices):
        return jax.lax.gather(
            operand,
            indices[..., None],
            dimensions,
            slice_sizes=(1, 6, 4),
            mode=mode,
        )

    operand = jnp.zeros((3, 6, 4), dtype=jnp.bfloat16)
    out_of_bounds = jnp.asarray([[-1, 3]], dtype=jnp.int32)

    with pytest.raises(JaxprConversionError) as raised:
        jax_function_to_tensor_program(gather, operand, out_of_bounds)

    assert raised.value.primitive == "gather"
    assert raised.value.reason == "only promise-in-bounds gather is supported"


def test_bfloat16_scalar_literal_lowers_as_a_generic_constant() -> None:
    argument = jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16)

    def increment(value):
        return value + jnp.asarray(1.0, dtype=jnp.bfloat16)

    program, actual = _execute(increment, (argument,))

    np.testing.assert_array_equal(actual[0], increment(argument))
    assert "convert" in serialize_tensor_program(program)


def test_integer_literal_retains_its_promoted_jax_dtype() -> None:
    argument = jnp.asarray([-1, 0, 2], dtype=jnp.int32)

    def is_negative(value):
        return value < 0

    program, actual = _execute(is_negative, (argument,))

    np.testing.assert_array_equal(actual[0], argument < 0)
    assert "int32" in serialize_tensor_program(program)


def test_additive_full_window_scatter_lowers_to_relation_fold_map() -> None:
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def scatter(operand, indices, updates):
        return jax.lax.scatter_add(
            operand,
            indices,
            updates,
            dimensions,
            mode="promise_in_bounds",
        )

    operand = jnp.ones((3, 4, 6), dtype=jnp.bfloat16)
    indices = jnp.asarray([[[1], [2]], [[1], [0]]], dtype=jnp.int32)
    updates = jnp.arange(2 * 2 * 4 * 6, dtype=jnp.bfloat16).reshape(2, 2, 4, 6) / 32

    program, actual = _execute(scatter, (operand, indices, updates))

    np.testing.assert_array_equal(actual[0], scatter(operand, indices, updates))
    assert tuple(type(operation) for operation in program.operations) == (
        RelationPrimitive,
        FoldPrimitive,
        MapPrimitive,
    )


def test_additive_scatter_rejects_non_promised_out_of_bounds_policy() -> None:
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def scatter(operand, indices, updates):
        return jax.lax.scatter_add(operand, indices, updates, dimensions, mode="clip")

    operand = jnp.zeros((3, 4, 6), dtype=jnp.bfloat16)
    indices = jnp.asarray([[[-1], [3]]], dtype=jnp.int32)
    updates = jnp.ones((1, 2, 4, 6), dtype=jnp.bfloat16)

    with pytest.raises(JaxprConversionError) as raised:
        jax_function_to_tensor_program(scatter, operand, indices, updates)

    assert raised.value.primitive == "scatter-add"
    assert raised.value.reason == "only promise-in-bounds scatter-add is supported"


def test_batched_additive_scatter_isolates_batches_and_sums_collisions() -> None:
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    def scatter(operand, indices, updates):
        return jax.lax.scatter_add(
            operand,
            indices,
            updates,
            dimensions,
            mode="promise_in_bounds",
        )

    operand = jnp.asarray([[10, 20, 30], [100, 200, 300]], dtype=jnp.bfloat16)
    indices = jnp.asarray([[[1], [1], [2]], [[0], [2], [2]]], dtype=jnp.int32)
    updates = jnp.asarray([[1, 2, 4], [8, 16, 32]], dtype=jnp.bfloat16)

    program, actual = _execute(scatter, (operand, indices, updates))

    np.testing.assert_array_equal(actual[0], scatter(operand, indices, updates))
    np.testing.assert_array_equal(actual[0], np.asarray([[10, 23, 34], [108, 200, 348]], dtype=np.float32))
    assert tuple(type(operation) for operation in program.operations) == (
        RelationPrimitive,
        FoldPrimitive,
        MapPrimitive,
    )


def test_rank_zero_literal_dot_lowers_to_pointwise_map() -> None:
    def scalar_dot(vector):
        return jax.lax.dot_general(
            jnp.asarray(1.5, dtype=jnp.float32),
            vector,
            (((), ()), ((), ())),
        )

    vector = jnp.arange(16, dtype=jnp.float32) / 8

    program, actual = _execute(scalar_dot, (vector,))

    np.testing.assert_array_equal(actual[0], scalar_dot(vector))
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], MapPrimitive)


def test_dot_general_keeps_unspecified_reduction_order_across_bf16_conversion() -> None:
    def projected(left, right):
        product = jax.lax.dot_general(
            left,
            right,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return product.astype(jnp.bfloat16)

    left = jnp.ones((4,), dtype=jnp.float32)
    right = jnp.ones((4,), dtype=jnp.float32)
    program = jax_function_to_tensor_program(projected, left, right)

    contract, conversion = program.operations
    assert program.numerical_policy.value == "source_ordered"
    assert isinstance(contract, ContractPrimitive)
    assert contract.order is EvaluationOrder.ORDER_FREE
    assert isinstance(conversion, MapPrimitive)
    assert conversion.expression.kind is ScalarExpressionKind.CONVERT
    assert conversion.expression.result_dtype is DType.BF16

    # Both sums are legal associations of the same realized FP32 terms. The
    # small pre-conversion difference crosses a BF16 midpoint, so the explicit
    # conversion remains a separate discontinuous Map rather than being hidden
    # by the Contract's reassociation permission.
    terms = tuple(np.float32(value) for value in (1.0, 2**-8 + 2**-23, 131072.0, -131072.0))
    left_associated = np.float32(np.float32(np.float32(terms[0] + terms[1]) + terms[2]) + terms[3])
    cancellation_first = np.float32(np.float32(terms[2] + terms[3]) + np.float32(terms[0] + terms[1]))

    assert left_associated == np.float32(1.0)
    assert cancellation_first > np.float32(1.0 + 2**-8)
    assert np.asarray(left_associated, dtype=jnp.bfloat16) == jnp.bfloat16(1.0)
    assert np.asarray(cancellation_first, dtype=jnp.bfloat16) == jnp.bfloat16(1.0 + 2**-7)


@pytest.mark.parametrize("factory", (rms_projection, layer_projection))
def test_jax_normalization_value_and_grad_lowers_to_acyclic_tasks(factory) -> None:
    config = RowProjectionConfig(rows=8, hidden=16, features=8)
    arguments = _arguments(config)
    function = projection_value_and_grad(factory(config))
    program = jax_function_to_tensor_program(function, *arguments, numerical_policy=NumericalPolicy.FAST)

    # Construction verifies the complete Event graph, including cycles introduced
    # by contracting multiple individually convex operation groups.
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    lowered = deserialize_tensor_program(json.dumps(request.algebra_record))
    inputs = {value.name: np.asarray(argument) for value, argument in zip(lowered.inputs, arguments, strict=True)}
    actual = tuple(execute_tensor_program(lowered, inputs).values())
    expected = jax.tree.leaves(function(*arguments))
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
@pytest.mark.parametrize("precision", (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST))
def test_dot_precision_survives_import_rewrites_and_roundtrip(dtype, precision) -> None:
    left = jnp.arange(12, dtype=dtype).reshape(3, 4) / 8
    right = jnp.arange(8, dtype=dtype).reshape(4, 2) / 8

    def forward(x, weight):
        return jnp.matmul(x.astype(jnp.float32), weight.astype(jnp.float32), precision=precision)

    program = jax_function_to_tensor_program(forward, left, right, numerical_policy=NumericalPolicy.FAST)
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    decoded = deserialize_tensor_program(json.dumps(request.algebra_record))
    contracts = tuple(operation for operation in decoded.operations if isinstance(operation, ContractPrimitive))
    expected_precision = (ContractOperandPrecision(precision.name.lower()),) * 2
    assert tuple(operation.operand_precision for operation in contracts) == (expected_precision,)
    assert decoded.numerical_policy is NumericalPolicy.FAST
    assert deserialize_tensor_program(serialize_tensor_program(decoded)) == decoded
    actual = tuple(
        execute_tensor_program(
            decoded,
            {value.name: np.asarray(argument) for value, argument in zip(decoded.inputs, (left, right), strict=True)},
        ).values()
    )
    np.testing.assert_allclose(actual[0], forward(left, right), rtol=2e-5, atol=2e-5)


def test_mixed_dot_precision_changes_source_identity_and_survives_jax_ad() -> None:
    left = jnp.ones((3, 4), dtype=jnp.float32)
    right = jnp.ones((4, 2), dtype=jnp.float32)

    def objective(x, weight, precision):
        return jnp.sum(jnp.matmul(x, weight, precision=precision))

    default = jax_function_to_tensor_program(lambda x, w: objective(x, w, None), left, right)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST)
    explicit = jax_function_to_tensor_program(lambda x, w: objective(x, w, precision), left, right)
    differentiated = jax_function_to_tensor_program(
        jax.value_and_grad(lambda x, w: objective(x, w, precision), argnums=(0, 1)),
        left,
        right,
    )
    assert default.semantic_fingerprint != explicit.semantic_fingerprint
    assert all(
        operation.operand_precision == (ContractOperandPrecision.HIGH, ContractOperandPrecision.HIGHEST)
        for operation in differentiated.operations
        if isinstance(operation, ContractPrimitive)
    )
    actual = tuple(
        execute_tensor_program(
            differentiated,
            {
                value.name: np.asarray(argument)
                for value, argument in zip(differentiated.inputs, (left, right), strict=True)
            },
        ).values()
    )
    expected = jax.tree.leaves(jax.value_and_grad(lambda x, w: objective(x, w, precision), argnums=(0, 1))(left, right))
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    "algorithm", (jax.lax.DotAlgorithmPreset.TF32_TF32_F32, jax.lax.DotAlgorithm(jnp.float32, jnp.float32, jnp.float32))
)
def test_explicit_dot_algorithm_is_rejected_instead_of_discarded(algorithm) -> None:
    arguments = (jnp.ones((2, 3)), jnp.ones((3, 4)))
    with pytest.raises(JaxprConversionError, match="explicit dot algorithms are not supported"):
        jax_function_to_tensor_program(lambda x, w: jnp.matmul(x, w, precision=algorithm), *arguments)
