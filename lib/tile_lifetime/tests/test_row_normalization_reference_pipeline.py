# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_map_load_contract import (
    Sm90MapLoadContractSchedule,
    generate_sm90_map_load_contract,
    run_sm90_map_load_contract,
)
from tile_lifetime.event_dataflow import EventDataflowProgram, ImperativeEventOpKind, verify_event_dataflow_program
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.row_normalization_jax import (
    RowProjectionConfig,
    layer_projection,
    projection_training_step,
    projection_value_and_grad,
    rms_projection,
)
from tile_lifetime.row_normalization_reference_pipeline import (
    compile_row_statistic_reference_pipeline,
)
from tile_lifetime.row_normalization_training import (
    build_one_fold_contract_fixture,
    build_two_fold_contract_fixture,
)
from tile_lifetime.sm90_fold_map_contract import (
    attach_sm90_fold_map_contract,
    discover_sm90_fold_map_contract_bindings,
    recover_sm90_fold_map_contract_binding,
)
from tile_lifetime.tensor_program import (
    FAST_FP32_SUM_ERROR_CONTRACT,
    FoldPrimitive,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    execute_tensor_program,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
    tensor_program_value_ordinals,
)


def test_one_aux_fold_dependent_map_has_an_executable_wgmma_fragment_load() -> None:
    row = TensorAxis(100, 128)
    reduction = TensorAxis(101, 128)
    feature = TensorAxis(102, 64)
    activation = ProgramValue("activation", (row, reduction), DType.BF16)
    fold_state = ProgramValue("fold_state", (row,), DType.FP32)
    weight = ProgramValue("weight", (reduction, feature), DType.BF16)
    output = ProgramValue("output", (row, feature), DType.BF16)
    inverse = scalar_unary(
        ScalarExpressionKind.RSQRT,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(fold_state.name),
                scalar_constant(float(reduction.extent)),
            ),
            scalar_constant(1e-5),
        ),
    )
    generated = generate_sm90_map_load_contract(
        expression=scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(activation.name), inverse),
        activation=activation,
        leaves=(activation, fold_state),
        row_axis=row,
        reduction_axis=reduction,
        weight_feature_axis=feature,
        output_row_axis=row,
        output_feature_axis=feature,
        weight=weight,
        output=output,
    )

    assert generated.executable_with_current_quack_adapter
    assert "args={'operand_0': 'mvec_fp32'}" in generated.source
    assert "cute.math.rsqrt" in generated.source


def test_identity_map_load_contract_needs_no_auxiliary_operand() -> None:
    row = TensorAxis(3, 64, "row")
    reduction = TensorAxis(5, 64, "reduction")
    feature = TensorAxis(7, 128, "feature")
    activation = ProgramValue("activation", (row, reduction), DType.BF16)
    weight = ProgramValue("weight", (reduction, feature), DType.BF16)
    output = ProgramValue("output", (row, feature), DType.BF16)

    generated = generate_sm90_map_load_contract(
        expression=scalar_input(activation.name),
        activation=activation,
        leaves=(activation,),
        row_axis=row,
        reduction_axis=reduction,
        weight_feature_axis=feature,
        output_row_axis=row,
        output_feature_axis=feature,
        weight=weight,
        output=output,
    )

    assert generated.operands == ()
    assert "def generated_transform(activation):" in generated.source
    assert "return activation" in generated.source


@pytest.mark.parametrize("rows,reduction_extent,supported", [(32, 64, False), (128, 32, False), (128, 64, True)])
def test_auxiliary_contract_admission_respects_provider_strip_geometry(rows, reduction_extent, supported) -> None:
    row, reduction, feature = TensorAxis(0, rows), TensorAxis(1, reduction_extent), TensorAxis(2, 256)
    activation = ProgramValue("activation", (row, reduction), DType.BF16)
    scale = ProgramValue("scale", (reduction,), DType.BF16)
    weight = ProgramValue("weight", (reduction, feature), DType.BF16)
    output = ProgramValue("output", (row, feature), DType.BF16)
    generated = generate_sm90_map_load_contract(
        expression=scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(activation.name), scalar_input(scale.name)),
        activation=activation,
        leaves=(activation, scale),
        row_axis=row,
        reduction_axis=reduction,
        weight_feature_axis=feature,
        output_row_axis=row,
        output_feature_axis=feature,
        weight=weight,
        output=output,
    )
    assert generated.executable_with_current_quack_adapter is supported


@pytest.mark.parametrize("auxiliary", [False, True])
def test_prepared_contract_plan_receives_the_transform_operand_bundle(auxiliary) -> None:
    activation = np.ones((128, 64), dtype=np.float32)
    scale = np.arange(64, dtype=np.float32)
    weight = np.ones((64, 256), dtype=np.float32)
    output = np.empty((128, 256), dtype=np.float32)

    class Transform:
        def bundle(self, tensor, operands, tile_m, tile_k):
            assert (tile_m, tile_k) == (128, 64)
            return tensor, tuple(operands.values())

    class PreparedPlan:
        def run(self, operand, rhs, *, out):
            # This fake models the external QuACK boundary: transform keywords
            # are not accepted, and an auxiliary signature requires bundled A.
            if auxiliary:
                tensor, (aux,) = operand
                np.copyto(out["D"], (tensor * aux) @ rhs)
            else:
                np.copyto(out["D"], operand @ rhs)

    run_sm90_map_load_contract(
        PreparedPlan(),
        Transform(),
        Sm90MapLoadContractSchedule(),
        activation,
        weight,
        output,
        {"scale": scale} if auxiliary else {},
    )
    np.testing.assert_array_equal(output, np.full((128, 256), scale.sum() if auxiliary else 64, dtype=np.float32))


def test_independently_typed_row_and_reduction_leaves_share_one_wgmma_schedule() -> None:
    row = TensorAxis(110, 128)
    reduction = TensorAxis(111, 64)
    feature = TensorAxis(112, 128)
    activation = ProgramValue("activation", (row, reduction), DType.BF16)
    fold_state = ProgramValue("fold_state", (row,), DType.FP32)
    feature_scale = ProgramValue("feature_scale", (reduction,), DType.BF16)
    weight = ProgramValue("weight", (reduction, feature), DType.BF16)
    output = ProgramValue("output", (row, feature), DType.BF16)
    expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(activation.name), scalar_input(fold_state.name)),
        scalar_input(feature_scale.name),
    )

    generated = generate_sm90_map_load_contract(
        expression=expression,
        activation=activation,
        leaves=(activation, fold_state, feature_scale),
        row_axis=row,
        reduction_axis=reduction,
        weight_feature_axis=feature,
        output_row_axis=row,
        output_feature_axis=feature,
        weight=weight,
        output=output,
    )

    assert generated.executable_with_current_quack_adapter
    assert "args={'operand_0': 'mvec_fp32', 'operand_1': 'kvec_bf16'}" in generated.source
    assert tuple(operand.value for operand in generated.operands) == (fold_state, feature_scale)
    assert (
        generated.schedule.fingerprint
        == generate_sm90_map_load_contract(
            expression=scalar_binary(ScalarExpressionKind.ADD, expression, scalar_constant(1.0)),
            activation=activation,
            leaves=(activation, fold_state, feature_scale),
            row_axis=row,
            reduction_axis=reduction,
            weight_feature_axis=feature,
            output_row_axis=row,
            output_feature_axis=feature,
            weight=weight,
            output=output,
        ).schedule.fingerprint
    )

    with pytest.raises(ValueError, match="leaves must be unique"):
        generate_sm90_map_load_contract(
            expression=expression,
            activation=activation,
            leaves=(activation, fold_state, feature_scale, feature_scale),
            row_axis=row,
            reduction_axis=reduction,
            weight_feature_axis=feature,
            output_row_axis=row,
            output_feature_axis=feature,
            weight=weight,
            output=output,
        )


def _reference(
    x: np.ndarray,
    feature_scale: np.ndarray,
    weight: np.ndarray,
    output_cotangent: np.ndarray,
    *,
    centered: bool,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = x - np.mean(x, axis=-1, keepdims=True) if centered else x
    inverse = np.reciprocal(np.sqrt(np.mean(local * local, axis=-1, keepdims=True) + epsilon))
    standardized = local * inverse
    scaled = standardized * feature_scale
    output = scaled @ weight
    projected = output_cotangent @ weight.T
    local_cotangent = projected * feature_scale
    correlation = np.mean(local_cotangent * standardized, axis=-1, keepdims=True)
    input_cotangent = inverse * (local_cotangent - standardized * correlation)
    if centered:
        input_cotangent -= inverse * np.mean(local_cotangent, axis=-1, keepdims=True)
    feature_scale_cotangent = np.sum(projected * standardized, axis=0)
    weight_cotangent = scaled.T @ output_cotangent
    return output, input_cotangent, feature_scale_cotangent, weight_cotangent


@pytest.mark.parametrize(("factory", "centered"), ((rms_projection, False), (layer_projection, True)))
def test_jax_row_projection_forward_and_pullback_match_independent_reference(factory, centered: bool) -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=3e-4)
    rng = np.random.default_rng(905)
    x = rng.normal(size=(config.rows, config.hidden)).astype(np.float32)
    feature_scale = rng.normal(size=(config.hidden,)).astype(np.float32)
    weight = rng.normal(size=(config.hidden, config.features)).astype(np.float32)
    output_cotangent = rng.normal(size=(config.rows, config.features)).astype(np.float32)
    expected = _reference(
        x,
        feature_scale,
        weight,
        output_cotangent,
        centered=centered,
        epsilon=config.epsilon,
    )

    output, cotangents = projection_training_step(factory(config))(
        jnp.asarray(x),
        jnp.asarray(feature_scale),
        jnp.asarray(weight),
        jnp.asarray(output_cotangent),
    )

    np.testing.assert_allclose(output, expected[0], rtol=3e-6, atol=3e-6)
    for actual, reference in zip(cotangents, expected[1:], strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("factory", (rms_projection, layer_projection))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_jax_value_and_grad_and_composed_pullback_have_the_same_cotangent(factory, dtype) -> None:
    config = RowProjectionConfig(rows=2, hidden=5, features=3, epsilon=2e-4)
    rng = np.random.default_rng(917)
    inputs = (
        jnp.asarray(rng.normal(size=(config.rows, config.hidden)).astype(np.float32), dtype=dtype),
        jnp.asarray(rng.normal(size=(config.hidden,)).astype(np.float32), dtype=dtype),
        jnp.asarray(rng.normal(size=(config.hidden, config.features)).astype(np.float32), dtype=dtype),
    )
    output_cotangent = jnp.asarray(rng.normal(size=(config.rows, config.features)).astype(np.float32))
    project = factory(config)

    objective, objective_cotangents = projection_value_and_grad(project)(*inputs, output_cotangent)
    output, composed_cotangents = projection_training_step(project)(*inputs, output_cotangent)

    expected_objective = jnp.sum(output * output_cotangent.astype(output.dtype))
    np.testing.assert_allclose(objective, expected_objective, rtol=2e-6, atol=2e-6)
    for objective_cotangent, composed_cotangent in zip(objective_cotangents, composed_cotangents, strict=True):
        np.testing.assert_allclose(objective_cotangent, composed_cotangent, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("factory", "builder"),
    (
        (rms_projection, build_one_fold_contract_fixture),
        (layer_projection, build_two_fold_contract_fixture),
    ),
)
def test_generic_algebra_matches_the_ordinary_jax_boundary(factory, builder) -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=7e-5)
    source = builder(
        rows=config.rows,
        hidden=config.hidden,
        features=config.features,
        epsilon=config.epsilon,
    )
    rng = np.random.default_rng(929)
    inputs = {
        "input": rng.normal(size=(config.rows, config.hidden)).astype(np.float32),
        "feature_scale": rng.normal(size=(config.hidden,)).astype(np.float32),
        "weight": rng.normal(size=(config.hidden, config.features)).astype(np.float32),
    }

    expected = factory(config)(inputs["input"], inputs["feature_scale"], inputs["weight"])
    actual = execute_tensor_program(source, inputs)["output"]

    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_imported_bf16_centered_projection_preserves_map_rounding() -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=7e-5)
    rng = np.random.default_rng(931)
    arguments = tuple(
        jnp.asarray(value, dtype=jnp.bfloat16)
        for value in (
            rng.normal(size=(config.rows, config.hidden)),
            rng.normal(size=(config.hidden,)),
            rng.normal(size=(config.hidden, config.features)) / np.sqrt(config.hidden),
        )
    )
    project = layer_projection(config)

    source = jax_function_to_tensor_program(
        project,
        *arguments,
        numerical_policy=NumericalPolicy.FAST,
    )
    actual = execute_tensor_program(
        source,
        {value.name: np.asarray(argument) for value, argument in zip(source.inputs, arguments, strict=True)},
    )[source.outputs[0].name]

    np.testing.assert_array_equal(actual, project(*arguments))
    assert source.outputs[0].dtype is DType.BF16
    assert any(
        operation.expression.kind is ScalarExpressionKind.CONVERT
        and operation.expression.operands[0].kind is ScalarExpressionKind.CONVERT
        for operation in source.operations
        if isinstance(operation, MapPrimitive)
    )


@pytest.mark.parametrize("builder", (build_one_fold_contract_fixture, build_two_fold_contract_fixture))
def test_generic_fold_map_contract_pipeline_has_exact_events_and_bounded_generations(builder) -> None:
    source = builder(
        rows=5,
        hidden=7,
        features=9,
        dtype=DType.FP32,
    )
    source = replace(source, numerical_policy=NumericalPolicy.FAST)

    compiled = compile_row_statistic_reference_pipeline(source, row_tile=2, feature_tile=4)

    request = compiled.forward_request
    covered_operations = tuple(ordinal for task in request.task_plan.tasks for ordinal in task.operation_ordinals)
    assert covered_operations == tuple(range(len(source.operations)))
    assert request.task_plan.tasks[0].structure.principal_kind.value == "fold"
    assert request.task_plan.tasks[1].structure.principal_kind.value == "contract"
    assert all(token not in request.canonical_json.lower() for token in ("rms", "layernorm", "normalization"))
    assert request.task_plan.bounded_buffers[0].capacity == 2
    assert max(request.task_plan.bounded_buffers[0].generations) > 0
    assert any(allocation.generation_count > 1 for allocation in request.event_allocations)
    verify_event_dataflow_program(
        EventDataflowProgram(request.task_plan.task_families, request.task_plan.dependences, request.event_plans)
    )
    imperative_kinds = {operation.kind for lowering in compiled.forward_imperative_events for operation in lowering}
    assert ImperativeEventOpKind.WAIT in imperative_kinds
    assert ImperativeEventOpKind.NOTIFY in imperative_kinds

    producer_live_outs = {
        access.value_id for access in request.task_plan.tasks[0].accesses if access.kind.value == "live_out"
    }
    if builder is build_one_fold_contract_fixture:
        # Input ordinal zero is deliberately carried through with the raw Fold
        # state: kernel 0 exposes x and s rather than only a normalized tensor.
        value_ordinals = tensor_program_value_ordinals(source)
        fold = next(operation for operation in source.operations if isinstance(operation, FoldPrimitive))
        assert producer_live_outs == {value_ordinals[source.inputs[0]], value_ordinals[fold.output]}


def test_scalar_map_mutation_changes_algebra_and_request_but_not_task_or_schedule_family() -> None:
    source = build_one_fold_contract_fixture(
        rows=4,
        hidden=8,
        features=6,
        dtype=DType.BF16,
    )
    source = replace(source, numerical_policy=NumericalPolicy.FAST)
    finalizer = source.operations[-2]
    assert isinstance(finalizer, MapPrimitive)
    standardized, feature_scale = finalizer.inputs
    changed_finalizer = replace(
        finalizer,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_unary(ScalarExpressionKind.TANH, scalar_input(standardized.name)),
            scalar_input(feature_scale.name),
        ),
    )
    changed = replace(source, operations=(*source.operations[:-2], changed_finalizer, source.operations[-1]))

    baseline = compile_row_statistic_reference_pipeline(source)
    mutated = compile_row_statistic_reference_pipeline(changed)

    assert baseline.forward_request.algebra_fingerprint != mutated.forward_request.algebra_fingerprint
    assert baseline.forward_request.fingerprint != mutated.forward_request.fingerprint
    assert baseline.forward_request.task_plan.fingerprint == mutated.forward_request.task_plan.fingerprint
    assert (
        baseline.forward_request.schedules[0].family.fingerprint
        == mutated.forward_request.schedules[0].family.fingerprint
    )
    assert baseline.forward_request.schedules[0].fingerprint == mutated.forward_request.schedules[0].fingerprint
    baseline_generated = attach_sm90_fold_map_contract(source, baseline.forward_request)
    mutated_generated = attach_sm90_fold_map_contract(changed, mutated.forward_request)
    assert baseline_generated.source_sha256 != mutated_generated.source_sha256
    assert baseline_generated.consumer is not None
    assert mutated_generated.consumer is not None
    assert "cute.math.tanh" not in baseline_generated.consumer.source
    assert "cute.math.tanh" in mutated_generated.consumer.source
    assert baseline_generated.consumer.schedule.fingerprint == mutated_generated.consumer.schedule.fingerprint


def test_dataflow_cut_ignores_an_independent_map_after_the_output_contract() -> None:
    source = build_one_fold_contract_fixture(
        rows=4,
        hidden=8,
        features=6,
    )
    source = replace(source, numerical_policy=NumericalPolicy.FAST)
    x = source.inputs[0]
    unused = ProgramValue("independent", x.axes, x.dtype)
    independent_map = MapPrimitive("independent map", (x,), unused, scalar_input(x.name))
    reordered = replace(source, operations=(*source.operations, independent_map))

    baseline = compile_row_statistic_reference_pipeline(source)
    with_independent_map = compile_row_statistic_reference_pipeline(reordered)

    assert baseline.forward_request.algebra_fingerprint != with_independent_map.forward_request.algebra_fingerprint
    assert baseline.forward_request.task_plan.fingerprint == with_independent_map.forward_request.task_plan.fingerprint
    assert (
        baseline.forward_request.schedules[0].family.fingerprint
        == with_independent_map.forward_request.schedules[0].family.fingerprint
    )


def test_two_stage_physical_request_does_not_claim_source_ordered_equivalence() -> None:
    source = build_one_fold_contract_fixture(
        rows=4,
        hidden=8,
        features=6,
    )

    with pytest.raises(ValueError, match="FAST numerical policy"):
        compile_row_statistic_reference_pipeline(source)


def test_fast_fp32_fold_contract_bounds_reassociated_realized_terms() -> None:
    terms = np.asarray([1.0, 1e-4, -1.0, 3e-4, 2e-4], dtype=np.float32)
    exact = float(np.sum(terms.astype(np.float64)))
    left = np.float32(0.0)
    for term in terms:
        left = np.float32(left + term)
    tree = np.float32(np.float32(terms[0] + terms[1]) + np.float32(terms[2] + np.float32(terms[3] + terms[4])))
    bound = FAST_FP32_SUM_ERROR_CONTRACT.absolute_error_bound(
        term_count=len(terms),
        absolute_sum=float(np.sum(np.abs(terms.astype(np.float64)))),
    )

    assert abs(float(left) - exact) <= bound
    assert abs(float(tree) - exact) <= bound


def test_fast_fold_error_contract_is_part_of_the_physical_request() -> None:
    source = replace(
        build_two_fold_contract_fixture(
            rows=4,
            hidden=8,
            features=6,
            dtype=DType.BF16,
        ),
        numerical_policy=NumericalPolicy.FAST,
    )

    request = compile_row_statistic_reference_pipeline(source).forward_request

    assert request.schedules[0].parameters["fold_error_contract"] == (FAST_FP32_SUM_ERROR_CONTRACT.canonical_record)


def test_common_request_attaches_to_two_generic_sm90_kernels() -> None:
    source = build_one_fold_contract_fixture(
        rows=4,
        hidden=8,
        features=6,
        dtype=DType.BF16,
    )
    source = replace(source, numerical_policy=NumericalPolicy.FAST)
    pipeline = compile_row_statistic_reference_pipeline(source, row_tile=2, feature_tile=4)

    generated = attach_sm90_fold_map_contract(source, pipeline.forward_request)

    assert generated.request_fingerprint == generated.physical_request.fingerprint
    assert generated.algebra_fingerprint == source.semantic_fingerprint
    assert generated.kernel_count == 2
    assert generated.performance_class == "sm90_wgmma_map_load_contract_v1"
    assert (generated.consumer.schedule.tile_m, generated.consumer.schedule.tile_n) == (128, 128)
    assert generated.intermediate_names == ("input", "sum_square", "inverse_scale")
    assert generated.fold_state_names == ("sum_square",)
    assert generated.row_materialization_names == ("inverse_scale",)
    assert generated.intermediate_storage == (
        "input_alias",
        "materialized_fp32",
        "materialized_fp32_row_map",
    )
    assert generated.source.count("__global__") == 1
    assert "shuttle_fold_producer" in generated.source
    assert generated.consumer is not None
    assert "@a_transform" in generated.consumer.source
    assert "@gemm_epilogue" in generated.consumer.source
    assert not generated.consumer.executable_with_current_quack_adapter
    assert "'operand_0': 'kvec_bf16'" in generated.consumer.source
    assert "'operand_1': 'mvec_fp32'" in generated.consumer.source
    assert "nvcuda::wmma" not in generated.source
    assert "fold_state_0[row]" in generated.source
    assert "row_map_0[row]" in generated.source
    assert "input[row * kReduction + reduction]" in generated.source
    assert all(token not in generated.source.lower() for token in ("rms", "layernorm", "attention", "softcap"))
    request = generated.physical_request
    assert len(request.event_realizations.audit.erased) == len(request.event_plans)
    assert len(request.event_realizations.erasure_witnesses) == len(request.event_plans)
    assert len(request.schedules[0].enforced_orders) == len(request.event_plans)
    assert request.schedules[0].parameters["fold_state_storage"] == "full_materialization"
    assert request.schedules[0].parameters["contract_implementation"] == "cute_wgmma_map_load_contract"
    assert request.schedules[0].parameters["contract_tile_m"] == generated.consumer.schedule.tile_m
    assert request.schedules[0].parameters["contract_tile_n"] == generated.consumer.schedule.tile_n
    assert request.schedules[0].parameters["contract_tile_k"] == generated.consumer.schedule.tile_k
    assert request.schedules[0].parameters["contract_pingpong"] == generated.consumer.schedule.pingpong
    assert tuple(result["storage"] for result in request.schedules[0].parameters["producer_results"]) == (
        "input_alias",
        "materialized_fp32",
        "materialized_fp32_row_map",
    )
    assert request.schedules[0].parameters["row_map_materializations"]
    assert len(request.task_plan.dependences) == 1
    assert request.task_plan.bounded_buffers == ()
    fragment = generated.consumer.canonical_fragment(
        component_id=0,
        schedule_fingerprint=request.schedules[0].fingerprint,
        value_ids=tensor_program_value_ordinals(source),
        kernel_symbol="compiled_wgmma_symbol",
        grid=(1, 2, 1),
        block=(384, 1, 1),
        dynamic_shared_memory_bytes=197_632,
        source_tasks=(recover_sm90_fold_map_contract_binding(source, request).task_ids[1],),
    )
    assert fragment.schedule_fingerprint == request.schedules[0].fingerprint
    assert fragment.launches[0].source_tasks == (1,)
    assert fragment.launches[0].kernel_symbol == "compiled_wgmma_symbol"
    assert fragment.launches[0].dynamic_shared_memory_bytes == 197_632
    assert set(fragment.slots_by_value_id) >= {
        tensor_program_value_ordinals(source)[value] for value in (*source.inputs, source.outputs[0])
    }


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_ordinary_jax_rms_projection_reaches_raw_input_fold_state_binding(dtype) -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=1e-5)
    arguments = (
        jnp.ones((config.rows, config.hidden), dtype=dtype),
        jnp.ones((config.hidden,), dtype=dtype),
        jnp.ones((config.hidden, config.features), dtype=dtype),
    )
    source = jax_function_to_tensor_program(
        rms_projection(config),
        *arguments,
        numerical_policy=NumericalPolicy.FAST,
    )

    pipeline = compile_row_statistic_reference_pipeline(source, row_tile=2, feature_tile=4)
    binding = recover_sm90_fold_map_contract_binding(source, pipeline.forward_request)
    if dtype is jnp.float32:
        with pytest.raises(ValueError, match="requires BF16 activation and weight operands"):
            attach_sm90_fold_map_contract(source, pipeline.forward_request)
        return
    generated = attach_sm90_fold_map_contract(source, pipeline.forward_request)

    assert binding.raw_input == source.inputs[0]
    assert binding.raw_input.dtype is (DType.BF16 if dtype is jnp.bfloat16 else DType.FP32)
    assert generated.intermediate_names == tuple(value.name for value in binding.boundary_values)
    assert generated.intermediate_storage == (
        "input_alias",
        "materialized_fp32",
        "materialized_fp32_row_map",
    )
    assert generated.kernel_count == 2
    assert generated.performance_class == "sm90_wgmma_map_load_contract_v1"
    assert any(
        operation.value == "relation"
        for task in pipeline.forward_request.task_plan.tasks
        for operation in task.operation_kinds
    )
    if dtype is jnp.bfloat16:
        assert "static_cast<float>" in generated.source
        consumer_accesses = pipeline.forward_request.task_plan.tasks[1].accesses
        assert any(access.kind.value == "recomputed" for access in consumer_accesses)


def test_two_fold_centered_statistic_uses_the_same_generic_sm90_region() -> None:
    source = build_two_fold_contract_fixture(
        rows=4,
        hidden=8,
        features=6,
        dtype=DType.BF16,
    )
    source = replace(source, numerical_policy=NumericalPolicy.FAST)
    pipeline = compile_row_statistic_reference_pipeline(source)

    binding = recover_sm90_fold_map_contract_binding(source, pipeline.forward_request)
    generated = attach_sm90_fold_map_contract(source, pipeline.forward_request)

    assert len(binding.folds) == 2
    assert (generated.consumer.schedule.tile_m, generated.consumer.schedule.tile_n) == (64, 256)
    assert generated.consumer.schedule.pingpong is False
    assert generated.consumer.schedule.threads_per_cta == 256
    assert binding.boundary_values == (
        source.inputs[0],
        *(fold.output for fold in binding.folds),
        *binding.row_materializations,
    )
    assert tuple(value.name for value in binding.row_materializations) == ("row_mean", "inverse_scale")
    assert generated.intermediate_storage == (
        "input_alias",
        "materialized_fp32",
        "materialized_fp32",
        "materialized_fp32_row_map",
        "materialized_fp32_row_map",
    )
    assert generated.performance_class == "sm90_wgmma_map_load_contract_v1"
    assert generated.physical_request.schedules[0].parameters["contract_threads"] == 256
    assert "float local_0" in generated.source
    assert "float local_1" in generated.source
    assert tuple(
        (operand.value.name, operand.value.dtype, operand.delivery) for operand in generated.consumer.operands
    ) == (
        ("feature_scale", DType.BF16, "kvec_bf16"),
        ("inverse_scale", DType.FP32, "mvec_fp32"),
        ("row_mean", DType.FP32, "mvec_fp32"),
    )
    assert "args={'operand_0': 'kvec_bf16', 'operand_1': 'mvec_fp32', 'operand_2': 'mvec_fp32'}" in (
        generated.consumer.source
    )
    assert "activation - operand_2" in generated.consumer.source
    assert "* operand_1" in generated.consumer.source
    assert "rsqrt" not in generated.consumer.source
    assert all(token not in generated.source.lower() for token in ("rms", "layernorm", "normalization"))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_ordinary_jax_layer_projection_reaches_raw_input_two_fold_state_binding(dtype) -> None:
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=1e-5)
    arguments = (
        jnp.ones((config.rows, config.hidden), dtype=dtype),
        jnp.ones((config.hidden,), dtype=dtype),
        jnp.ones((config.hidden, config.features), dtype=dtype),
    )
    source = jax_function_to_tensor_program(
        layer_projection(config),
        *arguments,
        numerical_policy=NumericalPolicy.FAST,
    )

    pipeline = compile_row_statistic_reference_pipeline(source, row_tile=2, feature_tile=4)
    binding = recover_sm90_fold_map_contract_binding(source, pipeline.forward_request)

    assert binding.raw_input == source.inputs[0]
    assert len(binding.folds) == 2
    assert binding.boundary_values == (
        binding.raw_input,
        *(fold.output for fold in binding.folds),
        *binding.row_materializations,
    )
    consumer_accesses = pipeline.forward_request.task_plan.tasks[1].accesses
    assert any(access.kind.value == "recomputed" for access in consumer_accesses)
    if dtype is jnp.float32:
        with pytest.raises(ValueError, match="requires BF16 activation and weight operands"):
            attach_sm90_fold_map_contract(source, pipeline.forward_request)
        return
    generated = attach_sm90_fold_map_contract(source, pipeline.forward_request)
    assert generated.intermediate_names == tuple(value.name for value in binding.boundary_values)
    assert generated.intermediate_storage == (
        "input_alias",
        "materialized_fp32",
        "materialized_fp32",
        "materialized_fp32_row_map",
        "materialized_fp32_row_map",
    )
    assert len(generated.consumer.operands) == 3


def test_sm90_region_discovery_is_independent_of_unrelated_tasks() -> None:
    source = replace(
        build_one_fold_contract_fixture(
            rows=4,
            hidden=8,
            features=6,
            dtype=DType.BF16,
        ),
        numerical_policy=NumericalPolicy.FAST,
    )
    request = compile_row_statistic_reference_pipeline(source).forward_request
    producer, consumer = request.task_plan.tasks
    residual = replace(consumer, id=2, family=replace(consumer.family, name="task_2"))
    residual_family = replace(
        request.schedules[0].family,
        name="residual_contract",
        tasks=(request.schedules[0].family.tasks[1],),
    )
    residual_schedule = replace(
        request.schedules[0],
        family=residual_family,
        task_ids=(2,),
        enforced_orders=(),
    )
    request_with_residual = replace(
        request,
        task_plan=replace(request.task_plan, tasks=(producer, consumer, residual)),
        schedules=(request.schedules[0], residual_schedule),
    )

    bindings = discover_sm90_fold_map_contract_bindings(source, request_with_residual)

    assert tuple(binding.task_ids for binding in bindings) == ((0, 1),)
    with pytest.raises(ValueError, match="does not yet execute residual tasks"):
        attach_sm90_fold_map_contract(source, request_with_residual)
