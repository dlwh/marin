# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComposition,
    CuteKernelComponentFragment,
    coalesce_private_map_fold_components,
    compose_cute_pipeline_launch_recipe,
    dependency_convex_component_fits_resource_envelope,
    dependency_convex_component_resource_record,
    dependency_convex_contract_phase_schedule,
    dependency_convex_reduction_phase_schedule,
    install_composed_cute_artifact,
    lower_canonical_kernel_composition,
    supports_same_coordinate_producer_reuse,
)
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    combine_physical_event_schedules,
)
from tile_lifetime.dense_algebra import dense_composition_tensor_program
from tile_lifetime.dense_flow_planner import (
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.dense_region import (
    DenseCompositionShape,
    DenseCompositionWeights,
    compile_dense_composition_reference,
    dense_composition_backward,
    dense_composition_forward,
    dense_composition_training_step,
    dense_composition_value_and_grad,
)
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.executable_normalized_exp_contract import prepare_sm90_normalized_exp_contract
from tile_lifetime.jax_normalized_exp_contract_pullback import trace_jax_owned_pullback
from tile_lifetime.jax_normalized_exp_contract_training import (
    JaxOwnedNormalizedExpValueAndGradTrace,
    prepare_sm90_split_normalized_exp_value_and_grad,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import (
    PhysicalNumericalPolicy,
    PhysicalTarget,
    TaskCapability,
    physical_schedule_is_eligible,
)
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    derive_streaming_physical_event_schedule,
    streaming_physical_event_schedule_record,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    execute_tensor_program,
    index_axis,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
    tensor_program_value_ordinals,
)


def _inputs() -> tuple[jax.Array, DenseCompositionWeights, DenseCompositionShape]:
    keys = jax.random.split(jax.random.key(17), 10)
    batch, sequence, model, heads, key_dimension, value_dimension, intermediate = (1, 4, 8, 2, 4, 3, 12)
    source = jax.random.normal(keys[0], (batch, sequence, model))
    weights = DenseCompositionWeights(
        input_scale=0.9 + 0.1 * jax.random.normal(keys[1], (model,)),
        query=0.1 * jax.random.normal(keys[2], (model, heads, key_dimension)),
        key=0.1 * jax.random.normal(keys[3], (model, heads, key_dimension)),
        value=0.1 * jax.random.normal(keys[4], (model, heads, value_dimension)),
        output=0.1 * jax.random.normal(keys[5], (heads, value_dimension, model)),
        mlp_scale=0.9 + 0.1 * jax.random.normal(keys[6], (model,)),
        gate=0.1 * jax.random.normal(keys[7], (model, intermediate)),
        up=0.1 * jax.random.normal(keys[8], (model, intermediate)),
        down=0.1 * jax.random.normal(keys[9], (intermediate, model)),
    )
    return source, weights, DenseCompositionShape(heads, key_dimension, value_dimension)


def _program_inputs(source: jax.Array, weights: DenseCompositionWeights) -> dict[str, np.ndarray]:
    sequence = source.shape[1]
    return {
        "source": np.asarray(source),
        "input_scale": np.asarray(weights.input_scale),
        "query_weight": np.asarray(weights.query),
        "key_weight": np.asarray(weights.key),
        "value_weight": np.asarray(weights.value),
        "output_weight": np.asarray(weights.output),
        "mlp_scale": np.asarray(weights.mlp_scale),
        "gate_weight": np.asarray(weights.gate),
        "up_weight": np.asarray(weights.up),
        "down_weight": np.asarray(weights.down),
        "query_position": np.arange(sequence, dtype=np.int32),
        "key_position": np.arange(sequence, dtype=np.int32),
    }


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_grouped_dense_composition_values_and_gradients_match_jax_attention(kv_heads: int) -> None:
    source, weights, shape = _inputs()
    shape = replace(shape, heads=4)
    weights = weights._replace(
        query=jnp.tile(weights.query, (1, 2, 1)),
        key=weights.key[:, :kv_heads],
        value=weights.value[:, :kv_heads],
        output=jnp.tile(weights.output, (2, 1, 1)),
    )

    def reference(x, parameters):
        normalized = x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + shape.epsilon)
        normalized = normalized * parameters.input_scale
        query = jnp.einsum("bsm,mhd->bshd", normalized, parameters.query)
        key = jnp.einsum("bsm,mhd->bshd", normalized, parameters.key)
        value = jnp.einsum("bsm,mhv->bshv", normalized, parameters.value)
        # JAX's independent GQA API requires matching Q/K/V feature dimensions.
        value = jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, 1)))
        attended = jax.nn.dot_product_attention(query, key, value, is_causal=True)[..., : shape.value_dimension]
        residual = x + jnp.einsum("bshv,hvm->bsm", attended, parameters.output)
        mlp = residual / jnp.sqrt(jnp.mean(residual**2, axis=-1, keepdims=True) + shape.epsilon)
        mlp = mlp * parameters.mlp_scale
        gate = mlp @ parameters.gate
        hidden = jax.nn.silu(gate) * (mlp @ parameters.up)
        return residual + hidden @ parameters.down

    actual = dense_composition_training_step(source, weights, shape)

    def objective(x, parameters):
        output = reference(x, parameters)
        return jnp.mean(output**2), output

    (loss, output), gradients = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(source, weights)
    expected = (loss, output, gradients)
    for generated, oracle in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(generated, oracle, rtol=1e-6, atol=1e-6)


def _program(source: jax.Array, weights: DenseCompositionWeights, shape: DenseCompositionShape):
    return dense_composition_tensor_program(
        batch_size=source.shape[0],
        sequence_length=source.shape[1],
        model_dimension=source.shape[2],
        heads=shape.heads,
        query_key_dimension=shape.query_key_dimension,
        value_dimension=shape.value_dimension,
        feed_forward_dimension=weights.gate.shape[1],
        epsilon=shape.epsilon,
        causal=shape.causal,
    )


def _tiles() -> dict[int, int]:
    return {0: 1, 1: 2, 2: 2, 3: 4, 4: 1, 5: 2, 6: 2, 7: 4}


def test_dense_composition_algebra_matches_ordinary_jax() -> None:
    source, weights, shape = _inputs()
    expected = np.asarray(dense_composition_forward(source, weights, shape))
    program = _program(source, weights, shape)

    actual = execute_tensor_program(program, _program_inputs(source, weights))["output"]
    error = np.abs(expected - actual)

    assert float(np.max(error)) < 1e-6
    assert float(np.mean(error)) < 1e-7
    assert all(
        isinstance(operation, (ContractPrimitive, MapPrimitive, FoldPrimitive)) for operation in program.operations
    )
    score = next(operation for operation in program.operations if operation.name == "score.contract")
    weighted = next(operation for operation in program.operations if operation.name == "value.contract")
    assert isinstance(score, ContractPrimitive)
    assert isinstance(weighted, ContractPrimitive)
    assert score.reduction_axes[0].extent == shape.query_key_dimension
    assert weighted.output.axes[-1].extent == shape.value_dimension
    assert score.reduction_axes != weighted.reduction_axes


def test_dense_composition_backward_and_training_are_jax_owned() -> None:
    source, weights, shape = _inputs()
    output = dense_composition_forward(source, weights, shape)
    cotangent = 2.0 * output / output.size

    source_gradient, weight_gradients = dense_composition_backward(source, weights, cotangent, shape)
    value, value_gradients = dense_composition_value_and_grad(source, weights, shape)
    loss, composed_output, composed_gradients = dense_composition_training_step(source, weights, shape)

    np.testing.assert_allclose(composed_output, output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loss, jnp.mean(output * output), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(value, loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(value_gradients[0], source_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(composed_gradients[0], source_gradient, rtol=1e-6, atol=1e-6)
    for actual, value_gradient, expected in zip(
        composed_gradients[1], value_gradients[1], weight_gradients, strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_gradient, expected, rtol=1e-6, atol=1e-6)

    def scalar_objective(delta: float) -> jax.Array:
        perturbed = source.at[0, 1, 2].add(delta)
        result = dense_composition_forward(perturbed, weights, shape)
        return jnp.mean(result * result)

    step = 1e-3
    finite_difference = (scalar_objective(step) - scalar_objective(-step)) / (2 * step)
    np.testing.assert_allclose(source_gradient[0, 1, 2], finite_difference, rtol=3e-3, atol=2e-4)
    jaxpr = jax.make_jaxpr(lambda x, parameters: dense_composition_training_step(x, parameters, shape))(source, weights)
    primitives = {equation.primitive.name for equation in jaxpr.jaxpr.eqns}
    assert "custom_vjp_call_jaxpr" not in primitives
    assert "custom_call" not in primitives


def test_jax_owned_dense_reverse_lowers_through_the_generic_executable_path() -> None:
    source, weights, shape = _inputs()
    output = dense_composition_forward(source, weights, shape)
    cotangent = 2.0 * output / output.size
    cases = (
        (
            lambda x, parameters: dense_composition_training_step(x, parameters, shape),
            (source, weights),
        ),
        (
            lambda x, parameters, output_cotangent: dense_composition_backward(x, parameters, output_cotangent, shape),
            (source, weights, cotangent),
        ),
    )

    for function, arguments in cases:
        program = jax_function_to_tensor_program(function, *arguments, numerical_policy=NumericalPolicy.FAST)
        flat_arguments = jax.tree_util.tree_leaves(arguments)
        actual = execute_tensor_program(
            program,
            {f"argument.{index}": np.asarray(value) for index, value in enumerate(flat_arguments)},
        )
        expected = jax.tree_util.tree_leaves(function(*arguments))
        for result, expected_value in zip(program.outputs, expected, strict=True):
            np.testing.assert_allclose(actual[result.name], expected_value, rtol=1e-6, atol=1e-6)

        tile_extent_by_axis = {
            axis.id: min(axis.extent, 2)
            for value in (
                *program.inputs,
                *(result for operation in program.operations for result in primitive_outputs(operation)),
            )
            for axis in value.axes
        }
        task_plan = derive_composed_task_plan(
            program,
            tile_extent_by_axis=tile_extent_by_axis,
            numerical_policy=PhysicalNumericalPolicy.FAST,
        )
        event_program = eventize_composed_task_plan(task_plan)
        request = physical_request_for_composed_tasks(
            algebra_record=program.canonical_record,
            algebra_fingerprint=program.semantic_fingerprint,
            task_plan=task_plan,
            event_program=event_program,
            target=PhysicalTarget.SM90,
            numerical_policy=PhysicalNumericalPolicy.FAST,
        )

        assert request.task_plan == task_plan
        assert request.event_plans == event_program.event_plans
        assert len(request.schedules) == len(task_plan.tasks)

        imported_request = jax_tensor_program_executable_request(
            program,
            tile_size=2,
            target=PhysicalTarget.SM90,
        )
        assert imported_request.normalized_exp_contract_regions
        assert len(imported_request.event_realizations.audit.entries) == len(imported_request.event_plans)


def test_dense_training_installs_disjoint_saved_state_normalized_exp_components() -> None:
    keys = jax.random.split(jax.random.key(91), 11)
    batch, sequence, model, heads, feature, intermediate = (1, 4, 8, 2, 128, 12)

    def array(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return (0.05 * jax.random.normal(key, shape)).astype(jnp.bfloat16)

    source = array(keys[0], (batch, sequence, model))
    weights = DenseCompositionWeights(
        input_scale=array(keys[1], (model,)),
        query=array(keys[2], (model, heads, feature)),
        key=array(keys[3], (model, heads, feature)),
        value=array(keys[4], (model, heads, feature)),
        output=array(keys[5], (heads, feature, model)),
        mlp_scale=array(keys[6], (model,)),
        gate=array(keys[7], (model, intermediate)),
        up=array(keys[8], (model, intermediate)),
        down=array(keys[9], (intermediate, model)),
    )
    shape = DenseCompositionShape(heads, feature, feature)
    primals = (source, *weights)

    def forward(x: jax.Array, *parameter_leaves: jax.Array) -> jax.Array:
        return dense_composition_forward(x, DenseCompositionWeights(*parameter_leaves), shape)

    def value_and_grad(x: jax.Array, *parameter_leaves: jax.Array):
        loss, gradients = jax.value_and_grad(
            lambda primal, *parameters: jnp.mean(forward(primal, *parameters).astype(jnp.float32) ** 2),
            argnums=tuple(range(len(primals))),
        )(x, *parameter_leaves)
        return loss, *gradients

    forward_program = jax_function_to_tensor_program(
        forward,
        *primals,
        numerical_policy=NumericalPolicy.FAST,
    )
    source_program = jax_function_to_tensor_program(
        value_and_grad,
        *primals,
        numerical_policy=NumericalPolicy.FAST,
    )
    output_cotangent = array(keys[10], (batch, sequence, model))
    trace = JaxOwnedNormalizedExpValueAndGradTrace(
        forward_program,
        source_program,
        trace_jax_owned_pullback(
            forward,
            *primals,
            output_cotangent=output_cotangent,
            numerical_policy=NumericalPolicy.FAST,
        ),
    )
    source_request = jax_tensor_program_executable_request(
        source_program,
        tile_size=128,
        target=PhysicalTarget.SM90,
    )
    plan = prepare_sm90_split_normalized_exp_value_and_grad(source_request, trace, tile_size=128)
    forward_schedule, pullback_schedule = plan.directional_schedules

    assert not set(forward_schedule.task_ids) & set(pullback_schedule.task_ids)
    assert plan.saved_values
    assert plan.crossing_event_names
    assert len(plan.composition.components) == 45
    co_launched = tuple(
        component.task_ids
        for component in plan.composition.components
        if len(component.task_ids) > 1
        and plan.physical_request.schedules[component.schedule_index].parameters.get("selection")
        == "contiguous_topological_dependency_convex_map_relation_boundary"
    )
    assert (11, 12) in co_launched
    assert (80, 81, 82, 83, 84) in co_launched
    normalized_exp_component_ids = {
        component.id
        for component in plan.composition.components
        if plan.physical_request.schedules[component.schedule_index].parameters.get("direction")
        in {"forward", "pullback"}
    }
    generic_records = tuple(
        dependency_convex_component_resource_record(plan.program, component)
        for component in plan.composition.components
        if component.id not in normalized_exp_component_ids
    )

    def image_count(maximum_live_inputs: int, maximum_live_outputs: int) -> int:
        return sum(
            (
                1
                if dependency_convex_component_fits_resource_envelope(
                    record,
                    maximum_operations=8,
                    maximum_live_inputs=maximum_live_inputs,
                    maximum_live_outputs=maximum_live_outputs,
                    maximum_reduction_depth=1,
                )
                else record["operation_count"]
            )
            for record in generic_records
        )

    assert image_count(8, 3) == 61
    assert image_count(4, 2) == 72
    component_16 = plan.composition.components[16]
    component_16_record = dependency_convex_component_resource_record(plan.program, component_16)
    assert component_16_record["operation_count"] == 5
    assert dependency_convex_component_fits_resource_envelope(
        component_16_record,
        maximum_operations=8,
        maximum_live_inputs=4,
        maximum_live_outputs=2,
        maximum_reduction_depth=1,
    )
    assert any(
        isinstance(plan.program.operations[ordinal], (ContractPrimitive, FoldPrimitive))
        and plan.program.operations[ordinal].accumulation_dtype is DType.BF16
        for ordinal in component_16.operation_ordinals
    )
    assert image_count(4, 2) + component_16_record["operation_count"] - 1 == 76
    assert image_count(8, 3) + 2 == 63
    assert image_count(8, 3) + 2 + 4 + 1 == 68
    records_by_boundary = {
        (len(record["live_inputs"]), len(record["live_outputs"]), record["maximum_reduction_depth"])
        for record in generic_records
    }
    assert {(4, 3, 1), (6, 2, 1), (6, 3, 2), (8, 3, 2)} <= records_by_boundary
    reduction_phase_schedules = tuple(
        schedule
        for component in plan.composition.components
        if (
            schedule := dependency_convex_reduction_phase_schedule(
                plan.program,
                plan.composition,
                component,
            )
        )
        is not None
    )
    assert tuple(schedule.source_component_id for schedule in reduction_phase_schedules) == (8, 15, 22, 34)
    assert tuple(
        tuple(len(phase.component.operation_ordinals) for phase in schedule.phases)
        for schedule in reduction_phase_schedules
    ) == ((3, 1), (3, 1), (5, 1), (7, 1))
    component_34_producer = reduction_phase_schedules[-1].phases[0].resource_record
    assert (len(component_34_producer["live_inputs"]), len(component_34_producer["live_outputs"])) == (8, 3)
    assert all(
        schedule.event_capacity == schedule.generation_count == 1
        and len(schedule.cut_value_ids) == 1
        and len(schedule.crossing_phase_edges) == 1
        for schedule in reduction_phase_schedules
    )
    assert all(
        dependency_convex_component_fits_resource_envelope(
            schedule.phases[0].resource_record,
            maximum_operations=8,
            maximum_live_inputs=8,
            maximum_live_outputs=3,
            maximum_reduction_depth=1,
        )
        and schedule.phases[1].resource_record["operation_count"] == 1
        and schedule.phases[1].resource_record["maximum_reduction_depth"] == 1
        for schedule in reduction_phase_schedules
    )
    split_generic_images = image_count(8, 3) - sum(
        sum(phase.resource_record["operation_count"] for phase in schedule.phases) - len(schedule.phases)
        for schedule in reduction_phase_schedules
    )
    assert split_generic_images == 47
    assert split_generic_images + 2 == 49
    assert split_generic_images + 2 + 4 + 1 == 54
    assert any(
        realization.kind is EventRealizationKind.ERASED_PROGRAM_ORDER
        for realization in plan.physical_request.event_realizations.audit.entries
    )
    assert all(not lowering.retained_operations for lowering in plan.composition.legalized_event_lowerings)
    assert tuple((value.shape, value.dtype) for value in plan.program.outputs) == tuple(
        (value.shape, value.dtype) for value in source_program.outputs
    )
    plan.require_whole_request_runtime()


def test_local_composition_rules_form_generic_contract_fold_tasks() -> None:
    source, weights, shape = _inputs()
    program = _program(source, weights, shape)
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    task_by_operation = {
        program.operations[ordinal].name: task for task in task_plan.tasks for ordinal in task.operation_ordinals
    }

    normalized_names = (
        "score.contract",
        "score.map",
        "score.maximum",
        "score.center",
        "score.exponential",
        "score.sum",
        "value.contract",
        "value.normalize",
    )
    assert len({task_by_operation[name].id for name in normalized_names}) == 1
    normalized_task = task_by_operation["score.contract"]
    assert tuple(phase.capability for phase in normalized_task.structure.phases) == (
        TaskCapability.MATRIX_CONTRACT,
        TaskCapability.SCALAR_MAP,
        TaskCapability.ASSOCIATIVE_FOLD,
        TaskCapability.SCALAR_MAP,
        TaskCapability.SCALAR_MAP,
        TaskCapability.ASSOCIATIVE_FOLD,
        TaskCapability.MATRIX_CONTRACT,
        TaskCapability.SCALAR_MAP,
    )

    contract_fold_contract_names = (
        "output.contract",
        "residual.add",
        "mlp.square",
        "mlp.fold",
        "mlp.finalize_fold",
        "mlp.feature_scale",
        "mlp.apply_fold",
        "gate.contract",
    )
    assert len({task_by_operation[name].id for name in contract_fold_contract_names}) == 1
    assert task_by_operation["activation.map"] is task_by_operation["down.contract"]


def test_dense_composition_reuses_sm90_contract_fold_schedule_by_axis_roles() -> None:
    program = dense_composition_tensor_program(
        batch_size=1,
        sequence_length=128,
        model_dimension=128,
        heads=1,
        query_key_dimension=128,
        value_dimension=128,
        feed_forward_dimension=256,
        causal=False,
        numerical_policy=NumericalPolicy.FAST,
    )
    tile_extent_by_axis = {
        axis.id: min(axis.extent, 128)
        for value in (
            *program.inputs,
            *(result for operation in program.operations for result in primitive_outputs(operation)),
        )
        for axis in value.axes
    }
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extent_by_axis,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    event_program = eventize_composed_task_plan(task_plan)
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=event_program,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    attached = attach_sm90_contract_fold_schedule(request)

    assert attached.physical_schedule.parameters["direction"] == "forward"
    required_capabilities = {
        capability
        for constraint in attached.physical_schedule.family.tasks
        for capability in constraint.required_capabilities
    }
    assert {
        TaskCapability.MATRIX_CONTRACT,
        TaskCapability.ASSOCIATIVE_FOLD,
        TaskCapability.SCALAR_MAP,
    } <= required_capabilities
    assert attached.physical_schedule.parameters["contract"] == {
        "tile_m": 128,
        "tile_n": 128,
        "threads": 384,
        "wgmma_instruction": "sm90.wgmma.mma_async",
        "accumulator_dtype": "fp32",
        "producer_feature_extent": 128,
        "consumer_feature_extent": 128,
        "owner_extent": 128,
        "streamed_extent": 128,
    }
    assert tuple(task_id for schedule in attached.request.schedules for task_id in schedule.task_ids) == (
        *attached.fused_task_ids,
        *(task.id for task in task_plan.tasks if task.id not in attached.fused_task_ids),
    )
    assert attached.request.event_plans == request.event_plans
    assert attached.request.event_allocations == request.event_allocations
    assert not attached.request.event_realizations.erasure_witnesses


def test_ordinary_jax_dense_composition_reaches_reusable_sm90_schedule() -> None:
    rng = np.random.default_rng(93)
    batch, sequence, model = 1, 4, 8
    heads, query_key_dimension, value_dimension, feed_forward = 2, 128, 128, 12
    source = rng.normal(size=(batch, sequence, model)).astype(np.float32)

    def weight(shape: tuple[int, ...]) -> np.ndarray:
        return (0.05 * rng.normal(size=shape)).astype(np.float32)

    weights = DenseCompositionWeights(
        input_scale=weight((model,)),
        query=weight((model, heads, query_key_dimension)),
        key=weight((model, heads, query_key_dimension)),
        value=weight((model, heads, value_dimension)),
        output=weight((heads, value_dimension, model)),
        mlp_scale=weight((model,)),
        gate=weight((model, feed_forward)),
        up=weight((model, feed_forward)),
        down=weight((feed_forward, model)),
    )
    shape = DenseCompositionShape(heads, query_key_dimension, value_dimension)

    def function(x, parameters):
        return dense_composition_forward(x, parameters, shape)

    imported = jax_function_to_tensor_program(function, source, weights, numerical_policy=NumericalPolicy.FAST)
    actual = execute_tensor_program(
        imported,
        {
            f"argument.{index}": np.asarray(value)
            for index, value in enumerate(jax.tree_util.tree_leaves((source, weights)))
        },
    )
    expected = np.asarray(function(source, weights))
    np.testing.assert_allclose(actual[imported.outputs[0].name], expected, rtol=1e-6, atol=1e-6)

    request = jax_tensor_program_executable_request(imported, tile_size=128, target=PhysicalTarget.SM90)
    attached = attach_sm90_contract_fold_schedule(request)

    assert request.provenance["index_domain_canonicalization"]["status"] == "applied"
    assert any(
        proof.get("neutral_map_operation_ordinals") for proof in request.provenance["layout_canonicalization"]["proofs"]
    )
    assert attached.physical_schedule.parameters["direction"] == "forward"
    required_capabilities = {
        capability
        for constraint in attached.physical_schedule.family.tasks
        for capability in constraint.required_capabilities
    }
    assert {
        TaskCapability.MATRIX_CONTRACT,
        TaskCapability.ASSOCIATIVE_FOLD,
        TaskCapability.SCALAR_MAP,
    } <= required_capabilities
    assert attached.physical_schedule.parameters["contract"] == {
        "tile_m": 64,
        "tile_n": 128,
        "threads": 256,
        "wgmma_instruction": "sm90.wgmma.mma_async",
        "accumulator_dtype": "fp32",
        "producer_feature_extent": 128,
        "consumer_feature_extent": 128,
        "owner_extent": sequence,
        "streamed_extent": sequence,
    }
    assert attached.request.event_plans == request.event_plans
    assert attached.request.event_allocations == request.event_allocations
    assert tuple(task_id for schedule in attached.request.schedules for task_id in schedule.task_ids) == (
        *attached.fused_task_ids,
        *(task.id for task in request.task_plan.tasks if task.id not in attached.fused_task_ids),
    )

    composition = lower_canonical_kernel_composition(attached.request)
    assert tuple(task_id for component in composition.components for task_id in component.task_ids) != tuple(
        task_id for schedule in attached.request.schedules for task_id in schedule.task_ids
    )
    assert sorted(task_id for component in composition.components for task_id in component.task_ids) == list(
        range(len(request.task_plan.tasks))
    )
    assert composition.concrete_events
    assert all(event.producer_component < event.consumer_component for event in composition.concrete_events)
    assert not composition.scheduled_request.event_realizations.audit.physical
    assert all(
        realization.kind.value in {"erased_program_order", "erased_stream_order"}
        for realization in composition.scheduled_request.event_realizations.audit.entries
    )
    normalized_exp = prepare_sm90_normalized_exp_contract(attached.request)
    assert normalized_exp.program.region.dimensions.relation_group_axes[0].extent == 1
    assert normalized_exp.program.region.producer.inputs[0].shape == (batch, heads, 1, sequence, query_key_dimension)

    program_record = composition.scheduled_request.algebra_record["program"]
    extent_by_axis = {axis["id"]: axis["extent"] for axis in program_record["axes"]}
    definitions = [*program_record["inputs"]]
    for operation in program_record["operations"]:
        definitions.extend(operation["results"])
    definition_by_id = {int(value["id"][1:]): value for value in definitions}
    fragments = []
    for component in composition.components:
        live_inputs = set(component.live_input_value_ids)
        live_outputs = set(component.live_output_value_ids)
        value_ids = tuple(sorted(live_inputs | live_outputs))
        slots = {
            value_id: CutePipelineSlot(
                f"local_{value_id}",
                {
                    "bf16": CutePipelineElementType.BF16,
                    "fp32": CutePipelineElementType.F32,
                }[definition_by_id[value_id]["dtype"]],
                tuple(extent_by_axis[axis] for axis in definition_by_id[value_id]["axes"]),
                CutePipelineSlotKind.SCRATCH,
                16,
            )
            for value_id in value_ids
        }
        arguments = tuple(
            CutePipelineSlotArgument(
                f"argument_{value_id}",
                slots[value_id].name,
                (
                    CutePipelineSlotAccess.READ_WRITE
                    if value_id in live_inputs and value_id in live_outputs
                    else CutePipelineSlotAccess.READ if value_id in live_inputs else CutePipelineSlotAccess.WRITE
                ),
            )
            for value_id in value_ids
        )
        fragments.append(
            CuteKernelComponentFragment(
                component.id,
                component.schedule_fingerprint,
                slots,
                (
                    CuteKernelLaunch(
                        f"local_launch_{component.id}",
                        f"kernel_{component.id}",
                        (1, 1, 1),
                        (128, 1, 1),
                        0,
                        (),
                        arguments,
                    ),
                ),
            )
        )
    private_fragments = tuple(
        replace(
            fragment,
            private_slots=(
                CutePipelineSlot(
                    "state_a", CutePipelineElementType.F32, (1,), CutePipelineSlotKind.SCRATCH, 16, reuse_group=17
                ),
            ),
            launches=(
                replace(
                    fragment.launches[0],
                    arguments=(
                        *fragment.launches[0].arguments,
                        CutePipelineSlotArgument("private_state", "state_a", CutePipelineSlotAccess.WRITE),
                    ),
                ),
            ),
        )
        for fragment in fragments
    )
    private_recipe = compose_cute_pipeline_launch_recipe(composition, private_fragments)
    private_groups = tuple(slot.reuse_group for slot in private_recipe.slots if "_private_" in slot.name)
    assert len(fragments) > 1
    assert private_groups == tuple(range(len(fragments)))

    recipe = compose_cute_pipeline_launch_recipe(composition, tuple(fragments))
    assert tuple(task for launch in recipe.launches for task in launch.source_tasks) == tuple(
        task for component in composition.components for task in component.task_ids
    )
    assert len(recipe.event_realizations) == len(request.event_plans)
    assert recipe.physical_event_provenance is not None
    assert recipe.physical_event_provenance["schedule"]["regions"][0]["realizations"][0]["kind"] == "erased_stream_order"
    assert all(realization.cover_pieces for realization in recipe.event_realizations)
    assert all(
        recipe.launches[producer].source_tasks and recipe.launches[consumer].source_tasks
        for realization in recipe.event_realizations
        for producer, _, consumer, _ in (realization.covering_launch_order,)
    )
    device_module = CutePipelineArtifact(
        request=normalized_exp.compile_request,
        host_object=b"joint-host-object",
        device_image=b"joint-device-image",
        kernel_names=tuple(launch.kernel_symbol for launch in recipe.launches),
        host_entry="joint_pipeline",
        compiler_ir="joint module",
        toolchain={"test": "structural"},
        source_digests={"test": "0" * 64},
    )
    artifact = install_composed_cute_artifact(device_module, composition, tuple(fragments))
    assert artifact.launch_recipe == recipe
    assert artifact.request.schedule["physical_schedules"] == tuple(
        schedule.canonical_record for schedule in composition.scheduled_request.schedules
    )
    assert artifact.request.algebra_identity["algebra_fingerprint"] == request.algebra_fingerprint
    assert artifact.request.physical_template == "canonical_schedule_component_pipeline_v1"


def test_composed_recipe_preserves_scalar_loss_rank_from_jax() -> None:
    source = jax_function_to_tensor_program(jnp.sum, jnp.ones((8,), dtype=jnp.float32))
    request = jax_tensor_program_executable_request(source, tile_size=128, target=PhysicalTarget.SM90)
    composition = lower_canonical_kernel_composition(request)
    values = {ordinal: value for value, ordinal in tensor_program_value_ordinals(source).items()}
    fragments = []
    for component in composition.components:
        inputs = set(component.live_input_value_ids)
        outputs = set(component.live_output_value_ids)
        slots = {
            ordinal: CutePipelineSlot(
                f"local_{ordinal}",
                CutePipelineElementType.F32,
                values[ordinal].shape or (1,),
                CutePipelineSlotKind.SCRATCH,
                16,
            )
            for ordinal in sorted(inputs | outputs)
        }
        fragments.append(
            CuteKernelComponentFragment(
                component.id,
                component.schedule_fingerprint,
                slots,
                (
                    CuteKernelLaunch(
                        f"local_launch_{component.id}",
                        "reduce",
                        (1, 1, 1),
                        (128, 1, 1),
                        0,
                        (),
                        tuple(
                            CutePipelineSlotArgument(
                                f"argument_{ordinal}",
                                slot.name,
                                CutePipelineSlotAccess.READ if ordinal in inputs else CutePipelineSlotAccess.WRITE,
                            )
                            for ordinal, slot in slots.items()
                        ),
                    ),
                ),
            )
        )
    recipe = compose_cute_pipeline_launch_recipe(composition, tuple(fragments))
    (public_output,) = (slot for slot in recipe.slots if slot.output_index == 0)
    assert public_output.shape == jax.eval_shape(jnp.sum, jnp.ones((8,), dtype=jnp.float32)).shape
    assert public_output.required_bytes == 4


def test_exact_affine_dependencies_and_event_generations_reach_physical_request() -> None:
    source, weights, shape = _inputs()
    executable = compile_dense_composition_reference(
        source,
        weights,
        shape,
        tile_extent_by_axis=_tiles(),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    program = executable.program
    task_by_operation = {
        program.operations[ordinal].name: task
        for task in executable.task_plan.tasks
        for ordinal in task.operation_ordinals
    }
    normalization = task_by_operation["input.apply_fold"]
    key_projection = task_by_operation["project.key"]
    dependence = next(
        dependence
        for dependence in executable.task_plan.dependences
        if dependence.relation.source == normalization.family and dependence.relation.target == key_projection.family
    )
    source_query_position = next(
        index for index, axis in enumerate(normalization.structure.owner_axes) if axis.axis_id == 1
    )
    target_key_position = next(
        index for index, axis in enumerate(key_projection.structure.owner_axes) if axis.axis_id == 2
    )
    assert all(
        pair.source[source_query_position] == pair.target[target_key_position] for pair in dependence.relation.pairs
    )

    assert len(executable.event_program.event_plans) == len(executable.task_plan.dependences)
    assert all(
        plan.required_dependence.relation.pairs == plan.scheduled_dependence.pairs
        for plan in executable.event_program.event_plans
    )
    request = executable.physical_request
    assert executable.program.numerical_policy.value == "fast"
    assert all(
        all(generation == 0 for generation in allocation.storage.generations) for allocation in request.event_allocations
    )
    assert all(allocation.generation_count == 1 for allocation in request.event_allocations)
    assert all(
        allocation.capacity == len(plan.domain.coordinates)
        for plan, allocation in zip(request.event_plans, request.event_allocations, strict=True)
    )
    assert json.loads(request.canonical_json)["schema_version"] == 1
    assert request.algebra_fingerprint == program.semantic_fingerprint
    assert request.provenance == {"source_boundary": "ordinary_jax", "differentiation_owner": "jax"}
    assert len(request.schedules) == len(executable.task_plan.tasks)
    assert tuple(task_id for schedule in request.schedules for task_id in schedule.task_ids) == tuple(
        range(len(executable.task_plan.tasks))
    )


def test_numerical_policy_changes_plan_identity_without_changing_algebra() -> None:
    source, weights, shape = _inputs()
    program = _program(source, weights, shape)
    fast = derive_composed_task_plan(
        program,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    source_ordered = derive_composed_task_plan(
        program,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )

    assert program.semantic_fingerprint == _program(source, weights, shape).semantic_fingerprint
    assert fast.fingerprint != source_ordered.fingerprint
    assert {task.structure.traversal.value for task in fast.tasks} == {"unordered"}
    assert {task.structure.traversal.value for task in source_ordered.tasks} == {"source_ordered"}
    assert len(eventize_composed_task_plan(fast).event_plans) == len(
        eventize_composed_task_plan(source_ordered).event_plans
    )


def test_scalar_ast_and_source_names_do_not_select_a_different_task_family() -> None:
    source, weights, shape = _inputs()
    program = _program(source, weights, shape)
    score_ordinal = next(index for index, operation in enumerate(program.operations) if operation.name == "score.map")
    score = program.operations[score_ordinal]
    assert isinstance(score, MapPrimitive)
    affine_score = replace(
        score,
        name="unrelated_frontend_name",
        expression=scalar_binary(ScalarExpressionKind.ADD, score.expression, scalar_constant(0.125)),
    )
    mutated = replace(
        program,
        operations=tuple(
            affine_score if ordinal == score_ordinal else replace(operation, name=f"renamed.{ordinal}")
            for ordinal, operation in enumerate(program.operations)
        ),
    )
    original_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    mutated_plan = derive_composed_task_plan(
        mutated,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    original_task = next(task for task in original_plan.tasks if score_ordinal in task.operation_ordinals)
    mutated_task = next(task for task in mutated_plan.tasks if score_ordinal in task.operation_ordinals)

    assert program.semantic_fingerprint != mutated.semantic_fingerprint
    assert original_plan.fingerprint == mutated_plan.fingerprint
    assert original_task.structure.schedule_fingerprint == mutated_task.structure.schedule_fingerprint


def test_extra_compatible_map_preserves_generic_contract_fold_fusion() -> None:
    source, weights, shape = _inputs()
    program = _program(source, weights, shape)
    center_ordinal = next(
        index for index, operation in enumerate(program.operations) if operation.name == "score.center"
    )
    exponential_ordinal = center_ordinal + 1
    center = program.operations[center_ordinal]
    exponential = program.operations[exponential_ordinal]
    assert isinstance(center, MapPrimitive)
    assert isinstance(exponential, MapPrimitive)
    shifted = ProgramValue("frontend.temporary", center.output.axes, center.output.dtype)
    inserted = MapPrimitive(
        "frontend.extra_map",
        (center.output,),
        shifted,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(center.output.name),
            scalar_constant(0.125),
        ),
    )
    rewritten_exponential = replace(
        exponential,
        inputs=(shifted,),
        expression=scalar_unary(ScalarExpressionKind.EXP, scalar_input(shifted.name)),
    )
    operations = list(program.operations)
    operations[exponential_ordinal] = rewritten_exponential
    operations.insert(exponential_ordinal, inserted)
    mutated = replace(program, operations=tuple(operations))

    plan = derive_composed_task_plan(
        mutated,
        tile_extent_by_axis=_tiles(),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    task_by_operation = {
        mutated.operations[ordinal].name: task for task in plan.tasks for ordinal in task.operation_ordinals
    }
    fused_names = (
        "score.contract",
        "score.maximum",
        "score.center",
        "frontend.extra_map",
        "score.exponential",
        "score.sum",
        "value.contract",
        "value.normalize",
    )
    fused_task = task_by_operation["frontend.extra_map"]
    assert all(task_by_operation[name] is fused_task for name in fused_names)
    request = physical_request_for_composed_tasks(
        algebra_record=mutated.canonical_record,
        algebra_fingerprint=mutated.semantic_fingerprint,
        task_plan=plan,
        event_program=eventize_composed_task_plan(plan),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    fused_schedule = next(schedule for schedule in request.schedules if fused_task.id in schedule.task_ids)
    assert fused_schedule.family.name == "contract_phase_dag"
    assert physical_schedule_is_eligible(fused_schedule, plan)


def test_fold_dependent_map_composition_is_discovered_from_ssa_and_axes() -> None:
    batch = TensorAxis(40, 2, "owner")
    row = TensorAxis(41, 3, "row")
    stream = TensorAxis(42, 5, "stream")
    feature = TensorAxis(43, 4, "feature")
    result = TensorAxis(44, 6, "result")
    left = ProgramValue("left", (batch, row, feature), DType.FP32)
    right = ProgramValue("right", (feature, stream), DType.FP32)
    payload = ProgramValue("payload", (stream, result), DType.FP32)
    scores = ProgramValue("scores", (batch, row, stream), DType.FP32)
    mapped = ProgramValue("mapped", scores.axes, DType.FP32)
    folded = ProgramValue("folded", (batch, row), DType.FP32)
    dependent = ProgramValue("dependent", scores.axes, DType.FP32)
    reassociated = ProgramValue("reassociated", scores.axes, DType.FP32)
    output = ProgramValue("output", (batch, row, result), DType.FP32)
    operations = (
        ContractPrimitive("a", (left, right), scores, (feature,), DType.FP32),
        MapPrimitive(
            "b",
            (scores,),
            mapped,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(scores.name), scalar_input(scores.name)),
        ),
        FoldPrimitive("c", mapped, folded, (stream,), FoldReducer.SUM, DType.FP32),
        MapPrimitive(
            "d",
            (mapped, folded),
            dependent,
            scalar_binary(ScalarExpressionKind.DIVIDE, scalar_input(mapped.name), scalar_input(folded.name)),
        ),
        MapPrimitive(
            "e",
            (dependent, mapped),
            reassociated,
            scalar_binary(ScalarExpressionKind.ADD, scalar_input(dependent.name), scalar_input(mapped.name)),
        ),
        ContractPrimitive("f", (reassociated, payload), output, (stream,), DType.FP32),
    )
    program = TensorProgram((left, right, payload), operations, (output,), NumericalPolicy.FAST)
    plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: 2 for axis in (batch, row, stream, feature, result)},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.operation_ordinals == tuple(range(len(operations)))
    assert tuple(axis.axis_id for axis in task.structure.owner_axes) == (batch.id, row.id, result.id)
    assert tuple(axis.axis_id for axis in task.structure.streamed_axes) == (stream.id,)
    assert tuple(axis.axis_id for axis in task.structure.contraction_axes) == (feature.id,)
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=plan,
        event_program=eventize_composed_task_plan(plan),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    assert physical_schedule_is_eligible(request.schedules[0], plan)


def test_generic_planner_tracks_every_result_of_a_relation() -> None:
    axis = TensorAxis(50, 4, "domain")
    source = ProgramValue("source", (axis,), DType.FP32)
    first = ProgramValue("first", (axis,), DType.FP32)
    second = ProgramValue("second", (axis,), DType.FP32)
    relation = RelationPrimitive(
        "split",
        (source,),
        (first, second),
        (axis,),
        (OperandIndexing(source, (axis,), (index_axis(axis),)),),
        result_sources=(source, source),
    )
    program = TensorProgram((source,), (relation,), (first, second), NumericalPolicy.FAST)

    plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: 2},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].operation_kinds[0].value == "relation"
    assert len(plan.tasks[0].accesses) == 3
    assert {access.value_id for access in plan.tasks[0].accesses} == {0, 1, 2}


def _separate_map_fold_composition(program: TensorProgram) -> CanonicalKernelComposition:
    policy = PhysicalNumericalPolicy(program.numerical_policy.value)
    tasks = derive_composed_task_plan(
        program,
        tile_extent_by_axis={
            axis.id: min(axis.extent, 2)
            for operation in program.operations
            for value in primitive_outputs(operation)
            for axis in value.axes
        },
        numerical_policy=policy,
        fusion_boundary_values=tuple(
            operation.output for operation in program.operations if isinstance(operation, MapPrimitive)
        ),
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=tasks,
        event_program=eventize_composed_task_plan(tasks),
        target=PhysicalTarget.SM90,
        numerical_policy=policy,
    )
    return lower_canonical_kernel_composition(request)


@pytest.mark.parametrize("retain_maps", (False, True))
def test_contract_phase_cut_preserves_live_maps_and_bf16_rounding(retain_maps: bool) -> None:
    row, reduction, column = TensorAxis(0, 3), TensorAxis(1, 8), TensorAxis(2, 16)
    source = ProgramValue("source", (row, reduction), DType.BF16)
    auxiliary = ProgramValue("auxiliary", source.axes, DType.BF16)
    scale = ProgramValue("scale", (reduction,), DType.BF16)
    weight = ProgramValue("weight", (reduction, column), DType.BF16)
    product = ProgramValue("product", source.axes, DType.BF16)
    scaled = ProgramValue("scaled", source.axes, DType.BF16)
    output = ProgramValue("output", (row, column), DType.BF16)
    operations = (
        MapPrimitive(
            "multiply",
            (source, auxiliary),
            product,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(source.name), scalar_input(auxiliary.name)),
        ),
        MapPrimitive(
            "scale",
            (product, scale),
            scaled,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(product.name), scalar_input(scale.name)),
        ),
        ContractPrimitive("contract", (scaled, weight), output, (reduction,), DType.FP32),
    )
    program = TensorProgram(
        (source, auxiliary, scale, weight),
        operations,
        (product, scaled, output) if retain_maps else (output,),
        NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    composition = lower_canonical_kernel_composition(request)
    assert len(composition.components) == 1
    component = composition.components[0]
    cut = dependency_convex_contract_phase_schedule(program, composition, component)
    assert cut is not None
    assert tuple(phase.component.operation_ordinals for phase in cut.phases) == ((0, 1), (2,))
    ordinals = tensor_program_value_ordinals(program)
    assert cut.phases[0].component.live_output_value_ids == tuple(
        ordinals[value] for value in ((product, scaled) if retain_maps else (scaled,))
    )
    assert set(cut.phases[1].component.live_input_value_ids) == {ordinals[scaled], ordinals[weight]}
    assert cut.phases[1].component.live_output_value_ids == (ordinals[output],)
    assert cut.cut_value_ids == (ordinals[scaled],)
    task = request.task_plan.tasks[component.task_ids[0]]
    assert task.structure is not None
    assert cut.crossing_phase_edges == tuple(edge for edge in task.structure.phase_edges if edge.target_phase == 2)
    assert tuple(phase for group in cut.phases for phase in group.physical_phases) == task.structure.phases

    values = {ordinal: value for value, ordinal in ordinals.items()}
    used_ids = {
        value_id
        for phase in cut.phases
        for value_id in (*phase.component.live_input_value_ids, *phase.component.live_output_value_ids)
    }
    slots = {
        value_id: CutePipelineSlot(
            f"local_{value_id}", CutePipelineElementType.BF16, values[value_id].shape, CutePipelineSlotKind.SCRATCH, 16
        )
        for value_id in used_ids
    }
    launches = tuple(
        CuteKernelLaunch(
            f"phase_{phase.id}",
            f"kernel_{phase.id}",
            (1, 1, 1),
            (128, 1, 1),
            0,
            (),
            tuple(
                CutePipelineSlotArgument(f"read_{value_id}", slots[value_id].name, CutePipelineSlotAccess.READ)
                for value_id in phase.component.live_input_value_ids
            )
            + tuple(
                CutePipelineSlotArgument(f"write_{value_id}", slots[value_id].name, CutePipelineSlotAccess.WRITE)
                for value_id in phase.component.live_output_value_ids
            ),
            source_tasks=component.task_ids,
        )
        for phase in cut.phases
    )
    # A real allocated provider region stays intact while the independent
    # materialized-cut obligation is proved by the composed launch region.
    provider_region = streaming_physical_event_schedule_record(
        derive_streaming_physical_event_schedule(
            StreamingContractFoldDescriptor(
                first_contract_name="first_contract",
                fold_update_name="state_update",
                second_contract_name="second_contract",
                finalize_name="finalize",
                fold_extent=512,
                resident_tile_size=128,
                streamed_tile_size=64,
                pipeline_depth=3,
                resident_reduction_dimension=128,
                streamed_reduction_dimension=128,
                output_dimension=128,
                element_bytes=2,
            )
        )
    )
    nested = combine_physical_event_schedules(provider_region)
    recipe = compose_cute_pipeline_launch_recipe(
        composition,
        (
            CuteKernelComponentFragment(
                component.id,
                component.schedule_fingerprint,
                slots,
                launches,
                physical_event_schedules=(nested,),
                materialized_cut_value_ids=cut.cut_value_ids,
            ),
        ),
    )
    assert recipe.physical_event_provenance is not None
    launch_events = recipe.physical_event_provenance["schedule"]["regions"][0]
    assert launch_events["predecessor_pairs"] == (((0,), (1,)),)
    assert provider_region["allocations"]
    assert recipe.physical_event_provenance["schedule"]["regions"][1] == nested
    assert all(launch.source_tasks == component.task_ids for launch in recipe.launches)
    assert {slot.output_index for slot in recipe.slots if slot.output_index is not None} == set(
        range(len(program.outputs))
    )

    rng = np.random.default_rng(43)
    inputs = {
        value.name: np.asarray(jnp.asarray(rng.normal(size=value.shape), dtype=jnp.bfloat16)) for value in program.inputs
    }
    environment = dict(inputs)
    for phase in cut.phases:
        phase_program = TensorProgram(
            tuple(values[value_id] for value_id in phase.component.live_input_value_ids),
            tuple(program.operations[ordinal] for ordinal in phase.component.operation_ordinals),
            tuple(values[value_id] for value_id in phase.component.live_output_value_ids),
            program.numerical_policy,
        )
        environment.update(execute_tensor_program(phase_program, environment))
    bf16 = inputs[source.name].dtype
    product_expected = (inputs[source.name].astype(np.float32) * inputs[auxiliary.name].astype(np.float32)).astype(bf16)
    scaled_expected = (product_expected.astype(np.float32) * inputs[scale.name].astype(np.float32)).astype(bf16)
    output_expected = (scaled_expected.astype(np.float32) @ inputs[weight.name].astype(np.float32)).astype(bf16)
    np.testing.assert_array_equal(environment[output.name], output_expected)
    if retain_maps:
        np.testing.assert_array_equal(environment[product.name], product_expected)
        np.testing.assert_array_equal(environment[scaled.name], scaled_expected)


def _scale_gradient_program(case: str = "private") -> TensorProgram:
    row, feature = TensorAxis(0, 3, "row"), TensorAxis(1, 5, "feature")
    producer_axes = (row,) if case == "broadcast" else (row, feature)
    source = ProgramValue("source", producer_axes, DType.BF16)
    cotangent = ProgramValue("cotangent", producer_axes, DType.BF16)
    product = ProgramValue("product", producer_axes, DType.BF16)
    scaled = ProgramValue("scaled", (row, feature), DType.FP32)
    output_axes = () if case == "scalar" else (feature,)
    reduced = ProgramValue("reduced", output_axes, DType.FP32)
    result = ProgramValue("result", output_axes, DType.FP32)
    operations = (
        MapPrimitive(
            "multiply",
            (source, cotangent),
            product,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(source.name), scalar_input(cotangent.name)),
        ),
        MapPrimitive(
            "scale",
            (product,),
            scaled,
            scalar_binary(ScalarExpressionKind.DIVIDE, scalar_input(product.name), scalar_constant(3.0)),
        ),
        FoldPrimitive(
            "sum",
            scaled,
            reduced,
            (row, feature) if case == "scalar" else (row,),
            FoldReducer.SUM,
            DType.FP64 if case == "wide_accumulation" else DType.FP32,
            order=EvaluationOrder.SOURCE_ORDERED if case == "ordered_fold" else EvaluationOrder.ORDER_FREE,
        ),
        MapPrimitive(
            "finalize",
            (reduced,),
            result,
            scalar_binary(ScalarExpressionKind.ADD, scalar_input(reduced.name), scalar_constant(1.0)),
        ),
    )
    return TensorProgram(
        (source, cotangent),
        operations,
        (result, product) if case == "external" else (result,),
        NumericalPolicy.SOURCE_ORDERED if case == "source_ordered" else NumericalPolicy.FAST,
    )


def test_private_scale_gradient_maps_move_to_fold_owner_preserving_events_and_rounding() -> None:
    program = _scale_gradient_program()
    baseline = _separate_map_fold_composition(program)
    treatment = coalesce_private_map_fold_components(program, baseline)
    assert tuple(component.operation_ordinals for component in treatment.components) == ((0, 1, 2), (3,))
    assert treatment.components[0].live_output_value_ids == baseline.components[2].live_output_value_ids
    assert treatment.scheduled_request.algebra_record == baseline.scheduled_request.algebra_record
    assert len(treatment.concrete_events) == 1
    assert len(treatment.legalized_event_lowerings) == len(baseline.legalized_event_lowerings)
    assert all(not lowering.retained_operations for lowering in treatment.legalized_event_lowerings)
    schedule = treatment.scheduled_request.schedules[treatment.components[0].schedule_index]
    orders = schedule.enforced_orders
    internal_orders = tuple(order for order in orders if order.source_task_id in (0, 1))
    assert {(order.source_task_id, order.target_task_id) for order in internal_orders} == {(0, 1), (1, 2)}
    assert schedule.parameters["execution_owner_task"] == 2
    family_to_task = {task.family: task.id for task in baseline.scheduled_request.task_plan.tasks}
    assert {(order.source_task_id, order.target_task_id): set(order.relation.pairs) for order in internal_orders} == {
        (family_to_task[dependence.relation.source], family_to_task[dependence.relation.target]): set(
            dependence.relation.pairs
        )
        for dependence in baseline.scheduled_request.task_plan.dependences
        if family_to_task[dependence.relation.target] <= 2
    }
    source = np.asarray(jnp.asarray(np.linspace(-1.3, 1.7, 15).reshape(3, 5), dtype=jnp.bfloat16))
    cotangent = np.asarray(jnp.asarray(np.linspace(0.2, 1.2, 15).reshape(3, 5), dtype=jnp.bfloat16))
    actual = execute_tensor_program(program, {"source": source, "cotangent": cotangent})["result"]
    product = (source.astype(np.float32) * cotangent.astype(np.float32)).astype(source.dtype)
    expected = np.sum(product.astype(np.float32) / np.float32(3.0), axis=0) + np.float32(1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "case", ("external", "broadcast", "source_ordered", "scalar", "ordered_fold", "wide_accumulation")
)
def test_private_map_fold_keeps_required_materialization_and_source_order(case: str) -> None:
    program = _scale_gradient_program(case)
    baseline = _separate_map_fold_composition(program)
    treatment = coalesce_private_map_fold_components(program, baseline)
    if case in ("source_ordered", "scalar", "ordered_fold", "wide_accumulation"):
        assert treatment.fingerprint == baseline.fingerprint
    else:
        # A private suffix may still fuse, but the shared or broadcast producer
        # must keep a separately materialized result.
        owner = next(component for component in treatment.components if 0 in component.operation_ordinals)
        assert owner.operation_ordinals == (0,)
        assert owner.live_output_value_ids == baseline.components[0].live_output_value_ids


@pytest.mark.parametrize("case", ("shared", "mixed_output_axes", "reduction_operand", "independent", "fold"))
def test_same_coordinate_reuse_requires_shared_producers_outside_reduction_coordinates(case: str) -> None:
    row, column, reduction = TensorAxis(0, 3), TensorAxis(1, 5), TensorAxis(2, 7)
    left = ProgramValue("left", (row, reduction), DType.BF16)
    weight = ProgramValue("weight", (reduction, column), DType.BF16)
    product = ProgramValue("product", (row, column), DType.BF16)
    scale = ProgramValue("scale", (column,), DType.BF16)
    scaled = ProgramValue("scaled", (row, column), DType.BF16)
    contract = ContractPrimitive("contract", (left, weight), product, (reduction,), DType.FP32)
    mapping = MapPrimitive(
        "scale",
        (product, scale),
        scaled,
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input("product"), scalar_input("scale")),
    )
    operations = (contract, mapping)
    inputs = (left, weight, scale)
    outputs = (product, scaled)
    if case == "mixed_output_axes":
        outputs = (product, scale)
    elif case == "reduction_operand":
        mapped_left = ProgramValue("mapped_left", left.axes, DType.BF16)
        producer = MapPrimitive("producer", (left,), mapped_left, scalar_input("left"))
        operations = (producer, replace(contract, inputs=(mapped_left, weight)), mapping)
    elif case == "independent":
        operations = (contract, replace(contract, name="second", output=scaled))
    elif case == "fold":
        reduced = ProgramValue("reduced", (row,), DType.BF16)
        operations = (contract, FoldPrimitive("fold", product, reduced, (column,), FoldReducer.SUM, DType.FP32))
        outputs = (product, reduced)
    assert supports_same_coordinate_producer_reuse(operations, inputs, outputs) is (case == "shared")
