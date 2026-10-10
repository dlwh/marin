# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    ReverseMaterializationPolicy,
    build_one_fold_contract_fixture,
    build_two_fold_contract_fixture,
    compile_fold_map_contract_training,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.row_normalization_jax import (
    RowProjectionConfig,
    layer_projection,
    projection_training_step,
    projection_value_and_grad,
)
from tile_lifetime.tensor_program import (
    MapPrimitive,
    NumericalPolicy,
    ScalarExpressionKind,
    execute_tensor_program,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
)


@pytest.mark.parametrize("builder", (build_one_fold_contract_fixture, build_two_fold_contract_fixture))
@pytest.mark.parametrize("policy", tuple(ReverseMaterializationPolicy))
def test_training_plan_executes_the_automatic_adjoint(builder, policy) -> None:
    source = builder(rows=4, hidden=7, features=5, epsilon=3e-4)
    plan = compile_fold_map_contract_training(source, materialization_policy=policy)
    rng = np.random.default_rng(31)
    inputs = {
        source.inputs[0].name: rng.normal(size=(4, 7)).astype(np.float32),
        source.inputs[1].name: rng.normal(size=(7,)).astype(np.float32),
        source.inputs[2].name: rng.normal(size=(7, 5)).astype(np.float32),
        plan.automatic_adjoint.output_cotangents[0].name: rng.normal(size=(4, 5)).astype(np.float32),
    }
    automatic = execute_tensor_program(plan.automatic_adjoint.program, inputs)
    backward_inputs = {value.name: inputs[value.name] for value in plan.backward.inputs if value.name in inputs}
    forward_values = execute_tensor_program(
        replace(source, outputs=tuple(operation.output for operation in source.operations)),
        {value.name: inputs[value.name] for value in source.inputs},
    )
    backward_inputs.update({name: forward_values[name] for name in plan.saved_values})
    actual = execute_tensor_program(plan.backward, backward_inputs)

    for gradient in plan.input_gradients:
        np.testing.assert_allclose(actual[gradient.name], automatic[gradient.name], rtol=2e-5, atol=2e-5)


def test_region_is_derived_from_dependencies_not_operation_positions_or_scalar_identity() -> None:
    source = build_two_fold_contract_fixture(rows=3, hidden=7, features=5)
    plan = compile_fold_map_contract_training(source)
    maps = [operation for operation in source.operations if isinstance(operation, MapPrimitive)]
    target = maps[-1]
    mutated = replace(
        source,
        operations=tuple(
            (
                replace(
                    operation,
                    expression=scalar_binary(ScalarExpressionKind.ADD, operation.expression, scalar_constant(0.25)),
                )
                if operation is target
                else operation
            )
            for operation in source.operations
        ),
    )
    changed = compile_fold_map_contract_training(mutated)

    assert len(plan.region.fold_states) == len(changed.region.fold_states) == 2
    assert plan.region.owner_axes == changed.region.owner_axes
    assert plan.region.contraction_axes == changed.region.contraction_axes
    assert plan.region.scalar_maps[-1].expression != changed.region.scalar_maps[-1].expression
    assert plan.forward_contract.shape == changed.forward_contract.shape
    assert plan.backward.semantic_fingerprint != changed.backward.semantic_fingerprint
    assert tuple(type(step) for step in plan.backward_steps) == tuple(type(step) for step in changed.backward_steps)


def test_materialization_policies_are_generic_graph_cuts() -> None:
    source = build_two_fold_contract_fixture(rows=8, hidden=16, features=24)
    save_all = compile_fold_map_contract_training(
        source, materialization_policy=ReverseMaterializationPolicy.SAVE_ALL_INTERMEDIATES
    )
    save_folds = compile_fold_map_contract_training(
        source, materialization_policy=ReverseMaterializationPolicy.SAVE_FOLD_STATES
    )
    recompute = compile_fold_map_contract_training(source, materialization_policy=ReverseMaterializationPolicy.RECOMPUTE)

    assert save_folds.saved_values == tuple(value.name for value in save_folds.region.fold_states)
    assert recompute.saved_values == ()
    assert len(save_all.recomputed_values) < len(save_folds.recomputed_values) < len(recompute.recomputed_values)
    assert save_all.input_gradients == save_folds.input_gradients == recompute.input_gradients
    assert len(recompute.backward_task_plan.tasks) > 1
    assert len(recompute.backward_event_program.event_plans) == len(recompute.backward_task_plan.dependences)


def test_natural_jax_value_and_grad_and_composed_boundary_agree() -> None:
    config = RowProjectionConfig(rows=2, hidden=5, features=3, epsilon=2e-4)
    rng = np.random.default_rng(917)
    arguments = (
        jnp.asarray(rng.normal(size=(config.rows, config.hidden)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(config.hidden,)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(config.hidden, config.features)), dtype=jnp.float32),
    )
    output_cotangent = jnp.asarray(rng.normal(size=(config.rows, config.features)), dtype=jnp.float32)
    project = layer_projection(config)
    objective, objective_gradients = projection_value_and_grad(project)(*arguments, output_cotangent)
    output, composed_gradients = projection_training_step(project)(*arguments, output_cotangent)

    np.testing.assert_allclose(objective, jnp.sum(output * output_cotangent), rtol=2e-6, atol=2e-6)
    for objective_gradient, composed_gradient in zip(objective_gradients, composed_gradients, strict=True):
        np.testing.assert_allclose(objective_gradient, composed_gradient, rtol=2e-6, atol=2e-6)

    imported = jax_function_to_tensor_program(project, *arguments, numerical_policy=NumericalPolicy.FAST)
    plan = compile_fold_map_contract_training(imported)
    assert len(plan.region.fold_states) == 2
    assert plan.region.contraction_axes == plan.region.contract.reduction_axes
    imported_inputs = {
        value.name: np.asarray(argument) for value, argument in zip(imported.inputs, arguments, strict=True)
    }
    all_forward = execute_tensor_program(
        replace(
            imported,
            outputs=tuple(output for operation in imported.operations for output in primitive_outputs(operation)),
        ),
        imported_inputs,
    )
    backward_inputs = {
        **{value.name: imported_inputs[value.name] for value in plan.backward.inputs if value.name in imported_inputs},
        **{name: all_forward[name] for name in plan.saved_values},
        plan.automatic_adjoint.output_cotangents[0].name: np.asarray(output_cotangent),
    }
    actual_gradients = execute_tensor_program(plan.backward, backward_inputs)
    for gradient, expected in zip(plan.input_gradients, composed_gradients, strict=True):
        np.testing.assert_allclose(actual_gradients[gradient.name], expected, rtol=2e-5, atol=2e-5)
