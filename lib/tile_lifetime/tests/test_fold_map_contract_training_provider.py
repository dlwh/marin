# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import tile_lifetime.fold_map_contract_training_provider as training_provider
from tile_lifetime.cuda_axis_fold_codegen import evaluate_axis_fold_program
from tile_lifetime.fold_map_contract_training_provider import provide_fold_map_contract_training
from tile_lifetime.jax_fold_map_contract_training import execute_jax_tensor_program_values
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.row_normalization_jax import RowProjectionConfig, layer_projection
from tile_lifetime.row_normalization_training import (
    ReverseMaterializationPolicy,
    compile_fold_map_contract_training,
)
from tile_lifetime.tensor_program import (
    FoldPrimitive,
    FoldReducer,
    NumericalPolicy,
    ProgramValue,
    TensorAxis,
    execute_tensor_program,
    primitive_outputs,
)


def _centered_plan():
    config = RowProjectionConfig(rows=3, hidden=7, features=5, epsilon=2e-4)
    rng = np.random.default_rng(1201)
    arguments = (
        jnp.asarray(rng.normal(size=(config.rows, config.hidden)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(config.hidden,)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(config.hidden, config.features)), dtype=jnp.float32),
    )
    program = jax_function_to_tensor_program(
        layer_projection(config),
        *arguments,
        numerical_policy=NumericalPolicy.FAST,
    )
    return compile_fold_map_contract_training(program), arguments


def test_provider_generates_every_automatic_adjoint_fold_from_scalar_dags() -> None:
    plan, arguments = _centered_plan()
    provider = provide_fold_map_contract_training(plan, target_name="shuttle.fold_map_contract_training")

    folds = tuple(step for step in plan.backward_steps if type(step).__name__ == "GeneratedFoldSkeleton")
    assert len(provider.regions) == 1
    region = provider.regions[0]
    assert len(region.fold_bindings) == len(folds) == 3
    assert provider.task_ids == tuple(task.id for task in plan.backward_task_plan.tasks)
    assert provider.event_count == len(plan.backward_event_program.event_plans)
    assert provider.saved_values == plan.saved_values
    assert provider.recomputed_values == plan.recomputed_values
    assert region.generated.source.count("__global__") == 3
    assert "fold_26_data" in region.generated.source
    assert all(token not in region.generated.source.lower() for token in ("rmsnorm", "layernorm", "normalization"))

    rng = np.random.default_rng(1203)
    source_inputs = {
        value.name: np.asarray(argument) for value, argument in zip(plan.source.inputs, arguments, strict=True)
    }
    forward_values = execute_tensor_program(
        replace(
            plan.source,
            outputs=(
                *plan.source.inputs,
                *(output for operation in plan.source.operations for output in primitive_outputs(operation)),
            ),
        ),
        source_inputs,
    )
    backward_inputs = {
        **{value.name: source_inputs[value.name] for value in plan.backward.inputs if value.name in source_inputs},
        **{name: forward_values[name] for name in plan.saved_values},
        plan.automatic_adjoint.output_cotangents[0].name: rng.normal(size=(3, 5)).astype(np.float32),
    }
    backward_values = execute_tensor_program(
        replace(
            plan.backward,
            outputs=(
                *plan.backward.inputs,
                *(output for operation in plan.backward.operations for output in primitive_outputs(operation)),
            ),
        ),
        backward_inputs,
    )
    for binding in region.fold_bindings:
        generated_inputs = {item.parameter: backward_values[item.value.name] for item in binding.inputs}
        actual = evaluate_axis_fold_program(binding.stage.program, generated_inputs)
        np.testing.assert_allclose(actual, backward_values[binding.fold.output.name], rtol=2e-5, atol=2e-5)


def test_provider_identity_tracks_materialization_cut_and_scalar_dag() -> None:
    plan, _ = _centered_plan()
    provider = provide_fold_map_contract_training(plan, target_name="shuttle.fold_map_contract_training")

    assert provider.regions[0].generated.semantic_fingerprints == tuple(
        binding.stage.program.semantic_fingerprint for binding in provider.regions[0].fold_bindings
    )
    assert len(set(provider.regions[0].generated.semantic_fingerprints)) == len(provider.regions[0].fold_bindings)
    assert provider.regions[0].generated.source_sha256


def test_recompute_cut_is_partitioned_at_contract_dependency() -> None:
    plan, _ = _centered_plan()
    recomputed = compile_fold_map_contract_training(
        plan.source,
        materialization_policy=ReverseMaterializationPolicy.RECOMPUTE,
    )
    provider = provide_fold_map_contract_training(recomputed, target_name="shuttle.recompute")

    assert tuple(len(region.fold_bindings) for region in provider.regions) == (2, 3)
    assert tuple(region.launch_operation_ordinal for region in provider.regions) == (0, 22)
    assert tuple(len(region.generated.outputs) for region in provider.regions) == (2, 3)
    assert "fold_28" not in {value.name for value in provider.regions[1].generated.inputs}


def test_generic_jax_executor_covers_terminal_maps_relations_and_contracts() -> None:
    plan, arguments = _centered_plan()
    rng = np.random.default_rng(1207)
    source_inputs = dict(zip((value.name for value in plan.source.inputs), arguments, strict=True))
    forward = execute_jax_tensor_program_values(plan.source, source_inputs)
    backward_inputs = {
        value.name: (
            source_inputs[value.name]
            if value.name in source_inputs
            else (
                forward[value.name]
                if value.name in forward
                else jnp.asarray(rng.normal(size=value.shape), dtype=jnp.float32)
            )
        )
        for value in plan.backward.inputs
    }
    actual = execute_jax_tensor_program_values(plan.backward, backward_inputs)
    expected = execute_tensor_program(plan.backward, backward_inputs)

    for output in plan.backward.outputs:
        np.testing.assert_allclose(actual[output.name], expected[output.name], rtol=2e-5, atol=2e-5)


def test_provider_rejects_unsupported_reducer_and_non_rank_two_domain_precisely() -> None:
    plan, _ = _centered_plan()
    fold = next(operation for operation in plan.backward.operations if isinstance(operation, FoldPrimitive))
    maximum = replace(fold, reducer=FoldReducer.MAXIMUM)
    maximum_plan = replace(
        plan,
        backward=replace(
            plan.backward,
            operations=tuple(maximum if operation is fold else operation for operation in plan.backward.operations),
        ),
    )
    with pytest.raises(ValueError, match="only supports sum reductions"):
        provide_fold_map_contract_training(maximum_plan, target_name="shuttle.maximum")

    extra_axis = TensorAxis(99, 1, "extra")
    rank_three_input = ProgramValue(fold.input.name, (*fold.input.axes, extra_axis), fold.input.dtype)
    rank_two_output = ProgramValue(fold.output.name, (*fold.output.axes, extra_axis), fold.output.dtype)
    rank_three = replace(fold, input=rank_three_input, output=rank_two_output)
    with pytest.raises(ValueError, match="contiguous rank-two input"):
        training_provider._axis_fold_stage(
            rank_three,
            producer_by_value={},
            operation_ordinal={id(rank_three): 0},
            threads=256,
        )
