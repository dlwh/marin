# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.event_dataflow import ImperativeEventOpKind
from tile_lifetime.jax_normalized_exp_contract_training import (
    prepare_sm90_jax_owned_normalized_exp_value_and_grad,
    prepare_sm90_split_normalized_exp_value_and_grad,
    trace_jax_owned_normalized_exp_value_and_grad,
)
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import (
    NumericalPolicy,
    execute_tensor_program,
    primitive_inputs,
    tensor_program_value_ordinals,
)
from tile_lifetime.terminal_gradient_layout import terminal_gradient_layout


def _grouped_normalized_exp_contract(*, score_transform: str, predicate_offset: int):
    def function(query, key, value):
        score = jnp.einsum("bqhgd,bkhd->bhgqk", query, key) * (128**-0.5)
        if score_transform == "tanh":
            score = 1.5 * jnp.tanh(score / 1.5)
        elif score_transform == "affine":
            score = 1.25 * score + 0.2
        query_position = jnp.arange(score.shape[-2])[:, None]
        key_position = jnp.arange(score.shape[-1])[None, :]
        score = jnp.where(key_position <= query_position + predicate_offset, score, -jnp.inf)
        row_maximum = jnp.max(score, axis=-1, keepdims=True)
        exponential = jnp.exp(score - row_maximum)
        probability = exponential / jnp.sum(exponential, axis=-1, keepdims=True)
        return jnp.einsum("bhgqk,bkhv->bqhgv", probability, value)

    return function


def _plan(*, score_transform: str, predicate_offset: int):
    shapes = ((1, 3, 2, 2, 128), (1, 4, 2, 128), (1, 4, 2, 128), (1, 3, 2, 2, 128))
    arguments = tuple(np.ones(shape, dtype=jnp.bfloat16) for shape in shapes)
    trace = trace_jax_owned_normalized_exp_value_and_grad(
        _grouped_normalized_exp_contract(
            score_transform=score_transform,
            predicate_offset=predicate_offset,
        ),
        *arguments[:3],
        output_cotangent=arguments[3],
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(
        trace.value_and_grad,
        tile_size=128,
        target=PhysicalTarget.SM90,
    )
    return prepare_sm90_jax_owned_normalized_exp_value_and_grad(request, trace, tile_size=128)


def test_natural_value_and_grad_reaches_exact_event_component_composition() -> None:
    plan = _plan(score_transform="identity", predicate_offset=1)

    assert plan.trace.canonical_record["differentiation_owner"] == "jax"
    assert plan.pullback_plan.replacement.identity == "jax_value_and_grad_of_fp32_output_inner_product"
    assert plan.pullback_plan.replacement.covered_task_ids == tuple(
        task.id for task in plan.source_request.task_plan.tasks
    )
    assert all(lowering.retained_operations for lowering in plan.preselection_event_lowerings)
    assert {
        operation.kind for lowering in plan.preselection_event_lowerings for operation in lowering.retained_operations
    } == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }
    assert plan.crossing_event_names
    pullback = plan.physical_request.normalized_exp_contract_pullbacks[0]
    pullback_schedule = next(
        schedule for schedule in plan.physical_request.schedules if schedule.parameters.get("direction") == "pullback"
    )
    operation_to_task = {
        operation: task.id for task in plan.physical_request.task_plan.tasks for operation in task.operation_ordinals
    }
    assert {operation_to_task[operation] for operation in pullback.operation_ordinals} <= set(pullback_schedule.task_ids)
    assert all(not lowering.retained_operations for lowering in plan.composition.legalized_event_lowerings)
    assert tuple((value.shape, value.dtype) for value in plan.physical_program.outputs) == tuple(
        (value.shape, value.dtype) for value in plan.trace.value_and_grad.outputs
    )
    plan.require_whole_request_runtime()


@pytest.mark.parametrize("predicate_offset", (0, 1))
def test_scalar_programs_and_affine_predicates_do_not_select_schedule_family(predicate_offset: int) -> None:
    plans = tuple(
        _plan(score_transform=score_transform, predicate_offset=predicate_offset)
        for score_transform in ("identity", "tanh", "affine")
    )
    lowerings = tuple(plan.pullback_plan.lowering for plan in plans)

    assert all(lowering.physical_schedule == lowerings[0].physical_schedule for lowering in lowerings)
    assert all(lowering.relation_layout == lowerings[0].relation_layout for lowering in lowerings)
    assert len({str(lowering.compile_request.scalar_programs["score_map"]) for lowering in lowerings}) == 3
    assert len({str(lowering.compile_request.scalar_programs["score_map_vjp"]) for lowering in lowerings}) == 3
    assert len({lowering.compile_request.digest for lowering in lowerings}) == 3
    assert all(plan.crossing_event_names for plan in plans)


def test_affine_predicate_changes_identity_without_changing_schedule_family() -> None:
    diagonal = _plan(score_transform="affine", predicate_offset=0)
    offset = _plan(score_transform="affine", predicate_offset=1)

    assert diagonal.pullback_plan.lowering.physical_schedule == offset.pullback_plan.lowering.physical_schedule
    assert diagonal.pullback_plan.lowering.relation_layout == offset.pullback_plan.lowering.relation_layout
    assert diagonal.pullback_plan.lowering.compile_request.digest != offset.pullback_plan.lowering.compile_request.digest
    assert diagonal.physical_request.algebra_fingerprint != offset.physical_request.algebra_fingerprint


def test_stale_pullback_capability_cannot_fall_back_to_graph_discovery() -> None:
    plan = _plan(score_transform="identity", predicate_offset=1)
    capability = plan.physical_request.normalized_exp_contract_pullbacks[0]
    stale = replace(
        plan.physical_request,
        normalized_exp_contract_pullbacks=(replace(capability, score_map_vjp_operation=10_000),),
    )

    with pytest.raises(ValueError, match="pullback operation ordinal is stale"):
        attach_sm90_contract_fold_schedule(stale)


def test_saved_state_split_installs_disjoint_schedules_with_crossing_events() -> None:
    shapes = ((1, 3, 2, 2, 128), (1, 4, 2, 128), (1, 4, 2, 128), (1, 3, 2, 2, 128))
    arguments = tuple(np.ones(shape, dtype=jnp.bfloat16) for shape in shapes)
    base = _grouped_normalized_exp_contract(score_transform="tanh", predicate_offset=1)

    def nonlinear_finalizer(query, key, value):
        normalized = base(query, key, value)
        return jnp.tanh(normalized) + 0.25 * normalized

    trace = trace_jax_owned_normalized_exp_value_and_grad(
        nonlinear_finalizer,
        *arguments[:3],
        output_cotangent=arguments[3],
        numerical_policy=NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(
        trace.value_and_grad,
        tile_size=128,
        target=PhysicalTarget.SM90,
    )
    plan = prepare_sm90_split_normalized_exp_value_and_grad(request, trace, tile_size=128)
    forward, pullback = plan.directional_schedules

    assert forward.parameters["direction"] == "forward"
    assert forward.parameters["execution_graph"] == "contract_fold_then_recomputed_map_contract"
    assert forward.parameters["pass_count"] == 2
    assert forward.parameters["selection"] == "typed_conversion_boundaries_and_task_resource_estimates"
    boundary = forward.parameters["conversion_boundary"]
    assert boundary["numerator"]["dtype"] == "bf16"
    assert boundary["denominator"]["dtype"] == "bf16"
    assert boundary["fold_state"]["dtype"] == "fp32"
    assert boundary == plan.forward_conversion_boundary.canonical_record
    assert pullback.parameters["direction"] == "pullback"
    assert not set(forward.task_ids) & set(pullback.task_ids)
    assert len(plan.replacement.saved_values) == 1
    assert plan.crossing_event_names
    assert any(event.producer_component != event.consumer_component for event in plan.composition.concrete_events)
    reverse_component = next(
        component
        for component in plan.composition.components
        if component.schedule_index == plan.physical_request.schedules.index(pullback)
    )
    values_by_ordinal = {ordinal: value for value, ordinal in tensor_program_value_ordinals(plan.program).items()}
    reverse_inputs = tuple(values_by_ordinal[ordinal] for ordinal in reverse_component.live_input_value_ids)
    reverse_outputs = tuple(values_by_ordinal[ordinal] for ordinal in reverse_component.live_output_value_ids)
    assert plan.saved_values[0] in reverse_inputs
    assert tuple((value.shape, value.dtype) for value in reverse_outputs) == tuple(
        (value.shape, value.dtype) for value in trace.value_and_grad.outputs[1:]
    )
    assert all(value in plan.program.outputs for value in reverse_outputs)
    capability = plan.physical_request.normalized_exp_contract_regions[0]
    producer_inputs = primitive_inputs(plan.program.operations[capability.producer_operation])
    consumer_inputs = primitive_inputs(plan.program.operations[capability.consumer_operation])
    primals = (
        producer_inputs[capability.producer_owner_operand],
        producer_inputs[1 - capability.producer_owner_operand],
        consumer_inputs[1 - capability.normalized_operand],
    )
    terminal_ordinals = plan.physical_request.normalized_exp_contract_pullbacks[0].terminal_contract_operations
    bindings = tuple(
        terminal_gradient_layout(plan.program, reverse_component.operation_ordinals, ordinal, primal, reverse_outputs)
        for ordinal, primal in zip(terminal_ordinals, primals, strict=True)
    )
    assert {binding.output for binding in bindings} == set(reverse_outputs)
    for binding, primal in zip(bindings, primals, strict=True):
        # This existing rank-five/rank-four grouped plan retains its declared
        # coordinate order, including renamed output axes and unit adjoint folds.
        physical = np.arange(np.prod(primal.shape), dtype=np.float32).astype(jnp.bfloat16).reshape(primal.shape)
        view = binding.bind_storage(primal.axes, physical.shape, tuple(s // physical.itemsize for s in physical.strides))
        actual = execute_tensor_program(view.transport_program(), {"physical_storage": physical.reshape(-1)})
        np.testing.assert_array_equal(actual[binding.output.name], physical)
    assert reverse_component.id == len(plan.composition.components) - 1
    assert len(plan.composition.components) >= 3

    execution_arguments = tuple(np.asarray(argument, dtype=np.float32) for argument in arguments)
    source_values = execute_tensor_program(
        trace.value_and_grad,
        dict(zip((value.name for value in trace.value_and_grad.inputs), execution_arguments, strict=True)),
    )
    executable_values = execute_tensor_program(
        plan.program,
        dict(zip((value.name for value in plan.program.inputs), execution_arguments, strict=True)),
    )
    for source_output, executable_output in zip(
        trace.value_and_grad.outputs,
        plan.program.outputs,
        strict=True,
    ):
        np.testing.assert_allclose(
            executable_values[executable_output.name],
            source_values[source_output.name],
        )
    plan.require_whole_request_runtime()


def test_conversion_faithful_split_schedule_does_not_dispatch_on_score_ast() -> None:
    shapes = ((1, 3, 2, 2, 128), (1, 4, 2, 128), (1, 4, 2, 128), (1, 3, 2, 2, 128))
    arguments = tuple(np.ones(shape, dtype=jnp.bfloat16) for shape in shapes)
    plans = []
    for score_transform in ("identity", "tanh", "affine"):
        trace = trace_jax_owned_normalized_exp_value_and_grad(
            _grouped_normalized_exp_contract(score_transform=score_transform, predicate_offset=1),
            *arguments[:3],
            output_cotangent=arguments[3],
            numerical_policy=NumericalPolicy.FAST,
        )
        request = jax_tensor_program_executable_request(
            trace.value_and_grad,
            tile_size=128,
            target=PhysicalTarget.SM90,
        )
        plans.append(prepare_sm90_split_normalized_exp_value_and_grad(request, trace, tile_size=128))

    forward_schedules = tuple(plan.directional_schedules[0] for plan in plans)
    assert all(schedule.family == forward_schedules[0].family for schedule in forward_schedules)
    assert all(
        schedule.parameters["contract"] == forward_schedules[0].parameters["contract"] for schedule in forward_schedules
    )
    assert all(schedule.parameters["pass_count"] == 2 for schedule in forward_schedules)
    assert all(schedule.parameters["producer_contract_traversals"] == 3 for schedule in forward_schedules)
    assert all(
        schedule.parameters["statistics_traversals"] == ("maximum", "centered_exponential_sum")
        for schedule in forward_schedules
    )
    assert len({plan.physical_request.algebra_fingerprint for plan in plans}) == 3
