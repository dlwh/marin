# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.backends.h100.generic_relation_simt import (
    generate_sm90_dependency_convex_simt,
    generate_sm90_relation_simt,
)
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.expert_executable_pipeline import (
    ExpertRouteSchedule,
    derive_expert_route_dataflow,
    execute_expert_route_forward,
    expert_physical_request,
    expert_tensor_program,
    generic_tensor_program_physical_request,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.moe_reference import (
    MoEDebugConfig,
    export_debug_moe_training_step_text,
    export_debug_moe_value_and_grad_text,
    moe_region,
    moe_training_step,
    moe_value_and_grad,
)
from tile_lifetime.physical_task_contract import (
    ParametricEventRealization,
    PhysicalNumericalPolicy,
    TaskOperationKind,
    TaskTraversal,
    instantiate_parametric_event_tensor,
    instantiate_task_dependence,
)
from tile_lifetime.relation import RelationPlacement
from tile_lifetime.tensor_program import (
    IndexExpressionKind,
    NumericalPolicy,
    RelationPrimitive,
    SelectionComparison,
    SelectionPrimitive,
    SelectionTieBreak,
    TransportPrimitive,
    deserialize_tensor_program,
    execute_tensor_program,
    serialize_tensor_program,
)


def test_generic_physical_request_preserves_source_vs_fast_policy_identity() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, _ = _jax_arguments(config)
    source_program = jax_function_to_tensor_program(moe_region(config), *inputs)
    fast_program = replace(source_program, numerical_policy=NumericalPolicy.FAST)

    source = generic_tensor_program_physical_request(source_program, schedule=_schedule())
    fast = generic_tensor_program_physical_request(fast_program, schedule=_schedule())

    assert source.numerical_policy is PhysicalNumericalPolicy.SOURCE_ORDERED
    assert fast.numerical_policy is PhysicalNumericalPolicy.FAST
    assert {task.structure.traversal for task in source.task_plan.tasks} == {TaskTraversal.SOURCE_ORDERED}
    assert TaskTraversal.UNORDERED in {task.structure.traversal for task in fast.task_plan.tasks}
    assert source.task_plan.fingerprint != fast.task_plan.fingerprint
    assert tuple(schedule.fingerprint for schedule in source.schedules) != tuple(
        schedule.fingerprint for schedule in fast.schedules
    )
    assert source.fingerprint != fast.fingerprint
    assert source.artifact_fingerprint != fast.artifact_fingerprint


def _jax_arguments(config: MoEDebugConfig):
    shapes = (
        (config.tokens, config.hidden),
        (config.hidden, config.experts),
        (config.intermediate, config.hidden),
        (config.intermediate, config.hidden),
        (config.hidden, config.intermediate),
        (config.experts, config.intermediate, config.hidden),
        (config.experts, config.intermediate, config.hidden),
        (config.experts, config.hidden, config.intermediate),
    )
    keys = jax.random.split(jax.random.key(23), len(shapes) + 1)
    inputs = tuple(
        jax.random.normal(key, shape, dtype=jnp.bfloat16) / 4 for key, shape in zip(keys[:-1], shapes, strict=True)
    )
    cotangent = jax.random.normal(
        keys[-1],
        (config.tokens, config.hidden),
        dtype=jnp.bfloat16,
    )
    return inputs, cotangent


def _schedule() -> ExpertRouteSchedule:
    return ExpertRouteSchedule(
        source_placement=RelationPlacement(
            rank_count=2,
            rank_by_item=(0, 1, 0, 1),
            local_item_by_item=(0, 0, 1, 1),
        ),
        destination_placement=RelationPlacement(
            rank_count=2,
            rank_by_item=(0, 0, 1, 1),
            local_item_by_item=(0, 1, 0, 1),
        ),
        tile_rows=2,
        macrobatch_rows=4,
    )


def test_jax_owned_reverse_and_composed_boundary_are_numerically_identical() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, cotangent = _jax_arguments(config)

    differentiated = moe_value_and_grad(config)(*inputs, cotangent)
    composed = moe_training_step(config)(*inputs, cotangent)

    expected_value = jnp.vdot(composed[0].astype(jnp.float32), cotangent.astype(jnp.float32))
    np.testing.assert_array_equal(differentiated[0], expected_value)
    for actual, expected in zip(differentiated[1:4], composed[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(differentiated[4:], composed[3:], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        assert np.count_nonzero(np.asarray(actual)) > 0


def test_jax_reverse_exports_have_no_custom_differentiation_boundary() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)

    for stablehlo in (
        export_debug_moe_value_and_grad_text(config),
        export_debug_moe_training_step_text(config),
    ):
        assert "stablehlo.custom_call" not in stablehlo
        assert "stablehlo.composite" in stablehlo
        assert "stablehlo.scatter" in stablehlo


def test_relation_transport_event_pipeline_matches_independent_routed_reference() -> None:
    rng = np.random.default_rng(29)
    expert_indices = np.asarray(
        ((0, 2), (1, 3), (3, 0), (2, 1)),
        dtype=np.int32,
    )
    route_weights = rng.random((4, 2), dtype=np.float32)
    route_weights /= np.sum(route_weights, axis=1, keepdims=True)
    source = rng.normal(size=(4, 4)).astype(np.float32)
    gate_weight = rng.normal(size=(4, 5, 4)).astype(np.float32)
    up_weight = rng.normal(size=(4, 5, 4)).astype(np.float32)
    down_weight = rng.normal(size=(4, 4, 5)).astype(np.float32)
    plan = derive_expert_route_dataflow(
        expert_indices,
        route_weights,
        hidden_dimension=4,
        schedule=_schedule(),
    )

    actual = execute_expert_route_forward(plan, source, gate_weight, up_weight, down_weight)
    expected = np.zeros_like(actual)
    for token in range(expert_indices.shape[0]):
        for slot in range(expert_indices.shape[1]):
            expert = expert_indices[token, slot]
            gate = gate_weight[expert] @ source[token]
            up = up_weight[expert] @ source[token]
            hidden = (gate / (1.0 + np.exp(-gate))) * up
            expected[token] += route_weights[token, slot] * (down_weight[expert] @ hidden)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_executable_algebra_carries_selection_and_assignment_as_generic_relations() -> None:
    program = expert_tensor_program(
        token_count=4,
        hidden_dimension=4,
        intermediate_dimension=6,
        expert_count=3,
        selected_count=2,
    )

    selection = next(operation for operation in program.operations if isinstance(operation, SelectionPrimitive))
    assert selection.comparison is SelectionComparison.TOTAL_ORDER
    assert selection.stable
    assert selection.tie_break is SelectionTieBreak.SOURCE_INDEX_ASCENDING
    relations = tuple(operation for operation in program.operations if isinstance(operation, RelationPrimitive))
    assert {relation.inputs[0].name for relation in relations} == {
        "routed.gate_weight",
        "routed.up_weight",
        "routed.down_weight",
    }
    for relation in relations:
        expert_index = relation.input_indexing[0].indices[0]
        assert expert_index.kind is IndexExpressionKind.TENSOR_ELEMENT
        assert expert_index.tensor is selection.indices
    transports = tuple(operation for operation in program.operations if isinstance(operation, TransportPrimitive))
    assert all(transport.routing_inputs == (selection.indices,) for transport in transports)
    serialized = serialize_tensor_program(program)
    assert "moe" not in serialized.lower()
    assert deserialize_tensor_program(serialized).semantic_fingerprint == program.semantic_fingerprint


def test_executable_algebra_matches_the_ordinary_jax_forward() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, _ = _jax_arguments(config)
    program = expert_tensor_program(
        token_count=config.tokens,
        hidden_dimension=config.hidden,
        intermediate_dimension=config.intermediate,
        expert_count=config.experts,
        selected_count=config.top_k,
    )

    actual = execute_tensor_program(
        program,
        {value.name: np.asarray(argument) for value, argument in zip(program.inputs, inputs, strict=True)},
    )
    expected_output, expected_indices, expected_weights = moe_region(config)(*inputs)

    np.testing.assert_array_equal(actual["output"], expected_output)
    np.testing.assert_array_equal(actual["selection.indices"], expected_indices)
    np.testing.assert_allclose(actual["route.weight"], expected_weights, rtol=1e-6, atol=1e-7)


def test_executable_selection_matches_jax_source_index_tie_break() -> None:
    config = MoEDebugConfig(tokens=2, hidden=2, intermediate=2, experts=4, top_k=2)
    program = expert_tensor_program(
        token_count=config.tokens,
        hidden_dimension=config.hidden,
        intermediate_dimension=config.intermediate,
        expert_count=config.experts,
        selected_count=config.top_k,
    )
    inputs = {value.name: np.zeros(value.shape, dtype=np.float32) for value in program.inputs}

    actual = execute_tensor_program(program, inputs)
    expected = moe_region(config)(*(jnp.zeros(value.shape, dtype=jnp.bfloat16) for value in program.inputs))

    np.testing.assert_array_equal(actual["selection.indices"], expected[1])
    np.testing.assert_array_equal(actual["selection.indices"], np.asarray(((0, 1), (0, 1)), dtype=np.int32))


def test_route_plan_preserves_exact_dependencies_and_bounded_generations() -> None:
    expert_indices = np.asarray(
        ((0, 2), (1, 3), (3, 0), (2, 1)),
        dtype=np.int32,
    )
    route_weights = np.full((4, 2), 0.5, dtype=np.float32)
    plan = derive_expert_route_dataflow(
        expert_indices,
        route_weights,
        hidden_dimension=4,
        schedule=_schedule(),
    )

    for pipeline in (plan.forward, plan.reverse):
        assert all(
            event.scheduled_dependence.pairs == event.required_dependence.relation.pairs
            for event in pipeline.program.event_plans
        )
        assert all(
            max(owned.runtime_inputs.event_generations, default=0)
            < plan.transport.template.epoch_protocol.generation_modulus
            for owned in pipeline.owned_events
        )
        assert all(
            len(set(owned.runtime_inputs.event_storage_slots)) == len(owned.runtime_inputs.event_storage_slots)
            for owned in pipeline.owned_events
        )

    serialized = plan.serialize()
    assert "moe" not in serialized.lower()
    record = json.loads(serialized)
    assert record["schema"] == "shuttle.generic-routed-executable-plan.v1"
    assert record["task_fingerprint"] == plan.task_fingerprint
    assert record["event_fingerprint"] == plan.event_fingerprint
    assert record["transport_template"]["generation_count"] == 2
    assert len(plan.task_fingerprint) == len(plan.event_fingerprint) == 64

    mutated = derive_expert_route_dataflow(
        np.roll(expert_indices, 1, axis=1),
        route_weights,
        hidden_dimension=4,
        schedule=_schedule(),
    )
    assert mutated.task_fingerprint == plan.task_fingerprint
    assert mutated.event_fingerprint == plan.event_fingerprint
    assert mutated.fingerprint != plan.fingerprint


def test_physical_request_keeps_runtime_routes_out_of_compiler_identity() -> None:
    program = expert_tensor_program(
        token_count=4,
        hidden_dimension=4,
        intermediate_dimension=6,
        expert_count=4,
        selected_count=2,
    )
    request = expert_physical_request(program, schedule=_schedule())
    first_indices = np.asarray(((0, 2), (1, 3), (3, 0), (2, 1)), dtype=np.int32)
    second_indices = np.roll(first_indices, 1, axis=1)
    ordinal = request.task_plan.dependence_programs[0].runtime_bindings[0].algebra_value_ordinal

    first = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=request.task_plan,
            algebra_values={ordinal: first_indices},
        )
        for dependence in request.task_plan.dependence_programs
    )
    second = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=request.task_plan,
            algebra_values={ordinal: second_indices},
        )
        for dependence in request.task_plan.dependence_programs
    )

    assert first[0].relation.pairs != second[0].relation.pairs
    assert first[1].relation.pairs != second[1].relation.pairs
    assert request.fingerprint == request.semantic_fingerprint
    assert request.canonical_record["task_plan"]["dependence_programs"]
    assert "edge_source_item" not in request.canonical_json
    assert all(
        event.realization is ParametricEventRealization.IMPERATIVE_WAIT_NOTIFY for event in request.event_programs
    )

    event_plan, allocation = instantiate_parametric_event_tensor(
        request.event_programs[0],
        task_plan=request.task_plan,
        algebra_values={ordinal: first_indices},
        name="event.runtime_route_to_expert",
    )
    assert event_plan.required_dependence.relation.pairs == first[0].relation.pairs
    assert allocation.capacity == 4 * 4 * 2
    assert allocation.generation_count == _schedule().generation_count


def test_reference_expert_request_rejects_fast_policy_instead_of_relabeling_it() -> None:
    fast = expert_tensor_program(
        token_count=4,
        hidden_dimension=4,
        intermediate_dimension=6,
        expert_count=4,
        selected_count=2,
        numerical_policy=NumericalPolicy.FAST,
    )

    with pytest.raises(ValueError, match="reference expert schedule requires SOURCE_ORDERED numerical policy"):
        expert_physical_request(fast, schedule=_schedule())


def test_imported_jax_boundaries_lower_through_generic_tasks() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, cotangent = _jax_arguments(config)
    boundaries = (
        (moe_region(config), inputs),
        (
            moe_training_step(config),
            (*tuple(value / 2 for value in inputs), cotangent / 8),
        ),
        (
            moe_value_and_grad(config),
            (*tuple(value / 2 for value in inputs), cotangent / 8),
        ),
    )

    for function, arguments in boundaries:
        program = jax_function_to_tensor_program(function, *arguments)
        actual = execute_tensor_program(
            program,
            {value.name: np.asarray(argument) for value, argument in zip(program.inputs, arguments, strict=True)},
        )
        expected = jax.tree.leaves(function(*arguments))
        for value, reference in zip(program.outputs, expected, strict=True):
            np.testing.assert_allclose(actual[value.name], reference, rtol=2e-5, atol=2e-5)

        request = generic_tensor_program_physical_request(program, schedule=_schedule())
        assert request.algebra_fingerprint == program.semantic_fingerprint
        covered_ordinals = tuple(
            sorted(ordinal for task in request.task_plan.tasks for ordinal in task.operation_ordinals)
        )
        assert covered_ordinals == tuple(range(len(program.operations)))
        assert any(len(task.operation_ordinals) > 1 for task in request.task_plan.tasks)
        assert len(request.event_plans) == len(request.task_plan.dependences)
        assert all(
            event.required_dependence.relation == dependence.relation
            for event, dependence in zip(request.event_plans, request.task_plan.dependences, strict=True)
        )
        assert "moe" not in request.canonical_json.lower()


def test_jax_owned_reverse_boundaries_generate_complete_sm90_programs() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, cotangent = _jax_arguments(config)

    for function in (moe_training_step(config), moe_value_and_grad(config)):
        program = jax_function_to_tensor_program(function, *inputs, cotangent)
        request = generic_tensor_program_physical_request(program, schedule=_schedule())
        generated = generate_sm90_relation_simt(request)

        assert tuple(launch.operation_ordinal for launch in generated.launches) == tuple(range(len(program.operations)))
        assert tuple(launch.task_id for launch in generated.launches) == tuple(
            task.id
            for operation_ordinal in range(len(program.operations))
            for task in request.task_plan.tasks
            if operation_ordinal in task.operation_ordinals
        )
        assert len(generated.runtime_event_records) == len(request.event_programs)


def test_jax_owned_reverse_boundaries_fuse_dependency_convex_components() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, cotangent = _jax_arguments(config)

    for function in (moe_training_step(config), moe_value_and_grad(config)):
        program = jax_function_to_tensor_program(function, *inputs, cotangent)
        request = generic_tensor_program_physical_request(program, schedule=_schedule())
        composition = lower_canonical_kernel_composition(request)
        generated = generate_sm90_dependency_convex_simt(request)
        covered_operations = tuple(
            operation_ordinal for launch in generated.launches for operation_ordinal in launch.operation_ordinals
        )

        assert tuple(sorted(covered_operations)) == tuple(range(len(program.operations)))
        assert len(covered_operations) == len(set(covered_operations))
        assert len(generated.launches) < len(program.operations)
        assert any(launch.fused for launch in generated.launches)
        assert all(
            not launch.fused
            for launch in generated.launches
            if isinstance(program.operations[launch.operation_ordinals[0]], (RelationPrimitive, SelectionPrimitive))
        )
        assert generated.request_fingerprint == request.fingerprint
        assert generated.composition_fingerprint == composition.fingerprint
        assert not composition.scheduled_request.event_realizations.audit.physical
        assert len(generated.runtime_event_records) == len(request.event_programs)
        canonical_events = cast(tuple[dict[str, Any], ...], request.canonical_record["event_programs"])
        assert all(
            record["event"] == canonical_events[index] for index, record in enumerate(generated.runtime_event_records)
        )
        assert "shuttle_sm90_dependency_convex_simt_run" in generated.source


def test_imported_jax_routing_stays_runtime_parametric_through_task_events() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    inputs, cotangent = _jax_arguments(config)
    forward = jax_function_to_tensor_program(moe_region(config), *inputs)
    training = jax_function_to_tensor_program(
        moe_training_step(config),
        *tuple(value / 2 for value in inputs),
        cotangent / 8,
    )
    forward_request = generic_tensor_program_physical_request(forward, schedule=_schedule())
    training_request = generic_tensor_program_physical_request(training, schedule=_schedule())
    generated_forward = generate_sm90_relation_simt(forward_request)
    assert tuple(launch.operation_ordinal for launch in generated_forward.launches) == tuple(
        range(len(forward.operations))
    )
    assert len(generated_forward.runtime_event_records) == len(forward_request.event_programs)
    assert "shuttle_sm90_relation_simt_run" in generated_forward.source

    dynamic_gathers = tuple(
        operation
        for operation in forward.operations
        if isinstance(operation, RelationPrimitive)
        and any(
            index.kind is IndexExpressionKind.TENSOR_ELEMENT
            for indexing in operation.input_indexing
            for index in indexing.indices
        )
    )
    assert dynamic_gathers
    forward_dependences = forward_request.task_plan.dependence_programs
    assert forward_dependences
    assert len(forward_request.event_programs) == len(forward_dependences)
    input_stage_count = sum(
        task.structure.principal_kind is TaskOperationKind.INPUT_STAGE for task in forward_request.task_plan.tasks
    )
    assert input_stage_count > 0
    assert input_stage_count <= len(forward_dependences)
    assert all(dependence.validity is None for dependence in forward_dependences)

    dependence_programs = training_request.task_plan.dependence_programs
    assert dependence_programs
    assert len(training_request.event_programs) == len(dependence_programs)
    assert any(dependence.validity is None for dependence in dependence_programs)
    assert any(dependence.validity is not None for dependence in dependence_programs)
    assert all(
        coordinate.lower_bound == 0 and coordinate.upper_bound > coordinate.lower_bound
        for dependence in dependence_programs
        for coordinate in (*dependence.source_coordinates, *dependence.target_coordinates)
    )

    first_routes = np.zeros((config.tokens, config.top_k), dtype=np.int32)
    second_routes = np.full((config.tokens, config.top_k), config.experts - 1, dtype=np.int32)

    def runtime_values(
        dependences,
        routes: np.ndarray,
    ) -> dict[int, np.ndarray]:
        return {
            binding.algebra_value_ordinal: np.broadcast_to(routes[..., None], binding.input.shape)
            for dependence in dependences
            for binding in dependence.runtime_bindings
        }

    forward_first_values = runtime_values(forward_dependences, first_routes)
    forward_second_values = runtime_values(forward_dependences, second_routes)
    forward_first = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=forward_request.task_plan,
            algebra_values=forward_first_values,
        )
        for dependence in forward_dependences
    )
    forward_second = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=forward_request.task_plan,
            algebra_values=forward_second_values,
        )
        for dependence in forward_dependences
    )
    assert all(
        left.relation.pairs != right.relation.pairs for left, right in zip(forward_first, forward_second, strict=True)
    )
    for index, (dependence, event) in enumerate(zip(forward_first, forward_request.event_programs, strict=True)):
        event_plan, allocation = instantiate_parametric_event_tensor(
            event,
            task_plan=forward_request.task_plan,
            algebra_values=forward_first_values,
            name=f"event.runtime_gather.{index}",
        )
        assert event_plan.required_dependence.relation == dependence.relation
        assert allocation.capacity == len(
            forward_request.task_plan.tasks[forward_dependences[index].target_task_id].family.coordinates
        )

    first_values = runtime_values(dependence_programs, first_routes)
    second_values = runtime_values(dependence_programs, second_routes)
    first = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=training_request.task_plan,
            algebra_values=first_values,
        )
        for dependence in dependence_programs
    )
    second = tuple(
        instantiate_task_dependence(
            dependence,
            task_plan=training_request.task_plan,
            algebra_values=second_values,
        )
        for dependence in dependence_programs
    )
    assert any(left.relation.pairs != right.relation.pairs for left, right in zip(first, second, strict=True))

    for index, (dependence, event) in enumerate(zip(first, training_request.event_programs, strict=True)):
        event_plan, allocation = instantiate_parametric_event_tensor(
            event,
            task_plan=training_request.task_plan,
            algebra_values=first_values,
            name=f"event.runtime_relation.{index}",
        )
        assert event_plan.required_dependence.relation == dependence.relation
        assert allocation.capacity == len(
            training_request.task_plan.tasks[dependence_programs[index].target_task_id].family.coordinates
        )

    reimported = jax_function_to_tensor_program(
        moe_training_step(config),
        *tuple(value / 3 for value in inputs),
        cotangent / 5,
    )
    assert generic_tensor_program_physical_request(reimported, schedule=_schedule()).fingerprint == (
        training_request.fingerprint
    )
    reimported_forward = jax_function_to_tensor_program(moe_region(config), *tuple(value / 7 for value in inputs))
    reimported_forward_request = generic_tensor_program_physical_request(reimported_forward, schedule=_schedule())
    assert reimported_forward_request.fingerprint == forward_request.fingerprint
    assert generate_sm90_relation_simt(reimported_forward_request).fingerprint == generated_forward.fingerprint
    assert "edge_destination_item" not in training_request.canonical_json
