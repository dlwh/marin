# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.dense_flow_planner import (
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.event_dataflow import EventGenerationPolicy, verify_event_dataflow_program
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.msa_frontend import (
    MSA_INPUT_NAMES,
    MSADebugConfig,
    msa_region,
    msa_selection,
    msa_training_step,
    msa_value_and_grad,
)
from tile_lifetime.msa_reference_pipeline import (
    build_msa_reference_oracle,
    derive_msa_reference_dataflow,
    execute_msa_reference_oracle,
    msa_reference_physical_request,
)
from tile_lifetime.physical_task_contract import (
    PhysicalNumericalPolicy,
    PhysicalTarget,
    RelationAccessClass,
    TaskTraversal,
    instantiate_parametric_event_tensor,
    instantiate_task_dependence,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    MapPrimitive,
    RelationPrimitive,
    SelectionComparison,
    SelectionPrimitive,
    SelectionTieBreak,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    selection_cutoff_ties,
    serialize_tensor_program,
    tensor_program_value_ordinals,
)


def _inputs(config: MSADebugConfig, *, seed: int) -> tuple[np.ndarray, ...]:
    keys = jax.random.split(jax.random.key(seed), len(MSA_INPUT_NAMES))
    shapes = (
        (config.query_length, config.hidden_dimension),
        (config.key_value_length, config.hidden_dimension),
        (config.hidden_dimension, config.query_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.index_dimension),
        (config.hidden_dimension, config.index_dimension),
    )
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    return tuple(
        np.asarray(jax.random.normal(key, shape, dtype=dtype), dtype=np.float32)
        for key, shape, dtype in zip(keys, shapes, dtypes, strict=True)
    )


def test_msa_reference_oracle_matches_ordinary_jax_forward_and_routes() -> None:
    config = MSADebugConfig()
    inputs = _inputs(config, seed=37)
    oracle = build_msa_reference_oracle(config)

    execution = execute_msa_reference_oracle(oracle, inputs)
    ordinary_jax = np.asarray(
        msa_region(config)(
            *(
                jnp.asarray(value, dtype=dtype)
                for value, dtype in zip(
                    inputs,
                    (jnp.bfloat16,) * 5 + (jnp.float32,) * 2,
                    strict=True,
                )
            )
        ),
        dtype=np.float32,
    )

    error = np.abs(ordinary_jax - execution.output)
    assert float(np.max(error)) <= 0.0625
    assert float(np.mean(error)) <= 0.005
    local_block = (np.arange(config.query_length) + config.query_position_offset) // config.block_size
    assert np.all(
        np.any(
            (execution.selection.indices == local_block[:, None, None]) & execution.selection.valid,
            axis=-1,
        )
    )
    assert execution.relation.route_count == int(np.count_nonzero(execution.selection.valid))
    canonical_output = np.asarray(
        execute_tensor_program(
            oracle.canonical_program,
            dict(zip(MSA_INPUT_NAMES, inputs, strict=True)),
        )["output"],
        dtype=np.float32,
    )
    canonical_error = np.abs(ordinary_jax - canonical_output)
    assert float(np.max(canonical_error)) <= 0.0625
    assert float(np.mean(canonical_error)) <= 0.005
    generic_kinds = {"Contract", "Map", "Fold", "Selection", "Relation", "RelationPredicate"}
    assert all(kind in generic_kinds for kind in oracle.operation_kinds)


def test_msa_canonical_forward_is_generic_serializable_algebra() -> None:
    config = MSADebugConfig()
    program = build_msa_reference_oracle(config).canonical_program

    restored = deserialize_tensor_program(serialize_tensor_program(program))

    assert restored.canonical_record == program.canonical_record
    assert restored.semantic_fingerprint == program.semantic_fingerprint
    assert tuple(value.shape for value in program.inputs) == (
        (config.query_length, config.hidden_dimension),
        (config.key_value_length, config.hidden_dimension),
        (config.hidden_dimension, config.query_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.head_dimension),
        (config.hidden_dimension, config.key_value_heads * config.index_dimension),
        (config.hidden_dimension, config.index_dimension),
    )
    selection = next(operation for operation in program.operations if isinstance(operation, SelectionPrimitive))
    assert selection.comparison is SelectionComparison.TOTAL_ORDER
    assert selection.stable
    assert selection.tie_break is SelectionTieBreak.SOURCE_INDEX_ASCENDING
    source_generated = tuple(
        operation
        for operation in program.operations
        if isinstance(operation, RelationPrimitive) and not operation.inputs
    )
    assert len(source_generated) == 3
    canonical_text = serialize_tensor_program(program).lower()
    assert all(token not in canonical_text for token in ("msa", "attention", "causal", "gqa", "mqa"))


def test_msa_reference_dataflow_uses_route_independent_parametric_dependencies() -> None:
    config = MSADebugConfig(
        query_length=4,
        key_value_length=4,
        hidden_dimension=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=2,
        index_dimension=2,
        block_size=1,
        selected_blocks=4,
    )
    oracle = build_msa_reference_oracle(config)
    execution = execute_msa_reference_oracle(oracle, _inputs(config, seed=71))
    other_execution = execute_msa_reference_oracle(oracle, _inputs(config, seed=72))

    dataflow = derive_msa_reference_dataflow(oracle)
    rebuilt_dataflow = derive_msa_reference_dataflow(oracle)

    verify_event_dataflow_program(dataflow.program)
    assert not np.array_equal(execution.selection.indices, other_execution.selection.indices)
    assert dataflow.program == rebuilt_dataflow.program
    selection = next(
        operation for operation in oracle.canonical_program.operations if isinstance(operation, SelectionPrimitive)
    )
    assert selection.output_validity is not None
    value_ordinals = tensor_program_value_ordinals(oracle.canonical_program)

    def exact_key_dependence(route_execution):
        return instantiate_task_dependence(
            dataflow.task_plan.dependence_programs[0],
            task_plan=dataflow.task_plan,
            algebra_values={
                value_ordinals[selection.indices]: route_execution.selection.indices,
                value_ordinals[selection.output_validity]: route_execution.selection.valid,
            },
        )

    exact = exact_key_dependence(execution)
    other_exact = exact_key_dependence(other_execution)
    expected_pairs = {
        ((int(execution.selection.indices[q, group, slot]), group), (q, group, local_head, slot))
        for q, group, slot in np.argwhere(execution.selection.valid)
        for local_head in range(config.query_heads // config.key_value_heads)
    }
    assert {(pair.source, pair.target) for pair in exact.relation.pairs} == expected_pairs
    assert exact.relation.pairs != other_exact.relation.pairs
    assert all(
        runtime.event_generations == (0,) * len(runtime.event_generations) for runtime in dataflow.event_runtime_inputs
    )
    assert dataflow.event_storage_capacity == sum(
        len(set(runtime.event_storage_slots)) for runtime in dataflow.event_runtime_inputs
    )
    assert all("msa" not in family.name.lower() for family in dataflow.program.task_families)
    score_family = next(family for family in dataflow.program.task_families if family.name == "score_contract")
    assert tuple(axis.name for axis in score_family.axes) == ("query", "group", "local_head", "selected_slot")

    selection_task = next(task for task in dataflow.task_plan.tasks if task.family.name == "bounded_selection")
    assert selection_task.structure.owner_axes[-1].extent == config.selected_blocks
    score_task = next(task for task in dataflow.task_plan.tasks if task.family.name == "score_contract")
    group_relation_index = next(
        index
        for index, operation in enumerate(oracle.canonical_program.operations)
        if isinstance(operation, RelationPrimitive) and operation.name == "group_query_relation"
    )
    expected_group_relation = oracle.canonical_program.canonical_record["operations"][group_relation_index]
    assert any(access.relation_record["algebra_operation"] == expected_group_relation for access in score_task.accesses)
    weighted_task = next(task for task in dataflow.task_plan.tasks if task.family.name == "weighted_contract")
    assert weighted_task.structure.owner_axes[-1].extent == config.head_dimension


def test_msa_physical_request_is_canonical_and_input_independent() -> None:
    config = MSADebugConfig()
    oracle = build_msa_reference_oracle(config)
    dataflow = derive_msa_reference_dataflow(oracle)

    first = msa_reference_physical_request(oracle, dataflow)
    second = msa_reference_physical_request(build_msa_reference_oracle(config), derive_msa_reference_dataflow(oracle))
    changed_oracle = build_msa_reference_oracle(MSADebugConfig(query_length=4, key_value_length=16))
    changed = msa_reference_physical_request(changed_oracle, derive_msa_reference_dataflow(changed_oracle))

    assert json.loads(first.canonical_json)["algebra_fingerprint"] == oracle.canonical_program.semantic_fingerprint
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.task_plan.fingerprint != changed.task_plan.fingerprint
    assert len(first.schedules) == len(first.task_plan.tasks)
    assert len(first.event_allocations) == len(first.event_plans)
    assert all(
        all(generation == 0 for generation in allocation.storage.generations) for allocation in first.event_allocations
    )
    assert all(
        all(slot < allocation.capacity for slot in allocation.storage.slots) for allocation in first.event_allocations
    )
    assert len(first.event_realizations.audit.physical) == len(first.event_plans)
    assert not first.event_realizations.audit.erased


def test_msa_physical_policy_changes_task_schedule_and_artifact_identity() -> None:
    oracle = build_msa_reference_oracle(MSADebugConfig())
    source_dataflow = derive_msa_reference_dataflow(oracle)
    source = msa_reference_physical_request(oracle, source_dataflow)
    fast_dataflow = derive_msa_reference_dataflow(
        oracle,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    fast = msa_reference_physical_request(
        oracle,
        fast_dataflow,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    assert all(task.structure.traversal is TaskTraversal.SOURCE_ORDERED for task in source.task_plan.tasks)
    assert all(task.structure.traversal is TaskTraversal.UNORDERED for task in fast.task_plan.tasks)
    assert source.task_plan.fingerprint != fast.task_plan.fingerprint
    assert tuple(schedule.fingerprint for schedule in source.schedules) != tuple(
        schedule.fingerprint for schedule in fast.schedules
    )
    assert source.fingerprint != fast.fingerprint
    assert source.artifact_fingerprint != fast.artifact_fingerprint

    with pytest.raises(ValueError, match="policy does not match its derived task plan"):
        msa_reference_physical_request(
            oracle,
            source_dataflow,
            numerical_policy=PhysicalNumericalPolicy.FAST,
        )


@pytest.mark.parametrize(
    ("factory", "expected_output_count"),
    (
        (msa_region, 1),
        (msa_value_and_grad, 8),
        (msa_training_step, 8),
    ),
)
def test_generic_jax_import_executes_and_lowers_msa_boundary(
    factory,
    expected_output_count,
) -> None:
    config = MSADebugConfig(
        query_length=4,
        key_value_length=4,
        hidden_dimension=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=2,
        index_dimension=2,
        block_size=1,
        selected_blocks=2,
    )
    arguments = tuple(
        jnp.asarray(value, dtype=dtype)
        for value, dtype in zip(_inputs(config, seed=89), (jnp.bfloat16,) * 5 + (jnp.float32,) * 2, strict=True)
    )
    if factory is not msa_region:
        output_cotangent = jnp.ones(
            (config.query_length, config.query_heads, config.head_dimension),
            jnp.bfloat16,
        )
        arguments = (*arguments, output_cotangent)

    function = factory(config)
    program = jax_function_to_tensor_program(function, *arguments)
    imported_results = execute_tensor_program(
        program,
        {value.name: np.asarray(argument) for value, argument in zip(program.inputs, arguments, strict=True)},
    )
    actual = tuple(imported_results[value.name] for value in program.outputs)
    expected = tuple(np.asarray(value) for value in jax.tree.leaves(function(*arguments)))

    assert len(actual) == len(expected) == expected_output_count
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)

    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={},
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    event_program = eventize_composed_task_plan(task_plan)
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=event_program,
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    assert any(
        access.relation_class is RelationAccessClass.BOUNDED_DYNAMIC
        for task in task_plan.tasks
        for access in task.accesses
    )

    verify_event_dataflow_program(event_program)
    assert any(isinstance(operation, ContractPrimitive) for operation in program.operations)
    assert any(isinstance(operation, FoldPrimitive) for operation in program.operations)
    assert any(isinstance(operation, MapPrimitive) for operation in program.operations)
    assert any(isinstance(operation, RelationPrimitive) for operation in program.operations)
    selection = next(operation for operation in program.operations if isinstance(operation, SelectionPrimitive))
    assert selection.comparison is SelectionComparison.TOTAL_ORDER
    assert selection.stable
    assert selection.tie_break is SelectionTieBreak.SOURCE_INDEX_ASCENDING
    assert request.algebra_fingerprint == program.semantic_fingerprint
    assert request.task_plan == task_plan
    assert len(request.schedules) == len(task_plan.tasks)
    assert len(request.event_plans) == len(request.event_allocations) == len(event_program.event_plans)
    assert task_plan.dependence_programs
    assert len(task_plan.dependence_programs) == len(request.event_programs)
    assert all(
        tuple(type(operation).__name__ for operation in dependence.relation_program.operations) == ("RelationPrimitive",)
        for dependence in task_plan.dependence_programs
    )
    assert all(
        all(generation == 0 for generation in allocation.storage.generations) for allocation in request.event_allocations
    )
    assert all(
        all(slot < allocation.capacity for slot in allocation.storage.slots) for allocation in request.event_allocations
    )
    assert json.loads(request.canonical_json)["algebra_fingerprint"] == program.semantic_fingerprint
    assert all(token not in request.canonical_json.lower() for token in ("msa", "attention", "workload"))


def test_msa_exact_cutoff_ties_preserve_jax_routes_and_report_split_equivalence_classes() -> None:
    config = MSADebugConfig(
        query_length=4,
        key_value_length=4,
        hidden_dimension=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=2,
        index_dimension=2,
        block_size=1,
        selected_blocks=2,
    )
    inputs = list(_inputs(config, seed=113))
    inputs[5] = np.zeros_like(inputs[5])
    inputs[6] = np.zeros_like(inputs[6])
    ranked_scores, jax_indices, jax_validity = msa_selection(config)(
        jnp.asarray(inputs[0], dtype=jnp.bfloat16),
        jnp.asarray(inputs[1], dtype=jnp.bfloat16),
        jnp.asarray(inputs[5], dtype=jnp.float32),
        jnp.asarray(inputs[6], dtype=jnp.float32),
    )
    ranked_scores = np.asarray(ranked_scores)
    jax_indices = np.asarray(jax_indices, dtype=np.int32)
    jax_validity = np.asarray(jax_validity, dtype=np.bool_)

    canonical = build_msa_reference_oracle(config).canonical_program
    selection = next(operation for operation in canonical.operations if isinstance(operation, SelectionPrimitive))
    assert selection.output_validity is not None
    diagnostic_program = TensorProgram(
        canonical.inputs,
        canonical.operations,
        (selection.score, selection.indices, selection.output_validity),
        canonical.numerical_policy,
    )
    diagnostic = execute_tensor_program(
        diagnostic_program,
        dict(zip(MSA_INPUT_NAMES, inputs, strict=True)),
    )
    canonical_indices = diagnostic[selection.indices.name]
    canonical_validity = diagnostic[selection.output_validity.name]
    np.testing.assert_array_equal(diagnostic[selection.score.name], ranked_scores)
    np.testing.assert_array_equal(canonical_validity, jax_validity)
    np.testing.assert_array_equal(
        canonical_indices,
        np.where(jax_validity, jax_indices, selection.invalid_index),
    )

    local_block = np.arange(config.query_length, dtype=np.int32)
    legal = np.arange(config.block_count)[None, None, :] <= local_block[:, None, None]
    reports = selection_cutoff_ties(
        selection,
        ranked_scores,
        jax_indices,
        validity=legal,
        selected_validity=jax_validity,
    )
    assert tuple(report.result_coordinate for report in reports) == ((2, 0), (3, 0))
    assert tuple(report.selected_source_indices for report in reports) == ((0,), (0,))
    assert tuple(report.excluded_source_indices for report in reports) == ((1,), (1, 2))


@pytest.mark.parametrize("factory", (msa_region, msa_value_and_grad, msa_training_step))
def test_generic_msa_gather_events_instantiate_exact_route_dependent_edges(factory) -> None:
    config = MSADebugConfig(
        query_length=4,
        key_value_length=4,
        hidden_dimension=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=2,
        index_dimension=2,
        block_size=1,
        selected_blocks=2,
    )
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    first_arguments = tuple(
        jnp.asarray(value, dtype=dtype) for value, dtype in zip(_inputs(config, seed=127), dtypes, strict=True)
    )
    second_arguments = tuple(
        jnp.asarray(value, dtype=dtype) for value, dtype in zip(_inputs(config, seed=131), dtypes, strict=True)
    )
    if factory is not msa_region:
        first_arguments = (
            *first_arguments,
            jnp.ones((config.query_length, config.query_heads, config.head_dimension), dtype=jnp.bfloat16),
        )
        second_arguments = (
            *second_arguments,
            -jnp.ones((config.query_length, config.query_heads, config.head_dimension), dtype=jnp.bfloat16),
        )
    program = jax_function_to_tensor_program(factory(config), *first_arguments)
    all_values = tuple(tensor_program_value_ordinals(program))
    tile_extent_by_axis = {axis.id: 1 for value in all_values for axis in value.axes}
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extent_by_axis,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    event_program = eventize_composed_task_plan(task_plan)
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=event_program,
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    value_ordinals = tensor_program_value_ordinals(program)
    value_by_ordinal = {ordinal: value for value, ordinal in value_ordinals.items()}
    runtime_values = tuple(
        dict.fromkeys(
            value_by_ordinal[binding.algebra_value_ordinal]
            for dependence in task_plan.dependence_programs
            for binding in dependence.runtime_bindings
        )
    )
    diagnostic_program = TensorProgram(
        program.inputs,
        program.operations,
        runtime_values,
        program.numerical_policy,
    )

    def instantiated(arguments):
        results = execute_tensor_program(
            diagnostic_program,
            {value.name: np.asarray(argument) for value, argument in zip(program.inputs, arguments, strict=True)},
        )
        algebra_values = {value_ordinals[value]: results[value.name] for value in runtime_values}
        dependences = tuple(
            instantiate_task_dependence(dependence, task_plan=task_plan, algebra_values=algebra_values)
            for dependence in task_plan.dependence_programs
        )
        events = tuple(
            instantiate_parametric_event_tensor(
                event,
                task_plan=task_plan,
                algebra_values=algebra_values,
                name=f"runtime.{index}",
            )
            for index, event in enumerate(request.event_programs)
        )
        return dependences, events

    first_dependences, first_events = instantiated(first_arguments)
    second_dependences, second_events = instantiated(second_arguments)
    assert tuple(dependence.relation.pairs for dependence in first_dependences) != tuple(
        dependence.relation.pairs for dependence in second_dependences
    )
    assert all(dependence.relation.pairs for dependence in first_dependences)
    assert all(len(event.domain.coordinates) == allocation.capacity for event, allocation in first_events)
    assert all(len(event.domain.coordinates) == allocation.capacity for event, allocation in second_events)

    first_event = request.event_programs[0]
    with pytest.raises(ValueError, match="storage must cover every consumer task"):
        replace(
            request,
            event_programs=(
                replace(
                    first_event,
                    capacity=first_event.capacity - 1,
                    slots=first_event.slots[:-1],
                    generations=first_event.generations[:-1],
                ),
                *request.event_programs[1:],
            ),
        )
    phased = replace(
        first_event,
        generations=(1,) * len(first_event.generations),
        generation_count=2,
        generation_policy=EventGenerationPolicy.PHASED,
    )
    generation_mutation = replace(request, event_programs=(phased, *request.event_programs[1:]))
    assert generation_mutation.fingerprint != request.fingerprint
