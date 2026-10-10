# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.backends.h100.triton_ordered_state import (
    generate_cuda_ordered_state_artifact,
    ordered_state_scan_component,
)
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.dense_flow_planner import derive_composed_task_plan
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.event_dataflow import ImperativeEventOpKind
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import (
    PhysicalNumericalPolicy,
    PhysicalTarget,
    TaskOperationKind,
    TaskTraversal,
    TaskValueAccessKind,
)
from tile_lifetime.stateful_scan_reference import (
    NATURAL_AFFINE_SCAN_INPUT_NAMES,
    NaturalAffineScanConfig,
    ScanDecayAxes,
    ScanDiagonalOperation,
    export_natural_affine_scan_training_step_text,
    export_natural_affine_scan_value_and_grad_text,
    natural_affine_scan_input_specifications,
    natural_affine_scan_program,
    natural_affine_scan_region,
    natural_affine_scan_training_step,
    natural_affine_scan_value_and_grad,
)
from tile_lifetime.stateful_scan_task_pipeline import (
    attach_chunked_scan_physical_request,
    attach_whole_program_scan_physical_request,
    compile_natural_affine_scan_reference_pipeline,
    derive_chunked_scan_reference_plan,
    derive_chunked_scan_tiling,
    extract_scan_state_region,
    refine_whole_program_scan_tasks,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    ScanDirection,
    ScanPrimitive,
    TensorProgram,
    execute_tensor_program,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_input,
    tensor_program_value_ordinals,
)


def _linear_matrix_scan(matrix, initial_state):
    def step(state, matrix_slice):
        next_state = jnp.einsum(
            "bhij,bhjv->bhiv",
            matrix_slice,
            state,
            preferred_element_type=jnp.float32,
        )
        return next_state, next_state

    return jax.lax.scan(step, initial_state, jnp.swapaxes(matrix, 0, 1))


def _linear_matrix_scan_training_step(matrix, initial_state, output_cotangent, state_cotangent):
    result, pullback = jax.vjp(_linear_matrix_scan, matrix, initial_state)
    return (*result, *pullback((state_cotangent, output_cotangent)))


def _inputs(config: NaturalAffineScanConfig) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(83)
    specifications = natural_affine_scan_input_specifications(config)
    values = []
    for index, specification in enumerate(specifications):
        value = rng.normal(size=specification.shape).astype(np.float32) * 0.15
        if index == 3:
            value = -np.abs(value)
        elif index == 4:
            value = 0.5 + value
        values.append(jnp.asarray(value, dtype=specification.dtype))
    return tuple(values)


def _direct_recurrence(config: NaturalAffineScanConfig, inputs: tuple[jax.Array, ...]) -> tuple[np.ndarray, np.ndarray]:
    query, key, value, log_decay, beta, initial_state = (np.asarray(item, dtype=np.float32) for item in inputs)
    state = initial_state.copy()
    outputs = []
    for position in range(config.sequence):
        decay = np.exp(log_decay[:, position], dtype=np.float32)
        if config.decay_axes is ScanDecayAxes.SCALAR:
            decay = decay[..., None]
        state *= decay[..., None]
        prediction = np.einsum("bhkv,bhrk->bhrv", state, key[:, position], optimize=False)
        delta = beta[:, position, :, :, None] * (value[:, position] - prediction)
        state += np.einsum("bhrk,bhrv->bhkv", key[:, position], delta, optimize=False)
        outputs.append(np.einsum("bhkv,bhk->bhv", state, query[:, position], optimize=False))
    return np.stack(outputs, axis=1), state


def test_natural_affine_scan_forward_matches_independent_recurrence() -> None:
    config = NaturalAffineScanConfig(
        batch=1,
        sequence=4,
        heads=2,
        key_dimension=3,
        value_dimension=5,
        update_rank=2,
        decay_axes=ScanDecayAxes.KEY,
    )
    inputs = _inputs(config)

    output, final_state = jax.jit(natural_affine_scan_region(config))(*inputs)
    expected_output, expected_state = _direct_recurrence(config, inputs)
    algebra_result = execute_tensor_program(
        natural_affine_scan_program(config),
        dict(zip(NATURAL_AFFINE_SCAN_INPUT_NAMES, inputs, strict=True)),
    )

    np.testing.assert_allclose(np.asarray(output), expected_output, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(final_state), expected_state, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(algebra_result["output"], expected_output, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(algebra_result["final_state"], expected_state, rtol=2e-5, atol=2e-5)


def test_ordinary_jax_scan_import_executes_and_retains_exact_chunk_events() -> None:
    config = NaturalAffineScanConfig(
        batch=1,
        sequence=4,
        heads=1,
        key_dimension=2,
        value_dimension=3,
        update_rank=1,
    )
    inputs = _inputs(config)
    function = natural_affine_scan_region(config)

    program = jax_function_to_tensor_program(function, *inputs)
    actual = execute_tensor_program(
        program,
        {value.name: np.asarray(argument) for value, argument in zip(program.inputs, inputs, strict=True)},
    )
    expected = jax.tree.leaves(function(*inputs))
    for result, reference in zip(actual.values(), expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)

    plan = derive_chunked_scan_reference_plan(program, chunk_extent=2, state_column_tile=2)
    carried_state = plan.task_plan.bounded_buffers[0]
    assert carried_state.capacity == 2
    for event in plan.event_plans:
        assert set(event.scheduled_dependence.pairs) == set(event.required_dependence.relation.pairs)
    for event, operations in zip(plan.event_plans, plan.imperative_events, strict=True):
        assert sum(operation.kind is ImperativeEventOpKind.WAIT for operation in operations) == len(
            event.trigger_relation.pairs
        )
        assert sum(operation.kind is ImperativeEventOpKind.NOTIFY for operation in operations) == len(
            event.notify_relation.pairs
        )


def test_ordinary_jax_scan_reverse_boundary_imports_without_custom_derivative() -> None:
    config = NaturalAffineScanConfig(sequence=3, heads=1, key_dimension=2, value_dimension=3)
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    output_cotangent = jnp.ones_like(output)
    final_state_cotangent = jnp.ones_like(final_state)
    arguments = (*inputs, output_cotangent, final_state_cotangent)
    function = natural_affine_scan_training_step(config)

    program = jax_function_to_tensor_program(function, *arguments)
    actual = execute_tensor_program(
        program,
        {value.name: np.asarray(argument) for value, argument in zip(program.inputs, arguments, strict=True)},
    )
    expected = jax.tree.leaves(function(*arguments))
    scans = tuple(operation for operation in program.operations if isinstance(operation, ScanPrimitive))

    assert tuple(scan.direction for scan in scans) == (ScanDirection.FORWARD, ScanDirection.REVERSE)
    for result, reference in zip(actual.values(), expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)


def test_jax_owned_reverse_scan_state_slice_lowers_to_generic_ordered_cuda() -> None:
    rng = np.random.default_rng(97)
    matrix = jnp.asarray(rng.normal(size=(1, 4, 1, 2, 2)), dtype=jnp.float32)
    initial_state = jnp.asarray(rng.normal(size=(1, 1, 2, 3)), dtype=jnp.float32)
    output_cotangent = jnp.asarray(rng.normal(size=(4, 1, 1, 2, 3)), dtype=jnp.float32)
    state_cotangent = jnp.asarray(rng.normal(size=(1, 1, 2, 3)), dtype=jnp.float32)
    imported = jax_function_to_tensor_program(
        _linear_matrix_scan_training_step,
        matrix,
        initial_state,
        output_cotangent,
        state_cotangent,
    )
    reverse_ordinal = next(
        ordinal
        for ordinal, operation in enumerate(imported.operations)
        if isinstance(operation, ScanPrimitive) and operation.direction is ScanDirection.REVERSE
    )
    reverse = extract_scan_state_region(imported, reverse_ordinal)
    scan = reverse.operations[0]
    assert isinstance(scan, ScanPrimitive)
    assert scan.direction is ScanDirection.REVERSE
    assert tuple(type(operation) for operation in scan.body.operations) == (MapPrimitive, ContractPrimitive)
    assert scan.emitted_outputs == ()

    region_inputs = []
    for value in reverse.inputs:
        region_inputs.append(rng.normal(size=value.shape).astype(np.float32) * 0.1)
    actual = next(
        iter(
            execute_tensor_program(
                reverse, dict(zip((value.name for value in reverse.inputs), region_inputs, strict=True))
            ).values()
        )
    )
    expected = region_inputs[0].copy()
    dy = region_inputs[1]
    matrices = region_inputs[-1]
    for position in range(scan.axis.extent - 1, -1, -1):
        expected = np.einsum(
            "bhij,bhiv->bhjv",
            matrices[position],
            expected + dy[position],
            optimize=False,
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    plan = derive_chunked_scan_reference_plan(reverse, chunk_extent=2, state_column_tile=3)
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=reverse.canonical_record,
        algebra_fingerprint=reverse.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={
            "automatic_differentiation": "jax",
            "source_fingerprint": imported.semantic_fingerprint,
            "source_operation_ordinal": reverse_ordinal,
        },
    )
    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(reverse, 0), request)

    assert plan.task_plan.tasks[0].structure.traversal is TaskTraversal.REVERSE
    assert len(plan.task_plan.tasks[0].structure.contraction_axes) == 1
    assert "int t = T - 1 - iteration;" in artifact.cuda_source
    assert request.provenance["automatic_differentiation"] == "jax"


def test_imported_jax_forward_scan_emits_relation_rich_saved_state_boundary() -> None:
    config = NaturalAffineScanConfig(sequence=4, heads=1, key_dimension=2, value_dimension=3)
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    imported = jax_function_to_tensor_program(
        natural_affine_scan_training_step(config),
        *inputs,
        jnp.ones_like(output),
        jnp.ones_like(final_state),
    )
    scan = next(
        operation
        for operation in imported.operations
        if isinstance(operation, ScanPrimitive) and operation.direction is ScanDirection.FORWARD
    )
    region = TensorProgram(
        primitive_inputs(scan),
        (scan,),
        primitive_outputs(scan),
        imported.numerical_policy,
    )
    plan = derive_chunked_scan_reference_plan(region, chunk_extent=2, state_column_tile=config.value_dimension)
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=region.canonical_record,
        algebra_fingerprint=region.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "ordinary_jax", "automatic_differentiation": "jax"},
    )

    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(region, 0), request)

    assert artifact.operand_names == tuple(value.name for value in primitive_inputs(scan))
    assert artifact.result_names == tuple(value.name for value in primitive_outputs(scan))
    assert "float state_values[K * 1]" in artifact.cuda_source
    assert f"constexpr int owners = {config.batch * config.heads * config.value_dimension};" in artifact.cuda_source
    assert request.provenance["source_boundary"] == "ordinary_jax"


def test_matrix_owner_scan_uses_resource_proof_beyond_4096_state_elements() -> None:
    config = NaturalAffineScanConfig(sequence=2, heads=1, key_dimension=256, value_dimension=17)
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    imported = jax_function_to_tensor_program(
        natural_affine_scan_training_step(config),
        *inputs,
        jnp.ones_like(output),
        jnp.ones_like(final_state),
    )
    scan = next(
        operation
        for operation in imported.operations
        if isinstance(operation, ScanPrimitive) and operation.direction is ScanDirection.REVERSE
    )
    region = TensorProgram(primitive_inputs(scan), (scan,), primitive_outputs(scan), imported.numerical_policy)
    plan = derive_chunked_scan_reference_plan(region, chunk_extent=2, state_column_tile=17)
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=region.canonical_record,
        algebra_fingerprint=region.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "ordinary_jax", "automatic_differentiation": "jax"},
    )

    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(region, 0), request)

    assert config.key_dimension * config.value_dimension == 4352
    assert "__shared__ float state_values[K * V]" in artifact.cuda_source
    assert "__syncthreads();" in artifact.cuda_source
    assert request.schedules[0].parameters["state_tile_elements"] == 4352


def test_full_jax_training_scan_pairs_form_persistent_canonical_components() -> None:
    config = NaturalAffineScanConfig(sequence=4, heads=1, key_dimension=2, value_dimension=3)
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    program = jax_function_to_tensor_program(
        natural_affine_scan_training_step(config),
        *inputs,
        jnp.ones_like(output),
        jnp.ones_like(final_state),
    )
    tile_extents = {axis.id: min(axis.extent, 2) for value in program.inputs for axis in value.axes}
    coarse = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extents,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    refined = refine_whole_program_scan_tasks(
        program,
        coarse,
        chunk_extent=2,
        state_column_tile=config.value_dimension,
    )
    request = attach_whole_program_scan_physical_request(
        program,
        refined,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "ordinary_jax", "automatic_differentiation": "jax"},
    )

    composition = lower_canonical_kernel_composition(request)
    scan_components = tuple(
        component
        for component in composition.components
        if any(isinstance(program.operations[ordinal], ScanPrimitive) for ordinal in set(component.operation_ordinals))
    )

    assert len(scan_components) == 2
    assert all(len(component.task_ids) == 2 for component in scan_components)
    assert all(len(set(component.operation_ordinals)) == 1 for component in scan_components)
    selected = tuple(
        ordered_state_scan_component(program, next(iter(set(component.operation_ordinals))))
        for component in scan_components
    )
    assert all(component.source_program_fingerprint == program.semantic_fingerprint for component in selected)
    assert tuple(component.scan.direction for component in selected) == (
        ScanDirection.FORWARD,
        ScanDirection.REVERSE,
    )
    assert all(component.program.operations == (component.scan,) for component in selected)
    assert all(
        not any(
            event.producer_component == component.id and event.consumer_component == component.id
            for event in composition.concrete_events
        )
        for component in scan_components
    )
    assert request.provenance["automatic_differentiation"] == "jax"


def test_natural_affine_scan_jax_reverse_matches_composed_boundary() -> None:
    config = NaturalAffineScanConfig(
        batch=1,
        sequence=4,
        heads=2,
        key_dimension=3,
        value_dimension=5,
        update_rank=2,
    )
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    rng = np.random.default_rng(89)
    output_cotangent = jnp.asarray(rng.normal(size=output.shape), dtype=output.dtype)
    state_cotangent = jnp.asarray(rng.normal(size=final_state.shape), dtype=final_state.dtype)

    objective, *objective_gradients = jax.jit(natural_affine_scan_value_and_grad(config))(
        *inputs,
        output_cotangent,
        state_cotangent,
    )
    composed_output, composed_state, *composed_gradients = jax.jit(natural_affine_scan_training_step(config))(
        *inputs,
        output_cotangent,
        state_cotangent,
    )

    expected_objective = jnp.vdot(composed_output, output_cotangent) + jnp.vdot(composed_state, state_cotangent)
    np.testing.assert_allclose(objective, expected_objective, rtol=1e-6, atol=1e-6)
    for actual, expected in zip(objective_gradients, composed_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        assert np.isfinite(np.asarray(actual)).all()


def test_natural_affine_scan_reverse_exports_as_ordinary_stablehlo() -> None:
    config = NaturalAffineScanConfig(sequence=3, heads=1, key_dimension=2, value_dimension=3)

    modules = (
        export_natural_affine_scan_value_and_grad_text(config),
        export_natural_affine_scan_training_step_text(config),
    )

    for module in modules:
        assert "stablehlo.while" in module
        assert "custom_call" not in module
        assert "custom_vjp" not in module


def test_chunked_scan_plan_has_exact_directional_state_dependencies() -> None:
    program = natural_affine_scan_program(
        NaturalAffineScanConfig(
            batch=1,
            sequence=7,
            heads=2,
            key_dimension=3,
            value_dimension=5,
            update_rank=2,
        ),
        direction=ScanDirection.REVERSE,
    )

    plan = derive_chunked_scan_reference_plan(program, chunk_extent=3, state_column_tile=4)
    chunk_task, commit_task = plan.task_plan.tasks
    carry, commit = plan.task_plan.dependences

    assert chunk_task.operation_kinds == (TaskOperationKind.SCAN,)
    assert chunk_task.structure.traversal is TaskTraversal.REVERSE
    assert tuple(axis.axis_id for axis in chunk_task.structure.owner_axes) == (0, 2, 4)
    assert tuple(axis.axis_id for axis in chunk_task.structure.streamed_axes) == (1,)
    assert tuple(axis.axis_id for axis in chunk_task.structure.contraction_axes) == (3, 5)
    assert commit_task.structure.traversal is TaskTraversal.REVERSE
    assert carry.relation.sources_for((0, 0, 1, 0)) == ((0, 0, 2, 0),)
    assert carry.relation.sources_for((0, 0, 0, 0)) == ((0, 0, 1, 0),)
    assert commit.relation.sources_for((0, 0, 0)) == ((0, 0, 0, 0),)
    for event in plan.event_plans:
        assert set(event.scheduled_dependence.pairs) == set(event.required_dependence.relation.pairs)


def test_chunked_scan_carried_state_uses_bounded_slots_and_generations() -> None:
    program = natural_affine_scan_program(
        NaturalAffineScanConfig(
            batch=1,
            sequence=13,
            heads=1,
            key_dimension=3,
            value_dimension=4,
            update_rank=1,
        )
    )

    plan = derive_chunked_scan_reference_plan(program, chunk_extent=3, state_column_tile=4)
    carried_state = plan.task_plan.bounded_buffers[0]

    assert carried_state.capacity == 2
    assert carried_state.slots == (0, 1, 0, 1, 0)
    assert carried_state.generations == (0, 0, 1, 1, 2)
    for event, operations in zip(plan.event_plans, plan.imperative_events, strict=True):
        assert sum(operation.kind is ImperativeEventOpKind.WAIT for operation in operations) == len(
            event.trigger_relation.pairs
        )
        assert sum(operation.kind is ImperativeEventOpKind.NOTIFY for operation in operations) == len(
            event.notify_relation.pairs
        )


def test_natural_affine_scan_algebra_preserves_typed_body_and_relations() -> None:
    config = NaturalAffineScanConfig(sequence=5, key_dimension=3, value_dimension=4, update_rank=2)

    program = natural_affine_scan_program(config)
    scan = program.operations[0]

    assert isinstance(scan, ScanPrimitive)
    assert scan.direction.value == "forward"
    assert scan.axis.extent == config.sequence
    assert scan.state[0].body_argument.axes == scan.state[0].body_result.axes
    assert tuple(binding.source.axes for binding in scan.scanned_inputs) == tuple(
        binding.indexing.domain_axes for binding in scan.scanned_inputs
    )
    decay = scan.body.operations[0]
    assert isinstance(decay, MapPrimitive)
    assert decay.expression.kind is ScalarExpressionKind.EXP
    assert [operation["kind"] for operation in program.canonical_record["operations"]] == ["scan"]
    renamed = replace(scan, name="arbitrary_ordered_program")
    assert derive_chunked_scan_tiling(renamed, chunk_extent=2, state_column_tile=4) == derive_chunked_scan_tiling(
        scan,
        chunk_extent=2,
        state_column_tile=4,
    )
    renamed_program = replace(program, operations=(renamed,))
    original_plan = derive_chunked_scan_reference_plan(program, chunk_extent=2, state_column_tile=4)
    renamed_plan = derive_chunked_scan_reference_plan(renamed_program, chunk_extent=2, state_column_tile=4)
    assert original_plan.task_plan.fingerprint == renamed_plan.task_plan.fingerprint


def test_scan_scalar_mutation_changes_algebra_but_reuses_task_and_schedule_structure() -> None:
    base_config = NaturalAffineScanConfig(sequence=5, key_dimension=3, value_dimension=4)
    mutated_config = NaturalAffineScanConfig(
        sequence=5,
        key_dimension=3,
        value_dimension=4,
        diagonal_operation=ScanDiagonalOperation.EXP_SQUARED,
    )

    base = compile_natural_affine_scan_reference_pipeline(base_config, chunk_extent=2, state_column_tile=4)
    mutated = compile_natural_affine_scan_reference_pipeline(mutated_config, chunk_extent=2, state_column_tile=4)

    assert base.algebra.semantic_fingerprint != mutated.algebra.semantic_fingerprint
    assert base.plan.task_plan.fingerprint == mutated.plan.task_plan.fingerprint
    assert (
        base.physical_request.schedules[0].family.fingerprint == mutated.physical_request.schedules[0].family.fingerprint
    )
    assert base.physical_request.fingerprint != mutated.physical_request.fingerprint


def test_scan_physical_request_erases_events_only_with_persistent_owner_order() -> None:
    pipeline = compile_natural_affine_scan_reference_pipeline(
        NaturalAffineScanConfig(sequence=7, key_dimension=3, value_dimension=5),
        chunk_extent=3,
        state_column_tile=4,
    )
    request = pipeline.physical_request

    assert request.algebra_fingerprint == pipeline.algebra.semantic_fingerprint
    assert request.task_plan.fingerprint == pipeline.plan.task_plan.fingerprint
    assert all(
        entry.kind is EventRealizationKind.ERASED_PROGRAM_ORDER for entry in request.event_realizations.audit.entries
    )
    assert tuple(witness.plan_name for witness in request.event_realizations.erasure_witnesses) == (
        "state_carry",
        "state_commit",
    )
    assert all(
        all(generation == 0 for generation in allocation.storage.generations) for allocation in request.event_allocations
    )
    assert len(request.fingerprint) == 64
    for forbidden in ("gdn", "gated_deltanet", "attention", "moe"):
        assert forbidden not in request.canonical_json.lower()


def test_scan_scalar_ast_changes_generated_kernel_without_changing_schedule_family() -> None:
    base_config = NaturalAffineScanConfig(sequence=5, key_dimension=8, value_dimension=16)
    composed_config = replace(base_config, diagonal_operation=ScanDiagonalOperation.EXP_SQUARED)
    base = compile_natural_affine_scan_reference_pipeline(base_config, chunk_extent=2, state_column_tile=16)
    composed = compile_natural_affine_scan_reference_pipeline(composed_config, chunk_extent=2, state_column_tile=16)

    base_artifact = generate_cuda_ordered_state_artifact(
        ordered_state_scan_component(base.algebra, 0), base.physical_request
    )
    composed_artifact = generate_cuda_ordered_state_artifact(
        ordered_state_scan_component(composed.algebra, 0), composed.physical_request
    )

    assert base.physical_request.schedules[0].family.fingerprint == (
        composed.physical_request.schedules[0].family.fingerprint
    )
    assert base_artifact.request_fingerprint != composed_artifact.request_fingerprint
    assert base_artifact.source_sha256 != composed_artifact.source_sha256
    assert "expf(value_3(" in base_artifact.cuda_source
    assert "expf(value_3(" in composed_artifact.cuda_source
    assert composed_artifact.cuda_source.count("expf(value_3(") > base_artifact.cuda_source.count("expf(value_3(")
    for forbidden in ("gdn", "gated_deltanet", "attention", "moe"):
        assert forbidden not in base_artifact.cuda_source.lower()


def test_scan_emitter_compiles_an_inserted_generic_map_without_topology_dispatch() -> None:
    base = compile_natural_affine_scan_reference_pipeline(
        NaturalAffineScanConfig(sequence=5, key_dimension=8, value_dimension=16),
        chunk_extent=2,
        state_column_tile=16,
    )
    scan = base.algebra.operations[0]
    assert isinstance(scan, ScanPrimitive)
    decay, decay_state, *tail = scan.body.operations
    assert isinstance(decay, MapPrimitive) and isinstance(decay_state, MapPrimitive)
    forwarded_decay = ProgramValue("forwarded_decay", decay.output.axes, decay.output.dtype)
    neutral = MapPrimitive(
        "arbitrary_neutral_map",
        (decay.output,),
        forwarded_decay,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(decay.output.name),
            scalar_constant(0.0),
        ),
    )
    state_input = next(value for value in decay_state.inputs if value != decay.output)
    rewritten_decay_state = replace(
        decay_state,
        inputs=(forwarded_decay, state_input),
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(forwarded_decay.name),
            scalar_input(state_input.name),
        ),
    )
    body = replace(scan.body, operations=(decay, neutral, rewritten_decay_state, *tail))
    mutated_algebra = replace(
        base.algebra, operations=(replace(scan, body=body),), numerical_policy=NumericalPolicy.SOURCE_ORDERED
    )
    plan = derive_chunked_scan_reference_plan(mutated_algebra, chunk_extent=2, state_column_tile=16)
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=mutated_algebra.canonical_record,
        algebra_fingerprint=mutated_algebra.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "generic_test"},
    )

    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(mutated_algebra, 0), request)

    assert plan.task_plan.fingerprint == base.plan.task_plan.fingerprint
    assert request.schedules[0].family.fingerprint == base.physical_request.schedules[0].family.fingerprint
    assert (
        artifact.source_sha256
        != generate_cuda_ordered_state_artifact(
            ordered_state_scan_component(base.algebra, 0), base.physical_request
        ).source_sha256
    )


def test_jax_owned_training_refines_both_scans_with_exact_events_and_bounded_state() -> None:
    config = NaturalAffineScanConfig(
        batch=1,
        sequence=5,
        heads=1,
        key_dimension=2,
        value_dimension=3,
        update_rank=1,
    )
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    arguments = (*inputs, jnp.ones_like(output), jnp.ones_like(final_state))
    program = jax_function_to_tensor_program(natural_affine_scan_training_step(config), *arguments)
    tile_extents = {axis.id: min(axis.extent, 2) for value in program.inputs for axis in value.axes}
    coarse = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extents,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )

    refined = refine_whole_program_scan_tasks(
        program,
        coarse,
        chunk_extent=2,
        state_column_tile=2,
    )

    scans = tuple(
        (ordinal, operation)
        for ordinal, operation in enumerate(program.operations)
        if isinstance(operation, ScanPrimitive)
    )
    assert tuple(scan.direction for _, scan in scans) == (ScanDirection.FORWARD, ScanDirection.REVERSE)
    assert len(refined.task_plan.tasks) == len(coarse.tasks) + len(scans)
    assert len(refined.task_plan.bounded_buffers) == 2
    assert tuple(buffer.capacity for buffer in refined.task_plan.bounded_buffers) == (2, 2)
    for ordinal, scan in scans:
        scan_tasks = tuple(task for task in refined.task_plan.tasks if task.operation_ordinals == (ordinal,))
        assert len(scan_tasks) == 2
        assert scan_tasks[0].family.name == f"scan.{ordinal}.chunks"
        assert scan_tasks[1].family.name == f"scan.{ordinal}.commit"
        expected = TaskTraversal.FORWARD if scan.direction is ScanDirection.FORWARD else TaskTraversal.REVERSE
        assert all(task.structure.traversal is expected for task in scan_tasks)
        assert {f"scan.{ordinal}.state_carry", f"scan.{ordinal}.state_commit"} <= {
            event.name for event in refined.event_plans
        }
    for event, operations in zip(refined.event_plans, refined.imperative_events, strict=True):
        assert set(event.scheduled_dependence.pairs) == set(event.required_dependence.relation.pairs)
        assert sum(operation.kind is ImperativeEventOpKind.WAIT for operation in operations) == len(
            event.trigger_relation.pairs
        )
        assert sum(operation.kind is ImperativeEventOpKind.NOTIFY for operation in operations) == len(
            event.notify_relation.pairs
        )


def test_scan_refinement_splices_forward_tapes_into_reverse_chunks_by_ordered_tile() -> None:
    config = NaturalAffineScanConfig(sequence=5, heads=1, key_dimension=2, value_dimension=3)
    inputs = _inputs(config)
    output, final_state = natural_affine_scan_region(config)(*inputs)
    program = jax_function_to_tensor_program(
        natural_affine_scan_training_step(config),
        *inputs,
        jnp.ones_like(output),
        jnp.ones_like(final_state),
    )
    coarse = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: min(axis.extent, 2) for value in program.inputs for axis in value.axes},
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    refined = refine_whole_program_scan_tasks(program, coarse, chunk_extent=2, state_column_tile=2)
    scan_ordinals = tuple(
        ordinal for ordinal, operation in enumerate(program.operations) if isinstance(operation, ScanPrimitive)
    )
    forward_family = next(
        task.family for task in refined.task_plan.tasks if task.family.name == f"scan.{scan_ordinals[0]}.chunks"
    )
    reverse_family = next(
        task.family for task in refined.task_plan.tasks if task.family.name == f"scan.{scan_ordinals[1]}.chunks"
    )
    tape_dependence = next(
        dependence
        for dependence in refined.task_plan.dependences
        if dependence.relation.source == forward_family and dependence.relation.target == reverse_family
    )

    assert tape_dependence.relation.pairs
    assert all(pair.source[2] == pair.target[2] for pair in tape_dependence.relation.pairs)


def test_scan_refinement_exposes_invariants_and_emitted_values_at_component_boundary() -> None:
    def affine_scan(initial, sequence, scale):
        def step(state, value):
            next_state = state * scale + value
            return next_state, state * value

        return jax.lax.scan(step, initial, sequence)

    initial = jnp.zeros((1, 1, 2, 3), dtype=jnp.float32)
    sequence = jnp.ones((5, 1, 1, 2, 3), dtype=jnp.float32)
    scale = jnp.full(initial.shape, 0.75, dtype=jnp.float32)
    program = jax_function_to_tensor_program(affine_scan, initial, sequence, scale)
    coarse = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: min(axis.extent, 2) for value in program.inputs for axis in value.axes},
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    refined = refine_whole_program_scan_tasks(program, coarse, chunk_extent=2, state_column_tile=3)
    scan_ordinal, scan = next(
        (ordinal, operation)
        for ordinal, operation in enumerate(program.operations)
        if isinstance(operation, ScanPrimitive)
    )
    chunk = next(task for task in refined.task_plan.tasks if task.family.name == f"scan.{scan_ordinal}.chunks")
    value_ordinals = tensor_program_value_ordinals(program)

    invariant_ids = {value_ordinals[binding.source] for binding in scan.invariants}
    emitted_ids = {value_ordinals[binding.destination] for binding in scan.emitted_outputs}
    assert invariant_ids <= {access.value_id for access in chunk.accesses if access.kind is TaskValueAccessKind.LIVE_IN}
    assert emitted_ids <= {access.value_id for access in chunk.accesses if access.kind is TaskValueAccessKind.LIVE_OUT}
