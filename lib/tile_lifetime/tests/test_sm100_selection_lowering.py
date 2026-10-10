# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tile_lifetime.backends.sm100.canonical_aot_artifact import (
    seal_sm100_canonical_aot_artifact,
    sm100_canonical_aot_compile_plan,
)
from tile_lifetime.backends.sm100.canonical_jax_ffi import generate_sm100_canonical_jax_ffi
from tile_lifetime.backends.sm100.canonical_simt_emitter import (
    SIMT_SCHEDULE_FAMILY,
    compose_sm100_canonical_simt_pipeline,
    emit_sm100_canonical_simt_components,
    lower_sm100_native_simt_kernel_composition,
    render_canonical_simt_kernel,
)
from tile_lifetime.backends.sm100.refined_schedule import (
    REFINED_SELECTION_NUMERICAL_REJECTION,
    attach_sm100_dependency_convex_gathered_normalized_exp_schedule,
    select_sm100_dependency_convex_gathered_normalized_exp_regions,
    select_sm100_dependency_convex_selection_regions,
    sm100_dependency_convex_selection_numerical_rejection,
)
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.dense_flow_planner import (
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.msa_frontend import MSADebugConfig, msa_preprojected_main_region, msa_region
from tile_lifetime.msa_reference_pipeline import (
    build_msa_reference_oracle,
    derive_msa_reference_dataflow,
    msa_reference_physical_request,
)
from tile_lifetime.physical_task_contract import (
    PhysicalNumericalPolicy,
    PhysicalTarget,
    TaskOperationKind,
    lower_executable_event_operations,
    physical_schedule_is_eligible,
)
from tile_lifetime.routed_attention import IndexDomainRestriction, ProjectedBlockSelectionProgram
from tile_lifetime.sm100_canonical_kernel_composition import (
    lower_sm100_canonical_kernel_composition,
    lower_sm100_canonical_simt_kernel_composition,
    select_sm100_projection_contract_lowerings,
)
from tile_lifetime.sm100_projected_routed_lowering import (
    lower_sm100_composed_physical_request,
    select_sm100_composed_task_lowerings,
)
from tile_lifetime.sm100_selection_lowering import (
    SM100SelectionStrategy,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
    select_sm100_selection_lowerings,
)
from tile_lifetime.tensor_program import (
    DType,
    OperandIndexing,
    ProgramValue,
    TensorAxis,
    TransportPrimitive,
    index_axis,
    index_tensor_element,
)


def _program(*, force_local_block: bool = True) -> ProjectedBlockSelectionProgram:
    return ProjectedBlockSelectionProgram(
        source_input="hidden",
        left_weight_input="index_q.weight",
        right_weight_input="index_k.weight",
        source_count=512,
        source_feature_count=3072,
        group_count=4,
        relation_feature_count=128,
        right_block_size=128,
        selected_count=4,
        score_scale=128**-0.5,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=force_local_block,
    )


def _physical_request(
    *,
    index_dimension: int = 128,
    numerical_policy: PhysicalNumericalPolicy = PhysicalNumericalPolicy.FAST,
):
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=128,
        index_dimension=index_dimension,
        block_size=128,
        selected_blocks=4,
    )
    oracle = build_msa_reference_oracle(config)
    return msa_reference_physical_request(
        oracle,
        derive_msa_reference_dataflow(oracle, numerical_policy=numerical_policy),
        numerical_policy=numerical_policy,
    )


def test_sm100_selection_attaches_to_generic_request_structure() -> None:
    request = _physical_request()

    materialized, fused = select_sm100_selection_lowerings(request)

    assert materialized.schedule.strategy is SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES
    assert fused.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
    assert materialized.physical_schedule.family == fused.physical_schedule.family
    assert physical_schedule_is_eligible(fused.physical_schedule, request.task_plan)
    assert tuple(request.task_plan.tasks[task_id].structure.principal_kind for task_id in fused.task_ids) == (
        TaskOperationKind.CONTRACT,
        TaskOperationKind.FOLD,
        TaskOperationKind.SELECTION,
    )
    assert fused.feature_extent == fused.right_block_size == 128
    assert fused.selected_extent == 4
    assert len(fused.physical_schedule.enforced_orders) == 2
    family_text = fused.physical_schedule.family.name.lower()
    assert all(token not in family_text for token in ("msa", "attention", "projected", "routed"))


def test_sm100_selection_eligibility_ignores_provenance_and_rejects_unsupported_geometry() -> None:
    request = _physical_request()
    renamed_origin = replace(request, provenance={"source_boundary": "unrelated_frontend"})

    first = select_sm100_selection_lowerings(request)
    second = select_sm100_selection_lowerings(renamed_origin)

    assert tuple(item.physical_schedule.family.fingerprint for item in first) == tuple(
        item.physical_schedule.family.fingerprint for item in second
    )
    assert tuple(item.task_ids for item in first) == tuple(item.task_ids for item in second)
    assert select_sm100_selection_lowerings(_physical_request(index_dimension=64)) == ()


def test_sm100_composition_uses_generic_dependences_between_independently_eligible_regions() -> None:
    request = _physical_request()

    lowering = select_sm100_composed_task_lowerings(request)

    assert len(lowering.candidates) == 4
    assert all(candidate.selection.task_ids == (5, 6, 7) for candidate in lowering.candidates)
    assert all(candidate.routed.task_ids == (8, 9, 10, 11, 12) for candidate in lowering.candidates)
    assert all(
        candidate.selection.selected_extent == candidate.routed.selected_extent for candidate in lowering.candidates
    )


def test_sm100_composition_installs_full_request_and_erases_only_fused_internal_events() -> None:
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=4,
    )
    oracle = build_msa_reference_oracle(config)
    source = msa_reference_physical_request(
        oracle,
        derive_msa_reference_dataflow(oracle, numerical_policy=PhysicalNumericalPolicy.FAST),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    candidates = select_sm100_composed_task_lowerings(source).candidates
    selected = next(
        candidate
        for candidate in candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
        and candidate.routed.schedule.orientation.value == "right_major"
    )

    lowered = lower_sm100_composed_physical_request(source, selected)
    request = lowered.request
    concrete_events = lower_executable_event_operations(request)

    assert tuple(schedule.task_ids for schedule in request.schedules) == (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5, 6, 7),
        (8, 9, 10, 11, 12),
    )
    assert len(request.event_realizations.audit.erased) == 8
    assert len(request.event_realizations.audit.physical) == 4
    assert all(
        event.realization.kind is EventRealizationKind.ERASED_PROGRAM_ORDER and not event.retained_operations
        for event in concrete_events
        if event.plan_name in {entry.plan_name for entry in request.event_realizations.audit.erased}
    )
    assert all(
        event.realization.kind is EventRealizationKind.PHYSICAL and event.retained_operations
        for event in concrete_events
        if event.plan_name in {entry.plan_name for entry in request.event_realizations.audit.physical}
    )
    assert all(event.realization.value == "imperative_wait_notify" for event in request.event_programs)
    assert request.algebra_fingerprint == source.algebra_fingerprint
    assert request.task_plan.fingerprint == source.task_plan.fingerprint
    assert request.fingerprint != source.fingerprint

    abi = lowered.emitter_abi
    assert abi.existing_emitter_constructor == {
        "head_dim": 128,
        "qheadperkv": 4,
        "m_block_size": 128,
        "n_block_size": 128,
        "use_prepare_scheduler": True,
    }
    assert "score_scale" not in abi.existing_emitter_constructor
    assert "causal" not in abi.existing_emitter_constructor
    assert len(abi.runtime_relations) == 2
    assert abi.residual_task_ids == (0, 1, 2, 3, 4)
    assert abi.retained_event_ids == (0, 1, 4, 5)
    assert not abi.complete_existing_emitter_coverage
    assert abi.runtime_relations[0].relation_program_fingerprint == abi.runtime_relations[1].relation_program_fingerprint
    assert abi.runtime_relations[0].algebra_value_ordinals == abi.runtime_relations[1].algebra_value_ordinals
    assert all(relation.event_realization == "imperative_wait_notify" for relation in abi.runtime_relations)
    assert any(program.operation.get("expression", {}).get("kind") == "select" for program in abi.scalar_programs)
    emitter_record = json.dumps(abi.canonical_record, sort_keys=True).lower()
    assert all(token not in emitter_record for token in ("msa", "attention", "causal", "score_scale"))


def test_sm100_canonical_kernel_composition_covers_residual_contracts_and_retained_events() -> None:
    source = _physical_request()
    selected = next(
        candidate
        for candidate in select_sm100_composed_task_lowerings(source).candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
        and candidate.routed.schedule.orientation.value == "right_major"
    )
    installed = lower_sm100_composed_physical_request(source, selected)

    composition = lower_sm100_canonical_kernel_composition(installed)

    assert composition.complete_task_and_event_coverage
    assert {item.task_id for item in composition.projection_lowerings} == set(installed.emitter_abi.residual_task_ids)
    assert {item.physical_schedule.family.name for item in composition.projection_lowerings} == {
        "sm100_contract_relations_maps_128"
    }
    component_task_sets = tuple(set(component.task_ids) for component in composition.components)
    assert set().union(*component_task_sets) == set(range(len(source.task_plan.tasks)))
    assert sum(map(len, component_task_sets)) == len(set().union(*component_task_sets))
    assert composition.concrete_events
    assert composition.parametric_events
    assert all(event.realization == "imperative_wait_notify" for event in composition.concrete_events)
    assert all(event.notify_operation_count and event.wait_operation_count for event in composition.concrete_events)
    assert all(event.relation.event_realization == "imperative_wait_notify" for event in composition.parametric_events)
    task_by_family = {task.family: task.id for task in source.task_plan.tasks}
    for event in composition.concrete_events:
        plan = source.event_plans[event.event_id]
        assert (
            task_by_family[plan.scheduled_dependence.source] in composition.components[event.producer_component].task_ids
        )
        assert (
            task_by_family[plan.scheduled_dependence.target] in composition.components[event.consumer_component].task_ids
        )
        assert event.event_id in composition.components[event.producer_component].concrete_notify_event_ids
        assert event.event_id in composition.components[event.consumer_component].concrete_wait_event_ids
    for event in composition.parametric_events:
        assert event.relation.source_task_id in composition.components[event.producer_component].task_ids
        assert event.relation.target_task_id in composition.components[event.consumer_component].task_ids
        assert (
            event.dependence_program_id in composition.components[event.producer_component].parametric_notify_event_ids
        )
        assert event.dependence_program_id in composition.components[event.consumer_component].parametric_wait_event_ids
    assert len(composition.fingerprint) == 64
    record = json.dumps(composition.canonical_record, sort_keys=True).lower()
    assert all(token not in record for token in ("msa", "attention", "causal", "gqa", "mqa"))


def test_sm100_canonical_simt_emitter_composes_full_request_and_event_order() -> None:
    source = _physical_request()
    selected = next(
        candidate
        for candidate in select_sm100_composed_task_lowerings(source).candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
        and candidate.routed.schedule.orientation.value == "right_major"
    )
    installed = lower_sm100_composed_physical_request(source, selected)
    tensor_core = lower_sm100_canonical_kernel_composition(installed)
    simt = lower_sm100_canonical_simt_kernel_composition(installed)

    tensor_core_coverage = emit_sm100_canonical_simt_components(tensor_core)
    coverage = emit_sm100_canonical_simt_components(simt)

    residual_tasks = set(installed.emitter_abi.residual_task_ids)
    residual_components = {component.id for component in simt.components if set(component.task_ids) <= residual_tasks}
    assert {item.component_id for item in coverage.generated} == {component.id for component in simt.components}
    assert coverage.complete
    assert not tensor_core_coverage.generated
    assert {component_id for component_id, _ in tensor_core_coverage.unsupported_components} == {
        component.id for component in tensor_core.components
    }
    assert all(
        simt.scheduled_request.schedules[component.schedule_index].family.name == SIMT_SCHEDULE_FAMILY
        for component in simt.components
        if component.id in residual_components
    )
    for generated in coverage.generated:
        component = simt.components[generated.component_id]
        assert generated.fragment.schedule_fingerprint == component.schedule_fingerprint
        assert len(generated.fragment.launches) == len(component.operation_ordinals)
        assert {task for launch in generated.fragment.launches for task in launch.source_tasks} == set(
            component.task_ids
        )
        assert all(launch.kernel_symbol in generated.source for launch in generated.fragment.launches)
        assert len(generated.source_sha256) == 64
    assert coverage.concrete_events == simt.concrete_events == tensor_core.concrete_events
    assert coverage.parametric_events == simt.parametric_events == tensor_core.parametric_events
    pipeline = compose_sm100_canonical_simt_pipeline(simt)
    assert {task for launch in pipeline.recipe.launches for task in launch.source_tasks} == set(
        range(len(source.task_plan.tasks))
    )
    assert all(
        pipeline.recipe.event_realizations[event.event_id].kind.value == "stream_order" for event in simt.concrete_events
    )
    assert {erasure.dependence_program_id for erasure in pipeline.runtime_event_erasures} == set(
        range(len(simt.parametric_events))
    )
    assert all(erasure.producer_launch < erasure.consumer_launch for erasure in pipeline.runtime_event_erasures)
    assert len(pipeline.source_sha256) == 64
    emitted_record = json.dumps(
        {
            "schedules": tuple(
                simt.scheduled_request.schedules[component.schedule_index].canonical_record
                for component in simt.components
                if component.id in residual_components
            ),
            "events": tuple(event.canonical_record for event in coverage.concrete_events),
            "runtime_events": tuple(event.canonical_record for event in coverage.parametric_events),
        },
        sort_keys=True,
    ).lower()
    assert all(token not in emitted_record for token in ("msa", "attention", "gqa", "mqa", "causal"))


def test_sm100_canonical_simt_pipeline_seals_complete_native_artifact(tmp_path: Path) -> None:
    source = _physical_request()
    selected = next(
        candidate
        for candidate in select_sm100_composed_task_lowerings(source).candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
        and candidate.routed.schedule.orientation.value == "right_major"
    )
    installed = lower_sm100_composed_physical_request(source, selected)
    composition = lower_sm100_canonical_simt_kernel_composition(installed)

    plan = sm100_canonical_aot_compile_plan(
        composition,
        output_directory=tmp_path,
        nvcc=Path("/opt/cuda/bin/nvcc"),
        cuda_include_directories=(tmp_path,),
    )
    artifact = seal_sm100_canonical_aot_artifact(
        plan,
        host_object=b"host-object",
        device_image=b"fatbin-image",
        nvcc_version="Cuda compilation tools, release 13.0",
    )

    assert len(plan.pipeline.recipe.launches) == len(plan.kernel_names) == 39
    assert set(plan.kernel_names) == {launch.kernel_symbol for launch in plan.pipeline.recipe.launches}
    assert tuple(plan.request.schedule["runtime_parametric_event_erasures"]) == tuple(
        erasure.canonical_record for erasure in plan.pipeline.runtime_event_erasures
    )
    assert all(erasure.producer_launch < erasure.consumer_launch for erasure in plan.pipeline.runtime_event_erasures)
    assert plan.cuda_include_directories == (tmp_path,)
    assert ("-I", str(tmp_path)) in tuple(zip(plan.host_compile_argv, plan.host_compile_argv[1:], strict=False))
    assert artifact.launch_recipe == plan.pipeline.recipe
    assert artifact.manifest["compiler_ir"]["format"] == "cuda-c++"
    assert artifact.manifest["device_image"]["kernels"] == plan.kernel_names
    transport = artifact.native_transport
    assert transport.startswith(b"SHUTGPU\0")
    assert int.from_bytes(transport[12:20], "little") == 6
    assert transport[20] == 5

    generated = generate_sm100_canonical_jax_ffi(
        artifact,
        target_name="shuttle.sm100_canonical_abi_conformance",
    )
    assert f'extern "C" int {artifact.host_entry}(void** arguments, cudaStream_t stream);' in generated.source
    assert "ffi::ScratchAllocator scratch" in generated.source
    assert generated.source.count("scratch.Allocate(") == sum(
        slot.kind.value == "scratch" for slot in artifact.launch_recipe.slots
    )
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL(shuttle_sm100_canonical_abi_conformance" in generated.source


def test_sm100_native_simt_pipeline_consumes_ordinary_jax_request() -> None:
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
    shapes = ((4, 4), (4, 4), (4, 4), (4, 2), (4, 2), (4, 2), (4, 2))
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    arguments = tuple(jnp.zeros(shape, dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    program = jax_function_to_tensor_program(msa_region(config), *arguments)
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={},
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=eventize_composed_task_plan(task_plan),
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )

    composition = lower_sm100_native_simt_kernel_composition(request)
    pipeline = compose_sm100_canonical_simt_pipeline(composition)

    assert composition.source_request_fingerprint == request.fingerprint
    assert {task for launch in pipeline.recipe.launches for task in launch.source_tasks} == set(
        range(len(task_plan.tasks))
    )
    assert len(pipeline.runtime_event_erasures) == len(request.event_programs) == 2
    assert all(erasure.producer_launch < erasure.consumer_launch for erasure in pipeline.runtime_event_erasures)
    assert pipeline.recipe.host_call_arguments


def test_sm100_dependency_convex_selection_requires_numerical_certificate() -> None:
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=4,
    )
    shapes = ((128, 128), (512, 128), (128, 1024), (128, 256), (128, 256), (128, 256), (128, 128))
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    arguments = tuple(jnp.zeros(shape, dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    program = jax_function_to_tensor_program(msa_region(config), *arguments)
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=eventize_composed_task_plan(task_plan),
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )

    assert select_sm100_dependency_convex_selection_regions(request) == ()
    assert sm100_dependency_convex_selection_numerical_rejection() == REFINED_SELECTION_NUMERICAL_REJECTION
    assert "tcgen05-to-canonical Contract error interval" in REFINED_SELECTION_NUMERICAL_REJECTION
    assert "scalar-Map interval propagation" in REFINED_SELECTION_NUMERICAL_REJECTION


def test_sm100_dependency_convex_gathered_normalized_exp_preserves_runtime_event_boundary() -> None:
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=4,
    )
    shapes = ((128, 128), (512, 128), (128, 1024), (128, 256), (128, 256), (128, 256), (128, 128))
    dtypes = (jnp.bfloat16,) * 5 + (jnp.float32,) * 2
    arguments = tuple(jnp.zeros(shape, dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    program = jax_function_to_tensor_program(msa_region(config), *arguments)
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=eventize_composed_task_plan(task_plan),
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    fallback = lower_sm100_native_simt_kernel_composition(request)
    fallback_pipeline = compose_sm100_canonical_simt_pipeline(fallback)
    selection_request = fallback.scheduled_request

    regions = select_sm100_dependency_convex_gathered_normalized_exp_regions(selection_request)

    assert len(regions) == 1
    region = regions[0]
    tasks = tuple(task_plan.tasks[task_id] for task_id in region.task_ids)
    assert tuple(task.structure.principal_kind for task in tasks).count(TaskOperationKind.CONTRACT) == 2
    assert tuple(task.structure.principal_kind for task in tasks).count(TaskOperationKind.FOLD) == 2
    assert len(region.gather_task_ids) == 2
    assert all(
        dependence.source_task_id not in region.task_ids and dependence.target_task_id in region.task_ids
        for dependence in region.refinement.coarse_plan.dependence_programs
    )
    assert len(region.refinement.coarse_plan.dependence_programs) == len(task_plan.dependence_programs)

    attached = attach_sm100_dependency_convex_gathered_normalized_exp_schedule(selection_request, region)
    composition = lower_canonical_kernel_composition(attached.scheduled_request)
    pipeline = compose_sm100_canonical_simt_pipeline(composition)
    fused_operations = sum(len(task_plan.tasks[task_id].operation_ordinals) for task_id in region.task_ids)

    assert len(composition.components) == len(fallback.components) - len(region.task_ids) + 1
    expected_launches = len(fallback_pipeline.recipe.launches) - fused_operations + 1
    assert len(pipeline.recipe.launches) == expected_launches
    assert len(pipeline.runtime_event_erasures) == len(task_plan.dependence_programs)
    assert all(erasure.producer_launch < erasure.consumer_launch for erasure in pipeline.runtime_event_erasures)
    fused_launches = tuple(launch for launch in pipeline.recipe.launches if launch.source_tasks == region.task_ids)
    assert len(fused_launches) == 1
    assert region.physical_schedule.parameters["matrix_instruction"] == "tcgen05_m64n128k16_bf16_fp32_pair"
    assert region.physical_schedule.parameters["owner_rows"] == 4
    assert region.physical_schedule.parameters["active_tensor_warps"] == 1
    assert region.physical_schedule.parameters["resident_warps_per_cta"] == 8
    assert region.physical_schedule.parameters["cta_barriers_per_component"] == 49
    assert region.physical_schedule.parameters["tensor_core_arrivals_per_component"] == 16
    assert region.physical_schedule.parameters["shared_memory_bytes"] == 51_344
    assert fused_launches[0].grid == (config.query_length * config.key_value_heads, 1, 1)
    assert fused_launches[0].dynamic_shared_memory_bytes == 4 * 128 * 4
    assert pipeline.source.count("SM100_MMA_F16BF16_SS") >= 1
    assert "wmma::" not in pipeline.source
    assert pipeline.source.count("fence_view_async_shared") >= 2
    assert pipeline.source.count("fence_barrier_init") >= 2
    assert "tensor_memory_output" in pipeline.source
    assert pipeline.source.count("normalized_wave_base < normalized_extent") == 3
    normalized_weight_line = next(
        line for line in pipeline.source.splitlines() if "const float normalized_weight" in line
    )
    assert "expf" in normalized_weight_line
    assert "row_sum" in normalized_weight_line
    assert "shared_right(value_feature, matrix_index / 128)" in pipeline.source
    assert "shuttle_sm100_refined_gathered_normalized_exp" in pipeline.source
    assert "msa" not in json.dumps(region.physical_schedule.canonical_record).lower()


def test_sm100_preprojected_main_boundary_uses_same_generic_schedule_families() -> None:
    config = MSADebugConfig(
        query_length=128,
        key_value_length=2048,
        hidden_dimension=128,
        query_heads=64,
        key_value_heads=4,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=16,
    )
    arguments = (
        jax.ShapeDtypeStruct((128, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 512), jnp.float32),
        jax.ShapeDtypeStruct((128, 128), jnp.float32),
        jax.ShapeDtypeStruct((128, 64, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048, 4, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048, 4, 128), jnp.bfloat16),
    )
    program = jax_function_to_tensor_program(msa_preprojected_main_region(config), *arguments)
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=eventize_composed_task_plan(task_plan),
        target=PhysicalTarget.SM100,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    fallback = lower_sm100_native_simt_kernel_composition(request)
    fallback_pipeline = compose_sm100_canonical_simt_pipeline(fallback)
    selection_request = fallback.scheduled_request
    normalized_exp = select_sm100_dependency_convex_gathered_normalized_exp_regions(selection_request)[0]
    scheduled = attach_sm100_dependency_convex_gathered_normalized_exp_schedule(
        selection_request,
        normalized_exp,
    ).scheduled_request
    pipeline = compose_sm100_canonical_simt_pipeline(lower_canonical_kernel_composition(scheduled))

    assert normalized_exp.physical_schedule.family.name == "sm100_dependency_convex_gathered_normalized_exp_contract"
    assert normalized_exp.physical_schedule.parameters["owner_rows"] == 16
    assert normalized_exp.physical_schedule.parameters["active_tensor_warps"] == 1
    assert normalized_exp.physical_schedule.parameters["cta_barriers_per_component"] == 181
    assert normalized_exp.physical_schedule.parameters["resident_ctas_by_shared_memory"] == 2
    assert normalized_exp.physical_schedule.parameters["shared_memory_bytes"] == 57_488
    fused_operations = sum(len(task_plan.tasks[task_id].operation_ordinals) for task_id in normalized_exp.task_ids)
    assert len(pipeline.recipe.launches) == len(fallback_pipeline.recipe.launches) - fused_operations + 1
    assert len(pipeline.runtime_event_erasures) == 2
    assert all(erasure.producer_launch < erasure.consumer_launch for erasure in pipeline.runtime_event_erasures)
    materialized_shapes = {slot.shape for slot in pipeline.recipe.slots}
    assert (config.query_length, config.key_value_heads, config.key_value_length) in materialized_shapes
    assert (
        config.query_length,
        config.key_value_heads,
        config.block_count,
        config.block_size,
    ) in materialized_shapes
    assert "msa" not in json.dumps(scheduled.canonical_record).lower()


def test_sm100_canonical_simt_transport_requires_collision_free_destination() -> None:
    row = TensorAxis(0, 3)
    column = TensorAxis(1, 4)
    source = ProgramValue("source", (row, column), DType.FP32)
    destination = ProgramValue("destination", (column, row), DType.FP32)
    operation = TransportPrimitive(
        "transpose placement",
        source,
        destination,
        (),
        (row, column),
        OperandIndexing(source, (row, column), (index_axis(row), index_axis(column))),
        OperandIndexing(destination, (row, column), (index_axis(column), index_axis(row))),
    )

    generated = render_canonical_simt_kernel("permutation_transport", operation, {source: 0, destination: 1})

    assert 'extern "C" __global__ void permutation_transport' in generated
    assert "argument_1" in generated

    route = ProgramValue("route", (row,), DType.INT32)
    dynamic_destination = replace(
        operation,
        routing_inputs=(route,),
        destination_indexing=OperandIndexing(
            destination,
            (row, column),
            (index_tensor_element(route, index_axis(row)), index_axis(row)),
        ),
    )
    with pytest.raises(ValueError, match="not proven injective"):
        render_canonical_simt_kernel(
            "dynamic_transport",
            dynamic_destination,
            {source: 0, route: 1, destination: 2},
        )


def test_sm100_projection_contract_schedule_selection_obeys_policy_and_geometry() -> None:
    fast = _physical_request()
    source_ordered = _physical_request(numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED)
    narrow = _physical_request(index_dimension=64)
    candidate_ids = tuple(
        task.id
        for task in fast.task_plan.tasks
        if task.structure.principal_kind is TaskOperationKind.CONTRACT
        and not task.structure.streamed_axes
        and not task.structure.reduction_axes
    )
    narrow_candidate_ids = tuple(
        task.id
        for task in narrow.task_plan.tasks
        if task.structure.principal_kind is TaskOperationKind.CONTRACT
        and not task.structure.streamed_axes
        and not task.structure.reduction_axes
    )

    assert {item.task_id for item in select_sm100_projection_contract_lowerings(fast, task_ids=candidate_ids)} == set(
        candidate_ids
    )
    assert select_sm100_projection_contract_lowerings(source_ordered, task_ids=candidate_ids) == ()
    narrow_selected = select_sm100_projection_contract_lowerings(narrow, task_ids=narrow_candidate_ids)
    assert set(item.task_id for item in narrow_selected) < set(narrow_candidate_ids)


def test_sm100_reassociated_composition_rejects_source_ordered_request() -> None:
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=4,
    )
    oracle = build_msa_reference_oracle(config)
    request = msa_reference_physical_request(oracle, derive_msa_reference_dataflow(oracle))
    fast_request = msa_reference_physical_request(
        oracle,
        derive_msa_reference_dataflow(oracle, numerical_policy=PhysicalNumericalPolicy.FAST),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    selected = select_sm100_composed_task_lowerings(fast_request).candidates[-1]

    assert select_sm100_composed_task_lowerings(request).candidates == ()
    with pytest.raises(ValueError, match="requires FAST numerical policy"):
        lower_sm100_composed_physical_request(request, selected)


def test_sm100_selection_candidates_make_score_materialization_explicit() -> None:
    program = _program()
    materialized, fused = tuple(
        lower_sm100_projected_selection(program, schedule) for schedule in default_sm100_selection_schedules()
    )

    assert materialized.schedule.strategy is SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES
    assert materialized.token_score_materialization_bytes == 512 * 4 * 512 * 4
    assert materialized.block_score_materialization_bytes == 512 * 4 * 4 * 4
    assert fused.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
    assert fused.token_score_materialization_bytes == 0
    assert fused.block_score_materialization_bytes == 512 * 4 * 4 * 4
    assert materialized.relation_index_bytes == fused.relation_index_bytes == 512 * 4 * 4 * 4
    assert "Fold -> top-k Selection -> Relation" in fused.dump()
    assert "output=score_descending" in fused.dump()
    assert "underfilled=explicit_invalid_slots:invalid=-1" in fused.dump()
    assert "external semantics: none" in fused.dump()


def test_sm100_selection_requires_the_semantic_local_edge_rule() -> None:
    with pytest.raises(ValueError, match="forced-local"):
        lower_sm100_projected_selection(_program(force_local_block=False), default_sm100_selection_schedules()[0])
