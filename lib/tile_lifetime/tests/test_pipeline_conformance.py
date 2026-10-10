# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from tile_lifetime.dense_flow_planner import (
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.event_dataflow import TaskFamily, TaskRelation
from tile_lifetime.expert_executable_pipeline import ExpertRouteSchedule, generic_tensor_program_physical_request
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.moe_reference import MoEDebugConfig, moe_value_and_grad
from tile_lifetime.native_task_plan_adapter import (
    canonical_task_event_plan_from_native_request,
    executable_physical_request_from_native_request,
)
from tile_lifetime.physical_artifact_compiler import decode_gpu_physical_artifact_request
from tile_lifetime.physical_schedule_semantics import (
    PhysicalSemanticsStatus,
    compare_physical_schedule_semantics,
    project_physical_schedule_semantics,
)
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    PhysicalNumericalPolicy,
    PhysicalTarget,
    TaskOperationKind,
)
from tile_lifetime.pipeline_conformance import (
    ConformanceStatus,
    CrossPipelinePhysicalRefinement,
    CrossPipelineTaskRefinement,
    PhysicalScheduleCover,
    compare_cross_pipeline_conformance,
)
from tile_lifetime.relation import RelationPlacement
from tile_lifetime.task_plan_refinement import (
    CanonicalTaskPlanRefinement,
    TaskRefinementBinding,
    identity_task_refinement_binding,
    whole_program_task_event_refinement,
)
from tile_lifetime.tensor_program import NumericalPolicy, RelationPrimitive, primitive_outputs

_FIXTURES = Path(__file__).parent / "fixtures/native_schema_v3"


def test_native_attention_proves_task_event_refinement_and_compares_physical_schema() -> None:
    def attention(query, key, value):
        query_f32 = query.astype(jnp.float32)
        key_f32 = key.astype(jnp.float32)
        value_f32 = value.astype(jnp.float32)
        scores = jnp.einsum("bhgqd,bhkd->bhgqk", query_f32, key_f32) * jnp.float32(0.125)
        query_index = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
        key_index = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
        predicate = jnp.broadcast_to(key_index <= query_index, scores.shape)
        masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
        maximum = jnp.max(masked, axis=4, keepdims=True)
        numerator = jnp.exp(masked - maximum)
        denominator = jnp.sum(numerator, axis=4, keepdims=True)
        weights = numerator / denominator
        return jnp.einsum("bhgqk,bhkv->bhgqv", weights, value_f32).astype(jnp.bfloat16)

    shapes = ((1, 2, 2, 17, 8), (1, 2, 17, 8), (1, 2, 17, 7))
    program = jax_function_to_tensor_program(
        attention,
        *(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in shapes),
        numerical_policy=NumericalPolicy.FAST,
    )
    values = (*program.inputs, *(value for operation in program.operations for value in primitive_outputs(operation)))
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: axis.extent for value in values for axis in value.axes},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    event_program = eventize_composed_task_plan(task_plan)
    executable = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=event_program,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    native = decode_gpu_physical_artifact_request((_FIXTURES / "jax-bf16-causal-gqa-online-axis4-sm90.json").read_text())
    native_plan = canonical_task_event_plan_from_native_request(native, algebra_program=program)
    assert len(native_plan.task_plan.tasks) == 1
    native_task = native_plan.task_plan.tasks[0]
    logical_family = TaskFamily("logical.fused.region", (), placement=native_task.family.placement)
    logical_task = replace(native_task, family=logical_family)
    logical_plan = CanonicalTaskPlan((logical_task,), ())
    refinement = CanonicalTaskPlanRefinement(
        algebra_fingerprint=program.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
        coarse_plan=logical_plan,
        refined_plan=executable.task_plan,
        bindings=tuple(
            TaskRefinementBinding(
                refined_task_id=task.id,
                coarse_task_id=logical_task.id,
                coordinate_relation=TaskRelation.from_pairs(
                    task.family,
                    logical_task.family,
                    ((task.family.coordinates[0], ()),),
                ),
            )
            for task in executable.task_plan.tasks
        ),
        coarse_events=native_plan.events,
        refined_events=executable.event_plans,
    )
    native_physical = executable_physical_request_from_native_request(native, algebra_program=program)
    native_coordinate_refinement = CanonicalTaskPlanRefinement(
        algebra_fingerprint=program.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
        coarse_plan=logical_plan,
        refined_plan=native_plan.task_plan,
        bindings=(
            TaskRefinementBinding(
                refined_task_id=native_task.id,
                coarse_task_id=logical_task.id,
                coordinate_relation=TaskRelation.from_pairs(
                    native_task.family,
                    logical_task.family,
                    tuple((coordinate, ()) for coordinate in native_task.family.coordinates),
                ),
            ),
        ),
        coarse_events=(),
        refined_events=native_plan.events,
    )
    physical_refinement = CrossPipelinePhysicalRefinement(
        executable_task_refinement=refinement,
        native_coordinate_refinement=native_coordinate_refinement,
        schedule_covers=(
            PhysicalScheduleCover(
                native_schedule_index=0,
                executable_schedule_indices=tuple(range(len(executable.schedules))),
                parameter_equivalences=(("pipeline_depth", "serial_tiles"),),
            ),
        ),
        executable_physical_fingerprint=executable.fingerprint,
        native_physical_fingerprint=native_physical.fingerprint,
    )

    comparison = compare_cross_pipeline_conformance(
        executable,
        native,
        algebra_program=program,
        refinement=refinement,
        native_physical_request=native_physical,
        physical_refinement=physical_refinement,
    )

    assert comparison.algebra.status is ConformanceStatus.EXACT
    assert comparison.ownership_task_plan.status is ConformanceStatus.PROVEN_REFINEMENT
    assert comparison.event_tensor.status is ConformanceStatus.PROVEN_REFINEMENT
    assert comparison.physical_request.status is ConformanceStatus.PROVEN_REFINEMENT
    assert comparison.physical_request.native_fingerprint == native_physical.fingerprint
    assert physical_refinement.fingerprint in comparison.physical_request.detail
    assert not comparison.source_containers.same_encoding_identity
    assert comparison.accepted

    mismatched_parameters = replace(
        physical_refinement,
        schedule_covers=(
            replace(
                physical_refinement.schedule_covers[0],
                parameter_equivalences=(("pipeline_depth", "workgroup_threads"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="physical schedule parameter 'pipeline_depth'"):
        compare_cross_pipeline_conformance(
            executable,
            native,
            algebra_program=program,
            refinement=refinement,
            native_physical_request=native_physical,
            physical_refinement=mismatched_parameters,
        )

    missing_event_cover = replace(
        physical_refinement,
        executable_task_refinement=replace(refinement, refined_events=()),
    )
    with pytest.raises(ValueError, match="does not cover executable Event allocations"):
        compare_cross_pipeline_conformance(
            executable,
            native,
            algebra_program=program,
            refinement=missing_event_cover.executable_task_refinement,
            native_physical_request=native_physical,
            physical_refinement=missing_event_cover,
        )


def test_native_natural_jax_moe_proves_logical_quotient_but_not_physical_schedule() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    shapes = (
        (4, 4),
        (4, 3),
        (6, 4),
        (6, 4),
        (4, 6),
        (3, 6, 4),
        (3, 6, 4),
        (3, 4, 6),
        (4, 4),
    )
    program = jax_function_to_tensor_program(
        moe_value_and_grad(config),
        *(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in shapes),
    )
    schedule = ExpertRouteSchedule(
        source_placement=RelationPlacement(2, (0, 1, 0, 1), (0, 0, 1, 1)),
        destination_placement=RelationPlacement(2, (0, 0, 1, 1), (0, 1, 0, 1)),
        tile_rows=2,
        macrobatch_rows=4,
    )
    executable = generic_tensor_program_physical_request(program, schedule=schedule)
    with gzip.open(_FIXTURES / "jax-bf16-moe-value-and-grad-small-sm90.json.gz", "rt") as stream:
        native = decode_gpu_physical_artifact_request(stream.read())
    native_plan = canonical_task_event_plan_from_native_request(native, algebra_program=program)
    native_physical = executable_physical_request_from_native_request(native, algebra_program=program)
    cross_pipeline_refinement = CrossPipelineTaskRefinement(
        whole_program_task_event_refinement(
            program,
            executable.task_plan,
            target=PhysicalTarget.SM90,
            numerical_policy=PhysicalNumericalPolicy.FAST,
            refined_events=executable.event_plans,
            refined_event_programs=executable.event_programs,
        ),
        whole_program_task_event_refinement(
            program,
            native_plan.task_plan,
            target=PhysicalTarget.SM90,
            numerical_policy=PhysicalNumericalPolicy.FAST,
            refined_events=native_plan.events,
            refined_event_programs=native_plan.event_programs,
        ),
    )
    physical_semantics = compare_physical_schedule_semantics(
        executable,
        cross_pipeline_refinement.executable_refinement,
        native_physical,
        cross_pipeline_refinement.native_refinement,
    )

    comparison = compare_cross_pipeline_conformance(
        executable,
        native,
        algebra_program=program,
        native_physical_request=native_physical,
        cross_pipeline_refinement=cross_pipeline_refinement,
        physical_semantics=physical_semantics,
    )

    assert comparison.algebra.status is ConformanceStatus.EXACT
    assert comparison.ownership_task_plan.status is ConformanceStatus.PROVEN_REFINEMENT
    assert comparison.event_tensor.status is ConformanceStatus.PROVEN_REFINEMENT
    assert comparison.physical_request.status is ConformanceStatus.MISMATCH
    assert not physical_semantics.accepted
    assert len(physical_semantics.owners) == 1
    assert physical_semantics.owners[0].status is PhysicalSemanticsStatus.MISMATCH
    assert physical_semantics.owners[0].differing_fields == (
        "phases",
        "ownership_and_tiles",
        "traversal_and_order",
        "dtype_and_layout",
        "relation_accesses",
        "resources",
        "schedule_implementation",
        "event_allocation_and_realization",
        "generated_module_bindings:unavailable",
    )
    assert "owner semantics: 0=" in comparison.physical_request.detail
    assert comparison.native_adaptation_error is None
    assert comparison.algebra.native_fingerprint == program.semantic_fingerprint
    assert comparison.physical_request.native_fingerprint == native_physical.fingerprint
    assert (
        sum(
            task.operation_kinds == (TaskOperationKind.IMPLEMENTATION_HELPER,)
            for task in native_physical.task_plan.tasks
        )
        == 34
    )
    helper_task_ids = {
        task.id
        for task in native_plan.task_plan.tasks
        if task.operation_kinds == (TaskOperationKind.IMPLEMENTATION_HELPER,)
    }
    assert {refinement.helper_task_id for refinement in native_plan.implementation_helper_refinements} == helper_task_ids
    for refinement in native_plan.implementation_helper_refinements:
        for operation_ordinal, owner_task_id in refinement.operation_owners:
            assert operation_ordinal in native_plan.task_plan.tasks[owner_task_id].operation_ordinals
        assert {owner for owner, _ in refinement.coordinate_covers} == {
            owner for _, owner in refinement.operation_owners
        }
    assert len(native_plan.task_plan.dependence_programs) == 8
    assert len(native_plan.event_programs) == 8
    assert tuple(event.dependence_program_index for event in native_plan.event_programs) == tuple(range(8))
    assert all(
        isinstance(dependence.relation_program.operations[0], RelationPrimitive)
        for dependence in native_plan.task_plan.dependence_programs
    )
    assert len(native_plan.runtime_relation_dependence_sites) == 11
    assert sum(site.internalized for site in native_plan.runtime_relation_dependence_sites) == 3
    assert {
        site.dependence_program_index for site in native_plan.runtime_relation_dependence_sites if not site.internalized
    } == set(range(8))
    assert sorted(ordinal for task in native_physical.task_plan.tasks for ordinal in task.operation_ordinals) == list(
        range(len(program.operations))
    )
    assert not comparison.source_containers.same_encoding_identity
    assert not comparison.accepted

    with pytest.raises(ValueError, match="not bound to the selected logical quotient proof"):
        compare_cross_pipeline_conformance(
            executable,
            native,
            algebra_program=program,
            native_physical_request=native_physical,
            cross_pipeline_refinement=cross_pipeline_refinement,
            physical_semantics=replace(physical_semantics, logical_quotient_fingerprint="0" * 64),
        )


def test_physical_semantics_retains_cross_owner_event_allocation_and_realization() -> None:
    program = jax_function_to_tensor_program(
        lambda value: (value + jnp.float32(1)) * jnp.float32(2),
        jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    values = (*program.inputs, *(value for operation in program.operations for value in primitive_outputs(operation)))
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: axis.extent for value in values for axis in value.axes},
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
    refinement = CanonicalTaskPlanRefinement(
        algebra_fingerprint=program.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
        coarse_plan=task_plan,
        refined_plan=task_plan,
        bindings=tuple(
            identity_task_refinement_binding(
                task.id,
                task.id,
                refined_family=task.family,
                coarse_family=task.family,
            )
            for task in task_plan.tasks
        ),
        coarse_events=request.event_plans,
        refined_events=request.event_plans,
    )

    owners = project_physical_schedule_semantics(
        request,
        refinement,
        generated_module_bindings={index: f"module-{index}" for index in range(len(request.schedules))},
    )

    assert len(owners) == 2
    assert tuple(len(owner.concrete_events) for owner in owners) == (1, 1)
    source_plan = cast(dict[str, object], owners[0].concrete_events[0]["plan"])
    target_plan = cast(dict[str, object], owners[1].concrete_events[0]["plan"])
    source_required = cast(dict[str, object], source_plan["required_dependence"])
    target_required = cast(dict[str, object], target_plan["required_dependence"])
    source_relation = cast(dict[str, object], source_required["relation"])
    target_relation = cast(dict[str, object], target_required["relation"])
    assert source_relation["source_task"] == {
        "owner_task": 0,
        "local_task_slot": 0,
    }
    assert source_relation["target_task"] == {
        "owner_task": 1,
        "local_task_slot": None,
    }
    assert target_relation["source_task"] == {
        "owner_task": 0,
        "local_task_slot": None,
    }
    assert target_relation["target_task"] == {
        "owner_task": 1,
        "local_task_slot": 0,
    }
    assert all(owner.generated_module_bindings is not None for owner in owners)

    comparison = compare_physical_schedule_semantics(
        request,
        refinement,
        request,
        refinement,
        executable_module_bindings={index: f"module-{index}" for index in range(len(request.schedules))},
        native_module_bindings={index: f"module-{index}" for index in range(len(request.schedules))},
    )
    assert comparison.accepted
    assert tuple(owner.status for owner in comparison.owners) == (
        PhysicalSemanticsStatus.EXACT,
        PhysicalSemanticsStatus.EXACT,
    )

    unbound = compare_physical_schedule_semantics(request, refinement, request, refinement)
    assert not unbound.accepted
    assert tuple(owner.status for owner in unbound.owners) == (
        PhysicalSemanticsStatus.UNAVAILABLE,
        PhysicalSemanticsStatus.UNAVAILABLE,
    )
    assert all(owner.differing_fields == ("generated_module_bindings:unavailable",) for owner in unbound.owners)
