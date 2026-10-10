# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare strict ordered split-attention launch evidence; never run exported code."""

import argparse
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from cute_launch_argument_capture import capture_strict_cute_launch_report
from phase_failure_evidence import write_report
from strict_launch_comparison import (
    ORDERED_COMPILED_STRIDE_FIELDS,
    LaunchScheduleEvidence,
    attention_precheck_function,
    bounded_report_bytes,
    component_boundary_values,
    numerical_error_records,
    ordered_slot_layouts,
    persist_precheck_failure,
    prepare_ordered_precheck_plan,
)

from tile_lifetime.backends.h100.cute_generic_tensor_program import compile_generic_kernel_composition
from tile_lifetime.backends.h100.cute_normalized_exp_training_component import (
    CompiledSm90NormalizedExpDirectionalComponent,
    _bound_output_view,
    _packed_view,
    sm90_split_normalized_exp_component_compiler,
)
from tile_lifetime.backends.h100.cute_streaming_backward_emitter import (
    CompiledH100NormalizedExponentialBackward,
    export_h100_normalized_exponential_backward_artifact,
)
from tile_lifetime.cute_pipeline_artifact import launch_dependency_event_schedule
from tile_lifetime.sm90_backward_accumulation import Sm90AccumulationMode, validate_ordered_backward_recipe

MAX_ABSOLUTE_ERROR = 6.103515625e-5
MAX_BOUNDARY_BYTES = 32 * 1024 * 1024


def _errors(actual, expected):
    return numerical_error_records(
        tuple(value.float().cpu().numpy() for value in actual), expected, gate=MAX_ABSOLUTE_ERROR
    )


def _input_bytes(bound):
    # Transfer typed CPU storage; no device float conversion or contiguous normalization.
    return {
        name: hashlib.sha256(value.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, value in bound.items()
    }


def _case(causal: bool, expanded: bool, record: dict) -> None:
    shape = (1, 256, 2, 128)
    generator = torch.Generator(device="cuda").manual_seed(20260925)
    arguments = tuple(
        torch.randn(item, device="cuda", dtype=torch.bfloat16, generator=generator) * scale
        for item, scale in ((shape, 0.1), (shape, 0.1), (shape, 0.1), (shape, 0.001))
    )
    examples = tuple(jnp.asarray(item.float().cpu().numpy(), dtype=jnp.bfloat16) for item in arguments)
    function = attention_precheck_function(causal, expanded)
    plan = prepare_ordered_precheck_plan(examples, causal=causal, expanded=expanded)
    record.update(shape=shape, causal=causal, expanded=expanded, program_fingerprint=plan.program.semantic_fingerprint)
    provider = sm90_split_normalized_exp_component_compiler(plan, accumulation=Sm90AccumulationMode.ORDERED)
    compiled = compile_generic_kernel_composition(
        plan.program, plan.composition, example_inputs=arguments, physical_component_compilers=(provider,)
    )

    def objective(q, k, v, cotangent):
        return jnp.sum(function(q, k, v).astype(jnp.float32) * cotangent.astype(jnp.float32))

    jax_result = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))(*examples)
    expected = (np.asarray(jax_result[0]), *(np.asarray(value) for value in jax_result[1]))
    environment = compiled.execute_environment(*arguments)
    record["public_first_errors"] = _errors(tuple(environment[value] for value in plan.program.outputs), expected)
    candidates = [
        component.physical_component
        for component in compiled.components
        if isinstance(component.physical_component, CompiledSm90NormalizedExpDirectionalComponent)
        and isinstance(component.physical_component.compiled, CompiledH100NormalizedExponentialBackward)
    ]
    if len(candidates) != 1:
        raise ValueError("precheck requires one actual ordered split backward provider")
    directional = candidates[0]
    owners = [
        component
        for component in plan.composition.components
        if component.task_ids
        in {
            schedule.task_ids
            for schedule in plan.directional_schedules
            if schedule.parameters.get("direction") == "pullback"
        }
    ]
    if len(owners) != 1:
        raise ValueError("ordered precheck requires one typed backward component owner")
    boundary_inputs, boundary_outputs = component_boundary_values(plan.program, owners[0])
    if boundary_inputs != directional.live_inputs or boundary_outputs != directional.live_outputs:
        raise ValueError("compiled provider and typed component live boundaries differ")
    record["component_boundary"] = {
        "operations": owners[0].operation_ordinals,
        "inputs": [value.name for value in boundary_inputs],
        "outputs": [value.name for value in boundary_outputs],
    }
    backward = directional.compiled
    accumulation = backward.template.lowering.accumulation_plan
    contributor_counts = [len(accumulation.contributors(owner)) for owner in range(accumulation.owner_blocks)]
    if max(contributor_counts) < 2:
        raise ValueError("ordered replay precheck must exercise multiple contributors per owner")
    record["accumulation_plan"] = accumulation.record
    record["contributor_counts"] = contributor_counts
    record["contributor_ranks"] = [
        [list(coordinate), rank] for coordinate, rank in accumulation.contributor_ranks.items()
    ]
    if directional.output_transports:
        raise ValueError("precheck does not admit canonical output transport kernels")
    inputs = {}
    for role in ("query", "key", "value", "output", "output_cotangent", "log_sum_exp", "normalized_output"):
        value = directional.role_values[role]
        inputs[role] = _packed_view(
            environment[value], value, directional.physical_axes[value], directional.physical_shapes[value]
        )

    def call(bound):
        backward(
            *(
                bound[role]
                for role in ("query", "key", "value", "output", "output_cotangent", "log_sum_exp", "normalized_output")
            )
        )

    # Export exactly the low-level callable. Canonical copies are outside this claim.
    artifact = export_h100_normalized_exponential_backward_artifact(
        backward,
        function_prefix="ordered_attention_precheck",
        source_slot_shapes={
            **{name: tuple(value.shape) for name, value in inputs.items()},
            **{
                name: tuple(getattr(backward, name).shape)
                for name in ("query_cotangent", "key_cotangent", "value_cotangent")
            },
        },
    )
    recipe = artifact.launch_recipe
    if recipe is None:
        raise ValueError("ordered provider omitted its low-level recipe")
    validate_ordered_backward_recipe(
        backward.template.lowering.accumulation_plan,
        recipe,
        expanded_cotangent=backward.normalized_output_cotangent is not None,
    )
    actual_expanded = backward.normalized_output_cotangent is not None
    record["actual_expanded_roles"] = actual_expanded
    record["optional_vjp_roles_exercised"] = actual_expanded
    record["nonlinear_source_maps_may_remain_outside_provider"] = expanded
    saved_layouts = ordered_slot_layouts(
        recipe,
        {
            slot.name: getattr(backward, ORDERED_COMPILED_STRIDE_FIELDS[slot.name])
            for slot in recipe.slots
            if slot.name in ORDERED_COMPILED_STRIDE_FIELDS
        },
    )
    saved_schedule = LaunchScheduleEvidence(
        recipe.digest, launch_dependency_event_schedule(recipe.launches, slots=recipe.slots), saved_layouts
    )
    record["saved_launch_schedule"] = {
        "recipe_digest": saved_schedule.recipe_digest,
        "schedule": saved_schedule.schedule,
    }
    boundary = sum(slot.required_bytes for slot in recipe.slots)
    if boundary > MAX_BOUNDARY_BYTES:
        raise ValueError("precheck boundary exceeds explicit capture budget")
    record["boundary_bytes"] = boundary

    def tensors(bound):
        return {
            slot.name: bound[slot.name] if slot.name in bound else getattr(backward, slot.name) for slot in recipe.slots
        }

    original_input_bytes = _input_bytes(inputs)
    record["input_byte_checks"] = []

    def check_inputs(bound, expected_bytes, phase):
        actual_bytes = _input_bytes(bound)
        record["input_byte_checks"].append({"phase": phase, "before": expected_bytes, "after": actual_bytes})
        if actual_bytes != expected_bytes:
            raise ValueError(f"read-only caller inputs changed bytes during {phase}")

    expected_low = tuple(
        value.detach().clone() for value in (backward.query_cotangent, backward.key_cotangent, backward.value_cotangent)
    )

    def check_low():
        actual = (backward.query_cotangent, backward.key_cotangent, backward.value_cotangent)
        errors = _errors(actual, tuple(value.float().cpu().numpy() for value in expected_low))
        if not all(
            torch.equal(left.view(torch.int16), right.view(torch.int16))
            for left, right in zip(actual, expected_low, strict=True)
        ):
            raise ValueError(f"ordered low-level replay changed boundary bytes: {errors}")
        return errors

    record["ordinary_replays"] = []
    for _ in range(2):
        call(inputs)
        record["ordinary_replays"].append(check_low())
        check_inputs(inputs, original_input_bytes, "ordinary")
    capture, graph = capture_strict_cute_launch_report(lambda: call(inputs), recipe, tensors(inputs), saved_schedule)
    record["capture"] = capture
    record["graph_replays"] = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        record["graph_replays"].append(check_low())
        check_inputs(inputs, original_input_bytes, "graph")
    # Keep original allocations alive; preserve shape/strides/values while rebinding caller inputs.
    relocated = {
        name: (
            torch.empty_strided(tuple(value.shape), tuple(value.stride()), dtype=value.dtype, device=value.device).copy_(
                value
            )
        )
        for name, value in inputs.items()
    }
    if any(
        relocated[name].data_ptr() == value.data_ptr() or not torch.equal(relocated[name], value)
        for name, value in inputs.items()
    ):
        raise ValueError("relocated caller input lacks a changed address or identical values")
    relocated_input_bytes = _input_bytes(relocated)
    if relocated_input_bytes != original_input_bytes:
        raise ValueError("relocated inputs changed exact bytes")
    call(relocated)
    record["relocated_errors"] = check_low()
    check_inputs(relocated, relocated_input_bytes, "relocated_ordinary")
    check_inputs(inputs, original_input_bytes, "original_after_relocation")
    moved, moved_graph = capture_strict_cute_launch_report(
        lambda: call(relocated), recipe, tensors(relocated), saved_schedule
    )
    record["relocated_capture"] = moved
    moved_graph.replay()
    torch.cuda.synchronize()
    record["relocated_graph_errors"] = check_low()
    check_inputs(relocated, relocated_input_bytes, "relocated_graph")
    check_inputs(inputs, original_input_bytes, "original_after_relocated_graph")
    record["canonical_copy_errors"] = []
    for role, source in zip(
        ("query_cotangent", "key_cotangent", "value_cotangent"),
        (backward.query_cotangent, backward.key_cotangent, backward.value_cotangent),
        strict=True,
    ):
        value = directional.role_values[role]
        canonical = _bound_output_view(source, directional.output_views[value])
        record["canonical_copy_errors"].extend(_errors((canonical,), (environment[value].float().cpu().numpy(),)))
    record["public_final_errors"] = _errors(compiled(*arguments), expected)
    check_inputs(inputs, original_input_bytes, "after_public_composition")
    record["native_execution_accepted"] = False
    record["status"] = "metadata_and_generated_checks_complete_alignment_unresolved"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "gate": MAX_ABSOLUTE_ERROR,
        "cases": [],
        "scope": "Compiler-owned generated calls and low-level ABI capture; canonical output copies checked separately.",
        "exported_or_native_execution": False,
        "prepared_coverage": "Causal, equal extents, multiple contributors; identity and nonlinear surrounding source.",
        "coverage_gaps": [
            "Plain FULL source preparation rejects before provider selection: explicit owner coordinate required.",
            "A retained fully covering predicate would select LOWER_TRIANGULAR, not prove FULL traversal.",
            "The prepared nonlinear source retains an identity physical VJP; optional VJP capture is not promised.",
        ],
    }
    try:
        for causal, expanded in ((True, False), (True, True)):
            case = {}
            report["cases"].append(case)
            _case(causal, expanded, case)
            bounded_report_bytes(report)
            write_report(args.json_output, report)
        report["status"] = "complete_with_unresolved_alignment_evidence"
        bounded_report_bytes(report)
        write_report(args.json_output, report)
    except Exception as error:
        persist_precheck_failure(args.json_output, report, error)
        raise


if __name__ == "__main__":
    main()
