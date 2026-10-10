# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark natural JAX value-and-grad lowered through canonical SM90 components."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flash_attn.cute.interface import flash_attn_func

from tile_lifetime.backends.h100.cute_generic_tensor_program import (
    build_generic_component_multi_image_artifact,
    compile_generic_kernel_composition,
)
from tile_lifetime.backends.h100.cute_normalized_exp_training_component import (
    CompiledSm90NormalizedExpDirectionalComponent,
    CompiledSm90NormalizedExpValueAndGradComponent,
    sm90_normalized_exp_value_and_grad_component_compiler,
    sm90_split_normalized_exp_component_compiler,
)
from tile_lifetime.h100_streaming_lowering import Sm90ProducerOperandOrientation
from tile_lifetime.jax_normalized_exp_contract_training import (
    prepare_sm90_jax_owned_normalized_exp_value_and_grad,
    prepare_sm90_split_normalized_exp_value_and_grad,
    trace_jax_owned_normalized_exp_value_and_grad,
)
from tile_lifetime.physical_task_contract import PhysicalTarget, canonical_fingerprint
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import FAST_FP32_SUM_ERROR_CONTRACT, NumericalPolicy

_MAX_ABSOLUTE_ERROR = 0.125
_MAX_RELATIVE_ERROR = 1.0e-3


def _samples(function, *, warmups: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    return samples


def _function(sequence: int, *, score_transform: str, predicate_offset: int):
    def function(query, key, value):
        score = jnp.einsum("bqhgd,bkhd->bhgqk", query, key) * (128**-0.5)
        if score_transform == "tanh":
            score = 10.0 * jnp.tanh(score / 10.0)
        elif score_transform == "affine":
            score = 1.25 * score + 0.2
        query_position = jnp.arange(sequence)[:, None]
        key_position = jnp.arange(sequence)[None, :]
        score = jnp.where(key_position <= query_position + predicate_offset, score, -jnp.inf)
        maximum = jnp.max(score, axis=-1, keepdims=True)
        exponential = jnp.exp(score - maximum)
        probability = exponential / jnp.sum(exponential, axis=-1, keepdims=True)
        return jnp.einsum("bhgqk,bkhv->bqhgv", probability, value)

    return function


def _error(actual: torch.Tensor, expected: np.ndarray) -> dict[str, float]:
    actual_array = actual.float().cpu().numpy()
    expected_array = expected.astype(np.float32)
    difference = np.abs(actual_array - expected_array)
    result = {
        "maximum_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference)),
        "allclose_atol_0.125_rtol_0.001": bool(
            np.allclose(
                actual_array,
                expected_array,
                atol=_MAX_ABSOLUTE_ERROR,
                rtol=_MAX_RELATIVE_ERROR,
            )
        ),
    }
    if actual_array.ndim == 0:
        result.update(actual=float(actual_array), reference=float(expected_array))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--score-transform", choices=("identity", "tanh", "affine"), default="identity")
    parser.add_argument("--predicate-offset", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--split", action="store_true")
    parser.add_argument(
        "--producer-orientation",
        choices=tuple(orientation.value for orientation in Sm90ProducerOperandOrientation),
        default=Sm90ProducerOperandOrientation.OWNER_STREAMED.value,
    )
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the SM90 value-and-grad benchmark requires CUDA")

    batch, key_value_heads, group_size, dimension = 1, 8, 4, 128
    generator = torch.Generator(device="cuda").manual_seed(20260903)
    query = torch.randn(
        (batch, args.sequence, key_value_heads, group_size, dimension),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        (batch, args.sequence, key_value_heads, dimension),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn_like(key)
    output_cotangent = torch.randn_like(query)
    function = _function(
        args.sequence,
        score_transform=args.score_transform,
        predicate_offset=args.predicate_offset,
    )
    numpy_inputs = tuple(tensor.float().cpu().numpy() for tensor in (query, key, value, output_cotangent))
    jax_examples = tuple(jnp.asarray(array, dtype=jnp.bfloat16) for array in numpy_inputs)
    trace = trace_jax_owned_normalized_exp_value_and_grad(
        function,
        *jax_examples[:3],
        output_cotangent=jax_examples[3],
        numerical_policy=NumericalPolicy.FAST,
    )
    source_request = jax_tensor_program_executable_request(
        trace.value_and_grad,
        tile_size=128,
        target=PhysicalTarget.SM90,
    )
    if args.split:
        plan = prepare_sm90_split_normalized_exp_value_and_grad(source_request, trace, tile_size=128)
        orientation = Sm90ProducerOperandOrientation(args.producer_orientation)
        if orientation is Sm90ProducerOperandOrientation.STREAMED_OWNER:
            forward_program = replace(
                plan.forward_plan.program,
                schedule=replace(
                    plan.forward_plan.program.schedule,
                    consumer_register_source=False,
                    producer_operand_orientation=orientation,
                ),
            )
            plan = replace(plan, forward_plan=replace(plan.forward_plan, program=forward_program))
        physical_program = plan.program
        component_compiler = sm90_split_normalized_exp_component_compiler(plan)
        replacement_fingerprint = canonical_fingerprint(
            {
                "covered_source_operations": plan.replacement.covered_source_operations,
                "source_boundary_inputs": tuple(value.name for value in plan.replacement.source_boundary_inputs),
                "source_boundary_outputs": tuple(value.name for value in plan.replacement.source_boundary_outputs),
                "saved_values": tuple(value.name for value in plan.replacement.saved_values),
            }
        )
    else:
        plan = prepare_sm90_jax_owned_normalized_exp_value_and_grad(source_request, trace, tile_size=128)
        physical_program = plan.physical_program
        component_compiler = sm90_normalized_exp_value_and_grad_component_compiler(plan)
        replacement_fingerprint = plan.pullback_plan.replacement.fingerprint
    compiled = compile_generic_kernel_composition(
        physical_program,
        plan.composition,
        example_inputs=(query, key, value, output_cotangent),
        physical_component_compilers=(component_compiler,),
    )
    artifact = build_generic_component_multi_image_artifact(
        compiled,
        function_prefix=f"shuttle_jax_value_and_grad_{args.score_transform}",
    )
    scalar_fold_component = next(
        component
        for component in compiled.components
        if any(primitive.private_scratch is not None for primitive in component.primitives)
    )
    scalar_fold_primitive = next(
        primitive for primitive in scalar_fold_component.primitives if primitive.private_scratch is not None
    )
    assert scalar_fold_primitive.private_scratch is not None
    scalar_fold_launches = tuple(
        launch
        for launch in artifact.launch_recipe.launches
        if scalar_fold_component.component_id
        in {
            component.id
            for component in plan.composition.components
            if set(launch.source_tasks) <= set(component.task_ids)
        }
    )

    generated_outputs: tuple[torch.Tensor, ...] = ()

    def generated() -> None:
        nonlocal generated_outputs
        generated_outputs = compiled(query, key, value, output_cotangent)

    jax_arguments = tuple(jax.device_put(value) for value in jax_examples)

    def objective(q, k, v, cotangent):
        return jnp.sum(function(q, k, v).astype(jnp.float32) * cotangent.astype(jnp.float32))

    jax_value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))
    jax_result = jax_value_and_grad(*jax_arguments)
    expected = (np.asarray(jax_result[0]), *(np.asarray(value) for value in jax_result[1]))
    generated()
    torch.cuda.synchronize()
    correctness = {
        name: _error(actual, reference)
        for name, actual, reference in zip(("loss", "query", "key", "value"), generated_outputs, expected, strict=True)
    }
    attention_component = next(
        component
        for component in compiled.components
        if isinstance(
            component.physical_component,
            (CompiledSm90NormalizedExpValueAndGradComponent, CompiledSm90NormalizedExpDirectionalComponent),
        )
        and (
            isinstance(component.physical_component, CompiledSm90NormalizedExpValueAndGradComponent)
            or "output" in component.physical_component.role_values
        )
    )
    physical_component = attention_component.physical_component
    assert isinstance(
        physical_component,
        (CompiledSm90NormalizedExpValueAndGradComponent, CompiledSm90NormalizedExpDirectionalComponent),
    )
    diagnostic_environment = compiled.execute_environment(query, key, value, output_cotangent)
    forward_output = (
        physical_component.output
        if isinstance(physical_component, CompiledSm90NormalizedExpValueAndGradComponent)
        else physical_component.role_values["output"]
    )
    generated_forward = diagnostic_environment[forward_output]
    generated_forward = (
        generated_forward.permute(0, 4, 1, 3, 2) if args.split else generated_forward.permute(0, 3, 1, 2, 4)
    )
    expected_forward = np.asarray(function(*jax_arguments[:3]))
    correctness["forward"] = _error(generated_forward, expected_forward)
    generated_forward_array = generated_forward.float().cpu().numpy()
    output_cotangent_array = output_cotangent.float().cpu().numpy()
    expected_forward_array = expected_forward.astype(np.float32)
    realized_terms = (generated_forward_array * output_cotangent_array).astype(np.float32)
    realized_serial_loss = float(np.sum(realized_terms, dtype=np.float32))
    generated_loss = float(generated_outputs[0].float().cpu())
    reference_loss = float(expected[0])
    forward_error_bound = float(
        np.sum(
            np.abs(output_cotangent_array) * np.abs(generated_forward_array - expected_forward_array),
            dtype=np.float64,
        )
    )
    fold_rounding_bound = FAST_FP32_SUM_ERROR_CONTRACT.absolute_error_bound(
        term_count=realized_terms.size,
        absolute_sum=float(np.sum(np.abs(realized_terms), dtype=np.float64)),
    )
    observed_loss_error = abs(generated_loss - reference_loss)
    loss_diagnostic = {
        "generated_loss": generated_loss,
        "independent_jax_loss": reference_loss,
        "serial_sum_of_generated_forward_terms": realized_serial_loss,
        "hierarchical_reduction_difference": generated_loss - realized_serial_loss,
        "forward_realization_objective_difference": realized_serial_loss - reference_loss,
        "observed_absolute_error": observed_loss_error,
        "observed_relative_error": observed_loss_error / abs(reference_loss),
        "forward_error_propagation_bound": forward_error_bound,
        "hierarchical_fold_rounding_bound": fold_rounding_bound,
        "propagated_fast_objective_bound": forward_error_bound + fold_rounding_bound,
        "observed_error_within_propagated_fast_bound": bool(
            observed_loss_error <= forward_error_bound + fold_rounding_bound
        ),
    }

    generated_samples = _samples(generated, warmups=args.warmups, repeats=args.repeats, iterations=args.iterations)
    oracle_samples: list[float] | None = None
    if args.predicate_offset == 0 and args.score_transform in {"identity", "tanh"}:
        flat_query = (
            query.view(batch, args.sequence, key_value_heads * group_size, dimension).detach().requires_grad_(True)
        )
        oracle_key = key.detach().requires_grad_(True)
        oracle_value = value.detach().requires_grad_(True)
        flat_cotangent = output_cotangent.view_as(flat_query)
        softcap = 10.0 if args.score_transform == "tanh" else 0.0

        def oracle() -> None:
            result = flash_attn_func(
                flat_query,
                oracle_key,
                oracle_value,
                softmax_scale=float(jnp.asarray(128**-0.5, dtype=jnp.bfloat16)),
                causal=True,
                softcap=softcap,
            )
            output = result[0] if isinstance(result, tuple) else result
            loss = torch.sum(output.float() * flat_cotangent.float())
            torch.autograd.grad(
                loss,
                (flat_query, oracle_key, oracle_value),
                retain_graph=False,
            )

        oracle_samples = _samples(oracle, warmups=args.warmups, repeats=args.repeats, iterations=args.iterations)

    generated_median = statistics.median(generated_samples)
    oracle_median = None if oracle_samples is None else statistics.median(oracle_samples)
    result = {
        "shape": {
            "batch": batch,
            "sequence": args.sequence,
            "key_value_heads": key_value_heads,
            "explicit_group": group_size,
            "dimension": dimension,
        },
        "score_transform": args.score_transform,
        "producer_orientation": args.producer_orientation,
        "predicate_offset": args.predicate_offset,
        "trace": plan.trace.canonical_record,
        "source_request_fingerprint": plan.source_request.fingerprint,
        "replacement_fingerprint": replacement_fingerprint,
        "physical_program_fingerprint": physical_program.semantic_fingerprint,
        "task_plan_fingerprint": plan.physical_request.task_plan.fingerprint,
        "event_plan_fingerprint": canonical_fingerprint(
            tuple(lowering.canonical_record for lowering in plan.preselection_event_lowerings)
        ),
        "component_composition_fingerprint": plan.composition.fingerprint,
        "artifact_request_digest": artifact.request.digest,
        "crossing_event_names": plan.crossing_event_names,
        "component_count": len(plan.composition.components),
        "launch_count": len(artifact.launch_recipe.launches),
        "event_realization_count": len(artifact.launch_recipe.event_realizations),
        "hierarchical_scalar_fold": {
            "partial_count": scalar_fold_primitive.private_scratch.numel(),
            "partial_tile_elements": 4096,
            "launches": tuple(
                {
                    "name": launch.name,
                    "grid": launch.grid,
                    "block": launch.block,
                    "source_tasks": launch.source_tasks,
                }
                for launch in scalar_fold_launches
            ),
        },
        "correctness": correctness,
        "loss_diagnostic": loss_diagnostic,
        "timing_ms": {
            "generated_samples": generated_samples,
            "generated_median": generated_median,
            "oracle_samples": oracle_samples,
            "oracle_median": oracle_median,
            "generated_over_oracle": None if oracle_median is None else generated_median / oracle_median,
        },
        "toolchain": {
            "gpu": torch.cuda.get_device_name(),
            "jax": jax.__version__,
            "torch": torch.__version__,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
