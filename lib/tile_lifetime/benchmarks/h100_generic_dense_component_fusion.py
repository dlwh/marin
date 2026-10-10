# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare physical-provider dense composition with JAX on H100."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from tile_lifetime.backends.h100.cute_contract_component import sm90_contract_component_compiler
from tile_lifetime.backends.h100.cute_generic_tensor_program import (
    build_generic_component_multi_image_artifact,
    compile_generic_kernel_composition,
)
from tile_lifetime.backends.h100.cute_normalized_exp_component import sm90_normalized_exp_component_compiler
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.dense_region import DenseCompositionShape, DenseCompositionWeights, dense_composition_forward
from tile_lifetime.index_domain_canonicalization import normalized_exp_task_algebra_from_record
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import PhysicalTarget, canonical_fingerprint
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import NumericalPolicy


def _latencies(call, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        result = call()
        result[0].block_until_ready() if isinstance(result[0], jax.Array) else torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter()
        result = call()
        result[0].block_until_ready() if isinstance(result[0], jax.Array) else torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e3)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--model", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--feed-forward", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()

    rng = np.random.default_rng(93)
    batch = 1

    def array(shape: tuple[int, ...]) -> np.ndarray:
        return (0.05 * rng.normal(size=shape)).astype(np.float32)

    source = rng.normal(size=(batch, args.sequence, args.model)).astype(np.float32)
    weights = DenseCompositionWeights(
        input_scale=array((args.model,)),
        query=array((args.model, args.heads, args.feature)),
        key=array((args.model, args.heads, args.feature)),
        value=array((args.model, args.heads, args.feature)),
        output=array((args.heads, args.feature, args.model)),
        mlp_scale=array((args.model,)),
        gate=array((args.model, args.feed_forward)),
        up=array((args.model, args.feed_forward)),
        down=array((args.feed_forward, args.model)),
    )
    if args.dtype == "bf16":
        source = jnp.asarray(source, dtype=jnp.bfloat16)
        weights = jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.bfloat16), weights)
    shape = DenseCompositionShape(args.heads, args.feature, args.feature)

    def function(x, parameters):
        return dense_composition_forward(x, parameters, shape)

    imported = jax_function_to_tensor_program(function, source, weights, numerical_policy=NumericalPolicy.FAST)
    request = jax_tensor_program_executable_request(imported, tile_size=128, target=PhysicalTarget.SM90)
    attached = attach_sm90_contract_fold_schedule(request)
    task_program = normalized_exp_task_algebra_from_record(attached.request.algebra_record).task_program
    composition = lower_canonical_kernel_composition(attached.request)
    arrays = tuple(np.asarray(value) for value in jax.tree_util.tree_leaves((source, weights)))
    torch_inputs = tuple(
        torch.asarray(np.asarray(value, dtype=np.float32), device="cuda").to(
            torch.bfloat16 if args.dtype == "bf16" else torch.float32
        )
        for value in arrays
    )
    physical = compile_generic_kernel_composition(
        task_program,
        composition,
        example_inputs=torch_inputs,
        physical_component_compilers=(
            sm90_normalized_exp_component_compiler(attached.request),
            sm90_contract_component_compiler(),
        ),
    )
    physical_artifact = build_generic_component_multi_image_artifact(
        physical,
        function_prefix="shuttle_dense_normalized_component",
    )

    (physical_output,) = physical(*torch_inputs)
    expected = np.asarray(function(source, weights), dtype=np.float32)
    physical_error = np.abs(physical_output.float().cpu().numpy() - expected)

    jax_inputs = jax.tree.map(jax.device_put, (source, weights))
    jax_function = jax.jit(function)
    physical_samples = _latencies(lambda: physical(*torch_inputs), warmup=args.warmup, iterations=args.iterations)
    jax_samples = _latencies(
        lambda: (jax_function(*jax_inputs),),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    physical_median = statistics.median(physical_samples)
    jax_median = statistics.median(jax_samples)
    print(
        json.dumps(
            {
                "shape": {
                    "batch": batch,
                    "sequence": args.sequence,
                    "model": args.model,
                    "heads": args.heads,
                    "feature": args.feature,
                    "feed_forward": args.feed_forward,
                    "dtype": args.dtype,
                },
                "algebra_fingerprint": task_program.semantic_fingerprint,
                "task_plan_fingerprint": attached.request.task_plan.fingerprint,
                "event_plan_fingerprint": canonical_fingerprint(
                    {"event_plans": attached.request.canonical_record["event_plans"]}
                ),
                "schedule_fingerprints": tuple(schedule.fingerprint for schedule in attached.request.schedules),
                "physical_component_ids": tuple(
                    component.component_id
                    for component in physical.components
                    if component.physical_component is not None
                ),
                "wgmma_contract_component_ids": tuple(
                    component.component_id
                    for component in physical.components
                    if component.physical_component is not None
                    and component.physical_component.artifact.request.physical_template
                    == "sm90_wgmma_map_load_contract_v1"
                ),
                "physical_launches": sum(len(component.kernel_names) for component in physical.components),
                "event_count": len(attached.request.event_plans),
                "artifact_event_realizations": len(physical_artifact.launch_recipe.event_realizations),
                "cross_component_events": len(composition.concrete_events),
                "numerics": {
                    "physical_max_abs": float(np.max(physical_error)),
                    "physical_mean_abs": float(np.mean(physical_error)),
                },
                "latency_ms": {
                    "jax_median": jax_median,
                    "physical_median": physical_median,
                    "physical_over_jax": physical_median / jax_median,
                },
                "toolchain": {
                    "jax": jax.__version__,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
