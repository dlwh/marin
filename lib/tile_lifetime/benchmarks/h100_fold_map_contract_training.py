# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise generic multi-output AxisFold training composition on H100."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.fold_map_contract_training_provider import provide_fold_map_contract_training
from tile_lifetime.jax_axis_fold_ffi import register_cuda_axis_fold_ffi
from tile_lifetime.jax_fold_map_contract_training import execute_fold_map_contract_training_jax
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.row_normalization_jax import RowProjectionConfig, layer_projection, projection_training_step
from tile_lifetime.row_normalization_training import compile_fold_map_contract_training
from tile_lifetime.tensor_program import FAST_FP32_SUM_ERROR_CONTRACT, NumericalPolicy


def _compile(source: str, directory: Path, *, architecture: str, nvcc: Path) -> ctypes.CDLL:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "generated_axis_fold_pipeline.cu"
    library_path = directory / "generated_axis_fold_pipeline.so"
    source_path.write_text(source + "\n")
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    command = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        f"-arch={architecture}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(include_directory),
        str(source_path),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
    )
    subprocess.run(command, check=True)
    return ctypes.CDLL(str(library_path))


def _errors(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {"maximum_absolute": float(difference.max(initial=0.0)), "mean_absolute": float(difference.mean())}


def _latency(function, arguments: tuple[jax.Array, ...], *, iterations: int) -> float:
    for _ in range(5):
        jax.block_until_ready(function(*arguments))
    started = time.perf_counter()
    for _ in range(iterations):
        result = function(*arguments)
    jax.block_until_ready(result)
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--input-scale", type=float, default=0.01)
    args = parser.parse_args()
    config = RowProjectionConfig(rows=args.rows, hidden=args.hidden, features=args.features, epsilon=2e-4)
    project = layer_projection(config)
    key = jax.random.key(43)
    keys = jax.random.split(key, 4)
    arguments = (
        (jax.random.normal(keys[0], (args.rows, args.hidden)) * args.input_scale).astype(jnp.float32),
        (jax.random.normal(keys[1], (args.hidden,)) * args.input_scale).astype(jnp.float32),
        (jax.random.normal(keys[2], (args.hidden, args.features)) * args.input_scale).astype(jnp.float32),
    )
    output_cotangent = (jax.random.normal(keys[3], (args.rows, args.features)) * args.input_scale).astype(jnp.float32)
    source = jax_function_to_tensor_program(project, *arguments, numerical_policy=NumericalPolicy.FAST)
    plan = compile_fold_map_contract_training(source)
    provider = provide_fold_map_contract_training(plan, target_name="shuttle.generic_fold_training")
    if len(provider.regions) != 1:
        raise ValueError("saved-state centered training should form one reverse Fold region")
    region = provider.regions[0]
    library = _compile(
        region.generated.source,
        args.output / "generated",
        architecture=args.architecture,
        nvcc=args.nvcc,
    )
    register_cuda_axis_fold_ffi(region.generated, library)
    cotangent_name = plan.automatic_adjoint.output_cotangents[0].name

    @jax.jit
    def generated(x, scale, weight, cotangent):
        outputs, gradients = execute_fold_map_contract_training_jax(
            provider,
            plan,
            {value.name: argument for value, argument in zip(plan.source.inputs, (x, scale, weight), strict=True)},
            {cotangent_name: cotangent},
        )
        return (*outputs, *gradients)

    natural = jax.jit(projection_training_step(project))
    generated_compiled = generated.lower(*arguments, output_cotangent).compile()
    natural_compiled = natural.lower(*arguments, output_cotangent).compile()
    actual = generated_compiled(*arguments, output_cotangent)
    natural_output, natural_gradients = natural_compiled(*arguments, output_cotangent)
    expected = (natural_output, *natural_gradients)
    jax.block_until_ready((actual, expected))
    errors = tuple(_errors(left, right) for left, right in zip(actual, expected, strict=True))
    if any(error["maximum_absolute"] > 2e-5 for error in errors):
        raise ValueError(f"generated training exceeded unchanged 2e-5 tolerance: {errors}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "shape": {"rows": args.rows, "hidden": args.hidden, "features": args.features},
        "input_scale": args.input_scale,
        "errors": dict(zip(("output", "input_gradient", "scale_gradient", "weight_gradient"), errors, strict=True)),
        "latency_ms": {
            "generated_composed": _latency(
                generated_compiled, (*arguments, output_cotangent), iterations=args.iterations
            ),
            "natural_jax": _latency(natural_compiled, (*arguments, output_cotangent), iterations=args.iterations),
        },
        "artifact": {
            "source_sha256": region.generated.source_sha256,
            "semantic_fingerprints": region.generated.semantic_fingerprints,
            "target": region.generated.target_name,
            "task_ids": provider.task_ids,
            "event_count": provider.event_count,
            "saved_values": provider.saved_values,
            "recomputed_values": provider.recomputed_values,
        },
        "fast_fold_error_contract": FAST_FP32_SUM_ERROR_CONTRACT.canonical_record,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "generated" / "generated_training_hlo.txt").write_text(generated_compiled.as_text())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
