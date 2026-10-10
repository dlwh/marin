# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the generic two-kernel Fold/Map/Contract SM90 attachment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_map_load_contract import Sm90MapLoadContractSchedule
from tile_lifetime.row_normalization_reference_pipeline import compile_row_statistic_reference_pipeline
from tile_lifetime.row_normalization_training import build_one_fold_contract_fixture, build_two_fold_contract_fixture
from tile_lifetime.sm90_fold_map_contract import attach_sm90_fold_map_contract
from tile_lifetime.tensor_program import FAST_FP32_SUM_ERROR_CONTRACT, NumericalPolicy

_torch_compile = torch.compile


def _compile_without_removed_limit(*compile_args, **compile_kwargs):
    kwargs = dict(compile_kwargs)
    kwargs.pop("recompile_limit", None)
    return _torch_compile(*compile_args, **kwargs)


torch.compile = _compile_without_removed_limit
transform_a_operand = importlib.import_module("quack.operand_transform").transform_a_operand


def _ordinary_jax(x, scale, weight, *, leading_fold: bool, epsilon: float):
    local = x.astype(jnp.float32)
    if leading_fold:
        local = jax.lax.reduce_precision(
            local - jnp.sum(local, axis=-1, keepdims=True) / x.shape[-1],
            exponent_bits=8,
            mantissa_bits=7,
        ).astype(jnp.bfloat16)
    squared = local.astype(jnp.float32) * local.astype(jnp.float32)
    state = jnp.sum(squared, axis=-1, keepdims=True, dtype=jnp.float32)
    inverse = jax.lax.rsqrt(state / x.shape[-1] + epsilon)
    mapped = jax.lax.reduce_precision(
        local.astype(jnp.float32) * inverse,
        exponent_bits=8,
        mantissa_bits=7,
    ).astype(jnp.bfloat16)
    mapped = jax.lax.reduce_precision(
        mapped.astype(jnp.float32) * scale.astype(jnp.float32),
        exponent_bits=8,
        mantissa_bits=7,
    ).astype(jnp.bfloat16)
    contracted = jnp.matmul(mapped, weight, preferred_element_type=jnp.float32)
    return jax.lax.reduce_precision(contracted, exponent_bits=8, mantissa_bits=7).astype(jnp.bfloat16)


def _torch_samples(function, *, warmup: int, samples: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return values


def _jax_samples(function, *, warmup: int, samples: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function().block_until_ready()
    values = []
    for _ in range(samples):
        before = time.perf_counter()
        for _ in range(iterations):
            result = function()
        result.block_until_ready()
        values.append((time.perf_counter() - before) * 1e3 / iterations)
    return values


def _interleaved_samples(torch_function, jax_function, *, warmup: int, samples: int, iterations: int):
    for _ in range(warmup):
        torch_function()
        jax_function().block_until_ready()
    torch.cuda.synchronize()
    generated = []
    oracle = []
    for sample in range(samples):
        order = ("generated", "oracle") if sample % 2 == 0 else ("oracle", "generated")
        for subject in order:
            if subject == "generated":
                generated.extend(_torch_samples(torch_function, warmup=0, samples=1, iterations=iterations))
            else:
                oracle.extend(_jax_samples(jax_function, warmup=0, samples=1, iterations=iterations))
    return generated, oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--folds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--threads", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--tile-m", type=int, choices=(64, 128))
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 192, 256))
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-pingpong", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epsilon = 1e-5
    source = replace(
        (build_one_fold_contract_fixture if args.folds == 1 else build_two_fold_contract_fixture)(
            rows=args.rows,
            hidden=args.hidden,
            features=args.features,
            epsilon=epsilon,
            dtype=DType.BF16,
        ),
        numerical_policy=NumericalPolicy.FAST,
    )
    planned = compile_row_statistic_reference_pipeline(source, row_tile=32, feature_tile=64)
    if (args.tile_m is None) != (args.tile_n is None):
        parser.error("--tile-m and --tile-n must be specified together")
    consumer_schedule = None
    if args.tile_m is not None:
        consumer_schedule = Sm90MapLoadContractSchedule(
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            cluster_m=args.cluster_m,
            cluster_n=args.cluster_n,
            pingpong=not args.no_pingpong,
        )
    generated = attach_sm90_fold_map_contract(
        source,
        planned.forward_request,
        threads=args.threads,
        consumer_schedule=consumer_schedule,
    )
    if generated.consumer is None:
        raise ValueError("benchmark requires the BF16 SM90 WGMMA Map-load/Contract component")
    extension = load_inline(
        name=f"shuttle_fold_map_contract_{generated.source_sha256[:12]}",
        cpp_sources="",
        cuda_sources=generated.source,
        extra_cuda_cflags=["-O3", "-arch=sm_90a"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    consumer_path = args.output / f"map_load_contract_{generated.consumer.source_sha256[:16]}.py"
    consumer_path.write_text(generated.consumer.source)
    specification = importlib.util.spec_from_file_location(consumer_path.stem, consumer_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import generated consumer {consumer_path}")
    consumer_module = importlib.util.module_from_spec(specification)
    sys.modules[consumer_path.stem] = consumer_module
    specification.loader.exec_module(consumer_module)

    rng = np.random.default_rng(20260903)
    arrays = (
        rng.normal(size=(args.rows, args.hidden)).astype(np.float32),
        rng.normal(size=(args.hidden,)).astype(np.float32),
        (rng.normal(size=(args.hidden, args.features)) / np.sqrt(args.hidden)).astype(np.float32),
    )
    torch_inputs = tuple(torch.asarray(value, device="cuda", dtype=torch.bfloat16) for value in arrays)
    jax_inputs = tuple(jnp.asarray(value, dtype=jnp.bfloat16) for value in arrays)
    leading_fold = args.folds == 2
    jax_function = jax.jit(
        lambda x, scale, weight: _ordinary_jax(
            x,
            scale,
            weight,
            leading_fold=leading_fold,
            epsilon=epsilon,
        )
    )

    output = torch.empty((args.rows, args.features), dtype=torch.bfloat16, device="cuda")
    physical_inputs = dict(zip((value.name for value in source.inputs), torch_inputs, strict=True))

    def consume(intermediates) -> torch.Tensor:
        physical_values = {
            **physical_inputs,
            **dict(zip(generated.intermediate_names[1:], intermediates, strict=True)),
        }
        transform_operands = {
            operand.parameter: physical_values[operand.value.name] for operand in generated.consumer.operands
        }
        activation_bundle = transform_a_operand(
            consumer_module.generated_transform,
            torch_inputs[0],
            transform_operands,
            generated.consumer.schedule.tile_m,
            generated.consumer.schedule.tile_k,
        )
        consumer_module.generated_epilogue.gemm(
            activation_bundle,
            torch_inputs[2].mT,
            output,
            None,
            epi_args={},
            transform_a=consumer_module.generated_transform,
            tile_M=generated.consumer.schedule.tile_m,
            tile_N=generated.consumer.schedule.tile_n,
            cluster_M=generated.consumer.schedule.cluster_m,
            cluster_N=generated.consumer.schedule.cluster_n,
            pingpong=generated.consumer.schedule.pingpong,
        )
        return output

    def producer_function():
        return extension.fold(*torch_inputs)

    profiled_intermediates = producer_function()

    def consumer_function() -> torch.Tensor:
        return consume(profiled_intermediates)

    def generated_function() -> torch.Tensor:
        return consume(producer_function())

    eager_output = torch.empty_like(output)

    def eager_function() -> torch.Tensor:
        local = torch_inputs[0].float()
        if leading_fold:
            local = (local - torch.sum(local, dim=-1, keepdim=True) / args.hidden).bfloat16()
        state = torch.sum(local.float() * local.float(), dim=-1, keepdim=True)
        inverse = torch.rsqrt(state / args.hidden + epsilon)
        mapped = (local.float() * inverse).bfloat16()
        mapped = (mapped * torch_inputs[1]).bfloat16()
        return torch.mm(mapped, torch_inputs[2], out=eager_output)

    actual = generated_function()
    produced_intermediates = extension.fold(*torch_inputs)
    produced_by_name = dict(zip(generated.intermediate_names[1:], produced_intermediates, strict=True))
    produced_fold_states = tuple(produced_by_name[name] for name in generated.fold_state_names)
    produced_row_materializations = tuple(produced_by_name[name] for name in generated.row_materialization_names)
    expected_row_sum = torch.sum(torch_inputs[0].float(), dim=-1)
    local_from_eager_state = (
        torch_inputs[0].float() - expected_row_sum[:, None] / args.hidden if leading_fold else torch_inputs[0].float()
    )
    if leading_fold:
        local_from_eager_state = local_from_eager_state.bfloat16()
        expected_fold_states = (
            expected_row_sum,
            torch.sum(local_from_eager_state.float() * local_from_eager_state.float(), dim=-1),
        )
    else:
        expected_fold_states = (torch.sum(local_from_eager_state.float() * local_from_eager_state.float(), dim=-1),)
    produced_local = torch_inputs[0].float()
    if leading_fold:
        produced_local = (produced_local - produced_row_materializations[0][:, None]).bfloat16()
    produced_mapped = (produced_local.float() * produced_row_materializations[-1][:, None]).bfloat16()
    produced_mapped = (produced_mapped * torch_inputs[1]).bfloat16()
    consumer_reference = torch.mm(produced_mapped, torch_inputs[2])
    expected = jax_function(*jax_inputs)
    actual_array = actual.float().cpu().numpy()
    expected_array = np.asarray(expected.astype(jnp.float32))
    difference = np.abs(actual_array - expected_array)
    eager_expected = eager_function()
    eager_difference = torch.abs(actual.float() - eager_expected.float())
    consumer_difference = torch.abs(actual.float() - consumer_reference.float())
    fold_state_errors = tuple(
        {
            "max_absolute_error": float(torch.max(torch.abs(actual_state - expected_state)).item()),
            "mean_absolute_error": float(torch.mean(torch.abs(actual_state - expected_state)).item()),
        }
        for actual_state, expected_state in zip(produced_fold_states, expected_fold_states, strict=True)
    )
    produced_row_sum = produced_fold_states[0] if leading_fold else None
    realized_fold_terms = [torch_inputs[0].float()]
    if leading_fold:
        assert produced_row_sum is not None
        realized_centered = (torch_inputs[0].float() - produced_row_sum[:, None] / args.hidden).bfloat16().float()
        realized_fold_terms.append(realized_centered * realized_centered)
    else:
        realized_fold_terms[0] = realized_fold_terms[0] * realized_fold_terms[0]
    fold_error_contract_checks = []
    coefficient = FAST_FP32_SUM_ERROR_CONTRACT.coefficient(args.hidden)
    for actual_state, terms in zip(produced_fold_states, realized_fold_terms, strict=True):
        exact_state = torch.sum(terms.double(), dim=-1)
        absolute_error = torch.abs(actual_state.double() - exact_state)
        bound = coefficient * torch.sum(torch.abs(terms.double()), dim=-1)
        fold_error_contract_checks.append(
            {
                "accepted": bool(torch.all(absolute_error <= bound).item()),
                "maximum_absolute_error": float(torch.max(absolute_error).item()),
                "maximum_bound": float(torch.max(bound).item()),
                "term_count": args.hidden,
            }
        )
    eager_close = bool(torch.allclose(actual.float(), eager_expected.float(), rtol=1e-2, atol=1e-2))
    jax_close = bool(np.allclose(actual_array, expected_array, rtol=1e-2, atol=1e-2))
    generated_samples = _torch_samples(generated_function, warmup=10, samples=args.samples, iterations=args.iterations)
    producer_samples = _torch_samples(producer_function, warmup=10, samples=args.samples, iterations=args.iterations)
    consumer_samples = _torch_samples(consumer_function, warmup=10, samples=args.samples, iterations=args.iterations)
    eager_samples = _torch_samples(eager_function, warmup=10, samples=args.samples, iterations=args.iterations)
    oracle_samples = _jax_samples(
        lambda: jax_function(*jax_inputs), warmup=10, samples=args.samples, iterations=args.iterations
    )
    interleaved_generated, interleaved_oracle = _interleaved_samples(
        generated_function,
        lambda: jax_function(*jax_inputs),
        warmup=5,
        samples=args.samples,
        iterations=args.iterations,
    )
    result = {
        "shape": {"rows": args.rows, "hidden": args.hidden, "features": args.features},
        "producer_threads": args.threads,
        "statistic": "second_moment" if args.folds == 1 else "centered_second_moment",
        "dtype": "bf16",
        "policy": "fast",
        "algebra_fingerprint": generated.algebra_fingerprint,
        "task_plan_fingerprint": generated.physical_request.task_plan.fingerprint,
        "event_tensor_plan_sha256": (
            hashlib.sha256(
                json.dumps(
                    generated.physical_request.canonical_record["event_plans"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        ),
        "schedule_fingerprint": generated.physical_request.schedules[0].fingerprint,
        "physical_request_fingerprint": generated.request_fingerprint,
        "generated_source_sha256": generated.source_sha256,
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "consumer_source_sha256": generated.consumer.source_sha256,
        "consumer_schedule_fingerprint": generated.consumer.schedule.fingerprint,
        "consumer_tile": {
            "m": generated.consumer.schedule.tile_m,
            "n": generated.consumer.schedule.tile_n,
            "k": generated.consumer.schedule.tile_k,
        },
        "producer_results": list(zip(generated.intermediate_names, generated.intermediate_storage, strict=True)),
        "performance_class": generated.performance_class,
        "max_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference)),
        "max_absolute_error_eager": float(torch.max(eager_difference).item()),
        "mean_absolute_error_eager": float(torch.mean(eager_difference).item()),
        "max_absolute_error_consumer_from_produced_states": float(torch.max(consumer_difference).item()),
        "mean_absolute_error_consumer_from_produced_states": float(torch.mean(consumer_difference).item()),
        "fold_state_errors": fold_state_errors,
        "fast_fold_error_contract": FAST_FP32_SUM_ERROR_CONTRACT.canonical_record,
        "fast_fold_error_contract_checks": fold_error_contract_checks,
        "weight_initialization": {"distribution": "normal", "standard_deviation": args.hidden**-0.5},
        "diagnostic_allclose_tolerance": {"rtol": 1e-2, "atol": 1e-2},
        "allclose_eager": eager_close,
        "allclose_ordinary_jax": jax_close,
        "generated_samples_ms": generated_samples,
        "producer_samples_ms": producer_samples,
        "consumer_samples_ms": consumer_samples,
        "eager_samples_ms": eager_samples,
        "ordinary_jax_samples_ms": oracle_samples,
        "interleaved_generated_samples_ms": interleaved_generated,
        "interleaved_ordinary_jax_samples_ms": interleaved_oracle,
        "generated_median_ms": statistics.median(generated_samples),
        "producer_median_ms": statistics.median(producer_samples),
        "consumer_median_ms": statistics.median(consumer_samples),
        "eager_median_ms": statistics.median(eager_samples),
        "ordinary_jax_median_ms": statistics.median(oracle_samples),
        "generated_over_eager": statistics.median(generated_samples) / statistics.median(eager_samples),
        "generated_over_ordinary_jax": statistics.median(generated_samples) / statistics.median(oracle_samples),
        "interleaved_generated_over_ordinary_jax": (
            statistics.median(interleaved_generated) / statistics.median(interleaved_oracle)
        ),
        "output_sha256": hashlib.sha256(actual.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "quack_version": importlib.metadata.version("quack-kernels"),
        "quack_revision": "84ef91df9bec87c7e4938517234fafb07ef844dd",
        "quack_patch": "lib/tile_lifetime/backends/h100/quack_fp32_row_scale.patch",
        "quack_patch_sha256": (
            hashlib.sha256(
                (Path(__file__).parents[1] / "backends/h100/quack_fp32_row_scale.patch").read_bytes()
            ).hexdigest()
        ),
        "gpu": torch.cuda.get_device_name(),
    }
    (args.output / "generated.cu").write_text(generated.source)
    (args.output / "generated_consumer.py").write_text(generated.consumer.source)
    (args.output / "physical_request.json").write_text(
        json.dumps(generated.physical_request.canonical_record, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
