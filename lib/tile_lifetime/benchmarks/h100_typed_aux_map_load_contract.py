# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the generic typed multi-auxiliary QuACK Map-load/Contract slice."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_map_load_contract import generate_sm90_map_load_contract
from tile_lifetime.tensor_program import (
    ProgramValue,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    TensorAxis,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_unary,
)


@dataclass(frozen=True)
class ShapeResult:
    rows: int
    reduction: int
    features: int
    generated_median_ms: float
    generated_samples_ms: tuple[float, ...]
    eager_map_contract_median_ms: float
    eager_map_contract_samples_ms: tuple[float, ...]
    bare_gemm_median_ms: float
    bare_gemm_samples_ms: tuple[float, ...]
    generated_over_eager_map_contract: float
    generated_over_bare_gemm: float
    max_absolute_error_full_reference: float
    mean_absolute_error_full_reference: float
    generated_tflops: float
    generated_source_sha256: str
    request_sha256: str
    schedule_fingerprint: str


def _samples(function, *, warmup: int, samples: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    return timings


def _benchmark_shape(
    rows: int,
    reduction: int,
    features: int,
    *,
    samples: int,
    iterations: int,
    output: Path,
) -> ShapeResult:
    row_axis = TensorAxis(0, rows, "row")
    reduction_axis = TensorAxis(1, reduction, "reduction")
    feature_axis = TensorAxis(2, features, "feature")
    activation = ProgramValue("activation", (row_axis, reduction_axis), DType.BF16)
    fold_state = ProgramValue("fold_state", (row_axis,), DType.FP32)
    feature_scale = ProgramValue("feature_scale", (reduction_axis,), DType.BF16)
    weight = ProgramValue("weight", (reduction_axis, feature_axis), DType.BF16)
    output_value = ProgramValue("output", (row_axis, feature_axis), DType.BF16)
    inverse = scalar_unary(
        ScalarExpressionKind.RSQRT,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(fold_state.name),
                scalar_constant(float(reduction)),
            ),
            scalar_constant(1e-5),
        ),
    )
    standardized = scalar_convert(
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(activation.name), inverse),
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )
    expression = scalar_convert(
        scalar_binary(ScalarExpressionKind.MULTIPLY, standardized, scalar_input(feature_scale.name)),
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )
    generated = generate_sm90_map_load_contract(
        expression=expression,
        activation=activation,
        leaves=(activation, fold_state, feature_scale),
        row_axis=row_axis,
        reduction_axis=reduction_axis,
        weight_feature_axis=feature_axis,
        output_row_axis=row_axis,
        output_feature_axis=feature_axis,
        weight=weight,
        output=output_value,
    )
    request = {
        "algebra": "bf16(bf16(activation * rsqrt(row_fold / reduction_extent + epsilon)) * feature_scale) -> Contract",
        "activation_axes": ["row", "reduction"],
        "weight_axes": ["reduction", "feature"],
        "output_axes": ["row", "feature"],
        "shape": [rows, reduction, features],
        "activation_dtype": "bf16",
        "fold_state_dtype": "fp32",
        "feature_scale_dtype": "bf16",
        "weight_dtype": "bf16",
        "output_dtype": "bf16",
        "aux_delivery": [asdict(operand) | {"value": operand.value.name} for operand in generated.operands],
        "schedule": asdict(generated.schedule),
    }
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
    request_sha256 = hashlib.sha256(request_json.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"m{rows}_k{reduction}_n{features}"
    (output / f"{stem}_request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    (output / f"{stem}_generated.py").write_text(generated.source)
    namespace: dict[str, object] = {}
    exec(generated.source, namespace)
    transform = namespace["generated_transform"]
    epilogue = namespace["generated_epilogue"]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260903 + rows + reduction + features)
    x = torch.randn((rows, reduction), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight_tensor = torch.randn((reduction, features), device="cuda", dtype=torch.bfloat16, generator=generator)
    fold_state_fp32 = torch.sum(x.float() * x.float(), dim=1)
    feature_scale_tensor = torch.randn((reduction,), device="cuda", dtype=torch.bfloat16, generator=generator)
    transform_a_operand = importlib.import_module("quack.operand_transform").transform_a_operand

    transformed = transform_a_operand(
        transform,
        x,
        {"operand_0": fold_state_fp32, "operand_1": feature_scale_tensor},
        generated.schedule.tile_m,
        generated.schedule.tile_k,
    )
    actual = torch.empty((rows, features), device="cuda", dtype=torch.bfloat16)
    eager_output = torch.empty_like(actual)
    bare_output = torch.empty_like(actual)
    standardized_tensor = (x.float() * torch.rsqrt(fold_state_fp32[:, None] / reduction + 1e-5)).bfloat16()
    reference_mapped = (standardized_tensor * feature_scale_tensor).bfloat16()

    def generated_function() -> torch.Tensor:
        epilogue.gemm(
            transformed,
            weight_tensor.mT,
            actual,
            None,
            epi_args={},
            transform_a=transform,
            tile_M=generated.schedule.tile_m,
            tile_N=generated.schedule.tile_n,
            cluster_M=generated.schedule.cluster_m,
            cluster_N=generated.schedule.cluster_n,
            pingpong=generated.schedule.pingpong,
        )
        return actual

    def eager_function() -> torch.Tensor:
        standardized_eager = (x.float() * torch.rsqrt(fold_state_fp32[:, None] / reduction + 1e-5)).bfloat16()
        mapped = (standardized_eager * feature_scale_tensor).bfloat16()
        return torch.mm(mapped, weight_tensor, out=eager_output)

    def bare_function() -> torch.Tensor:
        return torch.mm(reference_mapped, weight_tensor, out=bare_output)

    generated_function()
    torch.cuda.synchronize()
    full_reference = torch.mm(reference_mapped, weight_tensor).float()
    full_difference = torch.abs(actual.float() - full_reference)
    generated_samples = _samples(generated_function, warmup=5, samples=samples, iterations=iterations)
    eager_samples = _samples(eager_function, warmup=5, samples=samples, iterations=iterations)
    bare_samples = _samples(bare_function, warmup=5, samples=samples, iterations=iterations)
    generated_median = statistics.median(generated_samples)
    flop = 2.0 * rows * reduction * features
    return ShapeResult(
        rows=rows,
        reduction=reduction,
        features=features,
        generated_median_ms=generated_median,
        generated_samples_ms=tuple(generated_samples),
        eager_map_contract_median_ms=statistics.median(eager_samples),
        eager_map_contract_samples_ms=tuple(eager_samples),
        bare_gemm_median_ms=statistics.median(bare_samples),
        bare_gemm_samples_ms=tuple(bare_samples),
        generated_over_eager_map_contract=generated_median / statistics.median(eager_samples),
        generated_over_bare_gemm=generated_median / statistics.median(bare_samples),
        max_absolute_error_full_reference=float(torch.max(full_difference).item()),
        mean_absolute_error_full_reference=float(torch.mean(full_difference).item()),
        generated_tflops=flop / (generated_median * 1e9),
        generated_source_sha256=generated.source_sha256,
        request_sha256=request_sha256,
        schedule_fingerprint=generated.schedule.fingerprint,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_compile = torch.compile

    def compile_without_removed_limit(*compile_args, **compile_kwargs):
        compile_kwargs.pop("recompile_limit", None)
        return original_compile(*compile_args, **compile_kwargs)

    torch.compile = compile_without_removed_limit
    results = [
        _benchmark_shape(256, 512, 512, samples=args.samples, iterations=args.iterations, output=args.output),
        _benchmark_shape(2048, 4096, 4096, samples=args.samples, iterations=args.iterations, output=args.output),
    ]
    full_reference_exact = all(result.max_absolute_error_full_reference == 0.0 for result in results)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "quack_version": importlib.metadata.version("quack-kernels"),
        "quack_revision": "84ef91df9bec87c7e4938517234fafb07ef844dd",
        "quack_patch": "lib/tile_lifetime/backends/h100/quack_fp32_row_scale.patch",
        "quack_patch_sha256": "8813b2fb67690545e554119dc69d53a39ceb5239c41d6f7375435c2d032bd954",
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "samples": args.samples,
        "iterations_per_sample": args.iterations,
        "semantic_verdict": {
            "full_fp32_fold_state": "accepted" if full_reference_exact else "rejected",
            "two_independently_typed_auxiliary_values": "accepted" if full_reference_exact else "rejected",
            "auxiliary_materialization": "none: compact M and K vectors are passed directly",
        },
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "result.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
