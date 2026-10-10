# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the graph-derived SM90 Contract/Fold reverse pipeline against FA4."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from flash_attn.cute.interface import flash_attn_func

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_streaming_backward_emitter import compile_h100_normalized_exponential_backward
from tile_lifetime.h100_streaming_lowering import Sm90BackwardAccumulatorSchedule, Sm90ContractSchedule
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    normalized_exp_contract_pullback_program,
)
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)


def _samples(function, *, warmups: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
    }


def _program(sequence: int, *, scale: float, softcap: float | None):
    score_map = scaled_score_map(scale)
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=sequence,
        key_length=sequence,
        query_heads=32,
        key_value_heads=8,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    forward = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=2),
    )
    schedule = Sm90BackwardAccumulatorSchedule(
        contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=384),
    )
    backward = normalized_exp_contract_pullback_program(
        forward,
        schedule=schedule,
        traversal=ContractDomainTraversal.LOWER_TRIANGULAR,
    )
    return backward, schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--softcap", type=float)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the SM90 benchmark requires CUDA")

    scale = 1.0 / math.sqrt(128)
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    query = torch.randn((1, args.sequence, 32, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn((1, args.sequence, 8, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn_like(key)
    output_cotangent = torch.randn_like(query)

    oracle_query = query.detach().requires_grad_(True)
    oracle_key = key.detach().requires_grad_(True)
    oracle_value = value.detach().requires_grad_(True)
    oracle_output, log_sum_exp = flash_attn_func(
        oracle_query,
        oracle_key,
        oracle_value,
        softmax_scale=scale,
        causal=True,
        softcap=args.softcap or 0.0,
        return_lse=True,
    )
    oracle_gradients: tuple[torch.Tensor, ...] = ()

    def oracle() -> None:
        nonlocal oracle_gradients
        oracle_gradients = torch.autograd.grad(
            oracle_output,
            (oracle_query, oracle_key, oracle_value),
            output_cotangent,
            retain_graph=True,
        )

    backward, schedule = _program(args.sequence, scale=scale, softcap=args.softcap)
    compiled = compile_h100_normalized_exponential_backward(
        backward,
        query=query,
        key=key,
        value=value,
        output=oracle_output.detach(),
        output_cotangent=output_cotangent,
        log_sum_exp=log_sum_exp.detach(),
    )

    def generated() -> None:
        compiled(query, key, value, oracle_output, output_cotangent, log_sum_exp)

    generated()
    oracle()
    torch.cuda.synchronize()
    generated_gradients = (compiled.query_cotangent, compiled.key_cotangent, compiled.value_cotangent)
    correctness = {
        name: _error(actual, expected)
        for name, actual, expected in zip(("query", "key", "value"), generated_gradients, oracle_gradients, strict=True)
    }

    generated_samples = _samples(generated, warmups=args.warmups, repeats=args.repeats, iterations=args.iterations)
    oracle_samples = _samples(oracle, warmups=args.warmups, repeats=args.repeats, iterations=args.iterations)
    generated_median = statistics.median(generated_samples)
    oracle_median = statistics.median(oracle_samples)
    result = {
        "shape": {"batch": 1, "sequence": args.sequence, "query_heads": 32, "key_value_heads": 8, "dimension": 128},
        "graph_mutation": {"causal": True, "softcap": args.softcap},
        "physical_template": "contract_fold_dependent_map_contract_sm90_v1",
        "schedule": {
            "query_tile": schedule.contract.tile_m,
            "key_value_tile": schedule.contract.tile_n,
            "threads": 384,
            "query_stages": 2,
            "output_cotangent_stages": 2,
            "score_cotangent_stages": 2,
        },
        "correctness": correctness,
        "timing_ms": {
            "generated_samples": generated_samples,
            "oracle_samples": oracle_samples,
            "generated_median": generated_median,
            "oracle_median": oracle_median,
            "generated_over_oracle": generated_median / oracle_median,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
