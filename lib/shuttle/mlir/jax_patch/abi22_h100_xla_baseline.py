# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect stock ordinary-JAX host-blocking H100 timing samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxlib
except ModuleNotFoundError:
    jax = None
    jaxlib = None
    jnp = None

from abi22_h100_performance import (
    CELL_KEYS,
    EXPLORATORY_MODE,
    HOST_CLOCK,
    TE_UNAVAILABLE_BLOCKER,
    workload_digest,
)

WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 20
EPSILON = np.float32(1e-6)


@dataclass(frozen=True)
class Workload:
    case: str
    policy: str
    shape: tuple[int, ...]
    operands: tuple[np.ndarray, ...]


@dataclass
class CompiledCell:
    workload: Workload
    executable: Callable[..., Any]
    device_operands: tuple[jax.Array, ...]
    executable_digest: str
    samples: list[int]


def _bf16(value: object) -> np.ndarray:
    return np.asarray(value, dtype=ml_dtypes.bfloat16)


def _linear_bf16(start: float, span: float, count: int) -> np.ndarray:
    values = [
        np.float32(np.float32(start) + np.float32(np.float32(span) * np.float32(index) / np.float32(count - 1)))
        for index in range(count)
    ]
    return np.asarray(values, dtype=ml_dtypes.bfloat16)


def _norm_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = 14, 19
    x = _linear_bf16(-0.75, 1.625, rows * features)
    x[:features] = _linear_bf16(-0.001, 0.002, features)
    weight = _linear_bf16(0.5, 0.75, features)
    cotangent = _linear_bf16(-0.5, 1.125, rows * features)
    return x.reshape(2, 7, 19), weight, cotangent.reshape(2, 7, 19)


def _contract_inputs(columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, reduction = 5, 19
    x = _linear_bf16(-0.75, 1.625, rows * reduction)
    x[:reduction] = _linear_bf16(-0.001, 0.002, reduction)
    weight = _linear_bf16(0.5, 0.75, reduction)
    projection = _linear_bf16(-0.25, 0.625, reduction * columns)
    return x.reshape(rows, reduction), weight, projection.reshape(reduction, columns)


def workloads() -> tuple[Workload, ...]:
    """Return the exact eight benchmark coordinates in protocol order."""
    norm_x, norm_weight, cotangent = _norm_inputs()
    inline = _contract_inputs(7)
    compact = _contract_inputs(257)
    operands = {
        "rmsnorm_forward": (norm_x, norm_weight),
        "rmsnorm_backward": (norm_x, norm_weight, cotangent),
        "rmsnorm_contract_inline": inline,
        "rmsnorm_contract_compact": compact,
    }
    shapes = {
        "rmsnorm_forward": (2, 7, 19),
        "rmsnorm_backward": (2, 7, 19),
        "rmsnorm_contract_inline": (5, 19, 7),
        "rmsnorm_contract_compact": (5, 19, 257),
    }
    return tuple(Workload(case, policy, shapes[case], operands[case]) for case, policy in CELL_KEYS)


def _rmsnorm(x: jax.Array, weight: jax.Array) -> jax.Array:
    converted = x.astype(jnp.float32)
    inverse_root = jax.lax.rsqrt(jnp.mean(converted * converted, axis=-1, keepdims=True) + 1e-6)
    return (converted * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def _rmsnorm_backward(x: jax.Array, weight: jax.Array, cotangent: jax.Array) -> tuple[jax.Array, jax.Array]:
    _, pullback = jax.vjp(_rmsnorm, x, weight)
    return pullback(cotangent)


def _rmsnorm_contract(x: jax.Array, weight: jax.Array, projection: jax.Array) -> jax.Array:
    normalized = _rmsnorm(x, weight)
    return jnp.matmul(normalized, projection, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "rmsnorm_forward": _rmsnorm,
    "rmsnorm_backward": _rmsnorm_backward,
    "rmsnorm_contract_inline": _rmsnorm_contract,
    "rmsnorm_contract_compact": _rmsnorm_contract,
}


def _reference_norm_forward(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, ...]:
    rows = x.reshape(-1, x.shape[-1]).astype(np.float32)
    result = np.empty_like(rows, dtype=ml_dtypes.bfloat16)
    for row_index, row in enumerate(rows):
        square_sum = np.float32(0)
        for value in row:
            square_sum = np.float32(square_sum + np.float32(value * value))
        inverse_root = np.float32(1 / np.sqrt(np.float32(square_sum / np.float32(row.size) + EPSILON)))
        result[row_index] = _bf16(row * inverse_root * weight.astype(np.float32))
    return (result.reshape(x.shape),)


def _reference_norm_backward(x: np.ndarray, weight: np.ndarray, cotangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = x.reshape(-1, x.shape[-1]).astype(np.float32)
    upstream_rows = cotangent.reshape(rows.shape).astype(np.float32)
    scale = weight.astype(np.float32)
    input_cotangent = np.empty_like(rows, dtype=ml_dtypes.bfloat16)
    weight_cotangent = np.zeros(scale.shape, dtype=np.float32)
    for row_index, (row, upstream) in enumerate(zip(rows, upstream_rows, strict=True)):
        square_sum = np.float32(0)
        for value in row:
            square_sum = np.float32(square_sum + np.float32(value * value))
        inverse_root = np.float32(1 / np.sqrt(np.float32(square_sum / np.float32(row.size) + EPSILON)))
        inner_product = np.float32(0)
        for feature in range(row.size):
            inner_product = np.float32(inner_product + np.float32(row[feature] * upstream[feature] * scale[feature]))
            weight_cotangent[feature] = np.float32(
                weight_cotangent[feature] + np.float32(upstream[feature] * row[feature] * inverse_root)
            )
        derivative_scale = np.float32(inverse_root * inverse_root * inverse_root / np.float32(row.size))
        input_cotangent[row_index] = _bf16(upstream * scale * inverse_root - row * derivative_scale * inner_product)
    return input_cotangent.reshape(x.shape), _bf16(weight_cotangent)


def _pairwise_sum(values: np.ndarray) -> np.float32:
    leaves = [np.float32(value) for value in values]
    while len(leaves) > 1:
        leaves = [
            np.float32(leaves[index] + leaves[index + 1]) if index + 1 < len(leaves) else leaves[index]
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0]


def _reference_contract(
    x: np.ndarray, weight: np.ndarray, projection: np.ndarray, *, fast: bool
) -> tuple[np.ndarray, ...]:
    rows = x.reshape(-1, x.shape[-1]).astype(np.float32)
    scale = weight.astype(np.float32)
    matrix = projection.astype(np.float32)
    result = np.empty((rows.shape[0], matrix.shape[1]), dtype=ml_dtypes.bfloat16)
    for row_index, row in enumerate(rows):
        inverse_root = np.float32(1 / np.sqrt(np.float32(_pairwise_sum(row * row) / np.float32(row.size) + EPSILON)))
        normalized = _bf16(row * inverse_root * scale).astype(np.float32)
        for column in range(matrix.shape[1]):
            accumulators = np.zeros(32, dtype=np.float32)
            for reduction in range(row.size):
                lane = reduction % 32 if fast else 0
                accumulators[lane] = np.float32(
                    accumulators[lane] + np.float32(normalized[reduction] * matrix[reduction, column])
                )
            if fast:
                for offset in (16, 8, 4, 2, 1):
                    previous = accumulators.copy()
                    accumulators[: 32 - offset] = previous[: 32 - offset] + previous[offset:]
            result[row_index, column] = _bf16(accumulators[0])
    return (result.reshape((*x.shape[:-1], matrix.shape[1])),)


def _reference(workload: Workload) -> tuple[np.ndarray, ...]:
    if workload.case == "rmsnorm_forward":
        return _reference_norm_forward(*workload.operands)
    if workload.case == "rmsnorm_backward":
        return _reference_norm_backward(*workload.operands)
    return _reference_contract(*workload.operands, fast=workload.policy == "fast")


def _block(value: Any) -> None:
    leaves = jax.tree.leaves(value)
    for leaf in leaves:
        leaf.block_until_ready()


def _ready(value: Any) -> tuple[np.ndarray, ...]:
    leaves = jax.tree.leaves(value)
    _block(value)
    return tuple(np.asarray(leaf) for leaf in leaves)


def _array_digest(value: np.ndarray) -> str:
    if value.dtype != np.dtype(ml_dtypes.bfloat16):
        raise RuntimeError("XLA baseline output dtype is not BF16")
    return hashlib.sha256(value.view(np.uint16).astype("<u2", copy=False).tobytes(order="C")).hexdigest()


def _error(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]) -> tuple[float, int]:
    maximum_absolute_error = 0.0
    maximum_ulp = 0
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if actual_value.shape != expected_value.shape or actual_value.dtype != np.dtype(ml_dtypes.bfloat16):
            raise RuntimeError("XLA baseline output shape or dtype drifted")
        actual_f32 = actual_value.astype(np.float32)
        expected_f32 = expected_value.astype(np.float32)
        if not np.all(np.isfinite(actual_f32)):
            raise RuntimeError("XLA baseline output contains a non-finite value")
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(np.abs(actual_f32 - expected_f32), initial=0.0))
        )
        actual_bits = actual_value.view(np.uint16).astype(np.int32)
        expected_bits = expected_value.view(np.uint16).astype(np.int32)
        actual_ordered = np.where(actual_bits & 0x8000, 0x8000 - (actual_bits & 0x7FFF), 0x8000 + actual_bits)
        expected_ordered = np.where(expected_bits & 0x8000, 0x8000 - (expected_bits & 0x7FFF), 0x8000 + expected_bits)
        maximum_ulp = max(maximum_ulp, int(np.max(np.abs(actual_ordered - expected_ordered), initial=0)))
    return maximum_absolute_error, maximum_ulp


def _compile_cells(device: jax.Device) -> list[CompiledCell]:
    cells = []
    for workload in workloads():
        device_operands = tuple(jax.device_put(value, device) for value in workload.operands)
        compiled = jax.jit(_FUNCTIONS[workload.case]).lower(*device_operands).compile()
        executable_digest = hashlib.sha256(compiled.as_text().encode()).hexdigest()
        cells.append(CompiledCell(workload, compiled, device_operands, executable_digest, []))
    return cells


def _require_h100() -> jax.Device:
    if jax is None:
        raise RuntimeError("ordinary-JAX baseline requires JAX and jaxlib")
    if jax.default_backend() != "gpu":
        raise RuntimeError("ordinary-JAX baseline requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("ordinary-JAX baseline requires one visible H100")
    return devices[0]


def te_unavailable_status() -> dict[str, str]:
    """Return the audited TE 2.17 exploratory blocker."""
    return {
        "schema": "shuttle_h100_exploratory_unavailable_v1",
        "producer": "te",
        "protocol_mode": EXPLORATORY_MODE,
        "status": "unavailable",
        "blocker": TE_UNAVAILABLE_BLOCKER,
    }


def collect_report() -> dict[str, Any]:
    """Compile and time the exact ordinary-JAX H100 workload matrix."""
    device = _require_h100()
    cells = _compile_cells(device)
    for cell in cells:
        for _ in range(WARMUP_ITERATIONS):
            _block(cell.executable(*cell.device_operands))
    for round_index in range(MEASURED_ITERATIONS):
        for position in range(len(cells)):
            cell_index = position if round_index % 2 == 0 else len(cells) - 1 - position
            cell = cells[cell_index]
            start = time.perf_counter_ns()
            _block(cell.executable(*cell.device_operands))
            elapsed = time.perf_counter_ns() - start
            if elapsed <= 0:
                raise RuntimeError("host-blocking timer returned no elapsed time")
            cell.samples.append(elapsed)

    report_cells = []
    for cell_index, cell in enumerate(cells):
        actual = _ready(cell.executable(*cell.device_operands))
        expected = _reference(cell.workload)
        maximum_absolute_error, maximum_ulp = _error(actual, expected)
        operand_bytes = tuple(
            value.view(np.uint16).astype("<u2", copy=False).tobytes(order="C") for value in cell.workload.operands
        )
        report_cells.append(
            {
                "case": cell.workload.case,
                "policy": cell.workload.policy,
                "shape": list(cell.workload.shape),
                "samples_ns": cell.samples,
                "sample_rounds": list(range(MEASURED_ITERATIONS)),
                "sample_positions": [
                    cell_index if round_index % 2 == 0 else len(cells) - 1 - cell_index
                    for round_index in range(MEASURED_ITERATIONS)
                ],
                "output": {
                    "digests": [_array_digest(value) for value in actual],
                    "max_abs_error": maximum_absolute_error,
                    "max_bf16_ulp": maximum_ulp,
                },
                "identity": {
                    "workload_digest": workload_digest(
                        cell.workload.case, cell.workload.policy, cell.workload.shape, operand_bytes
                    ),
                    "executable_digest": cell.executable_digest,
                },
            }
        )
    compiler_options = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "platform_version": device.client.platform_version,
        "x64_enabled": bool(jax.config.x64_enabled),
    }
    canonical_options = json.dumps(compiler_options, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "shuttle_h100_exploratory_host_v1",
        "producer": "xla",
        "device": {"model": device.device_kind, "ordinal": device.id},
        "compiler": {
            "identity": f"ordinary-jax-{jax.__version__}-jaxlib-{jaxlib.__version__}",
            "options_digest": hashlib.sha256(canonical_options.encode()).hexdigest(),
        },
        "protocol": {
            "mode": EXPLORATORY_MODE,
            "clock": HOST_CLOCK,
            "counterbalance": "forward_reverse_alternating",
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "cells": report_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--te-status-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.te_status_output.exists():
        parser.error("output paths must not exist")
    arguments.output.write_text(json.dumps(collect_report(), indent=2, sort_keys=True) + "\n")
    arguments.te_status_output.write_text(json.dumps(te_unavailable_status(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
