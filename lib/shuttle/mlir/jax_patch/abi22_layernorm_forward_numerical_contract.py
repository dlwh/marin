# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prospective ABI22 LayerNorm-forward numerical contract."""

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

CONTRACT_PATH = Path(__file__).with_name("abi22-layernorm-forward-numerical-contract-v1.json")
CONTRACT_SCHEMA = "shuttle_abi22_layernorm_forward_numerical_contract_v1"
CASES = ("balanced", "positive_offset")
SHAPES = ((7, 13), (2048, 4096))
POLICIES = ("source_ordered", "fast")
EPSILON = np.float32(1.0e-5)
GENERATOR_SEED = 0x6A09E667
RELATIVE_SCALE_FLOOR = 2**-7
LIMITS = {
    "balanced": {
        "max_absolute_error": 2**-6,
        "mean_absolute_error": 1e-6,
        "relative_linf_error": 2**-7,
        "max_bfloat16_ulp_error": 16,
    },
    "positive_offset": {
        "max_absolute_error": 2**-4,
        "mean_absolute_error": 1e-6,
        "relative_linf_error": 2**-7,
        "max_bfloat16_ulp_error": 1,
    },
}


@dataclass(frozen=True)
class ErrorMetrics:
    """Whole-tensor BF16 deviations from an independent reference."""

    max_absolute_error: float
    mean_absolute_error: float
    relative_linf_error: float
    max_bfloat16_ulp_error: int


def _mix32(count: int, stream: int) -> np.ndarray:
    if count <= 0 or stream <= 0:
        raise ValueError("counter count and stream must be positive")
    mask = np.uint64(0xFFFFFFFF)
    values = (
        np.arange(count, dtype=np.uint64) + np.uint64(GENERATOR_SEED) + np.uint64(stream) * np.uint64(0x9E3779B9)
    ) & mask
    values ^= values >> np.uint64(16)
    values = (values * np.uint64(0x7FEB352D)) & mask
    values ^= values >> np.uint64(15)
    values = (values * np.uint64(0x846CA68B)) & mask
    values ^= values >> np.uint64(16)
    return values.astype(np.uint32)


def fixed_inputs(rows: int, features: int, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the held-out BF16 x, gamma, and beta arrays."""
    if (rows, features) not in SHAPES:
        raise ValueError(f"undeclared shape: {(rows, features)!r}")
    if case not in CASES:
        raise ValueError(f"undeclared case: {case!r}")
    x_quantized = _mix32(rows * features, 1).astype(np.int64) % 513 - 256
    gamma_quantized = 64 + _mix32(features, 2).astype(np.int64) % 129
    beta_origin = -64 if case == "balanced" else 1024
    beta_quantized = beta_origin + _mix32(features, 3).astype(np.int64) % 129
    scale = np.float32(1.0 / 128.0)
    x = (x_quantized.astype(np.float32) * scale).astype(ml_dtypes.bfloat16).reshape(rows, features)
    gamma = (gamma_quantized.astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    beta = (beta_quantized.astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    return x, gamma, beta


def binary64_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluate LayerNorm from the public BF16 inputs in binary64."""
    x64, gamma64, beta64 = _inputs(x, gamma, beta)
    features = x64.shape[1]
    mean = np.sum(x64, axis=1, dtype=np.float64) / features
    centered = x64 - mean[:, None]
    variance = np.sum(centered * centered, axis=1, dtype=np.float64) / features
    output = centered / np.sqrt(variance[:, None] + float(EPSILON))
    output = output * gamma64[None, :] + beta64[None, :]
    return output.astype(ml_dtypes.bfloat16)


def scheduled_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluate the emitted f32 operation sequence and declared Fold schedule."""
    x64, gamma64, beta64 = _inputs(x, gamma, beta)
    x32 = x64.astype(np.float32)
    gamma32 = gamma64.astype(np.float32)
    beta32 = beta64.astype(np.float32)
    features = x32.shape[1]
    summed = _scheduled_fold(x32)
    mean = np.divide(summed, np.float32(features), dtype=np.float32)
    centered = np.subtract(x32, mean[:, None], dtype=np.float32)
    squared = np.multiply(centered, centered, dtype=np.float32)
    squared_sum = _scheduled_fold(squared)
    variance = np.divide(squared_sum, np.float32(features), dtype=np.float32)
    root = np.sqrt(np.add(variance, EPSILON, dtype=np.float32), dtype=np.float32)
    inverse = np.divide(np.float32(1.0), root, dtype=np.float32)
    output = np.multiply(centered, inverse[:, None], dtype=np.float32)
    output = np.multiply(output, gamma32[None, :], dtype=np.float32)
    output = np.add(output, beta32[None, :], dtype=np.float32)
    return output.astype(ml_dtypes.bfloat16)


def _scheduled_fold(values: np.ndarray) -> np.ndarray:
    rows, features = values.shape
    tile = min(features, 256)
    threads = ((tile + 31) // 32) * 32
    serial_tiles = (features + tile - 1) // tile
    partials = np.zeros((rows, threads), dtype=np.float32)
    thread_starts = np.arange(threads, dtype=np.int64) * serial_tiles
    for leaf in range(serial_tiles):
        indices = thread_starts + leaf
        valid = indices < features
        partials[:, valid] = np.add(partials[:, valid], values[:, indices[valid]], dtype=np.float32)
    active = threads
    while active > 1:
        pairs = active // 2
        reduced = np.empty((rows, (active + 1) // 2), dtype=np.float32)
        reduced[:, :pairs] = np.add(
            partials[:, : pairs * 2 : 2],
            partials[:, 1 : pairs * 2 : 2],
            dtype=np.float32,
        )
        if active % 2:
            reduced[:, pairs] = partials[:, active - 1]
        partials = reduced
        active = (active + 1) // 2
    return partials[:, 0]


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> ErrorMetrics:
    """Measure all finite BF16 elements with no near-zero exception."""
    actual = _bf16_array(actual, "actual")
    reference = _bf16_array(reference, "reference")
    if actual.shape != reference.shape:
        raise ValueError("actual and reference shapes differ")
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    if not np.all(np.isfinite(actual64)) or not np.all(np.isfinite(reference64)):
        raise ValueError("actual and reference must be finite")
    absolute = np.abs(actual64 - reference64)
    maximum = float(np.max(absolute, initial=0.0))
    scale = max(float(np.max(np.abs(reference64), initial=0.0)), RELATIVE_SCALE_FLOOR)
    return ErrorMetrics(
        max_absolute_error=maximum,
        mean_absolute_error=float(np.mean(absolute)) if absolute.size else 0.0,
        relative_linf_error=maximum / scale,
        max_bfloat16_ulp_error=int(np.max(np.abs(_ordered_bf16(actual) - _ordered_bf16(reference)), initial=0)),
    )


def require_accepted(metrics: ErrorMetrics, case: str) -> None:
    """Apply the predeclared prospective limits for one output role."""
    if case not in CASES:
        raise ValueError(f"undeclared case: {case!r}")
    for name in ("max_absolute_error", "mean_absolute_error", "relative_linf_error"):
        value = getattr(metrics, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(metrics.max_bfloat16_ulp_error) is not int or metrics.max_bfloat16_ulp_error < 0:
        raise ValueError("max_bfloat16_ulp_error must be a nonnegative integer")
    for name, limit in LIMITS[case].items():
        if getattr(metrics, name) > limit:
            raise AssertionError(f"{case} {name} exceeds {limit}")


def array_digest(value: np.ndarray) -> str:
    """Hash the dtype, shape, and row-major bytes of one contract array."""
    value = _bf16_array(value, "value")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load and validate the sealed prospective contract."""
    document = json.loads(path.read_text())
    validate_contract(document)
    return document


def validate_contract(document: object) -> None:
    """Reject drift in the closed contract topology and numerical policy."""
    if type(document) is not dict or set(document) != {
        "cells",
        "contract_id",
        "generator",
        "model",
        "qualification_only",
        "schema",
    }:
        raise ValueError("contract schema is not closed")
    if document["schema"] != CONTRACT_SCHEMA or document["contract_id"] != "abi22_layernorm_forward_h100_v1":
        raise ValueError("contract identity differs")
    generator = document["generator"]
    if type(generator) is not dict or generator != {
        "algorithm": "mix32_power_of_two_bf16_v1",
        "seed": GENERATOR_SEED,
        "streams": {"beta": 3, "gamma": 2, "x": 1},
    }:
        raise ValueError("input generator differs")
    model = document["model"]
    if type(model) is not dict or model != {
        "epilogue": "sub_mul_mul_add_rn_f32_then_cvt_rn_bf16",
        "fold_schedule": "simt32_contiguous_leaf_then_adjacent_pair_tree_v1",
        "near_zero": "none",
        "rsqrt": "sqrt.rn.f32_then_div.rn.f32_from_one",
        "ulp": "raw_ordered_finite_bf16_all_elements",
    }:
        raise ValueError("Fold schedule model differs")
    cells = document["cells"]
    if type(cells) is not list or len(cells) != len(CASES) * len(SHAPES):
        raise ValueError("contract cells differ")
    expected_coordinates = {(case, shape) for case in CASES for shape in SHAPES}
    coordinates = set()
    for cell in cells:
        if type(cell) is not dict or set(cell) != {
            "binary64_output_digest",
            "case",
            "input_digests",
            "limits",
            "policies",
            "repeatability_runs",
            "schedule_metrics",
            "schedule_output_digest",
            "shape",
        }:
            raise ValueError("contract cell schema differs")
        case = cell["case"]
        shape = tuple(cell["shape"])
        coordinates.add((case, shape))
        if case not in CASES or shape not in SHAPES:
            raise ValueError("contract cell coordinate differs")
        if cell["limits"] != LIMITS[case] or cell["policies"] != list(POLICIES):
            raise ValueError("contract cell policy differs")
        if cell["repeatability_runs"] != 2:
            raise ValueError("contract repeatability differs")
        input_digests = cell["input_digests"]
        if type(input_digests) is not dict or set(input_digests) != {"beta", "gamma", "x"}:
            raise ValueError("contract input digests differ")
        digests = [
            *input_digests.values(),
            cell["binary64_output_digest"],
            cell["schedule_output_digest"],
        ]
        if not all(_is_sha256(digest) for digest in digests):
            raise ValueError("contract digest encoding differs")
        schedule_metrics = cell["schedule_metrics"]
        if type(schedule_metrics) is not dict or set(schedule_metrics) != {
            "max_absolute_error",
            "max_bfloat16_ulp_error",
            "mean_absolute_error",
            "relative_linf_error",
        }:
            raise ValueError("contract schedule metrics differ")
        require_accepted(ErrorMetrics(**schedule_metrics), case)
        sealed = _sealed_cell_evidence()[(case, shape)]
        if (
            input_digests != sealed["input_digests"]
            or cell["binary64_output_digest"] != sealed["binary64_output_digest"]
            or cell["schedule_output_digest"] != sealed["schedule_output_digest"]
            or schedule_metrics != sealed["schedule_metrics"]
        ):
            raise ValueError("contract cell evidence differs")
    if coordinates != expected_coordinates:
        raise ValueError("contract cell coverage differs")
    qualification = document["qualification_only"]
    if type(qualification) is not dict or qualification != {
        "job": "/dlwh/shuttle-layernorm-forward-h100-r3",
        "promotion_allowed": False,
        "purpose": "candidate_model_and_limit_design_only",
        "request_sha256": "9534f59f22cdd4f4dd28112de1185ffe40d72618a8933bb6ba5f3edbae73271b",
    }:
        raise ValueError("r3 qualification boundary differs")


def _inputs(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _bf16_array(x, "x", rank=2)
    gamma = _bf16_array(gamma, "gamma", rank=1)
    beta = _bf16_array(beta, "beta", rank=1)
    if gamma.shape != (x.shape[1],) or beta.shape != gamma.shape or tuple(x.shape) not in SHAPES:
        raise ValueError("LayerNorm input shapes differ from the contract")
    return x.astype(np.float64), gamma.astype(np.float64), beta.astype(np.float64)


def _bf16_array(value: np.ndarray, name: str, rank: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != ml_dtypes.bfloat16:
        raise ValueError(f"{name} must be a BF16 array")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if value.size == 0:
        raise ValueError(f"{name} must be nonempty")
    return value


def _ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.uint16).astype(np.int32)
    negative = (bits & 0x8000) != 0
    return np.where(negative, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@functools.cache
def _sealed_cell_evidence() -> dict[tuple[str, tuple[int, int]], dict[str, object]]:
    evidence = {}
    for case in CASES:
        for shape in SHAPES:
            inputs = fixed_inputs(*shape, case)
            binary64 = binary64_reference(*inputs)
            scheduled = scheduled_reference(*inputs)
            evidence[(case, shape)] = {
                "input_digests": dict(zip(("x", "gamma", "beta"), map(array_digest, inputs), strict=True)),
                "binary64_output_digest": array_digest(binary64),
                "schedule_output_digest": array_digest(scheduled),
                "schedule_metrics": asdict(error_metrics(scheduled, binary64)),
            }
    return evidence
