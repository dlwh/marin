# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prospective ABI22 LayerNorm backward/composed numerical contract."""

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

CONTRACT_PATH = Path(__file__).with_name("abi22-layernorm-bc-numerical-contract-v1.json")
CONTRACT_SCHEMA = "shuttle_abi22_layernorm_bc_numerical_contract_v1"
BOUNDARIES = ("backward", "composed")
SHAPES = ((7, 13), (2048, 4096))
POLICIES = ("source_ordered", "fast")
EPSILON = np.float32(1.0e-5)
GENERATOR_SEED = 0xBB67AE85
RELATIVE_SCALE_FLOOR = 2**-7
ROLE_LIMITS = {
    "y": {
        "max_absolute_error": 2**-4,
        "mean_absolute_error": 2**-16,
        "relative_linf_error": 2**-6,
        "max_bfloat16_ulp_error": 128,
    },
    "dx": {
        "max_absolute_error": 2**-4,
        "mean_absolute_error": 2**-14,
        "relative_linf_error": 2**-5,
        "max_bfloat16_ulp_error": 4,
    },
    "dgamma": {
        "max_absolute_error": 2.0,
        "mean_absolute_error": 2**-4,
        "relative_linf_error": 2**-6,
        "max_bfloat16_ulp_error": 2,
    },
    "dbeta": {
        "max_absolute_error": 2.0,
        "mean_absolute_error": 2**-4,
        "relative_linf_error": 2**-6,
        "max_bfloat16_ulp_error": 2,
    },
}
PUBLIC_ROLES = {
    "backward": {
        "inputs": ["x", "gamma", "dy"],
        "outputs": ["dx", "dgamma", "dbeta"],
    },
    "composed": {
        "inputs": ["x", "gamma", "beta", "dy"],
        "outputs": ["y", "dx", "dgamma", "dbeta"],
    },
}


@dataclass(frozen=True)
class ErrorMetrics:
    """BF16 deviations for one public result role."""

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


def fixed_inputs(rows: int, features: int) -> dict[str, np.ndarray]:
    """Return held-out BF16 x, dy, gamma, and beta arrays."""
    if (rows, features) not in SHAPES:
        raise ValueError(f"undeclared shape: {(rows, features)!r}")
    count = rows * features
    scale = np.float32(1.0 / 128.0)
    x = ((_mix32(count, 1).astype(np.int64) % 513 - 256).astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    dy = ((_mix32(count, 4).astype(np.int64) % 385 - 192).astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    gamma_bits = _mix32(features, 2).astype(np.int64)
    gamma_quantized = 64 + gamma_bits % 129
    gamma_quantized = np.where((gamma_bits & 0x100) == 0, gamma_quantized, -gamma_quantized)
    gamma = (gamma_quantized.astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    beta = ((_mix32(features, 3).astype(np.int64) % 257 - 128).astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
    if (rows, features) == (2048, 4096):
        dy = dy.reshape(rows, features)
        dy[:, 0] = ml_dtypes.bfloat16(0.0)
        dy[:9, 0] = np.array(
            [16384.0, 2**-11, -16384.0, 2**-11, 16384.0, 2**-11, -16384.0, 2**-11, 0.25],
            dtype=ml_dtypes.bfloat16,
        )
        dy = dy.reshape(-1)
    return {
        "x": x.reshape(rows, features),
        "gamma": gamma,
        "beta": beta,
        "dy": dy.reshape(rows, features),
    }


def binary64_reference(inputs: dict[str, np.ndarray], boundary: str) -> dict[str, np.ndarray]:
    """Evaluate the analytic LayerNorm F/VJP formulas in binary64."""
    x, gamma, beta, dy = _inputs(inputs)
    features = x.shape[1]
    mean = np.sum(x, axis=1, dtype=np.float64) / features
    centered = x - mean[:, None]
    variance = np.sum(centered * centered, axis=1, dtype=np.float64) / features
    inverse = 1.0 / np.sqrt(variance + float(EPSILON))
    normalized = centered * inverse[:, None]
    weighted = dy * gamma[None, :]
    dx = inverse[:, None] * (
        weighted
        - np.sum(weighted, axis=1, dtype=np.float64)[:, None] / features
        - normalized * (np.sum(weighted * normalized, axis=1, dtype=np.float64)[:, None] / features)
    )
    outputs = {
        "dx": dx,
        "dgamma": np.sum(dy * normalized, axis=0, dtype=np.float64),
        "dbeta": np.sum(dy, axis=0, dtype=np.float64),
    }
    if boundary == "composed":
        outputs = {"y": normalized * gamma[None, :] + beta[None, :], **outputs}
    elif boundary != "backward":
        raise ValueError(f"undeclared boundary: {boundary!r}")
    return {role: round_binary64_to_bf16(value) for role, value in outputs.items()}


def round_binary64_to_bf16(value: np.ndarray) -> np.ndarray:
    """Round finite binary64 values directly to BF16 with ties to even."""
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("binary64 values must be finite and nonempty")
    absolute = np.abs(value)
    _, exponent = np.frexp(absolute)
    spacing = np.ldexp(np.ones_like(absolute), np.maximum(exponent - 8, -133))
    rounded = np.copysign(np.rint(absolute / spacing) * spacing, value)
    bits = (rounded.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    return bits.view(ml_dtypes.bfloat16).reshape(value.shape)


def scheduled_reference(inputs: dict[str, np.ndarray], boundary: str) -> dict[str, np.ndarray]:
    """Evaluate the two-task f32 operation and Fold schedule exactly."""
    x64, gamma64, beta64, dy64 = _inputs(inputs)
    x = x64.astype(np.float32)
    gamma = gamma64.astype(np.float32)
    beta = beta64.astype(np.float32)
    dy = dy64.astype(np.float32)
    features = x.shape[1]

    mean_sum = _scheduled_fold_last(x)
    mean = np.divide(mean_sum, np.float32(features), dtype=np.float32)
    centered = np.subtract(x, mean[:, None], dtype=np.float32)
    squared = np.multiply(centered, centered, dtype=np.float32)
    variance_sum = _scheduled_fold_last(squared)
    variance = np.divide(variance_sum, np.float32(features), dtype=np.float32)
    shifted = np.add(variance, EPSILON, dtype=np.float32)
    root = np.sqrt(shifted, dtype=np.float32)
    inverse = np.divide(np.float32(1.0), root, dtype=np.float32)
    normalized = np.multiply(centered, inverse[:, None], dtype=np.float32)

    weighted = np.multiply(dy, gamma[None, :], dtype=np.float32)
    centered_weighted = np.multiply(centered, weighted, dtype=np.float32)
    correction_sum = _scheduled_fold_last(centered_weighted)
    inverse_over_shifted = np.divide(inverse, shifted, dtype=np.float32)
    correction_scale = np.multiply(np.float32(-0.5), inverse_over_shifted, dtype=np.float32)
    correction = np.multiply(correction_sum, correction_scale, dtype=np.float32)
    correction = np.divide(correction, np.float32(features), dtype=np.float32)
    correction = np.add(np.float32(0.0), correction, dtype=np.float32)
    direct = np.multiply(weighted, inverse[:, None], dtype=np.float32)
    centered_correction = np.multiply(centered, correction[:, None], dtype=np.float32)
    joined = np.add(direct, centered_correction, dtype=np.float32)
    repeated_centered_correction = np.multiply(correction[:, None], centered, dtype=np.float32)
    joined = np.add(joined, repeated_centered_correction, dtype=np.float32)
    negative = np.negative(joined, dtype=np.float32)
    row_correction = _scheduled_fold_last(negative)
    row_correction = np.divide(row_correction, np.float32(features), dtype=np.float32)
    row_correction = np.add(np.float32(0.0), row_correction, dtype=np.float32)
    dx = np.add(joined, row_correction[:, None], dtype=np.float32)

    dbeta = _scheduled_fold_first(dy)
    dbeta = np.add(np.float32(0.0), dbeta, dtype=np.float32)
    dgamma_input = np.multiply(normalized, dy, dtype=np.float32)
    dgamma = _scheduled_fold_first(dgamma_input)
    dgamma = np.add(np.float32(0.0), dgamma, dtype=np.float32)
    outputs = {"dx": dx, "dgamma": dgamma, "dbeta": dbeta}
    if boundary == "composed":
        y = np.multiply(normalized, gamma[None, :], dtype=np.float32)
        y = np.add(y, beta[None, :], dtype=np.float32)
        outputs = {"y": y, **outputs}
    elif boundary != "backward":
        raise ValueError(f"undeclared boundary: {boundary!r}")
    return {role: value.astype(ml_dtypes.bfloat16) for role, value in outputs.items()}


def _scheduled_fold_last(values: np.ndarray) -> np.ndarray:
    rows, extent = values.shape
    tile = min(extent, 256)
    threads = ((tile + 31) // 32) * 32
    serial_tiles = (extent + tile - 1) // tile
    partials = np.zeros((rows, threads), dtype=np.float32)
    starts = np.arange(threads, dtype=np.int64) * serial_tiles
    for leaf in range(serial_tiles):
        indices = starts + leaf
        valid = indices < extent
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


def _scheduled_fold_first(values: np.ndarray) -> np.ndarray:
    return _scheduled_fold_last(np.ascontiguousarray(values.T))


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> ErrorMetrics:
    """Measure every finite BF16 element with no near-zero exception."""
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
    ulps = np.abs(_ordered_bf16(actual) - _ordered_bf16(reference))
    return ErrorMetrics(
        max_absolute_error=maximum,
        mean_absolute_error=float(np.mean(absolute)) if absolute.size else 0.0,
        relative_linf_error=maximum / scale,
        max_bfloat16_ulp_error=int(np.max(ulps, initial=0)),
    )


def require_accepted(metrics: ErrorMetrics, role: str) -> None:
    """Apply the predeclared limits for one public result role."""
    if role not in ROLE_LIMITS:
        raise ValueError(f"undeclared role: {role!r}")
    for name in ("max_absolute_error", "mean_absolute_error", "relative_linf_error"):
        value = getattr(metrics, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(metrics.max_bfloat16_ulp_error) is not int or metrics.max_bfloat16_ulp_error < 0:
        raise ValueError("max_bfloat16_ulp_error must be a nonnegative integer")
    for name, limit in ROLE_LIMITS[role].items():
        if getattr(metrics, name) > limit:
            raise AssertionError(f"{role} {name} exceeds {limit}")


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


def resource_contract(boundary: str, shape: tuple[int, int]) -> dict[str, object]:
    """Return the exact two-launch slot and geometry contract."""
    rows, features = shape
    row_threads = ((min(features, 256) + 31) // 32) * 32
    feature_threads = ((min(rows, 256) + 31) // 32) * 32
    if boundary == "backward":
        slot_count = 8
        temporary_slots = [3, 4]
        operand_slots = [0, 1, 2]
        result_slots = [7, 6, 5]
        row_inputs = [0, 1, 2]
        row_outputs = [3, 4, 7]
        feature_inputs = [0, 2, 3, 4]
        feature_outputs = [5, 6]
    elif boundary == "composed":
        slot_count = 10
        temporary_slots = [4, 5]
        operand_slots = [0, 1, 2, 3]
        result_slots = [6, 9, 8, 7]
        row_inputs = [0, 1, 2, 3]
        row_outputs = [4, 5, 6, 9]
        feature_inputs = [0, 3, 4, 5]
        feature_outputs = [7, 8]
    else:
        raise ValueError(f"undeclared boundary: {boundary!r}")
    return {
        "launch_count": 2,
        "launches": [
            {
                "block": [row_threads, 1, 1],
                "dependencies": [],
                "dynamic_shared_bytes": row_threads * 4,
                "grid": [rows, 1, 1],
                "input_slots": row_inputs,
                "output_slots": row_outputs,
            },
            {
                "block": [feature_threads, 1, 1],
                "dependencies": [0],
                "dynamic_shared_bytes": feature_threads * 4,
                "grid": [features, 1, 1],
                "input_slots": feature_inputs,
                "output_slots": feature_outputs,
            },
        ],
        "public_binding_slots": {"operands": operand_slots, "results": result_slots},
        "slot_count": slot_count,
        "temporary_allocation_count": 2,
        "temporary_bytes": rows * 8,
        "temporary_slots": temporary_slots,
    }


def validate_contract(document: object) -> None:
    """Reject drift in the closed matrix, evidence, resources, and policy."""
    if type(document) is not dict or set(document) != {
        "cells",
        "contract_id",
        "generator",
        "hardware_evidence",
        "model",
        "schema",
    }:
        raise ValueError("contract schema is not closed")
    if document["schema"] != CONTRACT_SCHEMA or document["contract_id"] != "abi22_layernorm_bc_h100_v1":
        raise ValueError("contract identity differs")
    if document["hardware_evidence"] != "none_before_independent_approval":
        raise ValueError("hardware evidence boundary differs")
    if document["generator"] != {
        "algorithm": "mix32_power_of_two_bf16_with_large_dbeta_order_sentinel_v1",
        "seed": GENERATOR_SEED,
        "streams": {"beta": 3, "dy": 4, "gamma": 2, "x": 1},
    }:
        raise ValueError("input generator differs")
    if document["model"] != {
        "column_fold_schedule": "simt32_contiguous_leaf_then_adjacent_pair_tree_v1",
        "epilogue": "ordinary_jax_f32_vjp_operation_order_then_cvt_rn_bf16",
        "matrix": "backward_composed_x_7x13_2048x4096_x_source_ordered_fast",
        "near_zero": "none_all_metrics_cover_all_finite_elements",
        "repeatability_runs": 2,
        "resources": "two_launches_two_f32_row_stat_temporaries_rows_x_8_bytes",
        "row_fold_schedule": "simt32_contiguous_leaf_then_adjacent_pair_tree_v1",
        "rsqrt": "sqrt.rn.f32_then_div.rn.f32_from_one",
        "stores": "row_task_y_dx_feature_task_dbeta_dgamma_then_public_binding_projection",
        "ulp": "raw_ordered_finite_bf16_on_declared_reference_scope",
    }:
        raise ValueError("schedule model differs")
    cells = document["cells"]
    if type(cells) is not list or len(cells) != len(BOUNDARIES) * len(SHAPES):
        raise ValueError("contract cells differ")
    coordinates = set()
    sealed = _sealed_cell_evidence()
    for cell in cells:
        if type(cell) is not dict or set(cell) != {
            "boundary",
            "input_digests",
            "limits",
            "output_evidence",
            "policies",
            "public_roles",
            "repeatability_runs",
            "resources",
            "shape",
        }:
            raise ValueError("contract cell schema differs")
        boundary = cell["boundary"]
        shape = tuple(cell["shape"])
        coordinate = (boundary, shape)
        if boundary not in BOUNDARIES or shape not in SHAPES or coordinate in coordinates:
            raise ValueError("contract cell coordinate differs")
        coordinates.add(coordinate)
        roles = PUBLIC_ROLES[boundary]
        if cell["public_roles"] != roles or cell["policies"] != list(POLICIES):
            raise ValueError("public role or policy order differs")
        if cell["repeatability_runs"] != 2:
            raise ValueError("repeatability differs")
        if cell["limits"] != {role: ROLE_LIMITS[role] for role in roles["outputs"]}:
            raise ValueError("role limits differ")
        if cell["resources"] != resource_contract(boundary, shape):
            raise ValueError("resource contract differs")
        if cell["input_digests"] != sealed[coordinate]["input_digests"]:
            raise ValueError("input evidence differs")
        if cell["output_evidence"] != sealed[coordinate]["output_evidence"]:
            raise ValueError("output evidence differs")
        for role in roles["outputs"]:
            require_accepted(ErrorMetrics(**cell["output_evidence"][role]["schedule_metrics"]), role)
    if coordinates != {(boundary, shape) for boundary in BOUNDARIES for shape in SHAPES}:
        raise ValueError("contract cell coverage differs")


def _inputs(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if type(inputs) is not dict or set(inputs) != {"x", "gamma", "beta", "dy"}:
        raise ValueError("LayerNorm input roles differ")
    x = _bf16_array(inputs["x"], "x", 2)
    gamma = _bf16_array(inputs["gamma"], "gamma", 1)
    beta = _bf16_array(inputs["beta"], "beta", 1)
    dy = _bf16_array(inputs["dy"], "dy", 2)
    if tuple(x.shape) not in SHAPES or dy.shape != x.shape or gamma.shape != (x.shape[1],) or beta.shape != gamma.shape:
        raise ValueError("LayerNorm input shapes differ from the contract")
    return tuple(value.astype(np.float64) for value in (x, gamma, beta, dy))


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


@functools.cache
def _sealed_cell_evidence() -> dict[tuple[str, tuple[int, int]], dict[str, object]]:
    evidence = {}
    for shape in SHAPES:
        inputs = fixed_inputs(*shape)
        for boundary in BOUNDARIES:
            input_digests = {role: array_digest(inputs[role]) for role in PUBLIC_ROLES[boundary]["inputs"]}
            binary64 = binary64_reference(inputs, boundary)
            scheduled = scheduled_reference(inputs, boundary)
            output_evidence = {}
            for role in PUBLIC_ROLES[boundary]["outputs"]:
                output_evidence[role] = {
                    "binary64_digest": array_digest(binary64[role]),
                    "schedule_digest": array_digest(scheduled[role]),
                    "schedule_metrics": asdict(error_metrics(scheduled[role], binary64[role])),
                }
            evidence[(boundary, shape)] = {
                "input_digests": input_digests,
                "output_evidence": output_evidence,
            }
    return evidence
