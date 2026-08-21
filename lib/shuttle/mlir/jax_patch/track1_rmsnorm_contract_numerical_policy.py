# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent numerical policy for ordinary-JAX RMSNorm followed by Contract."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from abi15_h100_numerics import (
    bf16_bits_to_fraction,
    f32_bits_to_bf16_bits,
    f32_bits_to_fraction,
    fraction_to_f32_bits,
)

POLICY_FILENAME = "track1-rmsnorm-contract-h100-numerical-policy-v1.json"
POLICY_SHA256 = "6d1fcab38f21cbb3887811b60594ae06ac8978a1e9d33ccd42eb69374dc11fd2"
SHAPES = ((5, 19, 7), (5, 19, 257))
ROLES = ("result", "input_cotangent", "weight_cotangent", "projection_cotangent")
LIMITS = {
    "input_cotangent": {
        "maximum_absolute_error": 0.0625,
        "maximum_bf16_ulp_error": 8,
        "mean_absolute_error": 0.015625,
        "relative_linf_error": 0.03125,
    },
    "projection_cotangent": {
        "maximum_absolute_error": 0.0625,
        "maximum_bf16_ulp_error": 8,
        "mean_absolute_error": 0.015625,
        "relative_linf_error": 0.03125,
    },
    "result": {
        "maximum_absolute_error": 0.03125,
        "maximum_bf16_ulp_error": 4,
        "mean_absolute_error": 0.008,
        "relative_linf_error": 0.015625,
    },
    "weight_cotangent": {
        "maximum_absolute_error": 0.125,
        "maximum_bf16_ulp_error": 8,
        "mean_absolute_error": 0.03125,
        "relative_linf_error": 0.03125,
    },
}


@dataclass(frozen=True)
class NumericalMetrics:
    maximum_absolute_error: float
    mean_absolute_error: float
    relative_linf_error: float
    maximum_bf16_ulp_error: int


def deterministic_inputs(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact finite BF16 inputs for one structural boundary."""
    if shape not in SHAPES:
        raise ValueError("RMSNorm Contract shape is outside the numerical policy")
    rows, reduction, columns = shape
    x_ordinal = np.arange(rows * reduction, dtype=np.float32)
    x = (
        (
            np.where((x_ordinal.astype(np.int64) % 2) == 0, 1.0, -1.0)
            * (np.float32(0.125) + (x_ordinal % 23) / np.float32(8.0))
        )
        .astype(ml_dtypes.bfloat16)
        .reshape(rows, reduction)
    )
    weight = np.linspace(0.5, 1.25, reduction, dtype=np.float32).astype(ml_dtypes.bfloat16)
    projection_ordinal = np.arange(reduction * columns, dtype=np.float32)
    projection = (
        (
            np.where((projection_ordinal.astype(np.int64) % 3) == 0, -1.0, 1.0)
            * (np.float32(0.0625) + (projection_ordinal % 31) / np.float32(32.0))
        )
        .astype(ml_dtypes.bfloat16)
        .reshape(reduction, columns)
    )
    cotangent_ordinal = np.arange(rows * columns, dtype=np.float32)
    cotangent = (
        (
            np.where((cotangent_ordinal.astype(np.int64) % 5) == 0, -1.0, 1.0)
            * (np.float32(0.125) + (cotangent_ordinal % 17) / np.float32(16.0))
        )
        .astype(ml_dtypes.bfloat16)
        .reshape(rows, columns)
    )
    return x, weight, projection, cotangent


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def binary64_oracle(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Evaluate the RMSNorm, BF16 boundaries, Contract, and VJP equations."""
    x_bf16, weight_bf16, projection_bf16, cotangent_bf16 = deterministic_inputs(shape)
    x = x_bf16.astype(np.float64)
    weight = weight_bf16.astype(np.float64)
    projection = projection_bf16.astype(np.float64)
    cotangent = cotangent_bf16.astype(np.float64)
    inverse_root = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float64) + 1e-6)
    normalized = (x * inverse_root * weight).astype(ml_dtypes.bfloat16).astype(np.float64)
    result = (normalized @ projection).astype(ml_dtypes.bfloat16)
    projection_cotangent = (normalized.T @ cotangent).astype(ml_dtypes.bfloat16)
    normalized_cotangent = (cotangent @ projection.T).astype(ml_dtypes.bfloat16).astype(np.float64)
    weighted_cotangent = normalized_cotangent * weight
    correlation = np.sum(weighted_cotangent * x, axis=-1, keepdims=True, dtype=np.float64)
    input_cotangent = weighted_cotangent * inverse_root - (x * inverse_root**3 * correlation / np.float64(shape[1]))
    weight_cotangent = np.sum(normalized_cotangent * x * inverse_root, axis=0, dtype=np.float64)
    return {
        "result": result,
        "input_cotangent": input_cotangent,
        "weight_cotangent": weight_cotangent,
        "projection_cotangent": projection_cotangent,
    }


def _fold_sum(values: np.ndarray) -> np.float32:
    level = np.asarray(values, dtype=np.float32)
    while level.size > 1:
        width = (level.size + 1) // 2
        padded = np.zeros(width * 2, dtype=np.float32)
        padded[: level.size] = level
        level = np.add(padded[0::2], padded[1::2], dtype=np.float32)
    return np.float32(level[0] + np.float32(0.0))


def _fma_f32(accumulator_bits: int, lhs_bits: int, rhs_bits: int) -> int:
    exact = f32_bits_to_fraction(accumulator_bits) + bf16_bits_to_fraction(lhs_bits) * bf16_bits_to_fraction(rhs_bits)
    return fraction_to_f32_bits(exact)


def _add_f32(lhs_bits: int, rhs_bits: int) -> int:
    return fraction_to_f32_bits(f32_bits_to_fraction(lhs_bits) + f32_bits_to_fraction(rhs_bits))


def _contract_row(lhs: np.ndarray, rhs: np.ndarray, *, fast: bool) -> int:
    lhs_bits = lhs.view(np.uint16)
    rhs_bits = rhs.view(np.uint16)
    if not fast:
        accumulator = 0
        for left, right in zip(lhs_bits, rhs_bits, strict=True):
            accumulator = _fma_f32(accumulator, int(left), int(right))
        return f32_bits_to_bf16_bits(accumulator)
    lanes = [0] * 32
    for index, (left, right) in enumerate(zip(lhs_bits, rhs_bits, strict=True)):
        lane = index % 32
        lanes[lane] = _fma_f32(lanes[lane], int(left), int(right))
    for offset in (16, 8, 4, 2, 1):
        previous = lanes.copy()
        for lane in range(32 - offset):
            lanes[lane] = _add_f32(previous[lane], previous[lane + offset])
    return f32_bits_to_bf16_bits(lanes[0])


def emitted_primal_oracle(shape: tuple[int, int, int], *, fast: bool) -> np.ndarray:
    """Evaluate the specialized ABI22 Fold-to-Contract PTX recurrence."""
    x_bf16, weight_bf16, projection_bf16, _ = deterministic_inputs(shape)
    rows, reduction, columns = shape
    normalized = np.empty((rows, reduction), dtype=ml_dtypes.bfloat16)
    x = x_bf16.astype(np.float32)
    weight = weight_bf16.astype(np.float32)
    for row in range(rows):
        square_sum = _fold_sum(np.multiply(x[row], x[row], dtype=np.float32))
        mean_square = np.float32(square_sum / np.float32(reduction))
        root = np.sqrt(np.float32(mean_square + np.float32(1e-6)), dtype=np.float32)
        inverse_root = np.float32(np.float32(1.0) / root)
        normalized[row] = np.multiply(np.multiply(x[row], inverse_root, dtype=np.float32), weight, dtype=np.float32)
    result_bits = np.empty((rows, columns), dtype=np.uint16)
    for row in range(rows):
        for column in range(columns):
            result_bits[row, column] = _contract_row(normalized[row], projection_bf16[:, column], fast=fast)
    return result_bits.view(ml_dtypes.bfloat16)


def numerical_metrics(actual: np.ndarray, reference: np.ndarray) -> NumericalMetrics:
    observed = actual.astype(np.float64)
    absolute = np.abs(observed - reference.astype(np.float64))
    maximum = float(np.max(absolute, initial=0.0))
    scale = float(np.max(np.abs(reference), initial=0.0))

    def ordered(bits: np.ndarray) -> np.ndarray:
        return np.where(
            bits & np.uint16(0x8000),
            bits ^ np.uint16(0xFFFF),
            bits ^ np.uint16(0x8000),
        ).astype(np.int32)

    actual_bits = actual.view(np.uint16)
    rounded_bits = reference.astype(ml_dtypes.bfloat16).view(np.uint16)
    ulp = np.abs(ordered(actual_bits) - ordered(rounded_bits))
    return NumericalMetrics(
        maximum_absolute_error=maximum,
        mean_absolute_error=float(np.mean(absolute, dtype=np.float64)),
        relative_linf_error=maximum / scale if scale else 0.0,
        maximum_bf16_ulp_error=int(np.max(ulp, initial=0)),
    )


def _expected_policy() -> dict[str, object]:
    shapes = {}
    for shape in SHAPES:
        x, weight, projection, cotangent = deterministic_inputs(shape)
        analytic = binary64_oracle(shape)
        shapes["x".join(map(str, shape))] = {
            "binary64_rounded_outputs": {
                role: array_sha256(analytic[role].astype(ml_dtypes.bfloat16)) for role in ROLES
            },
            "inputs": {
                "cotangent": array_sha256(cotangent),
                "projection": array_sha256(projection),
                "weight": array_sha256(weight),
                "x": array_sha256(x),
            },
            "primal_emitted_outputs": {
                "fast": array_sha256(emitted_primal_oracle(shape, fast=True)),
                "source_ordered": array_sha256(emitted_primal_oracle(shape, fast=False)),
            },
        }
    return {
        "acceptance": {
            "composed_topology": "diagnostic_non_performant_not_architecture_target",
            "disabled_jax": "diagnostic_only",
            "independent_binary64_thresholds": LIMITS,
            "nonfinite_action": "reject",
            "primal_emitted_oracle_bitwise_required": True,
            "primal_topology": {
                "5x19x257": {"launches": 2, "temporary_bytes": 20, "temporaries": 1},
                "5x19x7": {"launches": 1, "temporary_bytes": 0, "temporaries": 0},
            },
            "repeat_bitwise_required": True,
        },
        "recurrence": {
            "contract_fast": "lane_mod_32_fma_rn_f32_then_shuffle_tree",
            "contract_source_ordered": "serial_ascending_fma_rn_f32",
            "epsilon": "1e-6",
            "normalized_boundary": "round_nearest_even_bf16",
            "rms_fold": "adjacent_pair_tree_positive_zero_padding",
            "rsqrt_realization": "sqrt_rn_f32_then_divide",
        },
        "schema": "track1_rmsnorm_contract_h100_numerical_policy",
        "schema_version": 1,
        "shapes": shapes,
        "status": "predeclared_unexecuted",
    }


def load_numerical_policy() -> dict[str, object]:
    """Load and verify the canonical predeclared policy."""
    raw = Path(__file__).with_name(POLICY_FILENAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise RuntimeError("RMSNorm Contract numerical policy SHA-256 differs")
    document = json.loads(raw)
    if raw != (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode():
        raise RuntimeError("RMSNorm Contract numerical policy is not canonical")
    if document != _expected_policy():
        raise RuntimeError("RMSNorm Contract numerical policy identity differs")
    return document
