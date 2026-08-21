# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable numerical policy for the Track 1 ordinary-JAX RMSNorm canary."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from abi21_rmsnorm_prehardware_contract import ANALYTIC_LIMITS, CONTRACT_FILENAME, CONTRACT_SHA256

POLICY_FILENAME = "track1-rmsnorm-h100-numerical-policy-v1.json"
POLICY_SHA256 = "4f3b354d8908101072c35b94fc308d500a64ebee22b9b88f475ac3137cdf7df9"
SHAPES = ((2, 7, 19), (3, 2, 5, 11, 263))


@dataclass(frozen=True)
class NumericalMetrics:
    """Per-role deviation from the independent binary64 equation."""

    maximum_absolute_error: float
    mean_absolute_error: float
    relative_linf_error: float
    maximum_bf16_ulp_error: int


def deterministic_inputs(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the canary's exact public BF16 inputs."""
    if shape not in SHAPES:
        raise ValueError("Track 1 RMSNorm shape is outside the numerical policy")
    size = int(np.prod(shape))
    ordinal = np.arange(size, dtype=np.float32)
    magnitude = 0.25 + (ordinal % 29) / 8.0
    sign = np.where((ordinal.astype(np.int64) % 2) == 0, 1.0, -1.0)
    x = (magnitude * sign).astype(ml_dtypes.bfloat16).reshape(shape)
    weight = np.linspace(0.5, 1.5, shape[-1], dtype=np.float32).astype(ml_dtypes.bfloat16)
    cotangent = (0.375 + (ordinal % 17) / 7.0).astype(ml_dtypes.bfloat16).reshape(shape)
    return x, weight, cotangent


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and canonical C-order bytes."""
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _fold_sum(values: np.ndarray) -> np.float32:
    tile = min(values.size, 256)
    threads = ((tile + 31) // 32) * 32
    serial_tiles = (values.size + tile - 1) // tile
    local_width = 1 << (serial_tiles - 1).bit_length()
    lanes = np.zeros((threads, local_width), dtype=np.float32)
    for thread in range(threads):
        for serial in range(serial_tiles):
            leaf = thread * serial_tiles + serial
            if leaf < values.size:
                lanes[thread, serial] = np.float32(values[leaf])
    while lanes.shape[1] > 1:
        lanes = np.add(lanes[:, 0::2], lanes[:, 1::2], dtype=np.float32)
    width = 1 << (min(values.size, threads) - 1).bit_length()
    level = lanes[:width, 0]
    while level.size > 1:
        level = np.add(level[0::2], level[1::2], dtype=np.float32)
    return np.float32(level[0] + np.float32(0.0))


def emitted_tree_oracle(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Evaluate the declared adjacent-tree and sqrt/div f32 recurrence."""
    x_bf16, weight_bf16, cotangent_bf16 = deterministic_inputs(shape)
    feature_count = shape[-1]
    rows = int(np.prod(shape[:-1]))
    x = x_bf16.astype(np.float32).reshape(rows, feature_count)
    weight = weight_bf16.astype(np.float32)
    cotangent = cotangent_bf16.astype(np.float32).reshape(rows, feature_count)
    result = np.empty_like(x)
    input_cotangent = np.empty_like(x)
    normalized = np.empty_like(x)
    for row in range(rows):
        squares = np.multiply(x[row], x[row], dtype=np.float32)
        mean_square = np.float32(_fold_sum(squares) / np.float32(feature_count))
        variance_epsilon = np.float32(mean_square + np.float32(1e-6))
        root = np.sqrt(variance_epsilon, dtype=np.float32)
        inverse_root = np.float32(np.float32(1.0) / root)
        inverse_over_variance = np.float32(inverse_root / variance_epsilon)
        variance_scale = np.float32(np.float32(-0.5) * inverse_over_variance)
        normalized[row] = np.multiply(x[row], inverse_root, dtype=np.float32)
        result[row] = np.multiply(normalized[row], weight, dtype=np.float32)
        weighted_cotangent = np.multiply(cotangent[row], weight, dtype=np.float32)
        correlation = _fold_sum(np.multiply(x[row], weighted_cotangent, dtype=np.float32))
        derivative_scale = np.float32(np.float32(correlation * variance_scale) / np.float32(feature_count))
        direct = np.multiply(weighted_cotangent, inverse_root, dtype=np.float32)
        first_path = np.multiply(x[row], derivative_scale, dtype=np.float32)
        second_path = np.multiply(derivative_scale, x[row], dtype=np.float32)
        input_cotangent[row] = np.add(np.add(direct, first_path, dtype=np.float32), second_path, dtype=np.float32)
    weight_cotangent = np.empty(feature_count, dtype=np.float32)
    products = np.multiply(normalized, cotangent, dtype=np.float32)
    for feature in range(feature_count):
        weight_cotangent[feature] = _fold_sum(products[:, feature])
    return {
        "result": result.reshape(shape).astype(ml_dtypes.bfloat16),
        "input_cotangent": input_cotangent.reshape(shape).astype(ml_dtypes.bfloat16),
        "weight_cotangent": weight_cotangent.astype(ml_dtypes.bfloat16),
    }


def binary64_oracle(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Evaluate independent binary64 RMSNorm and VJP equations."""
    x_bf16, weight_bf16, cotangent_bf16 = deterministic_inputs(shape)
    x = x_bf16.astype(np.float64)
    weight = weight_bf16.astype(np.float64)
    cotangent = cotangent_bf16.astype(np.float64)
    inverse_root = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float64) + 1e-6)
    result = x * inverse_root * weight
    weighted_cotangent = cotangent * weight
    projection = np.mean(weighted_cotangent * x, axis=-1, keepdims=True, dtype=np.float64)
    input_cotangent = weighted_cotangent * inverse_root - x * inverse_root**3 * projection
    prefix_axes = tuple(range(len(shape) - 1))
    weight_cotangent = np.sum(cotangent * x * inverse_root, axis=prefix_axes, dtype=np.float64)
    return {"result": result, "input_cotangent": input_cotangent, "weight_cotangent": weight_cotangent}


def numerical_metrics(actual: np.ndarray, reference: np.ndarray) -> NumericalMetrics:
    """Measure one finite public BF16 result against a binary64 reference."""
    observed = actual.astype(np.float64)
    absolute = np.abs(observed - reference)
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


def _policy_path() -> Path:
    return Path(__file__).with_name(POLICY_FILENAME)


def load_numerical_policy() -> dict[str, object]:
    """Load and verify the immutable predeclared canary policy."""
    raw = _policy_path().read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise RuntimeError("Track 1 RMSNorm numerical policy SHA-256 differs")
    document = json.loads(raw)
    if raw != (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode():
        raise RuntimeError("Track 1 RMSNorm numerical policy is not canonical")
    if (
        type(document) is not dict
        or set(document)
        != {"acceptance", "approved_source", "recurrence", "schema", "schema_version", "shapes", "status"}
        or document["schema"] != "track1_rmsnorm_h100_numerical_policy"
        or document["schema_version"] != 1
        or document["status"] != "predeclared_unexecuted"
        or document["approved_source"]
        != {
            "contract": CONTRACT_FILENAME,
            "imported_fields": ["epsilon", "independent_binary64_thresholds"],
            "schedule_exact_identity_imported": False,
            "sha256": CONTRACT_SHA256,
        }
    ):
        raise RuntimeError("Track 1 RMSNorm numerical policy identity differs")
    acceptance = document["acceptance"]
    if type(acceptance) is not dict or acceptance != {
        "disabled_jax": "diagnostic_only_optional_stronger_result",
        "enabled_emitted_tree_bitwise_required": True,
        "independent_binary64_thresholds": ANALYTIC_LIMITS,
        "nonfinite_action": "reject",
        "repeat_bitwise_required": True,
    }:
        raise RuntimeError("Track 1 RMSNorm numerical acceptance policy differs")
    recurrence = document["recurrence"]
    if type(recurrence) is not dict or recurrence != {
        "accumulator_type": "f32",
        "epsilon": "1e-6",
        "fold_initializer_placement": "single_final_f32_add",
        "lane_leaf_register_initialization": "positive_zero_before_conditional_load",
        "leaf_assignment": "contiguous_lane_order",
        "padding": "positive_zero_in_unloaded_lane_leaf_registers",
        "public_rounding": "round_nearest_even_bf16",
        "reduction_order": "tree_association_free_leaf_order_fixed",
        "rsqrt_realization": "sqrt_rn_f32_then_divide",
        "schedule_identity": "abi22_emitted_ptx_v1",
        "schedule_tree": "per_lane_adjacent_pair_then_shared_adjacent_pair",
    }:
        raise RuntimeError("Track 1 RMSNorm emitted-tree recurrence differs")
    shapes = document["shapes"]
    if type(shapes) is not dict or tuple(shapes) != tuple("x".join(map(str, shape)) for shape in SHAPES):
        raise RuntimeError("Track 1 RMSNorm numerical policy shape order differs")
    for shape in SHAPES:
        shape_id = "x".join(map(str, shape))
        x, weight, cotangent = deterministic_inputs(shape)
        schedule = emitted_tree_oracle(shape)
        analytic = binary64_oracle(shape)
        expected = {
            "binary64_rounded_outputs": {
                role: array_sha256(value.astype(ml_dtypes.bfloat16)) for role, value in analytic.items()
            },
            "emitted_tree_outputs": {role: array_sha256(value) for role, value in schedule.items()},
            "inputs": {
                "output_cotangent": array_sha256(cotangent),
                "weight": array_sha256(weight),
                "x": array_sha256(x),
            },
        }
        if shapes[shape_id] != expected:
            raise RuntimeError(f"Track 1 RMSNorm numerical policy role digests differ for {shape_id}")
    return document
