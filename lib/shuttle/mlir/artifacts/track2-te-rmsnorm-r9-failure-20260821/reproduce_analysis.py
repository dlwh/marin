# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the recoverable Track 2 r9 error envelope and v2 input digests."""

import hashlib
import json

import ml_dtypes
import numpy as np

ROWS = 2048
FEATURES = 4096
EPSILON = 1.0e-6
R9_DX_ABSOLUTE_LIMIT = 0.0625
R9_DX_ULP = 283
SPLITMIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)
SPLITMIX_SEEDS = {
    "x": 0x9E3779B97F4A7C15,
    "gamma": 0x243F6A8885A308D3,
    "dy": 0x13198A2E03707344,
}


def bf16_bytes(value: np.ndarray) -> bytes:
    return value.view(np.uint16).astype("<u2", copy=False).tobytes(order="C")


def sha256(value: np.ndarray) -> str:
    return hashlib.sha256(bf16_bytes(value)).hexdigest()


def reference(x: np.ndarray, gamma: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    gamma64 = gamma.astype(np.float64)
    dy64 = dy.astype(np.float64)
    inverse = np.reciprocal(np.sqrt(np.sum(x64 * x64, axis=1, keepdims=True, dtype=np.float64) / FEATURES + EPSILON))
    y = (x64 * inverse * gamma64).astype(ml_dtypes.bfloat16)
    correlation = np.sum(dy64 * x64 * gamma64, axis=1, keepdims=True, dtype=np.float64)
    dx = (dy64 * gamma64 * inverse - x64 * inverse**3 * correlation / FEATURES).astype(ml_dtypes.bfloat16)
    dgamma = np.sum(dy64 * x64 * inverse, axis=0, dtype=np.float64).astype(ml_dtypes.bfloat16)
    return y, dx, dgamma


def reversed_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specifications = (
        (-1.125, 0.5, (ROWS, FEATURES)),
        (0.375, 1.25, (FEATURES,)),
        (1.0, -0.75, (ROWS, FEATURES)),
    )
    return tuple(
        np.linspace(start, stop, int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for start, stop, shape in specifications
    )  # type: ignore[return-value]


def ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.uint16).astype(np.int32)
    return np.where((bits & 0x8000) != 0, 0x7FFF - (bits & 0x7FFF), 0x8000 + bits)


def inverse_ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.where(value >= 0x8000, value - 0x8000, 0x8000 | (0x7FFF - value)).astype(np.uint16)
    return bits.view(ml_dtypes.bfloat16)


def compatible_r9_references(dx: np.ndarray) -> dict[str, int | float | bool]:
    reference_values = dx.reshape(-1)
    reference_float = reference_values.astype(np.float32)
    ordered = ordered_bf16(reference_values)
    masks = []
    sign_crossings = 0
    maximum_actual = 0.0
    for direction in (-R9_DX_ULP, R9_DX_ULP):
        actual_ordered = ordered + direction
        valid = (actual_ordered >= 0) & (actual_ordered <= 0xFFFF)
        actual = inverse_ordered_bf16(np.clip(actual_ordered, 0, 0xFFFF)).astype(np.float32)
        compatible = valid & np.isfinite(actual) & (np.abs(actual - reference_float) <= R9_DX_ABSOLUTE_LIMIT)
        masks.append(compatible)
        sign_crossings += int(np.count_nonzero(compatible & (np.signbit(actual) != np.signbit(reference_float))))
        maximum_actual = max(maximum_actual, float(np.max(np.abs(actual[compatible]), initial=0.0)))
    compatible = masks[0] | masks[1]
    absolute_reference = np.abs(reference_float)
    return {
        "compatible_reference_count": int(np.count_nonzero(compatible)),
        "reference_element_count": int(reference_values.size),
        "compatible_fraction": float(np.mean(compatible)),
        "minimum_compatible_reference_magnitude": float(np.min(absolute_reference[compatible])),
        "maximum_compatible_reference_magnitude": float(np.max(absolute_reference[compatible])),
        "maximum_compatible_actual_magnitude": maximum_actual,
        "sign_crossing_count": sign_crossings,
        "all_compatible_pairs_same_sign": sign_crossings == 0,
    }


def splitmix_values(count: int, seed: int, lower: float, upper: float) -> np.ndarray:
    value = np.arange(count, dtype=np.uint64) + np.uint64(seed)
    value = (value ^ (value >> np.uint64(30))) * SPLITMIX_MULTIPLIER_1
    value = (value ^ (value >> np.uint64(27))) * SPLITMIX_MULTIPLIER_2
    value ^= value >> np.uint64(31)
    fraction = (value >> np.uint64(40)).astype(np.uint32).astype(np.float32) / np.float32(16777215.0)
    return (np.float32(lower) + np.float32(upper - lower) * fraction).astype(ml_dtypes.bfloat16)


def splitmix_pattern() -> dict[str, dict[str, str | int]]:
    x = splitmix_values(ROWS * FEATURES, SPLITMIX_SEEDS["x"], -1.0, 1.0).reshape(ROWS, FEATURES)
    gamma = splitmix_values(FEATURES, SPLITMIX_SEEDS["gamma"], 0.5, 1.5)
    dy = splitmix_values(ROWS * FEATURES, SPLITMIX_SEEDS["dy"], -1.0, 1.0).reshape(ROWS, FEATURES)
    y, dx, dgamma = reference(x, gamma, dy)
    return {
        "inputs": {
            role: {"sha256": sha256(value), "size": len(bf16_bytes(value))}
            for role, value in zip(("x", "gamma", "dy"), (x, gamma, dy), strict=True)
        },
        "references": {
            role: {"sha256": sha256(value), "size": len(bf16_bytes(value))}
            for role, value in zip(("y", "dx", "dgamma"), (y, dx, dgamma), strict=True)
        },
    }


def main() -> None:
    x, gamma, dy = reversed_inputs()
    _, dx, _ = reference(x, gamma, dy)
    absolute_dx = np.abs(dx.astype(np.float64))
    result = {
        "r9_compatibility_envelope": {
            **compatible_r9_references(dx),
            "reference_mean_absolute_value": float(np.mean(absolute_dx)),
            "reference_rms": float(np.sqrt(np.mean(absolute_dx * absolute_dx))),
            "reference_maximum_magnitude": float(np.max(absolute_dx)),
        },
        "third_pattern": splitmix_pattern(),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
