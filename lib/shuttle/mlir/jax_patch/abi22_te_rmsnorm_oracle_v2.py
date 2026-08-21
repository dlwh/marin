# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned matched Transformer Engine/XLA RMSNorm oracle protocol."""

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ml_dtypes
import numpy as np

ARCHITECTURE_SHAPE = (7, 13)
REPRESENTATIVE_SHAPE = (2048, 4096)
SHAPES = (ARCHITECTURE_SHAPE, REPRESENTATIVE_SHAPE)
BOUNDARIES = ("forward", "backward_recompute", "composed")
EPSILON = 1.0e-6
MEASURED_INVOCATIONS = 20
WARMUP_INVOCATIONS = 10
NORM_CONTRACT_BLOCKER = "te_2_17_rmsnorm_oracle_v2_does_not_cover_norm_contract"
QUALIFICATION_PATTERN = "qualification_linspace_v2"
HELDOUT_PATTERN = "heldout_reversed_linspace_v1"
ACCEPTANCE_PATTERN = "heldout_splitmix64_v1"
ACCEPTANCE_LIMITS = {
    "y": {
        "max_absolute_error": 0.0078125,
        "max_mean_absolute_error": 0.0078125,
        "absolute_floor": 0.015625,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 2,
    },
    "dx": {
        "max_absolute_error": 0.0625,
        "max_mean_absolute_error": 0.0625,
        "absolute_floor": 0.125,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 64,
    },
    "dgamma": {
        "max_absolute_error": 0.015625,
        "max_mean_absolute_error": 0.015625,
        "absolute_floor": 0.03125,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 8,
    },
}
NUMERICAL_POLICY = {
    "schema": "shuttle_abi22_rmsnorm_role_policy_v2",
    "qualification_patterns": [QUALIFICATION_PATTERN, HELDOUT_PATTERN],
    "acceptance_patterns": [ACCEPTANCE_PATTERN],
    "repeat_count": 2,
    "limits_by_role": ACCEPTANCE_LIMITS,
}
TE_SOURCE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
PRODUCERS = ("transformer_engine_2_17", "ordinary_jax_xla")
OUTPUT_ROLES = {
    "forward": ("y",),
    "backward_recompute": ("dx", "dgamma"),
    "composed": ("y", "dx", "dgamma"),
}
CALL_SEQUENCES = {
    "forward": ("rmsnorm_forward",),
    "backward_recompute": ("rmsnorm_forward_to_private_rsigma", "rmsnorm_backward"),
    "composed": ("rmsnorm_forward_to_public_y_and_private_rsigma", "rmsnorm_backward"),
}
INPUT_DIGESTS = {
    ARCHITECTURE_SHAPE: {
        "x": "bf45599484c0439c23fcd951f8f67645c1812c709002b7cbe3a7f7c67c69a9bb",
        "gamma": "909f58ed4a0ea4d2a436730afc3cb5c0960eb11019dfa2836d7827019ae87771",
        "dy": "7f1673f57926522d586d66896ff64949d88a02aa0bd43b246b519004ffa6f89f",
    },
    REPRESENTATIVE_SHAPE: {
        "x": "0f3b7cfa2054631d63038001deb298b086d0c2baffeaddd6082c4518df91e13c",
        "gamma": "e90137a7f830b75f00ee6dae7c89bbd7fba30dfb31ff7951fec129441c170821",
        "dy": "275b6bb7bf115e9110a4403637863cd0a3f0fb2cbb7b18ac9de05560db5a154c",
    },
}
REFERENCE_DIGESTS = {
    ARCHITECTURE_SHAPE: {
        "y": "5a78b29a63f3163c7012bcb40581b7290340f189c0e2f6b14036bb7ec1a99ab3",
        "dx": "79678506dafd3e75dc57813e76f46f9fcaa3f0b5b96d4243ebda9c9cb2a35302",
        "dgamma": "0b1c5df0409d00bf82a6d90748424f03019c6439d599003f28f9a71c79e7c7e9",
    },
    REPRESENTATIVE_SHAPE: {
        "y": "85924a7a28e8edb0a21aff788016315a0b804465ae04cac79c69517ff7c71284",
        "dx": "b832dffb3e31d5e5639c2e272bbd9e74cc5ec3a664298b02604448628c0fbdab",
        "dgamma": "7e8e78fc4dad00d99b614018ae47cfe67162a859cc10bab79c2c2b0ca73746b3",
    },
}
HELDOUT_INPUT_DIGESTS = {
    "x": "595fc9c13633ba99278591d4288b9150b1d7c71eb17bd9c5f0d4720e1762ea8b",
    "gamma": "bc3f6002b8b4c4718db1aeee0cf2b459c18ec80aa69958d2c8945be915a409d6",
    "dy": "237d26f9c6c048e5d3979da9e283aff46f062518d9a91ea5b74d2f8eaf80526d",
}
HELDOUT_REFERENCE_DIGESTS = {
    "y": "15a4e1660f2dc93b91841a0ac8c54c90372548a4b1269a09cc9c428cdc3e98f5",
    "dx": "d5e6be3d8cf52e4139235b9dde69a84db489a705580a376f08a613eb7beeab9a",
    "dgamma": "06498bae527509f636aae0baa0b247e657e5f6dc1f7453cf3279e53db90664e0",
}
ACCEPTANCE_INPUT_DIGESTS = {
    "x": "d5306b1c52e907c4f4df004e6f3a9c488f354acd80221eb5077e6f20a1c3f34e",
    "gamma": "7666bd787eded8f496d15ecaa0ff1a09e2785076f6f5e5db47737a95152b86ec",
    "dy": "2abeef4e00143b614ca27c6f310998493609de3490f4720538be507c4ddc4556",
}
ACCEPTANCE_REFERENCE_DIGESTS = {
    "y": "dbf0c5cc4900ca5498099bd458edfdf31200a692fd2996e66960e8b98b77078e",
    "dx": "938fe85ca5d7820ca1ae7dfb8acc970307ad88ec92d94b77bd024e904c2d8c55",
    "dgamma": "209869785a9897bfd13ff39665e0ad781d2bb14b3b7c15067a1f6c752c4601fb",
}
ELIGIBLE_COUNTS = {
    QUALIFICATION_PATTERN: {"y": 8_225_616, "dx": 6_523_647, "dgamma": 4_096},
    HELDOUT_PATTERN: {"y": 8_388_455, "dx": 5_397_253, "dgamma": 4_096},
    ACCEPTANCE_PATTERN: {"y": 8_305_853, "dx": 7_722_818, "dgamma": 4_091},
}
INPUT_RANGES = ((-0.75, 0.875), (-0.625, 1.0), (-0.5, 1.125))
POLICY_V3_ACCEPTANCE_PATTERN = "heldout_splitmix64_v2"
POLICY_V3_QUALIFICATION_PATTERNS = (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN)
POLICY_V3_SEED_PREIMAGES = {
    "x": "shuttle_abi22_rmsnorm_role_policy_v3/x",
    "gamma": "shuttle_abi22_rmsnorm_role_policy_v3/gamma",
    "dy": "shuttle_abi22_rmsnorm_role_policy_v3/dy",
}
POLICY_V3_SEEDS = {
    "x": 0x8AEBC21044665234,
    "gamma": 0xC234BF5558EEE5F8,
    "dy": 0x9AF2F87A8D1BB35C,
}
POLICY_V3_ACCEPTANCE_LIMITS = {
    "y": {
        "absolute_floor": 0.015625,
        "max_absolute_error_below_floor": 0.0078125,
        "max_absolute_error_above_floor": 0.03125,
        "max_mean_absolute_error": 0.00000095367431640625,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 2,
        "reference_magnitude_envelope": 4.0,
    },
    "dx": {
        "absolute_floor": 0.125,
        "max_absolute_error_below_floor": 0.0625,
        "max_absolute_error_above_floor": 1.0,
        "max_mean_absolute_error": 0.0000019073486328125,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 64,
        "reference_magnitude_envelope": 4.0,
    },
    "dgamma": {
        "absolute_floor": 0.03125,
        "max_absolute_error_below_floor": 0.015625,
        "max_absolute_error_above_floor": 4.0,
        "max_mean_absolute_error": 0.0009765625,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 8,
        "reference_magnitude_envelope": 128.0,
    },
}
NUMERICAL_POLICY_V3 = {
    "schema": "shuttle_abi22_rmsnorm_role_policy_v3",
    "qualification_patterns": list(POLICY_V3_QUALIFICATION_PATTERNS),
    "acceptance_patterns": [POLICY_V3_ACCEPTANCE_PATTERN],
    "repeat_count": 2,
    "generator": {
        "algorithm": "splitmix64_counter_high24_to_bfloat16_v1",
        "seed_derivation": "sha256_utf8_first_8_bytes_big_endian",
        "seed_preimages": POLICY_V3_SEED_PREIMAGES,
        "seeds": POLICY_V3_SEEDS,
    },
    "limits_by_role": POLICY_V3_ACCEPTANCE_LIMITS,
    "dgamma_mean_limit_basis": "prospective_l1_budget_4_over_4096_not_empirically_calibrated",
    "rejection_rule": "no_post_result_widening_or_promotion",
}
POLICY_V3_ACCEPTANCE_INPUT_DIGESTS = {
    "x": "94149f756a9a5cbd8d4ef1fd5b72284666f790ec06da9c4fd34f91981de72b0d",
    "gamma": "a6798be78031f1e5ea51e24649b26a9b952f5b00543a8bf878a1fc68dae96c8e",
    "dy": "16300ef5a34dbe3ae28fc07f6ade2e6f72f9ae63480bc34feb40581ee3bf3818",
}
POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS = {
    "y": "9adf2dbe23cbd85a4a2841e911a17fc769bad679764dc41a933fd29b3e2f5e7a",
    "dx": "7c4e9301b933be7ba6a94d30bf3e00a1e6071e31982c25ea1cafe74ce439fad3",
    "dgamma": "fc63fa70b69a390e51b5fc6a3a2f2c6e70373d1d4f9f79ff5c9f02681dc907e8",
}
POLICY_V3_REFERENCE_MAXIMA = {"y": 2.640625, "dx": 2.6875, "dgamma": 94.0}
POLICY_V3_PARTITION_COUNTS = {
    QUALIFICATION_PATTERN: {
        "y": {"below_floor": 162_992, "above_floor": 8_225_616},
        "dx": {"below_floor": 1_864_961, "above_floor": 6_523_647},
        "dgamma": {"below_floor": 0, "above_floor": 4_096},
    },
    HELDOUT_PATTERN: {
        "y": {"below_floor": 153, "above_floor": 8_388_455},
        "dx": {"below_floor": 2_991_355, "above_floor": 5_397_253},
        "dgamma": {"below_floor": 0, "above_floor": 4_096},
    },
    ACCEPTANCE_PATTERN: {
        "y": {"below_floor": 82_755, "above_floor": 8_305_853},
        "dx": {"below_floor": 665_790, "above_floor": 7_722_818},
        "dgamma": {"below_floor": 5, "above_floor": 4_091},
    },
    POLICY_V3_ACCEPTANCE_PATTERN: {
        "y": {"below_floor": 83_481, "above_floor": 8_305_127},
        "dx": {"below_floor": 662_862, "above_floor": 7_725_746},
        "dgamma": {"below_floor": 5, "above_floor": 4_091},
    },
}
POLICY_V4_ACCEPTANCE_PATTERN = "heldout_splitmix64_v3"
POLICY_V4_QUALIFICATION_PATTERNS = (
    QUALIFICATION_PATTERN,
    HELDOUT_PATTERN,
    ACCEPTANCE_PATTERN,
    POLICY_V3_ACCEPTANCE_PATTERN,
)
POLICY_V4_SEED_PREIMAGES = {
    "x": "shuttle_abi22_rmsnorm_role_policy_v4/x",
    "gamma": "shuttle_abi22_rmsnorm_role_policy_v4/gamma",
    "dy": "shuttle_abi22_rmsnorm_role_policy_v4/dy",
}
POLICY_V4_SEEDS = {
    "x": 0x628E8AB85F0D2BAD,
    "gamma": 0xD105A656E9DF5E9D,
    "dy": 0x717C9E023B564D40,
}
POLICY_V4_ACCEPTANCE_LIMITS = {role: dict(limits) for role, limits in POLICY_V3_ACCEPTANCE_LIMITS.items()}
NUMERICAL_POLICY_V4 = {
    "schema": "shuttle_abi22_rmsnorm_role_policy_v4",
    "qualification_patterns": list(POLICY_V4_QUALIFICATION_PATTERNS),
    "acceptance_patterns": [POLICY_V4_ACCEPTANCE_PATTERN],
    "repeat_count": 2,
    "generator": {
        "algorithm": "splitmix64_counter_high24_to_bfloat16_v1",
        "seed_derivation": "sha256_utf8_first_8_bytes_big_endian",
        "seed_preimages": POLICY_V4_SEED_PREIMAGES,
        "seeds": POLICY_V4_SEEDS,
    },
    "limits_by_role": POLICY_V4_ACCEPTANCE_LIMITS,
    "dgamma_mean_limit_basis": "prospective_l1_budget_4_over_4096_not_empirically_calibrated",
    "qualification_disposition": "diagnostic_only_cannot_reject_acceptance",
    "acceptance_repeatability": "exact_bfloat16_digest_per_provider_boundary_role",
    "rejection_rule": "acceptance_pattern_only_no_post_result_widening_or_promotion",
}
POLICY_V4_ACCEPTANCE_INPUT_DIGESTS = {
    "x": "1528589b5505fba379086deffe6c699c36114d3dc72600bbfe464e17947d71b2",
    "gamma": "7d44ce0c29c1cdd0b59ee5be57f1cc3656563d68095c9d73769bea144cc7bd09",
    "dy": "d1565501f4b951240b46d328a0459b81c1426a44f39cd0d6066cedb3eac0005d",
}
POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS = {
    "y": "437e9adc3e7cfd518132cfac8455d253a34bbfcc081109632611ff7ba4e62a70",
    "dx": "558ae600e406ade38183c4c9726bbfed16b60c07bb0888b4021b38c87408d3f8",
    "dgamma": "28d69308ae4d26785592deb62547d5db17ba50c20c8a21b6e4108349929f55f6",
}
POLICY_V4_REFERENCE_MAXIMA = {"y": 2.640625, "dx": 2.65625, "dgamma": 92.5}
POLICY_V4_PARTITION_COUNTS = {
    **POLICY_V3_PARTITION_COUNTS,
    POLICY_V4_ACCEPTANCE_PATTERN: {
        "y": {"below_floor": 84_000, "above_floor": 8_304_608},
        "dx": {"below_floor": 672_162, "above_floor": 7_716_446},
        "dgamma": {"below_floor": 4, "above_floor": 4_092},
    },
}


_RUNTIME_IDENTITY_SEAL = object()


class ValidatedTeRuntimeIdentity(Mapping[str, str]):
    """Immutable identity proven against resolved runner and TE library bytes."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str], seal: object) -> None:
        if seal is not _RUNTIME_IDENTITY_SEAL:
            raise TypeError("validated runtime identities are created only by manifest validation")
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("validated runtime identities are immutable")

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._values == dict(other)


class ValidatedExecutionPlan(dict[str, Any]):
    """Exact command matrix returned by the canonical plan derivation."""


class ValidatedQualificationPlan(dict[str, Any]):
    """Exact repeated qualification matrix for the frozen role policy."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bf16_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype != ml_dtypes.bfloat16:
        raise ValueError("array is not bfloat16")
    return array.view(np.uint16).tobytes(order="C")


def _fixed_inputs(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape not in SHAPES:
        raise ValueError("shape is outside the RMSNorm v2 contract")
    rows, features = shape
    dimensions = ((rows, features), (features,), (rows, features))
    return tuple(
        np.linspace(start, stop, math.prod(dimensions[index]), dtype=np.float32)
        .reshape(dimensions[index])
        .astype(ml_dtypes.bfloat16)
        for index, (start, stop) in enumerate(INPUT_RANGES)
    )  # type: ignore[return-value]


def _reference(
    x: np.ndarray, gamma: np.ndarray, dy: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    gamma64 = gamma.astype(np.float64)
    dy64 = dy.astype(np.float64)
    features = x.shape[1]
    inverse = np.reciprocal(np.sqrt(np.sum(x64 * x64, axis=1, keepdims=True, dtype=np.float64) / features + epsilon))
    y = (x64 * inverse * gamma64).astype(ml_dtypes.bfloat16)
    correlation = np.sum(dy64 * x64 * gamma64, axis=1, keepdims=True, dtype=np.float64)
    dx = (dy64 * gamma64 * inverse - x64 * inverse**3 * correlation / features).astype(ml_dtypes.bfloat16)
    dgamma = np.sum(dy64 * x64 * inverse, axis=0, dtype=np.float64).astype(ml_dtypes.bfloat16)
    return y, dx, dgamma


def _case_files(shape: tuple[int, int]) -> dict[str, tuple[int, str]]:
    rows, features = shape
    return {
        "x.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["x"]),
        "gamma.bf16": (features * 2, INPUT_DIGESTS[shape]["gamma"]),
        "dy.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["dy"]),
        "reference_y.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["y"]),
        "reference_dx.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["dx"]),
        "reference_dgamma.bf16": (features * 2, REFERENCE_DIGESTS[shape]["dgamma"]),
    }


def _verify_case_directory(case_root: Path) -> None:
    for shape in SHAPES:
        rows, features = shape
        directory = case_root / f"{rows}x{features}"
        for name, (size, digest) in _case_files(shape).items():
            path = directory / name
            if not path.is_file():
                raise ValueError(f"case file missing: {path}")
            payload = path.read_bytes()
            if len(payload) != size:
                raise ValueError(f"case file size drifted: {path}")
            if _sha256_bytes(payload) != digest:
                raise ValueError(f"case file digest drifted: {path}")


def prepare_cases(case_root: Path) -> dict[str, Any]:
    """Write the canonical BF16 operands and binary64-derived references."""
    for shape in SHAPES:
        rows, features = shape
        directory = case_root / f"{rows}x{features}"
        directory.mkdir(parents=True, exist_ok=True)
        x, gamma, dy = _fixed_inputs(shape)
        y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
        values = {
            "x.bf16": x,
            "gamma.bf16": gamma,
            "dy.bf16": dy,
            "reference_y.bf16": y,
            "reference_dx.bf16": dx,
            "reference_dgamma.bf16": dgamma,
        }
        for name, value in values.items():
            (directory / name).write_bytes(_bf16_bytes(value))
    _verify_case_directory(case_root)
    return {"input_digests": INPUT_DIGESTS, "reference_digests": REFERENCE_DIGESTS}


def _heldout_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = REPRESENTATIVE_SHAPE
    specifications = (
        (-1.125, 0.5, (rows, features)),
        (0.375, 1.25, (features,)),
        (1.0, -0.75, (rows, features)),
    )
    return tuple(
        np.linspace(start, stop, math.prod(shape), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for start, stop, shape in specifications
    )  # type: ignore[return-value]


def _heldout_case_files() -> dict[str, tuple[int, str]]:
    rows, features = REPRESENTATIVE_SHAPE
    return {
        "x.bf16": (rows * features * 2, HELDOUT_INPUT_DIGESTS["x"]),
        "gamma.bf16": (features * 2, HELDOUT_INPUT_DIGESTS["gamma"]),
        "dy.bf16": (rows * features * 2, HELDOUT_INPUT_DIGESTS["dy"]),
        "reference_y.bf16": (rows * features * 2, HELDOUT_REFERENCE_DIGESTS["y"]),
        "reference_dx.bf16": (rows * features * 2, HELDOUT_REFERENCE_DIGESTS["dx"]),
        "reference_dgamma.bf16": (features * 2, HELDOUT_REFERENCE_DIGESTS["dgamma"]),
    }


def _splitmix_values(count: int, seed: int, lower: float, upper: float) -> np.ndarray:
    value = np.arange(count, dtype=np.uint64) + np.uint64(seed)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    fraction = (value >> np.uint64(40)).astype(np.uint32).astype(np.float32) / np.float32(16777215.0)
    return (np.float32(lower) + np.float32(upper - lower) * fraction).astype(ml_dtypes.bfloat16)


def _acceptance_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = REPRESENTATIVE_SHAPE
    return (
        _splitmix_values(rows * features, 0x9E3779B97F4A7C15, -1.0, 1.0).reshape(rows, features),
        _splitmix_values(features, 0x243F6A8885A308D3, 0.5, 1.5),
        _splitmix_values(rows * features, 0x13198A2E03707344, -1.0, 1.0).reshape(rows, features),
    )


def _acceptance_case_files() -> dict[str, tuple[int, str]]:
    rows, features = REPRESENTATIVE_SHAPE
    return {
        "x.bf16": (rows * features * 2, ACCEPTANCE_INPUT_DIGESTS["x"]),
        "gamma.bf16": (features * 2, ACCEPTANCE_INPUT_DIGESTS["gamma"]),
        "dy.bf16": (rows * features * 2, ACCEPTANCE_INPUT_DIGESTS["dy"]),
        "reference_y.bf16": (rows * features * 2, ACCEPTANCE_REFERENCE_DIGESTS["y"]),
        "reference_dx.bf16": (rows * features * 2, ACCEPTANCE_REFERENCE_DIGESTS["dx"]),
        "reference_dgamma.bf16": (features * 2, ACCEPTANCE_REFERENCE_DIGESTS["dgamma"]),
    }


def _policy_v3_acceptance_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = REPRESENTATIVE_SHAPE
    return (
        _splitmix_values(rows * features, POLICY_V3_SEEDS["x"], -1.0, 1.0).reshape(rows, features),
        _splitmix_values(features, POLICY_V3_SEEDS["gamma"], 0.5, 1.5),
        _splitmix_values(rows * features, POLICY_V3_SEEDS["dy"], -1.0, 1.0).reshape(rows, features),
    )


def _policy_v3_acceptance_case_files() -> dict[str, tuple[int, str]]:
    rows, features = REPRESENTATIVE_SHAPE
    return {
        "x.bf16": (rows * features * 2, POLICY_V3_ACCEPTANCE_INPUT_DIGESTS["x"]),
        "gamma.bf16": (features * 2, POLICY_V3_ACCEPTANCE_INPUT_DIGESTS["gamma"]),
        "dy.bf16": (rows * features * 2, POLICY_V3_ACCEPTANCE_INPUT_DIGESTS["dy"]),
        "reference_y.bf16": (rows * features * 2, POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS["y"]),
        "reference_dx.bf16": (rows * features * 2, POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS["dx"]),
        "reference_dgamma.bf16": (features * 2, POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS["dgamma"]),
    }


def _verify_policy_cases(case_root: Path) -> None:
    for pattern, files in (
        (HELDOUT_PATTERN, _heldout_case_files()),
        (ACCEPTANCE_PATTERN, _acceptance_case_files()),
    ):
        for name, (size, digest) in files.items():
            path = case_root / pattern / name
            if not path.is_file():
                raise ValueError(f"case file missing: {path}")
            payload = path.read_bytes()
            if len(payload) != size:
                raise ValueError(f"case file size drifted: {path}")
            if _sha256_bytes(payload) != digest:
                raise ValueError(f"case file digest drifted: {path}")


def prepare_policy_cases(case_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Write both reviewed qualification and untouched acceptance inputs."""
    result = {}
    for pattern, inputs, input_digests, reference_digests in (
        (HELDOUT_PATTERN, _heldout_inputs(), HELDOUT_INPUT_DIGESTS, HELDOUT_REFERENCE_DIGESTS),
        (ACCEPTANCE_PATTERN, _acceptance_inputs(), ACCEPTANCE_INPUT_DIGESTS, ACCEPTANCE_REFERENCE_DIGESTS),
    ):
        directory = case_root / pattern
        directory.mkdir(parents=True, exist_ok=True)
        x, gamma, dy = inputs
        y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
        for name, value in {
            "x.bf16": x,
            "gamma.bf16": gamma,
            "dy.bf16": dy,
            "reference_y.bf16": y,
            "reference_dx.bf16": dx,
            "reference_dgamma.bf16": dgamma,
        }.items():
            (directory / name).write_bytes(_bf16_bytes(value))
        result[pattern] = {"input_digests": input_digests, "reference_digests": reference_digests}
    _verify_policy_cases(case_root)
    return result


def _verify_policy_v3_cases(case_root: Path) -> None:
    _verify_policy_cases(case_root)
    for name, (size, digest) in _policy_v3_acceptance_case_files().items():
        path = case_root / POLICY_V3_ACCEPTANCE_PATTERN / name
        if not path.is_file():
            raise ValueError(f"case file missing: {path}")
        payload = path.read_bytes()
        if len(payload) != size:
            raise ValueError(f"case file size drifted: {path}")
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"case file digest drifted: {path}")


def prepare_policy_v3_cases(case_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Write all qualification cases and the untouched v3 acceptance case."""
    result = prepare_policy_cases(case_root)
    directory = case_root / POLICY_V3_ACCEPTANCE_PATTERN
    directory.mkdir(parents=True, exist_ok=True)
    x, gamma, dy = _policy_v3_acceptance_inputs()
    y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
    for name, value in {
        "x.bf16": x,
        "gamma.bf16": gamma,
        "dy.bf16": dy,
        "reference_y.bf16": y,
        "reference_dx.bf16": dx,
        "reference_dgamma.bf16": dgamma,
    }.items():
        (directory / name).write_bytes(_bf16_bytes(value))
    result[POLICY_V3_ACCEPTANCE_PATTERN] = {
        "input_digests": POLICY_V3_ACCEPTANCE_INPUT_DIGESTS,
        "reference_digests": POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS,
    }
    _verify_policy_v3_cases(case_root)
    return result


def _policy_v4_acceptance_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = REPRESENTATIVE_SHAPE
    return (
        _splitmix_values(rows * features, POLICY_V4_SEEDS["x"], -1.0, 1.0).reshape(rows, features),
        _splitmix_values(features, POLICY_V4_SEEDS["gamma"], 0.5, 1.5),
        _splitmix_values(rows * features, POLICY_V4_SEEDS["dy"], -1.0, 1.0).reshape(rows, features),
    )


def _policy_v4_acceptance_case_files() -> dict[str, tuple[int, str]]:
    rows, features = REPRESENTATIVE_SHAPE
    return {
        "x.bf16": (rows * features * 2, POLICY_V4_ACCEPTANCE_INPUT_DIGESTS["x"]),
        "gamma.bf16": (features * 2, POLICY_V4_ACCEPTANCE_INPUT_DIGESTS["gamma"]),
        "dy.bf16": (rows * features * 2, POLICY_V4_ACCEPTANCE_INPUT_DIGESTS["dy"]),
        "reference_y.bf16": (rows * features * 2, POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS["y"]),
        "reference_dx.bf16": (rows * features * 2, POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS["dx"]),
        "reference_dgamma.bf16": (features * 2, POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS["dgamma"]),
    }


def _verify_policy_v4_cases(case_root: Path) -> None:
    _verify_policy_v3_cases(case_root)
    for name, (size, digest) in _policy_v4_acceptance_case_files().items():
        path = case_root / POLICY_V4_ACCEPTANCE_PATTERN / name
        if not path.is_file():
            raise ValueError(f"case file missing: {path}")
        payload = path.read_bytes()
        if len(payload) != size:
            raise ValueError(f"case file size drifted: {path}")
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"case file digest drifted: {path}")


def prepare_policy_v4_cases(case_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Write all diagnostic cases and the untouched v4 acceptance case."""
    result = prepare_policy_v3_cases(case_root)
    directory = case_root / POLICY_V4_ACCEPTANCE_PATTERN
    directory.mkdir(parents=True, exist_ok=True)
    x, gamma, dy = _policy_v4_acceptance_inputs()
    y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
    for name, value in {
        "x.bf16": x,
        "gamma.bf16": gamma,
        "dy.bf16": dy,
        "reference_y.bf16": y,
        "reference_dx.bf16": dx,
        "reference_dgamma.bf16": dgamma,
    }.items():
        (directory / name).write_bytes(_bf16_bytes(value))
    result[POLICY_V4_ACCEPTANCE_PATTERN] = {
        "input_digests": POLICY_V4_ACCEPTANCE_INPUT_DIGESTS,
        "reference_digests": POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS,
    }
    _verify_policy_v4_cases(case_root)
    return result


def _ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.uint16).astype(np.int32)
    return np.where((bits & 0x8000) != 0, 0x7FFF - (bits & 0x7FFF), 0x8000 + bits)


def epsilon_sensitivity(case_root: Path, shape: tuple[int, int]) -> dict[str, dict[str, float | int]]:
    """Measure the pinned epsilon reference against the same equations at epsilon zero."""
    _verify_case_directory(case_root)
    x, gamma, dy = _fixed_inputs(shape)
    zero_outputs = _reference(x, gamma, dy, 0.0)
    rows, features = shape
    shapes = ((rows, features), (rows, features), (features,))
    result = {}
    for role, output_shape, zero in zip(("y", "dx", "dgamma"), shapes, zero_outputs, strict=True):
        pinned = np.fromfile(case_root / f"{rows}x{features}" / f"reference_{role}.bf16", dtype=np.uint16)
        pinned = pinned.view(ml_dtypes.bfloat16).reshape(output_shape)
        result[role] = {
            "max_absolute_error": float(
                np.max(np.abs(pinned.astype(np.float32) - zero.astype(np.float32)), initial=0.0)
            ),
            "max_bfloat16_ulp": int(np.max(np.abs(_ordered_bf16(pinned) - _ordered_bf16(zero)), initial=0)),
        }
    return result


def execution_plan(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedExecutionPlan:
    """Return the exact closed producer command matrix after case preflight."""
    _verify_case_directory(case_root)
    runs = []
    for shape in SHAPES:
        rows, features = shape
        scope = "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_performance_qualification"
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                runs.append(
                    {
                        "shape": list(shape),
                        "scope": scope,
                        "boundary": boundary,
                        "producer": producer,
                        "argv": [
                            *prefix,
                            "--boundary",
                            boundary,
                            "--rows",
                            str(rows),
                            "--features",
                            str(features),
                            "--epsilon",
                            "1e-06",
                            "--case-directory",
                            str(case_root / f"{rows}x{features}"),
                            "--output",
                            str(case_root.parent / "raw" / f"{producer}-{rows}x{features}-{boundary}.json"),
                        ],
                    }
                )
    return ValidatedExecutionPlan(
        schema="shuttle_abi22_te_xla_rmsnorm_plan_v2",
        epsilon=EPSILON,
        excluded_capability=NORM_CONTRACT_BLOCKER,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_plan(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute every reviewed command once and stop on the first failed producer."""
    if not isinstance(plan, ValidatedExecutionPlan):
        raise ValueError("execution plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("execution plan executable identities drifted")
    case_root = Path(_string(plan.get("case_root"), "case_root"))
    expected = execution_plan(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        case_root,
    )
    if plan != expected:
        raise ValueError("execution plan differs from canonical 12-run derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} producer exited with status {status}")


def qualification_plan(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedQualificationPlan:
    """Return the repeated qualification and acceptance matrix for the role policy."""
    _verify_case_directory(case_root)
    _verify_policy_cases(case_root)
    rows, features = REPRESENTATIVE_SHAPE
    runs = []
    for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN):
        directory = case_root / (f"{rows}x{features}" if pattern == QUALIFICATION_PATTERN else pattern)
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                for repeat in range(NUMERICAL_POLICY["repeat_count"]):
                    runs.append(
                        {
                            "shape": [rows, features],
                            "scope": (
                                "representative_policy_acceptance"
                                if pattern == ACCEPTANCE_PATTERN
                                else "representative_policy_qualification"
                            ),
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "producer": producer,
                            "repeat": repeat,
                            "argv": [
                                *prefix,
                                "--boundary",
                                boundary,
                                "--rows",
                                str(rows),
                                "--features",
                                str(features),
                                "--epsilon",
                                "1e-06",
                                "--case-directory",
                                str(directory),
                                "--output",
                                str(
                                    case_root.parent
                                    / "raw"
                                    / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json"
                                ),
                            ],
                        }
                    )
    return ValidatedQualificationPlan(
        schema="shuttle_abi22_rmsnorm_policy_plan_v2",
        policy=NUMERICAL_POLICY,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_qualification_plan(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute only the exact canonical repeated qualification plan."""
    if not isinstance(plan, ValidatedQualificationPlan):
        raise ValueError("qualification plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("qualification plan executable identities drifted")
    expected = qualification_plan(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        Path(_string(plan.get("case_root"), "case_root")),
    )
    if plan != expected:
        raise ValueError("qualification plan differs from canonical repeated derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} qualification producer exited with status {status}")


def qualification_plan_v3(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedQualificationPlan:
    """Return the exact repeated four-pattern v3 policy matrix."""
    _verify_case_directory(case_root)
    _verify_policy_v3_cases(case_root)
    rows, features = REPRESENTATIVE_SHAPE
    runs = []
    for pattern in (*POLICY_V3_QUALIFICATION_PATTERNS, POLICY_V3_ACCEPTANCE_PATTERN):
        directory = case_root / (f"{rows}x{features}" if pattern == QUALIFICATION_PATTERN else pattern)
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                for repeat in range(NUMERICAL_POLICY_V3["repeat_count"]):
                    output = case_root.parent / "raw" / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json"
                    persistence = (
                        ["--output-files-directory", str(output.parent / f"{output.stem}-outputs")]
                        if producer == "ordinary_jax_xla"
                        else []
                    )
                    runs.append(
                        {
                            "shape": [rows, features],
                            "scope": (
                                "representative_policy_acceptance"
                                if pattern == POLICY_V3_ACCEPTANCE_PATTERN
                                else "representative_policy_qualification"
                            ),
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "producer": producer,
                            "repeat": repeat,
                            "argv": [
                                *prefix,
                                "--boundary",
                                boundary,
                                "--rows",
                                str(rows),
                                "--features",
                                str(features),
                                "--epsilon",
                                "1e-06",
                                "--case-directory",
                                str(directory),
                                "--output",
                                str(output),
                                *persistence,
                            ],
                        }
                    )
    return ValidatedQualificationPlan(
        schema="shuttle_abi22_rmsnorm_policy_plan_v3",
        policy=NUMERICAL_POLICY_V3,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_qualification_plan_v3(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute only an exact canonically derived v3 policy plan."""
    if not isinstance(plan, ValidatedQualificationPlan):
        raise ValueError("qualification plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("qualification plan executable identities drifted")
    expected = qualification_plan_v3(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        Path(_string(plan.get("case_root"), "case_root")),
    )
    if plan != expected:
        raise ValueError("qualification plan differs from canonical v3 derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} qualification producer exited with status {status}")


def qualification_plan_v4(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedQualificationPlan:
    """Return the exact repeated five-pattern v4 policy matrix."""
    _verify_case_directory(case_root)
    _verify_policy_v4_cases(case_root)
    rows, features = REPRESENTATIVE_SHAPE
    runs = []
    patterns = (*POLICY_V4_QUALIFICATION_PATTERNS, POLICY_V4_ACCEPTANCE_PATTERN)
    for pattern in patterns:
        directory = case_root / (f"{rows}x{features}" if pattern == QUALIFICATION_PATTERN else pattern)
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                for repeat in range(NUMERICAL_POLICY_V4["repeat_count"]):
                    output = case_root.parent / "raw" / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json"
                    persistence = (
                        ["--output-files-directory", str(output.parent / f"{output.stem}-outputs")]
                        if producer == "ordinary_jax_xla"
                        else []
                    )
                    runs.append(
                        {
                            "shape": [rows, features],
                            "scope": (
                                "representative_policy_acceptance"
                                if pattern == POLICY_V4_ACCEPTANCE_PATTERN
                                else "representative_policy_qualification"
                            ),
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "producer": producer,
                            "repeat": repeat,
                            "argv": [
                                *prefix,
                                "--boundary",
                                boundary,
                                "--rows",
                                str(rows),
                                "--features",
                                str(features),
                                "--epsilon",
                                "1e-06",
                                "--case-directory",
                                str(directory),
                                "--output",
                                str(output),
                                *persistence,
                            ],
                        }
                    )
    return ValidatedQualificationPlan(
        schema="shuttle_abi22_rmsnorm_policy_plan_v4",
        policy=NUMERICAL_POLICY_V4,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_qualification_plan_v4(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute only an exact canonically derived v4 policy plan."""
    if not isinstance(plan, ValidatedQualificationPlan):
        raise ValueError("qualification plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("qualification plan executable identities drifted")
    expected = qualification_plan_v4(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        Path(_string(plan.get("case_root"), "case_root")),
    )
    if plan != expected:
        raise ValueError("qualification plan differs from canonical v4 derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} qualification producer exited with status {status}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _load_document(value: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Path):
        return json.loads(value.read_text())
    return dict(value)


def validate_build_manifest(path: Path | Mapping[str, Any], runner: Path) -> dict[str, Any]:
    """Validate the versioned external TE build and runner-source closure."""
    document = _load_document(path)
    if set(document) != {"schema", "runner", "provider", "build", "runtime_identity"}:
        raise ValueError("build manifest fields drifted")
    if document["schema"] != "shuttle_abi22_te_rmsnorm_runner_build_v2":
        raise ValueError("build manifest schema drifted")
    runner_record = document["runner"]
    if set(runner_record) != {"path", "sha256"}:
        raise ValueError("runner build record drifted")
    if _sha(runner_record["sha256"], "runner digest") != _sha256_bytes(runner.read_bytes()):
        raise ValueError("runner source digest drifted")
    if document["provider"] != {"distribution": "transformer_engine", "version": "2.17.0"}:
        raise ValueError("build provider drifted")
    build = document["build"]
    if build != {
        "language": "c++20",
        "cuda_architecture": "sm_90a",
        "libraries": ["transformer_engine", "cudart"],
    }:
        raise ValueError("build contract drifted")
    if document["runtime_identity"] != "shuttle_abi22_te_rmsnorm_runtime_v2_required":
        raise ValueError("runtime identity requirement drifted")
    return document


def validate_runtime_manifest(document: Mapping[str, Any], runner: Path) -> ValidatedTeRuntimeIdentity:
    """Bind the report identity to the runner and resolved TE library bytes."""
    if set(document) != {"schema", "runner_sha256", "transformer_engine_library"}:
        raise ValueError("runtime manifest fields drifted")
    if document["schema"] != "shuttle_abi22_te_rmsnorm_runtime_v2":
        raise ValueError("runtime manifest schema drifted")
    runner_sha = _sha(document["runner_sha256"], "runner_sha256")
    if runner_sha != _sha256_bytes(runner.read_bytes()):
        raise ValueError("runner digest differs from resolved bytes")
    library_record = document["transformer_engine_library"]
    if not isinstance(library_record, Mapping) or set(library_record) != {"path", "sha256"}:
        raise ValueError("runtime library record drifted")
    library = Path(_string(library_record["path"], "runtime library path"))
    if not library.is_file():
        raise ValueError("runtime library path is not a regular file")
    library_sha = _sha(library_record["sha256"], "library digest")
    if library_sha != _sha256_bytes(library.read_bytes()):
        raise ValueError("library digest differs from resolved bytes")
    return ValidatedTeRuntimeIdentity(
        {
            "distribution": "transformer_engine",
            "version": "2.17.0",
            "source_commit": TE_SOURCE_COMMIT,
            "runner_sha256": runner_sha,
            "library_sha256": library_sha,
        },
        _RUNTIME_IDENTITY_SEAL,
    )


def _expected_inputs(shape: tuple[int, int], boundary: str) -> dict[str, str]:
    roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    return {role: INPUT_DIGESTS[shape][role] for role in roles}


def _expected_execution(boundary: str) -> dict[str, Any]:
    return {"call_sequence": list(CALL_SEQUENCES[boundary]), "rsigma_state": "invocation_local_not_public"}


def _output_shape(shape: tuple[int, int], role: str) -> tuple[int, ...]:
    return (shape[1],) if role == "dgamma" else shape


def _read_bf16_file(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"raw output is missing: {path}")
    value = np.fromfile(path, dtype=np.uint16)
    if value.size != math.prod(shape):
        raise ValueError(f"raw output size drifted: {path}")
    return value.view(ml_dtypes.bfloat16).reshape(shape)


def _json_number(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def error_metrics(actual: np.ndarray, reference: np.ndarray, *, role: str) -> dict[str, float | int | bool | str]:
    """Return whole-output and above-floor BF16 error metrics for one role."""
    if role not in ACCEPTANCE_LIMITS or actual.shape != reference.shape:
        raise ValueError("error metric role or shape drifted")
    actual_f32 = actual.astype(np.float32)
    reference_f32 = reference.astype(np.float32)
    finite = np.isfinite(actual_f32) & np.isfinite(reference_f32)
    difference = np.abs(actual_f32 - reference_f32)
    ulp = np.abs(_ordered_bf16(actual) - _ordered_bf16(reference))
    raw_index = int(np.argmax(ulp)) if ulp.size else 0
    floor = float(ACCEPTANCE_LIMITS[role]["absolute_floor"])
    eligible = finite & (np.abs(reference_f32) >= floor)
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count:
        relative = difference[eligible] / np.abs(reference_f32[eligible])
        max_relative = float(np.max(relative))
        max_masked_ulp = int(np.max(ulp[eligible]))
    else:
        max_relative = 0.0
        max_masked_ulp = 0
    return {
        "max_absolute_error": _json_number(float(np.max(difference, initial=0.0))),
        "mean_absolute_error": _json_number(float(np.mean(difference)) if difference.size else 0.0),
        "max_bfloat16_ulp": int(np.max(ulp, initial=0)),
        "raw_max_bfloat16_ulp_index": raw_index,
        "raw_max_bfloat16_ulp_actual": _json_number(float(actual_f32.flat[raw_index]) if actual_f32.size else 0.0),
        "raw_max_bfloat16_ulp_reference": _json_number(
            float(reference_f32.flat[raw_index]) if reference_f32.size else 0.0
        ),
        "absolute_floor": floor,
        "eligible_count": eligible_count,
        "max_relative_error_above_absolute_floor": _json_number(max_relative),
        "max_bfloat16_ulp_above_absolute_floor": max_masked_ulp,
        "all_finite": bool(np.all(finite)),
    }


def error_metrics_v3(actual: np.ndarray, reference: np.ndarray, *, role: str) -> dict[str, float | int | bool | str]:
    """Return exhaustive piecewise absolute and above-floor BF16 metrics."""
    if role not in POLICY_V3_ACCEPTANCE_LIMITS or actual.shape != reference.shape:
        raise ValueError("error metric role or shape drifted")
    actual_f32 = actual.astype(np.float32)
    reference_f32 = reference.astype(np.float32)
    finite = np.isfinite(actual_f32) & np.isfinite(reference_f32)
    difference = np.abs(actual_f32 - reference_f32)
    ulp = np.abs(_ordered_bf16(actual) - _ordered_bf16(reference))
    raw_index = int(np.argmax(ulp)) if ulp.size else 0
    floor = float(POLICY_V3_ACCEPTANCE_LIMITS[role]["absolute_floor"])
    below = np.abs(reference_f32) < floor
    above = ~below
    below_count = int(np.count_nonzero(below))
    above_count = int(np.count_nonzero(above))
    below_absolute = float(np.max(difference[below], initial=0.0))
    above_absolute = float(np.max(difference[above], initial=0.0))
    if above_count:
        relative = difference[above] / np.abs(reference_f32[above])
        max_relative = float(np.max(relative))
        max_masked_ulp = int(np.max(ulp[above]))
    else:
        max_relative = 0.0
        max_masked_ulp = 0
    return {
        "max_absolute_error": _json_number(float(np.max(difference, initial=0.0))),
        "mean_absolute_error": _json_number(float(np.mean(difference)) if difference.size else 0.0),
        "max_bfloat16_ulp": int(np.max(ulp, initial=0)),
        "raw_max_bfloat16_ulp_index": raw_index,
        "raw_max_bfloat16_ulp_actual": _json_number(float(actual_f32.flat[raw_index]) if actual_f32.size else 0.0),
        "raw_max_bfloat16_ulp_reference": _json_number(
            float(reference_f32.flat[raw_index]) if reference_f32.size else 0.0
        ),
        "absolute_floor": floor,
        "below_floor_count": below_count,
        "eligible_count": above_count,
        "max_absolute_error_below_floor": _json_number(below_absolute),
        "max_absolute_error_above_floor": _json_number(above_absolute),
        "max_relative_error_above_absolute_floor": _json_number(max_relative),
        "max_bfloat16_ulp_above_absolute_floor": max_masked_ulp,
        "all_finite": bool(np.all(finite)),
    }


def repeat_output_diagnostics_v3(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Return exact file-backed pairwise BF16 repeat diagnostics."""
    if first.dtype != ml_dtypes.bfloat16 or second.dtype != ml_dtypes.bfloat16 or first.shape != second.shape:
        raise ValueError("repeat diagnostic arrays must be shape-matched BF16")
    first_f32 = first.astype(np.float32)
    second_f32 = second.astype(np.float32)
    difference = np.abs(first_f32 - second_f32)
    ulp = np.abs(_ordered_bf16(first) - _ordered_bf16(second))
    mismatch = first.view(np.uint16) != second.view(np.uint16)
    mismatch_indices = np.flatnonzero(mismatch)
    first_index = int(mismatch_indices[0]) if mismatch_indices.size else 0
    maximum_index = int(np.argmax(difference)) if difference.size else 0
    return {
        "mismatch_count": int(mismatch_indices.size),
        "max_absolute_error": _json_number(float(np.max(difference, initial=0.0))),
        "max_bfloat16_ulp": int(np.max(ulp, initial=0)),
        "first_mismatch_index": first_index,
        "first_mismatch_repeat0": _json_number(float(first_f32.flat[first_index]) if first_f32.size else 0.0),
        "first_mismatch_repeat1": _json_number(float(second_f32.flat[first_index]) if second_f32.size else 0.0),
        "max_absolute_error_index": maximum_index,
        "max_absolute_error_repeat0": _json_number(float(first_f32.flat[maximum_index]) if first_f32.size else 0.0),
        "max_absolute_error_repeat1": _json_number(float(second_f32.flat[maximum_index]) if second_f32.size else 0.0),
    }


def error_metrics_v4(actual: np.ndarray, reference: np.ndarray, *, role: str) -> dict[str, float | int | bool | str]:
    """Return the frozen v4 piecewise role metrics."""
    return error_metrics_v3(actual, reference, role=role)


def repeat_output_diagnostics_v4(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Return file-backed repeat diagnostics for v4 evidence."""
    return repeat_output_diagnostics_v3(first, second)


def load_te_raw_cell(
    path: Path,
    case_directory: Path,
    *,
    expected_input_digests: Mapping[str, str] | None = None,
    expected_pattern: str | None = None,
) -> dict[str, Any]:
    """Convert one TE runner observation into a validated-report cell."""
    raw = json.loads(path.read_text())
    required = {
        "schema",
        "shape",
        "boundary",
        "epsilon",
        "warmup_invocations",
        "raw_host_blocking_ns",
        "raw_cuda_event_ns",
        "output_files",
        "workspace_bytes",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("raw TE cell fields drifted")
    if raw["schema"] != "shuttle_abi22_te_rmsnorm_raw_cell_v2" or raw["epsilon"] != EPSILON:
        raise ValueError("raw TE schema or epsilon drifted")
    shape = tuple(raw["shape"])
    boundary = raw["boundary"]
    if shape not in SHAPES or boundary not in BOUNDARIES:
        raise ValueError("raw TE coordinate is outside the matrix")
    expected_directory = f"{shape[0]}x{shape[1]}" if expected_pattern is None else expected_pattern
    if case_directory.name != expected_directory:
        raise ValueError("raw TE case directory does not match its shape")
    if expected_pattern is None:
        _verify_case_directory(case_directory.parent)
    else:
        _verify_policy_cases(case_directory.parent)
    roles = OUTPUT_ROLES[boundary]
    output_files = raw["output_files"]
    if not isinstance(output_files, Mapping) or tuple(sorted(output_files)) != tuple(sorted(roles)):
        raise ValueError("raw TE output roles drifted")
    outputs = []
    for role in roles:
        output_shape = _output_shape(shape, role)
        actual = _read_bf16_file(Path(_string(output_files[role], f"raw {role} path")), output_shape)
        reference = _read_bf16_file(case_directory / f"reference_{role}.bf16", output_shape)
        outputs.append(
            {
                "role": role,
                "digest": _sha256_bytes(_bf16_bytes(actual)),
                "reference_digest": _sha256_bytes(_bf16_bytes(reference)),
                **error_metrics(actual, reference, role=role),
            }
        )
    host = _positive_samples(raw["raw_host_blocking_ns"], "host", permitted_none=False)
    cuda = _positive_samples(raw["raw_cuda_event_ns"], "CUDA-event", permitted_none=False)
    assert host is not None and cuda is not None
    if raw["warmup_invocations"] != WARMUP_INVOCATIONS:
        raise ValueError("raw TE warmup drifted")
    workspace = raw["workspace_bytes"]
    if (
        not isinstance(workspace, Mapping)
        or set(workspace) != {"forward", "backward"}
        or any(type(value) is not int or value < 0 for value in workspace.values())
    ):
        raise ValueError("raw TE workspace metadata drifted")
    return {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_performance_qualification",
        "boundary": boundary,
        "input_digests": (
            _expected_inputs(shape, boundary)
            if expected_input_digests is None
            else {
                role: expected_input_digests[role]
                for role in (("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy"))
            }
        ),
        "execution": _expected_execution(boundary),
        "outputs": outputs,
        "timing": {
            "warmup_invocations": WARMUP_INVOCATIONS,
            "raw_host_blocking_ns": host,
            "median_host_blocking_ns": statistics.median(host),
            "raw_cuda_event_ns": cuda,
        },
    }


def load_te_raw_cell_v3(
    path: Path,
    case_directory: Path,
    *,
    expected_input_digests: Mapping[str, str],
    expected_pattern: str | None,
) -> dict[str, Any]:
    """Load one TE policy cell with v3 metrics while preserving v2 transport validation."""
    cell = load_te_raw_cell(
        path,
        case_directory,
        expected_input_digests=expected_input_digests,
        expected_pattern=expected_pattern,
    )
    raw = json.loads(path.read_text())
    outputs = []
    for role in OUTPUT_ROLES[cell["boundary"]]:
        shape = _output_shape(tuple(cell["shape"]), role)
        actual = _read_bf16_file(Path(_string(raw["output_files"][role], f"raw {role} path")), shape)
        reference = _read_bf16_file(case_directory / f"reference_{role}.bf16", shape)
        outputs.append(
            {
                "role": role,
                "digest": _sha256_bytes(_bf16_bytes(actual)),
                "reference_digest": _sha256_bytes(_bf16_bytes(reference)),
                **error_metrics_v3(actual, reference, role=role),
            }
        )
    cell["outputs"] = outputs
    return cell


def load_te_raw_cell_v4(
    path: Path,
    case_directory: Path,
    *,
    expected_input_digests: Mapping[str, str],
    expected_pattern: str | None,
) -> dict[str, Any]:
    """Load one TE policy cell with frozen v4 metrics."""
    cell = load_te_raw_cell(
        path,
        case_directory,
        expected_input_digests=expected_input_digests,
        expected_pattern=expected_pattern,
    )
    raw = json.loads(path.read_text())
    outputs = []
    for role in OUTPUT_ROLES[cell["boundary"]]:
        shape = _output_shape(tuple(cell["shape"]), role)
        actual = _read_bf16_file(Path(_string(raw["output_files"][role], f"raw {role} path")), shape)
        reference = _read_bf16_file(case_directory / f"reference_{role}.bf16", shape)
        outputs.append(
            {
                "role": role,
                "digest": _sha256_bytes(_bf16_bytes(actual)),
                "reference_digest": _sha256_bytes(_bf16_bytes(reference)),
                **error_metrics_v4(actual, reference, role=role),
            }
        )
    cell["outputs"] = outputs
    return cell


def assemble_report(
    producer: str,
    cells: list[dict[str, Any]],
    provider_identity: Mapping[str, str],
    device: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the six cell observations into the versioned report envelope."""
    report = {
        "schema": "shuttle_abi22_te_xla_rmsnorm_oracle_v2",
        "protocol_mode": "matched_host_blocking_v2",
        "producer": producer,
        "epsilon": EPSILON,
        "reference": {
            "implementation": "independent_numpy_binary64_closed_form_then_bfloat16",
            "input_generator": "numpy_linspace_float32_then_bfloat16_v2",
            "epsilon": EPSILON,
        },
        "provider_identity": dict(provider_identity),
        "device": dict(device),
        "capabilities": {
            "rmsnorm": list(BOUNDARIES),
            "composed_semantics": "forward_then_backward_with_invocation_local_rsigma",
            "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
        },
        "cells": cells,
    }
    return validate_report(
        report,
        expected_producer=producer,
        expected_te_identity=(
            provider_identity
            if producer == "transformer_engine_2_17" and isinstance(provider_identity, ValidatedTeRuntimeIdentity)
            else None
        ),
    )


def _validate_identity(
    identity: object,
    producer: str,
    expected_te_identity: ValidatedTeRuntimeIdentity | None,
) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError("provider_identity must be an object")
    if producer == "transformer_engine_2_17":
        fields = {"distribution", "version", "source_commit", "runner_sha256", "library_sha256"}
        if set(identity) != fields:
            raise ValueError("provider identity fields drifted")
        if identity.get("distribution") != "transformer_engine":
            raise ValueError("provider identity distribution drifted")
        if identity.get("version") != "2.17.0":
            raise ValueError("provider identity version drifted")
        if identity.get("source_commit") != TE_SOURCE_COMMIT:
            raise ValueError("provider identity source_commit drifted")
        _sha(identity.get("runner_sha256"), "provider identity runner_sha256")
        _sha(identity.get("library_sha256"), "provider identity library_sha256")
        if not isinstance(expected_te_identity, ValidatedTeRuntimeIdentity):
            raise ValueError("provider identity lacks a validated runtime manifest")
        if dict(identity) != dict(expected_te_identity):
            for field in ("version", "source_commit", "runner_sha256", "library_sha256"):
                if identity.get(field) != (expected_te_identity or {}).get(field):
                    raise ValueError(f"provider identity {field} drifted")
            raise ValueError("provider identity drifted")
        return
    expected = {"jax_version": "0.10.1", "jaxlib_version": "0.10.1", "xla_revision": XLA_REVISION}
    if dict(identity) != expected:
        for field, value in expected.items():
            if identity.get(field) != value:
                raise ValueError(f"provider identity {field} drifted")
        raise ValueError("provider identity drifted")


def _positive_samples(value: object, name: str, *, permitted_none: bool) -> list[int] | None:
    if value is None and permitted_none:
        return None
    if not isinstance(value, list) or len(value) != MEASURED_INVOCATIONS:
        raise ValueError(f"{name} samples must contain exactly 20 values")
    if any(type(sample) is not int or sample <= 0 for sample in value):
        raise ValueError(f"{name} samples must be positive integer nanoseconds")
    return value


def validate_report(
    document: Mapping[str, Any],
    *,
    expected_producer: str,
    expected_te_identity: ValidatedTeRuntimeIdentity | None = None,
) -> dict[str, Any]:
    """Validate one complete six-cell producer report."""
    if expected_producer not in PRODUCERS:
        raise ValueError("expected producer is outside the protocol")
    required = {
        "schema",
        "protocol_mode",
        "producer",
        "epsilon",
        "reference",
        "provider_identity",
        "device",
        "capabilities",
        "cells",
    }
    if set(document) != required:
        raise ValueError("report fields drifted")
    if document["schema"] != "shuttle_abi22_te_xla_rmsnorm_oracle_v2":
        raise ValueError("report schema drifted")
    if document["protocol_mode"] != "matched_host_blocking_v2":
        raise ValueError("protocol mode drifted")
    if document["producer"] != expected_producer:
        raise ValueError("producer drifted")
    if type(document["epsilon"]) is not float or document["epsilon"] != EPSILON:
        raise ValueError("epsilon drifted")
    if document["reference"] != {
        "implementation": "independent_numpy_binary64_closed_form_then_bfloat16",
        "input_generator": "numpy_linspace_float32_then_bfloat16_v2",
        "epsilon": EPSILON,
    }:
        raise ValueError("reference contract drifted")
    _validate_identity(document["provider_identity"], expected_producer, expected_te_identity)
    device = document["device"]
    if not isinstance(device, Mapping) or set(device) != {"model", "uuid", "compute_capability"}:
        raise ValueError("device identity drifted")
    for field in device:
        _string(device[field], f"device {field}")
    if document["capabilities"] != {
        "rmsnorm": list(BOUNDARIES),
        "composed_semantics": "forward_then_backward_with_invocation_local_rsigma",
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }:
        raise ValueError("norm_contract capability or RMSNorm boundary contract drifted")
    cells = document["cells"]
    if not isinstance(cells, list) or len(cells) != len(SHAPES) * len(BOUNDARIES):
        raise ValueError("cell matrix drifted")
    expected_coordinates = [(shape, boundary) for shape in SHAPES for boundary in BOUNDARIES]
    for cell, (shape, boundary) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("cell matrix contains a non-object")
        if tuple(cell.get("shape", ())) != shape or cell.get("boundary") != boundary:
            raise ValueError("cell matrix coordinate drifted")
        scope = "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_performance_qualification"
        if cell.get("scope") != scope:
            raise ValueError("cell scope drifted")
        if cell.get("input_digests") != _expected_inputs(shape, boundary):
            raise ValueError("cell input digest mismatch in input provenance")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("cell call sequence or rsigma state drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("cell output roles drifted")
        for output, role in zip(outputs, roles, strict=True):
            _sha(output.get("digest"), "output digest")
            reference_digest = _sha(output.get("reference_digest"), "reference digest")
            if reference_digest != REFERENCE_DIGESTS[shape][role]:
                raise ValueError("composed reference provenance drifted")
            absolute = output.get("max_absolute_error")
            ulp = output.get("max_bfloat16_ulp")
            if output.get("all_finite") is not True:
                raise ValueError("all_finite must be true")
            if type(absolute) not in (int, float) or not math.isfinite(float(absolute)) or absolute < 0:
                raise ValueError("max_absolute_error must be nonnegative and finite")
            if type(ulp) is not int or ulp < 0:
                raise ValueError("max_bfloat16_ulp must be a nonnegative integer")
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("timing warmup contract drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("median_host_blocking_ns does not match raw samples")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "CUDA-event",
            permitted_none=expected_producer == "ordinary_jax_xla",
        )
        if expected_producer == "ordinary_jax_xla" and cuda is not None:
            raise ValueError("XLA report must not contain CUDA-event samples")
    return dict(document)


def validate_qualification_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate repeated qualification patterns and the untouched acceptance pattern."""
    document = dict(value)
    if set(document) not in ({"schema", "cells"}, {"schema", "policy", "cells"}):
        raise ValueError("qualification evidence fields drifted")
    if document.get("schema") != "shuttle_abi22_rmsnorm_policy_evidence_v2":
        raise ValueError("qualification evidence schema drifted")
    if "policy" in document and document["policy"] != NUMERICAL_POLICY:
        raise ValueError("qualification numerical policy drifted")
    cells = document.get("cells")
    expected_coordinates = [
        (producer, pattern, boundary, repeat)
        for producer in PRODUCERS
        for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN)
        for boundary in BOUNDARIES
        for repeat in range(NUMERICAL_POLICY["repeat_count"])
    ]
    if not isinstance(cells, list) or len(cells) != len(expected_coordinates):
        raise ValueError("qualification cell matrix drifted")
    digests: dict[tuple[str, str, str, int, str], str] = {}
    for cell, (producer, pattern, boundary, repeat) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("qualification cell is not an object")
        if (
            cell.get("producer") != producer
            or cell.get("input_pattern") != pattern
            or cell.get("boundary") != boundary
            or cell.get("repeat") != repeat
            or tuple(cell.get("shape", ())) != REPRESENTATIVE_SHAPE
            or cell.get("scope")
            != (
                "representative_policy_acceptance"
                if pattern == ACCEPTANCE_PATTERN
                else "representative_policy_qualification"
            )
        ):
            raise ValueError("qualification coordinate drifted")
        if pattern == QUALIFICATION_PATTERN:
            input_digests = INPUT_DIGESTS[REPRESENTATIVE_SHAPE]
            reference_digests = REFERENCE_DIGESTS[REPRESENTATIVE_SHAPE]
        elif pattern == HELDOUT_PATTERN:
            input_digests = HELDOUT_INPUT_DIGESTS
            reference_digests = HELDOUT_REFERENCE_DIGESTS
        else:
            input_digests = ACCEPTANCE_INPUT_DIGESTS
            reference_digests = ACCEPTANCE_REFERENCE_DIGESTS
        if cell.get("input_digests") != {
            role: input_digests[role] for role in (("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy"))
        }:
            raise ValueError("qualification input provenance drifted")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("qualification call sequence drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("qualification output role drifted")
        for output, role in zip(outputs, roles, strict=True):
            digest = _sha(output.get("digest"), "qualification output digest")
            if _sha(output.get("reference_digest"), "qualification reference digest") != reference_digests[role]:
                raise ValueError("qualification reference digest drifted")
            absolute = output.get("max_absolute_error")
            mean_absolute = output.get("mean_absolute_error")
            ulp = output.get("max_bfloat16_ulp")
            all_finite = output.get("all_finite")
            if all_finite is not True:
                raise ValueError("qualification all_finite must be true")
            if type(absolute) not in (int, float) or not math.isfinite(float(absolute)) or absolute < 0:
                raise ValueError("qualification max_absolute_error must be finite and nonnegative")
            if type(mean_absolute) not in (int, float) or not math.isfinite(float(mean_absolute)) or mean_absolute < 0:
                raise ValueError("qualification mean_absolute_error must be finite and nonnegative")
            if type(ulp) is not int or ulp < 0:
                raise ValueError("qualification max_bfloat16_ulp must be nonnegative")
            limits = ACCEPTANCE_LIMITS[role]
            floor = output.get("absolute_floor")
            eligible_count = output.get("eligible_count")
            relative = output.get("max_relative_error_above_absolute_floor")
            masked_ulp = output.get("max_bfloat16_ulp_above_absolute_floor")
            if floor != limits["absolute_floor"]:
                raise ValueError("qualification absolute_floor drifted")
            if eligible_count != ELIGIBLE_COUNTS[pattern][role]:
                raise ValueError("qualification eligible_count drifted")
            if type(relative) not in (int, float) or not math.isfinite(float(relative)) or relative < 0:
                raise ValueError("qualification max_relative_error_above_absolute_floor must be finite")
            if type(masked_ulp) is not int or masked_ulp < 0:
                raise ValueError("qualification max_bfloat16_ulp_above_absolute_floor must be nonnegative")
            raw_index = output.get("raw_max_bfloat16_ulp_index")
            if type(raw_index) is not int or raw_index < 0:
                raise ValueError("qualification raw_max_bfloat16_ulp_index must be nonnegative")
            for field in ("raw_max_bfloat16_ulp_actual", "raw_max_bfloat16_ulp_reference"):
                raw_value = output.get(field)
                if type(raw_value) not in (int, float) or not math.isfinite(float(raw_value)):
                    raise ValueError(f"qualification {field} must be finite")
            coordinate = f"{producer}/{pattern}/{boundary}/repeat{repeat}/{role}"
            if pattern == ACCEPTANCE_PATTERN and float(absolute) > limits["max_absolute_error"]:
                raise ValueError(
                    f"{coordinate}: max_absolute_error {absolute} exceeds role limit {limits['max_absolute_error']}"
                )
            if pattern == ACCEPTANCE_PATTERN and float(mean_absolute) > limits["max_mean_absolute_error"]:
                raise ValueError(
                    f"{coordinate}: mean_absolute_error {mean_absolute} exceeds role limit "
                    f"{limits['max_mean_absolute_error']}"
                )
            if pattern == ACCEPTANCE_PATTERN and float(relative) > limits["max_relative_error_above_absolute_floor"]:
                raise ValueError(
                    f"{coordinate}: max_relative_error_above_absolute_floor {relative} exceeds role limit "
                    f"{limits['max_relative_error_above_absolute_floor']}"
                )
            if pattern == ACCEPTANCE_PATTERN and masked_ulp > limits["max_bfloat16_ulp_above_absolute_floor"]:
                raise ValueError(
                    f"{coordinate}: max_bfloat16_ulp_above_absolute_floor {masked_ulp} exceeds role limit "
                    f"{limits['max_bfloat16_ulp_above_absolute_floor']}"
                )
            digests[(producer, pattern, boundary, repeat, role)] = digest
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("qualification timing drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "qualification host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("qualification host median drifted")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "qualification CUDA-event",
            permitted_none=producer == "ordinary_jax_xla",
        )
        if (producer == "transformer_engine_2_17") != (cuda is not None):
            raise ValueError("qualification CUDA-event provider contract drifted")
    for producer in PRODUCERS:
        for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN):
            for boundary in BOUNDARIES:
                for role in OUTPUT_ROLES[boundary]:
                    if (
                        digests[(producer, pattern, boundary, 0, role)]
                        != digests[(producer, pattern, boundary, 1, role)]
                    ):
                        raise ValueError("qualification repeat output digest drifted")
    return {
        "schema": "shuttle_abi22_rmsnorm_policy_evidence_v2",
        "policy": NUMERICAL_POLICY,
        "cells": [dict(cell) for cell in cells],
    }


def _policy_v3_digests(pattern: str) -> tuple[Mapping[str, str], Mapping[str, str]]:
    if pattern == QUALIFICATION_PATTERN:
        return INPUT_DIGESTS[REPRESENTATIVE_SHAPE], REFERENCE_DIGESTS[REPRESENTATIVE_SHAPE]
    if pattern == HELDOUT_PATTERN:
        return HELDOUT_INPUT_DIGESTS, HELDOUT_REFERENCE_DIGESTS
    if pattern == ACCEPTANCE_PATTERN:
        return ACCEPTANCE_INPUT_DIGESTS, ACCEPTANCE_REFERENCE_DIGESTS
    if pattern == POLICY_V3_ACCEPTANCE_PATTERN:
        return POLICY_V3_ACCEPTANCE_INPUT_DIGESTS, POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS
    raise ValueError("policy v3 input pattern drifted")


def _finite_nonnegative(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return float(value)


def validate_policy_v3_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all v3 qualification cells and the untouched acceptance cells."""
    document = dict(value)
    if set(document) not in ({"schema", "cells"}, {"schema", "policy", "cells"}):
        raise ValueError("qualification evidence fields drifted")
    if document.get("schema") != "shuttle_abi22_rmsnorm_policy_evidence_v3":
        raise ValueError("qualification evidence schema drifted")
    if "policy" in document and document["policy"] != NUMERICAL_POLICY_V3:
        raise ValueError("qualification numerical policy drifted")
    patterns = (*POLICY_V3_QUALIFICATION_PATTERNS, POLICY_V3_ACCEPTANCE_PATTERN)
    expected_coordinates = [
        (producer, pattern, boundary, repeat)
        for producer in PRODUCERS
        for pattern in patterns
        for boundary in BOUNDARIES
        for repeat in range(NUMERICAL_POLICY_V3["repeat_count"])
    ]
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_coordinates):
        raise ValueError("qualification cell matrix drifted")
    digests: dict[tuple[str, str, str, int, str], str] = {}
    for cell, (producer, pattern, boundary, repeat) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("qualification cell is not an object")
        expected_scope = (
            "representative_policy_acceptance"
            if pattern == POLICY_V3_ACCEPTANCE_PATTERN
            else "representative_policy_qualification"
        )
        if (
            cell.get("producer") != producer
            or cell.get("input_pattern") != pattern
            or cell.get("boundary") != boundary
            or cell.get("repeat") != repeat
            or tuple(cell.get("shape", ())) != REPRESENTATIVE_SHAPE
            or cell.get("scope") != expected_scope
        ):
            raise ValueError("qualification coordinate drifted")
        input_digests, reference_digests = _policy_v3_digests(pattern)
        input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
        if cell.get("input_digests") != {role: input_digests[role] for role in input_roles}:
            raise ValueError("qualification input provenance drifted")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("qualification call sequence drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("qualification output role drifted")
        for output, role in zip(outputs, roles, strict=True):
            digest = _sha(output.get("digest"), "qualification output digest")
            if _sha(output.get("reference_digest"), "qualification reference digest") != reference_digests[role]:
                raise ValueError("qualification reference digest drifted")
            if output.get("all_finite") is not True:
                raise ValueError("qualification all_finite must be true")
            absolute = _finite_nonnegative(output.get("max_absolute_error"), "qualification max_absolute_error")
            mean_absolute = _finite_nonnegative(output.get("mean_absolute_error"), "qualification mean_absolute_error")
            below_absolute = _finite_nonnegative(
                output.get("max_absolute_error_below_floor"),
                "qualification max_absolute_error_below_floor",
            )
            above_absolute = _finite_nonnegative(
                output.get("max_absolute_error_above_floor"),
                "qualification max_absolute_error_above_floor",
            )
            relative = _finite_nonnegative(
                output.get("max_relative_error_above_absolute_floor"),
                "qualification max_relative_error_above_absolute_floor",
            )
            raw_ulp = output.get("max_bfloat16_ulp")
            masked_ulp = output.get("max_bfloat16_ulp_above_absolute_floor")
            if type(raw_ulp) is not int or raw_ulp < 0 or type(masked_ulp) is not int or masked_ulp < 0:
                raise ValueError("qualification BF16 ULP metrics must be nonnegative integers")
            limits = POLICY_V3_ACCEPTANCE_LIMITS[role]
            if output.get("absolute_floor") != limits["absolute_floor"]:
                raise ValueError("qualification absolute_floor drifted")
            counts = POLICY_V3_PARTITION_COUNTS[pattern][role]
            below_count = output.get("below_floor_count")
            above_count = output.get("eligible_count")
            if below_count != counts["below_floor"] or above_count != counts["above_floor"]:
                raise ValueError("qualification partition count drifted")
            output_size = REPRESENTATIVE_SHAPE[0] * REPRESENTATIVE_SHAPE[1]
            if role == "dgamma":
                output_size = REPRESENTATIVE_SHAPE[1]
            if below_count + above_count != output_size:
                raise ValueError("qualification partition counts do not cover the output")
            if absolute != max(below_absolute, above_absolute):
                raise ValueError("qualification whole max_absolute_error differs from partition maxima")
            raw_index = output.get("raw_max_bfloat16_ulp_index")
            if type(raw_index) is not int or raw_index < 0 or raw_index >= output_size:
                raise ValueError("qualification raw_max_bfloat16_ulp_index drifted")
            for field in ("raw_max_bfloat16_ulp_actual", "raw_max_bfloat16_ulp_reference"):
                raw_value = output.get(field)
                if type(raw_value) not in (int, float) or not math.isfinite(float(raw_value)):
                    raise ValueError(f"qualification {field} must be finite")
            coordinate = f"{producer}/{pattern}/{boundary}/repeat{repeat}/{role}"
            if pattern == POLICY_V3_ACCEPTANCE_PATTERN:
                for observed, field in (
                    (below_absolute, "max_absolute_error_below_floor"),
                    (above_absolute, "max_absolute_error_above_floor"),
                    (mean_absolute, "max_mean_absolute_error"),
                    (relative, "max_relative_error_above_absolute_floor"),
                    (masked_ulp, "max_bfloat16_ulp_above_absolute_floor"),
                ):
                    if observed > limits[field]:
                        raise ValueError(f"{coordinate}: {field} {observed} exceeds role limit {limits[field]}")
            digests[(producer, pattern, boundary, repeat, role)] = digest
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("qualification timing drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "qualification host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("qualification host median drifted")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "qualification CUDA-event",
            permitted_none=producer == "ordinary_jax_xla",
        )
        if (producer == "transformer_engine_2_17") != (cuda is not None):
            raise ValueError("qualification CUDA-event provider contract drifted")
    for producer in PRODUCERS:
        for pattern in patterns:
            for boundary in BOUNDARIES:
                for role in OUTPUT_ROLES[boundary]:
                    if (
                        digests[(producer, pattern, boundary, 0, role)]
                        != digests[(producer, pattern, boundary, 1, role)]
                    ):
                        raise ValueError("qualification repeat output digest drifted")
    return {
        "schema": "shuttle_abi22_rmsnorm_policy_evidence_v3",
        "policy": NUMERICAL_POLICY_V3,
        "cells": [dict(cell) for cell in cells],
    }


def _policy_v4_digests(pattern: str) -> tuple[Mapping[str, str], Mapping[str, str]]:
    if pattern in POLICY_V4_QUALIFICATION_PATTERNS:
        return _policy_v3_digests(pattern)
    if pattern == POLICY_V4_ACCEPTANCE_PATTERN:
        return POLICY_V4_ACCEPTANCE_INPUT_DIGESTS, POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS
    raise ValueError("policy v4 input pattern drifted")


def validate_policy_v4_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate v4 structure while gating numerics only on the untouched pattern."""
    document = dict(value)
    if set(document) not in ({"schema", "cells"}, {"schema", "policy", "cells"}):
        raise ValueError("qualification evidence fields drifted")
    if document.get("schema") != "shuttle_abi22_rmsnorm_policy_evidence_v4":
        raise ValueError("qualification evidence schema drifted")
    if "policy" in document and document["policy"] != NUMERICAL_POLICY_V4:
        raise ValueError("qualification numerical policy drifted")
    patterns = (*POLICY_V4_QUALIFICATION_PATTERNS, POLICY_V4_ACCEPTANCE_PATTERN)
    expected_coordinates = [
        (producer, pattern, boundary, repeat)
        for producer in PRODUCERS
        for pattern in patterns
        for boundary in BOUNDARIES
        for repeat in range(NUMERICAL_POLICY_V4["repeat_count"])
    ]
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_coordinates):
        raise ValueError("qualification cell matrix drifted")
    digests: dict[tuple[str, str, str, int, str], str] = {}
    for cell, (producer, pattern, boundary, repeat) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("qualification cell is not an object")
        expected_scope = (
            "representative_policy_acceptance"
            if pattern == POLICY_V4_ACCEPTANCE_PATTERN
            else "representative_policy_qualification"
        )
        if (
            cell.get("producer") != producer
            or cell.get("input_pattern") != pattern
            or cell.get("boundary") != boundary
            or cell.get("repeat") != repeat
            or tuple(cell.get("shape", ())) != REPRESENTATIVE_SHAPE
            or cell.get("scope") != expected_scope
        ):
            raise ValueError("qualification coordinate drifted")
        input_digests, reference_digests = _policy_v4_digests(pattern)
        input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
        if cell.get("input_digests") != {role: input_digests[role] for role in input_roles}:
            raise ValueError("qualification input provenance drifted")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("qualification call sequence drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("qualification output role drifted")
        for output, role in zip(outputs, roles, strict=True):
            digest = _sha(output.get("digest"), "qualification output digest")
            if _sha(output.get("reference_digest"), "qualification reference digest") != reference_digests[role]:
                raise ValueError("qualification reference digest drifted")
            if output.get("all_finite") is not True:
                raise ValueError("qualification all_finite must be true")
            absolute = _finite_nonnegative(output.get("max_absolute_error"), "qualification max_absolute_error")
            mean_absolute = _finite_nonnegative(output.get("mean_absolute_error"), "qualification mean_absolute_error")
            below_absolute = _finite_nonnegative(
                output.get("max_absolute_error_below_floor"),
                "qualification max_absolute_error_below_floor",
            )
            above_absolute = _finite_nonnegative(
                output.get("max_absolute_error_above_floor"),
                "qualification max_absolute_error_above_floor",
            )
            relative = _finite_nonnegative(
                output.get("max_relative_error_above_absolute_floor"),
                "qualification max_relative_error_above_absolute_floor",
            )
            raw_ulp = output.get("max_bfloat16_ulp")
            masked_ulp = output.get("max_bfloat16_ulp_above_absolute_floor")
            if type(raw_ulp) is not int or raw_ulp < 0 or type(masked_ulp) is not int or masked_ulp < 0:
                raise ValueError("qualification BF16 ULP metrics must be nonnegative integers")
            limits = POLICY_V4_ACCEPTANCE_LIMITS[role]
            if output.get("absolute_floor") != limits["absolute_floor"]:
                raise ValueError("qualification absolute_floor drifted")
            counts = POLICY_V4_PARTITION_COUNTS[pattern][role]
            below_count = output.get("below_floor_count")
            above_count = output.get("eligible_count")
            if below_count != counts["below_floor"] or above_count != counts["above_floor"]:
                raise ValueError("qualification partition count drifted")
            output_size = REPRESENTATIVE_SHAPE[1] if role == "dgamma" else math.prod(REPRESENTATIVE_SHAPE)
            if below_count + above_count != output_size:
                raise ValueError("qualification partition counts do not cover the output")
            if absolute != max(below_absolute, above_absolute):
                raise ValueError("qualification whole max_absolute_error differs from partition maxima")
            raw_index = output.get("raw_max_bfloat16_ulp_index")
            if type(raw_index) is not int or raw_index < 0 or raw_index >= output_size:
                raise ValueError("qualification raw_max_bfloat16_ulp_index drifted")
            for field in ("raw_max_bfloat16_ulp_actual", "raw_max_bfloat16_ulp_reference"):
                raw_value = output.get(field)
                if type(raw_value) not in (int, float) or not math.isfinite(float(raw_value)):
                    raise ValueError(f"qualification {field} must be finite")
            coordinate = f"{producer}/{pattern}/{boundary}/repeat{repeat}/{role}"
            if pattern == POLICY_V4_ACCEPTANCE_PATTERN:
                for observed, field in (
                    (below_absolute, "max_absolute_error_below_floor"),
                    (above_absolute, "max_absolute_error_above_floor"),
                    (mean_absolute, "max_mean_absolute_error"),
                    (relative, "max_relative_error_above_absolute_floor"),
                    (masked_ulp, "max_bfloat16_ulp_above_absolute_floor"),
                ):
                    if observed > limits[field]:
                        raise ValueError(f"{coordinate}: {field} {observed} exceeds role limit {limits[field]}")
            digests[(producer, pattern, boundary, repeat, role)] = digest
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("qualification timing drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "qualification host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("qualification host median drifted")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "qualification CUDA-event",
            permitted_none=producer == "ordinary_jax_xla",
        )
        if (producer == "transformer_engine_2_17") != (cuda is not None):
            raise ValueError("qualification CUDA-event provider contract drifted")

    qualification_drifts = []
    for producer in PRODUCERS:
        for pattern in patterns:
            for boundary in BOUNDARIES:
                for role in OUTPUT_ROLES[boundary]:
                    first = digests[(producer, pattern, boundary, 0, role)]
                    second = digests[(producer, pattern, boundary, 1, role)]
                    if first == second:
                        continue
                    if pattern == POLICY_V4_ACCEPTANCE_PATTERN:
                        raise ValueError("acceptance repeat output digest drifted")
                    qualification_drifts.append(
                        {
                            "producer": producer,
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "role": role,
                            "repeat0_digest": first,
                            "repeat1_digest": second,
                        }
                    )
    return {
        "schema": "shuttle_abi22_rmsnorm_policy_evidence_v4",
        "policy": NUMERICAL_POLICY_V4,
        "cells": [dict(cell) for cell in cells],
        "acceptance": {
            "status": "accepted",
            "input_pattern": POLICY_V4_ACCEPTANCE_PATTERN,
            "repeat_coordinate_count": sum(len(OUTPUT_ROLES[boundary]) for boundary in BOUNDARIES) * len(PRODUCERS),
        },
        "qualification": {
            "status": "diagnostic_only",
            "patterns": list(POLICY_V4_QUALIFICATION_PATTERNS),
            "repeat_digest_drifts": qualification_drifts,
        },
    }


def build_matched_acceptance(
    te_report: Mapping[str, Any],
    xla_report: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    expected_te_identity: ValidatedTeRuntimeIdentity,
) -> dict[str, Any]:
    """Build the representative numerical acceptance and clock-matched timing summary."""
    for report in (te_report, xla_report):
        cells = report.get("cells", [])
        if not any(tuple(cell.get("shape", ())) == REPRESENTATIVE_SHAPE for cell in cells):
            raise ValueError("representative cells are missing")
    te = validate_report(
        te_report,
        expected_producer="transformer_engine_2_17",
        expected_te_identity=expected_te_identity,
    )
    xla = validate_report(xla_report, expected_producer="ordinary_jax_xla")
    qualified = validate_qualification_evidence(qualification)
    if te["device"] != xla["device"]:
        raise ValueError("producer device identities differ")
    ratios = {}
    for boundary in BOUNDARIES:
        te_cell = next(
            cell for cell in te["cells"] if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        xla_cell = next(
            cell
            for cell in xla["cells"]
            if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        if te_cell["input_digests"] != xla_cell["input_digests"]:
            raise ValueError("producer input digests differ")
        ratios[boundary] = te_cell["timing"]["median_host_blocking_ns"] / xla_cell["timing"]["median_host_blocking_ns"]
    return {
        "schema": "shuttle_abi22_te_xla_rmsnorm_acceptance_v2",
        "status": "heldout_splitmix64_numerics_accepted",
        "accepted_input_pattern": ACCEPTANCE_PATTERN,
        "accepted_shape": list(REPRESENTATIVE_SHAPE),
        "accepted_boundaries": list(BOUNDARIES),
        "excluded_architecture_fixture_shape": list(ARCHITECTURE_SHAPE),
        "numerical_policy": NUMERICAL_POLICY,
        "metric_scope": {
            "whole_output": ["all_finite", "max_absolute_error", "mean_absolute_error"],
            "above_absolute_floor": [
                "eligible_count_nonzero",
                "max_relative_error",
                "max_bfloat16_ulp",
            ],
            "raw_global_bfloat16_ulp": "diagnostic_only",
        },
        "qualification": qualified,
        "performance": {
            "status": "reported_without_threshold",
            "clock": "host_blocking_ns",
            "input_pattern": QUALIFICATION_PATTERN,
            "median_ratio_te_over_xla": ratios,
        },
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }


def build_matched_acceptance_v3(
    te_report: Mapping[str, Any],
    xla_report: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    expected_te_identity: ValidatedTeRuntimeIdentity,
) -> dict[str, Any]:
    """Build the v3 SplitMix acceptance and qualification-timing summary."""
    for report in (te_report, xla_report):
        cells = report.get("cells", [])
        if not any(tuple(cell.get("shape", ())) == REPRESENTATIVE_SHAPE for cell in cells):
            raise ValueError("representative cells are missing")
    te = validate_report(
        te_report,
        expected_producer="transformer_engine_2_17",
        expected_te_identity=expected_te_identity,
    )
    xla = validate_report(xla_report, expected_producer="ordinary_jax_xla")
    qualified = validate_policy_v3_evidence(qualification)
    if te["device"] != xla["device"]:
        raise ValueError("producer device identities differ")
    ratios = {}
    for boundary in BOUNDARIES:
        te_cell = next(
            cell for cell in te["cells"] if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        xla_cell = next(
            cell
            for cell in xla["cells"]
            if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        if te_cell["input_digests"] != xla_cell["input_digests"]:
            raise ValueError("producer input digests differ")
        ratios[boundary] = te_cell["timing"]["median_host_blocking_ns"] / xla_cell["timing"]["median_host_blocking_ns"]
    return {
        "schema": "shuttle_abi22_te_xla_rmsnorm_acceptance_v3",
        "status": "heldout_splitmix64_v2_numerics_accepted",
        "accepted_input_pattern": POLICY_V3_ACCEPTANCE_PATTERN,
        "accepted_shape": list(REPRESENTATIVE_SHAPE),
        "accepted_boundaries": list(BOUNDARIES),
        "excluded_architecture_fixture_shape": list(ARCHITECTURE_SHAPE),
        "numerical_policy": NUMERICAL_POLICY_V3,
        "metric_scope": {
            "whole_output": ["all_finite", "mean_absolute_error"],
            "below_absolute_floor": ["exact_count", "max_absolute_error"],
            "above_absolute_floor": [
                "exact_count",
                "max_absolute_error",
                "max_relative_error",
                "max_bfloat16_ulp",
            ],
            "partition_closure": "whole_max_equals_max_of_disjoint_partition_maxima",
            "raw_global_bfloat16_ulp": "diagnostic_only",
            "repeatability": "exact_bfloat16_output_digest_per_coordinate",
        },
        "qualification": qualified,
        "performance": {
            "status": "reported_without_threshold",
            "clock": "host_blocking_ns",
            "input_pattern": QUALIFICATION_PATTERN,
            "median_ratio_te_over_xla": ratios,
        },
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }


def build_matched_acceptance_v4(
    te_report: Mapping[str, Any],
    xla_report: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    expected_te_identity: ValidatedTeRuntimeIdentity,
) -> dict[str, Any]:
    """Build v4 acceptance from the untouched pattern and diagnostic qualification."""
    for report in (te_report, xla_report):
        cells = report.get("cells", [])
        if not any(tuple(cell.get("shape", ())) == REPRESENTATIVE_SHAPE for cell in cells):
            raise ValueError("representative cells are missing")
    te = validate_report(
        te_report,
        expected_producer="transformer_engine_2_17",
        expected_te_identity=expected_te_identity,
    )
    xla = validate_report(xla_report, expected_producer="ordinary_jax_xla")
    qualified = validate_policy_v4_evidence(qualification)
    if te["device"] != xla["device"]:
        raise ValueError("producer device identities differ")
    ratios = {}
    for boundary in BOUNDARIES:
        te_cell = next(
            cell for cell in te["cells"] if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        xla_cell = next(
            cell
            for cell in xla["cells"]
            if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        if te_cell["input_digests"] != xla_cell["input_digests"]:
            raise ValueError("producer input digests differ")
        ratios[boundary] = te_cell["timing"]["median_host_blocking_ns"] / xla_cell["timing"]["median_host_blocking_ns"]
    return {
        "schema": "shuttle_abi22_te_xla_rmsnorm_acceptance_v4",
        "status": "heldout_splitmix64_v3_numerics_accepted",
        "accepted_input_pattern": POLICY_V4_ACCEPTANCE_PATTERN,
        "accepted_shape": list(REPRESENTATIVE_SHAPE),
        "accepted_boundaries": list(BOUNDARIES),
        "excluded_architecture_fixture_shape": list(ARCHITECTURE_SHAPE),
        "numerical_policy": NUMERICAL_POLICY_V4,
        "metric_scope": {
            "whole_output": ["all_finite", "mean_absolute_error"],
            "below_absolute_floor": ["exact_count", "max_absolute_error"],
            "above_absolute_floor": [
                "exact_count",
                "max_absolute_error",
                "max_relative_error",
                "max_bfloat16_ulp",
            ],
            "partition_closure": "whole_max_equals_max_of_disjoint_partition_maxima",
            "raw_global_bfloat16_ulp": "diagnostic_only",
            "repeatability": "exact_bfloat16_output_digest_per_acceptance_coordinate",
            "qualification_repeatability": "diagnostic_only_reported_without_rejection",
        },
        "qualification": qualified,
        "performance": {
            "status": "reported_without_threshold",
            "clock": "host_blocking_ns",
            "input_pattern": QUALIFICATION_PATTERN,
            "median_ratio_te_over_xla": ratios,
        },
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }
