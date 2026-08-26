# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prospective numerical contract for anonymous Contract/Fold composition."""

import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POLICY_FILENAME = "track1-generic-composition-h100-numerical-policy-v2.json"
POLICY_PATH = Path(__file__).with_name(POLICY_FILENAME)
POLICY_SHA256 = "78578508ee2106dccc205d6338a6ca94b0ee4d4562766b8a2bf11eae51e11984"
ORACLE_SOURCE_SHA256 = "493b1766f36b197fb0189df5a3e82631bb66526670f72ec21f0e9e675c0a8d65"
CELL_REPORT_PREFIX = "GENERIC_COMPOSITION_CELL "
MAX_CELL_REPORT_BYTES = 4096


@dataclass(frozen=True)
class Case:
    case_id: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    p_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    seed: int
    input_sha256: dict[str, str]
    rounded_f32_oracle_sha256: str
    stablehlo_sha256: str
    limits: dict[str, float]


def array_sha256(value: np.ndarray) -> str:
    """Hash one array with its exact dtype and shape."""
    original = np.asarray(value)
    dtype = original.dtype.newbyteorder("<")
    array = np.ascontiguousarray(original.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(dtype.str.encode())
    digest.update(np.asarray(original.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_inputs(
    case: Case,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the prospectively sealed public inputs for one case."""
    rng = np.random.default_rng(case.seed)
    a = (rng.standard_normal(case.a_shape) * 0.15 + 0.05).astype(np.float32)
    b = (rng.standard_normal(case.b_shape) * 0.12 - 0.03).astype(np.float32)
    p = (rng.standard_normal(case.p_shape) * 0.08 + 0.02).astype(np.float32)
    init = np.asarray(0, dtype=np.float32)
    values = {"a": a, "b": b, "p": p, "init": init}
    if {name: array_sha256(value) for name, value in values.items()} != case.input_sha256:
        raise ValueError("generic composition deterministic inputs differ from policy")
    return a, b, p, init


def binary64_oracle(
    a: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
    init: np.ndarray,
) -> np.ndarray:
    """Evaluate the anonymous public graph independently in binary64."""
    h = a.astype(np.float64) @ b.astype(np.float64)
    s = np.sum(h, axis=-1) + float(init)
    return (h * s[..., None]) @ p.astype(np.float64)


def oracle_source_sha256() -> str:
    """Hash the normalized source of the sealed binary64 oracle."""
    source = textwrap.dedent(inspect.getsource(binary64_oracle)).replace("\r\n", "\n").rstrip() + "\n"
    return hashlib.sha256(source.encode()).hexdigest()


def load_numerical_policy() -> tuple[dict[str, object], tuple[Case, ...]]:
    """Load and validate the immutable prospective policy."""
    raw = POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise ValueError("generic composition numerical policy digest differs")
    document = json.loads(raw)
    canonical = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if raw != canonical or type(document) is not dict:
        raise ValueError("generic composition numerical policy is not canonical")
    if set(document) != {
        "acceptance",
        "cases",
        "metric_contract",
        "oracle_contract",
        "schema",
        "schema_version",
        "semantic_contract",
        "version",
    }:
        raise ValueError("generic composition numerical policy schema is open")
    if (
        document["schema"] != "track1_generic_composition_h100_numerical_policy"
        or document["schema_version"] != 2
        or document["version"] != "prospective-v2"
    ):
        raise ValueError("generic composition numerical policy identity differs")
    expected_contract = {
        "groups": [
            {
                "kind": "contract_fold",
                "private_outputs": ["h", "s"],
                "sources": ["contract0", "fold"],
            },
            {
                "kind": "map_contract",
                "private_inputs": ["h", "s"],
                "public_outputs": ["y"],
                "sources": ["map", "contract1"],
            },
        ],
        "public_arguments": ["a", "b", "p", "init"],
        "public_results": ["y"],
    }
    if document["semantic_contract"] != expected_contract:
        raise ValueError("generic composition semantic contract differs")
    if document["acceptance"] != {
        "disabled_jax": "required_diagnostic_never_numerical_gate",
        "enabled_modes": ["source_ordered", "fast"],
        "enabled_thresholds": "case_limits",
        "nonfinite_action": "reject_all_modes",
        "raw_and_masked_bf16_ulp": "validated_diagnostic_only_no_magnitude_gate",
        "repeat_bitwise_required": True,
    }:
        raise ValueError("generic composition acceptance contract differs")
    if document["metric_contract"] != {
        "maximum_absolute_error": "max_abs_actual_f32_minus_binary64_reference",
        "mean_absolute_error": "mean_abs_actual_f32_minus_binary64_reference",
        "relative_linf_error": "maximum_absolute_error_over_max_reference_abs_or_smallest_positive_binary64",
        "runtime_metrics": "finite_recorded_values_not_byte_identity",
    }:
        raise ValueError("generic composition metric contract differs")
    if (
        document["oracle_contract"]
        != {
            "computation": "binary64_contract_fold_map_contract_v1",
            "identity_sha256": (
                "sha256(dtype_little_endian_string||shape_little_endian_i64||c_order_little_endian_bytes)"
            ),
            "platform_variation": "binary64_low_bits_may_vary_if_rounded_f32_identity_matches",
            "rounded_output": "ieee754_binary32_round_to_nearest_even",
            "source_normalization": "dedent_lf_strip_trailing_append_lf_utf8",
            "source_sha256": ORACLE_SOURCE_SHA256,
        }
        or oracle_source_sha256() != ORACLE_SOURCE_SHA256
    ):
        raise ValueError("generic composition oracle source identity differs")
    raw_cases = document["cases"]
    if type(raw_cases) is not list or len(raw_cases) != 2:
        raise ValueError("generic composition policy must contain two cases")
    cases = []
    for value in raw_cases:
        if type(value) is not dict or set(value) != {
            "a_shape",
            "b_shape",
            "case_id",
            "input_sha256",
            "limits",
            "p_shape",
            "result_shape",
            "rounded_f32_oracle_sha256",
            "seed",
            "stablehlo_sha256",
        }:
            raise ValueError("generic composition case schema differs")
        limits = value["limits"]
        if type(limits) is not dict or set(limits) != {
            "maximum_absolute_error",
            "mean_absolute_error",
            "relative_linf_error",
        }:
            raise ValueError("generic composition limits schema differs")
        if any(type(limit) not in (int, float) or limit <= 0 for limit in limits.values()):
            raise ValueError("generic composition limits must be positive")
        case = Case(
            case_id=value["case_id"],
            a_shape=tuple(value["a_shape"]),
            b_shape=tuple(value["b_shape"]),
            p_shape=tuple(value["p_shape"]),
            result_shape=tuple(value["result_shape"]),
            seed=value["seed"],
            input_sha256=dict(value["input_sha256"]),
            rounded_f32_oracle_sha256=value["rounded_f32_oracle_sha256"],
            stablehlo_sha256=value["stablehlo_sha256"],
            limits={name: float(limit) for name, limit in limits.items()},
        )
        inputs = deterministic_inputs(case)
        rounded = binary64_oracle(*inputs).astype(np.float32)
        if rounded.shape != case.result_shape or array_sha256(rounded) != case.rounded_f32_oracle_sha256:
            raise ValueError("generic composition binary64 oracle differs from policy")
        if not np.all(np.isfinite(rounded)):
            raise ValueError("generic composition oracle must be finite")
        cases.append(case)
    if [case.case_id for case in cases] != ["rank2-a", "rank3-b"]:
        raise ValueError("generic composition case order differs")
    return document, tuple(cases)


POLICY, CASES = load_numerical_policy()
SEMANTIC_CONTRACT = POLICY["semantic_contract"]
