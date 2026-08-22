# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prospective numerical contract for anonymous Contract/Fold composition."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POLICY_FILENAME = "track1-generic-composition-h100-numerical-policy-v1.json"
POLICY_PATH = Path(__file__).with_name(POLICY_FILENAME)
POLICY_SHA256 = "202413163a8c78067b88043db7207eafdeb5b58116341e186398572280466a21"


@dataclass(frozen=True)
class Case:
    case_id: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    p_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    seed: int
    input_sha256: dict[str, str]
    oracle_f32_sha256: str
    stablehlo_sha256: str
    limits: dict[str, float]


def array_sha256(value: np.ndarray) -> str:
    """Hash one array with its exact dtype and shape."""
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(original.dtype.str.encode())
    digest.update(np.asarray(original.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def load_numerical_policy() -> tuple[dict[str, object], tuple[Case, ...]]:
    """Load and validate the immutable prospective policy."""
    raw = POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise ValueError("generic composition numerical policy digest differs")
    document = json.loads(raw)
    canonical = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if raw != canonical or type(document) is not dict:
        raise ValueError("generic composition numerical policy is not canonical")
    if set(document) != {"cases", "schema", "schema_version", "semantic_contract", "version"}:
        raise ValueError("generic composition numerical policy schema is open")
    if (
        document["schema"] != "track1_generic_composition_h100_numerical_policy"
        or document["schema_version"] != 1
        or document["version"] != "prospective-v1"
    ):
        raise ValueError("generic composition numerical policy identity differs")
    expected_contract = {
        "groups": [
            {"kind": "contract_fold", "private_outputs": ["h", "s"], "sources": ["contract0", "fold"]},
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
            "oracle_f32_sha256",
            "p_shape",
            "result_shape",
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
            oracle_f32_sha256=value["oracle_f32_sha256"],
            stablehlo_sha256=value["stablehlo_sha256"],
            limits={name: float(limit) for name, limit in limits.items()},
        )
        inputs = deterministic_inputs(case)
        rounded = binary64_oracle(*inputs).astype(np.float32)
        if rounded.shape != case.result_shape or array_sha256(rounded) != case.oracle_f32_sha256:
            raise ValueError("generic composition binary64 oracle differs from policy")
        if not np.all(np.isfinite(rounded)) or np.any(rounded == 0):
            raise ValueError("generic composition oracle must bind finite nonzero full writes")
        cases.append(case)
    if [case.case_id for case in cases] != ["rank2-a", "rank3-b"]:
        raise ValueError("generic composition case order differs")
    return document, tuple(cases)


POLICY, CASES = load_numerical_policy()
SEMANTIC_CONTRACT = POLICY["semantic_contract"]
