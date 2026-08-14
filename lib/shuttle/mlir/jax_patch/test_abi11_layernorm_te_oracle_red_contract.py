# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Red contract for the missing ABI11 Transformer Engine LayerNorm oracle."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import pytest
from shuttle_jaxlib_gpu_layer_norm_acceptance import NUMERICAL_LIMITS

CONTRACT = Path(__file__).with_name("abi11-layernorm-bf16-te-2.17-oracle-red-v1.json")
METRICS = ("max_absolute_error", "mean_absolute_error", "relative_linf_error", "max_bfloat16_ulp_error")
INPUT_ROLES = ("x", "gamma", "beta", "dy")
OUTPUT_ROLES = ("y", "dx", "dgamma", "dbeta")
RELATIVE_SCALE_FLOOR = 2**-7


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate contract key: {key}")
        value[key] = item
    return value


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(), object_pairs_hook=_unique_object)


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _fixed_inputs(rows: int, columns: int) -> tuple[np.ndarray, ...]:
    shapes = ((rows, columns), (columns,), (columns,), (rows, columns))
    ranges = ((-0.75, 0.875), (-0.625, 1.0), (-0.25, 0.375), (-0.5, 0.75))
    return tuple(
        np.linspace(start, stop, math.prod(shape), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for shape, (start, stop) in zip(shapes, ranges, strict=True)
    )


def _reference_outputs(rows: int, columns: int) -> tuple[np.ndarray, ...]:
    x, gamma, beta, dy = (value.astype(np.float64) for value in _fixed_inputs(rows, columns))
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    inverse = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inverse
    y = normalized * gamma + beta
    scaled_cotangent = dy * gamma
    dx = (
        scaled_cotangent
        - np.mean(scaled_cotangent, axis=-1, keepdims=True)
        - normalized * np.mean(scaled_cotangent * normalized, axis=-1, keepdims=True)
    ) * inverse
    dgamma = np.sum(dy * normalized, axis=0)
    dbeta = np.sum(dy, axis=0)
    return tuple(np.asarray(value, dtype=ml_dtypes.bfloat16) for value in (y, dx, dgamma, dbeta))


def _dtype_floor(reference: np.ndarray) -> dict[str, float | int]:
    positive = np.nextafter(reference, np.array(np.inf, dtype=ml_dtypes.bfloat16)).astype(np.float64)
    negative = np.nextafter(reference, np.array(-np.inf, dtype=ml_dtypes.bfloat16)).astype(np.float64)
    reference64 = reference.astype(np.float64)
    spacing = np.maximum(np.abs(positive - reference64), np.abs(negative - reference64))
    maximum = float(np.max(spacing))
    scale = max(float(np.max(np.abs(reference64))), RELATIVE_SCALE_FLOOR)
    return {
        "max_absolute_error": maximum,
        "mean_absolute_error": float(np.mean(spacing)),
        "relative_linf_error": maximum / scale,
        "max_bfloat16_ulp_error": 1,
    }


def _require_objective(
    candidate: dict[str, float | int], expert: dict[str, float | int], floor: dict[str, float | int]
) -> None:
    for metric in METRICS:
        if candidate[metric] > max(expert[metric], floor[metric]):
            raise AssertionError(f"{metric} exceeds the matched expert-or-dtype-floor limit")


def test_contract_pins_the_audited_te_layernorm_api() -> None:
    contract = _load_contract()
    assert set(contract) == {
        "schema_version",
        "contract_id",
        "status",
        "source_base",
        "provider",
        "api",
        "runner",
        "coordinates",
        "inputs",
        "boundaries",
        "metrics",
        "objective",
        "artifacts",
        "claims",
    }
    assert contract["provider"]["commit"] == "2e559f062497bef768dfbe9d7e45548fadeca80a"
    assert contract["provider"]["source_files"][contract["api"]["header"]] == (
        "9533ef14eb870236c3200c9f143973e870ef52c4d07ddeb74fc2c4acd948fbc2"
    )
    signatures = contract["api"]["signatures"]
    assert set(signatures) == {
        "nvte_layernorm_fwd",
        "nvte_layernorm_bwd",
        "nvte_enable_cudnn_norm_fwd",
        "nvte_enable_cudnn_norm_bwd",
    }
    for record in signatures.values():
        assert hashlib.sha256(record["declaration"].encode()).hexdigest() == record["sha256"]
    assert "beta" in signatures["nvte_layernorm_fwd"]["declaration"]
    assert "mu" in signatures["nvte_layernorm_bwd"]["declaration"]
    assert "dbeta" in signatures["nvte_layernorm_bwd"]["declaration"]


def test_contract_binds_the_four_canonical_inputs_for_both_shapes() -> None:
    contract = _load_contract()
    assert contract["inputs"]["ranges"] == {
        "x": [-0.75, 0.875],
        "gamma": [-0.625, 1.0],
        "beta": [-0.25, 0.375],
        "dy": [-0.5, 0.75],
    }
    for rows, columns in ((2048, 4096), (7, 13)):
        records = contract["inputs"]["digests"][f"{rows}x{columns}"]
        assert tuple(records) == INPUT_ROLES
        for role, value in zip(INPUT_ROLES, _fixed_inputs(rows, columns), strict=True):
            assert records[role] == {
                "raw_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "array_sha256": _array_digest(value),
            }


def test_contract_separates_recompute_backward_from_saved_state_composed() -> None:
    contract = _load_contract()
    boundaries = contract["boundaries"]
    assert boundaries["forward"]["public_inputs"] == ["x", "gamma", "beta"]
    assert boundaries["forward"]["public_outputs"] == ["y"]
    assert boundaries["backward_recompute"]["public_inputs"] == ["x", "gamma", "dy"]
    assert boundaries["backward_recompute"]["public_outputs"] == ["dx", "dgamma", "dbeta"]
    assert "zero_beta" in boundaries["backward_recompute"]["calls"][0]
    assert "throwaway_y" in boundaries["backward_recompute"]["calls"][0]
    assert "recomputed_inside_the_boundary" in boundaries["backward_recompute"]["state"]
    assert boundaries["composed"]["public_inputs"] == ["x", "gamma", "beta", "dy"]
    assert boundaries["composed"]["public_outputs"] == ["y", "dx", "dgamma", "dbeta"]
    assert "saved_only_within_the_same_composed_invocation" in boundaries["composed"]["state"]
    assert (
        sum(len(pairs) for pairs in contract["coordinates"]["backend_pairs"].values())
        * len(contract["coordinates"]["shapes"])
        == contract["runner"]["expected_plan_records"]
    )


def test_contract_derives_metric_floors_and_enforces_every_matched_oracle() -> None:
    contract = _load_contract()
    assert contract["metrics"]["names"] == list(METRICS)
    assert contract["metrics"]["public_output_roles"] == {
        "forward": ["y"],
        "backward_recompute": ["dx", "dgamma", "dbeta"],
        "composed": ["y", "dx", "dgamma", "dbeta"],
    }
    assert contract["objective"]["expert_reference_limits"] == {
        shape: {
            role: {"max_absolute_error": limits[0], "mean_absolute_error": limits[1]}
            for role, limits in NUMERICAL_LIMITS[shape_id].items()
        }
        for shape_id, shape in (("44d152ecc3e9ff18", "2048x4096"), ("81928ab3539c0f03", "7x13"))
    }

    for rows, columns in ((2048, 4096), (7, 13)):
        floors = contract["metrics"]["dtype_floors"][f"{rows}x{columns}"]
        references = _reference_outputs(rows, columns)
        assert floors == {
            role: _dtype_floor(reference) for role, reference in zip(OUTPUT_ROLES, references, strict=True)
        }

    floor = contract["metrics"]["dtype_floors"]["7x13"]["y"]
    expert = {metric: 0 for metric in METRICS}
    _require_objective(dict(floor), expert, floor)
    rejected = dict(floor)
    rejected["max_bfloat16_ulp_error"] = 2
    with pytest.raises(AssertionError, match="matched expert-or-dtype-floor"):
        _require_objective(rejected, expert, floor)
    assert contract["objective"]["matched_oracle_selection"] == (
        "all_applicable_predeclared_backend_pairs_no_post_hoc_selector"
    )


def test_layernorm_te_runner_emits_the_closed_cpu_only_dry_run_plan() -> None:
    """Remain red until a runner implements the reviewed no-CUDA plan boundary."""
    contract = _load_contract()
    runner = CONTRACT.parent / Path(contract["runner"]["path"]).name
    assert runner.is_file(), "ABI11 Transformer Engine LayerNorm oracle runner is not implemented"
    result = subprocess.run(
        [sys.executable, str(runner), *contract["runner"]["dry_run_argv"]],
        cwd=CONTRACT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout, object_pairs_hook=_unique_object)
    assert set(plan) == {"schema_version", "contract_id", "hardware_required", "records"}
    assert plan["contract_id"] == contract["contract_id"]
    assert plan["hardware_required"] is False
    assert len(plan["records"]) == contract["runner"]["expected_plan_records"]
