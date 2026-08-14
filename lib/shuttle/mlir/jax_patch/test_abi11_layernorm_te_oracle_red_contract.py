# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract and runner tests for the ABI11 Transformer Engine LayerNorm oracle."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import pytest
from abi11_layernorm_te_oracle_runner import (
    ADAPTER,
    ADAPTER_SHA256,
    SUBMODULES,
    ExecutionConfig,
    _validate_adapter_metadata,
    _write_inputs,
    execute_record,
    validate_checkout_identity,
    verify_file_identity,
    verify_source_checkout,
)
from abi11_layernorm_te_oracle_runner import (
    load_contract as load_runner_contract,
)
from shuttle_jaxlib_gpu_layer_norm_acceptance import NUMERICAL_LIMITS

CONTRACT = Path(__file__).with_name("abi11-layernorm-bf16-te-2.17-oracle-red-v1.json")
RUNNER = Path(__file__).with_name("abi11_layernorm_te_oracle_runner.py")
METRICS = ("max_absolute_error", "mean_absolute_error", "relative_linf_error", "max_bfloat16_ulp_error")
INPUT_ROLES = ("x", "gamma", "beta", "dy")
OUTPUT_ROLES = ("y", "dx", "dgamma", "dbeta")
RELATIVE_SCALE_FLOOR = 2**-7
EXPECTED_SOURCE_BASE = {
    "marin_commit": "5a126aeb6e6069e854d7b3b32709a94eccb26a29",
    "marin_tree": "0571572dda7c61f238086afc569314363b289d25",
}
EXPECTED_PROVIDER = {
    "distribution": "transformer_engine",
    "version": "2.17.0",
    "tag": "v2.17",
    "commit": "2e559f062497bef768dfbe9d7e45548fadeca80a",
    "audit_checkout_state": "clean_detached_exact_commit_with_uninitialized_submodules",
    "source_files": {
        "transformer_engine/common/include/transformer_engine/normalization.h": (
            "9533ef14eb870236c3200c9f143973e870ef52c4d07ddeb74fc2c4acd948fbc2"
        ),
        "transformer_engine/common/include/transformer_engine/transformer_engine.h": (
            "29963fa4df150956b7775b0e80e88e657def80b22397eff7710a33c40b9a4c33"
        ),
        "transformer_engine/common/normalization/common.cpp": (
            "f7a24af63ca0110a888ebbf2a3ab5c209035be5973e708186a376d5990bf37a7"
        ),
        "transformer_engine/common/normalization/layernorm/ln_api.cpp": (
            "ad1abb6e76d0f2bdaea28877ab1eca741c797dc088278f2f760204cf222b9f53"
        ),
        "transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu": (
            "863a89caf4d930ef346facd08a3b3ac5cff3f71118c1199eaf21f30aaf7f8879"
        ),
        "transformer_engine/common/normalization/layernorm/ln_bwd_semi_cuda_kernel.cu": (
            "8c4549440568ca34bbe4e0f00b22f38e82bde0f68a732b0f1260ccd324128187"
        ),
    },
}
EXPECTED_SIGNATURES = {
    "nvte_layernorm_fwd": {
        "declaration": (
            "void nvte_layernorm_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta, "
            "const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma, NVTETensor workspace, "
            "const int multiprocessorCount, const bool zero_centered_gamma, cudaStream_t stream);"
        ),
        "sha256": "b0912b178faa23ca4450551f837726c95377a23ef5ee3e34b9b0792d5340d68d",
    },
    "nvte_layernorm_bwd": {
        "declaration": (
            "void nvte_layernorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor mu, "
            "const NVTETensor rsigma, const NVTETensor gamma, NVTETensor dx, NVTETensor dgamma, "
            "NVTETensor dbeta, NVTETensor workspace, const int multiprocessorCount, "
            "const bool zero_centered_gamma, cudaStream_t stream);"
        ),
        "sha256": "7d03f4f0450c0245d02f2ee2bfd7169f6a4a4ca059b3ed7b110581dca3d2339c",
    },
    "nvte_enable_cudnn_norm_fwd": {
        "declaration": "void nvte_enable_cudnn_norm_fwd(bool enable);",
        "sha256": "ec72fa17f51a067b9bfd11305ffd86abd982651809d373607b9bbd828ccdc544",
    },
    "nvte_enable_cudnn_norm_bwd": {
        "declaration": "void nvte_enable_cudnn_norm_bwd(bool enable);",
        "sha256": "e9eae4d1eb0be3e100f0708a83a6219a344b10ae30420eff0d07cb718872e1db",
    },
}
EXPECTED_API_INVARIANTS = {
    "x_rank": 2,
    "gamma_beta_same_shape_and_dtype": True,
    "epsilon_nonnegative": True,
    "mu_shape": "rows",
    "mu_dtype": "float32",
    "rsigma_shape": "rows",
    "rsigma_dtype": "float32",
    "dz_dtype_equals_gamma_dtype": True,
    "dx_dtype_equals_x_dtype": True,
    "dgamma_dtype_equals_gamma_dtype": True,
    "dbeta_dtype_equals_gamma_dtype": True,
    "zero_centered_gamma": False,
}
EXPECTED_EXECUTION = {
    "epsilon": 1e-5,
    "tensor_scaling_mode": "NVTE_DELAYED_TENSOR_SCALING",
    "tensor_layout": "row_major_contiguous_c_order",
    "backend_control": {
        "scope": "process_global",
        "isolation": "one_fresh_process_per_plan_record",
        "order": [
            "select_current_cuda_device",
            "read_cudaDeviceProp.multiProcessorCount",
            "nvte_enable_cudnn_norm_fwd(forward_backend_equals_cudnn)",
            "nvte_enable_cudnn_norm_bwd(backward_backend_equals_cudnn_null_maps_false)",
            "query_each_required_workspace_in_boundary_call_order",
            "execute_boundary_calls_in_declared_order",
        ],
    },
    "workspace": {
        "query": "call_same_nvte_entry_point_once_with_empty_workspace_tensor",
        "query_effect": "no_operation_workspace_shape_and_dtype_only",
        "required_dtype": "kNVTEByte",
        "allocation": "allocate_exact_returned_shape_and_dtype",
        "execution": "call_same_nvte_entry_point_again_with_allocated_workspace",
    },
    "multiprocessor_count": {
        "source": "cudaDeviceProp.multiProcessorCount",
        "device": "same_current_cuda_device_as_all_record_tensors_and_stream",
        "capture": "record_integer_before_workspace_query_and_execution",
    },
}
EXPECTED_TENSORS = {
    "x": {"shape": ["rows", "columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "gamma": {"shape": ["columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "beta": {"shape": ["columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "dy": {"shape": ["rows", "columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "y": {"shape": ["rows", "columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "dx": {"shape": ["rows", "columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "dgamma": {"shape": ["columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "dbeta": {"shape": ["columns"], "dtype": "kNVTEBFloat16", "layout": "row_major_contiguous_c_order"},
    "mu": {"shape": ["rows"], "dtype": "kNVTEFloat32", "layout": "row_major_contiguous_c_order"},
    "rsigma": {"shape": ["rows"], "dtype": "kNVTEFloat32", "layout": "row_major_contiguous_c_order"},
    "zero_beta": {
        "shape": ["columns"],
        "dtype": "kNVTEBFloat16",
        "layout": "row_major_contiguous_c_order",
        "value": "all_zero_bits",
    },
    "throwaway_y": {
        "shape": ["rows", "columns"],
        "dtype": "kNVTEBFloat16",
        "layout": "row_major_contiguous_c_order",
    },
    "forward_workspace": {
        "shape": "exact_forward_query_result",
        "dtype": "kNVTEByte",
        "layout": "row_major_contiguous_c_order",
    },
    "backward_workspace": {
        "shape": "exact_backward_query_result",
        "dtype": "kNVTEByte",
        "layout": "row_major_contiguous_c_order",
    },
}
EXPECTED_BACKEND_PAIRS = {
    "forward": [
        {"forward": "transformer_engine", "backward": None},
        {"forward": "cudnn", "backward": None},
    ],
    "backward_recompute": [
        {"forward": "transformer_engine", "backward": "transformer_engine"},
        {"forward": "transformer_engine", "backward": "cudnn"},
        {"forward": "cudnn", "backward": "transformer_engine"},
        {"forward": "cudnn", "backward": "cudnn"},
    ],
    "composed": [
        {"forward": "transformer_engine", "backward": "transformer_engine"},
        {"forward": "transformer_engine", "backward": "cudnn"},
        {"forward": "cudnn", "backward": "transformer_engine"},
        {"forward": "cudnn", "backward": "cudnn"},
    ],
}
EXPECTED_BOUNDARIES = {
    "forward": {
        "public_inputs": ["x", "gamma", "beta"],
        "public_outputs": ["y"],
        "calls": ["nvte_layernorm_fwd(x,gamma,beta,epsilon,y,mu,rsigma,forward_workspace,sm_count,false,stream)"],
        "state": "mu_and_rsigma_are_adapter_private",
    },
    "backward_recompute": {
        "public_inputs": ["x", "gamma", "dy"],
        "public_outputs": ["dx", "dgamma", "dbeta"],
        "calls": [
            "nvte_layernorm_fwd(x,gamma,zero_beta,epsilon,throwaway_y,mu,rsigma,forward_workspace,sm_count,false,stream)",
            "nvte_layernorm_bwd(dy,x,mu,rsigma,gamma,dx,dgamma,dbeta,backward_workspace,sm_count,false,stream)",
        ],
        "state": "zero_beta_is_implicit_publicly_and_mu_rsigma_are_recomputed_inside_the_boundary",
    },
    "composed": {
        "public_inputs": ["x", "gamma", "beta", "dy"],
        "public_outputs": ["y", "dx", "dgamma", "dbeta"],
        "calls": [
            "nvte_layernorm_fwd(x,gamma,beta,epsilon,y,mu,rsigma,forward_workspace,sm_count,false,stream)",
            "nvte_layernorm_bwd(dy,x,mu,rsigma,gamma,dx,dgamma,dbeta,backward_workspace,sm_count,false,stream)",
        ],
        "state": "mu_and_rsigma_are_saved_only_within_the_same_composed_invocation",
    },
}
EXPECTED_OBJECTIVE_POLICY = {
    "rule": (
        "for_every_qualified_matching_backend_pair_and_public_output_each_shuttle_metric_lte_max("
        "matched_te_metric,predeclared_shape_output_dtype_floor)"
    ),
    "aggregation": "metricwise_without_scalar_aggregation",
    "matched_oracle_selection": "all_applicable_predeclared_backend_pairs_no_post_hoc_selector",
    "expert_qualification": (
        "finite_outputs_and_current_abi11_predeclared_per_output_maximum_and_mean_absolute_reference_limits"
    ),
    "fast_policy": "identity_only_and_bitwise_equal_to_source_ordered_for_abi11",
    "timing": "not_authorized_not_part_of_this_contract",
}
EXPECTED_ARTIFACTS = {
    "compact_index_max_bytes": 10 * 1024 * 1024,
    "large_raw_arrays": "content_addressed_external_artifact_link_required_later",
    "wandb_status": "not_uploaded_not_authorized",
}
EXPECTED_CLAIMS = {
    "hardware_executed": False,
    "timing_collected": False,
    "authority_sealed": False,
    "scorecard_status_changed": False,
}
EXPECTED_PLAN_SHA256 = "f42f251961131e5b1c870161f48da14e018001e4aa762752d82d0a7a8c1a6965"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate contract key: {key}")
        value[key] = item
    return value


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(), object_pairs_hook=_unique_object)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    inverse = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + EXPECTED_EXECUTION["epsilon"])
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


@cache
def _expected_input_digests(rows: int, columns: int) -> dict[str, dict[str, str]]:
    return {
        role: {
            "raw_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "array_sha256": _array_digest(value),
        }
        for role, value in zip(INPUT_ROLES, _fixed_inputs(rows, columns), strict=True)
    }


@cache
def _expected_dtype_floors(rows: int, columns: int) -> dict[str, dict[str, float | int]]:
    return {
        role: _dtype_floor(reference)
        for role, reference in zip(OUTPUT_ROLES, _reference_outputs(rows, columns), strict=True)
    }


def _expected_expert_limits() -> dict[str, dict[str, dict[str, float]]]:
    return {
        shape: {
            role: {"max_absolute_error": limits[0], "mean_absolute_error": limits[1]}
            for role, limits in NUMERICAL_LIMITS[shape_id].items()
        }
        for shape_id, shape in (("44d152ecc3e9ff18", "2048x4096"), ("81928ab3539c0f03", "7x13"))
    }


def _expected_plan_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shape in ([2048, 4096], [7, 13]):
        for boundary in ("forward", "backward_recompute", "composed"):
            for backends in EXPECTED_BACKEND_PAIRS[boundary]:
                records.append(
                    {
                        "ordinal": len(records),
                        "shape": shape,
                        "boundary": boundary,
                        "forward_backend": backends["forward"],
                        "backward_backend": backends["backward"],
                    }
                )
    return records


def _validate_contract(contract: dict[str, Any]) -> None:
    assert set(contract) == {
        "schema_version",
        "contract_id",
        "status",
        "source_base",
        "provider",
        "api",
        "execution",
        "tensors",
        "runner",
        "coordinates",
        "inputs",
        "boundaries",
        "metrics",
        "objective",
        "artifacts",
        "claims",
    }
    assert contract["schema_version"] == 1
    assert contract["contract_id"] == "abi11_layernorm_bf16_te_2_17_oracle_red_v1"
    assert contract["status"] == "test_only_red_runner_implementation_absent"
    assert contract["source_base"] == EXPECTED_SOURCE_BASE
    assert contract["provider"] == EXPECTED_PROVIDER
    assert contract["api"] == {
        "header": "transformer_engine/common/include/transformer_engine/normalization.h",
        "signatures": EXPECTED_SIGNATURES,
        "invariants": EXPECTED_API_INVARIANTS,
    }
    for record in EXPECTED_SIGNATURES.values():
        assert hashlib.sha256(record["declaration"].encode()).hexdigest() == record["sha256"]
    assert contract["execution"] == EXPECTED_EXECUTION
    assert contract["tensors"] == EXPECTED_TENSORS

    expected_plan = _expected_plan_records()
    assert len(expected_plan) == 20
    assert _canonical_digest(expected_plan) == EXPECTED_PLAN_SHA256
    assert contract["runner"] == {
        "path": "lib/shuttle/mlir/jax_patch/abi11_layernorm_te_oracle_runner.py",
        "dry_run_argv": ["--contract", CONTRACT.name, "--print-plan"],
        "dry_run_requires_cuda": False,
        "expected_plan_records": len(expected_plan),
        "canonical_plan_sha256": EXPECTED_PLAN_SHA256,
        "canonical_plan": expected_plan,
    }
    assert contract["coordinates"] == {
        "shapes": [[2048, 4096], [7, 13]],
        "boundaries": ["forward", "backward_recompute", "composed"],
        "backend_pairs": EXPECTED_BACKEND_PAIRS,
    }

    inputs = contract["inputs"]
    assert set(inputs) == {"generator", "ranges", "digests"}
    assert inputs["generator"] == "numpy_2.3.5_linspace_float32_then_ml_dtypes_0.5.4_bfloat16"
    assert inputs["ranges"] == {
        "x": [-0.75, 0.875],
        "gamma": [-0.625, 1.0],
        "beta": [-0.25, 0.375],
        "dy": [-0.5, 0.75],
    }
    assert set(inputs["digests"]) == {"2048x4096", "7x13"}
    for rows, columns in ((2048, 4096), (7, 13)):
        records = inputs["digests"][f"{rows}x{columns}"]
        assert tuple(records) == INPUT_ROLES
        assert records == _expected_input_digests(rows, columns)

    assert contract["boundaries"] == EXPECTED_BOUNDARIES
    metrics = contract["metrics"]
    assert set(metrics) == {
        "reference",
        "names",
        "relative_scale_floor",
        "public_output_roles",
        "private_state_roles",
        "dtype_floors",
    }
    assert metrics["reference"] == "independent_numpy_float64_layernorm_closed_form_then_bfloat16_public_outputs"
    assert metrics["names"] == list(METRICS)
    assert metrics["relative_scale_floor"] == RELATIVE_SCALE_FLOOR
    assert metrics["public_output_roles"] == {
        "forward": ["y"],
        "backward_recompute": ["dx", "dgamma", "dbeta"],
        "composed": ["y", "dx", "dgamma", "dbeta"],
    }
    assert metrics["private_state_roles"] == [
        "mu",
        "rsigma",
        "throwaway_y",
        "forward_workspace",
        "backward_workspace",
    ]
    assert set(metrics["dtype_floors"]) == {"2048x4096", "7x13"}
    for rows, columns in ((2048, 4096), (7, 13)):
        assert metrics["dtype_floors"][f"{rows}x{columns}"] == _expected_dtype_floors(rows, columns)

    objective = dict(EXPECTED_OBJECTIVE_POLICY)
    objective["expert_reference_limits"] = _expected_expert_limits()
    assert contract["objective"] == objective
    assert contract["artifacts"] == EXPECTED_ARTIFACTS
    assert contract["claims"] == EXPECTED_CLAIMS


def _require_objective(
    candidate: dict[str, float | int], expert: dict[str, float | int], floor: dict[str, float | int]
) -> None:
    for metric in METRICS:
        if candidate[metric] > max(expert[metric], floor[metric]):
            raise AssertionError(f"{metric} exceeds the matched expert-or-dtype-floor limit")


def _replace_nested(contract: dict[str, Any], path: tuple[str | int, ...], replacement: Any) -> None:
    parent: Any = contract
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement


def test_contract_is_a_closed_te_layernorm_oracle_specification() -> None:
    _validate_contract(_load_contract())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("source_base", "marin_commit"), "0" * 40),
        (("provider", "source_files", "transformer_engine/common/normalization/layernorm/ln_api.cpp"), "0" * 64),
        (("execution", "epsilon"), 1e-4),
        (("execution", "tensor_scaling_mode"), "NVTE_MXFP8_1D_SCALING"),
        (("execution", "tensor_layout"), "column_major"),
        (("execution", "workspace", "query"), "execute_with_empty_workspace"),
        (("execution", "backend_control", "scope"), "thread_local"),
        (("execution", "multiprocessor_count", "source"), "hard_coded"),
        (("tensors", "x", "layout"), "column_major"),
        (("objective", "rule"), "accept_everything"),
        (("objective", "aggregation"), "scalar_average"),
        (("objective", "matched_oracle_selection"), "best_after_observation"),
        (("objective", "fast_policy"), "fastest_backend"),
        (("objective", "timing"), "authorized"),
        (("artifacts", "compact_index_max_bytes"), 20 * 1024 * 1024),
        (("artifacts", "large_raw_arrays"), "inline"),
        (("artifacts", "wandb_status"), "uploaded"),
        (("claims", "hardware_executed"), True),
        (("claims", "timing_collected"), True),
        (("claims", "authority_sealed"), True),
        (("claims", "scorecard_status_changed"), True),
    ],
)
def test_contract_rejects_policy_and_authority_leaf_mutations(path: tuple[str | int, ...], replacement: Any) -> None:
    mutated = copy.deepcopy(_load_contract())
    _replace_nested(mutated, path, replacement)
    with pytest.raises(AssertionError):
        _validate_contract(mutated)


def test_contract_rejects_coordinated_api_declaration_and_self_hash_mutation() -> None:
    mutated = copy.deepcopy(_load_contract())
    declaration = "void nvte_layernorm_fwd(const NVTETensor beta);"
    mutated["api"]["signatures"]["nvte_layernorm_fwd"] = {
        "declaration": declaration,
        "sha256": hashlib.sha256(declaration.encode()).hexdigest(),
    }
    mutated["provider"]["source_files"]["transformer_engine/common/normalization/layernorm/ln_api.cpp"] = "0" * 64
    with pytest.raises(AssertionError):
        _validate_contract(mutated)


def test_contract_rejects_coordinated_coordinate_relabel_and_rehashed_plan() -> None:
    mutated = copy.deepcopy(_load_contract())
    mutated["coordinates"]["boundaries"][1] = "backward_saved_state"
    for record in mutated["runner"]["canonical_plan"]:
        if record["boundary"] == "backward_recompute":
            record["boundary"] = "backward_saved_state"
    mutated["runner"]["canonical_plan_sha256"] = _canonical_digest(mutated["runner"]["canonical_plan"])
    with pytest.raises(AssertionError):
        _validate_contract(mutated)


def test_contract_rejects_coordinated_plan_reorder_and_rehash() -> None:
    mutated = copy.deepcopy(_load_contract())
    plan = mutated["runner"]["canonical_plan"]
    plan[0], plan[1] = plan[1], plan[0]
    for ordinal, record in enumerate(plan):
        record["ordinal"] = ordinal
    mutated["runner"]["canonical_plan_sha256"] = _canonical_digest(plan)
    with pytest.raises(AssertionError):
        _validate_contract(mutated)


def test_predeclared_metricwise_objective_checks_every_metric() -> None:
    contract = _load_contract()
    _validate_contract(contract)
    floor = contract["metrics"]["dtype_floors"]["7x13"]["y"]
    expert = {metric: 0 for metric in METRICS}
    _require_objective(dict(floor), expert, floor)
    for metric in METRICS:
        rejected = dict(floor)
        rejected[metric] = max(expert[metric], floor[metric]) + 1
        with pytest.raises(AssertionError, match=metric):
            _require_objective(rejected, expert, floor)


def test_layernorm_te_runner_emits_the_closed_cpu_only_dry_run_plan() -> None:
    contract = _load_contract()
    _validate_contract(contract)
    runner = CONTRACT.parent / Path(contract["runner"]["path"]).name
    result = subprocess.run(
        [sys.executable, str(runner), *contract["runner"]["dry_run_argv"]],
        cwd=CONTRACT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout, object_pairs_hook=_unique_object)
    assert plan == {
        "schema_version": 1,
        "contract_id": "abi11_layernorm_bf16_te_2_17_oracle_red_v1",
        "hardware_required": False,
        "records": _expected_plan_records(),
    }


def test_layernorm_te_runner_rejects_contract_byte_drift(tmp_path: Path) -> None:
    mutated = tmp_path / CONTRACT.name
    mutated.write_bytes(CONTRACT.read_bytes().replace(b'"timing_collected": false', b'"timing_collected": true'))
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--contract", str(mutated), "--print-plan"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "contract byte identity drifted" in result.stderr


def test_layernorm_te_runner_rejects_unpinned_source_before_hardware_boundary(tmp_path: Path) -> None:
    output = tmp_path / "execution"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--contract",
            str(CONTRACT),
            "--execute-record",
            "0",
            "--source-root",
            str(tmp_path / "missing-te"),
            "--adapter-binary",
            str(tmp_path / "missing-adapter"),
            "--adapter-binary-sha256",
            "0" * 64,
            "--transformer-engine-library",
            str(tmp_path / "missing-library"),
            "--transformer-engine-library-sha256",
            "0" * 64,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Transformer Engine source root is not a readable Git checkout" in result.stderr
    assert not output.exists()


def test_layernorm_te_runner_rejects_nonpinned_git_head(tmp_path: Path) -> None:
    source = tmp_path / "te"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    marker = source / "marker"
    marker.write_text("local source\n")
    subprocess.run(["git", "-C", str(source), "add", "marker"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "local source"], check=True)

    with pytest.raises(ValueError, match="checkout HEAD drifted"):
        verify_source_checkout(source, load_runner_contract(CONTRACT))


def test_layernorm_te_runner_rejects_caller_supplied_binary_digest_drift(tmp_path: Path) -> None:
    binary = tmp_path / "adapter"
    binary.write_bytes(b"adapter")
    binary.chmod(0o755)

    with pytest.raises(ValueError, match="adapter binary identity drifted"):
        verify_file_identity(binary, "0" * 64, "adapter binary", executable=True)


@pytest.mark.parametrize(
    ("ordinal", "roles"),
    [
        (10, {"x", "gamma", "beta"}),
        (12, {"x", "gamma", "dy"}),
        (16, {"x", "gamma", "beta", "dy"}),
    ],
)
def test_layernorm_te_runner_materializes_exact_public_input_roles(
    tmp_path: Path, ordinal: int, roles: set[str]
) -> None:
    contract = load_runner_contract(CONTRACT)
    record = contract["runner"]["canonical_plan"][ordinal]
    directory = tmp_path / record["boundary"]
    artifacts = _write_inputs(directory, record, contract)

    assert set(artifacts) == roles
    assert {path.stem for path in directory.iterdir()} == roles
    expected = contract["inputs"]["digests"]["7x13"]
    for role, artifact in artifacts.items():
        assert artifact["sha256"] == expected[role]["raw_sha256"]
        assert artifact["bytes"] == (7 * 13 * 2 if role in {"x", "dy"} else 13 * 2)


@pytest.mark.parametrize(
    ("branch", "status", "submodules", "message"),
    [
        ("main", "", [f"-{revision} {path}" for path, revision in SUBMODULES.items()], "must be detached"),
        ("", " M local", [f"-{revision} {path}" for path, revision in SUBMODULES.items()], "must be clean"),
        ("", "", [], "submodule identities drifted"),
        (
            "",
            "",
            [f" {revision} {path}" for path, revision in SUBMODULES.items()],
            "must remain uninitialized",
        ),
    ],
)
def test_layernorm_te_runner_rejects_checkout_state_mutations(
    branch: str, status: str, submodules: list[str], message: str
) -> None:
    contract = load_runner_contract(CONTRACT)
    with pytest.raises(ValueError, match=message):
        validate_checkout_identity(contract["provider"]["commit"], branch, status, submodules, contract)


def _valid_adapter_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "unsealed_local_hardware_observation",
        "coordinate": record,
        "epsilon": 1e-5,
        "scaling_mode": "NVTE_DELAYED_TENSOR_SCALING",
        "layout": "row_major_contiguous_c_order",
        "physical_sm_count": 1,
        "workspace_queries": {"forward": {"shape": [1], "dtype": "kNVTEByte"}, "backward": None},
        "timing_collected": False,
        "authority_sealed": False,
    }


def test_layernorm_te_runner_rejects_extra_public_output_role(tmp_path: Path) -> None:
    contract = load_runner_contract(CONTRACT)
    record = contract["runner"]["canonical_plan"][10]
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "y.bf16").write_bytes(bytes(7 * 13 * 2))
    (outputs / "dx.bf16").write_bytes(bytes(7 * 13 * 2))
    (outputs / "metadata.json").write_text("{}\n")

    with pytest.raises(ValueError, match="output directory entry set drifted"):
        _validate_adapter_metadata(_valid_adapter_metadata(record), record, 10, outputs, contract)


def test_layernorm_te_runner_rejects_symlinked_public_output(tmp_path: Path) -> None:
    contract = load_runner_contract(CONTRACT)
    record = contract["runner"]["canonical_plan"][10]
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    backing = tmp_path / "y.bf16"
    backing.write_bytes(bytes(7 * 13 * 2))
    (outputs / "y.bf16").symlink_to(backing)
    (outputs / "metadata.json").write_text("{}\n")

    with pytest.raises(ValueError, match="regular files"):
        _validate_adapter_metadata(_valid_adapter_metadata(record), record, 10, outputs, contract)


def test_layernorm_te_runner_rejects_wrong_public_output_size(tmp_path: Path) -> None:
    contract = load_runner_contract(CONTRACT)
    record = contract["runner"]["canonical_plan"][10]
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "y.bf16").write_bytes(b"too short")
    (outputs / "metadata.json").write_text("{}\n")

    with pytest.raises(ValueError, match="public output drifted: y"):
        _validate_adapter_metadata(_valid_adapter_metadata(record), record, 10, outputs, contract)


def _fake_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "fake_adapter.py"
    adapter.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ordinal", type=int, required=True)
parser.add_argument("--boundary", required=True)
parser.add_argument("--forward-backend", required=True)
parser.add_argument("--backward-backend", required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--columns", type=int, required=True)
parser.add_argument("--device", type=int, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--outputs", type=Path, required=True)
parser.add_argument("--te-library", type=Path, required=True)
args = parser.parse_args()
roles = {
    "forward": ({"x", "gamma", "beta"}, {"y"}),
    "backward_recompute": ({"x", "gamma", "dy"}, {"dx", "dgamma", "dbeta"}),
    "composed": ({"x", "gamma", "beta", "dy"}, {"y", "dx", "dgamma", "dbeta"}),
}
input_roles, output_roles = roles[args.boundary]
if {path.stem for path in args.inputs.iterdir()} != input_roles:
    raise SystemExit(3)
sizes = {
    "y": args.rows * args.columns * 2,
    "dx": args.rows * args.columns * 2,
    "dgamma": args.columns * 2,
    "dbeta": args.columns * 2,
}
for role in output_roles:
    (args.outputs / f"{role}.bf16").write_bytes(bytes([args.ordinal]) * sizes[role])
backward = None if args.backward_backend == "none" else args.backward_backend
metadata = {
    "schema_version": 1,
    "status": "unsealed_local_hardware_observation",
    "authority_sealed": False,
    "coordinate": {
        "ordinal": args.ordinal,
        "shape": [args.rows, args.columns],
        "boundary": args.boundary,
        "forward_backend": args.forward_backend,
        "backward_backend": backward,
    },
    "epsilon": 1e-5,
    "scaling_mode": "NVTE_DELAYED_TENSOR_SCALING",
    "layout": "row_major_contiguous_c_order",
    "physical_sm_count": 1,
    "workspace_queries": {
        "forward": {"shape": [1], "dtype": "kNVTEByte"},
        "backward": None if args.boundary == "forward" else {"shape": [2], "dtype": "kNVTEByte"},
    },
    "timing_collected": False,
}
(args.outputs / "metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\\n")
(args.outputs.parent / "observed-argv.json").write_text(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    adapter.chmod(0o755)
    return adapter


@pytest.mark.parametrize(
    ("ordinal", "input_roles", "output_roles"),
    [
        (10, {"x", "gamma", "beta"}, {"y"}),
        (12, {"x", "gamma", "dy"}, {"dx", "dgamma", "dbeta"}),
    ],
)
def test_layernorm_te_runner_executes_one_fake_record_and_binds_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ordinal: int,
    input_roles: set[str],
    output_roles: set[str],
) -> None:
    contract = load_runner_contract(CONTRACT)
    adapter = _fake_adapter(tmp_path)
    adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
    library = tmp_path / "libtransformer_engine.so"
    library.write_bytes(b"fake Transformer Engine library")
    library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    output = tmp_path / "execution"
    monkeypatch.setattr("abi11_layernorm_te_oracle_runner.verify_source_checkout", lambda *_args: None)

    result = execute_record(
        contract,
        ordinal,
        ExecutionConfig(
            source_root=tmp_path / "source",
            adapter_binary=adapter,
            adapter_binary_sha256=adapter_sha256,
            transformer_engine_library=library,
            transformer_engine_library_sha256=library_sha256,
            output=output,
            device=0,
        ),
    )

    assert set(result["input_artifacts"]) == input_roles
    assert set(result["output_artifacts"]) == output_roles
    assert result["contract_identity"] == {
        "contract_id": contract["contract_id"],
        "sha256": "5a3391f00bf5453d735cfe4359d8e8dff73fb70ccdf7908a0d19b592c413d005",
    }
    assert result["plan_identity"] == {
        "sha256": "f42f251961131e5b1c870161f48da14e018001e4aa762752d82d0a7a8c1a6965",
        "ordinal": ordinal,
    }
    assert result["build_identity"]["authority"] == "caller_supplied_digests_not_source_derived_build_proof"
    assert result["build_identity"]["adapter_binary_sha256"] == adapter_sha256
    assert result["build_identity"]["transformer_engine_library_sha256"] == library_sha256
    assert json.loads((output / "result.json").read_text()) == result
    observed_argv = json.loads((output / "observed-argv.json").read_text())
    assert observed_argv[-2:] == ["--te-library", str(library)]
    assert {path.stem for path in (output / "inputs").iterdir()} == input_roles
    assert result["execution_metadata_artifact"] == {
        "bytes": (output / "outputs/metadata.json").stat().st_size,
        "sha256": hashlib.sha256((output / "outputs/metadata.json").read_bytes()).hexdigest(),
    }


def test_layernorm_te_runner_pins_adapter_source_bytes() -> None:
    assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest() == ADAPTER_SHA256
