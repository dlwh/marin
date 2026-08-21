# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from abi22_shuttle_rmsnorm_representative import (
    ACCEPTANCE_PATTERN,
    BOUNDARIES,
    POLICIES,
    REPRESENTATIVE_SHAPE,
    RESOURCE_CONTRACT,
    RESULT_ROLES,
    ROLE_LIMITS,
    execution_plan,
    validate_output_artifact,
    validate_shuttle_evidence,
)

MLIR_ROOT = Path(__file__).resolve().parents[1]
R12_ROOT = MLIR_ROOT / "artifacts/track2-te-rmsnorm-r12-success-20260821"
R12_RESULT = R12_ROOT / "result-v4.json"
R12_RAW = R12_ROOT / "policy-raw-v4.json"
SHA256 = {
    "r12_result": "bce080011168c768c70a6aca8dcbae82f8d021c1f4fae7610e57476c2b4b2439",
    "r12_raw": "92fbb8ad362b8d28c005c4e2edd34e8690c79016f8a24c5a93d388b13563bcf3",
    "x": "1528589b5505fba379086deffe6c699c36114d3dc72600bbfe464e17947d71b2",
    "gamma": "7d44ce0c29c1cdd0b59ee5be57f1cc3656563d68095c9d73769bea144cc7bd09",
    "dy": "d1565501f4b951240b46d328a0459b81c1426a44f39cd0d6066cedb3eac0005d",
    "y": "437e9adc3e7cfd518132cfac8455d253a34bbfcc081109632611ff7ba4e62a70",
    "dx": "558ae600e406ade38183c4c9726bbfed16b60c07bb0888b4021b38c87408d3f8",
    "dgamma": "28d69308ae4d26785592deb62547d5db17ba50c20c8a21b6e4108349929f55f6",
}
EXPECTED_LIMITS = {
    "y": {
        "absolute_floor": 0.015625,
        "below_floor_count": 84_000,
        "eligible_count": 8_304_608,
        "max_absolute_error_below_floor": 0.0078125,
        "max_absolute_error_above_floor": 0.03125,
        "max_mean_absolute_error": 2**-20,
        "max_relative_error_above_floor": 0.5,
        "max_bfloat16_ulp_above_floor": 2,
    },
    "dx": {
        "absolute_floor": 0.125,
        "below_floor_count": 672_162,
        "eligible_count": 7_716_446,
        "max_absolute_error_below_floor": 0.0625,
        "max_absolute_error_above_floor": 1.0,
        "max_mean_absolute_error": 2**-19,
        "max_relative_error_above_floor": 0.5,
        "max_bfloat16_ulp_above_floor": 64,
    },
    "dgamma": {
        "absolute_floor": 0.03125,
        "below_floor_count": 4,
        "eligible_count": 4_092,
        "max_absolute_error_below_floor": 0.015625,
        "max_absolute_error_above_floor": 4.0,
        "max_mean_absolute_error": 2**-10,
        "max_relative_error_above_floor": 0.5,
        "max_bfloat16_ulp_above_floor": 8,
    },
}
EXPECTED_RESOURCES = {
    "forward": {
        "launch_count": 1,
        "slot_count": 3,
        "external_operand_count": 2,
        "external_result_count": 1,
        "temporary_allocation_count": 0,
        "temporary_bytes": 0,
    },
    "backward_recompute": {
        "launch_count": 3,
        "slot_count": 6,
        "external_operand_count": 3,
        "external_result_count": 2,
        "temporary_allocation_count": 1,
        "temporary_bytes": 8_192,
    },
    "composed": {
        "launch_count": 4,
        "slot_count": 7,
        "external_operand_count": 3,
        "external_result_count": 3,
        "temporary_allocation_count": 1,
        "temporary_bytes": 8_192,
    },
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(role: str) -> dict[str, object]:
    limits = EXPECTED_LIMITS[role]
    return {
        "all_finite": True,
        "absolute_floor": limits["absolute_floor"],
        "below_floor_count": limits["below_floor_count"],
        "eligible_count": limits["eligible_count"],
        "max_absolute_error": 0.0,
        "max_absolute_error_below_floor": 0.0,
        "max_absolute_error_above_floor": 0.0,
        "mean_absolute_error": 0.0,
        "max_relative_error_above_absolute_floor": 0.0,
        "max_bfloat16_ulp_above_absolute_floor": 0,
        "raw_max_bfloat16_ulp": 0,
    }


def _cell(policy: str, boundary: str, repeat: int) -> dict[str, object]:
    roles = RESULT_ROLES[boundary]
    outputs = []
    for index, role in enumerate(roles):
        digest = _sha(f"shuttle/{policy}/{boundary}/{role}")
        outputs.append(
            {
                "role": role,
                "public_result_index": index,
                "digest": digest,
                "reference_digest": SHA256[role],
                "artifact": {
                    "path": f"raw/{policy}/{boundary}/repeat-{repeat}/{role}.bf16",
                    "byte_count": (2048 * 4096 if role != "dgamma" else 4096) * 2,
                    "sha256": digest,
                },
                **_metrics(role),
            }
        )
    inputs = {"x": SHA256["x"], "gamma": SHA256["gamma"]}
    if boundary != "forward":
        inputs["dy"] = SHA256["dy"]
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_cell_v1",
        "producer": "shuttle_gpu_executable_bundle",
        "input_pattern": ACCEPTANCE_PATTERN,
        "shape": [2048, 4096],
        "epsilon": 1e-6,
        "policy": policy,
        "boundary": boundary,
        "repeat": repeat,
        "inputs": inputs,
        "call_sequence": {
            "forward": ["shuttle_rmsnorm_forward"],
            "backward_recompute": ["shuttle_rmsnorm_backward_recompute"],
            "composed": ["shuttle_rmsnorm_composed"],
        }[boundary],
        "outputs": outputs,
        "resources": copy.deepcopy(EXPECTED_RESOURCES[boundary]),
        "transport": {
            "contract_operation_count": 0,
            "code_bytes": 4096,
            "executable_digest": _sha(f"executable/{policy}/{boundary}"),
            "module_load_digest": _sha(f"module/{policy}/{boundary}"),
            "ptx_digests": [
                _sha(f"ptx/{policy}/{boundary}/{launch}")
                for launch in range(EXPECTED_RESOURCES[boundary]["launch_count"])
            ],
        },
        "timing": {
            "warmup_iterations": 10,
            "measured_iterations": 20,
            "host_clock": "steady_clock_execute_block_host_until_done_ns",
            "host_samples_ns": list(range(1000, 1020)),
            "cuda_event_status": "supplemental_not_acceptance",
            "cuda_event_samples_ns": list(range(500, 520)),
        },
    }


def _evidence() -> dict[str, object]:
    cells = [
        _cell(policy, boundary, repeat)
        for policy in ("source_ordered", "fast")
        for boundary in ("forward", "backward_recompute", "composed")
        for repeat in range(2)
    ]
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_evidence_v1",
        "status": "heldout_splitmix64_v3_shuttle_numerics_accepted",
        "producer": "shuttle_gpu_executable_bundle",
        "shape": [2048, 4096],
        "epsilon": 1e-6,
        "input_pattern": ACCEPTANCE_PATTERN,
        "compiler": {
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "compiler_identity": _sha("compiler"),
        },
        "external_evidence": {
            "transformer_engine_xla_status": "heldout_splitmix64_v3_numerics_accepted",
            "r12_result_sha256": SHA256["r12_result"],
            "r12_policy_raw_sha256": SHA256["r12_raw"],
            "comparison": "same_h100_host_blocking_reported_without_threshold",
        },
        "protocol": {
            "fresh_process_per_coordinate": True,
            "warmup_iterations": 10,
            "measured_iterations": 20,
            "acceptance_clock": "host_blocking",
            "cuda_event_status": "supplemental_not_acceptance",
        },
        "cells": cells,
        "scope": {
            "norm_contract": "excluded",
            "performance": "reported_without_threshold",
            "raw_outputs_persisted_before_validation": True,
        },
    }


def test_contract_freezes_r12_inputs_role_limits_resources_and_coordinates() -> None:
    assert ACCEPTANCE_PATTERN == "heldout_splitmix64_v3"
    assert REPRESENTATIVE_SHAPE == (2048, 4096)
    assert BOUNDARIES == ("forward", "backward_recompute", "composed")
    assert POLICIES == ("source_ordered", "fast")
    assert RESULT_ROLES == {
        "forward": ("y",),
        "backward_recompute": ("dx", "dgamma"),
        "composed": ("y", "dx", "dgamma"),
    }
    assert ROLE_LIMITS == EXPECTED_LIMITS
    assert RESOURCE_CONTRACT == EXPECTED_RESOURCES
    assert hashlib.sha256(R12_RESULT.read_bytes()).hexdigest() == SHA256["r12_result"]
    assert hashlib.sha256(R12_RAW.read_bytes()).hexdigest() == SHA256["r12_raw"]

    evidence = _evidence()
    assert validate_shuttle_evidence(evidence, r12_result=R12_RESULT, r12_policy_raw=R12_RAW) == evidence
    assert len(evidence["cells"]) == 12


def test_execution_plan_is_exact_fresh_process_matrix(tmp_path: Path) -> None:
    runner = tmp_path / "run-h100-shuttle-rmsnorm-representative.sh"
    case_root = tmp_path / "cases"
    raw_root = tmp_path / "raw"
    plan = execution_plan(runner=runner, case_root=case_root, raw_root=raw_root)
    assert plan["schema"] == "shuttle_abi22_rmsnorm_representative_plan_v1"
    assert plan["fresh_process_per_coordinate"] is True
    assert len(plan["runs"]) == 12
    expected = [
        (policy, boundary, repeat)
        for policy in ("source_ordered", "fast")
        for boundary in ("forward", "backward_recompute", "composed")
        for repeat in range(2)
    ]
    assert [(run["policy"], run["boundary"], run["repeat"]) for run in plan["runs"]] == expected
    for run, (policy, boundary, repeat) in zip(plan["runs"], expected, strict=True):
        assert run["argv"] == [
            str(runner),
            "--case-root",
            str(case_root),
            "--raw-root",
            str(raw_root),
            "--policy",
            policy,
            "--boundary",
            boundary,
            "--repeat",
            str(repeat),
        ]


@pytest.mark.parametrize("provider", ["source_ordered", "fast"])
@pytest.mark.parametrize("boundary", ["forward", "backward_recompute", "composed"])
@pytest.mark.parametrize("repeat", [0, 1])
def test_every_acceptance_cell_enforces_metrics_roles_and_resources(provider: str, boundary: str, repeat: int) -> None:
    baseline = _evidence()
    index = next(
        index
        for index, cell in enumerate(baseline["cells"])
        if (cell["policy"], cell["boundary"], cell["repeat"]) == (provider, boundary, repeat)
    )
    for role_index, role in enumerate(RESULT_ROLES[boundary]):
        limits = EXPECTED_LIMITS[role]
        mutations = {
            "all_finite": False,
            "below_floor_count": limits["below_floor_count"] + 1,
            "eligible_count": limits["eligible_count"] - 1,
            "max_absolute_error_below_floor": limits["max_absolute_error_below_floor"] * 2 + 1e-9,
            "max_absolute_error_above_floor": limits["max_absolute_error_above_floor"] * 2 + 1e-9,
            "mean_absolute_error": limits["max_mean_absolute_error"] * 2,
            "max_relative_error_above_absolute_floor": limits["max_relative_error_above_floor"] + 0.01,
            "max_bfloat16_ulp_above_absolute_floor": limits["max_bfloat16_ulp_above_floor"] + 1,
        }
        for field, value in mutations.items():
            changed = copy.deepcopy(baseline)
            changed["cells"][index]["outputs"][role_index][field] = value
            with pytest.raises(ValueError):
                validate_shuttle_evidence(changed, r12_result=R12_RESULT, r12_policy_raw=R12_RAW)
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["outputs"][role_index]["max_absolute_error"] = math.nan
        with pytest.raises(ValueError):
            validate_shuttle_evidence(changed, r12_result=R12_RESULT, r12_policy_raw=R12_RAW)

    for field in EXPECTED_RESOURCES[boundary]:
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["resources"][field] += 1
        with pytest.raises(ValueError):
            validate_shuttle_evidence(changed, r12_result=R12_RESULT, r12_policy_raw=R12_RAW)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_cell",
        "duplicate_cell",
        "wrong_shape",
        "wrong_epsilon",
        "wrong_pattern",
        "wrong_input",
        "wrong_reference",
        "wrong_role_order",
        "wrong_public_index",
        "repeat_drift",
        "contract_fusion",
        "wrong_warmup",
        "short_host_samples",
        "short_event_samples",
        "wrong_external_result",
        "wrong_external_raw",
    ],
)
def test_structural_transport_repeat_and_external_evidence_mutations_reject(mutation: str) -> None:
    changed = copy.deepcopy(_evidence())
    cell = changed["cells"][0]
    if mutation == "missing_cell":
        changed["cells"].pop()
    elif mutation == "duplicate_cell":
        changed["cells"][-1] = copy.deepcopy(changed["cells"][0])
    elif mutation == "wrong_shape":
        cell["shape"] = [2048, 4097]
    elif mutation == "wrong_epsilon":
        cell["epsilon"] = 1e-5
    elif mutation == "wrong_pattern":
        cell["input_pattern"] = "qualification_linspace_v2"
    elif mutation == "wrong_input":
        cell["inputs"]["x"] = "0" * 64
    elif mutation == "wrong_reference":
        cell["outputs"][0]["reference_digest"] = "0" * 64
    elif mutation == "wrong_role_order":
        backward = next(entry for entry in changed["cells"] if entry["boundary"] == "backward_recompute")
        backward["outputs"].reverse()
    elif mutation == "wrong_public_index":
        cell["outputs"][0]["public_result_index"] = 1
    elif mutation == "repeat_drift":
        repeated = next(
            entry
            for entry in changed["cells"]
            if entry["policy"] == cell["policy"] and entry["boundary"] == cell["boundary"] and entry["repeat"] == 1
        )
        repeated["outputs"][0]["digest"] = "0" * 64
    elif mutation == "contract_fusion":
        cell["transport"]["contract_operation_count"] = 1
    elif mutation == "wrong_warmup":
        cell["timing"]["warmup_iterations"] = 9
    elif mutation == "short_host_samples":
        cell["timing"]["host_samples_ns"].pop()
    elif mutation == "short_event_samples":
        cell["timing"]["cuda_event_samples_ns"].pop()
    elif mutation == "wrong_external_result":
        changed["external_evidence"]["r12_result_sha256"] = "0" * 64
    elif mutation == "wrong_external_raw":
        changed["external_evidence"]["r12_policy_raw_sha256"] = "0" * 64
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError):
        validate_shuttle_evidence(changed, r12_result=R12_RESULT, r12_policy_raw=R12_RAW)


def test_output_artifact_validation_rejects_case_alias_and_wrong_storage(tmp_path: Path) -> None:
    case_root = tmp_path / "cases"
    raw_root = tmp_path / "raw"
    case_root.mkdir()
    raw_root.mkdir()
    reference = case_root / "reference.bf16"
    reference.write_bytes(b"\x01\x02\x03\x04")
    output = raw_root / "output.bf16"
    output.write_bytes(b"\x04\x03\x02\x01")
    expected = {
        "path": str(output),
        "byte_count": 4,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    assert validate_output_artifact(expected, raw_root=raw_root, case_root=case_root) == expected

    output.unlink()
    output.hardlink_to(reference)
    aliased = copy.deepcopy(expected)
    aliased["sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_output_artifact(aliased, raw_root=raw_root, case_root=case_root)

    outside = copy.deepcopy(expected)
    outside["path"] = str(tmp_path / "outside.bf16")
    Path(outside["path"]).write_bytes(b"\x04\x03\x02\x01")
    with pytest.raises(ValueError):
        validate_output_artifact(outside, raw_root=raw_root, case_root=case_root)


def test_r12_external_evidence_is_revalidated_not_self_reported(tmp_path: Path) -> None:
    changed_result = json.loads(R12_RESULT.read_text())
    changed_result["evidence"]["acceptance"]["accepted_shape"] = [2048, 4097]
    changed_path = tmp_path / "result-v4.json"
    changed_path.write_text(json.dumps(changed_result, sort_keys=True, separators=(",", ":")))
    document = _evidence()
    document["external_evidence"]["r12_result_sha256"] = hashlib.sha256(changed_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_shuttle_evidence(document, r12_result=changed_path, r12_policy_raw=R12_RAW)
