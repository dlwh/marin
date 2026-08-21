# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from abi22_shuttle_rmsnorm_representative import (
    ACCEPTANCE_PATTERN,
    BINDING_CONTRACT,
    BOUNDARIES,
    POLICIES,
    REPRESENTATIVE_SHAPE,
    RESOURCE_CONTRACT,
    RESULT_ROLES,
    ROLE_LIMITS,
    execute_representative,
    execution_plan,
    prepare_representative_cases,
    validate_output_artifact,
    validate_shuttle_evidence,
)
from abi22_te_rmsnorm_oracle_v2 import error_metrics_v4, repeat_output_diagnostics_v4

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
FULL_BYTES = 2048 * 4096 * 2
FEATURE_BYTES = 4096 * 2
EXPECTED_BINDINGS = {
    "forward": {
        "operands": (
            {
                "role": "x",
                "runner_index": 0,
                "slot_ordinal": 0,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "gamma",
                "runner_index": 1,
                "slot_ordinal": 1,
                "required_bytes": FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
        ),
        "results": (
            {
                "role": "y",
                "public_result_index": 0,
                "runner_index": 0,
                "slot_ordinal": 2,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
        ),
    },
    "backward_recompute": {
        "operands": (
            {
                "role": "x",
                "runner_index": 0,
                "slot_ordinal": 0,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "gamma",
                "runner_index": 1,
                "slot_ordinal": 1,
                "required_bytes": FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
            {
                "role": "dy",
                "runner_index": 2,
                "slot_ordinal": 2,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
        ),
        "results": (
            {
                "role": "dx",
                "public_result_index": 0,
                "runner_index": 0,
                "slot_ordinal": 5,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dgamma",
                "public_result_index": 1,
                "runner_index": 1,
                "slot_ordinal": 4,
                "required_bytes": FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
        ),
    },
    "composed": {
        "operands": (
            {
                "role": "x",
                "runner_index": 0,
                "slot_ordinal": 0,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "gamma",
                "runner_index": 1,
                "slot_ordinal": 1,
                "required_bytes": FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
            {
                "role": "dy",
                "runner_index": 2,
                "slot_ordinal": 2,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
        ),
        "results": (
            {
                "role": "y",
                "public_result_index": 0,
                "runner_index": 0,
                "slot_ordinal": 4,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dx",
                "public_result_index": 1,
                "runner_index": 1,
                "slot_ordinal": 6,
                "required_bytes": FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dgamma",
                "public_result_index": 2,
                "runner_index": 2,
                "slot_ordinal": 5,
                "required_bytes": FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
        ),
    },
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _arguments(argv: list[str]) -> dict[str, str]:
    return {value[2:]: argv[index + 1] for index, value in enumerate(argv[:-1]) if value.startswith("--")}


def _fake_runner(
    calls: list[list[str]],
    argv: list[str],
    *,
    failure_coordinate: tuple[str, str, int] | None = None,
    omit_cell: bool = False,
    corrupt_output: bool = False,
    corrupt_artifact_size: bool = False,
    corrupt_artifact_sha: bool = False,
    raw_mutation: str | None = None,
) -> int:
    calls.append(argv)
    arguments = _arguments(argv)
    coordinate = (arguments["policy"], arguments["boundary"], int(arguments["repeat"]))
    if coordinate == failure_coordinate:
        return 17
    case_root = Path(arguments["case-root"])
    raw_root = Path(arguments["raw-root"])
    policy, boundary, repeat = coordinate
    directory = raw_root / policy / boundary / f"repeat-{repeat}"
    directory.mkdir(parents=True, exist_ok=True)
    if omit_cell and coordinate == ("source_ordered", "forward", 0):
        return 0
    external_bindings = []
    for binding in EXPECTED_BINDINGS[boundary]["operands"]:
        external_bindings.append({"kind": "operand", **binding})
    for binding in EXPECTED_BINDINGS[boundary]["results"]:
        role = binding["role"]
        output_path = directory / f"result-{binding['runner_index']}.bf16"
        shutil.copyfile(case_root / ACCEPTANCE_PATTERN / f"reference_{role}.bf16", output_path)
        if corrupt_output and coordinate == ("source_ordered", "forward", 0) and role == "y":
            output_path.write_bytes(b"\x00\x00" * (2048 * 4096))
        artifact = {
            "path": str(output_path),
            "byte_count": output_path.stat().st_size,
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        if corrupt_artifact_size and coordinate == ("source_ordered", "forward", 0) and role == "y":
            artifact["byte_count"] += 2
        if corrupt_artifact_sha and coordinate == ("source_ordered", "forward", 0) and role == "y":
            artifact["sha256"] = "0" * 64
        external_bindings.append({"kind": "result", **binding, "artifact": artifact})
    coordinate_number = POLICIES.index(policy) * len(BOUNDARIES) * 2 + BOUNDARIES.index(boundary) * 2 + repeat
    cell = {
        "schema": "shuttle_abi22_rmsnorm_representative_raw_cell_v1",
        "policy": policy,
        "boundary": boundary,
        "repeat": repeat,
        "runner_api": {
            "host": "GpuRunner::Execute_BlockHostUntilDone",
            "cuda_event": "GpuRunner::ExecuteTimed",
        },
        "external_bindings": external_bindings,
        "resources": copy.deepcopy(EXPECTED_RESOURCES[boundary]),
        "transport": {
            "contract_operation_count": 0,
            "code_bytes": 4096 + coordinate_number,
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
            "host_samples_ns": list(range(1000 + coordinate_number * 100, 1020 + coordinate_number * 100)),
            "cuda_event_status": "supplemental_not_acceptance",
            "cuda_event_samples_ns": list(range(500 + coordinate_number * 100, 520 + coordinate_number * 100)),
        },
    }
    first_coordinate = coordinate == ("source_ordered", "forward", 0)
    if raw_mutation == "backward_runner_index" and coordinate == ("source_ordered", "backward_recompute", 0):
        next(binding for binding in external_bindings if binding["kind"] == "result")["runner_index"] = 1
    elif raw_mutation == "composed_slot" and coordinate == ("source_ordered", "composed", 0):
        next(binding for binding in external_bindings if binding["kind"] == "result")["slot_ordinal"] = 3
    elif first_coordinate and raw_mutation not in (None, "backward_runner_index", "composed_slot"):
        operand = next(binding for binding in external_bindings if binding["kind"] == "operand")
        result = next(binding for binding in external_bindings if binding["kind"] == "result")
        if raw_mutation == "operand_bytes":
            operand["required_bytes"] += 2
        elif raw_mutation == "operand_type":
            operand["tensor_type"] = "bf16[4096,2048]"
        elif raw_mutation == "operand_alignment":
            operand["alignment"] = 4
        elif raw_mutation == "result_bytes":
            result["required_bytes"] += 2
        elif raw_mutation == "result_type":
            result["tensor_type"] = "bf16[4096,2048]"
        elif raw_mutation == "result_alignment":
            result["alignment"] = 4
        elif raw_mutation == "runner_api":
            cell["runner_api"]["host"] = "wall_clock_without_stream_fence"
        elif raw_mutation == "resource":
            cell["resources"]["launch_count"] += 1
        elif raw_mutation == "contract_transport":
            cell["transport"]["contract_operation_count"] = 1
        elif raw_mutation == "warmup_iterations":
            cell["timing"]["warmup_iterations"] = 9
        elif raw_mutation == "measured_iterations":
            cell["timing"]["measured_iterations"] = 19
        elif raw_mutation == "host_clock":
            cell["timing"]["host_clock"] = "wall_clock_ns"
        elif raw_mutation == "event_status":
            cell["timing"]["cuda_event_status"] = "acceptance_clock"
        elif raw_mutation == "negative_host_sample":
            cell["timing"]["host_samples_ns"][0] = -1
        elif raw_mutation == "nan_host_sample":
            cell["timing"]["host_samples_ns"][0] = math.nan
        elif raw_mutation == "negative_event_sample":
            cell["timing"]["cuda_event_samples_ns"][0] = -1
        elif raw_mutation == "nan_event_sample":
            cell["timing"]["cuda_event_samples_ns"][0] = math.nan
        else:
            raise AssertionError(raw_mutation)
    (directory / "cell.json").write_text(json.dumps(cell, sort_keys=True, separators=(",", ":")))
    return 0


@pytest.fixture(scope="module")
def prepared_case_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case_root = tmp_path_factory.mktemp("shuttle-representative-cases")
    prepared = prepare_representative_cases(case_root)
    assert prepared == {
        "input_digests": {role: SHA256[role] for role in ("x", "gamma", "dy")},
        "reference_digests": {role: SHA256[role] for role in ("y", "dx", "dgamma")},
    }
    expected_sizes = {
        "x.bf16": FULL_BYTES,
        "gamma.bf16": FEATURE_BYTES,
        "dy.bf16": FULL_BYTES,
        "reference_y.bf16": FULL_BYTES,
        "reference_dx.bf16": FULL_BYTES,
        "reference_dgamma.bf16": FEATURE_BYTES,
    }
    for name, size in expected_sizes.items():
        path = case_root / ACCEPTANCE_PATTERN / name
        assert path.stat().st_size == size
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == SHA256[name.removesuffix(".bf16").removeprefix("reference_")]
        )
    return case_root


@pytest.fixture(scope="module")
def executed_evidence(
    tmp_path_factory: pytest.TempPathFactory, prepared_case_root: Path
) -> tuple[dict[str, object], Path, Path, list[list[str]]]:
    workspace = tmp_path_factory.mktemp("shuttle-representative-execution")
    raw_root = workspace / "raw"
    reports_root = workspace / "reports"
    calls: list[list[str]] = []
    evidence = execute_representative(
        runner=workspace / "run-h100-shuttle-rmsnorm-representative.sh",
        case_root=prepared_case_root,
        raw_root=raw_root,
        reports_root=reports_root,
        r12_result=R12_RESULT,
        r12_policy_raw=R12_RAW,
        invoke=lambda argv: _fake_runner(calls, argv),
    )
    return evidence, raw_root, reports_root, calls


def _validate(
    evidence: dict[str, object],
    executed: tuple[dict[str, object], Path, Path, list[list[str]]],
    case_root: Path,
) -> dict[str, object]:
    return validate_shuttle_evidence(
        evidence,
        r12_result=R12_RESULT,
        r12_policy_raw=R12_RAW,
        raw_root=executed[1],
        case_root=case_root,
    )


def test_contract_freezes_r12_inputs_role_limits_resources_and_coordinates(
    executed_evidence: tuple[dict[str, object], Path, Path, list[list[str]]], prepared_case_root: Path
) -> None:
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
    assert BINDING_CONTRACT == EXPECTED_BINDINGS
    assert hashlib.sha256(R12_RESULT.read_bytes()).hexdigest() == SHA256["r12_result"]
    assert hashlib.sha256(R12_RAW.read_bytes()).hexdigest() == SHA256["r12_raw"]

    evidence = executed_evidence[0]
    assert _validate(evidence, executed_evidence, prepared_case_root) == evidence
    assert len(evidence["cells"]) == 12


def test_executor_invokes_exact_matrix_and_assembles_file_backed_evidence(
    executed_evidence: tuple[dict[str, object], Path, Path, list[list[str]]], prepared_case_root: Path
) -> None:
    evidence, raw_root, reports_root, calls = executed_evidence
    expected_plan = execution_plan(runner=Path(calls[0][0]), case_root=prepared_case_root, raw_root=raw_root)
    assert calls == [run["argv"] for run in expected_plan["runs"]]
    assert len(calls) == 12
    assert (
        validate_shuttle_evidence(
            evidence,
            r12_result=R12_RESULT,
            r12_policy_raw=R12_RAW,
            raw_root=raw_root,
            case_root=prepared_case_root,
        )
        == evidence
    )
    assert json.loads((reports_root / "shuttle-policy-raw-v1.json").read_text()) == evidence
    for cell in evidence["cells"]:
        raw_cell = json.loads(
            (raw_root / cell["policy"] / cell["boundary"] / f"repeat-{cell['repeat']}" / "cell.json").read_text()
        )
        assert cell["runner_api"] == raw_cell["runner_api"]
        assert cell["resources"] == raw_cell["resources"]
        assert cell["transport"] == raw_cell["transport"]
        assert cell["timing"] == raw_cell["timing"]
        contract = EXPECTED_BINDINGS[cell["boundary"]]["results"]
        assert [
            (output["role"], output["public_result_index"], output["runner_result_index"], output["slot_ordinal"])
            for output in cell["outputs"]
        ] == [
            (binding["role"], binding["public_result_index"], binding["runner_index"], binding["slot_ordinal"])
            for binding in contract
        ]
        expected_external_bindings = [
            {"kind": "operand", **binding} for binding in EXPECTED_BINDINGS[cell["boundary"]]["operands"]
        ] + [
            {"kind": "result", **binding, "artifact": output["artifact"]}
            for binding, output in zip(contract, cell["outputs"], strict=True)
        ]
        assert raw_cell["external_bindings"] == expected_external_bindings
        for output in cell["outputs"]:
            artifact = output["artifact"]
            assert validate_output_artifact(artifact, raw_root=raw_root, case_root=prepared_case_root) == artifact


@pytest.mark.parametrize(
    "failure",
    [
        "nonzero",
        "missing_cell",
        "artifact_size",
        "artifact_sha",
        "backward_runner_index",
        "composed_slot",
        "operand_bytes",
        "operand_type",
        "operand_alignment",
        "result_bytes",
        "result_type",
        "result_alignment",
        "runner_api",
        "resource",
        "contract_transport",
        "warmup_iterations",
        "measured_iterations",
        "host_clock",
        "event_status",
        "negative_host_sample",
        "nan_host_sample",
        "negative_event_sample",
        "nan_event_sample",
    ],
)
def test_executor_fails_closed_on_process_and_persisted_output_failures(
    tmp_path: Path, prepared_case_root: Path, failure: str
) -> None:
    raw_root = tmp_path / "raw"
    reports_root = tmp_path / "reports"
    kwargs: dict[str, object] = {}
    if failure == "nonzero":
        kwargs["failure_coordinate"] = ("source_ordered", "forward", 0)
    elif failure == "missing_cell":
        kwargs["omit_cell"] = True
    elif failure == "artifact_size":
        kwargs["corrupt_artifact_size"] = True
    elif failure == "artifact_sha":
        kwargs["corrupt_artifact_sha"] = True
    else:
        kwargs["raw_mutation"] = failure
    with pytest.raises(ValueError):
        execute_representative(
            runner=tmp_path / "runner",
            case_root=prepared_case_root,
            raw_root=raw_root,
            reports_root=reports_root,
            r12_result=R12_RESULT,
            r12_policy_raw=R12_RAW,
            invoke=lambda argv: _fake_runner([], argv, **kwargs),
        )


def test_executor_persists_raw_outputs_and_diagnostics_before_numerical_rejection(
    tmp_path: Path, prepared_case_root: Path
) -> None:
    raw_root = tmp_path / "raw"
    reports_root = tmp_path / "reports"
    with pytest.raises(ValueError):
        execute_representative(
            runner=tmp_path / "runner",
            case_root=prepared_case_root,
            raw_root=raw_root,
            reports_root=reports_root,
            r12_result=R12_RESULT,
            r12_policy_raw=R12_RAW,
            invoke=lambda argv: _fake_runner([], argv, corrupt_output=True),
        )
    raw_path = reports_root / "shuttle-policy-raw-v1.json"
    diagnostics_path = reports_root / "shuttle-repeat-diagnostics-v1.json"
    assert raw_path.is_file()
    assert diagnostics_path.is_file()
    raw = json.loads(raw_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    failed = raw["cells"][0]["outputs"][0]
    assert failed["max_absolute_error"] > EXPECTED_LIMITS["y"]["max_absolute_error_below_floor"]
    assert Path(failed["artifact"]["path"]).is_file()
    expected_coordinates = []
    cells = {(cell["policy"], cell["boundary"], cell["repeat"]): cell for cell in raw["cells"]}
    for policy in POLICIES:
        for boundary in BOUNDARIES:
            for role_index, role in enumerate(RESULT_ROLES[boundary]):
                repeat0 = cells[(policy, boundary, 0)]["outputs"][role_index]
                repeat1 = cells[(policy, boundary, 1)]["outputs"][role_index]
                shape = REPRESENTATIVE_SHAPE if role != "dgamma" else (REPRESENTATIVE_SHAPE[1],)
                first = np.fromfile(repeat0["artifact"]["path"], dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(shape)
                second = (
                    np.fromfile(repeat1["artifact"]["path"], dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(shape)
                )
                expected_coordinates.append(
                    {
                        "policy": policy,
                        "boundary": boundary,
                        "role": role,
                        "repeat0_artifact_sha256": repeat0["artifact"]["sha256"],
                        "repeat1_artifact_sha256": repeat1["artifact"]["sha256"],
                        **repeat_output_diagnostics_v4(first, second),
                    }
                )
    assert diagnostics == {
        "schema": "shuttle_abi22_rmsnorm_representative_repeat_diagnostics_v1",
        "coordinates": expected_coordinates,
    }
    failed_repeat = diagnostics["coordinates"][0]
    assert failed_repeat["mismatch_count"] > 0
    assert failed_repeat["max_absolute_error"] > 0
    assert failed_repeat["max_bfloat16_ulp"] > 0


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
def test_every_acceptance_cell_enforces_metrics_roles_and_resources(
    provider: str,
    boundary: str,
    repeat: int,
    executed_evidence: tuple[dict[str, object], Path, Path, list[list[str]]],
    prepared_case_root: Path,
) -> None:
    baseline = executed_evidence[0]
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
                _validate(changed, executed_evidence, prepared_case_root)
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["outputs"][role_index]["max_absolute_error"] = math.nan
        with pytest.raises(ValueError):
            _validate(changed, executed_evidence, prepared_case_root)
        changed = copy.deepcopy(baseline)
        output = changed["cells"][index]["outputs"][role_index]
        output["below_floor_count"] += 1
        output["eligible_count"] -= 1
        with pytest.raises(ValueError):
            _validate(changed, executed_evidence, prepared_case_root)
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["outputs"][role_index]["absolute_floor"] = limits["absolute_floor"] * 2
        with pytest.raises(ValueError):
            _validate(changed, executed_evidence, prepared_case_root)
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["outputs"][role_index]["max_absolute_error"] = (
            limits["max_absolute_error_below_floor"] / 2
        )
        with pytest.raises(ValueError):
            _validate(changed, executed_evidence, prepared_case_root)

    for field in EXPECTED_RESOURCES[boundary]:
        changed = copy.deepcopy(baseline)
        changed["cells"][index]["resources"][field] += 1
        with pytest.raises(ValueError):
            _validate(changed, executed_evidence, prepared_case_root)


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
        "wrong_runner_index",
        "wrong_slot_ordinal",
        "wrong_runner_api",
        "repeat_drift",
        "contract_fusion",
        "wrong_warmup",
        "short_host_samples",
        "short_event_samples",
        "wrong_external_result",
        "wrong_external_raw",
    ],
)
def test_structural_transport_repeat_and_external_evidence_mutations_reject(
    mutation: str,
    executed_evidence: tuple[dict[str, object], Path, Path, list[list[str]]],
    prepared_case_root: Path,
) -> None:
    changed = copy.deepcopy(executed_evidence[0])
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
    elif mutation == "wrong_runner_index":
        cell["outputs"][0]["runner_result_index"] = 1
    elif mutation == "wrong_slot_ordinal":
        cell["outputs"][0]["slot_ordinal"] += 1
    elif mutation == "wrong_runner_api":
        cell["runner_api"]["host"] = "stream_ordered_without_fence"
    elif mutation == "repeat_drift":
        repeated = next(
            entry
            for entry in changed["cells"]
            if entry["policy"] == cell["policy"] and entry["boundary"] == cell["boundary"] and entry["repeat"] == 1
        )
        output = repeated["outputs"][0]
        original = Path(output["artifact"]["path"])
        drift_path = original.with_name("result-0-drift.bf16")
        actual_bits = np.fromfile(original, dtype=np.uint16)
        reference_bits = np.fromfile(prepared_case_root / ACCEPTANCE_PATTERN / "reference_y.bf16", dtype=np.uint16)
        reference_f32 = reference_bits.view(ml_dtypes.bfloat16).astype(np.float32)
        index = int(np.flatnonzero(np.abs(reference_f32) >= EXPECTED_LIMITS["y"]["absolute_floor"])[0])
        actual_bits[index] ^= np.uint16(1)
        drift_path.write_bytes(actual_bits.tobytes())
        actual = actual_bits.view(ml_dtypes.bfloat16).reshape(REPRESENTATIVE_SHAPE)
        reference = reference_bits.view(ml_dtypes.bfloat16).reshape(REPRESENTATIVE_SHAPE)
        output.update(error_metrics_v4(actual, reference, role="y"))
        digest = hashlib.sha256(drift_path.read_bytes()).hexdigest()
        output["digest"] = digest
        output["artifact"] = {
            "path": str(drift_path),
            "byte_count": drift_path.stat().st_size,
            "sha256": digest,
        }
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
        _validate(changed, executed_evidence, prepared_case_root)


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


def test_r12_external_evidence_is_revalidated_not_self_reported(
    tmp_path: Path,
    executed_evidence: tuple[dict[str, object], Path, Path, list[list[str]]],
    prepared_case_root: Path,
) -> None:
    changed_result = json.loads(R12_RESULT.read_text())
    changed_result["evidence"]["acceptance"]["accepted_shape"] = [2048, 4097]
    changed_path = tmp_path / "result-v4.json"
    changed_path.write_text(json.dumps(changed_result, sort_keys=True, separators=(",", ":")))
    document = copy.deepcopy(executed_evidence[0])
    document["external_evidence"]["r12_result_sha256"] = hashlib.sha256(changed_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_shuttle_evidence(
            document,
            r12_result=changed_path,
            r12_policy_raw=R12_RAW,
            raw_root=executed_evidence[1],
            case_root=prepared_case_root,
        )
