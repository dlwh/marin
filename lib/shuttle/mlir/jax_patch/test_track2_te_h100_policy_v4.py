# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-level durability contract for RMSNorm role policy v4."""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import abi22_te_rmsnorm_oracle_v2 as oracle
import ml_dtypes
import numpy as np
import track2_te_h100_batch as batch
from track2_te_h100_batch import BatchPaths

ACCEPTANCE_PATTERN = "heldout_splitmix64_v3"
DEVICE = {"model": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test", "compute_capability": "9.0"}
XLA_RUNTIME = {
    "backend": "gpu",
    "device_count": 1,
    "device_id": 0,
    "device_kind": "NVIDIA H100 80GB HBM3",
    "jax_git_commit": oracle.JAX_REVISION,
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "platform_version": "CUDA 13.2",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arguments(argv: list[str]) -> dict[str, str]:
    return {value[2:]: argv[index + 1] for index, value in enumerate(argv[:-1]) if value.startswith("--")}


def _fake_invoke(
    paths: BatchPaths,
    calls: list[list[str]],
    argv: list[str],
    *,
    acceptance_drift: bool = True,
    qualification_drift: bool = False,
    alias_te_outputs: bool = False,
) -> int:
    calls.append(argv)
    arguments = _arguments(argv)
    shape = (int(arguments["rows"]), int(arguments["features"]))
    boundary = arguments["boundary"]
    case = Path(arguments["case-directory"])
    output = Path(arguments["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    roles = oracle.OUTPUT_ROLES[boundary]
    input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")

    if argv[0] == str(paths.te_runner):
        output_files = {}
        for role in roles:
            reference_path = case / f"reference_{role}.bf16"
            output_path = reference_path if alias_te_outputs else output.parent / f"{output.stem}-{role}.bf16"
            if not alias_te_outputs:
                shutil.copyfile(reference_path, output_path)
            output_files[role] = str(output_path)
        output.write_text(
            json.dumps(
                {
                    "schema": "shuttle_abi22_te_rmsnorm_raw_cell_v2",
                    "shape": list(shape),
                    "boundary": boundary,
                    "epsilon": oracle.EPSILON,
                    "warmup_invocations": 10,
                    "raw_host_blocking_ns": [100_000 + index for index in range(20)],
                    "raw_cuda_event_ns": [50_000 + index for index in range(20)],
                    "output_files": output_files,
                    "workspace_bytes": {"forward": 0, "backward": 0},
                },
                sort_keys=True,
            )
        )
        return 0

    error_metrics = getattr(oracle, "error_metrics_v4", None)
    if not callable(error_metrics):
        error_metrics = oracle.error_metrics
    output_files = {}
    outputs = []
    output_directory = Path(arguments.get("output-files-directory", paths.raw / f"{output.stem}-outputs"))
    output_directory.mkdir(parents=True, exist_ok=True)
    for role in roles:
        reference_path = case / f"reference_{role}.bf16"
        actual_path = output_directory / f"{role}.bf16"
        actual_path.write_bytes(reference_path.read_bytes())
        output_files[role] = str(actual_path)
        output_shape = shape if role != "dgamma" else (shape[1],)
        actual = np.fromfile(actual_path, dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(output_shape)
        reference = np.fromfile(reference_path, dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(output_shape)
        outputs.append(
            {
                "role": role,
                "digest": _digest(actual_path),
                "reference_digest": _digest(reference_path),
                **error_metrics(actual, reference, role=role),
            }
        )
    acceptance_coordinate = (
        acceptance_drift
        and any(ACCEPTANCE_PATTERN in argument for argument in argv)
        and boundary == "composed"
        and "repeat1" in output.name
    )
    qualification_coordinate = (
        qualification_drift
        and any(oracle.QUALIFICATION_PATTERN in argument for argument in argv)
        and boundary == "composed"
        and "repeat1" in output.name
    )
    if acceptance_coordinate or qualification_coordinate:
        actual_path = Path(output_files["y"])
        payload = bytearray(actual_path.read_bytes())
        payload[:2] = (int.from_bytes(payload[:2], "little") + 1).to_bytes(2, "little")
        actual_path.write_bytes(payload)
    scope = "architecture_fixture" if shape == oracle.ARCHITECTURE_SHAPE else "representative_performance_qualification"
    output.write_text(
        json.dumps(
            {
                "shape": list(shape),
                "scope": scope,
                "boundary": boundary,
                "input_digests": {role: _digest(case / f"{role}.bf16") for role in input_roles},
                "execution": {
                    "call_sequence": list(oracle.CALL_SEQUENCES[boundary]),
                    "rsigma_state": "invocation_local_not_public",
                },
                "outputs": outputs,
                "output_files": output_files,
                "observed_runtime": XLA_RUNTIME,
                "timing": {
                    "warmup_invocations": 10,
                    "raw_host_blocking_ns": [200_000 + index for index in range(20)],
                    "median_host_blocking_ns": statistics.median([200_000 + index for index in range(20)]),
                    "raw_cuda_event_ns": None,
                },
            },
            sort_keys=True,
        )
    )
    return 0


class Track2TeH100PolicyV4Test(unittest.TestCase):
    def test_batch_plan_and_cli_select_v4_explicitly(self) -> None:
        batch_plan_v4 = getattr(batch, "batch_plan_v4", None)
        self.assertTrue(callable(batch_plan_v4), "versioned v4 batch-plan API is missing")
        plan = batch_plan_v4(workspace=Path("/tmp/v4"), runner_repository=Path("/app"))
        self.assertEqual(plan["schema"], "shuttle_track2_te_h100_batch_plan_v4")
        self.assertEqual(plan["qualification"], {**oracle.NUMERICAL_POLICY_V4, "run_count": 60})
        v3_plan = batch.batch_plan_v3(workspace=Path("/tmp/v4"), runner_repository=Path("/app"))
        self.assertEqual(v3_plan["qualification"]["run_count"], 48)
        normalized = copy.deepcopy(plan)
        normalized["schema"] = v3_plan["schema"]
        normalized["qualification"] = v3_plan["qualification"]
        self.assertEqual(normalized, v3_plan)

        expected = {"schema": "v4-result"}
        stdout = io.StringIO()
        with (
            patch.object(batch, "execute_batch_v4", return_value=expected) as execute,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(
                batch.main(
                    [
                        "--workspace",
                        "/tmp/v4",
                        "--runner-repository",
                        "/app",
                        "--source-manifest",
                        "/app/source.json",
                        "--base-image",
                        batch.BASE_IMAGE,
                        "--execute-v4",
                    ]
                ),
                0,
            )
        execute.assert_called_once_with(
            workspace=Path("/tmp/v4"),
            runner_repository=Path("/app"),
            manifest_path=Path("/app/source.json"),
            observed_base_image=batch.BASE_IMAGE,
        )
        self.assertEqual(json.loads(stdout.getvalue().split("=", 1)[1]), expected)

    def test_execute_batch_v4_returns_only_v4_evidence(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        workspace = root / "workspace"
        repository = root / "repository"
        repository.mkdir()
        manifest = root / "source.json"
        expected_evidence = {"acceptance": {"schema": "shuttle_abi22_te_xla_rmsnorm_acceptance_v4"}}
        observed_plan = {}

        def hydrate(paths: BatchPaths, plan: dict[str, object]) -> dict[str, object]:
            observed_plan.update(plan)
            paths.te_runner.write_bytes(b"runner-binary")
            return {
                "device": DEVICE,
                "driver": "driver",
                "cuda_runtime_headers": {},
                "runtime_libraries": {},
                "transformer_engine": {},
            }

        with (
            patch.object(batch.platform, "system", return_value="Linux"),
            patch.object(batch, "validate_source_manifest", return_value={"commit": "c", "tree": "t"}),
            patch.object(batch, "_execute_hydration", side_effect=hydrate),
            patch.object(batch, "execute_oracle", side_effect=AssertionError("legacy v2 worker called")),
            patch.object(batch, "execute_oracle_v3", side_effect=AssertionError("legacy v3 worker called")),
            patch.object(batch, "execute_oracle_v4", return_value=expected_evidence) as execute_v4,
        ):
            result = batch.execute_batch_v4(
                workspace=workspace,
                runner_repository=repository,
                manifest_path=manifest,
                observed_base_image=batch.BASE_IMAGE,
            )

        self.assertEqual(result["schema"], "shuttle_track2_te_h100_batch_result_v4")
        self.assertEqual(result["evidence"], expected_evidence)
        self.assertEqual(observed_plan["schema"], "shuttle_track2_te_h100_batch_plan_v4")
        self.assertEqual(observed_plan["qualification"], {**oracle.NUMERICAL_POLICY_V4, "run_count": 60})
        execute_v4.assert_called_once()
        self.assertEqual(json.loads(BatchPaths.under(workspace).result.read_text()), result)

    def test_worker_reports_qualification_drift_without_rejecting_acceptance(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        paths = BatchPaths.under(Path(temporary.name) / "workspace")
        paths.workspace.mkdir()
        paths.reports.mkdir()
        paths.raw.mkdir()
        paths.te_runner.write_bytes(b"runner-binary")
        paths.te_library.parent.mkdir(parents=True)
        paths.te_library.write_bytes(b"resolved-te-library")
        calls: list[list[str]] = []

        result = batch.execute_oracle_v4(
            paths=paths,
            device=DEVICE,
            invoke=lambda argv: _fake_invoke(
                paths,
                calls,
                argv,
                acceptance_drift=False,
                qualification_drift=True,
            ),
        )

        self.assertEqual(result["acceptance"]["status"], "heldout_splitmix64_v3_numerics_accepted")
        drift = next(item for item in result["repeat_diagnostics"]["coordinates"] if item["mismatch_count"])
        self.assertEqual(
            result["qualification"]["qualification"]["repeat_digest_drifts"],
            [
                {
                    "producer": "ordinary_jax_xla",
                    "input_pattern": oracle.QUALIFICATION_PATTERN,
                    "boundary": "composed",
                    "role": "y",
                    "repeat0_digest": drift["repeat0_digest"],
                    "repeat1_digest": drift["repeat1_digest"],
                }
            ],
        )
        self.assertEqual(
            (drift["producer"], drift["input_pattern"], drift["boundary"], drift["role"]),
            ("ordinary_jax_xla", oracle.QUALIFICATION_PATTERN, "composed", "y"),
        )
        self.assertEqual(len(calls), 72)

    def test_worker_rejects_te_output_paths_that_alias_case_inputs(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        paths = BatchPaths.under(Path(temporary.name) / "workspace")
        paths.workspace.mkdir()
        paths.reports.mkdir()
        paths.raw.mkdir()
        paths.te_runner.write_bytes(b"runner-binary")
        paths.te_library.parent.mkdir(parents=True)
        paths.te_library.write_bytes(b"resolved-te-library")

        with self.assertRaisesRegex(ValueError, "output.*path|raw output"):
            batch.execute_oracle_v4(
                paths=paths,
                device=DEVICE,
                invoke=lambda argv: _fake_invoke(
                    paths,
                    [],
                    argv,
                    acceptance_drift=False,
                    alias_te_outputs=True,
                ),
            )

    def test_worker_persists_raw_outputs_and_repeat_diagnostics_before_rejection(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        paths = BatchPaths.under(Path(temporary.name) / "workspace")
        paths.workspace.mkdir()
        paths.reports.mkdir()
        paths.raw.mkdir()
        paths.te_runner.write_bytes(b"runner-binary")
        paths.te_library.parent.mkdir(parents=True)
        paths.te_library.write_bytes(b"resolved-te-library")
        calls: list[list[str]] = []
        stdout = io.StringIO()
        execute_oracle_v4 = getattr(batch, "execute_oracle_v4", None)
        self.assertTrue(callable(execute_oracle_v4), "versioned v4 worker API is missing")
        repeat_output_diagnostics_v4 = getattr(oracle, "repeat_output_diagnostics_v4", None)
        self.assertTrue(callable(repeat_output_diagnostics_v4), "versioned repeat diagnostic API is missing")

        with contextlib.redirect_stdout(stdout), self.assertRaisesRegex(ValueError, "repeat"):
            execute_oracle_v4(paths=paths, device=DEVICE, invoke=lambda argv: _fake_invoke(paths, calls, argv))

        lines = stdout.getvalue().splitlines()
        raw_lines = [line for line in lines if line.startswith("TRACK2_TE_POLICY_RAW_V4=")]
        repeat_lines = [line for line in lines if line.startswith("TRACK2_TE_REPEAT_DIAGNOSTICS_V4=")]
        report_lines = [line for line in lines if line.startswith("TRACK2_TE_BASE_REPORTS_V4=")]
        self.assertEqual(len(raw_lines), 1)
        self.assertEqual(len(repeat_lines), 1)
        self.assertEqual(len(report_lines), 1)
        raw = json.loads(raw_lines[0].split("=", 1)[1])
        diagnostics = json.loads(repeat_lines[0].split("=", 1)[1])
        self.assertEqual(json.loads((paths.reports / "policy-raw-v4.json").read_text()), raw)
        self.assertEqual(json.loads((paths.reports / "policy-repeat-diagnostics-v4.json").read_text()), diagnostics)
        self.assertEqual(raw["schema"], "shuttle_abi22_rmsnorm_policy_evidence_v4")
        self.assertEqual(len(raw["cells"]), 60)
        expected_coordinates = [
            (producer, pattern, boundary, role)
            for producer in oracle.PRODUCERS
            for pattern in (
                oracle.QUALIFICATION_PATTERN,
                oracle.HELDOUT_PATTERN,
                oracle.ACCEPTANCE_PATTERN,
                oracle.POLICY_V3_ACCEPTANCE_PATTERN,
                ACCEPTANCE_PATTERN,
            )
            for boundary in oracle.BOUNDARIES
            for role in oracle.OUTPUT_ROLES[boundary]
        ]
        self.assertEqual(
            [
                (item["producer"], item["input_pattern"], item["boundary"], item["role"])
                for item in diagnostics["coordinates"]
            ],
            expected_coordinates,
        )
        self.assertEqual(diagnostics["schema"], "shuttle_abi22_rmsnorm_repeat_diagnostics_v4")
        self.assertTrue(paths.te_report.is_file())
        self.assertTrue(paths.xla_report.is_file())
        te_report = json.loads(paths.te_report.read_text())
        xla_report = json.loads(paths.xla_report.read_text())
        self.assertEqual(te_report["producer"], "transformer_engine_2_17")
        self.assertEqual(xla_report["producer"], "ordinary_jax_xla")
        expected_base_coordinates = [
            (list(shape), boundary) for shape in oracle.SHAPES for boundary in oracle.BOUNDARIES
        ]
        for report in (te_report, xla_report):
            self.assertEqual(report["schema"], "shuttle_abi22_te_xla_rmsnorm_oracle_v2")
            self.assertEqual(report["device"], DEVICE)
            self.assertEqual(
                [(cell["shape"], cell["boundary"]) for cell in report["cells"]],
                expected_base_coordinates,
            )
        self.assertEqual(
            json.loads(report_lines[0].split("=", 1)[1]),
            {"ordinary_jax_xla": xla_report, "transformer_engine_2_17": te_report},
        )
        policy_cells = {
            (cell["producer"], cell["input_pattern"], cell["boundary"], cell["repeat"]): cell for cell in raw["cells"]
        }
        for item in diagnostics["coordinates"]:
            self.assertEqual(
                set(item),
                {
                    "producer",
                    "input_pattern",
                    "boundary",
                    "role",
                    "repeat0_digest",
                    "repeat1_digest",
                    "mismatch_count",
                    "max_absolute_error",
                    "max_bfloat16_ulp",
                    "first_mismatch_index",
                    "first_mismatch_repeat0",
                    "first_mismatch_repeat1",
                    "max_absolute_error_index",
                    "max_absolute_error_repeat0",
                    "max_absolute_error_repeat1",
                    "raw_output_files",
                },
            )
            producer = item["producer"]
            pattern = item["input_pattern"]
            boundary = item["boundary"]
            role = item["role"]
            expected_files = []
            output_shape = oracle.REPRESENTATIVE_SHAPE if role != "dgamma" else (oracle.REPRESENTATIVE_SHAPE[1],)
            for repeat in range(2):
                raw_cell = json.loads(
                    (paths.raw / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json").read_text()
                )
                expected_files.append(raw_cell["output_files"][role])
                cell = policy_cells[(producer, pattern, boundary, repeat)]
                output = next(output for output in cell["outputs"] if output["role"] == role)
                self.assertEqual(output["digest"], _digest(Path(expected_files[-1])))
            self.assertEqual(item["raw_output_files"], expected_files)
            self.assertTrue(all(Path(path).is_file() for path in expected_files))
            self.assertNotEqual(expected_files[0], expected_files[1])
            rows, features = oracle.REPRESENTATIVE_SHAPE
            case_directory = paths.cases / (f"{rows}x{features}" if pattern == oracle.QUALIFICATION_PATTERN else pattern)
            for path in map(Path, expected_files):
                self.assertTrue(path.resolve().is_relative_to(paths.raw.resolve()))
                self.assertFalse(path.resolve().is_relative_to(paths.cases.resolve()))
                self.assertNotEqual(path.stat().st_ino, (case_directory / f"reference_{role}.bf16").stat().st_ino)
            repeat_values = [
                np.fromfile(path, dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(output_shape)
                for path in expected_files
            ]
            expected_item = {
                "producer": producer,
                "input_pattern": pattern,
                "boundary": boundary,
                "role": role,
                "repeat0_digest": _digest(Path(expected_files[0])),
                "repeat1_digest": _digest(Path(expected_files[1])),
                **repeat_output_diagnostics_v4(*repeat_values),
                "raw_output_files": expected_files,
            }
            self.assertEqual(item, expected_item)
            if (producer, pattern, boundary, role) != (
                "ordinary_jax_xla",
                ACCEPTANCE_PATTERN,
                "composed",
                "y",
            ):
                self.assertEqual(item["mismatch_count"], 0)
                self.assertEqual(item["max_absolute_error"], 0.0)
                self.assertEqual(item["max_bfloat16_ulp"], 0)
        mismatch = next(
            item
            for item in diagnostics["coordinates"]
            if item["producer"] == "ordinary_jax_xla"
            and item["input_pattern"] == ACCEPTANCE_PATTERN
            and item["boundary"] == "composed"
            and item["role"] == "y"
        )
        self.assertEqual(mismatch["mismatch_count"], 1)
        self.assertGreater(mismatch["max_absolute_error"], 0)
        self.assertEqual(mismatch["max_bfloat16_ulp"], 1)
        self.assertEqual(mismatch["first_mismatch_index"], 0)
        self.assertTrue(all(Path(path).is_file() for path in mismatch["raw_output_files"]))
        self.assertEqual(len(calls), 72)


if __name__ == "__main__":
    unittest.main()
