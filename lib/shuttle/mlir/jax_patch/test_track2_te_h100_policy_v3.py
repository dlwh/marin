# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-level durability contract for RMSNorm role policy v3."""

import contextlib
import hashlib
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path

import abi22_te_rmsnorm_oracle_v2 as oracle
import ml_dtypes
import numpy as np
import track2_te_h100_batch as batch
from track2_te_h100_batch import BatchPaths

ACCEPTANCE_PATTERN = "heldout_splitmix64_v2"
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


def _fake_invoke(paths: BatchPaths, calls: list[list[str]], argv: list[str]) -> int:
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
                    "output_files": {role: str(case / f"reference_{role}.bf16") for role in roles},
                    "workspace_bytes": {"forward": 0, "backward": 0},
                },
                sort_keys=True,
            )
        )
        return 0

    error_metrics = getattr(oracle, "error_metrics_v3", None)
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
    if any(ACCEPTANCE_PATTERN in argument for argument in argv) and boundary == "composed" and "repeat1" in output.name:
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


class Track2TeH100PolicyV3Test(unittest.TestCase):
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
        execute_oracle_v3 = getattr(batch, "execute_oracle_v3", None)
        self.assertTrue(callable(execute_oracle_v3), "versioned v3 worker API is missing")
        repeat_output_diagnostics_v3 = getattr(oracle, "repeat_output_diagnostics_v3", None)
        self.assertTrue(callable(repeat_output_diagnostics_v3), "versioned repeat diagnostic API is missing")

        with contextlib.redirect_stdout(stdout), self.assertRaisesRegex(ValueError, "repeat"):
            execute_oracle_v3(paths=paths, device=DEVICE, invoke=lambda argv: _fake_invoke(paths, calls, argv))

        lines = stdout.getvalue().splitlines()
        raw_lines = [line for line in lines if line.startswith("TRACK2_TE_POLICY_RAW_V3=")]
        repeat_lines = [line for line in lines if line.startswith("TRACK2_TE_REPEAT_DIAGNOSTICS_V3=")]
        self.assertEqual(len(raw_lines), 1)
        self.assertEqual(len(repeat_lines), 1)
        raw = json.loads(raw_lines[0].split("=", 1)[1])
        diagnostics = json.loads(repeat_lines[0].split("=", 1)[1])
        self.assertEqual(json.loads((paths.reports / "policy-raw-v3.json").read_text()), raw)
        self.assertEqual(json.loads((paths.reports / "policy-repeat-diagnostics-v3.json").read_text()), diagnostics)
        self.assertEqual(raw["schema"], "shuttle_abi22_rmsnorm_policy_evidence_v3")
        self.assertEqual(len(raw["cells"]), 48)
        expected_coordinates = [
            (producer, pattern, boundary, role)
            for producer in oracle.PRODUCERS
            for pattern in (
                oracle.QUALIFICATION_PATTERN,
                oracle.HELDOUT_PATTERN,
                oracle.ACCEPTANCE_PATTERN,
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
        self.assertEqual(diagnostics["schema"], "shuttle_abi22_rmsnorm_repeat_diagnostics_v3")
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
            if producer == "ordinary_jax_xla":
                self.assertNotEqual(expected_files[0], expected_files[1])
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
                **repeat_output_diagnostics_v3(*repeat_values),
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
        self.assertEqual(len(calls), 60)


if __name__ == "__main__":
    unittest.main()
