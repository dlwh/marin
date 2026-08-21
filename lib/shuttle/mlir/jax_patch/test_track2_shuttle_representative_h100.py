# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from track2_h100_batch import BASE_IMAGE, main, representative_batch_plan
from track2_h100_iris import (
    build_launch_request,
    build_representative_launch_request,
    decode_representative_launch_request,
    representative_job_name,
    validate_representative_launch_request,
)

MARIN_REPOSITORY = Path(__file__).parents[4]


class ShuttleRepresentativeBatchTest(unittest.TestCase):
    def test_plan_reuses_hydration_and_runs_only_the_sealed_matrix(self) -> None:
        workspace = Path("/tmp/shuttle-track2-rmsnorm")
        plan = representative_batch_plan(workspace=workspace, runner_repository=Path("/app"))
        phases = plan["phases"]

        self.assertEqual(plan["schema"], "shuttle_track2_rmsnorm_representative_batch_plan_v1")
        self.assertEqual(plan["hardware"], {"gpu_count": 1, "gpu_model": "H100"})
        self.assertEqual(
            [phase["name"] for phase in phases],
            [
                "device",
                "source_permissions",
                "xla_init",
                "xla_remote",
                "xla_fetch",
                "xla_checkout",
                "xla_workspace_check",
                "xla_workspace_apply",
                "bazel_fetch",
                "shuttle_build",
                "shuttle_execute",
            ],
        )
        self.assertEqual(
            phases[1]["argv"],
            [
                "chmod",
                "755",
                "/app/lib/shuttle/mlir/test/run-h100-shuttle-rmsnorm-representative.sh",
                "/app/lib/shuttle/mlir/test/run-h100-shuttle-rmsnorm-representative-matrix.sh",
            ],
        )
        build = phases[9]["argv"]
        self.assertEqual(build[2], "build")
        self.assertEqual(build[-1], "@shuttle_mlir//:run_h100_shuttle_rmsnorm_representative_matrix")
        self.assertIn("--config=cuda", build)
        self.assertEqual(
            phases[10]["argv"],
            [
                "/tmp/shuttle-track2-rmsnorm/xla/bazel-bin/external/shuttle_mlir/"
                "run_h100_shuttle_rmsnorm_representative_matrix",
                "--python",
                "/opt/h100-evidence-runtime/bin/python",
                "--case-root",
                "/tmp/shuttle-track2-rmsnorm/cases",
                "--raw-root",
                "/tmp/shuttle-track2-rmsnorm/raw",
                "--reports-root",
                "/tmp/shuttle-track2-rmsnorm/reports",
                "--output",
                "/tmp/shuttle-track2-rmsnorm/reports/shuttle-representative.json",
            ],
        )
        self.assertTrue(all("xla-baseline" not in " ".join(phase["argv"]) for phase in phases))
        self.assertTrue(all("transformer_engine" not in " ".join(phase["argv"]) for phase in phases))

    def test_cli_requires_explicit_representative_mode(self) -> None:
        calls = []

        def fake_execute(**kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"schema": "shuttle_track2_rmsnorm_representative_batch_result_v1"}

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("track2_h100_batch.execute_representative_batch", side_effect=fake_execute),
        ):
            workspace = Path(temporary) / "work"
            self.assertEqual(
                main(
                    [
                        "--workspace",
                        str(workspace),
                        "--runner-repository",
                        "/app",
                        "--source-manifest",
                        "/app/.shuttle-track2-source-manifest.json",
                        "--base-image",
                        BASE_IMAGE,
                        "--execute-representative",
                    ]
                ),
                0,
            )
        self.assertEqual(
            calls,
            [
                {
                    "workspace": workspace,
                    "runner_repository": Path("/app"),
                    "manifest_path": Path("/app/.shuttle-track2-source-manifest.json"),
                    "observed_base_image": BASE_IMAGE,
                }
            ],
        )

    def test_request_changes_only_identity_and_explicit_entrypoint(self) -> None:
        baseline = build_launch_request(bundle_root=MARIN_REPOSITORY, run=1)
        representative = build_representative_launch_request(bundle_root=MARIN_REPOSITORY, run=1)
        validate_representative_launch_request(representative, run=1)
        raw = representative.SerializeToString(deterministic=True)
        self.assertEqual(decode_representative_launch_request(raw, run=1), representative)
        self.assertEqual(representative.name, representative_job_name(1))
        self.assertEqual(representative.entrypoint.run_command.argv[-1], "--execute-representative")

        normalized = deepcopy(representative)
        normalized.name = baseline.name
        normalized.entrypoint.run_command.argv[-1] = "--execute"
        self.assertEqual(
            normalized.SerializeToString(deterministic=True), baseline.SerializeToString(deterministic=True)
        )


if __name__ == "__main__":
    unittest.main()
