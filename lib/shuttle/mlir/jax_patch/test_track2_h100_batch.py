# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from track2_h100_batch import (
    BASE_IMAGE,
    TRACK2_HERMETIC_CUDA_VERSION,
    BatchPaths,
    batch_plan,
    canonical_json,
    validate_source_manifest,
)


def _manifest(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = root / ".shuttle-track2-source-manifest.json"
    manifest.write_bytes(
        canonical_json(
            {
                "commit": "1" * 40,
                "entries": [
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                ],
                "schema": "shuttle_track2_h100_source_manifest_v1",
                "tree": "2" * 40,
            }
        )
    )
    return manifest


def test_batch_plan_runs_xla_then_shuttle_on_one_workspace_and_device(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    compiler_identity = f"shuttle-abi22:{'a' * 40}:{'b' * 40}:{'c' * 64}"
    plan = batch_plan(
        workspace=workspace,
        runner_repository=Path("/app"),
        compiler_identity=compiler_identity,
    )
    phases = plan["phases"]

    assert plan["base_image"] == BASE_IMAGE
    assert plan["hardware"] == {"gpu_count": 1, "gpu_model": "H100"}
    assert [phase["name"] for phase in phases] == [
        "device",
        "xla_init",
        "xla_remote",
        "xla_fetch",
        "xla_checkout",
        "bazel_fetch",
        "xla",
        "shuttle",
    ]
    assert phases[1]["argv"] == ["git", "init", "--quiet"]
    assert phases[2]["argv"] == ["git", "remote", "add", "origin", "https://github.com/openxla/xla.git"]
    assert phases[3]["argv"][0:4] == ["git", "fetch", "--depth=1", "origin"]
    assert phases[4]["argv"] == ["git", "checkout", "--detach", "FETCH_HEAD"]
    assert phases[5]["argv"][0:7] == [
        "curl",
        "--fail",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
    ]
    xla_command = phases[6]["argv"]
    shuttle_command = phases[7]["argv"]
    assert "@shuttle_mlir//:abi22_h100_xla_baseline_test" in xla_command
    assert "@shuttle_mlir//:gpu_transport_v2_h100_performance_test" in shuttle_command
    assert xla_command[0:2] == shuttle_command[0:2]
    assert TRACK2_HERMETIC_CUDA_VERSION == "12.9.1"
    assert f"--repo_env=HERMETIC_CUDA_VERSION={TRACK2_HERMETIC_CUDA_VERSION}" in xla_command
    assert not any(value == "--repo_env=HERMETIC_CUDA_VERSION=13.0.0" for value in xla_command)
    assert any("SHUTTLE_H100_XLA_HOST_REPORT=" in value for value in xla_command)
    assert any("SHUTTLE_H100_XLA_HOST_REPORT=" in value for value in shuttle_command)
    assert any("SHUTTLE_H100_PERF_EXPLORATORY=1" in value for value in shuttle_command)
    assert f"--test_env=SHUTTLE_COMPILER_IDENTITY={compiler_identity}" in shuttle_command
    assert not any("SHUTTLE_H100_PERF_ACCEPTANCE" in value for value in shuttle_command)


def test_source_manifest_binds_every_declared_byte(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "lib/shuttle/mlir/probe.txt", b"reviewed bytes")

    identity = validate_source_manifest(tmp_path, manifest)

    assert identity == {
        "commit": "1" * 40,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tree": "2" * 40,
    }

    (tmp_path / "lib/shuttle/mlir/probe.txt").write_bytes(b"drifted bytes")
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        validate_source_manifest(tmp_path, manifest)


def test_batch_paths_keep_raw_reports_separate(tmp_path: Path) -> None:
    paths = BatchPaths.under(tmp_path)

    assert (
        len(
            {
                paths.xla_report,
                paths.te_status,
                paths.shuttle_report,
                paths.shuttle_validated,
                paths.exploratory_comparison,
                paths.result,
            }
        )
        == 6
    )
    assert json.loads(canonical_json({"b": 1, "a": 2})) == {"a": 2, "b": 1}
