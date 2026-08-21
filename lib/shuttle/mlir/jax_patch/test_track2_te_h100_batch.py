# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the sealed Track 2 TE/XLA H100 batch."""

import hashlib
from pathlib import Path

import pytest
from track2_te_h100_batch import (
    BASE_IMAGE,
    SHUTTLE_BLOCKER,
    TE_SOURCE_COMMIT,
    TE_SOURCE_TREE,
    TE_WHEEL_SHA256,
    TE_WHEEL_URL,
    BatchPaths,
    batch_plan,
    canonical_json,
    validate_source_manifest,
)


def _manifest(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = root / ".shuttle-track2-te-source-manifest.json"
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
                "schema": "shuttle_track2_te_h100_source_manifest_v1",
                "tree": "2" * 40,
            }
        )
    )
    return manifest


def test_batch_plan_pins_te_wheel_source_build_and_exact_v2_matrix(tmp_path: Path) -> None:
    workspace = tmp_path / "work"

    plan = batch_plan(workspace=workspace, runner_repository=Path("/app"))
    phases = plan["phases"]

    assert plan["schema"] == "shuttle_track2_te_h100_batch_plan_v1"
    assert plan["base_image"] == BASE_IMAGE
    assert plan["hardware"] == {"gpu_count": 1, "gpu_model": "H100"}
    assert plan["oracle"] == {
        "schema": "shuttle_abi22_te_xla_rmsnorm_plan_v2",
        "epsilon": 1e-6,
        "shapes": [[7, 13], [2048, 4096]],
        "representative_shape": [2048, 4096],
        "boundaries": ["forward", "backward_recompute", "composed"],
        "producer_order": ["transformer_engine_2_17", "ordinary_jax_xla"],
        "run_count": 12,
    }
    assert plan["shuttle"] == {"status": "pending", "blocker": SHUTTLE_BLOCKER}
    assert [phase["name"] for phase in phases] == [
        "device",
        "te_init",
        "te_remote",
        "te_fetch",
        "te_checkout",
        "te_wheel_fetch",
        "te_wheel_extract",
        "te_runner_build",
    ]
    assert phases[3]["argv"] == ["git", "fetch", "--depth=1", "origin", TE_SOURCE_COMMIT]
    assert phases[4]["argv"] == ["git", "checkout", "--detach", "FETCH_HEAD"]
    assert phases[5]["argv"] == [
        "curl",
        "--fail",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--output",
        str(BatchPaths.under(workspace).te_wheel),
        TE_WHEEL_URL,
    ]
    compile_argv = phases[7]["argv"]
    assert compile_argv[:4] == ["/usr/bin/g++", "-std=c++20", "-O3", "-Wall"]
    assert "-Werror" in compile_argv
    assert "-ltransformer_engine" in compile_argv
    assert "-lcudart" in compile_argv
    assert str(BatchPaths.under(workspace).te_runner) == compile_argv[-1]
    assert TE_WHEEL_SHA256 == "1c8e6627cf02358201f513ab902c7b4f82f179214bd775edc97ce0f2001d82da"
    assert TE_SOURCE_COMMIT == "2e559f062497bef768dfbe9d7e45548fadeca80a"
    assert TE_SOURCE_TREE == "c722c6cd46cc59a0dbabadd846af3f45b15c1a95"


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


def test_batch_paths_keep_raw_reports_and_runtime_identity_separate(tmp_path: Path) -> None:
    paths = BatchPaths.under(tmp_path)

    assert (
        len(
            {
                paths.te_report,
                paths.xla_report,
                paths.acceptance,
                paths.runtime_manifest,
                paths.result,
            }
        )
        == 5
    )
    assert paths.te_library.name == "libtransformer_engine.so"
    assert paths.raw.name == "raw"
