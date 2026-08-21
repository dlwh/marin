# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from iris.rpc import controller_pb2, job_pb2
from track2_h100_batch import BASE_IMAGE
from track2_h100_iris import (
    SOURCE_MANIFEST_NAME,
    _require_clean_source_closure,
    build_launch_request,
    decode_launch_request,
    job_name,
    main,
    validate_launch_request,
)

MARIN_REPOSITORY = Path(__file__).parents[4]


def _request() -> controller_pb2.Controller.LaunchJobRequest:
    return build_launch_request(bundle_root=MARIN_REPOSITORY, run=1)


def test_dry_run_emits_one_exact_zero_retry_h100_request(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "request.pb"

    assert main(["render", "--bundle-root", str(MARIN_REPOSITORY), "--run", "1", "--output", str(output)]) == 0

    raw = output.read_bytes()
    request = decode_launch_request(raw, run=1)
    summary = json.loads(capsys.readouterr().out)
    assert summary["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["bundle_sha256"] == request.bundle_id
    assert summary["source_commit"]
    assert summary["source_tree"]
    assert request.name == "/dlwh/shuttle-track2-h100-r1"
    assert request.task_image == BASE_IMAGE
    assert request.resources.device.gpu.variant == "H100"
    assert request.resources.device.gpu.count == 1
    assert request.replicas == 1
    assert request.max_retries_failure == 0
    assert request.max_retries_preemption == 0
    assert request.max_task_failures == 0
    assert request.fail_if_exists
    assert request.existing_job_policy == job_pb2.EXISTING_JOB_POLICY_ERROR
    assert request.priority_band == job_pb2.PRIORITY_BAND_BATCH
    assert list(request.environment.setup_scripts) == []
    assert request.entrypoint.run_command.argv[-1] == "--execute"
    assert request.bundle_id == hashlib.sha256(request.bundle_blob).hexdigest()

    with zipfile.ZipFile(io.BytesIO(request.bundle_blob)) as archive:
        manifest = json.loads(archive.read(SOURCE_MANIFEST_NAME))
        assert manifest["commit"] == summary["source_commit"]
        assert manifest["tree"] == summary["source_tree"]
        assert {entry["path"] for entry in manifest["entries"]} == set(archive.namelist()) - {SOURCE_MANIFEST_NAME}


@pytest.mark.parametrize("mutation", ["bundle", "gpu", "image", "preemption", "failure", "task_failure", "entrypoint"])
def test_request_rejects_execution_scope_mutations(mutation: str) -> None:
    request = deepcopy(_request())
    if mutation == "bundle":
        request.bundle_blob += b"drift"
    elif mutation == "gpu":
        request.resources.device.gpu.count = 2
    elif mutation == "image":
        request.task_image = "ghcr.io/marin-community/iris-task-h100-evidence:latest"
    elif mutation == "preemption":
        request.max_retries_preemption = 1
    elif mutation == "failure":
        request.max_retries_failure = 1
    elif mutation == "task_failure":
        request.max_task_failures = 1
    elif mutation == "entrypoint":
        request.entrypoint.run_command.argv[-1] = "--plan"

    with pytest.raises(ValueError, match="LaunchJobRequest"):
        validate_launch_request(request, run=1)


def test_render_is_create_only_and_submit_requires_reviewed_request_hash(tmp_path: Path) -> None:
    output = tmp_path / "request.pb"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        main(["render", "--bundle-root", str(MARIN_REPOSITORY), "--run", "1", "--output", str(output)])
    assert output.read_bytes() == b"keep"

    request = tmp_path / "reviewed.pb"
    request.write_bytes(_request().SerializeToString(deterministic=True))
    with pytest.raises(ValueError, match="reviewed request SHA-256"):
        main(
            [
                "submit",
                "--request",
                str(request),
                "--expected-request-sha256",
                "0" * 64,
                "--run",
                "1",
                "--config",
                "lib/iris/config/cw-us-east-02a.yaml",
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
    assert not (tmp_path / "receipt.json").exists()


def test_submit_reserves_receipt_before_external_rpc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = tmp_path / "reviewed.pb"
    raw = _request().SerializeToString(deterministic=True)
    request.write_bytes(raw)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"keep")
    submitted = False

    def fake_submit(*args: object, **kwargs: object) -> str:
        nonlocal submitted
        submitted = True
        return "/dlwh/shuttle-track2-h100-r1"

    monkeypatch.setattr("track2_h100_iris._submit", fake_submit)
    with pytest.raises(FileExistsError, match="receipt"):
        main(
            [
                "submit",
                "--request",
                str(request),
                "--expected-request-sha256",
                hashlib.sha256(raw).hexdigest(),
                "--run",
                "1",
                "--config",
                "lib/iris/config/cw-us-east-02a.yaml",
                "--receipt",
                str(receipt),
            ]
        )
    assert not submitted
    assert receipt.read_bytes() == b"keep"


def test_source_closure_rejects_dirty_tracked_and_untracked_bytes(tmp_path: Path) -> None:
    source = tmp_path / "lib/shuttle/mlir/source.cc"
    source.parent.mkdir(parents=True)
    source.write_text("clean\n")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", str(source.relative_to(tmp_path))], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Track 2 test",
            "-c",
            "user.email=track2@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    _require_clean_source_closure(tmp_path)

    source.write_text("dirty\n")
    with pytest.raises(RuntimeError, match="source closure is dirty"):
        _require_clean_source_closure(tmp_path)
    source.write_text("clean\n")
    (source.parent / "untracked.cc").write_text("untracked\n")
    with pytest.raises(RuntimeError, match="source closure is dirty"):
        _require_clean_source_closure(tmp_path)


def test_job_name_rejects_unbounded_run_ordinals() -> None:
    assert job_name(7) == "/dlwh/shuttle-track2-h100-r7"
    with pytest.raises(ValueError, match="positive integer"):
        job_name(0)
