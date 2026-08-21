# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the sealed Track 2 TE/XLA Iris request."""

import hashlib
import io
import json
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from iris.rpc import controller_pb2, job_pb2
from track2_te_h100_batch import BASE_IMAGE, SOURCE_MANIFEST_NAME
from track2_te_h100_iris import (
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


def test_render_emits_one_exact_batch_priority_zero_retry_h100_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "request.pb"

    assert main(["render", "--bundle-root", str(MARIN_REPOSITORY), "--run", "1", "--output", str(output)]) == 0

    raw = output.read_bytes()
    request = decode_launch_request(raw, run=1)
    summary = json.loads(capsys.readouterr().out)
    assert summary["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert request.name == "/dlwh/shuttle-track2-te-h100-r1"
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
    assert list(request.entrypoint.run_command.argv) == [
        "/opt/h100-evidence-runtime/bin/python",
        "/app/lib/shuttle/mlir/jax_patch/track2_te_h100_batch.py",
        "--workspace",
        "/tmp/shuttle-track2-te-h100",
        "--runner-repository",
        "/app",
        "--source-manifest",
        "/app/.shuttle-track2-te-source-manifest.json",
        "--base-image",
        (
            "ghcr.io/marin-community/iris-task-h100-evidence:"
            "5238c39e7a506b919fc803046de4a9dc2c29d02f"
            "@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef"
        ),
        "--execute",
    ]
    assert request.bundle_id == hashlib.sha256(request.bundle_blob).hexdigest()
    with zipfile.ZipFile(io.BytesIO(request.bundle_blob)) as archive:
        manifest = json.loads(archive.read(SOURCE_MANIFEST_NAME))
        assert (
            manifest["commit"]
            == subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=MARIN_REPOSITORY, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        assert (
            manifest["tree"]
            == subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=MARIN_REPOSITORY, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        members = {entry["path"] for entry in manifest["entries"]}
        assert members == set(archive.namelist()) - {SOURCE_MANIFEST_NAME}
        assert {
            "lib/shuttle/mlir/jax_patch/track2_te_h100_batch.py",
            "lib/shuttle/mlir/jax_patch/abi22_te_rmsnorm_oracle_v2.py",
            "lib/shuttle/mlir/jax_patch/abi22_xla_rmsnorm_oracle_v2.py",
            "lib/shuttle/mlir/jax_patch/abi22_te_rmsnorm_oracle_runner_v2.cpp",
            "lib/shuttle/mlir/jax_patch/abi22-te-rmsnorm-oracle-runner-build-v2.json",
        }.issubset(members)


@pytest.mark.parametrize(
    "mutation",
    [
        "bundle",
        "bundle_id",
        "name",
        "gpu",
        "gpu_variant",
        "cpu",
        "memory",
        "disk",
        "image",
        "bundle_init_image",
        "replicas",
        "timeout",
        "preemption",
        "failure",
        "task_failure",
        "fail_if_exists",
        "existing_policy",
        "environment",
        "entrypoint",
        "priority",
        "ports",
        "client_revision",
        "constraints",
        "scheduling_timeout",
        "coscheduling",
        "preemption_policy",
        "container_profile",
        "submit_argv",
        "federation",
        "exact_bundle_upload",
    ],
)
def test_request_rejects_execution_scope_mutations(mutation: str) -> None:
    request = deepcopy(_request())
    if mutation == "bundle":
        request.bundle_blob += b"drift"
    elif mutation == "bundle_id":
        request.bundle_id = "0" * 64
    elif mutation == "name":
        request.name = "/dlwh/other"
    elif mutation == "gpu":
        request.resources.device.gpu.count = 2
    elif mutation == "gpu_variant":
        request.resources.device.gpu.variant = "B200"
    elif mutation == "cpu":
        request.resources.cpu = 64
    elif mutation == "memory":
        request.resources.memory = "256GB"
    elif mutation == "disk":
        request.resources.disk = "1TB"
    elif mutation == "image":
        request.task_image = "ghcr.io/marin-community/iris-task-h100-evidence:latest"
    elif mutation == "bundle_init_image":
        request.bundle_init_image = "example.invalid/init@sha256:" + "0" * 64
    elif mutation == "replicas":
        request.replicas = 2
    elif mutation == "timeout":
        request.timeout.seconds += 1
    elif mutation == "preemption":
        request.max_retries_preemption = 1
    elif mutation == "failure":
        request.max_retries_failure = 1
    elif mutation == "task_failure":
        request.max_task_failures = 1
    elif mutation == "fail_if_exists":
        request.fail_if_exists = False
    elif mutation == "existing_policy":
        request.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED
    elif mutation == "environment":
        request.environment.setup_scripts.append("echo drift")
    elif mutation == "entrypoint":
        request.entrypoint.run_command.argv[-1] = "--plan"
    elif mutation == "priority":
        request.priority_band = job_pb2.PRIORITY_BAND_PRODUCTION
    elif mutation == "ports":
        request.ports.append("1234")
    elif mutation == "client_revision":
        request.client_revision_date = "1970-01-01"
    elif mutation == "constraints":
        request.constraints.add().SetInParent()
    elif mutation == "scheduling_timeout":
        request.scheduling_timeout.seconds = 1
    elif mutation == "coscheduling":
        request.coscheduling.SetInParent()
    elif mutation == "preemption_policy":
        request.preemption_policy = 1
    elif mutation == "container_profile":
        request.container_profile = 1
    elif mutation == "submit_argv":
        request.submit_argv.append("drift")
    elif mutation == "federation":
        request.federation.SetInParent()
    elif mutation == "exact_bundle_upload":
        request.exact_bundle_upload.SetInParent()

    with pytest.raises(ValueError, match="LaunchJobRequest"):
        validate_launch_request(request, run=1)


def _submit_arguments(request_path: Path, raw: bytes, receipt: Path) -> list[str]:
    return [
        "submit",
        "--request",
        str(request_path),
        "--expected-request-sha256",
        hashlib.sha256(raw).hexdigest(),
        "--run",
        "1",
        "--config",
        "lib/iris/config/cw-rno2a.yaml",
        "--receipt",
        str(receipt),
    ]


def test_submit_rejects_unreviewed_request_digest_before_receipt_or_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "reviewed.pb"
    raw = _request().SerializeToString(deterministic=True)
    request_path.write_bytes(raw)
    receipt = tmp_path / "receipt.json"
    submitted = False

    def fake_submit(*args: object, **kwargs: object) -> str:
        nonlocal submitted
        submitted = True
        return "/dlwh/shuttle-track2-te-h100-r1"

    monkeypatch.setattr("track2_te_h100_iris._submit", fake_submit)
    arguments = _submit_arguments(request_path, raw, receipt)
    arguments[arguments.index("--expected-request-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="reviewed request SHA-256"):
        main(arguments)
    assert not submitted
    assert not receipt.exists()


def test_submit_reserves_pending_receipt_before_rpc_and_finalizes_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "reviewed.pb"
    raw = _request().SerializeToString(deterministic=True)
    request_path.write_bytes(raw)
    receipt = tmp_path / "receipt.json"

    def fake_submit(*args: object, **kwargs: object) -> str:
        pending = json.loads(receipt.read_bytes())
        assert pending["status"] == "submission_pending"
        assert pending["request"]["request_sha256"] == hashlib.sha256(raw).hexdigest()
        return "/dlwh/shuttle-track2-te-h100-r1"

    monkeypatch.setattr("track2_te_h100_iris._submit", fake_submit)
    assert main(_submit_arguments(request_path, raw, receipt)) == 0
    final = json.loads(receipt.read_bytes())
    assert final["status"] == "submitted"
    assert final["job_id"] == "/dlwh/shuttle-track2-te-h100-r1"


@pytest.mark.parametrize("failure", ["rpc", "wrong_job"])
def test_submit_preserves_pending_receipt_on_uncertain_rpc_outcome(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "reviewed.pb"
    raw = _request().SerializeToString(deterministic=True)
    request_path.write_bytes(raw)
    receipt = tmp_path / "receipt.json"

    def fake_submit(*args: object, **kwargs: object) -> str:
        assert json.loads(receipt.read_bytes())["status"] == "submission_pending"
        if failure == "rpc":
            raise RuntimeError("transport failed after submission may have committed")
        return "/dlwh/unexpected"

    monkeypatch.setattr("track2_te_h100_iris._submit", fake_submit)
    with pytest.raises(RuntimeError):
        main(_submit_arguments(request_path, raw, receipt))
    assert json.loads(receipt.read_bytes())["status"] == "submission_pending"


def test_submit_refuses_existing_receipt_before_rpc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = tmp_path / "reviewed.pb"
    raw = _request().SerializeToString(deterministic=True)
    request_path.write_bytes(raw)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"keep")
    submitted = False

    def fake_submit(*args: object, **kwargs: object) -> str:
        nonlocal submitted
        submitted = True
        return "/dlwh/shuttle-track2-te-h100-r1"

    monkeypatch.setattr("track2_te_h100_iris._submit", fake_submit)
    with pytest.raises(FileExistsError, match="receipt"):
        main(_submit_arguments(request_path, raw, receipt))
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
            "user.name=Track 2 TE test",
            "-c",
            "user.email=track2-te@example.invalid",
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
    assert job_name(7) == "/dlwh/shuttle-track2-te-h100-r7"
    with pytest.raises(ValueError, match="positive integer"):
        job_name(0)
