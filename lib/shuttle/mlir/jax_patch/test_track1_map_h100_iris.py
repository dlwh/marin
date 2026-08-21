# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from iris.rpc import controller_pb2, job_pb2
from track1_map_h100_iris import (
    CLIENT_REVISION_DATE,
    Campaign,
    build_launch_request,
    decode_launch_request,
    job_name,
    main,
    validate_launch_request,
)
from track1_map_h100_prototype import BASE_IMAGE

MARIN_REPOSITORY = Path(__file__).parents[4]
AUTHORIZATION_PATH = MARIN_REPOSITORY / "scripts/shuttle/track1_map_h100_prototype_authorization.json"
SOURCE_MANIFEST_PATH = MARIN_REPOSITORY / "scripts/shuttle/track1_map_h100_source_manifest.json"


def _approved_authorization_sha256() -> str:
    return hashlib.sha256(AUTHORIZATION_PATH.read_bytes()).hexdigest()


def _request(retry: int = 1) -> controller_pb2.Controller.LaunchJobRequest:
    return build_launch_request(
        bundle_root=MARIN_REPOSITORY,
        authorization_path=AUTHORIZATION_PATH,
        approved_authorization_sha256=_approved_authorization_sha256(),
        campaign=Campaign.RMSNORM_CANARY,
        retry=retry,
    )


def test_dry_run_emits_and_decodes_the_real_fail_closed_launch_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "launch.pb"
    approved = _approved_authorization_sha256()

    assert (
        main(
            [
                "--dry-run",
                "--bundle-root",
                str(MARIN_REPOSITORY),
                "--authorization-file",
                str(AUTHORIZATION_PATH),
                "--approved-authorization-sha256",
                approved,
                "--campaign",
                Campaign.RMSNORM_CANARY.value,
                "--retry",
                "4",
                "--request-output",
                str(output),
            ]
        )
        == 0
    )

    raw = output.read_bytes()
    request = controller_pb2.Controller.LaunchJobRequest()
    request.ParseFromString(raw)
    decoded = decode_launch_request(
        raw, approved_authorization_sha256=approved, campaign=Campaign.RMSNORM_CANARY, retry=4
    )
    summary = json.loads(capsys.readouterr().out)

    assert request == decoded
    assert summary["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["bundle_sha256"] == request.bundle_id
    assert summary["authorization_sha256"] == approved
    assert summary["campaign"] == Campaign.RMSNORM_CANARY.value
    assert summary["job_name"] == job_name(Campaign.RMSNORM_CANARY, 4)
    assert summary["retry"] == 4

    assert request.name == "/dlwh/shuttle-track1-rmsnorm-h100-canary-r4"
    assert request.client_revision_date == CLIENT_REVISION_DATE
    assert request.task_image == BASE_IMAGE
    assert request.bundle_init_image == ""
    assert request.resources.cpu_millicores == 32_000
    assert request.resources.memory_bytes == 128 * 1024**3
    assert request.resources.disk_bytes == 500 * 1024**3
    assert request.resources.device.gpu.variant == "H100"
    assert request.resources.device.gpu.count == 1
    assert request.replicas == 1
    assert request.timeout.milliseconds == 4 * 60 * 60 * 1000
    assert request.max_retries_failure == 0
    assert request.max_retries_preemption == 0
    assert request.max_task_failures == 0
    assert request.fail_if_exists
    assert request.existing_job_policy == job_pb2.EXISTING_JOB_POLICY_ERROR
    assert request.priority_band == job_pb2.PRIORITY_BAND_BATCH
    assert list(request.environment.setup_scripts) == []
    assert dict(request.environment.env_vars) == {}
    assert list(request.entrypoint.setup_commands) == []
    command = list(request.entrypoint.run_command.argv)
    assert command[0] == "/opt/h100-evidence-runtime/bin/python"
    assert command[command.index("--python") + 1] == "/opt/h100-evidence-runtime/bin/python"
    assert command.count(approved) == 1
    assert command[-1] == "--execute"
    assert request.bundle_id == hashlib.sha256(request.bundle_blob).hexdigest()
    assert request.bundle_blob
    assert not request.HasField("exact_bundle_upload")

    manifest = json.loads(SOURCE_MANIFEST_PATH.read_bytes())
    expected_paths = {entry["path"] for entry in manifest["entries"]}
    expected_paths.update(
        {
            "scripts/shuttle/track1_map_h100_prototype_authorization.json",
            "scripts/shuttle/track1_map_h100_source_manifest.json",
        }
    )
    with zipfile.ZipFile(io.BytesIO(request.bundle_blob)) as archive:
        assert set(archive.namelist()) == expected_paths
        assert archive.read("scripts/shuttle/track1_map_h100_prototype_authorization.json") == (
            AUTHORIZATION_PATH.read_bytes()
        )


def test_request_bytes_are_deterministic_and_output_is_create_only(tmp_path: Path) -> None:
    approved = _approved_authorization_sha256()
    first = _request().SerializeToString(deterministic=True)
    second = _request().SerializeToString(deterministic=True)
    assert first == second
    assert _request(4).name == "/dlwh/shuttle-track1-rmsnorm-h100-canary-r4"
    assert _request(4).SerializeToString(deterministic=True) != first
    with pytest.raises(ValueError, match="protobuf is malformed"):
        decode_launch_request(b"\xff", approved_authorization_sha256=approved, campaign=Campaign.RMSNORM_CANARY, retry=1)
    with pytest.raises(ValueError, match="positive integer"):
        job_name(Campaign.RMSNORM_CANARY, 0)

    output = tmp_path / "launch.pb"
    output.write_bytes(b"do not replace")
    with pytest.raises(FileExistsError):
        main(
            [
                "--dry-run",
                "--bundle-root",
                str(MARIN_REPOSITORY),
                "--authorization-file",
                str(AUTHORIZATION_PATH),
                "--approved-authorization-sha256",
                approved,
                "--campaign",
                Campaign.RMSNORM_CANARY.value,
                "--retry",
                "1",
                "--request-output",
                str(output),
            ]
        )
    assert output.read_bytes() == b"do not replace"


def test_default_is_unsealed_before_bundle_or_request_output(tmp_path: Path) -> None:
    output = tmp_path / "launch.pb"
    with pytest.raises(RuntimeError, match="local dry-run only"):
        main(["--request-output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "authorization",
        "bundle_hash",
        "bundle_image",
        "bundle_integration_field",
        "bundle_zip",
        "client_revision",
        "cpu",
        "disk",
        "environment",
        "existing",
        "gpu",
        "image",
        "memory",
        "name",
        "priority",
        "replicas",
        "retry_failure",
        "retry_preemption",
        "setup",
        "task_failures",
        "timeout",
    ],
)
def test_decoded_request_rejects_every_launch_scope_mutation(mutation: str) -> None:
    approved = _approved_authorization_sha256()
    request = _request()
    changed = deepcopy(request)
    command = changed.entrypoint.run_command.argv
    if mutation == "authorization":
        command[list(command).index(approved)] = "f" * 64
    elif mutation == "bundle_hash":
        changed.bundle_id = "f" * 64
    elif mutation == "bundle_image":
        changed.bundle_init_image = BASE_IMAGE
    elif mutation == "bundle_integration_field":
        changed.exact_bundle_upload.bundle_id = changed.bundle_id
        changed.exact_bundle_upload.blob = changed.bundle_blob
    elif mutation == "bundle_zip":
        stream = io.BytesIO(changed.bundle_blob)
        with zipfile.ZipFile(stream, "a") as archive:
            archive.comment = b"alternate ZIP bytes"
        changed.bundle_blob = stream.getvalue()
        changed.bundle_id = hashlib.sha256(stream.getvalue()).hexdigest()
    elif mutation == "client_revision":
        changed.client_revision_date = "2099-12-31"
    elif mutation == "cpu":
        changed.resources.cpu_millicores = 31_000
    elif mutation == "disk":
        changed.resources.disk_bytes = 499 * 1024**3
    elif mutation == "environment":
        changed.environment.env_vars["UNREVIEWED"] = "1"
    elif mutation == "existing":
        changed.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_RECREATE
    elif mutation == "gpu":
        changed.resources.device.gpu.count = 2
    elif mutation == "image":
        changed.task_image = "ghcr.io/marin-community/iris-task-h100-evidence:latest"
    elif mutation == "memory":
        changed.resources.memory_bytes = 127 * 1024**3
    elif mutation == "name":
        changed.name = job_name(Campaign.RMSNORM_CANARY, 2)
    elif mutation == "priority":
        changed.priority_band = job_pb2.PRIORITY_BAND_INTERACTIVE
    elif mutation == "replicas":
        changed.replicas = 2
    elif mutation == "retry_failure":
        changed.max_retries_failure = 1
    elif mutation == "retry_preemption":
        changed.max_retries_preemption = 1
    elif mutation == "setup":
        changed.environment.setup_scripts.append("curl example.invalid")
    elif mutation == "task_failures":
        changed.max_task_failures = 1
    elif mutation == "timeout":
        changed.timeout.milliseconds += 1

    with pytest.raises(ValueError, match="LaunchJobRequest"):
        validate_launch_request(
            changed,
            approved_authorization_sha256=approved,
            campaign=Campaign.RMSNORM_CANARY,
            retry=1,
        )
