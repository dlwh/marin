# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the exact, locally reviewable Iris request for the Track 1 prototype."""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path

from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES, content_id
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, gpu_device
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_to_proto
from iris.version import client_revision_date
from rigging.timing import Duration
from track1_map_h100_prototype import (
    BASE_IMAGE,
    SOURCE_MANIFEST_RELATIVE,
    Authorization,
    canonical_json,
    load_authorization,
    require_source_bundle,
)

JOB_NAME = JobName.root("dlwh", "shuttle-track1-map-h100-prototype").to_wire()
AUTHORIZATION_RELATIVE = Path("scripts/shuttle/track1_map_h100_prototype_authorization.json")
REQUEST_SCHEMA = "track1_map_h100_iris_launch_request"
WORKSPACE = Path("/tmp/shuttle-track1-map-h100-prototype")
RUNNER_REPOSITORY = Path("/app")
PYTHON = Path("/opt/h100-evidence-runtime/bin/python")
CUDA_ROOT = Path("/usr/local/cuda-13.2")
TIMEOUT = Duration.from_hours(4)
FILE_MODE = 0o644
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SHA256 = re.compile(r"[0-9a-f]{64}")


def _entrypoint_command(approved_authorization_sha256: str) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER_REPOSITORY / "lib/shuttle/mlir/jax_patch/track1_map_h100_launcher.py"),
        "--authorization-file",
        str(RUNNER_REPOSITORY / AUTHORIZATION_RELATIVE),
        "--approved-authorization-sha256",
        approved_authorization_sha256,
        "--workspace",
        str(WORKSPACE),
        "--runner-repository",
        str(RUNNER_REPOSITORY),
        "--python",
        str(PYTHON),
        "--cuda-root",
        str(CUDA_ROOT),
        "--base-image",
        BASE_IMAGE,
        "--execute",
    ]


def _bundle_members(
    bundle_root: Path,
    authorization_path: Path,
    authorization: Authorization,
) -> list[tuple[str, bytes]]:
    expected_authorization_path = bundle_root / AUTHORIZATION_RELATIVE
    if authorization_path != expected_authorization_path:
        raise ValueError(f"authorization must be read from {AUTHORIZATION_RELATIVE}")
    manifest = require_source_bundle(bundle_root, authorization)
    entries = manifest["entries"]
    assert type(entries) is list
    relatives = [Path(entry["path"]) for entry in entries]
    relatives.extend((AUTHORIZATION_RELATIVE, SOURCE_MANIFEST_RELATIVE))
    members = []
    for relative in sorted(relatives):
        path = bundle_root / relative
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RuntimeError(f"Iris bundle member must be one regular file: {relative}")
        members.append((relative.as_posix(), path.read_bytes()))
    if len({name for name, _ in members}) != len(members):
        raise RuntimeError("Iris bundle closure contains a duplicate path")
    return members


def _exact_bundle(
    bundle_root: Path,
    authorization_path: Path,
    authorization: Authorization,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in _bundle_members(bundle_root, authorization_path, authorization):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | FILE_MODE) << 16
            archive.writestr(info, payload)
    blob = stream.getvalue()
    if len(blob) > MAX_BUNDLE_SIZE_BYTES:
        raise RuntimeError("exact Iris bundle exceeds the controller size limit")
    return blob


def build_launch_request(
    *,
    bundle_root: Path,
    authorization_path: Path,
    approved_authorization_sha256: str,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Build the real Iris request without constructing a client or issuing an RPC."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("Iris bundle root must be one directory, not a symlink")
    authorization = load_authorization(authorization_path, approved_authorization_sha256)
    blob = _exact_bundle(bundle_root, authorization_path, authorization)
    environment = job_pb2.EnvironmentConfig(setup_scripts=[])
    entrypoint = build_runtime_entrypoint(
        Entrypoint.from_command(*_entrypoint_command(approved_authorization_sha256)),
        environment,
    )
    resources = ResourceSpec(
        cpu=32,
        memory="128GB",
        disk="500GB",
        device=gpu_device("H100", 1),
    ).to_proto()
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JOB_NAME,
        entrypoint=entrypoint,
        resources=resources,
        environment=environment,
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        fail_if_exists=True,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        task_image=BASE_IMAGE,
        bundle_init_image=BASE_IMAGE,
        client_revision_date=client_revision_date(),
    )
    request.timeout.CopyFrom(duration_to_proto(TIMEOUT))
    request.exact_bundle_upload.bundle_id = content_id(blob)
    request.exact_bundle_upload.blob = blob
    validate_launch_request(request, approved_authorization_sha256=approved_authorization_sha256)
    return request


def _require_no_unknown_fields(message: object, name: str) -> None:
    known = type(message)()
    known.CopyFrom(message)
    known.DiscardUnknownFields()
    if known.SerializeToString(deterministic=True) != message.SerializeToString(deterministic=True):
        raise ValueError(f"LaunchJobRequest {name} contains unknown fields")


def _validated_bundle(blob: bytes, approved_authorization_sha256: str) -> str:
    if len(blob) > MAX_BUNDLE_SIZE_BYTES:
        raise ValueError("LaunchJobRequest bundle exceeds the controller size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("LaunchJobRequest bundle has duplicate members")
            payloads = {}
            for info in infos:
                path = Path(info.filename)
                if (
                    info.is_dir()
                    or info.filename.startswith("/")
                    or "\\" in info.filename
                    or any(part in ("", ".", "..") for part in path.parts)
                    or path.as_posix() != info.filename
                    or info.date_time != ZIP_DATE
                    or info.create_system != 3
                    or info.compress_type != zipfile.ZIP_STORED
                    or (info.external_attr >> 16) != stat.S_IFREG | FILE_MODE
                ):
                    raise ValueError("LaunchJobRequest bundle member metadata is not canonical")
                payloads[info.filename] = archive.read(info)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("LaunchJobRequest bundle is not a ZIP archive") from error

    try:
        authorization_raw = payloads[AUTHORIZATION_RELATIVE.as_posix()]
        manifest_raw = payloads[SOURCE_MANIFEST_RELATIVE.as_posix()]
        authorization = json.loads(authorization_raw)
        manifest = json.loads(manifest_raw)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("LaunchJobRequest bundle lacks its authorization closure") from error
    if hashlib.sha256(authorization_raw).hexdigest() != approved_authorization_sha256:
        raise ValueError("LaunchJobRequest bundle authorization differs from external approval")
    if type(authorization) is not dict or authorization_raw != canonical_json(authorization):
        raise ValueError("LaunchJobRequest bundle authorization is not canonical")
    if type(manifest) is not dict or manifest_raw != canonical_json(manifest):
        raise ValueError("LaunchJobRequest bundle manifest is not canonical")
    integration = authorization.get("integration")
    if type(integration) is not dict or hashlib.sha256(manifest_raw).hexdigest() != integration.get(
        "source_manifest_sha256"
    ):
        raise ValueError("LaunchJobRequest bundle manifest differs from authorization")
    entries = manifest.get("entries")
    if type(entries) is not list:
        raise ValueError("LaunchJobRequest bundle manifest entries are invalid")
    expected_paths = {AUTHORIZATION_RELATIVE.as_posix(), SOURCE_MANIFEST_RELATIVE.as_posix()}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("LaunchJobRequest bundle manifest entry is invalid")
        path = entry["path"]
        if type(path) is not str or path in expected_paths or path not in payloads:
            raise ValueError("LaunchJobRequest bundle manifest path is invalid")
        payload = payloads[path]
        if len(payload) != entry["size"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("LaunchJobRequest bundle member differs from its manifest")
        expected_paths.add(path)
    if set(payloads) != expected_paths:
        raise ValueError("LaunchJobRequest bundle contains an unauthorized member")
    return hashlib.sha256(blob).hexdigest()


def validate_launch_request(
    request: controller_pb2.Controller.LaunchJobRequest,
    *,
    approved_authorization_sha256: str,
) -> None:
    """Validate every execution-affecting field in a decoded Iris request."""
    if SHA256.fullmatch(approved_authorization_sha256) is None:
        raise ValueError("LaunchJobRequest approval SHA-256 is invalid")
    _require_no_unknown_fields(request, "root")
    if request.name != JOB_NAME:
        raise ValueError("LaunchJobRequest job name differs")
    if request.task_image != BASE_IMAGE or request.bundle_init_image != BASE_IMAGE:
        raise ValueError("LaunchJobRequest image differs")
    expected_resources = ResourceSpec(
        cpu=32,
        memory="128GB",
        disk="500GB",
        device=gpu_device("H100", 1),
    ).to_proto()
    if request.resources != expected_resources:
        raise ValueError("LaunchJobRequest resources differ")
    if request.replicas != 1:
        raise ValueError("LaunchJobRequest replica count differs")
    if request.timeout != duration_to_proto(TIMEOUT) or request.HasField("scheduling_timeout"):
        raise ValueError("LaunchJobRequest timeout differs")
    if request.max_retries_failure or request.max_retries_preemption or request.max_task_failures:
        raise ValueError("LaunchJobRequest retry or failure budget differs")
    if not request.fail_if_exists or request.existing_job_policy != job_pb2.EXISTING_JOB_POLICY_ERROR:
        raise ValueError("LaunchJobRequest existing-job policy differs")
    if request.environment != job_pb2.EnvironmentConfig(setup_scripts=[]):
        raise ValueError("LaunchJobRequest environment differs")
    expected_entrypoint = build_runtime_entrypoint(
        Entrypoint.from_command(*_entrypoint_command(approved_authorization_sha256)),
        job_pb2.EnvironmentConfig(setup_scripts=[]),
    )
    if request.entrypoint != expected_entrypoint:
        raise ValueError("LaunchJobRequest entrypoint differs")
    if (
        request.bundle_id
        or request.bundle_blob
        or request.ports
        or request.constraints
        or request.HasField("coscheduling")
        or request.preemption_policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED
        or request.priority_band != job_pb2.PRIORITY_BAND_BATCH
        or request.container_profile != job_pb2.CONTAINER_PROFILE_UNSPECIFIED
        or request.submit_argv
        or request.HasField("federation")
    ):
        raise ValueError("LaunchJobRequest contains an unauthorized launch option")
    if re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", request.client_revision_date) is None:
        raise ValueError("LaunchJobRequest client revision date is invalid")
    if not request.HasField("exact_bundle_upload"):
        raise ValueError("LaunchJobRequest exact bundle is missing")
    _require_no_unknown_fields(request.exact_bundle_upload, "exact bundle")
    bundle_sha256 = _validated_bundle(request.exact_bundle_upload.blob, approved_authorization_sha256)
    if request.exact_bundle_upload.bundle_id != bundle_sha256:
        raise ValueError("LaunchJobRequest exact bundle content ID differs")


def decode_launch_request(
    raw: bytes,
    *,
    approved_authorization_sha256: str,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Parse canonical protobuf bytes and revalidate their complete launch scope."""
    request = controller_pb2.Controller.LaunchJobRequest()
    try:
        request.ParseFromString(raw)
    except ValueError as error:
        raise ValueError("LaunchJobRequest protobuf is malformed") from error
    if request.SerializeToString(deterministic=True) != raw:
        raise ValueError("LaunchJobRequest protobuf is not canonical")
    validate_launch_request(request, approved_authorization_sha256=approved_authorization_sha256)
    return request


def _request_summary(
    raw: bytes,
    *,
    approved_authorization_sha256: str,
) -> dict[str, object]:
    request = decode_launch_request(raw, approved_authorization_sha256=approved_authorization_sha256)
    return {
        "authorization_sha256": approved_authorization_sha256,
        "bundle_sha256": request.exact_bundle_upload.bundle_id,
        "bundle_size": len(request.exact_bundle_upload.blob),
        "client_revision_date": request.client_revision_date,
        "image": request.task_image,
        "job_name": request.name,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_size": len(raw),
        "schema": REQUEST_SCHEMA,
        "schema_version": 1,
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"request output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--approved-authorization-sha256")
    parser.add_argument("--request-output", type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.dry_run:
        raise RuntimeError("Track 1 Iris submission is local dry-run only")
    required = {
        "bundle root": arguments.bundle_root,
        "authorization file": arguments.authorization_file,
        "approved authorization SHA-256": arguments.approved_authorization_sha256,
        "request output": arguments.request_output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"Iris dry-run is incomplete: {', '.join(missing)}")
    request = build_launch_request(
        bundle_root=arguments.bundle_root,
        authorization_path=arguments.authorization_file,
        approved_authorization_sha256=arguments.approved_authorization_sha256,
    )
    raw = request.SerializeToString(deterministic=True)
    summary = _request_summary(raw, approved_authorization_sha256=arguments.approved_authorization_sha256)
    _write_create_only(arguments.request_output, raw)
    print(canonical_json(summary).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
