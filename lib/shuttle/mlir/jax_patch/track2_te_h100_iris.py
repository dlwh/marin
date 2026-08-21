# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render and submit the sealed one-H100 Transformer Engine/XLA request."""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

from google.protobuf.message import DecodeError
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES, content_id
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, gpu_device
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_to_proto
from iris.version import client_revision_date
from rigging.timing import Duration
from track2_te_h100_batch import (
    BASE_IMAGE,
    SOURCE_MANIFEST_NAME,
    SOURCE_MANIFEST_SCHEMA,
    canonical_json,
)

REQUEST_SCHEMA = "shuttle_track2_te_h100_iris_request_v1"
REQUEST_SCHEMA_V3 = "shuttle_track2_te_h100_iris_request_v3"
REQUEST_SCHEMA_V4 = "shuttle_track2_te_h100_iris_request_v4"
RUNNER_REPOSITORY = Path("/app")
WORKSPACE = Path("/tmp/shuttle-track2-te-h100")
SOURCE_MANIFEST_PATH = RUNNER_REPOSITORY / SOURCE_MANIFEST_NAME
RUNTIME_PYTHON = Path("/opt/h100-evidence-runtime/bin/python")
TIMEOUT = Duration.from_hours(4)
CLIENT_REVISION_DATE = client_revision_date()
FILE_MODE = 0o644
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_RUN = 9999
LAUNCH_TIMEOUT_MS = 180_000


def job_name(run: int) -> str:
    """Return the deterministic job identity for one reviewed request."""
    if type(run) is not int or run < 1 or run > MAX_RUN:
        raise ValueError(f"run must be a positive integer no greater than {MAX_RUN}")
    return JobName.root("dlwh", f"shuttle-track2-te-h100-r{run}").to_wire()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _require_clean_source_closure(bundle_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", "lib/shuttle/mlir"],
        cwd=bundle_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Track 2 TE source closure is dirty")


def _source_paths(bundle_root: Path) -> list[Path]:
    _require_clean_source_closure(bundle_root)
    output = subprocess.run(
        ["git", "ls-files", "--cached", "-z", "--", "lib/shuttle/mlir"],
        cwd=bundle_root,
        check=True,
        capture_output=True,
    ).stdout
    relatives = sorted(Path(os.fsdecode(value)) for value in output.split(b"\0") if value)
    required = {
        Path("lib/shuttle/mlir/jax_patch/track2_te_h100_batch.py"),
        Path("lib/shuttle/mlir/jax_patch/abi22_te_rmsnorm_oracle_v2.py"),
        Path("lib/shuttle/mlir/jax_patch/abi22_xla_rmsnorm_oracle_v2.py"),
        Path("lib/shuttle/mlir/jax_patch/abi22_te_rmsnorm_oracle_runner_v2.cpp"),
        Path("lib/shuttle/mlir/jax_patch/abi22-te-rmsnorm-oracle-runner-build-v2.json"),
    }
    if not relatives or not required.issubset(relatives):
        raise RuntimeError("Track 2 TE source closure lacks a required runner or protocol file")
    return relatives


def _source_manifest(bundle_root: Path, relatives: list[Path]) -> bytes:
    entries = []
    for relative in relatives:
        path = bundle_root / relative
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RuntimeError(f"Track 2 TE bundle member must be one regular file: {relative}")
        payload = path.read_bytes()
        entries.append(
            {"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        )
    return canonical_json(
        {
            "commit": _git(bundle_root, "rev-parse", "HEAD"),
            "entries": entries,
            "schema": SOURCE_MANIFEST_SCHEMA,
            "tree": _git(bundle_root, "rev-parse", "HEAD^{tree}"),
        }
    )


def _zip_bytes(members: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(members):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | FILE_MODE) << 16
            archive.writestr(info, payload)
    return stream.getvalue()


def _exact_bundle(bundle_root: Path) -> bytes:
    relatives = _source_paths(bundle_root)
    members = [(relative.as_posix(), (bundle_root / relative).read_bytes()) for relative in relatives]
    members.append((SOURCE_MANIFEST_NAME, _source_manifest(bundle_root, relatives)))
    blob = _zip_bytes(members)
    if len(blob) > MAX_BUNDLE_SIZE_BYTES:
        raise RuntimeError("Track 2 TE Iris bundle exceeds the controller size limit")
    return blob


def _entrypoint_command() -> list[str]:
    return [
        str(RUNTIME_PYTHON),
        str(RUNNER_REPOSITORY / "lib/shuttle/mlir/jax_patch/track2_te_h100_batch.py"),
        "--workspace",
        str(WORKSPACE),
        "--runner-repository",
        str(RUNNER_REPOSITORY),
        "--source-manifest",
        str(SOURCE_MANIFEST_PATH),
        "--base-image",
        BASE_IMAGE,
        "--execute",
    ]


def _entrypoint_command_v3() -> list[str]:
    command = _entrypoint_command()
    command[-1] = "--execute-v3"
    return command


def _entrypoint_command_v4() -> list[str]:
    command = _entrypoint_command()
    command[-1] = "--execute-v4"
    return command


def build_launch_request(*, bundle_root: Path, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Build the batch-priority, zero-retry request without connecting."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("Iris bundle root must be one directory, not a symlink")
    blob = _exact_bundle(bundle_root)
    environment = job_pb2.EnvironmentConfig(setup_scripts=[])
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name(run),
        entrypoint=build_runtime_entrypoint(Entrypoint.from_command(*_entrypoint_command()), environment),
        resources=ResourceSpec(cpu=32, memory="128GB", disk="500GB", device=gpu_device("H100", 1)).to_proto(),
        environment=environment,
        bundle_id=content_id(blob),
        bundle_blob=blob,
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        fail_if_exists=True,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        task_image=BASE_IMAGE,
        client_revision_date=CLIENT_REVISION_DATE,
    )
    request.timeout.CopyFrom(duration_to_proto(TIMEOUT))
    validate_launch_request(request, run=run)
    return request


def build_launch_request_v3(*, bundle_root: Path, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Build the versioned v3 batch-priority request without connecting."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("Iris bundle root must be one directory, not a symlink")
    blob = _exact_bundle(bundle_root)
    environment = job_pb2.EnvironmentConfig(setup_scripts=[])
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name(run),
        entrypoint=build_runtime_entrypoint(Entrypoint.from_command(*_entrypoint_command_v3()), environment),
        resources=ResourceSpec(cpu=32, memory="128GB", disk="500GB", device=gpu_device("H100", 1)).to_proto(),
        environment=environment,
        bundle_id=content_id(blob),
        bundle_blob=blob,
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        fail_if_exists=True,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        task_image=BASE_IMAGE,
        client_revision_date=CLIENT_REVISION_DATE,
    )
    request.timeout.CopyFrom(duration_to_proto(TIMEOUT))
    validate_launch_request_v3(request, run=run)
    return request


def build_launch_request_v4(*, bundle_root: Path, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Build the versioned v4 batch-priority request without connecting."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("Iris bundle root must be one directory, not a symlink")
    blob = _exact_bundle(bundle_root)
    environment = job_pb2.EnvironmentConfig(setup_scripts=[])
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name(run),
        entrypoint=build_runtime_entrypoint(Entrypoint.from_command(*_entrypoint_command_v4()), environment),
        resources=ResourceSpec(cpu=32, memory="128GB", disk="500GB", device=gpu_device("H100", 1)).to_proto(),
        environment=environment,
        bundle_id=content_id(blob),
        bundle_blob=blob,
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        fail_if_exists=True,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        task_image=BASE_IMAGE,
        client_revision_date=CLIENT_REVISION_DATE,
    )
    request.timeout.CopyFrom(duration_to_proto(TIMEOUT))
    validate_launch_request_v4(request, run=run)
    return request


def _validated_bundle(blob: bytes) -> dict[str, object]:
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
        manifest_raw = payloads[SOURCE_MANIFEST_NAME]
        manifest = json.loads(manifest_raw)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("LaunchJobRequest bundle lacks its source manifest") from error
    if type(manifest) is not dict or manifest_raw != canonical_json(manifest):
        raise ValueError("LaunchJobRequest source manifest is not canonical")
    if set(manifest) != {"commit", "entries", "schema", "tree"}:
        raise ValueError("LaunchJobRequest source manifest fields differ")
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("LaunchJobRequest source manifest schema differs")
    expected = {SOURCE_MANIFEST_NAME}
    for entry in manifest.get("entries", []):
        if type(entry) is not dict or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("LaunchJobRequest source manifest entry is invalid")
        name = entry["path"]
        if type(name) is not str or name in expected or name not in payloads:
            raise ValueError("LaunchJobRequest source manifest path is invalid")
        payload = payloads[name]
        if len(payload) != entry["size"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("LaunchJobRequest source bundle differs from its manifest")
        expected.add(name)
    if set(payloads) != expected or blob != _zip_bytes(list(payloads.items())):
        raise ValueError("LaunchJobRequest bundle closure or ZIP bytes differ")
    return manifest


def _validate_launch_request(
    request: controller_pb2.Controller.LaunchJobRequest,
    *,
    run: int,
    entrypoint_command: list[str],
) -> None:
    known = type(request)()
    known.CopyFrom(request)
    known.DiscardUnknownFields()
    if known.SerializeToString(deterministic=True) != request.SerializeToString(deterministic=True):
        raise ValueError("LaunchJobRequest contains unknown fields")
    if request.name != job_name(run):
        raise ValueError("LaunchJobRequest name differs")
    if request.task_image != BASE_IMAGE or request.bundle_init_image:
        raise ValueError("LaunchJobRequest image differs")
    expected_resources = ResourceSpec(cpu=32, memory="128GB", disk="500GB", device=gpu_device("H100", 1)).to_proto()
    if request.resources != expected_resources:
        raise ValueError("LaunchJobRequest resources differ")
    if request.replicas != 1 or request.timeout != duration_to_proto(TIMEOUT):
        raise ValueError("LaunchJobRequest replica count or timeout differs")
    if request.max_retries_failure or request.max_retries_preemption or request.max_task_failures:
        raise ValueError("LaunchJobRequest retry or failure budget differs")
    if not request.fail_if_exists or request.existing_job_policy != job_pb2.EXISTING_JOB_POLICY_ERROR:
        raise ValueError("LaunchJobRequest existing-job policy differs")
    environment = job_pb2.EnvironmentConfig(setup_scripts=[])
    if request.environment != environment:
        raise ValueError("LaunchJobRequest environment differs")
    expected_entrypoint = build_runtime_entrypoint(Entrypoint.from_command(*entrypoint_command), environment)
    if request.entrypoint != expected_entrypoint:
        raise ValueError("LaunchJobRequest entrypoint differs")
    if (
        request.ports
        or request.constraints
        or request.HasField("scheduling_timeout")
        or request.HasField("coscheduling")
        or request.preemption_policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED
        or request.priority_band != job_pb2.PRIORITY_BAND_BATCH
        or request.container_profile != job_pb2.CONTAINER_PROFILE_UNSPECIFIED
        or request.submit_argv
        or request.HasField("federation")
        or request.HasField("exact_bundle_upload")
        or request.client_revision_date != CLIENT_REVISION_DATE
    ):
        raise ValueError("LaunchJobRequest contains an unauthorized launch option")
    _validated_bundle(request.bundle_blob)
    if request.bundle_id != content_id(request.bundle_blob):
        raise ValueError("LaunchJobRequest bundle content ID differs")


def validate_launch_request(request: controller_pb2.Controller.LaunchJobRequest, *, run: int) -> None:
    """Validate every v1 execution-affecting field before submission."""
    _validate_launch_request(request, run=run, entrypoint_command=_entrypoint_command())


def validate_launch_request_v3(request: controller_pb2.Controller.LaunchJobRequest, *, run: int) -> None:
    """Validate every v3 execution-affecting field before submission."""
    _validate_launch_request(request, run=run, entrypoint_command=_entrypoint_command_v3())


def validate_launch_request_v4(request: controller_pb2.Controller.LaunchJobRequest, *, run: int) -> None:
    """Validate every v4 execution-affecting field before submission."""
    _validate_launch_request(request, run=run, entrypoint_command=_entrypoint_command_v4())


def decode_launch_request(raw: bytes, *, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Decode canonical request bytes and revalidate their complete scope."""
    request = controller_pb2.Controller.LaunchJobRequest()
    try:
        request.ParseFromString(raw)
    except DecodeError as error:
        raise ValueError("LaunchJobRequest protobuf is malformed") from error
    if request.SerializeToString(deterministic=True) != raw:
        raise ValueError("LaunchJobRequest protobuf is not canonical")
    validate_launch_request(request, run=run)
    return request


def decode_launch_request_v3(raw: bytes, *, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Decode canonical v3 request bytes and revalidate their complete scope."""
    request = controller_pb2.Controller.LaunchJobRequest()
    try:
        request.ParseFromString(raw)
    except DecodeError as error:
        raise ValueError("LaunchJobRequest protobuf is malformed") from error
    if request.SerializeToString(deterministic=True) != raw:
        raise ValueError("LaunchJobRequest protobuf is not canonical")
    validate_launch_request_v3(request, run=run)
    return request


def decode_launch_request_v4(raw: bytes, *, run: int) -> controller_pb2.Controller.LaunchJobRequest:
    """Decode canonical v4 request bytes and revalidate their complete scope."""
    request = controller_pb2.Controller.LaunchJobRequest()
    try:
        request.ParseFromString(raw)
    except DecodeError as error:
        raise ValueError("LaunchJobRequest protobuf is malformed") from error
    if request.SerializeToString(deterministic=True) != raw:
        raise ValueError("LaunchJobRequest protobuf is not canonical")
    validate_launch_request_v4(request, run=run)
    return request


def _summary(raw: bytes, *, run: int) -> dict[str, object]:
    request = decode_launch_request(raw, run=run)
    manifest = _validated_bundle(request.bundle_blob)
    return {
        "bundle_sha256": request.bundle_id,
        "bundle_size": len(request.bundle_blob),
        "image": request.task_image,
        "job_name": request.name,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_size": len(raw),
        "run": run,
        "schema": REQUEST_SCHEMA,
        "source_commit": manifest["commit"],
        "source_tree": manifest["tree"],
    }


def _summary_v3(raw: bytes, *, run: int) -> dict[str, object]:
    request = decode_launch_request_v3(raw, run=run)
    manifest = _validated_bundle(request.bundle_blob)
    return {
        "bundle_sha256": request.bundle_id,
        "bundle_size": len(request.bundle_blob),
        "image": request.task_image,
        "job_name": request.name,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_size": len(raw),
        "run": run,
        "schema": REQUEST_SCHEMA_V3,
        "source_commit": manifest["commit"],
        "source_tree": manifest["tree"],
    }


def _summary_v4(raw: bytes, *, run: int) -> dict[str, object]:
    request = decode_launch_request_v4(raw, run=run)
    manifest = _validated_bundle(request.bundle_blob)
    return {
        "bundle_sha256": request.bundle_id,
        "bundle_size": len(request.bundle_blob),
        "image": request.task_image,
        "job_name": request.name,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_size": len(raw),
        "run": run,
        "schema": REQUEST_SCHEMA_V4,
        "source_commit": manifest["commit"],
        "source_tree": manifest["tree"],
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
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


def _reserve_create_only(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, FILE_MODE)
    except FileExistsError as error:
        raise FileExistsError(f"receipt already exists: {path}") from error


def _replace_reserved(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, payload)
    os.fsync(descriptor)


def _submit(request: controller_pb2.Controller.LaunchJobRequest, *, config: Path) -> str:
    with connect_controller(config_file=config) as endpoint:
        with rpc_client(endpoint.url, endpoint.credentials, timeout_ms=LAUNCH_TIMEOUT_MS) as client:
            response = client.launch_job(request, timeout_ms=LAUNCH_TIMEOUT_MS)
    return response.job_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("render", "render-v3", "render-v4"):
        render = subparsers.add_parser(name)
        render.add_argument("--bundle-root", required=True, type=Path)
        render.add_argument("--run", required=True, type=int)
        render.add_argument("--output", required=True, type=Path)
    for name in ("submit", "submit-v3", "submit-v4"):
        submit = subparsers.add_parser(name)
        submit.add_argument("--request", required=True, type=Path)
        submit.add_argument("--expected-request-sha256", required=True)
        submit.add_argument("--run", required=True, type=int)
        submit.add_argument("--config", required=True, type=Path)
        submit.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.action in ("render", "render-v3", "render-v4"):
        version = arguments.action.removeprefix("render")
        if version == "-v4":
            request = build_launch_request_v4(bundle_root=arguments.bundle_root, run=arguments.run)
        elif version == "-v3":
            request = build_launch_request_v3(bundle_root=arguments.bundle_root, run=arguments.run)
        else:
            request = build_launch_request(bundle_root=arguments.bundle_root, run=arguments.run)
        raw = request.SerializeToString(deterministic=True)
        _write_create_only(arguments.output, raw)
        if version == "-v4":
            summary = _summary_v4(raw, run=arguments.run)
        elif version == "-v3":
            summary = _summary_v3(raw, run=arguments.run)
        else:
            summary = _summary(raw, run=arguments.run)
        print(canonical_json(summary).decode(), end="")
        return 0

    raw = arguments.request.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if SHA256.fullmatch(arguments.expected_request_sha256) is None or observed_sha256 != (
        arguments.expected_request_sha256
    ):
        raise ValueError("reviewed request SHA-256 differs")
    version = arguments.action.removeprefix("submit")
    if version == "-v4":
        request = decode_launch_request_v4(raw, run=arguments.run)
        request_summary = _summary_v4(raw, run=arguments.run)
    elif version == "-v3":
        request = decode_launch_request_v3(raw, run=arguments.run)
        request_summary = _summary_v3(raw, run=arguments.run)
    else:
        request = decode_launch_request(raw, run=arguments.run)
        request_summary = _summary(raw, run=arguments.run)
    receipt_descriptor = _reserve_create_only(arguments.receipt)
    receipt_schema = (
        "shuttle_track2_te_h100_submit_receipt_v4"
        if version == "-v4"
        else (
            "shuttle_track2_te_h100_submit_receipt_v3"
            if version == "-v3"
            else "shuttle_track2_te_h100_submit_receipt_v1"
        )
    )
    pending = {
        "config": str(arguments.config),
        "request": request_summary,
        "schema": receipt_schema,
        "status": "submission_pending",
    }
    _replace_reserved(receipt_descriptor, canonical_json(pending))
    try:
        submitted_job = _submit(request, config=arguments.config)
        if submitted_job != request.name:
            raise RuntimeError("Iris returned an unexpected job identity")
        receipt = {
            "config": str(arguments.config),
            "job_id": submitted_job,
            "request": request_summary,
            "schema": receipt_schema,
            "status": "submitted",
        }
        _replace_reserved(receipt_descriptor, canonical_json(receipt))
    finally:
        os.close(receipt_descriptor)
    print(canonical_json(receipt).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
