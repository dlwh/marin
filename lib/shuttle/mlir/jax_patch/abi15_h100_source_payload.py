# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ABI15 source capsule and controller-verified raw export."""

import base64
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

CAPSULE_FILENAME = "abi15-h100-source-capsule.zip"
CAPSULE_MANIFEST_FILENAME = "abi15-h100-source-manifest.json"
ALLOWLIST_RELATIVE_PATH = "lib/shuttle/mlir/jax_patch/abi15-h100-source-allowlist-v1.json"
RAW_EXPORT_ARCHIVE_FILENAME = "abi15-h100-raw-artifacts.tar.gz"
RAW_EXPORT_MANIFEST_FILENAME = "RAW_EXPORT_MANIFEST"
READY_FILENAME = "RAW_EXPORT_READY"
ACK_FILENAME = "RAW_EXPORT_ACK"
CHUNK_BYTES = 256 * 1024
MAX_CAPSULE_BYTES = 20 * 1024 * 1024
MAX_CAPSULE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_CAPSULE_MEMBERS = 256
MAX_EXPORT_BYTES = 8 * 1024 * 1024 * 1024
MAX_EXPORT_MEMBERS = 100_000
MAX_METADATA_BYTES = 4 * 1024 * 1024
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PART = re.compile(r"abi15-h100-raw-artifacts\.tar\.gz\.part\.[0-9]{6}\Z")

REQUIRED_EXACT_PATHS = tuple(
    sorted(
        (
            ".agents/projects/tile_lifetime_compiler/shuttle_abi15_h100_composed_prerun_v1.json",
            "LICENSE",
            "lib/shuttle/mlir/jax_patch/0001-link-shuttle-xla-registry-adapter.patch",
            "lib/shuttle/mlir/jax_patch/0002-add-acceptance-observer-bridge.patch",
            "lib/shuttle/mlir/jax_patch/0003-link-shuttle-cuda-ffi.patch",
            "lib/shuttle/mlir/jax_patch/abi15-h100-composed-role-paths-v1.json",
            ALLOWLIST_RELATIVE_PATH,
            "lib/shuttle/mlir/jax_patch/abi15_h100_composed_contract.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_composed_fixture.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_evidence.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_evidence_subject.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_evidence_verify.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_gpu_build.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_gpu_transport.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_numerics.py",
            "lib/shuttle/mlir/jax_patch/abi15_h100_source_payload.py",
            "lib/shuttle/mlir/jax_patch/gpu_transport.py",
            "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration.py",
            "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration_abi11.py",
            "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration_abi15.py",
            "lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir",
        )
    )
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has an invalid digest")
    return value


def _relative_path(value: object, name: str) -> PurePosixPath:
    if type(value) is not str:
        raise ValueError(f"{name} must be a relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path


def _run_git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout.strip() if text else result.stdout


def _source_identity(repository: Path, commit: str) -> tuple[str, str]:
    commit = _sha(commit, _SHA1, "source commit")
    if _run_git(repository, "rev-parse", "HEAD") != commit:
        raise ValueError("source checkout differs from the requested commit")
    if _run_git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout must be globally clean")
    tree = _run_git(repository, "rev-parse", f"{commit}^{{tree}}")
    assert isinstance(tree, str)
    return commit, _sha(tree, _SHA1, "source tree")


def _tree_blobs(repository: Path, commit: str) -> dict[str, tuple[str, str]]:
    raw = _run_git(repository, "ls-tree", "-r", "-z", commit, text=False)
    assert isinstance(raw, bytes)
    records = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        if kind == b"blob":
            records[os.fsdecode(raw_path)] = (mode.decode(), object_id.decode())
    return records


def _git_blob(repository: Path, object_id: str) -> bytes:
    value = _run_git(repository, "cat-file", "blob", object_id, text=False)
    assert isinstance(value, bytes)
    return value


def _capsule_paths(repository: Path, records: dict[str, tuple[str, str]]) -> tuple[str, ...]:
    allowlist_record = records.get(ALLOWLIST_RELATIVE_PATH)
    if allowlist_record is None:
        raise ValueError("source commit omits the ABI15 source allowlist")
    raw = _git_blob(repository, allowlist_record[1])
    document = json.loads(raw)
    if raw != _canonical_json(document) or document != {
        "paths": list(REQUIRED_EXACT_PATHS),
        "schema_version": 1,
    }:
        raise ValueError("ABI15 source allowlist differs from the closed runtime path set")
    missing = [path for path in REQUIRED_EXACT_PATHS if path not in records]
    if missing:
        raise ValueError(f"source commit omits capsule paths: {missing}")
    return REQUIRED_EXACT_PATHS


def prepare_source_capsule(repository: Path, source_commit: str, output: Path) -> dict[str, Any]:
    """Create a deterministic allowlisted capsule from one clean Git commit."""
    commit, tree = _source_identity(repository, source_commit)
    if output.exists() or output.is_symlink():
        raise ValueError("source capsule output must be a fresh path")
    records = _tree_blobs(repository, commit)
    paths = _capsule_paths(repository, records)
    output.mkdir(parents=True)
    archive_path = output / CAPSULE_FILENAME
    member_records = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for relative in paths:
            mode, object_id = records[relative]
            if mode not in {"100644", "100755"}:
                raise ValueError("ABI15 source capsule permits only regular files")
            contents = _git_blob(repository, object_id)
            if len(contents) > MAX_CAPSULE_MEMBER_BYTES:
                raise ValueError("ABI15 source capsule member exceeds its bound")
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (0o100755 if mode == "100755" else 0o100644) << 16
            archive.writestr(info, contents)
            member_records.append(
                {
                    "mode": mode,
                    "path": relative,
                    "sha256": _sha256_bytes(contents),
                    "size": len(contents),
                }
            )
    if archive_path.stat().st_size > MAX_CAPSULE_BYTES or len(member_records) > MAX_CAPSULE_MEMBERS:
        raise ValueError("ABI15 source capsule exceeds its archive or member bound")
    manifest = {
        "archive": {
            "filename": CAPSULE_FILENAME,
            "sha256": _sha256_file(archive_path),
            "size": archive_path.stat().st_size,
        },
        "members": member_records,
        "schema_version": 1,
        "source": {"commit": commit, "tree": tree},
    }
    manifest_path = output / CAPSULE_MANIFEST_FILENAME
    manifest_path.write_bytes(_canonical_json(manifest))
    return {
        **manifest,
        "manifest_filename": CAPSULE_MANIFEST_FILENAME,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def restore_source_capsule(
    archive_path: Path,
    manifest_path: Path,
    output: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
) -> None:
    """Verify and restore a capsule without trusting archive filenames or modes."""
    if output.exists() or output.is_symlink():
        raise ValueError("capsule restore output must be fresh")
    raw = manifest_path.read_bytes()
    if _sha256_bytes(raw) != _sha(expected_manifest_sha256, _SHA256, "manifest SHA-256"):
        raise ValueError("source manifest differs from the trusted launch identity")
    manifest = json.loads(raw)
    if raw != _canonical_json(manifest) or set(manifest) != {
        "archive",
        "members",
        "schema_version",
        "source",
    }:
        raise ValueError("source manifest is not canonical or closed")
    expected_source = {
        "commit": _sha(expected_source_commit, _SHA1, "source commit"),
        "tree": _sha(expected_source_tree, _SHA1, "source tree"),
    }
    if manifest["schema_version"] != 1 or manifest["source"] != expected_source:
        raise ValueError("source manifest identity drift")
    archive_record = manifest["archive"]
    if archive_record != {
        "filename": CAPSULE_FILENAME,
        "sha256": _sha256_file(archive_path),
        "size": archive_path.stat().st_size,
    }:
        raise ValueError("source capsule archive identity drift")
    members = manifest["members"]
    if type(members) is not list or [record.get("path") for record in members] != list(REQUIRED_EXACT_PATHS):
        raise ValueError("source capsule member roster drift")
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != list(REQUIRED_EXACT_PATHS):
            raise ValueError("source capsule ZIP member roster drift")
        for info, record in zip(infos, members, strict=True):
            if info.is_dir() or info.file_size != record["size"] or info.file_size > MAX_CAPSULE_MEMBER_BYTES:
                raise ValueError("source capsule ZIP member metadata drift")
            contents = archive.read(info)
            if _sha256_bytes(contents) != record["sha256"]:
                raise ValueError("source capsule member bytes drift")
            mode = (info.external_attr >> 16) & 0o177777
            expected_mode = 0o100755 if record["mode"] == "100755" else 0o100644
            if mode != expected_mode:
                raise ValueError("source capsule member mode drift")
            relative = _relative_path(info.filename, "capsule member")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            target.chmod(0o755 if record["mode"] == "100755" else 0o644)


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir() or (root / "accepted_bundle.json").exists():
        raise ValueError("task raw artifact must be regular and cannot contain accepted_bundle.json")
    records = []
    total_size = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("raw export rejects symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("raw export rejects non-regular files")
        relative = path.relative_to(root).as_posix()
        _relative_path(relative, "artifact path")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode not in {0o644, 0o755}:
            raise ValueError("raw export rejects noncanonical modes")
        records.append(
            {
                "mode": f"{mode:04o}",
                "path": relative,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
        total_size += path.stat().st_size
        if len(records) > MAX_EXPORT_MEMBERS or total_size > MAX_EXPORT_BYTES:
            raise ValueError("raw export exceeds its member or size bound")
    if not records or not any(record["path"] == "raw-artifact-manifest.json" for record in records):
        raise ValueError("raw export requires the closed raw artifact manifest")
    return records


def _write_archive(path: Path, root: Path, records: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.GNU_FORMAT) as archive:
                for record in records:
                    info = tarfile.TarInfo(record["path"])
                    info.size = record["size"]
                    info.mode = int(record["mode"], 8)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with (root / record["path"]).open("rb") as stream:
                        archive.addfile(info, stream)


def _split_archive(path: Path, output: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("rb") as stream:
        for index, contents in enumerate(iter(lambda: stream.read(CHUNK_BYTES), b"")):
            filename = f"{RAW_EXPORT_ARCHIVE_FILENAME}.part.{index:06d}"
            part = output / filename
            part.write_bytes(contents)
            records.append(
                {
                    "filename": filename,
                    "index": index,
                    "sha256": _sha256_bytes(contents),
                    "size": len(contents),
                }
            )
    if not records:
        raise ValueError("raw export archive must create at least one part")
    return records


def prepare_raw_export(
    artifact: Path,
    output: Path,
    *,
    runner_exit_code: int,
    source_commit: str,
    source_tree: str,
    source_manifest_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Close raw task files into text-retrievable parts without acceptance."""
    if type(runner_exit_code) is not int or not 0 <= runner_exit_code <= 255:
        raise ValueError("runner exit code must be in [0, 255]")
    if output.exists() or output.is_symlink():
        raise ValueError("raw export output must be fresh")
    source = {
        "commit": _sha(source_commit, _SHA1, "source commit"),
        "manifest_sha256": _sha(source_manifest_sha256, _SHA256, "source manifest SHA-256"),
        "tree": _sha(source_tree, _SHA1, "source tree"),
    }
    authorization = _sha(authorization_sha256, _SHA256, "authorization SHA-256")
    records = _artifact_records(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name) / "export"
        temporary.mkdir()
        archive_path = temporary / RAW_EXPORT_ARCHIVE_FILENAME
        _write_archive(archive_path, artifact, records)
        archive = {
            "filename": RAW_EXPORT_ARCHIVE_FILENAME,
            "sha256": _sha256_file(archive_path),
            "size": archive_path.stat().st_size,
        }
        parts = _split_archive(archive_path, temporary)
        archive_path.unlink()
        manifest = {
            "archive": archive,
            "authorization_sha256": authorization,
            "members": records,
            "parts": parts,
            "runner_exit_code": runner_exit_code,
            "schema_version": 1,
            "source": source,
            "status": "raw_subject_artifact_only",
        }
        manifest_path = temporary / RAW_EXPORT_MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json(manifest))
        if manifest_path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("raw export manifest exceeds its metadata bound")
        ready = {
            "archive_sha256": archive["sha256"],
            "archive_size": archive["size"],
            "authorization_sha256": authorization,
            "manifest_sha256": _sha256_file(manifest_path),
            "part_count": len(parts),
            "runner_exit_code": runner_exit_code,
            "schema_version": 1,
            "status": "raw_ready_for_controller_verification",
        }
        (temporary / READY_FILENAME).write_bytes(_canonical_json(ready))
        temporary.replace(output)
    return ready


def _transport_bound(filename: str) -> int:
    if filename in {READY_FILENAME, RAW_EXPORT_MANIFEST_FILENAME}:
        return MAX_METADATA_BYTES
    if _PART.fullmatch(filename):
        return CHUNK_BYTES
    raise ValueError("raw export filename is not transportable")


def read_export_record(export: Path, filename: str, *, offset: int = 0) -> bytes:
    """Return one canonical base64 record for Iris text exec."""
    bound = _transport_bound(filename)
    if type(offset) is not int or offset < 0 or offset % CHUNK_BYTES != 0:
        raise ValueError("raw export offset must be a chunk boundary")
    path = export / filename
    if export.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > bound:
        raise ValueError("raw export file is missing, non-regular, or oversized")
    with path.open("rb") as stream:
        stream.seek(offset)
        contents = stream.read(CHUNK_BYTES)
    if not contents:
        raise ValueError("raw export record offset is beyond EOF")
    return _canonical_json(
        {
            "contents_base64": base64.b64encode(contents).decode("ascii"),
            "eof": offset + len(contents) == path.stat().st_size,
            "filename": filename,
            "offset": offset,
            "schema_version": 1,
            "sha256": _sha256_bytes(contents),
            "size": len(contents),
            "total_size": path.stat().st_size,
        }
    )


def decode_export_record(raw: bytes, *, expected_filename: str, expected_offset: int) -> dict[str, Any]:
    """Decode one contiguous text transport record."""
    if len(raw) > 2 * CHUNK_BYTES:
        raise ValueError("raw export record exceeds its text bound")
    record = json.loads(raw)
    if raw != _canonical_json(record) or set(record) != {
        "contents_base64",
        "eof",
        "filename",
        "offset",
        "schema_version",
        "sha256",
        "size",
        "total_size",
    }:
        raise ValueError("raw export record is not canonical or closed")
    if record["schema_version"] != 1 or record["filename"] != expected_filename or record["offset"] != expected_offset:
        raise ValueError("raw export record identity or offset drift")
    try:
        contents = base64.b64decode(record["contents_base64"], validate=True)
    except ValueError as error:
        raise ValueError("raw export record contains invalid base64") from error
    if type(record["size"]) is not int or record["size"] != len(contents) or not 0 < len(contents) <= CHUNK_BYTES:
        raise ValueError("raw export record size drift")
    if _sha256_bytes(contents) != _sha(record["sha256"], _SHA256, "record SHA-256"):
        raise ValueError("raw export record digest drift")
    if type(record["total_size"]) is not int or record["total_size"] < expected_offset + len(contents):
        raise ValueError("raw export record total size drift")
    if record["eof"] is not (expected_offset + len(contents) == record["total_size"]):
        raise ValueError("raw export EOF marker drift")
    return {**record, "contents": contents}


def _load_canonical(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"{name} is missing, non-regular, or oversized")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_json(value):
        raise ValueError(f"{name} must use canonical JSON")
    return value


def validate_retrieved_export(
    retrieval: Path,
    archive_output: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_source_manifest_sha256: str,
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    """Reconstruct and validate raw bytes before semantic controller verification."""
    if retrieval.is_symlink() or not retrieval.is_dir() or archive_output.exists() or archive_output.is_symlink():
        raise ValueError("retrieval input and archive output must be real and fresh")
    ready_path = retrieval / READY_FILENAME
    manifest_path = retrieval / RAW_EXPORT_MANIFEST_FILENAME
    ready = _load_canonical(ready_path, "READY")
    manifest = _load_canonical(manifest_path, "raw export manifest")
    if (
        ready.get("status") != "raw_ready_for_controller_verification"
        or manifest.get("status") != "raw_subject_artifact_only"
    ):
        raise ValueError("raw retrieval attempted to claim acceptance")
    expected_authorization = _sha(expected_authorization_sha256, _SHA256, "authorization SHA-256")
    expected_source = {
        "commit": _sha(expected_source_commit, _SHA1, "source commit"),
        "manifest_sha256": _sha(expected_source_manifest_sha256, _SHA256, "source manifest SHA-256"),
        "tree": _sha(expected_source_tree, _SHA1, "source tree"),
    }
    if (
        manifest.get("authorization_sha256") != expected_authorization
        or ready.get("authorization_sha256") != expected_authorization
    ):
        raise ValueError("raw retrieval authorization drift")
    if manifest.get("source") != expected_source or ready.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("raw retrieval source or manifest drift")
    parts = manifest.get("parts")
    if type(parts) is not list or not parts or ready.get("part_count") != len(parts):
        raise ValueError("raw retrieval part roster drift")
    expected_files = {READY_FILENAME, RAW_EXPORT_MANIFEST_FILENAME}
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with archive_output.open("wb") as output:
        for index, record in enumerate(parts):
            if type(record) is not dict or set(record) != {
                "filename",
                "index",
                "sha256",
                "size",
            }:
                raise ValueError("raw retrieval part schema drift")
            filename = f"{RAW_EXPORT_ARCHIVE_FILENAME}.part.{index:06d}"
            if record["index"] != index or record["filename"] != filename:
                raise ValueError("raw retrieval parts are missing or reordered")
            path = retrieval / filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
                raise ValueError("raw retrieval part file metadata drift")
            contents = path.read_bytes()
            if not contents or len(contents) > CHUNK_BYTES or _sha256_bytes(contents) != record["sha256"]:
                raise ValueError("raw retrieval part bytes drift")
            output.write(contents)
            digest.update(contents)
            size += len(contents)
            expected_files.add(filename)
    actual_files = {path.name for path in retrieval.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_files != expected_files:
        archive_output.unlink(missing_ok=True)
        raise ValueError("raw retrieval contains extra or missing transport files")
    archive = manifest.get("archive")
    if archive != {
        "filename": RAW_EXPORT_ARCHIVE_FILENAME,
        "sha256": digest.hexdigest(),
        "size": size,
    }:
        archive_output.unlink(missing_ok=True)
        raise ValueError("raw retrieval archive identity drift")
    if ready.get("archive_sha256") != digest.hexdigest() or ready.get("archive_size") != size:
        archive_output.unlink(missing_ok=True)
        raise ValueError("READY does not bind the reconstructed archive")
    members = manifest.get("members")
    if type(members) is not list or not members:
        archive_output.unlink(missing_ok=True)
        raise ValueError("raw export member roster is empty")
    observed = []
    try:
        with tarfile.open(archive_output, mode="r:gz") as archive_stream:
            tar_members = archive_stream.getmembers()
            if len(tar_members) != len(members):
                raise ValueError("raw archive member count drift")
            for tar_member, record in zip(tar_members, members, strict=True):
                if type(record) is not dict or set(record) != {
                    "mode",
                    "path",
                    "sha256",
                    "size",
                }:
                    raise ValueError("raw archive member schema drift")
                _relative_path(record["path"], "raw archive member")
                if not tar_member.isfile() or tar_member.name != record["path"] or tar_member.size != record["size"]:
                    raise ValueError("raw archive member identity drift")
                if tar_member.mode != int(record["mode"], 8):
                    raise ValueError("raw archive member mode drift")
                stream = archive_stream.extractfile(tar_member)
                if stream is None or _sha256_bytes(stream.read()) != record["sha256"]:
                    raise ValueError("raw archive member digest drift")
                observed.append(tar_member.name)
    except Exception:
        archive_output.unlink(missing_ok=True)
        raise
    if observed != sorted(set(observed)) or "accepted_bundle.json" in observed:
        archive_output.unlink(missing_ok=True)
        raise ValueError("raw archive paths are noncanonical or contain task acceptance")
    return manifest


def controller_ack(
    ready: dict[str, Any],
    *,
    verification_report_sha256: str,
    verifier_sha256: str,
) -> dict[str, Any]:
    """Build the ACK only after independent controller verification."""
    if ready.get("schema_version") != 1 or ready.get("status") != "raw_ready_for_controller_verification":
        raise ValueError("ACK requires one ABI15 raw READY record")
    return {
        "archive_sha256": _sha(ready.get("archive_sha256"), _SHA256, "archive SHA-256"),
        "authorization_sha256": _sha(ready.get("authorization_sha256"), _SHA256, "authorization SHA-256"),
        "manifest_sha256": _sha(ready.get("manifest_sha256"), _SHA256, "manifest SHA-256"),
        "schema_version": 1,
        "status": "controller_verified_raw_ack",
        "verification_report_sha256": _sha(verification_report_sha256, _SHA256, "verification report SHA-256"),
        "verifier_sha256": _sha(verifier_sha256, _SHA256, "verifier SHA-256"),
    }


def write_task_ack(export: Path, ack: dict[str, Any]) -> None:
    """Write a controller-produced ACK only when it binds the task's exact READY."""
    ready = _load_canonical(export / READY_FILENAME, "READY")
    expected_fields = {
        "archive_sha256",
        "authorization_sha256",
        "manifest_sha256",
        "schema_version",
        "status",
        "verification_report_sha256",
        "verifier_sha256",
    }
    if set(ack) != expected_fields or ack.get("status") != "controller_verified_raw_ack":
        raise ValueError("ACK must use the closed controller schema")
    for field in ("archive_sha256", "authorization_sha256", "manifest_sha256"):
        if ack.get(field) != ready.get(field):
            raise ValueError("ACK does not bind the task READY identities")
    for field in ("verification_report_sha256", "verifier_sha256"):
        _sha(ack.get(field), _SHA256, field)
    path = export / ACK_FILENAME
    if path.exists() or path.is_symlink():
        raise ValueError("raw ACK path must be fresh")
    temporary = export / f".{ACK_FILENAME}.tmp"
    temporary.write_bytes(_canonical_json(ack))
    temporary.replace(path)
