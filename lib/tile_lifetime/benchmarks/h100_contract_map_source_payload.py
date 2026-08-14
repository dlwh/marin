# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and restore an exact source capsule for the H100 evidence runner."""

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
ALLOWLIST_RELATIVE_PATH = Path("lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json")
CAPSULE_FILENAME = "h100-evidence-source-capsule.zip"
MANIFEST_FILENAME = "h100-evidence-source-manifest.json"
LAUNCHER_FILENAME = "h100_contract_map_source_payload.py"
RUNNER_RELATIVE_PATH = Path("lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py")
ARTIFACT_EXPORT_SCHEMA_VERSION = 1
ARTIFACT_EXPORT_ARCHIVE_FILENAME = "h100-evidence-artifacts.tar.gz"
ARTIFACT_EXPORT_MANIFEST_FILENAME = "h100-evidence-artifact-manifest.json"
ARTIFACT_EXPORT_READY_FILENAME = "EXPORT_READY"
ARTIFACT_EXPORT_ACK_FILENAME = "EXPORT_ACK"
ARTIFACT_EXPORT_CHUNK_BYTES = 256 * 1024
MAX_ARTIFACT_EXPORT_READY_BYTES = 16 * 1024
MAX_ARTIFACT_EXPORT_MANIFEST_BYTES = 1024 * 1024
MAX_ARTIFACT_EXPORT_RECORD_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_EXPORT_BYTES = 8 * 1024 * 1024 * 1024
MAX_ARTIFACT_EXPORT_MEMBERS = 100_000
MAX_ARTIFACT_EXPORT_PARTS = MAX_ARTIFACT_EXPORT_BYTES // ARTIFACT_EXPORT_CHUNK_BYTES
MISSING_ACCEPTED_BUNDLE_EXIT_CODE = 70
CUDA_BIN = Path("/usr/local/cuda-13.2/bin")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CAPSULE_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_MEMBERS = 10_000
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ARTIFACT_EXPORT_PART_PATTERN = re.compile(r"h100-evidence-artifacts\.tar\.gz\.part\.[0-9]{6}")
FORBIDDEN_PARTS = frozenset({".git", ".ssh", ".aws", ".gnupg", "__pycache__"})
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
    }
)
REQUIRED_EXACT_PATHS = (
    ".agents/projects/tile_lifetime_compiler/h100_contract_map_backend_evidence.md",
    "LICENSE",
    "lib/tile_lifetime/README.md",
    "lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py",
    "lib/tile_lifetime/benchmarks/h100_contract_map_backend_training.py",
    "lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json",
    "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py",
    "lib/tile_lifetime/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
)


def _run_git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout.rstrip("\n") if text else completed.stdout


def _require_sha(value: Any, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has invalid digest syntax: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a normalized relative path: {value!r}")
    path = PurePosixPath(value)
    if not value or str(path) != value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path: {value!r}")
    if FORBIDDEN_PARTS.intersection(path.parts) or path.name in FORBIDDEN_NAMES:
        raise ValueError(f"{name} names forbidden credential or repository state: {value!r}")
    return path


def _validate_symlink_target(path: PurePosixPath, target_value: Any) -> str:
    if not isinstance(target_value, str):
        raise ValueError(f"symlink target must be a string: {target_value!r}")
    target = PurePosixPath(target_value)
    normalized_target = posixpath.normpath(str(path.parent / target))
    if (
        not target_value
        or "\0" in target_value
        or target.is_absolute()
        or normalized_target == ".."
        or normalized_target.startswith("../")
    ):
        raise ValueError(f"source capsule symlink escapes the workspace: {path} -> {target_value}")
    return target_value


def _source_identity(source_root: Path, source_sha: str) -> tuple[str, str]:
    source_sha = _require_sha(source_sha, SHA1_PATTERN, "source SHA")
    if _run_git(source_root, "rev-parse", "--show-object-format") != "sha1":
        raise ValueError("source capsule supports only SHA-1 Git repositories")
    head = str(_run_git(source_root, "rev-parse", "HEAD"))
    if head != source_sha:
        raise ValueError(f"source checkout is {head}, expected {source_sha}")
    if _run_git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout must be globally clean before capsule preparation")
    tree = str(_run_git(source_root, "rev-parse", f"{source_sha}^{{tree}}"))
    return source_sha, _require_sha(tree, SHA1_PATTERN, "source tree")


def _git_tree_records(repository: Path, source_sha: str) -> dict[str, tuple[str, str]]:
    raw = _run_git(repository, "ls-tree", "-r", "-z", source_sha, text=False)
    assert isinstance(raw, bytes)
    records: dict[str, tuple[str, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", maxsplit=1)
        mode, object_type, object_id = metadata.split()
        source_path = os.fsdecode(raw_path)
        if object_type != b"blob":
            records[source_path] = (mode.decode(), "")
            continue
        records[source_path] = (mode.decode(), object_id.decode())
    return records


def _git_blob(repository: Path, object_id: str, source_path: str) -> bytes:
    if not object_id:
        raise ValueError(f"source capsule path is not a Git blob: {source_path}")
    contents = _run_git(repository, "cat-file", "blob", object_id, text=False)
    assert isinstance(contents, bytes)
    return contents


def _load_allowlist(
    source_root: Path,
    tree_records: dict[str, tuple[str, str]],
) -> tuple[str, ...]:
    allowlist_path = ALLOWLIST_RELATIVE_PATH.as_posix()
    if allowlist_path not in tree_records:
        raise ValueError("source commit omits the source capsule allowlist")
    raw = _git_blob(source_root, tree_records[allowlist_path][1], allowlist_path)
    try:
        allowlist = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("source capsule allowlist is not valid JSON") from error
    if not isinstance(allowlist, dict) or set(allowlist) != {"exact", "recursive", "schema_version"}:
        raise ValueError("source capsule allowlist must use the closed schema")
    if _canonical_json(allowlist) != raw:
        raise ValueError("source capsule allowlist must use canonical JSON encoding")
    if type(allowlist["schema_version"]) is not int or allowlist["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"source capsule allowlist schema_version must be {SCHEMA_VERSION}")
    exact = allowlist["exact"]
    recursive = allowlist["recursive"]
    if not isinstance(exact, list) or not exact or not all(isinstance(value, str) for value in exact):
        raise ValueError("source capsule exact allowlist must be a nonempty string list")
    if not isinstance(recursive, list) or len(recursive) != 1:
        raise ValueError("source capsule recursive allowlist must contain exactly one package rule")
    rule = recursive[0]
    if rule != {"path": "lib/tile_lifetime/src/tile_lifetime", "suffix": ".py"}:
        raise ValueError("source capsule recursive allowlist must cover exactly the tile_lifetime Python package")

    paths = {_validate_relative_path(value, "allowlist path").as_posix() for value in exact}
    package_paths = {
        path
        for path, (_mode, object_id) in tree_records.items()
        if object_id and path.startswith(f"{rule['path']}/") and path.endswith(rule["suffix"])
    }
    paths.update(package_paths)
    if ALLOWLIST_RELATIVE_PATH.as_posix() not in paths:
        raise ValueError("source capsule allowlist must include itself")
    if exact != sorted(set(exact)):
        raise ValueError("source capsule exact allowlist must be unique and source ordered")
    if tuple(exact) != REQUIRED_EXACT_PATHS:
        raise ValueError("source capsule exact allowlist drifted from the reviewed runtime/config path set")

    missing = sorted(path for path in paths if path not in tree_records or not tree_records[path][1])
    if missing:
        raise ValueError(f"source capsule allowlist names paths absent from the immutable source tree: {missing}")
    return tuple(sorted(paths))


def _capsule_records(
    source_root: Path,
    paths: tuple[str, ...],
    tree_records: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records: list[dict[str, Any]] = []
    contents_by_path: dict[str, bytes] = {}
    total_size = 0
    for source_path in paths:
        path = _validate_relative_path(source_path, "capsule path")
        mode, object_id = tree_records[source_path]
        blob = _git_blob(source_root, object_id, source_path)
        if mode in {"100644", "100755"}:
            member_type = "file"
            contents = blob
        elif mode == "120000":
            member_type = "symlink"
            contents = blob
            try:
                target = contents.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"source capsule symlink target is not UTF-8: {source_path}") from error
            _validate_symlink_target(path, target)
        else:
            raise ValueError(f"source capsule does not support Git mode {mode} at {source_path}")
        if len(contents) > MAX_MEMBER_BYTES:
            raise ValueError(f"source capsule member {source_path} exceeds {MAX_MEMBER_BYTES} bytes")
        total_size += len(contents)
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"source capsule expands beyond {MAX_EXTRACTED_BYTES} bytes")
        contents_by_path[source_path] = contents
        records.append(
            {
                "mode": mode,
                "path": source_path,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size": len(contents),
                "type": member_type,
            }
        )
    if len(records) > MAX_MEMBERS:
        raise ValueError(f"source capsule exceeds {MAX_MEMBERS} members")
    return records, contents_by_path


def _write_capsule(path: Path, contents_by_path: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents in sorted(contents_by_path.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, contents)


def prepare_source_payload(source_root: Path, source_sha: str, output_directory: Path) -> dict[str, Any]:
    """Create an immutable allowlisted source capsule from a clean exact checkout."""
    source_root = source_root.resolve()
    output_directory = output_directory.resolve()
    source_sha, source_tree = _source_identity(source_root, source_sha)
    tree_records = _git_tree_records(source_root, source_sha)
    allowlisted_paths = _load_allowlist(source_root, tree_records)
    records, contents_by_path = _capsule_records(source_root, allowlisted_paths, tree_records)
    if output_directory.exists():
        raise ValueError(f"payload output already exists: {output_directory}")
    if output_directory.is_relative_to(source_root):
        raise ValueError("payload output must be outside the exact source checkout")
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output_directory.name}.", dir=output_directory.parent) as temp_name:
        payload = Path(temp_name) / "payload"
        payload.mkdir()
        capsule_path = payload / CAPSULE_FILENAME
        _write_capsule(capsule_path, contents_by_path)
        if capsule_path.stat().st_size > MAX_CAPSULE_BYTES:
            raise ValueError(f"source capsule exceeds {MAX_CAPSULE_BYTES} bytes")
        manifest = {
            "archive": {"filename": CAPSULE_FILENAME, "sha256": _sha256(capsule_path)},
            "members": records,
            "schema_version": SCHEMA_VERSION,
            "source": {"commit": source_sha, "tree": source_tree},
        }
        manifest_path = payload / MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json(manifest))
        launcher_relative_path = "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py"
        (payload / LAUNCHER_FILENAME).write_bytes(contents_by_path[launcher_relative_path])
        payload_size = sum(path.stat().st_size for path in payload.iterdir())
        if payload_size > MAX_PAYLOAD_BYTES:
            raise ValueError(f"source capsule transport exceeds {MAX_PAYLOAD_BYTES} bytes before Iris compression")
        if _source_identity(source_root, source_sha) != (source_sha, source_tree):
            raise ValueError("source identity changed while preparing the capsule")
        payload.replace(output_directory)

    result = dict(manifest)
    result["launcher_sha256"] = _sha256(output_directory / LAUNCHER_FILENAME)
    result["manifest_sha256"] = _sha256(output_directory / MANIFEST_FILENAME)
    result["payload_directory"] = str(output_directory)
    return result


def _validate_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    expected_manifest_sha256 = _require_sha(expected_manifest_sha256, SHA256_PATTERN, "manifest SHA-256")
    expected_source_sha = _require_sha(expected_source_sha, SHA1_PATTERN, "source SHA")
    expected_source_tree = _require_sha(expected_source_tree, SHA1_PATTERN, "source tree")
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("source manifest SHA-256 does not match the trusted launch identity")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("source manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {"archive", "members", "schema_version", "source"}:
        raise ValueError("source manifest must use the closed schema")
    if _canonical_json(manifest) != raw:
        raise ValueError("source manifest must use canonical JSON encoding")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"source manifest schema_version must be {SCHEMA_VERSION}")
    if manifest["source"] != {"commit": expected_source_sha, "tree": expected_source_tree}:
        raise ValueError("source manifest commit or tree differs from the trusted launch identity")
    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {"filename", "sha256"}:
        raise ValueError("source manifest archive record must use the closed schema")
    if archive["filename"] != CAPSULE_FILENAME:
        raise ValueError(f"source manifest archive filename must be {CAPSULE_FILENAME}")
    _require_sha(archive["sha256"], SHA256_PATTERN, "capsule SHA-256")
    members = manifest["members"]
    if not isinstance(members, list) or not members or len(members) > MAX_MEMBERS:
        raise ValueError("source manifest members must be a bounded nonempty list")
    paths: set[str] = set()
    total_size = 0
    for record in members:
        if not isinstance(record, dict) or set(record) != {"mode", "path", "sha256", "size", "type"}:
            raise ValueError("source capsule member must use the closed schema")
        path = _validate_relative_path(record["path"], "capsule path")
        if record["path"] in paths:
            raise ValueError(f"source capsule repeats path: {record['path']}")
        paths.add(record["path"])
        if record["type"] == "file":
            if record["mode"] not in {"100644", "100755"}:
                raise ValueError(f"source capsule file has invalid mode: {record}")
        elif record["type"] == "symlink":
            if record["mode"] != "120000":
                raise ValueError(f"source capsule symlink has invalid mode: {record}")
        else:
            raise ValueError(f"source capsule member has invalid type: {record}")
        if type(record["size"]) is not int or not 0 <= record["size"] <= MAX_MEMBER_BYTES:
            raise ValueError(f"source capsule member has invalid size: {record}")
        _require_sha(record["sha256"], SHA256_PATTERN, "capsule member SHA-256")
        total_size += record["size"]
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"source capsule expands beyond {MAX_EXTRACTED_BYTES} bytes")
        if record["type"] == "symlink" and not path.parts:
            raise ValueError("source capsule symlink path must be nonempty")
    if [record["path"] for record in members] != sorted(paths):
        raise ValueError("source capsule members must be in canonical path order")
    return manifest


def load_and_verify_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    """Load one trusted canonical source manifest without importing project code."""
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("source manifest must be a regular file")
    raw = manifest_path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"source manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    return _validate_manifest(
        raw,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_sha=expected_source_sha,
        expected_source_tree=expected_source_tree,
    )


def _safe_parent(root: Path, path: PurePosixPath) -> None:
    parent = root
    for part in path.parts[:-1]:
        parent /= part
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"source capsule parent is not a real directory: {path}")
        parent.mkdir(exist_ok=True)


def _verify_extracted_source(source_root: Path, manifest: dict[str, Any]) -> None:
    records = {record["path"]: record for record in manifest["members"]}
    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != set(records):
        raise ValueError(
            f"source capsule file set differs from manifest: missing={sorted(set(records) - actual_paths)}, "
            f"extra={sorted(actual_paths - set(records))}"
        )
    expected_directories = {
        parent.as_posix()
        for source_path in records
        for parent in PurePosixPath(source_path).parents
        if parent.as_posix() != "."
    }
    actual_directories = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_directories != expected_directories:
        raise ValueError("source capsule directory set differs from manifest paths")
    for source_path, record in records.items():
        path = source_root.joinpath(*PurePosixPath(source_path).parts)
        if record["type"] == "symlink":
            if not path.is_symlink():
                raise ValueError(f"source capsule member is not the required symlink: {source_path}")
            contents = os.readlink(path).encode()
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source capsule member is not a regular file: {source_path}")
            expected_mode = 0o755 if record["mode"] == "100755" else 0o644
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise ValueError(f"source capsule member mode differs from manifest: {source_path}")
            contents = path.read_bytes()
        if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"source capsule member content differs from manifest: {source_path}")


def restore_and_verify_source_payload(
    workspace: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> tuple[Path, dict[str, Any]]:
    """Extract and verify an exact allowlisted source capsule."""
    workspace = workspace.resolve()
    manifest_path = workspace / MANIFEST_FILENAME
    capsule_path = workspace / CAPSULE_FILENAME
    manifest = load_and_verify_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_sha=expected_source_sha,
        expected_source_tree=expected_source_tree,
    )
    if capsule_path.is_symlink() or not capsule_path.is_file():
        raise ValueError("source capsule must be a regular file")
    if capsule_path.stat().st_size > MAX_CAPSULE_BYTES:
        raise ValueError(f"source capsule exceeds {MAX_CAPSULE_BYTES} bytes")
    if _sha256(capsule_path) != manifest["archive"]["sha256"]:
        raise ValueError("source capsule SHA-256 does not match the source manifest")
    source_root = workspace / "source"
    if source_root.exists() or source_root.is_symlink():
        raise ValueError("source capsule extraction root must not already exist")
    temporary = Path(tempfile.mkdtemp(prefix=".source-capsule-", dir=workspace))
    try:
        records = {record["path"]: record for record in manifest["members"]}
        with zipfile.ZipFile(capsule_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != set(records):
                raise ValueError("source capsule members differ from the closed manifest")
            for member in members:
                record = records[member.filename]
                transport_mode = member.external_attr >> 16
                if member.is_dir() or stat.S_IFMT(transport_mode) != stat.S_IFREG:
                    raise ValueError(f"source capsule transport member is not regular: {member.filename}")
                if member.file_size != record["size"]:
                    raise ValueError(f"source capsule member size differs from manifest: {member.filename}")
                contents = archive.read(member)
                if hashlib.sha256(contents).hexdigest() != record["sha256"]:
                    raise ValueError(f"source capsule member SHA-256 differs from manifest: {member.filename}")
                relative = _validate_relative_path(member.filename, "capsule path")
                destination = temporary.joinpath(*relative.parts)
                _safe_parent(temporary, relative)
                if destination.exists() or destination.is_symlink():
                    raise ValueError(f"source capsule member repeats extraction path: {member.filename}")
                if record["type"] == "symlink":
                    try:
                        target = contents.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(f"source capsule symlink target is not UTF-8: {member.filename}") from error
                    destination.symlink_to(_validate_symlink_target(relative, target))
                else:
                    destination.write_bytes(contents)
                    destination.chmod(0o755 if record["mode"] == "100755" else 0o644)
        _verify_extracted_source(temporary, manifest)
        temporary.replace(source_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return source_root, manifest


def runner_arguments(
    runtime_python: Path,
    source_root: Path,
    source_sha: str,
    source_tree: str,
    manifest_path: Path,
    manifest_sha256: str,
    artifact_directory: Path,
    require_jax_version: str,
) -> tuple[str, ...]:
    if not require_jax_version.strip():
        raise ValueError("required JAX version must be nonempty")
    if not runtime_python.is_absolute() or not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise ValueError(f"runtime Python must be an absolute executable file: {runtime_python}")
    return (
        str(runtime_python),
        str(source_root / RUNNER_RELATIVE_PATH),
        "--execute",
        "--source-root",
        str(source_root),
        "--source-sha",
        source_sha,
        "--source-tree",
        source_tree,
        "--source-capsule-manifest",
        str(manifest_path),
        "--source-capsule-manifest-sha256",
        manifest_sha256,
        "--artifact-directory",
        str(artifact_directory),
        "--require-jax-version",
        require_jax_version,
        "--git",
        "/usr/bin/git",
        "--nvidia-smi",
        "/usr/bin/nvidia-smi",
        "--nvcc",
        str(CUDA_BIN / "nvcc"),
        "--ptxas",
        str(CUDA_BIN / "ptxas"),
        "--cuobjdump",
        str(CUDA_BIN / "cuobjdump"),
        "--ncu",
        "/usr/local/bin/ncu",
        "--nsys",
        "/usr/local/bin/nsys",
    )


def _artifact_file_records(artifact_directory: Path) -> list[dict[str, Any]]:
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    total_size = 0
    for path in sorted(artifact_directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact export rejects symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact export rejects non-regular files: {path}")
        relative = path.relative_to(artifact_directory).as_posix()
        _validate_relative_path(relative, "artifact export path")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode not in {0o644, 0o755}:
            raise ValueError(f"artifact export rejects noncanonical mode {mode:04o}: {path}")
        records.append(
            {
                "mode": f"{mode:04o}",
                "path": relative,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
        if len(records) > MAX_ARTIFACT_EXPORT_MEMBERS:
            raise ValueError("artifact export exceeds the member-count bound")
        total_size += path.stat().st_size
        if total_size > MAX_ARTIFACT_EXPORT_BYTES:
            raise ValueError("artifact export exceeds the uncompressed size bound")
    return records


def _write_artifact_archive(
    path: Path,
    artifact_directory: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.GNU_FORMAT) as archive:
                for record in records:
                    source = artifact_directory / record["path"]
                    info = tarfile.TarInfo(record["path"])
                    info.size = record["size"]
                    info.mode = int(record["mode"], 8)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)


def _split_artifact_archive(path: Path, export_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for index, contents in enumerate(iter(lambda: stream.read(ARTIFACT_EXPORT_CHUNK_BYTES), b"")):
            filename = f"{ARTIFACT_EXPORT_ARCHIVE_FILENAME}.part.{index:06d}"
            part = export_directory / filename
            part.write_bytes(contents)
            records.append(
                {
                    "filename": filename,
                    "index": index,
                    "sha256": hashlib.sha256(contents).hexdigest(),
                    "size": len(contents),
                }
            )
            if len(records) > MAX_ARTIFACT_EXPORT_PARTS:
                raise ValueError("artifact export archive exceeds the transport-part bound")
    if not records:
        raise ValueError("artifact export archive must produce at least one transport part")
    return records


def prepare_artifact_export(
    artifact_directory: Path,
    export_directory: Path,
    *,
    runner_exit_code: int,
    source_sha: str,
    source_tree: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Seal task-local evidence into bounded chunks before the Iris task exits."""
    if type(runner_exit_code) is not int or not 0 <= runner_exit_code <= 255:
        raise ValueError("runner exit code must be an integer in [0, 255]")
    source_sha = _require_sha(source_sha, SHA1_PATTERN, "source SHA")
    source_tree = _require_sha(source_tree, SHA1_PATTERN, "source tree")
    source_manifest_sha256 = _require_sha(source_manifest_sha256, SHA256_PATTERN, "source manifest SHA-256")
    artifact_directory = artifact_directory.absolute()
    export_directory = export_directory.absolute()
    if artifact_directory.is_symlink():
        raise ValueError("artifact directory must not be a symlink")
    if export_directory.exists():
        raise ValueError(f"artifact export directory already exists: {export_directory}")
    if export_directory.is_relative_to(artifact_directory) or artifact_directory.is_relative_to(export_directory):
        raise ValueError("artifact and export directories must not contain one another")

    records = _artifact_file_records(artifact_directory)
    accepted = next((record for record in records if record["path"] == "accepted_bundle.json"), None)
    wrapper_exit_code = runner_exit_code
    if runner_exit_code == 0 and accepted is None:
        wrapper_exit_code = MISSING_ACCEPTED_BUNDLE_EXIT_CODE
    export_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{export_directory.name}.", dir=export_directory.parent) as temp_name:
        temporary = Path(temp_name) / "export"
        temporary.mkdir()
        archive_path = temporary / ARTIFACT_EXPORT_ARCHIVE_FILENAME
        _write_artifact_archive(archive_path, artifact_directory, records)
        archive = {
            "filename": ARTIFACT_EXPORT_ARCHIVE_FILENAME,
            "sha256": _sha256(archive_path),
            "size": archive_path.stat().st_size,
        }
        parts = _split_artifact_archive(archive_path, temporary)
        manifest = {
            "accepted_bundle": accepted,
            "archive": archive,
            "members": records,
            "parts": parts,
            "runner_exit_code": runner_exit_code,
            "schema_version": ARTIFACT_EXPORT_SCHEMA_VERSION,
            "source": {
                "commit": source_sha,
                "manifest_sha256": source_manifest_sha256,
                "tree": source_tree,
            },
            "wrapper_exit_code": wrapper_exit_code,
        }
        manifest_path = temporary / ARTIFACT_EXPORT_MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json(manifest))
        if manifest_path.stat().st_size > MAX_ARTIFACT_EXPORT_MANIFEST_BYTES:
            raise ValueError("artifact export manifest exceeds the transport metadata bound")
        ready = {
            "accepted_bundle_present": accepted is not None,
            "archive_sha256": archive["sha256"],
            "archive_size": archive["size"],
            "manifest_sha256": _sha256(manifest_path),
            "part_count": len(parts),
            "ready_filename": ARTIFACT_EXPORT_READY_FILENAME,
            "runner_exit_code": runner_exit_code,
            "schema_version": ARTIFACT_EXPORT_SCHEMA_VERSION,
            "wrapper_exit_code": wrapper_exit_code,
        }
        ready_path = temporary / ARTIFACT_EXPORT_READY_FILENAME
        ready_path.write_bytes(_canonical_json(ready))
        temporary.replace(export_directory)
    return ready


def _validate_artifact_export_ready(ready: dict[str, Any]) -> None:
    if set(ready) != {
        "accepted_bundle_present",
        "archive_sha256",
        "archive_size",
        "manifest_sha256",
        "part_count",
        "ready_filename",
        "runner_exit_code",
        "schema_version",
        "wrapper_exit_code",
    }:
        raise ValueError("artifact export READY has unexpected fields")
    if type(ready.get("accepted_bundle_present")) is not bool:
        raise ValueError("artifact export READY accepted-bundle status must be boolean")
    if type(ready.get("archive_size")) is not int or not 0 < ready["archive_size"] <= MAX_ARTIFACT_EXPORT_BYTES:
        raise ValueError("artifact export READY archive size is invalid")
    if type(ready.get("part_count")) is not int or not 1 <= ready["part_count"] <= MAX_ARTIFACT_EXPORT_PARTS:
        raise ValueError("artifact export READY part count is invalid")
    if ready.get("ready_filename") != ARTIFACT_EXPORT_READY_FILENAME:
        raise ValueError("artifact export READY filename identity is invalid")
    for name in ("runner_exit_code", "wrapper_exit_code"):
        if type(ready.get(name)) is not int or not 0 <= ready[name] <= 255:
            raise ValueError(f"artifact export READY {name} is invalid")
    if type(ready.get("schema_version")) is not int or ready["schema_version"] != ARTIFACT_EXPORT_SCHEMA_VERSION:
        raise ValueError("artifact export READY has an unsupported schema")
    _require_sha(ready.get("archive_sha256"), SHA256_PATTERN, "archive SHA-256")
    _require_sha(ready.get("manifest_sha256"), SHA256_PATTERN, "export manifest SHA-256")


def artifact_export_ack(ready: dict[str, Any]) -> bytes:
    """Return the exact acknowledgement bytes for a validated export."""
    _validate_artifact_export_ready(ready)
    return _canonical_json(
        {
            "archive_sha256": _require_sha(ready.get("archive_sha256"), SHA256_PATTERN, "archive SHA-256"),
            "manifest_sha256": _require_sha(ready.get("manifest_sha256"), SHA256_PATTERN, "export manifest SHA-256"),
            "schema_version": ARTIFACT_EXPORT_SCHEMA_VERSION,
        }
    )


def _artifact_export_transport_bound(filename: str) -> int:
    if filename == ARTIFACT_EXPORT_READY_FILENAME:
        return MAX_ARTIFACT_EXPORT_READY_BYTES
    if filename == ARTIFACT_EXPORT_MANIFEST_FILENAME:
        return MAX_ARTIFACT_EXPORT_MANIFEST_BYTES
    if ARTIFACT_EXPORT_PART_PATTERN.fullmatch(filename):
        return ARTIFACT_EXPORT_CHUNK_BYTES
    raise ValueError(f"artifact export filename is not transportable: {filename!r}")


def read_artifact_export_record(export_directory: Path, filename: str, *, offset: int = 0) -> bytes:
    """Encode one bounded export-file window as canonical base64-safe text for Iris exec."""
    bound = _artifact_export_transport_bound(filename)
    if type(offset) is not int or offset < 0 or offset % ARTIFACT_EXPORT_CHUNK_BYTES != 0:
        raise ValueError("artifact export read offset must be a nonnegative chunk boundary")
    export_directory = export_directory.absolute()
    if export_directory.is_symlink() or not export_directory.is_dir():
        raise ValueError("artifact export directory must be a regular directory")
    path = export_directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact export file is missing or not regular: {filename}")
    total_size = path.stat().st_size
    if total_size > bound or offset >= total_size:
        raise ValueError(f"artifact export file exceeds its transport bound: {filename}")
    with path.open("rb") as stream:
        stream.seek(offset)
        contents = stream.read(ARTIFACT_EXPORT_CHUNK_BYTES)
    end = offset + len(contents)
    return _canonical_json(
        {
            "contents_base64": base64.b64encode(contents).decode("ascii"),
            "eof": end == total_size,
            "filename": filename,
            "offset": offset,
            "schema_version": ARTIFACT_EXPORT_SCHEMA_VERSION,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "size": len(contents),
            "total_size": total_size,
        }
    )


def decode_artifact_export_record(record_path: Path, retrieval_directory: Path) -> dict[str, Any]:
    """Strictly decode one controller-captured text record into a fresh retrieval file."""
    record_path = record_path.absolute()
    if (
        record_path.is_symlink()
        or not record_path.is_file()
        or record_path.stat().st_size > MAX_ARTIFACT_EXPORT_RECORD_BYTES
    ):
        raise ValueError("artifact export transport record is missing, non-regular, or oversized")
    raw = record_path.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or raw != _canonical_json(record):
        raise ValueError("artifact export transport record must use canonical JSON")
    if set(record) != {
        "contents_base64",
        "eof",
        "filename",
        "offset",
        "schema_version",
        "sha256",
        "size",
        "total_size",
    }:
        raise ValueError("artifact export transport record has unexpected fields")
    if type(record.get("schema_version")) is not int or record["schema_version"] != ARTIFACT_EXPORT_SCHEMA_VERSION:
        raise ValueError("artifact export transport record has an unsupported schema")
    filename = record.get("filename")
    if not isinstance(filename, str):
        raise ValueError("artifact export transport filename must be a string")
    bound = _artifact_export_transport_bound(filename)
    offset = record.get("offset")
    total_size = record.get("total_size")
    eof = record.get("eof")
    if (
        type(offset) is not int
        or offset < 0
        or offset % ARTIFACT_EXPORT_CHUNK_BYTES != 0
        or type(total_size) is not int
        or not 0 < total_size <= bound
        or type(eof) is not bool
    ):
        raise ValueError("artifact export transport window identity is invalid")
    encoded = record.get("contents_base64")
    if not isinstance(encoded, str):
        raise ValueError("artifact export contents must be base64 text")
    try:
        contents = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("artifact export contents are not canonical base64") from error
    if base64.b64encode(contents).decode("ascii") != encoded:
        raise ValueError("artifact export contents are not canonical base64")
    size = record.get("size")
    if type(size) is not int or not 0 < size <= ARTIFACT_EXPORT_CHUNK_BYTES or size != len(contents):
        raise ValueError("artifact export transport size is invalid")
    end = offset + size
    if end > total_size or eof is not (end == total_size) or (not eof and size != ARTIFACT_EXPORT_CHUNK_BYTES):
        raise ValueError("artifact export transport window is not contiguous with the declared file")
    if hashlib.sha256(contents).hexdigest() != _require_sha(
        record.get("sha256"), SHA256_PATTERN, "transport record SHA-256"
    ):
        raise ValueError("artifact export transport digest mismatch")
    retrieval_directory = retrieval_directory.absolute()
    if retrieval_directory.exists():
        if retrieval_directory.is_symlink() or not retrieval_directory.is_dir():
            raise ValueError("artifact retrieval directory must be a regular directory")
    else:
        retrieval_directory.mkdir(parents=True)
    output = retrieval_directory / filename
    partial = retrieval_directory / f".{filename}.partial"
    if output.exists() or output.is_symlink():
        raise ValueError(f"artifact retrieval output already exists: {filename}")
    if offset == 0:
        if partial.exists() or partial.is_symlink():
            raise ValueError(f"artifact retrieval partial output already exists: {filename}")
        partial.write_bytes(contents)
    else:
        if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != offset:
            raise ValueError("artifact export transport windows are missing, duplicated, or reordered")
        with partial.open("ab") as stream:
            stream.write(contents)
    if eof:
        partial.replace(output)
    return {
        "eof": eof,
        "filename": filename,
        "next_offset": end,
        "sha256": record["sha256"],
        "size": size,
        "total_size": total_size,
    }


def write_artifact_export_ack(
    export_directory: Path,
    *,
    archive_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically acknowledge only the exact READY identities supplied by the controller."""
    archive_sha256 = _require_sha(archive_sha256, SHA256_PATTERN, "archive SHA-256")
    manifest_sha256 = _require_sha(manifest_sha256, SHA256_PATTERN, "manifest SHA-256")
    export_directory = export_directory.absolute()
    ready_path = export_directory / ARTIFACT_EXPORT_READY_FILENAME
    if (
        ready_path.is_symlink()
        or not ready_path.is_file()
        or ready_path.stat().st_size > MAX_ARTIFACT_EXPORT_READY_BYTES
    ):
        raise ValueError("artifact export READY is missing, non-regular, or oversized")
    raw = ready_path.read_bytes()
    ready = json.loads(raw)
    if not isinstance(ready, dict) or raw != _canonical_json(ready):
        raise ValueError("artifact export READY must use canonical JSON")
    expected = artifact_export_ack(ready)
    if ready.get("archive_sha256") != archive_sha256 or ready.get("manifest_sha256") != manifest_sha256:
        raise ValueError("artifact acknowledgement does not bind the exact export identities")
    ack_path = export_directory / ARTIFACT_EXPORT_ACK_FILENAME
    if ack_path.exists() or ack_path.is_symlink():
        raise ValueError("artifact acknowledgement already exists")
    temporary = export_directory / f".{ARTIFACT_EXPORT_ACK_FILENAME}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("artifact acknowledgement temporary path already exists")
    temporary.write_bytes(expected)
    temporary.replace(ack_path)
    return {"ack_sha256": _sha256(ack_path), "archive_sha256": archive_sha256, "manifest_sha256": manifest_sha256}


def validate_retrieved_artifact_export(
    retrieval_directory: Path,
    archive_output: Path,
    *,
    expected_source_sha: str,
    expected_source_tree: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate controller-retrieved export parts and reconstruct the exact archive."""
    expected_source_sha = _require_sha(expected_source_sha, SHA1_PATTERN, "source SHA")
    expected_source_tree = _require_sha(expected_source_tree, SHA1_PATTERN, "source tree")
    expected_source_manifest_sha256 = _require_sha(
        expected_source_manifest_sha256, SHA256_PATTERN, "source manifest SHA-256"
    )
    retrieval_directory = retrieval_directory.absolute()
    archive_output = archive_output.absolute()
    if retrieval_directory.is_symlink() or not retrieval_directory.is_dir():
        raise ValueError("artifact retrieval directory must be a regular directory")
    if archive_output.exists() or archive_output.is_relative_to(retrieval_directory):
        raise ValueError("artifact archive output must be fresh and outside the retrieval directory")

    ready_path = retrieval_directory / ARTIFACT_EXPORT_READY_FILENAME
    manifest_path = retrieval_directory / ARTIFACT_EXPORT_MANIFEST_FILENAME
    for path, name in ((ready_path, "READY"), (manifest_path, "manifest")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact export {name} must be a regular file")
    if ready_path.stat().st_size > MAX_ARTIFACT_EXPORT_READY_BYTES:
        raise ValueError("artifact export READY exceeds the metadata size bound")
    if manifest_path.stat().st_size > MAX_ARTIFACT_EXPORT_MANIFEST_BYTES:
        raise ValueError("artifact export manifest exceeds the metadata size bound")
    ready = json.loads(ready_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(ready, dict) or ready_path.read_bytes() != _canonical_json(ready):
        raise ValueError("artifact export READY must use canonical JSON")
    if not isinstance(manifest, dict) or manifest_path.read_bytes() != _canonical_json(manifest):
        raise ValueError("artifact export manifest must use canonical JSON")
    _validate_artifact_export_ready(ready)
    if set(manifest) != {
        "accepted_bundle",
        "archive",
        "members",
        "parts",
        "runner_exit_code",
        "schema_version",
        "source",
        "wrapper_exit_code",
    }:
        raise ValueError("artifact export manifest has unexpected fields")
    if _sha256(manifest_path) != _require_sha(ready.get("manifest_sha256"), SHA256_PATTERN, "export manifest SHA-256"):
        raise ValueError("artifact export READY does not bind the retrieved manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != ARTIFACT_EXPORT_SCHEMA_VERSION:
        raise ValueError("artifact export manifest has an unsupported schema")
    expected_source = {
        "commit": expected_source_sha,
        "manifest_sha256": expected_source_manifest_sha256,
        "tree": expected_source_tree,
    }
    if manifest.get("source") != expected_source:
        raise ValueError("artifact export source identity differs from the authorized capsule")

    parts = manifest.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_ARTIFACT_EXPORT_PARTS:
        raise ValueError("artifact export has an invalid part count")
    expected_files = {ARTIFACT_EXPORT_READY_FILENAME, ARTIFACT_EXPORT_MANIFEST_FILENAME}
    total_size = 0
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_output.name}.", dir=archive_output.parent, delete=False
    ) as stream:
        temporary_archive = Path(stream.name)
        try:
            for index, record in enumerate(parts):
                expected_filename = f"{ARTIFACT_EXPORT_ARCHIVE_FILENAME}.part.{index:06d}"
                if not isinstance(record, dict) or set(record) != {"filename", "index", "sha256", "size"}:
                    raise ValueError("artifact export part record has unexpected fields")
                if (
                    record.get("filename") != expected_filename
                    or type(record.get("index")) is not int
                    or record["index"] != index
                ):
                    raise ValueError("artifact export parts are not canonical and contiguous")
                size = record.get("size")
                if type(size) is not int or not 0 < size <= ARTIFACT_EXPORT_CHUNK_BYTES:
                    raise ValueError("artifact export part has an invalid size")
                part = retrieval_directory / expected_filename
                if part.is_symlink() or not part.is_file() or part.stat().st_size != size:
                    raise ValueError("artifact export part is missing or has the wrong size")
                if _sha256(part) != _require_sha(record.get("sha256"), SHA256_PATTERN, "part SHA-256"):
                    raise ValueError("artifact export part digest mismatch")
                with part.open("rb") as part_stream:
                    shutil.copyfileobj(part_stream, stream)
                expected_files.add(expected_filename)
                total_size += size
                if total_size > MAX_ARTIFACT_EXPORT_BYTES:
                    raise ValueError("artifact export archive exceeds the size bound")
        except Exception:
            temporary_archive.unlink(missing_ok=True)
            raise
    actual_files = {path.name for path in retrieval_directory.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_files != expected_files or any(
        path.is_symlink() or not path.is_file() for path in retrieval_directory.iterdir()
    ):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact retrieval file set differs from the sealed export")

    archive = manifest.get("archive")
    if not isinstance(archive, dict) or set(archive) != {"filename", "sha256", "size"}:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export archive record has unexpected fields")
    if (
        archive.get("filename") != ARTIFACT_EXPORT_ARCHIVE_FILENAME
        or type(archive.get("size")) is not int
        or archive["size"] != total_size
    ):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export archive size or name mismatch")
    archive_sha256 = _require_sha(archive.get("sha256"), SHA256_PATTERN, "archive SHA-256")
    if _sha256(temporary_archive) != archive_sha256 or ready.get("archive_sha256") != archive_sha256:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export archive digest mismatch")
    if (
        type(ready.get("archive_size")) is not int
        or ready["archive_size"] != total_size
        or type(ready.get("part_count")) is not int
        or ready["part_count"] != len(parts)
    ):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export READY does not bind the archive layout")

    members = manifest.get("members")
    if not isinstance(members, list) or len(members) > MAX_ARTIFACT_EXPORT_MEMBERS:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export has an invalid member count")
    member_paths: list[str] = []
    try:
        with tarfile.open(temporary_archive, "r:gz") as archive_stream:
            tar_members = archive_stream.getmembers()
            if len(tar_members) != len(members):
                raise ValueError("artifact archive member count differs from the manifest")
            for record, member in zip(members, tar_members, strict=True):
                if not isinstance(record, dict) or set(record) != {"mode", "path", "sha256", "size"}:
                    raise ValueError("artifact export member record has unexpected fields")
                member_path = str(_validate_relative_path(record.get("path"), "artifact member path"))
                if member_path != member.name or not member.isfile():
                    raise ValueError("artifact archive member path or type differs from the manifest")
                if record.get("mode") not in {"0644", "0755"} or member.mode != int(record["mode"], 8):
                    raise ValueError("artifact archive member mode differs from the manifest")
                size = record.get("size")
                if type(size) is not int or size < 0 or member.size != size:
                    raise ValueError("artifact archive member size differs from the manifest")
                extracted = archive_stream.extractfile(member)
                if extracted is None:
                    raise ValueError("artifact archive member cannot be read")
                digest = hashlib.sha256()
                while block := extracted.read(1024 * 1024):
                    digest.update(block)
                if digest.hexdigest() != _require_sha(record.get("sha256"), SHA256_PATTERN, "member SHA-256"):
                    raise ValueError("artifact archive member digest differs from the manifest")
                member_paths.append(member_path)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    if member_paths != sorted(set(member_paths)):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact archive member paths are not sorted and unique")
    accepted = next((record for record in members if record["path"] == "accepted_bundle.json"), None)
    if (
        manifest.get("accepted_bundle") != accepted
        or type(ready.get("accepted_bundle_present")) is not bool
        or ready["accepted_bundle_present"] is not (accepted is not None)
    ):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export does not bind the accepted bundle record")
    runner_exit_code = manifest.get("runner_exit_code")
    wrapper_exit_code = manifest.get("wrapper_exit_code")
    if type(runner_exit_code) is not int or not 0 <= runner_exit_code <= 255:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export runner exit code is invalid")
    if type(wrapper_exit_code) is not int or not 0 <= wrapper_exit_code <= 255:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export wrapper exit code is invalid")
    expected_wrapper_exit = (
        MISSING_ACCEPTED_BUNDLE_EXIT_CODE if runner_exit_code == 0 and accepted is None else runner_exit_code
    )
    if wrapper_exit_code != expected_wrapper_exit:
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export wrapper exit code is inconsistent")
    if (
        type(ready.get("runner_exit_code")) is not int
        or ready["runner_exit_code"] != runner_exit_code
        or type(ready.get("wrapper_exit_code")) is not int
        or ready["wrapper_exit_code"] != wrapper_exit_code
    ):
        temporary_archive.unlink(missing_ok=True)
        raise ValueError("artifact export READY exit status is inconsistent")
    temporary_archive.replace(archive_output)
    return {"manifest": manifest, "ready": ready}


def wait_for_artifact_ack(
    export_directory: Path,
    ready: dict[str, Any],
    *,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
) -> None:
    """Keep the task alive until the controller acknowledges exact retrieved bytes."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("artifact acknowledgement timeout must be positive")
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0:
        raise ValueError("artifact acknowledgement polling interval must be positive")
    expected = artifact_export_ack(ready)
    ack_path = export_directory.absolute() / ARTIFACT_EXPORT_ACK_FILENAME
    deadline = time.monotonic() + timeout_seconds
    observed_mismatch = False
    while time.monotonic() < deadline:
        if ack_path.is_symlink():
            raise ValueError("artifact acknowledgement must be a regular file")
        if ack_path.is_file():
            if ack_path.read_bytes() == expected:
                return
            observed_mismatch = True
        time.sleep(poll_seconds)
    if observed_mismatch:
        raise ValueError("artifact acknowledgement does not bind the exact export identities")
    raise TimeoutError("timed out waiting for artifact export acknowledgement")


def capsule_python_path(source_root: Path, inherited: str | None) -> str:
    """Return the closed capsule import roots followed by any image paths."""
    roots = (str(source_root), str(source_root / "lib/tile_lifetime/src"))
    return os.pathsep.join((*roots, *((inherited,) if inherited else ())))


def verify_launcher_identity(path: Path, expected_sha256: str) -> None:
    """Reject execution when the transported stdlib launcher changed."""
    expected_sha256 = _require_sha(expected_sha256, SHA256_PATTERN, "launcher SHA-256")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("source capsule launcher differs from the trusted launch identity")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--source-sha", required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    verify = subparsers.add_parser("verify-export")
    verify.add_argument("--retrieval-directory", type=Path, required=True)
    verify.add_argument("--archive-output", type=Path, required=True)
    verify.add_argument("--ack-output", type=Path, required=True)
    verify.add_argument("--source-sha", required=True)
    verify.add_argument("--source-tree", required=True)
    verify.add_argument("--source-manifest-sha256", required=True)
    read_export = subparsers.add_parser("read-export")
    read_export.add_argument("--launcher-sha256", required=True)
    read_export.add_argument("--export-directory", type=Path, required=True)
    read_export.add_argument("--filename", required=True)
    read_export.add_argument("--offset", type=int, required=True)
    decode_export = subparsers.add_parser("decode-export-record")
    decode_export.add_argument("--record", type=Path, required=True)
    decode_export.add_argument("--retrieval-directory", type=Path, required=True)
    ack_export = subparsers.add_parser("ack-export")
    ack_export.add_argument("--launcher-sha256", required=True)
    ack_export.add_argument("--export-directory", type=Path, required=True)
    ack_export.add_argument("--archive-sha256", required=True)
    ack_export.add_argument("--manifest-sha256", required=True)
    for command in ("run", "run-export"):
        run = subparsers.add_parser(command)
        run.add_argument("--workspace", type=Path, required=True)
        run.add_argument("--launcher-sha256", required=True)
        run.add_argument("--manifest-sha256", required=True)
        run.add_argument("--source-sha", required=True)
        run.add_argument("--source-tree", required=True)
        run.add_argument("--artifact-directory", type=Path, required=True)
        run.add_argument("--require-jax-version", required=True)
        run.add_argument("--runtime-python", type=Path, required=True)
        if command == "run-export":
            run.add_argument("--export-directory", type=Path, required=True)
            run.add_argument("--ack-timeout-seconds", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.command == "prepare":
        print(
            json.dumps(prepare_source_payload(args.source_root, args.source_sha, args.output_directory), sort_keys=True)
        )
        return
    if args.command == "verify-export":
        validated = validate_retrieved_artifact_export(
            args.retrieval_directory,
            args.archive_output,
            expected_source_sha=args.source_sha,
            expected_source_tree=args.source_tree,
            expected_source_manifest_sha256=args.source_manifest_sha256,
        )
        ack_output = args.ack_output.resolve()
        if ack_output.exists():
            raise ValueError(f"artifact acknowledgement output already exists: {ack_output}")
        ack_output.parent.mkdir(parents=True, exist_ok=True)
        ack_output.write_bytes(artifact_export_ack(validated["ready"]))
        print(
            json.dumps(
                {
                    "ack_sha256": _sha256(ack_output),
                    "archive_sha256": validated["ready"]["archive_sha256"],
                    "manifest_sha256": validated["ready"]["manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "decode-export-record":
        print(json.dumps(decode_artifact_export_record(args.record, args.retrieval_directory), sort_keys=True))
        return
    verify_launcher_identity(Path(__file__).resolve(), args.launcher_sha256)
    if args.command == "read-export":
        sys.stdout.buffer.write(read_artifact_export_record(args.export_directory, args.filename, offset=args.offset))
        sys.stdout.buffer.flush()
        return
    if args.command == "ack-export":
        print(
            json.dumps(
                write_artifact_export_ack(
                    args.export_directory,
                    archive_sha256=args.archive_sha256,
                    manifest_sha256=args.manifest_sha256,
                ),
                sort_keys=True,
            )
        )
        return
    workspace = args.workspace.resolve()
    source_root, _manifest = restore_and_verify_source_payload(
        workspace,
        expected_manifest_sha256=args.manifest_sha256,
        expected_source_sha=args.source_sha,
        expected_source_tree=args.source_tree,
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = capsule_python_path(source_root, os.environ.get("PYTHONPATH"))
    arguments = runner_arguments(
        args.runtime_python,
        source_root,
        args.source_sha,
        args.source_tree,
        workspace / MANIFEST_FILENAME,
        args.manifest_sha256,
        args.artifact_directory.resolve(),
        args.require_jax_version,
    )
    if args.command == "run":
        os.execv(arguments[0], arguments)
    export_directory = args.export_directory.resolve()
    if export_directory.exists():
        raise ValueError(f"artifact export directory already exists: {export_directory}")
    completed = subprocess.run(arguments, check=False)
    runner_exit_code = completed.returncode if completed.returncode >= 0 else min(255, 128 - completed.returncode)
    ready = prepare_artifact_export(
        args.artifact_directory,
        export_directory,
        runner_exit_code=runner_exit_code,
        source_sha=args.source_sha,
        source_tree=args.source_tree,
        source_manifest_sha256=args.manifest_sha256,
    )
    print(json.dumps({"export_ready": ready}, sort_keys=True), flush=True)
    wait_for_artifact_ack(export_directory, ready, timeout_seconds=args.ack_timeout_seconds)
    raise SystemExit(ready["wrapper_exit_code"])


if __name__ == "__main__":
    main()
