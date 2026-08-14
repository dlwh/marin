# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed raw-evidence schemas for the non-authoritative ABI15 H100 checkpoint."""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from abi15_h100_architecture_contract import (
    CONTRACT_SHA256 as ARCHITECTURE_BUILD_CONTRACT_SHA256,
)
from abi15_h100_architecture_contract import (
    load_architecture_build_contract,
)
from abi15_h100_composed_contract import CONTRACT_SHA256, FIXTURE_SHA256, load_contract

RAW_MANIFEST_FILENAME = "raw-artifact-manifest.json"
ROLE_MANIFEST_FILENAME = "abi15-h100-composed-role-paths-v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_STATIC_ROLES = {
    "architecture_build_contract": "contract/architecture-build.json",
    "baseline_report": "process/baseline.json",
    "build_manifest": "build/build-manifest.json",
    "cache_populate_after": "cache/populate-after.json",
    "cache_populate_before": "cache/populate-before.json",
    "cache_reuse_after": "cache/reuse-after.json",
    "cache_reuse_before": "cache/reuse-before.json",
    "cuda_pjrt_plugin": "runtime/xla-cuda-plugin.so",
    "cuda_pjrt_record": "runtime/cuda-pjrt-RECORD",
    "cuda_pjrt_wheel": "runtime/jax-cuda13-pjrt.whl",
    "cuda_plugin_extension": "runtime/cuda-plugin-extension.so",
    "cuda_plugin_record": "runtime/cuda-plugin-RECORD",
    "cuda_plugin_wheel": "runtime/jax-cuda13-plugin.whl",
    "fixture": "contract/fixture.mlir",
    "jax_common": "runtime/libjax-common.so",
    "jax_extension": "runtime/jax-extension.so",
    "jaxlib_wheel": "runtime/jaxlib.whl",
    "populate_report": "process/populate.json",
    "prerun_contract": "contract/prerun.json",
    "ptxas_records": "assembly/ptxas-records.json",
    "registration_transcript": "runtime/registration.json",
    "reuse_report": "process/reuse.json",
}


@dataclass(frozen=True)
class RoleManifest:
    """Closed generators for the exact raw artifact role roster."""

    policies: tuple[str, ...]
    datasets: tuple[str, ...]
    output_roles: tuple[str, ...]
    populate_runs: tuple[str, ...]
    reuse_runs: tuple[str, ...]
    calls: tuple[tuple[str, tuple[int, ...]], ...]
    static_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ArtifactFile:
    """One verified regular file in the raw subject artifact."""

    role: str
    path: PurePosixPath
    size: int
    sha256: str


@dataclass(frozen=True)
class RawArtifact:
    """One closed subject artifact with no acceptance claim."""

    root: Path
    manifest_sha256: str
    files: Mapping[str, ArtifactFile]


@dataclass(frozen=True)
class ProcessReport:
    """Stable process and cache transition fields emitted by one worker."""

    process: str
    pid: int
    cache_hits: int
    observer_events: int
    cache_before: tuple[tuple[str, str], ...]
    cache_after: tuple[tuple[str, str], ...]
    runtime_binaries: tuple[tuple[str, str], ...]
    runtime_mappings: tuple[tuple[str, str, int, int], ...]


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


def _require_sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_revision(value: object, name: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full Git SHA-1")
    return value


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} must use the closed schema")
    return value


def _strings(value: object, expected: tuple[str, ...], name: str) -> tuple[str, ...]:
    if type(value) is not list or tuple(value) != expected:
        raise ValueError(f"{name} differs from the closed ordered values")
    return expected


def _relative_path(value: object, name: str) -> PurePosixPath:
    if type(value) is not str:
        raise ValueError(f"{name} must be a relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path


def load_role_manifest(path: Path | None = None) -> RoleManifest:
    """Load and semantically close the ABI15 role generator manifest."""
    selected = Path(__file__).with_name(ROLE_MANIFEST_FILENAME) if path is None else path
    raw = selected.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("ABI15 role manifest is not JSON") from error
    if raw != _canonical_json(document):
        raise ValueError("ABI15 role manifest must use canonical JSON")
    root = _closed(
        document,
        {
            "calls",
            "datasets",
            "output_roles",
            "policies",
            "populate_runs",
            "reuse_runs",
            "schema_version",
            "static_roles",
            "status",
        },
        "role manifest",
    )
    if root["schema_version"] != 1 or root["status"] != "raw_subject_artifact_only":
        raise ValueError("ABI15 role manifest schema or status drift")
    policies = _strings(root["policies"], ("source_ordered", "fast"), "policies")
    datasets = _strings(root["datasets"], ("dense", "sparse_policy_discriminator"), "datasets")
    outputs = _strings(root["output_roles"], ("primal", "lhs_cotangent", "rhs_cotangent"), "outputs")
    populate_runs = _strings(root["populate_runs"], ("first", "sequential", "thread_0", "thread_1"), "runs")
    reuse_runs = _strings(root["reuse_runs"], ("reuse",), "reuse runs")
    calls = _closed(root["calls"], {"forward", "backward"}, "calls")
    if calls != {"backward": list(range(10)), "forward": [0, 1]}:
        raise ValueError("ABI15 call ordinals drift")
    static = root["static_roles"]
    if static != _STATIC_ROLES or list(static) != sorted(static):
        raise ValueError("ABI15 static roles differ from the exact sorted mapping")
    normalized = tuple(
        (role, _relative_path(relative, f"static role {role}").as_posix()) for role, relative in static.items()
    )
    if len({path for _, path in normalized}) != len(normalized) or "accepted_bundle" in static:
        raise ValueError("ABI15 static roles repeat a path or include an acceptance claim")
    return RoleManifest(
        policies,
        datasets,
        outputs,
        populate_runs,
        reuse_runs,
        (("forward", (0, 1)), ("backward", tuple(range(10)))),
        normalized,
    )


def expected_role_paths(manifest: RoleManifest) -> dict[str, str]:
    """Expand the exact role and relative-path roster."""
    roles = dict(manifest.static_roles)
    for dataset in manifest.datasets:
        for input_role in ("lhs", "rhs", "output_cotangent"):
            roles[f"input_{dataset}_{input_role}"] = f"inputs/{dataset}/{input_role}.bf16"
        for output in manifest.output_roles:
            roles[f"baseline_{dataset}_{output}"] = f"baseline/{dataset}/{output}.bf16"
    for policy in manifest.policies:
        roles[f"compiled_mlir_{policy}"] = f"compiler/{policy}/generated.mlir"
        roles[f"observer_populate_{policy}"] = f"observer/{policy}/populate.jsonl"
        roles[f"observer_reuse_{policy}"] = f"observer/{policy}/reuse.jsonl"
        roles[f"cache_entry_{policy}"] = f"cache/entries/{policy}.bin"
        for call, ordinals in manifest.calls:
            roles[f"transport_{policy}_{call}"] = f"compiler/{policy}/{call}.bin"
            roles[f"descriptor_{policy}_{call}"] = f"compiler/{policy}/{call}.json"
            for ordinal in ordinals:
                roles[f"ptx_{policy}_{call}_{ordinal:02d}"] = f"assembly/{policy}/{call}/entry-{ordinal:02d}.ptx"
                roles[f"cubin_{policy}_{call}_{ordinal:02d}"] = f"assembly/{policy}/{call}/entry-{ordinal:02d}.cubin"
        for dataset in manifest.datasets:
            for output in manifest.output_roles:
                roles[f"reference_{policy}_{dataset}_{output}"] = f"reference/{policy}/{dataset}/{output}.bf16"
            for run in (*manifest.populate_runs, *manifest.reuse_runs):
                for output in manifest.output_roles:
                    roles[f"output_{policy}_{dataset}_{run}_{output}"] = (
                        f"outputs/{policy}/{dataset}/{run}/{output}.bf16"
                    )
    if "accepted_bundle" in roles or len(set(roles.values())) != len(roles):
        raise ValueError("ABI15 role expansion contains acceptance or path aliasing")
    return dict(sorted(roles.items()))


def validate_unsealed_prerun_contract() -> dict[str, Any]:
    """Return the immutable local contract only while hardware authority is null."""
    load_contract()
    path = (
        Path(__file__).parents[4] / ".agents/projects/tile_lifetime_compiler/shuttle_abi15_h100_composed_prerun_v1.json"
    )
    document = json.loads(path.read_bytes())
    authority = document.get("hardware_authority")
    if authority != {
        "image_digest": None,
        "launch_authorization_sha256": None,
        "source_capsule_sha256": None,
        "status": "not_authorized",
    }:
        raise ValueError("ABI15 local contract must remain explicitly unsealed")
    return document


def require_sealed_hardware_authority() -> None:
    """Stop the pre-image harness until a later checkpoint seals authority."""
    document = validate_unsealed_prerun_contract()
    if document["hardware_authority"]["status"] == "not_authorized":
        raise RuntimeError("ABI15 H100 launch authority is explicitly unsealed")
    raise RuntimeError("ABI15 immutable pre-run contract cannot itself authorize a launch")


def seal_raw_artifact(
    root: Path,
    role_paths: Mapping[str, str],
    *,
    harness_commit: str,
    harness_tree: str,
) -> str:
    """Write a raw subject manifest without creating an acceptance claim."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("raw artifact root must be a real directory")
    commit = _require_revision(harness_commit, "harness commit")
    tree = _require_revision(harness_tree, "harness tree")
    expected = expected_role_paths(load_role_manifest())
    if dict(role_paths) != expected:
        raise ValueError("raw artifact roles differ from the closed ABI15 roster")
    manifest_path = root / RAW_MANIFEST_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("raw artifact manifest already exists")
    files = []
    for role, relative in expected.items():
        path = root / _relative_path(relative, f"role {role}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"raw artifact role is missing or non-regular: {role}")
        files.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    document = {
        "files": files,
        "harness": {"commit": commit, "tree": tree},
        "prerun_contract_sha256": CONTRACT_SHA256,
        "schema_version": 1,
        "status": "raw_subject_artifact_only",
    }
    manifest_path.write_bytes(_canonical_json(document))
    return _sha256_file(manifest_path)


def load_raw_artifact(root: Path, *, expected_harness_commit: str, expected_harness_tree: str) -> RawArtifact:
    """Validate every raw artifact byte and the exact closed role roster."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("raw artifact root must be a real directory")
    manifest_path = root / RAW_MANIFEST_FILENAME
    raw = manifest_path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical_json(document):
        raise ValueError("raw artifact manifest must use canonical JSON")
    manifest = _closed(
        document,
        {"files", "harness", "prerun_contract_sha256", "schema_version", "status"},
        "raw artifact manifest",
    )
    if manifest["schema_version"] != 1 or manifest["status"] != "raw_subject_artifact_only":
        raise ValueError("raw artifact status cannot claim acceptance")
    if manifest["prerun_contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("raw artifact contract identity drift")
    expected_harness = {
        "commit": _require_revision(expected_harness_commit, "expected harness commit"),
        "tree": _require_revision(expected_harness_tree, "expected harness tree"),
    }
    if manifest["harness"] != expected_harness:
        raise ValueError("raw artifact harness identity drift")
    records = manifest["files"]
    if type(records) is not list:
        raise ValueError("raw artifact files must be a list")
    expected = expected_role_paths(load_role_manifest())
    files = {}
    covered = {RAW_MANIFEST_FILENAME}
    for record_value in records:
        record = _closed(record_value, {"path", "role", "sha256", "size"}, "raw artifact file")
        role = record["role"]
        if type(role) is not str or role in files or expected.get(role) != record["path"]:
            raise ValueError("raw artifact contains an unknown, duplicate, or remapped role")
        relative = _relative_path(record["path"], f"role {role}")
        path = root / relative
        size = record["size"]
        digest = _require_sha(record["sha256"], f"role {role} digest")
        if type(size) is not int or size < 0 or path.is_symlink() or not path.is_file():
            raise ValueError(f"raw artifact role is missing or invalid: {role}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"raw artifact role bytes drifted: {role}")
        files[role] = ArtifactFile(role, relative, size, digest)
        covered.add(relative.as_posix())
    if set(files) != set(expected):
        raise ValueError("raw artifact role roster drift")
    all_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("raw artifact rejects symlinks")
    actual = {path.relative_to(root).as_posix() for path in all_paths if path.is_file()}
    if actual != covered:
        raise ValueError("raw artifact contains an unmanifested or missing file")
    expected_directories = {
        parent.as_posix()
        for relative in covered
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    actual_directories = {path.relative_to(root).as_posix() for path in all_paths if path.is_dir()}
    if actual_directories != expected_directories:
        raise ValueError("raw artifact contains an empty or unmanifested directory")
    if (
        files["prerun_contract"].sha256 != CONTRACT_SHA256
        or files["fixture"].sha256 != FIXTURE_SHA256
        or files["architecture_build_contract"].sha256 != ARCHITECTURE_BUILD_CONTRACT_SHA256
    ):
        raise ValueError("raw artifact changed an immutable contract or fixture")
    load_architecture_build_contract(root / files["architecture_build_contract"].path)
    return RawArtifact(root.resolve(), _sha256_bytes(raw), files)


def validate_process_reports(reports: tuple[ProcessReport, ...]) -> None:
    """Validate the baseline/populate/reuse process and cache state machine."""
    if tuple(report.process for report in reports) != ("baseline", "populate", "reuse"):
        raise ValueError("process reports must be baseline, populate, reuse")
    if any(type(report.pid) is not int or report.pid <= 0 for report in reports):
        raise ValueError("process reports require positive integer PIDs")
    if len({report.pid for report in reports}) != 3:
        raise ValueError("baseline, populate, and reuse must use distinct processes")
    baseline, populate, reuse = reports
    expected_keys = ("fast", "source_ordered")
    if baseline.cache_hits != 0 or baseline.observer_events != 0 or baseline.cache_before or baseline.cache_after:
        raise ValueError("baseline must not use Shuttle observer or persistent cache")
    if populate.cache_hits != 0 or populate.observer_events != 6 or populate.cache_before:
        raise ValueError("populate must start empty, miss twice, and record six observer phases")
    if tuple(key for key, _ in populate.cache_after) != expected_keys:
        raise ValueError("populate must create exactly the FAST and SOURCE_ORDERED cache entries")
    if reuse.cache_hits != 2 or reuse.observer_events != 0:
        raise ValueError("reuse must hit twice without observer activity")
    if reuse.cache_before != populate.cache_after or reuse.cache_after != populate.cache_after:
        raise ValueError("reuse must preserve the exact populated cache bytes")
    for report in reports:
        for _, digest in (*report.cache_before, *report.cache_after):
            _require_sha(digest, "cache entry digest")
    runtime_keys = (
        "cuda_pjrt_plugin_sha256",
        "cuda_pjrt_record_sha256",
        "cuda_plugin_extension_sha256",
        "cuda_plugin_record_sha256",
        "jax_common_sha256",
        "jax_extension_sha256",
    )
    identities = [report.runtime_binaries for report in reports]
    if any(tuple(key for key, _ in identity) != runtime_keys for identity in identities) or len(set(identities)) != 1:
        raise ValueError("process reports must bind the same exact loaded runtime bytes")
    for _, digest in identities[0]:
        _require_sha(digest, "loaded runtime digest")
    mapping_roles = ("cuda_pjrt_plugin", "cuda_plugin_extension", "jax_common", "jax_extension")
    for report in reports:
        if tuple(item[0] for item in report.runtime_mappings) != mapping_roles:
            raise ValueError("process report loaded mapping role roster drift")
        for role, path, start, end in report.runtime_mappings:
            if not path.startswith("/") or type(start) is not int or type(end) is not int or not 0 < start < end:
                raise ValueError(f"process report loaded mapping range drift: {role}")
    mapping_paths = [tuple((role, path) for role, path, _, _ in report.runtime_mappings) for report in reports]
    if len(set(mapping_paths)) != 1:
        raise ValueError("process reports loaded different runtime mapping paths")
    expected_names = {
        "cuda_pjrt_plugin": "xla_cuda_plugin.so",
        "cuda_plugin_extension": "cuda_plugin_extension.so",
        "jax_common": "libjax_common.so",
        "jax_extension": "_jax.so",
    }
    if any(Path(path).name != expected_names[role] for role, path in mapping_paths[0]):
        raise ValueError("process report loaded mapping filename drift")
