# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authority-separated evidence bundle for the ABI 10 H100 forward cell."""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import ml_dtypes
import numpy as np
from acceptance_contract import EVENT_FIELDS, ObserverIdentity
from gpu_transport import GpuTransport, load_gpu_transport
from target1_abi10_h100_forward_contract import (
    CANONICAL_OPTIONS,
    FINAL_MODULE_SHA256,
    GENERATED_BUNDLE_SHA256,
    INPUT_ARRAY_DIGESTS,
    INPUT_RAW_SHA256,
    REFERENCE_LIMITS,
    REFERENCE_OUTPUT_ARRAY_DIGEST,
    TRANSFORMER_ENGINE_COMMIT,
    TUNING_SHA256,
)
from target1_acceptance_contract import target1_expectation
from target1_numerical_oracle import array_digest, error_metrics
from target1_te_oracle_harness import validate_result

AUTHORIZATION_PATH = Path(__file__).with_name("target1-abi10-h100-forward-authorization-v1.json")
EXPECTED_AUTHORIZATION_SHA256 = "UNSEALED"
AUTHORIZATION_ID = "target1_abi10_h100_forward_authorization_v1"
ARTIFACT_MANIFEST = "artifact-manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX16 = re.compile(r"[0-9a-f]{16}\Z")
JOB_ID = re.compile(r"/[a-z0-9_-]+/abi10-h100-[0-9a-f]{12}-(subject|verify)-[0-9a-f]{12}\Z")
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "shuttle-kernels"
WANDB_ARTIFACT_TYPE = "shuttle-h100-evidence"


@dataclass(frozen=True)
class ArtifactFile:
    """One regular file covered by the closed evidence manifest."""

    role: str
    path: PurePosixPath
    size: int
    sha256: str


@dataclass(frozen=True)
class Authorization:
    """Exact checked-in pre-hardware authorization bytes and decoded fields."""

    path: Path
    sha256: str
    canonical_json: str

    def decoded(self) -> dict[str, Any]:
        """Return a new validated copy so callers cannot mutate the trust root."""
        return _validate_authorization_document(_parse_json(self.canonical_json))


@dataclass(frozen=True)
class ArtifactBundle:
    """Verified local bytes downloaded from one exact W&B artifact version."""

    root: Path
    manifest_sha256: str
    files: Mapping[str, ArtifactFile]
    collector_job_id: str

    def path_for(self, role: str) -> Path:
        """Return the verified path for a manifest role."""
        record = self.files[role]
        path = self.root / record.path
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.size
            or _file_sha256(path) != record.sha256
        ):
            raise ValueError(f"artifact role bytes changed after manifest validation: {role}")
        return path


@dataclass(frozen=True)
class IrisJobReceipt:
    """Controller-exported launch and attempt identity for one completed job."""

    cluster: str
    job_id: str
    submitting_user: str
    bundle_id: str
    authorization_sha256: str
    entrypoint: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    task_image: str
    gpu_count: int
    gpu_variant: str
    replicas: int
    existing_job_policy: str
    fail_if_exists: bool
    submitted_at_ms: int
    started_at_ms: int
    finished_at_ms: int
    state: str
    attempt_id: int
    attempt_uid: str
    worker_id: str
    submission_log_seq: int
    postseal_log_seq: int
    postseal_log_json: str


@dataclass(frozen=True)
class WandbArtifactReceipt:
    """Exact immutable-version W&B artifact identity observed by the verifier."""

    entity: str
    project: str
    name: str
    version: str
    digest: str
    manifest_sha256: str


OUTPUT_ROLES = (
    "ordinary_jax_output",
    "shuttle_sequential_0",
    "shuttle_sequential_1",
    "shuttle_sequential_2",
    "shuttle_concurrent_0",
    "shuttle_concurrent_1",
    "shuttle_reuse_check",
)
REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "input_x",
        "input_gamma",
        "reference_output",
        *OUTPUT_ROLES,
        "gpu_transport",
        "decoded_descriptor",
        "ptxas_records",
        "jax_extension",
        "jax_common",
        "cuda_plugin_extension",
        "registration_transcript",
        "observer_populate",
        "observer_reuse",
        "baseline_report",
        "cache_populate_report",
        "cache_reuse_report",
        "cache_entry",
        "gpu_build_manifest",
        "build_manifest",
        "te_run_plan",
        "te_execution_identity",
        *(
            f"te_case_{shape}_{role}"
            for shape in ("2048x4096", "7x13")
            for role in ("x", "gamma", "dy", "reference_y", "reference_dx", "reference_dgamma")
        ),
        "te_runner",
        "te_harness",
        "te_result_0",
        "te_result_1",
        "te_result_2",
        "te_result_3",
        *(f"ptx_{index:02d}" for index in range(19)),
        *(f"cubin_{index:02d}" for index in range(19)),
    },
)


def load_authorization() -> Authorization:
    """Load only the independently reviewed, digest-pinned authorization."""
    if EXPECTED_AUTHORIZATION_SHA256 == "UNSEALED":
        raise RuntimeError("the H100 authorization has not been sealed for hardware allocation")
    return _load_authorization(AUTHORIZATION_PATH, EXPECTED_AUTHORIZATION_SHA256)


def load_iris_receipt(path: Path) -> IrisJobReceipt:
    """Load one closed controller/log export without trusting annotations."""
    value = _closed(
        _load_json(path),
        {
            "cluster",
            "job_id",
            "submitting_user",
            "bundle_id",
            "authorization_sha256",
            "entrypoint",
            "environment",
            "task_image",
            "gpu_count",
            "gpu_variant",
            "replicas",
            "existing_job_policy",
            "fail_if_exists",
            "submitted_at_ms",
            "started_at_ms",
            "finished_at_ms",
            "state",
            "attempt_id",
            "attempt_uid",
            "worker_id",
            "submission_log_seq",
            "postseal_log_seq",
            "postseal_log_json",
        },
        "Iris receipt",
    )
    entrypoint = _sequence(value.pop("entrypoint"), "Iris receipt.entrypoint")
    environment_value = _sequence(value.pop("environment"), "Iris receipt.environment")
    environment = []
    for index, item in enumerate(environment_value):
        pair = _sequence(item, f"Iris receipt.environment[{index}]")
        if len(pair) != 2 or any(type(part) is not str for part in pair):
            raise ValueError("Iris receipt environment entries must be string pairs")
        environment.append((pair[0], pair[1]))
    return IrisJobReceipt(
        entrypoint=tuple(entrypoint),
        environment=tuple(environment),
        **value,
    )


def load_wandb_receipt(path: Path) -> WandbArtifactReceipt:
    """Load the exact artifact-version response exported by the collector."""
    value = _closed(
        _load_json(path),
        {"entity", "project", "name", "version", "digest", "manifest_sha256"},
        "W&B receipt",
    )
    return WandbArtifactReceipt(**value)


def _load_authorization(path: Path, expected_sha256: str) -> Authorization:
    actual = _file_sha256(path)
    _equal(actual, _sha(expected_sha256, "expected authorization SHA-256"), "authorization file SHA-256")
    canonical_json = path.read_text()
    _validate_authorization_document(_parse_json(canonical_json))
    return Authorization(path.resolve(), actual, canonical_json)


def _validate_authorization_document(document: dict[str, Any]) -> dict[str, Any]:
    root = _closed(
        document,
        {
            "schema_version",
            "authorization_id",
            "status",
            "challenge",
            "evidence_revision",
            "evidence_tree",
            "coordinate",
            "source",
            "compiler",
            "inputs",
            "reference",
            "generated_bundle",
            "toolchain",
            "collectors",
            "iris",
            "wandb",
            "claims",
        },
        "authorization",
    )
    _equal(root["schema_version"], 1, "authorization.schema_version")
    _equal(root["authorization_id"], AUTHORIZATION_ID, "authorization.authorization_id")
    _equal(root["status"], "sealed_before_hardware_allocation", "authorization.status")
    _sha(root["challenge"], "authorization.challenge")
    _revision(root["evidence_revision"], "authorization.evidence_revision")
    _revision(root["evidence_tree"], "authorization.evidence_tree")
    _equal(
        root["coordinate"],
        {
            "target_id": 1,
            "boundary": "forward",
            "shape": [2048, 4096],
            "shape_id": "rmsnorm_bf16_2048x4096",
            "policy": "source_ordered",
            "hardware_class": "h100",
        },
        "authorization.coordinate",
    )
    _validate_authorized_source(root["source"])
    _validate_authorized_collectors(root["collectors"])
    _validate_authorized_compiler(root["compiler"])
    _validate_authorized_inputs(root["inputs"])
    _validate_authorized_reference(root["reference"])
    _validate_authorized_bundle(root["generated_bundle"])
    _validate_authorized_toolchain(root["toolchain"])
    _validate_authorized_iris(root["iris"], root["challenge"])
    _equal(
        root["wandb"],
        {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "artifact_name": "target1-abi10-h100-forward-" + root["challenge"][:12],
            "artifact_type": WANDB_ARTIFACT_TYPE,
            "exact_version_required": True,
        },
        "authorization.wandb",
    )
    _equal(
        root["claims"],
        {
            "timing_permitted": False,
            "offline_json_can_accept": False,
            "required_receipts": ["iris_subject", "wandb_exact_version", "iris_verifier"],
            "maximum_claim": "controller_observed_content_addressed_independently_revalidated_externally_exported",
        },
        "authorization.claims",
    )
    return root


def authorization_sha256() -> str:
    """Return the digest embedded into the Iris job ID and argv."""
    return load_authorization().sha256


def load_artifact_bundle(root: Path, authorization: Authorization) -> ArtifactBundle:
    """Verify every byte in one downloaded exact-version artifact directory."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence artifact root must be a real directory")
    manifest_path = root / ARTIFACT_MANIFEST
    manifest = _closed(
        _load_json(manifest_path),
        {
            "schema_version",
            "authorization_id",
            "authorization_sha256",
            "challenge",
            "collector_job_id",
            "files",
        },
        "artifact manifest",
    )
    _equal(manifest["schema_version"], 1, "artifact manifest.schema_version")
    document = authorization.decoded()
    _equal(manifest["authorization_id"], document["authorization_id"], "artifact manifest.authorization_id")
    _equal(manifest["authorization_sha256"], authorization.sha256, "artifact manifest.authorization_sha256")
    _equal(manifest["challenge"], document["challenge"], "artifact manifest.challenge")
    _job_id(manifest["collector_job_id"], document, "artifact manifest.collector_job_id")
    records = _sequence(manifest["files"], "artifact manifest.files")
    if not records:
        raise ValueError("artifact manifest must cover at least one file")
    files: dict[str, ArtifactFile] = {}
    covered_paths = {ARTIFACT_MANIFEST}
    for index, value in enumerate(records):
        name = f"artifact manifest.files[{index}]"
        record = _closed(value, {"role", "path", "size", "sha256"}, name)
        role = _nonempty(record["role"], f"{name}.role")
        relative = _relative_path(record["path"], f"{name}.path")
        if role in files or relative.as_posix() in covered_paths:
            raise ValueError("artifact manifest contains a duplicate role or path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact file is missing or not regular: {relative}")
        size = _positive_integer(record["size"], f"{name}.size")
        digest = _sha(record["sha256"], f"{name}.sha256")
        if path.stat().st_size != size or _file_sha256(path) != digest:
            raise ValueError(f"artifact file bytes differ from manifest: {relative}")
        files[role] = ArtifactFile(role, relative, size, digest)
        covered_paths.add(relative.as_posix())
    all_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("artifact directory must not contain symlinks")
    actual_paths = {path.relative_to(root).as_posix() for path in all_paths if path.is_file()}
    if actual_paths != covered_paths:
        raise ValueError("artifact directory contains an unmanifested or missing file")
    if files.keys() != REQUIRED_ARTIFACT_ROLES:
        raise ValueError("artifact manifest role roster differs from the sealed evidence protocol")
    return ArtifactBundle(root.resolve(), _file_sha256(manifest_path), files, manifest["collector_job_id"])


def write_artifact_manifest(
    root: Path,
    collector_job_id: str,
    role_paths: Mapping[str, str],
    authorization: Authorization,
) -> ArtifactBundle:
    """Seal a complete raw artifact directory before its one W&B upload."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence artifact root must be a real directory")
    manifest_path = root / ARTIFACT_MANIFEST
    if manifest_path.exists():
        raise ValueError("evidence artifact manifest already exists")
    document = authorization.decoded()
    _job_id(collector_job_id, document, "artifact collector job ID")
    if "-subject-" not in collector_job_id:
        raise ValueError("artifact collector job must be the subject role")
    if type(role_paths) is not dict or set(role_paths) != REQUIRED_ARTIFACT_ROLES:
        raise ValueError("artifact role map differs from the sealed evidence role roster")
    files = []
    observed_paths = set()
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        relative_path = _relative_path(role_paths[role], f"artifact role path {role}")
        relative = relative_path.as_posix()
        if relative in observed_paths:
            raise ValueError("artifact role map contains a duplicate path")
        observed_paths.add(relative)
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact role is missing or not regular: {role}")
        files.append({"role": role, "path": relative, "size": path.stat().st_size, "sha256": _file_sha256(path)})
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if actual_paths != observed_paths:
        raise ValueError("artifact directory contains a file outside the role map")
    manifest = {
        "schema_version": 1,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": authorization.sha256,
        "challenge": document["challenge"],
        "collector_job_id": collector_job_id,
        "files": files,
    }
    manifest_path.write_text(_canonical_json(manifest))
    return load_artifact_bundle(root, authorization)


def validate_authority_receipts(
    subject: IrisJobReceipt,
    verifier: IrisJobReceipt,
    wandb: WandbArtifactReceipt,
    artifact: ArtifactBundle,
    authorization: Authorization,
) -> None:
    """Join live-exported Iris and W&B receipts to the verified local bytes."""
    document = authorization.decoded()
    expected_authorization = authorization.sha256
    validate_subject_receipt(subject, wandb, artifact, authorization)
    for name, receipt in (("subject", subject), ("verifier", verifier)):
        _validate_receipt_types(receipt, name)
        _equal(receipt.cluster, document["iris"]["cluster"], f"{name} receipt.cluster")
        _job_id(receipt.job_id, document, f"{name} receipt.job_id")
        _equal(receipt.submitting_user, document["iris"]["submitting_user"], f"{name} receipt.submitting_user")
        _sha(receipt.bundle_id, f"{name} receipt.bundle_id")
        _equal(receipt.authorization_sha256, expected_authorization, f"{name} receipt.authorization_sha256")
        _entrypoint_binds(receipt.entrypoint, expected_authorization, document, name)
        _equal(receipt.environment, (("CUDA_VISIBLE_DEVICES", "0"),), f"{name} receipt.environment")
        _equal(receipt.task_image, document["iris"]["task_image"], f"{name} receipt.task_image")
        _equal(receipt.gpu_count, document["iris"]["gpu_count"], f"{name} receipt.gpu_count")
        _equal(receipt.gpu_variant, "H100", f"{name} receipt.gpu_variant")
        _equal(receipt.replicas, 1, f"{name} receipt.replicas")
        _equal(receipt.existing_job_policy, "error", f"{name} receipt.existing_job_policy")
        _equal(receipt.fail_if_exists, True, f"{name} receipt.fail_if_exists")
        if not 0 < receipt.submitted_at_ms <= receipt.started_at_ms <= receipt.finished_at_ms:
            raise ValueError(f"{name} receipt timestamps are not ordered")
        _equal(receipt.state, "succeeded", f"{name} receipt.state")
        if receipt.attempt_id < 0 or not HEX16.fullmatch(receipt.attempt_uid):
            raise ValueError(f"{name} receipt attempt identity is malformed")
        _nonempty(receipt.worker_id, f"{name} receipt.worker_id")
        if receipt.submission_log_seq <= 0 or receipt.postseal_log_seq <= receipt.submission_log_seq:
            raise ValueError(f"{name} receipt log sequence is not a pre/post execution pair")
    if subject.job_id == verifier.job_id or subject.attempt_uid == verifier.attempt_uid:
        raise ValueError("subject and verifier must be independent Iris executions")
    _equal(subject.bundle_id, verifier.bundle_id, "subject/verifier workspace bundle ID")
    _equal(subject.job_id, artifact.collector_job_id, "artifact collector job ID")
    if "-subject-" not in subject.job_id or "-verify-" not in verifier.job_id:
        raise ValueError("subject and verifier Iris job roles are reversed")
    _validate_postseal_log(verifier, wandb, expected_authorization, "verifier")


def validate_subject_receipt(
    subject: IrisJobReceipt,
    wandb: WandbArtifactReceipt,
    artifact: ArtifactBundle,
    authorization: Authorization,
) -> None:
    """Join the completed subject job and exact W&B version before verification."""
    document = authorization.decoded()
    _validate_receipt_types(subject, "subject")
    _equal(subject.cluster, document["iris"]["cluster"], "subject receipt.cluster")
    _job_id(subject.job_id, document, "subject receipt.job_id")
    if "-subject-" not in subject.job_id:
        raise ValueError("subject receipt has the wrong Iris job role")
    _equal(subject.submitting_user, document["iris"]["submitting_user"], "subject receipt.submitting_user")
    _sha(subject.bundle_id, "subject receipt.bundle_id")
    _equal(subject.authorization_sha256, authorization.sha256, "subject receipt.authorization_sha256")
    _entrypoint_binds(subject.entrypoint, authorization.sha256, document, "subject")
    _equal(subject.environment, (("CUDA_VISIBLE_DEVICES", "0"),), "subject receipt.environment")
    _equal(subject.task_image, document["iris"]["task_image"], "subject receipt.task_image")
    _equal(subject.gpu_count, document["iris"]["gpu_count"], "subject receipt.gpu_count")
    _equal(subject.gpu_variant, "H100", "subject receipt.gpu_variant")
    _equal(subject.replicas, 1, "subject receipt.replicas")
    _equal(subject.existing_job_policy, "error", "subject receipt.existing_job_policy")
    _equal(subject.fail_if_exists, True, "subject receipt.fail_if_exists")
    if not 0 < subject.submitted_at_ms <= subject.started_at_ms <= subject.finished_at_ms:
        raise ValueError("subject receipt timestamps are not ordered")
    _equal(subject.state, "succeeded", "subject receipt.state")
    if subject.attempt_id < 0 or not HEX16.fullmatch(subject.attempt_uid):
        raise ValueError("subject receipt attempt identity is malformed")
    _nonempty(subject.worker_id, "subject receipt.worker_id")
    if subject.submission_log_seq <= 0 or subject.postseal_log_seq <= subject.submission_log_seq:
        raise ValueError("subject receipt log sequence is not a pre/post execution pair")
    _equal(subject.job_id, artifact.collector_job_id, "artifact collector job ID")
    _equal(wandb.entity, WANDB_ENTITY, "wandb receipt.entity")
    _equal(wandb.project, WANDB_PROJECT, "wandb receipt.project")
    _equal(wandb.name, document["wandb"]["artifact_name"], "wandb receipt.name")
    if not re.fullmatch(r"v[0-9]+", wandb.version):
        raise ValueError("wandb receipt must name an exact immutable version")
    _nonempty(wandb.digest, "wandb receipt.digest")
    _equal(wandb.manifest_sha256, artifact.manifest_sha256, "wandb receipt.manifest_sha256")
    _validate_postseal_log(subject, wandb, authorization.sha256, "subject")


def validate_numerical_artifacts(artifact: ArtifactBundle, authorization: Authorization) -> dict[str, object]:
    """Recompute the forward numerical result from manifested raw BF16 bytes."""
    document = authorization.decoded()
    x = _read_bfloat16(artifact.path_for("input_x"), (2048, 4096), "input_x")
    gamma = _read_bfloat16(artifact.path_for("input_gamma"), (4096,), "input_gamma")
    reference = _read_bfloat16(artifact.path_for("reference_output"), (2048, 4096), "reference_output")
    _equal(_raw_sha256(x), document["inputs"]["raw_sha256"][0], "input_x raw SHA-256")
    _equal(_raw_sha256(gamma), document["inputs"]["raw_sha256"][1], "input_gamma raw SHA-256")
    _equal(array_digest(x), document["inputs"]["array_sha256"][0], "input_x array digest")
    _equal(array_digest(gamma), document["inputs"]["array_sha256"][1], "input_gamma array digest")
    _equal(array_digest(reference), document["reference"]["array_sha256"], "reference output array digest")

    outputs = {role: _read_bfloat16(artifact.path_for(role), (2048, 4096), role) for role in OUTPUT_ROLES}
    ordinary = outputs["ordinary_jax_output"]
    for role, output in outputs.items():
        if not np.array_equal(output.view(np.uint16), ordinary.view(np.uint16)):
            raise ValueError(f"{role} differs bitwise from same-device ordinary JAX")
    measured = error_metrics(ordinary, reference)
    metrics = {
        "max_absolute_error": measured.max_absolute_error,
        "mean_absolute_error": measured.mean_absolute_error,
        "relative_linf_error": measured.relative_linf_error,
        "max_bfloat16_ulp_error": measured.max_bfloat16_ulp_error,
    }
    for name, limit in document["reference"]["limits"].items():
        if metrics[name] > limit:
            raise ValueError(f"ordinary JAX {name} exceeds the sealed independent-reference limit")
    return {
        "ordinary_jax_raw_sha256": _raw_sha256(ordinary),
        "reference_raw_sha256": _raw_sha256(reference),
        "metrics": metrics,
        "sequential_invocations": 3,
        "concurrent_invocations": 2,
        "bitwise_repeatability": True,
    }


def validate_authorized_checkout(authorization: Authorization, repository: Path) -> None:
    """Require the verifier to run from the exact clean preauthorized source tree."""
    document = authorization.decoded()
    revision = _git(repository, "rev-parse", "HEAD")
    base_revision = document["source"]["marin_commit"]
    _equal(_git(repository, "rev-parse", f"{revision}~2"), base_revision, "authorization predecessor commit")
    _equal(
        _git(repository, "rev-parse", f"{base_revision}^{{tree}}"),
        document["source"]["marin_tree"],
        "authorized Marin tree",
    )
    if _git(repository, "status", "--porcelain"):
        raise ValueError("authorized Marin checkout is dirty")
    authorization_relative = AUTHORIZATION_PATH.relative_to(repository).as_posix()
    evidence_relative = Path(__file__).relative_to(repository).as_posix()
    changed = set(_git(repository, "diff", "--name-only", f"{base_revision}..{revision}").splitlines())
    _equal(changed, {authorization_relative, evidence_relative}, "post-authorization changed path set")
    for relative, digest in document["collectors"].items():
        if not relative.startswith("lib/"):
            continue
        path = _relative_path(relative, f"authorization.collectors.{relative}")
        base_bytes = subprocess.run(
            ["git", "show", f"{base_revision}:{path.as_posix()}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if _bytes_sha256(base_bytes) != digest:
            raise ValueError(f"authorized collector bytes differ: {relative}")
        current = repository / path
        if relative == evidence_relative:
            expected = base_bytes.replace(
                b'EXPECTED_AUTHORIZATION_SHA256 = "UNSEALED"',
                f'EXPECTED_AUTHORIZATION_SHA256 = "{authorization.sha256}"'.encode(),
            )
            if current.read_bytes() != expected:
                raise ValueError("authority module differs from the one-token sealed transformation")
        elif current.is_symlink() or not current.is_file() or _file_sha256(current) != digest:
            raise ValueError(f"authorized collector working bytes differ: {relative}")
    patch_paths = tuple(sorted((repository / "lib/shuttle/mlir/jax_patch").glob("000[1-3]-*.patch")))
    if len(patch_paths) != 3:
        raise ValueError("authorized checkout does not contain the three exact JAX patches")
    _equal(
        [_file_sha256(path) for path in patch_paths],
        document["source"]["jax_patch_sha256"],
        "authorized ordered JAX patch digests",
    )


def validate_gpu_artifacts(artifact: ArtifactBundle, authorization: Authorization) -> GpuTransport:
    """Join the independent decoder, PTX files, cubins, and ptxas records."""
    document = authorization.decoded()
    transport = load_gpu_transport(artifact.path_for("gpu_transport"))
    expected = document["generated_bundle"]
    _equal(transport.transport_root, expected["transport_sha256"], "GPU transport SHA-256")
    _equal(transport.device_root, expected["device_fingerprint"], "GPU device fingerprint")
    _equal(len(transport.launches), expected["launch_count"], "GPU launch count")
    _equal(len(transport.slots), expected["slot_count"], "GPU slot count")
    _equal(
        [launch.code_digest for launch in transport.launches],
        expected["ptx_slice_sha256"],
        "ordered GPU PTX slice digests",
    )
    descriptor = _canonical_json(gpu_transport_descriptor(transport))
    if artifact.path_for("decoded_descriptor").read_text() != descriptor:
        raise ValueError("decoded GPU descriptor differs from the independent decoder")
    records = _closed(
        _load_json(artifact.path_for("ptxas_records")),
        {"schema_version", "ptxas_version", "ptxas_sha256", "target", "entries"},
        "ptxas records",
    )
    _equal(records["schema_version"], 1, "ptxas records.schema_version")
    _equal(records["ptxas_version"], document["toolchain"]["ptxas_version"], "ptxas records.ptxas_version")
    _equal(records["ptxas_sha256"], document["toolchain"]["ptxas_sha256"], "ptxas records.ptxas_sha256")
    _equal(records["target"], document["toolchain"]["target"], "ptxas records.target")
    entries = _sequence(records["entries"], "ptxas records.entries")
    if len(entries) != len(transport.launches):
        raise ValueError("ptxas records must cover every GPU launch ordinal")
    for ordinal, (value, launch) in enumerate(zip(entries, transport.launches, strict=True)):
        name = f"ptxas records.entries[{ordinal}]"
        record = _closed(
            value,
            {"ordinal", "ptx_sha256", "cubin_sha256", "exit_code", "spill_bytes", "register_count"},
            name,
        )
        _equal(record["ordinal"], ordinal, f"{name}.ordinal")
        ptx = artifact.path_for(f"ptx_{ordinal:02d}").read_bytes()
        cubin = artifact.path_for(f"cubin_{ordinal:02d}").read_bytes()
        if ptx != launch.code:
            raise ValueError(f"PTX ordinal {ordinal} differs from the decoded GPU transport")
        _equal(record["ptx_sha256"], _bytes_sha256(ptx), f"{name}.ptx_sha256")
        _equal(record["cubin_sha256"], _bytes_sha256(cubin), f"{name}.cubin_sha256")
        _equal(record["exit_code"], 0, f"{name}.exit_code")
        _equal(record["spill_bytes"], 0, f"{name}.spill_bytes")
        if type(record["register_count"]) is not int or record["register_count"] <= 0:
            raise ValueError(f"{name}.register_count must be a positive integer")
    return transport


def validate_observer_and_cache_artifacts(artifact: ArtifactBundle, authorization: Authorization) -> dict[str, object]:
    """Validate fresh-process cache behavior and one exact compiler invocation."""
    document = authorization.decoded()
    reports = {
        role: _load_json(artifact.path_for(role))
        for role in ("baseline_report", "cache_populate_report", "cache_reuse_report")
    }
    baseline = _closed(
        reports["baseline_report"],
        {"worker", "pid", "observer_events", "runtime_binaries"},
        "baseline report",
    )
    populate = _closed(
        reports["cache_populate_report"],
        {
            "worker",
            "pid",
            "cache_hits",
            "cache_name",
            "cache_size",
            "cache_sha256",
            "observer_event_count",
            "runtime_binaries",
        },
        "cache populate report",
    )
    reuse = _closed(
        reports["cache_reuse_report"],
        {
            "worker",
            "pid",
            "cache_hits",
            "cache_name",
            "cache_size",
            "cache_sha256",
            "observer_event_count",
            "runtime_binaries",
        },
        "cache reuse report",
    )
    _equal(baseline["worker"], "baseline", "baseline report.worker")
    _equal(populate["worker"], "populate", "cache populate report.worker")
    _equal(reuse["worker"], "reuse", "cache reuse report.worker")
    pids = [baseline["pid"], populate["pid"], reuse["pid"]]
    if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != 3:
        raise ValueError("baseline, populate, and reuse reports must name distinct positive process IDs")
    expected_binaries = {
        "jax_extension_sha256": _file_sha256(artifact.path_for("jax_extension")),
        "jax_common_sha256": _file_sha256(artifact.path_for("jax_common")),
        "cuda_plugin_extension_sha256": _file_sha256(artifact.path_for("cuda_plugin_extension")),
    }
    for name, report in reports.items():
        identity = _closed(
            report["runtime_binaries"],
            set(expected_binaries),
            f"{name}.runtime_binaries",
        )
        _equal(identity, expected_binaries, f"{name}.runtime_binaries")
    _equal(baseline["observer_events"], 0, "baseline report.observer_events")
    _equal(populate["cache_hits"], 0, "cache populate report.cache_hits")
    _equal(reuse["cache_hits"], 1, "cache reuse report.cache_hits")
    _equal(populate["observer_event_count"], 3, "cache populate report.observer_event_count")
    _equal(reuse["observer_event_count"], 0, "cache reuse report.observer_event_count")
    for field in ("cache_name", "cache_size", "cache_sha256"):
        _equal(reuse[field], populate[field], f"cache reuse report.{field}")
    cache = artifact.path_for("cache_entry")
    _equal(cache.stat().st_size, populate["cache_size"], "cache entry size")
    _equal(_file_sha256(cache), populate["cache_sha256"], "cache entry SHA-256")
    _sha(populate["cache_sha256"], "cache populate report.cache_sha256")
    if not re.fullmatch(r"jit_forward-[0-9a-f]{64}-cache", _nonempty(populate["cache_name"], "cache name")):
        raise ValueError("cache entry name differs from the public forward cache key")

    events = _load_event_transcript(artifact.path_for("observer_populate"), "observer populate")
    if _load_event_transcript(artifact.path_for("observer_reuse"), "observer reuse"):
        raise ValueError("cache reuse emitted Shuttle observer events")
    if tuple(event["phase"] for event in events) != ("algebra_coverage", "generated_bundle", "final_erasure"):
        raise ValueError("GPU observer phases differ from the exact successful pipeline")
    if len({event["invocation_id"] for event in events}) != 1:
        raise ValueError("GPU observer phases do not share one invocation ID")
    tuning = json.dumps(json.loads(CANONICAL_OPTIONS)["tuning"], sort_keys=True, separators=(",", ":"))
    identity = ObserverIdentity(
        policy="source_ordered",
        policy_digest=hashlib.sha256(CANONICAL_OPTIONS.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(tuning.encode()).hexdigest(),
        canonical_options=CANONICAL_OPTIONS,
        canonical_tuning=tuning,
    )
    _equal(identity.tuning_digest, TUNING_SHA256, "sealed ABI 10 tuning digest")
    fixture = target1_expectation("44d152ecc3e9ff18", "forward")
    coverage = fixture.coverage_manifest(identity)
    for event in events:
        _equal(event["policy"], identity.policy, "GPU observer policy")
        _equal(event["policy_digest"], identity.policy_digest, "GPU observer policy digest")
        _equal(event["tuning_digest"], identity.tuning_digest, "GPU observer tuning digest")
        _equal(event["failure_pass"], "", "GPU observer failure pass")
    for event in events[:2]:
        _equal(event["region_membership"], fixture.region_membership, "GPU observer region membership")
        _equal(event["coverage_manifest"], coverage, "GPU observer coverage manifest")
        _equal(event["unsupported_fingerprint"], fixture.unsupported_fingerprint, "GPU unsupported fingerprint")
        _equal(event["no_shuttle_semantics"], False, "GPU pre-final erasure status")
    _equal(events[0]["normalized_module_fingerprint"], "", "algebra observer fingerprint")
    _equal(
        events[1]["normalized_module_fingerprint"],
        document["compiler"]["generated_bundle_sha256"],
        "generated-bundle observer fingerprint",
    )
    final = events[2]
    _equal(final["region_membership"], "", "final observer region membership")
    _equal(final["coverage_manifest"], "", "final observer coverage manifest")
    _equal(final["unsupported_fingerprint"], "", "final observer unsupported fingerprint")
    _equal(final["normalized_module_fingerprint"], document["compiler"]["final_module_sha256"], "final fingerprint")
    _equal(final["no_shuttle_semantics"], True, "final erasure status")
    return {"cache_name": populate["cache_name"], "cache_sha256": populate["cache_sha256"], "events": 3}


def validate_registration_artifacts(artifact: ArtifactBundle) -> dict[str, object]:
    """Bind the real CUDA extension bytes to its captured PJRT registration."""
    transcript = _closed(
        _load_json(artifact.path_for("registration_transcript")),
        {
            "schema_version",
            "plugin_extension_sha256",
            "exported_targets",
            "upstream_target_count",
            "registration",
        },
        "registration transcript",
    )
    _equal(transcript["schema_version"], 1, "registration transcript.schema_version")
    _equal(
        transcript["plugin_extension_sha256"],
        _file_sha256(artifact.path_for("cuda_plugin_extension")),
        "registration plugin extension SHA-256",
    )
    targets = _sequence(transcript["exported_targets"], "registration transcript.exported_targets")
    if "shuttle.gpu.executable_bundle.v1" not in targets or len(targets) != len(set(targets)):
        raise ValueError("registration transcript does not contain one unique Shuttle target")
    if type(transcript["upstream_target_count"]) is not int or transcript["upstream_target_count"] < 0:
        raise ValueError("registration transcript upstream target count is malformed")
    _equal(len(targets), transcript["upstream_target_count"] + 1, "registration exported target count")
    registration = _closed(
        transcript["registration"],
        {"target", "api_version", "traits", "stage_pointers"},
        "registration transcript.registration",
    )
    _equal(registration["target"], "shuttle.gpu.executable_bundle.v1", "registration target")
    _equal(registration["api_version"], 1, "registration API version")
    _equal(registration["traits"], 0, "registration traits")
    pointers = _closed(
        registration["stage_pointers"],
        {"instantiate", "prepare", "initialize", "execute"},
        "registration stage pointers",
    )
    values = list(pointers.values())
    if any(type(value) is not int or value <= 0 for value in values) or len(set(values)) != 4:
        raise ValueError("registration stage pointers must be four distinct nonzero addresses")
    return transcript


def validate_gpu_build_artifacts(artifact: ArtifactBundle, authorization: Authorization) -> dict[str, Any]:
    """Bind rebuilt JAX/plugin files to the exact authorized source and toolchain."""
    document = authorization.decoded()
    manifest = _closed(
        _load_json(artifact.path_for("gpu_build_manifest")),
        {
            "schema_version",
            "jax_commit",
            "jax_tree",
            "xla_commit",
            "jax_patch_sha256",
            "bazel_version",
            "bazel_sha256",
            "ptxas_version",
            "ptxas_sha256",
            "jax_extension_sha256",
            "jax_common_sha256",
            "cuda_plugin_extension_sha256",
            "commands",
        },
        "GPU build manifest",
    )
    _equal(manifest["schema_version"], 1, "GPU build manifest.schema_version")
    for field in ("jax_commit", "jax_tree", "xla_commit", "jax_patch_sha256"):
        _equal(manifest[field], document["source"][field], f"GPU build manifest.{field}")
    _equal(manifest["ptxas_version"], document["toolchain"]["ptxas_version"], "GPU build manifest.ptxas_version")
    _equal(manifest["ptxas_sha256"], document["toolchain"]["ptxas_sha256"], "GPU build manifest.ptxas_sha256")
    _equal(
        manifest["jax_extension_sha256"],
        _file_sha256(artifact.path_for("jax_extension")),
        "rebuilt JAX extension SHA-256",
    )
    _equal(
        manifest["jax_common_sha256"],
        _file_sha256(artifact.path_for("jax_common")),
        "rebuilt JAX common library SHA-256",
    )
    _equal(
        manifest["cuda_plugin_extension_sha256"],
        _file_sha256(artifact.path_for("cuda_plugin_extension")),
        "rebuilt CUDA plugin extension SHA-256",
    )
    _equal(manifest["bazel_version"], document["toolchain"]["bazel_version"], "GPU build manifest.bazel_version")
    _equal(manifest["bazel_sha256"], document["toolchain"]["bazel_sha256"], "GPU build manifest.bazel_sha256")
    commands = _sequence(manifest["commands"], "GPU build manifest.commands")
    if commands != document["iris"]["build_commands"]:
        raise ValueError("GPU build commands differ from the pre-hardware authorization")
    return manifest


def validate_te_artifacts(artifact: ArtifactBundle, authorization: Authorization) -> list[dict[str, Any]]:
    """Validate the four sealed 2048-forward TE/cuDNN backend-pair results."""
    document = authorization.decoded()
    for role, source in (
        ("te_case_2048x4096_x", "input_x"),
        ("te_case_2048x4096_gamma", "input_gamma"),
        ("te_case_2048x4096_reference_y", "reference_output"),
    ):
        if artifact.path_for(role).read_bytes() != artifact.path_for(source).read_bytes():
            raise ValueError(f"{role} differs from the canonical raw evidence bytes")
    plan_path = artifact.path_for("te_run_plan")
    identity_path = artifact.path_for("te_execution_identity")
    if _file_sha256(artifact.path_for("te_runner")) != document["collectors"]["te_runner"]:
        raise ValueError("TE runner bytes differ from the authorization")
    if _file_sha256(artifact.path_for("te_harness")) != document["collectors"]["te_harness"]:
        raise ValueError("TE harness bytes differ from the authorization")
    if _file_sha256(artifact.path_for("build_manifest")) != document["collectors"]["te_build_manifest"]:
        raise ValueError("TE build manifest differs from the authorization")
    with chdir(artifact.root):
        results = [
            validate_result(artifact.path_for(f"te_result_{index}"), plan_path, identity_path) for index in range(4)
        ]
    coordinates = []
    for index, result in enumerate(results):
        _equal(result["boundary"], "forward", f"TE result {index} boundary")
        _equal(result["shape"], [2048, 4096], f"TE result {index} shape")
        coordinates.append((result["backends"]["forward"], result["backends"]["backward"]))
    _equal(
        set(coordinates),
        {
            ("transformer_engine", "transformer_engine"),
            ("cudnn", "cudnn"),
            ("transformer_engine", "cudnn"),
            ("cudnn", "transformer_engine"),
        },
        "TE forward backend-pair coverage",
    )
    return results


def _validate_authorized_compiler(value: object) -> None:
    compiler = _closed(
        value,
        {
            "canonical_options_sha256",
            "canonical_options",
            "pipeline_abi_version",
            "generated_bundle_sha256",
            "final_module_sha256",
            "transport_schema_version",
            "device_schema_version",
            "invocation_abi_schema_version",
            "bundle_schema_version",
            "custom_call_target",
        },
        "authorization.compiler",
    )
    _equal(compiler["pipeline_abi_version"], 10, "authorization.compiler.pipeline_abi_version")
    _equal(compiler["canonical_options"], CANONICAL_OPTIONS, "authorization.compiler.canonical_options")
    _equal(
        compiler["canonical_options_sha256"],
        hashlib.sha256(CANONICAL_OPTIONS.encode()).hexdigest(),
        "authorization.compiler.canonical_options_sha256",
    )
    _equal(
        compiler["generated_bundle_sha256"], GENERATED_BUNDLE_SHA256, "authorization.compiler.generated_bundle_sha256"
    )
    _equal(compiler["final_module_sha256"], FINAL_MODULE_SHA256, "authorization.compiler.final_module_sha256")
    _equal(compiler["transport_schema_version"], 2, "authorization.compiler.transport_schema_version")
    _equal(compiler["device_schema_version"], 3, "authorization.compiler.device_schema_version")
    _equal(compiler["invocation_abi_schema_version"], 3, "authorization.compiler.invocation_abi_schema_version")
    _equal(compiler["bundle_schema_version"], 2, "authorization.compiler.bundle_schema_version")
    _equal(
        compiler["custom_call_target"], "shuttle.gpu.executable_bundle.v1", "authorization.compiler.custom_call_target"
    )


def _validate_authorized_source(value: object) -> None:
    source = _closed(
        value,
        {
            "marin_commit",
            "marin_tree",
            "jax_commit",
            "jax_tree",
            "xla_commit",
            "transformer_engine_commit",
            "jax_patch_sha256",
        },
        "authorization.source",
    )
    for field in (
        "marin_commit",
        "marin_tree",
        "jax_commit",
        "jax_tree",
        "xla_commit",
        "transformer_engine_commit",
    ):
        _revision(source[field], f"authorization.source.{field}")
    _equal(source["jax_commit"], "619764c15117fbefc4ba13ab941871cb514c23f6", "authorization.source.jax_commit")
    _equal(source["jax_tree"], "56d186bccb48075374f32e46d2e090731b66e1ea", "authorization.source.jax_tree")
    _equal(source["xla_commit"], "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69", "authorization.source.xla_commit")
    _equal(
        source["transformer_engine_commit"], TRANSFORMER_ENGINE_COMMIT, "authorization.source.transformer_engine_commit"
    )
    patches = _sequence(source["jax_patch_sha256"], "authorization.source.jax_patch_sha256")
    if len(patches) != 3:
        raise ValueError("authorization.source.jax_patch_sha256 must contain the three ordered patches")
    for index, item in enumerate(patches):
        _sha(item, f"authorization.source.jax_patch_sha256[{index}]")


def _validate_authorized_inputs(value: object) -> None:
    inputs = _closed(value, {"array_sha256", "raw_sha256"}, "authorization.inputs")
    for field in ("array_sha256", "raw_sha256"):
        items = _sequence(inputs[field], f"authorization.inputs.{field}")
        if len(items) != 2:
            raise ValueError(f"authorization.inputs.{field} must contain x and gamma")
        for index, item in enumerate(items):
            _sha(item, f"authorization.inputs.{field}[{index}]")
    _equal(inputs["array_sha256"], list(INPUT_ARRAY_DIGESTS), "authorization.inputs.array_sha256")
    _equal(inputs["raw_sha256"], list(INPUT_RAW_SHA256), "authorization.inputs.raw_sha256")


def _validate_authorized_collectors(value: object) -> None:
    collectors = _closed(
        value,
        {
            "lib/shuttle/mlir/jax_patch/target1_abi10_h100_evidence.py",
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_subject.py",
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_build.py",
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_verify.py",
            "lib/shuttle/mlir/jax_patch/gpu_transport.py",
            "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration.py",
            "lib/shuttle/mlir/jax_patch/target1-abi10-h100-forward-role-paths-v1.json",
            "te_runner",
            "te_harness",
            "te_build_manifest",
        },
        "authorization.collectors",
    )
    for name, digest in collectors.items():
        _sha(digest, f"authorization.collectors.{name}")


def _validate_authorized_reference(value: object) -> None:
    reference = _closed(value, {"array_sha256", "limits"}, "authorization.reference")
    _equal(reference["array_sha256"], REFERENCE_OUTPUT_ARRAY_DIGEST, "authorization.reference.array_sha256")
    _equal(reference["limits"], REFERENCE_LIMITS, "authorization.reference.limits")


def _validate_authorized_bundle(value: object) -> None:
    bundle = _closed(
        value,
        {"transport_sha256", "device_fingerprint", "launch_count", "slot_count", "ptx_slice_sha256"},
        "authorization.generated_bundle",
    )
    _sha(bundle["transport_sha256"], "authorization.generated_bundle.transport_sha256")
    _sha(bundle["device_fingerprint"], "authorization.generated_bundle.device_fingerprint")
    _equal(bundle["launch_count"], 19, "authorization.generated_bundle.launch_count")
    _equal(bundle["slot_count"], 21, "authorization.generated_bundle.slot_count")
    slices = _sequence(bundle["ptx_slice_sha256"], "authorization.generated_bundle.ptx_slice_sha256")
    if len(slices) != 19:
        raise ValueError("authorization.generated_bundle.ptx_slice_sha256 must contain 19 ordinals")
    for index, item in enumerate(slices):
        _sha(item, f"authorization.generated_bundle.ptx_slice_sha256[{index}]")


def _validate_authorized_toolchain(value: object) -> None:
    toolchain = _closed(
        value,
        {"bazel_version", "bazel_sha256", "ptxas_version", "ptxas_sha256", "target"},
        "authorization.toolchain",
    )
    _equal(toolchain["bazel_version"], "7.4.1", "authorization.toolchain.bazel_version")
    _equal(
        toolchain["bazel_sha256"],
        "c97f02133adce63f0c28678ac1f21d65fa8255c80429b588aeeba8a1fac6202b",
        "authorization.toolchain.bazel_sha256",
    )
    _equal(toolchain["ptxas_version"], "13.2.78", "authorization.toolchain.ptxas_version")
    _sha(toolchain["ptxas_sha256"], "authorization.toolchain.ptxas_sha256")
    _equal(toolchain["target"], "sm_90", "authorization.toolchain.target")


def _validate_authorized_iris(value: object, challenge: str) -> None:
    iris = _closed(
        value,
        {
            "cluster",
            "submitting_user",
            "gpu_count",
            "visible_gpu",
            "subject_job_id",
            "verifier_job_id",
            "subject_command",
            "verifier_command",
            "existing_job_policy",
            "task_image",
            "build_commands",
        },
        "authorization.iris",
    )
    _equal(iris["cluster"], "cw-us-east-02a", "authorization.iris.cluster")
    _equal(iris["submitting_user"], "dlwh", "authorization.iris.submitting_user")
    _equal(iris["gpu_count"], 8, "authorization.iris.gpu_count")
    _equal(iris["visible_gpu"], 0, "authorization.iris.visible_gpu")
    for role in ("subject", "verifier"):
        job_id = _nonempty(iris[f"{role}_job_id"], f"authorization.iris.{role}_job_id")
        marker = "subject" if role == "subject" else "verify"
        if not JOB_ID.fullmatch(job_id) or f"abi10-h100-{challenge[:12]}-{marker}-" not in job_id:
            raise ValueError(f"authorization.iris.{role}_job_id does not bind the challenge and role")
        command = _sequence(iris[f"{role}_command"], f"authorization.iris.{role}_command")
        if not command or any(type(item) is not str or not item for item in command):
            raise ValueError(f"authorization.iris.{role}_command must be a nonempty argv list")
    _equal(iris["existing_job_policy"], "error", "authorization.iris.existing_job_policy")
    if not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", _nonempty(iris["task_image"], "authorization.iris.task_image")):
        raise ValueError("authorization.iris.task_image must be digest-pinned")
    commands = _sequence(iris["build_commands"], "authorization.iris.build_commands")
    if not commands or any(type(command) is not str or not command for command in commands):
        raise ValueError("authorization.iris.build_commands must be a nonempty string list")


def _validate_digest_mapping(value: object, name: str) -> None:
    if type(value) is not dict or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    for key, item in value.items():
        _nonempty(key, f"{name} key")
        _sha(item, f"{name}.{key}")


def _relative_path(value: object, name: str) -> PurePosixPath:
    text = _nonempty(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{name} is not a canonical relative path")
    return path


def _job_id(value: object, authorization: Mapping[str, Any], name: str) -> str:
    item = _nonempty(value, name)
    role = "subject" if "subject" in name or "collector" in name else "verifier"
    if item != authorization["iris"][f"{role}_job_id"]:
        raise ValueError(f"{name} differs from the exact authorized Iris job ID")
    return item


def _entrypoint_binds(
    entrypoint: tuple[str, ...], authorization_sha: str, authorization: Mapping[str, Any], name: str
) -> None:
    if type(entrypoint) is not tuple or not entrypoint or any(type(item) is not str or not item for item in entrypoint):
        raise ValueError(f"{name} receipt entrypoint is malformed")
    role = "subject" if name == "subject" else "verify"
    expected = tuple(
        authorization_sha if item == "${AUTHORIZATION_SHA256}" else item
        for item in authorization["iris"][f"{'subject' if role == 'subject' else 'verifier'}_command"]
    )
    _equal(entrypoint, expected, f"{name} receipt command")


def _validate_receipt_types(receipt: IrisJobReceipt, name: str) -> None:
    for field in (
        "submitted_at_ms",
        "started_at_ms",
        "finished_at_ms",
        "attempt_id",
        "submission_log_seq",
        "postseal_log_seq",
        "gpu_count",
        "replicas",
    ):
        if type(getattr(receipt, field)) is not int:
            raise ValueError(f"{name} receipt.{field} must be an integer")
    if type(receipt.fail_if_exists) is not bool:
        raise ValueError(f"{name} receipt.fail_if_exists must be a boolean")


def _validate_postseal_log(
    receipt: IrisJobReceipt,
    wandb: WandbArtifactReceipt,
    authorization_sha256: str,
    role: str,
) -> None:
    log_role = "subject" if role == "subject" else "verify"
    value = _closed(
        _parse_json(receipt.postseal_log_json),
        {
            "schema_version",
            "role",
            "authorization_sha256",
            "wandb_entity",
            "wandb_project",
            "wandb_name",
            "wandb_version",
            "wandb_digest",
            "artifact_manifest_sha256",
            "verdict",
        },
        f"{role} postseal log",
    )
    _equal(value["schema_version"], 1, f"{role} postseal log.schema_version")
    _equal(value["role"], log_role, f"{role} postseal log.role")
    _equal(value["authorization_sha256"], authorization_sha256, f"{role} postseal log.authorization_sha256")
    _equal(value["wandb_entity"], wandb.entity, f"{role} postseal log.wandb_entity")
    _equal(value["wandb_project"], wandb.project, f"{role} postseal log.wandb_project")
    _equal(value["wandb_name"], wandb.name, f"{role} postseal log.wandb_name")
    _equal(value["wandb_version"], wandb.version, f"{role} postseal log.wandb_version")
    _equal(value["wandb_digest"], wandb.digest, f"{role} postseal log.wandb_digest")
    _equal(
        value["artifact_manifest_sha256"],
        wandb.manifest_sha256,
        f"{role} postseal log.artifact_manifest_sha256",
    )
    _equal(value["verdict"], "uploaded" if role == "subject" else "accepted", f"{role} postseal log.verdict")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing or not regular: {path}")
    return _parse_json(path.read_text())


def _load_event_transcript(path: Path, name: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    events = []
    for index, record in enumerate(value):
        if type(record) is not list or len(record) != len(EVENT_FIELDS):
            raise ValueError(f"{name}[{index}] differs from the native observer schema")
        events.append(dict(zip(EVENT_FIELDS, record, strict=True)))
    return events


def _parse_json(value: str) -> dict[str, Any]:
    decoded = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if type(decoded) is not dict:
        raise ValueError("JSON document root must be an object")
    return decoded


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bfloat16(path: Path, shape: tuple[int, ...], name: str) -> np.ndarray:
    expected_size = int(np.prod(shape, dtype=np.int64)) * 2
    data = path.read_bytes()
    if len(data) != expected_size:
        raise ValueError(f"{name} has the wrong BF16 byte size")
    return np.frombuffer(data, dtype=ml_dtypes.bfloat16).reshape(shape)


def _raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def gpu_transport_descriptor(transport: GpuTransport) -> dict[str, object]:
    """Return the canonical JSON-ready projection stored beside raw transport."""
    return {
        "schema_version": 1,
        "transport_root": transport.transport_root,
        "device_root": transport.device_root,
        "invocation_root": transport.invocation_root,
        "bundle_root": transport.bundle_root,
        "code_digest": transport.code_digest,
        "temporary_bytes": transport.temporary_bytes,
        "launches": [
            {
                "ordinal": launch.ordinal,
                "source_task": launch.source_task,
                "code_offset": launch.code_offset,
                "code_length": launch.code_length,
                "code_digest": launch.code_digest,
                "grid": list(launch.grid),
                "block": list(launch.block),
                "dynamic_shared_memory_bytes": launch.dynamic_shared_memory_bytes,
                "input_slots": list(launch.input_slots),
                "output_slots": list(launch.output_slots),
                "dependencies": list(launch.dependencies),
            }
            for launch in transport.launches
        ],
        "slots": [
            {
                "ordinal": slot.ordinal,
                "source_buffer": slot.source_buffer,
                "element_type": slot.element_type.value,
                "dimensions": list(slot.dimensions),
                "required_bytes": slot.required_bytes,
                "byte_strides": list(slot.byte_strides),
                "alignment": slot.alignment,
                "storage": slot.storage.value,
                "binding_kind": slot.binding_kind.value,
                "binding_index": slot.binding_index,
            }
            for slot in transport.slots
        ],
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields differ from the closed schema")
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    item = _nonempty(value, name)
    if not SHA256.fullmatch(item):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return item


def _revision(value: object, name: str) -> str:
    item = _nonempty(value, name)
    if not re.fullmatch(r"[0-9a-f]{40}", item):
        raise ValueError(f"{name} must be a full Git revision")
    return item


def _equal(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} differs from the sealed value")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
