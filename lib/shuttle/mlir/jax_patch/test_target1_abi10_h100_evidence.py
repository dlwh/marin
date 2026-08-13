# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from target1_abi10_h100_evidence import (
    AUTHORIZATION_ID,
    REQUIRED_ARTIFACT_ROLES,
    WANDB_ARTIFACT_TYPE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    IrisJobReceipt,
    WandbArtifactReceipt,
    _load_authorization,
    load_artifact_bundle,
    validate_authority_receipts,
    validate_registration_artifacts,
)
from target1_abi10_h100_forward_contract import (
    CANONICAL_OPTIONS,
    FINAL_MODULE_SHA256,
    GENERATED_BUNDLE_SHA256,
    INPUT_ARRAY_DIGESTS,
    INPUT_RAW_SHA256,
    REFERENCE_LIMITS,
    REFERENCE_OUTPUT_ARRAY_DIGEST,
    TRANSFORMER_ENGINE_COMMIT,
)

CHALLENGE = "3" * 64
SHA = "a" * 64
REVISION = "b" * 40
PTX_SLICES = [hashlib.sha256(f"ptx-{index}".encode()).hexdigest() for index in range(19)]
SUBJECT_JOB = f"/dlwh/abi10-h100-{CHALLENGE[:12]}-subject-123456789abc"
VERIFIER_JOB = f"/dlwh/abi10-h100-{CHALLENGE[:12]}-verify-123456789abc"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _authorization_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "authorization_id": AUTHORIZATION_ID,
        "status": "sealed_before_hardware_allocation",
        "challenge": CHALLENGE,
        "evidence_revision": REVISION,
        "evidence_tree": REVISION,
        "coordinate": {
            "target_id": 1,
            "boundary": "forward",
            "shape": [2048, 4096],
            "shape_id": "rmsnorm_bf16_2048x4096",
            "policy": "source_ordered",
            "hardware_class": "h100",
        },
        "source": {
            "marin_commit": REVISION,
            "marin_tree": REVISION,
            "jax_commit": "619764c15117fbefc4ba13ab941871cb514c23f6",
            "jax_tree": "56d186bccb48075374f32e46d2e090731b66e1ea",
            "xla_commit": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
            "transformer_engine_commit": TRANSFORMER_ENGINE_COMMIT,
            "jax_patch_sha256": [SHA, "b" * 64, "c" * 64],
        },
        "compiler": {
            "pipeline_abi_version": 10,
            "canonical_options": CANONICAL_OPTIONS,
            "canonical_options_sha256": hashlib.sha256(CANONICAL_OPTIONS.encode()).hexdigest(),
            "generated_bundle_sha256": GENERATED_BUNDLE_SHA256,
            "final_module_sha256": FINAL_MODULE_SHA256,
            "transport_schema_version": 2,
            "device_schema_version": 3,
            "invocation_abi_schema_version": 3,
            "bundle_schema_version": 2,
            "custom_call_target": "shuttle.gpu.executable_bundle.v1",
        },
        "inputs": {
            "array_sha256": list(INPUT_ARRAY_DIGESTS),
            "raw_sha256": list(INPUT_RAW_SHA256),
        },
        "reference": {
            "array_sha256": REFERENCE_OUTPUT_ARRAY_DIGEST,
            "limits": REFERENCE_LIMITS,
        },
        "generated_bundle": {
            "transport_sha256": "7eac8ca775032acb6cc1d9c440804d490b84edbde5ee65aeb924aa7bc4e95e27",
            "device_fingerprint": "80d41a3db8d128bbc3851ab8f17d9e1b6167f3d80ba95cc2068c6e0090147f51",
            "launch_count": 19,
            "slot_count": 21,
            "ptx_slice_sha256": PTX_SLICES,
        },
        "toolchain": {
            "bazel_version": "7.4.1",
            "bazel_sha256": "c97f02133adce63f0c28678ac1f21d65fa8255c80429b588aeeba8a1fac6202b",
            "ptxas_version": "13.2.78",
            "ptxas_sha256": "a0212ca1a74ca4877e8d063baa1dd0b075fcd255fb6ab9e8a6bf98a276a2d6e8",
            "target": "sm_90",
        },
        "collectors": {
            "lib/shuttle/mlir/jax_patch/target1_abi10_h100_evidence.py": SHA,
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_subject.py": SHA,
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_build.py": SHA,
            "lib/shuttle/mlir/jax_patch/shuttle_h100_evidence_verify.py": SHA,
            "lib/shuttle/mlir/jax_patch/gpu_transport.py": SHA,
            "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration.py": SHA,
            "lib/shuttle/mlir/jax_patch/target1-abi10-h100-forward-role-paths-v1.json": SHA,
            "te_runner": SHA,
            "te_harness": SHA,
            "te_build_manifest": SHA,
        },
        "iris": {
            "cluster": "cw-us-east-02a",
            "submitting_user": "dlwh",
            "gpu_count": 8,
            "visible_gpu": 0,
            "subject_job_id": SUBJECT_JOB,
            "verifier_job_id": VERIFIER_JOB,
            "subject_command": [
                ".venv/bin/python",
                "-m",
                "shuttle_h100_evidence_subject",
                "--authorization-sha256",
                "${AUTHORIZATION_SHA256}",
                "--challenge",
                CHALLENGE,
            ],
            "verifier_command": [
                ".venv/bin/python",
                "-m",
                "shuttle_h100_evidence_verify",
                "--authorization-sha256",
                "${AUTHORIZATION_SHA256}",
                "--challenge",
                CHALLENGE,
            ],
            "existing_job_policy": "error",
            "task_image": "registry/image@sha256:" + "d" * 64,
            "build_commands": ["build exact artifacts"],
        },
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "artifact_name": "target1-abi10-h100-forward-" + CHALLENGE[:12],
            "artifact_type": WANDB_ARTIFACT_TYPE,
            "exact_version_required": True,
        },
        "claims": {
            "timing_permitted": False,
            "offline_json_can_accept": False,
            "required_receipts": ["iris_subject", "wandb_exact_version", "iris_verifier"],
            "maximum_claim": "controller_observed_content_addressed_independently_revalidated_externally_exported",
        },
    }


def _authorization(tmp_path: Path):
    path = tmp_path / "authorization.json"
    _write_json(path, _authorization_document())
    return _load_authorization(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _artifact(tmp_path: Path, authorization, overrides: dict[str, bytes] | None = None):
    root = tmp_path / "artifact"
    root.mkdir()
    files = []
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        path = root / role
        path.write_bytes((overrides or {}).get(role, ("raw evidence " + role).encode()))
        files.append(
            {
                "role": role,
                "path": role,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_json(
        root / "artifact-manifest.json",
        {
            "schema_version": 1,
            "authorization_id": AUTHORIZATION_ID,
            "authorization_sha256": authorization.sha256,
            "challenge": CHALLENGE,
            "collector_job_id": SUBJECT_JOB,
            "files": files,
        },
    )
    return load_artifact_bundle(root, authorization)


def _iris(role: str, authorization) -> IrisJobReceipt:
    postseal = {
        "schema_version": 1,
        "role": role,
        "authorization_sha256": authorization.sha256,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": "target1-abi10-h100-forward-" + CHALLENGE[:12],
        "wandb_version": "v0",
        "wandb_digest": "wandb-digest",
        "artifact_manifest_sha256": "MANIFEST",
        "verdict": "uploaded" if role == "subject" else "accepted",
    }
    return IrisJobReceipt(
        cluster="cw-us-east-02a",
        job_id=SUBJECT_JOB if role == "subject" else VERIFIER_JOB,
        submitting_user="dlwh",
        bundle_id=SHA,
        authorization_sha256=authorization.sha256,
        entrypoint=tuple(
            authorization.sha256 if item == "${AUTHORIZATION_SHA256}" else item
            for item in authorization.decoded()["iris"][f"{'subject' if role == 'subject' else 'verifier'}_command"]
        ),
        environment=(("CUDA_VISIBLE_DEVICES", "0"),),
        task_image="registry/image@sha256:" + "d" * 64,
        gpu_count=8,
        gpu_variant="H100",
        replicas=1,
        existing_job_policy="error",
        fail_if_exists=True,
        submitted_at_ms=1,
        started_at_ms=2,
        finished_at_ms=3,
        state="succeeded",
        attempt_id=0,
        attempt_uid="1" * 16 if role == "subject" else "2" * 16,
        worker_id=role,
        submission_log_seq=10,
        postseal_log_seq=20,
        postseal_log_json=json.dumps(postseal, sort_keys=True),
    )


def _postseal(role: str, authorization, manifest_sha256: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "role": role,
            "authorization_sha256": authorization.sha256,
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_name": "target1-abi10-h100-forward-" + CHALLENGE[:12],
            "wandb_version": "v0",
            "wandb_digest": "wandb-digest",
            "artifact_manifest_sha256": manifest_sha256,
            "verdict": "uploaded" if role == "subject" else "accepted",
        },
        sort_keys=True,
    )


def test_external_authority_join_accepts_exact_bytes_and_receipts(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    receipt = WandbArtifactReceipt(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="target1-abi10-h100-forward-" + CHALLENGE[:12],
        version="v0",
        digest="wandb-digest",
        manifest_sha256=artifact.manifest_sha256,
    )

    subject = replace(
        _iris("subject", authorization),
        postseal_log_json=_postseal("subject", authorization, artifact.manifest_sha256),
    )
    verifier = replace(
        _iris("verify", authorization),
        postseal_log_json=_postseal("verify", authorization, artifact.manifest_sha256),
    )
    validate_authority_receipts(
        subject,
        verifier,
        receipt,
        artifact,
        authorization,
    )


def test_artifact_rejects_byte_substitution_and_unmanifested_file(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    artifact.path_for("gpu_transport").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="bytes differ"):
        load_artifact_bundle(artifact.root, authorization)

    (artifact.root / artifact.files["gpu_transport"].path).write_bytes(b"raw evidence gpu_transport")
    (artifact.root / "unmanifested").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unmanifested"):
        load_artifact_bundle(artifact.root, authorization)


def test_artifact_rejects_unmanifested_symlink(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    (artifact.root / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_artifact_bundle(artifact.root, authorization)


def test_artifact_rejects_a_different_authorization_even_when_self_consistent(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    changed = _authorization_document()
    changed["challenge"] = "4" * 64
    changed["iris"]["subject_job_id"] = "/dlwh/abi10-h100-" + "4" * 12 + "-subject-123456789abc"
    changed["iris"]["verifier_job_id"] = "/dlwh/abi10-h100-" + "4" * 12 + "-verify-123456789abc"
    changed["iris"]["subject_command"][-1] = "4" * 64
    changed["iris"]["verifier_command"][-1] = "4" * 64
    changed["wandb"]["artifact_name"] = "target1-abi10-h100-forward-" + "4" * 12
    changed_path = tmp_path / "changed-authorization.json"
    _write_json(changed_path, changed)
    changed_authorization = _load_authorization(changed_path, hashlib.sha256(changed_path.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="authorization_sha256"):
        load_artifact_bundle(artifact.root, changed_authorization)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("job_id", SUBJECT_JOB, "exact authorized Iris job ID"),
        ("attempt_uid", "1" * 16, "independent"),
        ("entrypoint", ("python", "verify.py"), "command differs"),
        ("state", "failed", "state"),
    ],
)
def test_receipts_reject_nonindependent_or_unbound_verifier(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    wandb = WandbArtifactReceipt(
        WANDB_ENTITY,
        WANDB_PROJECT,
        "target1-abi10-h100-forward-" + CHALLENGE[:12],
        "v0",
        "wandb-digest",
        artifact.manifest_sha256,
    )
    subject = replace(
        _iris("subject", authorization),
        postseal_log_json=_postseal("subject", authorization, artifact.manifest_sha256),
    )
    verifier = replace(
        _iris("verify", authorization),
        postseal_log_json=_postseal("verify", authorization, artifact.manifest_sha256),
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        validate_authority_receipts(subject, verifier, wandb, artifact, authorization)


def test_authorization_rejects_semantic_drift_with_rewritten_json(tmp_path: Path) -> None:
    document = _authorization_document()
    drifted = copy.deepcopy(document)
    drifted["reference"]["array_sha256"] = SHA
    path = tmp_path / "authorization.json"
    _write_json(path, drifted)

    with pytest.raises(ValueError, match=r"reference\.array_sha256"):
        _load_authorization(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_registration_transcript_is_bound_to_actual_plugin_bytes(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    plugin_bytes = b"real plugin bytes"
    transcript = {
        "schema_version": 1,
        "plugin_extension_sha256": hashlib.sha256(plugin_bytes).hexdigest(),
        "exported_targets": ["built_in", "shuttle.gpu.executable_bundle.v1"],
        "upstream_target_count": 1,
        "registration": {
            "target": "shuttle.gpu.executable_bundle.v1",
            "api_version": 1,
            "traits": 0,
            "stage_pointers": {"instantiate": 1, "prepare": 2, "initialize": 3, "execute": 4},
        },
    }
    artifact = _artifact(
        tmp_path,
        authorization,
        {
            "cuda_plugin_extension": plugin_bytes,
            "registration_transcript": (json.dumps(transcript, indent=2, sort_keys=True) + "\n").encode(),
        },
    )

    validate_registration_artifacts(artifact)
    (artifact.root / artifact.files["cuda_plugin_extension"].path).write_bytes(b"substituted plugin")
    with pytest.raises(ValueError, match="changed after manifest"):
        validate_registration_artifacts(artifact)


def test_authorization_rejects_malformed_image_digest(tmp_path: Path) -> None:
    document = _authorization_document()
    document["iris"]["task_image"] = "registry/image@sha256:not-a-digest"
    path = tmp_path / "authorization.json"
    _write_json(path, document)
    with pytest.raises(ValueError, match="digest-pinned"):
        _load_authorization(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_receipt_rejects_noninteger_runtime_identity(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    artifact = _artifact(tmp_path, authorization)
    wandb = WandbArtifactReceipt(
        WANDB_ENTITY,
        WANDB_PROJECT,
        "target1-abi10-h100-forward-" + CHALLENGE[:12],
        "v0",
        "wandb-digest",
        artifact.manifest_sha256,
    )
    subject = replace(
        _iris("subject", authorization),
        submitted_at_ms=True,
        postseal_log_json=_postseal("subject", authorization, artifact.manifest_sha256),
    )
    verifier = replace(
        _iris("verify", authorization),
        postseal_log_json=_postseal("verify", authorization, artifact.manifest_sha256),
    )
    with pytest.raises(ValueError, match="submitted_at_ms must be an integer"):
        validate_authority_receipts(subject, verifier, wandb, artifact, authorization)
