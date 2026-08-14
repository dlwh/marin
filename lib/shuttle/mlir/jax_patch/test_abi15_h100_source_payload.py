# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip and mutation tests for the ABI15 capsule and raw export."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from abi15_h100_source_payload import (
    ACK_FILENAME,
    READY_FILENAME,
    REQUIRED_EXACT_PATHS,
    controller_ack,
    decode_export_record,
    prepare_raw_export,
    prepare_source_capsule,
    read_export_record,
    restore_source_capsule,
    validate_retrieved_export,
    write_task_ack,
)

SHA = "a" * 64


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    for relative in REQUIRED_EXACT_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n")
    allowlist = repository / "lib/shuttle/mlir/jax_patch/abi15-h100-source-allowlist-v1.json"
    allowlist.write_text(
        json.dumps(
            {"paths": list(REQUIRED_EXACT_PATHS), "schema_version": 1},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _git(repository, "add", *REQUIRED_EXACT_PATHS)
    _git(repository, "commit", "-q", "-m", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def _raw_artifact(root: Path) -> None:
    root.mkdir()
    (root / "raw-artifact-manifest.json").write_text("{}\n")
    nested = root / "nested/result.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"".join(hashlib.sha256(index.to_bytes(4, "little")).digest() for index in range(20_000)))


def _copy_transport(export: Path, retrieval: Path) -> None:
    retrieval.mkdir()
    for path in export.iterdir():
        if path.name != ACK_FILENAME:
            shutil.copyfile(path, retrieval / path.name)


def test_source_capsule_round_trip_binds_exact_commit_tree_and_files(
    tmp_path: Path,
) -> None:
    repository, commit = _source_repository(tmp_path)
    output = tmp_path / "capsule"

    prepared = prepare_source_capsule(repository, commit, output)
    restored = tmp_path / "restored"
    restore_source_capsule(
        output / prepared["archive"]["filename"],
        output / prepared["manifest_filename"],
        restored,
        expected_manifest_sha256=prepared["manifest_sha256"],
        expected_source_commit=commit,
        expected_source_tree=_git(repository, "rev-parse", "HEAD^{tree}"),
    )

    assert sorted(path.relative_to(restored).as_posix() for path in restored.rglob("*") if path.is_file()) == list(
        REQUIRED_EXACT_PATHS
    )


def test_source_capsule_rejects_dirty_checkout_and_allowlist_drift(
    tmp_path: Path,
) -> None:
    repository, commit = _source_repository(tmp_path)
    (repository / REQUIRED_EXACT_PATHS[0]).write_text("dirty\n")
    with pytest.raises(ValueError):
        prepare_source_capsule(repository, commit, tmp_path / "dirty")
    _git(repository, "checkout", "--", REQUIRED_EXACT_PATHS[0])
    allowlist = repository / "lib/shuttle/mlir/jax_patch/abi15-h100-source-allowlist-v1.json"
    document = json.loads(allowlist.read_text())
    document["paths"].pop()
    allowlist.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    _git(repository, "add", allowlist.relative_to(repository).as_posix())
    _git(repository, "commit", "-q", "-m", "drift")
    with pytest.raises(ValueError):
        prepare_source_capsule(repository, _git(repository, "rev-parse", "HEAD"), tmp_path / "drift")


def test_raw_export_round_trip_requires_controller_verification_before_ack(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _raw_artifact(artifact)
    export = tmp_path / "export"
    ready = prepare_raw_export(
        artifact,
        export,
        runner_exit_code=0,
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_manifest_sha256="d" * 64,
        authorization_sha256="e" * 64,
    )
    retrieval = tmp_path / "retrieval"
    _copy_transport(export, retrieval)
    archive = tmp_path / "retrieved.tar.gz"

    verified = validate_retrieved_export(
        retrieval,
        archive,
        expected_source_commit="b" * 40,
        expected_source_tree="c" * 40,
        expected_source_manifest_sha256="d" * 64,
        expected_authorization_sha256="e" * 64,
    )
    ack = controller_ack(
        ready,
        verification_report_sha256="f" * 64,
        verifier_sha256="1" * 64,
    )
    write_task_ack(export, ack)

    assert verified["archive"]["sha256"] == ready["archive_sha256"]
    assert json.loads((export / ACK_FILENAME).read_text()) == ack
    assert not (artifact / "accepted_bundle.json").exists()


def test_raw_export_rejects_task_authored_acceptance(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _raw_artifact(artifact)
    (artifact / "accepted_bundle.json").write_text("{}\n")

    with pytest.raises(ValueError):
        prepare_raw_export(
            artifact,
            tmp_path / "export",
            runner_exit_code=0,
            source_commit="b" * 40,
            source_tree="c" * 40,
            source_manifest_sha256="d" * 64,
            authorization_sha256="e" * 64,
        )


@pytest.mark.parametrize("mutation", ["part", "order", "manifest"])
def test_retrieved_export_rejects_rehashed_or_transport_mutation(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "artifact"
    _raw_artifact(artifact)
    export = tmp_path / "export"
    prepare_raw_export(
        artifact,
        export,
        runner_exit_code=0,
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_manifest_sha256="d" * 64,
        authorization_sha256="e" * 64,
    )
    retrieval = tmp_path / "retrieval"
    _copy_transport(export, retrieval)
    manifest_path = retrieval / "RAW_EXPORT_MANIFEST"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "part":
        part = retrieval / manifest["parts"][0]["filename"]
        part.write_bytes(part.read_bytes() + b"x")
    elif mutation == "order":
        manifest["parts"] = list(reversed(manifest["parts"]))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        ready_path = retrieval / READY_FILENAME
        ready = json.loads(ready_path.read_text())
        ready["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ready_path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    else:
        manifest["authorization_sha256"] = "2" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        ready_path = retrieval / READY_FILENAME
        ready = json.loads(ready_path.read_text())
        ready["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ready_path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError):
        validate_retrieved_export(
            retrieval,
            tmp_path / "archive.tar.gz",
            expected_source_commit="b" * 40,
            expected_source_tree="c" * 40,
            expected_source_manifest_sha256="d" * 64,
            expected_authorization_sha256="e" * 64,
        )


def test_export_record_round_trip_enforces_contiguous_offset_and_digest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _raw_artifact(artifact)
    export = tmp_path / "export"
    prepare_raw_export(
        artifact,
        export,
        runner_exit_code=0,
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_manifest_sha256="d" * 64,
        authorization_sha256="e" * 64,
    )
    encoded = read_export_record(export, READY_FILENAME)

    decoded = decode_export_record(encoded, expected_filename=READY_FILENAME, expected_offset=0)

    assert decoded["contents"] == (export / READY_FILENAME).read_bytes()
    mutated = json.loads(encoded)
    mutated["offset"] = 1
    with pytest.raises(ValueError):
        decode_export_record(
            (json.dumps(mutated, indent=2, sort_keys=True) + "\n").encode(),
            expected_filename=READY_FILENAME,
            expected_offset=0,
        )
