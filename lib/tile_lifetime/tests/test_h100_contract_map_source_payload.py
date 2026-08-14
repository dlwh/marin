# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest
from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES
from iris.cluster.client.bundle import create_workspace_zip

payload = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_source_payload")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")


def _source_repository(root: Path, *, symlink_license: bool = False) -> tuple[Path, str, str]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Capsule Test")
    _git(source, "config", "user.email", "capsule@example.com")
    for relative in payload.REQUIRED_EXACT_PATHS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json":
            allowlist = {
                "exact": list(payload.REQUIRED_EXACT_PATHS),
                "recursive": [{"path": "lib/tile_lifetime/src/tile_lifetime", "suffix": ".py"}],
                "schema_version": payload.SCHEMA_VERSION,
            }
            path.write_bytes(payload._canonical_json(allowlist))
        elif relative == "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py":
            path.write_bytes(Path(payload.__file__).read_bytes())
        elif relative == "lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py":
            path.write_text(
                """import json
import sys
from importlib.metadata import version
from pathlib import Path

import jax
import numpy
import shuttle

artifact_directory = Path(sys.argv[sys.argv.index("--artifact-directory") + 1])
artifact_directory.mkdir(parents=True)
(artifact_directory / "runtime-imports.json").write_text(
    json.dumps(
        {
            "jax": jax.__version__,
            "numpy": numpy.__version__,
            "python": sys.executable,
            "shuttle": version("marin-shuttle"),
        },
        sort_keys=True,
    )
)
(artifact_directory / "accepted_bundle.json").write_text(
    json.dumps({"schema": "capsule-test-accepted-bundle"}, sort_keys=True)
)
"""
            )
        elif relative == "LICENSE" and symlink_license:
            path.symlink_to("pyproject.toml")
        else:
            path.write_text(f"capsule fixture: {relative}\n")
    package = source / "lib/tile_lifetime/src/tile_lifetime"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("\n")
    executable = package / "generated.py"
    executable.write_text("VALUE = 1\n")
    executable.chmod(0o755)
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "capsule source")
    return source, _git(source, "rev-parse", "HEAD"), _git(source, "rev-parse", "HEAD^{tree}")


def _prepared_remote(tmp_path: Path, *, symlink_license: bool = False) -> tuple[Path, dict, str, str]:
    source, commit, tree = _source_repository(tmp_path, symlink_license=symlink_license)
    prepared = tmp_path / "prepared"
    result = payload.prepare_source_payload(source, commit, prepared)
    remote = tmp_path / "remote"
    with zipfile.ZipFile(io.BytesIO(create_workspace_zip(prepared))) as bundle:
        bundle.extractall(remote)
    return remote, result, commit, tree


def _restore(remote: Path, result: dict, commit: str, tree: str) -> tuple[Path, dict]:
    return payload.restore_and_verify_source_payload(
        remote,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha=commit,
        expected_source_tree=tree,
    )


def test_source_capsule_round_trip_restores_exact_files_modes_and_symlinks(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path, symlink_license=True)

    source_root, manifest = _restore(remote, result, commit, tree)

    assert manifest["source"] == {"commit": commit, "tree": tree}
    expected = {record["path"] for record in manifest["members"]}
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert actual == expected
    assert os.readlink(source_root / "LICENSE") == "pyproject.toml"
    assert stat.S_IMODE((source_root / "lib/tile_lifetime/src/tile_lifetime/generated.py").stat().st_mode) == 0o755


def test_source_capsule_creation_rejects_dirty_global_checkout(tmp_path):
    source, commit, _tree = _source_repository(tmp_path)
    (source / "untracked-secret").write_text("not transportable\n")

    with pytest.raises(ValueError, match="globally clean"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


def test_source_capsule_creation_rejects_worktree_mutation_during_preparation(tmp_path, monkeypatch):
    source, commit, _tree = _source_repository(tmp_path)
    load_allowlist = payload._load_allowlist

    def mutate_after_identity(source_root, tree_records):
        paths = load_allowlist(source_root, tree_records)
        (source_root / "LICENSE").write_text("changed during preparation\n")
        return paths

    monkeypatch.setattr(payload, "_load_allowlist", mutate_after_identity)

    with pytest.raises(ValueError, match="globally clean"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


def test_source_capsule_creation_rejects_allowlist_drift(tmp_path):
    source, _commit, _tree = _source_repository(tmp_path)
    allowlist_path = source / payload.ALLOWLIST_RELATIVE_PATH
    allowlist = json.loads(allowlist_path.read_bytes())
    allowlist["exact"].remove("uv.lock")
    allowlist_path.write_bytes(payload._canonical_json(allowlist))
    _git(source, "add", allowlist_path.relative_to(source).as_posix())
    _git(source, "commit", "--quiet", "-m", "drift allowlist")
    commit = _git(source, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="drifted"):
        payload.prepare_source_payload(source, commit, tmp_path / "prepared")


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "mode"])
def test_source_capsule_verifier_rejects_extracted_file_set_or_identity_changes(tmp_path, mutation):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    source_root, manifest = _restore(remote, result, commit, tree)
    target = source_root / "lib/tile_lifetime/src/tile_lifetime/generated.py"
    if mutation == "changed":
        target.write_text("VALUE = 2\n")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (source_root / "unexpected.py").write_text("unexpected\n")
    else:
        target.chmod(0o644)

    with pytest.raises(ValueError, match=r"file set|mode|content"):
        payload._verify_extracted_source(source_root, manifest)


def test_source_capsule_rejects_corruption_before_extraction(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    capsule = remote / payload.CAPSULE_FILENAME
    capsule.write_bytes(capsule.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="capsule SHA-256"):
        _restore(remote, result, commit, tree)

    assert not (remote / "source").exists()


def test_source_capsule_rejects_member_hash_mismatch_with_matching_archive_hash(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path) as archive:
        entries = {member.filename: archive.read(member) for member in archive.infolist()}
    entries["LICENSE"] = b"changed\n"
    payload._write_capsule(capsule_path, entries)
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match=r"member size|member SHA-256"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )


def test_source_capsule_rejects_traversal_with_matching_transport_hashes(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path, "a") as archive:
        info = zipfile.ZipInfo("../credential", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        archive.writestr(info, b"secret")
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"].insert(
        0,
        {
            "mode": "0644",
            "path": "../credential",
            "sha256": hashlib.sha256(b"secret").hexdigest(),
            "size": 6,
            "type": "file",
        },
    )
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match="normalized relative path"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )
    assert not (remote / "source").exists()
    assert not (remote.parent / "credential").exists()


def test_source_capsule_rejects_escaping_symlink_with_matching_transport_hashes(tmp_path):
    remote, _result, commit, tree = _prepared_remote(tmp_path, symlink_license=True)
    capsule_path = remote / payload.CAPSULE_FILENAME
    with zipfile.ZipFile(capsule_path) as archive:
        entries = {member.filename: archive.read(member) for member in archive.infolist()}
    entries["LICENSE"] = b"../outside"
    payload._write_capsule(capsule_path, entries)
    manifest_path = remote / payload.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    record = next(record for record in manifest["members"] if record["path"] == "LICENSE")
    record["size"] = len(entries["LICENSE"])
    record["sha256"] = hashlib.sha256(entries["LICENSE"]).hexdigest()
    manifest["archive"]["sha256"] = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
    raw = payload._canonical_json(manifest)
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match="escapes"):
        payload.restore_and_verify_source_payload(
            remote,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha=commit,
            expected_source_tree=tree,
        )


def test_source_capsule_runner_command_fixes_manifest_execute_and_tool_paths(tmp_path):
    arguments = payload.runner_arguments(
        Path(sys.executable),
        tmp_path,
        "1" * 40,
        "2" * 40,
        tmp_path / payload.MANIFEST_FILENAME,
        "3" * 64,
        tmp_path.parent / "artifacts",
        "0.10.1",
    )

    assert arguments[0] == sys.executable
    assert arguments[2:4] == ("--execute", "--source-root")
    assert arguments[arguments.index("--source-tree") + 1] == "2" * 40
    assert arguments[arguments.index("--source-capsule-manifest-sha256") + 1] == "3" * 64
    assert arguments[arguments.index("--nvcc") + 1] == "/usr/local/cuda-13.2/bin/nvcc"
    assert arguments[arguments.index("--ncu") + 1] == "/usr/local/bin/ncu"
    assert arguments[arguments.index("--nsys") + 1] == "/usr/local/bin/nsys"

    with pytest.raises(ValueError, match="absolute executable file"):
        payload.runner_arguments(
            Path("python"),
            tmp_path,
            "1" * 40,
            "2" * 40,
            tmp_path / payload.MANIFEST_FILENAME,
            "3" * 64,
            tmp_path.parent / "artifacts",
            "0.10.1",
        )


def test_source_capsule_launcher_uses_selected_runtime_without_path_fallback(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    selected_python = Path(sys.executable)
    bootstrap_python = selected_python.resolve()
    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "unqualified-python-ran"
    poison_python = poison / "python"
    poison_python.write_text(f"#!/bin/sh\ntouch {marker}\nexit 97\n")
    poison_python.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PATH"] = str(poison)
    environment.pop("PYTHONPATH", None)

    subprocess.run(
        (
            str(bootstrap_python),
            str(remote / payload.LAUNCHER_FILENAME),
            "run",
            "--workspace",
            str(remote),
            "--launcher-sha256",
            result["launcher_sha256"],
            "--manifest-sha256",
            result["manifest_sha256"],
            "--source-sha",
            commit,
            "--source-tree",
            tree,
            "--artifact-directory",
            str(artifacts),
            "--require-jax-version",
            "0.10.1",
            "--runtime-python",
            str(selected_python),
        ),
        cwd=remote,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    imports = json.loads((artifacts / "runtime-imports.json").read_text())
    assert Path(imports["python"]) == selected_python
    assert imports["jax"] == "0.10.1"
    assert imports["numpy"]
    assert imports["shuttle"] == "0.1.0"
    assert not marker.exists()


def test_source_capsule_artifact_export_is_closed_and_requires_exact_ack(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "accepted_bundle.json").write_text('{"accepted":true}\n')
    nested = artifacts / "profiles"
    nested.mkdir()
    (nested / "kernel.sass").write_bytes(b"exact-sass\n")
    export = tmp_path / "export"

    ready = payload.prepare_artifact_export(
        artifacts,
        export,
        runner_exit_code=0,
        source_sha="1" * 40,
        source_tree="2" * 40,
        source_manifest_sha256="3" * 64,
    )

    assert ready["runner_exit_code"] == 0
    assert ready["wrapper_exit_code"] == 0
    manifest_path = export / payload.ARTIFACT_EXPORT_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["members"] == [
        {
            "mode": "0644",
            "path": "accepted_bundle.json",
            "sha256": hashlib.sha256(b'{"accepted":true}\n').hexdigest(),
            "size": 18,
        },
        {
            "mode": "0644",
            "path": "profiles/kernel.sass",
            "sha256": hashlib.sha256(b"exact-sass\n").hexdigest(),
            "size": 11,
        },
    ]
    archive = b"".join((export / part["filename"]).read_bytes() for part in manifest["parts"])
    assert hashlib.sha256(archive).hexdigest() == manifest["archive"]["sha256"]
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    for filename in (
        payload.ARTIFACT_EXPORT_READY_FILENAME,
        payload.ARTIFACT_EXPORT_MANIFEST_FILENAME,
        *(part["filename"] for part in manifest["parts"]),
    ):
        (retrieval / filename).write_bytes((export / filename).read_bytes())
    archive_path = tmp_path / "retrieved.tar.gz"
    ack_path = tmp_path / "retrieved.ack"
    payload.main(
        (
            "verify-export",
            "--retrieval-directory",
            str(retrieval),
            "--archive-output",
            str(archive_path),
            "--ack-output",
            str(ack_path),
            "--source-sha",
            "1" * 40,
            "--source-tree",
            "2" * 40,
            "--source-manifest-sha256",
            "3" * 64,
        )
    )
    validation_summary = json.loads(capsys.readouterr().out)
    assert validation_summary["archive_sha256"] == ready["archive_sha256"]
    assert validation_summary["manifest_sha256"] == ready["manifest_sha256"]
    assert archive_path.read_bytes() == archive
    assert ack_path.read_bytes() == payload.artifact_export_ack(ready)
    with tarfile.open(archive_path, "r:gz") as tar:
        assert tuple(member.name for member in tar.getmembers()) == (
            "accepted_bundle.json",
            "profiles/kernel.sass",
        )

    (export / payload.ARTIFACT_EXPORT_ACK_FILENAME).write_text("wrong\n")
    with pytest.raises(ValueError, match="exact export identities"):
        payload.wait_for_artifact_ack(export, ready, timeout_seconds=0.01, poll_seconds=0.001)
    (export / payload.ARTIFACT_EXPORT_ACK_FILENAME).write_bytes(payload.artifact_export_ack(ready))
    payload.wait_for_artifact_ack(export, ready, timeout_seconds=0.01, poll_seconds=0.001)


@pytest.mark.parametrize("mutation", ["corrupt_part", "missing_part", "extra_file", "wrong_source", "boolean_schema"])
def test_source_capsule_retrieved_export_rejects_transport_drift(tmp_path, mutation):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "accepted_bundle.json").write_text('{"accepted":true}\n')
    export = tmp_path / "export"
    payload.prepare_artifact_export(
        artifacts,
        export,
        runner_exit_code=0,
        source_sha="1" * 40,
        source_tree="2" * 40,
        source_manifest_sha256="3" * 64,
    )
    manifest = json.loads((export / payload.ARTIFACT_EXPORT_MANIFEST_FILENAME).read_bytes())
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    filenames = [
        payload.ARTIFACT_EXPORT_READY_FILENAME,
        payload.ARTIFACT_EXPORT_MANIFEST_FILENAME,
        *(part["filename"] for part in manifest["parts"]),
    ]
    for filename in filenames:
        (retrieval / filename).write_bytes((export / filename).read_bytes())
    expected_source = "1" * 40
    if mutation == "corrupt_part":
        (retrieval / manifest["parts"][0]["filename"]).write_bytes(b"corrupt")
    elif mutation == "missing_part":
        (retrieval / manifest["parts"][0]["filename"]).unlink()
    elif mutation == "extra_file":
        (retrieval / "unsealed").write_text("extra")
    elif mutation == "wrong_source":
        expected_source = "4" * 40
    else:
        manifest_path = retrieval / payload.ARTIFACT_EXPORT_MANIFEST_FILENAME
        mutated_manifest = json.loads(manifest_path.read_bytes())
        mutated_manifest["schema_version"] = True
        manifest_path.write_bytes(payload._canonical_json(mutated_manifest))
        ready_path = retrieval / payload.ARTIFACT_EXPORT_READY_FILENAME
        mutated_ready = json.loads(ready_path.read_bytes())
        mutated_ready["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        mutated_ready["schema_version"] = True
        ready_path.write_bytes(payload._canonical_json(mutated_ready))

    with pytest.raises(ValueError):
        payload.validate_retrieved_artifact_export(
            retrieval,
            tmp_path / "retrieved.tar.gz",
            expected_source_sha=expected_source,
            expected_source_tree="2" * 40,
            expected_source_manifest_sha256="3" * 64,
        )


def test_source_capsule_artifact_export_rejects_noncanonical_file_mode(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    accepted = artifacts / "accepted_bundle.json"
    accepted.write_text('{"accepted":true}\n')
    accepted.chmod(0o600)

    with pytest.raises(ValueError, match="noncanonical mode"):
        payload.prepare_artifact_export(
            artifacts,
            tmp_path / "export",
            runner_exit_code=0,
            source_sha="1" * 40,
            source_tree="2" * 40,
            source_manifest_sha256="3" * 64,
        )


def test_source_capsule_artifact_export_turns_missing_success_bundle_into_failure(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    ready = payload.prepare_artifact_export(
        artifacts,
        tmp_path / "export",
        runner_exit_code=0,
        source_sha="1" * 40,
        source_tree="2" * 40,
        source_manifest_sha256="3" * 64,
    )

    assert ready["runner_exit_code"] == 0
    assert ready["wrapper_exit_code"] != 0
    assert ready["accepted_bundle_present"] is False


def test_source_capsule_run_export_keeps_process_alive_until_exact_ack(tmp_path):
    remote, result, commit, tree = _prepared_remote(tmp_path)
    artifacts = tmp_path / "artifacts"
    export = tmp_path / "export"
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        (
            sys.executable,
            str(remote / payload.LAUNCHER_FILENAME),
            "run-export",
            "--workspace",
            str(remote),
            "--launcher-sha256",
            result["launcher_sha256"],
            "--manifest-sha256",
            result["manifest_sha256"],
            "--source-sha",
            commit,
            "--source-tree",
            tree,
            "--artifact-directory",
            str(artifacts),
            "--export-directory",
            str(export),
            "--ack-timeout-seconds",
            "10",
            "--require-jax-version",
            "0.10.1",
            "--runtime-python",
            sys.executable,
        ),
        cwd=remote,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready_path = export / payload.ARTIFACT_EXPORT_READY_FILENAME
    deadline = time.monotonic() + 10
    while not ready_path.is_file() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_path.is_file()
    assert process.poll() is None
    ready = json.loads(ready_path.read_bytes())
    (export / payload.ARTIFACT_EXPORT_ACK_FILENAME).write_text("wrong\n")
    time.sleep(0.05)
    assert process.poll() is None
    (export / payload.ARTIFACT_EXPORT_ACK_FILENAME).write_bytes(payload.artifact_export_ack(ready))

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, stderr
    assert payload.ARTIFACT_EXPORT_READY_FILENAME in stdout


def test_source_capsule_python_path_prefers_capsule_package_and_benchmark_roots(tmp_path):
    assert payload.capsule_python_path(tmp_path, "/image/site-packages") == os.pathsep.join(
        (str(tmp_path), str(tmp_path / "lib/tile_lifetime/src"), "/image/site-packages")
    )


def test_source_capsule_launcher_identity_rejects_tampering(tmp_path):
    launcher = tmp_path / payload.LAUNCHER_FILENAME
    launcher.write_text("trusted\n")
    identity = hashlib.sha256(launcher.read_bytes()).hexdigest()

    payload.verify_launcher_identity(launcher, identity)
    launcher.write_text("tampered\n")
    with pytest.raises(ValueError, match="trusted launch identity"):
        payload.verify_launcher_identity(launcher, identity)


def test_real_repository_capsule_iris_bundle_is_complete_and_under_limit(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    source = tmp_path / "source-repository"
    subprocess.run(
        ("git", "clone", "--quiet", "--no-local", "--depth=1", repository.as_uri(), str(source)),
        check=True,
    )
    source_sha = _git(source, "rev-parse", "HEAD")
    source_tree = _git(source, "rev-parse", "HEAD^{tree}")
    prepared = tmp_path / "prepared"
    result = payload.prepare_source_payload(source, source_sha, prepared)

    bundle_bytes = create_workspace_zip(prepared)
    assert len(bundle_bytes) < MAX_BUNDLE_SIZE_BYTES
    assert {path.name for path in prepared.iterdir()} == {
        payload.CAPSULE_FILENAME,
        payload.LAUNCHER_FILENAME,
        payload.MANIFEST_FILENAME,
    }
    launcher_record = next(
        record
        for record in result["members"]
        if record["path"] == "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py"
    )
    assert result["launcher_sha256"] == launcher_record["sha256"]
    remote = tmp_path / "remote"
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
        bundle.extractall(remote)
    source_root, manifest = _restore(remote, result, source_sha, source_tree)

    payload._verify_extracted_source(source_root, manifest)
    assert len(manifest["members"]) >= 150
    assert (source_root / payload.RUNNER_RELATIVE_PATH).is_file()

    audit_script = """
import importlib
import sys
from pathlib import Path
runner = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner")
importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")
tool = Path("/capsule-audit-tool")
config = runner.RunnerConfig(
    source_root=Path(sys.argv[1]),
    source_sha=sys.argv[2],
    source_tree=sys.argv[3],
    source_capsule_manifest=Path(sys.argv[4]),
    source_capsule_manifest_sha256=sys.argv[5],
    artifact_directory=Path(sys.argv[1]).parent / "artifacts",
    tools=runner.ToolPaths(tool, tool, tool, tool, tool, tool, tool),
    require_jax_version="0.10.1",
)
runner.audit_imported_local_modules(config)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = payload.capsule_python_path(source_root, environment.get("PYTHONPATH"))
    subprocess.run(
        (
            sys.executable,
            "-c",
            audit_script,
            str(source_root),
            source_sha,
            source_tree,
            str(remote / payload.MANIFEST_FILENAME),
            result["manifest_sha256"],
        ),
        cwd=remote,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
