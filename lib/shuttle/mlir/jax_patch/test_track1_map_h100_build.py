# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from track1_map_h100_build import (
    JAX_COMMIT,
    JAX_PATCH_SHA256,
    JAX_TREE,
    XLA_COMMIT,
    XLA_PATCH_SHA256,
    BuildInputs,
    BuildToolDigests,
    apply_source_patches,
    inspect_build_inputs,
    offline_wheel_command,
    require_offline_build_inputs,
    require_source_checkouts_unchanged,
    source_checkouts_identity,
)

MARIN_REPOSITORY = Path(__file__).parents[4]
PATCH_SOURCE_CAPSULES = MARIN_REPOSITORY / "lib/shuttle/mlir/jax_patch/testdata"


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Track One", "-c", "user.email=track1@example.invalid", *arguments],
        cwd=repository,
        check=True,
    )


def _git_repository(repository: Path, files: dict[str, str]) -> None:
    repository.mkdir(exist_ok=True)
    for relative_path, content in files.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if relative_path == "build/build.py":
            path.chmod(0o755)
    _run_git(repository, "init", "-q")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "-qm", "fixture")


def _inputs(tmp_path: Path) -> BuildInputs:
    return BuildInputs(
        bazel=tmp_path / "bazel",
        bazel_sha256=hashlib.sha256(b"bazel").hexdigest(),
        cuda_root=tmp_path / "cuda",
        image_archive=tmp_path / "base-image.oci",
        image_sha256=hashlib.sha256(b"oci").hexdigest(),
        jax_commit=JAX_COMMIT,
        jax_patch_sha256=JAX_PATCH_SHA256,
        jax_repository=tmp_path / "jax",
        jax_status="",
        jax_tree=JAX_TREE,
        marin_repository=MARIN_REPOSITORY,
        nvcc_sha256=hashlib.sha256(b"nvcc").hexdigest(),
        platform="linux",
        ptxas_sha256=hashlib.sha256(b"ptxas").hexdigest(),
        xla_commit=XLA_COMMIT,
        xla_patch_sha256=XLA_PATCH_SHA256,
        xla_repository=tmp_path / "xla",
        xla_status="",
    )


def _materialize(inputs: BuildInputs) -> None:
    inputs.bazel.write_bytes(b"bazel")
    inputs.bazel.chmod(0o755)
    inputs.image_archive.write_bytes(b"oci")
    (inputs.cuda_root / "bin").mkdir(parents=True)
    (inputs.cuda_root / "bin/ptxas").write_bytes(b"ptxas")
    (inputs.cuda_root / "bin/ptxas").chmod(0o755)
    (inputs.cuda_root / "bin/nvcc").write_bytes(b"nvcc")
    (inputs.cuda_root / "bin/nvcc").chmod(0o755)
    (inputs.jax_repository / "build").mkdir(parents=True, exist_ok=True)
    (inputs.jax_repository / "build/build.py").write_text("# pinned JAX builder\n")
    (inputs.jax_repository / "build/build.py").chmod(0o755)


def _tool_digests(inputs: BuildInputs) -> BuildToolDigests:
    return BuildToolDigests(
        bazel=inputs.bazel_sha256,
        image=inputs.image_sha256,
        nvcc=inputs.nvcc_sha256,
        ptxas=inputs.ptxas_sha256,
    )


def _patch_source_repository(tmp_path: Path, capsule_name: str) -> Path:
    capsule = PATCH_SOURCE_CAPSULES / capsule_name
    raw_manifest = capsule.joinpath("source.json").read_bytes()
    manifest = json.loads(raw_manifest)
    assert raw_manifest == json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    repository = tmp_path / capsule_name
    repository.mkdir()
    for entry in manifest["entries"]:
        source = capsule / entry["path"]
        assert source.stat().st_size == entry["size"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        assert (
            subprocess.run(["git", "hash-object", source], check=True, capture_output=True, text=True).stdout.strip()
            == entry["blob"]
        )
        destination = repository / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    _run_git(repository, "init", "-q")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "-qm", f"exact {manifest['commit']} patch source capsule")
    return repository


def test_exact_pinned_source_capsules_accept_both_complete_ordered_patch_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax_repository = _patch_source_repository(tmp_path, "jax_619764c1")
    xla_repository = _patch_source_repository(tmp_path, "xla_9b6359")
    inputs = replace(_inputs(tmp_path), jax_repository=jax_repository, xla_repository=xla_repository)
    monkeypatch.setattr("track1_map_h100_build._require_live_source_snapshot", lambda value: None)

    apply_source_patches(inputs)

    jax_module = jax_repository.joinpath("jaxlib/jax.cc").read_text()
    assert '#include "shuttle/Transforms/XlaRegistration.h"' in jax_module
    assert "registerShuttleXlaTransform()" in jax_module
    xla_client = xla_repository.joinpath("xla/pjrt/c_api_client/pjrt_c_api_client.cc").read_text()
    assert "ApplyStablehloModuleTransform" in xla_client


def test_offline_build_preflight_is_closed_over_source_tools_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    monkeypatch.setattr("track1_map_h100_build._require_live_source_snapshot", lambda value: None)
    require_offline_build_inputs(inputs, expected_tool_sha256=_tool_digests(inputs))

    mutations = (
        (replace(inputs, platform="darwin"), "Linux"),
        (replace(inputs, jax_commit="d" * 40), "JAX commit"),
        (replace(inputs, jax_tree="e" * 40), "JAX tree"),
        (replace(inputs, xla_commit="f" * 40), "XLA commit"),
        (replace(inputs, jax_status=" M jaxlib/BUILD"), "JAX checkout"),
        (replace(inputs, xla_status="?? xla/pjrt/new.cc"), "XLA checkout"),
        (replace(inputs, jax_patch_sha256=JAX_PATCH_SHA256[:-1]), "JAX patch"),
        (replace(inputs, xla_patch_sha256=XLA_PATCH_SHA256[::-1]), "XLA patch"),
    )
    for mutated, message in mutations:
        with pytest.raises(RuntimeError, match=message):
            require_offline_build_inputs(mutated, expected_tool_sha256=_tool_digests(inputs))

    drifted_tools = replace(_tool_digests(inputs), image="9" * 64)
    with pytest.raises(RuntimeError, match="tool or image"):
        require_offline_build_inputs(inputs, expected_tool_sha256=drifted_tools)

    inputs.image_archive.unlink()
    with pytest.raises(RuntimeError, match="image archive"):
        require_offline_build_inputs(inputs, expected_tool_sha256=_tool_digests(inputs))


def test_offline_build_gate_rehashes_live_inputs_and_requires_executable_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    monkeypatch.setattr("track1_map_h100_build._require_live_source_snapshot", lambda value: None)
    expected = _tool_digests(inputs)

    inputs.bazel.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="changed after inspection"):
        require_offline_build_inputs(inputs, expected_tool_sha256=expected)

    inputs.bazel.write_bytes(b"bazel")
    inputs.jax_repository.joinpath("build/build.py").chmod(0o644)
    with pytest.raises(RuntimeError, match="build entrypoint must be executable"):
        require_offline_build_inputs(inputs, expected_tool_sha256=expected)


def test_inspection_reads_separate_git_repositories_and_live_file_hashes(tmp_path: Path) -> None:
    jax_repository = tmp_path / "jax"
    xla_repository = tmp_path / "xla"
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    _git_repository(jax_repository, {"build/build.py": "#!/usr/bin/env python3\n"})
    _git_repository(xla_repository, {"xla/pjrt/marker": "xla\n"})

    inspected = inspect_build_inputs(
        marin_repository=MARIN_REPOSITORY,
        jax_repository=jax_repository,
        bazel=inputs.bazel,
        cuda_root=inputs.cuda_root,
        image_archive=inputs.image_archive,
        xla_repository=xla_repository,
    )

    assert inspected.jax_status == ""
    assert inspected.xla_status == ""
    assert inspected.bazel_sha256 == hashlib.sha256(b"bazel").hexdigest()
    assert inspected.xla_repository == xla_repository

    (xla_repository / "xla/pjrt/marker").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="source changed after inspection"):
        apply_source_patches(inspected)
    assert inspect_build_inputs(
        marin_repository=MARIN_REPOSITORY,
        jax_repository=jax_repository,
        bazel=inputs.bazel,
        cuda_root=inputs.cuda_root,
        image_archive=inputs.image_archive,
        xla_repository=xla_repository,
    ).xla_status.startswith("M ")


def test_hydration_source_identity_rejects_tracked_untracked_and_head_drift(tmp_path: Path) -> None:
    jax = tmp_path / "jax"
    xla = tmp_path / "xla"
    _git_repository(jax, {"source.cc": "jax\n"})
    _git_repository(xla, {"source.cc": "xla\n"})
    (jax / "source.cc").write_text("approved jax patch\n")
    (xla / "source.cc").write_text("approved xla patch\n")
    approved = source_checkouts_identity(jax_repository=jax, xla_repository=xla)

    require_source_checkouts_unchanged(approved, jax_repository=jax, xla_repository=xla)

    for mutation in ("tracked", "untracked", "head"):
        case_jax = tmp_path / f"jax-{mutation}"
        case_xla = tmp_path / f"xla-{mutation}"
        _git_repository(case_jax, {"source.cc": "jax\n"})
        _git_repository(case_xla, {"source.cc": "xla\n"})
        (case_jax / "source.cc").write_text("approved jax patch\n")
        (case_xla / "source.cc").write_text("approved xla patch\n")
        expected = source_checkouts_identity(jax_repository=case_jax, xla_repository=case_xla)
        if mutation == "tracked":
            (case_jax / "source.cc").write_text("hydration mutation\n")
        elif mutation == "untracked":
            (case_xla / "injected.bzl").write_text("hydration mutation\n")
        else:
            _run_git(case_jax, "add", "source.cc")
            _run_git(case_jax, "commit", "-m", "unexpected hydration commit")

        with pytest.raises(RuntimeError, match="hydration changed patched JAX or XLA source"):
            require_source_checkouts_unchanged(expected, jax_repository=case_jax, xla_repository=case_xla)


def test_source_patching_checks_both_repositories_before_applying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    run = Mock()
    monkeypatch.setattr("track1_map_h100_build.subprocess.run", run)
    monkeypatch.setattr("track1_map_h100_build._require_live_source_snapshot", lambda value: None)

    apply_source_patches(inputs)

    commands = [call.args[0] for call in run.call_args_list]
    assert commands[0][0:3] == ["git", "apply", "--check"]
    assert commands[1][0:3] == ["git", "apply", "--check"]
    assert commands[2][0:2] == ["git", "apply"]
    assert commands[3][0:2] == ["git", "apply"]
    assert run.call_args_list[0].kwargs["cwd"] == inputs.jax_repository
    assert run.call_args_list[1].kwargs["cwd"] == inputs.xla_repository
    assert run.call_args_list[2].kwargs["cwd"] == inputs.jax_repository
    assert run.call_args_list[3].kwargs["cwd"] == inputs.xla_repository


def test_source_patch_check_failure_applies_neither_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    run = Mock(side_effect=[None, RuntimeError("xla check failed")])
    monkeypatch.setattr("track1_map_h100_build.subprocess.run", run)
    monkeypatch.setattr("track1_map_h100_build._require_live_source_snapshot", lambda value: None)

    with pytest.raises(RuntimeError, match="xla check failed"):
        apply_source_patches(inputs)

    assert run.call_count == 2


def test_wheel_command_is_gpu_observable_and_download_disabled(tmp_path: Path) -> None:
    command = offline_wheel_command(
        jax_repository=tmp_path / "jax",
        marin_repository=tmp_path / "marin",
        bazel=tmp_path / "bazel",
        bazel_output_root=tmp_path / "bazel-output",
        wheel_output=tmp_path / "wheels",
        xla_repository=tmp_path / "xla",
    )

    assert command[:3] == [
        str(tmp_path / "jax/build/build.py"),
        "build",
        "--wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt",
    ]
    assert "--cuda_version=13.0.0" in command
    assert "--cuda_compute_capabilities=9.0" in command
    assert "--python_version=3.12" in command
    assert "--disable_nccl" in command
    assert "--bazel_options=--define=SHUTTLE_TEST_OBSERVER=1" in command
    assert "--bazel_options=--repository_disable_download" in command
    assert all("LOCAL_CUDA_PATH" not in argument for argument in command)
    assert f"--bazel_options=--override_repository=xla={tmp_path / 'xla'}" in command
    assert f"--bazel_options=--override_repository=shuttle_mlir={tmp_path / 'marin/lib/shuttle/mlir'}" in command
    assert all("http://" not in argument and "https://" not in argument for argument in command)
