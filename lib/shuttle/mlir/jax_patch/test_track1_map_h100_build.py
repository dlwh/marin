# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
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
    offline_wheel_command,
    require_offline_build_inputs,
)


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
    (inputs.jax_repository / "build").mkdir(parents=True)
    (inputs.jax_repository / "build/build.py").write_text("# pinned JAX builder\n")


def _tool_digests(inputs: BuildInputs) -> BuildToolDigests:
    return BuildToolDigests(
        bazel=inputs.bazel_sha256,
        image=inputs.image_sha256,
        nvcc=inputs.nvcc_sha256,
        ptxas=inputs.ptxas_sha256,
    )


def test_offline_build_preflight_is_closed_over_source_tools_and_image(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
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


def test_source_patching_checks_both_repositories_before_applying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    run = Mock()
    monkeypatch.setattr("track1_map_h100_build.subprocess.run", run)

    apply_source_patches(inputs, tmp_path / "marin")

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

    with pytest.raises(RuntimeError, match="xla check failed"):
        apply_source_patches(inputs, tmp_path / "marin")

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
    assert "--cuda_compute_capabilities=9.0" in command
    assert "--disable_nccl" in command
    assert "--bazel_options=--define=SHUTTLE_TEST_OBSERVER=1" in command
    assert "--bazel_options=--repository_disable_download" in command
    assert f"--bazel_options=--override_repository=xla={tmp_path / 'xla'}" in command
    assert f"--bazel_options=--override_repository=shuttle_mlir={tmp_path / 'marin/lib/shuttle/mlir'}" in command
    assert all("http://" not in argument and "https://" not in argument for argument in command)
