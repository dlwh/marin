# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from track1_map_h100_build import (
    JAX_COMMIT,
    JAX_TREE,
    XLA_COMMIT,
    BuildInputs,
    offline_wheel_command,
    require_offline_build_inputs,
)


def _inputs(tmp_path: Path) -> BuildInputs:
    return BuildInputs(
        bazel=tmp_path / "bazel",
        cuda_root=tmp_path / "cuda",
        image_archive=tmp_path / "base-image.oci",
        jax_commit=JAX_COMMIT,
        jax_repository=tmp_path / "jax",
        jax_tree=JAX_TREE,
        patch_sha256=("a" * 64, "b" * 64, "c" * 64),
        platform="linux",
        xla_commit=XLA_COMMIT,
    )


def _materialize(inputs: BuildInputs) -> None:
    inputs.bazel.write_bytes(b"bazel")
    inputs.image_archive.write_bytes(b"oci")
    (inputs.cuda_root / "bin").mkdir(parents=True)
    (inputs.cuda_root / "bin/ptxas").write_bytes(b"ptxas")
    (inputs.cuda_root / "bin/nvcc").write_bytes(b"nvcc")
    (inputs.jax_repository / "build").mkdir(parents=True)
    (inputs.jax_repository / "build/build.py").write_text("# pinned JAX builder\n")


def test_offline_build_preflight_is_closed_over_source_tools_and_image(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _materialize(inputs)
    require_offline_build_inputs(inputs, expected_patch_sha256=inputs.patch_sha256)

    mutations = (
        (replace(inputs, platform="darwin"), "Linux"),
        (replace(inputs, jax_commit="d" * 40), "JAX commit"),
        (replace(inputs, jax_tree="e" * 40), "JAX tree"),
        (replace(inputs, xla_commit="f" * 40), "XLA commit"),
        (replace(inputs, patch_sha256=("a" * 64, "b" * 64)), "patch"),
    )
    for mutated, message in mutations:
        with pytest.raises(RuntimeError, match=message):
            require_offline_build_inputs(mutated, expected_patch_sha256=inputs.patch_sha256)

    inputs.image_archive.unlink()
    with pytest.raises(RuntimeError, match="image archive"):
        require_offline_build_inputs(inputs, expected_patch_sha256=inputs.patch_sha256)


def test_wheel_command_is_gpu_observable_and_download_disabled(tmp_path: Path) -> None:
    command = offline_wheel_command(
        jax_repository=tmp_path / "jax",
        marin_repository=tmp_path / "marin",
        bazel=tmp_path / "bazel",
        bazel_output_root=tmp_path / "bazel-output",
        wheel_output=tmp_path / "wheels",
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
    assert f"--bazel_options=--override_repository=shuttle_mlir={tmp_path / 'marin/lib/shuttle/mlir'}" in command
    assert all("http://" not in argument and "https://" not in argument for argument in command)
