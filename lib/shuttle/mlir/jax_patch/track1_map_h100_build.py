# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline build preflight for the narrow Track 1 H100 subject."""

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

JAX_COMMIT = "619764c15117fbefc4ba13ab941871cb514c23f6"
JAX_TREE = "56d186bccb48075374f32e46d2e090731b66e1ea"
XLA_COMMIT = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
PATCH_NAMES = (
    "0001-link-shuttle-xla-registry-adapter.patch",
    "0002-add-acceptance-observer-bridge.patch",
    "0003-link-shuttle-cuda-ffi.patch",
)


@dataclass(frozen=True)
class BuildInputs:
    bazel: Path
    cuda_root: Path
    image_archive: Path
    jax_commit: str
    jax_repository: Path
    jax_tree: str
    patch_sha256: tuple[str, ...]
    platform: str
    xla_commit: str


def _regular(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"offline {name} must be one regular file")


def require_offline_build_inputs(
    inputs: BuildInputs,
    *,
    expected_patch_sha256: tuple[str, ...],
) -> None:
    """Reject incomplete or drifting inputs before an offline wheel build."""
    if inputs.platform != "linux":
        raise RuntimeError("offline CUDA wheel construction requires Linux")
    if inputs.jax_commit != JAX_COMMIT:
        raise RuntimeError("offline JAX commit differs from the pinned source")
    if inputs.jax_tree != JAX_TREE:
        raise RuntimeError("offline JAX tree differs from the pinned source")
    if inputs.xla_commit != XLA_COMMIT:
        raise RuntimeError("offline XLA commit differs from the pinned source")
    if inputs.patch_sha256 != expected_patch_sha256 or len(inputs.patch_sha256) != len(PATCH_NAMES):
        raise RuntimeError("offline JAX patch roster or bytes differ")
    _regular(inputs.bazel, "Bazel binary")
    _regular(inputs.cuda_root / "bin/ptxas", "ptxas binary")
    _regular(inputs.cuda_root / "bin/nvcc", "nvcc binary")
    _regular(inputs.image_archive, "image archive")
    _regular(inputs.jax_repository / "build/build.py", "JAX build entrypoint")


def offline_wheel_command(
    *,
    jax_repository: Path,
    marin_repository: Path,
    bazel: Path,
    bazel_output_root: Path,
    wheel_output: Path,
) -> list[str]:
    """Return the closed, download-disabled patched wheel build argv."""
    return [
        str(jax_repository / "build/build.py"),
        "build",
        "--wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt",
        f"--bazel_path={bazel}",
        f"--output_path={wheel_output}",
        "--cuda_version=13.0.0",
        "--cuda_compute_capabilities=9.0",
        "--disable_nccl",
        "--bazel_options=--repo_env=ML_WHEEL_TYPE=release",
        "--bazel_options=--define=SHUTTLE_TEST_OBSERVER=1",
        "--bazel_options=--repository_disable_download",
        f"--bazel_startup_options=--output_user_root={bazel_output_root}",
        f"--bazel_options=--override_repository=shuttle_mlir={marin_repository / 'lib/shuttle/mlir'}",
    ]


def patch_sha256(marin_repository: Path) -> tuple[str, ...]:
    patch_root = marin_repository / "lib/shuttle/mlir/jax_patch"
    return tuple(hashlib.sha256((patch_root / name).read_bytes()).hexdigest() for name in PATCH_NAMES)


def _git(repository: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", revision],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inspect_build_inputs(
    *,
    marin_repository: Path,
    jax_repository: Path,
    bazel: Path,
    cuda_root: Path,
    image_archive: Path,
) -> BuildInputs:
    """Inspect local inputs without downloading, building, or starting an image."""
    if not (jax_repository / "build/build.py").is_file():
        jax_commit = "missing"
        jax_tree = "missing"
        xla_commit = "missing"
    else:
        jax_commit = _git(jax_repository, "HEAD")
        jax_tree = _git(jax_repository, "HEAD^{tree}")
        xla_commit = _git(jax_repository / "third_party/xla", "HEAD")
    return BuildInputs(
        bazel=bazel,
        cuda_root=cuda_root,
        image_archive=image_archive,
        jax_commit=jax_commit,
        jax_repository=jax_repository,
        jax_tree=jax_tree,
        patch_sha256=patch_sha256(marin_repository),
        platform=platform.system().lower(),
        xla_commit=xla_commit,
    )
