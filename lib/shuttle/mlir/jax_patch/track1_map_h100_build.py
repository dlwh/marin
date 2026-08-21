# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline build preflight for the narrow Track 1 H100 subject."""

import hashlib
import os
import platform
import stat
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
JAX_PATCH_SHA256 = (
    "1e8b1400ee05bf8c8037277046113e1f1b1fae5ef56077899b7e853855e3424e",
    "c4d5bc4aaa4b72ee7e2ecb44d28719be99da8defcf01ebd107308df41cb71942",
    "a98f04ac7afc6e5d8df63f2679cf89bc7cd0b29509a393da86f352cb6ba2ec9f",
)
XLA_PATCH_NAMES = (
    "0001-add-stablehlo-module-transform-hook.patch",
    "0002-anchor-lit-labels-to-xla-repository.patch",
    "0003-test-c-api-transform-boundary.patch",
)
XLA_PATCH_SHA256 = (
    "8e50020606758a431f7e96b80e2720ef5ca40986746449cc6a20ab6ae9c7bb1e",
    "e4f3121f3123d7e2ee781cc5ee92f1ddeb0df662b6702d37615bcae821ecbc99",
    "a0aa22a0c4189be6a76c3ad4969b2983ea3c6f84cc277ab7ecfbd786581c0151",
)
HERMETIC_CUDA_VERSION = "13.0.0"
HERMETIC_CUDA_COMPUTE_CAPABILITIES = ("9.0",)


@dataclass(frozen=True)
class BuildToolDigests:
    bazel: str
    image: str
    nvcc: str
    ptxas: str


@dataclass(frozen=True)
class BuildInputs:
    bazel: Path
    bazel_sha256: str
    cuda_root: Path
    image_archive: Path
    image_sha256: str
    jax_commit: str
    jax_patch_sha256: tuple[str, ...]
    jax_repository: Path
    jax_status: str
    jax_tree: str
    marin_repository: Path
    nvcc_sha256: str
    platform: str
    ptxas_sha256: str
    xla_commit: str
    xla_patch_sha256: tuple[str, ...]
    xla_repository: Path
    xla_status: str


@dataclass(frozen=True)
class SourceCheckoutsIdentity:
    jax_commit: str
    jax_diff_sha256: str
    jax_status_sha256: str
    jax_tree: str
    jax_untracked_sha256: str
    xla_commit: str
    xla_diff_sha256: str
    xla_status_sha256: str
    xla_tree: str
    xla_untracked_sha256: str


def _regular(path: Path, name: str, *, executable: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"offline {name} must be one regular file")
    if executable and not os.access(path, os.X_OK):
        raise RuntimeError(f"offline {name} must be executable")


def require_offline_build_inputs(
    inputs: BuildInputs,
    *,
    expected_tool_sha256: BuildToolDigests,
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
    if inputs.jax_status:
        raise RuntimeError("offline JAX checkout must be clean")
    if inputs.xla_status:
        raise RuntimeError("offline XLA checkout must be clean")
    if inputs.jax_patch_sha256 != JAX_PATCH_SHA256:
        raise RuntimeError("offline JAX patch roster or bytes differ")
    if inputs.xla_patch_sha256 != XLA_PATCH_SHA256:
        raise RuntimeError("offline XLA patch roster or bytes differ")
    _require_live_source_snapshot(inputs)
    _regular(inputs.bazel, "Bazel binary", executable=True)
    _regular(inputs.cuda_root / "bin/ptxas", "ptxas binary", executable=True)
    _regular(inputs.cuda_root / "bin/nvcc", "nvcc binary", executable=True)
    _regular(inputs.image_archive, "image archive")
    _regular(inputs.jax_repository / "build/build.py", "JAX build entrypoint", executable=True)
    observed_tools = BuildToolDigests(
        bazel=inputs.bazel_sha256,
        image=inputs.image_sha256,
        nvcc=inputs.nvcc_sha256,
        ptxas=inputs.ptxas_sha256,
    )
    if observed_tools != expected_tool_sha256:
        raise RuntimeError("offline tool or image bytes differ from launch authority")
    live_tools = BuildToolDigests(
        bazel=_file_sha256(inputs.bazel),
        image=_file_sha256(inputs.image_archive),
        nvcc=_file_sha256(inputs.cuda_root / "bin/nvcc"),
        ptxas=_file_sha256(inputs.cuda_root / "bin/ptxas"),
    )
    if live_tools != observed_tools:
        raise RuntimeError("offline tool or image changed after inspection")
    if patch_sha256(inputs.marin_repository) != inputs.jax_patch_sha256:
        raise RuntimeError("offline JAX patches changed after inspection")
    if xla_patch_sha256(inputs.marin_repository) != inputs.xla_patch_sha256:
        raise RuntimeError("offline XLA patches changed after inspection")


def offline_wheel_command(
    *,
    jax_repository: Path,
    marin_repository: Path,
    bazel: Path,
    bazel_output_root: Path,
    wheel_output: Path,
    xla_repository: Path,
) -> list[str]:
    """Return the closed, download-disabled patched wheel build argv."""
    return [
        str(jax_repository / "build/build.py"),
        "build",
        "--wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt",
        f"--bazel_path={bazel}",
        f"--output_path={wheel_output}",
        "--python_version=3.12",
        f"--cuda_version={HERMETIC_CUDA_VERSION}",
        f"--cuda_compute_capabilities={','.join(HERMETIC_CUDA_COMPUTE_CAPABILITIES)}",
        "--disable_nccl",
        "--bazel_options=--repo_env=ML_WHEEL_TYPE=release",
        "--bazel_options=--define=SHUTTLE_TEST_OBSERVER=1",
        "--bazel_options=--repository_disable_download",
        f"--bazel_startup_options=--output_user_root={bazel_output_root}",
        f"--bazel_options=--override_repository=xla={xla_repository}",
        f"--bazel_options=--override_repository=shuttle_mlir={marin_repository / 'lib/shuttle/mlir'}",
    ]


def patch_sha256(marin_repository: Path) -> tuple[str, ...]:
    patch_root = marin_repository / "lib/shuttle/mlir/jax_patch"
    return tuple(hashlib.sha256((patch_root / name).read_bytes()).hexdigest() for name in PATCH_NAMES)


def xla_patch_sha256(marin_repository: Path) -> tuple[str, ...]:
    patch_root = marin_repository / "lib/shuttle/mlir/xla_patch"
    return tuple(hashlib.sha256((patch_root / name).read_bytes()).hexdigest() for name in XLA_PATCH_NAMES)


def apply_source_patches(inputs: BuildInputs) -> None:
    """Check both exact patch sets before modifying either clean checkout."""
    _require_live_source_snapshot(inputs)
    if patch_sha256(inputs.marin_repository) != JAX_PATCH_SHA256:
        raise RuntimeError("offline JAX patches changed before application")
    if xla_patch_sha256(inputs.marin_repository) != XLA_PATCH_SHA256:
        raise RuntimeError("offline XLA patches changed before application")
    jax_patches = [inputs.marin_repository / "lib/shuttle/mlir/jax_patch" / name for name in PATCH_NAMES]
    xla_patches = [inputs.marin_repository / "lib/shuttle/mlir/xla_patch" / name for name in XLA_PATCH_NAMES]
    commands = (
        (inputs.jax_repository, ["git", "apply", "--check", *map(str, jax_patches)]),
        (inputs.xla_repository, ["git", "apply", "--check", *map(str, xla_patches)]),
    )
    for repository, command in commands:
        subprocess.run(command, cwd=repository, check=True)
    for repository, command in commands:
        subprocess.run([*command[:2], *command[3:]], cwd=repository, check=True)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _untracked_sha256(repository: Path) -> str:
    digest = hashlib.sha256()
    raw_paths = _git_bytes(repository, "ls-files", "--others", "--exclude-standard", "-z")
    for raw_path in sorted(path for path in raw_paths.split(b"\0") if path):
        relative = Path(os.fsdecode(raw_path))
        path = repository / relative
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"hydrated source contains a non-regular untracked file: {relative}")
        digest.update(len(raw_path).to_bytes(8, "little"))
        digest.update(raw_path)
        digest.update(stat.S_IMODE(metadata.st_mode).to_bytes(4, "little"))
        digest.update(metadata.st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_checkouts_identity(*, jax_repository: Path, xla_repository: Path) -> SourceCheckoutsIdentity:
    """Snapshot all live tracked and non-ignored untracked JAX/XLA source state."""
    return SourceCheckoutsIdentity(
        jax_commit=_git(jax_repository, "rev-parse", "HEAD"),
        jax_diff_sha256=hashlib.sha256(
            _git_bytes(jax_repository, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        ).hexdigest(),
        jax_status_sha256=hashlib.sha256(
            _git_bytes(jax_repository, "status", "--porcelain=v1", "--untracked-files=all")
        ).hexdigest(),
        jax_tree=_git(jax_repository, "rev-parse", "HEAD^{tree}"),
        jax_untracked_sha256=_untracked_sha256(jax_repository),
        xla_commit=_git(xla_repository, "rev-parse", "HEAD"),
        xla_diff_sha256=hashlib.sha256(
            _git_bytes(xla_repository, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        ).hexdigest(),
        xla_status_sha256=hashlib.sha256(
            _git_bytes(xla_repository, "status", "--porcelain=v1", "--untracked-files=all")
        ).hexdigest(),
        xla_tree=_git(xla_repository, "rev-parse", "HEAD^{tree}"),
        xla_untracked_sha256=_untracked_sha256(xla_repository),
    )


def require_source_checkouts_unchanged(
    expected: SourceCheckoutsIdentity,
    *,
    jax_repository: Path,
    xla_repository: Path,
) -> None:
    """Reject source drift between the approved patch application and offline build."""
    observed = source_checkouts_identity(jax_repository=jax_repository, xla_repository=xla_repository)
    if observed != expected:
        raise RuntimeError("network hydration changed patched JAX or XLA source")


def _require_live_source_snapshot(inputs: BuildInputs) -> None:
    live = (
        _git(inputs.jax_repository, "rev-parse", "HEAD"),
        _git(inputs.jax_repository, "rev-parse", "HEAD^{tree}"),
        _git(inputs.jax_repository, "status", "--porcelain"),
        _git(inputs.xla_repository, "rev-parse", "HEAD"),
        _git(inputs.xla_repository, "status", "--porcelain"),
    )
    expected = (
        inputs.jax_commit,
        inputs.jax_tree,
        inputs.jax_status,
        inputs.xla_commit,
        inputs.xla_status,
    )
    if live != expected:
        raise RuntimeError("offline source changed after inspection")


def inspect_build_inputs(
    *,
    marin_repository: Path,
    jax_repository: Path,
    bazel: Path,
    cuda_root: Path,
    image_archive: Path,
    xla_repository: Path,
) -> BuildInputs:
    """Inspect local inputs without downloading, building, or starting an image."""
    if not (jax_repository / "build/build.py").is_file():
        jax_commit = "missing"
        jax_tree = "missing"
        jax_status = "missing"
    else:
        jax_commit = _git(jax_repository, "rev-parse", "HEAD")
        jax_tree = _git(jax_repository, "rev-parse", "HEAD^{tree}")
        jax_status = _git(jax_repository, "status", "--porcelain")
    if not xla_repository.is_dir():
        xla_commit = "missing"
        xla_status = "missing"
    else:
        xla_commit = _git(xla_repository, "rev-parse", "HEAD")
        xla_status = _git(xla_repository, "status", "--porcelain")
    ptxas = cuda_root / "bin/ptxas"
    nvcc = cuda_root / "bin/nvcc"
    return BuildInputs(
        bazel=bazel,
        bazel_sha256=_file_sha256(bazel),
        cuda_root=cuda_root,
        image_archive=image_archive,
        image_sha256=_file_sha256(image_archive),
        jax_commit=jax_commit,
        jax_patch_sha256=patch_sha256(marin_repository),
        jax_repository=jax_repository,
        jax_status=jax_status,
        jax_tree=jax_tree,
        marin_repository=marin_repository,
        nvcc_sha256=_file_sha256(nvcc),
        platform=platform.system().lower(),
        ptxas_sha256=_file_sha256(ptxas),
        xla_commit=xla_commit,
        xla_patch_sha256=xla_patch_sha256(marin_repository),
        xla_repository=xla_repository,
        xla_status=xla_status,
    )


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
