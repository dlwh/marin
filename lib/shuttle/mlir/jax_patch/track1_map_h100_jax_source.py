# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install and verify the pinned pure-Python JAX source without a package index."""

import argparse
import ast
import importlib
import json
import os
import stat
import sysconfig
from pathlib import Path

EXPECTED_JAX_VERSION = "0.10.1"
SOURCE_PATH_FILE = "00-track1-pinned-jax.pth"
DEPENDENCY_PATH_FILE = "10-track1-base-runtime.pth"


def _regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"JAX source member is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"JAX source member must be one regular file: {path}")


def _source_version(version_path: Path) -> str:
    _regular_file(version_path)
    tree = ast.parse(version_path.read_text(), filename=str(version_path))
    versions = []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "_version" and isinstance(statement.value, ast.Constant):
            versions.append(statement.value.value)
    if versions != [EXPECTED_JAX_VERSION]:
        raise RuntimeError(f"JAX source base version must be exactly {EXPECTED_JAX_VERSION}")
    return EXPECTED_JAX_VERSION


def _require_repository(repository: Path) -> Path:
    if not repository.is_absolute() or repository.is_symlink() or not repository.is_dir():
        raise RuntimeError("JAX source repository must be one absolute directory, not a symlink")
    package = repository / "jax"
    if package.is_symlink() or not package.is_dir():
        raise RuntimeError("JAX source package must be one directory, not a symlink")
    _regular_file(package / "__init__.py")
    _source_version(package / "version.py")
    return repository.resolve()


def install_jax_source(
    repository: Path,
    site_packages: Path,
    dependency_site_packages: Path,
) -> tuple[Path, Path]:
    """Expose exact JAX source and immutable-image dependencies without resolution."""
    source = _require_repository(repository)
    if site_packages.is_symlink() or not site_packages.is_dir():
        raise RuntimeError("JAX source install target must be one site-packages directory")
    if (
        not dependency_site_packages.is_absolute()
        or dependency_site_packages.is_symlink()
        or not dependency_site_packages.is_dir()
    ):
        raise RuntimeError("JAX source dependency site-packages must be one absolute directory, not a symlink")
    installed = (site_packages / SOURCE_PATH_FILE, site_packages / DEPENDENCY_PATH_FILE)
    if any(path.exists() or path.is_symlink() for path in installed):
        raise RuntimeError("JAX source path file already exists")
    created = []
    try:
        for path, target in zip(installed, (source, dependency_site_packages.resolve()), strict=True):
            with path.open("x", encoding="utf-8") as stream:
                stream.write(f"{target}{os.linesep}")
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return installed


def require_jax_source(repository: Path) -> dict[str, str]:
    """Require the imported JAX package to be the exact pinned source checkout."""
    source = _require_repository(repository)
    jax = importlib.import_module("jax")
    version = importlib.import_module("jax.version")
    origin = Path(jax.__file__).resolve()
    expected_origin = source / "jax/__init__.py"
    if origin != expected_origin:
        raise RuntimeError(
            f"JAX runtime origin differs from pinned source: expected {expected_origin}, observed {origin}"
        )
    observed_version = getattr(version, "_version", None)
    if observed_version != EXPECTED_JAX_VERSION:
        raise RuntimeError(
            f"JAX runtime base version differs: expected {EXPECTED_JAX_VERSION}, observed {observed_version}"
        )
    return {"base_version": observed_version, "origin": str(origin)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "verify"))
    parser.add_argument("repository", type=Path)
    parser.add_argument("--dependency-site-packages", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.action == "install":
        if arguments.dependency_site_packages is None:
            parser.error("install requires --dependency-site-packages")
        install_jax_source(
            arguments.repository,
            Path(sysconfig.get_path("purelib")),
            arguments.dependency_site_packages,
        )
    else:
        if arguments.dependency_site_packages is not None:
            parser.error("verify does not accept --dependency-site-packages")
        print(json.dumps(require_jax_source(arguments.repository), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
