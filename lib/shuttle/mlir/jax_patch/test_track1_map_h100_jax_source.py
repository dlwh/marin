# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest
from track1_map_h100_jax_source import EXPECTED_JAX_VERSION, install_jax_source


def _fake_jax(repository: Path, version: str = "0.10.1") -> None:
    package = repository / "jax"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "import compiled_dependency\nimport runtime_dependency\nfrom .version import _version\n__version__ = _version\n"
    )
    package.joinpath("version.py").write_text(f"_version = {version!r}\n")


def test_source_install_is_offline_and_imports_the_exact_pinned_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "jax-source"
    site_packages = tmp_path / "site-packages"
    dependency_site_packages = tmp_path / "base-runtime-site-packages"
    site_packages.mkdir()
    dependency_site_packages.mkdir()
    site_packages.joinpath("compiled_dependency.py").write_text("SOURCE = 'child'\n")
    dependency_site_packages.joinpath("compiled_dependency.py").write_text("SOURCE = 'base'\n")
    dependency_site_packages.joinpath("runtime_dependency.py").write_text("PINNED = True\n")
    _fake_jax(repository)

    installed = install_jax_source(repository, site_packages, dependency_site_packages)

    assert [path.read_text() for path in installed] == [
        f"{repository.resolve()}\n",
        f"{dependency_site_packages.resolve()}\n",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import pathlib, site; "
                f"site.addsitedir({str(site_packages)!r}); "
                "import jax; "
                f"assert jax.version._version == {EXPECTED_JAX_VERSION!r}; "
                "assert jax.compiled_dependency.SOURCE == 'child'; "
                f"assert pathlib.Path(jax.__file__).resolve().is_relative_to({str(repository.resolve())!r})"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation", ["missing", "symlink", "version"])
def test_source_install_rejects_noncanonical_package(tmp_path: Path, mutation: str) -> None:
    repository = tmp_path / "jax-source"
    site_packages = tmp_path / "site-packages"
    dependency_site_packages = tmp_path / "base-runtime-site-packages"
    site_packages.mkdir()
    dependency_site_packages.mkdir()
    _fake_jax(repository, version="0.10.0" if mutation == "version" else "0.10.1")
    if mutation == "missing":
        repository.joinpath("jax/__init__.py").unlink()
    elif mutation == "symlink":
        source = repository / "jax/version.py"
        target = repository / "version-target.py"
        source.rename(target)
        source.symlink_to(target)

    with pytest.raises(RuntimeError, match="JAX source"):
        install_jax_source(repository, site_packages, dependency_site_packages)

    assert not any(site_packages.iterdir())


@pytest.mark.parametrize("mutation", ["missing", "symlink"])
def test_source_install_rejects_unbound_dependency_directory(tmp_path: Path, mutation: str) -> None:
    repository = tmp_path / "jax-source"
    site_packages = tmp_path / "site-packages"
    dependency_site_packages = tmp_path / "base-runtime-site-packages"
    site_packages.mkdir()
    _fake_jax(repository)
    if mutation == "symlink":
        target = tmp_path / "dependency-target"
        target.mkdir()
        dependency_site_packages.symlink_to(target)

    with pytest.raises(RuntimeError, match="dependency site-packages"):
        install_jax_source(repository, site_packages, dependency_site_packages)

    assert not any(site_packages.iterdir())
