# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

import pytest
import track1_map_h100_prototype as prototype
from track1_tile_lifetime_source import (
    DIST_INFO_DIRECTORY,
    ENTRY_POINT_GROUP,
    ENTRY_POINT_NAME,
    ENTRY_POINT_VALUE,
    SOURCE_PATH_FILE,
    install_tile_lifetime_source,
    require_tile_lifetime_source,
)


def _repository(path: Path) -> Path:
    repository = path / "tile-lifetime"
    package = repository / "src/tile_lifetime/backends/h100"
    package.mkdir(parents=True)
    (repository / "src/tile_lifetime/__init__.py").write_text("")
    (repository / "src/tile_lifetime/backends/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "provider.py").write_text("def compile_normalized_exp_contract_artifact(serialized):\n    return None\n")
    (repository / "pyproject.toml").write_text(
        "[project]\n"
        'name = "marin-tile-lifetime"\n'
        'version = "0.1.0"\n'
        f'[project.entry-points."{ENTRY_POINT_GROUP}"]\n'
        f'{ENTRY_POINT_NAME} = "{ENTRY_POINT_VALUE}"\n'
    )
    return repository


def test_install_exposes_source_and_standard_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repository(tmp_path)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    source_path, dist_info = install_tile_lifetime_source(repository, site_packages)

    assert source_path == site_packages / SOURCE_PATH_FILE
    assert source_path.read_text() == f"{repository / 'src'}\n"
    assert dist_info == site_packages / DIST_INFO_DIRECTORY
    assert (dist_info / "RECORD").is_file()
    monkeypatch.syspath_prepend(str(repository / "src"))
    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    distribution = importlib.metadata.distribution("marin-tile-lifetime")
    matches = [entry_point for entry_point in distribution.entry_points if entry_point.group == ENTRY_POINT_GROUP]
    assert [(entry_point.name, entry_point.value) for entry_point in matches] == [(ENTRY_POINT_NAME, ENTRY_POINT_VALUE)]
    assert callable(matches[0].load())
    sys.modules.pop("tile_lifetime.backends.h100.provider", None)
    sys.modules.pop("tile_lifetime.backends.h100", None)
    sys.modules.pop("tile_lifetime.backends", None)
    sys.modules.pop("tile_lifetime", None)


def test_install_rejects_changed_provider_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    pyproject = repository / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace(ENTRY_POINT_VALUE, "example:provider"))
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    with pytest.raises(RuntimeError, match="identity or physical-provider entry point differs"):
        install_tile_lifetime_source(repository, site_packages)


def test_sealed_provider_source_closure_imports_the_real_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marin_repository = Path(__file__).parents[4]
    repository = tmp_path / "tile-lifetime"
    for relative in prototype.SOURCE_FILES:
        if not relative.is_relative_to("lib/tile_lifetime"):
            continue
        destination = repository / relative.relative_to("lib/tile_lifetime")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marin_repository / relative, destination)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    install_tile_lifetime_source(repository, site_packages)
    monkeypatch.syspath_prepend(str(repository / "src"))
    monkeypatch.syspath_prepend(str(site_packages))
    for name in tuple(sys.modules):
        if name == "tile_lifetime" or name.startswith("tile_lifetime."):
            sys.modules.pop(name)
    importlib.invalidate_caches()

    observed = require_tile_lifetime_source(repository)

    assert observed["entry_point"] == ENTRY_POINT_VALUE
    fast_divmod = importlib.import_module("tile_lifetime.cute_fast_divmod")
    assert fast_divmod.encode_cute_fast_divmod_divisor(3).hex() == "030000005655555501010000"
