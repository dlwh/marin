# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install and verify the sealed Tile Lifetime physical-provider source."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import stat
import sysconfig
import tomllib
from base64 import urlsafe_b64encode
from pathlib import Path

EXPECTED_DISTRIBUTION = "marin-tile-lifetime"
EXPECTED_VERSION = "0.1.0"
ENTRY_POINT_GROUP = "shuttle.gpu_physical_artifact_compilers"
ENTRY_POINT_NAME = "sm90_normalized_exp_contract"
ENTRY_POINT_VALUE = "tile_lifetime.backends.h100.provider:compile_normalized_exp_contract_artifact"
SOURCE_PATH_FILE = "20-track1-tile-lifetime.pth"
DIST_INFO_DIRECTORY = "marin_tile_lifetime-0.1.0.dist-info"


def _regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"Tile Lifetime source member is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"Tile Lifetime source member must be one regular file: {path}")


def _require_repository(repository: Path) -> Path:
    if not repository.is_absolute() or repository.is_symlink() or not repository.is_dir():
        raise RuntimeError("Tile Lifetime repository must be one absolute directory, not a symlink")
    pyproject = repository / "pyproject.toml"
    package = repository / "src/tile_lifetime"
    _regular_file(pyproject)
    _regular_file(package / "__init__.py")
    document = tomllib.loads(pyproject.read_text())
    project = document.get("project")
    if not isinstance(project, dict):
        raise RuntimeError("Tile Lifetime pyproject must contain project metadata")
    entry_points = project.get("entry-points")
    expected_entry_points = {ENTRY_POINT_GROUP: {ENTRY_POINT_NAME: ENTRY_POINT_VALUE}}
    if (
        project.get("name") != EXPECTED_DISTRIBUTION
        or project.get("version") != EXPECTED_VERSION
        or entry_points != expected_entry_points
    ):
        raise RuntimeError("Tile Lifetime package identity or physical-provider entry point differs")
    return repository.resolve()


def _record_hash(value: bytes) -> str:
    digest = urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def install_tile_lifetime_source(repository: Path, site_packages: Path) -> tuple[Path, Path]:
    """Expose exact provider source and its standard distribution entry point."""
    source = _require_repository(repository)
    if site_packages.is_symlink() or not site_packages.is_dir():
        raise RuntimeError("Tile Lifetime install target must be one site-packages directory")
    source_path = site_packages / SOURCE_PATH_FILE
    dist_info = site_packages / DIST_INFO_DIRECTORY
    if source_path.exists() or source_path.is_symlink() or dist_info.exists() or dist_info.is_symlink():
        raise RuntimeError("Tile Lifetime source distribution already exists")

    source_path_value = f"{source / 'src'}{os.linesep}".encode()
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {EXPECTED_DISTRIBUTION}\n"
        f"Version: {EXPECTED_VERSION}\n"
        "Requires-Python: >=3.12\n"
    ).encode()
    entry_points = (f"[{ENTRY_POINT_GROUP}]\n" f"{ENTRY_POINT_NAME} = {ENTRY_POINT_VALUE}\n").encode()
    installer = b"track1-sealed-source\n"
    members = {
        source_path.relative_to(site_packages).as_posix(): source_path_value,
        f"{DIST_INFO_DIRECTORY}/METADATA": metadata,
        f"{DIST_INFO_DIRECTORY}/entry_points.txt": entry_points,
        f"{DIST_INFO_DIRECTORY}/INSTALLER": installer,
    }
    created: list[Path] = []
    try:
        with source_path.open("xb") as stream:
            stream.write(source_path_value)
        created.append(source_path)
        dist_info.mkdir()
        created.append(dist_info)
        for relative, value in members.items():
            if relative == SOURCE_PATH_FILE:
                continue
            path = site_packages / relative
            with path.open("xb") as stream:
                stream.write(value)
        record_path = dist_info / "RECORD"
        rows = [(relative, _record_hash(value), str(len(value))) for relative, value in sorted(members.items())]
        rows.append((record_path.relative_to(site_packages).as_posix(), "", ""))
        with record_path.open("x", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
    except BaseException:
        for path in reversed(created):
            if path.is_dir():
                for member in path.iterdir():
                    member.unlink()
                path.rmdir()
            else:
                path.unlink()
        raise
    return source_path, dist_info


def require_tile_lifetime_source(repository: Path) -> dict[str, str]:
    """Require the exact source origin and installed generic provider entry point."""
    source = _require_repository(repository)
    package = importlib.import_module("tile_lifetime")
    origin = Path(package.__file__).resolve()
    expected_origin = source / "src/tile_lifetime/__init__.py"
    if origin != expected_origin:
        raise RuntimeError(f"Tile Lifetime runtime origin differs: expected {expected_origin}, observed {origin}")
    distribution = importlib.metadata.distribution(EXPECTED_DISTRIBUTION)
    if distribution.version != EXPECTED_VERSION:
        raise RuntimeError("Tile Lifetime installed distribution version differs")
    matches = [
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == ENTRY_POINT_GROUP and entry_point.name == ENTRY_POINT_NAME
    ]
    if len(matches) != 1 or matches[0].value != ENTRY_POINT_VALUE or not callable(matches[0].load()):
        raise RuntimeError("Tile Lifetime generic physical-provider entry point differs")
    return {
        "distribution": EXPECTED_DISTRIBUTION,
        "entry_point": ENTRY_POINT_VALUE,
        "origin": str(origin),
        "version": distribution.version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "verify"))
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.action == "install":
        install_tile_lifetime_source(arguments.repository, Path(sysconfig.get_path("purelib")))
    else:
        print(json.dumps(require_tile_lifetime_source(arguments.repository), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
