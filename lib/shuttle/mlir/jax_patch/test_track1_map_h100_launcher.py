# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_track1_map_h100_prototype import _document
from track1_map_h100_prototype import BASE_IMAGE, canonical_json

JAX_PATCH = Path(__file__).parent


def test_launcher_fails_before_importing_subject_or_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    imported = []
    original_import = builtins.__import__

    def record_import(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        return original_import(name, *args, **kwargs)

    sys.modules.pop("track1_map_h100_launcher", None)
    sys.modules.pop("track1_map_h100_prototype", None)
    monkeypatch.setattr(builtins, "__import__", record_import)
    launcher = importlib.import_module("track1_map_h100_launcher")
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        launcher.main(["--require-launch-ready"])

    assert "track1_map_h100_subject" not in imported
    assert all(name != "jax" and not name.startswith("jax.") for name in imported)


def test_launcher_reaches_fail_closed_boundary_with_only_stdlib_imports() -> None:
    environment = {"PYTHONPATH": str(JAX_PATCH)}
    result = subprocess.run(
        [sys.executable, "-S", str(JAX_PATCH / "track1_map_h100_launcher.py"), "--require-launch-ready"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "explicitly unsealed" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_launcher_dry_run_emits_canonical_plan_without_importing_subject_or_jax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    authorization = tmp_path / "authorization.json"
    raw = canonical_json(_document())
    authorization.write_bytes(raw)
    imported = []
    original_import = builtins.__import__

    def record_import(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        return original_import(name, *args, **kwargs)

    sys.modules.pop("track1_map_h100_prototype", None)
    sys.modules.pop("track1_map_h100_launcher", None)
    monkeypatch.setattr(builtins, "__import__", record_import)
    launcher = importlib.import_module("track1_map_h100_launcher")
    assert (
        launcher.main(
            [
                "--authorization-file",
                str(authorization),
                "--approved-authorization-sha256",
                hashlib.sha256(raw).hexdigest(),
                "--workspace",
                str(tmp_path / "workspace"),
                "--runner-repository",
                "/app",
                "--python",
                "/usr/bin/python3.12",
                "--cuda-root",
                "/usr/local/cuda-13.2",
                "--base-image",
                BASE_IMAGE,
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.encode()
    assert output == canonical_json(json.loads(output))
    assert "track1_map_h100_subject" not in imported
    assert all(name != "jax" and not name.startswith("jax.") for name in imported)
