# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import sys

import pytest


def test_launcher_fails_before_importing_subject_or_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    imported = []
    original_import = builtins.__import__

    def record_import(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        return original_import(name, *args, **kwargs)

    sys.modules.pop("track1_map_h100_launcher", None)
    monkeypatch.setattr(builtins, "__import__", record_import)
    launcher = importlib.import_module("track1_map_h100_launcher")
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        launcher.main(["--require-launch-ready"])

    assert "track1_map_h100_subject" not in imported
    assert all(name != "jax" and not name.startswith("jax.") for name in imported)
