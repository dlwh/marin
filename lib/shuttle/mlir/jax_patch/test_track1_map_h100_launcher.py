# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins

import pytest
from track1_map_h100_launcher import main


def test_launcher_fails_before_importing_subject_or_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    imported = []
    original_import = builtins.__import__

    def record_import(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        main(["--require-launch-ready"])

    assert "track1_map_h100_subject" not in imported
    assert "jax" not in imported
