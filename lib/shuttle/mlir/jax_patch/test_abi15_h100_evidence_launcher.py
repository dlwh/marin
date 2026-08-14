# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from abi15_h100_evidence_launcher import main


def test_stdlib_launcher_stops_before_importing_jax_writing_or_starting_a_backend(tmp_path: Path) -> None:
    before_modules = set(sys.modules)
    sentinel = tmp_path / "must-not-exist"

    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        main()

    assert not sentinel.exists()
    assert set(sys.modules) - before_modules == set()
