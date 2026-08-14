# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from abi15_h100_evidence_subject import run_orchestrator, worker_command

SHA = "a" * 64
COMMIT = "b" * 40
TREE = "c" * 40


def test_worker_command_binds_fresh_process_role_paths_and_authorization(
    tmp_path: Path,
) -> None:
    command = worker_command(
        "populate",
        tmp_path / "output",
        tmp_path / "cache",
        tmp_path / "report.json",
        SHA,
    )

    assert command[0] == sys.executable
    assert command[command.index("--authorization-sha256") + 1] == SHA
    assert command[command.index("--worker") + 1] == "populate"
    assert command[command.index("--cache-directory") + 1] == str(tmp_path / "cache")
    assert command[command.index("--report") + 1] == str(tmp_path / "report.json")


def test_subject_stops_before_build_copy_or_hardware_while_authority_is_unsealed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    build = tmp_path / "build"
    output.mkdir()
    build.mkdir()
    marker = build / "would-copy"
    marker.write_text("must not copy\n")

    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        run_orchestrator(
            output,
            build,
            authorization_sha256=SHA,
            harness_commit=COMMIT,
            harness_tree=TREE,
        )

    assert not any(output.iterdir())
    assert not (output / "accepted_bundle.json").exists()
