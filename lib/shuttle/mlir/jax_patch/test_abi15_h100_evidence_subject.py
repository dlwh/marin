# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
from abi15_h100_evidence_subject import _mapping_record, _one_mapped, run_orchestrator, worker_command

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
    assert Path(command[1]).name == "abi15_h100_evidence_launcher.py"
    assert "abi15_h100_evidence_subject.py" not in command[1]
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


def test_mapping_identity_binds_existing_inode_device_path_and_bytes(tmp_path: Path) -> None:
    runtime = tmp_path / "_jax.so"
    runtime.write_bytes(b"runtime")
    runtime_stat = runtime.stat()
    device = f"{os.major(runtime_stat.st_dev):02x}:{os.minor(runtime_stat.st_dev):02x}"

    record = _mapping_record("jax_extension", runtime, ((0x1000, 0x2000, device, runtime_stat.st_ino),))

    assert record[1:4] == (str(runtime), device, runtime_stat.st_ino)
    with pytest.raises(RuntimeError):
        _mapping_record("jax_extension", tmp_path / "missing/_jax.so", ((0x1000, 0x2000, device, 1),))
    with pytest.raises(RuntimeError):
        _mapping_record("jax_extension", runtime, ((0x1000, 0x2000, device, runtime_stat.st_ino + 1),))


def test_mapped_runtime_role_rejects_same_basename_alias(tmp_path: Path) -> None:
    first = tmp_path / "first/_jax.so"
    second = tmp_path / "second/_jax.so"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(RuntimeError, match="exactly one"):
        _one_mapped({first: (), second: ()}, "_jax.so")
