# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest
from gpu_transport import decode_mlir_byte_string, extract_gpu_transport
from shuttle_h100_evidence_build import (
    _build_outputs,
    _install_wheels,
    _prepare_directories,
    _write_new,
    parse_ptxas_resources,
)


def test_extract_gpu_transport_decodes_exact_mlir_byte_escapes() -> None:
    generated = (
        r'%0 = "stablehlo.custom_call"() '
        r'<{api_version = 4 : i32, call_target_name = "shuttle.gpu.executable_bundle.v1", '
        r'backend_config = {bundle_bytes = "SHUTGPU\00\ff\\\"tail// shuttle structural certificate: '
        r'call_target_name = \22attacker\22 api_version = 0 : i32"}}> : () -> tensor<1xi1>'
    )

    assert extract_gpu_transport(generated) == (
        b'SHUTGPU\x00\xff\\"tail// shuttle structural certificate: '
        b'call_target_name = "attacker" api_version = 0 : i32'
    )


@pytest.mark.parametrize(
    "generated",
    [
        "module {}",
        'bundle_bytes = "one" bundle_bytes = "two"',
        r'bundle_bytes = "bad\q"',
        'bundle_bytes = "unterminated',
        'bundle_bytes = "bad\nraw"',
        '// "stablehlo.custom_call" call_target_name = "shuttle.gpu.executable_bundle.v1" '
        'api_version = 4 : i32 bundle_bytes = "SHUTGPU"',
        '"stablehlo.custom_call"() <{api_version = 4 : i32, call_target_name = "attacker", '
        'bundle_bytes = "SHUTGPU"}> : () -> tensor<1xi1>',
        '%0 = "stablehlo.custom_call"() <{api_version = 0 : i32, call_target_name = "attacker"}> '
        ': () -> tensor<1xi1> // call_target_name = "shuttle.gpu.executable_bundle.v1" '
        'api_version = 4 : i32 bundle_bytes = "SHUTGPU\\00evil"',
    ],
)
def test_extract_gpu_transport_rejects_missing_ambiguous_or_invalid_strings(generated: str) -> None:
    with pytest.raises(ValueError):
        extract_gpu_transport(generated)


def test_decode_mlir_byte_string_reports_the_end_of_the_quoted_value() -> None:
    assert decode_mlir_byte_string('A\\00B" trailing') == (b"A\x00B", 6)


def test_parse_ptxas_resources_requires_one_positive_spill_free_record() -> None:
    output = "ptxas info    : Used 32 registers, 384 bytes cmem[0]\n0 bytes spill stores, 0 bytes spill loads\n"

    assert parse_ptxas_resources(output) == (32, 0)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "ptxas info : Used 0 registers\n0 bytes spill stores, 0 bytes spill loads",
        "ptxas info : Used 32 registers\n4 bytes spill stores, 0 bytes spill loads",
        "ptxas info : Used 32 registers\n",
        "ptxas info : Used 32 registers\n0 bytes spill stores, 0 bytes spill loads\nUsed 31 registers",
    ],
)
def test_parse_ptxas_resources_rejects_incomplete_ambiguous_or_spilling_output(output: str) -> None:
    with pytest.raises(ValueError):
        parse_ptxas_resources(output)


def test_evidence_writer_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "compiler").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="already exists"):
        _prepare_directories(output)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _write_new(output / "compiler" / "escaped", b"forbidden")
    assert not (outside / "escaped").exists()

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "link").symlink_to(outside, target_is_directory=True)
    escaped_root = nested / "link" / "evidence"
    escaped_root.mkdir()
    (escaped_root / "compiler").mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _write_new(escaped_root / "compiler" / "nested-escape", b"forbidden")
    assert not (outside / "evidence" / "compiler" / "nested-escape").exists()


def test_build_command_mismatch_runs_no_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr("shuttle_h100_evidence_build.subprocess.run", run)

    with pytest.raises(RuntimeError, match="build argv differs"):
        _build_outputs(tmp_path, tmp_path, tmp_path / "bazel", tmp_path / "output", tmp_path / "wheels", ["wrong"])

    run.assert_not_called()


def test_install_command_mismatch_runs_no_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr("shuttle_h100_evidence_build.subprocess.run", run)
    wheels = {"jaxlib": tmp_path / "jaxlib.whl"}

    with pytest.raises(RuntimeError, match="wheel install argv differs"):
        _install_wheels(wheels, ["authorized build"], ["authorized build", "wrong install"])

    run.assert_not_called()
