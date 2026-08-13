# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from shuttle_h100_evidence_build import decode_mlir_byte_string, extract_gpu_transport, parse_ptxas_resources


def test_extract_gpu_transport_decodes_exact_mlir_byte_escapes() -> None:
    generated = (
        r"stablehlo.custom_call @shuttle.gpu.executable_bundle.v1 "
        r'{backend_config = {bundle_bytes = "SHUTGPU\00\ff\\\"tail"}}'
    )

    assert extract_gpu_transport(generated) == b'SHUTGPU\x00\xff\\"tail'


@pytest.mark.parametrize(
    "generated",
    [
        "module {}",
        'bundle_bytes = "one" bundle_bytes = "two"',
        r'bundle_bytes = "bad\q"',
        'bundle_bytes = "unterminated',
        'bundle_bytes = "bad\nraw"',
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
