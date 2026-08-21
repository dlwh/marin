# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ABI22 H100 performance report protocol."""

from __future__ import annotations

import copy

import pytest
from abi22_h100_performance import CELL_KEYS, build_acceptance, validate_report

HEX = "0" * 64
MEASURED_ITERATIONS = 6


def _resources(case: str) -> dict[str, int]:
    return {
        "rmsnorm_forward": {
            "code_bytes": 1024,
            "launch_count": 1,
            "slot_count": 3,
            "external_operand_count": 2,
            "external_result_count": 1,
            "temporary_allocation_count": 0,
            "temporary_bytes": 0,
        },
        "rmsnorm_backward": {
            "code_bytes": 4096,
            "launch_count": 3,
            "slot_count": 6,
            "external_operand_count": 3,
            "external_result_count": 2,
            "temporary_allocation_count": 1,
            "temporary_bytes": 56,
        },
        "rmsnorm_contract_inline": {
            "code_bytes": 2048,
            "launch_count": 1,
            "slot_count": 4,
            "external_operand_count": 3,
            "external_result_count": 1,
            "temporary_allocation_count": 0,
            "temporary_bytes": 0,
        },
        "rmsnorm_contract_compact": {
            "code_bytes": 3072,
            "launch_count": 2,
            "slot_count": 5,
            "external_operand_count": 3,
            "external_result_count": 1,
            "temporary_allocation_count": 1,
            "temporary_bytes": 20,
        },
    }[case]


def _report(producer: str, base_ns: int) -> dict[str, object]:
    cells = []
    for cell_index, (case, policy) in enumerate(CELL_KEYS):
        result_count = 2 if case == "rmsnorm_backward" else 1
        cells.append(
            {
                "case": case,
                "policy": policy,
                "shape": {
                    "rmsnorm_forward": [2, 7, 19],
                    "rmsnorm_backward": [2, 7, 19],
                    "rmsnorm_contract_inline": [5, 19, 7],
                    "rmsnorm_contract_compact": [5, 19, 257],
                }[case],
                "samples_ns": [base_ns + cell_index * 10 + round_index for round_index in range(MEASURED_ITERATIONS)],
                "sample_rounds": list(range(MEASURED_ITERATIONS)),
                "sample_positions": [
                    cell_index if round_index % 2 == 0 else len(CELL_KEYS) - 1 - cell_index
                    for round_index in range(MEASURED_ITERATIONS)
                ],
                "resources": _resources(case),
                "output": {
                    "digests": [HEX for _ in range(result_count)],
                    "max_abs_error": 0.0,
                    "max_bf16_ulp": 1 if policy == "source_ordered" else 4,
                },
                "identity": {
                    "fixture_digest": HEX,
                    "executable_digest": HEX,
                    "ptx_digests": [HEX] * _resources(case)["launch_count"],
                    "module_load_digest": HEX,
                },
            }
        )
    return {
        "schema": "shuttle_h100_performance_v1",
        "producer": producer,
        "device": {"model": "NVIDIA H100 80GB HBM3", "ordinal": 0},
        "compiler": {"identity": f"{producer}-compiler", "options_digest": HEX},
        "protocol": {
            "clock": "cuda_event",
            "counterbalance": "forward_reverse_alternating",
            "warmup_iterations": 3,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "cells": cells,
    }


def test_complete_shuttle_report_preserves_raw_counterbalanced_samples() -> None:
    report = validate_report(_report("shuttle", 100), expected_producer="shuttle")

    assert [(cell["case"], cell["policy"]) for cell in report["cells"]] == list(CELL_KEYS)
    assert report["cells"][0]["samples_ns"] == [100, 101, 102, 103, 104, 105]
    assert report["protocol"]["warmup_iterations"] == 3


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("missing-cell", "complete case/policy matrix"),
        ("short-samples", "measured_iterations"),
        ("wrong-position", "counterbalanced position"),
        ("nonpositive-sample", "positive integer nanoseconds"),
        ("missing-ptx", "PTX digest"),
        ("wrong-launch-count", "resource counters"),
        ("loose-source-ordered-error", "BF16 ULP"),
    ],
)
def test_shuttle_report_rejects_partial_or_untrustworthy_measurements(mutation: str, diagnostic: str) -> None:
    report = _report("shuttle", 100)
    cells = report["cells"]
    assert isinstance(cells, list)
    if mutation == "missing-cell":
        cells.pop()
    elif mutation == "short-samples":
        cells[0]["samples_ns"].pop()
    elif mutation == "wrong-position":
        cells[0]["sample_positions"][1] = 0
    elif mutation == "nonpositive-sample":
        cells[0]["samples_ns"][0] = 0
    elif mutation == "missing-ptx":
        cells[0]["identity"]["ptx_digests"] = []
    elif mutation == "wrong-launch-count":
        cells[0]["resources"]["launch_count"] = 2
    elif mutation == "loose-source-ordered-error":
        cells[0]["output"]["max_bf16_ulp"] = 2

    with pytest.raises(ValueError, match=diagnostic):
        validate_report(report, expected_producer="shuttle")


def test_acceptance_requires_independent_complete_xla_and_te_inputs() -> None:
    shuttle = _report("shuttle", 100)
    xla = _report("xla", 200)
    te = _report("te", 150)

    accepted = build_acceptance(shuttle, xla, te)
    assert accepted["schema"] == "shuttle_h100_performance_acceptance_v1"
    assert len(accepted["comparisons"]) == len(CELL_KEYS)
    assert accepted["comparisons"][0]["shuttle_median_ns"] == 102.5
    assert accepted["comparisons"][0]["xla_median_ns"] == 202.5
    assert accepted["comparisons"][0]["te_median_ns"] == 152.5

    partial_xla = copy.deepcopy(xla)
    partial_xla["cells"].pop()
    with pytest.raises(ValueError, match=r"xla.*complete case/policy matrix"):
        build_acceptance(shuttle, partial_xla, te)
    with pytest.raises(ValueError, match="xla baseline is required"):
        build_acceptance(shuttle, None, te)
    with pytest.raises(ValueError, match="te baseline is required"):
        build_acceptance(shuttle, xla, None)
