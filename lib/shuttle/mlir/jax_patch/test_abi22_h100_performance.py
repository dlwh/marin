# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ABI22 H100 performance report protocol."""

from __future__ import annotations

import copy
import unittest

from abi22_h100_performance import (
    CELL_KEYS,
    build_acceptance,
    build_exploratory,
    validate_exploratory_report,
    validate_report,
)

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
                    "workload_digest": HEX,
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


def _shuttle_host_report(base_ns: int) -> dict[str, object]:
    report = _report("shuttle", base_ns)
    report["host_protocol"] = {
        "mode": "exploratory_host_v1",
        "clock": "host_block_until_ready",
        "counterbalance": "forward_reverse_alternating",
        "warmup_iterations": 3,
        "measured_iterations": MEASURED_ITERATIONS,
    }
    for cell_index, cell in enumerate(report["cells"]):
        cell["host_samples_ns"] = [
            base_ns + 1_000 + cell_index * 10 + round_index for round_index in range(MEASURED_ITERATIONS)
        ]
    return report


def _exploratory_report(producer: str, base_ns: int) -> dict[str, object]:
    accepted = _report(producer, base_ns)
    cells = []
    for cell in accepted["cells"]:
        cells.append(
            {
                "case": cell["case"],
                "policy": cell["policy"],
                "shape": cell["shape"],
                "samples_ns": cell["samples_ns"],
                "sample_rounds": cell["sample_rounds"],
                "sample_positions": cell["sample_positions"],
                "output": cell["output"],
                "identity": {
                    "workload_digest": cell["identity"]["workload_digest"],
                    "executable_digest": cell["identity"]["executable_digest"],
                },
            }
        )
    return {
        "schema": "shuttle_h100_exploratory_host_v1",
        "producer": producer,
        "device": accepted["device"],
        "compiler": accepted["compiler"],
        "protocol": {
            "mode": "exploratory_host_v1",
            "clock": "host_block_until_ready",
            "counterbalance": "forward_reverse_alternating",
            "warmup_iterations": 3,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "cells": cells,
    }


def _te_unavailable() -> dict[str, object]:
    return {
        "schema": "shuttle_h100_exploratory_unavailable_v1",
        "producer": "te",
        "protocol_mode": "exploratory_host_v1",
        "status": "unavailable",
        "blocker": "te_2_17_runner_lacks_norm_contract_cells",
    }


class PerformanceProtocolTest(unittest.TestCase):
    def test_complete_shuttle_report_preserves_raw_counterbalanced_samples(self) -> None:
        report = validate_report(_report("shuttle", 100), expected_producer="shuttle")

        self.assertEqual([(cell["case"], cell["policy"]) for cell in report["cells"]], list(CELL_KEYS))
        self.assertEqual(report["cells"][0]["samples_ns"], [100, 101, 102, 103, 104, 105])
        self.assertEqual(report["protocol"]["warmup_iterations"], 3)

    def test_shuttle_report_rejects_partial_or_untrustworthy_measurements(self) -> None:
        mutations = [
            ("missing-cell", "complete case/policy matrix"),
            ("short-samples", "measured_iterations"),
            ("wrong-position", "counterbalanced position"),
            ("nonpositive-sample", "positive integer nanoseconds"),
            ("missing-ptx", "PTX digest"),
            ("wrong-launch-count", "resource counters"),
            ("loose-source-ordered-error", "BF16 ULP"),
            ("odd-sample-count", "positive and even"),
        ]
        for mutation, diagnostic in mutations:
            with self.subTest(mutation=mutation):
                report = _report("shuttle", 100)
                cells = report["cells"]
                self.assertIsInstance(cells, list)
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
                elif mutation == "odd-sample-count":
                    report["protocol"]["measured_iterations"] = 3
                    for cell in cells:
                        for field in ("samples_ns", "sample_rounds", "sample_positions"):
                            cell[field] = cell[field][:3]

                with self.assertRaisesRegex(ValueError, diagnostic):
                    validate_report(report, expected_producer="shuttle")

    def test_acceptance_requires_independent_complete_xla_and_te_inputs(self) -> None:
        shuttle = _report("shuttle", 100)
        xla = _report("xla", 200)
        te = _report("te", 150)

        accepted = build_acceptance(shuttle, xla, te)
        self.assertEqual(accepted["schema"], "shuttle_h100_performance_acceptance_v1")
        self.assertEqual(len(accepted["comparisons"]), len(CELL_KEYS))
        self.assertEqual(accepted["comparisons"][0]["shuttle_median_ns"], 102.5)
        self.assertEqual(accepted["comparisons"][0]["xla_median_ns"], 202.5)
        self.assertEqual(accepted["comparisons"][0]["te_median_ns"], 152.5)

        partial_xla = copy.deepcopy(xla)
        partial_xla["cells"].pop()
        with self.assertRaisesRegex(ValueError, r"xla.*complete case/policy matrix"):
            build_acceptance(shuttle, partial_xla, te)
        with self.assertRaisesRegex(ValueError, "xla baseline is required"):
            build_acceptance(shuttle, None, te)
        with self.assertRaisesRegex(ValueError, "te baseline is required"):
            build_acceptance(shuttle, xla, None)

        mismatched_xla = copy.deepcopy(xla)
        mismatched_xla["cells"][0]["identity"]["workload_digest"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "xla rmsnorm_forward source_ordered workload does not match"):
            build_acceptance(shuttle, mismatched_xla, te)

        with self.assertRaisesRegex(ValueError, "schema"):
            build_acceptance(shuttle, _exploratory_report("xla", 200), te)

    def test_exploratory_host_comparison_uses_only_matched_host_samples(self) -> None:
        shuttle = _shuttle_host_report(100)
        xla = _exploratory_report("xla", 200)

        explored = build_exploratory(shuttle, xla, _te_unavailable())

        self.assertEqual(explored["schema"], "shuttle_h100_exploratory_comparison_v1")
        self.assertEqual(explored["protocol"]["clock"], "host_block_until_ready")
        self.assertEqual(explored["comparisons"][0]["shuttle_median_ns"], 1102.5)
        self.assertEqual(explored["comparisons"][0]["xla_median_ns"], 202.5)
        self.assertEqual(explored["te"], _te_unavailable())

    def test_exploratory_host_comparison_rejects_partial_or_cross_clock_inputs(self) -> None:
        shuttle = _shuttle_host_report(100)
        xla = _exploratory_report("xla", 200)

        missing_host = copy.deepcopy(shuttle)
        del missing_host["cells"][0]["host_samples_ns"]
        with self.assertRaisesRegex(ValueError, "host_samples_ns"):
            build_exploratory(missing_host, xla, _te_unavailable())

        wrong_clock = copy.deepcopy(xla)
        wrong_clock["protocol"]["clock"] = "cuda_event"
        with self.assertRaisesRegex(ValueError, "host_block_until_ready"):
            validate_exploratory_report(wrong_clock, expected_producer="xla")

        mismatched_workload = copy.deepcopy(xla)
        mismatched_workload["cells"][0]["identity"]["workload_digest"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "workload does not match"):
            build_exploratory(shuttle, mismatched_workload, _te_unavailable())

        incomplete_xla = copy.deepcopy(xla)
        incomplete_xla["cells"].pop()
        with self.assertRaisesRegex(ValueError, "complete case/policy matrix"):
            build_exploratory(shuttle, incomplete_xla, _te_unavailable())

        with self.assertRaisesRegex(ValueError, "TE exploratory blocker"):
            build_exploratory(shuttle, xla, None)


if __name__ == "__main__":
    unittest.main()
