# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import unittest

import ml_dtypes
import numpy as np
from abi22_layernorm_forward_numerical_contract import (
    CASES,
    SHAPES,
    ErrorMetrics,
    array_digest,
    binary64_reference,
    error_metrics,
    fixed_inputs,
    load_contract,
    require_accepted,
    scheduled_reference,
    validate_contract,
)

ABI11_ARRAY_DIGESTS = {
    "47c7e1bda25abe1f0a2e0f2c34154f0484d367d1309ae2ddb08450ed9eb4b5df",
    "a69953eeff9555b4a0cf42fbe2c978a4c3559eb76329b437f6bd32cdaa284e7a",
    "74bf1cbac5e6448acebd23da739e2f12a1a9fd4ee2d3a98364a04abc60ddf2fa",
    "000ad6855ebae83591386e08d580bbb9d9e8b37c60f34fe4d3f809b5d432afb8",
    "5702474f3719c6c8fc4da160d9021ee2bb07df6e5d58138366d2dcdcceb8fa2a",
    "7e63bd3ff9da45ea73fe965f0ca25c5b868a892ec2361aa303292d2631b73434",
}


class LayerNormForwardNumericalContractTest(unittest.TestCase):
    def test_models_and_digests_are_frozen_before_hardware(self) -> None:
        contract = load_contract()
        for cell in contract["cells"]:
            rows, features = cell["shape"]
            inputs = fixed_inputs(rows, features, cell["case"])
            self.assertEqual(
                dict(zip(("x", "gamma", "beta"), map(array_digest, inputs), strict=True)),
                cell["input_digests"],
            )
            self.assertFalse(set(cell["input_digests"].values()) & ABI11_ARRAY_DIGESTS)
            analytic = binary64_reference(*inputs)
            scheduled = scheduled_reference(*inputs)
            self.assertEqual(array_digest(analytic), cell["binary64_output_digest"])
            self.assertEqual(array_digest(scheduled), cell["schedule_output_digest"])
            metrics = error_metrics(scheduled, analytic)
            self.assertEqual(metrics.__dict__, cell["schedule_metrics"])
            require_accepted(metrics, cell["case"])

    def test_rejects_nonfinite_and_gross_near_zero_errors(self) -> None:
        reference = np.array([-2.6077032089233398e-08], dtype=ml_dtypes.bfloat16)
        gross = np.array([math.ldexp(1.0, -13)], dtype=ml_dtypes.bfloat16)
        metrics = error_metrics(gross, reference)
        self.assertGreater(metrics.mean_absolute_error, 1e-6)
        self.assertGreater(metrics.relative_linf_error, 2**-7)
        self.assertGreater(metrics.max_bfloat16_ulp_error, 16)
        with self.assertRaises(AssertionError):
            require_accepted(metrics, "positive_offset")
        for value in (math.inf, math.nan):
            actual = np.array([value], dtype=ml_dtypes.bfloat16)
            with self.assertRaisesRegex(ValueError, "finite"):
                error_metrics(actual, reference)

    def test_rejects_missing_mean_variance_and_affine_indexing(self) -> None:
        x, gamma, beta = fixed_inputs(7, 13, "balanced")
        x64 = x.astype(np.float64)
        gamma64 = gamma.astype(np.float64)
        beta64 = beta.astype(np.float64)
        mean = np.sum(x64, axis=1, dtype=np.float64) / x.shape[1]
        centered = x64 - mean[:, None]
        variance = np.sum(centered * centered, axis=1, dtype=np.float64) / x.shape[1]
        inverse = 1.0 / np.sqrt(variance[:, None] + 1.0e-5)
        wrong_outputs = (
            (x64 * inverse * gamma64[None, :] + beta64[None, :]).astype(ml_dtypes.bfloat16),
            (centered * gamma64[None, :] + beta64[None, :]).astype(ml_dtypes.bfloat16),
            (centered * inverse * np.roll(gamma64, 1)[None, :] + np.roll(beta64, 1)[None, :]).astype(ml_dtypes.bfloat16),
        )
        reference = binary64_reference(x, gamma, beta)
        for wrong in wrong_outputs:
            with self.assertRaises(AssertionError):
                require_accepted(error_metrics(wrong, reference), "balanced")

    def test_rejects_serial_fold_schedule_digest(self) -> None:
        contract = load_contract()
        cell = next(cell for cell in contract["cells"] if cell["case"] == "balanced" and cell["shape"] == [2048, 4096])
        x, gamma, beta = fixed_inputs(2048, 4096, "balanced")
        x32 = x.astype(np.float32)
        gamma32 = gamma.astype(np.float32)
        beta32 = beta.astype(np.float32)
        mean = np.add.accumulate(x32, axis=1, dtype=np.float32)[:, -1] / np.float32(x.shape[1])
        centered = x32 - mean[:, None]
        variance = np.add.accumulate(centered * centered, axis=1, dtype=np.float32)[:, -1] / np.float32(x.shape[1])
        serial = (
            centered * (np.float32(1.0) / np.sqrt(variance[:, None] + np.float32(1.0e-5))) * gamma32[None, :]
            + beta32[None, :]
        ).astype(ml_dtypes.bfloat16)
        self.assertNotEqual(array_digest(serial), cell["schedule_output_digest"])

    def test_rejects_limit_and_topology_mutations(self) -> None:
        contract = load_contract()
        changed = {**contract, "schema": "drift"}
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_contract(changed)
        changed = {**contract, "cells": [dict(cell) for cell in contract["cells"]]}
        changed["cells"][0]["limits"] = {
            **changed["cells"][0]["limits"],
            "max_absolute_error": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "policy"):
            validate_contract(changed)
        changed = copy.deepcopy(contract)
        changed["model"]["epilogue"] = "drift"
        with self.assertRaisesRegex(ValueError, "model"):
            validate_contract(changed)
        changed = copy.deepcopy(contract)
        changed["qualification_only"]["purpose"] = "acceptance"
        with self.assertRaisesRegex(ValueError, "qualification"):
            validate_contract(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["binary64_output_digest"] = "0" * 63
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_contract(changed)

    def test_rejects_invalid_metric_fields(self) -> None:
        invalid_metrics = (
            ErrorMetrics(math.nan, 0.0, 0.0, 0),
            ErrorMetrics(0.0, math.inf, 0.0, 0),
            ErrorMetrics(0.0, 0.0, -1.0, 0),
            ErrorMetrics(0.0, 0.0, 0.0, -1),
        )
        for metrics in invalid_metrics:
            with self.assertRaises(ValueError):
                require_accepted(metrics, "balanced")

    def test_declares_all_cases_shapes_and_policies(self) -> None:
        contract = load_contract()
        coordinates = {(cell["case"], tuple(cell["shape"])) for cell in contract["cells"]}
        self.assertEqual(coordinates, {(case, shape) for case in CASES for shape in SHAPES})
        self.assertTrue(all(cell["policies"] == ["source_ordered", "fast"] for cell in contract["cells"]))
        self.assertTrue(all(cell["repeatability_runs"] == 2 for cell in contract["cells"]))


if __name__ == "__main__":
    unittest.main()
