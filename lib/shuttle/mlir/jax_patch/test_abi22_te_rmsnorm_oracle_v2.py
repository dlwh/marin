# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the versioned TE 2.17/XLA RMSNorm oracle lane."""

import copy
import hashlib
import statistics
import unittest
from pathlib import Path

from abi22_te_rmsnorm_oracle_v2 import (
    ARCHITECTURE_SHAPE,
    BOUNDARIES,
    EPSILON,
    NORM_CONTRACT_BLOCKER,
    REPRESENTATIVE_SHAPE,
    build_matched_acceptance,
    execution_plan,
    validate_report,
)

TE_RUNNER = Path(__file__).with_name("abi22_te_rmsnorm_oracle_runner_v2.cpp")


def _cell(shape: tuple[int, int], boundary: str, *, provider: str) -> dict:
    roles = {
        "forward": ("y",),
        "backward_recompute": ("dx", "dgamma"),
        "composed": ("y", "dx", "dgamma"),
    }[boundary]
    host_samples = [100_000 + index * 1_000 for index in range(20)]
    return {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_acceptance",
        "boundary": boundary,
        "input_digest": hashlib.sha256(f"{shape}/{boundary}".encode()).hexdigest(),
        "outputs": [
            {
                "role": role,
                "digest": hashlib.sha256(f"{provider}/{shape}/{boundary}/{role}".encode()).hexdigest(),
                "max_absolute_error": 0.0,
                "max_bfloat16_ulp": 0,
            }
            for role in roles
        ],
        "timing": {
            "warmup_invocations": 10,
            "raw_host_blocking_ns": host_samples,
            "median_host_blocking_ns": statistics.median(host_samples),
            "raw_cuda_event_ns": (
                [50_000 + index * 100 for index in range(20)] if provider == "transformer_engine_2_17" else None
            ),
        },
    }


def _report(provider: str) -> dict:
    return {
        "schema": "shuttle_abi22_te_xla_rmsnorm_oracle_v2",
        "protocol_mode": "matched_host_blocking_v2",
        "producer": provider,
        "epsilon": EPSILON,
        "provider_identity": (
            {
                "distribution": "transformer_engine",
                "version": "2.17.0",
                "source_commit": "2e559f062497bef768dfbe9d7e45548fadeca80a",
                "runner_sha256": "1" * 64,
                "library_sha256": "2" * 64,
            }
            if provider == "transformer_engine_2_17"
            else {
                "jax_version": "0.10.1",
                "jaxlib_version": "0.10.1",
                "xla_revision": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
            }
        ),
        "device": {
            "model": "NVIDIA H100 80GB HBM3",
            "uuid": "GPU-test",
            "compute_capability": "9.0",
        },
        "capabilities": {
            "rmsnorm": ["forward", "backward_recompute", "composed"],
            "composed_semantics": "forward_then_backward_with_invocation_local_rsigma",
            "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
        },
        "cells": [
            _cell(shape, boundary, provider=provider)
            for shape in (ARCHITECTURE_SHAPE, REPRESENTATIVE_SHAPE)
            for boundary in BOUNDARIES
        ],
    }


class TeRmsnormOracleV2Test(unittest.TestCase):
    def test_plan_separates_architecture_from_representative_acceptance(self) -> None:
        plan = execution_plan("./abi22_te_rmsnorm_oracle_runner_v2")

        self.assertEqual(plan["schema"], "shuttle_abi22_te_xla_rmsnorm_plan_v2")
        self.assertEqual(plan["epsilon"], 1.0e-6)
        self.assertEqual(len(plan["runs"]), 12)
        self.assertEqual(
            {(tuple(run["shape"]), run["scope"], run["boundary"], run["producer"]) for run in plan["runs"]},
            {
                (shape, scope, boundary, producer)
                for shape, scope in (
                    (ARCHITECTURE_SHAPE, "architecture_fixture"),
                    (REPRESENTATIVE_SHAPE, "representative_acceptance"),
                )
                for boundary in BOUNDARIES
                for producer in ("transformer_engine_2_17", "ordinary_jax_xla")
            },
        )
        te_runs = [run for run in plan["runs"] if run["producer"] == "transformer_engine_2_17"]
        self.assertTrue(all(run["argv"][run["argv"].index("--epsilon") + 1] == "1e-06" for run in te_runs))
        self.assertTrue(all("contract" not in run["boundary"] for run in plan["runs"]))
        self.assertEqual(plan["excluded_capability"], NORM_CONTRACT_BLOCKER)

    def test_reports_require_the_closed_te_and_xla_matrix(self) -> None:
        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            validated = validate_report(_report(producer), expected_producer=producer)
            self.assertEqual(len(validated["cells"]), 6)

        for mutation, diagnostic in (
            (lambda report: report.update(epsilon=1.0e-5), "epsilon"),
            (lambda report: report["cells"].pop(), "matrix"),
            (lambda report: report["cells"][0].update(scope="representative_acceptance"), "scope"),
            (
                lambda report: report["capabilities"]["norm_contract"].update(status="supported"),
                "norm_contract",
            ),
            (lambda report: report["cells"][0]["timing"].update(warmup_invocations=0), "warmup"),
            (lambda report: report["cells"][0]["timing"]["raw_host_blocking_ns"].pop(), "samples"),
            (lambda report: report["cells"][0]["outputs"][0].update(role="dx"), "roles"),
        ):
            report = _report("transformer_engine_2_17")
            mutation(report)
            with self.assertRaisesRegex(ValueError, diagnostic):
                validate_report(report, expected_producer="transformer_engine_2_17")

    def test_acceptance_uses_only_complete_representative_cells_and_matched_host_clock(self) -> None:
        te = _report("transformer_engine_2_17")
        xla = _report("ordinary_jax_xla")

        accepted = build_matched_acceptance(te, xla)

        self.assertEqual(accepted["status"], "representative_numerics_accepted")
        self.assertEqual(accepted["accepted_shape"], [2048, 4096])
        self.assertEqual(accepted["accepted_boundaries"], list(BOUNDARIES))
        self.assertEqual(accepted["excluded_architecture_fixture_shape"], [7, 13])
        self.assertEqual(accepted["performance"]["clock"], "host_blocking_ns")
        self.assertEqual(set(accepted["performance"]["median_ratio_te_over_xla"]), set(BOUNDARIES))
        self.assertEqual(accepted["performance"]["status"], "reported_without_threshold")
        self.assertEqual(accepted["norm_contract"]["blocker"], NORM_CONTRACT_BLOCKER)

        architecture_only_te = copy.deepcopy(te)
        architecture_only_xla = copy.deepcopy(xla)
        architecture_only_te["cells"] = architecture_only_te["cells"][:3]
        architecture_only_xla["cells"] = architecture_only_xla["cells"][:3]
        with self.assertRaisesRegex(ValueError, "representative"):
            build_matched_acceptance(architecture_only_te, architecture_only_xla)

        mismatched = copy.deepcopy(xla)
        mismatched["device"]["uuid"] = "GPU-other"
        with self.assertRaisesRegex(ValueError, "device"):
            build_matched_acceptance(te, mismatched)

        mismatched = copy.deepcopy(xla)
        mismatched["cells"][3]["input_digest"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input digest"):
            build_matched_acceptance(te, mismatched)

    def test_runner_source_uses_exact_te_api_and_both_timing_observations(self) -> None:
        source = TE_RUNNER.read_text()

        self.assertIn("nvte_rmsnorm_fwd", source)
        self.assertIn("nvte_rmsnorm_bwd", source)
        self.assertIn("nvte_enable_cudnn_norm_fwd(false)", source)
        self.assertIn("nvte_enable_cudnn_norm_bwd(false)", source)
        self.assertIn("cudaEventElapsedTime", source)
        self.assertIn("std::chrono::steady_clock", source)
        self.assertNotIn("norm_contract", source)


if __name__ == "__main__":
    unittest.main()
