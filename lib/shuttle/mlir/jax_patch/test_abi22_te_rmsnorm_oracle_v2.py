# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the versioned TE 2.17/XLA RMSNorm oracle lane."""

import copy
import hashlib
import statistics
import unittest
from pathlib import Path

from abi22_te_rmsnorm_oracle_v2 import (
    ACCEPTANCE_LIMITS,
    ARCHITECTURE_SHAPE,
    BOUNDARIES,
    EPSILON,
    INPUT_DIGESTS,
    NORM_CONTRACT_BLOCKER,
    REFERENCE_DIGESTS,
    REPRESENTATIVE_SHAPE,
    build_matched_acceptance,
    execute_plan,
    execution_plan,
    validate_build_manifest,
    validate_report,
)

TE_RUNNER = Path(__file__).with_name("abi22_te_rmsnorm_oracle_runner_v2.cpp")
BUILD_MANIFEST = Path(__file__).with_name("abi22-te-rmsnorm-oracle-runner-build-v2.json")


def _cell(shape: tuple[int, int], boundary: str, *, provider: str) -> dict:
    roles = {
        "forward": ("y",),
        "backward_recompute": ("dx", "dgamma"),
        "composed": ("y", "dx", "dgamma"),
    }[boundary]
    host_scale = 1 if provider == "transformer_engine_2_17" else 2
    host_samples = [host_scale * (100_000 + index * 1_000) for index in range(20)]
    input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    call_sequence = {
        "forward": ["rmsnorm_forward"],
        "backward_recompute": ["rmsnorm_forward_to_private_rsigma", "rmsnorm_backward"],
        "composed": ["rmsnorm_forward_to_public_y_and_private_rsigma", "rmsnorm_backward"],
    }[boundary]
    return {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_acceptance",
        "boundary": boundary,
        "input_digests": {role: INPUT_DIGESTS[shape][role] for role in input_roles},
        "execution": {
            "call_sequence": call_sequence,
            "rsigma_state": "invocation_local_not_public",
        },
        "outputs": [
            {
                "role": role,
                "digest": REFERENCE_DIGESTS[shape][role],
                "reference_digest": REFERENCE_DIGESTS[shape][role],
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
        "reference": {
            "implementation": "independent_numpy_binary64_closed_form_then_bfloat16",
            "input_generator": "numpy_linspace_float32_then_bfloat16_v2",
            "epsilon": EPSILON,
        },
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
    def test_canonical_bf16_inputs_and_independent_references_are_pinned(self) -> None:
        self.assertEqual(
            INPUT_DIGESTS,
            {
                (7, 13): {
                    "x": "bf45599484c0439c23fcd951f8f67645c1812c709002b7cbe3a7f7c67c69a9bb",
                    "gamma": "909f58ed4a0ea4d2a436730afc3cb5c0960eb11019dfa2836d7827019ae87771",
                    "dy": "7f1673f57926522d586d66896ff64949d88a02aa0bd43b246b519004ffa6f89f",
                },
                (2048, 4096): {
                    "x": "0f3b7cfa2054631d63038001deb298b086d0c2baffeaddd6082c4518df91e13c",
                    "gamma": "e90137a7f830b75f00ee6dae7c89bbd7fba30dfb31ff7951fec129441c170821",
                    "dy": "275b6bb7bf115e9110a4403637863cd0a3f0fb2cbb7b18ac9de05560db5a154c",
                },
            },
        )
        self.assertEqual(
            REFERENCE_DIGESTS,
            {
                (7, 13): {
                    "y": "5a78b29a63f3163c7012bcb40581b7290340f189c0e2f6b14036bb7ec1a99ab3",
                    "dx": "79678506dafd3e75dc57813e76f46f9fcaa3f0b5b96d4243ebda9c9cb2a35302",
                    "dgamma": "0b1c5df0409d00bf82a6d90748424f03019c6439d599003f28f9a71c79e7c7e9",
                },
                (2048, 4096): {
                    "y": "85924a7a28e8edb0a21aff788016315a0b804465ae04cac79c69517ff7c71284",
                    "dx": "b832dffb3e31d5e5639c2e272bbd9e74cc5ec3a664298b02604448628c0fbdab",
                    "dgamma": "7e8e78fc4dad00d99b614018ae47cfe67162a859cc10bab79c2c2b0ca73746b3",
                },
            },
        )

    def test_plan_separates_architecture_from_representative_acceptance(self) -> None:
        plan = execution_plan(
            "./abi22_te_rmsnorm_oracle_runner_v2",
            "./abi22_xla_rmsnorm_oracle_v2.py",
            "./python",
        )

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
        for run in plan["runs"]:
            rows, features = run["shape"]
            prefix = (
                ["./abi22_te_rmsnorm_oracle_runner_v2"]
                if run["producer"] == "transformer_engine_2_17"
                else ["./python", "./abi22_xla_rmsnorm_oracle_v2.py"]
            )
            self.assertEqual(
                run["argv"],
                [
                    *prefix,
                    "--boundary",
                    run["boundary"],
                    "--rows",
                    str(rows),
                    "--features",
                    str(features),
                    "--epsilon",
                    "1e-06",
                    "--case-directory",
                    f"cases/{rows}x{features}",
                    "--output",
                    f"raw/{run['producer']}-{rows}x{features}-{run['boundary']}.json",
                ],
            )
        self.assertTrue(all("contract" not in run["boundary"] for run in plan["runs"]))
        self.assertEqual(plan["excluded_capability"], NORM_CONTRACT_BLOCKER)

        invoked: list[list[str]] = []

        def invoke(argv: list[str]) -> int:
            invoked.append(argv)
            return 0

        execute_plan(plan, invoke)
        self.assertEqual(invoked, [run["argv"] for run in plan["runs"]])

        with self.assertRaisesRegex(RuntimeError, "ordinary_jax_xla"):
            execute_plan(plan, lambda argv: 7 if "ordinary_jax_xla" in argv[-1] else 0)

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
            (lambda report: report["cells"][0]["timing"]["raw_cuda_event_ns"].pop(), "CUDA-event"),
            (lambda report: report["cells"][0]["outputs"][0].update(role="dx"), "roles"),
            (
                lambda report: report["cells"][3]["outputs"][0].update(max_absolute_error=0.5),
                "max_absolute_error",
            ),
            (
                lambda report: report["cells"][3]["outputs"][0].update(max_bfloat16_ulp=100),
                "max_bfloat16_ulp",
            ),
            (
                lambda report: report["cells"][5]["outputs"][0].update(reference_digest="f" * 64),
                "composed",
            ),
            (
                lambda report: report["cells"][5]["input_digests"].update(x="e" * 64),
                "input provenance",
            ),
            (
                lambda report: report["cells"][4]["execution"]["call_sequence"].reverse(),
                "call sequence",
            ),
            (
                lambda report: report["cells"][5]["execution"].update(rsigma_state="saved_across_invocations"),
                "rsigma",
            ),
        ):
            report = _report("transformer_engine_2_17")
            mutation(report)
            with self.assertRaisesRegex(ValueError, diagnostic):
                validate_report(report, expected_producer="transformer_engine_2_17")

        for cell_index in (3, 4, 5):
            for field, value in (
                ("max_absolute_error", ACCEPTANCE_LIMITS["max_absolute_error"] + 0.01),
                ("max_bfloat16_ulp", ACCEPTANCE_LIMITS["max_bfloat16_ulp"] + 1),
                ("max_absolute_error", float("nan")),
            ):
                report = _report("transformer_engine_2_17")
                report["cells"][cell_index]["outputs"][0][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_report(report, expected_producer="transformer_engine_2_17")

    def test_reports_pin_provider_identity_and_supplemental_cuda_events(self) -> None:
        for path, value, diagnostic in (
            (("version",), "2.16.0", "version"),
            (("source_commit",), "0" * 40, "source_commit"),
            (("runner_sha256",), "missing", "runner_sha256"),
            (("library_sha256",), "missing", "library_sha256"),
        ):
            report = _report("transformer_engine_2_17")
            report["provider_identity"][path[0]] = value
            with self.assertRaisesRegex(ValueError, diagnostic):
                validate_report(report, expected_producer="transformer_engine_2_17")

        for field, value in (
            ("jax_version", "0.10.0"),
            ("jaxlib_version", "0.10.0"),
            ("xla_revision", "0" * 40),
        ):
            report = _report("ordinary_jax_xla")
            report["provider_identity"][field] = value
            with self.assertRaisesRegex(ValueError, field):
                validate_report(report, expected_producer="ordinary_jax_xla")

        for samples in (None, [1] * 19, [1] * 19 + [0]):
            report = _report("transformer_engine_2_17")
            report["cells"][0]["timing"]["raw_cuda_event_ns"] = samples
            with self.assertRaisesRegex(ValueError, "CUDA-event"):
                validate_report(report, expected_producer="transformer_engine_2_17")

        report = _report("ordinary_jax_xla")
        report["cells"][0]["timing"]["raw_cuda_event_ns"] = [1] * 20
        with self.assertRaisesRegex(ValueError, "CUDA-event"):
            validate_report(report, expected_producer="ordinary_jax_xla")

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
        self.assertEqual(
            accepted["performance"]["median_ratio_te_over_xla"],
            {boundary: 0.5 for boundary in BOUNDARIES},
        )
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
        mismatched["cells"][3]["input_digests"]["x"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input digest"):
            build_matched_acceptance(te, mismatched)

        changed_cuda = copy.deepcopy(te)
        changed_cuda["cells"][3]["timing"]["raw_cuda_event_ns"] = [1_000_000 + index for index in range(20)]
        self.assertEqual(
            build_matched_acceptance(changed_cuda, xla)["performance"]["median_ratio_te_over_xla"],
            {boundary: 0.5 for boundary in BOUNDARIES},
        )

    def test_runner_build_manifest_closes_the_external_te_binary(self) -> None:
        manifest = validate_build_manifest(BUILD_MANIFEST, TE_RUNNER)

        self.assertEqual(manifest["schema"], "shuttle_abi22_te_rmsnorm_runner_build_v2")
        self.assertEqual(manifest["provider"], {"distribution": "transformer_engine", "version": "2.17.0"})
        self.assertEqual(manifest["runner"]["sha256"], hashlib.sha256(TE_RUNNER.read_bytes()).hexdigest())
        self.assertEqual(manifest["build"]["language"], "c++20")
        self.assertEqual(manifest["build"]["cuda_architecture"], "sm_90a")
        self.assertIn("transformer_engine", manifest["build"]["libraries"])
        self.assertIn("cudart", manifest["build"]["libraries"])

        mutated = copy.deepcopy(manifest)
        mutated["runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "runner.*digest"):
            validate_build_manifest(mutated, TE_RUNNER)

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
