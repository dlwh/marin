# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the scale-consistent RMSNorm role policy v4."""

import copy
import hashlib
import statistics
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import abi22_te_rmsnorm_oracle_v2 as oracle
import ml_dtypes
import numpy as np

ACCEPTANCE_PATTERN = "heldout_splitmix64_v3"
QUALIFICATION_PATTERNS = (
    "qualification_linspace_v2",
    "heldout_reversed_linspace_v1",
    "heldout_splitmix64_v1",
    "heldout_splitmix64_v2",
)
INPUT_DIGESTS = {
    "x": "1528589b5505fba379086deffe6c699c36114d3dc72600bbfe464e17947d71b2",
    "gamma": "7d44ce0c29c1cdd0b59ee5be57f1cc3656563d68095c9d73769bea144cc7bd09",
    "dy": "d1565501f4b951240b46d328a0459b81c1426a44f39cd0d6066cedb3eac0005d",
}
REFERENCE_DIGESTS = {
    "y": "437e9adc3e7cfd518132cfac8455d253a34bbfcc081109632611ff7ba4e62a70",
    "dx": "558ae600e406ade38183c4c9726bbfed16b60c07bb0888b4021b38c87408d3f8",
    "dgamma": "28d69308ae4d26785592deb62547d5db17ba50c20c8a21b6e4108349929f55f6",
}
SEED_PREIMAGES = {
    "x": "shuttle_abi22_rmsnorm_role_policy_v4/x",
    "gamma": "shuttle_abi22_rmsnorm_role_policy_v4/gamma",
    "dy": "shuttle_abi22_rmsnorm_role_policy_v4/dy",
}
SEEDS = {
    "x": 0x628E8AB85F0D2BAD,
    "gamma": 0xD105A656E9DF5E9D,
    "dy": 0x717C9E023B564D40,
}
LIMITS = {
    "y": {
        "absolute_floor": 0.015625,
        "max_absolute_error_below_floor": 0.0078125,
        "max_absolute_error_above_floor": 0.03125,
        "max_mean_absolute_error": 0.00000095367431640625,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 2,
        "reference_magnitude_envelope": 4.0,
    },
    "dx": {
        "absolute_floor": 0.125,
        "max_absolute_error_below_floor": 0.0625,
        "max_absolute_error_above_floor": 1.0,
        "max_mean_absolute_error": 0.0000019073486328125,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 64,
        "reference_magnitude_envelope": 4.0,
    },
    "dgamma": {
        "absolute_floor": 0.03125,
        "max_absolute_error_below_floor": 0.015625,
        "max_absolute_error_above_floor": 4.0,
        "max_mean_absolute_error": 0.0009765625,
        "max_relative_error_above_absolute_floor": 0.5,
        "max_bfloat16_ulp_above_absolute_floor": 8,
        "reference_magnitude_envelope": 128.0,
    },
}
ABOVE_COUNTS = {"y": 8_304_608, "dx": 7_716_446, "dgamma": 4_092}
BELOW_COUNTS = {"y": 84_000, "dx": 672_162, "dgamma": 4}
OUTPUT_SIZES = {"y": 8_388_608, "dx": 8_388_608, "dgamma": 4_096}
REFERENCE_MAXIMA = {"y": 2.640625, "dx": 2.65625, "dgamma": 92.5}
PARTITION_COUNTS = {
    "qualification_linspace_v2": {
        "y": {"below_floor": 162_992, "above_floor": 8_225_616},
        "dx": {"below_floor": 1_864_961, "above_floor": 6_523_647},
        "dgamma": {"below_floor": 0, "above_floor": 4_096},
    },
    "heldout_reversed_linspace_v1": {
        "y": {"below_floor": 153, "above_floor": 8_388_455},
        "dx": {"below_floor": 2_991_355, "above_floor": 5_397_253},
        "dgamma": {"below_floor": 0, "above_floor": 4_096},
    },
    "heldout_splitmix64_v1": {
        "y": {"below_floor": 82_755, "above_floor": 8_305_853},
        "dx": {"below_floor": 665_790, "above_floor": 7_722_818},
        "dgamma": {"below_floor": 5, "above_floor": 4_091},
    },
    "heldout_splitmix64_v2": {
        "y": {"below_floor": 83_481, "above_floor": 8_305_127},
        "dx": {"below_floor": 662_862, "above_floor": 7_725_746},
        "dgamma": {"below_floor": 5, "above_floor": 4_091},
    },
    ACCEPTANCE_PATTERN: {
        "y": {"below_floor": 84_000, "above_floor": 8_304_608},
        "dx": {"below_floor": 672_162, "above_floor": 7_716_446},
        "dgamma": {"below_floor": 4, "above_floor": 4_092},
    },
}


def _api(name: str) -> Callable[..., Any]:
    value = getattr(oracle, name, None)
    if not callable(value):
        raise AssertionError(f"RMSNorm policy v4 API is missing: {name}")
    return value


def _v4_value(name: str) -> Any:
    if not hasattr(oracle, name):
        raise AssertionError(f"RMSNorm policy v4 value is missing: {name}")
    return getattr(oracle, name)


def _pattern_digests(pattern: str) -> tuple[dict[str, str], dict[str, str]]:
    if pattern == oracle.QUALIFICATION_PATTERN:
        return oracle.INPUT_DIGESTS[oracle.REPRESENTATIVE_SHAPE], oracle.REFERENCE_DIGESTS[oracle.REPRESENTATIVE_SHAPE]
    if pattern == oracle.HELDOUT_PATTERN:
        return oracle.HELDOUT_INPUT_DIGESTS, oracle.HELDOUT_REFERENCE_DIGESTS
    if pattern == oracle.ACCEPTANCE_PATTERN:
        return oracle.ACCEPTANCE_INPUT_DIGESTS, oracle.ACCEPTANCE_REFERENCE_DIGESTS
    if pattern == oracle.POLICY_V3_ACCEPTANCE_PATTERN:
        return oracle.POLICY_V3_ACCEPTANCE_INPUT_DIGESTS, oracle.POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS
    if pattern == ACCEPTANCE_PATTERN:
        return INPUT_DIGESTS, REFERENCE_DIGESTS
    raise AssertionError(pattern)


def _policy_evidence() -> dict[str, Any]:
    partition_counts = _v4_value("POLICY_V4_PARTITION_COUNTS")
    cells = []
    for producer in oracle.PRODUCERS:
        for pattern in (*QUALIFICATION_PATTERNS, ACCEPTANCE_PATTERN):
            input_digests, reference_digests = _pattern_digests(pattern)
            for boundary in oracle.BOUNDARIES:
                roles = oracle.OUTPUT_ROLES[boundary]
                input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
                for repeat in range(2):
                    cells.append(
                        {
                            "shape": list(oracle.REPRESENTATIVE_SHAPE),
                            "scope": (
                                "representative_policy_acceptance"
                                if pattern == ACCEPTANCE_PATTERN
                                else "representative_policy_qualification"
                            ),
                            "producer": producer,
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "repeat": repeat,
                            "input_digests": {role: input_digests[role] for role in input_roles},
                            "execution": {
                                "call_sequence": list(oracle.CALL_SEQUENCES[boundary]),
                                "rsigma_state": "invocation_local_not_public",
                            },
                            "outputs": [
                                {
                                    "role": role,
                                    "digest": reference_digests[role],
                                    "reference_digest": reference_digests[role],
                                    "max_absolute_error": 0.0,
                                    "mean_absolute_error": 0.0,
                                    "max_bfloat16_ulp": 0,
                                    "raw_max_bfloat16_ulp_index": 0,
                                    "raw_max_bfloat16_ulp_actual": 0.0,
                                    "raw_max_bfloat16_ulp_reference": 0.0,
                                    "absolute_floor": LIMITS[role]["absolute_floor"],
                                    "below_floor_count": partition_counts[pattern][role]["below_floor"],
                                    "eligible_count": partition_counts[pattern][role]["above_floor"],
                                    "max_absolute_error_below_floor": 0.0,
                                    "max_absolute_error_above_floor": 0.0,
                                    "max_relative_error_above_absolute_floor": 0.0,
                                    "max_bfloat16_ulp_above_absolute_floor": 0,
                                    "all_finite": True,
                                }
                                for role in roles
                            ],
                            "timing": {
                                "warmup_invocations": 10,
                                "raw_host_blocking_ns": [100_000 + index for index in range(20)],
                                "median_host_blocking_ns": statistics.median([100_000 + index for index in range(20)]),
                                "raw_cuda_event_ns": (
                                    [50_000 + index for index in range(20)]
                                    if producer == "transformer_engine_2_17"
                                    else None
                                ),
                            },
                        }
                    )
    return {"schema": "shuttle_abi22_rmsnorm_policy_evidence_v4", "cells": cells}


class TeRmsnormPolicyV4Test(unittest.TestCase):
    def test_policy_freezes_standard_separation_and_fifth_untouched_input(self) -> None:
        self.assertEqual(
            _v4_value("NUMERICAL_POLICY_V4"),
            {
                "schema": "shuttle_abi22_rmsnorm_role_policy_v4",
                "qualification_patterns": list(QUALIFICATION_PATTERNS),
                "acceptance_patterns": [ACCEPTANCE_PATTERN],
                "repeat_count": 2,
                "generator": {
                    "algorithm": "splitmix64_counter_high24_to_bfloat16_v1",
                    "seed_derivation": "sha256_utf8_first_8_bytes_big_endian",
                    "seed_preimages": SEED_PREIMAGES,
                    "seeds": SEEDS,
                },
                "limits_by_role": LIMITS,
                "dgamma_mean_limit_basis": "prospective_l1_budget_4_over_4096_not_empirically_calibrated",
                "qualification_disposition": "diagnostic_only_cannot_reject_acceptance",
                "acceptance_repeatability": "exact_bfloat16_digest_per_provider_boundary_role",
                "rejection_rule": "acceptance_pattern_only_no_post_result_widening_or_promotion",
            },
        )
        self.assertEqual(_v4_value("POLICY_V4_ACCEPTANCE_INPUT_DIGESTS"), INPUT_DIGESTS)
        self.assertEqual(_v4_value("POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS"), REFERENCE_DIGESTS)
        self.assertEqual(_v4_value("POLICY_V4_REFERENCE_MAXIMA"), REFERENCE_MAXIMA)
        self.assertEqual(_v4_value("POLICY_V4_PARTITION_COUNTS"), PARTITION_COUNTS)
        for role in OUTPUT_SIZES:
            self.assertEqual(BELOW_COUNTS[role] + ABOVE_COUNTS[role], OUTPUT_SIZES[role])
        for role, preimage in SEED_PREIMAGES.items():
            self.assertEqual(int.from_bytes(hashlib.sha256(preimage.encode()).digest()[:8], "big"), SEEDS[role])

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case_root = Path(temporary.name) / "cases"
        oracle.prepare_cases(case_root)
        _api("prepare_policy_v4_cases")(case_root)
        directory = case_root / ACCEPTANCE_PATTERN
        for name, digest in {
            "x.bf16": INPUT_DIGESTS["x"],
            "gamma.bf16": INPUT_DIGESTS["gamma"],
            "dy.bf16": INPUT_DIGESTS["dy"],
            "reference_y.bf16": REFERENCE_DIGESTS["y"],
            "reference_dx.bf16": REFERENCE_DIGESTS["dx"],
            "reference_dgamma.bf16": REFERENCE_DIGESTS["dgamma"],
        }.items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)

        plan = _api("qualification_plan_v4")("./te", "./xla.py", "./python", case_root)
        self.assertEqual(plan["schema"], "shuttle_abi22_rmsnorm_policy_plan_v4")
        self.assertEqual(len(plan["runs"]), 60)
        self.assertEqual(
            [(run["input_pattern"], run["boundary"], run["producer"], run["repeat"]) for run in plan["runs"]],
            [
                (pattern, boundary, producer, repeat)
                for pattern in (*QUALIFICATION_PATTERNS, ACCEPTANCE_PATTERN)
                for boundary in oracle.BOUNDARIES
                for producer in oracle.PRODUCERS
                for repeat in range(2)
            ],
        )
        rows, features = oracle.REPRESENTATIVE_SHAPE
        for run in plan["runs"]:
            pattern = run["input_pattern"]
            directory = case_root / (f"{rows}x{features}" if pattern == oracle.QUALIFICATION_PATTERN else pattern)
            prefix = ["./te"] if run["producer"] == "transformer_engine_2_17" else ["./python", "./xla.py"]
            output = (
                case_root.parent
                / "raw"
                / (f"policy-{run['producer']}-{pattern}-{run['boundary']}" f"-repeat{run['repeat']}.json")
            )
            output_files = output.parent / f"{output.stem}-outputs"
            persistence_argv = (
                ["--output-files-directory", str(output_files)] if run["producer"] == "ordinary_jax_xla" else []
            )
            self.assertEqual(
                run,
                {
                    "shape": [rows, features],
                    "scope": (
                        "representative_policy_acceptance"
                        if pattern == ACCEPTANCE_PATTERN
                        else "representative_policy_qualification"
                    ),
                    "input_pattern": pattern,
                    "boundary": run["boundary"],
                    "producer": run["producer"],
                    "repeat": run["repeat"],
                    "argv": [
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
                        str(directory),
                        "--output",
                        str(output),
                        *persistence_argv,
                    ],
                },
            )
        invoked: list[list[str]] = []
        _api("execute_qualification_plan_v4")(plan, lambda argv: invoked.append(argv) or 0)
        self.assertEqual(invoked, [run["argv"] for run in plan["runs"]])
        plan["runs"][0]["argv"][-1] = str(case_root.parent / "raw" / "substituted.json")
        with self.assertRaisesRegex(ValueError, "canonical|plan"):
            _api("execute_qualification_plan_v4")(plan, lambda argv: 0)

    def test_piecewise_metrics_cover_every_element_and_close_whole_maximum(self) -> None:
        actual = np.asarray([0.0244140625, -0.25, 1.03125], dtype=ml_dtypes.bfloat16)
        reference = np.asarray([-0.00677490234375, -0.25, 1.0], dtype=ml_dtypes.bfloat16)
        metrics = _api("error_metrics_v4")(actual, reference, role="y")

        self.assertEqual(metrics["below_floor_count"], 1)
        self.assertEqual(metrics["eligible_count"], 2)
        self.assertEqual(metrics["below_floor_count"] + metrics["eligible_count"], actual.size)
        self.assertEqual(metrics["max_absolute_error_below_floor"], 0.03118896484375)
        self.assertEqual(metrics["max_absolute_error_above_floor"], 0.03125)
        self.assertEqual(metrics["max_absolute_error"], 0.03125)
        self.assertEqual(metrics["mean_absolute_error"], 0.02081298828125)
        self.assertEqual(metrics["max_bfloat16_ulp"], 30_887)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_index"], 0)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_actual"], 0.0244140625)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_reference"], -0.00677490234375)
        self.assertEqual(
            metrics["max_absolute_error"],
            max(metrics["max_absolute_error_below_floor"], metrics["max_absolute_error_above_floor"]),
        )
        self.assertEqual(metrics["max_relative_error_above_absolute_floor"], 0.03125)
        self.assertEqual(metrics["max_bfloat16_ulp_above_absolute_floor"], 4)
        self.assertIs(metrics["all_finite"], True)
        self.assertGreater(metrics["max_absolute_error_below_floor"], LIMITS["y"]["max_absolute_error_below_floor"])

        for nonfinite in (float("nan"), float("inf")):
            nonfinite_actual = np.asarray([nonfinite, 1.0], dtype=ml_dtypes.bfloat16)
            finite_reference = np.asarray([0.0, 1.0], dtype=ml_dtypes.bfloat16)
            self.assertIs(
                _api("error_metrics_v4")(nonfinite_actual, finite_reference, role="y")["all_finite"],
                False,
            )

    def test_validator_mutation_locks_every_acceptance_gate_and_repeat_digest(self) -> None:
        validate = _api("validate_policy_v4_evidence")
        evidence = _policy_evidence()
        validated = validate(evidence)
        self.assertEqual(validated["policy"], _v4_value("NUMERICAL_POLICY_V4"))
        self.assertEqual(
            validated["acceptance"],
            {
                "status": "accepted",
                "input_pattern": ACCEPTANCE_PATTERN,
                "repeat_coordinate_count": 12,
            },
        )
        self.assertEqual(
            validated["qualification"],
            {
                "status": "diagnostic_only",
                "patterns": list(QUALIFICATION_PATTERNS),
                "repeat_digest_drifts": [],
            },
        )
        acceptance_cells = [
            (index, cell) for index, cell in enumerate(validated["cells"]) if cell["input_pattern"] == ACCEPTANCE_PATTERN
        ]
        for index, cell in acceptance_cells:
            for output_index, output in enumerate(cell["outputs"]):
                role = output["role"]
                mutations = (
                    ("max_absolute_error_below_floor", LIMITS[role]["max_absolute_error_below_floor"] + 1e-6),
                    ("max_absolute_error_above_floor", LIMITS[role]["max_absolute_error_above_floor"] + 1e-6),
                    ("mean_absolute_error", LIMITS[role]["max_mean_absolute_error"] + 1e-9),
                    ("max_relative_error_above_absolute_floor", 0.5001),
                    ("max_bfloat16_ulp_above_absolute_floor", LIMITS[role]["max_bfloat16_ulp_above_absolute_floor"] + 1),
                    ("all_finite", False),
                )
                for field, value in mutations:
                    corrupted = copy.deepcopy(evidence)
                    corrupted["cells"][index]["outputs"][output_index][field] = value
                    if field.startswith("max_absolute_error_"):
                        below = corrupted["cells"][index]["outputs"][output_index]["max_absolute_error_below_floor"]
                        above = corrupted["cells"][index]["outputs"][output_index]["max_absolute_error_above_floor"]
                        corrupted["cells"][index]["outputs"][output_index]["max_absolute_error"] = max(below, above)
                    with self.assertRaises(ValueError, msg=f"{role}/{field}"):
                        validate(corrupted)

                for field in ("below_floor_count", "eligible_count"):
                    corrupted = copy.deepcopy(evidence)
                    corrupted["cells"][index]["outputs"][output_index][field] += 1
                    with self.assertRaises(ValueError, msg=f"{role}/{field}"):
                        validate(corrupted)

                count_swap = copy.deepcopy(evidence)
                count_output = count_swap["cells"][index]["outputs"][output_index]
                count_output["below_floor_count"] += 1
                count_output["eligible_count"] -= 1
                with self.assertRaisesRegex(ValueError, "count"):
                    validate(count_swap)

                for field in (
                    "max_absolute_error",
                    "mean_absolute_error",
                    "max_absolute_error_below_floor",
                    "max_absolute_error_above_floor",
                    "max_relative_error_above_absolute_floor",
                ):
                    nonfinite = copy.deepcopy(evidence)
                    nonfinite["cells"][index]["outputs"][output_index][field] = float("nan")
                    with self.assertRaisesRegex(ValueError, "finite|absolute|relative"):
                        validate(nonfinite)

                corrupted = copy.deepcopy(evidence)
                corrupted["cells"][index]["outputs"][output_index]["max_absolute_error"] = 0.125
                with self.assertRaisesRegex(ValueError, "partition|whole|max_absolute"):
                    validate(corrupted)

        acceptance_repeat_coordinates = [
            (producer, pattern, boundary, role)
            for producer in oracle.PRODUCERS
            for pattern in (ACCEPTANCE_PATTERN,)
            for boundary in oracle.BOUNDARIES
            for role in oracle.OUTPUT_ROLES[boundary]
        ]
        for producer, pattern, boundary, role in acceptance_repeat_coordinates:
            repeated = copy.deepcopy(evidence)
            cell = next(
                cell
                for cell in repeated["cells"]
                if cell["producer"] == producer
                and cell["input_pattern"] == pattern
                and cell["boundary"] == boundary
                and cell["repeat"] == 1
            )
            output = next(output for output in cell["outputs"] if output["role"] == role)
            output["digest"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "repeat"):
                validate(repeated)

        role_swap = copy.deepcopy(evidence)
        composed = next(
            cell
            for cell in role_swap["cells"]
            if cell["producer"] == "transformer_engine_2_17"
            and cell["input_pattern"] == ACCEPTANCE_PATTERN
            and cell["boundary"] == "composed"
            and cell["repeat"] == 0
        )
        composed["outputs"][0], composed["outputs"][1] = composed["outputs"][1], composed["outputs"][0]
        with self.assertRaisesRegex(ValueError, "role"):
            validate(role_swap)

    def test_qualification_metrics_and_repeat_drift_are_reported_but_non_gating(self) -> None:
        validate = _api("validate_policy_v4_evidence")
        qualification_coordinates = [
            (producer, pattern, boundary, role)
            for producer in oracle.PRODUCERS
            for pattern in QUALIFICATION_PATTERNS
            for boundary in oracle.BOUNDARIES
            for role in oracle.OUTPUT_ROLES[boundary]
        ]
        for producer, pattern, boundary, role in qualification_coordinates:
            evidence = _policy_evidence()
            cells = [
                cell
                for cell in evidence["cells"]
                if cell["producer"] == producer and cell["input_pattern"] == pattern and cell["boundary"] == boundary
            ]
            for cell in cells:
                output = next(output for output in cell["outputs"] if output["role"] == role)
                output["max_absolute_error_below_floor"] = 1024.0
                output["max_absolute_error_above_floor"] = 1024.0
                output["max_absolute_error"] = 1024.0
                output["mean_absolute_error"] = 1024.0
                output["max_relative_error_above_absolute_floor"] = 1024.0
                output["max_bfloat16_ulp_above_absolute_floor"] = 65_535
                output["digest"] = ("e" if cell["repeat"] == 0 else "f") * 64
            repeat1 = next(cell for cell in cells if cell["repeat"] == 1)
            self.assertEqual(next(output for output in repeat1["outputs"] if output["role"] == role)["digest"], "f" * 64)

            validated = validate(evidence)
            self.assertEqual(validated["acceptance"]["status"], "accepted")
            self.assertEqual(
                validated["qualification"]["repeat_digest_drifts"],
                [
                    {
                        "producer": producer,
                        "input_pattern": pattern,
                        "boundary": boundary,
                        "role": role,
                        "repeat0_digest": "e" * 64,
                        "repeat1_digest": "f" * 64,
                    }
                ],
            )

    def test_validator_binds_complete_matrix_provenance_roles_and_partitions(self) -> None:
        validate = _api("validate_policy_v4_evidence")
        evidence = _policy_evidence()
        for cells in (
            evidence["cells"][:-1],
            [*evidence["cells"], copy.deepcopy(evidence["cells"][-1])],
            [evidence["cells"][1], evidence["cells"][0], *evidence["cells"][2:]],
        ):
            corrupted = copy.deepcopy(evidence)
            corrupted["cells"] = cells
            with self.assertRaisesRegex(ValueError, "cell|coordinate|matrix"):
                validate(corrupted)

        for index, cell in enumerate(evidence["cells"]):
            input_role = next(iter(cell["input_digests"]))
            mutations = (
                ("shape", [1, 1]),
                ("scope", "wrong_scope"),
                ("input_digests", {**cell["input_digests"], input_role: "0" * 64}),
                (
                    "execution",
                    {"call_sequence": ["wrong_call"], "rsigma_state": "invocation_local_not_public"},
                ),
            )
            for field, value in mutations:
                corrupted = copy.deepcopy(evidence)
                corrupted["cells"][index][field] = value
                with self.assertRaisesRegex(ValueError, "coordinate|input|call|provenance|scope|shape"):
                    validate(corrupted)

            role_order = copy.deepcopy(evidence)
            outputs = role_order["cells"][index]["outputs"]
            if len(outputs) == 1:
                outputs[0]["role"] = "dx"
            else:
                outputs[0], outputs[1] = outputs[1], outputs[0]
            with self.assertRaisesRegex(ValueError, "role"):
                validate(role_order)

            for output_index in range(len(cell["outputs"])):
                bad_reference = copy.deepcopy(evidence)
                bad_reference["cells"][index]["outputs"][output_index]["reference_digest"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "reference"):
                    validate(bad_reference)

                bad_count = copy.deepcopy(evidence)
                count_output = bad_count["cells"][index]["outputs"][output_index]
                count_output["below_floor_count"] += 1
                count_output["eligible_count"] -= 1
                with self.assertRaisesRegex(ValueError, "count"):
                    validate(bad_count)

                bad_closure = copy.deepcopy(evidence)
                bad_closure["cells"][index]["outputs"][output_index]["max_absolute_error"] = 1.0
                with self.assertRaisesRegex(ValueError, "partition|whole|max_absolute"):
                    validate(bad_closure)

    def test_repeat_diagnostics_reports_pairwise_drift_before_rejection(self) -> None:
        first = np.asarray([0.0, 1.0, -2.0], dtype=ml_dtypes.bfloat16)
        second = np.asarray([0.0, 1.0078125, -1.0], dtype=ml_dtypes.bfloat16)
        diagnostics = _api("repeat_output_diagnostics_v4")(first, second)

        self.assertEqual(diagnostics["mismatch_count"], 2)
        self.assertEqual(diagnostics["max_absolute_error"], 1.0)
        self.assertGreater(diagnostics["max_bfloat16_ulp"], 0)
        self.assertEqual(diagnostics["first_mismatch_index"], 1)
        self.assertEqual(diagnostics["first_mismatch_repeat0"], 1.0)
        self.assertEqual(diagnostics["first_mismatch_repeat1"], 1.0078125)
        self.assertEqual(diagnostics["max_absolute_error_index"], 2)
        self.assertEqual(diagnostics["max_absolute_error_repeat0"], -2.0)
        self.assertEqual(diagnostics["max_absolute_error_repeat1"], -1.0)
        for field in diagnostics:
            if isinstance(diagnostics[field], float):
                self.assertTrue(np.isfinite(diagnostics[field]))


if __name__ == "__main__":
    unittest.main()
