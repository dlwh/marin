# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the scale-consistent RMSNorm role policy v3."""

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

ACCEPTANCE_PATTERN = "heldout_splitmix64_v2"
QUALIFICATION_PATTERNS = (
    "qualification_linspace_v2",
    "heldout_reversed_linspace_v1",
    "heldout_splitmix64_v1",
)
INPUT_DIGESTS = {
    "x": "94149f756a9a5cbd8d4ef1fd5b72284666f790ec06da9c4fd34f91981de72b0d",
    "gamma": "a6798be78031f1e5ea51e24649b26a9b952f5b00543a8bf878a1fc68dae96c8e",
    "dy": "16300ef5a34dbe3ae28fc07f6ade2e6f72f9ae63480bc34feb40581ee3bf3818",
}
REFERENCE_DIGESTS = {
    "y": "9adf2dbe23cbd85a4a2841e911a17fc769bad679764dc41a933fd29b3e2f5e7a",
    "dx": "7c4e9301b933be7ba6a94d30bf3e00a1e6071e31982c25ea1cafe74ce439fad3",
    "dgamma": "fc63fa70b69a390e51b5fc6a3a2f2c6e70373d1d4f9f79ff5c9f02681dc907e8",
}
SEED_PREIMAGES = {
    "x": "shuttle_abi22_rmsnorm_role_policy_v3/x",
    "gamma": "shuttle_abi22_rmsnorm_role_policy_v3/gamma",
    "dy": "shuttle_abi22_rmsnorm_role_policy_v3/dy",
}
SEEDS = {
    "x": 0x8AEBC21044665234,
    "gamma": 0xC234BF5558EEE5F8,
    "dy": 0x9AF2F87A8D1BB35C,
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
ABOVE_COUNTS = {"y": 8_305_127, "dx": 7_725_746, "dgamma": 4_091}
BELOW_COUNTS = {"y": 83_481, "dx": 662_862, "dgamma": 5}
OUTPUT_SIZES = {"y": 8_388_608, "dx": 8_388_608, "dgamma": 4_096}
REFERENCE_MAXIMA = {"y": 2.640625, "dx": 2.6875, "dgamma": 94.0}
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
    ACCEPTANCE_PATTERN: {
        "y": {"below_floor": 83_481, "above_floor": 8_305_127},
        "dx": {"below_floor": 662_862, "above_floor": 7_725_746},
        "dgamma": {"below_floor": 5, "above_floor": 4_091},
    },
}


def _api(name: str) -> Callable[..., Any]:
    value = getattr(oracle, name, None)
    if not callable(value):
        raise AssertionError(f"RMSNorm policy v3 API is missing: {name}")
    return value


def _v3_value(name: str) -> Any:
    if not hasattr(oracle, name):
        raise AssertionError(f"RMSNorm policy v3 value is missing: {name}")
    return getattr(oracle, name)


def _pattern_digests(pattern: str) -> tuple[dict[str, str], dict[str, str]]:
    if pattern == oracle.QUALIFICATION_PATTERN:
        return oracle.INPUT_DIGESTS[oracle.REPRESENTATIVE_SHAPE], oracle.REFERENCE_DIGESTS[oracle.REPRESENTATIVE_SHAPE]
    if pattern == oracle.HELDOUT_PATTERN:
        return oracle.HELDOUT_INPUT_DIGESTS, oracle.HELDOUT_REFERENCE_DIGESTS
    if pattern == oracle.ACCEPTANCE_PATTERN:
        return oracle.ACCEPTANCE_INPUT_DIGESTS, oracle.ACCEPTANCE_REFERENCE_DIGESTS
    if pattern == ACCEPTANCE_PATTERN:
        return INPUT_DIGESTS, REFERENCE_DIGESTS
    raise AssertionError(pattern)


def _policy_evidence() -> dict[str, Any]:
    partition_counts = _v3_value("POLICY_V3_PARTITION_COUNTS")
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
    return {"schema": "shuttle_abi22_rmsnorm_policy_evidence_v3", "cells": cells}


class TeRmsnormPolicyV3Test(unittest.TestCase):
    def test_policy_freezes_piecewise_limits_and_fourth_untouched_input(self) -> None:
        self.assertEqual(
            _v3_value("NUMERICAL_POLICY_V3"),
            {
                "schema": "shuttle_abi22_rmsnorm_role_policy_v3",
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
                "rejection_rule": "no_post_result_widening_or_promotion",
            },
        )
        self.assertEqual(_v3_value("POLICY_V3_ACCEPTANCE_INPUT_DIGESTS"), INPUT_DIGESTS)
        self.assertEqual(_v3_value("POLICY_V3_ACCEPTANCE_REFERENCE_DIGESTS"), REFERENCE_DIGESTS)
        self.assertEqual(_v3_value("POLICY_V3_REFERENCE_MAXIMA"), REFERENCE_MAXIMA)
        self.assertEqual(_v3_value("POLICY_V3_PARTITION_COUNTS"), PARTITION_COUNTS)
        for role in OUTPUT_SIZES:
            self.assertEqual(BELOW_COUNTS[role] + ABOVE_COUNTS[role], OUTPUT_SIZES[role])
        for role, preimage in SEED_PREIMAGES.items():
            self.assertEqual(int.from_bytes(hashlib.sha256(preimage.encode()).digest()[:8], "big"), SEEDS[role])

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case_root = Path(temporary.name) / "cases"
        oracle.prepare_cases(case_root)
        _api("prepare_policy_v3_cases")(case_root)
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

        plan = _api("qualification_plan_v3")("./te", "./xla.py", "./python", case_root)
        self.assertEqual(plan["schema"], "shuttle_abi22_rmsnorm_policy_plan_v3")
        self.assertEqual(len(plan["runs"]), 48)
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
                    ],
                },
            )
        invoked: list[list[str]] = []
        _api("execute_qualification_plan_v3")(plan, lambda argv: invoked.append(argv) or 0)
        self.assertEqual(invoked, [run["argv"] for run in plan["runs"]])
        plan["runs"][0]["argv"][-1] = str(case_root.parent / "raw" / "substituted.json")
        with self.assertRaisesRegex(ValueError, "canonical|plan"):
            _api("execute_qualification_plan_v3")(plan, lambda argv: 0)

    def test_piecewise_metrics_cover_every_element_and_close_whole_maximum(self) -> None:
        actual = np.asarray([0.0244140625, -0.25, 1.03125], dtype=ml_dtypes.bfloat16)
        reference = np.asarray([-0.00677490234375, -0.25, 1.0], dtype=ml_dtypes.bfloat16)
        metrics = _api("error_metrics_v3")(actual, reference, role="y")

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
                _api("error_metrics_v3")(nonfinite_actual, finite_reference, role="y")["all_finite"],
                False,
            )

    def test_validator_mutation_locks_every_acceptance_gate_and_repeat_digest(self) -> None:
        validate = _api("validate_policy_v3_evidence")
        evidence = _policy_evidence()
        validated = validate(evidence)
        self.assertEqual(validated["policy"], _v3_value("NUMERICAL_POLICY_V3"))
        acceptance_cells = [
            (index, cell)
            for index, cell in enumerate(validated["cells"])
            if cell["input_pattern"] == ACCEPTANCE_PATTERN and cell["repeat"] == 0
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
                    corrupted = copy.deepcopy(validated)
                    corrupted["cells"][index]["outputs"][output_index][field] = value
                    if field.startswith("max_absolute_error_"):
                        below = corrupted["cells"][index]["outputs"][output_index]["max_absolute_error_below_floor"]
                        above = corrupted["cells"][index]["outputs"][output_index]["max_absolute_error_above_floor"]
                        corrupted["cells"][index]["outputs"][output_index]["max_absolute_error"] = max(below, above)
                    with self.assertRaises(ValueError, msg=f"{role}/{field}"):
                        validate(corrupted)

                for field in ("below_floor_count", "eligible_count"):
                    corrupted = copy.deepcopy(validated)
                    corrupted["cells"][index]["outputs"][output_index][field] += 1
                    with self.assertRaises(ValueError, msg=f"{role}/{field}"):
                        validate(corrupted)

                count_swap = copy.deepcopy(validated)
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
                    nonfinite = copy.deepcopy(validated)
                    nonfinite["cells"][index]["outputs"][output_index][field] = float("nan")
                    with self.assertRaisesRegex(ValueError, "finite|absolute|relative"):
                        validate(nonfinite)

                corrupted = copy.deepcopy(validated)
                corrupted["cells"][index]["outputs"][output_index]["max_absolute_error"] = 0.125
                with self.assertRaisesRegex(ValueError, "partition|whole|max_absolute"):
                    validate(corrupted)

        structural_cells = [(index, cell) for index, cell in enumerate(validated["cells"]) if cell["repeat"] == 0]
        for index, cell in structural_cells:
            for output_index, output in enumerate(cell["outputs"]):
                coordinate = f"{cell['producer']}/{cell['input_pattern']}/{cell['boundary']}/{output['role']}"
                count_swap = copy.deepcopy(validated)
                count_output = count_swap["cells"][index]["outputs"][output_index]
                count_output["below_floor_count"] += 1
                count_output["eligible_count"] -= 1
                with self.assertRaisesRegex(ValueError, "count", msg=coordinate):
                    validate(count_swap)

                for field in (
                    "max_absolute_error",
                    "mean_absolute_error",
                    "max_absolute_error_below_floor",
                    "max_absolute_error_above_floor",
                    "max_relative_error_above_absolute_floor",
                ):
                    nonfinite = copy.deepcopy(validated)
                    nonfinite["cells"][index]["outputs"][output_index][field] = float("nan")
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite|absolute|relative",
                        msg=f"{coordinate}/{field}",
                    ):
                        validate(nonfinite)

                false_finite = copy.deepcopy(validated)
                false_finite["cells"][index]["outputs"][output_index]["all_finite"] = False
                with self.assertRaisesRegex(ValueError, "finite", msg=coordinate):
                    validate(false_finite)

                inconsistent_whole = copy.deepcopy(validated)
                inconsistent_whole["cells"][index]["outputs"][output_index]["max_absolute_error"] = 0.125
                with self.assertRaisesRegex(
                    ValueError,
                    "partition|whole|max_absolute",
                    msg=coordinate,
                ):
                    validate(inconsistent_whole)

        repeated = copy.deepcopy(validated)
        repeat_coordinates = [
            (producer, pattern, boundary, role)
            for producer in oracle.PRODUCERS
            for pattern in (*QUALIFICATION_PATTERNS, ACCEPTANCE_PATTERN)
            for boundary in oracle.BOUNDARIES
            for role in oracle.OUTPUT_ROLES[boundary]
        ]
        for producer, pattern, boundary, role in repeat_coordinates:
            repeated = copy.deepcopy(validated)
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

        role_swap = copy.deepcopy(validated)
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

    def test_repeat_diagnostics_reports_pairwise_drift_before_rejection(self) -> None:
        first = np.asarray([0.0, 1.0, -2.0], dtype=ml_dtypes.bfloat16)
        second = np.asarray([0.0, 1.0078125, -1.0], dtype=ml_dtypes.bfloat16)
        diagnostics = _api("repeat_output_diagnostics_v3")(first, second)

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
