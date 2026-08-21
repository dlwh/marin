# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the versioned TE 2.17/XLA RMSNorm oracle lane."""

import copy
import hashlib
import statistics
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import ml_dtypes
import numpy as np
from abi22_te_rmsnorm_oracle_v2 import (
    ACCEPTANCE_INPUT_DIGESTS,
    ACCEPTANCE_LIMITS,
    ACCEPTANCE_PATTERN,
    ACCEPTANCE_REFERENCE_DIGESTS,
    ARCHITECTURE_SHAPE,
    BOUNDARIES,
    ELIGIBLE_COUNTS,
    EPSILON,
    HELDOUT_INPUT_DIGESTS,
    HELDOUT_PATTERN,
    HELDOUT_REFERENCE_DIGESTS,
    INPUT_DIGESTS,
    NORM_CONTRACT_BLOCKER,
    NUMERICAL_POLICY,
    QUALIFICATION_PATTERN,
    REFERENCE_DIGESTS,
    REPRESENTATIVE_SHAPE,
    ValidatedTeRuntimeIdentity,
    build_matched_acceptance,
    epsilon_sensitivity,
    error_metrics,
    execute_plan,
    execute_qualification_plan,
    execution_plan,
    prepare_cases,
    prepare_policy_cases,
    qualification_plan,
    validate_build_manifest,
    validate_qualification_evidence,
    validate_report,
    validate_runtime_manifest,
)

TE_RUNNER = Path(__file__).with_name("abi22_te_rmsnorm_oracle_runner_v2.cpp")
BUILD_MANIFEST = Path(__file__).with_name("abi22-te-rmsnorm-oracle-runner-build-v2.json")


TE_IDENTITY: dict | None = None


def _expected_te_identity() -> Mapping[str, str]:
    if TE_IDENTITY is None:
        raise AssertionError("test runtime identity was not prepared")
    return TE_IDENTITY


def _validate(report: dict, producer: str) -> dict:
    return validate_report(
        report,
        expected_producer=producer,
        expected_te_identity=_expected_te_identity() if producer == "transformer_engine_2_17" else None,
    )


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
        "scope": "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_performance_qualification",
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
                "mean_absolute_error": 0.0,
                "absolute_floor": ACCEPTANCE_LIMITS[role]["absolute_floor"],
                "eligible_count": 1,
                "max_relative_error_above_absolute_floor": 0.0,
                "max_bfloat16_ulp_above_absolute_floor": 0,
                "all_finite": True,
                "raw_max_bfloat16_ulp_index": 0,
                "raw_max_bfloat16_ulp_actual": 0.0,
                "raw_max_bfloat16_ulp_reference": 0.0,
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
            dict(_expected_te_identity())
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


def _qualification_evidence() -> dict:
    cells = []
    for provider in ("transformer_engine_2_17", "ordinary_jax_xla"):
        for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN):
            if pattern == QUALIFICATION_PATTERN:
                input_digests = INPUT_DIGESTS[REPRESENTATIVE_SHAPE]
                reference_digests = REFERENCE_DIGESTS[REPRESENTATIVE_SHAPE]
            elif pattern == HELDOUT_PATTERN:
                input_digests = HELDOUT_INPUT_DIGESTS
                reference_digests = HELDOUT_REFERENCE_DIGESTS
            else:
                input_digests = ACCEPTANCE_INPUT_DIGESTS
                reference_digests = ACCEPTANCE_REFERENCE_DIGESTS
            for boundary in BOUNDARIES:
                for repeat in range(2):
                    cell = _cell(REPRESENTATIVE_SHAPE, boundary, provider=provider)
                    cell.update(
                        producer=provider,
                        input_pattern=pattern,
                        repeat=repeat,
                        scope=(
                            "representative_policy_acceptance"
                            if pattern == ACCEPTANCE_PATTERN
                            else "representative_policy_qualification"
                        ),
                    )
                    cell["input_digests"] = {role: input_digests[role] for role in cell["input_digests"]}
                    for output in cell["outputs"]:
                        output["digest"] = reference_digests[output["role"]]
                        output["reference_digest"] = reference_digests[output["role"]]
                        output["eligible_count"] = ELIGIBLE_COUNTS[pattern][output["role"]]
                    cells.append(cell)
    return validate_qualification_evidence({"schema": "shuttle_abi22_rmsnorm_policy_evidence_v2", "cells": cells})


class TeRmsnormOracleV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global TE_IDENTITY
        cls.runtime_directory = tempfile.TemporaryDirectory()
        library = Path(cls.runtime_directory.name) / "libtransformer_engine.so"
        library.write_bytes(b"sealed-transformer-engine-2.17-library")
        runtime_manifest = {
            "schema": "shuttle_abi22_te_rmsnorm_runtime_v2",
            "runner_sha256": hashlib.sha256(TE_RUNNER.read_bytes()).hexdigest(),
            "transformer_engine_library": {
                "path": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            },
        }
        TE_IDENTITY = validate_runtime_manifest(runtime_manifest, TE_RUNNER)
        cls.runtime_manifest = runtime_manifest

    @classmethod
    def tearDownClass(cls) -> None:
        global TE_IDENTITY
        TE_IDENTITY = None
        cls.runtime_directory.cleanup()

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

    def test_plan_separates_architecture_from_representative_performance_qualification(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case_root = Path(temporary.name) / "cases"
        prepared = prepare_cases(case_root)
        self.assertEqual(prepared["input_digests"], INPUT_DIGESTS)
        self.assertEqual(prepared["reference_digests"], REFERENCE_DIGESTS)
        for shape in (ARCHITECTURE_SHAPE, REPRESENTATIVE_SHAPE):
            rows, features = shape
            directory = case_root / f"{rows}x{features}"
            expected = {
                "x.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["x"]),
                "gamma.bf16": (features * 2, INPUT_DIGESTS[shape]["gamma"]),
                "dy.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["dy"]),
                "reference_y.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["y"]),
                "reference_dx.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["dx"]),
                "reference_dgamma.bf16": (features * 2, REFERENCE_DIGESTS[shape]["dgamma"]),
            }
            for name, (size, digest) in expected.items():
                path = directory / name
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, size)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

        sensitivity = epsilon_sensitivity(case_root, REPRESENTATIVE_SHAPE)
        self.assertEqual(
            sensitivity,
            {
                "y": {"max_absolute_error": 1.36328125, "max_bfloat16_ulp": 231},
                "dx": {"max_absolute_error": 189.0, "max_bfloat16_ulp": 34089},
                "dgamma": {"max_absolute_error": 4.0, "max_bfloat16_ulp": 1},
            },
        )
        self.assertGreater(sensitivity["y"]["max_absolute_error"], ACCEPTANCE_LIMITS["y"]["max_absolute_error"])
        plan = execution_plan(
            "./abi22_te_rmsnorm_oracle_runner_v2",
            "./abi22_xla_rmsnorm_oracle_v2.py",
            "./python",
            case_root,
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
                    (REPRESENTATIVE_SHAPE, "representative_performance_qualification"),
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
                    str(case_root / f"{rows}x{features}"),
                    "--output",
                    str(case_root.parent / "raw" / f"{run['producer']}-{rows}x{features}-{run['boundary']}.json"),
                ],
            )
            self.assertTrue(Path(run["argv"][-1]).is_absolute())
        self.assertTrue(all("contract" not in run["boundary"] for run in plan["runs"]))
        self.assertEqual(plan["excluded_capability"], NORM_CONTRACT_BLOCKER)

        (case_root / "2048x4096/x.bf16").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "case.*digest|case.*size"):
            execution_plan(
                "./abi22_te_rmsnorm_oracle_runner_v2",
                "./abi22_xla_rmsnorm_oracle_v2.py",
                "./python",
                case_root,
            )
        prepare_cases(case_root)
        (case_root / "7x13/gamma.bf16").unlink()
        with self.assertRaisesRegex(ValueError, "case.*missing"):
            execution_plan(
                "./abi22_te_rmsnorm_oracle_runner_v2",
                "./abi22_xla_rmsnorm_oracle_v2.py",
                "./python",
                case_root,
            )
        prepare_cases(case_root)

        invoked: list[list[str]] = []
        execute_plan(plan, lambda argv: invoked.append(argv) or 0)
        self.assertEqual(invoked, [run["argv"] for run in plan["runs"]])

        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            with self.assertRaisesRegex(RuntimeError, producer):
                execute_plan(plan, lambda argv, selected=producer: 7 if selected in argv[-1] else 0)

    def test_role_policy_uses_qualification_inputs_and_one_untouched_acceptance_pattern(self) -> None:
        self.assertEqual(
            NUMERICAL_POLICY,
            {
                "schema": "shuttle_abi22_rmsnorm_role_policy_v2",
                "qualification_patterns": ["qualification_linspace_v2", "heldout_reversed_linspace_v1"],
                "acceptance_patterns": ["heldout_splitmix64_v1"],
                "repeat_count": 2,
                "limits_by_role": {
                    "y": {
                        "max_absolute_error": 0.0078125,
                        "max_mean_absolute_error": 0.0078125,
                        "absolute_floor": 0.015625,
                        "max_relative_error_above_absolute_floor": 0.5,
                        "max_bfloat16_ulp_above_absolute_floor": 2,
                    },
                    "dx": {
                        "max_absolute_error": 0.0625,
                        "max_mean_absolute_error": 0.0625,
                        "absolute_floor": 0.125,
                        "max_relative_error_above_absolute_floor": 0.5,
                        "max_bfloat16_ulp_above_absolute_floor": 64,
                    },
                    "dgamma": {
                        "max_absolute_error": 0.015625,
                        "max_mean_absolute_error": 0.015625,
                        "absolute_floor": 0.03125,
                        "max_relative_error_above_absolute_floor": 0.5,
                        "max_bfloat16_ulp_above_absolute_floor": 8,
                    },
                },
            },
        )
        self.assertEqual(
            HELDOUT_INPUT_DIGESTS,
            {
                "x": "595fc9c13633ba99278591d4288b9150b1d7c71eb17bd9c5f0d4720e1762ea8b",
                "gamma": "bc3f6002b8b4c4718db1aeee0cf2b459c18ec80aa69958d2c8945be915a409d6",
                "dy": "237d26f9c6c048e5d3979da9e283aff46f062518d9a91ea5b74d2f8eaf80526d",
            },
        )
        self.assertEqual(
            ACCEPTANCE_INPUT_DIGESTS,
            {
                "x": "d5306b1c52e907c4f4df004e6f3a9c488f354acd80221eb5077e6f20a1c3f34e",
                "gamma": "7666bd787eded8f496d15ecaa0ff1a09e2785076f6f5e5db47737a95152b86ec",
                "dy": "2abeef4e00143b614ca27c6f310998493609de3490f4720538be507c4ddc4556",
            },
        )
        self.assertEqual(
            ACCEPTANCE_REFERENCE_DIGESTS,
            {
                "y": "dbf0c5cc4900ca5498099bd458edfdf31200a692fd2996e66960e8b98b77078e",
                "dx": "938fe85ca5d7820ca1ae7dfb8acc970307ad88ec92d94b77bd024e904c2d8c55",
                "dgamma": "209869785a9897bfd13ff39665e0ad781d2bb14b3b7c15067a1f6c752c4601fb",
            },
        )
        self.assertEqual(
            ELIGIBLE_COUNTS,
            {
                QUALIFICATION_PATTERN: {"y": 8_225_616, "dx": 6_523_647, "dgamma": 4_096},
                HELDOUT_PATTERN: {"y": 8_388_455, "dx": 5_397_253, "dgamma": 4_096},
                ACCEPTANCE_PATTERN: {"y": 8_305_853, "dx": 7_722_818, "dgamma": 4_091},
            },
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case_root = Path(temporary.name) / "cases"
        prepare_cases(case_root)
        prepared = prepare_policy_cases(case_root)
        self.assertEqual(
            prepared,
            {
                HELDOUT_PATTERN: {
                    "input_digests": HELDOUT_INPUT_DIGESTS,
                    "reference_digests": HELDOUT_REFERENCE_DIGESTS,
                },
                ACCEPTANCE_PATTERN: {
                    "input_digests": ACCEPTANCE_INPUT_DIGESTS,
                    "reference_digests": ACCEPTANCE_REFERENCE_DIGESTS,
                },
            },
        )
        for pattern, inputs, references in (
            (HELDOUT_PATTERN, HELDOUT_INPUT_DIGESTS, HELDOUT_REFERENCE_DIGESTS),
            (ACCEPTANCE_PATTERN, ACCEPTANCE_INPUT_DIGESTS, ACCEPTANCE_REFERENCE_DIGESTS),
        ):
            directory = case_root / pattern
            for name, digest in {
                "x.bf16": inputs["x"],
                "gamma.bf16": inputs["gamma"],
                "dy.bf16": inputs["dy"],
                "reference_y.bf16": references["y"],
                "reference_dx.bf16": references["dx"],
                "reference_dgamma.bf16": references["dgamma"],
            }.items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)
        plan = qualification_plan("./te", "./xla.py", "./python", case_root)
        self.assertEqual(plan["schema"], "shuttle_abi22_rmsnorm_policy_plan_v2")
        self.assertEqual(len(plan["runs"]), 36)
        self.assertEqual(
            {(run["input_pattern"], run["boundary"], run["producer"], run["repeat"]) for run in plan["runs"]},
            {
                (pattern, boundary, producer, repeat)
                for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN, ACCEPTANCE_PATTERN)
                for boundary in BOUNDARIES
                for producer in ("transformer_engine_2_17", "ordinary_jax_xla")
                for repeat in range(2)
            },
        )
        for run in plan["runs"]:
            self.assertEqual(run["shape"], [2048, 4096])
            self.assertEqual(run["argv"][run["argv"].index("--epsilon") + 1], "1e-06")
            expected_directory = "2048x4096" if run["input_pattern"] == QUALIFICATION_PATTERN else run["input_pattern"]
            self.assertEqual(Path(run["argv"][run["argv"].index("--case-directory") + 1]).name, expected_directory)
            self.assertIn(f"repeat{run['repeat']}", Path(run["argv"][-1]).name)
            self.assertTrue(Path(run["argv"][-1]).is_absolute())
        acceptance_directory = case_root / ACCEPTANCE_PATTERN
        (acceptance_directory / "x.bf16").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "case.*digest|case.*size"):
            qualification_plan("./te", "./xla.py", "./python", case_root)
        prepare_policy_cases(case_root)
        invoked: list[list[str]] = []
        execute_qualification_plan(plan, lambda argv: invoked.append(argv) or 0)
        self.assertEqual(invoked, [run["argv"] for run in plan["runs"]])

    def test_reports_require_the_closed_te_and_xla_matrix(self) -> None:
        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            validated = _validate(_report(producer), producer)
            self.assertEqual(len(validated["cells"]), 6)

        for mutation, diagnostic in (
            (lambda report: report.update(epsilon=1.0e-5), "epsilon"),
            (lambda report: report["cells"].pop(), "matrix"),
            (
                lambda report: report["cells"][0].update(scope="representative_performance_qualification"),
                "scope",
            ),
            (
                lambda report: report["capabilities"]["norm_contract"].update(status="supported"),
                "norm_contract",
            ),
            (lambda report: report["cells"][0]["timing"].update(warmup_invocations=0), "warmup"),
            (lambda report: report["cells"][0]["timing"]["raw_host_blocking_ns"].pop(), "samples"),
            (lambda report: report["cells"][0]["timing"]["raw_cuda_event_ns"].pop(), "CUDA-event"),
            (lambda report: report["cells"][0]["outputs"][0].update(role="dx"), "roles"),
            (lambda report: report["cells"][3]["outputs"][0].update(all_finite=False), "all_finite"),
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
                _validate(report, "transformer_engine_2_17")

        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            qualification_only = _report(producer)
            qualification_only["cells"][3]["outputs"][0].update(
                max_absolute_error=0.5,
                mean_absolute_error=0.25,
                max_bfloat16_ulp=10_000,
                max_relative_error_above_absolute_floor=2.0,
                max_bfloat16_ulp_above_absolute_floor=10_000,
            )
            _validate(qualification_only, producer)

        wrong_sequences = {
            "forward": ["rmsnorm_backward"],
            "backward_recompute": ["rmsnorm_backward", "rmsnorm_forward_to_private_rsigma"],
            "composed": ["rmsnorm_backward", "rmsnorm_forward_to_public_y_and_private_rsigma"],
        }
        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            for cell_index, cell in enumerate(_report(producer)["cells"]):
                report = _report(producer)
                report["cells"][cell_index]["execution"]["call_sequence"] = wrong_sequences[cell["boundary"]]
                with self.assertRaisesRegex(ValueError, "call sequence"):
                    _validate(report, producer)

        qualified = _qualification_evidence()
        self.assertEqual(qualified["policy"], NUMERICAL_POLICY)
        for mutation, diagnostic in (
            (lambda evidence: evidence["cells"][0]["outputs"][0].update(role="dx"), "role"),
            (
                lambda evidence: evidence["cells"][6]["outputs"][0].update(max_absolute_error=float("inf")),
                "finite",
            ),
            (lambda evidence: evidence["cells"][1]["outputs"][0].update(digest="d" * 64), "repeat"),
            (lambda evidence: evidence["cells"][0].update(input_pattern=HELDOUT_PATTERN), "coordinate"),
        ):
            evidence = copy.deepcopy(qualified)
            mutation(evidence)
            with self.assertRaisesRegex(ValueError, diagnostic):
                validate_qualification_evidence(evidence)
        for index, cell in enumerate(qualified["cells"]):
            if cell["input_pattern"] != ACCEPTANCE_PATTERN or cell["repeat"] != 0:
                continue
            for output_index, output in enumerate(cell["outputs"]):
                role = output["role"]
                for field, value in (
                    ("max_absolute_error", ACCEPTANCE_LIMITS[role]["max_absolute_error"] + 1.0e-6),
                    (
                        "mean_absolute_error",
                        ACCEPTANCE_LIMITS[role]["max_mean_absolute_error"] + 1.0e-6,
                    ),
                    ("mean_absolute_error", float("nan")),
                    (
                        "max_relative_error_above_absolute_floor",
                        ACCEPTANCE_LIMITS[role]["max_relative_error_above_absolute_floor"] + 1.0e-6,
                    ),
                    (
                        "max_bfloat16_ulp_above_absolute_floor",
                        ACCEPTANCE_LIMITS[role]["max_bfloat16_ulp_above_absolute_floor"] + 1,
                    ),
                    ("eligible_count", 0),
                    ("eligible_count", ELIGIBLE_COUNTS[ACCEPTANCE_PATTERN][role] + 1),
                    ("absolute_floor", ACCEPTANCE_LIMITS[role]["absolute_floor"] * 2),
                    ("all_finite", False),
                ):
                    evidence = copy.deepcopy(qualified)
                    evidence["cells"][index]["outputs"][output_index][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        validate_qualification_evidence(evidence)

        for index, cell in enumerate(qualified["cells"]):
            for output_index, _output in enumerate(cell["outputs"]):
                evidence = copy.deepcopy(qualified)
                evidence["cells"][index]["outputs"][output_index]["eligible_count"] += 1
                with self.assertRaisesRegex(ValueError, "eligible_count"):
                    validate_qualification_evidence(evidence)

        diagnostic = copy.deepcopy(qualified)
        reversed_dx = next(
            output
            for cell in diagnostic["cells"]
            if cell["producer"] == "transformer_engine_2_17"
            and cell["input_pattern"] == HELDOUT_PATTERN
            and cell["boundary"] == "backward_recompute"
            and cell["repeat"] == 0
            for output in cell["outputs"]
            if output["role"] == "dx"
        )
        reversed_dx.update(max_absolute_error=0.0625, mean_absolute_error=0.0625, max_bfloat16_ulp=283)
        validate_qualification_evidence(diagnostic)

        global_ulp_is_diagnostic = copy.deepcopy(qualified)
        acceptance_output = next(
            output
            for cell in global_ulp_is_diagnostic["cells"]
            if cell["input_pattern"] == ACCEPTANCE_PATTERN
            for output in cell["outputs"]
        )
        acceptance_output["max_bfloat16_ulp"] = 10_000
        validate_qualification_evidence(global_ulp_is_diagnostic)

        acceptance_repeat = copy.deepcopy(qualified)
        repeated_output = next(
            output
            for cell in acceptance_repeat["cells"]
            if cell["producer"] == "ordinary_jax_xla"
            and cell["input_pattern"] == ACCEPTANCE_PATTERN
            and cell["boundary"] == "composed"
            and cell["repeat"] == 1
            for output in cell["outputs"]
            if output["role"] == "dx"
        )
        repeated_output["digest"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "repeat"):
            validate_qualification_evidence(acceptance_repeat)

        distinct = copy.deepcopy(qualified)
        for index in (2, 3):
            distinct["cells"][index]["outputs"][0][
                "digest"
            ] = "b09429e74cce38c20cc84b56a7d99aa51ded89a0e3d1a942e96eb43390965249"
        for index in (4, 5):
            distinct["cells"][index]["outputs"][1][
                "digest"
            ] = "502041f57708e86411c5e2fafe675aba5208a89194cc82b3f8db43e0b5f6ace5"
        validate_qualification_evidence(distinct)

        for producer in ("transformer_engine_2_17", "ordinary_jax_xla"):
            report = _report(producer)
            report["cells"][4]["outputs"][0][
                "digest"
            ] = "b09429e74cce38c20cc84b56a7d99aa51ded89a0e3d1a942e96eb43390965249"
            report["cells"][5]["outputs"][1][
                "digest"
            ] = "502041f57708e86411c5e2fafe675aba5208a89194cc82b3f8db43e0b5f6ace5"
            _validate(report, producer)
            report["cells"][4]["outputs"][0]["digest"] = "not-a-sha"
            with self.assertRaisesRegex(ValueError, "output digest"):
                _validate(report, producer)

    def test_error_metrics_separates_global_ulp_from_the_absolute_floor_mask(self) -> None:
        reference = np.array([-0.01, 0.125, 1.0], dtype=ml_dtypes.bfloat16)
        actual = np.array([-0.0625, 0.0625, 1.0078125], dtype=ml_dtypes.bfloat16)

        metrics = error_metrics(actual, reference, role="dx")

        self.assertEqual(metrics["all_finite"], True)
        self.assertEqual(metrics["max_absolute_error"], 0.0625)
        self.assertAlmostEqual(metrics["mean_absolute_error"], 0.040934244791666664)
        self.assertEqual(metrics["max_bfloat16_ulp"], 348)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_index"], 0)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_actual"], -0.0625)
        self.assertEqual(metrics["raw_max_bfloat16_ulp_reference"], -0.010009765625)
        self.assertEqual(metrics["absolute_floor"], 0.125)
        self.assertEqual(metrics["eligible_count"], 2)
        self.assertEqual(metrics["max_relative_error_above_absolute_floor"], 0.5)
        self.assertEqual(metrics["max_bfloat16_ulp_above_absolute_floor"], 128)

    def test_reports_pin_provider_identity_and_supplemental_cuda_events(self) -> None:
        for path, value, diagnostic in (
            (("version",), "2.16.0", "version"),
            (("source_commit",), "0" * 40, "source_commit"),
            (("runner_sha256",), "3" * 64, "runner_sha256"),
            (("library_sha256",), "4" * 64, "library_sha256"),
        ):
            report = _report("transformer_engine_2_17")
            report["provider_identity"][path[0]] = value
            with self.assertRaisesRegex(ValueError, diagnostic):
                _validate(report, "transformer_engine_2_17")

        for field, value in (
            ("jax_version", "0.10.0"),
            ("jaxlib_version", "0.10.0"),
            ("xla_revision", "0" * 40),
        ):
            report = _report("ordinary_jax_xla")
            report["provider_identity"][field] = value
            with self.assertRaisesRegex(ValueError, field):
                _validate(report, "ordinary_jax_xla")

        for samples in (None, [1] * 19, [1] * 19 + [0]):
            report = _report("transformer_engine_2_17")
            report["cells"][0]["timing"]["raw_cuda_event_ns"] = samples
            with self.assertRaisesRegex(ValueError, "CUDA-event"):
                _validate(report, "transformer_engine_2_17")

        report = _report("ordinary_jax_xla")
        report["cells"][0]["timing"]["raw_cuda_event_ns"] = [1] * 20
        with self.assertRaisesRegex(ValueError, "CUDA-event"):
            _validate(report, "ordinary_jax_xla")

    def test_acceptance_uses_only_complete_representative_cells_and_matched_host_clock(self) -> None:
        te = _report("transformer_engine_2_17")
        xla = _report("ordinary_jax_xla")

        qualification = _qualification_evidence()
        accepted = build_matched_acceptance(
            te, xla, qualification=qualification, expected_te_identity=_expected_te_identity()
        )

        self.assertEqual(accepted["status"], "heldout_splitmix64_numerics_accepted")
        self.assertEqual(accepted["accepted_input_pattern"], ACCEPTANCE_PATTERN)
        self.assertEqual(
            accepted["metric_scope"],
            {
                "whole_output": ["all_finite", "max_absolute_error", "mean_absolute_error"],
                "above_absolute_floor": [
                    "eligible_count_nonzero",
                    "max_relative_error",
                    "max_bfloat16_ulp",
                ],
                "raw_global_bfloat16_ulp": "diagnostic_only",
            },
        )
        self.assertEqual(accepted["accepted_shape"], [2048, 4096])
        self.assertEqual(accepted["accepted_boundaries"], list(BOUNDARIES))
        self.assertEqual(accepted["excluded_architecture_fixture_shape"], [7, 13])
        self.assertEqual(accepted["numerical_policy"], NUMERICAL_POLICY)
        self.assertEqual(accepted["qualification"], qualification)
        self.assertEqual(accepted["performance"]["clock"], "host_blocking_ns")
        self.assertEqual(accepted["performance"]["input_pattern"], QUALIFICATION_PATTERN)
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
            build_matched_acceptance(
                architecture_only_te,
                architecture_only_xla,
                qualification=qualification,
                expected_te_identity=_expected_te_identity(),
            )

        mismatched = copy.deepcopy(xla)
        mismatched["device"]["uuid"] = "GPU-other"
        with self.assertRaisesRegex(ValueError, "device"):
            build_matched_acceptance(
                te, mismatched, qualification=qualification, expected_te_identity=_expected_te_identity()
            )

        mismatched = copy.deepcopy(xla)
        mismatched["cells"][3]["input_digests"]["x"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input digest"):
            build_matched_acceptance(
                te, mismatched, qualification=qualification, expected_te_identity=_expected_te_identity()
            )

        changed_cuda = copy.deepcopy(te)
        changed_cuda["cells"][3]["timing"]["raw_cuda_event_ns"] = [1_000_000 + index for index in range(20)]
        self.assertEqual(
            build_matched_acceptance(
                changed_cuda,
                xla,
                qualification=qualification,
                expected_te_identity=_expected_te_identity(),
            )["performance"]["median_ratio_te_over_xla"],
            {boundary: 0.5 for boundary in BOUNDARIES},
        )

        corrupt_median = copy.deepcopy(te)
        corrupt_median["cells"][3]["timing"]["median_host_blocking_ns"] += 1
        with self.assertRaisesRegex(ValueError, "median_host_blocking_ns"):
            build_matched_acceptance(
                corrupt_median,
                xla,
                qualification=qualification,
                expected_te_identity=_expected_te_identity(),
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

        runtime = validate_runtime_manifest(self.runtime_manifest, TE_RUNNER)
        self.assertEqual(runtime, _expected_te_identity())
        with self.assertRaisesRegex(TypeError, "created only"):
            ValidatedTeRuntimeIdentity(dict(runtime), object())
        with self.assertRaisesRegex(TypeError, "immutable"):
            runtime._values = {}  # type: ignore[misc]
        mutated_runtime = copy.deepcopy(self.runtime_manifest)
        mutated_runtime["transformer_engine_library"]["sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "library.*digest"):
            validate_runtime_manifest(mutated_runtime, TE_RUNNER)

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
