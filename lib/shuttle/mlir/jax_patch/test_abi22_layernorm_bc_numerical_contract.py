# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import math
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np
from abi22_layernorm_bc_numerical_contract import (
    PUBLIC_ROLES,
    SHAPES,
    ErrorMetrics,
    array_digest,
    binary64_reference,
    error_metrics,
    fixed_inputs,
    load_contract,
    require_accepted,
    round_binary64_to_bf16,
    scheduled_reference,
    validate_contract,
)


class LayerNormBCNumericalContractTest(unittest.TestCase):
    def test_models_digests_roles_and_resources_are_frozen_before_hardware(self) -> None:
        contract = load_contract()
        self.assertEqual(len(contract["cells"]), 4)
        evidence = {}
        for cell in contract["cells"]:
            boundary = cell["boundary"]
            shape = tuple(cell["shape"])
            inputs = fixed_inputs(*shape)
            self.assertEqual(
                {role: array_digest(inputs[role]) for role in PUBLIC_ROLES[boundary]["inputs"]},
                cell["input_digests"],
            )
            self.assertEqual(set(cell["input_digests"]), set(PUBLIC_ROLES[boundary]["inputs"]))
            analytic = binary64_reference(inputs, boundary)
            scheduled = scheduled_reference(inputs, boundary)
            self.assertEqual(list(scheduled), PUBLIC_ROLES[boundary]["outputs"])
            self.assertEqual(cell["resources"]["temporary_bytes"], shape[0] * 8)
            for role in PUBLIC_ROLES[boundary]["outputs"]:
                sealed = cell["output_evidence"][role]
                self.assertEqual(array_digest(analytic[role]), sealed["binary64_digest"])
                self.assertEqual(array_digest(scheduled[role]), sealed["schedule_digest"])
                metrics = error_metrics(scheduled[role], analytic[role])
                self.assertEqual(metrics.__dict__, sealed["schedule_metrics"])
                require_accepted(metrics, role)
                evidence[(boundary, shape, role)] = scheduled[role]
        for shape in SHAPES:
            for role in ("dx", "dgamma", "dbeta"):
                self.assertTrue(
                    np.array_equal(
                        evidence[("backward", shape, role)],
                        evidence[("composed", shape, role)],
                    )
                )

    def test_inputs_are_fresh_and_all_streams_are_semantically_live(self) -> None:
        directory = Path(__file__).parent
        forward = json.loads((directory / "abi22-layernorm-forward-numerical-contract-v1.json").read_text())
        prior = {digest for cell in forward["cells"] for digest in cell["input_digests"].values()}
        target1 = json.loads((directory / "target1-rowwise-bf16-numerical-oracle-v1.json").read_text())
        prior.update(digest for digests in target1["inputs"]["digests"].values() for digest in digests)
        rmsnorm = json.loads((directory / "track1-rmsnorm-h100-numerical-policy-v1.json").read_text())
        prior.update(digest for shape in rmsnorm["shapes"].values() for digest in shape["inputs"].values())
        for shape in SHAPES:
            inputs = fixed_inputs(*shape)
            self.assertFalse({array_digest(value) for value in inputs.values()} & prior)
        inputs = fixed_inputs(7, 13)
        expected = scheduled_reference(inputs, "composed")
        mutations = {
            "x": np.roll(inputs["x"], 1, axis=1),
            "gamma": np.roll(inputs["gamma"], 1),
            "beta": np.roll(inputs["beta"], 1),
            "dy": np.roll(inputs["dy"], 1, axis=0),
        }
        for role, mutation in mutations.items():
            changed = {**inputs, role: mutation}
            actual = scheduled_reference(changed, "composed")
            self.assertTrue(
                any(array_digest(actual[name]) != array_digest(expected[name]) for name in expected),
                role,
            )

    def test_wrong_vjp_terms_axes_and_affine_roles_are_rejected(self) -> None:
        inputs = fixed_inputs(7, 13)
        reference = binary64_reference(inputs, "composed")
        x = inputs["x"].astype(np.float64)
        dy = inputs["dy"].astype(np.float64)
        gamma = inputs["gamma"].astype(np.float64)
        beta = inputs["beta"].astype(np.float64)
        mean = np.mean(x, axis=1)
        centered = x - mean[:, None]
        inverse = 1.0 / np.sqrt(np.mean(centered * centered, axis=1) + float(np.float32(1.0e-5)))
        normalized = centered * inverse[:, None]
        weighted = dy * gamma[None, :]
        wrong = {
            "dx": (inverse[:, None] * (weighted - np.mean(weighted, axis=1)[:, None])).astype(ml_dtypes.bfloat16),
            "dgamma": np.sum(dy * np.roll(normalized, 1, axis=1), axis=0).astype(ml_dtypes.bfloat16),
            "dbeta": np.sum(dy, axis=1).astype(ml_dtypes.bfloat16),
            "y": (normalized * np.roll(gamma, 1)[None, :] + np.roll(beta, 1)[None, :]).astype(ml_dtypes.bfloat16),
        }
        for role, actual in wrong.items():
            if actual.shape != reference[role].shape:
                with self.assertRaisesRegex(ValueError, "shapes differ"):
                    error_metrics(actual, reference[role])
            else:
                with self.assertRaises(AssertionError):
                    require_accepted(error_metrics(actual, reference[role]), role)

    def test_schedule_is_not_reassociated_binary64_on_large_boundary(self) -> None:
        inputs = fixed_inputs(2048, 4096)
        scheduled = scheduled_reference(inputs, "composed")
        analytic = binary64_reference(inputs, "composed")
        self.assertNotEqual(array_digest(scheduled["y"]), array_digest(analytic["y"]))
        self.assertNotEqual(array_digest(scheduled["dx"]), array_digest(analytic["dx"]))
        self.assertNotEqual(array_digest(scheduled["dbeta"]), array_digest(analytic["dbeta"]))
        self.assertEqual(error_metrics(scheduled["dbeta"], analytic["dbeta"]).max_bfloat16_ulp_error, 1)

    def test_binary64_oracle_rounds_directly_to_bf16(self) -> None:
        value = np.array([1.00390625 + math.ldexp(1.0, -30)], dtype=np.float64)
        direct = round_binary64_to_bf16(value)
        via_binary32 = value.astype(np.float32).astype(ml_dtypes.bfloat16)
        self.assertEqual(int(direct.view(np.uint16)[0]), 0x3F81)
        self.assertEqual(int(via_binary32.view(np.uint16)[0]), 0x3F80)

    def test_no_near_zero_or_nonfinite_escape(self) -> None:
        reference = np.array([-2.6077032089233398e-08], dtype=ml_dtypes.bfloat16)
        gross = np.array([math.ldexp(1.0, -8)], dtype=ml_dtypes.bfloat16)
        metrics = error_metrics(gross, reference)
        self.assertGreater(metrics.max_bfloat16_ulp_error, 128)
        with self.assertRaises(AssertionError):
            require_accepted(metrics, "y")
        for value in (math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "finite"):
                error_metrics(np.array([value], dtype=ml_dtypes.bfloat16), reference)

    def test_contract_rejects_matrix_role_resource_limit_and_evidence_drift(self) -> None:
        contract = load_contract()
        mutations = []
        changed = copy.deepcopy(contract)
        changed["cells"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][1] = copy.deepcopy(changed["cells"][0])
        mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["public_roles"]["outputs"] = ["dbeta", "dgamma", "dx"]
        mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["resources"]["launch_count"] = 1
        mutations.append(changed)
        for path, value in (
            (("slot_count",), 9),
            (("temporary_slots", 0), 4),
            (("public_binding_slots", "results", 0), 6),
            (("launches", 1, "dependencies"), []),
            (("launches", 0, "grid", 0), 8),
            (("launches", 0, "block", 0), 64),
            (("launches", 0, "dynamic_shared_bytes"), 256),
            (("launches", 1, "output_slots", 0), 6),
        ):
            changed = copy.deepcopy(contract)
            target = changed["cells"][0]["resources"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["limits"]["dx"]["max_absolute_error"] = 1.0
        mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["input_digests"]["x"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(contract)
        changed["cells"][0]["output_evidence"]["dx"]["schedule_metrics"]["mean_absolute_error"] = 1e-12
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(ValueError):
                validate_contract(changed)

    def test_metrics_reject_invalid_values(self) -> None:
        for metrics in (
            ErrorMetrics(-1.0, 0.0, 0.0, 0),
            ErrorMetrics(0.0, math.nan, 0.0, 0),
            ErrorMetrics(0.0, 0.0, math.inf, 0),
            ErrorMetrics(0.0, 0.0, 0.0, -1),
        ):
            with self.assertRaises(ValueError):
                require_accepted(metrics, "dx")


if __name__ == "__main__":
    unittest.main()
