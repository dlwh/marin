# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
from track1_rmsnorm_contract_numerical_policy import (
    MASKED_ULP_ABSOLUTE_FLOORS,
    POLICY_FILENAME,
    POLICY_SHA256,
    SHAPES,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    emitted_primal_oracle,
    load_numerical_policy,
    numerical_metrics,
)


def test_policy_binds_inputs_oracles_and_exact_epsilon() -> None:
    policy = load_numerical_policy()

    assert POLICY_FILENAME == "track1-rmsnorm-contract-h100-numerical-policy-v2.json"
    assert policy["schema_version"] == 2
    assert policy["recurrence"]["epsilon"] == "1e-6"
    assert policy["status"] == "predeclared_unexecuted"
    assert policy["acceptance"]["coordinate_order"] == "all_primal_before_any_composed"
    assert policy["acceptance"]["disabled_jax"] == "diagnostic_only_never_threshold_gated"
    assert policy["acceptance"]["raw_full_range_bf16_ulp"] == "diagnostic_only"
    assert len(POLICY_SHA256) == 64
    for shape in SHAPES:
        x, weight, projection, cotangent = deterministic_inputs(shape)
        rows, reduction, columns = shape
        assert x.shape == (rows, reduction)
        assert weight.shape == (reduction,)
        assert projection.shape == (reduction, columns)
        assert cotangent.shape == (rows, columns)
        assert {value.dtype for value in (x, weight, projection, cotangent)} == {np.dtype(ml_dtypes.bfloat16)}
        assert all(np.isfinite(value.astype(np.float32)).all() for value in (x, weight, projection, cotangent))

        analytic = binary64_oracle(shape)
        assert tuple(analytic) == ("result", "input_cotangent", "weight_cotangent", "projection_cotangent")
        assert analytic["result"].shape == (rows, columns)
        assert analytic["input_cotangent"].shape == (rows, reduction)
        assert analytic["weight_cotangent"].shape == (reduction,)
        assert analytic["projection_cotangent"].shape == (reduction, columns)
        for fast in (False, True):
            primal = emitted_primal_oracle(shape, fast=fast)
            assert primal.shape == (rows, columns)
            assert primal.dtype == np.dtype(ml_dtypes.bfloat16)

        shape_policy = policy["shapes"]["x".join(map(str, shape))]
        assert shape_policy["inputs"] == {
            "cotangent": array_sha256(cotangent),
            "projection": array_sha256(projection),
            "weight": array_sha256(weight),
            "x": array_sha256(x),
        }


def test_v2_inputs_are_new_and_exactly_bound() -> None:
    expected = {
        (5, 19, 7): {
            "cotangent": "c76e12b59c7d74279ed779afaea929e9545b94a27ce88728a7060f405dbc1b9f",
            "projection": "c548904f19130005325fb2b6f01a54cae6e3a8c1fbe73558c93ec67aff1c61f2",
            "weight": "f478adc355a7291649090d43f0b0635bbef2636254eeb14a9af964f8ab5394c0",
            "x": "535415147e92e7151300e1c3e3ec1a5fa926a1356993b1db61d17ccf5c42bc94",
        },
        (5, 19, 257): {
            "cotangent": "bd579fc07c4def281a0f67fd5f6e37115306ce5ce804475432e5fa83243b06c4",
            "projection": "a58b622fb22def1896432cb65828ae95c2f850dad3728f270b708a6ea1a131f6",
            "weight": "f478adc355a7291649090d43f0b0635bbef2636254eeb14a9af964f8ab5394c0",
            "x": "535415147e92e7151300e1c3e3ec1a5fa926a1356993b1db61d17ccf5c42bc94",
        },
    }

    for shape, identities in expected.items():
        x, weight, projection, cotangent = deterministic_inputs(shape)
        assert {
            "cotangent": array_sha256(cotangent),
            "projection": array_sha256(projection),
            "weight": array_sha256(weight),
            "x": array_sha256(x),
        } == identities


def test_piecewise_ulp_ignores_only_reference_values_below_the_role_floor() -> None:
    reference = np.asarray([0.0001, 1.0], dtype=np.float64)
    actual = np.asarray([0.0, 1.0], dtype=ml_dtypes.bfloat16)

    metrics = numerical_metrics(
        actual,
        reference,
        masked_ulp_absolute_floor=MASKED_ULP_ABSOLUTE_FLOORS["input_cotangent"],
    )

    assert metrics.maximum_bf16_ulp_error > 8
    assert metrics.maximum_masked_bf16_ulp_error == 0
