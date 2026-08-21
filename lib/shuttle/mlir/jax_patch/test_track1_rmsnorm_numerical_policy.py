# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
from track1_rmsnorm_numerical_policy import (
    ANALYTIC_LIMITS,
    POLICY_SHA256,
    array_sha256,
    binary64_oracle,
    emitted_tree_oracle,
    load_numerical_policy,
    numerical_metrics,
)


def test_policy_predeclares_rank_three_and_rank_five_role_digests() -> None:
    policy = load_numerical_policy()

    assert len(POLICY_SHA256) == 64
    assert policy["status"] == "predeclared_unexecuted"
    assert policy["approved_source"]["schedule_exact_identity_imported"] is False
    assert policy["recurrence"]["schedule_identity"] == "abi22_emitted_ptx_v1"
    assert policy["recurrence"]["lane_leaf_register_initialization"] == ("positive_zero_before_conditional_load")
    assert tuple(policy["shapes"]) == ("2x7x19", "3x2x5x11x263")
    assert policy["shapes"]["3x2x5x11x263"]["emitted_tree_outputs"]["input_cotangent"] == (
        "b7b633ae8f64da15b63b74be5c8582c18ec9c8d3a978bac3edb646168ea97b31"
    )


def test_predeclared_emitted_tree_roles_satisfy_every_independent_limit() -> None:
    policy = load_numerical_policy()
    for shape, shape_id in (((2, 7, 19), "2x7x19"), ((3, 2, 5, 11, 263), "3x2x5x11x263")):
        emitted = emitted_tree_oracle(shape)
        analytic = binary64_oracle(shape)
        for role in ("result", "input_cotangent", "weight_cotangent"):
            assert array_sha256(emitted[role]) == policy["shapes"][shape_id]["emitted_tree_outputs"][role]
            assert array_sha256(analytic[role].astype(ml_dtypes.bfloat16)) == (
                policy["shapes"][shape_id]["binary64_rounded_outputs"][role]
            )
            metrics = numerical_metrics(emitted[role], analytic[role])
            assert metrics.maximum_absolute_error <= ANALYTIC_LIMITS[role]["maximum_absolute_error"]
            assert metrics.maximum_bf16_ulp_error <= ANALYTIC_LIMITS[role]["maximum_bf16_ulp_error"]
            assert metrics.mean_absolute_error <= ANALYTIC_LIMITS[role]["mean_absolute_error"]
            assert metrics.relative_linf_error <= ANALYTIC_LIMITS[role]["relative_linf_error"]
