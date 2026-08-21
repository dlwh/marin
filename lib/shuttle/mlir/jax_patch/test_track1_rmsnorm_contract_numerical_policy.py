# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
from track1_rmsnorm_contract_numerical_policy import (
    POLICY_SHA256,
    SHAPES,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    emitted_primal_oracle,
    load_numerical_policy,
)


def test_policy_binds_inputs_oracles_and_exact_epsilon() -> None:
    policy = load_numerical_policy()

    assert policy["recurrence"]["epsilon"] == "1e-6"
    assert policy["status"] == "predeclared_unexecuted"
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
