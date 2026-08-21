# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ordinary-JAX H100 baseline producer."""

import unittest

import abi22_h100_xla_baseline as baseline
import ml_dtypes
import numpy as np
from abi22_h100_performance import CELL_KEYS, workload_digest


class _Result:
    def __init__(self) -> None:
        self.cast_dtype: object | None = None

    def astype(self, dtype: object) -> "_Result":
        self.cast_dtype = dtype
        return self


class _Jnp:
    float32 = object()
    bfloat16 = object()

    def __init__(self) -> None:
        self.call: tuple[object, object, object] | None = None
        self.result = _Result()

    def matmul(self, lhs: object, rhs: object, *, preferred_element_type: object) -> _Result:
        self.call = (lhs, rhs, preferred_element_type)
        return self.result


class XlaBaselineTest(unittest.TestCase):
    def test_workloads_match_the_complete_bf16_benchmark_matrix(self) -> None:
        matrix = baseline.workloads()

        self.assertEqual([(cell.case, cell.policy) for cell in matrix], list(CELL_KEYS))
        self.assertEqual(
            [cell.shape for cell in matrix],
            [
                (2, 7, 19),
                (2, 7, 19),
                (2, 7, 19),
                (2, 7, 19),
                (5, 19, 7),
                (5, 19, 7),
                (5, 19, 257),
                (5, 19, 257),
            ],
        )
        self.assertTrue(
            all(all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in cell.operands) for cell in matrix)
        )
        for source_ordered, fast in zip(matrix[::2], matrix[1::2], strict=True):
            self.assertTrue(
                all(
                    np.array_equal(source_value, fast_value)
                    for source_value, fast_value in zip(source_ordered.operands, fast.operands, strict=True)
                )
            )

    def test_contract_keeps_bf16_operands_and_requests_f32_accumulation(self) -> None:
        fake_jnp = _Jnp()
        normalized = object()
        projection = object()
        old_jnp = baseline.jnp
        old_rmsnorm = baseline._rmsnorm
        baseline.jnp = fake_jnp
        baseline._rmsnorm = lambda x, weight: normalized
        try:
            result = baseline._rmsnorm_contract(object(), object(), projection)
        finally:
            baseline.jnp = old_jnp
            baseline._rmsnorm = old_rmsnorm

        self.assertIs(result, fake_jnp.result)
        self.assertEqual(fake_jnp.call, (normalized, projection, fake_jnp.float32))
        self.assertIs(result.cast_dtype, fake_jnp.bfloat16)

    def test_workload_digest_is_canonical_little_endian_bf16(self) -> None:
        digest = workload_digest(
            "probe",
            "fast",
            (2, 2),
            (bytes.fromhex("0100020003000400"), bytes.fromhex("05000600")),
        )

        self.assertEqual(digest, "1a39c50a5f3ab75d02bad51a11a33175b8910c8292d7570ccc64d76f932ee1c2")

    def test_te_status_names_the_missing_norm_contract_coverage(self) -> None:
        self.assertEqual(
            baseline.te_unavailable_status(),
            {
                "schema": "shuttle_h100_exploratory_unavailable_v1",
                "producer": "te",
                "protocol_mode": "exploratory_host_v1",
                "status": "unavailable",
                "blocker": "te_2_17_runner_lacks_required_shapes_epsilon_and_norm_contract",
            },
        )


if __name__ == "__main__":
    unittest.main()
