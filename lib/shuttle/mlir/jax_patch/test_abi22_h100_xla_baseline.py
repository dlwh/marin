# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ordinary-JAX H100 baseline producer."""

import ml_dtypes
import numpy as np
from abi22_h100_performance import CELL_KEYS, workload_digest
from abi22_h100_xla_baseline import te_unavailable_status, workloads


def test_workloads_match_the_complete_bf16_benchmark_matrix() -> None:
    matrix = workloads()

    assert [(cell.case, cell.policy) for cell in matrix] == list(CELL_KEYS)
    assert [cell.shape for cell in matrix] == [
        (2, 7, 19),
        (2, 7, 19),
        (2, 7, 19),
        (2, 7, 19),
        (5, 19, 7),
        (5, 19, 7),
        (5, 19, 257),
        (5, 19, 257),
    ]
    assert all(all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in cell.operands) for cell in matrix)
    for source_ordered, fast in zip(matrix[::2], matrix[1::2], strict=True):
        assert all(
            np.array_equal(source_value, fast_value)
            for source_value, fast_value in zip(source_ordered.operands, fast.operands, strict=True)
        )


def test_workload_digest_is_canonical_little_endian_bf16() -> None:
    digest = workload_digest(
        "probe",
        "fast",
        (2, 2),
        (bytes.fromhex("0100020003000400"), bytes.fromhex("05000600")),
    )

    assert digest == "1a39c50a5f3ab75d02bad51a11a33175b8910c8292d7570ccc64d76f932ee1c2"


def test_te_status_names_the_missing_norm_contract_coverage() -> None:
    assert te_unavailable_status() == {
        "schema": "shuttle_h100_exploratory_unavailable_v1",
        "producer": "te",
        "protocol_mode": "exploratory_host_v1",
        "status": "unavailable",
        "blocker": "te_2_17_runner_lacks_norm_contract_cells",
    }
