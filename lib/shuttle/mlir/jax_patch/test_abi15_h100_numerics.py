# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import pytest
from abi15_h100_numerics import (
    COLUMNS,
    REDUCTION,
    ROWS,
    Inputs,
    ReductionPolicy,
    dense_inputs,
    f32_bits_to_fraction,
    fraction_to_bf16_bits,
    fraction_to_f32_bits,
    reference,
    sparse_policy_inputs,
)


def test_binary32_rounding_handles_ties_and_exact_cancellation() -> None:
    large = Fraction(1 << 25)
    rounded = fraction_to_f32_bits(large + 1)
    assert f32_bits_to_fraction(rounded) == large
    cancelled = fraction_to_f32_bits(f32_bits_to_fraction(rounded) - large)
    assert cancelled == 0
    assert fraction_to_bf16_bits(Fraction(1)) == 0x3F80


def test_sparse_inputs_distinguish_source_ordered_and_fast() -> None:
    inputs = sparse_policy_inputs()
    source_ordered = reference(inputs, ReductionPolicy.SOURCE_ORDERED)
    fast = reference(inputs, ReductionPolicy.FAST)
    assert source_ordered.lhs_cotangent[0] == 0x0000
    assert fast.lhs_cotangent[0] == 0x3F80
    assert source_ordered.rhs_cotangent[0] == 0x0000
    assert fast.rhs_cotangent[0] == 0x3F80
    assert source_ordered != fast


def test_dense_reference_closes_all_public_result_shapes() -> None:
    inputs = dense_inputs()
    outputs = reference(inputs, ReductionPolicy.SOURCE_ORDERED)
    assert len(outputs.primal) == ROWS * COLUMNS
    assert len(outputs.lhs_cotangent) == ROWS * REDUCTION
    assert len(outputs.rhs_cotangent) == REDUCTION * COLUMNS
    assert len(set(outputs.primal)) > 16
    assert len(set(outputs.lhs_cotangent)) > 16
    assert len(set(outputs.rhs_cotangent)) > 16


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (Inputs((), (0,) * (REDUCTION * COLUMNS), (0,) * (ROWS * COLUMNS)), "lhs"),
        (Inputs((0,) * (ROWS * REDUCTION), (), (0,) * (ROWS * COLUMNS)), "rhs"),
        (
            Inputs((0,) * (ROWS * REDUCTION), (0,) * (REDUCTION * COLUMNS), ()),
            "output cotangent",
        ),
    ],
)
def test_reference_rejects_mismatched_public_shapes(inputs: Inputs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reference(inputs, ReductionPolicy.SOURCE_ORDERED)
