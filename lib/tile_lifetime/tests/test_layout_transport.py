# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime.layout_transport import canonical_layout_transport_program
from tile_lifetime.tensor_program import DType, ProgramValue, TensorAxis, execute_tensor_program


@pytest.mark.parametrize("physical_order", [(0, 3, 1, 2), (0, 1, 2, 3), (2, 0, 3, 1)])
def test_physical_layout_transport_restores_canonical_values(physical_order: tuple[int, ...]) -> None:
    axes = tuple(TensorAxis(index, extent, f"dimension_{index}") for index, extent in enumerate((1, 2, 5, 3)))
    output = ProgramValue("canonical", axes, DType.FP32)
    expected = np.arange(30, dtype=np.float32).reshape(output.shape)
    physical = np.ascontiguousarray(expected.transpose(physical_order))
    program = canonical_layout_transport_program(
        output,
        tuple(axes[index] for index in physical_order),
        tuple(stride // physical.itemsize for stride in physical.strides),
    )
    actual = execute_tensor_program(program, {"physical_storage": physical.reshape(-1)})[output.name]
    np.testing.assert_array_equal(actual, expected)


def test_physical_layout_transport_preserves_strided_gradient_storage() -> None:
    row, head, feature = TensorAxis(0, 3, "row"), TensorAxis(1, 2, "head"), TensorAxis(2, 5, "feature")
    output = ProgramValue("gradient", (row, head, feature), DType.FP32)
    expected = np.arange(30, dtype=np.float32).reshape(output.shape)
    # The physical tensor is a view of already-canonical dense gradient storage.
    physical = expected.transpose(1, 0, 2)
    program = canonical_layout_transport_program(
        output, (head, row, feature), tuple(stride // physical.itemsize for stride in physical.strides)
    )
    actual = execute_tensor_program(program, {"physical_storage": expected.reshape(-1)})[output.name]
    np.testing.assert_array_equal(actual, expected)


def test_physical_layout_transport_rejects_overlapping_storage() -> None:
    axes = (TensorAxis(0, 2, "row"), TensorAxis(1, 2, "column"))
    with pytest.raises(ValueError, match="dense nonoverlapping"):
        canonical_layout_transport_program(ProgramValue("result", axes, DType.FP32), axes, (1, 1))
