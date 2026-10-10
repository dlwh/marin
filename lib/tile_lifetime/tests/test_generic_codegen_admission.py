# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.backends.h100.generic_codegen_admission import (
    UnsupportedContractLoweringError,
    require_generic_codegen_admission,
)
from tile_lifetime.canonical_kernel_composition import (
    lower_canonical_kernel_composition,
)
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.sm90_contract_binding import sm90_contract_component_binding
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    scalar_binary,
    scalar_input,
)


@pytest.mark.parametrize(
    "component_id,phase_id,ordinals,contracts,reductions",
    [
        (17, None, (50, 53, 54, 55, 57), (57,), ((4096,),)),
        (23, 0, (65, 67, 68, 72, 74), (65, 67), ((14336,), (14336,))),
        (19, None, (56,), (56,), ((1, 2048),)),
    ],
)
def test_full_dense_unlowered_contracts_report_exact_source_boundary(
    component_id, phase_id, ordinals, contracts, reductions
):
    program = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    with pytest.raises(UnsupportedContractLoweringError) as raised:
        require_generic_codegen_admission(
            ((ordinal, program.operations[ordinal]) for ordinal in ordinals),
            component_id=component_id,
            phase_id=phase_id,
        )
    error = raised.value
    assert error.component_id == component_id
    assert error.phase_id == phase_id
    assert tuple(ordinal for ordinal, _ in error.contracts) == contracts
    assert (
        tuple(tuple(axis.extent for axis in operation.reduction_axes) for _, operation in error.contracts) == reductions
    )
    assert all(operation is program.operations[ordinal] for ordinal, operation in error.contracts)


@pytest.mark.parametrize("reduction", (1, 8, 14336))
def test_contract_rejection_has_no_small_shape_exception(reduction):
    row, inner, column = TensorAxis(0, 2), TensorAxis(1, reduction), TensorAxis(2, 16)
    left = ProgramValue("left", (row, inner), DType.BF16)
    right = ProgramValue("right", (inner, column), DType.BF16)
    product = ProgramValue("product", (row, column), DType.BF16)
    contract = ContractPrimitive("matmul", (left, right), product, (inner,), DType.FP32)
    with pytest.raises(UnsupportedContractLoweringError) as raised:
        require_generic_codegen_admission(((7, contract),))
    assert raised.value.contracts == ((7, contract),)


def test_admitted_physical_contract_keeps_generic_maps_and_reference_execution():
    row, inner, column = TensorAxis(0, 2), TensorAxis(1, 8), TensorAxis(2, 16)
    left = ProgramValue("left", (row, inner), DType.BF16)
    right = ProgramValue("right", (inner, column), DType.BF16)
    product = ProgramValue("product", (row, column), DType.BF16)
    squared = ProgramValue("squared", product.axes, DType.BF16)
    program = TensorProgram(
        (left, right),
        (
            ContractPrimitive("matmul", (left, right), product, (inner,), DType.FP32),
            MapPrimitive(
                "square",
                (product,),
                squared,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(product.name), scalar_input(product.name)),
            ),
        ),
        (squared,),
        NumericalPolicy.FAST,
    )
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    composition = lower_canonical_kernel_composition(request)
    assert tuple(component.operation_ordinals for component in composition.components) == ((0,), (1,))
    assert sm90_contract_component_binding(program, composition.components[0]) is not None
    # Only the Map residue reaches generic code generation after physical admission.
    require_generic_codegen_admission(((1, program.operations[1]),))
    left_array = np.arange(16, dtype=np.float32).reshape(left.shape) / 8
    right_array = np.arange(128, dtype=np.float32).reshape(right.shape) / 16
    actual = execute_tensor_program(program, {left.name: left_array, right.name: right_array})[squared.name]
    product_array = np.asarray(left_array @ right_array, dtype=jnp.bfloat16)
    expected = np.asarray(np.square(product_array.astype(np.float32)), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(actual, expected)
