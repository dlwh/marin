# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU structural admission for contiguous SM90 Map-load/Contract components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.backends.h100.cute_map_load_contract import (
    GeneratedSm90MapLoadContract,
    generate_sm90_map_load_contract,
)
from tile_lifetime.canonical_kernel_composition import CanonicalKernelComponent
from tile_lifetime.sm90_fold_map_contract import _fused_scalar_expression, _scalar_leaves
from tile_lifetime.tensor_program import (
    ContractOperandPrecision,
    ContractPrimitive,
    DType,
    EvaluationOrder,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpression,
    TensorAxis,
    TensorProgram,
    primitive_outputs,
)


class MatrixStorageOrder(StrEnum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"


@dataclass(frozen=True)
class Sm90MatrixView:
    """A zero-offset matrix interpretation of canonical contiguous source storage."""

    source_shape: tuple[int, ...]
    shape: tuple[int, int]
    order: MatrixStorageOrder

    @property
    def storage_shape(self) -> tuple[int, int]:
        return self.shape if self.order is MatrixStorageOrder.ROW_MAJOR else self.shape[::-1]

    @property
    def strides(self) -> tuple[int, int]:
        return (self.shape[1], 1) if self.order is MatrixStorageOrder.ROW_MAJOR else (1, self.shape[0])

    @property
    def canonical_record(self) -> dict[str, object]:
        return {"source_shape": self.source_shape, "shape": self.shape, "strides": self.strides, "offset": 0}

    def validate_source_storage(self, shape: tuple[int, ...], strides: tuple[int, ...], offset: int) -> None:
        """Reject layouts that cannot share the proved matrix's allocation."""
        if shape != self.source_shape or len(strides) != len(shape) or offset != 0:
            raise ValueError("SM90 matrix view requires its exact source shape and zero storage offset")
        if math.prod(shape) != math.prod(self.shape) or any(
            extent > 1 and stride != math.prod(shape[index + 1 :])
            for index, (extent, stride) in enumerate(zip(shape, strides, strict=True))
        ):
            raise ValueError("SM90 matrix view requires canonical contiguous source storage")


@dataclass(frozen=True)
class Sm90ContractComponentBinding:
    """One BF16 Contract with proved matrix views for the QuACK WGMMA mainloop."""

    component_id: int
    operation_ordinals: tuple[int, ...]
    activation: ProgramValue
    scalar_leaves: tuple[ProgramValue, ...]
    weight: ProgramValue
    output: ProgramValue
    rows: int
    reduction: int
    features: int
    expression: ScalarExpression
    generated: GeneratedSm90MapLoadContract
    activation_view: Sm90MatrixView
    weight_view: Sm90MatrixView
    output_view: Sm90MatrixView


def sm90_contract_component_binding(
    program: TensorProgram,
    component: CanonicalKernelComponent,
) -> Sm90ContractComponentBinding | None:
    """Return the WGMMA binding when generic axes, layout, and dtypes permit it."""
    operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
    contracts = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))
    if len(contracts) != 1 or any(
        not isinstance(operation, (MapPrimitive, ContractPrimitive)) for operation in operations
    ):
        return None
    contract = contracts[0]
    if operations[-1] is not contract or len(contract.inputs) != 2:
        return None
    if (
        program.numerical_policy is not NumericalPolicy.FAST
        or contract.accumulation_dtype not in {DType.BF16, DType.FP32}
        or contract.output.dtype is not DType.BF16
        or contract.order is not EvaluationOrder.ORDER_FREE
        or not contract.reduction_axes
        or any(precision is not ContractOperandPrecision.DEFAULT for precision in contract.operand_precision)
    ):
        return None
    domains = _matrix_domains(contract)
    activation_index = 0
    activation_order = weight_order = MatrixStorageOrder.ROW_MAJOR
    if domains is None:
        # Producer Maps are proved only for the original MK/KN interpretation.
        # Newly oriented operands use an identity transform on one Contract.
        if len(operations) != 1:
            return None
        oriented = _oriented_matrix_domains(contract)
        if oriented is None:
            return None
        domains, activation_index, activation_order, weight_order = oriented
    left, weight = contract.inputs[activation_index], contract.inputs[1 - activation_index]
    if weight.dtype is not DType.BF16:
        return None
    row_domains, reduction_domains, feature_domains = domains
    producer_by_value = {value: operation for operation in operations for value in primitive_outputs(operation)}
    allowed = set(operations[:-1])
    leaves = tuple(sorted(_scalar_leaves(left, producer_by_value, allowed), key=lambda value: value.name))
    full_shape_leaves = tuple(value for value in leaves if value.axes == left.axes and value.dtype is DType.BF16)
    if len(full_shape_leaves) != 1:
        return None
    activation = full_shape_leaves[0]

    row_operand_axes = left.axes[: len(row_domains)]
    reduction_operand_axes = left.axes[len(row_domains) :]
    if any(value is not activation and value.axes not in {row_operand_axes, reduction_operand_axes} for value in leaves):
        return None

    axis_ids = {
        axis.id
        for value in (
            *program.inputs,
            *(value for operation in program.operations for value in primitive_outputs(operation)),
        )
        for axis in value.axes
    }
    row_axis = TensorAxis(max(axis_ids, default=-1) + 1, math.prod(axis.extent for axis in row_domains), "wgmma_row")
    reduction_axis = TensorAxis(row_axis.id + 1, math.prod(axis.extent for axis in reduction_domains), "wgmma_reduction")
    feature_axis = TensorAxis(row_axis.id + 2, math.prod(axis.extent for axis in feature_domains), "wgmma_feature")
    if reduction_axis.extent % 8 or feature_axis.extent % 8:
        return None
    if activation_order is MatrixStorageOrder.COLUMN_MAJOR and row_axis.extent % 8:
        return None

    proxy_by_value = {
        value: ProgramValue(
            value.name,
            (
                (row_axis, reduction_axis)
                if value is activation
                else (row_axis,) if value.axes == row_operand_axes else (reduction_axis,)
            ),
            value.dtype,
        )
        for value in leaves
    }
    proxy_activation = proxy_by_value[activation]
    proxy_weight = ProgramValue(weight.name, (reduction_axis, feature_axis), weight.dtype)
    proxy_output = ProgramValue(contract.output.name, (row_axis, feature_axis), contract.output.dtype)
    expression = _fused_scalar_expression(left, producer_by_value, allowed)
    generated = generate_sm90_map_load_contract(
        expression=expression,
        activation=proxy_activation,
        leaves=tuple(proxy_by_value[value] for value in leaves),
        row_axis=row_axis,
        reduction_axis=reduction_axis,
        weight_feature_axis=feature_axis,
        output_row_axis=row_axis,
        output_feature_axis=feature_axis,
        weight=proxy_weight,
        output=proxy_output,
    )
    if not generated.executable_with_current_quack_adapter:
        return None
    return Sm90ContractComponentBinding(
        component_id=component.id,
        operation_ordinals=component.operation_ordinals,
        activation=activation,
        scalar_leaves=leaves,
        weight=weight,
        output=contract.output,
        rows=row_axis.extent,
        reduction=reduction_axis.extent,
        features=feature_axis.extent,
        expression=expression,
        generated=generated,
        activation_view=Sm90MatrixView(activation.shape, (row_axis.extent, reduction_axis.extent), activation_order),
        weight_view=Sm90MatrixView(weight.shape, (reduction_axis.extent, feature_axis.extent), weight_order),
        output_view=Sm90MatrixView(
            contract.output.shape, (row_axis.extent, feature_axis.extent), MatrixStorageOrder.ROW_MAJOR
        ),
    )


def _oriented_matrix_domains(
    contract: ContractPrimitive,
) -> (
    tuple[
        tuple[tuple[TensorAxis, ...], tuple[TensorAxis, ...], tuple[TensorAxis, ...]],
        int,
        MatrixStorageOrder,
        MatrixStorageOrder,
    ]
    | None
):
    domains = tuple(_operand_domains(contract, index) for index in range(2))
    if any(domain is None for domain in domains):
        return None
    # Unit axes have no address variation. All other axes must remain in their
    # source block order, including the ordered multi-axis K flattening.
    operands = tuple(tuple(axis for axis in domain if axis.extent != 1) for domain in domains if domain is not None)
    output = tuple(axis for axis in contract.output.axes if axis.extent != 1)
    reduction = tuple(axis for axis in contract.reduction_axes if axis.extent != 1)
    for activation_index in range(2):
        left, right = operands[activation_index], operands[1 - activation_index]
        rows = tuple(axis for axis in output if axis in left and axis not in right)
        features = tuple(axis for axis in output if axis in right and axis not in left)
        if not rows or not features or output != (*rows, *features):
            continue
        left_order = _matrix_storage_order(left, rows, reduction)
        right_order = _matrix_storage_order(right, reduction, features)
        if left_order is not None and right_order is not None:
            return (rows, reduction, features), activation_index, left_order, right_order
    return None


def _matrix_storage_order(
    domains: tuple[TensorAxis, ...], rows: tuple[TensorAxis, ...], columns: tuple[TensorAxis, ...]
) -> MatrixStorageOrder | None:
    if domains == (*rows, *columns):
        return MatrixStorageOrder.ROW_MAJOR
    if domains == (*columns, *rows):
        return MatrixStorageOrder.COLUMN_MAJOR
    return None


def _matrix_domains(
    contract: ContractPrimitive,
) -> tuple[tuple[TensorAxis, ...], tuple[TensorAxis, ...], tuple[TensorAxis, ...]] | None:
    left_domains = _operand_domains(contract, 0)
    right_domains = _operand_domains(contract, 1)
    if left_domains is None or right_domains is None:
        return None
    reduction = contract.reduction_axes
    rows = tuple(axis for axis in contract.output.axes if axis in left_domains and axis not in right_domains)
    features = tuple(axis for axis in contract.output.axes if axis in right_domains and axis not in left_domains)
    if not rows or not features:
        return None
    if contract.output.axes != (*rows, *features):
        return None
    if left_domains != (*rows, *reduction) or right_domains != (*reduction, *features):
        return None
    return rows, reduction, features


def _operand_domains(contract: ContractPrimitive, input_index: int) -> tuple[TensorAxis, ...] | None:
    operand = contract.inputs[input_index]
    mappings = {mapping.operand_axis: mapping for mapping in contract.index_maps_for_input(input_index)}
    domains = []
    for axis in operand.axes:
        mapping = mappings.get(axis)
        if mapping is None:
            if axis not in (*contract.output.axes, *contract.reduction_axes):
                return None
            domains.append(axis)
            continue
        if mapping.divisor != 1 or mapping.offset != 0 or mapping.modulus is not None:
            return None
        if mapping.domain_axis.extent != mapping.operand_axis.extent:
            return None
        domains.append(mapping.domain_axis)
    if len(domains) != len(set(domains)):
        return None
    return tuple(domains)
