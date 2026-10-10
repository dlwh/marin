# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Relations from dense strided physical storage to canonical tensors."""

import math

from tile_lifetime.tensor_program import (
    IndexExpressionKind,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    TensorAxis,
    TensorProgram,
    index_axis,
    index_binary,
    index_constant,
)


def canonical_layout_transport_program(
    output: ProgramValue, physical_axes: tuple[TensorAxis, ...], physical_strides: tuple[int, ...]
) -> TensorProgram:
    """Copy dense physical storage into the output's canonical axis order."""
    if len(physical_axes) != len(output.axes) or set(physical_axes) != set(output.axes):
        raise ValueError("layout transport requires the same canonical and physical axes")
    ordered = sorted(zip(physical_axes, physical_strides, strict=True), key=lambda item: item[1])
    expected_stride = 1
    for axis, stride in ordered:
        if axis.extent == 1:
            continue
        if stride != expected_stride:
            raise ValueError("layout transport requires dense nonoverlapping physical storage")
        expected_stride *= axis.extent
    storage_axis = TensorAxis(max((axis.id for axis in output.axes), default=-1) + 1, math.prod(output.shape), "storage")
    storage = ProgramValue("physical_storage", (storage_axis,), output.dtype)
    linear = index_constant(0)
    for axis, stride in zip(physical_axes, physical_strides, strict=True):
        linear = index_binary(
            IndexExpressionKind.ADD,
            linear,
            index_binary(IndexExpressionKind.MULTIPLY, index_axis(axis), index_constant(stride)),
        )
    relation = RelationPrimitive(
        name="canonical_layout_transport",
        inputs=(storage,),
        outputs=(output,),
        domain_axes=output.axes,
        input_indexing=(OperandIndexing(storage, output.axes, (linear,)),),
    )
    return TensorProgram((storage,), (relation,), (output,), NumericalPolicy.FAST)
