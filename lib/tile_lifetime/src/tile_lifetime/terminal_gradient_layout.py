# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact terminal gradient Relations bound to dense physical output storage."""

import math
from dataclasses import dataclass

from tile_lifetime.layout_relation_canonicalization import (
    BijectiveLayoutRelation,
    LayoutRelationKind,
    bijective_layout_relation,
)
from tile_lifetime.layout_transport import canonical_layout_transport_program
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    NumericalPolicy,
    ProgramValue,
    RelationPrimitive,
    TensorAxis,
    TensorProgram,
    primitive_inputs,
)


@dataclass(frozen=True)
class DenseOutputView:
    """Canonical result strides over one unchanged, zero-offset allocation."""

    output: ProgramValue
    source_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    result_strides: tuple[int, ...]

    def __post_init__(self) -> None:
        if math.prod(self.source_shape) != math.prod(self.output.shape):
            raise ValueError("output view must preserve the complete storage range")
        # This existing transport proof rejects overlap, padding and negative
        # nonunit strides. Unit strides are immaterial to element addresses.
        self.transport_program()

    @property
    def contiguous(self) -> bool:
        stride = 1
        for extent, actual in reversed(tuple(zip(self.output.shape, self.result_strides, strict=True))):
            if extent != 1 and actual != stride:
                return False
            stride *= extent
        return True

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "output": self.output.name,
            "dtype": self.output.dtype.value,
            "result_axes": tuple(axis.id for axis in self.output.axes),
            "result_shape": self.output.shape,
            "result_strides": self.result_strides,
            "source_shape": self.source_shape,
            "source_strides": self.source_strides,
            "storage_offset": 0,
        }

    def validate_storage(
        self, shape: tuple[int, ...], strides: tuple[int, ...], storage_offset: int, dtype: DType
    ) -> None:
        """Require the exact allocation-relative view used at compilation."""
        if (shape, strides, storage_offset, dtype) != (self.source_shape, self.source_strides, 0, self.output.dtype):
            raise ValueError("physical output storage differs from the bound layout")

    def transport_program(self) -> TensorProgram:
        """Return the explicit copy Relation for these same canonical strides."""
        return canonical_layout_transport_program(self.output, self.output.axes, self.result_strides)


@dataclass(frozen=True)
class TerminalGradientLayout:
    """One admitted Contract's unique, scalar-free route to a live output."""

    source_fingerprint: str
    terminal_contract_ordinal: int
    source: ProgramValue
    output: ProgramValue
    operation_ordinals: tuple[int, ...]
    steps: tuple[BijectiveLayoutRelation | FoldPrimitive, ...]

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "terminal_contract_ordinal": self.terminal_contract_ordinal,
            "operation_ordinals": self.operation_ordinals,
            "source": self.source.name,
            "output": self.output.name,
            "steps": tuple(
                step.kind.value if isinstance(step, BijectiveLayoutRelation) else "singleton_sum" for step in self.steps
            ),
        }

    def bind_storage(
        self, physical_axes: tuple[TensorAxis, ...], shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> DenseOutputView:
        """Compose the proved Relations with the actual dense physical layout."""
        view = bind_dense_output_view(self.source, physical_axes, shape, strides)
        current = self.source
        result_strides = view.result_strides
        for step in self.steps:
            if isinstance(step, FoldPrimitive):
                result_strides = tuple(result_strides[current.axes.index(axis)] for axis in step.output.axes)
                current = step.output
                continue
            result = step.relation.outputs[0]
            if step.kind is LayoutRelationKind.ROW_MAJOR_RESHAPE:
                result_strides = _reshape_strides(current.shape, result_strides, result.shape)
            else:
                mapped = [1] * len(result.axes)
                for source_dimension, result_dimension in enumerate(step.source_to_result_dimensions):
                    if result_dimension is not None:
                        mapped[result_dimension] = result_strides[source_dimension]
                result_strides = tuple(mapped)
            current = result
        return DenseOutputView(self.output, shape, strides, result_strides)


def terminal_gradient_layout(
    program: TensorProgram,
    operation_ordinals: tuple[int, ...],
    terminal_contract_ordinal: int,
    primal: ProgramValue,
    live_outputs: tuple[ProgramValue, ...],
) -> TerminalGradientLayout:
    """Bind an admitted terminal Contract to one exact live-output view chain."""
    owned = set(operation_ordinals)
    terminal = program.operations[terminal_contract_ordinal]
    if terminal_contract_ordinal not in owned or not isinstance(terminal, ContractPrimitive):
        raise ValueError("terminal gradient requires an owned Contract")
    source = terminal.output
    if source.axes != primal.axes or source.dtype is not primal.dtype:
        raise ValueError("terminal gradient must preserve its admitted primal axes and dtype")
    consumers: dict[ProgramValue, list[int]] = {}
    for ordinal in operation_ordinals:
        for value in primitive_inputs(program.operations[ordinal]):
            consumers.setdefault(value, []).append(ordinal)
    value = source
    ordinals: list[int] = []
    steps: list[BijectiveLayoutRelation | FoldPrimitive] = []
    while value not in live_outputs:
        following = consumers.get(value, ())
        if len(following) != 1 or following[0] in ordinals:
            raise ValueError("terminal gradient must have one unambiguous live-output path")
        ordinal = following[0]
        operation = program.operations[ordinal]
        if isinstance(operation, RelationPrimitive):
            proof = bijective_layout_relation(operation)
            if operation.inputs != (value,):
                raise ValueError("terminal gradient Relation must consume the selected value")
            steps.append(proof)
            value = operation.outputs[0]
        elif isinstance(operation, FoldPrimitive) and (
            program.numerical_policy is NumericalPolicy.FAST
            and operation.reducer is FoldReducer.SUM
            and operation.order is EvaluationOrder.ORDER_FREE
            and operation.accumulation_dtype is operation.input.dtype is operation.output.dtype
            and all(axis.extent == 1 for axis in operation.reduction_axes)
        ):
            # Existing gradient capabilities admit this unit-domain adjoint.
            # Nonunit group sums stay in the source program after the live cut.
            steps.append(operation)
            value = operation.output
        else:
            raise ValueError("terminal gradient path requires a bijective Relation or same-dtype singleton SUM")
        ordinals.append(ordinal)
    if consumers.get(value):
        raise ValueError("terminal gradient live output also has an owned continuation")
    return TerminalGradientLayout(
        program.semantic_fingerprint, terminal_contract_ordinal, source, value, tuple(ordinals), tuple(steps)
    )


def bind_dense_output_view(
    output: ProgramValue,
    physical_axes: tuple[TensorAxis, ...],
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> DenseOutputView:
    """Expand packed storage and restore the declared logical axis order."""
    if len(physical_axes) != len(output.axes) or set(physical_axes) != set(output.axes):
        raise ValueError("physical output view must cover every logical axis exactly")
    expanded = _reshape_strides(shape, strides, tuple(axis.extent for axis in physical_axes))
    result = tuple(expanded[physical_axes.index(axis)] for axis in output.axes)
    return DenseOutputView(output, shape, strides, result)


def _reshape_strides(shape: tuple[int, ...], strides: tuple[int, ...], result_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != len(strides) or math.prod(shape) != math.prod(result_shape):
        raise ValueError("output reshape must preserve the complete storage range")
    if any(extent <= 0 for extent in (*shape, *result_shape)) or any(stride <= 0 for stride in strides):
        raise ValueError("output reshape requires positive extents and strides")
    # An order-preserving view may split/merge a contiguous run, but must never
    # cross a stride discontinuity. Ignore unit axes, whose coordinates are zero.
    chunks: list[tuple[int, int]] = []
    for extent, stride in reversed(tuple(zip(shape, strides, strict=True))):
        if extent == 1:
            continue
        if chunks and stride == chunks[-1][0] * chunks[-1][1]:
            size, base = chunks[-1]
            chunks[-1] = (size * extent, base)
        else:
            chunks.append((extent, stride))
    result = [1] * len(result_shape)
    dimension = len(result_shape) - 1
    for size, base in chunks:
        consumed = 1
        while dimension >= 0 and (consumed < size or result_shape[dimension] == 1):
            result[dimension] = consumed * base
            consumed *= result_shape[dimension]
            dimension -= 1
        if consumed != size:
            raise ValueError("output reshape would require a hidden copy across strided axes")
    if any(extent != 1 for extent in result_shape[: dimension + 1]):
        raise ValueError("output reshape does not cover every source element")
    return tuple(result)
