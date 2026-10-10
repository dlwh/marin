# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic typed conversion from a JAX-owned JAXpr to Shuttle tensor algebra."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.extend.core import ClosedJaxpr, DropVar, Literal, Var

from shuttle.ir import DType
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractOperandPrecision,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    ScanDirection,
    ScanInputBinding,
    ScanInvariantBinding,
    ScanOutputBinding,
    ScanPrimitive,
    ScanStateBinding,
    SelectionComparison,
    SelectionOrder,
    SelectionPrimitive,
    SelectionTieBreak,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    index_tensor_element,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_select,
    scalar_unary,
)


class JaxprConversionError(ValueError):
    """Precise rejection at one unsupported JAXpr equation or boundary value."""

    def __init__(
        self,
        primitive: str,
        equation: int | None,
        reason: str,
        *,
        equation_path: tuple[int, ...] | None = None,
    ):
        self.primitive = primitive
        self.equation = equation
        self.equation_path = equation_path
        self.reason = reason
        if equation_path is not None:
            location = "equation " + " -> ".join(str(index) for index in equation_path)
        else:
            location = "boundary" if equation is None else f"equation {equation}"
        super().__init__(f"JAXpr {location} primitive {primitive!r}: {reason}")


@dataclass(frozen=True)
class _PrimitiveReference:
    name: str


@dataclass(frozen=True)
class _FlatEquation:
    primitive: _PrimitiveReference
    invars: tuple[Var | Literal, ...]
    outvars: tuple[Var | DropVar, ...]
    params: dict[str, Any]
    path: tuple[int, ...]


@dataclass(frozen=True)
class _FlatJaxpr:
    invars: tuple[Var, ...]
    outvars: tuple[Var | Literal, ...]
    eqns: tuple[_FlatEquation, ...]


@dataclass(frozen=True)
class _FlatClosedJaxpr:
    jaxpr: _FlatJaxpr
    consts: tuple[()] = ()


@dataclass
class _DimensionClasses:
    parent: dict[tuple[Var, int], tuple[Var, int]]
    extent: dict[tuple[Var, int], int]

    def find(self, node: tuple[Var, int]) -> tuple[Var, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[Var, int], right: tuple[Var, int], *, equation: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.extent[left_root] != self.extent[right_root]:
            raise JaxprConversionError(
                "shape_relation",
                equation,
                f"cannot identify dimensions of extents {self.extent[left_root]} and {self.extent[right_root]}",
            )
        self.parent[right_root] = left_root


_ELEMENTWISE_BINARY = {
    "add": ScalarExpressionKind.ADD,
    "add_any": ScalarExpressionKind.ADD,
    "sub": ScalarExpressionKind.SUBTRACT,
    "mul": ScalarExpressionKind.MULTIPLY,
    "div": ScalarExpressionKind.DIVIDE,
    "max": ScalarExpressionKind.MAXIMUM,
    "min": ScalarExpressionKind.MINIMUM,
    "lt": ScalarExpressionKind.LESS,
    "le": ScalarExpressionKind.LESS_EQUAL,
    "gt": ScalarExpressionKind.GREATER,
    "ge": ScalarExpressionKind.GREATER_EQUAL,
    "eq": ScalarExpressionKind.EQUAL,
    "ne": ScalarExpressionKind.NOT_EQUAL,
}

_ELEMENTWISE_UNARY = {
    "neg": ScalarExpressionKind.NEGATE,
    "exp": ScalarExpressionKind.EXP,
    "log": ScalarExpressionKind.LOG,
    "rsqrt": ScalarExpressionKind.RSQRT,
    "tanh": ScalarExpressionKind.TANH,
}

_VIEW_PRIMITIVES = frozenset(
    {"broadcast_in_dim", "identity_alias", "reshape", "slice", "squeeze", "stop_gradient", "transpose"}
)
_FOLD_PRIMITIVES = frozenset({"reduce_sum", "reduce_max"})
_SUPPORTED_PRIMITIVES = frozenset(
    {
        *_ELEMENTWISE_BINARY,
        *_ELEMENTWISE_UNARY,
        *_VIEW_PRIMITIVES,
        *_FOLD_PRIMITIVES,
        "convert_element_type",
        "reduce_precision",
        "select_n",
        "dot_general",
        "gather",
        "iota",
        "integer_pow",
        "logistic",
        "scatter-add",
        "scan",
        "top_k",
    }
)


def jax_function_to_tensor_program(
    function: Callable[..., Any],
    *arguments: Any,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> TensorProgram:
    """Trace an ordinary JAX callable, then convert its already-differentiated JAXpr.

    Callers form ``jax.vjp`` or ``jax.value_and_grad`` before passing the
    callable here. This function invokes no differentiation or custom rule.
    """
    return jaxpr_to_tensor_program(jax.make_jaxpr(function)(*arguments), numerical_policy=numerical_policy)


def jaxpr_to_tensor_program(
    closed_jaxpr: ClosedJaxpr,
    *,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> TensorProgram:
    """Convert a closed, first-order JAXpr into generic Shuttle algebra."""
    flattened = _flatten_closed_jaxpr(closed_jaxpr)
    equations = flattened.jaxpr.eqns
    for equation in equations:
        if equation.primitive.name not in _SUPPORTED_PRIMITIVES:
            raise JaxprConversionError(
                equation.primitive.name,
                equation.path[0],
                "no generic Shuttle algebra conversion is registered",
                equation_path=equation.path,
            )

    dimensions = _dimension_classes(flattened)
    axes = _logical_axes(flattened, dimensions)
    values: dict[Var, ProgramValue] = {}
    inputs = tuple(
        _program_value(variable, axes, name=f"argument.{index}") for index, variable in enumerate(flattened.jaxpr.invars)
    )
    values.update(zip(flattened.jaxpr.invars, inputs, strict=True))
    operations: list[TensorPrimitive] = []
    temporary_index = 0

    for equation_index, equation in enumerate(equations):
        primitive = equation.primitive.name
        output_variables = tuple(variable for variable in equation.outvars if not isinstance(variable, DropVar))
        if len(output_variables) != len(equation.outvars):
            raise JaxprConversionError(primitive, equation_index, "discarded JAXpr results are not supported")
        outputs = tuple(
            _program_value(variable, axes, name=f"equation.{equation_index}.result.{result_index}")
            for result_index, variable in enumerate(output_variables)
        )
        if primitive == "scan":
            operations.append(
                _scan_operation(
                    equation,
                    equation_index=equation_index,
                    inputs=tuple(_required_value(atom, values, primitive, equation_index) for atom in equation.invars),
                    outputs=outputs,
                    numerical_policy=numerical_policy,
                )
            )
            values.update(zip(output_variables, outputs, strict=True))
            continue
        if primitive == "top_k":
            if len(outputs) != 2:
                raise JaxprConversionError(primitive, equation_index, "top_k requires value and index results")
            input_atom = equation.invars[0]
            if isinstance(input_atom, Literal):
                raise JaxprConversionError(primitive, equation_index, "top_k requires a tensor operand")
            operations.append(
                _selection_operation(
                    equation,
                    equation_index=equation_index,
                    input_value=values[input_atom],
                    outputs=(outputs[0], outputs[1]),
                )
            )
            values.update(zip(output_variables, outputs, strict=True))
            continue
        if len(outputs) != 1:
            raise JaxprConversionError(primitive, equation_index, "the initial converter requires one result")
        output = outputs[0]

        if (
            primitive in _ELEMENTWISE_BINARY
            or primitive in _ELEMENTWISE_UNARY
            or primitive in {"integer_pow", "logistic"}
            or primitive
            in {
                "convert_element_type",
                "reduce_precision",
                "select_n",
            }
        ):
            operation, added_relations, temporary_index = _elementwise_operation(
                equation,
                equation_index=equation_index,
                output=output,
                values=values,
                temporary_index=temporary_index,
            )
            operations.extend(added_relations)
            operations.append(operation)
        elif primitive == "gather":
            operations.append(
                _gather_operation(
                    equation,
                    equation_index=equation_index,
                    inputs=(
                        _required_value(equation.invars[0], values, primitive, equation_index),
                        _required_value(equation.invars[1], values, primitive, equation_index),
                    ),
                    output=output,
                )
            )
        elif primitive == "scatter-add":
            operations.extend(
                _scatter_add_operations(
                    equation,
                    equation_index=equation_index,
                    inputs=tuple(_required_value(atom, values, primitive, equation_index) for atom in equation.invars),
                    output=output,
                )
            )
        elif primitive in _VIEW_PRIMITIVES:
            input_atom = equation.invars[0]
            if isinstance(input_atom, Literal):
                operations.append(
                    MapPrimitive(
                        f"equation.{equation_index}.{primitive}",
                        (),
                        output,
                        _literal_expression(input_atom, primitive, equation_index),
                    )
                )
            else:
                operations.append(
                    _view_operation(
                        equation,
                        equation_index=equation_index,
                        input_value=values[input_atom],
                        output=output,
                    )
                )
        elif primitive in _FOLD_PRIMITIVES:
            operations.append(
                _fold_operation(
                    equation,
                    equation_index=equation_index,
                    input_value=_required_value(equation.invars[0], values, primitive, equation_index),
                    output=output,
                    numerical_policy=numerical_policy,
                )
            )
        elif primitive == "dot_general":
            _contract_operand_precision(equation.params.get("precision"), equation_index)
            if any(isinstance(atom, Literal) for atom in equation.invars):
                scalar_dot, added_relations, temporary_index = _scalar_dot_operation(
                    equation,
                    equation_index=equation_index,
                    output=output,
                    values=values,
                    temporary_index=temporary_index,
                )
                operations.extend(added_relations)
                operations.append(scalar_dot)
            else:
                operations.append(
                    _contract_operation(
                        equation,
                        equation_index=equation_index,
                        inputs=(
                            _required_value(equation.invars[0], values, primitive, equation_index),
                            _required_value(equation.invars[1], values, primitive, equation_index),
                        ),
                        output=output,
                    )
                )
        else:
            assert primitive == "iota"
            operations.append(_iota_operation(equation, equation_index=equation_index, output=output))
        values[output_variables[0]] = output

    program_outputs: list[ProgramValue] = []
    for output in flattened.jaxpr.outvars:
        if isinstance(output, Literal):
            raise JaxprConversionError("literal", None, "literal program results are not supported")
        program_outputs.append(values[output])
    return TensorProgram(inputs, tuple(operations), tuple(program_outputs), numerical_policy)


def _required_value(
    atom: Var | Literal,
    values: dict[Var, ProgramValue],
    primitive: str,
    equation: int,
) -> ProgramValue:
    if isinstance(atom, Literal):
        raise JaxprConversionError(primitive, equation, "tensor operand cannot be a scalar literal")
    return values[atom]


def _flatten_closed_jaxpr(closed_jaxpr: ClosedJaxpr) -> _FlatClosedJaxpr:
    """Inline structurally nested single-call JAXprs without interpreting their names."""
    equations: list[_FlatEquation] = []

    def inline(
        nested: ClosedJaxpr,
        actual_inputs: tuple[Var | Literal, ...],
        result_targets: tuple[Var | DropVar, ...],
        path: tuple[int, ...],
    ) -> None:
        if nested.consts or nested.jaxpr.constvars:
            outer = path[0] if path else None
            raise JaxprConversionError(
                "constvar",
                outer,
                "captured array constants are not supported",
                equation_path=path or None,
            )
        if len(nested.jaxpr.invars) != len(actual_inputs) or len(nested.jaxpr.outvars) != len(result_targets):
            outer = path[0] if path else None
            raise JaxprConversionError(
                "call_jaxpr",
                outer,
                "nested JAXpr boundary arity does not match its call equation",
                equation_path=path or None,
            )
        substitutions: dict[Var, Var | Literal | DropVar] = dict(zip(nested.jaxpr.invars, actual_inputs, strict=True))
        terminal_targets = dict(zip(nested.jaxpr.outvars, result_targets, strict=True))
        for nested_index, equation in enumerate(nested.jaxpr.eqns):
            equation_path = (*path, nested_index)
            equation_inputs = tuple(
                atom if isinstance(atom, Literal) else substitutions.get(atom, atom) for atom in equation.invars
            )
            equation_outputs = tuple(terminal_targets.get(atom, atom) for atom in equation.outvars)
            substitutions.update(zip(equation.outvars, equation_outputs, strict=True))
            called = equation.params.get("jaxpr")
            if isinstance(called, ClosedJaxpr) and equation.primitive.name != "scan":
                inline(called, equation_inputs, equation_outputs, equation_path)
                continue
            equations.append(
                _FlatEquation(
                    _PrimitiveReference(equation.primitive.name),
                    equation_inputs,
                    equation_outputs,
                    dict(equation.params),
                    equation_path,
                )
            )
        for nested_output, result_target in zip(nested.jaxpr.outvars, result_targets, strict=True):
            source = (
                nested_output if isinstance(nested_output, Literal) else substitutions.get(nested_output, nested_output)
            )
            if source == result_target:
                continue
            equations.append(
                _FlatEquation(
                    _PrimitiveReference("identity_alias"),
                    (source,),
                    (result_target,),
                    {},
                    (*path, len(nested.jaxpr.eqns)),
                )
            )

    inline(
        closed_jaxpr,
        tuple(closed_jaxpr.jaxpr.invars),
        tuple(closed_jaxpr.jaxpr.outvars),
        (),
    )
    return _FlatClosedJaxpr(
        _FlatJaxpr(tuple(closed_jaxpr.jaxpr.invars), tuple(closed_jaxpr.jaxpr.outvars), tuple(equations))
    )


def _dimension_classes(closed_jaxpr: _FlatClosedJaxpr) -> _DimensionClasses:
    variables = (
        *closed_jaxpr.jaxpr.invars,
        *(
            variable
            for equation in closed_jaxpr.jaxpr.eqns
            for variable in equation.outvars
            if isinstance(variable, Var)
        ),
    )
    parent: dict[tuple[Var, int], tuple[Var, int]] = {}
    extent: dict[tuple[Var, int], int] = {}
    for variable in variables:
        for dimension, size in enumerate(_shape(variable)):
            node = (variable, dimension)
            parent[node] = node
            extent[node] = size
    classes = _DimensionClasses(parent, extent)
    for equation_index, equation in enumerate(closed_jaxpr.jaxpr.eqns):
        output_variables = tuple(variable for variable in equation.outvars if isinstance(variable, Var))
        if equation.primitive.name == "scan":
            _union_scan_dimensions(classes, equation, output_variables, equation=equation_index)
            continue
        if equation.primitive.name == "top_k" and len(output_variables) == 2:
            input_variables = tuple(variable for variable in equation.invars if isinstance(variable, Var))
            if len(input_variables) != 1:
                continue
            axis = equation.params.get("axis")
            if not isinstance(axis, int):
                continue
            input_variable = input_variables[0]
            for output in output_variables:
                for dimension in range(len(_shape(input_variable))):
                    if dimension == axis:
                        continue
                    classes.union((input_variable, dimension), (output, dimension), equation=equation_index)
            for dimension in range(len(_shape(output_variables[0]))):
                classes.union(
                    (output_variables[0], dimension),
                    (output_variables[1], dimension),
                    equation=equation_index,
                )
            continue
        if len(output_variables) != 1:
            continue
        output = output_variables[0]
        primitive = equation.primitive.name
        variable_inputs = tuple(variable for variable in equation.invars if isinstance(variable, Var))
        if (
            primitive in _ELEMENTWISE_BINARY
            or primitive in _ELEMENTWISE_UNARY
            or primitive in {"integer_pow", "logistic"}
            or primitive
            in {
                "convert_element_type",
                "reduce_precision",
                "select_n",
            }
        ):
            for input_variable in variable_inputs:
                _union_broadcast_dimensions(classes, input_variable, output, equation=equation_index)
        elif primitive == "broadcast_in_dim" and variable_inputs:
            dimensions = _integer_tuple(equation.params.get("broadcast_dimensions"), primitive, equation_index)
            for input_dimension, output_dimension in enumerate(dimensions):
                if _shape(variable_inputs[0])[input_dimension] == _shape(output)[output_dimension]:
                    classes.union(
                        (variable_inputs[0], input_dimension),
                        (output, output_dimension),
                        equation=equation_index,
                    )
        elif primitive in {"identity_alias", "reshape", "squeeze", "stop_gradient"} and variable_inputs:
            _union_reshape_dimensions(classes, variable_inputs[0], output, equation=equation_index)
        elif primitive == "transpose" and variable_inputs:
            permutation = _integer_tuple(equation.params.get("permutation"), primitive, equation_index)
            for output_dimension, input_dimension in enumerate(permutation):
                classes.union(
                    (variable_inputs[0], input_dimension),
                    (output, output_dimension),
                    equation=equation_index,
                )
        elif primitive == "slice" and variable_inputs:
            for dimension, (input_extent, output_extent) in enumerate(
                zip(_shape(variable_inputs[0]), _shape(output), strict=True)
            ):
                if input_extent == output_extent:
                    classes.union(
                        (variable_inputs[0], dimension),
                        (output, dimension),
                        equation=equation_index,
                    )
        elif primitive in _FOLD_PRIMITIVES and variable_inputs:
            reduced = set(_integer_tuple(equation.params.get("axes"), primitive, equation_index))
            output_dimension = 0
            for input_dimension in range(len(_shape(variable_inputs[0]))):
                if input_dimension in reduced:
                    continue
                classes.union(
                    (variable_inputs[0], input_dimension),
                    (output, output_dimension),
                    equation=equation_index,
                )
                output_dimension += 1
        elif primitive == "dot_general" and len(variable_inputs) == 2:
            _union_dot_dimensions(
                classes,
                variable_inputs[0],
                variable_inputs[1],
                output,
                equation.params.get("dimension_numbers"),
                equation=equation_index,
            )
        elif primitive == "dot_general" and len(variable_inputs) == 1:
            _union_broadcast_dimensions(classes, variable_inputs[0], output, equation=equation_index)
        elif primitive == "scatter-add" and len(variable_inputs) == 3:
            _union_scatter_add_dimensions(
                classes,
                variable_inputs[0],
                variable_inputs[1],
                variable_inputs[2],
                output,
                equation,
                equation=equation_index,
            )
    return classes


def _union_scan_dimensions(
    classes: _DimensionClasses,
    scan: _FlatEquation,
    outputs: tuple[Var, ...],
    *,
    equation: int,
) -> None:
    num_consts = _static_nonnegative_integer(scan.params.get("num_consts"), "num_consts", equation)
    num_carry = _static_nonnegative_integer(scan.params.get("num_carry"), "num_carry", equation)
    length = _static_nonnegative_integer(scan.params.get("length"), "length", equation)
    inputs = tuple(atom for atom in scan.invars if isinstance(atom, Var))
    if len(inputs) != len(scan.invars):
        raise JaxprConversionError("scan", equation, "scan tensor operands cannot be scalar literals")
    if len(inputs) < num_consts + num_carry or len(outputs) < num_carry:
        raise JaxprConversionError("scan", equation, "scan boundary arity disagrees with its bindings")
    for initial, final in zip(inputs[num_consts : num_consts + num_carry], outputs[:num_carry], strict=True):
        if _shape(initial) != _shape(final):
            raise JaxprConversionError("scan", equation, "scan carried-state shapes must agree")
        for dimension in range(len(_shape(initial))):
            classes.union((initial, dimension), (final, dimension), equation=equation)

    scanned = inputs[num_consts + num_carry :]
    emitted = outputs[num_carry:]
    sequence_values = (*scanned, *emitted)
    if not sequence_values:
        return
    if any(not _shape(value) or _shape(value)[0] != length for value in sequence_values):
        raise JaxprConversionError("scan", equation, "scan tensors must have the declared leading extent")
    first = sequence_values[0]
    for value in sequence_values[1:]:
        classes.union((first, 0), (value, 0), equation=equation)


def _logical_axes(closed_jaxpr: _FlatClosedJaxpr, dimensions: _DimensionClasses) -> dict[tuple[Var, int], TensorAxis]:
    variables = (
        *closed_jaxpr.jaxpr.invars,
        *(
            variable
            for equation in closed_jaxpr.jaxpr.eqns
            for variable in equation.outvars
            if isinstance(variable, Var)
        ),
    )
    axis_by_root: dict[tuple[Var, int], TensorAxis] = {}
    result: dict[tuple[Var, int], TensorAxis] = {}
    for variable in variables:
        for dimension, size in enumerate(_shape(variable)):
            node = (variable, dimension)
            root = dimensions.find(node)
            if root not in axis_by_root:
                axis_by_root[root] = TensorAxis(len(axis_by_root), size)
            result[node] = axis_by_root[root]
    return result


def _program_value(variable: Var, axes: dict[tuple[Var, int], TensorAxis], *, name: str) -> ProgramValue:
    return ProgramValue(
        name,
        tuple(axes[(variable, dimension)] for dimension in range(len(_shape(variable)))),
        _dtype(variable.aval.dtype, primitive="boundary", equation=None),
    )


def _elementwise_operation(
    equation: Any,
    *,
    equation_index: int,
    output: ProgramValue,
    values: dict[Var, ProgramValue],
    temporary_index: int,
) -> tuple[MapPrimitive, tuple[RelationPrimitive, ...], int]:
    inputs: list[ProgramValue] = []
    expressions: list[ScalarExpression] = []
    relations: list[RelationPrimitive] = []
    broadcasted_by_variable: dict[Var, ProgramValue] = {}
    for operand_index, atom in enumerate(equation.invars):
        if isinstance(atom, Literal):
            expressions.append(_literal_expression(atom, equation.primitive.name, equation_index))
            continue
        if atom in broadcasted_by_variable:
            value = broadcasted_by_variable[atom]
        else:
            source = values[atom]
            if source.axes == output.axes:
                value = source
            else:
                value = ProgramValue(f"broadcast.{temporary_index}", output.axes, source.dtype)
                temporary_index += 1
                relations.append(
                    _broadcast_relation(
                        f"equation.{equation_index}.operand.{operand_index}.broadcast",
                        source,
                        value,
                    )
                )
            broadcasted_by_variable[atom] = value
            inputs.append(value)
        expressions.append(scalar_input(value.name))

    primitive = equation.primitive.name
    if primitive in _ELEMENTWISE_BINARY:
        expression = scalar_binary(_ELEMENTWISE_BINARY[primitive], expressions[0], expressions[1])
    elif primitive in _ELEMENTWISE_UNARY:
        expression = scalar_unary(_ELEMENTWISE_UNARY[primitive], expressions[0])
    elif primitive == "convert_element_type":
        expression = scalar_convert(
            expressions[0],
            result_dtype=output.dtype,
            semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        )
    elif primitive == "reduce_precision":
        exponent_bits = equation.params.get("exponent_bits")
        mantissa_bits = equation.params.get("mantissa_bits")
        if (exponent_bits, mantissa_bits) != (8, 7):
            raise JaxprConversionError(
                primitive,
                equation_index,
                "only the e8m7 format corresponding to BF16 is supported",
            )
        expression = scalar_convert(
            scalar_convert(
                expressions[0],
                result_dtype=DType.BF16,
                semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
            ),
            result_dtype=output.dtype,
            semantics=ScalarConversionSemantics.EXACT,
        )
    elif primitive == "integer_pow":
        expression = _integer_power_expression(
            expressions[0],
            equation.params.get("y"),
            equation_index,
            dtype=output.dtype,
        )
    elif primitive == "logistic":
        negated = scalar_unary(ScalarExpressionKind.NEGATE, expressions[0])
        denominator = scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            scalar_unary(ScalarExpressionKind.EXP, negated),
        )
        expression = scalar_binary(ScalarExpressionKind.DIVIDE, scalar_constant(1.0), denominator)
    else:
        assert primitive == "select_n"
        if len(expressions) != 3:
            raise JaxprConversionError(primitive, equation_index, "only a Boolean two-way select is supported")
        expression = scalar_select(expressions[0], expressions[2], expressions[1])
    unique_inputs = tuple(dict.fromkeys(inputs))
    return (
        MapPrimitive(f"equation.{equation_index}.{primitive}", unique_inputs, output, expression),
        tuple(relations),
        temporary_index,
    )


def _integer_power_expression(
    operand: ScalarExpression,
    exponent: object,
    equation: int,
    *,
    dtype: DType,
) -> ScalarExpression:
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise JaxprConversionError("integer_pow", equation, "exponent must be a static integer")
    if abs(exponent) > 8:
        raise JaxprConversionError(
            "integer_pow", equation, "only exponents with absolute value at most eight are supported"
        )
    one = scalar_constant(1 if dtype is DType.INT32 else 1.0)
    if dtype not in {DType.INT32, DType.FP32}:
        one = scalar_convert(
            one,
            result_dtype=dtype,
            semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        )
    magnitude = one
    for _ in range(abs(exponent)):
        magnitude = scalar_binary(ScalarExpressionKind.MULTIPLY, magnitude, operand)
    if exponent < 0:
        return scalar_binary(ScalarExpressionKind.DIVIDE, one, magnitude)
    return magnitude


def _literal_expression(literal: Literal, primitive: str, equation: int) -> ScalarExpression:
    value = np.asarray(literal.val)
    if value.shape:
        raise JaxprConversionError(primitive, equation, "only scalar JAXpr literals are supported")
    if value.dtype.kind == "b":
        expression = scalar_constant(bool(value))
        source_dtype = DType.BOOL
    elif value.dtype.kind == "f" or str(value.dtype) == "bfloat16":
        expression = scalar_constant(float(value))
        source_dtype = DType.FP32
    elif value.dtype.kind in "iu":
        expression = scalar_constant(int(value))
        source_dtype = DType.INT32
    else:
        raise JaxprConversionError(primitive, equation, f"unsupported scalar literal dtype {value.dtype}")
    target_dtype = _dtype(literal.aval.dtype, primitive, equation)
    if source_dtype is target_dtype:
        return expression
    return scalar_convert(
        expression,
        result_dtype=target_dtype,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )


def _broadcast_relation(name: str, source: ProgramValue, output: ProgramValue) -> RelationPrimitive:
    rank_offset = len(output.axes) - len(source.axes)
    if rank_offset < 0:
        raise ValueError("an elementwise operand cannot have greater rank than its output")
    indices: list[IndexExpression] = []
    for source_dimension, source_axis in enumerate(source.axes):
        output_dimension = source_dimension + rank_offset
        output_axis = output.axes[output_dimension]
        if source_axis.extent == 1:
            indices.append(index_constant(0))
        elif source_axis.extent == output_axis.extent:
            indices.append(index_axis(output_axis))
        else:
            raise ValueError("elementwise operands must obey trailing-dimension broadcasting")
    return RelationPrimitive(
        name,
        (source,),
        (output,),
        output.axes,
        (OperandIndexing(source, output.axes, tuple(indices)),),
    )


def _view_operation(
    equation: Any,
    *,
    equation_index: int,
    input_value: ProgramValue,
    output: ProgramValue,
) -> RelationPrimitive:
    primitive = equation.primitive.name
    if primitive == "broadcast_in_dim":
        dimensions = _integer_tuple(equation.params.get("broadcast_dimensions"), primitive, equation_index)
        indices = tuple(
            index_constant(0) if axis.extent == 1 else index_axis(output.axes[dimensions[input_dimension]])
            for input_dimension, axis in enumerate(input_value.axes)
        )
    elif primitive == "transpose":
        permutation = _integer_tuple(equation.params.get("permutation"), primitive, equation_index)
        indices = tuple(
            index_axis(output.axes[permutation.index(input_dimension)]) for input_dimension in range(len(permutation))
        )
    elif primitive == "slice":
        starts = _integer_tuple(equation.params.get("start_indices"), primitive, equation_index)
        strides = _integer_tuple(equation.params.get("strides"), primitive, equation_index)
        indices = tuple(
            index_binary(
                IndexExpressionKind.ADD,
                index_constant(start),
                index_binary(IndexExpressionKind.MULTIPLY, index_axis(axis), index_constant(stride)),
            )
            for start, stride, axis in zip(starts, strides, output.axes, strict=True)
        )
    else:
        if primitive == "reshape" and equation.params.get("dimensions") is not None:
            raise JaxprConversionError(primitive, equation_index, "reshape with an implicit permutation is unsupported")
        indices = _reshape_indices(input_value.axes, output.axes)
    return RelationPrimitive(
        f"equation.{equation_index}.{primitive}",
        (input_value,),
        (output,),
        output.axes,
        (OperandIndexing(input_value, output.axes, indices),),
    )


def _reshape_indices(
    source_axes: tuple[TensorAxis, ...], output_axes: tuple[TensorAxis, ...]
) -> tuple[IndexExpression, ...]:
    linear = index_constant(0)
    for dimension, axis in enumerate(output_axes):
        stride = int(np.prod([remaining.extent for remaining in output_axes[dimension + 1 :]], dtype=np.int64))
        term = index_axis(axis)
        if stride != 1:
            term = index_binary(IndexExpressionKind.MULTIPLY, term, index_constant(stride))
        linear = index_binary(IndexExpressionKind.ADD, linear, term)
    indices: list[IndexExpression] = []
    for dimension, axis in enumerate(source_axes):
        stride = int(np.prod([remaining.extent for remaining in source_axes[dimension + 1 :]], dtype=np.int64))
        index = linear
        if stride != 1:
            index = index_binary(IndexExpressionKind.FLOOR_DIVIDE, index, index_constant(stride))
        if axis.extent != 1:
            index = index_binary(IndexExpressionKind.MODULO, index, index_constant(axis.extent))
        else:
            index = index_constant(0)
        indices.append(index)
    return tuple(indices)


def _fold_operation(
    equation: Any,
    *,
    equation_index: int,
    input_value: ProgramValue,
    output: ProgramValue,
    numerical_policy: NumericalPolicy,
) -> FoldPrimitive:
    axes = _integer_tuple(equation.params.get("axes"), equation.primitive.name, equation_index)
    reducer = FoldReducer.SUM if equation.primitive.name == "reduce_sum" else FoldReducer.MAXIMUM
    order = (
        EvaluationOrder.SOURCE_ORDERED
        if numerical_policy is NumericalPolicy.SOURCE_ORDERED
        else EvaluationOrder.ORDER_FREE
    )
    return FoldPrimitive(
        f"equation.{equation_index}.{equation.primitive.name}",
        input_value,
        output,
        tuple(input_value.axes[dimension] for dimension in axes),
        reducer,
        output.dtype,
        order,
    )


def _scan_operation(
    equation: _FlatEquation,
    *,
    equation_index: int,
    inputs: tuple[ProgramValue, ...],
    outputs: tuple[ProgramValue, ...],
    numerical_policy: NumericalPolicy,
) -> ScanPrimitive:
    body_jaxpr = equation.params.get("jaxpr")
    if not isinstance(body_jaxpr, ClosedJaxpr):
        raise JaxprConversionError("scan", equation_index, "scan requires a closed body JAXpr")
    num_consts = _static_nonnegative_integer(equation.params.get("num_consts"), "num_consts", equation_index)
    num_carry = _static_nonnegative_integer(equation.params.get("num_carry"), "num_carry", equation_index)
    length = _static_nonnegative_integer(equation.params.get("length"), "length", equation_index)
    reverse = equation.params.get("reverse")
    if not isinstance(reverse, bool):
        raise JaxprConversionError("scan", equation_index, "reverse must be a static Boolean")
    if num_carry == 0:
        raise JaxprConversionError("scan", equation_index, "generic Scan requires at least one carried state")
    if length == 0:
        raise JaxprConversionError("scan", equation_index, "zero-length scans are not supported")
    if len(inputs) < num_consts + num_carry or len(outputs) < num_carry:
        raise JaxprConversionError("scan", equation_index, "scan boundary arity disagrees with its bindings")

    try:
        body = jaxpr_to_tensor_program(body_jaxpr, numerical_policy=numerical_policy)
    except JaxprConversionError as error:
        nested_path = error.equation_path
        if nested_path is None:
            nested_path = () if error.equation is None else (error.equation,)
        raise JaxprConversionError(
            error.primitive,
            equation.path[0],
            error.reason,
            equation_path=(*equation.path, *nested_path),
        ) from error
    if len(body.inputs) != len(inputs) or len(body.outputs) != len(outputs):
        raise JaxprConversionError("scan", equation_index, "scan body arity disagrees with its outer boundary")

    invariant_sources = inputs[:num_consts]
    initial_states = inputs[num_consts : num_consts + num_carry]
    scanned_sources = inputs[num_consts + num_carry :]
    final_states = outputs[:num_carry]
    emitted_destinations = outputs[num_carry:]
    invariant_arguments = body.inputs[:num_consts]
    state_arguments = body.inputs[num_consts : num_consts + num_carry]
    scanned_arguments = body.inputs[num_consts + num_carry :]
    state_results = body.outputs[:num_carry]
    emitted_results = body.outputs[num_carry:]
    if len(scanned_sources) != len(scanned_arguments) or len(emitted_destinations) != len(emitted_results):
        raise JaxprConversionError("scan", equation_index, "scan body sequence arity disagrees with its outer boundary")

    scan_axis_candidates = tuple(source.axes[0] for source in scanned_sources) + tuple(
        destination.axes[0] for destination in emitted_destinations
    )
    if not scan_axis_candidates:
        raise JaxprConversionError("scan", equation_index, "scan requires a scanned input or emitted output")
    scan_axis = scan_axis_candidates[0]
    if any(axis != scan_axis or axis.extent != length for axis in scan_axis_candidates):
        raise JaxprConversionError("scan", equation_index, "scan tensors do not share the declared leading axis")

    return ScanPrimitive(
        name=f"equation.{equation_index}.scan",
        axis=scan_axis,
        body=body,
        state=tuple(
            ScanStateBinding(initial, body_argument, body_result, final)
            for initial, body_argument, body_result, final in zip(
                initial_states, state_arguments, state_results, final_states, strict=True
            )
        ),
        invariants=tuple(
            ScanInvariantBinding(source, body_argument)
            for source, body_argument in zip(invariant_sources, invariant_arguments, strict=True)
        ),
        scanned_inputs=tuple(
            ScanInputBinding(
                source,
                body_argument,
                OperandIndexing(source, source.axes, tuple(index_axis(axis) for axis in source.axes)),
            )
            for source, body_argument in zip(scanned_sources, scanned_arguments, strict=True)
        ),
        emitted_outputs=tuple(
            ScanOutputBinding(
                body_result,
                destination,
                OperandIndexing(
                    destination,
                    destination.axes,
                    tuple(index_axis(axis) for axis in destination.axes),
                ),
            )
            for body_result, destination in zip(emitted_results, emitted_destinations, strict=True)
        ),
        direction=ScanDirection.REVERSE if reverse else ScanDirection.FORWARD,
        order=EvaluationOrder.SOURCE_ORDERED,
    )


def _static_nonnegative_integer(value: object, parameter: str, equation: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise JaxprConversionError("scan", equation, f"{parameter} must be a static nonnegative integer")
    return value


def _gather_operation(
    equation: Any,
    *,
    equation_index: int,
    inputs: tuple[ProgramValue, ProgramValue],
    output: ProgramValue,
) -> RelationPrimitive:
    operand, start_indices = inputs
    if start_indices.dtype is not DType.INT32:
        raise JaxprConversionError("gather", equation_index, "gather indices must be INT32")
    (
        offset_dims,
        collapsed_dimension,
        indexed_dimension,
        batch_output_dims,
        operand_batch_by_indices,
    ) = _full_slice_gather_structure(
        equation,
        operand_shape=operand.shape,
        indices_shape=start_indices.shape,
        output_shape=output.shape,
        equation_index=equation_index,
    )
    batch_indices = tuple(index_axis(output.axes[dimension]) for dimension in batch_output_dims)
    dynamic_index = index_tensor_element(start_indices, *batch_indices, index_constant(0))
    offset_by_operand = {
        operand_dimension: output_dimension
        for operand_dimension, output_dimension in zip(
            (
                dimension
                for dimension in range(len(operand.axes))
                if dimension != collapsed_dimension and dimension not in operand_batch_by_indices
            ),
            offset_dims,
            strict=True,
        )
    }
    operand_indices = tuple(
        (
            dynamic_index
            if operand_dimension == indexed_dimension
            else (
                index_axis(output.axes[batch_output_dims[operand_batch_by_indices[operand_dimension]]])
                if operand_dimension in operand_batch_by_indices
                else index_axis(output.axes[offset_by_operand[operand_dimension]])
            )
        )
        for operand_dimension in range(len(operand.axes))
    )
    return RelationPrimitive(
        f"equation.{equation_index}.gather",
        inputs,
        (output,),
        output.axes,
        (
            OperandIndexing(operand, output.axes, operand_indices),
            OperandIndexing(start_indices, output.axes, (*batch_indices, index_constant(0))),
        ),
        result_sources=(operand,),
        order=EvaluationOrder.ORDER_FREE,
    )


def _full_slice_gather_structure(
    equation: Any,
    *,
    operand_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    equation_index: int,
) -> tuple[tuple[int, ...], int, int, tuple[int, ...], dict[int, int]]:
    dimensions = equation.params.get("dimension_numbers")
    if dimensions is None:
        raise JaxprConversionError("gather", equation_index, "gather dimension_numbers are required")
    operand_batching = tuple(int(value) for value in dimensions.operand_batching_dims)
    indices_batching = tuple(int(value) for value in dimensions.start_indices_batching_dims)
    if len(operand_batching) != len(indices_batching):
        raise JaxprConversionError("gather", equation_index, "gather batching dimensions must be paired")
    if any(dimension >= len(indices_shape) - 1 for dimension in indices_batching):
        raise JaxprConversionError("gather", equation_index, "gather index batching dimension is out of range")
    if any(
        operand_shape[operand_dimension] != indices_shape[index_dimension]
        for operand_dimension, index_dimension in zip(operand_batching, indices_batching, strict=True)
    ):
        raise JaxprConversionError("gather", equation_index, "gather batch extents must agree")
    collapsed = tuple(int(value) for value in dimensions.collapsed_slice_dims)
    indexed = tuple(int(value) for value in dimensions.start_index_map)
    if len(collapsed) != 1 or collapsed != indexed:
        raise JaxprConversionError(
            "gather",
            equation_index,
            "full-slice gather requires one indexed collapsed operand dimension",
        )
    slice_sizes = _integer_tuple(equation.params.get("slice_sizes"), "gather", equation_index)
    if len(slice_sizes) != len(operand_shape):
        raise JaxprConversionError("gather", equation_index, "gather slice rank must match operand rank")
    collapsed_dimension = collapsed[0]
    if collapsed_dimension in operand_batching:
        raise JaxprConversionError("gather", equation_index, "indexed and batched operand dimensions must differ")
    for dimension, (slice_size, extent) in enumerate(zip(slice_sizes, operand_shape, strict=True)):
        expected = 1 if dimension == collapsed_dimension or dimension in operand_batching else extent
        if slice_size != expected:
            raise JaxprConversionError(
                "gather",
                equation_index,
                "gather must take full slices outside its indexed collapsed dimension",
            )
    if not indices_shape or indices_shape[-1] != 1:
        raise JaxprConversionError("gather", equation_index, "gather requires a trailing singleton index vector")
    offset_dims = tuple(int(value) for value in dimensions.offset_dims)
    if len(offset_dims) != len(operand_shape) - 1 - len(operand_batching):
        raise JaxprConversionError("gather", equation_index, "gather offset dimensions must cover every full slice")
    batch_output_dims = tuple(dimension for dimension in range(len(output_shape)) if dimension not in offset_dims)
    if tuple(output_shape[dimension] for dimension in batch_output_dims) != indices_shape[:-1]:
        raise JaxprConversionError("gather", equation_index, "gather output batch shape disagrees with its indices")
    mode = equation.params.get("mode")
    if getattr(mode, "name", None) != "PROMISE_IN_BOUNDS":
        raise JaxprConversionError("gather", equation_index, "only promise-in-bounds gather is supported")
    return (
        offset_dims,
        collapsed_dimension,
        indexed[0],
        batch_output_dims,
        dict(zip(operand_batching, indices_batching, strict=True)),
    )


def _scatter_add_operations(
    equation: Any,
    *,
    equation_index: int,
    inputs: tuple[ProgramValue, ...],
    output: ProgramValue,
) -> tuple[RelationPrimitive, FoldPrimitive, MapPrimitive]:
    if len(inputs) != 3:
        raise JaxprConversionError("scatter-add", equation_index, "scatter-add requires operand, indices, and updates")
    operand, scatter_indices, updates = inputs
    (
        window_to_operand,
        batch_update_to_operand,
        _inserted_dimension,
        scatter_dimension,
        route_dimensions,
        index_to_update,
    ) = _scatter_add_structure(
        equation,
        operand_shape=operand.shape,
        indices_shape=scatter_indices.shape,
        updates_shape=updates.shape,
        equation_index=equation_index,
    )
    if scatter_indices.dtype is not DType.INT32:
        raise JaxprConversionError("scatter-add", equation_index, "scatter indices must be INT32")
    reduction_dimensions = tuple(
        dimension for dimension in route_dimensions if dimension not in dict(batch_update_to_operand)
    )
    if not reduction_dimensions:
        raise JaxprConversionError("scatter-add", equation_index, "scatter-add requires a nonempty reduction domain")
    route_axes = tuple(updates.axes[dimension] for dimension in reduction_dimensions)
    domain_axes = (*output.axes, *route_axes)
    output_axis_by_update = {
        update_dimension: output.axes[operand_dimension]
        for update_dimension, operand_dimension in (*window_to_operand, *batch_update_to_operand)
    }
    route_axis_by_update = dict(zip(reduction_dimensions, route_axes, strict=True))
    update_indices = tuple(
        (
            index_axis(output_axis_by_update[dimension])
            if dimension in output_axis_by_update
            else index_axis(route_axis_by_update[dimension])
        )
        for dimension in range(len(updates.axes))
    )
    operand_batch_by_index = {
        index_dimension: operand_dimension
        for index_dimension, update_dimension in enumerate(index_to_update)
        for batch_update_dimension, operand_dimension in batch_update_to_operand
        if update_dimension == batch_update_dimension
    }
    scatter_index_expressions = tuple(
        (
            index_axis(output.axes[operand_batch_by_index[index_dimension]])
            if index_dimension in operand_batch_by_index
            else index_axis(route_axis_by_update[update_dimension])
        )
        for index_dimension, update_dimension in enumerate(index_to_update)
    )
    dynamic_destination = index_tensor_element(
        scatter_indices,
        *scatter_index_expressions,
        index_constant(0),
    )
    predicate = index_compare(
        IndexPredicateKind.EQUAL,
        index_axis(output.axes[scatter_dimension]),
        dynamic_destination,
    )
    contributions = ProgramValue(
        f"equation.{equation_index}.scatter.contributions",
        domain_axes,
        updates.dtype,
    )
    relation = RelationPrimitive(
        f"equation.{equation_index}.scatter.relation",
        (updates, scatter_indices),
        (contributions,),
        domain_axes,
        (
            OperandIndexing(updates, domain_axes, update_indices),
            OperandIndexing(
                scatter_indices,
                domain_axes,
                (*scatter_index_expressions, index_constant(0)),
            ),
        ),
        result_sources=(updates,),
        predicate=predicate,
        order=EvaluationOrder.SOURCE_ORDERED,
    )
    summed_updates = ProgramValue(f"equation.{equation_index}.scatter.sum", output.axes, output.dtype)
    fold = FoldPrimitive(
        f"equation.{equation_index}.scatter.fold",
        contributions,
        summed_updates,
        route_axes,
        FoldReducer.SUM,
        output.dtype,
        EvaluationOrder.SOURCE_ORDERED,
    )
    finalizer = MapPrimitive(
        f"equation.{equation_index}.scatter.add",
        (operand, summed_updates),
        output,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input(operand.name),
            scalar_input(summed_updates.name),
        ),
    )
    return relation, fold, finalizer


def _scatter_add_structure(
    equation: Any,
    *,
    operand_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    updates_shape: tuple[int, ...],
    equation_index: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    dimensions = equation.params.get("dimension_numbers")
    if dimensions is None:
        raise JaxprConversionError("scatter-add", equation_index, "scatter dimension_numbers are required")
    operand_batching = tuple(int(value) for value in dimensions.operand_batching_dims)
    indices_batching = tuple(int(value) for value in dimensions.scatter_indices_batching_dims)
    if len(operand_batching) != len(indices_batching):
        raise JaxprConversionError("scatter-add", equation_index, "scatter batching dimensions must be paired")
    inserted = tuple(int(value) for value in dimensions.inserted_window_dims)
    scattered = tuple(int(value) for value in dimensions.scatter_dims_to_operand_dims)
    if len(inserted) != 1 or inserted != scattered:
        raise JaxprConversionError(
            "scatter-add",
            equation_index,
            "scatter-add requires one indexed inserted operand dimension",
        )
    window = tuple(int(value) for value in dimensions.update_window_dims)
    route = tuple(dimension for dimension in range(len(updates_shape)) if dimension not in window)
    if not indices_shape or indices_shape[-1] != 1:
        raise JaxprConversionError(
            "scatter-add", equation_index, "scatter-add requires a trailing singleton index vector"
        )
    if tuple(updates_shape[dimension] for dimension in route) != indices_shape[:-1]:
        raise JaxprConversionError("scatter-add", equation_index, "scatter-add route shape disagrees with its indices")
    index_to_update = route
    batch_update_to_operand = tuple(
        (index_to_update[index_dimension], operand_dimension)
        for operand_dimension, index_dimension in zip(operand_batching, indices_batching, strict=True)
    )
    if any(operand_dimension == inserted[0] for operand_dimension in operand_batching):
        raise JaxprConversionError("scatter-add", equation_index, "indexed and batched operand dimensions must differ")
    if any(
        operand_shape[operand_dimension] != indices_shape[index_dimension]
        for operand_dimension, index_dimension in zip(operand_batching, indices_batching, strict=True)
    ):
        raise JaxprConversionError("scatter-add", equation_index, "scatter batch extents must agree")
    window_operand_dimensions = tuple(
        dimension
        for dimension in range(len(operand_shape))
        if dimension != inserted[0] and dimension not in operand_batching
    )
    if len(window) != len(window_operand_dimensions) or tuple(updates_shape[dimension] for dimension in window) != tuple(
        operand_shape[dimension] for dimension in window_operand_dimensions
    ):
        raise JaxprConversionError(
            "scatter-add",
            equation_index,
            "scatter-add updates must contain full windows outside the indexed dimension",
        )
    if getattr(equation.params.get("mode"), "name", None) != "PROMISE_IN_BOUNDS":
        raise JaxprConversionError("scatter-add", equation_index, "only promise-in-bounds scatter-add is supported")
    update_jaxpr = equation.params.get("update_jaxpr")
    update_equations = () if update_jaxpr is None else tuple(update_jaxpr.eqns)
    if len(update_equations) != 1 or update_equations[0].primitive.name != "add":
        raise JaxprConversionError("scatter-add", equation_index, "scatter update computation must be scalar addition")
    if equation.params.get("update_consts"):
        raise JaxprConversionError("scatter-add", equation_index, "scatter update constants are not supported")
    return (
        tuple(zip(window, window_operand_dimensions, strict=True)),
        batch_update_to_operand,
        inserted[0],
        scattered[0],
        route,
        index_to_update,
    )


def _selection_operation(
    equation: Any,
    *,
    equation_index: int,
    input_value: ProgramValue,
    outputs: tuple[ProgramValue, ProgramValue],
) -> SelectionPrimitive:
    primitive = equation.primitive.name
    axis = equation.params.get("axis")
    count = equation.params.get("k")
    if not isinstance(axis, int) or axis < 0 or axis >= len(input_value.axes):
        raise JaxprConversionError(primitive, equation_index, "top_k axis must be a valid static dimension")
    if not isinstance(count, int) or count <= 0:
        raise JaxprConversionError(primitive, equation_index, "top_k count must be a positive static integer")
    values, indices = outputs
    return SelectionPrimitive(
        f"equation.{equation_index}.top_k",
        input_value,
        values,
        indices,
        input_value.axes[axis],
        count,
        order=SelectionOrder.DESCENDING,
        comparison=SelectionComparison.TOTAL_ORDER,
        stable=True,
        tie_break=SelectionTieBreak.SOURCE_INDEX_ASCENDING,
        evaluation_order=EvaluationOrder.SOURCE_ORDERED,
    )


def _scalar_dot_operation(
    equation: Any,
    *,
    equation_index: int,
    output: ProgramValue,
    values: dict[Var, ProgramValue],
    temporary_index: int,
) -> tuple[MapPrimitive, tuple[RelationPrimitive, ...], int]:
    dimensions = _dot_dimension_numbers(equation.params.get("dimension_numbers"), equation_index)
    if any(dimensions_part for pair in dimensions for dimensions_part in pair):
        raise JaxprConversionError(
            "dot_general",
            equation_index,
            "a scalar dot operand requires empty contraction and batch dimensions",
        )
    literals = tuple(atom for atom in equation.invars if isinstance(atom, Literal))
    variables = tuple(atom for atom in equation.invars if isinstance(atom, Var))
    if len(literals) != 1 or len(variables) != 1 or literals[0].aval.shape:
        raise JaxprConversionError(
            "dot_general",
            equation_index,
            "only one rank-zero literal multiplied by one tensor is supported",
        )
    source = values[variables[0]]
    relations: tuple[RelationPrimitive, ...] = ()
    if source.axes != output.axes:
        broadcast = ProgramValue(f"broadcast.{temporary_index}", output.axes, source.dtype)
        temporary_index += 1
        relations = (_broadcast_relation(f"equation.{equation_index}.scalar_dot.broadcast", source, broadcast),)
        source = broadcast
    literal = _literal_expression(literals[0], "dot_general", equation_index)
    expression = scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(source.name), literal)
    return (
        MapPrimitive(f"equation.{equation_index}.scalar_dot", (source,), output, expression),
        relations,
        temporary_index,
    )


def _contract_operand_precision(precision: object, equation: int) -> tuple[ContractOperandPrecision, ...]:
    if precision is None:
        return (ContractOperandPrecision.DEFAULT, ContractOperandPrecision.DEFAULT)
    if (
        not isinstance(precision, tuple)
        or len(precision) != 2
        or any(not isinstance(operand, jax.lax.Precision) for operand in precision)
    ):
        raise JaxprConversionError("dot_general", equation, "explicit dot algorithms are not supported")
    return tuple(ContractOperandPrecision(operand.name.lower()) for operand in precision)


def _contract_operation(
    equation: Any,
    *,
    equation_index: int,
    inputs: tuple[ProgramValue, ProgramValue],
    output: ProgramValue,
) -> ContractPrimitive:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = _dot_dimension_numbers(
        equation.params.get("dimension_numbers"),
        equation_index,
    )
    contraction_axes = tuple(inputs[0].axes[int(dimension)] for dimension in lhs_contract)
    output_dimension = len(lhs_batch)
    input_index_maps: list[tuple[AxisIndexMap, ...]] = []
    for value, contracted, batched in (
        (inputs[0], lhs_contract, lhs_batch),
        (inputs[1], rhs_contract, rhs_batch),
    ):
        mappings: list[AxisIndexMap] = []
        for dimension, operand_axis in enumerate(value.axes):
            if dimension in contracted or dimension in batched:
                continue
            mappings.append(AxisIndexMap(output.axes[output_dimension], operand_axis))
            output_dimension += 1
        input_index_maps.append(tuple(mappings))
    preferred = equation.params.get("preferred_element_type")
    accumulation_dtype = output.dtype if preferred is None else _dtype(preferred, "dot_general", equation_index)
    return ContractPrimitive(
        f"equation.{equation_index}.dot_general",
        inputs,
        output,
        contraction_axes,
        accumulation_dtype,
        tuple(input_index_maps),
        # JAX dot_general specifies dimensions and accumulation type, but no
        # observable reduction association. An enclosing source-ordered
        # program therefore cannot strengthen this Contract to source-ordered.
        order=EvaluationOrder.ORDER_FREE,
        operand_precision=_contract_operand_precision(equation.params.get("precision"), equation_index),
    )


def _iota_operation(equation: Any, *, equation_index: int, output: ProgramValue) -> RelationPrimitive:
    if output.dtype is not DType.INT32:
        raise JaxprConversionError("iota", equation_index, "the initial source-free iota requires INT32")
    dimension = equation.params.get("dimension")
    if not isinstance(dimension, int):
        raise JaxprConversionError("iota", equation_index, "iota dimension must be a static integer")
    return RelationPrimitive(
        f"equation.{equation_index}.iota",
        (),
        (output,),
        output.axes,
        (),
        (index_axis(output.axes[dimension]),),
    )


def _union_broadcast_dimensions(
    classes: _DimensionClasses,
    input_variable: Var,
    output_variable: Var,
    *,
    equation: int,
) -> None:
    input_shape = _shape(input_variable)
    output_shape = _shape(output_variable)
    offset = len(output_shape) - len(input_shape)
    if offset < 0:
        raise JaxprConversionError("broadcast", equation, "operand rank exceeds result rank")
    for input_dimension, input_extent in enumerate(input_shape):
        output_dimension = input_dimension + offset
        if input_extent == output_shape[output_dimension]:
            classes.union((input_variable, input_dimension), (output_variable, output_dimension), equation=equation)
        elif input_extent != 1:
            raise JaxprConversionError("broadcast", equation, "incompatible elementwise broadcast extents")


def _union_reshape_dimensions(
    classes: _DimensionClasses,
    input_variable: Var,
    output_variable: Var,
    *,
    equation: int,
) -> None:
    input_dimensions = [index for index, extent in enumerate(_shape(input_variable)) if extent != 1]
    output_dimensions = [index for index, extent in enumerate(_shape(output_variable)) if extent != 1]
    if len(input_dimensions) != len(output_dimensions):
        return
    for input_dimension, output_dimension in zip(input_dimensions, output_dimensions, strict=True):
        if _shape(input_variable)[input_dimension] == _shape(output_variable)[output_dimension]:
            classes.union((input_variable, input_dimension), (output_variable, output_dimension), equation=equation)


def _union_dot_dimensions(
    classes: _DimensionClasses,
    lhs: Var,
    rhs: Var,
    output: Var,
    dimension_numbers: object,
    *,
    equation: int,
) -> None:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = _dot_dimension_numbers(dimension_numbers, equation)
    if len(lhs_contract) != len(rhs_contract) or len(lhs_batch) != len(rhs_batch):
        raise JaxprConversionError("dot_general", equation, "paired dot dimensions must have equal arity")
    for lhs_dimension, rhs_dimension in zip(lhs_contract, rhs_contract, strict=True):
        classes.union((lhs, lhs_dimension), (rhs, rhs_dimension), equation=equation)
    output_dimension = 0
    for lhs_dimension, rhs_dimension in zip(lhs_batch, rhs_batch, strict=True):
        classes.union((lhs, lhs_dimension), (rhs, rhs_dimension), equation=equation)
        classes.union((lhs, lhs_dimension), (output, output_dimension), equation=equation)
        output_dimension += 1


def _union_scatter_add_dimensions(
    classes: _DimensionClasses,
    operand: Var,
    scatter_indices: Var,
    updates: Var,
    output: Var,
    scatter_equation: Any,
    *,
    equation: int,
) -> None:
    window_to_operand, batch_update_to_operand, _, _, _route_dimensions, index_to_update = _scatter_add_structure(
        scatter_equation,
        operand_shape=_shape(operand),
        indices_shape=_shape(scatter_indices),
        updates_shape=_shape(updates),
        equation_index=equation,
    )
    for dimension in range(len(_shape(operand))):
        classes.union((operand, dimension), (output, dimension), equation=equation)
    for update_dimension, operand_dimension in (*window_to_operand, *batch_update_to_operand):
        classes.union((updates, update_dimension), (output, operand_dimension), equation=equation)
    for indices_dimension, update_dimension in enumerate(index_to_update):
        if update_dimension not in dict(batch_update_to_operand):
            classes.union((scatter_indices, indices_dimension), (updates, update_dimension), equation=equation)


def _dot_dimension_numbers(
    value: object,
    equation: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise JaxprConversionError("dot_general", equation, "invalid dimension_numbers")
    contraction, batch = value
    if not isinstance(contraction, Sequence) or len(contraction) != 2:
        raise JaxprConversionError("dot_general", equation, "invalid contraction dimensions")
    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise JaxprConversionError("dot_general", equation, "invalid batch dimensions")
    return (
        (
            _integer_tuple(contraction[0], "dot_general", equation),
            _integer_tuple(contraction[1], "dot_general", equation),
        ),
        (_integer_tuple(batch[0], "dot_general", equation), _integer_tuple(batch[1], "dot_general", equation)),
    )


def _integer_tuple(value: object, primitive: str, equation: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence):
        raise JaxprConversionError(primitive, equation, "dimension parameter must be a static sequence")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) for item in value):
        raise JaxprConversionError(primitive, equation, "dimension parameters cannot be Boolean")
    return result


def _shape(atom: Var) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in atom.aval.shape)


def _dtype(dtype: object, primitive: str, equation: int | None) -> DType:
    normalized = np.dtype(dtype)
    mapping = {
        np.dtype(np.bool_): DType.BOOL,
        np.dtype(np.float32): DType.FP32,
        np.dtype(np.float64): DType.FP64,
        np.dtype(np.int32): DType.INT32,
    }
    if str(normalized) == "bfloat16":
        return DType.BF16
    if normalized not in mapping:
        raise JaxprConversionError(primitive, equation, f"unsupported dtype {normalized}")
    return mapping[normalized]
