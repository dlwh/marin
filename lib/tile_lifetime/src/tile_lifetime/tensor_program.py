# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small backend-independent tensor semantics used by synthesis prototypes."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import ml_dtypes
import numpy as np

from shuttle.ir import DType


@dataclass(frozen=True)
class TensorAxis:
    """One logical dimension with stable identity."""

    id: int
    extent: int
    label: str | None = None

    def __post_init__(self) -> None:
        if self.extent <= 0:
            raise ValueError("tensor-axis extents must be positive")


@dataclass(frozen=True)
class ProgramValue:
    """A named tensor value indexed by logical axes."""

    name: str
    axes: tuple[TensorAxis, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"value {self.name!r} repeats a logical axis")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.extent for axis in self.axes)


class ScalarExpressionKind(StrEnum):
    """Scalar operations allowed in a pointwise map."""

    INPUT = "input"
    CONSTANT = "constant"
    CONVERT = "convert"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    NEGATE = "negate"
    EXP = "exp"
    LOG = "log"
    RSQRT = "rsqrt"
    TANH = "tanh"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    SELECT = "select"


class ScalarConversionSemantics(StrEnum):
    """Finite-precision semantics carried by a scalar conversion."""

    EXACT = "exact"
    ROUND_NEAREST_EVEN = "round_nearest_even"


@dataclass(frozen=True)
class ScalarExpression:
    """A scalar expression evaluated pointwise after logical-axis broadcasting."""

    kind: ScalarExpressionKind
    operands: tuple["ScalarExpression", ...] = ()
    input_name: str | None = None
    constant: float | int | bool | None = None
    result_dtype: DType | None = None
    conversion_semantics: ScalarConversionSemantics | None = None

    def __post_init__(self) -> None:
        if self.kind is ScalarExpressionKind.INPUT:
            if (
                self.input_name is None
                or self.operands
                or self.constant is not None
                or self.result_dtype is not None
                or self.conversion_semantics is not None
            ):
                raise ValueError("input scalar expressions require only an input name")
            return
        if self.kind is ScalarExpressionKind.CONSTANT:
            if (
                self.constant is None
                or self.operands
                or self.input_name is not None
                or self.result_dtype is not None
                or self.conversion_semantics is not None
            ):
                raise ValueError("constant scalar expressions require only a literal")
            return
        expected_arity = {
            ScalarExpressionKind.CONVERT: 1,
            ScalarExpressionKind.ADD: 2,
            ScalarExpressionKind.SUBTRACT: 2,
            ScalarExpressionKind.MULTIPLY: 2,
            ScalarExpressionKind.DIVIDE: 2,
            ScalarExpressionKind.NEGATE: 1,
            ScalarExpressionKind.EXP: 1,
            ScalarExpressionKind.LOG: 1,
            ScalarExpressionKind.RSQRT: 1,
            ScalarExpressionKind.TANH: 1,
            ScalarExpressionKind.MAXIMUM: 2,
            ScalarExpressionKind.MINIMUM: 2,
            ScalarExpressionKind.LESS: 2,
            ScalarExpressionKind.LESS_EQUAL: 2,
            ScalarExpressionKind.GREATER: 2,
            ScalarExpressionKind.GREATER_EQUAL: 2,
            ScalarExpressionKind.EQUAL: 2,
            ScalarExpressionKind.NOT_EQUAL: 2,
            ScalarExpressionKind.SELECT: 3,
        }[self.kind]
        if len(self.operands) != expected_arity or self.input_name is not None or self.constant is not None:
            raise ValueError(f"{self.kind.value} scalar expressions require {expected_arity} operands")
        if self.kind is ScalarExpressionKind.CONVERT:
            if self.result_dtype is None or self.conversion_semantics is None:
                raise ValueError("convert scalar expressions require a result dtype and conversion semantics")
            return
        if self.result_dtype is not None or self.conversion_semantics is not None:
            raise ValueError(f"{self.kind.value} scalar expressions cannot carry conversion metadata")


def serialize_scalar_expression(expression: ScalarExpression) -> str:
    """Serialize one scalar AST for transport through a physical-plan record."""

    def encode(node: ScalarExpression) -> dict[str, object]:
        encoded: dict[str, object] = {"kind": node.kind.value}
        if node.input_name is not None:
            encoded["input"] = node.input_name
        if node.constant is not None:
            encoded["constant"] = node.constant
        if node.result_dtype is not None:
            encoded["result_dtype"] = node.result_dtype.value
        if node.conversion_semantics is not None:
            encoded["conversion_semantics"] = node.conversion_semantics.value
        if node.operands:
            encoded["operands"] = [encode(operand) for operand in node.operands]
        return encoded

    return json.dumps(encode(expression), sort_keys=True, separators=(",", ":"))


def deserialize_scalar_expression(serialized: str) -> ScalarExpression:
    """Restore a scalar AST emitted by :func:`serialize_scalar_expression`."""

    def decode(encoded: object) -> ScalarExpression:
        if not isinstance(encoded, dict) or not isinstance(encoded.get("kind"), str):
            raise ValueError("serialized scalar expression must be an object with a string kind")
        kind = ScalarExpressionKind(encoded["kind"])
        operands_value = encoded.get("operands", ())
        if not isinstance(operands_value, (list, tuple)):
            raise ValueError("serialized scalar-expression operands must be a sequence")
        operands = tuple(decode(operand) for operand in operands_value)
        input_name = encoded.get("input")
        constant = encoded.get("constant")
        result_dtype = encoded.get("result_dtype")
        conversion_semantics = encoded.get("conversion_semantics")
        if input_name is not None and not isinstance(input_name, str):
            raise ValueError("serialized scalar-expression inputs must be strings")
        if constant is not None and not isinstance(constant, (bool, int, float)):
            raise ValueError("serialized scalar-expression constants must be numeric or Boolean")
        if result_dtype is not None and not isinstance(result_dtype, str):
            raise ValueError("serialized scalar-expression result dtypes must be strings")
        if conversion_semantics is not None and not isinstance(conversion_semantics, str):
            raise ValueError("serialized scalar-expression conversion semantics must be strings")
        return ScalarExpression(
            kind=kind,
            operands=operands,
            input_name=input_name,
            constant=constant,
            result_dtype=None if result_dtype is None else DType(result_dtype),
            conversion_semantics=(
                None if conversion_semantics is None else ScalarConversionSemantics(conversion_semantics)
            ),
        )

    return decode(json.loads(serialized))


def scalar_input(name: str) -> ScalarExpression:
    """Reference one pointwise-map input."""
    return ScalarExpression(kind=ScalarExpressionKind.INPUT, input_name=name)


def scalar_constant(value: float | int | bool) -> ScalarExpression:
    """Construct one scalar literal."""
    return ScalarExpression(kind=ScalarExpressionKind.CONSTANT, constant=value)


def scalar_convert(
    operand: ScalarExpression,
    *,
    result_dtype: DType,
    semantics: ScalarConversionSemantics,
) -> ScalarExpression:
    """Construct a typed scalar conversion with explicit rounding semantics."""
    return ScalarExpression(
        kind=ScalarExpressionKind.CONVERT,
        operands=(operand,),
        result_dtype=result_dtype,
        conversion_semantics=semantics,
    )


def scalar_unary(kind: ScalarExpressionKind, operand: ScalarExpression) -> ScalarExpression:
    """Construct a unary scalar expression."""
    if kind not in (
        ScalarExpressionKind.EXP,
        ScalarExpressionKind.LOG,
        ScalarExpressionKind.RSQRT,
        ScalarExpressionKind.TANH,
        ScalarExpressionKind.NEGATE,
    ):
        raise ValueError(f"{kind.value} is not unary")
    return ScalarExpression(kind=kind, operands=(operand,))


def scalar_binary(
    kind: ScalarExpressionKind,
    left: ScalarExpression,
    right: ScalarExpression,
) -> ScalarExpression:
    """Construct a binary scalar expression."""
    if kind not in {
        ScalarExpressionKind.ADD,
        ScalarExpressionKind.SUBTRACT,
        ScalarExpressionKind.MULTIPLY,
        ScalarExpressionKind.DIVIDE,
        ScalarExpressionKind.MAXIMUM,
        ScalarExpressionKind.MINIMUM,
        ScalarExpressionKind.LESS,
        ScalarExpressionKind.LESS_EQUAL,
        ScalarExpressionKind.GREATER,
        ScalarExpressionKind.GREATER_EQUAL,
        ScalarExpressionKind.EQUAL,
        ScalarExpressionKind.NOT_EQUAL,
    }:
        raise ValueError(f"{kind.value} is not binary")
    return ScalarExpression(kind=kind, operands=(left, right))


def scalar_select(
    predicate: ScalarExpression,
    when_true: ScalarExpression,
    when_false: ScalarExpression,
) -> ScalarExpression:
    """Construct a pointwise selection."""
    return ScalarExpression(
        kind=ScalarExpressionKind.SELECT,
        operands=(predicate, when_true, when_false),
    )


def evaluate_scalar_expression(expression: ScalarExpression, inputs: Mapping[str, Any]) -> Any:
    """Evaluate the closed scalar AST on Python scalars or NumPy-compatible arrays."""
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return inputs[expression.input_name]
        except KeyError as error:
            raise ValueError(f"missing scalar-expression input {expression.input_name!r}") from error
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant

    operands = tuple(evaluate_scalar_expression(operand, inputs) for operand in expression.operands)
    if kind is ScalarExpressionKind.CONVERT:
        assert expression.result_dtype is not None
        dtype = {
            DType.BOOL: np.bool_,
            DType.BF16: ml_dtypes.bfloat16,
            DType.FP32: np.float32,
            DType.FP64: np.float64,
            DType.INT32: np.int32,
        }[expression.result_dtype]
        return np.asarray(operands[0]).astype(dtype)
    if kind is ScalarExpressionKind.ADD:
        return np.add(operands[0], operands[1])
    if kind is ScalarExpressionKind.SUBTRACT:
        return np.subtract(operands[0], operands[1])
    if kind is ScalarExpressionKind.MULTIPLY:
        return np.multiply(operands[0], operands[1])
    if kind is ScalarExpressionKind.DIVIDE:
        return np.divide(operands[0], operands[1])
    if kind is ScalarExpressionKind.NEGATE:
        return np.negative(operands[0])
    if kind is ScalarExpressionKind.EXP:
        return np.exp(operands[0])
    if kind is ScalarExpressionKind.LOG:
        return np.log(operands[0])
    if kind is ScalarExpressionKind.RSQRT:
        return np.reciprocal(np.sqrt(operands[0]))
    if kind is ScalarExpressionKind.TANH:
        return np.tanh(operands[0])
    if kind is ScalarExpressionKind.MAXIMUM:
        return np.maximum(operands[0], operands[1])
    if kind is ScalarExpressionKind.MINIMUM:
        return np.minimum(operands[0], operands[1])
    if kind is ScalarExpressionKind.LESS:
        return np.less(operands[0], operands[1])
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return np.less_equal(operands[0], operands[1])
    if kind is ScalarExpressionKind.GREATER:
        return np.greater(operands[0], operands[1])
    if kind is ScalarExpressionKind.GREATER_EQUAL:
        return np.greater_equal(operands[0], operands[1])
    if kind is ScalarExpressionKind.EQUAL:
        return np.equal(operands[0], operands[1])
    if kind is ScalarExpressionKind.NOT_EQUAL:
        return np.not_equal(operands[0], operands[1])
    assert kind is ScalarExpressionKind.SELECT
    return np.where(operands[0], operands[1], operands[2])


def infer_scalar_expression_dtype(
    expression: ScalarExpression,
    inputs: Mapping[str, DType],
) -> DType:
    """Infer the result type of a scalar AST under deterministic literal typing."""
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return inputs[expression.input_name]
        except KeyError as error:
            raise ValueError(f"missing scalar-expression type for {expression.input_name!r}") from error
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return DType.BOOL
        return DType.INT32 if isinstance(expression.constant, int) else DType.FP32
    if kind is ScalarExpressionKind.CONVERT:
        assert expression.result_dtype is not None
        infer_scalar_expression_dtype(expression.operands[0], inputs)
        return expression.result_dtype
    operand_types = tuple(infer_scalar_expression_dtype(operand, inputs) for operand in expression.operands)
    if kind in {
        ScalarExpressionKind.LESS,
        ScalarExpressionKind.LESS_EQUAL,
        ScalarExpressionKind.GREATER,
        ScalarExpressionKind.GREATER_EQUAL,
        ScalarExpressionKind.EQUAL,
        ScalarExpressionKind.NOT_EQUAL,
    }:
        _promote_scalar_dtypes(operand_types[0], operand_types[1])
        return DType.BOOL
    if kind is ScalarExpressionKind.SELECT:
        if operand_types[0] is not DType.BOOL:
            raise ValueError("scalar select predicates must be Boolean")
        return _promote_scalar_dtypes(operand_types[1], operand_types[2])
    if kind in {
        ScalarExpressionKind.NEGATE,
        ScalarExpressionKind.EXP,
        ScalarExpressionKind.LOG,
        ScalarExpressionKind.RSQRT,
        ScalarExpressionKind.TANH,
    }:
        if operand_types[0] not in {DType.BF16, DType.FP32, DType.FP64}:
            raise ValueError(f"{kind.value} requires a floating-point operand")
        return operand_types[0]
    return _promote_scalar_dtypes(operand_types[0], operand_types[1])


def _promote_scalar_dtypes(left: DType, right: DType) -> DType:
    if left is right:
        return left
    floating = (DType.BF16, DType.FP32, DType.FP64)
    if left in floating and right in floating:
        return floating[max(floating.index(left), floating.index(right))]
    raise ValueError(f"scalar operation has incompatible operand dtypes {left.value} and {right.value}")


class NumericalPolicy(StrEnum):
    """Program-level finite-precision transformation policy."""

    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True)
class AssociativeFoldErrorContract:
    """Forward error bound for a reassociated sum of realized scalar terms.

    For finite terms, no overflow, and round-to-nearest addition, every binary
    association of ``n`` FP32 terms differs from their exact real sum by at
    most ``gamma(n - 1) * sum(abs(term))``. The contract begins after each
    term's scalar Map has been evaluated, so it does not hide upstream Map
    rounding or discontinuous downstream conversions.
    """

    accumulation_dtype: DType
    unit_roundoff: float

    def __post_init__(self) -> None:
        if self.accumulation_dtype is not DType.FP32:
            raise ValueError("the current associative Fold error contract is derived for FP32")
        if not 0.0 < self.unit_roundoff < 1.0:
            raise ValueError("unit roundoff must be finite and between zero and one")

    def coefficient(self, term_count: int) -> float:
        """Return gamma(n - 1) for one valid binary association."""
        if term_count <= 0:
            raise ValueError("a Fold error bound requires at least one term")
        rounded_operations = term_count - 1
        product = rounded_operations * self.unit_roundoff
        if product >= 1.0:
            raise ValueError("the standard summation error bound requires (n - 1) * u < 1")
        return product / (1.0 - product)

    def absolute_error_bound(self, *, term_count: int, absolute_sum: float) -> float:
        """Bound error against the exact sum of the realized terms."""
        if not math.isfinite(absolute_sum) or absolute_sum < 0.0:
            raise ValueError("the Fold absolute sum must be finite and nonnegative")
        return self.coefficient(term_count) * absolute_sum

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the artifact-stable statement of this numerical contract."""
        return {
            "accumulation_dtype": self.accumulation_dtype.value,
            "bound": "gamma(n-1)*sum(abs(realized_values))",
            "rounding": "round_nearest_even",
            "unit_roundoff": self.unit_roundoff,
        }


FAST_FP32_SUM_ERROR_CONTRACT = AssociativeFoldErrorContract(DType.FP32, 2.0**-24)


class EvaluationOrder(StrEnum):
    """Order constraint attached to an individual algebra primitive."""

    SOURCE_ORDERED = "source_ordered"
    ORDER_FREE = "order_free"


class ContractOperandPrecision(StrEnum):
    """Source multiplication precision request, independent of reduction order.

    DEFAULT permits tensor-core realization; it does not prescribe TF32.
    HIGH and HIGHEST constrain operand approximation, not the instruction family.
    """

    DEFAULT = "default"
    HIGH = "high"
    HIGHEST = "highest"


@dataclass(frozen=True)
class ContractPrimitive:
    """A multilinear contraction with explicit key and reduction axes."""

    name: str
    inputs: tuple[ProgramValue, ...]
    output: ProgramValue
    reduction_axes: tuple[TensorAxis, ...]
    accumulation_dtype: DType
    input_index_maps: tuple[tuple["AxisIndexMap", ...], ...] = ()
    order: EvaluationOrder = EvaluationOrder.ORDER_FREE
    operand_precision: tuple[ContractOperandPrecision, ...] = ()

    def __post_init__(self) -> None:
        if not self.operand_precision:
            object.__setattr__(self, "operand_precision", (ContractOperandPrecision.DEFAULT,) * len(self.inputs))
        if len(self.operand_precision) != len(self.inputs):
            raise ValueError("a contraction requires one precision request per operand")
        if len(self.inputs) < 2:
            raise ValueError("a contraction requires at least two inputs")
        input_axes = {axis for value in self.inputs for axis in value.axes}
        if not set(self.reduction_axes) <= input_axes:
            raise ValueError("contraction reduction axes must occur in its inputs")
        if set(self.output.axes) & set(self.reduction_axes):
            raise ValueError("contraction output cannot retain a reduction axis")
        if self.input_index_maps and len(self.input_index_maps) != len(self.inputs):
            raise ValueError("contraction index-map groups must match its inputs")
        iteration_axes = set(self.output.axes) | set(self.reduction_axes)
        for value, mappings in zip(self.inputs, self.input_index_maps, strict=False):
            mapped_operand_axes: set[TensorAxis] = set()
            for mapping in mappings:
                if mapping.domain_axis not in iteration_axes:
                    raise ValueError("contraction index-map domain axes must occur in the output or reduction domain")
                if mapping.operand_axis not in value.axes:
                    raise ValueError("contraction index-map operand axes must occur in the mapped input")
                if mapping.operand_axis in mapped_operand_axes:
                    raise ValueError("a contraction input axis can be indexed only once")
                mapped_operand_axes.add(mapping.operand_axis)

    def index_maps_for_input(self, input_index: int) -> tuple["AxisIndexMap", ...]:
        """Return the logical indexing relations applied to one operand."""
        if not self.input_index_maps:
            return ()
        return self.input_index_maps[input_index]


@dataclass(frozen=True)
class AxisIndexMap:
    """Map one output-domain index onto an operand axis.

    The initial affine integer form is deliberately small: it covers grouped
    head sharing and block-oriented relations without naming either workload.
    For a domain index ``i``, the operand index is
    ``((i // divisor) + offset) % modulus`` when a modulus is present.
    """

    domain_axis: TensorAxis
    operand_axis: TensorAxis
    divisor: int = 1
    offset: int = 0
    modulus: int | None = None

    def __post_init__(self) -> None:
        if self.divisor <= 0:
            raise ValueError("axis index-map divisors must be positive")
        if self.modulus is not None and self.modulus <= 0:
            raise ValueError("axis index-map moduli must be positive")
        mapped_min = self.offset
        mapped_max = (self.domain_axis.extent - 1) // self.divisor + self.offset
        if self.modulus is None and (mapped_min < 0 or mapped_max >= self.operand_axis.extent):
            raise ValueError("axis index map can address outside the operand axis")
        if self.modulus is not None and self.modulus > self.operand_axis.extent:
            raise ValueError("axis index-map modulus exceeds the operand axis")

    def indices(self) -> tuple[int, ...]:
        """Materialize the finite index relation for reference execution."""
        values = tuple(index // self.divisor + self.offset for index in range(self.domain_axis.extent))
        if self.modulus is None:
            return values
        return tuple(value % self.modulus for value in values)


class IndexExpressionKind(StrEnum):
    """Integer/Boolean expressions used by operand indexing relations."""

    AXIS = "axis"
    CONSTANT = "constant"
    TENSOR_ELEMENT = "tensor_element"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    FLOOR_DIVIDE = "floor_divide"
    MODULO = "modulo"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SELECT = "select"


@dataclass(frozen=True)
class IndexExpression:
    """One expression in a tensor operand's logical indexing map."""

    kind: IndexExpressionKind
    operands: tuple["IndexExpression", ...] = ()
    axis: TensorAxis | None = None
    constant: int | bool | None = None
    tensor: ProgramValue | None = None
    predicate: "IndexPredicate | None" = None

    def __post_init__(self) -> None:
        if self.kind is IndexExpressionKind.AXIS:
            if (
                self.axis is None
                or self.operands
                or self.constant is not None
                or self.tensor is not None
                or self.predicate is not None
            ):
                raise ValueError("axis index expressions require only an axis")
            return
        if self.kind is IndexExpressionKind.CONSTANT:
            if (
                self.constant is None
                or self.operands
                or self.axis is not None
                or self.tensor is not None
                or self.predicate is not None
            ):
                raise ValueError("constant index expressions require only an integer or Boolean literal")
            return
        if self.kind is IndexExpressionKind.TENSOR_ELEMENT:
            if self.tensor is None or self.axis is not None or self.constant is not None or self.predicate is not None:
                raise ValueError("tensor-element index expressions require a tensor and index operands")
            if len(self.operands) != len(self.tensor.axes):
                raise ValueError("tensor-element index count must equal the tensor rank")
            if self.tensor.dtype not in {DType.INT32, DType.BOOL}:
                raise ValueError("tensor-element index expressions require an INT32 or Boolean tensor")
            return
        if self.kind is IndexExpressionKind.SELECT:
            if (
                self.predicate is None
                or len(self.operands) != 2
                or self.axis is not None
                or self.constant is not None
                or self.tensor is not None
            ):
                raise ValueError("select index expressions require a predicate and two value operands")
            return
        if (
            len(self.operands) != 2
            or self.axis is not None
            or self.constant is not None
            or self.tensor is not None
            or self.predicate is not None
        ):
            raise ValueError(f"{self.kind.value} index expressions require exactly two operands")


def index_axis(axis: TensorAxis) -> IndexExpression:
    """Reference one logical domain axis in an indexing expression."""
    return IndexExpression(IndexExpressionKind.AXIS, axis=axis)


def index_constant(value: int | bool) -> IndexExpression:
    """Construct an integer or Boolean indexing literal."""
    return IndexExpression(IndexExpressionKind.CONSTANT, constant=value)


def index_tensor_element(tensor: ProgramValue, *indices: IndexExpression) -> IndexExpression:
    """Read an INT32 or Boolean tensor at dynamically computed logical indices."""
    return IndexExpression(IndexExpressionKind.TENSOR_ELEMENT, operands=tuple(indices), tensor=tensor)


def index_binary(
    kind: IndexExpressionKind,
    left: IndexExpression,
    right: IndexExpression,
) -> IndexExpression:
    """Construct a binary indexing expression."""
    if kind in {
        IndexExpressionKind.AXIS,
        IndexExpressionKind.CONSTANT,
        IndexExpressionKind.TENSOR_ELEMENT,
        IndexExpressionKind.SELECT,
    }:
        raise ValueError(f"{kind.value} is not a binary index expression")
    return IndexExpression(kind, operands=(left, right))


def index_select(
    predicate: "IndexPredicate",
    when_true: IndexExpression,
    when_false: IndexExpression,
) -> IndexExpression:
    """Select one of two integer/Boolean index expressions."""
    return IndexExpression(IndexExpressionKind.SELECT, operands=(when_true, when_false), predicate=predicate)


class IndexPredicateKind(StrEnum):
    """Predicates restricting an algebra primitive's logical domain."""

    CONSTANT = "constant"
    IS_TRUE = "is_true"
    NOT = "not"
    AND = "and"
    OR = "or"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"


@dataclass(frozen=True)
class IndexPredicate:
    """A generic predicate over index expressions and nested predicates."""

    kind: IndexPredicateKind
    expressions: tuple[IndexExpression, ...] = ()
    predicates: tuple["IndexPredicate", ...] = ()
    constant: bool | None = None

    def __post_init__(self) -> None:
        if self.kind is IndexPredicateKind.CONSTANT:
            if self.constant is None or self.expressions or self.predicates:
                raise ValueError("constant index predicates require only a Boolean literal")
            return
        if self.constant is not None:
            raise ValueError(f"{self.kind.value} index predicates cannot carry a literal")
        if self.kind is IndexPredicateKind.IS_TRUE:
            if len(self.expressions) != 1 or self.predicates:
                raise ValueError("is-true predicates require one index expression")
            return
        if self.kind is IndexPredicateKind.NOT:
            if len(self.predicates) != 1 or self.expressions:
                raise ValueError("not predicates require one nested predicate")
            return
        if self.kind in {IndexPredicateKind.AND, IndexPredicateKind.OR}:
            if len(self.predicates) != 2 or self.expressions:
                raise ValueError(f"{self.kind.value} predicates require two nested predicates")
            return
        if len(self.expressions) != 2 or self.predicates:
            raise ValueError(f"{self.kind.value} predicates require two index expressions")


def index_compare(
    kind: IndexPredicateKind,
    left: IndexExpression,
    right: IndexExpression,
) -> IndexPredicate:
    """Construct an index comparison."""
    if kind not in {
        IndexPredicateKind.EQUAL,
        IndexPredicateKind.NOT_EQUAL,
        IndexPredicateKind.LESS,
        IndexPredicateKind.LESS_EQUAL,
        IndexPredicateKind.GREATER,
        IndexPredicateKind.GREATER_EQUAL,
    }:
        raise ValueError(f"{kind.value} is not an index comparison")
    return IndexPredicate(kind, expressions=(left, right))


@dataclass(frozen=True)
class OperandIndexing:
    """Map a primitive domain onto every logical axis of one operand."""

    operand: ProgramValue
    domain_axes: tuple[TensorAxis, ...]
    indices: tuple[IndexExpression, ...]
    predicate: IndexPredicate | None = None

    def __post_init__(self) -> None:
        if len(set(self.domain_axes)) != len(self.domain_axes):
            raise ValueError("operand-indexing domain axes must be unique")
        if len(self.indices) != len(self.operand.axes):
            raise ValueError("operand-indexing expressions must cover every operand axis")
        referenced_axes = set().union(*(_index_expression_axes(index) for index in self.indices), set())
        if self.predicate is not None:
            referenced_axes.update(_index_predicate_axes(self.predicate))
        if not referenced_axes <= set(self.domain_axes):
            raise ValueError("operand indexing references an axis outside its domain")


def _index_expression_axes(expression: IndexExpression) -> set[TensorAxis]:
    axes = {expression.axis} if expression.axis is not None else set()
    for operand in expression.operands:
        axes.update(_index_expression_axes(operand))
    if expression.predicate is not None:
        axes.update(_index_predicate_axes(expression.predicate))
    return axes


def _index_expression_tensors(expression: IndexExpression) -> set[ProgramValue]:
    tensors = {expression.tensor} if expression.tensor is not None else set()
    for operand in expression.operands:
        tensors.update(_index_expression_tensors(operand))
    if expression.predicate is not None:
        tensors.update(_index_predicate_tensors(expression.predicate))
    return tensors


def _index_predicate_axes(predicate: IndexPredicate) -> set[TensorAxis]:
    axes: set[TensorAxis] = set()
    for expression in predicate.expressions:
        axes.update(_index_expression_axes(expression))
    for nested in predicate.predicates:
        axes.update(_index_predicate_axes(nested))
    return axes


def _index_predicate_tensors(predicate: IndexPredicate) -> set[ProgramValue]:
    tensors: set[ProgramValue] = set()
    for expression in predicate.expressions:
        tensors.update(_index_expression_tensors(expression))
    for nested in predicate.predicates:
        tensors.update(_index_predicate_tensors(nested))
    return tensors


@dataclass(frozen=True)
class MapPrimitive:
    """A pointwise map over explicitly broadcast tensor inputs and an optional index domain."""

    name: str
    inputs: tuple[ProgramValue, ...]
    output: ProgramValue
    expression: ScalarExpression
    output_conversion: ScalarConversionSemantics = ScalarConversionSemantics.ROUND_NEAREST_EVEN
    predicate: IndexPredicate | None = None
    rejected_value: float | int | bool | None = None

    def __post_init__(self) -> None:
        input_names = {value.name for value in self.inputs}
        referenced_names = scalar_expression_inputs(self.expression)
        if self.predicate is not None:
            referenced_names |= {value.name for value in _index_predicate_tensors(self.predicate)}
        if referenced_names != input_names:
            raise ValueError(
                f"map {self.name!r} expression inputs {sorted(referenced_names)} do not match "
                f"declared inputs {sorted(input_names)}"
            )
        output_axes = set(self.output.axes)
        if any(not set(value.axes) <= output_axes for value in self.inputs):
            raise ValueError(f"map {self.name!r} inputs do not broadcast to its output axes")
        if (self.predicate is None) != (self.rejected_value is None):
            raise ValueError("Map predicate and rejected value must be specified together")
        if self.predicate is not None and not _index_predicate_axes(self.predicate) <= output_axes:
            raise ValueError("Map predicate references an axis outside its output domain")
        if isinstance(self.rejected_value, float) and math.isnan(self.rejected_value):
            raise ValueError("Map rejected value cannot be NaN")
        expression_dtype = infer_scalar_expression_dtype(
            self.expression, {value.name: value.dtype for value in self.inputs}
        )
        if expression_dtype is not self.output.dtype and {
            expression_dtype,
            self.output.dtype,
        } - {DType.BF16, DType.FP32, DType.FP64}:
            raise ValueError(
                f"map {self.name!r} expression has dtype {expression_dtype.value}, "
                f"which cannot be converted to output dtype {self.output.dtype.value}"
            )
        exact_widening = {
            (DType.BF16, DType.FP32),
            (DType.BF16, DType.FP64),
            (DType.FP32, DType.FP64),
        }
        if (
            expression_dtype is not self.output.dtype
            and self.output_conversion is ScalarConversionSemantics.EXACT
            and (expression_dtype, self.output.dtype) not in exact_widening
        ):
            raise ValueError(
                f"map {self.name!r} cannot convert {expression_dtype.value} to " f"{self.output.dtype.value} exactly"
            )


class FoldReducer(StrEnum):
    """Associative reduction operators supported by the semantic IR."""

    MAXIMUM = "maximum"
    SUM = "sum"


@dataclass(frozen=True)
class FoldPrimitive:
    """An unordered reduction over one or more logical axes."""

    name: str
    input: ProgramValue
    output: ProgramValue
    reduction_axes: tuple[TensorAxis, ...]
    reducer: FoldReducer
    accumulation_dtype: DType
    order: EvaluationOrder = EvaluationOrder.ORDER_FREE

    def __post_init__(self) -> None:
        if not self.reduction_axes or not set(self.reduction_axes) <= set(self.input.axes):
            raise ValueError("fold axes must be a nonempty subset of input axes")
        expected_axes = tuple(axis for axis in self.input.axes if axis not in self.reduction_axes)
        if self.output.axes != expected_axes:
            raise ValueError("fold output axes must equal the unreduced input axes in order")


class SelectionOrder(StrEnum):
    """Ordering of valid Selection results."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class SelectionComparison(StrEnum):
    """Comparison semantics used by Selection."""

    NUMERIC = "numeric"
    TOTAL_ORDER = "total_order"


class SelectionTieBreak(StrEnum):
    """Tie-breaking rule among equal Selection keys."""

    SOURCE_INDEX_ASCENDING = "source_index_ascending"


class UnderfilledSelectionPolicy(StrEnum):
    """Observable representation of an underfilled Selection."""

    EXPLICIT_INVALID_SLOTS = "explicit_invalid_slots"


@dataclass(frozen=True)
class SelectionPrimitive:
    """A bounded ordered selection over one logical tensor dimension."""

    name: str
    score: ProgramValue
    values: ProgramValue
    indices: ProgramValue
    dimension: TensorAxis
    count: int
    validity: ProgramValue | None = None
    output_validity: ProgramValue | None = None
    order: SelectionOrder = SelectionOrder.DESCENDING
    comparison: SelectionComparison = SelectionComparison.TOTAL_ORDER
    stable: bool = True
    tie_break: SelectionTieBreak = SelectionTieBreak.SOURCE_INDEX_ASCENDING
    underfilled_policy: UnderfilledSelectionPolicy = UnderfilledSelectionPolicy.EXPLICIT_INVALID_SLOTS
    invalid_index: int = -1
    invalid_value: float = float("-inf")
    evaluation_order: EvaluationOrder = EvaluationOrder.SOURCE_ORDERED

    def __post_init__(self) -> None:
        if self.dimension not in self.score.axes:
            raise ValueError("Selection dimension must occur in the score operand")
        if self.count <= 0 or self.count > self.dimension.extent:
            raise ValueError("Selection count must be positive and no larger than its dimension")
        if self.values.dtype != self.score.dtype:
            raise ValueError("Selection values must have the score dtype")
        if self.score.dtype not in {DType.BF16, DType.FP32, DType.FP64}:
            raise ValueError("Selection scores must be floating point")
        if self.indices.dtype is not DType.INT32:
            raise ValueError("Selection indices must be INT32")
        if self.validity is not None and (
            self.validity.axes != self.score.axes or self.validity.dtype is not DType.BOOL
        ):
            raise ValueError("Selection validity must be a Boolean tensor with the score axes")
        if len(self.values.axes) != len(self.score.axes):
            raise ValueError("Selection value rank must equal the score rank")
        expected_axes = tuple(
            output_axis if axis == self.dimension else axis
            for axis, output_axis in zip(self.score.axes, self.values.axes, strict=True)
        )
        selected_axis_index = self.score.axes.index(self.dimension)
        selected_axis = self.values.axes[selected_axis_index]
        if selected_axis.extent != self.count:
            raise ValueError("Selection output dimension must have the selected count extent")
        for index, (source_axis, result_axis) in enumerate(zip(self.score.axes, self.values.axes, strict=True)):
            if index != selected_axis_index and source_axis != result_axis:
                raise ValueError("Selection must preserve every non-selected logical axis")
        if self.indices.axes != expected_axes:
            raise ValueError("Selection values and indices must have identical logical axes")
        if self.output_validity is not None:
            if self.output_validity.axes != expected_axes or self.output_validity.dtype is not DType.BOOL:
                raise ValueError("Selection output validity must be Boolean with the result axes")
        if self.invalid_index >= 0:
            raise ValueError("the explicit invalid Selection index must be negative")
        if not math.isfinite(self.invalid_value) and not math.isinf(self.invalid_value):
            raise ValueError("Selection invalid values cannot be NaN")

    @property
    def inputs(self) -> tuple[ProgramValue, ...]:
        """Return score and optional validity operands."""
        if self.validity is None:
            return (self.score,)
        return (self.score, self.validity)

    @property
    def outputs(self) -> tuple[ProgramValue, ...]:
        """Return value, source-index, and optional validity results."""
        if self.output_validity is None:
            return (self.values, self.indices)
        return (self.values, self.indices, self.output_validity)


@dataclass(frozen=True)
class SelectionCutoffTie:
    """One exact score equivalence class crossing a Selection cutoff."""

    result_coordinate: tuple[int, ...]
    score_dtype: str
    cutoff_score_bits: str
    selected_source_indices: tuple[int, ...]
    excluded_source_indices: tuple[int, ...]


@dataclass(frozen=True)
class RelationPrimitive:
    """Apply typed operand index relations over an explicit logical domain."""

    name: str
    inputs: tuple[ProgramValue, ...]
    outputs: tuple[ProgramValue, ...]
    domain_axes: tuple[TensorAxis, ...]
    input_indexing: tuple[OperandIndexing, ...]
    result_expressions: tuple[IndexExpression | IndexPredicate | None, ...] = ()
    result_sources: tuple[ProgramValue | None, ...] = ()
    predicate: IndexPredicate | None = None
    order: EvaluationOrder = EvaluationOrder.ORDER_FREE

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("Relation requires at least one output")
        if len(set(self.domain_axes)) != len(self.domain_axes):
            raise ValueError("Relation domain axes must be unique")
        if tuple(indexing.operand for indexing in self.input_indexing) != self.inputs:
            raise ValueError("Relation input-indexing records must correspond to inputs in order")
        if any(indexing.domain_axes != self.domain_axes for indexing in self.input_indexing):
            raise ValueError("Relation operand indexing must use the declared domain")
        if any(output.axes != self.domain_axes for output in self.outputs):
            raise ValueError("Relation outputs must be indexed by the complete declared domain")
        if self.result_expressions and len(self.result_expressions) != len(self.outputs):
            raise ValueError("Relation result expressions must correspond to outputs in order")
        if self.result_sources and len(self.result_sources) != len(self.outputs):
            raise ValueError("Relation result sources must correspond to outputs in order")
        expressions = self.resolved_result_expressions
        sources = self.resolved_result_sources
        indexed_operands = {indexing.operand for indexing in self.input_indexing}
        for output, expression, source in zip(self.outputs, expressions, sources, strict=True):
            if (expression is None) == (source is None):
                raise ValueError("each Relation result requires exactly one expression or indexed source")
            if source is not None and (source not in self.inputs or source not in indexed_operands):
                raise ValueError("Relation result sources must be declared indexed inputs")
            if source is not None and source.dtype != output.dtype:
                raise ValueError("Relation result source and output dtypes must match")
            if isinstance(expression, IndexPredicate) and output.dtype is not DType.BOOL:
                raise ValueError("Relation predicate results require a Boolean output")
            if isinstance(expression, IndexExpression) and output.dtype not in {DType.INT32, DType.BOOL}:
                raise ValueError("Relation index-expression results require an INT32 or Boolean output")
        referenced = set().union(*(_indexing_tensors(indexing) for indexing in self.input_indexing), set())
        if self.predicate is not None:
            referenced.update(_index_predicate_tensors(self.predicate))
        for expression in self.result_expressions:
            if isinstance(expression, IndexExpression):
                referenced.update(_index_expression_tensors(expression))
            elif isinstance(expression, IndexPredicate):
                referenced.update(_index_predicate_tensors(expression))
        if not referenced <= set(self.inputs):
            raise ValueError("Relation indexing references a tensor that is not a declared input")

    @property
    def resolved_result_expressions(self) -> tuple[IndexExpression | IndexPredicate | None, ...]:
        """Return one explicit expression slot per result."""
        return self.result_expressions or (None,) * len(self.outputs)

    @property
    def resolved_result_sources(self) -> tuple[ProgramValue | None, ...]:
        """Resolve omitted result sources only when dtype makes the source unique."""
        if self.result_sources:
            return self.result_sources
        sources: list[ProgramValue | None] = []
        for output, expression in zip(self.outputs, self.resolved_result_expressions, strict=True):
            if expression is not None:
                sources.append(None)
                continue
            candidates = tuple(value for value in self.inputs if value.dtype == output.dtype)
            if len(candidates) != 1:
                raise ValueError(
                    f"Relation result {output.name!r} has {len(candidates)} compatible indexed sources; "
                    "bind result_sources explicitly"
                )
            sources.append(candidates[0])
        return tuple(sources)


def _indexing_tensors(indexing: OperandIndexing) -> set[ProgramValue]:
    tensors: set[ProgramValue] = set()
    for expression in indexing.indices:
        tensors.update(_index_expression_tensors(expression))
    if indexing.predicate is not None:
        tensors.update(_index_predicate_tensors(indexing.predicate))
    return tensors


class ScanDirection(StrEnum):
    """Traversal direction along a Scan dimension."""

    FORWARD = "forward"
    REVERSE = "reverse"


@dataclass(frozen=True)
class ScanStateBinding:
    """Bind one outer carried state to its body argument/result and final result."""

    initial: ProgramValue
    body_argument: ProgramValue
    body_result: ProgramValue
    final: ProgramValue


@dataclass(frozen=True)
class ScanInvariantBinding:
    """Bind one loop-invariant outer tensor to a body argument."""

    source: ProgramValue
    body_argument: ProgramValue


@dataclass(frozen=True)
class ScanInputBinding:
    """Bind one outer scanned tensor to a body argument."""

    source: ProgramValue
    body_argument: ProgramValue
    indexing: OperandIndexing

    def __post_init__(self) -> None:
        if self.indexing.operand != self.source:
            raise ValueError("Scan input indexing must index its source")


@dataclass(frozen=True)
class ScanOutputBinding:
    """Bind one body-emitted value to an outer result tensor."""

    body_result: ProgramValue
    destination: ProgramValue
    indexing: OperandIndexing

    def __post_init__(self) -> None:
        if self.indexing.operand != self.destination:
            raise ValueError("Scan output indexing must index its destination")


@dataclass(frozen=True)
class ScanPrimitive:
    """An explicit state-carrying ordered recurrence with a generic algebra body."""

    name: str
    axis: TensorAxis
    body: "TensorProgram"
    state: tuple[ScanStateBinding, ...]
    scanned_inputs: tuple[ScanInputBinding, ...]
    emitted_outputs: tuple[ScanOutputBinding, ...]
    invariants: tuple[ScanInvariantBinding, ...] = ()
    direction: ScanDirection = ScanDirection.FORWARD
    order: EvaluationOrder = EvaluationOrder.SOURCE_ORDERED

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("Scan requires at least one carried state")
        body_inputs = set(self.body.inputs)
        body_outputs = set(self.body.outputs)
        for binding in self.state:
            if binding.body_argument not in body_inputs or binding.body_result not in body_outputs:
                raise ValueError("Scan state bindings must name body inputs and outputs")
            state_values = (binding.initial, binding.body_argument, binding.body_result, binding.final)
            if len({value.dtype for value in state_values}) != 1 or len({value.shape for value in state_values}) != 1:
                raise ValueError("Scan carried-state shapes and dtypes must agree across the body boundary")
        for binding in self.invariants:
            if binding.body_argument not in body_inputs:
                raise ValueError("Scan invariant bindings must name a body input")
            if (
                binding.source.shape != binding.body_argument.shape
                or binding.source.dtype is not binding.body_argument.dtype
            ):
                raise ValueError("Scan invariant source and body-argument shapes and dtypes must agree")
        for binding in self.scanned_inputs:
            if binding.body_argument not in body_inputs:
                raise ValueError("Scan input bindings must name a body input")
            self._validate_slice_binding(binding.indexing, binding.body_argument, "input")
        for binding in self.emitted_outputs:
            if binding.body_result not in body_outputs:
                raise ValueError("Scan output bindings must name a body output")
            self._validate_slice_binding(binding.indexing, binding.body_result, "output")
            if binding.destination.dtype is not binding.body_result.dtype:
                raise ValueError("Scan emitted body and destination dtypes must match")
        body_input_bindings = (
            tuple(binding.body_argument for binding in self.state)
            + tuple(binding.body_argument for binding in self.invariants)
            + tuple(binding.body_argument for binding in self.scanned_inputs)
        )
        if len(set(body_input_bindings)) != len(body_input_bindings) or set(body_input_bindings) != body_inputs:
            raise ValueError("Scan bindings must cover every body input exactly")
        body_output_bindings = tuple(binding.body_result for binding in self.state) + tuple(
            binding.body_result for binding in self.emitted_outputs
        )
        if set(body_output_bindings) != body_outputs:
            raise ValueError("Scan bindings must cover every body output")
        declared_outer_inputs = (
            {binding.initial for binding in self.state}
            | {binding.source for binding in self.invariants}
            | {binding.source for binding in self.scanned_inputs}
        )
        for binding in (*self.scanned_inputs, *self.emitted_outputs):
            if not _indexing_tensors(binding.indexing) <= declared_outer_inputs:
                raise ValueError("Scan indexing references an undeclared outer input")

    def _validate_slice_binding(
        self,
        indexing: OperandIndexing,
        body_value: ProgramValue,
        description: str,
    ) -> None:
        if self.axis not in indexing.domain_axes:
            raise ValueError(f"Scan {description} indexing domain must contain the scan axis")
        slice_shape = tuple(axis.extent for axis in indexing.domain_axes if axis != self.axis)
        if slice_shape != body_value.shape:
            raise ValueError(
                f"Scan {description} indexing slice shape {slice_shape} does not match body shape {body_value.shape}"
            )

    @property
    def inputs(self) -> tuple[ProgramValue, ...]:
        """Return invariant tensors, initial states, then scanned tensors."""
        return (
            tuple(binding.source for binding in self.invariants)
            + tuple(binding.initial for binding in self.state)
            + tuple(binding.source for binding in self.scanned_inputs)
        )

    @property
    def outputs(self) -> tuple[ProgramValue, ...]:
        """Return final states followed by emitted tensors."""
        return tuple(binding.final for binding in self.state) + tuple(
            binding.destination for binding in self.emitted_outputs
        )


@dataclass(frozen=True)
class TransportPrimitive:
    """Move a payload between logical placements without applying a reduction."""

    name: str
    source: ProgramValue
    destination: ProgramValue
    routing_inputs: tuple[ProgramValue, ...]
    domain_axes: tuple[TensorAxis, ...]
    source_indexing: OperandIndexing
    destination_indexing: OperandIndexing
    predicate: IndexPredicate | None = None
    order: EvaluationOrder = EvaluationOrder.SOURCE_ORDERED

    def __post_init__(self) -> None:
        if self.source.dtype != self.destination.dtype:
            raise ValueError("Transport source and destination dtypes must match")
        if self.source_indexing.operand != self.source:
            raise ValueError("Transport source indexing must index the source")
        if self.destination_indexing.operand != self.destination:
            raise ValueError("Transport destination indexing must index the destination")
        if self.source_indexing.domain_axes != self.domain_axes:
            raise ValueError("Transport source indexing must use the transport domain")
        if self.destination_indexing.domain_axes != self.domain_axes:
            raise ValueError("Transport destination indexing must use the transport domain")
        declared = {self.source, *self.routing_inputs}
        referenced = _indexing_tensors(self.source_indexing) | _indexing_tensors(self.destination_indexing)
        if self.predicate is not None:
            referenced |= _index_predicate_tensors(self.predicate)
        if not referenced <= declared:
            raise ValueError("Transport indexing references an undeclared routing input")

    @property
    def inputs(self) -> tuple[ProgramValue, ...]:
        """Return payload followed by dynamic routing operands."""
        return (self.source, *self.routing_inputs)

    @property
    def outputs(self) -> tuple[ProgramValue, ...]:
        """Return the transported payload result."""
        return (self.destination,)


TensorPrimitive = (
    ContractPrimitive
    | MapPrimitive
    | FoldPrimitive
    | SelectionPrimitive
    | RelationPrimitive
    | ScanPrimitive
    | TransportPrimitive
)


@dataclass(frozen=True)
class TensorProgram:
    """A closed acyclic program in the generic Shuttle tensor algebra."""

    inputs: tuple[ProgramValue, ...]
    operations: tuple[TensorPrimitive, ...]
    outputs: tuple[ProgramValue, ...]
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED

    def __post_init__(self) -> None:
        available = {value.name: value for value in self.inputs}
        if len(available) != len(self.inputs):
            raise ValueError("tensor program input names must be unique")
        for operation in self.operations:
            operation_inputs = primitive_inputs(operation)
            missing = tuple(value.name for value in operation_inputs if value.name not in available)
            if missing:
                raise ValueError(f"operation {operation.name!r} reads unavailable values {missing}")
            incompatible = tuple(value.name for value in operation_inputs if available[value.name] != value)
            if incompatible:
                raise ValueError(f"operation {operation.name!r} reads values with incompatible types {incompatible}")
            for result in primitive_outputs(operation):
                if result.name in available:
                    raise ValueError(f"tensor value {result.name!r} has multiple definitions")
                available[result.name] = result
        missing_outputs = tuple(
            value.name for value in self.outputs if value.name not in available or available[value.name] != value
        )
        if missing_outputs:
            raise ValueError(f"tensor program outputs are unavailable: {missing_outputs}")

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the canonical, alpha-renamed semantic record."""
        return canonical_tensor_program_record(self)

    @property
    def semantic_fingerprint(self) -> str:
        """Return the SHA-256 identity of the canonical semantic record."""
        return tensor_program_fingerprint(self)


def primitive_inputs(operation: TensorPrimitive) -> tuple[ProgramValue, ...]:
    """Return the tensor operands consumed by one primitive."""
    if isinstance(operation, FoldPrimitive):
        return (operation.input,)
    return operation.inputs


def primitive_outputs(operation: TensorPrimitive) -> tuple[ProgramValue, ...]:
    """Return tensor results in their stable result-ordinal order."""
    if isinstance(operation, (ContractPrimitive, MapPrimitive, FoldPrimitive)):
        return (operation.output,)
    return operation.outputs


def tensor_program_value_ordinals(program: TensorProgram) -> dict[ProgramValue, int]:
    """Map values to stable ordinals: inputs first, then primitive results."""
    values = (*program.inputs, *(value for operation in program.operations for value in primitive_outputs(operation)))
    return {value: ordinal for ordinal, value in enumerate(values)}


def execute_tensor_program(program: TensorProgram, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Execute generic tensor algebra as a small, deterministic NumPy reference."""
    environment: dict[ProgramValue, np.ndarray] = {}
    for value in program.inputs:
        if value.name not in inputs:
            raise ValueError(f"missing tensor-program input {value.name!r}")
        array = np.asarray(inputs[value.name])
        if array.shape != value.shape:
            raise ValueError(f"tensor-program input {value.name!r} has shape {array.shape}, expected {value.shape}")
        environment[value] = array
    for operation in program.operations:
        results = _execute_primitive(operation, environment)
        expected = primitive_outputs(operation)
        if len(results) != len(expected):
            raise ValueError(f"primitive {operation.name!r} returned the wrong number of results")
        for value, result in zip(expected, results, strict=True):
            array = np.asarray(result)
            if array.shape != value.shape:
                raise ValueError(
                    f"primitive {operation.name!r} result {value.name!r} has shape {array.shape}, expected {value.shape}"
                )
            environment[value] = array.astype(_numpy_dtype(value.dtype), copy=False)
    return {value.name: environment[value] for value in program.outputs}


def _execute_primitive(
    operation: TensorPrimitive,
    environment: Mapping[ProgramValue, np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(operation, ContractPrimitive):
        arrays: list[np.ndarray] = []
        axes: list[tuple[TensorAxis, ...]] = []
        accumulation_dtype = _numpy_dtype(operation.accumulation_dtype)
        if accumulation_dtype == np.dtype(ml_dtypes.bfloat16):
            accumulation_dtype = np.dtype(np.float32)
        for input_index, value in enumerate(operation.inputs):
            array = environment[value].astype(accumulation_dtype, copy=False)
            value_axes = list(value.axes)
            for mapping in operation.index_maps_for_input(input_index):
                dimension = value_axes.index(mapping.operand_axis)
                array = np.take(array, mapping.indices(), axis=dimension)
                value_axes[dimension] = mapping.domain_axis
            arrays.append(array)
            axes.append(tuple(value_axes))
        labels = {
            axis: index for index, axis in enumerate(dict.fromkeys((*operation.output.axes, *operation.reduction_axes)))
        }
        arguments: list[Any] = []
        for array, value_axes in zip(arrays, axes, strict=True):
            arguments.extend((array, [labels[axis] for axis in value_axes]))
        arguments.append([labels[axis] for axis in operation.output.axes])
        result = np.einsum(*arguments, optimize=False)
        return (np.asarray(result),)
    if isinstance(operation, MapPrimitive):
        scalar_inputs = {
            value.name: _broadcast_value(environment[value], value.axes, operation.output.axes)
            for value in operation.inputs
        }
        result = evaluate_scalar_expression(operation.expression, scalar_inputs)
        result = np.broadcast_to(np.asarray(result), operation.output.shape)
        if operation.predicate is not None:
            coordinates = _domain_coordinates(operation.output.axes)
            allowed = _evaluate_index_predicate(operation.predicate, coordinates, environment)
            assert allowed is not None and operation.rejected_value is not None
            result = np.where(allowed, result, operation.rejected_value)
        return (result,)
    if isinstance(operation, FoldPrimitive):
        array = environment[operation.input].astype(_numpy_dtype(operation.accumulation_dtype), copy=False)
        dimensions = tuple(operation.input.axes.index(axis) for axis in operation.reduction_axes)
        if operation.reducer is FoldReducer.SUM:
            return (np.sum(array, axis=dimensions, dtype=_numpy_dtype(operation.accumulation_dtype)),)
        return (np.max(array, axis=dimensions),)
    if isinstance(operation, SelectionPrimitive):
        return _execute_selection(operation, environment)
    if isinstance(operation, RelationPrimitive):
        return _execute_relation(operation, environment)
    if isinstance(operation, ScanPrimitive):
        return _execute_scan(operation, environment)
    assert isinstance(operation, TransportPrimitive)
    return (_execute_transport(operation, environment),)


def _numpy_dtype(dtype: DType) -> Any:
    return {
        DType.BOOL: np.bool_,
        DType.BF16: ml_dtypes.bfloat16,
        DType.FP32: np.float32,
        DType.FP64: np.float64,
        DType.INT32: np.int32,
    }[dtype]


def _broadcast_value(
    array: np.ndarray,
    source_axes: tuple[TensorAxis, ...],
    output_axes: tuple[TensorAxis, ...],
) -> np.ndarray:
    positions = [output_axes.index(axis) for axis in source_axes]
    permutation = tuple(index for index, _ in sorted(enumerate(positions), key=lambda item: item[1]))
    ordered = np.transpose(array, permutation) if permutation != tuple(range(array.ndim)) else array
    ordered_positions = sorted(positions)
    shape = [1] * len(output_axes)
    for dimension, position in enumerate(ordered_positions):
        shape[position] = ordered.shape[dimension]
    return np.broadcast_to(ordered.reshape(shape), tuple(axis.extent for axis in output_axes))


def _execute_selection(
    operation: SelectionPrimitive,
    environment: Mapping[ProgramValue, np.ndarray],
) -> tuple[np.ndarray, ...]:
    score = np.moveaxis(environment[operation.score], operation.score.axes.index(operation.dimension), -1)
    valid = (
        np.ones(score.shape, dtype=np.bool_)
        if operation.validity is None
        else np.moveaxis(environment[operation.validity], operation.score.axes.index(operation.dimension), -1)
    )
    flat_score = score.reshape(-1, score.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    selected_values = np.full((flat_score.shape[0], operation.count), operation.invalid_value, dtype=score.dtype)
    selected_indices = np.full((flat_score.shape[0], operation.count), operation.invalid_index, dtype=np.int32)
    selected_validity = np.zeros((flat_score.shape[0], operation.count), dtype=np.bool_)
    reverse = operation.order is SelectionOrder.DESCENDING
    for row in range(flat_score.shape[0]):
        legal = np.flatnonzero(flat_valid[row])
        if operation.comparison is SelectionComparison.NUMERIC and np.any(np.isnan(flat_score[row, legal])):
            raise ValueError("numeric Selection comparison does not define an order for NaN")

        def score_key(index: int, scores: np.ndarray = flat_score[row]) -> Any:
            if operation.comparison is SelectionComparison.TOTAL_ORDER:
                return _total_order_key(scores[index])
            return float(scores[index])

        ordered = sorted(
            legal.tolist(),
            key=lambda index: (score_key(index), -index if reverse else index),
            reverse=reverse,
        )[: operation.count]
        selected_values[row, : len(ordered)] = flat_score[row, ordered]
        selected_indices[row, : len(ordered)] = ordered
        selected_validity[row, : len(ordered)] = True
    result_shape = (*score.shape[:-1], operation.count)
    dimension = operation.score.axes.index(operation.dimension)
    outputs = (
        np.moveaxis(selected_values.reshape(result_shape), -1, dimension),
        np.moveaxis(selected_indices.reshape(result_shape), -1, dimension),
    )
    if operation.output_validity is None:
        return outputs
    return (*outputs, np.moveaxis(selected_validity.reshape(result_shape), -1, dimension))


def _total_order_key(value: Any) -> tuple[int, int]:
    scalar = np.asarray(value)
    if scalar.dtype.itemsize <= 4:
        bits = int(np.asarray(scalar, dtype=np.float32).view(np.uint32))
        sign = 1 << 31
        return (0, (~bits & 0xFFFFFFFF) if bits & sign else bits | sign)
    bits = int(np.asarray(scalar, dtype=np.float64).view(np.uint64))
    sign = 1 << 63
    return (0, (~bits & 0xFFFFFFFFFFFFFFFF) if bits & sign else bits | sign)


def selection_cutoff_ties(
    operation: SelectionPrimitive,
    score: np.ndarray,
    selected_indices: np.ndarray,
    *,
    validity: np.ndarray | None = None,
    selected_validity: np.ndarray | None = None,
) -> tuple[SelectionCutoffTie, ...]:
    """Report exact score ties split by a bounded Selection cutoff.

    Rows without an equivalence class containing both selected and excluded
    source indices are omitted. The score bit pattern keeps signed zero and
    NaN payload distinctions visible when total-order comparison is requested.
    """
    score = np.asarray(score)
    selected_indices = np.asarray(selected_indices)
    expected_score_shape = tuple(axis.extent for axis in operation.score.axes)
    expected_selected_shape = tuple(axis.extent for axis in operation.indices.axes)
    if score.shape != expected_score_shape:
        raise ValueError(f"Selection score has shape {score.shape}, expected {expected_score_shape}")
    if selected_indices.shape != expected_selected_shape:
        raise ValueError(f"Selection indices have shape {selected_indices.shape}, expected {expected_selected_shape}")
    validity = np.ones(score.shape, dtype=np.bool_) if validity is None else np.asarray(validity, dtype=np.bool_)
    if validity.shape != score.shape:
        raise ValueError("Selection validity must have the score shape")
    selected_validity = (
        selected_indices != operation.invalid_index
        if selected_validity is None
        else np.asarray(selected_validity, dtype=np.bool_)
    )
    if selected_validity.shape != selected_indices.shape:
        raise ValueError("Selection output validity must have the selected-index shape")

    dimension = operation.score.axes.index(operation.dimension)
    flat_score = np.moveaxis(score, dimension, -1).reshape(-1, operation.dimension.extent)
    flat_validity = np.moveaxis(validity, dimension, -1).reshape(-1, operation.dimension.extent)
    flat_indices = np.moveaxis(selected_indices, dimension, -1).reshape(-1, operation.count)
    flat_selected_validity = np.moveaxis(selected_validity, dimension, -1).reshape(-1, operation.count)
    row_shape = tuple(extent for index, extent in enumerate(expected_score_shape) if index != dimension)
    reports = []
    for row, (scores, legal_mask, indices, output_validity) in enumerate(
        zip(flat_score, flat_validity, flat_indices, flat_selected_validity, strict=True)
    ):
        selected = tuple(int(index) for index in indices[output_validity])
        if not selected:
            continue
        if any(index < 0 or index >= operation.dimension.extent for index in selected):
            raise ValueError("valid Selection output contains an out-of-domain source index")
        cutoff_index = selected[-1]
        if operation.comparison is SelectionComparison.TOTAL_ORDER:
            cutoff_key = _total_order_key(scores[cutoff_index])
            tied = tuple(
                int(index) for index in np.flatnonzero(legal_mask) if _total_order_key(scores[index]) == cutoff_key
            )
        else:
            cutoff = scores[cutoff_index]
            if np.isnan(cutoff):
                raise ValueError("numeric Selection comparison does not define an order for NaN")
            tied = tuple(int(index) for index in np.flatnonzero(legal_mask) if scores[index] == cutoff)
        selected_set = set(selected)
        selected_tied = tuple(index for index in selected if index in tied)
        excluded_tied = tuple(index for index in tied if index not in selected_set)
        if not selected_tied or not excluded_tied:
            continue
        reports.append(
            SelectionCutoffTie(
                result_coordinate=tuple(int(index) for index in np.unravel_index(row, row_shape)),
                score_dtype=score.dtype.str,
                cutoff_score_bits=np.asarray(scores[cutoff_index]).tobytes().hex(),
                selected_source_indices=selected_tied,
                excluded_source_indices=excluded_tied,
            )
        )
    return tuple(reports)


def _execute_relation(
    operation: RelationPrimitive,
    environment: Mapping[ProgramValue, np.ndarray],
) -> tuple[np.ndarray, ...]:
    coordinates = _domain_coordinates(operation.domain_axes)
    allowed = _evaluate_index_predicate(operation.predicate, coordinates, environment)
    gathered = {
        indexing.operand: _gather_operand(indexing, coordinates, environment, allowed)
        for indexing in operation.input_indexing
    }
    expressions = operation.resolved_result_expressions
    sources = operation.resolved_result_sources
    results: list[np.ndarray] = []
    for output, expression, source in zip(operation.outputs, expressions, sources, strict=True):
        if isinstance(expression, IndexExpression):
            value = _evaluate_index_expression(expression, coordinates, environment)
        elif isinstance(expression, IndexPredicate):
            value = _evaluate_index_predicate(expression, coordinates, environment)
        else:
            assert source is not None
            value = gathered[source]
        result = np.broadcast_to(np.asarray(value), output.shape)
        if allowed is not None:
            result = np.where(allowed, result, np.zeros((), dtype=result.dtype))
        results.append(result)
    return tuple(results)


def _domain_coordinates(domain_axes: tuple[TensorAxis, ...]) -> dict[TensorAxis, np.ndarray]:
    shape = tuple(axis.extent for axis in domain_axes)
    return {
        axis: np.broadcast_to(
            np.arange(axis.extent, dtype=np.int32).reshape(
                tuple(axis.extent if position == index else 1 for position in range(len(domain_axes)))
            ),
            shape,
        )
        for index, axis in enumerate(domain_axes)
    }


def _gather_operand(
    indexing: OperandIndexing,
    coordinates: Mapping[TensorAxis, np.ndarray],
    environment: Mapping[ProgramValue, np.ndarray],
    additional_allowed: Any | None = None,
) -> np.ndarray:
    domain_shape = tuple(axis.extent for axis in indexing.domain_axes)
    indices = tuple(
        np.broadcast_to(_evaluate_index_expression(index, coordinates, environment), domain_shape)
        for index in indexing.indices
    )
    allowed = _evaluate_index_predicate(indexing.predicate, coordinates, environment)
    if additional_allowed is not None:
        allowed = additional_allowed if allowed is None else np.logical_and(allowed, additional_allowed)
    if allowed is not None:
        mask = np.broadcast_to(np.asarray(allowed, dtype=np.bool_), domain_shape)
        indices = tuple(np.where(mask, index, 0) for index in indices)
    gathered = environment[indexing.operand][indices]
    if allowed is not None:
        gathered = np.where(allowed, gathered, np.zeros((), dtype=gathered.dtype))
    return gathered


def _evaluate_index_expression(
    expression: IndexExpression,
    coordinates: Mapping[TensorAxis, np.ndarray],
    environment: Mapping[ProgramValue, np.ndarray],
) -> Any:
    if expression.kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return coordinates[expression.axis]
    if expression.kind is IndexExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    operands = tuple(_evaluate_index_expression(value, coordinates, environment) for value in expression.operands)
    if expression.kind is IndexExpressionKind.TENSOR_ELEMENT:
        assert expression.tensor is not None
        return environment[expression.tensor][operands]
    if expression.kind is IndexExpressionKind.ADD:
        return np.add(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.SUBTRACT:
        return np.subtract(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.MULTIPLY:
        return np.multiply(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.FLOOR_DIVIDE:
        return np.floor_divide(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.MODULO:
        return np.mod(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.MINIMUM:
        return np.minimum(operands[0], operands[1])
    if expression.kind is IndexExpressionKind.MAXIMUM:
        return np.maximum(operands[0], operands[1])
    assert expression.kind is IndexExpressionKind.SELECT
    assert expression.predicate is not None
    predicate = _evaluate_index_predicate(expression.predicate, coordinates, environment)
    return np.where(predicate, operands[0], operands[1])


def _evaluate_index_predicate(
    predicate: IndexPredicate | None,
    coordinates: Mapping[TensorAxis, np.ndarray],
    environment: Mapping[ProgramValue, np.ndarray],
) -> Any | None:
    if predicate is None:
        return None
    if predicate.kind is IndexPredicateKind.CONSTANT:
        assert predicate.constant is not None
        return predicate.constant
    expressions = tuple(
        _evaluate_index_expression(expression, coordinates, environment) for expression in predicate.expressions
    )
    nested = tuple(_evaluate_index_predicate(value, coordinates, environment) for value in predicate.predicates)
    if predicate.kind is IndexPredicateKind.IS_TRUE:
        return np.asarray(expressions[0], dtype=np.bool_)
    if predicate.kind is IndexPredicateKind.NOT:
        return np.logical_not(nested[0])
    if predicate.kind is IndexPredicateKind.AND:
        return np.logical_and(nested[0], nested[1])
    if predicate.kind is IndexPredicateKind.OR:
        return np.logical_or(nested[0], nested[1])
    if predicate.kind is IndexPredicateKind.EQUAL:
        return np.equal(expressions[0], expressions[1])
    if predicate.kind is IndexPredicateKind.NOT_EQUAL:
        return np.not_equal(expressions[0], expressions[1])
    if predicate.kind is IndexPredicateKind.LESS:
        return np.less(expressions[0], expressions[1])
    if predicate.kind is IndexPredicateKind.LESS_EQUAL:
        return np.less_equal(expressions[0], expressions[1])
    if predicate.kind is IndexPredicateKind.GREATER:
        return np.greater(expressions[0], expressions[1])
    assert predicate.kind is IndexPredicateKind.GREATER_EQUAL
    return np.greater_equal(expressions[0], expressions[1])


def _execute_scan(
    operation: ScanPrimitive,
    environment: Mapping[ProgramValue, np.ndarray],
) -> tuple[np.ndarray, ...]:
    states = [environment[binding.initial] for binding in operation.state]
    emitted = [
        np.zeros(binding.destination.shape, dtype=_numpy_dtype(binding.destination.dtype))
        for binding in operation.emitted_outputs
    ]
    written = [np.zeros(binding.destination.shape, dtype=np.bool_) for binding in operation.emitted_outputs]
    input_gathers = [
        (
            _gather_operand(binding.indexing, _domain_coordinates(binding.indexing.domain_axes), environment),
            binding.indexing.domain_axes.index(operation.axis),
        )
        for binding in operation.scanned_inputs
    ]
    indices = range(operation.axis.extent)
    if operation.direction is ScanDirection.REVERSE:
        indices = reversed(range(operation.axis.extent))
    for index in indices:
        body_inputs: dict[str, np.ndarray] = {
            binding.body_argument.name: state for binding, state in zip(operation.state, states, strict=True)
        }
        body_inputs.update({binding.body_argument.name: environment[binding.source] for binding in operation.invariants})
        for binding, (gathered, dimension) in zip(operation.scanned_inputs, input_gathers, strict=True):
            body_inputs[binding.body_argument.name] = np.take(gathered, index, axis=dimension)
        body_result = execute_tensor_program(operation.body, body_inputs)
        states = [body_result[binding.body_result.name] for binding in operation.state]
        for output_index, binding in enumerate(operation.emitted_outputs):
            coordinates = _domain_coordinates(binding.indexing.domain_axes)
            dimension = binding.indexing.domain_axes.index(operation.axis)
            destination_indices = tuple(
                np.take(
                    np.broadcast_to(
                        _evaluate_index_expression(index_expression, coordinates, environment),
                        tuple(axis.extent for axis in binding.indexing.domain_axes),
                    ),
                    index,
                    axis=dimension,
                ).astype(np.intp, copy=False)
                for index_expression in binding.indexing.indices
            )
            value = body_result[binding.body_result.name]
            allowed = _evaluate_index_predicate(binding.indexing.predicate, coordinates, environment)
            if allowed is not None:
                mask = np.take(
                    np.broadcast_to(allowed, tuple(axis.extent for axis in binding.indexing.domain_axes)),
                    index,
                    axis=dimension,
                )
                destination_indices = tuple(indices[mask] for indices in destination_indices)
                value = value[mask]
            if np.any(written[output_index][destination_indices]):
                raise ValueError("Scan output indexing writes one destination element more than once")
            emitted[output_index][destination_indices] = value
            written[output_index][destination_indices] = True
    if any(not np.all(output_written) for output_written in written):
        raise ValueError("Scan output indexing does not define every destination element")
    return (*states, *emitted)


def _execute_transport(
    operation: TransportPrimitive,
    environment: Mapping[ProgramValue, np.ndarray],
) -> np.ndarray:
    coordinates = _domain_coordinates(operation.domain_axes)
    allowed = _evaluate_index_predicate(operation.predicate, coordinates, environment)
    destination_allowed = _evaluate_index_predicate(operation.destination_indexing.predicate, coordinates, environment)
    if destination_allowed is not None:
        allowed = destination_allowed if allowed is None else np.logical_and(allowed, destination_allowed)
    source = _gather_operand(operation.source_indexing, coordinates, environment, allowed)
    destination_indices = tuple(
        np.broadcast_to(
            np.asarray(_evaluate_index_expression(index, coordinates, environment), dtype=np.intp),
            tuple(axis.extent for axis in operation.domain_axes),
        )
        for index in operation.destination_indexing.indices
    )
    source_allowed = _evaluate_index_predicate(operation.source_indexing.predicate, coordinates, environment)
    if source_allowed is not None:
        allowed = source_allowed if allowed is None else np.logical_and(allowed, source_allowed)
    if allowed is not None:
        mask = np.broadcast_to(np.asarray(allowed, dtype=np.bool_), tuple(axis.extent for axis in operation.domain_axes))
        destination_indices = tuple(index[mask] for index in destination_indices)
        source = source[mask]
    linear = np.ravel_multi_index(destination_indices, operation.destination.shape)
    if np.unique(linear).size != linear.size:
        raise ValueError("Transport destination relation is not injective; use an explicit Fold for reduction")
    destination = np.zeros(operation.destination.shape, dtype=environment[operation.source].dtype)
    destination[destination_indices] = source
    return destination


def scalar_expression_inputs(expression: ScalarExpression) -> set[str]:
    """Return the names referenced by one scalar expression."""
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return {expression.input_name}
    names: set[str] = set()
    for operand in expression.operands:
        names.update(scalar_expression_inputs(operand))
    return names


def canonical_tensor_program_record(program: TensorProgram) -> dict[str, object]:
    """Return a lossless semantic record with incidental names alpha-renamed.

    Input, operation, and result tuple order is semantic. Value ordinals are
    assigned to inputs first and then to each primitive result. Axis ordinals
    are assigned on first structural use. Operation names, value names, axis
    ids, and axis labels are provenance and therefore do not affect identity.
    """
    record, _, _ = _encode_program(program)
    return record


def canonical_operand_indexing_record(
    program: TensorProgram,
    indexing: OperandIndexing,
) -> dict[str, object]:
    """Encode one indexing map in the alpha-renamed context of its program."""
    _, value_ids, axis_ids = _encode_program(program)

    def axis_id(axis: TensorAxis) -> str:
        try:
            return axis_ids[axis]
        except KeyError as error:
            raise ValueError("operand indexing references an axis outside the tensor program") from error

    try:
        return _encode_operand_indexing(indexing, value_ids, axis_id)
    except KeyError as error:
        raise ValueError("operand indexing references a value outside the tensor program") from error


def canonical_axis_references(
    program: TensorProgram,
    axes: tuple[TensorAxis, ...],
) -> tuple[str, ...]:
    """Return alpha-renamed axis references in one program's canonical namespace."""
    _, _, axis_ids = _encode_program(program)
    try:
        return tuple(axis_ids[axis] for axis in axes)
    except KeyError as error:
        raise ValueError("canonical axis reference names an axis outside the tensor program") from error


def serialize_tensor_program(program: TensorProgram) -> str:
    """Serialize a tensor program to canonical semantic JSON."""
    return json.dumps(canonical_tensor_program_record(program), sort_keys=True, separators=(",", ":"), allow_nan=False)


def deserialize_tensor_program(serialized: str) -> TensorProgram:
    """Restore a tensor program from :func:`serialize_tensor_program`."""
    encoded = json.loads(serialized)
    if not isinstance(encoded, dict):
        raise ValueError("serialized tensor program must be a JSON object")
    program, _, _ = _decode_program(encoded)
    return program


def tensor_program_fingerprint(program: TensorProgram) -> str:
    """Return a canonical semantic SHA-256 digest."""
    return hashlib.sha256(serialize_tensor_program(program).encode()).hexdigest()


def _encode_program(
    program: TensorProgram,
) -> tuple[dict[str, object], dict[ProgramValue, str], dict[TensorAxis, str]]:
    value_ids: dict[ProgramValue, str] = {}
    axis_ids: dict[TensorAxis, str] = {}

    def axis_id(axis: TensorAxis) -> str:
        if axis not in axis_ids:
            axis_ids[axis] = f"a{len(axis_ids)}"
        return axis_ids[axis]

    def value_definition(value: ProgramValue) -> dict[str, object]:
        if value in value_ids:
            raise ValueError(f"tensor value {value.name!r} has multiple canonical definitions")
        identifier = f"v{len(value_ids)}"
        value_ids[value] = identifier
        return {
            "id": identifier,
            "axes": [axis_id(axis) for axis in value.axes],
            "dtype": value.dtype.value,
        }

    inputs = [value_definition(value) for value in program.inputs]
    operations: list[dict[str, object]] = []
    for operation in program.operations:
        input_ids = [value_ids[value] for value in primitive_inputs(operation)]
        results = [value_definition(value) for value in primitive_outputs(operation)]
        operation_record: dict[str, object] = {
            "kind": _primitive_kind(operation),
            "inputs": input_ids,
            "results": results,
        }
        _encode_operation_fields(operation, operation_record, value_ids, axis_id)
        operations.append(operation_record)
    axes = [
        {"id": identifier, "extent": axis.extent}
        for axis, identifier in sorted(axis_ids.items(), key=lambda item: int(item[1][1:]))
    ]
    record: dict[str, object] = {
        "format": "shuttle_tensor_program_v1",
        "numerical_policy": program.numerical_policy.value,
        "axes": axes,
        "inputs": inputs,
        "operations": operations,
        "outputs": [value_ids[value] for value in program.outputs],
    }
    return record, value_ids, axis_ids


def _primitive_kind(operation: TensorPrimitive) -> str:
    return {
        ContractPrimitive: "contract",
        MapPrimitive: "map",
        FoldPrimitive: "fold",
        SelectionPrimitive: "selection",
        RelationPrimitive: "relation",
        ScanPrimitive: "scan",
        TransportPrimitive: "transport",
    }[type(operation)]


def _encode_operation_fields(
    operation: TensorPrimitive,
    record: dict[str, object],
    value_ids: dict[ProgramValue, str],
    axis_id: Any,
) -> None:
    if isinstance(operation, ContractPrimitive):
        input_index_maps = operation.input_index_maps or ((),) * len(operation.inputs)
        record.update(
            reduction_axes=[axis_id(axis) for axis in operation.reduction_axes],
            accumulation_dtype=operation.accumulation_dtype.value,
            operand_precision=[precision.value for precision in operation.operand_precision],
            input_index_maps=[
                [
                    {
                        "domain_axis": axis_id(mapping.domain_axis),
                        "operand_axis": axis_id(mapping.operand_axis),
                        "divisor": mapping.divisor,
                        "offset": mapping.offset,
                        "modulus": mapping.modulus,
                    }
                    for mapping in mappings
                ]
                for mappings in input_index_maps
            ],
            order=operation.order.value,
        )
        return
    if isinstance(operation, MapPrimitive):
        record["expression"] = _encode_scalar_expression(operation.expression, value_ids)
        record["output_conversion"] = operation.output_conversion.value
        if operation.predicate is not None:
            assert operation.rejected_value is not None
            record["predicate"] = _encode_index_predicate(operation.predicate, value_ids, axis_id)
            record["rejected_value"] = _encode_number(operation.rejected_value)
        return
    if isinstance(operation, FoldPrimitive):
        record.update(
            reduction_axes=[axis_id(axis) for axis in operation.reduction_axes],
            reducer=operation.reducer.value,
            accumulation_dtype=operation.accumulation_dtype.value,
            order=operation.order.value,
        )
        return
    if isinstance(operation, SelectionPrimitive):
        record.update(
            dimension=axis_id(operation.dimension),
            count=operation.count,
            has_validity=operation.validity is not None,
            has_output_validity=operation.output_validity is not None,
            order=operation.order.value,
            comparison=operation.comparison.value,
            stable=operation.stable,
            tie_break=operation.tie_break.value,
            underfilled_policy=operation.underfilled_policy.value,
            invalid_index=operation.invalid_index,
            invalid_value=_encode_number(operation.invalid_value),
            evaluation_order=operation.evaluation_order.value,
        )
        return
    if isinstance(operation, RelationPrimitive):
        record.update(
            domain_axes=[axis_id(axis) for axis in operation.domain_axes],
            input_indexing=[_encode_operand_indexing(value, value_ids, axis_id) for value in operation.input_indexing],
            result_expressions=[
                _encode_relation_result_expression(expression, value_ids, axis_id)
                for expression in operation.resolved_result_expressions
            ],
            result_sources=[
                None if source is None else value_ids[source] for source in operation.resolved_result_sources
            ],
            predicate=_encode_index_predicate(operation.predicate, value_ids, axis_id),
            order=operation.order.value,
        )
        return
    if isinstance(operation, ScanPrimitive):
        body_record, body_values, _ = _encode_program(operation.body)
        record.update(
            axis=axis_id(operation.axis),
            direction=operation.direction.value,
            order=operation.order.value,
            body=body_record,
            state=[
                {
                    "initial": value_ids[binding.initial],
                    "body_argument": body_values[binding.body_argument],
                    "body_result": body_values[binding.body_result],
                    "final": value_ids[binding.final],
                }
                for binding in operation.state
            ],
            invariants=[
                {
                    "source": value_ids[binding.source],
                    "body_argument": body_values[binding.body_argument],
                }
                for binding in operation.invariants
            ],
            scanned_inputs=[
                {
                    "source": value_ids[binding.source],
                    "body_argument": body_values[binding.body_argument],
                    "indexing": _encode_operand_indexing(binding.indexing, value_ids, axis_id),
                }
                for binding in operation.scanned_inputs
            ],
            emitted_outputs=[
                {
                    "body_result": body_values[binding.body_result],
                    "destination": value_ids[binding.destination],
                    "indexing": _encode_operand_indexing(binding.indexing, value_ids, axis_id),
                }
                for binding in operation.emitted_outputs
            ],
        )
        return
    assert isinstance(operation, TransportPrimitive)
    record.update(
        domain_axes=[axis_id(axis) for axis in operation.domain_axes],
        routing_inputs=[value_ids[value] for value in operation.routing_inputs],
        source_indexing=_encode_operand_indexing(operation.source_indexing, value_ids, axis_id),
        destination_indexing=_encode_operand_indexing(operation.destination_indexing, value_ids, axis_id),
        predicate=_encode_index_predicate(operation.predicate, value_ids, axis_id),
        order=operation.order.value,
    )


def _encode_scalar_expression(
    expression: ScalarExpression,
    value_ids: Mapping[ProgramValue, str],
) -> dict[str, object]:
    names = {value.name: identifier for value, identifier in value_ids.items()}
    record: dict[str, object] = {"kind": expression.kind.value}
    if expression.input_name is not None:
        try:
            record["input"] = names[expression.input_name]
        except KeyError as error:
            raise ValueError(f"scalar expression refers to unknown value {expression.input_name!r}") from error
    if expression.constant is not None:
        record["constant"] = _encode_number(expression.constant)
    if expression.result_dtype is not None:
        record["result_dtype"] = expression.result_dtype.value
    if expression.conversion_semantics is not None:
        record["conversion_semantics"] = expression.conversion_semantics.value
    if expression.operands:
        record["operands"] = [_encode_scalar_expression(operand, value_ids) for operand in expression.operands]
    return record


def _encode_number(value: float | bool | int) -> object:
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if math.isnan(value):
        return {"float": "nan"}
    if math.isinf(value):
        return {"float": "positive_infinity" if value > 0 else "negative_infinity"}
    return value


def _encode_index_expression(
    expression: IndexExpression,
    value_ids: Mapping[ProgramValue, str],
    axis_id: Any,
) -> dict[str, object]:
    record: dict[str, object] = {"kind": expression.kind.value}
    if expression.axis is not None:
        record["axis"] = axis_id(expression.axis)
    if expression.constant is not None:
        record["constant"] = expression.constant
    if expression.tensor is not None:
        record["tensor"] = value_ids[expression.tensor]
    if expression.predicate is not None:
        record["predicate"] = _encode_index_predicate(expression.predicate, value_ids, axis_id)
    if expression.operands:
        record["operands"] = [_encode_index_expression(operand, value_ids, axis_id) for operand in expression.operands]
    return record


def _encode_index_predicate(
    predicate: IndexPredicate | None,
    value_ids: Mapping[ProgramValue, str],
    axis_id: Any,
) -> dict[str, object] | None:
    if predicate is None:
        return None
    record: dict[str, object] = {"kind": predicate.kind.value}
    if predicate.constant is not None:
        record["constant"] = predicate.constant
    if predicate.expressions:
        record["expressions"] = [
            _encode_index_expression(expression, value_ids, axis_id) for expression in predicate.expressions
        ]
    if predicate.predicates:
        record["predicates"] = [_encode_index_predicate(nested, value_ids, axis_id) for nested in predicate.predicates]
    return record


def _encode_operand_indexing(
    indexing: OperandIndexing,
    value_ids: Mapping[ProgramValue, str],
    axis_id: Any,
) -> dict[str, object]:
    return {
        "operand": value_ids[indexing.operand],
        "domain_axes": [axis_id(axis) for axis in indexing.domain_axes],
        "indices": [_encode_index_expression(index, value_ids, axis_id) for index in indexing.indices],
        "predicate": _encode_index_predicate(indexing.predicate, value_ids, axis_id),
    }


def _encode_relation_result_expression(
    expression: IndexExpression | IndexPredicate | None,
    value_ids: Mapping[ProgramValue, str],
    axis_id: Any,
) -> dict[str, object] | None:
    if expression is None:
        return None
    if isinstance(expression, IndexExpression):
        return {"kind": "index", "value": _encode_index_expression(expression, value_ids, axis_id)}
    return {"kind": "predicate", "value": _encode_index_predicate(expression, value_ids, axis_id)}


def _decode_program(
    record: dict[str, object],
) -> tuple[TensorProgram, dict[str, ProgramValue], dict[str, TensorAxis]]:
    if record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("unsupported tensor-program serialization format")
    encoded_axes = _record_list(record, "axes")
    axes: dict[str, TensorAxis] = {}
    for ordinal, encoded_axis in enumerate(encoded_axes):
        axis_record = _object_record(encoded_axis, "axis")
        identifier = _record_string(axis_record, "id")
        extent = _record_int(axis_record, "extent")
        if identifier in axes:
            raise ValueError("serialized tensor-program axis ids must be unique")
        axes[identifier] = TensorAxis(ordinal, extent)

    values: dict[str, ProgramValue] = {}

    def define_value(encoded: object) -> ProgramValue:
        value_record = _object_record(encoded, "value")
        identifier = _record_string(value_record, "id")
        if identifier in values:
            raise ValueError("serialized tensor-program value ids must be unique")
        axis_names = _record_list(value_record, "axes")
        try:
            value_axes = tuple(axes[_string(axis, "value axis id")] for axis in axis_names)
        except KeyError as error:
            raise ValueError(f"serialized value references unknown axis {error.args[0]!r}") from error
        value = ProgramValue(identifier, value_axes, DType(_record_string(value_record, "dtype")))
        values[identifier] = value
        return value

    inputs = tuple(define_value(encoded) for encoded in _record_list(record, "inputs"))
    operations: list[TensorPrimitive] = []
    for ordinal, encoded_operation in enumerate(_record_list(record, "operations")):
        operation_record = _object_record(encoded_operation, "operation")
        kind = _record_string(operation_record, "kind")
        input_values = tuple(_value_references(operation_record, "inputs", values))
        results = tuple(define_value(encoded) for encoded in _record_list(operation_record, "results"))
        name = f"op{ordinal}.{kind}"
        operation = _decode_operation(name, kind, operation_record, input_values, results, values, axes)
        operations.append(operation)
    output_values = tuple(_value_references(record, "outputs", values))
    program = TensorProgram(
        inputs=inputs,
        operations=tuple(operations),
        outputs=output_values,
        numerical_policy=NumericalPolicy(_record_string(record, "numerical_policy")),
    )
    return program, values, axes


def _decode_operation(
    name: str,
    kind: str,
    record: dict[str, object],
    inputs: tuple[ProgramValue, ...],
    results: tuple[ProgramValue, ...],
    values: dict[str, ProgramValue],
    axes: dict[str, TensorAxis],
) -> TensorPrimitive:
    if kind == "contract":
        input_index_maps = tuple(
            tuple(
                AxisIndexMap(
                    domain_axis=axes[_record_string(mapping_record, "domain_axis")],
                    operand_axis=axes[_record_string(mapping_record, "operand_axis")],
                    divisor=_record_int(mapping_record, "divisor"),
                    offset=_record_int(mapping_record, "offset"),
                    modulus=_optional_int(mapping_record.get("modulus"), "index-map modulus"),
                )
                for mapping_record in (
                    _object_record(mapping, "axis index map") for mapping in _list(group, "index-map group")
                )
            )
            for group in _record_list(record, "input_index_maps")
        )
        return ContractPrimitive(
            name,
            inputs,
            _single_result(results, kind),
            _axis_references(record, "reduction_axes", axes),
            DType(_record_string(record, "accumulation_dtype")),
            input_index_maps,
            EvaluationOrder(_record_string(record, "order")),
            tuple(ContractOperandPrecision(precision) for precision in _record_list(record, "operand_precision")),
        )
    if kind == "map":
        predicate = _decode_index_predicate(record.get("predicate"), values, axes)
        rejected_value_record = record.get("rejected_value")
        return MapPrimitive(
            name,
            inputs,
            _single_result(results, kind),
            _decode_scalar_expression(_object_record(record.get("expression"), "scalar expression"), values),
            ScalarConversionSemantics(_record_string(record, "output_conversion")),
            predicate,
            None if rejected_value_record is None else _decode_number(rejected_value_record),
        )
    if kind == "fold":
        if len(inputs) != 1:
            raise ValueError("serialized Fold must have one input")
        return FoldPrimitive(
            name,
            inputs[0],
            _single_result(results, kind),
            _axis_references(record, "reduction_axes", axes),
            FoldReducer(_record_string(record, "reducer")),
            DType(_record_string(record, "accumulation_dtype")),
            EvaluationOrder(_record_string(record, "order")),
        )
    if kind == "selection":
        has_validity = _record_bool(record, "has_validity")
        has_output_validity = _record_bool(record, "has_output_validity")
        if len(inputs) != 1 + int(has_validity) or len(results) != 2 + int(has_output_validity):
            raise ValueError("serialized Selection operand/result arity does not match its validity flags")
        return SelectionPrimitive(
            name=name,
            score=inputs[0],
            validity=inputs[1] if has_validity else None,
            values=results[0],
            indices=results[1],
            output_validity=results[2] if has_output_validity else None,
            dimension=axes[_record_string(record, "dimension")],
            count=_record_int(record, "count"),
            order=SelectionOrder(_record_string(record, "order")),
            comparison=SelectionComparison(_record_string(record, "comparison")),
            stable=_record_bool(record, "stable"),
            tie_break=SelectionTieBreak(_record_string(record, "tie_break")),
            underfilled_policy=UnderfilledSelectionPolicy(_record_string(record, "underfilled_policy")),
            invalid_index=_record_int(record, "invalid_index"),
            invalid_value=float(_decode_number(record.get("invalid_value"))),
            evaluation_order=EvaluationOrder(_record_string(record, "evaluation_order")),
        )
    if kind == "relation":
        return RelationPrimitive(
            name=name,
            inputs=inputs,
            outputs=results,
            domain_axes=_axis_references(record, "domain_axes", axes),
            input_indexing=tuple(
                _decode_operand_indexing(_object_record(encoded, "operand indexing"), values, axes)
                for encoded in _record_list(record, "input_indexing")
            ),
            result_expressions=tuple(
                _decode_relation_result_expression(encoded, values, axes)
                for encoded in _record_list(record, "result_expressions")
            ),
            result_sources=tuple(
                None if encoded is None else values[_string(encoded, "Relation result source")]
                for encoded in _record_list(record, "result_sources")
            ),
            predicate=_decode_index_predicate(record.get("predicate"), values, axes),
            order=EvaluationOrder(_record_string(record, "order")),
        )
    if kind == "scan":
        body_record = _object_record(record.get("body"), "Scan body")
        body, body_values, _ = _decode_program(body_record)
        state = tuple(
            ScanStateBinding(
                initial=values[_record_string(binding, "initial")],
                body_argument=body_values[_record_string(binding, "body_argument")],
                body_result=body_values[_record_string(binding, "body_result")],
                final=values[_record_string(binding, "final")],
            )
            for binding in (_object_record(encoded, "Scan state binding") for encoded in _record_list(record, "state"))
        )
        invariants = tuple(
            ScanInvariantBinding(
                source=values[_record_string(binding, "source")],
                body_argument=body_values[_record_string(binding, "body_argument")],
            )
            for binding in (
                _object_record(encoded, "Scan invariant binding") for encoded in _record_list(record, "invariants")
            )
        )
        scanned_inputs = tuple(
            ScanInputBinding(
                source=values[_record_string(binding, "source")],
                body_argument=body_values[_record_string(binding, "body_argument")],
                indexing=_decode_operand_indexing(
                    _object_record(binding.get("indexing"), "Scan input indexing"), values, axes
                ),
            )
            for binding in (
                _object_record(encoded, "Scan input binding") for encoded in _record_list(record, "scanned_inputs")
            )
        )
        emitted_outputs = tuple(
            ScanOutputBinding(
                body_result=body_values[_record_string(binding, "body_result")],
                destination=values[_record_string(binding, "destination")],
                indexing=_decode_operand_indexing(
                    _object_record(binding.get("indexing"), "Scan output indexing"), values, axes
                ),
            )
            for binding in (
                _object_record(encoded, "Scan output binding") for encoded in _record_list(record, "emitted_outputs")
            )
        )
        operation = ScanPrimitive(
            name=name,
            axis=axes[_record_string(record, "axis")],
            body=body,
            state=state,
            scanned_inputs=scanned_inputs,
            emitted_outputs=emitted_outputs,
            invariants=invariants,
            direction=ScanDirection(_record_string(record, "direction")),
            order=EvaluationOrder(_record_string(record, "order")),
        )
        if operation.inputs != inputs or operation.outputs != results:
            raise ValueError("serialized Scan bindings do not match its operand/result lists")
        return operation
    if kind == "transport":
        if not inputs or len(results) != 1:
            raise ValueError("serialized Transport requires a source and one destination")
        operation = TransportPrimitive(
            name=name,
            source=inputs[0],
            destination=results[0],
            routing_inputs=tuple(_value_references(record, "routing_inputs", values)),
            domain_axes=_axis_references(record, "domain_axes", axes),
            source_indexing=_decode_operand_indexing(
                _object_record(record.get("source_indexing"), "Transport source indexing"), values, axes
            ),
            destination_indexing=_decode_operand_indexing(
                _object_record(record.get("destination_indexing"), "Transport destination indexing"), values, axes
            ),
            predicate=_decode_index_predicate(record.get("predicate"), values, axes),
            order=EvaluationOrder(_record_string(record, "order")),
        )
        if operation.inputs != inputs:
            raise ValueError("serialized Transport routing inputs do not match its operands")
        return operation
    raise ValueError(f"unsupported serialized tensor primitive {kind!r}")


def _decode_scalar_expression(record: dict[str, object], values: Mapping[str, ProgramValue]) -> ScalarExpression:
    input_id = record.get("input")
    input_name: str | None = None
    if input_id is not None:
        identifier = _string(input_id, "scalar input id")
        try:
            input_name = values[identifier].name
        except KeyError as error:
            raise ValueError(f"scalar expression references unknown value {identifier!r}") from error
    constant = _decode_number(record["constant"]) if "constant" in record else None
    return ScalarExpression(
        kind=ScalarExpressionKind(_record_string(record, "kind")),
        operands=tuple(
            _decode_scalar_expression(_object_record(encoded, "scalar operand"), values)
            for encoded in _list(record.get("operands", []), "scalar operands")
        ),
        input_name=input_name,
        constant=constant,
        result_dtype=DType(_string(record["result_dtype"], "scalar result dtype")) if "result_dtype" in record else None,
        conversion_semantics=(
            ScalarConversionSemantics(_string(record["conversion_semantics"], "conversion semantics"))
            if "conversion_semantics" in record
            else None
        ),
    )


def _decode_number(encoded: object) -> float | bool | int:
    if isinstance(encoded, (bool, int, float)):
        return encoded
    record = _object_record(encoded, "encoded numeric literal")
    spelling = _record_string(record, "float")
    return {
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": float("-inf"),
    }[spelling]


def _decode_index_expression(
    record: dict[str, object],
    values: Mapping[str, ProgramValue],
    axes: Mapping[str, TensorAxis],
) -> IndexExpression:
    return IndexExpression(
        kind=IndexExpressionKind(_record_string(record, "kind")),
        operands=tuple(
            _decode_index_expression(_object_record(encoded, "index-expression operand"), values, axes)
            for encoded in _list(record.get("operands", []), "index-expression operands")
        ),
        axis=axes[_string(record["axis"], "index-expression axis")] if "axis" in record else None,
        constant=record.get("constant") if isinstance(record.get("constant"), (bool, int)) else None,
        tensor=values[_string(record["tensor"], "index-expression tensor")] if "tensor" in record else None,
        predicate=_decode_index_predicate(record.get("predicate"), values, axes),
    )


def _decode_index_predicate(
    encoded: object,
    values: Mapping[str, ProgramValue],
    axes: Mapping[str, TensorAxis],
) -> IndexPredicate | None:
    if encoded is None:
        return None
    record = _object_record(encoded, "index predicate")
    return IndexPredicate(
        kind=IndexPredicateKind(_record_string(record, "kind")),
        expressions=tuple(
            _decode_index_expression(_object_record(item, "predicate expression"), values, axes)
            for item in _list(record.get("expressions", []), "predicate expressions")
        ),
        predicates=tuple(
            predicate
            for item in _list(record.get("predicates", []), "nested predicates")
            if (predicate := _decode_index_predicate(item, values, axes)) is not None
        ),
        constant=record.get("constant") if isinstance(record.get("constant"), bool) else None,
    )


def _decode_operand_indexing(
    record: dict[str, object],
    values: Mapping[str, ProgramValue],
    axes: Mapping[str, TensorAxis],
) -> OperandIndexing:
    return OperandIndexing(
        operand=values[_record_string(record, "operand")],
        domain_axes=tuple(axes[_string(axis, "indexing domain axis")] for axis in _record_list(record, "domain_axes")),
        indices=tuple(
            _decode_index_expression(_object_record(encoded, "index expression"), values, axes)
            for encoded in _record_list(record, "indices")
        ),
        predicate=_decode_index_predicate(record.get("predicate"), values, axes),
    )


def _decode_relation_result_expression(
    encoded: object,
    values: Mapping[str, ProgramValue],
    axes: Mapping[str, TensorAxis],
) -> IndexExpression | IndexPredicate | None:
    if encoded is None:
        return None
    record = _object_record(encoded, "Relation result expression")
    kind = _record_string(record, "kind")
    if kind == "index":
        return _decode_index_expression(_object_record(record.get("value"), "index result expression"), values, axes)
    if kind == "predicate":
        predicate = _decode_index_predicate(record.get("value"), values, axes)
        if predicate is None:
            raise ValueError("Relation predicate result cannot be null")
        return predicate
    raise ValueError(f"unsupported Relation result-expression kind {kind!r}")


def _object_record(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"serialized {description} must be an object")
    return value


def _record_list(record: Mapping[str, object], key: str) -> list[object]:
    if key not in record:
        raise ValueError(f"serialized record is missing {key!r}")
    return _list(record[key], key)


def _list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"serialized {description} must be a list")
    return value


def _record_string(record: Mapping[str, object], key: str) -> str:
    if key not in record:
        raise ValueError(f"serialized record is missing {key!r}")
    return _string(record[key], key)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"serialized {description} must be a string")
    return value


def _record_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"serialized {key} must be an integer")
    return value


def _optional_int(value: object, description: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"serialized {description} must be an integer or null")
    return value


def _record_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"serialized {key} must be Boolean")
    return value


def _value_references(
    record: Mapping[str, object],
    key: str,
    values: Mapping[str, ProgramValue],
) -> list[ProgramValue]:
    identifiers = (_string(value, f"{key} value id") for value in _record_list(record, key))
    try:
        return [values[identifier] for identifier in identifiers]
    except KeyError as error:
        raise ValueError(f"serialized program references unknown value {error.args[0]!r}") from error


def _axis_references(
    record: Mapping[str, object],
    key: str,
    axes: Mapping[str, TensorAxis],
) -> tuple[TensorAxis, ...]:
    identifiers = (_string(value, f"{key} axis id") for value in _record_list(record, key))
    try:
        return tuple(axes[identifier] for identifier in identifiers)
    except KeyError as error:
        raise ValueError(f"serialized program references unknown axis {error.args[0]!r}") from error


def _single_result(results: tuple[ProgramValue, ...], kind: str) -> ProgramValue:
    if len(results) != 1:
        raise ValueError(f"serialized {kind} must have one result")
    return results[0]
