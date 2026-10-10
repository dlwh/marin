# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonicalize index-generated ``select(value, -inf)`` domains.

This module recognizes a semantic property of generic tensor algebra, not an
attention pattern: an integer-affine predicate selects the finite domain of a
floating-point value and rejects the complement with negative infinity.  The
source :class:`TensorProgram` remains executable and unchanged; the derived
restriction is the physicalizable representation of the same domain.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from shuttle.ir import DType
from tile_lifetime.normalized_exp_contract_region import (
    AffineIndexExpression,
    AffineIndexPredicate,
    IndexComparisonKind,
    NormalizedExpContractRegion,
    normalized_exp_contract_region,
)
from tile_lifetime.tensor_program import (
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    deserialize_tensor_program,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    primitive_inputs,
    primitive_outputs,
)

DOMAIN_POSITION_COORDINATE_BASIS = "domain_position"
TENSOR_AXIS_ID_COORDINATE_BASIS = "tensor_axis_id"


@dataclass(frozen=True)
class DomainCoordinateRemapProof:
    """Proof that portable domain positions were rebound to typed axis ids."""

    source_basis: str
    target_basis: str
    position_to_axis_id: tuple[tuple[int, int], ...]
    domain_extents: tuple[int, ...]


@dataclass(frozen=True)
class IndexDomainRestriction:
    """An exact integer-affine domain recovered from a generic scalar select."""

    masked_value: ProgramValue
    selected_value: ProgramValue
    domain_axes: tuple[TensorAxis, ...]
    streamed_axes: tuple[TensorAxis, ...]
    predicate: IndexPredicate
    affine_predicate: AffineIndexPredicate
    source_fingerprint: str
    select_operation_ordinal: int
    predicate_operation_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.masked_value.axes != self.domain_axes or self.selected_value.axes != self.domain_axes:
            raise ValueError("domain restriction values must use the complete restricted domain")
        if (
            not self.streamed_axes
            or len(set(self.streamed_axes)) != len(self.streamed_axes)
            or not set(self.streamed_axes) <= set(self.domain_axes)
        ):
            raise ValueError("domain restriction streamed axes must be a nonempty domain subset")
        affine_dimensions = {
            dimension
            for expression in (self.affine_predicate.left, self.affine_predicate.right)
            for dimension, _ in expression.coefficients
        }
        domain_axis_ids = {axis.id for axis in self.domain_axes}
        if not affine_dimensions <= domain_axis_ids:
            raise ValueError("domain restriction affine predicate references an axis outside its domain")

    @property
    def portable_affine_predicate(self) -> AffineIndexPredicate:
        """Return the predicate in canonical artifact domain-position coordinates."""
        return affine_predicate_in_domain_position_basis(self.affine_predicate, self.domain_axes)

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return provenance and semantic identity without tensor value names."""
        return {
            "format": "shuttle_index_domain_restriction_v1",
            "source_fingerprint": self.source_fingerprint,
            "domain_rank": len(self.domain_axes),
            "domain_extents": tuple(axis.extent for axis in self.domain_axes),
            "streamed_dimensions": tuple(self.domain_axes.index(axis) for axis in self.streamed_axes),
            "coordinate_basis": DOMAIN_POSITION_COORDINATE_BASIS,
            "predicate": self.portable_affine_predicate.canonical_record,
            "select_operation_ordinal": self.select_operation_ordinal,
            "predicate_operation_ordinals": self.predicate_operation_ordinals,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable digest for this source-derived restriction."""
        encoded = json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class _AffineComparison:
    kind: IndexPredicateKind
    left: AffineIndexExpression
    right: AffineIndexExpression


class NormalizedExpDomainCanonicalizationError(ValueError):
    """The program has no unique supported normalized-exp index domain."""


@dataclass(frozen=True)
class DomainRestrictedNormalizedExpProgram:
    """Executable source plus its canonical domain-restricted task algebra."""

    source: TensorProgram
    task_program: TensorProgram
    restriction: IndexDomainRestriction
    region: NormalizedExpContractRegion

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return canonical algebra identity, excluding source/debug provenance."""
        return {
            "format": "shuttle_domain_restricted_tensor_program_v1",
            "program": self.task_program.canonical_record,
            "domain_restrictions": (
                {
                    "domain_extents": tuple(axis.extent for axis in self.restriction.domain_axes),
                    "streamed_dimensions": tuple(
                        self.restriction.domain_axes.index(axis) for axis in self.restriction.streamed_axes
                    ),
                    "coordinate_basis": DOMAIN_POSITION_COORDINATE_BASIS,
                    "predicate": self.restriction.portable_affine_predicate.canonical_record,
                },
            ),
        }

    @property
    def semantic_fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class CanonicalNormalizedExpTaskAlgebra:
    """Typed task algebra reconstructed from an executable-request record."""

    task_program: TensorProgram
    region: NormalizedExpContractRegion
    domain_coordinate_remap: DomainCoordinateRemapProof


def normalized_exp_task_algebra_from_record(record: Mapping[str, Any]) -> CanonicalNormalizedExpTaskAlgebra:
    """Reconstruct the typed region consumed by a physical adapter."""
    if record.get("format") != "shuttle_domain_restricted_tensor_program_v1":
        raise ValueError("request algebra is not a domain-restricted tensor program")
    encoded_program = record.get("program")
    if not isinstance(encoded_program, Mapping):
        raise ValueError("domain-restricted request must contain one canonical tensor program")
    restrictions = record.get("domain_restrictions")
    if not isinstance(restrictions, (tuple, list)) or len(restrictions) != 1:
        raise ValueError("current normalized-exp adapter requires exactly one domain restriction")
    restriction = restrictions[0]
    if not isinstance(restriction, Mapping):
        raise ValueError("domain restriction must be a canonical mapping")
    task_program = deserialize_tensor_program(json.dumps(encoded_program, sort_keys=True, separators=(",", ":")))
    region = normalized_exp_contract_region(task_program)
    domain_axes = region.score_map.output.axes
    domain_extents = restriction.get("domain_extents")
    if not isinstance(domain_extents, (tuple, list)) or not all(type(extent) is int for extent in domain_extents):
        raise ValueError("domain restriction extents must be an integer sequence")
    if tuple(domain_extents) != tuple(axis.extent for axis in domain_axes):
        raise ValueError("domain restriction extents do not match the reconstructed score domain")
    if restriction.get("coordinate_basis") != DOMAIN_POSITION_COORDINATE_BASIS:
        raise ValueError("domain restriction predicate must use the explicit domain-position coordinate basis")
    portable_predicate = _affine_predicate_from_record(restriction.get("predicate"))
    position_to_axis_id = tuple((position, axis.id) for position, axis in enumerate(domain_axes))
    predicate = _remap_affine_predicate(portable_predicate, dict(position_to_axis_id))
    if (
        region.score_map.predicate is None
        or _affine_predicate(region.score_map.predicate, domain_axes) != portable_predicate
    ):
        raise ValueError("domain predicate record disagrees with the predicated score Map")
    proof = DomainCoordinateRemapProof(
        source_basis=DOMAIN_POSITION_COORDINATE_BASIS,
        target_basis=TENSOR_AXIS_ID_COORDINATE_BASIS,
        position_to_axis_id=position_to_axis_id,
        domain_extents=tuple(domain_extents),
    )
    return CanonicalNormalizedExpTaskAlgebra(task_program, replace(region, domain_predicate=predicate), proof)


def _affine_predicate_from_record(encoded: object) -> AffineIndexPredicate:
    if not isinstance(encoded, Mapping):
        raise ValueError("domain restriction predicate must be a canonical mapping")
    try:
        comparison = IndexComparisonKind(encoded["comparison"])
        left = _affine_expression_from_record(encoded["left"])
        right = _affine_expression_from_record(encoded["right"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("domain restriction predicate record is malformed") from error
    return AffineIndexPredicate(comparison, left, right)


def _affine_expression_from_record(encoded: object) -> AffineIndexExpression:
    if not isinstance(encoded, Mapping):
        raise ValueError("affine index expression must be a canonical mapping")
    coefficients = encoded.get("coefficients")
    constant = encoded.get("constant")
    if not isinstance(coefficients, (tuple, list)) or type(constant) is not int:
        raise ValueError("affine index expression record is malformed")
    decoded: list[tuple[int, int]] = []
    for item in coefficients:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or type(item[0]) is not int or type(item[1]) is not int:
            raise ValueError("affine index coefficient must be an integer pair")
        decoded.append((item[0], item[1]))
    return AffineIndexExpression(tuple(decoded), constant)


def _remap_affine_predicate(
    predicate: AffineIndexPredicate,
    dimension_map: Mapping[int, int],
) -> AffineIndexPredicate:
    def remap(expression: AffineIndexExpression) -> AffineIndexExpression:
        missing = tuple(dimension for dimension, _ in expression.coefficients if dimension not in dimension_map)
        if missing:
            raise ValueError(f"affine predicate references dimensions outside its coordinate basis: {missing}")
        coefficients: dict[int, int] = {}
        for dimension, coefficient in expression.coefficients:
            mapped = dimension_map[dimension]
            coefficients[mapped] = coefficients.get(mapped, 0) + coefficient
        return AffineIndexExpression(
            tuple(sorted((dimension, coefficient) for dimension, coefficient in coefficients.items() if coefficient)),
            expression.constant,
        )

    return AffineIndexPredicate(predicate.comparison, remap(predicate.left), remap(predicate.right))


def affine_predicate_in_domain_position_basis(
    predicate: AffineIndexPredicate,
    domain_axes: tuple[TensorAxis, ...],
) -> AffineIndexPredicate:
    """Convert a typed axis-id predicate to the portable artifact basis."""
    return _remap_affine_predicate(
        predicate,
        {axis.id: position for position, axis in enumerate(domain_axes)},
    )


def canonicalize_negative_infinity_domain(
    program: TensorProgram,
    masked_value: ProgramValue,
    *,
    streamed_axes: tuple[TensorAxis, ...],
) -> IndexDomainRestriction:
    """Recover an affine domain from ``select(predicate, value, -inf)``.

    Relation-only views and broadcasts are followed through exact
    :class:`OperandIndexing` records.  The selected value must already span the
    select domain: this prevents silently changing scalar broadcast semantics.
    """
    producers, operation_ordinals = _producer_records(program)
    operation = producers.get(masked_value)
    if not isinstance(operation, MapPrimitive):
        raise ValueError("index-domain candidate must be produced by a scalar Map")
    if operation.expression.kind is not ScalarExpressionKind.SELECT:
        raise ValueError("index-domain candidate Map must yield a two-way scalar select")

    predicate_expression, when_true, when_false = operation.expression.operands
    predicate_value = _input_value(predicate_expression, operation, "select predicate")
    true_is_rejected = _is_negative_infinity(when_true, operation, producers, set())
    false_is_rejected = _is_negative_infinity(when_false, operation, producers, set())
    if true_is_rejected == false_is_rejected:
        raise ValueError("index-domain select must have exactly one negative-infinity rejected branch")
    selected_expression = when_false if true_is_rejected else when_true
    selected_value = _input_value(selected_expression, operation, "selected score")
    if selected_value.axes != masked_value.axes:
        raise ValueError("index-domain selected value must preserve every select domain axis")

    visited: set[int] = set()
    coordinates = {axis: AffineIndexExpression(((dimension, 1),)) for dimension, axis in enumerate(masked_value.axes)}
    comparison = _index_predicate_from_value(
        predicate_value,
        coordinates,
        producers,
        operation_ordinals,
        visited,
        masked_value.axes,
    )
    predicate = index_compare(
        comparison.kind,
        _affine_as_index(comparison.left, masked_value.axes),
        _affine_as_index(comparison.right, masked_value.axes),
    )
    if true_is_rejected:
        predicate = IndexPredicate(IndexPredicateKind.NOT, predicates=(predicate,))
    position_predicate = _affine_predicate(predicate, masked_value.axes)
    affine_predicate = _remap_affine_predicate(
        position_predicate,
        {position: axis.id for position, axis in enumerate(masked_value.axes)},
    )
    return IndexDomainRestriction(
        masked_value=masked_value,
        selected_value=selected_value,
        domain_axes=masked_value.axes,
        streamed_axes=streamed_axes,
        predicate=predicate,
        affine_predicate=affine_predicate,
        source_fingerprint=program.semantic_fingerprint,
        select_operation_ordinal=operation_ordinals[id(operation)],
        predicate_operation_ordinals=tuple(sorted(visited)),
    )


def attach_index_domain_restriction(
    region: NormalizedExpContractRegion,
    restriction: IndexDomainRestriction,
) -> NormalizedExpContractRegion:
    """Attach a recovered domain to an unguarded normalized-exp region."""
    if region.score_map.output != restriction.masked_value:
        raise ValueError("domain predicate must be attached to the region's score Map output")
    if region.score_map.predicate != restriction.predicate or region.score_map.rejected_value != -math.inf:
        raise ValueError("region score Map must carry the recovered predicate and negative-infinity complement")
    if region.dimensions.normalized_axes != restriction.streamed_axes:
        raise ValueError("domain restriction streamed axes must equal the normalized-exp Fold axes")
    if region.domain_predicate is not None and region.domain_predicate != restriction.affine_predicate:
        raise ValueError("normalized-exp region already carries a different domain predicate")
    return replace(region, domain_predicate=restriction.affine_predicate)


def canonicalize_normalized_exp_index_domain(program: TensorProgram) -> DomainRestrictedNormalizedExpProgram:
    """Move one exact ``select(value, -inf)`` into a normalized-exp domain.

    The source remains the executable numerical oracle.  ``task_program``
    omits the index-generation/select materialization because ``region`` binds
    the same predicate to the normalized Fold/weighted-Contract domain.
    """
    try:
        return _canonicalize_normalized_exp_index_domain(program)
    except ValueError as error:
        raise NormalizedExpDomainCanonicalizationError(str(error)) from error


def _canonicalize_normalized_exp_index_domain(program: TensorProgram) -> DomainRestrictedNormalizedExpProgram:
    producers, operation_ordinals = _producer_records(program)
    uses: dict[ProgramValue, list[TensorPrimitive]] = {}
    for operation in program.operations:
        for value in primitive_inputs(operation):
            uses.setdefault(value, []).append(operation)
    candidates: list[tuple[MapPrimitive, FoldPrimitive, MapPrimitive]] = []
    for operation in program.operations:
        if not isinstance(operation, MapPrimitive) or operation.expression.kind is not ScalarExpressionKind.SELECT:
            continue
        consumers = uses.get(operation.output, [])
        maxima = tuple(
            consumer
            for consumer in consumers
            if isinstance(consumer, FoldPrimitive) and consumer.reducer is FoldReducer.MAXIMUM
        )
        centers = tuple(
            consumer
            for consumer in consumers
            if isinstance(consumer, MapPrimitive)
            and consumer.expression.kind is ScalarExpressionKind.SUBTRACT
            and any(_maximum_fold_source(value, producers) is not None for value in consumer.inputs)
        )
        # A differentiated program also uses the masked value in the source
        # maximum's pullback.  Those auxiliary consumers do not change the
        # forward normalized-exponential domain and remain explicit in the
        # task program; only the maximum and centering uses are rebound to the
        # finite score value here.
        if len(maxima) == 1 and len(centers) == 1:
            candidates.append((operation, maxima[0], centers[0]))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one select(value, -inf) feeding a maximum Fold and centering Map, found {len(candidates)}"
        )
    select, maximum, center = candidates[0]
    restriction = canonicalize_negative_infinity_domain(
        program,
        select.output,
        streamed_axes=maximum.reduction_axes,
    )
    score_map = producers.get(restriction.selected_value)
    if not isinstance(score_map, MapPrimitive):
        raise ValueError("normalized-exp selected score must be produced by a scalar Map")
    predicated_score_map = replace(
        score_map,
        name=f"{score_map.name}.predicated_domain",
        output=select.output,
        predicate=restriction.predicate,
        rejected_value=-math.inf,
    )
    rewritten: list[TensorPrimitive] = []
    for operation in program.operations:
        if operation is select:
            rewritten.append(predicated_score_map)
            continue
        if operation is center:
            maximum_inputs = tuple(value for value in center.inputs if _maximum_fold_source(value, producers) is maximum)
            if len(maximum_inputs) != 1:
                raise ValueError("normalized-exp centering Map must consume one maximum Fold finalizer")
            maximum_input = maximum_inputs[0]
            rewritten.append(
                replace(
                    center,
                    inputs=tuple(maximum.output if value is maximum_input else value for value in center.inputs),
                    expression=_rename_scalar_expression_input(
                        center.expression,
                        maximum_input.name,
                        maximum.output.name,
                    ),
                )
            )
            continue
        rewritten.append(operation)
    task_program = _remove_dead_tensor_operations(
        TensorProgram(program.inputs, tuple(rewritten), program.outputs, program.numerical_policy)
    )
    region = attach_index_domain_restriction(normalized_exp_contract_region(task_program), restriction)
    if operation_ordinals[id(select)] != restriction.select_operation_ordinal:
        raise ValueError("index-domain select provenance changed during canonicalization")
    return DomainRestrictedNormalizedExpProgram(program, task_program, restriction, region)


def _maximum_fold_source(
    value: ProgramValue,
    producers: dict[ProgramValue, TensorPrimitive],
) -> FoldPrimitive | None:
    """Follow an exact ``maximum(fold, -inf)`` scalar finalizer.

    JAX emits this finalizer for BF16 softmax even though the Fold identity is
    already negative infinity. It remains generic scalar algebra rather than
    becoming a workload-specific physical feature.
    """
    producer = producers.get(value)
    if isinstance(producer, FoldPrimitive):
        return producer if producer.reducer is FoldReducer.MAXIMUM else None
    if (
        isinstance(producer, MapPrimitive)
        and producer.expression.kind is ScalarExpressionKind.INPUT
        and len(producer.inputs) == 1
    ):
        folded = producers.get(producer.inputs[0])
        return folded if isinstance(folded, FoldPrimitive) and folded.reducer is FoldReducer.MAXIMUM else None
    if not isinstance(producer, MapPrimitive) or producer.expression.kind is not ScalarExpressionKind.MAXIMUM:
        return None
    if len(producer.inputs) != 1 or len(producer.expression.operands) != 2:
        return None
    dynamic, constant = producer.expression.operands
    if dynamic.kind is not ScalarExpressionKind.INPUT:
        dynamic, constant = constant, dynamic
    if dynamic.kind is not ScalarExpressionKind.INPUT or not _is_negative_infinity(constant, producer, producers, set()):
        return None
    source = _input_value(dynamic, producer, "maximum Fold finalizer")
    folded = producers.get(source)
    return folded if isinstance(folded, FoldPrimitive) and folded.reducer is FoldReducer.MAXIMUM else None


def _rename_scalar_expression_input(
    expression: ScalarExpression,
    source_name: str,
    result_name: str,
) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        if expression.input_name != source_name:
            return expression
        return replace(expression, input_name=result_name)
    return replace(
        expression,
        operands=tuple(
            _rename_scalar_expression_input(operand, source_name, result_name) for operand in expression.operands
        ),
    )


def _remove_dead_tensor_operations(program: TensorProgram) -> TensorProgram:
    live_values = set(program.outputs)
    retained: list[TensorPrimitive] = []
    for operation in reversed(program.operations):
        if not any(output in live_values for output in primitive_outputs(operation)):
            continue
        retained.append(operation)
        live_values.update(primitive_inputs(operation))
    retained.reverse()
    return TensorProgram(program.inputs, tuple(retained), program.outputs, program.numerical_policy)


def _producer_records(
    program: TensorProgram,
) -> tuple[dict[ProgramValue, TensorPrimitive], dict[int, int]]:
    producers: dict[ProgramValue, TensorPrimitive] = {}
    ordinals: dict[int, int] = {}
    for ordinal, operation in enumerate(program.operations):
        ordinals[id(operation)] = ordinal
        for output in primitive_outputs(operation):
            producers[output] = operation
    return producers, ordinals


def _input_value(expression: ScalarExpression, operation: MapPrimitive, description: str) -> ProgramValue:
    if expression.kind is not ScalarExpressionKind.INPUT or expression.input_name is None:
        raise ValueError(f"index-domain {description} must be one tensor input to the select Map")
    matches = tuple(value for value in operation.inputs if value.name == expression.input_name)
    if len(matches) != 1:
        raise ValueError(f"index-domain {description} does not identify one select operand")
    return matches[0]


def _is_negative_infinity(
    expression: ScalarExpression,
    operation: MapPrimitive,
    producers: dict[ProgramValue, TensorPrimitive],
    active: set[ProgramValue],
) -> bool:
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return isinstance(expression.constant, float) and expression.constant == -math.inf
    if expression.kind is ScalarExpressionKind.CONVERT:
        return _is_negative_infinity(expression.operands[0], operation, producers, active)
    if expression.kind is not ScalarExpressionKind.INPUT:
        return False
    value = _input_value(expression, operation, "select branch")
    return _negative_infinity_value(value, producers, active)


def _negative_infinity_value(
    value: ProgramValue,
    producers: dict[ProgramValue, TensorPrimitive],
    active: set[ProgramValue],
) -> bool:
    if value in active:
        raise ValueError("index-domain rejected-value lineage contains a cycle")
    active.add(value)
    producer = producers.get(value)
    if isinstance(producer, MapPrimitive):
        result = _is_negative_infinity(producer.expression, producer, producers, active)
    elif isinstance(producer, RelationPrimitive):
        result_sources = producer.resolved_result_sources
        result_index = producer.outputs.index(value)
        source = result_sources[result_index]
        result = source is not None and _negative_infinity_value(source, producers, active)
    else:
        result = False
    active.remove(value)
    return result


def _index_predicate_from_value(
    value: ProgramValue,
    coordinates: dict[TensorAxis, AffineIndexExpression],
    producers: dict[ProgramValue, TensorPrimitive],
    operation_ordinals: dict[int, int],
    visited: set[int],
    root_axes: tuple[TensorAxis, ...],
) -> _AffineComparison:
    producer = producers.get(value)
    if producer is None:
        raise ValueError("index-domain predicate depends on a program input rather than index generation")
    visited.add(operation_ordinals[id(producer)])
    if isinstance(producer, RelationPrimitive):
        expression, source, source_coordinates = _relation_result(
            producer,
            value,
            coordinates,
            root_axes,
        )
        if isinstance(expression, IndexPredicate):
            return _comparison_from_index_predicate(expression, coordinates, root_axes)
        if isinstance(expression, IndexExpression):
            raise ValueError("index-domain predicate Relation must yield a typed comparison, not an integer value")
        assert source is not None
        return _index_predicate_from_value(
            source,
            source_coordinates,
            producers,
            operation_ordinals,
            visited,
            root_axes,
        )
    if not isinstance(producer, MapPrimitive):
        raise ValueError("index-domain predicate lineage contains a non-Map, non-Relation primitive")
    if value.dtype is not DType.BOOL:
        raise ValueError("index-domain predicate Map must produce a Boolean value")
    scalar_inputs = {input_value.name: input_value for input_value in producer.inputs}
    return _predicate_from_scalar_expression(
        producer.expression,
        scalar_inputs,
        coordinates,
        producers,
        operation_ordinals,
        visited,
        root_axes,
    )


def _predicate_from_scalar_expression(
    expression: ScalarExpression,
    scalar_inputs: dict[str, ProgramValue],
    coordinates: dict[TensorAxis, AffineIndexExpression],
    producers: dict[ProgramValue, TensorPrimitive],
    operation_ordinals: dict[int, int],
    visited: set[int],
    root_axes: tuple[TensorAxis, ...],
) -> _AffineComparison:
    comparisons = {
        ScalarExpressionKind.EQUAL: IndexPredicateKind.EQUAL,
        ScalarExpressionKind.NOT_EQUAL: IndexPredicateKind.NOT_EQUAL,
        ScalarExpressionKind.LESS: IndexPredicateKind.LESS,
        ScalarExpressionKind.LESS_EQUAL: IndexPredicateKind.LESS_EQUAL,
        ScalarExpressionKind.GREATER: IndexPredicateKind.GREATER,
        ScalarExpressionKind.GREATER_EQUAL: IndexPredicateKind.GREATER_EQUAL,
    }
    kind = comparisons.get(expression.kind)
    if kind is None or len(expression.operands) != 2:
        raise ValueError(f"unsupported index-domain scalar predicate {expression.kind.value!r}")
    operands = tuple(
        _affine_from_scalar_expression(
            operand,
            scalar_inputs,
            coordinates,
            producers,
            operation_ordinals,
            visited,
            root_axes,
        )
        for operand in expression.operands
    )
    return _AffineComparison(kind, operands[0], operands[1])


def _affine_from_scalar_expression(
    expression: ScalarExpression,
    scalar_inputs: dict[str, ProgramValue],
    coordinates: dict[TensorAxis, AffineIndexExpression],
    producers: dict[ProgramValue, TensorPrimitive],
    operation_ordinals: dict[int, int],
    visited: set[int],
    root_axes: tuple[TensorAxis, ...],
) -> AffineIndexExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        value = scalar_inputs[expression.input_name]
        return _affine_from_value(value, coordinates, producers, operation_ordinals, visited, root_axes)
    if expression.kind is ScalarExpressionKind.CONSTANT:
        if type(expression.constant) is not int:
            raise ValueError("index-domain affine constants must be exact integers")
        return AffineIndexExpression(constant=expression.constant)
    if expression.kind is ScalarExpressionKind.CONVERT:
        return _affine_from_scalar_expression(
            expression.operands[0], scalar_inputs, coordinates, producers, operation_ordinals, visited, root_axes
        )
    if expression.kind in {
        ScalarExpressionKind.ADD,
        ScalarExpressionKind.SUBTRACT,
        ScalarExpressionKind.MULTIPLY,
    }:
        left, right = (
            _affine_from_scalar_expression(
                operand, scalar_inputs, coordinates, producers, operation_ordinals, visited, root_axes
            )
            for operand in expression.operands
        )
        return _combine_affine(expression.kind, left, right)
    raise ValueError(f"unsupported affine index scalar operation {expression.kind.value!r}")


def _affine_from_value(
    value: ProgramValue,
    coordinates: dict[TensorAxis, AffineIndexExpression],
    producers: dict[ProgramValue, TensorPrimitive],
    operation_ordinals: dict[int, int],
    visited: set[int],
    root_axes: tuple[TensorAxis, ...],
) -> AffineIndexExpression:
    producer = producers.get(value)
    if producer is None:
        raise ValueError("affine index value depends on a program input")
    visited.add(operation_ordinals[id(producer)])
    if isinstance(producer, RelationPrimitive):
        expression, source, source_coordinates = _relation_result(
            producer,
            value,
            coordinates,
            root_axes,
        )
        if isinstance(expression, IndexExpression):
            return _affine_from_index_expression(expression, coordinates, root_axes)
        if expression is not None:
            raise ValueError("affine index value is produced by a predicate expression")
        assert source is not None
        return _affine_from_value(source, source_coordinates, producers, operation_ordinals, visited, root_axes)
    if isinstance(producer, MapPrimitive):
        scalar_inputs = {input_value.name: input_value for input_value in producer.inputs}
        return _affine_from_scalar_expression(
            producer.expression,
            scalar_inputs,
            coordinates,
            producers,
            operation_ordinals,
            visited,
            root_axes,
        )
    raise ValueError("affine index lineage contains a non-Map, non-Relation primitive")


def _relation_result(
    relation: RelationPrimitive,
    value: ProgramValue,
    coordinates: dict[TensorAxis, AffineIndexExpression],
    root_axes: tuple[TensorAxis, ...],
) -> tuple[IndexExpression | IndexPredicate | None, ProgramValue | None, dict[TensorAxis, AffineIndexExpression]]:
    result_index = relation.outputs.index(value)
    expression = relation.resolved_result_expressions[result_index]
    source = relation.resolved_result_sources[result_index]
    if expression is not None:
        return expression, None, coordinates
    assert source is not None
    indexing = next(item for item in relation.input_indexing if item.operand == source)
    source_coordinates = {
        axis: _affine_from_index_expression(index, coordinates, root_axes)
        for axis, index in zip(source.axes, indexing.indices, strict=True)
    }
    return None, source, source_coordinates


def _affine_from_index_expression(
    expression: IndexExpression,
    coordinates: dict[TensorAxis, AffineIndexExpression],
    root_axes: tuple[TensorAxis, ...],
) -> AffineIndexExpression:
    if expression.kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return coordinates[expression.axis]
    if expression.kind is IndexExpressionKind.CONSTANT:
        if type(expression.constant) is not int:
            raise ValueError("affine index expressions require integer constants")
        return AffineIndexExpression(constant=expression.constant)
    if expression.kind in {
        IndexExpressionKind.ADD,
        IndexExpressionKind.SUBTRACT,
        IndexExpressionKind.MULTIPLY,
    }:
        left, right = (_affine_from_index_expression(operand, coordinates, root_axes) for operand in expression.operands)
        scalar_kind = {
            IndexExpressionKind.ADD: ScalarExpressionKind.ADD,
            IndexExpressionKind.SUBTRACT: ScalarExpressionKind.SUBTRACT,
            IndexExpressionKind.MULTIPLY: ScalarExpressionKind.MULTIPLY,
        }[expression.kind]
        return _combine_affine(scalar_kind, left, right)
    left, right = (_affine_from_index_expression(operand, coordinates, root_axes) for operand in expression.operands)
    if right.coefficients or right.constant <= 0:
        raise ValueError(f"affine {expression.kind.value} requires a positive integer divisor/modulus")
    if expression.kind is IndexExpressionKind.FLOOR_DIVIDE:
        divisor = right.constant
        quotient_constant, remainder_constant = divmod(left.constant, divisor)
        quotient_coefficients: list[tuple[int, int]] = []
        remainder_coefficients: list[tuple[int, int]] = []
        for dimension, coefficient in left.coefficients:
            quotient_coefficient, remainder_coefficient = divmod(coefficient, divisor)
            if quotient_coefficient:
                quotient_coefficients.append((dimension, quotient_coefficient))
            if remainder_coefficient:
                remainder_coefficients.append((dimension, remainder_coefficient))
        remainder = AffineIndexExpression(tuple(remainder_coefficients), remainder_constant)
        minimum, maximum = _affine_bounds(remainder, root_axes)
        if minimum < 0 or maximum >= divisor:
            raise ValueError("floor-divide index expression is not affine over the complete domain")
        return AffineIndexExpression(tuple(quotient_coefficients), quotient_constant)
    if expression.kind is IndexExpressionKind.MODULO:
        minimum, maximum = _affine_bounds(left, root_axes)
        if 0 <= minimum and maximum < right.constant:
            return left
        remainder = AffineIndexExpression(
            tuple(
                (dimension, coefficient % right.constant)
                for dimension, coefficient in left.coefficients
                if coefficient % right.constant
            ),
            left.constant % right.constant,
        )
        minimum, maximum = _affine_bounds(remainder, root_axes)
        if 0 <= minimum and maximum < right.constant:
            return remainder
        raise ValueError("modulo index expression is not affine over the complete domain")
    raise ValueError(f"unsupported affine index expression {expression.kind.value!r}")


def _combine_affine(
    kind: ScalarExpressionKind,
    left: AffineIndexExpression,
    right: AffineIndexExpression,
) -> AffineIndexExpression:
    if kind is ScalarExpressionKind.MULTIPLY:
        if left.coefficients and right.coefficients:
            raise ValueError("multiplication of two varying index expressions is not affine")
        varying, constant = (left, right.constant) if left.coefficients else (right, left.constant)
        return AffineIndexExpression(
            tuple(
                (dimension, coefficient * constant)
                for dimension, coefficient in varying.coefficients
                if coefficient * constant
            ),
            varying.constant * constant,
        )
    sign = -1 if kind is ScalarExpressionKind.SUBTRACT else 1
    coefficients: dict[int, int] = dict(left.coefficients)
    for dimension, coefficient in right.coefficients:
        coefficients[dimension] = coefficients.get(dimension, 0) + sign * coefficient
    return AffineIndexExpression(
        tuple((dimension, coefficient) for dimension, coefficient in sorted(coefficients.items()) if coefficient),
        left.constant + sign * right.constant,
    )


def _affine_bounds(expression: AffineIndexExpression, axes: tuple[TensorAxis, ...]) -> tuple[int, int]:
    minimum = expression.constant
    maximum = expression.constant
    for dimension, coefficient in expression.coefficients:
        contribution = coefficient * (axes[dimension].extent - 1)
        minimum += min(0, contribution)
        maximum += max(0, contribution)
    return minimum, maximum


def _affine_as_index(expression: AffineIndexExpression, axes: tuple[TensorAxis, ...]) -> IndexExpression:
    result = index_constant(expression.constant)
    for dimension, coefficient in expression.coefficients:
        term = index_axis(axes[dimension])
        if coefficient != 1:
            term = index_binary(IndexExpressionKind.MULTIPLY, term, index_constant(coefficient))
        result = index_binary(IndexExpressionKind.ADD, result, term)
    return result


def _comparison_from_index_predicate(
    predicate: IndexPredicate,
    coordinates: dict[TensorAxis, AffineIndexExpression],
    domain_axes: tuple[TensorAxis, ...],
) -> _AffineComparison:
    if predicate.predicates or len(predicate.expressions) != 2:
        raise ValueError("index-domain predicate Relation must yield one integer comparison")
    left, right = (_affine_from_index_expression(item, coordinates, domain_axes) for item in predicate.expressions)
    return _AffineComparison(predicate.kind, left, right)


def _affine_predicate(predicate: IndexPredicate, axes: tuple[TensorAxis, ...]) -> AffineIndexPredicate:
    inverted = False
    if predicate.kind is IndexPredicateKind.NOT:
        predicate = predicate.predicates[0]
        inverted = True
    if predicate.predicates or len(predicate.expressions) != 2:
        raise ValueError("physical affine domains require one integer comparison")
    coordinates = {axis: AffineIndexExpression(((dimension, 1),)) for dimension, axis in enumerate(axes)}
    left, right = (_affine_from_index_expression(expression, coordinates, axes) for expression in predicate.expressions)
    kind = predicate.kind
    if inverted:
        kind = {
            IndexPredicateKind.LESS: IndexPredicateKind.GREATER_EQUAL,
            IndexPredicateKind.LESS_EQUAL: IndexPredicateKind.GREATER,
            IndexPredicateKind.GREATER: IndexPredicateKind.LESS_EQUAL,
            IndexPredicateKind.GREATER_EQUAL: IndexPredicateKind.LESS,
        }.get(kind, kind)
        if kind in {IndexPredicateKind.EQUAL, IndexPredicateKind.NOT_EQUAL}:
            raise ValueError("physical affine domains cannot represent an inverted equality as one half-space")
    if kind is IndexPredicateKind.LESS:
        right = replace(right, constant=right.constant - 1)
    elif kind is IndexPredicateKind.LESS_EQUAL:
        pass
    elif kind is IndexPredicateKind.GREATER:
        left, right = right, replace(left, constant=left.constant - 1)
    elif kind is IndexPredicateKind.GREATER_EQUAL:
        left, right = right, left
    else:
        raise ValueError(f"physical affine domains do not support {kind.value!r} comparisons")
    return AffineIndexPredicate(IndexComparisonKind.SIGNED_LESS_EQUAL, left, right)
