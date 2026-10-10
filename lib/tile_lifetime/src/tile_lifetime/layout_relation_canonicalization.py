# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonicalize exact layout Relations around generic algebra primitives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum

from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    IndexExpression,
    IndexExpressionKind,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    index_axis,
    index_binary,
    index_constant,
    primitive_inputs,
    primitive_outputs,
)


class LayoutRelationKind(StrEnum):
    """Semantically distinct exact layout relations admitted by this pass."""

    AXIS_PERMUTATION = "axis_permutation"
    ROW_MAJOR_RESHAPE = "row_major_reshape"
    BROADCAST_PROJECTION = "broadcast_projection"


@dataclass(frozen=True)
class BijectiveLayoutRelation:
    """A complete scalar-free Relation with a proved dense index bijection."""

    relation: RelationPrimitive
    kind: LayoutRelationKind
    source_to_result_dimensions: tuple[int | None, ...]


def bijective_layout_relation(relation: RelationPrimitive) -> BijectiveLayoutRelation:
    """Prove a permutation (including unit axes) or exact row-major reshape."""
    if not _complete_layout_relation(relation):
        raise ValueError("layout Relation requires a complete unpredicated single-source domain")
    source, result = relation.inputs[0], relation.outputs[0]
    mapping: list[int | None] = []
    used: set[int] = set()
    for axis, expression in zip(source.axes, relation.input_indexing[0].indices, strict=True):
        expression = _unit_index(expression)
        if axis.extent == 1 and expression == index_constant(0):
            mapping.append(None)
            continue
        if expression.kind is not IndexExpressionKind.AXIS:
            break
        dimension = result.axes.index(expression.axis)
        if dimension in used or result.axes[dimension].extent != axis.extent:
            break
        used.add(dimension)
        mapping.append(dimension)
    else:
        if all(axis.extent == 1 or dimension in used for dimension, axis in enumerate(result.axes)):
            return BijectiveLayoutRelation(relation, LayoutRelationKind.AXIS_PERMUTATION, tuple(mapping))
    reshape = _row_major_reshape(relation)
    if reshape is not None:
        return BijectiveLayoutRelation(relation, LayoutRelationKind.ROW_MAJOR_RESHAPE, reshape)
    raise ValueError("layout Relation has no exact symbolic dense bijection")


def _complete_layout_relation(relation: RelationPrimitive) -> bool:
    if (
        len(relation.inputs) != 1
        or len(relation.outputs) != 1
        or len(relation.input_indexing) != 1
        or relation.predicate is not None
        or relation.resolved_result_sources != relation.inputs
        or relation.resolved_result_expressions != (None,)
    ):
        return False
    source, result = relation.inputs[0], relation.outputs[0]
    indexing = relation.input_indexing[0]
    return (
        indexing.operand == source
        and indexing.predicate is None
        and relation.domain_axes == indexing.domain_axes == result.axes
        and source.dtype is result.dtype
        and _element_count(source.shape) == _element_count(result.shape)
    )


def _unit_index(expression: IndexExpression) -> IndexExpression:
    if expression.kind is IndexExpressionKind.AXIS and expression.axis.extent == 1:
        return index_constant(0)
    if expression.kind not in (IndexExpressionKind.ADD, IndexExpressionKind.MULTIPLY):
        return expression
    left, right = map(_unit_index, expression.operands)
    zero, one = index_constant(0), index_constant(1)
    if expression.kind is IndexExpressionKind.ADD:
        if left == zero:
            return right
        if right == zero:
            return left
    else:
        if left == zero or right == zero:
            return zero
        if left == one:
            return right
        if right == one:
            return left
    return replace(expression, operands=(left, right))


@dataclass(frozen=True)
class LayoutRelationProof:
    """Machine-checkable provenance for one layout relation chain."""

    kind: LayoutRelationKind
    relation_operation_ordinals: tuple[int, ...]
    source_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    source_to_result_dimensions: tuple[int | None, ...]
    contract_operation_ordinals: tuple[int, ...]
    action: str
    neutral_map_operation_ordinals: tuple[int, ...] = ()

    @property
    def canonical_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind.value,
            "relation_operation_ordinals": self.relation_operation_ordinals,
            "source_shape": self.source_shape,
            "result_shape": self.result_shape,
            "source_to_result_dimensions": self.source_to_result_dimensions,
            "contract_operation_ordinals": self.contract_operation_ordinals,
            "action": self.action,
        }
        if self.neutral_map_operation_ordinals:
            record["neutral_map_operation_ordinals"] = self.neutral_map_operation_ordinals
        return record


@dataclass(frozen=True)
class LayoutRelationCanonicalization:
    """A rewritten program plus exact source-anchored layout proofs."""

    source_fingerprint: str
    program: TensorProgram
    proofs: tuple[LayoutRelationProof, ...]

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "shuttle_layout_relation_canonicalization_v1",
            "source_fingerprint": self.source_fingerprint,
            "result_fingerprint": self.program.semantic_fingerprint,
            "proofs": tuple(proof.canonical_record for proof in self.proofs),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def canonicalize_contract_layout_relations(program: TensorProgram) -> LayoutRelationCanonicalization:
    """Absorb exact Contract result views and scalar-Map broadcast views.

    Rank-changing row-major reshapes are retained as public layout views.  In
    particular, splitting one logical dimension into explicit group/local
    dimensions is mathematical indexing information; it is not recovered as a
    named execution mode.
    """
    producers, uses, operation_ordinals = _program_graph(program)
    absorbed_contracts: dict[int, ContractPrimitive] = {}
    skipped_relations: set[int] = set()
    proofs: list[LayoutRelationProof] = []

    for operation in program.operations:
        if not isinstance(operation, ContractPrimitive):
            continue
        current = operation.output
        chain: list[RelationPrimitive] = []
        source_to_current = {axis: axis for axis in current.axes}
        while True:
            consumers = uses.get(current, ())
            if len(consumers) != 1 or not isinstance(consumers[0], RelationPrimitive):
                break
            relation = consumers[0]
            mapping = _axis_permutation(relation, current)
            if mapping is None or set(current.axes) != set(relation.outputs[0].axes):
                break
            chain.append(relation)
            source_to_current = {
                source_axis: mapping[current_axis] for source_axis, current_axis in source_to_current.items()
            }
            current = relation.outputs[0]
        if not chain:
            continue
        remapped_index_maps = tuple(
            tuple(_remap_axis_index_map(mapping, source_to_current) for mapping in mappings)
            for mappings in operation.input_index_maps
        )
        absorbed_contracts[id(operation)] = replace(
            operation,
            output=current,
            input_index_maps=remapped_index_maps,
        )
        skipped_relations.update(id(relation) for relation in chain)
        first = chain[0]
        proofs.append(
            LayoutRelationProof(
                LayoutRelationKind.AXIS_PERMUTATION,
                tuple(operation_ordinals[id(relation)] for relation in chain),
                operation.output.shape,
                current.shape,
                tuple(current.axes.index(source_to_current[axis]) for axis in operation.output.axes),
                (operation_ordinals[id(operation)],),
                "absorbed_into_contract_result",
            )
        )
        assert first.inputs[0] == operation.output

    rewritten: list[TensorPrimitive] = []
    for operation in program.operations:
        replacement = absorbed_contracts.get(id(operation))
        if replacement is not None:
            rewritten.append(replacement)
            continue
        if id(operation) in skipped_relations:
            continue
        if isinstance(operation, MapPrimitive):
            operation, map_proofs = _elide_map_broadcasts(
                operation,
                producers,
                operation_ordinals,
                blocked_relation_ids=skipped_relations,
            )
            proofs.extend(map_proofs)
        rewritten.append(operation)

    transformed = _remove_dead_operations(
        TensorProgram(program.inputs, tuple(rewritten), program.outputs, program.numerical_policy)
    )
    retained_operation_ids = {id(operation) for operation in transformed.operations}
    for operation in program.operations:
        if not isinstance(operation, RelationPrimitive) or id(operation) not in retained_operation_ids:
            continue
        permutation = _axis_permutation(operation, operation.inputs[0]) if len(operation.inputs) == 1 else None
        if permutation is not None:
            proofs.append(
                LayoutRelationProof(
                    LayoutRelationKind.AXIS_PERMUTATION,
                    (operation_ordinals[id(operation)],),
                    operation.inputs[0].shape,
                    operation.outputs[0].shape,
                    tuple(operation.outputs[0].axes.index(permutation[axis]) for axis in operation.inputs[0].axes),
                    _downstream_contract_ordinals(operation.outputs[0], uses, operation_ordinals),
                    "retained_as_public_layout_view",
                )
            )
            continue
        reshape_mapping = _row_major_reshape(operation)
        if reshape_mapping is None:
            continue
        proofs.append(
            LayoutRelationProof(
                LayoutRelationKind.ROW_MAJOR_RESHAPE,
                (operation_ordinals[id(operation)],),
                operation.inputs[0].shape,
                operation.outputs[0].shape,
                reshape_mapping,
                _downstream_contract_ordinals(operation.outputs[0], uses, operation_ordinals),
                "retained_as_public_layout_view",
            )
        )
    return LayoutRelationCanonicalization(program.semantic_fingerprint, transformed, tuple(proofs))


def _program_graph(
    program: TensorProgram,
) -> tuple[dict[ProgramValue, TensorPrimitive], dict[ProgramValue, tuple[TensorPrimitive, ...]], dict[int, int]]:
    producers: dict[ProgramValue, TensorPrimitive] = {}
    mutable_uses: dict[ProgramValue, list[TensorPrimitive]] = {}
    ordinals: dict[int, int] = {}
    for ordinal, operation in enumerate(program.operations):
        ordinals[id(operation)] = ordinal
        for value in primitive_outputs(operation):
            producers[value] = operation
        for value in primitive_inputs(operation):
            mutable_uses.setdefault(value, []).append(operation)
    return producers, {value: tuple(items) for value, items in mutable_uses.items()}, ordinals


def _axis_permutation(
    relation: RelationPrimitive,
    source: ProgramValue,
) -> dict[TensorAxis, TensorAxis] | None:
    if not _complete_layout_relation(relation) or relation.inputs != (source,):
        return None
    indexing = relation.input_indexing[0]
    mapping: dict[TensorAxis, TensorAxis] = {}
    used_result_axes: set[TensorAxis] = set()
    remaining_unit_axes = [axis for axis in relation.domain_axes if axis.extent == 1]
    for source_axis, expression in zip(source.axes, indexing.indices, strict=True):
        result_axis: TensorAxis | None = None
        if expression.kind is IndexExpressionKind.AXIS:
            result_axis = expression.axis
        elif expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and source_axis.extent == 1:
            result_axis = next((axis for axis in remaining_unit_axes if axis not in used_result_axes), None)
        if result_axis is None or result_axis.extent != source_axis.extent or result_axis in used_result_axes:
            return None
        mapping[source_axis] = result_axis
        used_result_axes.add(result_axis)
    if used_result_axes != set(relation.domain_axes):
        return None
    return mapping


def _remap_axis_index_map(
    mapping: AxisIndexMap,
    source_to_result: dict[TensorAxis, TensorAxis],
) -> AxisIndexMap:
    return replace(mapping, domain_axis=source_to_result.get(mapping.domain_axis, mapping.domain_axis))


def _elide_map_broadcasts(
    operation: MapPrimitive,
    producers: dict[ProgramValue, TensorPrimitive],
    operation_ordinals: dict[int, int],
    *,
    blocked_relation_ids: set[int],
) -> tuple[MapPrimitive, tuple[LayoutRelationProof, ...]]:
    replacements: dict[str, ProgramValue] = {}
    proofs: list[LayoutRelationProof] = []
    for value in operation.inputs:
        root = value
        chain: list[RelationPrimitive] = []
        neutral_map_ordinals: list[int] = []
        while True:
            producer = producers.get(root)
            if isinstance(producer, MapPrimitive):
                passthrough = _neutral_scalar_map_source(producer, root)
                if passthrough is None:
                    break
                neutral_map_ordinals.append(operation_ordinals[id(producer)])
                root = passthrough
                continue
            if not isinstance(producer, RelationPrimitive) or id(producer) in blocked_relation_ids:
                break
            source_to_result = _broadcast_projection(producer, root)
            permutation = _axis_permutation(producer, producer.inputs[0]) if producer.inputs else None
            reshape = _row_major_reshape(producer)
            if source_to_result is None and permutation is None and reshape is None:
                break
            chain.append(producer)
            root = producer.inputs[0]
        if not chain or not set(root.axes) <= set(operation.output.axes):
            continue
        replacements[value.name] = root
        proofs.append(
            LayoutRelationProof(
                LayoutRelationKind.BROADCAST_PROJECTION,
                tuple(operation_ordinals[id(relation)] for relation in reversed(chain)),
                root.shape,
                value.shape,
                tuple(operation.output.axes.index(axis) for axis in root.axes),
                (),
                "elided_into_map_broadcast",
                tuple(reversed(neutral_map_ordinals)),
            )
        )
    if not replacements:
        return operation, ()
    rewritten_inputs = tuple(dict.fromkeys(replacements.get(value.name, value) for value in operation.inputs))
    rewritten_expression = _rename_scalar_inputs(
        operation.expression,
        {name: value.name for name, value in replacements.items()},
    )
    return replace(operation, inputs=rewritten_inputs, expression=rewritten_expression), tuple(proofs)


def _neutral_scalar_map_source(operation: MapPrimitive, result: ProgramValue) -> ProgramValue | None:
    """Return the sole input when a scalar Map is an exact floating-point identity."""
    if operation.output != result or len(operation.inputs) != 1:
        return None
    source = operation.inputs[0]
    if source.axes != result.axes or source.dtype is not result.dtype:
        return None
    expression = operation.expression
    if expression.kind is ScalarExpressionKind.INPUT and expression.input_name == source.name:
        return source
    if expression.kind is not ScalarExpressionKind.MAXIMUM:
        return None
    left, right = expression.operands
    pairs = ((left, right), (right, left))
    if any(
        value.kind is ScalarExpressionKind.INPUT
        and value.input_name == source.name
        and neutral.kind is ScalarExpressionKind.CONSTANT
        and isinstance(neutral.constant, float)
        and math.isinf(neutral.constant)
        and neutral.constant < 0
        for value, neutral in pairs
    ):
        return source
    return None


def _broadcast_projection(
    relation: RelationPrimitive,
    result: ProgramValue,
) -> tuple[int | None, ...] | None:
    if (
        relation.predicate is not None
        or len(relation.inputs) != 1
        or relation.outputs != (result,)
        or relation.resolved_result_sources != relation.inputs
        or relation.resolved_result_expressions != (None,)
    ):
        return None
    source = relation.inputs[0]
    if source.dtype is not result.dtype or any(axis.extent != 1 and axis not in result.axes for axis in source.axes):
        return None
    indexing = relation.input_indexing[0]
    if indexing.predicate is not None:
        return None
    dimensions: list[int | None] = []
    for axis, expression in zip(source.axes, indexing.indices, strict=True):
        if expression.kind is IndexExpressionKind.AXIS and expression.axis == axis:
            dimensions.append(result.axes.index(axis))
            continue
        if expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and axis.extent == 1:
            dimensions.append(None)
            continue
        return None
    return tuple(dimensions)


def _rename_scalar_inputs(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return replace(expression, input_name=names.get(expression.input_name, expression.input_name))
    return replace(expression, operands=tuple(_rename_scalar_inputs(operand, names) for operand in expression.operands))


def _row_major_reshape(relation: RelationPrimitive) -> tuple[int | None, ...] | None:
    if not _complete_layout_relation(relation):
        return None
    source, result = relation.inputs[0], relation.outputs[0]
    expected = _row_major_source_indices(source.axes, result.axes)
    if relation.input_indexing[0].indices != expected:
        return None
    return tuple(result.axes.index(axis) if axis in result.axes else None for axis in source.axes)


def _downstream_contract_ordinals(
    value: ProgramValue,
    uses: dict[ProgramValue, tuple[TensorPrimitive, ...]],
    operation_ordinals: dict[int, int],
) -> tuple[int, ...]:
    pending = [value]
    visited_values: set[ProgramValue] = set()
    contracts: set[int] = set()
    while pending:
        current = pending.pop()
        if current in visited_values:
            continue
        visited_values.add(current)
        for operation in uses.get(current, ()):
            if isinstance(operation, ContractPrimitive):
                contracts.add(operation_ordinals[id(operation)])
                continue
            if (
                isinstance(operation, MapPrimitive)
                and len(operation.inputs) == 1
                and operation.inputs[0] == current
                and operation.output.axes == current.axes
            ):
                pending.append(operation.output)
    return tuple(sorted(contracts))


def _row_major_source_indices(
    source_axes: tuple[TensorAxis, ...],
    result_axes: tuple[TensorAxis, ...],
) -> tuple[IndexExpression, ...]:
    linear = index_constant(0)
    for dimension, axis in enumerate(result_axes):
        stride = _element_count(tuple(remaining.extent for remaining in result_axes[dimension + 1 :]))
        term = index_axis(axis)
        if stride != 1:
            term = index_binary(IndexExpressionKind.MULTIPLY, term, index_constant(stride))
        linear = index_binary(IndexExpressionKind.ADD, linear, term)
    indices: list[IndexExpression] = []
    for dimension, axis in enumerate(source_axes):
        stride = _element_count(tuple(remaining.extent for remaining in source_axes[dimension + 1 :]))
        index = linear
        if stride != 1:
            index = index_binary(IndexExpressionKind.FLOOR_DIVIDE, index, index_constant(stride))
        if axis.extent != 1:
            index = index_binary(IndexExpressionKind.MODULO, index, index_constant(axis.extent))
        else:
            index = index_constant(0)
        indices.append(index)
    return tuple(indices)


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def _remove_dead_operations(program: TensorProgram) -> TensorProgram:
    live_values = set(program.outputs)
    retained: list[TensorPrimitive] = []
    for operation in reversed(program.operations):
        if not any(output in live_values for output in primitive_outputs(operation)):
            continue
        retained.append(operation)
        live_values.update(primitive_inputs(operation))
    retained.reverse()
    return TensorProgram(program.inputs, tuple(retained), program.outputs, program.numerical_policy)
