# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical owner/time cover derivation for exact Event relations."""

from __future__ import annotations

from dataclasses import replace

from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineExpression,
    CuteAffineExpressionKind,
    CuteAffineMap,
    CuteIntegerSet,
    CuteStructuralEventCover,
    affine_constant,
)


def affine_expression_from_record(record: object) -> CuteAffineExpression:
    """Convert Shuttle's structured symbol-free affine AST to transport form."""
    if not isinstance(record, dict):
        raise ValueError("source Event affine expressions must be structured")
    kind = record.get("kind")
    if kind == "dimension":
        position = record.get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError("source Event affine dimensions must be integers")
        return CuteAffineExpression(CuteAffineExpressionKind.DIMENSION, position)
    if kind == "constant":
        value = record.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("source Event affine constants must be integers")
        return affine_constant(value)
    binary_kinds = {
        "add": CuteAffineExpressionKind.ADD,
        "multiply": CuteAffineExpressionKind.MULTIPLY,
        "modulo": CuteAffineExpressionKind.MODULO,
        "floor_divide": CuteAffineExpressionKind.FLOOR_DIVIDE,
        "ceil_divide": CuteAffineExpressionKind.CEIL_DIVIDE,
    }
    expression_kind = binary_kinds.get(kind)
    if expression_kind is None:
        raise ValueError(f"unsupported source Event affine expression {kind!r}")
    return _binary(
        expression_kind,
        affine_expression_from_record(record.get("left")),
        affine_expression_from_record(record.get("right")),
    )


def derive_lpt_algebraic_refinement_covers(
    cover: CuteStructuralEventCover,
    *,
    producer_row: CuteAffineExpression,
    relation_dimensions: int,
    batch: int,
    shared_relation: int,
    relation_group: int,
    owner_extent: int,
    producer_feature_extent: int,
    consumer_feature_extent: int,
    owner_tile: int,
) -> tuple[CuteStructuralEventCover, ...]:
    """Derive exact common-CTA maps for the LPT online refinement schedule."""
    extents = (
        batch,
        shared_relation,
        relation_group,
        owner_extent,
        producer_feature_extent,
        consumer_feature_extent,
        owner_tile,
    )
    if relation_dimensions <= 0 or any(extent <= 0 for extent in extents):
        raise ValueError("algebraic Event covers require positive relation and schedule extents")

    query = _modulo(producer_row, owner_extent)
    relation = _modulo(_floor_divide(producer_row, owner_extent), relation_group)
    shared = _modulo(
        _floor_divide(producer_row, owner_extent * relation_group),
        shared_relation,
    )
    batch_index = _floor_divide(producer_row, owner_extent * relation_group * shared_relation)
    head_batch = _add(_multiply(shared_relation, batch_index), shared)
    packed_owner = _floor_divide(
        _add(_multiply(relation_group, query), relation),
        owner_tile,
    )

    owner_blocks = (owner_extent * relation_group + owner_tile - 1) // owner_tile
    head_batches = batch * shared_relation
    bytes_per_shared_relation = owner_extent * (producer_feature_extent + consumer_feature_extent) * 2
    l2_section = (
        1
        if bytes_per_shared_relation > 50 * 1024 * 1024
        else _floor_power_of_two((50 * 1024 * 1024) // bytes_per_shared_relation)
    )
    full_sections, residual = divmod(head_batches, l2_section)
    phase = CuteAffineMap(relation_dimensions, (affine_constant(0),))
    pieces: list[CuteStructuralEventCover] = []

    def append_piece(section: int, width: int, base: int) -> None:
        section_head = _add(head_batch, affine_constant(-base))
        reversed_owner = _add(affine_constant(owner_blocks - 1), _multiply(-1, packed_owner))
        launch_owner = _add(
            affine_constant(section * l2_section * owner_blocks),
            _add(_multiply(width, reversed_owner), section_head),
        )
        predicate = CuteIntegerSet(
            relation_dimensions,
            (
                section_head,
                _add(affine_constant(width - 1), _multiply(-1, section_head)),
            ),
            (False, False),
        )
        owner = CuteAffineMap(relation_dimensions, (launch_owner, affine_constant(0), affine_constant(0)))
        pieces.append(
            replace(
                cover,
                relation_subset=predicate,
                producer_owner=owner,
                consumer_owner=owner,
                producer_time=phase,
                consumer_time=phase,
            )
        )

    for section in range(full_sections):
        append_piece(section, l2_section, section * l2_section)
    if residual:
        append_piece(full_sections, residual, full_sections * l2_section)
    if not pieces:
        raise ValueError("algebraic Event refinement has an empty physical owner domain")
    if len(pieces) == 1:
        pieces[0] = replace(pieces[0], relation_subset=None)
    return tuple(pieces)


def derive_single_tile_algebraic_refinement_covers(
    cover: CuteStructuralEventCover,
    *,
    producer_row: CuteAffineExpression,
    relation_dimensions: int,
    batch: int,
    shared_relation: int,
    relation_group: int,
    owner_extent: int,
    owner_tile: int,
) -> tuple[CuteStructuralEventCover, ...]:
    """Derive the common-CTA map for the ordinary tiled scheduler."""
    extents = (batch, shared_relation, relation_group, owner_extent, owner_tile)
    if relation_dimensions <= 0 or any(extent <= 0 for extent in extents):
        raise ValueError("algebraic Event covers require positive relation and schedule extents")

    query = _modulo(producer_row, owner_extent)
    relation = _modulo(_floor_divide(producer_row, owner_extent), relation_group)
    shared = _modulo(
        _floor_divide(producer_row, owner_extent * relation_group),
        shared_relation,
    )
    batch_index = _floor_divide(producer_row, owner_extent * relation_group * shared_relation)
    packed_owner = _floor_divide(
        _add(_multiply(relation_group, query), relation),
        owner_tile,
    )
    owner = CuteAffineMap(relation_dimensions, (packed_owner, shared, batch_index))
    phase = CuteAffineMap(relation_dimensions, (affine_constant(0),))
    return (
        replace(
            cover,
            producer_owner=owner,
            consumer_owner=owner,
            producer_time=phase,
            consumer_time=phase,
        ),
    )


def _binary(
    kind: CuteAffineExpressionKind,
    left: CuteAffineExpression,
    right: CuteAffineExpression,
) -> CuteAffineExpression:
    return CuteAffineExpression(kind, operands=(left, right))


def _add(left: CuteAffineExpression, right: CuteAffineExpression) -> CuteAffineExpression:
    return _binary(CuteAffineExpressionKind.ADD, left, right)


def _multiply(value: int, expression: CuteAffineExpression) -> CuteAffineExpression:
    return _binary(CuteAffineExpressionKind.MULTIPLY, affine_constant(value), expression)


def _floor_divide(expression: CuteAffineExpression, divisor: int) -> CuteAffineExpression:
    return _binary(CuteAffineExpressionKind.FLOOR_DIVIDE, expression, affine_constant(divisor))


def _modulo(expression: CuteAffineExpression, divisor: int) -> CuteAffineExpression:
    return _binary(CuteAffineExpressionKind.MODULO, expression, affine_constant(divisor))


def _floor_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("physical owner swizzle requires a positive L2 section")
    return 1 << (value.bit_length() - 1)
