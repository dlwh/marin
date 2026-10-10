# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.index_domain_canonicalization import (
    canonicalize_normalized_exp_index_domain,
    normalized_exp_task_algebra_from_record,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.layout_relation_canonicalization import (
    LayoutRelationKind,
    canonicalize_contract_layout_relations,
)
from tile_lifetime.streaming_attention_backward_reference import (
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    RelationPrimitive,
    ScalarExpressionKind,
    execute_tensor_program,
    index_axis,
    index_compare,
    index_constant,
)
from tile_lifetime.tensor_program_rewrites import sink_fold_invariant_division_through_contract


def _arguments(config: StreamingAttentionBackwardDebugConfig) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(73)
    return (
        rng.normal(size=(config.batch, config.query_length, config.query_heads, config.head_dimension)).astype(
            jnp.bfloat16
        ),
        rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
            jnp.bfloat16
        ),
        rng.normal(size=(config.batch, config.key_length, config.key_value_heads, config.head_dimension)).astype(
            jnp.bfloat16
        ),
    )


def test_contract_layout_canonicalization_exposes_generic_forward_region_exactly() -> None:
    config = StreamingAttentionBackwardDebugConfig()
    arguments = _arguments(config)
    imported = jax_function_to_tensor_program(
        causal_gqa_attention(config),
        *arguments,
        numerical_policy=NumericalPolicy.FAST,
    )
    source = sink_fold_invariant_division_through_contract(imported).program
    canonicalized = canonicalize_contract_layout_relations(source)

    environment = {f"argument.{index}": value for index, value in enumerate(arguments)}
    expected = execute_tensor_program(source, environment)
    actual = execute_tensor_program(canonicalized.program, environment)
    for output in source.outputs:
        np.testing.assert_array_equal(actual[output.name], expected[output.name])

    operations = canonicalized.program.operations
    producer = next(operation for operation in operations if isinstance(operation, ContractPrimitive))
    score_map = next(
        operation
        for operation in operations
        if isinstance(operation, MapPrimitive)
        and operation.expression.kind is ScalarExpressionKind.MULTIPLY
        and producer.output in operation.inputs
    )
    maximum = next(
        operation
        for operation in operations
        if isinstance(operation, FoldPrimitive) and operation.input.name.endswith("result.0")
    )
    center = next(
        operation
        for operation in operations
        if isinstance(operation, MapPrimitive) and operation.expression.kind is ScalarExpressionKind.SUBTRACT
    )
    assert producer.output in score_map.inputs
    assert maximum.output in center.inputs
    assert any(
        proof.kind is LayoutRelationKind.ROW_MAJOR_RESHAPE
        and proof.action == "retained_as_public_layout_view"
        and proof.source_shape == arguments[0].shape
        for proof in canonicalized.proofs
    )
    assert canonicalized.source_fingerprint == source.semantic_fingerprint
    assert canonicalized.fingerprint

    domain = canonicalize_normalized_exp_index_domain(canonicalized.program)
    assert domain.region.domain_predicate is not None
    assert domain.restriction.portable_affine_predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((4, 1),), "constant": 0},
        "right": {"coefficients": ((3, 1),), "constant": 0},
    }
    assert not any(
        isinstance(operation, MapPrimitive) and operation.expression.kind is ScalarExpressionKind.SELECT
        for operation in domain.task_program.operations
    )
    assert any(isinstance(operation, RelationPrimitive) for operation in domain.task_program.operations)
    decoded = normalized_exp_task_algebra_from_record(domain.canonical_record)
    assert decoded.task_program.semantic_fingerprint == domain.task_program.semantic_fingerprint
    assert decoded.region.domain_predicate is not None
    decoded_axes = decoded.region.score_map.output.axes
    assert decoded_axes[4].id != 4
    assert decoded_axes[3].id != 3
    assert decoded.region.domain_predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((decoded_axes[4].id, 1),), "constant": 0},
        "right": {"coefficients": ((decoded_axes[3].id, 1),), "constant": 0},
    }
    assert decoded.domain_coordinate_remap.position_to_axis_id == tuple(
        (position, axis.id) for position, axis in enumerate(decoded_axes)
    )
    assert decoded.domain_coordinate_remap.source_basis == "domain_position"
    assert decoded.domain_coordinate_remap.target_basis == "tensor_axis_id"
    assert tuple(axis.extent for axis in decoded.region.dimensions.normalized_axes) == tuple(
        axis.extent for axis in domain.region.dimensions.normalized_axes
    )


def test_contract_layout_canonicalization_absorbs_harmless_relation_chain() -> None:
    def twice_transposed_contract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        product = jnp.einsum("bi,ji->bj", left, right)
        return jnp.transpose(jnp.transpose(product, (1, 0)), (1, 0)) + 0.25

    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    source = jax_function_to_tensor_program(twice_transposed_contract, left, right)
    canonicalized = canonicalize_contract_layout_relations(source)

    expected = execute_tensor_program(source, {"argument.0": left, "argument.1": right})
    actual = execute_tensor_program(canonicalized.program, {"argument.0": left, "argument.1": right})
    np.testing.assert_array_equal(actual[source.outputs[0].name], expected[source.outputs[0].name])
    assert not any(isinstance(operation, RelationPrimitive) for operation in canonicalized.program.operations)
    assert canonicalized.proofs[0].kind is LayoutRelationKind.AXIS_PERMUTATION


def test_contract_layout_canonicalization_retains_non_bijective_slice() -> None:
    def sliced_contract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("bi,ji->bj", left, right)[:, ::2]

    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    source = jax_function_to_tensor_program(sliced_contract, left, right)
    canonicalized = canonicalize_contract_layout_relations(source)

    expected = execute_tensor_program(source, {"argument.0": left, "argument.1": right})
    actual = execute_tensor_program(canonicalized.program, {"argument.0": left, "argument.1": right})
    np.testing.assert_array_equal(actual[source.outputs[0].name], expected[source.outputs[0].name])
    relation = next(
        operation for operation in canonicalized.program.operations if isinstance(operation, RelationPrimitive)
    )
    assert relation.outputs[0] == source.outputs[0]
    assert not canonicalized.proofs


@pytest.mark.parametrize("predicate_location", ("relation", "input"))
def test_predicated_reshape_has_no_bijective_layout_proof(predicate_location) -> None:
    source = jax_function_to_tensor_program(lambda x: x.reshape(2, 2, 3), np.zeros((2, 6), np.float32))
    relation = source.operations[0]
    predicate = index_compare(IndexPredicateKind.LESS, index_axis(relation.domain_axes[0]), index_constant(1))
    if predicate_location == "relation":
        relation = replace(relation, predicate=predicate)
    else:
        relation = replace(relation, input_indexing=(replace(relation.input_indexing[0], predicate=predicate),))
    predicated = replace(source, operations=(relation,))
    canonical = canonicalize_contract_layout_relations(predicated)
    assert canonical.program == predicated
    assert not canonical.proofs
