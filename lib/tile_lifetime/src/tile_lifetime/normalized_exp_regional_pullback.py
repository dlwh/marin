# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace one embedded JAX reverse slice with its typed algebraic pullback."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tile_lifetime.normalized_exp_contract_region import (
    NormalizedExpContractPullback,
    NormalizedExpContractRegion,
    NormalizedExpContractRegionCapability,
    derive_normalized_exp_contract_pullback,
    normalized_exp_contract_pullback_tensor_program,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorPrimitive,
    TensorProgram,
    index_axis,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_input,
    scalar_unary,
)
from tile_lifetime.tensor_program_rewrites import remove_dead_operations


@dataclass(frozen=True)
class NormalizedExpRegionalPullbackReplacement:
    """A dependency-derived source cut and its typed replacement boundary."""

    program: TensorProgram
    pullback: NormalizedExpContractPullback
    covered_source_operations: tuple[int, ...]
    source_boundary_inputs: tuple[ProgramValue, ...]
    source_boundary_outputs: tuple[ProgramValue, ...]
    saved_values: tuple[ProgramValue, ...] = ()


@dataclass(frozen=True)
class NormalizedExpSavedStateCut:
    """Disjoint forward/reverse programs joined by an algebra-derived state cut."""

    forward: TensorProgram
    reverse: TensorProgram
    pullback: NormalizedExpContractPullback
    public_output: ProgramValue
    saved_values: tuple[ProgramValue, ...]
    recomputed_values: tuple[ProgramValue, ...]
    reverse_outputs: tuple[ProgramValue, ...]

    def __post_init__(self) -> None:
        forward_outputs = set(self.forward.outputs)
        if self.public_output not in forward_outputs or not set(self.saved_values) <= forward_outputs:
            raise ValueError("normalized-exp saved-state values must be produced by the forward program")
        if not set(self.saved_values) <= set(self.reverse.inputs):
            raise ValueError("normalized-exp reverse program must consume every saved-state value")
        forward_operation_outputs = {
            value for operation in self.forward.operations for value in primitive_outputs(operation)
        }
        reverse_operation_outputs = {
            value for operation in self.reverse.operations for value in primitive_outputs(operation)
        }
        if forward_operation_outputs & reverse_operation_outputs:
            raise ValueError("normalized-exp forward and reverse capabilities must be disjoint")


def split_normalized_exp_saved_state(
    region: NormalizedExpContractRegion,
    pullback: NormalizedExpContractPullback | None = None,
) -> NormalizedExpSavedStateCut:
    """Derive a compact forward/reverse cut from pullback value dependencies.

    The reverse capability recomputes the producer Contract, scalar score Map,
    and normalized weights from producer operands plus a saved rowwise
    log-normalizer.  It saves the normalized Contract output only when the
    post-finalizer VJP cannot use the public result directly.
    """
    pullback = derive_normalized_exp_contract_pullback(region) if pullback is None else pullback
    if pullback.region != region:
        raise ValueError("normalized-exp saved-state cut requires the pullback of the supplied region")

    log_normalizer = ProgramValue(
        "normalized_exp_contract.saved_log_normalizer",
        region.fold_state.row_sum_exp.axes,
        region.fold_state.row_sum_exp.dtype,
    )
    log_normalizer_map = MapPrimitive(
        "materialize normalized-exp row state",
        (region.fold_state.row_sum_exp, region.fold_state.row_max),
        log_normalizer,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_unary(ScalarExpressionKind.LOG, scalar_input(region.fold_state.row_sum_exp.name)),
            scalar_input(region.fold_state.row_max.name),
        ),
    )

    normalized_output = pullback.normalized_output.output
    can_reuse_public_output = _normalized_output_can_alias_public_output(pullback)
    saved_normalized_output = None if can_reuse_public_output else normalized_output
    forward_operations: tuple[TensorPrimitive, ...] = (
        region.producer,
        region.score_map,
        region.fold_state.maximum,
        region.fold_state.center,
        region.fold_state.exponential,
        region.fold_state.normalizer,
        region.consumer,
        pullback.normalized_output,
        pullback.post_normalization,
        log_normalizer_map,
    )
    forward_inputs = _external_inputs(forward_operations)
    saved_values = (log_normalizer,) if saved_normalized_output is None else (log_normalizer, saved_normalized_output)
    forward = TensorProgram(
        forward_inputs,
        forward_operations,
        (pullback.post_normalization.output, *saved_values),
        NumericalPolicy.FAST,
    )

    normalized_boundary = pullback.post_normalization.output if can_reuse_public_output else normalized_output
    substitutions = {
        pullback.normalized_output.output: normalized_boundary,
        region.fold_state.row_sum_exp: log_normalizer,
    }
    recomputed_weight = ProgramValue(
        "normalized_exp_contract.recomputed_normalized_weight",
        region.score_map.output.axes,
        pullback.normalized_weight.dtype,
    )
    recomputed_weight_map = MapPrimitive(
        "recompute normalized weight from saved row state",
        (region.score_map.output, log_normalizer),
        recomputed_weight,
        scalar_unary(
            ScalarExpressionKind.EXP,
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input(region.score_map.output.name),
                scalar_input(log_normalizer.name),
            ),
        ),
    )
    substitutions[pullback.normalized_weight] = recomputed_weight

    closed = normalized_exp_contract_pullback_tensor_program(pullback)
    reverse_tail = tuple(
        _substitute_primitive(operation, substitutions)
        for operation in closed.operations
        if operation
        not in {
            region.producer,
            region.score_map,
            region.fold_state.maximum,
            region.fold_state.center,
            region.fold_state.exponential,
            region.fold_state.normalizer,
            region.consumer,
            region.finalizer,
            pullback.normalized_output,
            pullback.normalized_weight_map,
        }
    )
    recomputed_score = _fresh_primitive_outputs(region.producer, ".reverse_recompute")
    recompute_substitutions = {region.producer.output: recomputed_score[0]}
    recomputed_producer = _substitute_primitive(region.producer, recompute_substitutions, output_values=recomputed_score)
    recomputed_score_map_outputs = _fresh_primitive_outputs(region.score_map, ".reverse_recompute")
    recompute_substitutions[region.score_map.output] = recomputed_score_map_outputs[0]
    recomputed_score_map = _substitute_primitive(
        region.score_map,
        recompute_substitutions,
        output_values=recomputed_score_map_outputs,
    )
    recomputed_weight_map = _substitute_primitive(recomputed_weight_map, recompute_substitutions)
    substitutions.update(recompute_substitutions)
    substitutions[recomputed_weight] = recomputed_weight_map.output
    reverse_tail = tuple(_substitute_primitive(operation, substitutions) for operation in reverse_tail)
    reverse_operations = (recomputed_producer, recomputed_score_map, recomputed_weight_map, *reverse_tail)
    reverse_outputs = tuple(substitutions.get(value, value) for value in closed.outputs[:3])
    reverse_inputs = _external_inputs(reverse_operations)
    reverse = TensorProgram(reverse_inputs, reverse_operations, reverse_outputs, NumericalPolicy.FAST)
    operation_by_name = {operation.name: operation for operation in reverse_operations}
    split_pullback = replace(
        pullback,
        normalized_output=region.finalizer if can_reuse_public_output else pullback.normalized_output,
        post_normalization_vjp=operation_by_name[pullback.post_normalization_vjp.name],
        output_product=operation_by_name[pullback.output_product.name],
        output_dependent_fold=operation_by_name[pullback.output_dependent_fold.name],
        consumer_weight_cotangent=operation_by_name[pullback.consumer_weight_cotangent.name],
        normalized_weight=recomputed_weight_map.output,
        normalized_weight_map=recomputed_weight_map,
        normalized_exp_pullback=operation_by_name[pullback.normalized_exp_pullback.name],
        score_map_vjp=operation_by_name[pullback.score_map_vjp.name],
        terminal_gradients=tuple(
            replace(
                gradient,
                contract=operation_by_name[gradient.contract.name],
                destination=substitutions.get(gradient.destination, gradient.destination),
            )
            for gradient in pullback.terminal_gradients
        ),
    )
    return NormalizedExpSavedStateCut(
        forward=forward,
        reverse=reverse,
        pullback=split_pullback,
        public_output=pullback.post_normalization.output,
        saved_values=saved_values,
        recomputed_values=(recomputed_producer.output, recomputed_score_map.output, recomputed_weight_map.output),
        reverse_outputs=reverse_outputs,
    )


def replace_normalized_exp_regional_pullback(
    source: TensorProgram,
    region: NormalizedExpContractRegion,
    capability: NormalizedExpContractRegionCapability,
) -> NormalizedExpRegionalPullbackReplacement:
    """Replace the reverse consumers of a declared region without workload matching.

    The source roles are recovered from Contract incidence and exact tensor
    domains.  The replacement reverse algebra is always obtained by automatic
    differentiation of ``region``.
    """
    operations = source.operations
    query, key = region.producer.inputs
    value = region.consumer.inputs[1]
    score = region.producer.output
    finalizer_ordinal = capability.finalizer_operation

    weight_cotangent = _one_contract(
        operations,
        after=finalizer_ordinal,
        operand=value,
        output_shape=score.shape,
        description="consumer-weight cotangent",
    )
    output_cotangent = next(item for item in weight_cotangent.inputs if item != value)
    value_gradient = _one_contract(
        operations,
        after=finalizer_ordinal,
        operand=output_cotangent,
        output_shape=value.shape,
        excluded=(weight_cotangent,),
        description="consumer-operand terminal gradient",
    )
    query_gradient = _one_contract(
        operations,
        after=finalizer_ordinal,
        operand=key,
        output_shape=query.shape,
        description="producer-owner terminal gradient",
    )
    key_gradient = _one_contract(
        operations,
        after=finalizer_ordinal,
        operand=query,
        output_shape=key.shape,
        excluded=(region.producer,),
        description="producer-streamed terminal gradient",
    )
    query_raw = next(item for item in query_gradient.inputs if item != key)
    key_raw = next(item for item in key_gradient.inputs if item != query)
    if query_raw != key_raw:
        raise ValueError("normalized-exp terminal producer gradients do not share one score cotangent")

    forward_ordinals = {
        capability.producer_operation,
        capability.score_operation,
        *capability.fold_operations,
        capability.consumer_operation,
        capability.finalizer_operation,
        *capability.finalizer_tail_operations,
    }
    forward_values = {output for ordinal in forward_ordinals for output in primitive_outputs(operations[ordinal])}
    boundaries = {query, key, value, output_cotangent, *forward_values}
    producer_by_value = {
        output: (ordinal, operation)
        for ordinal, operation in enumerate(operations)
        for output in primitive_outputs(operation)
    }
    reverse_ordinals = _ancestor_operation_cut(
        (query_gradient.output, key_gradient.output, value_gradient.output),
        boundaries=boundaries,
        producer_by_value=producer_by_value,
    )
    required = {
        operations.index(weight_cotangent),
        operations.index(query_gradient),
        operations.index(key_gradient),
        operations.index(value_gradient),
    }
    if not required <= reverse_ordinals:
        raise ValueError("normalized-exp reverse cut does not contain every terminal Contract")

    pullback = derive_normalized_exp_contract_pullback(region)
    old_cotangent = pullback.output_cotangent
    post_vjp = replace(
        pullback.post_normalization_vjp,
        inputs=tuple(
            output_cotangent if item == old_cotangent else item for item in pullback.post_normalization_vjp.inputs
        ),
        expression=_rename_scalar_input(
            pullback.post_normalization_vjp.expression,
            old_cotangent.name,
            output_cotangent.name,
        ),
    )
    pullback = replace(pullback, output_cotangent=output_cotangent, post_normalization_vjp=post_vjp)
    boundary_axes = (*output_cotangent.axes[1:], output_cotangent.axes[0])
    reverse_output_cotangent = replace(
        output_cotangent,
        name=f"{output_cotangent.name}.normalized_exp_reverse_boundary",
        axes=boundary_axes,
    )
    reverse_boundary = RelationPrimitive(
        "normalized-exp reverse cotangent boundary",
        (output_cotangent,),
        (reverse_output_cotangent,),
        boundary_axes,
        (
            OperandIndexing(
                output_cotangent,
                boundary_axes,
                tuple(index_axis(axis) for axis in output_cotangent.axes),
            ),
        ),
    )
    pullback = replace(
        pullback,
        output_cotangent=reverse_output_cotangent,
        post_normalization_vjp=_substitute_primitive(
            pullback.post_normalization_vjp,
            {output_cotangent: reverse_output_cotangent},
        ),
    )
    split = split_normalized_exp_saved_state(region, pullback)
    replacement_operations = split.reverse.operations
    replacement_outputs = split.reverse.outputs
    source_outputs = (query_gradient.output, key_gradient.output, value_gradient.output)
    output_relations = tuple(
        RelationPrimitive(
            f"normalized-exp regional terminal {index} boundary",
            (replacement_output,),
            (source_output,),
            source_output.axes,
            (
                OperandIndexing(
                    replacement_output,
                    source_output.axes,
                    tuple(index_axis(axis) for axis in source_output.axes),
                ),
            ),
        )
        for index, (replacement_output, source_output) in enumerate(
            zip(replacement_outputs, source_outputs, strict=True)
        )
    )

    output_cotangent_producer = producer_by_value.get(output_cotangent)
    if output_cotangent_producer is None:
        raise ValueError("normalized-exp output cotangent must be produced inside the containing program")
    insertion_ordinal = output_cotangent_producer[0] + 1
    forward_state_operations = tuple(
        operation
        for operation in split.forward.operations
        if operation
        not in {
            region.producer,
            region.score_map,
            region.fold_state.maximum,
            region.fold_state.center,
            region.fold_state.exponential,
            region.fold_state.normalizer,
            region.consumer,
            region.finalizer,
            pullback.post_normalization,
        }
    )
    forward_state_insertion = capability.finalizer_operation + 1
    rewritten: list[TensorPrimitive] = []
    for ordinal, operation in enumerate(operations):
        if ordinal == forward_state_insertion:
            rewritten.extend(forward_state_operations)
        if ordinal == insertion_ordinal:
            rewritten.extend((reverse_boundary, *replacement_operations, *output_relations))
        if ordinal not in reverse_ordinals:
            rewritten.append(operation)
    if forward_state_insertion == len(operations):
        rewritten.extend(forward_state_operations)
    if insertion_ordinal == len(operations):
        rewritten.extend((reverse_boundary, *replacement_operations, *output_relations))
    program = remove_dead_operations(
        TensorProgram(source.inputs, tuple(rewritten), source.outputs, source.numerical_policy)
    )
    return NormalizedExpRegionalPullbackReplacement(
        program,
        split.pullback,
        tuple(sorted(reverse_ordinals)),
        (query, key, value, output_cotangent),
        source_outputs,
        split.saved_values,
    )


def _one_contract(
    operations: tuple[TensorPrimitive, ...],
    *,
    after: int,
    operand: ProgramValue,
    output_shape: tuple[int, ...],
    description: str,
    excluded: tuple[ContractPrimitive, ...] = (),
) -> ContractPrimitive:
    matches = tuple(
        operation
        for ordinal, operation in enumerate(operations)
        if ordinal > after
        and isinstance(operation, ContractPrimitive)
        and operation not in excluded
        and operand in operation.inputs
        and operation.output.shape == output_shape
    )
    if len(matches) != 1:
        raise ValueError(f"normalized-exp regional {description} requires one Contract, found {len(matches)}")
    return matches[0]


def _ancestor_operation_cut(
    roots: tuple[ProgramValue, ...],
    *,
    boundaries: set[ProgramValue],
    producer_by_value: dict[ProgramValue, tuple[int, TensorPrimitive]],
) -> set[int]:
    pending = list(roots)
    covered: set[int] = set()
    while pending:
        value = pending.pop()
        if value in boundaries:
            continue
        producer = producer_by_value.get(value)
        if producer is None:
            raise ValueError(f"normalized-exp reverse slice escapes through undeclared value {value.name!r}")
        ordinal, operation = producer
        if ordinal in covered:
            continue
        covered.add(ordinal)
        pending.extend(primitive_inputs(operation))
    return covered


def _rename_scalar_input(expression: ScalarExpression, source: str, target: str) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        return replace(expression, input_name=target) if expression.input_name == source else expression
    return replace(
        expression,
        operands=tuple(_rename_scalar_input(operand, source, target) for operand in expression.operands),
    )


def _normalized_output_can_alias_public_output(pullback: NormalizedExpContractPullback) -> bool:
    """Prove that the normalized value and public value share exact storage."""
    expression = pullback.post_normalization.expression
    while expression.kind is ScalarExpressionKind.CONVERT:
        expression = expression.operands[0]
    return (
        expression.kind is ScalarExpressionKind.INPUT
        and expression.input_name == pullback.normalized_output.output.name
        and pullback.post_normalization.output.dtype == pullback.normalized_output.output.dtype
    )


def _external_inputs(operations: tuple[TensorPrimitive, ...]) -> tuple[ProgramValue, ...]:
    produced: set[ProgramValue] = set()
    inputs: list[ProgramValue] = []
    for operation in operations:
        for value in primitive_inputs(operation):
            if value not in produced and value not in inputs:
                inputs.append(value)
        produced.update(primitive_outputs(operation))
    return tuple(inputs)


def _fresh_primitive_outputs(operation: TensorPrimitive, suffix: str) -> tuple[ProgramValue, ...]:
    return tuple(replace(value, name=f"{value.name}{suffix}") for value in primitive_outputs(operation))


def _substitute_expression_inputs(
    expression: ScalarExpression,
    substitutions: dict[ProgramValue, ProgramValue],
) -> ScalarExpression:
    names = {source.name: target.name for source, target in substitutions.items()}
    if expression.kind is ScalarExpressionKind.INPUT and expression.input_name in names:
        return replace(expression, input_name=names[expression.input_name])
    return replace(
        expression,
        operands=tuple(_substitute_expression_inputs(operand, substitutions) for operand in expression.operands),
    )


def _substitute_primitive(
    operation: TensorPrimitive,
    substitutions: dict[ProgramValue, ProgramValue],
    *,
    output_values: tuple[ProgramValue, ...] | None = None,
) -> TensorPrimitive:
    inputs = tuple(substitutions.get(value, value) for value in primitive_inputs(operation))
    outputs = primitive_outputs(operation) if output_values is None else output_values
    if isinstance(operation, ContractPrimitive):
        return replace(operation, inputs=inputs, output=outputs[0])
    if isinstance(operation, MapPrimitive):
        return replace(
            operation,
            inputs=inputs,
            output=outputs[0],
            expression=_substitute_expression_inputs(operation.expression, substitutions),
        )
    if isinstance(operation, FoldPrimitive):
        return replace(operation, input=inputs[0], output=outputs[0])
    if isinstance(operation, RelationPrimitive):
        indexing_by_value = {item.operand: item for item in operation.input_indexing}
        return replace(
            operation,
            inputs=inputs,
            outputs=outputs,
            input_indexing=tuple(
                replace(indexing_by_value[original], operand=replacement)
                for original, replacement in zip(operation.inputs, inputs, strict=True)
            ),
            result_sources=tuple(
                substitutions.get(value, value) if value is not None else None
                for value in operation.resolved_result_sources
            ),
        )
    raise TypeError(f"normalized-exp saved-state substitution does not support {type(operation).__name__}")
