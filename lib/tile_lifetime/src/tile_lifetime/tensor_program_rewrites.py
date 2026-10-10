# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic algebraic rewrites over executable tensor programs."""

import math
from dataclasses import dataclass, replace

from shuttle.ir import DType
from tile_lifetime.plan import NumericalEquivalence, RewriteExplanation
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    IndexExpressionKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    index_axis,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_input,
    serialize_scalar_expression,
)


@dataclass(frozen=True)
class TensorProgramRewrite:
    """A transformed program and machine-readable rewrite explanations."""

    source_fingerprint: str
    program: TensorProgram
    rewrites: tuple[RewriteExplanation, ...]


@dataclass(frozen=True)
class FoldInvariantDivisionBoundary:
    """Typed source boundary for a Fold-invariant division feeding a Contract."""

    contract_operation: int
    normalized_operand: int
    division_operation: int
    numerator: ProgramValue
    denominator: ProgramValue
    fold_state: ProgramValue
    denominator_path: tuple[tuple[int, str, str, str], ...]
    division_expression: ScalarExpression

    @property
    def canonical_record(self) -> dict[str, object]:
        def value_record(value: ProgramValue) -> dict[str, object]:
            return {
                "name": value.name,
                "shape": value.shape,
                "dtype": value.dtype.value,
            }

        return {
            "format": "shuttle_fold_invariant_division_boundary_v1",
            "contract_operation": self.contract_operation,
            "normalized_operand": self.normalized_operand,
            "division_operation": self.division_operation,
            "numerator": value_record(self.numerator),
            "denominator": value_record(self.denominator),
            "fold_state": value_record(self.fold_state),
            "denominator_path": tuple(
                {
                    "operation": ordinal,
                    "kind": kind,
                    "input_dtype": input_dtype,
                    "output_dtype": output_dtype,
                }
                for ordinal, kind, input_dtype, output_dtype in self.denominator_path
            ),
            "division_expression": serialize_scalar_expression(self.division_expression),
        }


def fold_invariant_division_boundaries(program: TensorProgram) -> tuple[FoldInvariantDivisionBoundary, ...]:
    """Return exact typed division boundaries eligible for the FAST sink rewrite.

    The record retains the source Map and every Relation/conversion between its
    denominator and Fold state. Physical schedules can therefore choose a
    conversion-faithful two-pass realization without reconstructing a workload
    name or interpreting a score program.
    """
    producer_by_value = {
        output.name: (ordinal, operation)
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    boundaries: list[FoldInvariantDivisionBoundary] = []
    for contract_ordinal, operation in enumerate(program.operations):
        if not isinstance(operation, ContractPrimitive):
            continue
        for input_index, divided in enumerate(operation.inputs):
            produced = producer_by_value.get(divided.name)
            if produced is None or not isinstance(produced[1], MapPrimitive):
                continue
            division_ordinal, division = produced
            if division.expression.kind is not ScalarExpressionKind.DIVIDE:
                continue
            left, right = division.expression.operands
            if left.kind is not ScalarExpressionKind.INPUT or right.kind is not ScalarExpressionKind.INPUT:
                continue
            inputs = {value.name: value for value in division.inputs}
            numerator = inputs.get(left.input_name or "")
            denominator = inputs.get(right.input_name or "")
            if numerator is None or denominator is None or numerator.axes != divided.axes:
                continue
            current = denominator
            path: list[tuple[int, str, str, str]] = []
            while True:
                source = producer_by_value.get(current.name)
                if source is None:
                    break
                ordinal, producer = source
                if isinstance(producer, FoldPrimitive):
                    if set(current.axes) & set(operation.reduction_axes):
                        break
                    boundaries.append(
                        FoldInvariantDivisionBoundary(
                            contract_operation=contract_ordinal,
                            normalized_operand=input_index,
                            division_operation=division_ordinal,
                            numerator=numerator,
                            denominator=denominator,
                            fold_state=current,
                            denominator_path=tuple(path),
                            division_expression=division.expression,
                        )
                    )
                    break
                if isinstance(producer, MapPrimitive):
                    if producer.expression.kind is not ScalarExpressionKind.CONVERT or len(producer.inputs) != 1:
                        break
                    path.append(
                        (
                            ordinal,
                            "convert",
                            producer.inputs[0].dtype.value,
                            producer.output.dtype.value,
                        )
                    )
                    current = producer.inputs[0]
                    continue
                if isinstance(producer, RelationPrimitive):
                    if producer.predicate is not None or len(producer.inputs) != 1:
                        break
                    result_index = producer.outputs.index(current)
                    source_value = producer.resolved_result_sources[result_index]
                    if source_value is None:
                        break
                    path.append(
                        (
                            ordinal,
                            "relation",
                            source_value.dtype.value,
                            current.dtype.value,
                        )
                    )
                    current = source_value
                    continue
                break
    return tuple(boundaries)


def compose_scalar_map_chains(program: TensorProgram) -> TensorProgramRewrite:
    """Inline compatible pointwise Map producers into their scalar consumers.

    Differentiated JAX programs retain score-transform intermediates for the
    reverse pass.  This rewrite therefore leaves every producer in place and
    only gives each downstream Map an equivalent, transitively composed scalar
    expression.  Region extraction can then see the complete score program
    without requiring its intermediates to have a single use.
    """
    source_fingerprint = program.semantic_fingerprint
    if program.numerical_policy is not NumericalPolicy.FAST:
        return TensorProgramRewrite(source_fingerprint, program, ())

    producer_by_value: dict[str, TensorPrimitive] = {}
    producer_ordinal_by_value: dict[str, int] = {}
    rewritten: list[TensorPrimitive] = []
    composed_ordinals: list[tuple[int, int]] = []
    for ordinal, operation in enumerate(program.operations):
        if isinstance(operation, MapPrimitive):
            simplified_expression = _simplify_scalar_expression(operation.expression)
            if simplified_expression != operation.expression:
                operation = replace(operation, expression=simplified_expression)
        if not isinstance(operation, MapPrimitive) or len(operation.inputs) != 1:
            rewritten.append(operation)
            for output in primitive_outputs(operation):
                producer_by_value[output.name] = operation
                producer_ordinal_by_value[output.name] = ordinal
            continue
        input_value = operation.inputs[0]
        producer = producer_by_value.get(input_value.name)
        if not _composable_scalar_map(producer, input_value, operation):
            rewritten.append(operation)
            producer_by_value[operation.output.name] = operation
            producer_ordinal_by_value[operation.output.name] = ordinal
            continue
        assert isinstance(producer, MapPrimitive)
        expression = _substitute_scalar_input(
            operation.expression,
            input_name=input_value.name,
            replacement=producer.expression,
        )
        composed = replace(operation, inputs=producer.inputs, expression=expression)
        rewritten.append(composed)
        producer_by_value[composed.output.name] = composed
        producer_ordinal_by_value[composed.output.name] = ordinal
        composed_ordinals.append((producer_ordinal_by_value[input_value.name], ordinal))

    if not composed_ordinals:
        return TensorProgramRewrite(source_fingerprint, program, ())
    transformed = TensorProgram(program.inputs, tuple(rewritten), program.outputs, program.numerical_policy)
    explanation = RewriteExplanation(
        name="compose_compatible_scalar_map_chains",
        applied=True,
        original_fragment=("intermediate = scalar_map(x)", "result = scalar_map(intermediate)"),
        transformed_fragment=("result = composed_scalar_map(x)",),
        semantic_properties=(
            "each composed Map has one tensor input",
            "producer and consumer domains and scalar dtypes are identical",
            "producer operations remain available to residual reverse consumers",
            f"composed producer/consumer ordinals: {tuple(composed_ordinals)}",
        ),
        legality_checks=(
            "the request uses FAST numerical policy",
            "composition substitutes the producer AST for exactly its consumer input",
            "no tensor indexing or broadcasting is changed",
        ),
        estimated_benefit="exposes complete scalar programs to generic fused-region extraction",
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=(
            "removes intermediate FP32 materialization boundaries and may change compiler-selected rounding; "
            "SOURCE_ORDERED policy retains the original Map chain"
        ),
    )
    return TensorProgramRewrite(source_fingerprint, transformed, (explanation,))


def _simplify_scalar_expression(expression: ScalarExpression) -> ScalarExpression:
    """Canonicalize exact scalar identities before composing adjacent Maps."""
    operands = tuple(_simplify_scalar_expression(operand) for operand in expression.operands)
    expression = replace(expression, operands=operands)
    if expression.kind is not ScalarExpressionKind.MAXIMUM or len(operands) != 2:
        return expression
    left, right = operands
    if _is_scalar_constant(left, -math.inf):
        return right
    if _is_scalar_constant(right, -math.inf):
        return left
    return expression


def _is_scalar_constant(expression: ScalarExpression, value: float) -> bool:
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return expression.constant == value
    if expression.kind is ScalarExpressionKind.CONVERT and len(expression.operands) == 1:
        return _is_scalar_constant(expression.operands[0], value)
    return False


def _composable_scalar_map(
    producer: TensorPrimitive | None,
    intermediate: ProgramValue,
    consumer: MapPrimitive,
) -> bool:
    return (
        isinstance(producer, MapPrimitive)
        and len(producer.inputs) == 1
        and producer.output == intermediate
        and producer.output.axes == consumer.output.axes
        and producer.output.dtype is consumer.output.dtype
        and producer.output.dtype is not DType.BOOL
    )


def _substitute_scalar_input(
    expression: ScalarExpression,
    *,
    input_name: str,
    replacement: ScalarExpression,
) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        return replacement if expression.input_name == input_name else expression
    return replace(
        expression,
        operands=tuple(
            _substitute_scalar_input(operand, input_name=input_name, replacement=replacement)
            for operand in expression.operands
        ),
    )


def fold_widening_converts_into_contracts(program: TensorProgram) -> TensorProgramRewrite:
    """Represent private exact BF16-to-FP32 widening in accumulator semantics."""
    source_fingerprint = program.semantic_fingerprint
    uses: dict[ProgramValue, list[TensorPrimitive]] = {}
    producers: dict[ProgramValue, TensorPrimitive] = {}
    for operation in program.operations:
        for value in primitive_inputs(operation):
            uses.setdefault(value, []).append(operation)
        for value in primitive_outputs(operation):
            producers[value] = operation

    removed: set[int] = set()
    rewritten: list[TensorPrimitive] = []
    converted_ordinals: list[int] = []
    operation_ordinals = {id(operation): ordinal for ordinal, operation in enumerate(program.operations)}
    for operation in program.operations:
        if isinstance(operation, FoldPrimitive):
            conversion = producers.get(operation.input)
            if not _is_private_exact_accumulator_widening(conversion, operation.input, uses):
                rewritten.append(operation)
                continue
            assert isinstance(conversion, MapPrimitive)
            rewritten.append(replace(operation, input=conversion.inputs[0]))
            removed.add(id(conversion))
            converted_ordinals.append(operation_ordinals[id(conversion)])
            continue
        if not isinstance(operation, ContractPrimitive):
            rewritten.append(operation)
            continue
        inputs = list(operation.inputs)
        changed = False
        for input_index, value in enumerate(inputs):
            conversion = producers.get(value)
            if not _is_private_exact_accumulator_widening(conversion, value, uses):
                continue
            assert isinstance(conversion, MapPrimitive)
            inputs[input_index] = conversion.inputs[0]
            removed.add(id(conversion))
            converted_ordinals.append(operation_ordinals[id(conversion)])
            changed = True
        rewritten.append(replace(operation, inputs=tuple(inputs)) if changed else operation)

    if not removed:
        return TensorProgramRewrite(source_fingerprint, program, ())
    transformed = TensorProgram(
        program.inputs,
        tuple(operation for operation in rewritten if id(operation) not in removed),
        program.outputs,
        program.numerical_policy,
    )
    explanation = RewriteExplanation(
        name="fold_exact_widening_convert_into_contract",
        applied=True,
        original_fragment=("wide_operand = convert(narrow_operand)", "y = Contract(wide_operand, ... )"),
        transformed_fragment=("y = Contract(narrow_operand, ...; wide accumulation)",),
        semantic_properties=(
            "BF16 values are represented exactly in FP32",
            "Contract accumulation dtype remains unchanged",
            f"converted source operation ordinals: {tuple(sorted(converted_ordinals))}",
        ),
        legality_checks=(
            "conversion result has exactly one use",
            "the sole user is the rewritten Contract",
            "conversion is BF16 to FP32 round-to-nearest-even",
        ),
        estimated_benefit="exposes the physical BF16 Contract operand without materializing an FP32 tensor",
        numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
        numerical_effect="none; every BF16 value is exactly representable in FP32",
    )
    return TensorProgramRewrite(source_fingerprint, transformed, (explanation,))


def _is_private_exact_accumulator_widening(
    operation: TensorPrimitive | None,
    output: ProgramValue,
    uses: dict[ProgramValue, list[TensorPrimitive]],
) -> bool:
    if not isinstance(operation, MapPrimitive) or operation.output != output or len(operation.inputs) != 1:
        return False
    expression = operation.expression
    return (
        expression.kind is ScalarExpressionKind.CONVERT
        and expression.result_dtype is DType.FP32
        and expression.conversion_semantics is ScalarConversionSemantics.ROUND_NEAREST_EVEN
        and operation.inputs[0].dtype is DType.BF16
        and output.dtype is DType.FP32
        and len(uses.get(output, ())) == 1
        and isinstance(uses[output][0], (ContractPrimitive, FoldPrimitive))
    )


def sink_fold_invariant_division_through_contract(program: TensorProgram) -> TensorProgramRewrite:
    """Move a Fold-derived invariant division through a linear Contract.

    This applies the real-algebra identity ``Contract(x / s, w) =
    Contract(x, w) / s`` when ``s`` is invariant over the Contract reduction
    axes. The transformation is deliberately available only under FAST
    numerical policy because it changes finite-precision division and
    accumulation order.
    """
    source_fingerprint = program.semantic_fingerprint
    if program.numerical_policy is not NumericalPolicy.FAST:
        return TensorProgramRewrite(source_fingerprint, program, ())

    producer_by_value = {
        output.name: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    rewritten: list[TensorPrimitive] = []
    explanations: list[RewriteExplanation] = []
    for ordinal, operation in enumerate(program.operations):
        if not isinstance(operation, ContractPrimitive):
            rewritten.append(operation)
            continue
        match = _fold_invariant_division(operation, producer_by_value)
        if match is None:
            rewritten.append(operation)
            continue
        input_index, numerator, normalizer = match
        canonical_axes, domain_axis_replacements = _canonical_contract_output_axes(operation)
        accumulator = ProgramValue(
            f"{operation.output.name}.unnormalized.{ordinal}",
            canonical_axes,
            operation.output.dtype,
        )
        contract_inputs = list(operation.inputs)
        contract_inputs[input_index] = numerator
        unnormalized = replace(
            operation,
            name=f"{operation.name}.unnormalized",
            inputs=tuple(contract_inputs),
            output=accumulator,
            input_index_maps=tuple(
                tuple(
                    replace(mapping, domain_axis=domain_axis_replacements.get(mapping.domain_axis, mapping.domain_axis))
                    for mapping in mappings
                )
                for mappings in operation.input_index_maps
            ),
        )
        if not set(normalizer.axes) <= set(canonical_axes):
            raise ValueError("Fold owner axes cannot be mapped onto the Contract output domain")
        finalized = (
            operation.output
            if canonical_axes == operation.output.axes
            else ProgramValue(f"{operation.output.name}.finalized.{ordinal}", canonical_axes, operation.output.dtype)
        )
        finalizer = MapPrimitive(
            f"{operation.name}.fold_finalize",
            (accumulator, normalizer),
            finalized,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(accumulator.name),
                scalar_input(normalizer.name),
            ),
        )
        rewritten.append(unnormalized)
        rewritten.append(finalizer)
        if finalized != operation.output:
            rewritten.append(
                _project_canonical_contract_output(
                    finalized,
                    operation.output,
                    domain_axis_replacements=domain_axis_replacements,
                    name=f"{operation.name}.output_relation",
                )
            )
        explanations.append(
            RewriteExplanation(
                name="move_fold_invariant_division_through_contract",
                applied=True,
                original_fragment=(
                    "normalized = numerator / fold_state",
                    "y = Contract(normalized, other_operands)",
                ),
                transformed_fragment=(
                    "accumulator = Contract(numerator, other_operands)",
                    "y = accumulator / fold_state",
                ),
                semantic_properties=(
                    "Contract is linear in each operand",
                    "Fold state is invariant over the Contract reduction axes",
                    "operand index relations map the Fold owner axes onto the Contract output",
                ),
                legality_checks=(
                    "division Map has one tensor numerator and one Fold-derived denominator",
                    "numerator and divided operand have identical logical axes and dtype",
                    "numerical policy permits finite-precision reassociation",
                ),
                estimated_benefit="avoids materializing a normalized Contract operand",
                numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
                numerical_effect=(
                    "moves division after Contract accumulation and can change finite-precision rounding; "
                    "SOURCE_ORDERED policy retains the original program"
                ),
            )
        )

    transformed = TensorProgram(
        program.inputs,
        tuple(rewritten),
        program.outputs,
        program.numerical_policy,
    )
    transformed = remove_dead_operations(transformed)
    return TensorProgramRewrite(source_fingerprint, transformed, tuple(explanations))


def _fold_invariant_division(
    contract: ContractPrimitive,
    producer_by_value: dict[str, TensorPrimitive],
) -> tuple[int, ProgramValue, ProgramValue] | None:
    matches: list[tuple[int, ProgramValue, ProgramValue]] = []
    for input_index, value in enumerate(contract.inputs):
        division = producer_by_value.get(value.name)
        if not isinstance(division, MapPrimitive) or division.expression.kind is not ScalarExpressionKind.DIVIDE:
            continue
        left, right = division.expression.operands
        if left.kind is not ScalarExpressionKind.INPUT or right.kind is not ScalarExpressionKind.INPUT:
            continue
        assert left.input_name is not None and right.input_name is not None
        inputs = {input_value.name: input_value for input_value in division.inputs}
        numerator = inputs.get(left.input_name)
        denominator = inputs.get(right.input_name)
        if numerator is None or denominator is None:
            continue
        normalizer = _broadcast_fold_source(denominator, producer_by_value)
        if normalizer is None or set(normalizer.axes) & set(contract.reduction_axes):
            continue
        if numerator.axes != value.axes or numerator.dtype is not value.dtype:
            continue
        matches.append((input_index, numerator, normalizer))
    if len(matches) != 1:
        return None
    return matches[0]


def _broadcast_fold_source(
    value: ProgramValue,
    producer_by_value: dict[str, TensorPrimitive],
) -> ProgramValue | None:
    current = value
    while True:
        producer = producer_by_value.get(current.name)
        if isinstance(producer, FoldPrimitive):
            return current
        if (
            isinstance(producer, MapPrimitive)
            and producer.expression.kind is ScalarExpressionKind.CONVERT
            and len(producer.inputs) == 1
        ):
            # A Fold-state storage conversion remains invariant over the
            # downstream Contract reduction. Under FAST, keep the Fold state
            # in its accumulation dtype and let the post-Contract finalizer
            # perform the result conversion once.
            current = producer.inputs[0]
            continue
        if not isinstance(producer, RelationPrimitive) or producer.predicate is not None:
            return None
        try:
            output_index = producer.outputs.index(current)
        except ValueError:
            return None
        if producer.resolved_result_expressions[output_index] is not None:
            return None
        source = producer.resolved_result_sources[output_index]
        if source is None:
            return None
        indexing = next((item for item in producer.input_indexing if item.operand == source), None)
        if indexing is None or indexing.predicate is not None:
            return None
        for axis, expression in zip(source.axes, indexing.indices, strict=True):
            if expression.kind is IndexExpressionKind.AXIS and expression.axis == axis:
                continue
            if expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and axis.extent == 1:
                continue
            return None
        current = source


def _canonical_contract_output_axes(
    operation: ContractPrimitive,
) -> tuple[tuple[TensorAxis, ...], dict[TensorAxis, TensorAxis]]:
    replacements: dict[TensorAxis, TensorAxis] = {}
    for input_index in range(len(operation.inputs)):
        for mapping in operation.index_maps_for_input(input_index):
            previous = replacements.get(mapping.domain_axis)
            if previous is not None and previous != mapping.operand_axis:
                raise ValueError("Contract output axis has incompatible operand index relations")
            replacements[mapping.domain_axis] = mapping.operand_axis
    canonical_axes = tuple(replacements.get(axis, axis) for axis in operation.output.axes)
    if len(set(canonical_axes)) != len(canonical_axes):
        raise ValueError("Contract output index relations do not define unique canonical axes")
    return canonical_axes, replacements


def _project_canonical_contract_output(
    source: ProgramValue,
    output: ProgramValue,
    *,
    domain_axis_replacements: dict[TensorAxis, TensorAxis],
    name: str,
) -> RelationPrimitive:
    source_for_domain = {replacement: domain for domain, replacement in domain_axis_replacements.items()}
    indices = tuple(index_axis(source_for_domain.get(axis, axis)) for axis in source.axes)
    return RelationPrimitive(
        name,
        (source,),
        (output,),
        output.axes,
        (OperandIndexing(source, output.axes, indices),),
        result_sources=(source,),
    )


def remove_dead_operations(program: TensorProgram) -> TensorProgram:
    """Drop pure tensor operations whose results cannot reach a program output.

    TensorProgram primitives are referentially transparent.  Liveness is
    therefore an algebraic quotient, not a scheduling choice, and is useful at
    boundaries where one frontend retains an unused reduction initializer as a
    full tensor while another represents the live Fold initializer explicitly
    as a scalar.
    """
    live_values = set(program.outputs)
    retained: list[TensorPrimitive] = []
    for operation in reversed(program.operations):
        if not any(output in live_values for output in primitive_outputs(operation)):
            continue
        retained.append(operation)
        live_values.update(primitive_inputs(operation))
    retained.reverse()
    return TensorProgram(program.inputs, tuple(retained), program.outputs, program.numerical_policy)
