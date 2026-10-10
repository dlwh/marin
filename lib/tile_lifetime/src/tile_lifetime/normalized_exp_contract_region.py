# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared algebra and ownership for normalized-exponential contractions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, TypeVar

from shuttle.ir import DType
from tile_lifetime.autodiff import adjoint_contract_operand_precision, scalar_expression_vjp
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    MapPrimitive,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    canonical_axis_references,
    index_axis,
    index_binary,
    index_constant,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_expression_inputs,
    scalar_input,
    tensor_program_value_ordinals,
)


class ContractDomainTraversal(StrEnum):
    """Traversal of the producer/consumer relation, independent of workload names."""

    FULL = "full"
    LOWER_TRIANGULAR = "lower_triangular"


class IndexComparisonKind(StrEnum):
    """Typed comparisons over logical Contract coordinates."""

    SIGNED_LESS_EQUAL = "signed_less_equal"


@dataclass(frozen=True)
class AffineIndexExpression:
    """A canonical affine integer expression keyed by ``TensorAxis.id``."""

    coefficients: tuple[tuple[int, int], ...] = ()
    constant: int = 0

    def __post_init__(self) -> None:
        if any(
            type(dimension) is not int or type(coefficient) is not int or coefficient == 0
            for dimension, coefficient in self.coefficients
        ):
            raise ValueError("affine index coefficients must be nonzero integer dimension/coefficient pairs")
        dimensions = tuple(dimension for dimension, _ in self.coefficients)
        if dimensions != tuple(sorted(dimensions)) or len(set(dimensions)) != len(dimensions):
            raise ValueError("affine index dimensions must be unique and sorted")
        if type(self.constant) is not int:
            raise ValueError("affine index constants must be integers")

    @property
    def canonical_record(self) -> dict[str, object]:
        return {"coefficients": self.coefficients, "constant": self.constant}


@dataclass(frozen=True)
class AffineIndexPredicate:
    """One typed affine comparison selecting a Contract iteration domain."""

    comparison: IndexComparisonKind
    left: AffineIndexExpression
    right: AffineIndexExpression

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "comparison": self.comparison.value,
            "left": self.left.canonical_record,
            "right": self.right.canonical_record,
        }


class ReverseContractFoldOrder(StrEnum):
    """Finite-precision Fold order for a relation-mapped reverse traversal."""

    STREAMED_AXES_MAJOR_MAPPED_RELATION_MINOR_TREE = "streamed_axes_major_mapped_relation_minor_tree"


@dataclass(frozen=True)
class NormalizedExpFoldState:
    """The two Folds and scalar Maps implementing a normalized exponential."""

    maximum: FoldPrimitive
    center: MapPrimitive
    exponential: MapPrimitive
    normalizer: FoldPrimitive

    @property
    def row_max(self) -> ProgramValue:
        return self.maximum.output

    @property
    def row_sum_exp(self) -> ProgramValue:
        return self.normalizer.output


@dataclass(frozen=True)
class NormalizedExpContractionDimensions:
    """Distinct reduction and result dimensions of the two contractions."""

    producer_features: tuple[TensorAxis, ...]
    normalized_axes: tuple[TensorAxis, ...]
    consumer_features: tuple[TensorAxis, ...]
    shared_relation_axes: tuple[TensorAxis, ...] = ()
    relation_group_axes: tuple[TensorAxis, ...] = ()

    def __post_init__(self) -> None:
        if not self.producer_features:
            raise ValueError("normalized-exp producer Contract must have a feature reduction")
        if not self.normalized_axes:
            raise ValueError("normalized-exp Fold must have a reduction axis")
        if not self.consumer_features:
            raise ValueError("normalized-exp consumer Contract must produce a feature axis")
        dimensions = (
            *self.producer_features,
            *self.normalized_axes,
            *self.consumer_features,
            *self.shared_relation_axes,
            *self.relation_group_axes,
        )
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("producer features, normalized axes, consumer features, and relation axes must be distinct")


@dataclass(frozen=True)
class NormalizedExpContractRegionCapability:
    """Canonical operation/value roles declared by algebra planning.

    Operation ordinals refer to the containing canonical TensorProgram or its
    source-preserving native lowering. Value ordinals use that boundary's
    canonical SSA namespace. The physical provider validates these references;
    it never rediscovers the region by matching graph topology.
    """

    ordinal: int
    producer_operation: int
    producer_owner_operand: int
    score_operation: int
    score_value: int
    fold_operations: tuple[int, ...]
    maximum_value: int
    normalizer_value: int
    normalized_weight_value: int
    normalized_weight_operation: int
    consumer_operation: int
    normalized_operand: int
    finalizer_value: int
    finalizer_operation: int
    finalizer_tail_operations: tuple[int, ...]
    output_result: int | None
    producer_features: tuple[str, ...]
    normalized_axes: tuple[str, ...]
    consumer_features: tuple[str, ...]
    shared_relation_axes: tuple[str, ...] = ()
    relation_group_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        integers = (
            self.ordinal,
            self.producer_operation,
            self.producer_owner_operand,
            self.score_operation,
            self.score_value,
            self.maximum_value,
            self.normalizer_value,
            self.normalized_weight_value,
            self.normalized_weight_operation,
            self.consumer_operation,
            self.normalized_operand,
            self.finalizer_value,
            self.finalizer_operation,
            *self.finalizer_tail_operations,
            *self.fold_operations,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("normalized-exp Contract capability ordinals must be nonnegative integers")
        if self.output_result is not None and (type(self.output_result) is not int or self.output_result < 0):
            raise ValueError("normalized-exp Contract public result ordinal must be nonnegative")
        if not self.fold_operations or len(set(self.fold_operations)) != len(self.fold_operations):
            raise ValueError("normalized-exp Contract capability requires distinct Fold-program operations")

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "producer_operation": self.producer_operation,
            "producer_owner_operand": self.producer_owner_operand,
            "score_operation": self.score_operation,
            "score_value": self.score_value,
            "fold_operations": self.fold_operations,
            "maximum_value": self.maximum_value,
            "normalizer_value": self.normalizer_value,
            "normalized_weight_value": self.normalized_weight_value,
            "normalized_weight_operation": self.normalized_weight_operation,
            "consumer_operation": self.consumer_operation,
            "normalized_operand": self.normalized_operand,
            "finalizer_value": self.finalizer_value,
            "finalizer_operation": self.finalizer_operation,
            "finalizer_tail_operations": self.finalizer_tail_operations,
            "output_result": self.output_result,
            "producer_features": self.producer_features,
            "normalized_axes": self.normalized_axes,
            "consumer_features": self.consumer_features,
            "shared_relation_axes": self.shared_relation_axes,
            "relation_group_axes": self.relation_group_axes,
        }


@dataclass(frozen=True)
class NormalizedExpContractPullbackCapability:
    """Exact operation roles for the pullback of a declared forward region."""

    ordinal: int
    forward_region_ordinal: int
    normalized_output_operation: int
    post_normalization_vjp_operation: int
    output_product_operation: int
    output_dependent_fold_operation: int
    consumer_weight_cotangent_operation: int
    normalized_weight_operation: int
    normalized_exp_pullback_operation: int
    score_map_vjp_operation: int
    terminal_contract_operations: tuple[int, ...]

    def __post_init__(self) -> None:
        ordinals = (
            self.ordinal,
            self.forward_region_ordinal,
            self.normalized_output_operation,
            self.post_normalization_vjp_operation,
            self.output_product_operation,
            self.output_dependent_fold_operation,
            self.consumer_weight_cotangent_operation,
            self.normalized_weight_operation,
            self.normalized_exp_pullback_operation,
            self.score_map_vjp_operation,
            *self.terminal_contract_operations,
        )
        if any(type(value) is not int or value < 0 for value in ordinals):
            raise ValueError("normalized-exp pullback capability ordinals must be nonnegative integers")
        if len(self.terminal_contract_operations) != 3:
            raise ValueError("normalized-exp pullback capability requires three terminal gradient Contracts")

    @property
    def operation_ordinals(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.normalized_output_operation,
                    self.post_normalization_vjp_operation,
                    self.output_product_operation,
                    self.output_dependent_fold_operation,
                    self.consumer_weight_cotangent_operation,
                    self.normalized_weight_operation,
                    self.normalized_exp_pullback_operation,
                    self.score_map_vjp_operation,
                    *self.terminal_contract_operations,
                )
            )
        )

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "forward_region_ordinal": self.forward_region_ordinal,
            "normalized_output_operation": self.normalized_output_operation,
            "post_normalization_vjp_operation": self.post_normalization_vjp_operation,
            "output_product_operation": self.output_product_operation,
            "output_dependent_fold_operation": self.output_dependent_fold_operation,
            "consumer_weight_cotangent_operation": self.consumer_weight_cotangent_operation,
            "normalized_weight_operation": self.normalized_weight_operation,
            "normalized_exp_pullback_operation": self.normalized_exp_pullback_operation,
            "score_map_vjp_operation": self.score_map_vjp_operation,
            "terminal_contract_operations": self.terminal_contract_operations,
        }


@dataclass(frozen=True)
class NormalizedExpContractRegion:
    """A Contract/Map/normalized-exp-Fold/Contract/Map algebraic region.

    The region retains the original :class:`AxisIndexMap` values on each
    Contract. Head sharing, sharding, and other operand reuse are therefore
    properties of index relations rather than semantic modes.
    """

    producer: ContractPrimitive
    score_map: MapPrimitive
    fold_state: NormalizedExpFoldState
    consumer: ContractPrimitive
    finalizer: MapPrimitive
    dimensions: NormalizedExpContractionDimensions
    domain_predicate: AffineIndexPredicate | None = None

    def __post_init__(self) -> None:
        maximum = self.fold_state.maximum
        center = self.fold_state.center
        exponential = self.fold_state.exponential
        normalizer = self.fold_state.normalizer
        if self.score_map.predicate is not None and self.score_map.rejected_value != -math.inf:
            raise ValueError("normalized-exp score Map predicate must reject with negative infinity")
        if self.producer.output not in self.score_map.inputs or maximum.input != self.score_map.output:
            raise ValueError("score Map must consume the producer Contract and feed the maximum Fold domain")
        if maximum.reducer is not FoldReducer.MAXIMUM:
            raise ValueError("normalized-exp maximum state must use a maximum Fold")
        if normalizer.reducer is not FoldReducer.SUM:
            raise ValueError("normalized-exp normalizer state must use a sum Fold")
        if maximum.reduction_axes != self.dimensions.normalized_axes:
            raise ValueError("normalized-exp dimensions disagree with the maximum Fold")
        if normalizer.reduction_axes != self.dimensions.normalized_axes:
            raise ValueError("normalized-exp Folds must reduce the same logical axes")
        if self.producer.reduction_axes != self.dimensions.producer_features:
            raise ValueError("normalized-exp dimensions disagree with the producer Contract")
        if self.consumer.reduction_axes != self.dimensions.normalized_axes:
            raise ValueError("consumer Contract must reduce the normalized-exponential axes")
        _expect_expression(
            center,
            ScalarExpressionKind.SUBTRACT,
            (self.score_map.output.name, maximum.output.name),
        )
        _expect_expression(exponential, ScalarExpressionKind.EXP, (center.output.name,))
        if normalizer.input != exponential.output or self.consumer.inputs[0] != exponential.output:
            raise ValueError("exponential values must feed both the sum Fold and consumer Contract")
        if self.consumer.output not in self.finalizer.inputs or normalizer.output not in self.finalizer.inputs:
            raise ValueError("finalizer must consume the weighted Contract and normalized-exp sum")
        if self.finalizer.output.axes != self.consumer.output.axes:
            raise ValueError("finalizer must preserve the consumer Contract output domain")
        owner_axes = set(maximum.output.axes)
        expected_consumer_features = tuple(axis for axis in self.consumer.output.axes if axis not in owner_axes)
        if self.dimensions.consumer_features != expected_consumer_features:
            raise ValueError("normalized-exp dimensions disagree with the consumer Contract result features")

    @property
    def producer_index_relations(self) -> tuple[tuple[AxisIndexMap, ...], ...]:
        return tuple(self.producer.index_maps_for_input(index) for index in range(len(self.producer.inputs)))

    @property
    def consumer_index_relations(self) -> tuple[tuple[AxisIndexMap, ...], ...]:
        return tuple(self.consumer.index_maps_for_input(index) for index in range(len(self.consumer.inputs)))

    @property
    def owner_axes(self) -> tuple[TensorAxis, ...]:
        return self.fold_state.normalizer.output.axes


@dataclass(frozen=True)
class ForwardOwnershipPlan:
    """Owner-computes plan for the shared forward region."""

    owner_axes: tuple[TensorAxis, ...]
    streamed_axes: tuple[TensorAxis, ...]
    producer_reduction_axes: tuple[TensorAxis, ...]
    consumer_feature_axes: tuple[TensorAxis, ...]
    producer_index_relations: tuple[tuple[AxisIndexMap, ...], ...]
    consumer_index_relations: tuple[tuple[AxisIndexMap, ...], ...]


ScheduleT = TypeVar("ScheduleT")
ReverseScheduleT = TypeVar("ReverseScheduleT")


@dataclass(frozen=True)
class NormalizedExpContractProgram(Generic[ScheduleT]):
    """One algebraic normalized-exp Contract region and its chosen schedule."""

    source: TensorProgram
    region: NormalizedExpContractRegion
    ownership: ForwardOwnershipPlan
    schedule: ScheduleT
    materialized_values: tuple[ProgramValue, ...]

    def __post_init__(self) -> None:
        expected_ownership = derive_forward_ownership_plan(self.region)
        if self.ownership != expected_ownership:
            raise ValueError("forward ownership must be derived from the normalized-exp Contract region")

    @property
    def canonical_region_record(self) -> dict[str, object]:
        """Return the schedule-independent canonical record for this region."""
        return canonical_normalized_exp_contract_record(self)

    @property
    def region_fingerprint(self) -> str:
        """Return the schedule-independent semantic fingerprint for this region."""
        return normalized_exp_contract_fingerprint(self)


def normalized_exp_contract_region_capability(
    program: NormalizedExpContractProgram[object],
    *,
    ordinal: int = 0,
) -> NormalizedExpContractRegionCapability:
    """Encode exact source roles from an already typed algebraic region."""
    return normalized_exp_region_capability(program.source, program.region, ordinal=ordinal)


def normalized_exp_region_capability(
    source: TensorProgram,
    region: NormalizedExpContractRegion,
    *,
    ordinal: int = 0,
) -> NormalizedExpContractRegionCapability:
    """Encode exact roles for a normalized-exp region in a TensorProgram."""
    operations = source.operations
    value_ordinals = tensor_program_value_ordinals(source)

    def operation_ordinal(operation: TensorPrimitive) -> int:
        outputs = primitive_outputs(operation)
        matches = tuple(
            index
            for index, candidate in enumerate(operations)
            if type(candidate) is type(operation) and primitive_outputs(candidate) == outputs
        )
        if len(matches) != 1:
            raise ValueError("normalized-exp region operation is not unique in its source TensorProgram")
        return matches[0]

    try:
        producer_operation = operation_ordinal(region.producer)
        producer_owner_operand = operations[producer_operation].inputs.index(region.producer.inputs[0])
        consumer_operation = operation_ordinal(region.consumer)
        normalized_operand = operations[consumer_operation].inputs.index(region.fold_state.exponential.output)
    except ValueError as error:
        raise ValueError("normalized-exp region consumer does not bind its declared normalized value") from error
    producer_by_value = {output: operation for operation in operations for output in primitive_outputs(operation)}
    output_candidates: list[tuple[int, tuple[TensorPrimitive, ...]]] = []
    for output_result, output in enumerate(source.outputs):
        current = output
        tail: list[TensorPrimitive] = []
        visited: set[ProgramValue] = set()
        while current != region.finalizer.output:
            if current in visited:
                break
            visited.add(current)
            operation = producer_by_value.get(current)
            if isinstance(operation, MapPrimitive) and len(operation.inputs) == 1:
                current = operation.inputs[0]
            elif isinstance(operation, RelationPrimitive) and len(operation.inputs) == 1:
                current = operation.inputs[0]
            else:
                break
            tail.append(operation)
        if current == region.finalizer.output:
            output_candidates.append((output_result, tuple(reversed(tail))))
    if len(output_candidates) > 1:
        raise ValueError(
            "normalized-exp region finalizer reaches multiple public results; declare a pullback boundary explicitly"
        )
    output_result, finalizer_tail = output_candidates[0] if output_candidates else (None, ())
    return NormalizedExpContractRegionCapability(
        ordinal=ordinal,
        producer_operation=producer_operation,
        producer_owner_operand=producer_owner_operand,
        score_operation=operation_ordinal(region.score_map),
        score_value=value_ordinals[region.score_map.output],
        fold_operations=tuple(
            operation_ordinal(operation)
            for operation in (
                region.fold_state.maximum,
                region.fold_state.center,
                region.fold_state.exponential,
                region.fold_state.normalizer,
            )
        ),
        maximum_value=value_ordinals[region.fold_state.maximum.output],
        normalizer_value=value_ordinals[region.fold_state.normalizer.output],
        normalized_weight_value=value_ordinals[region.fold_state.exponential.output],
        normalized_weight_operation=operation_ordinal(region.fold_state.exponential),
        consumer_operation=consumer_operation,
        normalized_operand=normalized_operand,
        finalizer_value=value_ordinals[region.finalizer.output],
        finalizer_operation=operation_ordinal(region.finalizer),
        finalizer_tail_operations=tuple(operation_ordinal(operation) for operation in finalizer_tail),
        output_result=output_result,
        producer_features=canonical_axis_references(source, region.dimensions.producer_features),
        normalized_axes=canonical_axis_references(source, region.dimensions.normalized_axes),
        consumer_features=canonical_axis_references(source, region.dimensions.consumer_features),
        shared_relation_axes=canonical_axis_references(source, region.dimensions.shared_relation_axes),
        relation_group_axes=canonical_axis_references(source, region.dimensions.relation_group_axes),
    )


def normalized_exp_pullback_capability(
    source: TensorProgram,
    pullback: NormalizedExpContractPullback,
    *,
    ordinal: int = 0,
    forward_region_ordinal: int = 0,
) -> NormalizedExpContractPullbackCapability:
    """Bind a typed pullback's roles into its containing TensorProgram."""

    def operation_ordinal(operation: TensorPrimitive) -> int:
        output_names = tuple(output.name for output in primitive_outputs(operation))
        matches = tuple(
            index
            for index, candidate in enumerate(source.operations)
            if type(candidate) is type(operation)
            and tuple(output.name for output in primitive_outputs(candidate)) == output_names
        )
        if len(matches) != 1:
            raise ValueError(
                "normalized-exp pullback operation is not unique in its source TensorProgram: "
                f"outputs={output_names}, matches={matches}"
            )
        return matches[0]

    producer_by_value = {
        output.name: operation for operation in source.operations for output in primitive_outputs(operation)
    }

    def terminal_contract_ordinal(gradient: TerminalGradientContract) -> int:
        destination = gradient.destination
        source_destination = next(
            (value for value in (*source.inputs, *source.outputs) if value.name == destination.name),
            None,
        )
        if source_destination is None:
            source_destination = next(
                (
                    value
                    for operation in source.operations
                    for value in primitive_outputs(operation)
                    if value.name == destination.name
                ),
                None,
            )
        if source_destination is None:
            input_names = tuple(value.name for value in primitive_inputs(gradient.contract))
            matches = tuple(
                index
                for index, operation in enumerate(source.operations)
                if isinstance(operation, ContractPrimitive)
                and tuple(value.name for value in primitive_inputs(operation)) == input_names
            )
            if len(matches) != 1:
                raise ValueError(
                    "normalized-exp terminal gradient does not rebind through layout canonicalization: "
                    f"destination={destination.name}, inputs={input_names}, matches={matches}"
                )
            return matches[0]
        pending = [source_destination]
        visited: set[str] = set()
        contracts: set[int] = set()
        while pending:
            value = pending.pop()
            if value.name in visited:
                continue
            visited.add(value.name)
            producer = producer_by_value.get(value.name)
            if producer is None:
                continue
            if isinstance(producer, ContractPrimitive):
                contracts.add(source.operations.index(producer))
                continue
            pending.extend(primitive_inputs(producer))
        if len(contracts) != 1:
            raise ValueError(
                "normalized-exp terminal gradient destination does not bind one terminal Contract: "
                f"destination={destination.name}, contracts={tuple(sorted(contracts))}"
            )
        return next(iter(contracts))

    terminal_contracts = tuple(terminal_contract_ordinal(gradient) for gradient in pullback.terminal_gradients)
    if len(set(terminal_contracts)) != len(terminal_contracts):
        raise ValueError("normalized-exp pullback terminal destinations do not bind distinct Contracts")

    return NormalizedExpContractPullbackCapability(
        ordinal=ordinal,
        forward_region_ordinal=forward_region_ordinal,
        normalized_output_operation=operation_ordinal(pullback.normalized_output),
        post_normalization_vjp_operation=operation_ordinal(pullback.post_normalization_vjp),
        output_product_operation=operation_ordinal(pullback.output_product),
        output_dependent_fold_operation=operation_ordinal(pullback.output_dependent_fold),
        consumer_weight_cotangent_operation=operation_ordinal(pullback.consumer_weight_cotangent),
        normalized_weight_operation=operation_ordinal(pullback.normalized_weight_map),
        normalized_exp_pullback_operation=operation_ordinal(pullback.normalized_exp_pullback),
        score_map_vjp_operation=operation_ordinal(pullback.score_map_vjp),
        terminal_contract_operations=terminal_contracts,
    )


def canonical_normalized_exp_contract_record(
    program: NormalizedExpContractProgram[object],
) -> dict[str, object]:
    """Project a normalized-exp region into the TensorProgram canonical namespace."""
    dimensions = program.region.dimensions
    return {
        "format": "shuttle_normalized_exp_contract_region_v1",
        "algebra": program.source.canonical_record,
        "contraction_dimensions": {
            "producer_features": canonical_axis_references(program.source, dimensions.producer_features),
            "normalized_axes": canonical_axis_references(program.source, dimensions.normalized_axes),
            "consumer_features": canonical_axis_references(program.source, dimensions.consumer_features),
            "shared_relation_axes": canonical_axis_references(program.source, dimensions.shared_relation_axes),
            "relation_group_axes": canonical_axis_references(program.source, dimensions.relation_group_axes),
        },
        "domain_predicate": _canonical_domain_predicate(program),
    }


def normalized_exp_contract_fingerprint(program: NormalizedExpContractProgram[object]) -> str:
    """Hash the schedule-independent canonical normalized-exp region record."""
    payload = json.dumps(
        canonical_normalized_exp_contract_record(program),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_domain_predicate(program: NormalizedExpContractProgram[object]) -> dict[str, object] | None:
    predicate = program.region.domain_predicate
    if predicate is None:
        return None
    axes_by_id: dict[int, TensorAxis] = {}
    for axis in program.region.producer.output.axes:
        if axis.id in axes_by_id:
            raise ValueError("normalized-exp producer output axis ids must be unique")
        axes_by_id[axis.id] = axis

    def expression_record(expression: AffineIndexExpression) -> dict[str, object]:
        try:
            axes = tuple(axes_by_id[dimension] for dimension, _ in expression.coefficients)
        except KeyError as error:
            raise ValueError("normalized-exp domain predicate references an axis outside the producer output") from error
        axis_references = canonical_axis_references(program.source, axes)
        return {
            "coefficients": tuple(
                (axis_reference, coefficient)
                for axis_reference, (_, coefficient) in zip(axis_references, expression.coefficients, strict=True)
            ),
            "constant": expression.constant,
        }

    return {
        "comparison": predicate.comparison.value,
        "left": expression_record(predicate.left),
        "right": expression_record(predicate.right),
    }


@dataclass(frozen=True)
class TerminalGradientContract:
    """One terminal Contract and its relation to the destination operand."""

    operand: ProgramValue
    contract: ContractPrimitive
    destination: ProgramValue
    destination_index_relations: tuple[AxisIndexMap, ...] = ()


@dataclass(frozen=True)
class NormalizedExpContractPullback:
    """Reverse algebra derived from :class:`NormalizedExpContractRegion`."""

    region: NormalizedExpContractRegion
    output_cotangent: ProgramValue
    normalized_output: MapPrimitive
    post_normalization: MapPrimitive
    post_normalization_vjp: MapPrimitive
    output_product: MapPrimitive
    output_dependent_fold: FoldPrimitive
    consumer_weight_cotangent: ContractPrimitive
    normalized_weight: ProgramValue
    normalized_weight_map: MapPrimitive
    normalized_exp_pullback: MapPrimitive
    score_map_vjp: MapPrimitive
    terminal_gradients: tuple[TerminalGradientContract, ...]

    @property
    def output_scale(self) -> float:
        """Return the legacy linear-finalizer coefficient when one exists."""
        return normalized_exp_finalizer_scale(self.region)


@dataclass(frozen=True)
class ReverseOwnershipPlan:
    """Relation-derived owner and stream axes for both reverse traversals."""

    left_owner_axes: tuple[TensorAxis, ...]
    right_owner_axes: tuple[TensorAxis, ...]
    left_streamed_axes: tuple[TensorAxis, ...]
    right_streamed_axes: tuple[TensorAxis, ...]
    traversal: ContractDomainTraversal
    producer_index_relations: tuple[tuple[AxisIndexMap, ...], ...]
    consumer_index_relations: tuple[tuple[AxisIndexMap, ...], ...]

    @property
    def index_relations(self) -> tuple[tuple[tuple[AxisIndexMap, ...], ...], ...]:
        return (self.producer_index_relations, self.consumer_index_relations)


@dataclass(frozen=True)
class NormalizedExpContractPullbackProgram(Generic[ScheduleT, ReverseScheduleT]):
    """A shared normalized-exp Contract region and its scheduled pullback.

    This is the backend boundary for reverse lowering.  It deliberately
    contains no workload provenance or named differentiation mode: the
    pullback algebra, relation-derived ownership, finite-precision Fold order,
    and selected physical schedule are the complete contract.
    """

    forward: NormalizedExpContractProgram[ScheduleT]
    pullback: NormalizedExpContractPullback
    ownership: ReverseOwnershipPlan
    right_owner_fold_order: ReverseContractFoldOrder
    schedule: ReverseScheduleT
    materialized_values: tuple[ProgramValue, ...]

    def __post_init__(self) -> None:
        if self.pullback.region != self.forward.region:
            raise ValueError("normalized-exp pullback must derive from the program's forward region")
        expected_ownership = derive_reverse_ownership_plan(
            self.pullback,
            traversal=self.ownership.traversal,
        )
        if self.ownership != expected_ownership:
            raise ValueError("reverse ownership must be derived from the normalized-exp Contract pullback")

    @property
    def dimensions(self) -> NormalizedExpContractionDimensions:
        return self.forward.region.dimensions


def normalized_exp_contract_region(
    program: TensorProgram,
    *,
    output: ProgramValue | None = None,
) -> NormalizedExpContractRegion:
    """Extract one region by following value producers, independent of operation order."""
    producer_by_value = {
        output.name: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    if output is not None:
        candidates = (output,)
    else:
        candidates = tuple(operation.output for operation in program.operations if isinstance(operation, MapPrimitive))
    regions: list[NormalizedExpContractRegion] = []
    rejections: list[str] = []
    for candidate in candidates:
        try:
            region = _region_from_finalizer_output(candidate, producer_by_value)
        except ValueError as error:
            rejections.append(f"{candidate.name}: {error}")
            continue
        if region not in regions:
            regions.append(region)
    if len(regions) == 1:
        return regions[0]
    if not regions:
        detail = rejections[0] if output is not None and rejections else "no finalizer closes the local algebra"
        raise ValueError(f"no normalized-exp Contract region found: {detail}")
    raise ValueError("multiple normalized-exp Contract regions found; select one finalizer output explicitly")


def _region_from_finalizer_output(
    output: ProgramValue,
    producer_by_value: dict[str, TensorPrimitive],
) -> NormalizedExpContractRegion:
    finalizer = producer_by_value.get(output.name)
    if not isinstance(finalizer, MapPrimitive):
        raise ValueError("candidate output is not produced by a finalizer Map")
    finalizer_producers = tuple(producer_by_value.get(value.name) for value in finalizer.inputs)
    consumers = tuple(operation for operation in finalizer_producers if isinstance(operation, ContractPrimitive))
    normalizers = tuple(
        operation
        for operation in finalizer_producers
        if isinstance(operation, FoldPrimitive) and operation.reducer is FoldReducer.SUM
    )
    if len(consumers) != 1 or len(normalizers) != 1:
        raise ValueError("finalizer inputs do not identify one weighted Contract and one sum Fold")
    consumer = consumers[0]
    normalizer = normalizers[0]

    exponential = producer_by_value.get(normalizer.input.name)
    if not isinstance(exponential, MapPrimitive) or exponential.expression.kind is not ScalarExpressionKind.EXP:
        raise ValueError("sum Fold input is not produced by an exponential Map")
    if not exponential.inputs:
        raise ValueError("exponential Map has no centered-score input")
    center = producer_by_value.get(exponential.inputs[0].name)
    if not isinstance(center, MapPrimitive) or center.expression.kind is not ScalarExpressionKind.SUBTRACT:
        raise ValueError("exponential Map input is not produced by a centering Map")

    center_producers = tuple(producer_by_value.get(value.name) for value in center.inputs)
    maxima = tuple(
        operation
        for operation in center_producers
        if isinstance(operation, FoldPrimitive) and operation.reducer is FoldReducer.MAXIMUM
    )
    direct_score_maps = tuple(operation for operation in center_producers if isinstance(operation, MapPrimitive))
    if len(maxima) != 1 or len(direct_score_maps) != 1:
        raise ValueError("centering Map inputs do not identify one score domain and one maximum Fold")
    maximum = maxima[0]
    score_map = direct_score_maps[0]

    score_producers = tuple(producer_by_value.get(value.name) for value in score_map.inputs)
    producers = tuple(operation for operation in score_producers if isinstance(operation, ContractPrimitive))
    if len(producers) != 1:
        raise ValueError("score Map inputs do not identify one producer Contract")
    producer = producers[0]
    normalized_axes = set(maximum.reduction_axes)
    streamed_inputs = tuple(
        input_index
        for input_index, value in enumerate(producer.inputs)
        if normalized_axes & set(value.axes)
        or any(mapping.domain_axis in normalized_axes for mapping in producer.index_maps_for_input(input_index))
    )
    if len(streamed_inputs) != 1:
        raise ValueError("producer Contract must identify one operand over the normalized axes")
    if streamed_inputs[0] == 0:
        input_index_maps = producer.input_index_maps or ((),) * len(producer.inputs)
        producer = replace(
            producer,
            inputs=(producer.inputs[1], producer.inputs[0]),
            input_index_maps=(input_index_maps[1], input_index_maps[0]),
        )
    try:
        normalized_input_index = consumer.inputs.index(exponential.output)
    except ValueError as error:
        raise ValueError("exponential values do not feed the weighted Contract") from error
    if normalized_input_index != 0:
        permutation = (
            normalized_input_index,
            *(index for index in range(len(consumer.inputs)) if index != normalized_input_index),
        )
        input_index_maps = consumer.input_index_maps or ((),) * len(consumer.inputs)
        consumer = replace(
            consumer,
            inputs=tuple(consumer.inputs[index] for index in permutation),
            input_index_maps=tuple(input_index_maps[index] for index in permutation),
        )
    owner_axes = set(maximum.output.axes)
    dimensions = NormalizedExpContractionDimensions(
        producer_features=producer.reduction_axes,
        normalized_axes=maximum.reduction_axes,
        consumer_features=tuple(axis for axis in consumer.output.axes if axis not in owner_axes),
    )
    return NormalizedExpContractRegion(
        producer=producer,
        score_map=score_map,
        fold_state=NormalizedExpFoldState(maximum, center, exponential, normalizer),
        consumer=consumer,
        finalizer=finalizer,
        dimensions=dimensions,
    )


def derive_forward_ownership_plan(region: NormalizedExpContractRegion) -> ForwardOwnershipPlan:
    """Derive forward ownership entirely from Fold and Contract dimensions."""
    return ForwardOwnershipPlan(
        owner_axes=region.owner_axes,
        streamed_axes=region.dimensions.normalized_axes,
        producer_reduction_axes=region.dimensions.producer_features,
        consumer_feature_axes=region.dimensions.consumer_features,
        producer_index_relations=region.producer_index_relations,
        consumer_index_relations=region.consumer_index_relations,
    )


def derive_normalized_exp_contract_pullback(
    region: NormalizedExpContractRegion,
) -> NormalizedExpContractPullback:
    """Differentiate the shared region into explicit Fold/Map/Contract algebra."""
    if len(region.producer.inputs) != 2:
        raise ValueError("normalized-exp pullback currently supports a two-input producer Contract")
    if len(region.consumer.inputs) != 2:
        raise ValueError("normalized-exp pullback currently supports a two-input consumer Contract")
    consumer_operand = region.consumer.inputs[1]
    accumulation_dtype = region.consumer.accumulation_dtype
    output_cotangent = ProgramValue(
        "cotangent.normalized_exp_contract.output",
        region.finalizer.output.axes,
        accumulation_dtype,
    )
    normalized_output_value = ProgramValue(
        "normalized_exp_contract.normalized_output",
        region.consumer.output.axes,
        accumulation_dtype,
    )
    normalized_output = MapPrimitive(
        "normalize weighted Contract",
        (region.consumer.output, region.fold_state.row_sum_exp),
        normalized_output_value,
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input(region.consumer.output.name),
            scalar_input(region.fold_state.row_sum_exp.name),
        ),
    )
    post_normalization_expression = normalized_exp_post_finalizer_expression(
        region,
        normalized_output_name=normalized_output_value.name,
    )
    post_normalization_input_names = scalar_expression_inputs(post_normalization_expression)
    post_normalization_inputs = (
        normalized_output_value,
        *(
            value
            for value in region.finalizer.inputs
            if value not in (region.consumer.output, region.fold_state.row_sum_exp)
            and value.name in post_normalization_input_names
        ),
    )
    post_normalization = MapPrimitive(
        "post-normalization finalizer Map",
        post_normalization_inputs,
        region.finalizer.output,
        post_normalization_expression,
    )
    differentiable_finalizer = _without_terminal_storage_conversion(
        post_normalization_expression,
        output_dtype=region.finalizer.output.dtype,
    )
    normalized_output_cotangent_value = ProgramValue(
        "cotangent.normalized_exp_contract.normalized_output",
        normalized_output_value.axes,
        accumulation_dtype,
    )
    post_normalization_vjp_expression = scalar_expression_vjp(
        differentiable_finalizer,
        input_name=normalized_output_value.name,
        cotangent_name=output_cotangent.name,
    )
    finalizer_vjp_input_names = scalar_expression_inputs(post_normalization_vjp_expression)
    finalizer_vjp_inputs = (
        output_cotangent,
        *(value for value in post_normalization_inputs if value.name in finalizer_vjp_input_names),
    )
    post_normalization_vjp = MapPrimitive(
        "automatically differentiated post-normalization finalizer VJP",
        finalizer_vjp_inputs,
        normalized_output_cotangent_value,
        post_normalization_vjp_expression,
    )
    output_product_value = ProgramValue(
        "normalized_output_times_cotangent",
        region.finalizer.output.axes,
        accumulation_dtype,
    )
    output_product = MapPrimitive(
        "multiply normalized output by its cotangent",
        (normalized_output_value, normalized_output_cotangent_value),
        output_product_value,
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(normalized_output_value.name),
            scalar_input(normalized_output_cotangent_value.name),
        ),
    )
    output_feature_axes = region.dimensions.consumer_features
    output_dot_value = ProgramValue(
        "normalized_exp_contract.output_dot_cotangent",
        tuple(axis for axis in output_product_value.axes if axis not in output_feature_axes),
        accumulation_dtype,
    )
    output_dependent_fold = FoldPrimitive(
        "fold final output product",
        output_product_value,
        output_dot_value,
        output_feature_axes,
        FoldReducer.SUM,
        accumulation_dtype,
    )

    weight_cotangent_value = ProgramValue(
        "normalized_exp_contract.weight_cotangent",
        region.producer.output.axes,
        accumulation_dtype,
    )
    consumer_relations = tuple(
        relation
        for relation in region.consumer.index_maps_for_input(1)
        if relation.domain_axis in (*weight_cotangent_value.axes, *output_feature_axes)
    )
    consumer_weight_cotangent = ContractPrimitive(
        "consumer weight cotangent Contract",
        (normalized_output_cotangent_value, consumer_operand),
        weight_cotangent_value,
        output_feature_axes,
        accumulation_dtype,
        input_index_maps=((), consumer_relations),
        operand_precision=adjoint_contract_operand_precision(region.consumer, 0),
    )
    normalized_weight = ProgramValue(
        "normalized_exp_contract.normalized_weight",
        region.producer.output.axes,
        accumulation_dtype,
    )
    normalized_weight_map = MapPrimitive(
        "normalize exponential weight",
        (region.fold_state.exponential.output, region.fold_state.row_sum_exp),
        normalized_weight,
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input(region.fold_state.exponential.output.name),
            scalar_input(region.fold_state.row_sum_exp.name),
        ),
    )
    mapped_score_cotangent = ProgramValue(
        "cotangent.normalized_exp_contract.mapped_score",
        region.producer.output.axes,
        accumulation_dtype,
    )
    normalized_exp_pullback = MapPrimitive(
        "normalized-exp pullback",
        (normalized_weight, weight_cotangent_value, output_dot_value),
        mapped_score_cotangent,
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(normalized_weight.name),
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input(weight_cotangent_value.name),
                scalar_input(output_dot_value.name),
            ),
        ),
    )

    raw_score_cotangent = ProgramValue(
        "cotangent.normalized_exp_contract.raw_score",
        region.producer.output.axes,
        accumulation_dtype,
    )
    vjp_expression = scalar_expression_vjp(
        region.score_map.expression,
        input_name=region.producer.output.name,
        cotangent_name=mapped_score_cotangent.name,
    )
    vjp_inputs = (
        mapped_score_cotangent,
        *(
            value
            for value in region.score_map.inputs
            if value.name in scalar_expression_inputs(vjp_expression) and value != region.producer.output
        ),
    )
    if region.producer.output.name in scalar_expression_inputs(vjp_expression):
        vjp_inputs = (*vjp_inputs, region.producer.output)
    score_map_vjp = MapPrimitive(
        "automatically differentiated score Map VJP",
        vjp_inputs,
        raw_score_cotangent,
        vjp_expression,
        predicate=region.score_map.predicate,
        rejected_value=0.0 if region.score_map.predicate is not None else None,
    )

    terminal_gradients = (
        _terminal_contract_gradient(
            contract=region.producer,
            output_cotangent=raw_score_cotangent,
            input_index=0,
            name="terminal producer-left gradient Contract",
            accumulation_dtype=accumulation_dtype,
        ),
        _terminal_contract_gradient(
            contract=region.producer,
            output_cotangent=raw_score_cotangent,
            input_index=1,
            name="terminal producer-right gradient Contract",
            accumulation_dtype=accumulation_dtype,
        ),
        _terminal_contract_gradient(
            contract=region.consumer,
            output_cotangent=normalized_output_cotangent_value,
            input_index=1,
            name="terminal consumer-operand gradient Contract",
            accumulation_dtype=accumulation_dtype,
            other_operand=normalized_weight,
        ),
    )
    return NormalizedExpContractPullback(
        region=region,
        output_cotangent=output_cotangent,
        normalized_output=normalized_output,
        post_normalization=post_normalization,
        post_normalization_vjp=post_normalization_vjp,
        output_product=output_product,
        output_dependent_fold=output_dependent_fold,
        consumer_weight_cotangent=consumer_weight_cotangent,
        normalized_weight=normalized_weight,
        normalized_weight_map=normalized_weight_map,
        normalized_exp_pullback=normalized_exp_pullback,
        score_map_vjp=score_map_vjp,
        terminal_gradients=terminal_gradients,
    )


def derive_reverse_ownership_plan(
    pullback: NormalizedExpContractPullback,
    *,
    traversal: ContractDomainTraversal,
) -> ReverseOwnershipPlan:
    """Derive both reverse owner traversals from operand-axis relations."""
    region = pullback.region
    left, right = region.producer.inputs
    reduction_axes = set(region.dimensions.producer_features)
    left_owner_axes = tuple(axis for axis in left.axes if axis not in reduction_axes)
    right_owner_axes = tuple(axis for axis in right.axes if axis not in reduction_axes)
    left_axes = set(left.axes)
    right_axes = set(right.axes)
    right_mapped_domains = {relation.domain_axis for relation in region.producer.index_maps_for_input(1)}
    left_streamed_axes = tuple(axis for axis in region.producer.output.axes if axis not in left_axes)
    right_streamed_axes = tuple(
        axis for axis in region.producer.output.axes if axis not in right_axes or axis in right_mapped_domains
    )
    return ReverseOwnershipPlan(
        left_owner_axes=left_owner_axes,
        right_owner_axes=right_owner_axes,
        left_streamed_axes=left_streamed_axes,
        right_streamed_axes=right_streamed_axes,
        traversal=traversal,
        producer_index_relations=region.producer_index_relations,
        consumer_index_relations=region.consumer_index_relations,
    )


def normalized_exp_contract_pullback_tensor_program(
    pullback: NormalizedExpContractPullback,
) -> TensorProgram:
    """Close a normalized-exponential pullback into generic tensor algebra.

    Relation-mapped operand gradients are expressed as a generic gather into
    an explicit relation-group axis followed by a sum Fold.  The resulting
    program contains no attention mode or physical schedule choices.
    """
    region = pullback.region
    operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive | RelationPrimitive] = [
        region.producer,
        region.score_map,
    ]
    operations.extend(
        (
            region.fold_state.maximum,
            region.fold_state.center,
            region.fold_state.exponential,
            region.fold_state.normalizer,
            region.consumer,
            region.finalizer,
            pullback.normalized_output,
            pullback.post_normalization_vjp,
            pullback.output_product,
            pullback.output_dependent_fold,
            pullback.consumer_weight_cotangent,
            pullback.normalized_weight_map,
            pullback.normalized_exp_pullback,
            pullback.score_map_vjp,
        )
    )
    outputs: list[ProgramValue] = []
    next_axis_id = 1 + max(
        axis.id
        for value in (
            *region.producer.inputs,
            region.producer.output,
            region.consumer.inputs[1],
            region.finalizer.output,
        )
        for axis in value.axes
    )
    for gradient_index, gradient in enumerate(pullback.terminal_gradients):
        operations.append(gradient.contract)
        if not gradient.destination_index_relations:
            if gradient.contract.output != gradient.destination:
                raise ValueError("an unmapped terminal gradient Contract must directly produce its destination")
            outputs.append(gradient.destination)
            continue

        relations = gradient.destination_index_relations
        relation_by_expanded_axis = {relation.domain_axis: relation for relation in relations}
        if len(relation_by_expanded_axis) != len(relations):
            raise ValueError("terminal gradient relations must map distinct expanded axes")
        group_axes: list[TensorAxis] = []
        for relation_index, relation in enumerate(relations):
            if relation.offset != 0 or relation.modulus is not None:
                raise ValueError("terminal gradient relation inversion requires zero-offset non-modular mappings")
            if relation.domain_axis.extent != relation.operand_axis.extent * relation.divisor:
                raise ValueError("terminal gradient relation must exactly cover its destination axis")
            group_axes.append(
                TensorAxis(
                    next_axis_id + relation_index,
                    relation.divisor,
                    f"terminal_gradient_{gradient_index}_relation_group_{relation_index}",
                )
            )
        next_axis_id += len(group_axes)
        gather_axes = (*gradient.destination.axes, *group_axes)
        input_indices: list[IndexExpression] = []
        for expanded_axis in gradient.contract.output.axes:
            relation = relation_by_expanded_axis.get(expanded_axis)
            if relation is None:
                try:
                    input_indices.append(
                        index_axis(next(axis for axis in gradient.destination.axes if axis == expanded_axis))
                    )
                except StopIteration as error:
                    raise ValueError(
                        "terminal gradient Contract axis is absent from its destination relation"
                    ) from error
                continue
            relation_index = relations.index(relation)
            input_indices.append(
                index_binary(
                    IndexExpressionKind.ADD,
                    index_binary(
                        IndexExpressionKind.MULTIPLY,
                        index_axis(relation.operand_axis),
                        index_constant(relation.divisor),
                    ),
                    index_axis(group_axes[relation_index]),
                )
            )
        gathered = ProgramValue(
            f"terminal_gradient_{gradient_index}.relation_grouped",
            gather_axes,
            gradient.contract.output.dtype,
        )
        operations.append(
            RelationPrimitive(
                f"terminal gradient {gradient_index} destination relation",
                (gradient.contract.output,),
                (gathered,),
                gather_axes,
                (
                    OperandIndexing(
                        gradient.contract.output,
                        gather_axes,
                        tuple(input_indices),
                    ),
                ),
            )
        )
        operations.append(
            FoldPrimitive(
                f"terminal gradient {gradient_index} relation Fold",
                gathered,
                gradient.destination,
                tuple(group_axes),
                FoldReducer.SUM,
                gradient.contract.accumulation_dtype,
            )
        )
        outputs.append(gradient.destination)

    available_names: set[str] = set()
    inputs: list[ProgramValue] = []
    for operation in operations:
        operation_inputs = (operation.input,) if isinstance(operation, FoldPrimitive) else operation.inputs
        for value in operation_inputs:
            if value.name not in available_names:
                inputs.append(value)
                available_names.add(value.name)
        operation_outputs = (
            (operation.output,)
            if isinstance(operation, (ContractPrimitive, MapPrimitive, FoldPrimitive))
            else operation.outputs
        )
        available_names.update(value.name for value in operation_outputs)
    return TensorProgram(
        inputs=tuple(inputs),
        operations=tuple(operations),
        outputs=(*outputs, pullback.output_dependent_fold.output),
    )


def normalized_exp_finalizer_scale(region: NormalizedExpContractRegion) -> float:
    """Return the coefficient when the post-normalization Map is linear."""
    normalized_output_name = "normalized_exp_contract.normalized_output"
    expression = normalized_exp_post_finalizer_expression(region, normalized_output_name=normalized_output_name)
    expression = _without_terminal_storage_conversion(expression, output_dtype=region.finalizer.output.dtype)
    scale = 1.0
    if expression.kind is ScalarExpressionKind.MULTIPLY:
        left, right = expression.operands
        if left.constant is not None:
            scale = float(left.constant)
            expression = right
        elif right.constant is not None:
            scale = float(right.constant)
            expression = left
        else:
            raise ValueError("normalized-exp finalizer scale must be a scalar constant")
    if expression.kind is not ScalarExpressionKind.INPUT or expression.input_name != normalized_output_name:
        raise ValueError("normalized-exp finalizer is not a linear Map of the normalized Contract result")
    if not math.isfinite(scale):
        raise ValueError("normalized-exp finalizer scale must be finite")
    return scale


def normalized_exp_post_finalizer_expression(
    region: NormalizedExpContractRegion,
    *,
    normalized_output_name: str,
) -> ScalarExpression:
    """Factor a finalizer ``h(accumulator / normalizer)`` into the scalar AST ``h(U)``.

    Direct uses of the unnormalized accumulator or Fold state are rejected.
    Such a general ``g(accumulator, normalizer)`` is not invariant to the
    maximum used to stabilize the exponential and therefore needs the explicit
    maximum-Fold VJP rather than the normalized-exp pullback used here.
    """

    quotient_count = 0

    def rewrite(expression: ScalarExpression) -> ScalarExpression:
        nonlocal quotient_count
        if expression.kind is ScalarExpressionKind.DIVIDE:
            numerator, denominator = expression.operands
            if (
                numerator.kind is ScalarExpressionKind.INPUT
                and numerator.input_name == region.consumer.output.name
                and denominator.kind is ScalarExpressionKind.INPUT
                and denominator.input_name == region.fold_state.row_sum_exp.name
            ):
                quotient_count += 1
                return scalar_input(normalized_output_name)
        if expression.kind is ScalarExpressionKind.INPUT and expression.input_name in {
            region.consumer.output.name,
            region.fold_state.row_sum_exp.name,
        }:
            raise ValueError(
                "normalized-exp pullback requires finalizer uses of the weighted Contract and Fold state "
                "to factor through their quotient"
            )
        return ScalarExpression(
            kind=expression.kind,
            operands=tuple(rewrite(operand) for operand in expression.operands),
            input_name=expression.input_name,
            constant=expression.constant,
            result_dtype=expression.result_dtype,
            conversion_semantics=expression.conversion_semantics,
        )

    rewritten = rewrite(region.finalizer.expression)
    rewritten_inputs = scalar_expression_inputs(rewritten)
    auxiliary_inputs = {
        value.name
        for value in region.finalizer.inputs
        if value not in (region.consumer.output, region.fold_state.row_sum_exp)
    }
    if quotient_count == 0 or normalized_output_name not in rewritten_inputs:
        raise ValueError("normalized-exp pullback requires a post-normalization scalar Map of accumulator / normalizer")
    unavailable_inputs = rewritten_inputs - {normalized_output_name} - auxiliary_inputs
    if unavailable_inputs:
        raise ValueError(f"post-normalization scalar Map references unavailable inputs {sorted(unavailable_inputs)}")
    return rewritten


def _without_terminal_storage_conversion(
    expression: ScalarExpression,
    *,
    output_dtype: DType,
) -> ScalarExpression:
    if expression.kind is not ScalarExpressionKind.CONVERT:
        return expression
    expected_semantics = (
        ScalarConversionSemantics.ROUND_NEAREST_EVEN if output_dtype is DType.BF16 else ScalarConversionSemantics.EXACT
    )
    if expression.result_dtype is not output_dtype or expression.conversion_semantics is not expected_semantics:
        raise ValueError("normalized-exp pullback does not support this terminal storage conversion")
    return expression.operands[0]


def normalized_exp_contract_pullback_program(
    forward: NormalizedExpContractProgram[ScheduleT],
    *,
    schedule: ReverseScheduleT,
    traversal: ContractDomainTraversal,
) -> NormalizedExpContractPullbackProgram[ScheduleT, ReverseScheduleT]:
    """Differentiate and schedule one shared normalized-exp Contract region."""
    pullback = derive_normalized_exp_contract_pullback(forward.region)
    return NormalizedExpContractPullbackProgram(
        forward=forward,
        pullback=pullback,
        ownership=derive_reverse_ownership_plan(pullback, traversal=traversal),
        right_owner_fold_order=ReverseContractFoldOrder.STREAMED_AXES_MAJOR_MAPPED_RELATION_MINOR_TREE,
        schedule=schedule,
        materialized_values=(
            *(gradient.destination for gradient in pullback.terminal_gradients),
            pullback.output_dependent_fold.output,
        ),
    )


def verify_normalized_exp_score_map_vjp(
    program: NormalizedExpContractPullbackProgram[ScheduleT, ReverseScheduleT],
) -> None:
    """Verify that reverse scalar algebra is the VJP of the forward score Map."""
    region = program.forward.region
    expected_expression = scalar_expression_vjp(
        region.score_map.expression,
        input_name=region.producer.output.name,
        cotangent_name=program.pullback.normalized_exp_pullback.output.name,
    )
    score_map_vjp = program.pullback.score_map_vjp
    if score_map_vjp.expression != expected_expression:
        raise ValueError("score Map VJP is not the derivative of the recovered forward scalar AST")
    expected_inputs = scalar_expression_inputs(expected_expression)
    actual_inputs = tuple(value.name for value in score_map_vjp.inputs)
    if len(actual_inputs) != len(expected_inputs) or set(actual_inputs) != expected_inputs:
        raise ValueError(
            "score Map VJP inputs do not match its derived scalar expression: "
            f"expected {tuple(sorted(expected_inputs))}, found {actual_inputs}"
        )
    if score_map_vjp.output.axes != region.score_map.output.axes:
        raise ValueError("score Map VJP must preserve the forward score Map output domain")


def verify_normalized_exp_finalizer_vjp(
    program: NormalizedExpContractPullbackProgram[ScheduleT, ReverseScheduleT],
) -> None:
    """Verify that reverse finalization is the VJP of the source post-normalization AST."""
    pullback = program.pullback
    region = program.forward.region
    normalized_output = pullback.normalized_output.output
    expected_finalizer = normalized_exp_post_finalizer_expression(
        region,
        normalized_output_name=normalized_output.name,
    )
    if pullback.post_normalization.expression != expected_finalizer:
        raise ValueError("post-normalization Map is not factored from the recovered forward finalizer AST")
    expected_vjp = scalar_expression_vjp(
        _without_terminal_storage_conversion(expected_finalizer, output_dtype=region.finalizer.output.dtype),
        input_name=normalized_output.name,
        cotangent_name=pullback.output_cotangent.name,
    )
    if pullback.post_normalization_vjp.expression != expected_vjp:
        raise ValueError("finalizer Map VJP is not the derivative of the recovered post-normalization scalar AST")
    expected_inputs = scalar_expression_inputs(expected_vjp)
    actual_inputs = tuple(value.name for value in pullback.post_normalization_vjp.inputs)
    if len(actual_inputs) != len(expected_inputs) or set(actual_inputs) != expected_inputs:
        raise ValueError(
            "finalizer Map VJP inputs do not match its derived scalar expression: "
            f"expected {tuple(sorted(expected_inputs))}, found {actual_inputs}"
        )
    if pullback.post_normalization_vjp.output.axes != region.finalizer.output.axes:
        raise ValueError("finalizer Map VJP must preserve the forward finalizer output domain")


def is_identity_post_normalization_pullback(
    program: NormalizedExpContractPullbackProgram[ScheduleT, ReverseScheduleT],
) -> bool:
    """Return whether the finalizer VJP forwards the output cotangent unchanged."""
    expression = program.pullback.post_normalization_vjp.expression
    return (
        expression.kind is ScalarExpressionKind.INPUT and expression.input_name == program.pullback.output_cotangent.name
    )


def relation_group_size(relations: tuple[AxisIndexMap, ...]) -> int:
    """Return the uniform reuse factor encoded by one floor-division relation."""
    if len(relations) != 1:
        raise ValueError("SM90 packed operands require one axis index relation")
    relation = relations[0]
    if relation.offset != 0 or relation.modulus is not None:
        raise ValueError("SM90 packed operands require a zero-offset floor-division index relation")
    if relation.domain_axis.extent != relation.operand_axis.extent * relation.divisor:
        raise ValueError("floor-division index relation must exactly cover its operand axis")
    return relation.divisor


def _terminal_contract_gradient(
    *,
    contract: ContractPrimitive,
    output_cotangent: ProgramValue,
    input_index: int,
    name: str,
    accumulation_dtype: DType,
    other_operand: ProgramValue | None = None,
) -> TerminalGradientContract:
    """Differentiate one binary Contract operand over its relation-expanded domain."""
    if len(contract.inputs) != 2:
        raise ValueError("terminal Contract gradients currently require two inputs")
    operand = contract.inputs[input_index]
    relations = contract.index_maps_for_input(input_index)
    expanded_axes = _expanded_destination_axes(operand.axes, relations)
    destination = ProgramValue(f"cotangent.{operand.name}", operand.axes, accumulation_dtype)
    expanded = destination
    if relations:
        expanded = ProgramValue(f"{destination.name}.relation_expanded", expanded_axes, accumulation_dtype)
    other_index = 1 - input_index
    other = contract.inputs[other_index] if other_operand is None else other_operand
    if other.axes != contract.inputs[other_index].axes:
        raise ValueError("replacement Contract-gradient operand must preserve the forward operand axes")
    iteration_axes = (*contract.output.axes, *contract.reduction_axes)
    reduction_axes = tuple(axis for axis in iteration_axes if axis not in expanded_axes)
    gradient_contract = ContractPrimitive(
        name,
        (output_cotangent, other),
        expanded,
        reduction_axes,
        accumulation_dtype,
        input_index_maps=((), contract.index_maps_for_input(other_index)),
        operand_precision=adjoint_contract_operand_precision(contract, input_index),
    )
    return TerminalGradientContract(operand, gradient_contract, destination, relations)


def _expanded_destination_axes(
    destination_axes: tuple[TensorAxis, ...],
    relations: tuple[AxisIndexMap, ...],
) -> tuple[TensorAxis, ...]:
    replacement = {relation.operand_axis: relation.domain_axis for relation in relations}
    return tuple(replacement.get(axis, axis) for axis in destination_axes)


def _expect_expression(
    operation: MapPrimitive,
    kind: ScalarExpressionKind,
    input_names: tuple[str, ...],
) -> None:
    expression = operation.expression
    if expression.kind is not kind:
        raise ValueError(f"Map {operation.name!r} must use {kind.value}")
    actual_names = tuple(operand.input_name for operand in expression.operands)
    if actual_names != input_names:
        raise ValueError(f"Map {operation.name!r} expected scalar inputs {input_names}, found {actual_names}")
