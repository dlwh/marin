# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a physical normalized-exponential Contract region from typed Shuttle IR."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from shuttle.ir import DType
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.cute_pipeline_artifact import CuteAffineMap
from tile_lifetime.normalized_exp_contract_region import (
    AffineIndexExpression,
    AffineIndexPredicate,
    IndexComparisonKind,
    NormalizedExpContractionDimensions,
    NormalizedExpContractProgram,
    NormalizedExpContractRegion,
    NormalizedExpFoldState,
    derive_forward_ownership_plan,
)
from tile_lifetime.physical_artifact_compiler import (
    GpuPhysicalArtifactRequest,
    ShuttleNormalizedExpContractPullbackRecord,
    ShuttleNormalizedExpContractRegionRecord,
    ShuttleOperationRecord,
    ShuttleTypeRecord,
    ShuttleValueRecord,
)
from tile_lifetime.physical_event_cover import affine_expression_from_record
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_expression_inputs,
    scalar_input,
    scalar_select,
    scalar_unary,
)

_DOMAIN_PREDICATE_INPUT = "contract.domain_predicate"
_SCORE_INPUT = "contract.producer"
_CONSUMER_ACCUMULATOR_INPUT = "contract.consumer_accumulator"
_NORMALIZER_INPUT = "contract.normalizer"
_MAPPED_SCORE_COTANGENT_INPUT = "cotangent.normalized_exp_contract.mapped_score"
_FINALIZER_OUTPUT_INPUT = "contract.finalizer_output"
_FINALIZER_COTANGENT_INPUT = "cotangent.contract.finalizer_output"

ScheduleT = TypeVar("ScheduleT")


@dataclass(frozen=True)
class SourceBoundBuffer:
    """One exact public or temporary buffer from the MLIR materialization plan."""

    ordinal: int
    type: ShuttleTypeRecord
    live_in: bool
    live_out: bool
    storage: str
    source_argument: int | None
    producer: int | None
    consumers: tuple[int, ...]


@dataclass(frozen=True)
class SourceBoundPhase:
    """One typed phase-local iteration domain in a composed native task."""

    source_operation: int
    kind: str
    domain_shape: tuple[int, ...]
    reduction_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class SourceBoundPhaseEdge:
    """One exact internal SSA use and its affine consumer access."""

    source_phase: int
    source_result: int
    target_phase: int
    target_operand: int
    access_map: CuteAffineMap


@dataclass(frozen=True)
class SourceBoundResultBinding:
    """Bind one externally live phase result to a materialized buffer."""

    phase: int
    source_operation: int
    result: int
    buffer: int


@dataclass(frozen=True)
class SourceBoundTask:
    """One certified generic composition task from the materialization plan."""

    ordinal: int
    source_operations: tuple[int, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    dependencies: tuple[int, ...]
    domain_shape: tuple[int, ...]
    reduction_dimensions: tuple[int, ...]
    owner_dimensions: tuple[int, ...]
    vector_dimensions: tuple[int, ...]
    stream_dimensions: tuple[int, ...]
    state_inputs: tuple[int, ...]
    state_outputs: tuple[int, ...]
    producer_contraction_dimensions: tuple[int, ...]
    consumer_contraction_dimensions: tuple[int, ...]
    phases: tuple[SourceBoundPhase, ...] = ()
    phase_edges: tuple[SourceBoundPhaseEdge, ...] = ()
    result_bindings: tuple[SourceBoundResultBinding, ...] = ()
    source_expansions: tuple[object | None, ...] = ()


@dataclass(frozen=True)
class SourceBoundScheduleTask:
    """One verified schedule record bound to its generic composition task."""

    ordinal: int
    source_task: int
    kind: str
    domain_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    serial_tiles: int
    workgroup_threads: int
    subgroup_size: int
    scratch_bytes: int
    input_buffers: tuple[int, ...]
    output_buffers: tuple[int, ...]
    dependencies: tuple[int, ...]
    reduction_axis: int | None
    owner_dimensions: tuple[int, ...]
    vector_dimensions: tuple[int, ...]
    stream_dimensions: tuple[int, ...]
    state_inputs: tuple[int, ...]
    state_outputs: tuple[int, ...]
    producer_contraction_dimensions: tuple[int, ...]
    consumer_contraction_dimensions: tuple[int, ...]
    online_mma: bool


@dataclass(frozen=True)
class NormalizedExpContractGeometry:
    """Contraction dimensions derived from affine operand relations."""

    batch_shape: tuple[int, ...]
    shared_relation_extent: int
    relation_group_shape: tuple[int, ...]
    owner_extent: int
    streamed_extent: int
    producer_feature_extent: int
    consumer_feature_extent: int
    batch_dimensions: tuple[int, ...]
    shared_relation_dimension: int
    relation_group_dimensions: tuple[int, ...]
    owner_dimension: int
    streamed_dimension: int
    producer_feature_dimension: int
    normalized_dimension: int
    consumer_feature_dimension: int

    @property
    def batch_size(self) -> int:
        return math.prod(self.batch_shape)

    @property
    def relation_group_size(self) -> int:
        return math.prod(self.relation_group_shape)

    @property
    def expanded_relation_extent(self) -> int:
        return self.shared_relation_extent * self.relation_group_size


@dataclass(frozen=True)
class PhysicalContractDomainPredicate:
    """One source relation restricting the streamed Contract domain."""

    relation: ShuttleOperationRecord
    source_operations: tuple[ShuttleOperationRecord, ...]
    predicate: AffineIndexPredicate


class PhysicalNormalizedExpContractNotApplicable(ValueError):
    """The typed region does not contain this physical task family."""


@dataclass(frozen=True)
class PhysicalNormalizedExpContractRegion:
    """Generic algebra, index relations, scalar program, and source boundary."""

    producer: ShuttleOperationRecord
    score_program: tuple[ShuttleOperationRecord, ...]
    maximum: ShuttleOperationRecord
    exponential: ShuttleOperationRecord
    normalizer: ShuttleOperationRecord
    consumer: ShuttleOperationRecord
    finalizer: ShuttleOperationRecord
    producer_left_argument: int
    producer_right_argument: int
    consumer_argument: int
    output_result: int | None
    public_output_dimensions: tuple[int, ...]
    producer_to_consumer_dimensions: tuple[tuple[int, int], ...]
    geometry: NormalizedExpContractGeometry
    score_expression: ScalarExpression
    finalizer_expression: ScalarExpression
    domain_predicate: PhysicalContractDomainPredicate | None
    buffers: tuple[SourceBoundBuffer, ...]

    @property
    def public_inputs(self) -> tuple[SourceBoundBuffer, ...]:
        return tuple(buffer for buffer in self.buffers if buffer.live_in)

    @property
    def public_outputs(self) -> tuple[SourceBoundBuffer, ...]:
        return tuple(buffer for buffer in self.buffers if buffer.live_out)


@dataclass(frozen=True)
class _NormalizedExpForwardBinding:
    producer: ShuttleOperationRecord
    maximum: ShuttleOperationRecord
    exponential: ShuttleOperationRecord
    normalizer: ShuttleOperationRecord
    consumer: ShuttleOperationRecord
    normalized_operand: int
    score_value: int
    normalizer_value: int


@dataclass(frozen=True)
class _ProducerOrientation:
    owner_operand: int
    stream_operand: int
    common_dimensions: tuple[int, ...]
    relation_group_dimensions: tuple[int, ...]
    owner_dimension: int
    streamed_dimension: int


@dataclass(frozen=True)
class PhysicalNormalizedExpContractPullback:
    """A source-certified pullback around the shared forward region."""

    forward: PhysicalNormalizedExpContractRegion
    tasks: tuple[SourceBoundTask, ...]
    output_cotangent_argument: int
    forward_output_buffer: int
    query_cotangent_buffer: int
    key_cotangent_buffer: int
    value_cotangent_buffer: int
    maximum_state_buffer: int
    normalizer_state_buffer: int
    output_dependent_state_buffer: int
    score_map_vjp_expression: ScalarExpression
    normalized_exp_pullback_expression: ScalarExpression


@dataclass(frozen=True)
class PhysicalSavedStateReverseBoundary:
    """A certified six-input/three-output reverse Contract/Fold boundary."""

    inputs: tuple[SourceBoundBuffer, ...]
    outputs: tuple[SourceBoundBuffer, ...]
    owner_task: SourceBoundTask
    streamed_task: SourceBoundTask
    producer_owner_input: SourceBoundBuffer
    producer_stream_input: SourceBoundBuffer
    consumer_input: SourceBoundBuffer
    finalizer_output_input: SourceBoundBuffer
    finalizer_cotangent_input: SourceBoundBuffer
    saved_state_input: SourceBoundBuffer
    producer_owner_cotangent_output: SourceBoundBuffer
    producer_stream_cotangent_output: SourceBoundBuffer
    consumer_cotangent_output: SourceBoundBuffer
    output_dependent_state_output: SourceBoundBuffer
    producer: ShuttleOperationRecord
    producer_owner_operand_index: int
    consumer_weight_cotangent: ShuttleOperationRecord
    consumer_output_cotangent_operand_index: int
    output_dependent_fold: ShuttleOperationRecord
    score_expression: ScalarExpression
    score_map_vjp_expression: ScalarExpression
    normalized_exp_pullback_expression: ScalarExpression
    output_product_expression: ScalarExpression
    domain_predicate: PhysicalContractDomainPredicate | None


@dataclass(frozen=True)
class PhysicalNormalizedExpForwardMatch:
    """A typed forward region accepted by the normalized-exp task family."""

    region: PhysicalNormalizedExpContractRegion


@dataclass(frozen=True)
class PhysicalNormalizedExpPullbackMatch:
    """A typed value-and-pullback region accepted by the task family."""

    pullback: PhysicalNormalizedExpContractPullback


type PhysicalNormalizedExpContractMatch = PhysicalNormalizedExpForwardMatch | PhysicalNormalizedExpPullbackMatch


@dataclass(frozen=True)
class _TerminalContractBinding:
    contract: ShuttleOperationRecord
    direct_argument: int
    algebra_operand: int


@dataclass(frozen=True)
class _OutputDependentFoldBinding:
    operation: ShuttleOperationRecord
    arguments: frozenset[int]


@dataclass(frozen=True)
class _SavedStateReverseAlgebra:
    producer_owner_input: SourceBoundBuffer
    producer_stream_input: SourceBoundBuffer
    consumer_input: SourceBoundBuffer
    finalizer_output_input: SourceBoundBuffer
    finalizer_cotangent_input: SourceBoundBuffer
    saved_state_input: SourceBoundBuffer
    producer_owner_cotangent_output: SourceBoundBuffer
    producer_stream_cotangent_output: SourceBoundBuffer
    consumer_cotangent_output: SourceBoundBuffer
    output_dependent_state_output: SourceBoundBuffer
    producer: ShuttleOperationRecord
    producer_owner_operand_index: int
    consumer_weight_cotangent: ShuttleOperationRecord
    consumer_output_cotangent_operand_index: int
    output_dependent_fold: ShuttleOperationRecord
    score_expression: ScalarExpression
    score_map_vjp_expression: ScalarExpression
    normalized_exp_pullback_expression: ScalarExpression
    output_product_expression: ScalarExpression
    domain_predicate: PhysicalContractDomainPredicate | None


def normalized_exp_contract_program(
    region: PhysicalNormalizedExpContractRegion,
    *,
    schedule: ScheduleT,
) -> NormalizedExpContractProgram[ScheduleT]:
    """Translate typed physical algebra into a backend-neutral Contract/Fold program.

    Shared and relation-group axes remain explicit in the algebra. Both scalar
    Maps remain the exact source ASTs; the conversion does not recover or name
    a workload.
    """
    geometry = region.geometry
    batch = tuple(TensorAxis(index, extent, f"batch_{index}") for index, extent in enumerate(geometry.batch_shape))
    shared_relation = TensorAxis(len(batch), geometry.shared_relation_extent, "shared_relation")
    relation_groups = tuple(
        TensorAxis(len(batch) + index + 1, extent, f"relation_group_{index}")
        for index, extent in enumerate(geometry.relation_group_shape)
    )
    owner = TensorAxis(len(batch) + len(relation_groups) + 1, geometry.owner_extent, "owner")
    streamed = TensorAxis(len(batch) + len(relation_groups) + 2, geometry.streamed_extent, "streamed")
    producer_feature = TensorAxis(
        len(batch) + len(relation_groups) + 3,
        geometry.producer_feature_extent,
        "producer_feature",
    )
    consumer_feature = TensorAxis(
        len(batch) + len(relation_groups) + 4,
        geometry.consumer_feature_extent,
        "consumer_feature",
    )

    producer_owner_input = ProgramValue(
        "contract.producer_owner_input",
        (*batch, shared_relation, *relation_groups, owner, producer_feature),
        DType.BF16,
    )
    producer_stream_input = ProgramValue(
        "contract.producer_stream_input",
        (*batch, shared_relation, streamed, producer_feature),
        DType.BF16,
    )
    consumer_input = ProgramValue(
        "contract.consumer_input",
        (*batch, shared_relation, streamed, consumer_feature),
        DType.BF16,
    )
    raw_score = ProgramValue(
        "contract.producer",
        (*batch, shared_relation, *relation_groups, owner, streamed),
        DType.FP32,
    )
    mapped_score = ProgramValue("contract.mapped_score", raw_score.axes, DType.FP32)
    row_maximum = ProgramValue(
        "contract.maximum",
        (*batch, shared_relation, *relation_groups, owner),
        DType.FP32,
    )
    centered_score = ProgramValue("contract.centered_score", raw_score.axes, DType.FP32)
    exponential = ProgramValue("contract.exponential", raw_score.axes, DType.FP32)
    normalizer = ProgramValue("contract.normalizer", row_maximum.axes, DType.FP32)
    accumulator = ProgramValue(
        "contract.consumer_accumulator",
        (*batch, shared_relation, *relation_groups, owner, consumer_feature),
        DType.FP32,
    )
    output = ProgramValue("contract.finalized_output", accumulator.axes, DType.BF16)
    producer = ContractPrimitive(
        "producer Contract",
        (producer_owner_input, producer_stream_input),
        raw_score,
        (producer_feature,),
        DType.FP32,
    )
    score_map = MapPrimitive(
        "score scalar Map",
        (raw_score,),
        mapped_score,
        _rename_scalar_expression_inputs(region.score_expression, {"contract.producer": raw_score.name}),
    )
    maximum = FoldPrimitive(
        "normalized-exp maximum Fold",
        mapped_score,
        row_maximum,
        (streamed,),
        FoldReducer.MAXIMUM,
        DType.FP32,
    )
    center = MapPrimitive(
        "center score Map",
        (mapped_score, row_maximum),
        centered_score,
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_input(mapped_score.name),
            scalar_input(row_maximum.name),
        ),
    )
    exponential_map = MapPrimitive(
        "normalized exponential Map",
        (centered_score,),
        exponential,
        scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered_score.name)),
    )
    normalizer_fold = FoldPrimitive(
        "normalized-exp sum Fold",
        exponential,
        normalizer,
        (streamed,),
        FoldReducer.SUM,
        DType.FP32,
    )
    consumer = ContractPrimitive(
        "weighted consumer Contract",
        (exponential, consumer_input),
        accumulator,
        (streamed,),
        DType.FP32,
    )
    finalizer = MapPrimitive(
        "finalizer scalar Map",
        (accumulator, normalizer),
        output,
        _rename_scalar_expression_inputs(
            _compose_finalizer_with_normalized_consumer(region.finalizer_expression),
            {
                "contract.consumer_accumulator": accumulator.name,
                "contract.normalizer": normalizer.name,
            },
        ),
    )
    algebra = NormalizedExpContractRegion(
        producer=producer,
        score_map=score_map,
        fold_state=NormalizedExpFoldState(maximum, center, exponential_map, normalizer_fold),
        consumer=consumer,
        finalizer=finalizer,
        dimensions=NormalizedExpContractionDimensions(
            producer_features=(producer_feature,),
            normalized_axes=(streamed,),
            consumer_features=(consumer_feature,),
            shared_relation_axes=(shared_relation,),
            relation_group_axes=relation_groups,
        ),
        domain_predicate=(
            None
            if region.domain_predicate is None
            else _remap_affine_index_predicate(
                region.domain_predicate.predicate,
                {
                    **{
                        source_dimension: axis.id
                        for source_dimension, axis in zip(geometry.batch_dimensions, batch, strict=True)
                    },
                    geometry.shared_relation_dimension: shared_relation.id,
                    **{
                        source_dimension: axis.id
                        for source_dimension, axis in zip(
                            geometry.relation_group_dimensions, relation_groups, strict=True
                        )
                    },
                    geometry.owner_dimension: owner.id,
                    geometry.streamed_dimension: streamed.id,
                },
            )
        ),
    )
    source = TensorProgram(
        inputs=(producer_owner_input, producer_stream_input, consumer_input),
        operations=(producer, score_map, maximum, center, exponential_map, normalizer_fold, consumer, finalizer),
        outputs=(output,),
        numerical_policy=NumericalPolicy.FAST,
    )
    return NormalizedExpContractProgram(
        source=source,
        region=algebra,
        ownership=derive_forward_ownership_plan(algebra),
        schedule=schedule,
        materialized_values=(output,),
    )


def _rename_scalar_expression_inputs(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return scalar_input(names[expression.input_name])
        except KeyError as error:
            raise ValueError(f"scalar AST references unavailable physical input {expression.input_name!r}") from error
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_rename_scalar_expression_inputs(operand, names) for operand in expression.operands),
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def _compose_finalizer_with_normalized_consumer(expression: ScalarExpression) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        if expression.input_name == _CONSUMER_ACCUMULATOR_INPUT:
            return scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(_CONSUMER_ACCUMULATOR_INPUT),
                scalar_input(_NORMALIZER_INPUT),
            )
        return expression
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_compose_finalizer_with_normalized_consumer(operand) for operand in expression.operands),
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def source_bound_buffers(request: GpuPhysicalArtifactRequest) -> tuple[SourceBoundBuffer, ...]:
    """Decode exact source-bound buffer ordinals from a materialization plan."""
    buffers: list[SourceBoundBuffer] = []
    for operation in request.materialization.body().operations:
        if operation.name != "shuttle.materialization_buffer":
            continue
        attributes = operation.attributes
        tensor_type = attributes.get("tensor_type")
        if not isinstance(tensor_type, dict):
            raise ValueError("materialization buffer has no typed tensor_type")
        storage = _mlir_attribute(attributes.get("storage"), "materialization buffer storage")
        ordinal = _integer(attributes.get("ordinal"), "materialization buffer ordinal")
        producer_value = attributes.get("producer")
        producer = None if producer_value is None else _integer(producer_value, "materialization buffer producer")
        source_argument_value = attributes.get("source_argument")
        source_argument = (
            None
            if source_argument_value is None
            else _integer(source_argument_value, "materialization buffer source argument")
        )
        buffers.append(
            SourceBoundBuffer(
                ordinal=ordinal,
                type=_type_record(tensor_type),
                live_in=_boolean(attributes.get("live_in"), "materialization buffer live_in"),
                live_out=_boolean(attributes.get("live_out"), "materialization buffer live_out"),
                storage=storage,
                source_argument=source_argument,
                producer=producer,
                consumers=_integer_array(attributes.get("consumers"), "materialization buffer consumers"),
            )
        )
    buffers.sort(key=lambda buffer: buffer.ordinal)
    if [buffer.ordinal for buffer in buffers] != list(range(len(buffers))):
        raise ValueError("materialization buffers must have dense source-bound ordinals")
    return tuple(buffers)


def normalized_exp_sufficient_state_cut_types(
    region: PhysicalNormalizedExpContractRegion,
) -> tuple[ShuttleTypeRecord, ShuttleTypeRecord, ShuttleTypeRecord]:
    """Return the exact score, maximum, and normalizer types at the online cut.

    These are source tensor types, so their axis order follows the Fold's
    affine relations rather than the backend's canonical owner-axis order.
    """
    score_values = {
        result.id: result.type for operation in (*region.score_program, region.producer) for result in operation.results
    }
    if not region.maximum.operands:
        raise ValueError("normalized-exponential maximum Fold has no score operand")
    try:
        score_type = score_values[region.maximum.operands[0]]
    except KeyError as error:
        raise ValueError("normalized-exponential score cut is not typed by the source score program") from error

    if region.maximum is region.normalizer:
        if len(region.maximum.results) != 2:
            raise ValueError("combined normalized-exponential Fold must return maximum and normalizer state")
        maximum_type, normalizer_type = (result.type for result in region.maximum.results)
    else:
        if len(region.maximum.results) != 1 or len(region.normalizer.results) != 1:
            raise ValueError("normalized-exponential Folds must each return one sufficient state")
        maximum_type = region.maximum.results[0].type
        normalizer_type = region.normalizer.results[0].type
    return score_type, maximum_type, normalizer_type


def normalized_exp_consumer_dimension_roles(
    region: PhysicalNormalizedExpContractRegion,
) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
    """Map producer batch/relation/owner roles into the consumer domain."""
    relation = dict(region.producer_to_consumer_dimensions)
    producer_dimensions = {
        *region.geometry.batch_dimensions,
        region.geometry.shared_relation_dimension,
        *region.geometry.relation_group_dimensions,
        region.geometry.owner_dimension,
        region.geometry.streamed_dimension,
    }
    if len(relation) != len(region.producer_to_consumer_dimensions) or set(relation) != producer_dimensions:
        raise ValueError("normalized score relation does not map every producer score dimension exactly once")
    if len(set(relation.values())) != len(relation):
        raise ValueError("normalized score relation does not map producer dimensions one-to-one")
    return (
        tuple(relation[dimension] for dimension in region.geometry.batch_dimensions),
        relation[region.geometry.shared_relation_dimension],
        tuple(relation[dimension] for dimension in region.geometry.relation_group_dimensions),
        relation[region.geometry.owner_dimension],
    )


def source_bound_tasks(request: GpuPhysicalArtifactRequest) -> tuple[SourceBoundTask, ...]:
    """Decode the exact source-derived composition graph."""
    tasks: list[SourceBoundTask] = []
    for operation in request.materialization.body().operations:
        if operation.name != "shuttle.materialization_task":
            continue
        attributes = operation.attributes
        if "composition_sources" not in attributes and "source" not in attributes:
            raise ValueError("materialization task has no source-bound composition")
        if "composition_sources" in attributes and "source" in attributes:
            composition_sources = attributes["composition_sources"]
            if not isinstance(composition_sources, list) or attributes["source"] not in composition_sources:
                raise ValueError("materialization task principal source is outside its source-bound composition")
        encoded_sources = attributes.get("composition_sources")
        if encoded_sources is None:
            encoded_sources = [attributes["source"]]
        if not isinstance(encoded_sources, list):
            raise ValueError("materialization task composition sources must be an array")
        source_operations = tuple(
            _source_operation_ordinal(source, "materialization task composition source") for source in encoded_sources
        )
        encoded_expansion_group = attributes.get("source_expansion_group")
        encoded_expansion = attributes.get("source_expansion")
        if encoded_expansion_group is not None and encoded_expansion is not None:
            raise ValueError("materialization task source expansion cannot be both singular and grouped")
        if encoded_expansion_group is not None:
            if not isinstance(encoded_expansion_group, list) or len(encoded_expansion_group) != len(encoded_sources):
                raise ValueError("materialization task source expansion group must align with composition sources")
            source_expansions = tuple(encoded_expansion_group)
        elif encoded_expansion is not None:
            if len(encoded_sources) != 1:
                raise ValueError("materialization task source expansion requires one source")
            source_expansions = (encoded_expansion,)
        else:
            source_expansions = (None,) * len(encoded_sources)
        source_identities = tuple(
            json.dumps((source, expansion), sort_keys=True, separators=(",", ":"))
            for source, expansion in zip(encoded_sources, source_expansions, strict=True)
        )
        if len(set(source_identities)) != len(source_identities):
            raise ValueError("materialization task composition sources must bind distinct operations")
        phases, phase_edges, result_bindings = _source_bound_composition_phases(attributes)
        domain_shape = _integer_array(attributes.get("domain_shape"), "materialization task domain")
        reduction_dimensions = _integer_array(attributes.get("reduction_dimensions"), "materialization task reductions")
        partition_names = (
            "composition_owner_dimensions",
            "online_vector_dimensions",
            "composition_stream_dimensions",
            "composition_producer_contraction_dimensions",
            "composition_consumer_contraction_dimensions",
        )
        explicit_partition = tuple(name in attributes for name in partition_names)
        if ("composition_state_inputs" in attributes or "composition_state_outputs" in attributes) and not all(
            explicit_partition
        ):
            raise ValueError("stateful materialization tasks require an explicit composition partition")
        if any(explicit_partition):
            if "composition_owner_dimensions" not in attributes or "composition_stream_dimensions" not in attributes:
                raise ValueError("partial composition partitions require explicit owner and stream dimensions")
            owner_dimensions = _integer_array(
                attributes.get("composition_owner_dimensions"), "composition owner dimensions"
            )
            vector_dimensions = _optional_integer_array(
                attributes.get("online_vector_dimensions"), "composition vector dimensions"
            )
            stream_dimensions = _integer_array(
                attributes.get("composition_stream_dimensions"), "composition stream dimensions"
            )
            producer_contraction_dimensions = _optional_integer_array(
                attributes.get("composition_producer_contraction_dimensions"),
                "composition producer contraction dimensions",
            )
            consumer_contraction_dimensions = _optional_integer_array(
                attributes.get("composition_consumer_contraction_dimensions"),
                "composition consumer contraction dimensions",
            )
        else:
            # Stateless generic Contract/Fold tasks carry the exact domain and
            # reduction dimensions but deliberately do not claim an online
            # ownership partition.  This complement is only a structural task
            # partition; physical ownership is derived later from algebraic
            # indexing relations and contraction geometry.
            stream_dimensions = reduction_dimensions
            owner_dimensions = tuple(axis for axis in range(len(domain_shape)) if axis not in set(reduction_dimensions))
            vector_dimensions = ()
            producer_contraction_dimensions = ()
            consumer_contraction_dimensions = ()
        tasks.append(
            SourceBoundTask(
                ordinal=_integer(attributes.get("ordinal"), "materialization task ordinal"),
                source_operations=source_operations,
                inputs=_integer_array(attributes.get("input_buffers"), "materialization task inputs"),
                outputs=_integer_array(attributes.get("output_buffers"), "materialization task outputs"),
                dependencies=_integer_array(attributes.get("dependencies"), "materialization task dependencies"),
                domain_shape=domain_shape,
                reduction_dimensions=reduction_dimensions,
                owner_dimensions=owner_dimensions,
                vector_dimensions=vector_dimensions,
                stream_dimensions=stream_dimensions,
                state_inputs=_optional_integer_array(
                    attributes.get("composition_state_inputs"), "composition state inputs"
                ),
                state_outputs=_optional_integer_array(
                    attributes.get("composition_state_outputs"), "composition state outputs"
                ),
                producer_contraction_dimensions=producer_contraction_dimensions,
                consumer_contraction_dimensions=consumer_contraction_dimensions,
                phases=phases,
                phase_edges=phase_edges,
                result_bindings=result_bindings,
                source_expansions=source_expansions,
            )
        )
    tasks.sort(key=lambda task: task.ordinal)
    if [task.ordinal for task in tasks] != list(range(len(tasks))):
        raise ValueError("materialization tasks must have dense source-bound ordinals")
    for task in tasks:
        rank = len(task.domain_shape)
        axes = (
            *task.reduction_dimensions,
            *task.owner_dimensions,
            *task.vector_dimensions,
            *task.stream_dimensions,
            *task.producer_contraction_dimensions,
            *task.consumer_contraction_dimensions,
        )
        if any(axis < 0 or axis >= rank for axis in axes):
            raise ValueError("composition dimensions must index the task domain")
        partitions = (task.owner_dimensions, task.vector_dimensions, task.stream_dimensions)
        partition_axes = tuple(axis for partition in partitions for axis in partition)
        if (
            any(len(set(partition)) != len(partition) for partition in partitions)
            or len(set(partition_axes)) != len(partition_axes)
            or set(partition_axes) != set(range(rank))
        ):
            raise ValueError(
                "composition owner, vector, and stream dimensions must form a disjoint exhaustive partition"
            )
        has_state = bool(task.state_inputs or task.state_outputs)
        if not has_state and task.stream_dimensions != task.reduction_dimensions:
            raise ValueError("stateless composition stream dimensions must equal task reduction dimensions")
        if not set(task.state_inputs).issubset(task.inputs):
            raise ValueError("composition state inputs must be task inputs")
        if not set(task.state_outputs).issubset(task.outputs):
            raise ValueError("composition state outputs must be task outputs")
        if task.phases:
            if tuple(phase.source_operation for phase in task.phases) != task.source_operations:
                raise ValueError("composition phases must preserve source operation order")
            if tuple(binding.buffer for binding in task.result_bindings) != task.outputs:
                raise ValueError("composition result bindings must preserve task output order")
    return tuple(tasks)


def _source_bound_composition_phases(
    attributes: Mapping[str, Any],
) -> tuple[
    tuple[SourceBoundPhase, ...],
    tuple[SourceBoundPhaseEdge, ...],
    tuple[SourceBoundResultBinding, ...],
]:
    encoded_phases = attributes.get("composition_phases")
    encoded_edges = attributes.get("composition_phase_edges")
    encoded_bindings = attributes.get("composition_result_bindings")
    if encoded_phases is None and encoded_edges is None and encoded_bindings is None:
        return (), (), ()
    if (
        not isinstance(encoded_phases, list)
        or not isinstance(encoded_edges, list)
        or not isinstance(encoded_bindings, list)
    ):
        raise ValueError("composition phase, edge, and result records must be present together as arrays")
    phases = tuple(
        SourceBoundPhase(
            source_operation=_source_operation_ordinal(phase.get("source"), "composition phase source"),
            kind=_mlir_attribute(phase.get("kind"), "composition phase kind"),
            domain_shape=_integer_array(phase.get("domain_shape"), "composition phase domain"),
            reduction_dimensions=_integer_array(phase.get("reduction_dimensions"), "composition phase reductions"),
        )
        for phase in encoded_phases
        if isinstance(phase, dict)
    )
    if len(phases) != len(encoded_phases):
        raise ValueError("composition phases must be typed dictionaries")
    edges = tuple(
        SourceBoundPhaseEdge(
            source_phase=_integer(edge.get("source_phase"), "composition edge source phase"),
            source_result=_integer(edge.get("source_result"), "composition edge source result"),
            target_phase=_integer(edge.get("target_phase"), "composition edge target phase"),
            target_operand=_integer(edge.get("target_operand"), "composition edge target operand"),
            access_map=_composition_affine_map(edge.get("access_map")),
        )
        for edge in encoded_edges
        if isinstance(edge, dict)
    )
    if len(edges) != len(encoded_edges):
        raise ValueError("composition phase edges must be typed dictionaries")
    bindings = tuple(
        SourceBoundResultBinding(
            phase=_integer(binding.get("phase"), "composition result phase"),
            source_operation=_source_operation_ordinal(binding.get("source"), "composition result source"),
            result=_integer(binding.get("result"), "composition result ordinal"),
            buffer=_integer(binding.get("buffer"), "composition result buffer"),
        )
        for binding in encoded_bindings
        if isinstance(binding, dict)
    )
    if len(bindings) != len(encoded_bindings):
        raise ValueError("composition result bindings must be typed dictionaries")
    for phase in phases:
        if any(extent <= 0 for extent in phase.domain_shape):
            raise ValueError("composition phases require static domains")
        if any(dimension < 0 or dimension >= len(phase.domain_shape) for dimension in phase.reduction_dimensions) or len(
            set(phase.reduction_dimensions)
        ) != len(phase.reduction_dimensions):
            raise ValueError("composition phase reductions must uniquely index its domain")
    for edge in edges:
        if min(edge.source_phase, edge.source_result, edge.target_phase, edge.target_operand) < 0:
            raise ValueError("composition phase edges require non-negative ordinals")
        if edge.source_phase >= len(phases) or edge.target_phase >= len(phases):
            raise ValueError("composition phase edge references an unavailable phase")
    for binding in bindings:
        if min(binding.phase, binding.source_operation, binding.result, binding.buffer) < 0:
            raise ValueError("composition result bindings require non-negative ordinals")
        if binding.phase >= len(phases) or binding.source_operation != phases[binding.phase].source_operation:
            raise ValueError("composition result binding disagrees with its phase source")
    return phases, edges, bindings


def _composition_affine_map(encoded: object) -> CuteAffineMap:
    if (
        not isinstance(encoded, dict)
        or encoded.get("kind") != "affine_map"
        or type(encoded.get("dimensions")) is not int
        or encoded.get("symbols") != 0
        or not isinstance(encoded.get("results"), list)
    ):
        raise ValueError("composition phase access must be a symbol-free affine map")
    return CuteAffineMap(
        encoded["dimensions"],
        tuple(affine_expression_from_record(expression) for expression in encoded["results"]),
    )


def source_bound_schedule_tasks(request: GpuPhysicalArtifactRequest) -> tuple[SourceBoundScheduleTask, ...]:
    """Decode schedule records and prove their exact materialization-task provenance."""
    source_tasks = source_bound_tasks(request)
    schedules: list[SourceBoundScheduleTask] = []
    source_task_by_ordinal = {task.ordinal: task for task in source_tasks}
    for operation in request.schedule.body().operations:
        if operation.name != "shuttle.schedule_task":
            continue
        attributes = operation.attributes
        source_task_ordinal = _integer(attributes.get("source_task"), "schedule source task")
        source_task = source_task_by_ordinal.get(source_task_ordinal)
        if source_task is None:
            raise ValueError(f"schedule task binds unknown materialization task {source_task_ordinal}")
        schedules.append(
            SourceBoundScheduleTask(
                ordinal=_integer(attributes.get("ordinal"), "schedule task ordinal"),
                source_task=source_task_ordinal,
                kind=_mlir_attribute(attributes.get("kind"), "schedule task kind"),
                domain_shape=_integer_array(attributes.get("domain_shape"), "schedule task domain"),
                grid_shape=_integer_array(attributes.get("grid_shape"), "schedule task grid"),
                tile_shape=_integer_array(attributes.get("tile_shape"), "schedule task tile"),
                serial_tiles=_integer(attributes.get("serial_tiles"), "schedule serial tiles"),
                workgroup_threads=_integer(attributes.get("workgroup_threads"), "schedule workgroup threads"),
                subgroup_size=_integer(attributes.get("subgroup_size"), "schedule subgroup size"),
                scratch_bytes=_integer(attributes.get("scratch_bytes"), "schedule scratch bytes"),
                input_buffers=_integer_array(attributes.get("input_buffers"), "schedule task inputs"),
                output_buffers=_integer_array(attributes.get("output_buffers"), "schedule task outputs"),
                dependencies=_integer_array(attributes.get("dependencies"), "schedule task dependencies"),
                reduction_axis=(
                    None
                    if attributes.get("reduction_axis") is None
                    else _integer(attributes.get("reduction_axis"), "schedule reduction axis")
                ),
                owner_dimensions=(
                    source_task.owner_dimensions
                    if "composition_owner_dimensions" not in attributes
                    else _integer_array(attributes.get("composition_owner_dimensions"), "schedule owner dimensions")
                ),
                vector_dimensions=(
                    source_task.vector_dimensions
                    if "online_vector_dimensions" not in attributes
                    else _integer_array(attributes.get("online_vector_dimensions"), "schedule vector dimensions")
                ),
                stream_dimensions=(
                    source_task.stream_dimensions
                    if "composition_stream_dimensions" not in attributes
                    else _integer_array(attributes.get("composition_stream_dimensions"), "schedule stream dimensions")
                ),
                state_inputs=_optional_integer_array(
                    attributes.get("composition_state_inputs"), "schedule state inputs"
                ),
                state_outputs=_optional_integer_array(
                    attributes.get("composition_state_outputs"), "schedule state outputs"
                ),
                producer_contraction_dimensions=(
                    source_task.producer_contraction_dimensions
                    if "composition_producer_contraction_dimensions" not in attributes
                    else _integer_array(
                        attributes.get("composition_producer_contraction_dimensions"),
                        "schedule producer contraction dimensions",
                    )
                ),
                consumer_contraction_dimensions=(
                    source_task.consumer_contraction_dimensions
                    if "composition_consumer_contraction_dimensions" not in attributes
                    else _integer_array(
                        attributes.get("composition_consumer_contraction_dimensions"),
                        "schedule consumer contraction dimensions",
                    )
                ),
                online_mma=_optional_boolean(attributes.get("online_mma"), "schedule online MMA"),
            )
        )
    schedules.sort(key=lambda schedule: schedule.ordinal)
    if [schedule.ordinal for schedule in schedules] != list(range(len(schedules))):
        raise ValueError("schedule tasks must have dense source-bound ordinals")
    if len(schedules) != len(source_tasks):
        raise ValueError("schedule plan must bind every materialization task exactly once")
    for schedule, task in zip(schedules, source_tasks, strict=True):
        if schedule.source_task != task.ordinal:
            raise ValueError(
                f"schedule task {schedule.ordinal} binds materialization task {schedule.source_task}, "
                f"expected {task.ordinal}"
            )
        expected = (
            ("domain", schedule.domain_shape, task.domain_shape),
            ("inputs", schedule.input_buffers, task.inputs),
            ("outputs", schedule.output_buffers, task.outputs),
            ("dependencies", schedule.dependencies, task.dependencies),
            ("owner dimensions", schedule.owner_dimensions, task.owner_dimensions),
            ("vector dimensions", schedule.vector_dimensions, task.vector_dimensions),
            ("stream dimensions", schedule.stream_dimensions, task.stream_dimensions),
            ("state inputs", schedule.state_inputs, task.state_inputs),
            ("state outputs", schedule.state_outputs, task.state_outputs),
            (
                "producer contraction dimensions",
                schedule.producer_contraction_dimensions,
                task.producer_contraction_dimensions,
            ),
            (
                "consumer contraction dimensions",
                schedule.consumer_contraction_dimensions,
                task.consumer_contraction_dimensions,
            ),
        )
        for label, actual, source in expected:
            if actual != source:
                raise ValueError(f"schedule task {schedule.ordinal} {label} disagree with its materialization task")
        expected_reduction_axis = task.stream_dimensions[-1] if task.stream_dimensions else None
        if schedule.reduction_axis != expected_reduction_axis:
            raise ValueError("schedule reduction axis must be the final streamed dimension")
        compact_contract_tile = (
            not task.producer_contraction_dimensions
            and not task.consumer_contraction_dimensions
            and len(schedule.tile_shape) == 1
        )
        if len(schedule.tile_shape) != len(task.domain_shape) and not compact_contract_tile:
            raise ValueError("composition schedule tile rank must equal its task-domain rank")
        if not schedule.grid_shape or any(extent <= 0 for extent in schedule.grid_shape):
            raise ValueError("composition schedule grid extents must be positive")
        if compact_contract_tile:
            invalid_tile = schedule.tile_shape[0] <= 0 or schedule.tile_shape[0] > math.prod(task.domain_shape)
        else:
            invalid_tile = any(
                tile <= 0 or tile > extent for tile, extent in zip(schedule.tile_shape, task.domain_shape, strict=True)
            )
        if invalid_tile:
            raise ValueError("composition schedule tiles must be positive and bounded by the task domain")
        if schedule.serial_tiles <= 0 or schedule.workgroup_threads <= 0 or schedule.subgroup_size <= 0:
            raise ValueError("composition schedule execution geometry must be positive")
        if schedule.workgroup_threads > schedule.subgroup_size and schedule.workgroup_threads % schedule.subgroup_size:
            raise ValueError("composition schedule threads must be one partial or a whole number of subgroups")
        if schedule.scratch_bytes < 0:
            raise ValueError("composition schedule scratch bytes cannot be negative")
        if any(schedule.tile_shape[dimension] != task.domain_shape[dimension] for dimension in task.vector_dimensions):
            raise ValueError("composition schedule must keep each vector dimension within one physical tile")
    return tuple(schedules)


def source_argument_buffers(
    request: GpuPhysicalArtifactRequest,
    buffers: tuple[SourceBoundBuffer, ...],
) -> dict[int, SourceBoundBuffer]:
    """Bind live-in buffers to exact Region block arguments."""
    arguments = request.region.body().arguments
    by_argument: dict[int, SourceBoundBuffer] = {}
    for buffer in buffers:
        if not buffer.live_in:
            if buffer.source_argument is not None:
                raise ValueError(f"non-live-in source buffer {buffer.ordinal} has a source argument")
            continue
        argument = buffer.source_argument
        if argument is None:
            raise ValueError(f"live-in source buffer {buffer.ordinal} has no source argument")
        if argument < 0 or argument >= len(arguments):
            raise ValueError(f"source buffer {buffer.ordinal} argument {argument} is outside the Region boundary")
        if argument in by_argument:
            raise ValueError(f"Region argument {argument} is bound by multiple source buffers")
        if buffer.type != arguments[argument].type:
            raise ValueError(f"source buffer {buffer.ordinal} type disagrees with Region argument {argument}")
        if buffer.ordinal >= len(request.binding_indices) or request.binding_indices[buffer.ordinal] != argument:
            raise ValueError(
                f"source buffer {buffer.ordinal} physical binding disagrees with Region argument {argument}"
            )
        by_argument[argument] = buffer
    if set(by_argument) != set(range(len(arguments))):
        raise ValueError("live-in source buffers must bind every Region argument exactly once")
    return by_argument


def source_result_buffers(
    request: GpuPhysicalArtifactRequest,
    buffers: tuple[SourceBoundBuffer, ...],
) -> dict[int, SourceBoundBuffer]:
    """Bind live-out buffers to exact Region yield results."""
    body = request.region.body()
    if not body.operations or body.operations[-1].name != "shuttle.yield":
        raise ValueError("source Region must terminate with shuttle.yield")
    yielded = body.operations[-1].operands
    value_types = {argument.id: argument.type for argument in body.arguments}
    for operation in body.operations:
        value_types.update((result.id, result.type) for result in operation.results)

    by_result: dict[int, SourceBoundBuffer] = {}
    abi_bindings: set[int] = set()
    for buffer in buffers:
        result = (
            request.region_result_indices[buffer.ordinal]
            if buffer.ordinal < len(request.region_result_indices)
            else None
        )
        if not buffer.live_out:
            if result is not None:
                raise ValueError(f"non-live-out source buffer {buffer.ordinal} has a Region result")
            continue
        if result is None:
            raise ValueError(f"live-out source buffer {buffer.ordinal} has no Region result")
        abi_binding = request.binding_indices[buffer.ordinal] if buffer.ordinal < len(request.binding_indices) else None
        if abi_binding is None or abi_binding < 0:
            raise ValueError(f"live-out source buffer {buffer.ordinal} has no public ABI result binding")
        if abi_binding in abi_bindings:
            raise ValueError(f"public ABI result binding {abi_binding} is used by multiple source buffers")
        abi_bindings.add(abi_binding)
        if result < 0 or result >= len(yielded):
            raise ValueError(f"source buffer {buffer.ordinal} result {result} is outside the Region boundary")
        if result in by_result:
            raise ValueError(f"Region result {result} is bound by multiple source buffers")
        yielded_type = value_types.get(yielded[result])
        if yielded_type is None or buffer.type != yielded_type:
            raise ValueError(f"source buffer {buffer.ordinal} type disagrees with Region result {result}")
        by_result[result] = buffer
    if set(by_result) != set(range(len(yielded))):
        raise ValueError("live-out source buffers must bind every Region result exactly once")
    if abi_bindings != set(range(len(yielded))):
        raise ValueError("live-out source buffers must bind every public ABI result exactly once")
    return by_result


def _required_source_argument(buffer: SourceBoundBuffer) -> int:
    if buffer.source_argument is None:
        raise ValueError(f"live-in source buffer {buffer.ordinal} has no source argument")
    return buffer.source_argument


def _affine_expression_dimensions(expression: object) -> frozenset[int]:
    if not isinstance(expression, dict):
        raise ValueError("affine indexing expression must be structured")
    kind = expression.get("kind")
    if kind == "dimension":
        return frozenset((_integer(expression.get("position"), "affine dimension position"),))
    if kind in ("constant", "symbol"):
        return frozenset()
    if kind in ("add", "multiply", "modulo", "floor_divide", "ceil_divide"):
        return _affine_expression_dimensions(expression.get("left")) | _affine_expression_dimensions(
            expression.get("right")
        )
    raise ValueError(f"unsupported affine indexing expression {kind!r}")


def _affine_map_dimensions(affine_map: dict[str, Any]) -> frozenset[int]:
    results = affine_map.get("results")
    if not isinstance(results, list):
        raise ValueError("affine map has no structured results")
    return frozenset(dimension for expression in results for dimension in _affine_expression_dimensions(expression))


def _nearest_contract(
    value: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> ShuttleOperationRecord:
    worklist = [value]
    visited: set[int] = set()
    candidates: list[ShuttleOperationRecord] = []
    while worklist:
        current = worklist.pop()
        if current in visited:
            continue
        visited.add(current)
        operation = producer_by_value.get(current)
        if operation is None:
            continue
        if operation.name == "shuttle.contract":
            if all(candidate is not operation for candidate in candidates):
                candidates.append(operation)
            continue
        worklist.extend(operation.operands)
    if len(candidates) != 1:
        raise ValueError(f"expected one nearest terminal Contract, found {len(candidates)}")
    return candidates[0]


def _terminal_result_value(
    request: GpuPhysicalArtifactRequest,
    buffers: tuple[SourceBoundBuffer, ...],
    buffer_ordinal: int,
) -> int:
    if len(request.region_result_indices) != len(buffers):
        raise ValueError("saved-state reverse buffers do not have complete Region-result identities")
    result_index = request.region_result_indices[buffer_ordinal]
    body = request.region.body()
    if not body.operations or body.operations[-1].name != "shuttle.yield":
        raise ValueError("saved-state reverse Region has no source-bound yield")
    yielded = body.operations[-1].operands
    if result_index is None or result_index < 0 or result_index >= len(yielded):
        raise ValueError(f"terminal source buffer {buffer_ordinal} has no Region-result identity")
    return yielded[result_index]


def _terminal_contract_binding(
    contract: ShuttleOperationRecord,
    task: SourceBoundTask,
    producer_by_value: dict[int, ShuttleOperationRecord],
    argument_index: dict[int, int],
    bf16_arguments: frozenset[int],
) -> _TerminalContractBinding:
    maps = _affine_maps(contract)
    if len(contract.operands) != 2 or len(contract.results) != 1 or len(maps) != 3:
        raise ValueError("terminal reverse Contract must have two operands and one result relation")
    owner = frozenset(task.owner_dimensions)
    vector = frozenset(task.vector_dimensions)
    stream = frozenset(task.stream_dimensions)
    full_domain = owner | vector | stream
    operand_dimensions = tuple(_affine_map_dimensions(affine_map) for affine_map in maps[:2])
    result_dimensions = _affine_map_dimensions(maps[2])
    if result_dimensions != owner | vector:
        raise ValueError("terminal reverse Contract result relation does not match owner/vector ownership")
    candidates: list[_TerminalContractBinding] = []
    for direct_operand in range(2):
        algebra_operand = 1 - direct_operand
        direct_roots = _argument_roots(contract.operands[direct_operand], producer_by_value, argument_index)
        algebra_roots = _argument_roots(contract.operands[algebra_operand], producer_by_value, argument_index)
        if (
            len(direct_roots) != 1
            or not direct_roots.issubset(bf16_arguments)
            or len(algebra_roots) < 2
            or not (vector | stream).issubset(operand_dimensions[direct_operand])
            or operand_dimensions[algebra_operand] != owner | stream
            or not operand_dimensions[direct_operand].issubset(full_domain)
        ):
            continue
        candidates.append(
            _TerminalContractBinding(
                contract=contract,
                direct_argument=next(iter(direct_roots)),
                algebra_operand=contract.operands[algebra_operand],
            )
        )
    if len(candidates) != 1:
        raise ValueError(f"terminal reverse Contract has {len(candidates)} unambiguous direct operand relations")
    return candidates[0]


def _two_argument_contracts(
    value: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
    argument_index: dict[int, int],
    bf16_arguments: frozenset[int],
) -> tuple[tuple[ShuttleOperationRecord, frozenset[int]], ...]:
    candidates: list[tuple[ShuttleOperationRecord, frozenset[int]]] = []
    for operation in _ancestor_operations(value, producer_by_value):
        if operation.name != "shuttle.contract" or len(operation.operands) != 2 or len(operation.results) != 1:
            continue
        operand_roots = tuple(
            _argument_roots(operand, producer_by_value, argument_index) for operand in operation.operands
        )
        if (
            any(len(roots) != 1 or not roots.issubset(bf16_arguments) for roots in operand_roots)
            or operand_roots[0] == operand_roots[1]
        ):
            continue
        maps = _affine_maps(operation)
        if len(maps) != 3:
            raise ValueError("two-input Contract must carry two operand relations and one result relation")
        left = _affine_map_dimensions(maps[0])
        right = _affine_map_dimensions(maps[1])
        result = _affine_map_dimensions(maps[2])
        if not ((left & right) - result) or not ((left & result) - right) or not ((right & result) - left):
            continue
        candidates.append((operation, operand_roots[0] | operand_roots[1]))
    return tuple(candidates)


def _output_dependent_fold_arguments(
    operations: tuple[ShuttleOperationRecord, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
    argument_index: dict[int, int],
    bf16_arguments: frozenset[int],
    state_type: ShuttleTypeRecord,
) -> _OutputDependentFoldBinding:
    candidates: list[_OutputDependentFoldBinding] = []
    for operation in operations:
        if _fold_reducer(operation) != "sum" or len(operation.results) != 1 or operation.results[0].type != state_type:
            continue
        roots = frozenset(
            root
            for operand in operation.operands
            for root in _argument_roots(operand, producer_by_value, argument_index)
            if root in bf16_arguments
        )
        contract_ancestors = tuple(
            ancestor
            for operand in operation.operands
            for ancestor in _ancestor_operations(operand, producer_by_value)
            if ancestor.name == "shuttle.contract"
        )
        if len(roots) == 2 and not contract_ancestors:
            candidates.append(_OutputDependentFoldBinding(operation, roots))
    if len(candidates) != 1:
        raise ValueError(f"expected one output-dependent Fold over two source arguments, found {len(candidates)}")
    return candidates[0]


def _score_expression_from_saved_state_recompute(
    normalized_weight: int,
    producer: ShuttleOperationRecord,
    saved_state_argument: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
    argument_index: dict[int, int],
) -> tuple[ScalarExpression, PhysicalContractDomainPredicate | None]:
    producer_result = _single_result(producer)
    score_program = _ancestor_operations(normalized_weight, producer_by_value)
    domain_predicate = _domain_predicate(score_program, producer, producer_by_value)
    candidates: dict[int, ScalarExpression] = {}
    for operation in _ancestor_operations(normalized_weight, producer_by_value):
        if operation.name != "shuttle.map":
            continue
        operand_roots = tuple(
            _argument_roots(operand, producer_by_value, argument_index) for operand in operation.operands
        )
        if not any(saved_state_argument in roots for roots in operand_roots):
            continue
        for operand, roots in zip(operation.operands, operand_roots, strict=True):
            if saved_state_argument in roots:
                continue
            try:
                expression = _scalar_expression_from_graph(
                    operand,
                    {producer_result: _SCORE_INPUT},
                    producer_by_value,
                    domain_predicate,
                )
            except ValueError:
                continue
            if scalar_expression_inputs(expression) == {_SCORE_INPUT}:
                candidates[operand] = expression
    if len(candidates) != 1:
        raise ValueError(
            f"saved-state normalized-exp recomputation has {len(candidates)} unambiguous transformed-score branches"
        )
    return _without_domain_predicate(next(iter(candidates.values()))), domain_predicate


def _output_product_expression(
    fold: ShuttleOperationRecord,
    finalizer_output_argument: int,
    finalizer_cotangent_argument: int,
    body_arguments: tuple[ShuttleValueRecord, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> ScalarExpression:
    if len(fold.operands) != 1:
        raise ValueError("output-dependent Fold must consume one scalar Map result")
    expression = _scalar_expression_from_graph(
        fold.operands[0],
        {
            body_arguments[finalizer_output_argument].id: _FINALIZER_OUTPUT_INPUT,
            body_arguments[finalizer_cotangent_argument].id: _FINALIZER_COTANGENT_INPUT,
        },
        producer_by_value,
        None,
    )
    expected_inputs = {_FINALIZER_OUTPUT_INPUT, _FINALIZER_COTANGENT_INPUT}
    if scalar_expression_inputs(expression) != expected_inputs:
        raise ValueError("output-dependent Fold scalar AST must consume finalizer output and cotangent exactly")
    if not _binary_scalar_inputs(
        expression,
        ScalarExpressionKind.MULTIPLY,
        _FINALIZER_OUTPUT_INPUT,
        _FINALIZER_COTANGENT_INPUT,
    ):
        raise ValueError("current saved-state reverse lowering requires the unit-finalizer O*dO Fold Map")
    return expression


def _reverse_score_pullback_expressions(
    raw_score_cotangent: int,
    normalized_weight: int,
    consumer_weight_cotangent: ShuttleOperationRecord,
    output_dependent_fold: ShuttleOperationRecord,
    producer: ShuttleOperationRecord,
    score_expression: ScalarExpression,
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> tuple[ScalarExpression, ScalarExpression]:
    expected_score_vjp = scalar_expression_vjp(
        score_expression,
        input_name=_SCORE_INPUT,
        cotangent_name=_MAPPED_SCORE_COTANGENT_INPUT,
    )
    candidate_values = {
        raw_score_cotangent,
        *(
            result.id
            for operation in _ancestor_operations(raw_score_cotangent, producer_by_value)
            for result in operation.results
        ),
    }
    decoded_candidates: list[tuple[int, ScalarExpression]] = []
    for mapped_score_cotangent in candidate_values - {_single_result(producer)}:
        try:
            expression = _scalar_expression_from_graph(
                raw_score_cotangent,
                {
                    mapped_score_cotangent: _MAPPED_SCORE_COTANGENT_INPUT,
                    _single_result(producer): _SCORE_INPUT,
                },
                producer_by_value,
                None,
            )
        except ValueError:
            continue
        if expression == expected_score_vjp:
            decoded_candidates.append((mapped_score_cotangent, expression))
    if len(decoded_candidates) != 1:
        raise ValueError(
            f"reverse score pullback has {len(decoded_candidates)} scalar subgraphs matching the generic score-Map VJP"
        )
    mapped_score_cotangent, score_map_vjp_expression = decoded_candidates[0]
    normalized_exp_pullback_expression = _scalar_expression_from_graph(
        mapped_score_cotangent,
        {
            normalized_weight: "normalized_exp_contract.normalized_weight",
            _single_result(consumer_weight_cotangent): "normalized_exp_contract.weight_cotangent",
            _single_result(output_dependent_fold): "normalized_exp_contract.output_dot_cotangent",
        },
        producer_by_value,
        None,
    )
    normalized_weight_input = "normalized_exp_contract.normalized_weight"
    weight_cotangent_input = "normalized_exp_contract.weight_cotangent"
    output_dot_input = "normalized_exp_contract.output_dot_cotangent"
    difference = scalar_binary(
        ScalarExpressionKind.SUBTRACT,
        scalar_input(weight_cotangent_input),
        scalar_input(output_dot_input),
    )
    expected_normalized_pullbacks = (
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(normalized_weight_input),
            difference,
        ),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            difference,
            scalar_input(normalized_weight_input),
        ),
    )
    if normalized_exp_pullback_expression not in expected_normalized_pullbacks:
        raise ValueError("current saved-state reverse lowering requires the unit-finalizer normalized-exp pullback")
    return score_map_vjp_expression, normalized_exp_pullback_expression


def _binary_scalar_inputs(
    expression: ScalarExpression,
    kind: ScalarExpressionKind,
    left: str,
    right: str,
) -> bool:
    if expression.kind is not kind:
        return False
    inputs = tuple(
        operand.input_name if operand.kind is ScalarExpressionKind.INPUT else None for operand in expression.operands
    )
    return inputs in ((left, right), (right, left))


def _saved_state_reverse_operands(
    request: GpuPhysicalArtifactRequest,
    buffers: tuple[SourceBoundBuffer, ...],
    owner: SourceBoundTask,
    streamed: SourceBoundTask,
    owner_data_output: int,
    streamed_data_outputs: tuple[int, int],
) -> _SavedStateReverseAlgebra:
    body = request.region.body()
    operations = tuple(operation for operation in body.operations if operation.results)
    producer_by_value = {result.id: operation for operation in operations for result in operation.results}
    argument_index = {argument.id: index for index, argument in enumerate(body.arguments)}
    by_argument = source_argument_buffers(request, buffers)
    bf16_arguments = frozenset(
        argument for argument, buffer in by_argument.items() if buffer.type.element_type == "bf16"
    )

    owner_contract = _nearest_contract(_terminal_result_value(request, buffers, owner_data_output), producer_by_value)
    owner_binding = _terminal_contract_binding(owner_contract, owner, producer_by_value, argument_index, bf16_arguments)
    streamed_bindings = tuple(
        _terminal_contract_binding(
            _nearest_contract(_terminal_result_value(request, buffers, output), producer_by_value),
            streamed,
            producer_by_value,
            argument_index,
            bf16_arguments,
        )
        for output in streamed_data_outputs
    )
    streamed_direct = frozenset(binding.direct_argument for binding in streamed_bindings)
    if len(streamed_direct) != 2 or owner_binding.direct_argument in streamed_direct:
        raise ValueError("reverse terminal Contracts do not expose three distinct direct source operands")

    if len(owner.state_outputs) != 1:
        raise ValueError("saved-state reverse owner task must produce one output-dependent Fold state")
    fold_binding = _output_dependent_fold_arguments(
        operations,
        producer_by_value,
        argument_index,
        bf16_arguments,
        buffers[owner.state_outputs[0]].type,
    )
    fold_arguments = fold_binding.arguments
    cotangent_candidates = fold_arguments & streamed_direct
    if len(cotangent_candidates) != 1:
        raise ValueError("output-dependent Fold does not identify one streamed terminal source operand")
    finalizer_cotangent = next(iter(cotangent_candidates))
    finalizer_outputs = fold_arguments - cotangent_candidates
    if len(finalizer_outputs) != 1:
        raise ValueError("output-dependent Fold does not identify one finalized output source")
    finalizer_output = next(iter(finalizer_outputs))
    producer_owner_candidates = streamed_direct - cotangent_candidates
    if len(producer_owner_candidates) != 1:
        raise ValueError("streamed terminal Contracts do not identify one producer-owner source")
    producer_owner = next(iter(producer_owner_candidates))
    producer_stream = owner_binding.direct_argument

    pair_contracts = _two_argument_contracts(
        owner_binding.algebra_operand,
        producer_by_value,
        argument_index,
        bf16_arguments,
    )
    consumer_candidates = tuple(
        roots - {finalizer_cotangent}
        for _, roots in pair_contracts
        if finalizer_cotangent in roots and len(roots - {finalizer_cotangent}) == 1
    )
    if len(consumer_candidates) != 1:
        raise ValueError(
            f"reverse scalar pullback has {len(consumer_candidates)} Contract relations pairing the finalizer cotangent"
        )
    consumer_input = next(iter(consumer_candidates[0]))

    producer_pairs = tuple(
        operation for operation, roots in pair_contracts if roots == frozenset((producer_owner, producer_stream))
    )
    if len(producer_pairs) != 1:
        raise ValueError(f"reverse scalar pullback has {len(producer_pairs)} producer Contract relations")

    consumer_contracts = tuple(
        operation for operation, roots in pair_contracts if roots == frozenset((finalizer_cotangent, consumer_input))
    )
    if len(consumer_contracts) != 1:
        raise ValueError(f"reverse scalar pullback has {len(consumer_contracts)} consumer-weight Contract relations")
    producer_operand_roots = tuple(
        _argument_roots(operand, producer_by_value, argument_index) for operand in producer_pairs[0].operands
    )
    producer_owner_operand_indices = tuple(
        index for index, roots in enumerate(producer_operand_roots) if roots == frozenset((producer_owner,))
    )
    consumer_operand_roots = tuple(
        _argument_roots(operand, producer_by_value, argument_index) for operand in consumer_contracts[0].operands
    )
    consumer_output_cotangent_operand_indices = tuple(
        index for index, roots in enumerate(consumer_operand_roots) if roots == frozenset((finalizer_cotangent,))
    )
    if len(producer_owner_operand_indices) != 1 or len(consumer_output_cotangent_operand_indices) != 1:
        raise ValueError("reverse source Contracts do not bind their operand relations uniquely")

    f32_arguments = tuple(argument for argument, buffer in by_argument.items() if buffer.type.element_type == "f32")
    if len(f32_arguments) != 1:
        raise ValueError("saved-state reverse scalar pullback requires one FP32 source state")
    saved_state_argument = f32_arguments[0]
    if by_argument[saved_state_argument].ordinal not in owner.state_inputs:
        raise ValueError("saved normalized-exp state is not an owner-task state input")

    consumer_terminal = tuple(
        (output, binding)
        for output, binding in zip(streamed_data_outputs, streamed_bindings, strict=True)
        if binding.direct_argument == finalizer_cotangent
    )
    producer_stream_terminal = tuple(
        (output, binding)
        for output, binding in zip(streamed_data_outputs, streamed_bindings, strict=True)
        if binding.direct_argument == producer_owner
    )
    if len(consumer_terminal) != 1 or len(producer_stream_terminal) != 1:
        raise ValueError("streamed terminal Contracts do not bind consumer and producer-stream cotangents uniquely")

    score_expression, domain_predicate = _score_expression_from_saved_state_recompute(
        consumer_terminal[0][1].algebra_operand,
        producer_pairs[0],
        saved_state_argument,
        producer_by_value,
        argument_index,
    )
    score_map_vjp_expression, normalized_exp_pullback_expression = _reverse_score_pullback_expressions(
        owner_binding.algebra_operand,
        consumer_terminal[0][1].algebra_operand,
        consumer_contracts[0],
        fold_binding.operation,
        producer_pairs[0],
        score_expression,
        producer_by_value,
    )
    output_product_expression = _output_product_expression(
        fold_binding.operation,
        finalizer_output,
        finalizer_cotangent,
        body.arguments,
        producer_by_value,
    )

    ordered_arguments = (
        producer_owner,
        producer_stream,
        consumer_input,
        finalizer_output,
        finalizer_cotangent,
    )
    if len(set(ordered_arguments)) != 5 or frozenset(ordered_arguments) != bf16_arguments:
        raise ValueError("reverse scalar pullback does not bind every BF16 source argument exactly once")
    return _SavedStateReverseAlgebra(
        producer_owner_input=by_argument[producer_owner],
        producer_stream_input=by_argument[producer_stream],
        consumer_input=by_argument[consumer_input],
        finalizer_output_input=by_argument[finalizer_output],
        finalizer_cotangent_input=by_argument[finalizer_cotangent],
        saved_state_input=by_argument[saved_state_argument],
        producer_owner_cotangent_output=buffers[owner_data_output],
        producer_stream_cotangent_output=buffers[producer_stream_terminal[0][0]],
        consumer_cotangent_output=buffers[consumer_terminal[0][0]],
        output_dependent_state_output=buffers[owner.state_outputs[0]],
        producer=producer_pairs[0],
        producer_owner_operand_index=producer_owner_operand_indices[0],
        consumer_weight_cotangent=consumer_contracts[0],
        consumer_output_cotangent_operand_index=consumer_output_cotangent_operand_indices[0],
        output_dependent_fold=fold_binding.operation,
        score_expression=score_expression,
        score_map_vjp_expression=score_map_vjp_expression,
        normalized_exp_pullback_expression=normalized_exp_pullback_expression,
        output_product_expression=output_product_expression,
        domain_predicate=domain_predicate,
    )


def physical_saved_state_reverse_boundary(
    request: GpuPhysicalArtifactRequest,
) -> PhysicalSavedStateReverseBoundary:
    """Inspect the historical saved-state benchmark boundary.

    This diagnostic adapter is intentionally not reachable from the canonical
    physical-provider match. Reverse compilation requires a typed pullback
    capability rather than this structural reconstruction.
    """
    if request.materialization.attributes.get("stateful_contract_fold_pullback") is not True:
        raise ValueError("materialization is not a certified stateful Contract/Fold pullback")
    buffers = source_bound_buffers(request)
    source_argument_buffers(request, buffers)
    tasks = source_bound_tasks(request)
    inputs = tuple(buffer for buffer in buffers if buffer.live_in)
    outputs = tuple(buffer for buffer in buffers if buffer.live_out)
    if len(inputs) != 6 or len(outputs) != 3:
        raise ValueError("saved-state reverse physical boundary requires six inputs and three outputs")
    if sum(buffer.type.element_type == "f32" for buffer in inputs) != 1 or any(
        buffer.type.element_type != "bf16" for buffer in inputs if buffer.type.element_type != "f32"
    ):
        raise ValueError("saved-state reverse physical boundary requires five BF16 inputs and one FP32 state input")
    if any(buffer.type.element_type != "bf16" for buffer in outputs):
        raise ValueError("saved-state reverse physical boundary requires three BF16 outputs")

    owner_candidates = tuple(
        task for task in tasks if task.state_inputs and task.state_outputs and len(task.reduction_dimensions) == 1
    )
    streamed_candidates = tuple(
        task for task in tasks if task.state_inputs and not task.state_outputs and len(task.reduction_dimensions) >= 2
    )
    if len(owner_candidates) != 1 or len(streamed_candidates) != 1 or len(tasks) != 2:
        raise ValueError("saved-state reverse physical boundary requires one owner and one streamed adjoint task")
    owner = owner_candidates[0]
    streamed = streamed_candidates[0]
    if owner.dependencies or streamed.dependencies != (owner.ordinal,):
        raise ValueError("saved-state reverse physical task dependencies disagree with state flow")
    if not set(owner.state_inputs).issubset(streamed.state_inputs) or not set(owner.state_outputs).issubset(
        streamed.state_inputs
    ):
        raise ValueError("streamed adjoint task does not consume the complete saved and output-dependent state")
    owner_data_outputs = tuple(ordinal for ordinal in owner.outputs if ordinal not in owner.state_outputs)
    streamed_data_outputs = tuple(ordinal for ordinal in streamed.outputs if ordinal not in streamed.state_outputs)
    if len(owner_data_outputs) != 1 or len(streamed_data_outputs) != 2:
        raise ValueError("saved-state reverse tasks must produce one owner and two streamed terminal Contracts")
    if set((*owner_data_outputs, *streamed_data_outputs)) != {buffer.ordinal for buffer in outputs}:
        raise ValueError("saved-state reverse task outputs do not close the three-result public boundary")
    state_ordinals = set((*owner.state_inputs, *owner.state_outputs, *streamed.state_inputs))
    if any(buffers[ordinal].type.element_type != "f32" for ordinal in state_ordinals):
        raise ValueError("saved-state reverse Fold state must use FP32 storage")
    reverse = _saved_state_reverse_operands(
        request,
        buffers,
        owner,
        streamed,
        owner_data_outputs[0],
        streamed_data_outputs,
    )
    return PhysicalSavedStateReverseBoundary(
        inputs=inputs,
        outputs=outputs,
        owner_task=owner,
        streamed_task=streamed,
        producer_owner_input=reverse.producer_owner_input,
        producer_stream_input=reverse.producer_stream_input,
        consumer_input=reverse.consumer_input,
        finalizer_output_input=reverse.finalizer_output_input,
        finalizer_cotangent_input=reverse.finalizer_cotangent_input,
        saved_state_input=reverse.saved_state_input,
        producer_owner_cotangent_output=reverse.producer_owner_cotangent_output,
        producer_stream_cotangent_output=reverse.producer_stream_cotangent_output,
        consumer_cotangent_output=reverse.consumer_cotangent_output,
        output_dependent_state_output=reverse.output_dependent_state_output,
        producer=reverse.producer,
        producer_owner_operand_index=reverse.producer_owner_operand_index,
        consumer_weight_cotangent=reverse.consumer_weight_cotangent,
        consumer_output_cotangent_operand_index=reverse.consumer_output_cotangent_operand_index,
        output_dependent_fold=reverse.output_dependent_fold,
        score_expression=reverse.score_expression,
        score_map_vjp_expression=reverse.score_map_vjp_expression,
        normalized_exp_pullback_expression=reverse.normalized_exp_pullback_expression,
        output_product_expression=reverse.output_product_expression,
        domain_predicate=reverse.domain_predicate,
    )


def physical_saved_state_reverse_region(
    boundary: PhysicalSavedStateReverseBoundary,
) -> PhysicalNormalizedExpContractRegion:
    """Reconstruct the shared generic forward algebra needed by a saved-state pullback."""
    producer_maps = _affine_maps(boundary.producer)
    consumer_pullback_maps = _affine_maps(boundary.consumer_weight_cotangent)
    if len(producer_maps) != 3 or len(consumer_pullback_maps) != 3:
        raise ValueError("saved-state reverse source Contracts must carry two operand maps and one result map")

    owner_operand = boundary.producer_owner_operand_index
    stream_operand = 1 - owner_operand
    producer = ShuttleOperationRecord(
        boundary.producer.name,
        dict(
            boundary.producer.attributes,
            indexing_maps=[producer_maps[owner_operand], producer_maps[stream_operand], producer_maps[2]],
        ),
        (boundary.producer.operands[owner_operand], boundary.producer.operands[stream_operand]),
        boundary.producer.results,
        boundary.producer.regions,
    )
    output_cotangent_operand = boundary.consumer_output_cotangent_operand_index
    consumer_operand = 1 - output_cotangent_operand
    consumer = ShuttleOperationRecord(
        boundary.consumer_weight_cotangent.name,
        dict(
            boundary.consumer_weight_cotangent.attributes,
            indexing_maps=[
                consumer_pullback_maps[2],
                consumer_pullback_maps[consumer_operand],
                consumer_pullback_maps[output_cotangent_operand],
            ],
        ),
        boundary.consumer_weight_cotangent.operands,
        boundary.consumer_weight_cotangent.results,
        boundary.consumer_weight_cotangent.regions,
    )
    geometry = _saved_state_reverse_geometry(boundary, producer_maps, consumer_pullback_maps)
    finalizer_expression = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input(_CONSUMER_ACCUMULATOR_INPUT),
        scalar_input(_NORMALIZER_INPUT),
    )
    buffers = tuple(
        sorted(
            {
                buffer.ordinal: buffer
                for buffer in (
                    *boundary.inputs,
                    *boundary.outputs,
                    boundary.output_dependent_state_output,
                )
            }.values(),
            key=lambda buffer: buffer.ordinal,
        )
    )
    return PhysicalNormalizedExpContractRegion(
        producer=producer,
        score_program=(boundary.producer,),
        maximum=boundary.output_dependent_fold,
        exponential=boundary.output_dependent_fold,
        normalizer=boundary.output_dependent_fold,
        consumer=consumer,
        finalizer=boundary.output_dependent_fold,
        producer_left_argument=_required_source_argument(boundary.producer_owner_input),
        producer_right_argument=_required_source_argument(boundary.producer_stream_input),
        consumer_argument=_required_source_argument(boundary.consumer_input),
        output_result=0,
        public_output_dimensions=contract_indexing_dimensions(consumer)[2],
        producer_to_consumer_dimensions=tuple(
            zip(contract_indexing_dimensions(producer)[2], contract_indexing_dimensions(consumer)[0], strict=True)
        ),
        geometry=geometry,
        score_expression=boundary.score_expression,
        finalizer_expression=finalizer_expression,
        domain_predicate=boundary.domain_predicate,
        buffers=buffers,
    )


def _saved_state_reverse_geometry(
    boundary: PhysicalSavedStateReverseBoundary,
    producer_maps: tuple[dict[str, Any], ...],
    consumer_pullback_maps: tuple[dict[str, Any], ...],
) -> NormalizedExpContractGeometry:
    owner_operand = boundary.producer_owner_operand_index
    stream_operand = 1 - owner_operand
    owner_map = producer_maps[owner_operand]
    stream_map = producer_maps[stream_operand]
    score_map = producer_maps[2]
    owner_dimensions = _direct_dimensions(owner_map)
    stream_dimensions = _direct_dimensions(stream_map)
    score_dimensions = _direct_dimensions(score_map)
    iterator_kinds = boundary.producer.attributes.get("iterator_kinds")
    if not isinstance(iterator_kinds, list) or iterator_kinds.count("reduction") != 1:
        raise ValueError("saved-state producer Contract must expose one reduction iterator")
    producer_feature_dimension = iterator_kinds.index("reduction")
    common_dimensions = tuple(
        dimension for dimension in score_dimensions if dimension in owner_dimensions and dimension in stream_dimensions
    )
    owner_only_dimensions = tuple(
        dimension
        for dimension in score_dimensions
        if dimension in owner_dimensions and dimension not in stream_dimensions
    )
    streamed_dimensions = tuple(
        dimension
        for dimension in score_dimensions
        if dimension in stream_dimensions and dimension not in owner_dimensions
    )
    if not common_dimensions or not owner_only_dimensions or len(streamed_dimensions) != 1:
        raise ValueError("saved-state producer Contract does not expose owner, stream, and shared relation axes")
    owner_dimension = owner_only_dimensions[-1]
    relation_group_dimensions = owner_only_dimensions[:-1]
    streamed_dimension = streamed_dimensions[0]

    consumer_iterator_kinds = boundary.consumer_weight_cotangent.attributes.get("iterator_kinds")
    if not isinstance(consumer_iterator_kinds, list) or consumer_iterator_kinds.count("reduction") != 1:
        raise ValueError("saved-state consumer-weight Contract must expose one reduction iterator")
    consumer_feature_dimension = consumer_iterator_kinds.index("reduction")
    output_cotangent_map = consumer_pullback_maps[boundary.consumer_output_cotangent_operand_index]
    consumer_operand_map = consumer_pullback_maps[1 - boundary.consumer_output_cotangent_operand_index]
    if _direct_dimensions(consumer_pullback_maps[2]) != score_dimensions:
        raise ValueError("consumer-weight Contract result relation must match the producer score relation")
    output_feature_extent = _extent_for_dimension(
        boundary.finalizer_cotangent_input.type.shape,
        output_cotangent_map,
        consumer_feature_dimension,
    )
    consumer_feature_extent = _extent_for_dimension(
        boundary.consumer_input.type.shape,
        consumer_operand_map,
        consumer_feature_dimension,
    )
    if output_feature_extent != consumer_feature_extent:
        raise ValueError("consumer-weight Contract operands disagree on the consumer feature extent")

    return NormalizedExpContractGeometry(
        batch_shape=tuple(
            _extent_for_dimension(boundary.producer_owner_input.type.shape, owner_map, dimension)
            for dimension in common_dimensions[:-1]
        ),
        shared_relation_extent=_extent_for_dimension(
            boundary.producer_owner_input.type.shape,
            owner_map,
            common_dimensions[-1],
        ),
        relation_group_shape=tuple(
            _extent_for_dimension(boundary.producer_owner_input.type.shape, owner_map, dimension)
            for dimension in relation_group_dimensions
        ),
        owner_extent=_extent_for_dimension(
            boundary.producer_owner_input.type.shape,
            owner_map,
            owner_dimension,
        ),
        streamed_extent=_extent_for_dimension(
            boundary.producer_stream_input.type.shape,
            stream_map,
            streamed_dimension,
        ),
        producer_feature_extent=_extent_for_dimension(
            boundary.producer_owner_input.type.shape,
            owner_map,
            producer_feature_dimension,
        ),
        consumer_feature_extent=consumer_feature_extent,
        batch_dimensions=common_dimensions[:-1],
        shared_relation_dimension=common_dimensions[-1],
        relation_group_dimensions=relation_group_dimensions,
        owner_dimension=owner_dimension,
        streamed_dimension=streamed_dimension,
        producer_feature_dimension=producer_feature_dimension,
        normalized_dimension=streamed_dimension,
        consumer_feature_dimension=consumer_feature_dimension,
    )


def physical_normalized_exp_contract_pullback(
    request: GpuPhysicalArtifactRequest,
) -> PhysicalNormalizedExpContractPullback:
    """Inspect the historical combined forward/pullback benchmark boundary.

    This remains available to the focused reference tests only. The physical
    provider must not infer a pullback from state flags or task topology.
    """
    forward = physical_normalized_exp_contract_region(request)
    buffers = forward.buffers
    tasks = source_bound_tasks(request)
    forward_task = _single_task(
        tasks,
        lambda task: bool(task.state_outputs) and not task.state_inputs and len(task.reduction_dimensions) == 1,
        "producer-state",
    )
    query_task = _single_task(
        tasks,
        lambda task: bool(task.state_inputs) and bool(task.state_outputs) and len(task.reduction_dimensions) == 1,
        "owner-adjoint",
    )
    key_value_task = _single_task(
        tasks,
        lambda task: bool(task.state_inputs) and not task.state_outputs and len(task.reduction_dimensions) >= 2,
        "streamed-adjoint",
    )
    if len(forward.public_inputs) != 4 or len(forward.public_outputs) != 4:
        raise ValueError("current SM90 value-and-pullback boundary requires 4 inputs and 4 outputs")
    state_buffers = tuple(buffer for buffer in buffers if buffer.storage.endswith("<temporary>"))
    if len(state_buffers) != 3 or any(buffer.type.element_type != "f32" for buffer in state_buffers):
        raise ValueError("current SM90 value-and-pullback requires exactly three source-derived FP32 states")
    if len(forward_task.state_outputs) != 2 or len(query_task.state_outputs) != 1:
        raise ValueError("current SM90 schedule requires two forward states and one output-dependent state")
    if set(query_task.state_inputs) != set(forward_task.state_outputs):
        raise ValueError("owner-adjoint task must consume the producer Fold state")
    if set(key_value_task.state_inputs) != set((*forward_task.state_outputs, *query_task.state_outputs)):
        raise ValueError("streamed-adjoint task must consume all sufficient Fold state")
    if query_task.dependencies != (forward_task.ordinal,) or set(key_value_task.dependencies) != {
        forward_task.ordinal,
        query_task.ordinal,
    }:
        raise ValueError("stateful Contract/Fold pullback dependencies disagree with state flow")
    if forward_task.domain_shape != query_task.domain_shape:
        raise ValueError("producer-state and owner-adjoint tasks must share one contraction domain")
    if forward_task.stream_dimensions != query_task.stream_dimensions:
        raise ValueError("producer-state and owner-adjoint tasks must share streamed dimensions")

    forward_output_buffer = _single_data_output(forward_task, buffers, "producer-state")
    query_cotangent_buffer = _single_data_output(query_task, buffers, "owner-adjoint")
    key_value_outputs = tuple(
        ordinal for ordinal in key_value_task.outputs if ordinal not in key_value_task.state_outputs
    )
    if len(key_value_outputs) != 2 or any(not buffers[ordinal].live_out for ordinal in key_value_outputs):
        raise ValueError("streamed-adjoint task must produce two terminal live-out Contracts")
    output_cotangent = tuple(
        buffer
        for buffer in forward.public_inputs
        if buffer.ordinal not in forward_task.inputs
        and buffer.ordinal in query_task.inputs
        and buffer.ordinal in key_value_task.inputs
        and buffer.type == buffers[forward_output_buffer].type
    )
    if len(output_cotangent) != 1:
        raise ValueError("could not derive the output cotangent from task dependencies and boundary types")

    key_cotangent_buffer, value_cotangent_buffer = _key_value_cotangent_buffers(request, forward, key_value_outputs)

    geometry = forward.geometry
    owner_state_shape = (
        *geometry.batch_shape,
        geometry.shared_relation_extent,
        *geometry.relation_group_shape,
        geometry.owner_extent,
    )
    for ordinal in (*forward_task.state_outputs, *query_task.state_outputs):
        if buffers[ordinal].type.element_type != "f32" or buffers[ordinal].type.shape != owner_state_shape:
            raise ValueError(f"source state buffer {ordinal} does not match the relation-derived owner domain")
    if output_cotangent[0].type != buffers[forward_output_buffer].type:
        raise ValueError("output cotangent and output boundaries disagree")
    producer_left_buffer = forward.public_inputs[forward.producer_left_argument]
    producer_right_buffer = forward.public_inputs[forward.producer_right_argument]
    consumer_buffer = forward.public_inputs[forward.consumer_argument]
    if buffers[query_cotangent_buffer].type != producer_left_buffer.type:
        raise ValueError("output cotangent/output and query/query-cotangent boundaries disagree")
    if (
        buffers[key_cotangent_buffer].type != producer_right_buffer.type
        or buffers[value_cotangent_buffer].type != consumer_buffer.type
    ):
        raise ValueError("key/value terminal cotangents disagree with their operands")

    return PhysicalNormalizedExpContractPullback(
        forward=forward,
        tasks=tasks,
        output_cotangent_argument=_required_source_argument(output_cotangent[0]),
        forward_output_buffer=forward_output_buffer,
        query_cotangent_buffer=query_cotangent_buffer,
        key_cotangent_buffer=key_cotangent_buffer,
        value_cotangent_buffer=value_cotangent_buffer,
        maximum_state_buffer=forward_task.state_outputs[0],
        normalizer_state_buffer=forward_task.state_outputs[1],
        output_dependent_state_buffer=query_task.state_outputs[0],
        score_map_vjp_expression=scalar_expression_vjp(
            forward.score_expression,
            input_name=_SCORE_INPUT,
            cotangent_name=_MAPPED_SCORE_COTANGENT_INPUT,
        ),
        normalized_exp_pullback_expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("normalized_exp_contract.normalized_weight"),
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input("normalized_exp_contract.weight_cotangent"),
                scalar_input("normalized_exp_contract.output_dot_cotangent"),
            ),
        ),
    )


def _single_task(
    tasks: tuple[SourceBoundTask, ...],
    predicate: Any,
    label: str,
) -> SourceBoundTask:
    matches = tuple(task for task in tasks if predicate(task))
    if len(matches) != 1:
        raise ValueError(f"expected one structural {label} task, found {len(matches)}")
    return matches[0]


def _single_data_output(
    task: SourceBoundTask,
    buffers: tuple[SourceBoundBuffer, ...],
    label: str,
) -> int:
    outputs = tuple(ordinal for ordinal in task.outputs if ordinal not in task.state_outputs)
    if len(outputs) != 1 or not buffers[outputs[0]].live_out:
        raise ValueError(f"structural {label} task must have one non-state live output")
    return outputs[0]


def _key_value_cotangent_buffers(
    request: GpuPhysicalArtifactRequest,
    forward: PhysicalNormalizedExpContractRegion,
    candidate_buffers: tuple[int, int],
) -> tuple[int, int]:
    if len(request.region_result_indices) != len(forward.buffers):
        raise ValueError("physical source buffers do not have complete region-result identities")
    body = request.region.body()
    if body.operations[-1].name != "shuttle.yield":
        raise ValueError("normalized-exp pullback region has no source-bound yield")
    producer_by_value = {result.id: operation for operation in body.operations for result in operation.results}
    argument_index = {argument.id: index for index, argument in enumerate(body.arguments)}
    by_direct_owner_operand: dict[bool, int] = {}
    for buffer_ordinal in candidate_buffers:
        contract = _nearest_contract(
            _terminal_result_value(request, forward.buffers, buffer_ordinal),
            producer_by_value,
        )
        direct_roots = tuple(
            _argument_roots(operand, producer_by_value, argument_index) for operand in contract.operands
        )
        has_direct_owner_operand = frozenset((forward.producer_left_argument,)) in direct_roots
        if has_direct_owner_operand in by_direct_owner_operand:
            raise ValueError("streamed terminal Contracts do not have distinct direct operand relations")
        by_direct_owner_operand[has_direct_owner_operand] = buffer_ordinal
    if set(by_direct_owner_operand) != {False, True}:
        raise ValueError("could not distinguish streamed terminal Contracts by their direct operand relations")
    return by_direct_owner_operand[True], by_direct_owner_operand[False]


def _declared_normalized_exp_contract_pullback(
    request: GpuPhysicalArtifactRequest,
    capability: ShuttleNormalizedExpContractPullbackRecord,
) -> PhysicalNormalizedExpContractPullback:
    """Bind a planner-declared pullback without recovering roles from topology."""
    try:
        forward_capability = request.normalized_exp_contract_regions[capability.forward_region_ordinal]
    except IndexError as error:
        raise ValueError("planner-declared pullback references a stale forward region") from error
    forward = physical_normalized_exp_contract_region(request, forward_capability)
    operations = tuple(operation for operation in request.region.body().operations if operation.results)
    tasks = source_bound_tasks(request)
    buffers = forward.buffers

    def operation(ordinal: int, expected_name: str, role: str) -> ShuttleOperationRecord:
        try:
            candidate = operations[ordinal]
        except IndexError as error:
            raise ValueError(f"planner-declared {role} operation ordinal is stale") from error
        if candidate.name != expected_name:
            raise ValueError(f"planner-declared {role} does not reference {expected_name}")
        return candidate

    def task(ordinal: int, role: str) -> SourceBoundTask:
        try:
            return tasks[ordinal]
        except IndexError as error:
            raise ValueError(f"planner-declared {role} task ordinal is stale") from error

    def buffer(ordinal: int, role: str) -> SourceBoundBuffer:
        try:
            return buffers[ordinal]
        except IndexError as error:
            raise ValueError(f"planner-declared {role} buffer ordinal is stale") from error

    operation(capability.normalized_output_operation, "shuttle.map", "normalized output")
    operation(capability.post_normalization_vjp_operation, "shuttle.map", "finalizer VJP")
    operation(capability.output_product_operation, "shuttle.map", "output product")
    operation(
        capability.output_dependent_fold_operation,
        "shuttle.fold",
        "output-dependent Fold",
    )
    operation(
        capability.consumer_weight_cotangent_operation,
        "shuttle.contract",
        "consumer-weight cotangent",
    )
    operation(capability.normalized_weight_operation, "shuttle.map", "normalized weight")
    normalized_exp_pullback = operation(
        capability.normalized_exp_pullback_operation,
        "shuttle.map",
        "normalized-exp pullback",
    )
    score_map_vjp = operation(capability.score_map_vjp_operation, "shuttle.map", "score-Map VJP")
    terminal_contracts = tuple(
        operation(ordinal, "shuttle.contract", f"terminal gradient {index}")
        for index, ordinal in enumerate(capability.terminal_contract_operations)
    )
    if len(terminal_contracts) != 3 or len(set(capability.terminal_contract_operations)) != 3:
        raise ValueError("planner-declared pullback must bind three distinct terminal Contracts")

    forward_task = task(capability.forward_task, "forward")
    owner_task = task(capability.owner_pullback_task, "owner pullback")
    streamed_task = task(capability.streamed_pullback_task, "streamed pullback")
    task_ordinals = {forward_task.ordinal, owner_task.ordinal, streamed_task.ordinal}
    if len(task_ordinals) != 3 or capability.objective_task in task_ordinals:
        raise ValueError("planner-declared pullback tasks must have distinct ownership")
    if capability.objective_task is not None:
        task(capability.objective_task, "objective")

    maximum_state = buffer(capability.maximum_state_buffer, "maximum state")
    normalizer_state = buffer(capability.normalizer_state_buffer, "normalizer state")
    output_dependent_state = buffer(capability.output_dependent_state_buffer, "output-dependent state")
    output_cotangent = buffer(capability.output_cotangent_buffer, "output cotangent")
    terminal_buffers = tuple(
        buffer(ordinal, f"terminal cotangent {index}")
        for index, ordinal in enumerate(capability.terminal_cotangent_buffers)
    )
    if len(terminal_buffers) != 3 or len({item.ordinal for item in terminal_buffers}) != 3:
        raise ValueError("planner-declared pullback must bind three distinct terminal cotangent buffers")
    if capability.forward_result_buffer is None:
        raise ValueError("current SM90 value-and-pullback artifact requires a declared forward result buffer")
    forward_result = buffer(capability.forward_result_buffer, "forward result")

    if set((maximum_state.ordinal, normalizer_state.ordinal)) != set(forward_task.state_outputs):
        raise ValueError("planner-declared forward Fold states disagree with the forward task")
    if not {maximum_state.ordinal, normalizer_state.ordinal}.issubset(owner_task.state_inputs):
        raise ValueError("planner-declared owner pullback task does not consume the forward Fold states")
    if output_dependent_state.ordinal not in owner_task.state_outputs:
        raise ValueError("planner-declared output-dependent state is not produced by the owner pullback task")
    if not {
        maximum_state.ordinal,
        normalizer_state.ordinal,
        output_dependent_state.ordinal,
    }.issubset(streamed_task.state_inputs):
        raise ValueError("planner-declared streamed pullback task does not consume the sufficient state")
    if forward_result.producer != forward_task.ordinal:
        raise ValueError("planner-declared forward result is not produced by the forward task")
    if terminal_buffers[0].producer != owner_task.ordinal or any(
        item.producer != streamed_task.ordinal for item in terminal_buffers[1:]
    ):
        raise ValueError("planner-declared terminal cotangent buffers disagree with pullback task ownership")
    if output_cotangent.source_argument is None or not output_cotangent.live_in:
        raise ValueError("planner-declared output cotangent is not a public source argument")
    if output_cotangent.ordinal not in owner_task.inputs or output_cotangent.ordinal not in streamed_task.inputs:
        raise ValueError("planner-declared pullback tasks do not consume the output cotangent")

    owner_sources = set(owner_task.source_operations)
    streamed_sources = set(streamed_task.source_operations)
    if capability.terminal_contract_operations[0] not in owner_sources or not set(
        capability.terminal_contract_operations[1:]
    ).issubset(streamed_sources):
        raise ValueError("planner-declared terminal Contracts are absent from their pullback tasks")
    if capability.output_dependent_fold_operation not in owner_sources:
        raise ValueError("planner-declared output-dependent Fold is absent from the owner pullback task")

    producer_by_value = {result.id: candidate for candidate in operations for result in candidate.results}
    mapped_cotangent_inputs = tuple(
        operand
        for operand in score_map_vjp.operands
        if normalized_exp_pullback in _ancestor_operations(operand, producer_by_value)
    )
    if len(mapped_cotangent_inputs) != 1:
        raise ValueError("planner-declared score-Map VJP does not consume one normalized-exp cotangent")
    score_map_vjp_expression = _scalar_expression_from_graph(
        _single_result(score_map_vjp),
        {
            mapped_cotangent_inputs[0]: _MAPPED_SCORE_COTANGENT_INPUT,
            _single_result(forward.producer): _SCORE_INPUT,
        },
        producer_by_value,
        None,
    )
    expected_score_vjp = scalar_expression_vjp(
        forward.score_expression,
        input_name=_SCORE_INPUT,
        cotangent_name=_MAPPED_SCORE_COTANGENT_INPUT,
    )
    if score_map_vjp_expression != expected_score_vjp:
        raise ValueError("planner-declared score-Map VJP disagrees with generic scalar differentiation")
    normalized_exp_pullback_expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input("normalized_exp_contract.normalized_weight"),
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_input("normalized_exp_contract.weight_cotangent"),
            scalar_input("normalized_exp_contract.output_dot_cotangent"),
        ),
    )

    geometry = forward.geometry
    owner_state_shape = (
        *geometry.batch_shape,
        geometry.shared_relation_extent,
        *geometry.relation_group_shape,
        geometry.owner_extent,
    )
    if any(
        state.type.element_type != "f32" or state.type.shape != owner_state_shape
        for state in (maximum_state, normalizer_state, output_dependent_state)
    ):
        raise ValueError("planner-declared pullback states disagree with the relation-derived owner domain")
    query = forward.public_inputs[forward.producer_left_argument]
    key = forward.public_inputs[forward.producer_right_argument]
    value = forward.public_inputs[forward.consumer_argument]
    if (
        terminal_buffers[0].type != query.type
        or terminal_buffers[1].type != key.type
        or terminal_buffers[2].type != value.type
    ):
        raise ValueError("planner-declared terminal cotangents disagree with their Contract operands")
    if forward_result.type != output_cotangent.type:
        raise ValueError("planner-declared forward result and output cotangent types disagree")
    return PhysicalNormalizedExpContractPullback(
        forward=forward,
        tasks=tasks,
        output_cotangent_argument=output_cotangent.source_argument,
        forward_output_buffer=forward_result.ordinal,
        query_cotangent_buffer=terminal_buffers[0].ordinal,
        key_cotangent_buffer=terminal_buffers[1].ordinal,
        value_cotangent_buffer=terminal_buffers[2].ordinal,
        maximum_state_buffer=maximum_state.ordinal,
        normalizer_state_buffer=normalizer_state.ordinal,
        output_dependent_state_buffer=output_dependent_state.ordinal,
        score_map_vjp_expression=score_map_vjp_expression,
        normalized_exp_pullback_expression=normalized_exp_pullback_expression,
    )


def physical_normalized_exp_contract_match(
    request: GpuPhysicalArtifactRequest,
) -> PhysicalNormalizedExpContractMatch | None:
    """Classify a request through typed algebra and task-plan extraction.

    ``None`` means another provider may handle the request. Once the algebra
    matches this task family, malformed schedules and unsupported physical
    policies remain errors at this boundary.
    """
    if request.target != "sm_90":
        return None
    if not request.normalized_exp_contract_regions:
        return None
    if len(request.normalized_exp_contract_regions) != 1:
        raise ValueError(
            "one physical artifact request currently requires exactly one planner-declared "
            "normalized-exp Contract region"
        )
    try:
        region = physical_normalized_exp_contract_region(
            request,
            request.normalized_exp_contract_regions[0],
        )
    except PhysicalNormalizedExpContractNotApplicable:
        return None
    if request.normalized_exp_contract_pullbacks:
        if len(request.normalized_exp_contract_pullbacks) != 1:
            raise ValueError("one physical artifact request currently requires exactly one declared pullback")
        return PhysicalNormalizedExpPullbackMatch(
            _declared_normalized_exp_contract_pullback(
                request,
                request.normalized_exp_contract_pullbacks[0],
            )
        )
    tasks = source_bound_tasks(request)
    if any(task.state_inputs or task.state_outputs for task in tasks):
        raise ValueError(
            "stateful normalized-exp compilation requires a planner-declared pullback capability; "
            "the legacy structural reverse adapter is diagnostic-only"
        )
    return PhysicalNormalizedExpForwardMatch(region)


def physical_normalized_exp_contract_region(
    request: GpuPhysicalArtifactRequest,
    capability: ShuttleNormalizedExpContractRegionRecord | None = None,
) -> PhysicalNormalizedExpContractRegion:
    """Bind one planner-declared normalized-exp Contract region to source IR."""
    if request.target != "sm_90":
        raise ValueError("the current normalized-exponential physical template requires sm_90")
    if capability is None:
        if not request.normalized_exp_contract_regions:
            raise PhysicalNormalizedExpContractNotApplicable(
                "request has no planner-declared normalized-exp Contract capability"
            )
        if len(request.normalized_exp_contract_regions) != 1:
            raise ValueError("select one planner-declared normalized-exp Contract region explicitly")
        capability = request.normalized_exp_contract_regions[0]
    body = request.region.body()
    operations = tuple(operation for operation in body.operations if operation.results)
    producer_by_value = {result.id: operation for operation in operations for result in operation.results}
    argument_index = {argument.id: index for index, argument in enumerate(body.arguments)}

    def result(operation_ordinal: int, result_ordinal: int) -> tuple[ShuttleOperationRecord, int]:
        operation = operations[operation_ordinal]
        return operation, operation.results[result_ordinal].id

    try:
        producer = operations[capability.producer_operation]
        consumer = operations[capability.consumer_operation]
        fold_operations = tuple(operations[index] for index in capability.fold_operations)
        maximum, _ = result(capability.maximum_operation, capability.maximum_result)
        normalizer, normalizer_value = result(capability.normalizer_operation, capability.normalizer_result)
        exponential, normalized_weight_value = result(
            capability.normalized_weight_operation,
            capability.normalized_weight_result,
        )
        finalizer, finalizer_value = result(capability.finalizer_operation, capability.finalizer_result)
        _, score_value = result(capability.score_operation, capability.score_result)
    except (IndexError, KeyError) as error:
        raise ValueError("planner-declared normalized-exp source ordinal is stale") from error
    if producer.name != "shuttle.contract" or consumer.name != "shuttle.contract":
        raise ValueError("planner-declared normalized-exp Contract role does not reference a Contract")
    if any(operation.name not in {"shuttle.fold", "shuttle.map", "shuttle.relation"} for operation in fold_operations):
        raise ValueError("planner-declared normalized-exp Fold program contains an unsupported capability")
    if maximum not in fold_operations or normalizer not in fold_operations:
        raise ValueError("planner-declared normalized-exp state values do not belong to its Fold program")
    if capability.normalized_operand >= len(consumer.operands):
        raise ValueError("planner-declared normalized-exp consumer operand is outside the Contract")
    if consumer.operands[capability.normalized_operand] != normalized_weight_value:
        raise ValueError("planner-declared normalized weight does not bind the declared consumer operand")
    binding = _NormalizedExpForwardBinding(
        producer=producer,
        maximum=maximum,
        exponential=exponential,
        normalizer=normalizer,
        consumer=consumer,
        normalized_operand=capability.normalized_operand,
        score_value=score_value,
        normalizer_value=normalizer_value,
    )
    if _mlir_attribute(request.region.attributes.get("policy"), "region policy") != "#shuttle.policy<fast>":
        raise ValueError("online normalized-exponential contraction requires FAST numerical policy")
    producer = binding.producer
    maximum = binding.maximum
    exponential = binding.exponential
    normalizer = binding.normalizer
    consumer = binding.consumer

    if len(producer.operands) != 2 or len(consumer.operands) != 2:
        raise ValueError("SM90 normalized-exponential lowering requires two-input Contracts")
    producer_roots = tuple(_argument_roots(operand, producer_by_value, argument_index) for operand in producer.operands)
    if any(len(roots) != 1 for roots in producer_roots):
        raise ValueError("producer Contract operands must each derive from one region argument")
    normalized_operand = binding.normalized_operand
    consumer_operand = consumer.operands[1 - normalized_operand]
    value_roots = _argument_roots(consumer_operand, producer_by_value, argument_index)
    if len(value_roots) != 1:
        raise ValueError("weighted Contract operand must derive from one region argument")
    consumer_argument = next(iter(value_roots))
    producer_arguments = tuple(next(iter(roots)) for roots in producer_roots)
    if len({*producer_arguments, consumer_argument}) != 3:
        raise ValueError("normalized-exponential Contracts require three distinct source operands")

    yield_operation = body.operations[-1]
    if yield_operation.name != "shuttle.yield":
        raise ValueError("Shuttle region is missing its source-bound yield")
    if capability.output_result is not None:
        if capability.output_result >= len(yield_operation.operands):
            raise ValueError("planner-declared normalized-exp result ordinal is stale")
        if yield_operation.operands[capability.output_result] != finalizer_value:
            raise ValueError("planner-declared normalized-exp finalizer does not bind its result")
    if finalizer.name != "shuttle.map" or consumer not in _ancestor_operations(
        finalizer_value,
        producer_by_value,
    ):
        raise ValueError("planner-declared normalized-exp finalizer does not consume the weighted Contract")

    consumer_result_dimensions = contract_indexing_dimensions(consumer)[2]
    public_output_lineage = _tensor_dimension_lineage(
        _single_result(finalizer),
        root_value=_single_result(consumer),
        root_dimensions=consumer_result_dimensions,
        producer_by_value=producer_by_value,
    )
    if len(public_output_lineage) != len(finalizer.results[0].type.shape) or any(
        dimension is None for dimension in public_output_lineage
    ):
        raise ValueError("finalizer output must preserve every weighted Contract result dimension")
    public_output_dimensions = tuple(dimension for dimension in public_output_lineage if dimension is not None)
    if len(set(public_output_dimensions)) != len(public_output_dimensions) or set(public_output_dimensions) != set(
        consumer_result_dimensions
    ):
        raise ValueError("finalizer output must be a permutation of the weighted Contract result dimensions")

    geometry, orientation, producer_to_consumer_dimensions = _contraction_geometry(
        producer,
        consumer,
        body.arguments[producer_arguments[0]],
        body.arguments[producer_arguments[1]],
        body.arguments[consumer_argument],
        normalized_operand,
        producer_by_value,
        public_output_dimensions,
    )
    producer_left_argument = producer_arguments[orientation.owner_operand]
    producer_right_argument = producer_arguments[orientation.stream_operand]
    score_ancestors = _ancestor_operations(binding.score_value, producer_by_value)
    domain_predicate = _domain_predicate(score_ancestors, producer, producer_by_value)
    score_expression = _scalar_expression_from_graph(
        binding.score_value,
        {_single_result(producer): _SCORE_INPUT},
        producer_by_value,
        domain_predicate,
    )
    score_expression = _without_domain_predicate(score_expression)
    if scalar_expression_inputs(score_expression) != {_SCORE_INPUT}:
        raise ValueError("score scalar AST must be closed over exactly the producer Contract result")
    finalizer_expression = _scalar_expression_from_graph(
        _single_result(finalizer),
        {
            _single_result(consumer): _CONSUMER_ACCUMULATOR_INPUT,
            binding.normalizer_value: _NORMALIZER_INPUT,
        },
        producer_by_value,
        None,
    )
    finalizer_inputs = scalar_expression_inputs(finalizer_expression)
    if _CONSUMER_ACCUMULATOR_INPUT not in finalizer_inputs or not finalizer_inputs <= {
        _CONSUMER_ACCUMULATOR_INPUT,
        _NORMALIZER_INPUT,
    }:
        raise ValueError(
            "finalizer scalar AST must be closed over the weighted Contract and optional normalized-exp Fold state"
        )

    return PhysicalNormalizedExpContractRegion(
        producer=producer,
        score_program=tuple(reversed(score_ancestors)),
        maximum=maximum,
        exponential=exponential,
        normalizer=normalizer,
        consumer=consumer,
        finalizer=finalizer,
        producer_left_argument=producer_left_argument,
        producer_right_argument=producer_right_argument,
        consumer_argument=consumer_argument,
        output_result=capability.output_result,
        public_output_dimensions=public_output_dimensions,
        producer_to_consumer_dimensions=producer_to_consumer_dimensions,
        geometry=geometry,
        score_expression=score_expression,
        finalizer_expression=finalizer_expression,
        domain_predicate=domain_predicate,
        buffers=source_bound_buffers(request),
    )


def _bounded_diagnostic(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 12]}...+{len(value) - limit}"


def _affine_maps_diagnostic(maps: tuple[dict[str, Any], ...]) -> str:
    records = tuple(
        f"map{index}(direct={_direct_dimensions_diagnostic(affine_map)},raw="
        f"{_bounded_diagnostic(json.dumps(affine_map, sort_keys=True, separators=(',', ':')), 1_024)})"
        for index, affine_map in enumerate(maps)
    )
    return _bounded_diagnostic(repr(records), 4_096)


def _contract_maps_diagnostic(operation: ShuttleOperationRecord) -> str:
    encoded = operation.attributes.get("indexing_maps")
    return _bounded_diagnostic(json.dumps(encoded, sort_keys=True, separators=(",", ":")), 4_096)


def _direct_dimensions_diagnostic(affine_map: dict[str, Any]) -> str:
    try:
        return repr(_direct_dimensions(affine_map))
    except ValueError as error:
        return f"error({error})"


def _geometry_map_parse_diagnostic(
    producer: ShuttleOperationRecord,
    consumer: ShuttleOperationRecord,
) -> str:
    return _bounded_diagnostic(
        "producer_raw=" f"{_contract_maps_diagnostic(producer)};consumer_raw={_contract_maps_diagnostic(consumer)}",
        8_192,
    )


def _compact_projection_diagnostic(dimensions: tuple[int, ...], projected: tuple[int, ...]) -> str:
    accepted = _is_contiguous_projection(dimensions, projected)
    return f"{'accepted' if accepted else 'rejected'}(dimensions={dimensions},projected={projected})"


def _contraction_geometry(
    producer: ShuttleOperationRecord,
    consumer: ShuttleOperationRecord,
    producer_left: ShuttleValueRecord,
    producer_right: ShuttleValueRecord,
    consumer_operand: ShuttleValueRecord,
    normalized_operand: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
    public_output_dimensions: tuple[int, ...] | None = None,
) -> tuple[
    NormalizedExpContractGeometry,
    _ProducerOrientation,
    tuple[tuple[int, int], ...],
]:
    try:
        producer_maps = _affine_maps(producer)
        consumer_maps = _affine_maps(consumer)
    except ValueError as error:
        diagnostic = _geometry_map_parse_diagnostic(producer, consumer)
        raise ValueError(f"Contract affine map extraction failed: {error}; {diagnostic}") from error
    if len(producer_maps) != 3 or len(consumer_maps) != 3:
        raise ValueError(
            "two-input Contracts require two operand maps and one result map; "
            f"producer={_affine_maps_diagnostic(producer_maps)};consumer={_affine_maps_diagnostic(consumer_maps)}"
        )
    try:
        operand_dimensions = tuple(_direct_dimensions(affine_map) for affine_map in producer_maps[:2])
        score_dimensions = _direct_dimensions(producer_maps[2])
        consumer_dimensions = tuple(_direct_dimensions(affine_map) for affine_map in consumer_maps)
    except ValueError as error:
        raise ValueError(
            f"Contract affine map dimension extraction failed: {error}; "
            f"producer={_affine_maps_diagnostic(producer_maps)};consumer={_affine_maps_diagnostic(consumer_maps)}"
        ) from error
    iterator_kinds = producer.attributes.get("iterator_kinds")
    if not isinstance(iterator_kinds, list) or iterator_kinds.count("reduction") != 1:
        raise ValueError("producer Contract must expose one reduction iterator")
    producer_feature_dimension = iterator_kinds.index("reduction")
    value_map = consumer_maps[1 - normalized_operand]
    value_dimensions = consumer_dimensions[1 - normalized_operand]
    result_dimensions = consumer_dimensions[2]
    normalized_dimensions = consumer_dimensions[normalized_operand]
    normalized_lineage = _tensor_dimension_lineage(
        consumer.operands[normalized_operand],
        root_value=_single_result(producer),
        root_dimensions=score_dimensions,
        producer_by_value=producer_by_value,
    )
    if len(normalized_lineage) != len(normalized_dimensions):
        raise ValueError("normalized score lineage rank does not match its weighted Contract relation")
    producer_to_consumer_dimensions = tuple(
        (producer_dimension, consumer_dimension)
        for producer_dimension, consumer_dimension in zip(normalized_lineage, normalized_dimensions, strict=True)
        if producer_dimension is not None
    )
    producer_to_consumer_dimension = dict(producer_to_consumer_dimensions)
    if len(producer_to_consumer_dimension) != len(score_dimensions):
        raise ValueError("normalized score relation must project each producer dimension exactly once")
    consumer_feature_dimensions = tuple(
        dimension
        for dimension in result_dimensions
        if dimension in value_dimensions and dimension not in normalized_dimensions
    )
    normalized_dimensions_only = tuple(
        dimension
        for dimension in normalized_dimensions
        if dimension in value_dimensions and dimension not in result_dimensions
    )
    if len(consumer_feature_dimensions) != 1 or len(normalized_dimensions_only) != 1:
        raise ValueError("consumer Contract must expose distinct normalized and value feature dimensions")
    consumer_feature_dimension = consumer_feature_dimensions[0]
    consumer_feature_extent = _extent_for_dimension(consumer_operand.type.shape, value_map, consumer_feature_dimension)
    consumer_streamed_extent = _extent_for_dimension(
        consumer_operand.type.shape,
        value_map,
        normalized_dimensions_only[0],
    )

    producer_operands = (producer_left, producer_right)
    orientations: list[_ProducerOrientation] = []
    candidate_diagnostics: list[str] = []
    for owner_operand in range(2):
        stream_operand = 1 - owner_operand
        owner_dimensions = operand_dimensions[owner_operand]
        stream_dimensions = operand_dimensions[stream_operand]
        common_dimensions = tuple(
            dimension
            for dimension in owner_dimensions
            if dimension in score_dimensions and dimension in stream_dimensions
        )
        owner_only_dimensions = tuple(
            dimension
            for dimension in owner_dimensions
            if dimension in score_dimensions and dimension not in stream_dimensions
        )
        streamed_dimensions = tuple(
            dimension
            for dimension in score_dimensions
            if dimension in stream_dimensions and dimension not in owner_dimensions
        )
        streamed_extent: int | None = None
        streamed_extent_check = "not_attempted(requires_one_streamed_dimension)"
        if len(streamed_dimensions) == 1:
            streamed_extent = _extent_for_dimension(
                producer_operands[stream_operand].type.shape,
                producer_maps[stream_operand],
                streamed_dimensions[0],
            )
            streamed_extent_check = (
                f"{'accepted' if streamed_extent == consumer_streamed_extent else 'rejected'}"
                f"(producer={streamed_extent},consumer={consumer_streamed_extent})"
            )
        # The owner operand can contain both an independently tiled owner axis
        # and relation-group axes.  Their extents and position in the Contract
        # result do not distinguish those roles: q-major storage, for example,
        # projects ``q, batch, head, group, feature``.  The physical relation
        # layout does.  Its shared and group axes must be one compact projection
        # in both the owner input and the public output, leaving the owner axis
        # independently strided.  The weighted Contract may produce another
        # axis order before the finalizer's index relation.  Derive compactness
        # at the physical boundary instead of assigning roles by operand or
        # intermediate-result position.
        owner_axis_candidates = []
        split_diagnostics: list[str] = []
        if common_dimensions:
            shared_relation_dimension = common_dimensions[-1]
            for owner_dimension in owner_only_dimensions:
                relation_group_dimensions = tuple(
                    dimension for dimension in owner_only_dimensions if dimension != owner_dimension
                )
                packed_relation_dimensions = (shared_relation_dimension, *relation_group_dimensions)
                producer_projection = _compact_projection_diagnostic(
                    owner_dimensions,
                    packed_relation_dimensions,
                )
                try:
                    public_output_packed_dimensions = tuple(
                        producer_to_consumer_dimension[dimension] for dimension in packed_relation_dimensions
                    )
                except KeyError as error:
                    public_output_projection = f"rejected(unmapped_producer_dimension=d{error.args[0]})"
                    public_output_packed_dimensions = ()
                else:
                    public_output_projection = (
                        "not_checked(finalizer_lineage_unavailable)"
                        if public_output_dimensions is None
                        else _compact_projection_diagnostic(
                            public_output_dimensions,
                            public_output_packed_dimensions,
                        )
                    )
                split_diagnostics.append(
                    f"owner=d{owner_dimension},groups={relation_group_dimensions},packed={packed_relation_dimensions},"
                    f"producer_compact={producer_projection},"
                    f"public_output_packed={public_output_packed_dimensions},"
                    f"public_output_compact={public_output_projection}"
                )
                if not _is_contiguous_projection(owner_dimensions, packed_relation_dimensions):
                    continue
                if not public_output_packed_dimensions:
                    continue
                if public_output_dimensions is not None and not _is_contiguous_projection(
                    public_output_dimensions, public_output_packed_dimensions
                ):
                    continue
                owner_axis_candidates.append((owner_dimension, relation_group_dimensions))
        else:
            split_diagnostics.append("not_attempted(missing_shared_relation_dimension)")
        candidate_diagnostics.append(
            _bounded_diagnostic(
                f"owner_operand={owner_operand},stream_operand={stream_operand},common={common_dimensions},"
                f"owner_only={owner_only_dimensions},stream_only={streamed_dimensions},"
                f"streamed_extent_check={streamed_extent_check},splits=({';'.join(split_diagnostics)})",
                4_096,
            )
        )
        if (
            not common_dimensions
            or not owner_only_dimensions
            or len(streamed_dimensions) != 1
            or streamed_extent != consumer_streamed_extent
        ):
            continue
        if len(owner_axis_candidates) != 1:
            continue
        owner_dimension, relation_group_dimensions = owner_axis_candidates[0]
        orientations.append(
            _ProducerOrientation(
                owner_operand=owner_operand,
                stream_operand=stream_operand,
                common_dimensions=common_dimensions,
                relation_group_dimensions=relation_group_dimensions,
                owner_dimension=owner_dimension,
                streamed_dimension=streamed_dimensions[0],
            )
        )
    if not orientations:
        lineage = tuple(sorted(producer_to_consumer_dimension.items()))
        dimensions_diagnostic = _bounded_diagnostic(
            f"producer_dimensions=(operands={operand_dimensions},result={score_dimensions});"
            f"consumer_dimensions=(normalized={normalized_dimensions},value={value_dimensions},result={result_dimensions});"
            f"producer_to_consumer_lineage={lineage};"
            f"consumer_streamed=(dimension=d{normalized_dimensions_only[0]},extent={consumer_streamed_extent})",
            4_096,
        )
        candidates_diagnostic = _bounded_diagnostic(" | ".join(candidate_diagnostics), 8_192)
        maps_diagnostic = (
            f"producer_maps={_affine_maps_diagnostic(producer_maps)};"
            f"consumer_maps={_affine_maps_diagnostic(consumer_maps)}"
        )
        raise ValueError(
            "producer Contract does not expose owner, stream, and shared relation axes; "
            f"{dimensions_diagnostic};candidates=({candidates_diagnostic});{maps_diagnostic}"
        )
    if len(orientations) > 1:
        signatures = {
            (
                tuple(
                    _extent_for_dimension(
                        producer_operands[candidate.owner_operand].type.shape,
                        producer_maps[candidate.owner_operand],
                        dimension,
                    )
                    for dimension in candidate.common_dimensions
                ),
                tuple(
                    _extent_for_dimension(
                        producer_operands[candidate.owner_operand].type.shape,
                        producer_maps[candidate.owner_operand],
                        dimension,
                    )
                    for dimension in (*candidate.relation_group_dimensions, candidate.owner_dimension)
                ),
                _extent_for_dimension(
                    producer_operands[candidate.stream_operand].type.shape,
                    producer_maps[candidate.stream_operand],
                    candidate.streamed_dimension,
                ),
            )
            for candidate in orientations
        }
        if len(signatures) != 1:
            raise ValueError("producer Contract owner and stream relations are ambiguous")
        orientation = next(candidate for candidate in orientations if candidate.owner_operand == 0)
    else:
        orientation = orientations[0]

    owner_map = producer_maps[orientation.owner_operand]
    stream_map = producer_maps[orientation.stream_operand]
    owner_input = producer_operands[orientation.owner_operand]
    stream_input = producer_operands[orientation.stream_operand]
    common_dimensions = orientation.common_dimensions
    owner_dimension = orientation.owner_dimension
    group_dimensions = orientation.relation_group_dimensions
    streamed_dimension = orientation.streamed_dimension
    owner_extent = _extent_for_dimension(owner_input.type.shape, owner_map, owner_dimension)
    streamed_extent = _extent_for_dimension(stream_input.type.shape, stream_map, streamed_dimension)
    producer_feature_extent = _extent_for_dimension(owner_input.type.shape, owner_map, producer_feature_dimension)
    common_extents = tuple(
        _extent_for_dimension(owner_input.type.shape, owner_map, dimension) for dimension in common_dimensions
    )
    group_extents = tuple(
        _extent_for_dimension(owner_input.type.shape, owner_map, dimension) for dimension in group_dimensions
    )
    if common_extents[-1] <= 0:
        raise ValueError("shared relation axis must have positive extent")
    return (
        NormalizedExpContractGeometry(
            batch_shape=common_extents[:-1],
            shared_relation_extent=common_extents[-1],
            relation_group_shape=group_extents,
            owner_extent=owner_extent,
            streamed_extent=streamed_extent,
            producer_feature_extent=producer_feature_extent,
            consumer_feature_extent=consumer_feature_extent,
            batch_dimensions=common_dimensions[:-1],
            shared_relation_dimension=common_dimensions[-1],
            relation_group_dimensions=group_dimensions,
            owner_dimension=owner_dimension,
            streamed_dimension=streamed_dimension,
            producer_feature_dimension=producer_feature_dimension,
            normalized_dimension=normalized_dimensions_only[0],
            consumer_feature_dimension=consumer_feature_dimension,
        ),
        orientation,
        producer_to_consumer_dimensions,
    )


def _tensor_dimension_lineage(
    value: int,
    *,
    root_value: int,
    root_dimensions: tuple[int, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> tuple[int | None, ...]:
    """Trace root tensor axes through pointwise and structural score dataflow."""
    if value == root_value:
        return root_dimensions
    operation = producer_by_value.get(value)
    if operation is None or operation.name in ("shuttle.contract", "shuttle.fold"):
        return ()

    operand_lineages = tuple(
        _tensor_dimension_lineage(
            operand,
            root_value=root_value,
            root_dimensions=root_dimensions,
            producer_by_value=producer_by_value,
        )
        for operand in operation.operands
    )
    carrying = tuple(index for index, lineage in enumerate(operand_lineages) if lineage)
    if len(carrying) != 1:
        if not carrying:
            return ()
        raise ValueError("score index dataflow must have one producer-derived tensor operand")
    operand_index = carrying[0]
    operand_lineage = operand_lineages[operand_index]
    operand_operation = producer_by_value.get(operation.operands[operand_index])
    operand_type = (
        operand_operation.results[0].type
        if operand_operation is not None and len(operand_operation.results) == 1
        else None
    )
    if operand_type is None:
        raise ValueError("score index dataflow operand type is unavailable")
    if len(operation.results) != 1:
        raise ValueError("score index dataflow operation must produce one tensor")

    if operation.name == "shuttle.map":
        semantics = _map_semantics(operation)
        if semantics == "reshape":
            return _reshape_dimension_lineage(
                operand_lineage,
                operand_type.shape,
                operation.results[0].type.shape,
            )
        maps = operation.attributes.get("indexing_maps")
        if maps is None:
            if operand_type.shape != operation.results[0].type.shape:
                raise ValueError("pointwise score Map without index relations must preserve its tensor shape")
            return operand_lineage
        affine_maps = _affine_maps(operation)
        if len(affine_maps) != len(operation.operands) + 1:
            raise ValueError("score Map must carry one affine relation per operand and result")
        return _transfer_affine_dimension_lineage(
            operand_lineage,
            affine_maps[operand_index],
            affine_maps[-1],
        )

    if operation.name != "shuttle.relation":
        raise ValueError(f"unsupported score index dataflow operation {operation.name!r}")
    semantics = _relation_semantics(operation)
    if semantics in ("convert", "select", "add"):
        if operand_type.shape != operation.results[0].type.shape:
            raise ValueError(f"score {semantics} relation must preserve its producer-derived operand shape")
        return operand_lineage
    if semantics == "broadcast_in_dim":
        parameters = tuple(
            _relation_integer_parameter(parameter) for parameter in operation.attributes.get("parameters", ())
        )
        if not parameters and not operand_lineage:
            return (None,) * len(operation.results[0].type.shape)
        if len(parameters) != len(operand_lineage):
            raise ValueError("score broadcast relation must map every producer-derived operand axis")
        if len(set(parameters)) != len(parameters) or any(
            dimension < 0 or dimension >= len(operation.results[0].type.shape) for dimension in parameters
        ):
            raise ValueError("score broadcast relation has invalid result dimensions")
        result: list[int | None] = [None] * len(operation.results[0].type.shape)
        for operand_dimension, result_dimension in enumerate(parameters):
            result[result_dimension] = operand_lineage[operand_dimension]
        return tuple(result)
    if semantics == "reshape":
        return _reshape_dimension_lineage(
            operand_lineage,
            operand_type.shape,
            operation.results[0].type.shape,
        )
    if semantics == "transpose":
        parameters = tuple(
            _relation_integer_parameter(parameter) for parameter in operation.attributes.get("parameters", ())
        )
        if sorted(parameters) != list(range(len(operand_lineage))):
            raise ValueError("score transpose relation must carry an exact permutation")
        return tuple(operand_lineage[operand_dimension] for operand_dimension in parameters)
    raise ValueError(f"unsupported score index dataflow relation {semantics!r}")


def _transfer_affine_dimension_lineage(
    operand_lineage: tuple[int | None, ...],
    operand_map: dict[str, Any],
    result_map: dict[str, Any],
) -> tuple[int | None, ...]:
    operand_expressions = operand_map.get("results")
    if not isinstance(operand_expressions, list):
        raise ValueError("score Map operand relation has no structured results")
    result_dimensions = _direct_dimensions(result_map)
    if len(operand_expressions) != len(operand_lineage):
        raise ValueError("score Map operand relation rank does not match its tensor rank")
    domain_lineage: dict[int, int | None] = {}
    for expression, lineage in zip(operand_expressions, operand_lineage, strict=True):
        dimensions = _affine_expression_dimensions(expression)
        if len(dimensions) != 1:
            raise ValueError("score Map operand axis must derive from exactly one affine domain dimension")
        dimension = next(iter(dimensions))
        existing = domain_lineage.get(dimension)
        if dimension in domain_lineage and existing != lineage:
            raise ValueError("score Map operand relation aliases distinct producer axes")
        domain_lineage[dimension] = lineage
    try:
        return tuple(domain_lineage[dimension] for dimension in result_dimensions)
    except KeyError as error:
        raise ValueError(f"score Map result introduces unbound affine dimension d{error.args[0]}") from error


def _affine_expression_dimensions(expression: object) -> frozenset[int]:
    if not isinstance(expression, dict):
        raise ValueError("score Map affine relation has an invalid expression")
    kind = expression.get("kind")
    if kind == "dimension":
        return frozenset((_integer(expression.get("position"), "affine dimension position"),))
    if kind == "constant":
        return frozenset()
    operands = expression.get("operands")
    if not isinstance(operands, list):
        left = expression.get("left")
        right = expression.get("right")
        if left is None or right is None:
            raise ValueError(f"score Map affine relation has unsupported expression {kind!r}")
        operands = [left, right]
    return frozenset(dimension for operand in operands for dimension in _affine_expression_dimensions(operand))


def _map_semantics(operation: ShuttleOperationRecord) -> str:
    encoded = operation.attributes.get("semantics")
    if encoded is None:
        return "pointwise"
    value = _mlir_attribute(encoded, "Map semantics")
    prefix = "#shuttle.map_semantics<"
    if not value.startswith(prefix) or not value.endswith(">"):
        raise ValueError(f"invalid Map semantics {value!r}")
    return value[len(prefix) : -1]


def _reshape_dimension_lineage(
    operand_lineage: tuple[int | None, ...],
    operand_shape: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> tuple[int | None, ...]:
    if operand_shape == result_shape:
        return operand_lineage
    if math.prod(operand_shape) != math.prod(result_shape):
        raise ValueError("score reshape relation must preserve element count")
    operand_nonunit = tuple(
        (extent, lineage) for extent, lineage in zip(operand_shape, operand_lineage, strict=True) if extent != 1
    )
    result_nonunit = tuple(extent for extent in result_shape if extent != 1)
    if tuple(extent for extent, _ in operand_nonunit) != result_nonunit:
        raise ValueError("score reshape relation may only insert or remove unit axes")
    lineages = iter(lineage for _, lineage in operand_nonunit)
    return tuple(None if extent == 1 else next(lineages) for extent in result_shape)


def _is_contiguous_projection(dimensions: tuple[int, ...], projected: tuple[int, ...]) -> bool:
    if not projected:
        return True
    positions = tuple(
        index for dimension in projected for index, candidate in enumerate(dimensions) if candidate == dimension
    )
    return len(positions) == len(projected) and positions == tuple(range(positions[0], positions[0] + len(positions)))


def _ancestor_operations(
    value: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> tuple[ShuttleOperationRecord, ...]:
    operation = producer_by_value.get(value)
    if operation is None:
        return ()
    ancestors = [operation]
    for operand in operation.operands:
        for ancestor in _ancestor_operations(operand, producer_by_value):
            if all(existing is not ancestor for existing in ancestors):
                ancestors.append(ancestor)
    return tuple(ancestors)


def _ancestor_operations_before_contract(
    value: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> tuple[ShuttleOperationRecord, ...]:
    operation = producer_by_value.get(value)
    if operation is None or operation.name == "shuttle.contract":
        return ()
    ancestors = [operation]
    for operand in operation.operands:
        for ancestor in _ancestor_operations_before_contract(operand, producer_by_value):
            if all(existing is not ancestor for existing in ancestors):
                ancestors.append(ancestor)
    return tuple(ancestors)


def _argument_roots(
    value: int,
    producer_by_value: dict[int, ShuttleOperationRecord],
    argument_index: dict[int, int],
) -> frozenset[int]:
    if value in argument_index:
        return frozenset((argument_index[value],))
    operation = producer_by_value.get(value)
    if operation is None:
        return frozenset()
    return frozenset(
        root for operand in operation.operands for root in _argument_roots(operand, producer_by_value, argument_index)
    )


def _fold_reducer(operation: ShuttleOperationRecord) -> str | None:
    if operation.name != "shuttle.fold":
        return None
    names = {nested.name for nested in operation.body().operations}
    if "arith.maximumf" in names:
        return "maximum"
    if "arith.addf" in names:
        return "sum"
    return None


def _contains_scalar_operation(operation: ShuttleOperationRecord, name: str) -> bool:
    return any(nested.name == name for nested in operation.walk())


def _domain_predicate(
    score_program: tuple[ShuttleOperationRecord, ...],
    producer: ShuttleOperationRecord,
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> PhysicalContractDomainPredicate | None:
    candidates: list[ShuttleOperationRecord] = []
    for operation in score_program:
        if operation.name != "shuttle.relation":
            continue
        semantics = _mlir_attribute(operation.attributes.get("semantics"), "relation semantics")
        if semantics != "#shuttle.relation_semantics<select>":
            continue
        candidates.append(operation)
    if len(candidates) > 1:
        raise ValueError(
            f"the current normalized-exp Contract template supports one score-domain selection, found {len(candidates)}"
        )
    if not candidates:
        return None
    select = candidates[0]
    if len(select.operands) != 3:
        raise ValueError("score-domain select relation must consume predicate, selected score, and rejected score")
    result_dimensions = contract_indexing_dimensions(producer)[2]
    if len(result_dimensions) != len(select.results[0].type.shape):
        raise ValueError("score-domain rank must match the producer Contract result relation")
    lineage_producers = {**producer_by_value, **{result.id: producer for result in producer.results}}
    select_dimensions = _tensor_dimension_lineage(
        _single_result(select),
        root_value=_single_result(producer),
        root_dimensions=result_dimensions,
        producer_by_value=lineage_producers,
    )
    if len(select_dimensions) != len(select.results[0].type.shape) or any(
        dimension is None for dimension in select_dimensions
    ):
        raise ValueError("score-domain select must preserve every producer Contract iteration dimension")
    coordinates = tuple(_affine_dimension(dimension) for dimension in select_dimensions if dimension is not None)
    predicate = _affine_index_predicate(select.operands[0], coordinates, producer_by_value)
    source_operations = tuple(reversed(_ancestor_operations(select.operands[0], producer_by_value)))
    return PhysicalContractDomainPredicate(select, source_operations, predicate)


def _affine_index_predicate(
    value: int,
    coordinates: tuple[AffineIndexExpression, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> AffineIndexPredicate:
    operation = producer_by_value.get(value)
    if operation is None:
        raise ValueError("score-domain predicate has no source relation")
    semantics = _relation_semantics(operation)
    if semantics in ("reshape", "broadcast_in_dim"):
        if len(operation.operands) != 1:
            raise ValueError(f"score-domain {semantics} relation must consume one operand")
        return _affine_index_predicate(
            operation.operands[0],
            _relation_operand_coordinates(operation, coordinates, producer_by_value),
            producer_by_value,
        )
    if semantics != "compare" or len(operation.operands) != 2:
        raise ValueError(f"unsupported score-domain predicate relation {semantics!r}")
    parameters = operation.attributes.get("parameters", ())
    if not isinstance(parameters, (list, tuple)) or tuple(parameters) != ("SIGNED", "LE"):
        raise ValueError("current SM90 affine domain codegen supports signed less-equal comparisons")
    return AffineIndexPredicate(
        IndexComparisonKind.SIGNED_LESS_EQUAL,
        _affine_index_value(operation.operands[0], coordinates, producer_by_value),
        _affine_index_value(operation.operands[1], coordinates, producer_by_value),
    )


def _affine_index_value(
    value: int,
    coordinates: tuple[AffineIndexExpression, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> AffineIndexExpression:
    operation = producer_by_value.get(value)
    if operation is None:
        raise ValueError("score-domain affine value has no source relation")
    semantics = _relation_semantics(operation)
    if semantics == "iota":
        parameters = operation.attributes.get("parameters", ())
        if not isinstance(parameters, (list, tuple)) or len(parameters) != 1:
            raise ValueError("score-domain iota relation must carry one dimension")
        dimension = _relation_integer_parameter(parameters[0])
        if dimension < 0 or dimension >= len(coordinates):
            raise ValueError("score-domain iota dimension is out of range")
        return coordinates[dimension]
    if semantics == "constant":
        parameters = operation.attributes.get("parameters", ())
        if not isinstance(parameters, (list, tuple)) or len(parameters) != 1:
            raise ValueError("score-domain constant relation must carry one value")
        return AffineIndexExpression(constant=_relation_integer_parameter(parameters[0]))
    if semantics in ("reshape", "broadcast_in_dim", "convert"):
        if len(operation.operands) != 1:
            raise ValueError(f"score-domain {semantics} relation must consume one operand")
        operand_coordinates = (
            coordinates
            if semantics == "convert"
            else _relation_operand_coordinates(operation, coordinates, producer_by_value)
        )
        return _affine_index_value(operation.operands[0], operand_coordinates, producer_by_value)
    if semantics == "add":
        if len(operation.operands) != 2:
            raise ValueError("score-domain add relation must consume two operands")
        return _add_affine_index_expressions(
            _affine_index_value(operation.operands[0], coordinates, producer_by_value),
            _affine_index_value(operation.operands[1], coordinates, producer_by_value),
        )
    raise ValueError(f"unsupported affine score-domain relation {semantics!r}")


def _relation_operand_coordinates(
    operation: ShuttleOperationRecord,
    result_coordinates: tuple[AffineIndexExpression, ...],
    producer_by_value: dict[int, ShuttleOperationRecord],
) -> tuple[AffineIndexExpression, ...]:
    if len(operation.operands) != 1:
        raise ValueError("coordinate projection requires a unary relation")
    semantics = _relation_semantics(operation)
    # Operand types are recovered from the relation result graph by the caller;
    # broadcasts carry an explicit exact axis map, while reshapes are admitted
    # only when they insert or remove singleton dimensions.
    if semantics == "broadcast_in_dim":
        parameters = operation.attributes.get("parameters", ())
        if not isinstance(parameters, (list, tuple)):
            raise ValueError("score-domain broadcast dimensions must be an array")
        dimensions = tuple(_relation_integer_parameter(parameter) for parameter in parameters)
        if any(dimension < 0 or dimension >= len(result_coordinates) for dimension in dimensions):
            raise ValueError("score-domain broadcast dimension is out of range")
        return tuple(result_coordinates[dimension] for dimension in dimensions)
    if semantics != "reshape":
        raise ValueError(f"relation {semantics!r} has no coordinate projection")
    operand_operation = producer_by_value.get(operation.operands[0])
    if operand_operation is None:
        raise ValueError("score-domain reshape operand has no source relation")
    operand_result = next(
        (result for result in operand_operation.results if result.id == operation.operands[0]),
        None,
    )
    if operand_result is None:
        raise ValueError("score-domain reshape operand type is unavailable")
    operand_shape = operand_result.type.shape
    result_shape = operation.results[0].type.shape
    operand_nonunit = tuple((index, extent) for index, extent in enumerate(operand_shape) if extent != 1)
    result_nonunit = tuple((index, extent) for index, extent in enumerate(result_shape) if extent != 1)
    if tuple(extent for _, extent in operand_nonunit) != tuple(extent for _, extent in result_nonunit):
        raise ValueError("current SM90 affine domain codegen supports reshapes that only insert/remove unit axes")
    mapped = {
        operand_axis: result_coordinates[result_axis]
        for (operand_axis, _), (result_axis, _) in zip(operand_nonunit, result_nonunit, strict=True)
    }
    return tuple(mapped.get(axis, AffineIndexExpression()) for axis in range(len(operand_shape)))


def _relation_semantics(operation: ShuttleOperationRecord) -> str:
    encoded = _mlir_attribute(operation.attributes.get("semantics"), "relation semantics")
    prefix = "#shuttle.relation_semantics<"
    if not encoded.startswith(prefix) or not encoded.endswith(">"):
        raise ValueError(f"invalid relation semantics {encoded!r}")
    return encoded[len(prefix) : -1]


def _relation_integer_parameter(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, dict) and type(value.get("value")) is int:
        return value["value"]
    raise ValueError("score-domain affine relation requires exact integer parameters")


def source_relation_semantics(operation: ShuttleOperationRecord) -> tuple[str, tuple[int, ...]]:
    """Decode one typed source Relation kind and its integer parameters."""
    if operation.name != "shuttle.relation":
        raise ValueError("source Relation decoding requires a shuttle.relation operation")
    parameters = operation.attributes.get("parameters", ())
    if not isinstance(parameters, (list, tuple)):
        raise ValueError("source Relation parameters must be a structured sequence")
    return _relation_semantics(operation), tuple(_relation_integer_parameter(value) for value in parameters)


def _affine_dimension(dimension: int) -> AffineIndexExpression:
    return AffineIndexExpression(((dimension, 1),))


def _add_affine_index_expressions(
    left: AffineIndexExpression,
    right: AffineIndexExpression,
) -> AffineIndexExpression:
    coefficients: dict[int, int] = {}
    for dimension, coefficient in (*left.coefficients, *right.coefficients):
        coefficients[dimension] = coefficients.get(dimension, 0) + coefficient
    return AffineIndexExpression(
        tuple((dimension, coefficient) for dimension, coefficient in sorted(coefficients.items()) if coefficient),
        left.constant + right.constant,
    )


def _remap_affine_index_predicate(
    predicate: AffineIndexPredicate,
    dimensions: dict[int, int],
) -> AffineIndexPredicate:
    def remap(expression: AffineIndexExpression) -> AffineIndexExpression:
        try:
            coefficients = tuple(
                sorted((dimensions[dimension], coefficient) for dimension, coefficient in expression.coefficients)
            )
        except KeyError as error:
            raise ValueError(f"score-domain predicate references non-iteration dimension d{error.args[0]}") from error
        return AffineIndexExpression(coefficients, expression.constant)

    return AffineIndexPredicate(predicate.comparison, remap(predicate.left), remap(predicate.right))


def _scalar_expression_from_graph(
    value: int,
    roots: dict[int, str],
    producer_by_value: dict[int, ShuttleOperationRecord],
    domain_predicate: PhysicalContractDomainPredicate | None,
) -> ScalarExpression:
    root = roots.get(value)
    if root is not None:
        return scalar_input(root)
    operation = producer_by_value.get(value)
    if operation is None:
        raise ValueError(f"scalar AST value {value} has no source operation or bound input")
    if operation.name == "shuttle.map":
        operands = tuple(
            _scalar_expression_from_graph(operand, roots, producer_by_value, domain_predicate)
            for operand in operation.operands
        )
        return _map_scalar_expression(operation, operands)
    if domain_predicate is not None and operation is domain_predicate.relation:
        if operation.results and operation.results[0].type.element_type in ("i1", "pred"):
            return scalar_input(_DOMAIN_PREDICATE_INPUT)
        candidates: list[ScalarExpression] = []
        for operand in operation.operands:
            try:
                candidate = _scalar_expression_from_graph(operand, roots, producer_by_value, domain_predicate)
            except ValueError:
                continue
            if scalar_expression_inputs(candidate) & set(roots.values()):
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("domain relation does not identify one producer-derived scalar value")
    if operation.name == "shuttle.relation":
        semantics = _relation_semantics(operation)
        if semantics not in ("broadcast_in_dim", "reshape"):
            raise ValueError(f"unsupported scalar dataflow relation {semantics!r}")
        if len(operation.operands) != 1:
            raise ValueError(f"scalar dataflow {semantics} relation must consume one operand")
        return _scalar_expression_from_graph(operation.operands[0], roots, producer_by_value, domain_predicate)
    if operation.name == "arith.constant":
        return scalar_constant(_constant_value(operation))
    raise ValueError(f"unsupported scalar graph operation {operation.name!r}")


def _map_scalar_expression(
    operation: ShuttleOperationRecord,
    operands: tuple[ScalarExpression, ...],
) -> ScalarExpression:
    block = operation.body()
    if len(block.arguments) != len(operands):
        raise ValueError("scalar Map body arguments do not match its tensor operands")
    expressions = {argument.id: operand for argument, operand in zip(block.arguments, operands, strict=True)}
    for nested in block.operations:
        if nested.name == "shuttle.yield":
            if len(nested.operands) != 1 or nested.operands[0] not in expressions:
                raise ValueError("scalar Map must yield one decoded scalar value")
            return expressions[nested.operands[0]]
        if len(nested.results) != 1:
            raise ValueError(f"scalar operation {nested.name!r} must produce one value")
        result = nested.results[0].id
        try:
            arguments = tuple(expressions[operand] for operand in nested.operands)
        except KeyError as error:
            raise ValueError(f"scalar operation {nested.name!r} references an unavailable SSA value") from error
        expressions[result] = _scalar_operation_expression(nested, arguments)
    raise ValueError("scalar Map body has no shuttle.yield")


def scalar_expression_from_source_map(
    operation: ShuttleOperationRecord,
    input_names: tuple[str, ...],
) -> ScalarExpression:
    """Decode one typed Shuttle scalar Map body into the shared scalar AST."""
    if operation.name != "shuttle.map":
        raise ValueError("scalar source decoding requires a shuttle.map operation")
    return _map_scalar_expression(operation, tuple(scalar_input(name) for name in input_names))


def _scalar_operation_expression(
    operation: ShuttleOperationRecord,
    operands: tuple[ScalarExpression, ...],
) -> ScalarExpression:
    binary = {
        "arith.addf": ScalarExpressionKind.ADD,
        "arith.subf": ScalarExpressionKind.SUBTRACT,
        "arith.mulf": ScalarExpressionKind.MULTIPLY,
        "arith.divf": ScalarExpressionKind.DIVIDE,
    }
    if operation.name == "arith.constant":
        if operands:
            raise ValueError("arith.constant cannot consume scalar operands")
        constant = scalar_constant(_constant_value(operation))
        encoded = operation.attributes.get("value")
        if isinstance(encoded, dict) and encoded.get("element_type") == "bf16":
            return scalar_convert(
                constant,
                result_dtype=DType.BF16,
                semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
            )
        return constant
    if operation.name == "shuttle.scalar_convert":
        if len(operands) != 1 or len(operation.results) != 1:
            raise ValueError("shuttle.scalar_convert must consume and produce one scalar value")
        semantics = _mlir_attribute(operation.attributes.get("semantics"), "scalar conversion semantics")
        try:
            conversion_semantics = ScalarConversionSemantics(semantics.split("<")[-1].rstrip(">"))
        except ValueError as error:
            raise ValueError(f"unsupported scalar conversion semantics {semantics!r}") from error
        element_type = operation.results[0].type.element_type
        result_dtype = {
            "bf16": DType.BF16,
            "f32": DType.FP32,
        }.get(element_type)
        if result_dtype is None:
            raise ValueError(f"unsupported scalar conversion result type {element_type!r}")
        return scalar_convert(operands[0], result_dtype=result_dtype, semantics=conversion_semantics)
    if operation.name in binary:
        if len(operands) != 2:
            raise ValueError(f"{operation.name} must consume two scalar operands")
        return scalar_binary(binary[operation.name], *operands)
    if operation.name == "arith.negf":
        if len(operands) != 1:
            raise ValueError("arith.negf must consume one scalar operand")
        return scalar_unary(ScalarExpressionKind.NEGATE, operands[0])
    if operation.name == "math.tanh":
        if len(operands) != 1:
            raise ValueError("math.tanh must consume one scalar operand")
        return scalar_unary(ScalarExpressionKind.TANH, operands[0])
    if operation.name == "math.exp":
        if len(operands) != 1:
            raise ValueError("math.exp must consume one scalar operand")
        return scalar_unary(ScalarExpressionKind.EXP, operands[0])
    if operation.name == "arith.cmpf":
        if len(operands) != 2:
            raise ValueError("arith.cmpf must consume two scalar operands")
        predicate = _comparison_predicate(operation)
        if predicate in ("ole", "ule", "le"):
            return scalar_binary(ScalarExpressionKind.LESS_EQUAL, *operands)
        if predicate in ("oge", "uge", "ge"):
            return scalar_binary(ScalarExpressionKind.LESS_EQUAL, operands[1], operands[0])
        if predicate in ("oeq", "ueq", "eq"):
            return scalar_binary(ScalarExpressionKind.EQUAL, *operands)
        raise ValueError(f"unsupported CuTe scalar comparison predicate {predicate!r}")
    if operation.name in ("arith.select", "select"):
        if len(operands) != 3:
            raise ValueError(f"{operation.name} must consume predicate, true, and false values")
        return scalar_select(*operands)
    raise ValueError(f"unsupported CuTe scalar operation {operation.name!r}")


def _comparison_predicate(operation: ShuttleOperationRecord) -> str:
    encoded = operation.attributes.get("predicate")
    if type(encoded) is int:
        try:
            return {
                1: "oeq",
                3: "oge",
                5: "ole",
                8: "ueq",
                10: "uge",
                12: "ule",
            }[encoded]
        except KeyError as error:
            raise ValueError(f"unsupported numeric arith.cmpf predicate {encoded}") from error
    if isinstance(encoded, str):
        return encoded.lower()
    if isinstance(encoded, dict):
        value = encoded.get("value")
        if isinstance(value, str):
            return value.lower().split("<")[-1].rstrip(">")
    raise ValueError("arith.cmpf must carry a typed predicate")


def _constant_value(operation: ShuttleOperationRecord) -> float | bool:
    encoded = operation.attributes.get("value")
    if not isinstance(encoded, dict):
        raise ValueError("physical scalar constant must use exact typed encoding")
    kind = encoded.get("kind")
    element_type = encoded.get("element_type")
    bits = encoded.get("bits")
    if kind == "float" and element_type == "f32" and isinstance(bits, str):
        return struct.unpack(">f", bytes.fromhex(bits.zfill(8)))[0]
    if kind == "float" and element_type == "f64" and isinstance(bits, str):
        return struct.unpack(">d", bytes.fromhex(bits.zfill(16)))[0]
    if kind == "float" and element_type == "bf16" and isinstance(bits, str):
        return struct.unpack(">f", bytes.fromhex(bits.zfill(4) + "0000"))[0]
    if kind in ("integer", "bool") and element_type in ("i1", "pred"):
        if isinstance(bits, str):
            return bool(int(bits, 16))
        value = encoded.get("value")
        if type(value) is int:
            return bool(value)
    raise ValueError("current CuTe scalar constants must be BF16, f32, f64, or Boolean")


def _without_domain_predicate(expression: ScalarExpression) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT and expression.input_name == _DOMAIN_PREDICATE_INPUT:
        raise ValueError("domain predicate may only guard a score selection")
    if expression.kind is ScalarExpressionKind.SELECT:
        predicate, when_true, when_false = expression.operands
        if predicate.kind is ScalarExpressionKind.INPUT and predicate.input_name == _DOMAIN_PREDICATE_INPUT:
            if _is_negative_infinity(when_false):
                return _without_domain_predicate(when_true)
            if _is_negative_infinity(when_true):
                return _without_domain_predicate(when_false)
            raise ValueError("domain predicate selection must reject scores with negative infinity")
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_without_domain_predicate(operand) for operand in expression.operands),
        input_name=expression.input_name,
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def _is_negative_infinity(expression: ScalarExpression) -> bool:
    return (
        expression.kind is ScalarExpressionKind.CONSTANT
        and isinstance(expression.constant, float)
        and expression.constant == float("-inf")
    )


def _single_result(operation: ShuttleOperationRecord) -> int:
    if len(operation.results) != 1:
        raise ValueError(f"{operation.name} must produce one value for scalar AST recovery")
    return operation.results[0].id


def _affine_maps(operation: ShuttleOperationRecord) -> tuple[dict[str, Any], ...]:
    maps = operation.attributes.get("indexing_maps")
    if not isinstance(maps, list) or any(not isinstance(value, dict) for value in maps):
        raise ValueError(f"{operation.name} has no typed affine indexing maps")
    return tuple(maps)


def contract_indexing_dimensions(operation: ShuttleOperationRecord) -> tuple[tuple[int, ...], ...]:
    """Return each Contract operand/result relation as direct domain dimensions."""
    return tuple(_direct_dimensions(affine_map) for affine_map in _affine_maps(operation))


def _direct_dimensions(affine_map: dict[str, Any]) -> tuple[int, ...]:
    results = affine_map.get("results")
    if not isinstance(results, list):
        raise ValueError("affine map has no structured results")
    dimensions: list[int] = []
    for expression in results:
        if not isinstance(expression, dict) or expression.get("kind") != "dimension":
            raise ValueError("Contract operand maps currently require direct dimension projections")
        dimensions.append(_integer(expression.get("position"), "affine dimension position"))
    return tuple(dimensions)


def _extent_for_dimension(shape: tuple[int, ...], affine_map: dict[str, Any], dimension: int) -> int:
    dimensions = _direct_dimensions(affine_map)
    positions = tuple(index for index, mapped_dimension in enumerate(dimensions) if mapped_dimension == dimension)
    if len(positions) != 1:
        raise ValueError(f"affine dimension d{dimension} is not projected exactly once")
    return shape[positions[0]]


def _type_record(encoded: dict[str, Any]) -> ShuttleTypeRecord:
    kind = encoded.get("kind")
    element_type = encoded.get("element_type")
    shape = encoded.get("shape")
    if kind != "ranked_tensor" or not isinstance(element_type, str) or not isinstance(shape, list):
        raise ValueError("materialization tensor_type must be a ranked tensor")
    return ShuttleTypeRecord(
        kind=kind,
        element_type=element_type,
        shape=_integer_array(shape, "materialization tensor shape"),
    )


def _mlir_attribute(value: object, label: str) -> str:
    if not isinstance(value, dict) or value.get("kind") != "mlir_attribute" or not isinstance(value.get("value"), str):
        raise ValueError(f"{label} must be a typed MLIR attribute")
    return value["value"]


def _source_operation_ordinal(value: object, label: str) -> int:
    encoded = _mlir_attribute(value, label)
    prefix = "#shuttle.source_ref<"
    if not encoded.startswith(prefix) or not encoded.endswith(">"):
        raise ValueError(f"{label} must bind a source operation in the physical region")
    coordinates = encoded[len(prefix) : -1].split(", ")
    if not all(coordinate.isdecimal() for coordinate in coordinates):
        raise ValueError(f"{label} has a malformed operation/result coordinate")
    if len(coordinates) == 1:
        return int(coordinates[0])
    if len(coordinates) == 4 and coordinates[:2] == ["0", "0"]:
        return int(coordinates[2])
    raise ValueError(f"{label} does not bind a top-level physical-region operation")


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _integer_array(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(element) is not int for element in value):
        raise ValueError(f"{label} must be an integer array")
    return tuple(int(element) for element in value)


def _optional_integer_array(value: object, label: str) -> tuple[int, ...]:
    return () if value is None else _integer_array(value, label)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be Boolean")
    return value


def _optional_boolean(value: object, label: str) -> bool:
    return False if value is None else _boolean(value, label)
