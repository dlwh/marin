# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive streaming reverse programs from generic normalized-exp tensor algebra."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, TypeVar

import numpy as np

from shuttle.ir import DType
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.event_dataflow import TaskAxis, TaskFamily, TaskRelation
from tile_lifetime.fold_placement import (
    FoldAttachment,
    FoldResultDisposition,
    OwnerTileAvailability,
    attach_fold_to_owner_preparation,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractProgram,
    NormalizedExpContractPullback,
    ReverseContractFoldOrder,
    ReverseOwnershipPlan,
    derive_normalized_exp_contract_pullback,
    derive_reverse_ownership_plan,
    normalized_exp_finalizer_scale,
    relation_group_size,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.shared_reverse_fusion import SharedReverseFusionPlan, plan_shared_producer_reverse_fusion
from tile_lifetime.streaming_attention import (
    AttentionScoreAxis,
    StreamingAttentionExecution,
    StreamingTileSchedule,
    _align_array,
    _contract,
    _evaluate_map,
    _slice_array,
    _validated_inputs,
    execute_tensor_program,
)
from tile_lifetime.tensor_program import (
    FoldPrimitive,
    MapPrimitive,
    ProgramValue,
    TensorProgram,
    scalar_expression_inputs,
)


class StreamingAttentionBackwardStage(StrEnum):
    """Generic physical stages needed by the reverse streamed Fold."""

    LOAD_QUERY_AND_STATE = "load_query_and_state"
    LOAD_KEY_VALUE = "load_key_value"
    RECOMPUTE_QK = "recompute_qk"
    RECOMPUTE_PROBABILITY = "recompute_probability"
    DV_CONTRACT = "dv_contract"
    DP_CONTRACT = "dp_contract"
    SCORE_MAP_VJP = "score_map_vjp"
    DQ_CONTRACT = "dq_contract"
    DK_CONTRACT = "dk_contract"


class StreamingAttentionBackwardReassociation(StrEnum):
    """Finite-precision order used for gradient accumulation."""

    DETERMINISTIC_TREE = "deterministic_tree"


class StreamingAttentionBackwardMaximumVJP(StrEnum):
    """Treatment of the maximum Fold introduced by JAX's normalized-exp VJP."""

    NORMALIZED_EXP_INVARIANT = "normalized_exp_invariant"
    JAX_EQUAL_SPLIT = "jax_equal_split"


class StreamingAttentionBackwardProvenance(StrEnum):
    """Origin of the visible generic reverse algebra."""

    REFERENCE_SYMBOLIC_VJP = "reference_symbolic_vjp"
    JAX_VJP_HLO_RECOVERY = "jax_vjp_hlo_recovery"


StreamingAttentionBackwardDomainTraversal = ContractDomainTraversal


StreamingAttentionBackwardFoldOrder = ReverseContractFoldOrder


ForwardScheduleT = TypeVar("ForwardScheduleT")


@dataclass(frozen=True)
class StreamingAttentionBackwardTileSchedule:
    """A deterministic reverse schedule over generic Contract/Fold axes.

    ``query_heads_per_key_value_tile`` is derived from the QK input index map.
    It coalesces all query rows that consume the same K/V tile into one physical
    Contract in both reverse traversals.  This is an axis-relation
    transformation, not an attention-name dispatch.
    """

    query_tile_size: int
    key_value_tile_size: int
    query_heads_per_key_value_tile: int
    domain_traversal: StreamingAttentionBackwardDomainTraversal
    key_value_fold_order: StreamingAttentionBackwardFoldOrder
    ownership: ReverseOwnershipPlan
    query_owner_attachments: tuple[FoldAttachment, ...] = ()

    def __post_init__(self) -> None:
        extents = (
            self.query_tile_size,
            self.key_value_tile_size,
            self.query_heads_per_key_value_tile,
        )
        if any(extent <= 0 for extent in extents):
            raise ValueError("streaming backward tile extents must be positive")
        if self.domain_traversal is not self.ownership.traversal:
            raise ValueError("reverse tile traversal must match the relation-derived ownership plan")

    @property
    def contract_tile_m(self) -> int:
        return self.query_tile_size

    @property
    def contract_tile_n(self) -> int:
        return self.key_value_tile_size

    @property
    def index_relation_group_size(self) -> int:
        return self.query_heads_per_key_value_tile

    @property
    def right_owner_fold_order(self) -> ReverseContractFoldOrder:
        return self.key_value_fold_order


@dataclass(frozen=True)
class StreamingAttentionBackwardWorkEstimate:
    """Static Contract counts and temporary footprint for one reverse schedule."""

    logical_query_key_tile_pairs: int
    fully_restricted_tile_pairs: int
    query_gradient_contract_invocations: int
    scalar_head_query_gradient_contract_invocations: int
    full_domain_query_gradient_contract_invocations: int
    full_domain_scalar_head_query_gradient_contract_invocations: int
    key_value_gradient_contract_invocations: int
    scalar_head_key_value_contract_invocations: int
    full_domain_scalar_head_key_value_contract_invocations: int
    packed_query_rows: int
    peak_score_tile_elements: int
    peak_query_tile_elements: int
    key_value_gradient_accumulator_elements: int

    @property
    def query_gradient_contract_invocation_reduction(self) -> float:
        return self.scalar_head_query_gradient_contract_invocations / self.query_gradient_contract_invocations

    @property
    def query_gradient_contract_invocation_reduction_from_full_scalar(self) -> float:
        return (
            self.full_domain_scalar_head_query_gradient_contract_invocations / self.query_gradient_contract_invocations
        )

    @property
    def key_value_contract_invocation_reduction(self) -> float:
        return self.scalar_head_key_value_contract_invocations / self.key_value_gradient_contract_invocations

    @property
    def key_value_contract_invocation_reduction_from_full_scalar(self) -> float:
        return self.full_domain_scalar_head_key_value_contract_invocations / self.key_value_gradient_contract_invocations


@dataclass(frozen=True)
class StreamingAttentionBackwardProgram(Generic[ForwardScheduleT]):
    """A reverse stream with visible scalar and contraction algebra.

    Accepted end-to-end paths recover this structure from JAX VJP HLO.  The
    local symbolic derivation remains a component oracle for physical schedule
    work until that recovery boundary is connected.
    """

    forward: NormalizedExpContractProgram[ForwardScheduleT]
    pullback: NormalizedExpContractPullback
    stages: tuple[StreamingAttentionBackwardStage, ...]
    materialized_values: tuple[ProgramValue, ...]
    provenance: StreamingAttentionBackwardProvenance
    accumulation_dtype: DType = DType.FP32
    reassociation: StreamingAttentionBackwardReassociation = StreamingAttentionBackwardReassociation.DETERMINISTIC_TREE
    maximum_vjp: StreamingAttentionBackwardMaximumVJP = StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT

    @property
    def output_cotangent(self) -> ProgramValue:
        return self.pullback.output_cotangent

    @property
    def output_dot_map(self) -> MapPrimitive:
        return self.pullback.output_product

    @property
    def output_dot_fold(self) -> FoldPrimitive:
        return self.pullback.output_dependent_fold

    @property
    def score_map_vjp(self) -> MapPrimitive:
        return self.pullback.score_map_vjp

    @property
    def output_scale(self) -> float:
        return self.pullback.output_scale


@dataclass(frozen=True)
class StreamingAttentionBackwardExecution:
    """Input cotangents produced by the generic reverse stream."""

    query_cotangent: np.ndarray
    key_cotangent: np.ndarray
    value_cotangent: np.ndarray


def derive_streaming_attention_backward(
    forward: NormalizedExpContractProgram[ForwardScheduleT],
) -> StreamingAttentionBackwardProgram[ForwardScheduleT]:
    """Build a reference reverse program used to validate JAX-VJP recovery."""
    pullback = derive_normalized_exp_contract_pullback(forward.region)
    cotangents = tuple(gradient.destination for gradient in pullback.terminal_gradients)
    return StreamingAttentionBackwardProgram(
        forward=forward,
        pullback=pullback,
        stages=tuple(StreamingAttentionBackwardStage),
        materialized_values=(*cotangents, pullback.output_dependent_fold.output),
        provenance=StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP,
    )


def eliminate_normalized_exp_maximum_vjp(
    program: StreamingAttentionBackwardProgram,
    *,
    numerical_policy: NumericalPolicy,
) -> StreamingAttentionBackwardProgram:
    """Eliminate the explicit maximum VJP using normalized-exp invariance.

    JAX differentiates the source max Fold and assigns its cotangent equally to
    tied maxima.  For a normalized exponential, adding a row constant leaves
    the result unchanged, so the complete maximum-cotangent path cancels over
    the reals.  The cancellation changes floating-point operation order and is
    therefore unavailable under ``BITWISE_EXACT``.
    """
    if program.maximum_vjp is StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT:
        return program
    if program.maximum_vjp is not StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT:
        raise ValueError(f"unsupported maximum VJP {program.maximum_vjp.value!r}")
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("bitwise policy requires preserving JAX's explicit equal-split maximum VJP")
    return replace(
        program,
        maximum_vjp=StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT,
    )


def verify_streaming_attention_backward_score_map_vjp(
    program: StreamingAttentionBackwardProgram,
) -> None:
    """Verify that the reverse score Map is derived from the forward scalar AST.

    Physical streaming templates may lower only a constrained scalar primitive
    set, but they must not substitute a fixed workload formula for the recovered
    program. This check keeps the semantic mutation boundary explicit before a
    backend binds the scalar derivative into its tile schedule.
    """
    forward_map = program.forward.region.score_map
    expected_expression = scalar_expression_vjp(
        forward_map.expression,
        input_name=program.forward.region.producer.output.name,
        cotangent_name=program.pullback.normalized_exp_pullback.output.name,
    )
    if program.score_map_vjp.expression != expected_expression:
        raise ValueError("score Map VJP is not the derivative of the recovered forward scalar AST")
    expected_inputs = scalar_expression_inputs(expected_expression)
    actual_inputs = tuple(value.name for value in program.score_map_vjp.inputs)
    if len(actual_inputs) != len(expected_inputs) or set(actual_inputs) != expected_inputs:
        raise ValueError(
            "score Map VJP inputs do not match its derived scalar expression: "
            f"expected {tuple(sorted(expected_inputs))}, found {actual_inputs}"
        )
    if program.score_map_vjp.output.axes != forward_map.output.axes:
        raise ValueError("score Map VJP must preserve the forward score Map output domain")


def derive_streaming_attention_backward_tile_schedule(
    program: StreamingAttentionBackwardProgram,
    *,
    query_tile_size: int,
    key_value_tile_size: int,
    domain_traversal: StreamingAttentionBackwardDomainTraversal,
) -> StreamingAttentionBackwardTileSchedule:
    """Coalesce query heads that share one operand through a Contract index map."""
    matching_maps = program.forward.region.producer.index_maps_for_input(1)
    group_size = relation_group_size(matching_maps)
    output_dot_attachment = attach_fold_to_owner_preparation(
        program.output_dot_map,
        program.output_dot_fold,
        owner_axes=program.output_dot_fold.output.axes,
        input_availability=tuple(
            OwnerTileAvailability(value, program.output_dot_fold.reduction_axes)
            for value in program.output_dot_map.inputs
        ),
        result_disposition=FoldResultDisposition.MATERIALIZE_FOR_CONSUMERS,
    )
    return StreamingAttentionBackwardTileSchedule(
        query_tile_size=query_tile_size,
        key_value_tile_size=key_value_tile_size,
        query_heads_per_key_value_tile=group_size,
        domain_traversal=domain_traversal,
        key_value_fold_order=ReverseContractFoldOrder.STREAMED_AXES_MAJOR_MAPPED_RELATION_MINOR_TREE,
        query_owner_attachments=(output_dot_attachment,),
        ownership=derive_reverse_ownership_plan(program.pullback, traversal=domain_traversal),
    )


def estimate_streaming_attention_backward_work(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
) -> StreamingAttentionBackwardWorkEstimate:
    """Count physical reverse Contracts without executing a target backend."""
    query, key = program.forward.region.producer.inputs
    query_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key.axes if axis.label == AttentionScoreAxis.KEY.value)
    batch_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.BATCH.value)
    query_head = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.HEAD.value)
    query_feature = next(axis for axis in query.axes if axis.label == "key_feature")
    key_value_head = next(axis for axis in key.axes if axis.label == "key_value_head")
    expected_head_group = query_head.extent // key_value_head.extent
    if schedule.query_heads_per_key_value_tile != expected_head_group:
        raise ValueError("work estimate schedule disagrees with the Contract head index relation")
    if query_axis.extent % schedule.query_tile_size or key_axis.extent % schedule.key_value_tile_size:
        raise ValueError("work estimates require tile-aligned query and key domains")
    query_tiles = query_axis.extent // schedule.query_tile_size
    key_tiles = key_axis.extent // schedule.key_value_tile_size
    all_tile_pairs = query_tiles * key_tiles
    if schedule.domain_traversal is StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR:
        valid_tile_pairs = sum(
            max(0, query_tiles - (key_start // schedule.query_tile_size))
            for key_start in range(0, key_axis.extent, schedule.key_value_tile_size)
        )
    else:
        valid_tile_pairs = all_tile_pairs
    if valid_tile_pairs <= 0 or valid_tile_pairs > all_tile_pairs:
        raise ValueError("domain traversal produced an invalid Q/K tile count")

    batch_count = batch_axis.extent
    logical_tile_pairs = batch_count * query_head.extent * valid_tile_pairs
    fully_restricted = batch_count * query_head.extent * (all_tile_pairs - valid_tile_pairs)
    grouped_query_contracts = batch_count * key_value_head.extent * valid_tile_pairs * 3
    scalar_query_contracts = logical_tile_pairs * 3
    full_domain_scalar_query_contracts = batch_count * query_head.extent * all_tile_pairs * 3
    grouped_key_value_contracts = batch_count * key_value_head.extent * valid_tile_pairs * 4
    scalar_key_value_contracts = logical_tile_pairs * 4
    full_domain_scalar_key_value_contracts = batch_count * query_head.extent * all_tile_pairs * 4
    packed_query_rows = schedule.query_tile_size * schedule.query_heads_per_key_value_tile
    return StreamingAttentionBackwardWorkEstimate(
        logical_query_key_tile_pairs=logical_tile_pairs,
        fully_restricted_tile_pairs=fully_restricted,
        query_gradient_contract_invocations=grouped_query_contracts,
        scalar_head_query_gradient_contract_invocations=scalar_query_contracts,
        full_domain_query_gradient_contract_invocations=(batch_count * key_value_head.extent * all_tile_pairs * 3),
        full_domain_scalar_head_query_gradient_contract_invocations=full_domain_scalar_query_contracts,
        key_value_gradient_contract_invocations=grouped_key_value_contracts,
        scalar_head_key_value_contract_invocations=scalar_key_value_contracts,
        full_domain_scalar_head_key_value_contract_invocations=full_domain_scalar_key_value_contracts,
        packed_query_rows=packed_query_rows,
        peak_score_tile_elements=packed_query_rows * schedule.key_value_tile_size,
        peak_query_tile_elements=packed_query_rows * query_feature.extent,
        key_value_gradient_accumulator_elements=2 * schedule.key_value_tile_size * query_feature.extent,
    )


def derive_streaming_attention_backward_fusion_plan(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
    *,
    local_capacity_bytes: int,
) -> SharedReverseFusionPlan:
    """Test a five-Contract owner-computes traversal over the reverse tile relation.

    The relation connects query-gradient and key/value-gradient owners. The
    generic fusion planner rejects a connected component when its deterministic
    accumulator frontier cannot remain local; it never inserts atomics or
    partial-gradient materialization implicitly.
    """
    query, key = program.forward.region.producer.inputs
    query_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key.axes if axis.label == AttentionScoreAxis.KEY.value)
    batch_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.BATCH.value)
    query_feature = next(axis for axis in query.axes if axis.label == "key_feature")
    key_value_head = next(axis for axis in key.axes if axis.label == "key_value_head")
    if program.accumulation_dtype is not DType.FP32:
        raise ValueError("the first fused reverse ownership model requires FP32 accumulators")
    if query_axis.extent % schedule.query_tile_size or key_axis.extent % schedule.key_value_tile_size:
        raise ValueError("fused reverse ownership requires tile-aligned query and key domains")

    query_tiles = query_axis.extent // schedule.query_tile_size
    key_tiles = key_axis.extent // schedule.key_value_tile_size
    query_owners = TaskFamily(
        "query-gradient owners",
        (
            TaskAxis("batch", batch_axis.extent),
            TaskAxis("key_value_head", key_value_head.extent),
            TaskAxis("query_tile", query_tiles),
        ),
    )
    key_value_owners = TaskFamily(
        "key/value-gradient owners",
        (
            TaskAxis("batch", batch_axis.extent),
            TaskAxis("key_value_head", key_value_head.extent),
            TaskAxis("key_value_tile", key_tiles),
        ),
    )
    pairs = []
    for batch in range(batch_axis.extent):
        for head in range(key_value_head.extent):
            for query_tile in range(query_tiles):
                query_stop = (query_tile + 1) * schedule.query_tile_size
                for key_value_tile in range(key_tiles):
                    key_start = key_value_tile * schedule.key_value_tile_size
                    if (
                        schedule.domain_traversal is StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR
                        and key_start >= query_stop
                    ):
                        continue
                    pairs.append(((batch, head, query_tile), (batch, head, key_value_tile)))
    relation = TaskRelation.from_pairs(query_owners, key_value_owners, tuple(pairs))
    packed_query_rows = schedule.query_tile_size * schedule.query_heads_per_key_value_tile
    return plan_shared_producer_reverse_fusion(
        relation,
        source_accumulator_elements=packed_query_rows * query_feature.extent,
        target_accumulator_elements=2 * schedule.key_value_tile_size * query_feature.extent,
        transient_edge_elements=4 * packed_query_rows * schedule.key_value_tile_size,
        accumulator_bytes_per_element=4,
        local_capacity_bytes=local_capacity_bytes,
        baseline_contracts_per_edge=7,
        fused_contracts_per_edge=5,
    )


def execute_streaming_attention_backward(
    program: StreamingAttentionBackwardProgram,
    inputs: dict[str, np.ndarray],
    forward_execution: StreamingAttentionExecution,
    output_cotangent: np.ndarray,
) -> StreamingAttentionBackwardExecution:
    """Execute the reference reverse, recomputing primal state for the generic finalizer VJP."""
    forward = program.forward
    region = forward.region
    values = _validated_inputs(forward.source, inputs)
    query_value, key_value = region.producer.inputs
    value_value = region.consumer.inputs[1]
    query = values[query_value.name]
    key = values[key_value.name]
    value = values[value_value.name]
    if output_cotangent.shape != region.finalizer.output.shape:
        raise ValueError(
            f"output cotangent has shape {output_cotangent.shape}, expected {region.finalizer.output.shape}"
        )
    if forward_execution.output.shape != region.finalizer.output.shape:
        raise ValueError("saved streaming output has the wrong shape")
    if forward_execution.row_max.shape != region.fold_state.row_max.shape:
        raise ValueError("saved row maximum has the wrong shape")
    if forward_execution.row_sum_exp.shape != region.fold_state.row_sum_exp.shape:
        raise ValueError("saved row sum-exp has the wrong shape")

    query_axis = next(axis for axis in query_value.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key_value.axes if axis.label == AttentionScoreAxis.KEY.value)
    row_axes = region.fold_state.row_max.axes
    score_axes = region.producer.output.axes
    query_cotangent = np.zeros(query_value.shape, dtype=np.float32)
    key_cotangent = np.zeros(key_value.shape, dtype=np.float32)
    value_cotangent = np.zeros(value_value.shape, dtype=np.float32)
    unnormalized_state = execute_tensor_program(
        TensorProgram(
            inputs=forward.source.inputs,
            operations=tuple(
                operation for operation in forward.source.operations if operation.output != region.finalizer.output
            ),
            outputs=(region.consumer.output, region.fold_state.row_sum_exp),
        ),
        inputs,
    )
    normalized_output = execute_tensor_program(
        TensorProgram(
            inputs=program.pullback.normalized_output.inputs,
            operations=(program.pullback.normalized_output,),
            outputs=(program.pullback.normalized_output.output,),
        ),
        {
            region.consumer.output.name: unnormalized_state[region.consumer.output.name],
            region.fold_state.row_sum_exp.name: unnormalized_state[region.fold_state.row_sum_exp.name],
        },
    )[program.pullback.normalized_output.output.name]
    finalizer_vjp_bindings = {program.output_cotangent.name: output_cotangent}
    if program.pullback.normalized_output.output in program.pullback.post_normalization_vjp.inputs:
        finalizer_vjp_bindings[program.pullback.normalized_output.output.name] = normalized_output
    normalized_output_cotangent = execute_tensor_program(
        TensorProgram(
            inputs=program.pullback.post_normalization_vjp.inputs,
            operations=(program.pullback.post_normalization_vjp,),
            outputs=(program.pullback.post_normalization_vjp.output,),
        ),
        finalizer_vjp_bindings,
    )[program.pullback.post_normalization_vjp.output.name]
    output_dot_state = execute_tensor_program(
        TensorProgram(
            inputs=program.output_dot_map.inputs,
            operations=(program.output_dot_map, program.output_dot_fold),
            outputs=(program.output_dot_fold.output,),
        ),
        {
            program.pullback.normalized_output.output.name: normalized_output,
            program.pullback.post_normalization_vjp.output.name: normalized_output_cotangent,
        },
    )[program.output_dot_fold.output.name]

    for query_start in range(0, query_axis.extent, forward.schedule.query_tile_size):
        query_stop = min(query_start + forward.schedule.query_tile_size, query_axis.extent)
        query_slices = {query_axis: slice(query_start, query_stop)}
        query_tile = _slice_array(query, query_value, query_slices).astype(np.float32)
        normalized_output_cotangent_tile = _slice_array(
            normalized_output_cotangent,
            program.pullback.post_normalization_vjp.output,
            query_slices,
        ).astype(np.float32)
        row_slices = tuple(slice(query_start, query_stop) if axis == query_axis else slice(None) for axis in row_axes)
        output_dot_slices = tuple(
            slice(query_start, query_stop) if axis == query_axis else slice(None)
            for axis in program.output_dot_fold.output.axes
        )
        row_max = forward_execution.row_max[row_slices]
        row_sum_exp = forward_execution.row_sum_exp[row_slices]
        output_dot = output_dot_state[output_dot_slices]
        query_gradient_tile = np.zeros(query_tile.shape, dtype=np.float32)

        def reverse_tile_data(
            key_start: int,
            current_query_start: int,
            current_query_stop: int,
            current_query_tile: np.ndarray,
            current_row_max: np.ndarray,
            current_row_sum_exp: np.ndarray,
            current_normalized_output_cotangent: np.ndarray,
        ):
            key_stop = min(key_start + forward.schedule.key_value_tile_size, key_axis.extent)
            tile_slices = {
                query_axis: slice(current_query_start, current_query_stop),
                key_axis: slice(key_start, key_stop),
            }
            key_tile = _slice_array(key, key_value, tile_slices).astype(np.float32)
            value_tile = _slice_array(value, value_value, tile_slices).astype(np.float32)
            raw_score = _contract(region.producer, (current_query_tile, key_tile))
            score_bindings = {region.producer.output.name: raw_score}
            for score_input in region.score_map.inputs[1:]:
                score_bindings[score_input.name] = _slice_array(
                    values[score_input.name],
                    score_input,
                    tile_slices,
                )
            mapped_score = _evaluate_map(region.score_map, score_bindings)
            exponential = np.exp(mapped_score - _align_array(current_row_max, row_axes, score_axes))
            probability = exponential / _align_array(current_row_sum_exp, row_axes, score_axes)
            key_head_map = region.producer.index_maps_for_input(1)[0]
            head_indices = key_head_map.indices()
            key_expanded = np.take(key_tile, head_indices, axis=key_value.axes.index(key_head_map.operand_axis))
            value_expanded = np.take(value_tile, head_indices, axis=value_value.axes.index(key_head_map.operand_axis))
            d_probability = np.einsum("bqhv,bkhv->bhqk", current_normalized_output_cotangent, value_expanded)
            return (
                key_stop,
                score_bindings,
                mapped_score,
                exponential,
                probability,
                head_indices,
                key_expanded,
                d_probability,
            )

        global_d_sum = None
        global_d_maximum = None
        global_tie_count = None
        if program.maximum_vjp is StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT:
            row_shape = (*_align_array(row_sum_exp, row_axes, score_axes).shape[:-1], 1)
            global_d_sum = np.zeros(row_shape, dtype=np.float32)
            global_tie_count = np.zeros(row_shape, dtype=np.int32)
            aligned_sum = _align_array(row_sum_exp, row_axes, score_axes)
            maximum = _align_array(row_max, row_axes, score_axes)
            for key_start in range(0, key_axis.extent, forward.schedule.key_value_tile_size):
                _, _, mapped_score, exponential, _, _, _, d_probability = reverse_tile_data(
                    key_start,
                    query_start,
                    query_stop,
                    query_tile,
                    row_max,
                    row_sum_exp,
                    normalized_output_cotangent_tile,
                )
                global_d_sum += np.sum(
                    -d_probability * exponential / np.square(aligned_sum),
                    axis=-1,
                    keepdims=True,
                )
                global_tie_count += np.sum(mapped_score == maximum, axis=-1, keepdims=True)
            global_d_maximum = np.zeros(row_shape, dtype=np.float32)
            for key_start in range(0, key_axis.extent, forward.schedule.key_value_tile_size):
                _, _, _, exponential, _, _, _, d_probability = reverse_tile_data(
                    key_start,
                    query_start,
                    query_stop,
                    query_tile,
                    row_max,
                    row_sum_exp,
                    normalized_output_cotangent_tile,
                )
                d_centered = (d_probability / aligned_sum + global_d_sum) * exponential
                global_d_maximum -= np.sum(d_centered, axis=-1, keepdims=True)
            if np.any(global_tie_count <= 0):
                raise ValueError("JAX maximum VJP has a row without a maximum tie")

        for key_start in range(0, key_axis.extent, forward.schedule.key_value_tile_size):
            (
                key_stop,
                score_bindings,
                mapped_score,
                exponential,
                probability,
                head_indices,
                key_expanded,
                d_probability,
            ) = reverse_tile_data(
                key_start,
                query_start,
                query_stop,
                query_tile,
                row_max,
                row_sum_exp,
                normalized_output_cotangent_tile,
            )
            if program.maximum_vjp is StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT:
                assert global_d_sum is not None
                assert global_d_maximum is not None
                assert global_tie_count is not None
                aligned_sum = _align_array(row_sum_exp, row_axes, score_axes)
                d_centered = (d_probability / aligned_sum + global_d_sum) * exponential
                maximum = _align_array(row_max, row_axes, score_axes)
                maximum_ties = mapped_score == maximum
                d_mapped_score = d_centered + maximum_ties * (global_d_maximum / global_tie_count)
            else:
                d_mapped_score = probability * (d_probability - output_dot[:, :, :, None].transpose(0, 2, 1, 3))
            score_vjp_bindings = dict(score_bindings)
            score_vjp_bindings[program.pullback.normalized_exp_pullback.output.name] = d_mapped_score
            d_raw_score = _evaluate_map(program.score_map_vjp, score_vjp_bindings)
            query_gradient_tile += np.einsum("bhqk,bkhd->bqhd", d_raw_score, key_expanded)
            d_key_expanded = np.einsum("bhqk,bqhd->bkhd", d_raw_score, query_tile)
            d_value_expanded = np.einsum(
                "bhqk,bqhv->bkhv",
                probability,
                normalized_output_cotangent_tile,
            )
            key_axis_index = key_value.axes.index(key_axis)
            key_destination = [slice(None)] * key_cotangent.ndim
            value_destination = [slice(None)] * value_cotangent.ndim
            key_destination[key_axis_index] = slice(key_start, key_stop)
            value_destination[value_value.axes.index(key_axis)] = slice(key_start, key_stop)
            for query_head, key_value_head in enumerate(head_indices):
                key_cotangent[tuple(key_destination)][:, :, key_value_head, :] += d_key_expanded[:, :, query_head, :]
                value_cotangent[tuple(value_destination)][:, :, key_value_head, :] += d_value_expanded[
                    :, :, query_head, :
                ]
        query_destination = [slice(None)] * query_cotangent.ndim
        query_destination[query_value.axes.index(query_axis)] = slice(query_start, query_stop)
        query_cotangent[tuple(query_destination)] = query_gradient_tile
    return StreamingAttentionBackwardExecution(
        query_cotangent=query_cotangent,
        key_cotangent=key_cotangent,
        value_cotangent=value_cotangent,
    )


def _normalized_output_scale(forward: NormalizedExpContractProgram[StreamingTileSchedule]) -> float:
    return normalized_exp_finalizer_scale(forward.region)
