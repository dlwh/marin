# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic task planning plus a quarantined historical dense replay planner.

``derive_composed_task_plan`` and ``physical_request_for_composed_tasks`` are
the canonical executable path.  The ``compile_reference_*`` entrypoint below
retains the old eight-skeleton replay solely for comparison tests and the
explicit FA3 oracle benchmark.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter
from math import ceil, prod

from shuttle.ir import DType
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_flow import (
    ErasedDenseFlowProgram,
    FlowContract,
    FlowDomainRestriction,
    FlowFold,
    FlowMap,
    FlowMapIteration,
    FlowValue,
    validate_erased_dense_flow,
)
from tile_lifetime.event_buffering import physical_event_realization, verify_event_realizations
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
)
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.physical_task_contract import (
    BoundedTaskCoordinate,
    CanonicalTaskPlan,
    DependenceRuntimeBinding,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    ParametricEventDerivation,
    ParametricEventRealization,
    ParametricEventTensorPlan,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskDependenceProgram,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccess,
    TaskValueAccessKind,
    TiledTask,
    physical_phase_iteration_space,
)
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    ReductionSkeleton,
    ReferenceAttentionSkeleton,
    ReferenceRegionPlan,
    RewriteExplanation,
)
from tile_lifetime.semantic_erasure import validate_reference_plan_semantic_erasure
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    ScanPrimitive,
    SelectionPrimitive,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    index_axis,
    index_binary,
    index_constant,
    primitive_inputs,
    primitive_outputs,
    serialize_scalar_expression,
    tensor_program_value_ordinals,
)

H100_GEMM_TILE_SHAPE = (128, 256, 64)
REFERENCE_STREAMING_BACKEND = "reference_online_fold_oracle"
MAXIMUM_FUSED_PHASES = 64
MAXIMUM_FUSED_REGISTERS_PER_THREAD = 128


@dataclass(frozen=True)
class _ErasedDenseRegion:
    input_view: FlowMap
    initial_projection: FlowContract
    initial_partition: FlowMap
    initial_pair_maps: tuple[FlowMap, FlowMap]
    score_contract: FlowContract
    score_map: FlowMap
    score_domain: FlowDomainRestriction
    maximum_fold: FlowFold
    center_map: FlowMap
    exponential_map: FlowMap
    sum_fold: FlowFold
    value_contract: FlowContract
    normalize_map: FlowMap
    attention_view: FlowMap
    output_contract: FlowContract
    first_add: FlowMap
    first_square: FlowMap
    first_fold: FlowFold
    first_row_finalize: FlowMap
    first_feature_scale: FlowMap
    first_normalized_scale: FlowMap
    gate_contract: FlowContract
    pairwise_map: FlowMap
    down_contract: FlowContract
    second_add: FlowMap
    second_square: FlowMap
    second_fold: FlowFold
    second_row_finalize: FlowMap
    second_feature_scale: FlowMap
    second_normalized_scale: FlowMap
    next_view: FlowMap
    next_projection: FlowContract
    next_partition: FlowMap
    next_pair_maps: tuple[FlowMap, FlowMap]


def compile_reference_erased_dense_transformer_region(
    erased: ErasedDenseFlowProgram,
    *,
    row_scale_placement: RowScalePlacement,
) -> ReferenceRegionPlan:
    """Build the historical eight-skeleton reference replay plan."""
    validate_erased_dense_flow(erased)
    region = _recover_erased_region(erased.operations)
    plan = replace(
        _plan_erased_region(region, row_scale_placement=row_scale_placement),
        semantic_erasure_report=erased.report,
    )
    validate_reference_plan_semantic_erasure(plan)
    return plan


def _recover_erased_region(operations) -> _ErasedDenseRegion:
    producer_by_value = {output.name: operation for operation in operations for output in _operation_outputs(operation)}
    consumers_by_value: dict[str, list[object]] = {}
    for operation in operations:
        for input_value in _operation_inputs(operation):
            consumers_by_value.setdefault(input_value.name, []).append(operation)

    score_domain = _one(
        (operation for operation in operations if isinstance(operation, FlowDomainRestriction)),
        "normalized weighted Fold domain restriction",
    )
    score_map = _producer(producer_by_value, score_domain.input, FlowMap, "score Map")
    score_contract = _producer(producer_by_value, score_map.inputs[0], FlowContract, "score Contract")
    maximum_fold = _one(
        (
            operation
            for operation in consumers_by_value.get(score_domain.output.name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "maximum"
        ),
        "maximum Fold",
    )
    center_map = _one(
        (
            operation
            for operation in consumers_by_value.get(score_domain.output.name, ())
            if isinstance(operation, FlowMap) and maximum_fold.output in operation.inputs
        ),
        "center Map",
    )
    exponential_map = _one(
        (
            operation
            for operation in consumers_by_value.get(center_map.outputs[0].name, ())
            if isinstance(operation, FlowMap)
            and any(expression.kind is ScalarExpressionKind.EXP for expression in operation.expressions)
        ),
        "exponential Map",
    )
    sum_fold = _one(
        (
            operation
            for operation in consumers_by_value.get(exponential_map.outputs[0].name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "sum"
        ),
        "sum Fold",
    )
    value_contract = _one(
        (
            operation
            for operation in consumers_by_value.get(exponential_map.outputs[0].name, ())
            if isinstance(operation, FlowContract)
        ),
        "weighted-value Contract",
    )
    normalize_map = _one(
        (
            operation
            for operation in consumers_by_value.get(value_contract.output.name, ())
            if isinstance(operation, FlowMap) and sum_fold.output in operation.inputs
        ),
        "normalized weighted Fold finalization",
    )

    initial_pair_q = _producer(producer_by_value, score_contract.inputs[0], FlowMap, "query pair Map")
    initial_pair_k = _producer(producer_by_value, score_contract.inputs[1], FlowMap, "key pair Map")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in (initial_pair_q, initial_pair_k)):
        raise ValueError("score Contract inputs are not produced by adjacent-pair Maps")
    query_partition_output = _one(
        (
            input_value
            for input_value in initial_pair_q.inputs
            if isinstance(producer_by_value.get(input_value.name), FlowMap)
            and producer_by_value[input_value.name].iteration is FlowMapIteration.PARTITION
        ),
        "query projection partition output",
    )
    key_partition_output = _one(
        (
            input_value
            for input_value in initial_pair_k.inputs
            if isinstance(producer_by_value.get(input_value.name), FlowMap)
            and producer_by_value[input_value.name].iteration is FlowMapIteration.PARTITION
        ),
        "key projection partition output",
    )
    initial_partition = _producer(producer_by_value, query_partition_output, FlowMap, "initial partition Map")
    if producer_by_value.get(key_partition_output.name) is not initial_partition:
        raise ValueError("query and key pair Maps do not share one projection partition")
    initial_projection = _producer(
        producer_by_value,
        initial_partition.inputs[0],
        FlowContract,
        "initial projection Contract",
    )
    input_view = _producer(producer_by_value, initial_projection.inputs[0], FlowMap, "initial input view")

    attention_view = _one(
        (
            operation
            for operation in consumers_by_value.get(normalize_map.outputs[0].name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.VIEW
        ),
        "attention output view",
    )
    output_contract = _one(
        (
            operation
            for operation in consumers_by_value.get(attention_view.outputs[0].name, ())
            if isinstance(operation, FlowContract)
        ),
        "attention output Contract",
    )
    first = _recover_fold_scale_contract(output_contract, producer_by_value, consumers_by_value)
    gate_contract = _consumer_contract(first.normalized_scale.outputs[0], consumers_by_value, "first scaled Contract")
    pairwise_map = _one(
        (
            operation
            for operation in consumers_by_value.get(gate_contract.output.name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.ADJACENT_PAIR
        ),
        "expanded projection pairwise Map",
    )
    down_contract = _consumer_contract(pairwise_map.outputs[0], consumers_by_value, "pairwise consumer Contract")
    second = _recover_fold_scale_contract(down_contract, producer_by_value, consumers_by_value)
    next_view = _one(
        (
            operation
            for operation in consumers_by_value.get(second.normalized_scale.outputs[0].name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.VIEW
        ),
        "following projection input view",
    )
    next_projection = _consumer_contract(next_view.outputs[0], consumers_by_value, "following projection Contract")
    next_partition = _one(
        (
            operation
            for operation in consumers_by_value.get(next_projection.output.name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.PARTITION
        ),
        "following projection partition",
    )
    next_pairs = tuple(
        operation
        for output in next_partition.outputs
        for operation in consumers_by_value.get(output.name, ())
        if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.ADJACENT_PAIR
    )
    if len(next_pairs) != 2:
        raise ValueError(f"following partition must feed two adjacent-pair Maps, found {len(next_pairs)}")
    _validate_generic_dataflow(
        initial_projection,
        initial_partition,
        (initial_pair_q, initial_pair_k),
        score_contract,
        score_map,
        score_domain,
        maximum_fold,
        exponential_map,
        sum_fold,
        value_contract,
        normalize_map,
        pairwise_map,
        next_projection,
        next_partition,
        next_pairs,
    )
    return _ErasedDenseRegion(
        input_view,
        initial_projection,
        initial_partition,
        (initial_pair_q, initial_pair_k),
        score_contract,
        score_map,
        score_domain,
        maximum_fold,
        center_map,
        exponential_map,
        sum_fold,
        value_contract,
        normalize_map,
        attention_view,
        output_contract,
        first.add,
        first.square,
        first.fold,
        first.row_finalize,
        first.feature_scale,
        first.normalized_scale,
        gate_contract,
        pairwise_map,
        down_contract,
        second.add,
        second.square,
        second.fold,
        second.row_finalize,
        second.feature_scale,
        second.normalized_scale,
        next_view,
        next_projection,
        next_partition,
        next_pairs,
    )


@dataclass(frozen=True)
class _FlowFoldScaleContract:
    add: FlowMap
    square: FlowMap
    fold: FlowFold
    row_finalize: FlowMap
    feature_scale: FlowMap
    normalized_scale: FlowMap


def _recover_fold_scale_contract(
    producer_contract: FlowContract,
    producer_by_value: dict[str, object],
    consumers_by_value: dict[str, list[object]],
) -> _FlowFoldScaleContract:
    add = _one(
        (
            operation
            for operation in consumers_by_value.get(producer_contract.output.name, ())
            if isinstance(operation, FlowMap) and _single_expression_kind(operation, ScalarExpressionKind.ADD)
        ),
        "tile-local add after Contract",
    )
    square = _one(
        (
            operation
            for operation in consumers_by_value.get(add.outputs[0].name, ())
            if isinstance(operation, FlowMap) and _is_square_map(operation)
        ),
        "pointwise square before Fold",
    )
    fold = _one(
        (
            operation
            for operation in consumers_by_value.get(square.outputs[0].name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "sum"
        ),
        "sum Fold after square",
    )
    row_finalize = _one(
        (operation for operation in consumers_by_value.get(fold.output.name, ()) if isinstance(operation, FlowMap)),
        "row-scalar Fold finalization",
    )
    feature_scale = _one(
        (
            operation
            for operation in consumers_by_value.get(add.outputs[0].name, ())
            if isinstance(operation, FlowMap)
            and operation is not square
            and _single_expression_kind(operation, ScalarExpressionKind.MULTIPLY)
        ),
        "feature-scale Map",
    )
    normalized_scale = _one(
        (
            operation
            for operation in consumers_by_value.get(feature_scale.outputs[0].name, ())
            if isinstance(operation, FlowMap) and row_finalize.outputs[0] in operation.inputs
        ),
        "row-scale Map",
    )
    if producer_by_value.get(normalized_scale.outputs[0].name) is not normalized_scale:
        raise ValueError("row-scale Map output is not present in generic Flow dataflow")
    return _FlowFoldScaleContract(add, square, fold, row_finalize, feature_scale, normalized_scale)


def _operation_inputs(operation) -> tuple[FlowValue, ...]:
    if isinstance(operation, FlowFold):
        return (operation.input,)
    if isinstance(operation, FlowDomainRestriction):
        return (operation.input,)
    return operation.inputs


def _operation_outputs(operation) -> tuple[FlowValue, ...]:
    if isinstance(operation, FlowContract):
        return (operation.output,)
    if isinstance(operation, FlowFold):
        return (operation.output,)
    if isinstance(operation, FlowDomainRestriction):
        return (operation.output,)
    return operation.outputs


def _producer(producers: dict[str, object], value: FlowValue, expected_type, description: str):
    operation = producers.get(value.name)
    if not isinstance(operation, expected_type):
        actual = "none" if operation is None else type(operation).__name__
        raise ValueError(f"{description} producer is {actual}, expected {expected_type.__name__}")
    return operation


def _one(candidates, description: str):
    matches = tuple(candidates)
    if len(matches) != 1:
        raise ValueError(f"expected one {description}, found {len(matches)}")
    return matches[0]


def _consumer_contract(
    value: FlowValue,
    consumers_by_value: dict[str, list[object]],
    description: str,
) -> FlowContract:
    return _one(
        (operation for operation in consumers_by_value.get(value.name, ()) if isinstance(operation, FlowContract)),
        description,
    )


def _single_expression_kind(operation: FlowMap, kind: ScalarExpressionKind) -> bool:
    return len(operation.expressions) == 1 and operation.expressions[0].kind is kind


def _is_square_map(operation: FlowMap) -> bool:
    if not _single_expression_kind(operation, ScalarExpressionKind.MULTIPLY):
        return False
    expression = operation.expressions[0]
    return len(expression.operands) == 2 and expression.operands[0] == expression.operands[1]


def _validate_generic_dataflow(
    initial_projection: FlowContract,
    initial_partition: FlowMap,
    initial_pairs: tuple[FlowMap, FlowMap],
    score_contract: FlowContract,
    score_map: FlowMap,
    score_domain: FlowDomainRestriction,
    maximum_fold: FlowFold,
    exponential_map: FlowMap,
    sum_fold: FlowFold,
    value_contract: FlowContract,
    normalize_map: FlowMap,
    pairwise_map: FlowMap,
    next_projection: FlowContract,
    next_partition: FlowMap,
    next_pairs: tuple[FlowMap, FlowMap],
) -> None:
    if initial_partition.iteration is not FlowMapIteration.PARTITION:
        raise ValueError("initial contraction is not followed by a generic partition Map")
    if initial_partition.inputs != (initial_projection.output,):
        raise ValueError("initial partition does not consume the initial Contract")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in initial_pairs):
        raise ValueError("initial pair transforms are not generic adjacent-pair Maps")
    if score_contract.inputs[:1] != (initial_pairs[0].outputs[0],):
        raise ValueError("score Contract does not consume the first pairwise output")
    if score_map.inputs != (score_contract.output,) or score_domain.input != score_map.outputs[0]:
        raise ValueError("score Map and DomainRestriction are disconnected")
    if maximum_fold.input != score_domain.output or maximum_fold.reducer.value != "maximum":
        raise ValueError("normalized exponential does not begin with a maximum Fold")
    if sum_fold.input != exponential_map.outputs[0] or sum_fold.reducer.value != "sum":
        raise ValueError("normalized exponential does not contain a sum Fold")
    if value_contract.inputs[0] != exponential_map.outputs[0]:
        raise ValueError("value Contract does not consume normalized-exponential contributions")
    if normalize_map.inputs != (value_contract.output, sum_fold.output):
        raise ValueError("normalized weighted Fold does not finalize by its sum state")
    if pairwise_map.iteration is not FlowMapIteration.ADJACENT_PAIR or len(pairwise_map.expressions) != 1:
        raise ValueError("expanded projection is not finalized by one generic adjacent-pair Map")
    if next_partition.inputs != (next_projection.output,) or next_partition.iteration is not FlowMapIteration.PARTITION:
        raise ValueError("following projection does not feed a generic partition Map")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in next_pairs):
        raise ValueError("following pair transforms are not generic adjacent-pair Maps")


def _plan_erased_region(region: _ErasedDenseRegion, *, row_scale_placement: RowScalePlacement) -> ReferenceRegionPlan:
    initial_projection = _projection_pairwise_linear_contract(
        region.initial_projection,
        region.initial_partition,
        region.initial_pair_maps,
        input_value=region.initial_projection.inputs[0].name,
        input_layout="bsh_contiguous",
        cluster_shape=_projection_cluster_shape(region.initial_projection.output.shape[-1]),
    )
    streaming = _reference_attention_oracle_skeleton(region)

    first_scaled = region.first_feature_scale.outputs[0]
    first_partials = f"{region.first_square.inputs[0].name}.fold_partials"
    first_row_scalar = region.first_row_finalize.outputs[0]
    output_contract = _contract_maps_fold(
        region.output_contract,
        region.first_add,
        region.first_feature_scale,
        region.first_square,
        partials=first_partials,
    )
    first_reduction = _fold_reduction(region.first_fold, region.first_row_finalize, first_partials)
    gate_contract = GemmSkeleton(
        name=f"contract_{row_scale_placement.value}_row_scale_pairwise_map",
        input=first_scaled.name,
        weight=region.gate_contract.inputs[1].name,
        output=region.pairwise_map.outputs[0].name,
        shape=_gemm_shape(region.gate_contract),
        accumulation_dtype=region.gate_contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=_prologue_cluster_shape(region.gate_contract.output.shape[1]),
        pingpong=False,
        prologue=_row_scale_preparation(
            first_scaled,
            first_row_scalar,
            region.first_normalized_scale.outputs[0],
            row_scale_placement,
        ),
        epilogue=(
            *_row_scale_finalization(region.gate_contract.output, first_row_scalar, row_scale_placement),
            Attachment(
                "pairwise_map",
                AttachmentSite.GEMM_EPILOGUE,
                (region.gate_contract.output.name,),
                (region.pairwise_map.outputs[0].name,),
                (("expression_ast", serialize_scalar_expression(region.pairwise_map.expressions[0])),),
            ),
        ),
    )

    second_scaled = region.second_feature_scale.outputs[0]
    second_partials = f"{region.second_square.inputs[0].name}.fold_partials"
    second_row_scalar = region.second_row_finalize.outputs[0]
    down_contract = _contract_maps_fold(
        region.down_contract,
        region.second_add,
        region.second_feature_scale,
        region.second_square,
        partials=second_partials,
    )
    second_reduction = _fold_reduction(region.second_fold, region.second_row_finalize, second_partials)
    next_contract = _projection_pairwise_linear_contract(
        region.next_projection,
        region.next_partition,
        region.next_pair_maps,
        input_value=second_scaled.name,
        input_layout="row_major_mk",
        cluster_shape=_projection_cluster_shape(region.next_projection.output.shape[-1]),
        prologue=(
            *_row_scale_preparation(
                second_scaled,
                second_row_scalar,
                region.second_normalized_scale.outputs[0],
                row_scale_placement,
            ),
            Attachment(
                "view",
                AttachmentSite.GEMM_PROLOGUE,
                (
                    (
                        region.second_normalized_scale.outputs[0].name
                        if row_scale_placement is RowScalePlacement.CONSUMER_PROLOGUE
                        else second_scaled.name
                    ),
                ),
                (region.next_view.outputs[0].name,),
            ),
        ),
        epilogue_prefix=_row_scale_finalization(
            region.next_projection.output,
            second_row_scalar,
            row_scale_placement,
        ),
    )

    skeletons = (
        initial_projection,
        streaming,
        output_contract,
        first_reduction,
        gate_contract,
        down_contract,
        second_reduction,
        next_contract,
    )
    materializations = _materializations(
        region,
        first_scaled,
        first_partials,
        second_scaled,
        second_partials,
        row_scale_placement,
    )
    rewrites = (
        _explanation("attach_partition_and_pairwise_linear_maps"),
        _explanation("derive_streaming_normalized_weighted_fold"),
        _explanation("place_maps_and_partial_folds_around_contracts"),
    )
    return ReferenceRegionPlan(skeletons, materializations, rewrites)


def _projection_pairwise_linear_contract(
    contract: FlowContract,
    partition: FlowMap,
    pair_maps: tuple[FlowMap, FlowMap],
    *,
    input_value: str,
    input_layout: str,
    cluster_shape: tuple[int, int, int],
    prologue: tuple[Attachment, ...] = (),
    epilogue_prefix: tuple[Attachment, ...] = (),
) -> GemmSkeleton:
    batch, sequence, hidden = contract.inputs[0].shape
    pair_attachments = tuple(
        Attachment(
            "pairwise_linear_map",
            AttachmentSite.GEMM_EPILOGUE,
            tuple(value.name for value in operation.inputs),
            tuple(value.name for value in operation.outputs),
            tuple(
                (f"expression_ast.{index}", serialize_scalar_expression(expression))
                for index, expression in enumerate(operation.expressions)
            ),
        )
        for operation in pair_maps
    )
    return GemmSkeleton(
        name="contract_partition_pairwise_linear_maps",
        input=input_value,
        weight=contract.inputs[1].name,
        output=contract.output.name,
        shape=(batch * sequence, contract.inputs[1].shape[1], hidden),
        accumulation_dtype=contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout=input_layout,
        output_layout="fa3_bshd_last_dimension_contiguous",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=cluster_shape,
        pingpong=False,
        prologue=prologue,
        epilogue=(
            *epilogue_prefix,
            Attachment(
                "partition",
                AttachmentSite.GEMM_EPILOGUE,
                (contract.output.name,),
                tuple(value.name for value in partition.outputs),
                partition.attributes,
            ),
            *pair_attachments,
        ),
    )


def _reference_attention_oracle_skeleton(region: _ErasedDenseRegion) -> ReferenceAttentionSkeleton:
    """Adapt generic Flow to the fixed fields required by the FA3 oracle API."""
    query, key = region.score_contract.inputs
    value = region.value_contract.inputs[1]
    output = region.normalize_map.outputs[0]
    scale = _reference_score_scale(region.score_map.expressions[0], region.score_contract.output.name)
    causal = region.score_domain.predicate.kind is ScalarExpressionKind.LESS_EQUAL
    head_dimension = query.shape[-1]
    query_block_size = 192 if head_dimension == 64 else 128
    key_value_block_size = 128 if causal else (192 if head_dimension == 64 else 176)
    query_heads = query.shape[2]
    key_value_heads = key.shape[2]
    return ReferenceAttentionSkeleton(
        name="streaming_normalized_weighted_fold",
        query=query.name,
        key=key.name,
        value=value.name,
        output=output.name,
        score_value=region.score_domain.output.name,
        probability_value=region.exponential_map.outputs[0].name,
        query_block_size=query_block_size,
        key_value_block_size=key_value_block_size,
        head_dimension=head_dimension,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        causal=causal,
        scale=scale,
        backend=REFERENCE_STREAMING_BACKEND,
        input_layout="fa3_bshd_last_dimension_contiguous",
        output_layout="bshd_contiguous",
        pipeline_stages=2,
        producer_threads=32,
        consumer_threads=(query_block_size // 64) * 128,
        pack_gqa=query_heads != key_value_heads,
        mma_pv_is_rs=causal,
        intra_warpgroup_overlap=True,
        persistent_scheduler=True,
        register_estimate=168 if head_dimension == 128 else None,
        online_state=(
            region.maximum_fold.output.name,
            region.sum_fold.output.name,
            region.value_contract.output.name,
        ),
        attachments=(
            Attachment(
                "score_map",
                AttachmentSite.ATTENTION_SCORE_TRANSFORM,
                (region.score_contract.output.name,),
                (region.score_map.outputs[0].name,),
            ),
            Attachment(
                "domain_restriction",
                AttachmentSite.ATTENTION_SCORE_TRANSFORM,
                (region.score_map.outputs[0].name,),
                (region.score_domain.output.name,),
            ),
            Attachment(
                "online_fold_update",
                AttachmentSite.ATTENTION_ONLINE_UPDATE,
                (region.score_domain.output.name, value.name),
                (
                    region.maximum_fold.output.name,
                    region.sum_fold.output.name,
                    region.value_contract.output.name,
                ),
            ),
            Attachment(
                "fold_finalize",
                AttachmentSite.ATTENTION_OUTPUT_TRANSFORM,
                (region.value_contract.output.name, region.sum_fold.output.name),
                (output.name,),
            ),
        ),
    )


def _contract_maps_fold(
    contract: FlowContract,
    add: FlowMap,
    feature_scale: FlowMap,
    square: FlowMap,
    *,
    partials: str,
) -> GemmSkeleton:
    contract_output = contract.output
    forwarded = next(value for value in add.inputs if value != contract_output)
    feature_vector = next(value for value in feature_scale.inputs if value != square.inputs[0])
    return GemmSkeleton(
        name="contract_maps_and_fold_partials",
        input=contract.inputs[0].name,
        weight=contract.inputs[1].name,
        output=feature_scale.outputs[0].name,
        shape=_gemm_shape(contract),
        accumulation_dtype=contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=(1, 1, 1),
        pingpong=False,
        epilogue=(
            Attachment(
                "add",
                AttachmentSite.GEMM_EPILOGUE,
                (contract_output.name, forwarded.name),
                (add.outputs[0].name,),
            ),
            Attachment(
                "multiply",
                AttachmentSite.GEMM_EPILOGUE,
                (add.outputs[0].name, feature_vector.name),
                (feature_scale.outputs[0].name,),
                (("input.1_delivery", "row"),),
            ),
            Attachment(
                "partial_sum_square",
                AttachmentSite.GEMM_EPILOGUE,
                (square.inputs[0].name,),
                (partials,),
            ),
        ),
    )


def _fold_reduction(fold: FlowFold, finalize: FlowMap, partials: str) -> ReductionSkeleton:
    return ReductionSkeleton(
        name="combine_fold_partials",
        input=partials,
        output=finalize.outputs[0].name,
        operator=_render_scalar_expression(finalize.expressions[0], {fold.output.name: "sum"}),
        reduction_dtype=fold.accumulation_dtype,
    )


def _row_scale_preparation(
    input_value: FlowValue,
    row_scalar: FlowValue,
    output_value: FlowValue,
    placement: RowScalePlacement,
) -> tuple[Attachment, ...]:
    if placement is not RowScalePlacement.CONSUMER_PROLOGUE:
        return ()
    return (
        Attachment(
            "scale_row",
            AttachmentSite.GEMM_PROLOGUE,
            (input_value.name, row_scalar.name),
            (output_value.name,),
        ),
    )


def _row_scale_finalization(
    contract_output: FlowValue,
    row_scalar: FlowValue,
    placement: RowScalePlacement,
) -> tuple[Attachment, ...]:
    if placement is RowScalePlacement.CONSUMER_PROLOGUE:
        return ()
    return (
        Attachment(
            "scale_row",
            AttachmentSite.GEMM_EPILOGUE,
            (contract_output.name, row_scalar.name),
            (contract_output.name,),
        ),
    )


def _materializations(
    region: _ErasedDenseRegion,
    first_scaled: FlowValue,
    first_partials: str,
    second_scaled: FlowValue,
    second_partials: str,
    placement: RowScalePlacement,
) -> tuple[MaterializationRecord, ...]:
    output: list[MaterializationRecord] = [
        _record(
            region.input_view.outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=region.input_view.inputs[0].name,
        ),
        *_projection_records(region.initial_projection, region.initial_partition, region.initial_pair_maps),
        *_streaming_records(region),
        _record(
            region.attention_view.outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=region.attention_view.inputs[0].name,
        ),
        _record(region.output_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.first_add.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.first_normalized_scale.outputs[0], _scale_disposition(placement)),
        _record(first_scaled, MaterializationDisposition.MATERIALIZE),
        MaterializationRecord(
            first_partials,
            _partial_shape(region.first_square.inputs[0]),
            region.first_fold.accumulation_dtype,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "partial Fold buffer",
        ),
        _record(region.gate_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.pairwise_map.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.down_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.second_add.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.second_normalized_scale.outputs[0], _scale_disposition(placement)),
        _record(second_scaled, MaterializationDisposition.MATERIALIZE),
        MaterializationRecord(
            second_partials,
            _partial_shape(region.second_square.inputs[0]),
            region.second_fold.accumulation_dtype,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "partial Fold buffer",
        ),
        _record(region.next_view.outputs[0], MaterializationDisposition.ALIAS, alias_of=region.next_view.inputs[0].name),
        *_projection_records(region.next_projection, region.next_partition, region.next_pair_maps),
    ]
    return tuple(output)


def _projection_records(
    contract: FlowContract,
    partition: FlowMap,
    pair_maps: tuple[FlowMap, FlowMap],
) -> tuple[MaterializationRecord, ...]:
    return (
        _record(contract.output, MaterializationDisposition.MATERIALIZE),
        *(_record(value, MaterializationDisposition.EPILOGUE_ONLY) for value in partition.outputs[:2]),
        _record(
            pair_maps[0].outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=contract.output.name,
        ),
        _record(
            pair_maps[1].outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=contract.output.name,
        ),
        _record(partition.outputs[2], MaterializationDisposition.ALIAS, alias_of=contract.output.name),
    )


def _streaming_records(region: _ErasedDenseRegion) -> tuple[MaterializationRecord, ...]:
    return (
        _record(region.score_domain.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.exponential_map.outputs[0], MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.maximum_fold.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.sum_fold.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.value_contract.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.normalize_map.outputs[0], MaterializationDisposition.MATERIALIZE),
    )


def _record(
    value: FlowValue,
    disposition: MaterializationDisposition,
    *,
    alias_of: str | None = None,
) -> MaterializationRecord:
    return MaterializationRecord(
        value.name,
        value.shape,
        value.dtype,
        disposition,
        "derived from generic Flow",
        alias_of,
    )


def _scale_disposition(placement: RowScalePlacement) -> MaterializationDisposition:
    if placement is RowScalePlacement.CONSUMER_PROLOGUE:
        return MaterializationDisposition.PROLOGUE_ONLY
    return MaterializationDisposition.EPILOGUE_ONLY


def _partial_shape(value: FlowValue) -> tuple[int, int]:
    return value.shape[0], (value.shape[1] + H100_GEMM_TILE_SHAPE[1] - 1) // H100_GEMM_TILE_SHAPE[1]


def _gemm_shape(contract: FlowContract) -> tuple[int, int, int]:
    return contract.output.shape[0], contract.output.shape[1], contract.inputs[0].shape[1]


def _projection_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 6_144 else 1, 1)


def _prologue_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 16_384 else 1, 1)


def _reference_score_scale(expression: ScalarExpression, score_name: str) -> float:
    """Recover the scalar accepted by the historical FA3 oracle interface."""
    if expression.kind is not ScalarExpressionKind.MULTIPLY:
        raise ValueError("score Map must multiply the score Contract by a scalar")
    left, right = expression.operands
    literal = right if left.input_name == score_name else left
    if literal.kind is not ScalarExpressionKind.CONSTANT or not isinstance(literal.constant, float):
        raise ValueError("score Map scale is not a floating-point literal")
    return literal.constant


def _render_scalar_expression(
    expression: ScalarExpression,
    aliases: dict[str, str],
    parent_precedence: int = 0,
) -> str:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return aliases.get(expression.input_name, expression.input_name)
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return str(expression.constant)
    if expression.kind is ScalarExpressionKind.RSQRT:
        return f"rsqrt({_render_scalar_expression(expression.operands[0], aliases)})"
    operators = {
        ScalarExpressionKind.ADD: ("+", 10),
        ScalarExpressionKind.SUBTRACT: ("-", 10),
        ScalarExpressionKind.MULTIPLY: ("*", 20),
        ScalarExpressionKind.DIVIDE: ("/", 20),
    }
    operator = operators.get(expression.kind)
    if operator is None:
        raise ValueError(f"unsupported row-finalization expression {expression.kind.value}")
    symbol, precedence = operator
    left = _render_scalar_expression(expression.operands[0], aliases, precedence)
    right = _render_scalar_expression(expression.operands[1], aliases, precedence + 1)
    rendered = f"{left} {symbol} {right}"
    return f"({rendered})" if precedence < parent_precedence else rendered


def _explanation(name: str) -> RewriteExplanation:
    return RewriteExplanation(
        name,
        True,
        ("generic Flow",),
        ("bounded tile-lifetime skeletons",),
        ("Map/Contract/Fold/DomainRestriction dataflow",),
        ("machine-checked semantic-name erasure",),
        "avoid activation-sized intermediates",
        NumericalEquivalence.ALGEBRAICALLY_EXACT,
        "finite-precision order is controlled by the selected row-scale policy",
    )


@dataclass(frozen=True)
class DenseCompositionExecutable:
    """Executable-reference boundaries for one generic tensor composition."""

    program: TensorProgram
    task_plan: CanonicalTaskPlan
    event_program: EventDataflowProgram
    physical_request: ExecutablePhysicalRequest


def derive_composed_task_plan(
    program: TensorProgram,
    *,
    tile_extent_by_axis: Mapping[int, int],
    numerical_policy: PhysicalNumericalPolicy,
    fusion_boundary_values: tuple[ProgramValue, ...] = (),
) -> CanonicalTaskPlan:
    """Fuse local generic regions and derive ownership-indexed task families."""
    operation_index = {id(operation): index for index, operation in enumerate(program.operations)}
    producer_by_value = {
        output.name: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    consumers_by_value: dict[str, list[TensorPrimitive]] = defaultdict(list)
    for operation in program.operations:
        for value in primitive_inputs(operation):
            consumers_by_value[value.name].append(operation)

    predecessors = {
        ordinal: {
            operation_index[id(producer_by_value[value.name])]
            for value in primitive_inputs(operation)
            if value.name in producer_by_value
        }
        for ordinal, operation in enumerate(program.operations)
    }
    groups: list[set[int]] = []
    assigned: set[int] = set()

    for value in fusion_boundary_values:
        producer = producer_by_value.get(value.name)
        if producer is None:
            continue
        ordinal = operation_index[id(producer)]
        if ordinal in assigned:
            continue
        groups.append({ordinal})
        assigned.add(ordinal)

    for terminal_ordinal, terminal in enumerate(program.operations):
        if not isinstance(terminal, ContractPrimitive):
            continue
        candidate = _contract_fold_bridge_candidate(
            terminal_ordinal,
            program=program,
            producer_by_value=producer_by_value,
            consumers_by_value=consumers_by_value,
            operation_index=operation_index,
            forbidden=assigned,
        )
        if candidate is None or candidate & assigned:
            continue
        if not _fusion_preserves_acyclic_quotient(groups, candidate, predecessors):
            continue
        groups.append(candidate)
        assigned.update(candidate)

    for ordinal, operation in enumerate(program.operations):
        if ordinal in assigned or not isinstance(operation, FoldPrimitive):
            continue
        group = {ordinal}
        _include_map_ancestry_until_contract(
            operation.input,
            producer_by_value=producer_by_value,
            operation_index=operation_index,
            forbidden=assigned,
            group=group,
        )
        if not _within_fusion_resource_bounds(group, program):
            group = {ordinal}
        elif not _is_dependency_convex(
            group,
            program=program,
            producer_by_value=producer_by_value,
            consumers_by_value=consumers_by_value,
            operation_index=operation_index,
        ):
            group = {ordinal}
        if not _fusion_preserves_acyclic_quotient(groups, group, predecessors):
            group = {ordinal}
        groups.append(group)
        assigned.update(group)

    for ordinal, operation in enumerate(program.operations):
        if ordinal in assigned or not isinstance(operation, ContractPrimitive):
            continue
        group = {ordinal}
        for value in operation.inputs:
            _include_map_ancestry(
                value,
                producer_by_value=producer_by_value,
                operation_index=operation_index,
                forbidden=assigned,
                group=group,
            )
        if not _within_fusion_resource_bounds(group, program):
            group = {ordinal}
        elif not _is_dependency_convex(
            group,
            program=program,
            producer_by_value=producer_by_value,
            consumers_by_value=consumers_by_value,
            operation_index=operation_index,
        ):
            group = {ordinal}
        if not _fusion_preserves_acyclic_quotient(groups, group, predecessors):
            group = {ordinal}
        groups.append(group)
        assigned.update(group)

    groups.extend({ordinal} for ordinal in range(len(program.operations)) if ordinal not in assigned)
    ordered_groups = tuple(tuple(sorted(group)) for group in sorted(groups, key=min))
    value_id = _value_ids(program)
    task_for_operation: dict[int, int] = {}
    tasks: list[TiledTask] = []
    for task_id, ordinals in enumerate(ordered_groups):
        for ordinal in ordinals:
            task_for_operation[ordinal] = task_id
        tasks.append(
            _tiled_task(
                task_id,
                ordinals,
                program,
                consumers_by_value=consumers_by_value,
                operation_index=operation_index,
                value_id=value_id,
                tile_extent_by_axis=tile_extent_by_axis,
                numerical_policy=numerical_policy,
            )
        )

    dependences = _exact_task_dependences(
        program,
        tuple(tasks),
        task_for_operation=task_for_operation,
    )
    task_plan = CanonicalTaskPlan(tuple(tasks), dependences)
    return materialize_runtime_relation_dependences(
        program,
        task_plan,
        tile_extent_by_axis=tile_extent_by_axis,
    )


@dataclass(frozen=True)
class RuntimeRelationDependenceSite:
    """One algebraic runtime Relation edge and its task-level realization."""

    relation_operation_ordinal: int
    source_value_ordinal: int
    source_task_id: int
    target_task_id: int
    dependence_program_index: int | None

    @property
    def internalized(self) -> bool:
        """Return whether fusion makes this exact dependence intra-task."""
        return self.dependence_program_index is None

    @property
    def canonical_record(self) -> dict[str, int | bool | None]:
        """Return stable provenance for this runtime-dependence proof."""
        return {
            "relation_operation": self.relation_operation_ordinal,
            "source_value": self.source_value_ordinal,
            "source_task": self.source_task_id,
            "target_task": self.target_task_id,
            "dependence_program": self.dependence_program_index,
            "internalized": self.internalized,
        }


def materialize_runtime_relation_dependences(
    program: TensorProgram,
    task_plan: CanonicalTaskPlan,
    *,
    tile_extent_by_axis: Mapping[int, int],
) -> CanonicalTaskPlan:
    """Replace tensor-indexed Relation edges with exact runtime programs."""
    task_for_operation = {ordinal: task.id for task in task_plan.tasks for ordinal in task.operation_ordinals}
    producer_for_value = {
        output: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    value_ordinals = tensor_program_value_ordinals(program)
    tasks = list(task_plan.tasks)
    staged_inputs: dict[ProgramValue, int] = {}

    def source_task_id(value: ProgramValue) -> int:
        producer = producer_for_value.get(value)
        if producer is not None:
            return task_for_operation[producer]
        if value not in program.inputs:
            raise ValueError(f"runtime Relation source {value.name!r} has no producer")
        existing = staged_inputs.get(value)
        if existing is not None:
            return existing
        task_id = len(tasks)
        owner_axes = tuple(
            TaskAxisTile(axis.id, axis.extent, min(axis.extent, tile_extent_by_axis.get(axis.id, axis.extent)))
            for axis in value.axes
        )
        structure = PhysicalTaskStructure(
            principal_kind=TaskOperationKind.INPUT_STAGE,
            phases=(PhysicalPhase(0, TaskCapability.INPUT_STAGING, (), value.shape, ()),),
            phase_edges=(),
            owner_axes=owner_axes,
            streamed_axes=(),
            reduction_axes=(),
            contraction_axes=(),
            traversal=TaskTraversal.SOURCE_ORDERED,
            element_types=(value.dtype.value,),
            layout_classes=("logical_row_major",),
            resources=TaskResourceEstimate(8, 0, 0),
        )
        family = TaskFamily(
            f"task.input_stage.{program.inputs.index(value)}",
            tuple(TaskAxis(f"axis.{axis.axis_id}", ceil(axis.extent / axis.tile_extent)) for axis in owner_axes),
        )
        tasks.append(
            TiledTask(
                task_id,
                family,
                (),
                (TaskOperationKind.INPUT_STAGE,),
                structure,
                (
                    TaskValueAccess(
                        value_ordinals[value],
                        TaskValueAccessKind.LIVE_OUT,
                        0,
                        RelationAccessClass.BOUNDED_DYNAMIC,
                        {
                            "input_axes": tuple(axis.id for axis in value.axes),
                            "input_shape": value.shape,
                            "layout": "logical_row_major",
                        },
                    ),
                ),
            )
        )
        staged_inputs[value] = task_id
        return task_id

    crossing_values: dict[tuple[int, int], set[ProgramValue]] = defaultdict(set)
    for consumer_ordinal, operation in enumerate(program.operations):
        target_task_id = task_for_operation[consumer_ordinal]
        for value in primitive_inputs(operation):
            producer = producer_for_value.get(value)
            if producer is None:
                continue
            source_id = task_for_operation[producer]
            if source_id != target_task_id:
                crossing_values[(source_id, target_task_id)].add(value)

    dependence_programs = list(task_plan.dependence_programs)
    dynamically_covered: dict[tuple[int, int], set[ProgramValue]] = defaultdict(set)
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    for target_ordinal, operation in enumerate(program.operations):
        if not isinstance(operation, RelationPrimitive):
            continue
        target_task = tasks[task_for_operation[target_ordinal]]
        for indexing in operation.input_indexing:
            runtime_values = set().union(
                *(_index_expression_values(expression) for expression in indexing.indices),
                set(),
            )
            if indexing.predicate is not None:
                runtime_values.update(_index_predicate_values(indexing.predicate))
            if operation.predicate is not None:
                runtime_values.update(_index_predicate_values(operation.predicate))
            if not runtime_values:
                continue
            runtime_inputs = tuple(value for value in operation.inputs if value in runtime_values)
            source_id = source_task_id(indexing.operand)
            if source_id == target_task.id:
                continue
            source_task = tasks[source_id]
            dependence_programs.append(
                _runtime_relation_dependence_program(
                    operation,
                    runtime_inputs=runtime_inputs,
                    runtime_value_ordinals=value_ordinals,
                    source_task=source_task,
                    target_task=target_task,
                    source_shape=indexing.operand.shape,
                    source_expressions=indexing.indices,
                    validity=_predicate_conjunction(indexing.predicate, operation.predicate),
                    visibility=visibility,
                )
            )
            dynamically_covered[(source_id, target_task.id)].add(indexing.operand)

    if not dependence_programs and not staged_inputs:
        return task_plan
    task_id_for_family_name = {task.family.name: task.id for task in tasks}
    static_dependences = tuple(
        dependence
        for dependence in task_plan.dependences
        if not (
            (
                edge := (
                    task_id_for_family_name[dependence.relation.source.name],
                    task_id_for_family_name[dependence.relation.target.name],
                )
            )
            in dynamically_covered
            and crossing_values[edge] <= dynamically_covered[edge]
        )
    )
    return CanonicalTaskPlan(
        tuple(tasks),
        static_dependences,
        task_plan.bounded_buffers,
        tuple(dependence_programs),
    )


def audit_runtime_relation_dependences(
    program: TensorProgram,
    task_plan: CanonicalTaskPlan,
) -> tuple[RuntimeRelationDependenceSite, ...]:
    """Prove every dynamic Relation source is parametric or intra-task."""
    task_for_operation = {ordinal: task.id for task in task_plan.tasks for ordinal in task.operation_ordinals}
    producer_for_value = {
        output: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    value_ordinals = tensor_program_value_ordinals(program)
    staged_input_tasks = {
        access.value_id: task.id
        for task in task_plan.tasks
        if task.operation_kinds == (TaskOperationKind.INPUT_STAGE,)
        for access in task.accesses
        if access.kind is TaskValueAccessKind.LIVE_OUT
    }
    available_programs = set(range(len(task_plan.dependence_programs)))
    sites: list[RuntimeRelationDependenceSite] = []
    for target_ordinal, operation in enumerate(program.operations):
        if not isinstance(operation, RelationPrimitive):
            continue
        target_task_id = task_for_operation[target_ordinal]
        for indexing in operation.input_indexing:
            runtime_values = set().union(
                *(_index_expression_values(expression) for expression in indexing.indices),
                set(),
            )
            if indexing.predicate is not None:
                runtime_values.update(_index_predicate_values(indexing.predicate))
            if operation.predicate is not None:
                runtime_values.update(_index_predicate_values(operation.predicate))
            if not runtime_values:
                continue
            producer = producer_for_value.get(indexing.operand)
            source_value_ordinal = value_ordinals[indexing.operand]
            source_task_id = (
                task_for_operation[producer] if producer is not None else staged_input_tasks[source_value_ordinal]
            )
            if source_task_id == target_task_id:
                sites.append(
                    RuntimeRelationDependenceSite(
                        target_ordinal,
                        source_value_ordinal,
                        source_task_id,
                        target_task_id,
                        None,
                    )
                )
                continue
            runtime_binding_ordinals = tuple(
                value_ordinals[value] for value in operation.inputs if value in runtime_values
            )
            candidates = tuple(
                index
                for index in available_programs
                if (dependence := task_plan.dependence_programs[index]).source_task_id == source_task_id
                and dependence.target_task_id == target_task_id
                and tuple(binding.algebra_value_ordinal for binding in dependence.runtime_bindings)
                == runtime_binding_ordinals
            )
            if len(candidates) != 1:
                raise ValueError("runtime Relation edge does not have one exact parametric dependence program")
            program_index = candidates[0]
            available_programs.remove(program_index)
            sites.append(
                RuntimeRelationDependenceSite(
                    target_ordinal,
                    source_value_ordinal,
                    source_task_id,
                    target_task_id,
                    program_index,
                )
            )
    if available_programs:
        raise ValueError("task plan contains parametric dependence programs without runtime Relation provenance")
    return tuple(sites)


def _runtime_relation_dependence_program(
    relation: RelationPrimitive,
    *,
    runtime_inputs: tuple[ProgramValue, ...],
    runtime_value_ordinals: Mapping[ProgramValue, int],
    source_task: TiledTask,
    target_task: TiledTask,
    source_shape: tuple[int, ...],
    source_expressions: tuple[IndexExpression, ...],
    validity: IndexPredicate | None,
    visibility: MemoryVisibility,
) -> TaskDependenceProgram:
    domain_axis = {axis.id: axis for axis in relation.domain_axes}

    def coordinate_values(task: TiledTask, prefix: str) -> tuple[ProgramValue, ...]:
        return tuple(
            ProgramValue(f"{prefix}.{position}", relation.domain_axes, DType.INT32)
            for position, _ in enumerate(task.structure.owner_axes)
        )

    def target_expressions() -> tuple[IndexExpression, ...]:
        expressions = []
        for owned in target_task.structure.owner_axes:
            axis = domain_axis.get(owned.axis_id)
            if axis is None:
                raise ValueError("runtime Relation dependence cannot project a target owner axis outside its domain")
            expression = index_axis(axis)
            if owned.tile_extent != 1:
                expression = index_binary(
                    IndexExpressionKind.FLOOR_DIVIDE,
                    expression,
                    index_constant(owned.tile_extent),
                )
            expressions.append(expression)
        return tuple(expressions)

    source_owner_axes = source_task.structure.owner_axes
    if len(source_expressions) != len(source_owner_axes):
        singleton_owner = all(axis.extent == 1 for axis in source_task.family.axes)
        flattened_source = (
            len(source_owner_axes) == 1
            and source_owner_axes[0].extent == prod(source_shape)
            and len(source_expressions) == len(source_shape)
        )
        if singleton_owner:
            source_expressions = tuple(index_constant(0) for _ in source_owner_axes)
        elif not flattened_source:
            raise ValueError("runtime Relation source indexing rank does not match the producer task's owned axes")
        else:
            linearized = index_constant(0)
            for expression, extent in zip(source_expressions, source_shape, strict=True):
                linearized = index_binary(
                    IndexExpressionKind.ADD,
                    index_binary(IndexExpressionKind.MULTIPLY, linearized, index_constant(extent)),
                    expression,
                )
            source_expressions = (linearized,)
    source_expressions = tuple(
        (
            index_binary(IndexExpressionKind.FLOOR_DIVIDE, expression, index_constant(axis.tile_extent))
            if axis.tile_extent != 1
            else expression
        )
        for expression, axis in zip(source_expressions, source_owner_axes, strict=True)
    )
    source_values = coordinate_values(source_task, "source")
    target_values = coordinate_values(target_task, "target")
    validity_value = None if validity is None else ProgramValue("validity", relation.domain_axes, DType.BOOL)
    input_indexing = {indexing.operand: indexing for indexing in relation.input_indexing}
    result_expressions = (*source_expressions, *target_expressions(), *((validity,) if validity is not None else ()))
    relation_operation = RelationPrimitive(
        "runtime.task_relation",
        runtime_inputs,
        (*source_values, *target_values, *((validity_value,) if validity_value is not None else ())),
        relation.domain_axes,
        tuple(input_indexing[value] for value in runtime_inputs),
        result_expressions=result_expressions,
        order=EvaluationOrder.ORDER_FREE,
    )
    relation_program = TensorProgram(
        runtime_inputs,
        (relation_operation,),
        relation_operation.outputs,
        NumericalPolicy.SOURCE_ORDERED,
    )

    def bounded(values: tuple[ProgramValue, ...], task: TiledTask) -> tuple[BoundedTaskCoordinate, ...]:
        return tuple(
            BoundedTaskCoordinate(value, 0, ceil(axis.extent / axis.tile_extent))
            for value, axis in zip(values, task.structure.owner_axes, strict=True)
        )

    return TaskDependenceProgram(
        source_task.id,
        target_task.id,
        relation_program,
        tuple(DependenceRuntimeBinding(value, runtime_value_ordinals[value]) for value in runtime_inputs),
        bounded(source_values, source_task),
        bounded(target_values, target_task),
        validity_value,
        visibility,
    )


def _index_predicate_values(predicate: IndexPredicate) -> set[ProgramValue]:
    values = set().union(*(_index_expression_values(expression) for expression in predicate.expressions), set())
    for nested in predicate.predicates:
        values.update(_index_predicate_values(nested))
    return values


def _index_expression_values(expression: IndexExpression) -> set[ProgramValue]:
    values = {expression.tensor} if expression.tensor is not None else set()
    for operand in expression.operands:
        values.update(_index_expression_values(operand))
    if expression.predicate is not None:
        values.update(_index_predicate_values(expression.predicate))
    return values


def _predicate_conjunction(left: IndexPredicate | None, right: IndexPredicate | None) -> IndexPredicate | None:
    if left is None:
        return right
    if right is None:
        return left
    return IndexPredicate(IndexPredicateKind.AND, predicates=(left, right))


def eventize_composed_task_plan(task_plan: CanonicalTaskPlan) -> EventDataflowProgram:
    """Factor every exact inter-task dependence into one EventTensor."""
    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event.{ordinal}",
            memory_scope=EventMemoryScope.DEVICE,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        for ordinal, dependence in enumerate(task_plan.dependences)
    )
    return EventDataflowProgram(task_plan.task_families, task_plan.dependences, event_plans)


def physical_request_for_composed_tasks(
    *,
    algebra_record,
    algebra_fingerprint: str,
    task_plan: CanonicalTaskPlan,
    event_program: EventDataflowProgram,
    target: PhysicalTarget,
    numerical_policy: PhysicalNumericalPolicy,
) -> ExecutablePhysicalRequest:
    """Create a canonical request without using a workload identity as dispatch."""
    schedules = tuple(
        PhysicalSchedule(
            PhysicalScheduleFamily(
                f"{task.structure.principal_kind.value}_phase_dag",
                target,
                (_constraint_for_task(task),),
            ),
            (task.id,),
            {"pipeline_depth": 1},
        )
        for task in task_plan.tasks
    )
    realizations = verify_event_realizations(
        event_program,
        tuple(
            physical_event_realization(
                plan,
                mechanism="bounded_counter",
                reason="no selected physical order has yet proved this exact inter-task dependence redundant",
            )
            for plan in event_program.event_plans
        ),
    )
    allocations = tuple(
        EventTensorAllocation(
            EventStorageBinding(
                plan.domain,
                tuple(range(len(plan.domain.coordinates))),
                (0,) * len(plan.domain.coordinates),
            ),
            capacity=len(plan.domain.coordinates),
            generation_count=1,
        )
        for plan in event_program.event_plans
    )
    event_programs = tuple(
        ParametricEventTensorPlan(
            dependence_program_index=index,
            derivation=ParametricEventDerivation.EXACT_CONSUMER,
            capacity=len(task_plan.tasks[dependence.target_task_id].family.coordinates),
            slots=tuple(range(len(task_plan.tasks[dependence.target_task_id].family.coordinates))),
            generations=(0,) * len(task_plan.tasks[dependence.target_task_id].family.coordinates),
            generation_count=1,
            memory_scope=EventMemoryScope.DEVICE,
            generation_policy=EventGenerationPolicy.PER_INVOCATION,
            scheduling_mode=EventSchedulingMode.DYNAMIC,
            realization=ParametricEventRealization.IMPERATIVE_WAIT_NOTIFY,
        )
        for index, dependence in enumerate(task_plan.dependence_programs)
    )
    return ExecutablePhysicalRequest(
        algebra_record=algebra_record,
        algebra_fingerprint=algebra_fingerprint,
        task_plan=task_plan,
        event_plans=event_program.event_plans,
        event_allocations=allocations,
        target=target,
        numerical_policy=numerical_policy,
        schedules=schedules,
        event_realizations=EventRealizationProof(realizations, ()),
        provenance={"source_boundary": "ordinary_jax", "differentiation_owner": "jax"},
        event_programs=event_programs,
    )


def _contract_fold_bridge_candidate(
    terminal_ordinal: int,
    *,
    program: TensorProgram,
    producer_by_value: Mapping[str, TensorPrimitive],
    consumers_by_value: Mapping[str, list[TensorPrimitive]],
    operation_index: Mapping[int, int],
    forbidden: set[int],
) -> set[int] | None:
    terminal = program.operations[terminal_ordinal]
    assert isinstance(terminal, ContractPrimitive)
    candidate = {terminal_ordinal}
    upstream_contracts: set[int] = set()
    for value in terminal.inputs:
        _collect_map_fold_ancestry(
            value,
            producer_by_value=producer_by_value,
            operation_index=operation_index,
            group=candidate,
            boundary_contracts=upstream_contracts,
            crossed_phase=False,
        )
    folds = tuple(
        program.operations[ordinal] for ordinal in candidate if isinstance(program.operations[ordinal], FoldPrimitive)
    )
    if not upstream_contracts or not folds:
        return None
    terminal_reductions = set(terminal.reduction_axes)
    if not any(terminal_reductions & set(fold.reduction_axes) for fold in folds):
        return None

    candidate.update(upstream_contracts)
    for upstream_ordinal in upstream_contracts:
        upstream = program.operations[upstream_ordinal]
        assert isinstance(upstream, ContractPrimitive)
        for value in upstream.inputs:
            _include_map_ancestry(
                value,
                producer_by_value=producer_by_value,
                operation_index=operation_index,
                forbidden=forbidden,
                group=candidate,
            )
    candidate = _include_closed_map_fold_consumers(
        candidate,
        program=program,
        producer_by_value=producer_by_value,
        operation_index=operation_index,
        forbidden=forbidden,
    )

    if not _within_fusion_resource_bounds(candidate, program):
        return None
    boundary_outputs = _boundary_outputs(
        tuple(program.operations[ordinal] for ordinal in sorted(candidate)),
        program=program,
        ordinal_set=candidate,
        consumers_by_value=consumers_by_value,
        operation_index=operation_index,
    )
    if not boundary_outputs or not _common_axes(boundary_outputs):
        return None
    owner_axes, streamed_axes, reduction_axes, contraction_axes = _fusion_axis_roles(
        tuple(program.operations[ordinal] for ordinal in sorted(candidate)),
        boundary_outputs,
    )
    roles = (*owner_axes, *streamed_axes, *reduction_axes, *contraction_axes)
    if len(roles) != len(set(roles)):
        return None
    if not _is_dependency_convex(
        candidate,
        program=program,
        producer_by_value=producer_by_value,
        consumers_by_value=consumers_by_value,
        operation_index=operation_index,
    ):
        return None
    return candidate


def _include_closed_map_fold_consumers(
    candidate: set[int],
    *,
    program: TensorProgram,
    producer_by_value: Mapping[str, TensorPrimitive],
    operation_index: Mapping[int, int],
    forbidden: set[int],
) -> set[int]:
    """Absorb scalar/fold consumers whose produced inputs are all local."""
    result = set(candidate)
    changed = True
    while changed:
        changed = False
        for ordinal, operation in enumerate(program.operations):
            if (
                ordinal in result
                or ordinal in forbidden
                or not isinstance(
                    operation,
                    (MapPrimitive, FoldPrimitive),
                )
            ):
                continue
            input_producers = tuple(producer_by_value.get(value.name) for value in primitive_inputs(operation))
            produced_inputs = tuple(producer for producer in input_producers if producer is not None)
            if not produced_inputs or not any(operation_index[id(producer)] in result for producer in produced_inputs):
                continue
            # This expansion is deliberately a *closed* continuation of the
            # candidate.  A Map that also consumes a live-in belongs at the
            # component boundary: absorbing it would make an otherwise valid
            # Contract/Fold region depend on an unrelated downstream input
            # (for example the cotangent used by a scalar training loss).
            if len(produced_inputs) != len(input_producers):
                continue
            if any(operation_index[id(producer)] not in result for producer in produced_inputs):
                continue
            tentative = {*result, ordinal}
            if not _within_fusion_resource_bounds(tentative, program):
                continue
            result = tentative
            changed = True
    return result


def _fusion_preserves_acyclic_quotient(
    groups: list[set[int]],
    candidate: set[int],
    predecessors: Mapping[int, set[int]],
) -> bool:
    # Convexity in the original DAG is insufficient: two disjoint convex
    # groups can have edges in both directions after contraction.
    representative = {ordinal: ordinal for ordinal in predecessors}
    for group in (*groups, candidate):
        representative.update(dict.fromkeys(group, min(group)))
    quotient: dict[int, set[int]] = defaultdict(set)
    for ordinal, sources in predecessors.items():
        target = representative[ordinal]
        quotient[target].update(representative[source] for source in sources if representative[source] != target)
    try:
        TopologicalSorter(quotient).prepare()
    except CycleError:
        return False
    return True


def _is_dependency_convex(
    ordinals: set[int],
    *,
    program: TensorProgram,
    producer_by_value: Mapping[str, TensorPrimitive],
    consumers_by_value: Mapping[str, list[TensorPrimitive]],
    operation_index: Mapping[int, int],
) -> bool:
    ancestors = set(ordinals)
    ancestor_frontier = list(ordinals)
    while ancestor_frontier:
        ordinal = ancestor_frontier.pop()
        for value in primitive_inputs(program.operations[ordinal]):
            producer = producer_by_value.get(value.name)
            if producer is None:
                continue
            producer_ordinal = operation_index[id(producer)]
            if producer_ordinal not in ancestors:
                ancestors.add(producer_ordinal)
                ancestor_frontier.append(producer_ordinal)

    descendants = set(ordinals)
    descendant_frontier = list(ordinals)
    while descendant_frontier:
        ordinal = descendant_frontier.pop()
        for output in primitive_outputs(program.operations[ordinal]):
            for consumer in consumers_by_value[output.name]:
                consumer_ordinal = operation_index[id(consumer)]
                if consumer_ordinal not in descendants:
                    descendants.add(consumer_ordinal)
                    descendant_frontier.append(consumer_ordinal)
    return not ((ancestors & descendants) - ordinals)


def _collect_map_fold_ancestry(
    value: ProgramValue,
    *,
    producer_by_value: Mapping[str, TensorPrimitive],
    operation_index: Mapping[int, int],
    group: set[int],
    boundary_contracts: set[int],
    crossed_phase: bool,
) -> None:
    producer = producer_by_value.get(value.name)
    if producer is None:
        return
    ordinal = operation_index[id(producer)]
    if ordinal in group or ordinal in boundary_contracts:
        return
    if isinstance(producer, ContractPrimitive):
        if crossed_phase:
            boundary_contracts.add(ordinal)
        return
    if not isinstance(producer, (MapPrimitive, FoldPrimitive)):
        return
    group.add(ordinal)
    for input_value in primitive_inputs(producer):
        _collect_map_fold_ancestry(
            input_value,
            producer_by_value=producer_by_value,
            operation_index=operation_index,
            group=group,
            boundary_contracts=boundary_contracts,
            crossed_phase=True,
        )


def _within_fusion_resource_bounds(ordinals: set[int], program: TensorProgram) -> bool:
    operations = tuple(program.operations[ordinal] for ordinal in ordinals)
    return len(operations) <= MAXIMUM_FUSED_PHASES and _estimated_registers(operations) <= (
        MAXIMUM_FUSED_REGISTERS_PER_THREAD
    )


def _estimated_registers(operations: tuple[TensorPrimitive, ...]) -> int:
    """Estimate scalar/register pressure by algebraic capability.

    Index-only Relations change addressing without retaining a tensor-sized
    value. Contracts and Folds retain accumulators, while scalar Maps add only
    short-lived scalar state. This is intentionally an algebra estimate rather
    than a count of frontend equations: inserting reshape or broadcast
    relations must not by itself make the same physical region unfusible.
    """
    return 32 + sum(
        (
            8
            if isinstance(operation, ContractPrimitive)
            else (4 if isinstance(operation, FoldPrimitive) else 2 if isinstance(operation, MapPrimitive) else 0)
        )
        for operation in operations
    )


def _include_map_ancestry_until_contract(
    value: ProgramValue,
    *,
    producer_by_value: Mapping[str, TensorPrimitive],
    operation_index: Mapping[int, int],
    forbidden: set[int],
    group: set[int],
) -> None:
    producer = producer_by_value.get(value.name)
    if producer is None:
        return
    ordinal = operation_index[id(producer)]
    if ordinal in forbidden or ordinal in group:
        return
    if isinstance(producer, ContractPrimitive):
        group.add(ordinal)
        return
    if not isinstance(producer, MapPrimitive):
        return
    group.add(ordinal)
    for input_value in primitive_inputs(producer):
        _include_map_ancestry_until_contract(
            input_value,
            producer_by_value=producer_by_value,
            operation_index=operation_index,
            forbidden=forbidden,
            group=group,
        )


def _include_map_ancestry(
    value: ProgramValue,
    *,
    producer_by_value: Mapping[str, TensorPrimitive],
    operation_index: Mapping[int, int],
    forbidden: set[int],
    group: set[int],
) -> None:
    producer = producer_by_value.get(value.name)
    if not isinstance(producer, MapPrimitive) and not _is_identity_relation(producer):
        return
    ordinal = operation_index[id(producer)]
    if ordinal in forbidden or ordinal in group:
        return
    group.add(ordinal)
    for input_value in primitive_inputs(producer):
        _include_map_ancestry(
            input_value,
            producer_by_value=producer_by_value,
            operation_index=operation_index,
            forbidden=forbidden,
            group=group,
        )


def _is_identity_relation(operation: TensorPrimitive | None) -> bool:
    """Whether a Relation only renames values in the same logical index space."""
    if not isinstance(operation, RelationPrimitive) or operation.predicate is not None:
        return False
    if any(output.axes != operation.domain_axes for output in operation.outputs):
        return False
    for indexing in operation.input_indexing:
        if indexing.predicate is not None or indexing.operand.axes != operation.domain_axes:
            return False
        if any(
            expression.kind is not IndexExpressionKind.AXIS or expression.axis != axis
            for expression, axis in zip(indexing.indices, indexing.operand.axes, strict=True)
        ):
            return False
    return all(
        source is not None and source.axes == output.axes
        for source, output in zip(operation.resolved_result_sources, operation.outputs, strict=True)
    )


def _tiled_task(
    task_id: int,
    ordinals: tuple[int, ...],
    program: TensorProgram,
    *,
    consumers_by_value: Mapping[str, list[TensorPrimitive]],
    operation_index: Mapping[int, int],
    value_id: Mapping[str, int],
    tile_extent_by_axis: Mapping[int, int],
    numerical_policy: PhysicalNumericalPolicy,
) -> TiledTask:
    operations = tuple(program.operations[ordinal] for ordinal in ordinals)
    ordinal_set = set(ordinals)
    external_outputs = _boundary_outputs(
        operations,
        program=program,
        ordinal_set=ordinal_set,
        consumers_by_value=consumers_by_value,
        operation_index=operation_index,
    )
    if not external_outputs:
        external_outputs = primitive_outputs(operations[-1])
    owner_axes, streamed_axes, reduction_axes, contraction_axes = _fusion_axis_roles(operations, external_outputs)
    owner_tiles = _axis_tiles(owner_axes, tile_extent_by_axis)
    structure = PhysicalTaskStructure(
        principal_kind=_operation_kind(
            next((operation for operation in operations if isinstance(operation, ContractPrimitive)), operations[-1])
        ),
        phases=tuple(
            PhysicalPhase(
                index,
                _capability(operation),
                (ordinal,),
                *physical_phase_iteration_space(operation),
            )
            for index, (ordinal, operation) in enumerate(zip(ordinals, operations, strict=True))
        ),
        phase_edges=_phase_edges(operations),
        owner_axes=owner_tiles,
        streamed_axes=_axis_tiles(streamed_axes, tile_extent_by_axis),
        reduction_axes=_axis_tiles(reduction_axes, tile_extent_by_axis),
        contraction_axes=_axis_tiles(contraction_axes, tile_extent_by_axis),
        traversal=(
            TaskTraversal.SOURCE_ORDERED
            if numerical_policy is PhysicalNumericalPolicy.SOURCE_ORDERED
            else TaskTraversal.UNORDERED
        ),
        element_types=tuple(
            dict.fromkeys(output.dtype.value for operation in operations for output in primitive_outputs(operation))
        ),
        layout_classes=("logical_row_major",),
        resources=TaskResourceEstimate(_estimated_registers(operations), 0, 0),
    )
    family = TaskFamily(
        f"task.{task_id}",
        tuple(TaskAxis(f"axis.{axis.axis_id}", ceil(axis.extent / axis.tile_extent)) for axis in owner_tiles),
        placement=None,
    )
    accesses = task_value_accesses(program, ordinals)
    return TiledTask(
        task_id,
        family,
        ordinals,
        tuple(_operation_kind(operation) for operation in operations),
        structure,
        accesses,
    )


def _common_axes(values: tuple[ProgramValue, ...]) -> tuple[TensorAxis, ...]:
    common = set(values[0].axes)
    for value in values[1:]:
        common.intersection_update(value.axes)
    return tuple(axis for axis in values[0].axes if axis in common)


def _boundary_outputs(
    operations: tuple[TensorPrimitive, ...],
    *,
    program: TensorProgram,
    ordinal_set: set[int],
    consumers_by_value: Mapping[str, list[TensorPrimitive]],
    operation_index: Mapping[int, int],
) -> tuple[ProgramValue, ...]:
    program_outputs = {value.name for value in program.outputs}
    return tuple(
        output
        for operation in operations
        for output in primitive_outputs(operation)
        if output.name in program_outputs
        or any(operation_index[id(consumer)] not in ordinal_set for consumer in consumers_by_value[output.name])
    )


def _fusion_axis_roles(
    operations: tuple[TensorPrimitive, ...],
    boundary_outputs: tuple[ProgramValue, ...],
) -> tuple[tuple[TensorAxis, ...], tuple[TensorAxis, ...], tuple[TensorAxis, ...], tuple[TensorAxis, ...]]:
    fold_axes = tuple(
        dict.fromkeys(
            axis for operation in operations if isinstance(operation, FoldPrimitive) for axis in operation.reduction_axes
        )
    )
    contract_axes = tuple(
        dict.fromkeys(
            axis
            for operation in operations
            if isinstance(operation, ContractPrimitive)
            for axis in operation.reduction_axes
        )
    )
    fold_axis_set = set(fold_axes)
    contract_axis_set = set(contract_axes)
    streamed_axes = tuple(axis for axis in fold_axes if axis in contract_axis_set)
    reduction_axes = tuple(axis for axis in fold_axes if axis not in contract_axis_set)
    contraction_axes = tuple(axis for axis in contract_axes if axis not in fold_axis_set)
    non_owner_axes = {*streamed_axes, *reduction_axes, *contraction_axes}
    owner_axes = tuple(axis for axis in _common_axes(boundary_outputs) if axis not in non_owner_axes)
    return owner_axes, streamed_axes, reduction_axes, contraction_axes


def _axis_tiles(axes: tuple[TensorAxis, ...], tile_extent_by_axis: Mapping[int, int]) -> tuple[TaskAxisTile, ...]:
    return tuple(
        TaskAxisTile(axis.id, axis.extent, min(axis.extent, tile_extent_by_axis.get(axis.id, axis.extent)))
        for axis in axes
    )


def _operation_kind(operation: TensorPrimitive) -> TaskOperationKind:
    if isinstance(operation, ContractPrimitive):
        return TaskOperationKind.CONTRACT
    if isinstance(operation, FoldPrimitive):
        return TaskOperationKind.FOLD
    if isinstance(operation, MapPrimitive):
        return TaskOperationKind.MAP
    if isinstance(operation, SelectionPrimitive):
        return TaskOperationKind.SELECTION
    if isinstance(operation, ScanPrimitive):
        return TaskOperationKind.SCAN
    if isinstance(operation, RelationPrimitive):
        return TaskOperationKind.RELATION
    return TaskOperationKind.TRANSPORT


def _capability(operation: TensorPrimitive) -> TaskCapability:
    if isinstance(operation, ContractPrimitive):
        return TaskCapability.MATRIX_CONTRACT
    if isinstance(operation, FoldPrimitive):
        return TaskCapability.ASSOCIATIVE_FOLD
    if isinstance(operation, MapPrimitive):
        return TaskCapability.SCALAR_MAP
    if isinstance(operation, SelectionPrimitive):
        return TaskCapability.BOUNDED_SELECTION
    if isinstance(operation, ScanPrimitive):
        return TaskCapability.ORDERED_STATE
    if isinstance(operation, RelationPrimitive):
        return TaskCapability.INDEX_RELATION
    return TaskCapability.DATA_TRANSPORT


def _phase_edges(operations: tuple[TensorPrimitive, ...]) -> tuple[PhysicalPhaseEdge, ...]:
    producer_phase = {
        output.name: phase for phase, operation in enumerate(operations) for output in primitive_outputs(operation)
    }
    edges = {
        PhysicalPhaseEdge(
            producer_phase[value.name],
            phase,
            _input_relation_class(operation, value),
        )
        for phase, operation in enumerate(operations)
        for value in primitive_inputs(operation)
        if value.name in producer_phase
    }
    return tuple(sorted(edges))


def _value_ids(program: TensorProgram) -> dict[str, int]:
    values = (*program.inputs, *(output for operation in program.operations for output in primitive_outputs(operation)))
    return {value.name: index for index, value in enumerate(values)}


def task_value_accesses(
    program: TensorProgram,
    operation_ordinals: tuple[int, ...],
) -> tuple[TaskValueAccess, ...]:
    """Derive the exact live/temporary value boundary for one task component."""
    if not operation_ordinals or len(set(operation_ordinals)) != len(operation_ordinals):
        raise ValueError("task component operation ordinals must be nonempty and unique")
    if any(ordinal < 0 or ordinal >= len(program.operations) for ordinal in operation_ordinals):
        raise ValueError("task component operation ordinal is outside the algebra program")
    operation_index = {id(operation): index for index, operation in enumerate(program.operations)}
    consumers_by_value: dict[str, list[TensorPrimitive]] = defaultdict(list)
    for operation in program.operations:
        for value in primitive_inputs(operation):
            consumers_by_value[value.name].append(operation)
    operations = tuple(program.operations[ordinal] for ordinal in operation_ordinals)
    return _task_accesses(
        operations,
        ordinal_set=set(operation_ordinals),
        consumers_by_value=consumers_by_value,
        operation_index=operation_index,
        value_id=_value_ids(program),
        output_names={value.name for value in program.outputs},
    )


def _task_accesses(
    operations: tuple[TensorPrimitive, ...],
    *,
    ordinal_set: set[int],
    consumers_by_value: Mapping[str, list[TensorPrimitive]],
    operation_index: Mapping[int, int],
    value_id: Mapping[str, int],
    output_names: set[str],
) -> tuple[TaskValueAccess, ...]:
    producer_phase = {
        output.name: phase for phase, operation in enumerate(operations) for output in primitive_outputs(operation)
    }
    accesses: list[TaskValueAccess] = []
    seen_inputs: set[tuple[str, int]] = set()
    for phase, operation in enumerate(operations):
        for value in primitive_inputs(operation):
            if value.name in producer_phase or (value.name, phase) in seen_inputs:
                continue
            seen_inputs.add((value.name, phase))
            accesses.append(
                TaskValueAccess(
                    value_id[value.name],
                    TaskValueAccessKind.LIVE_IN,
                    phase,
                    _input_relation_class(operation, value),
                    _input_relation_record(operation, value, primitive_outputs(operation)[0]),
                )
            )
        for output in primitive_outputs(operation):
            outside = output.name in output_names or any(
                operation_index[id(consumer)] not in ordinal_set for consumer in consumers_by_value[output.name]
            )
            kind = TaskValueAccessKind.LIVE_OUT if outside else TaskValueAccessKind.TEMPORARY
            accesses.append(
                TaskValueAccess(
                    value_id[output.name],
                    kind,
                    phase,
                    RelationAccessClass.IDENTITY,
                    _value_relation_record(output, output),
                )
            )
    return tuple(accesses)


def _input_relation_class(operation: TensorPrimitive, value: ProgramValue) -> RelationAccessClass:
    if isinstance(operation, ContractPrimitive):
        input_index = operation.inputs.index(value)
        if operation.index_maps_for_input(input_index):
            return RelationAccessClass.AFFINE
    if isinstance(operation, RelationPrimitive):
        indexing = operation.input_indexing[operation.inputs.index(value)]
        runtime_values = set().union(
            *(_index_expression_values(expression) for expression in indexing.indices),
            set(),
        )
        if indexing.predicate is not None:
            runtime_values.update(_index_predicate_values(indexing.predicate))
        if operation.predicate is not None:
            runtime_values.update(_index_predicate_values(operation.predicate))
        if runtime_values:
            return RelationAccessClass.BOUNDED_DYNAMIC
    return (
        RelationAccessClass.IDENTITY
        if any(value.axes == output.axes for output in primitive_outputs(operation))
        else RelationAccessClass.AFFINE
    )


def _input_relation_record(
    operation: TensorPrimitive,
    value: ProgramValue,
    target: ProgramValue,
) -> dict[str, object]:
    record = _value_relation_record(value, target)
    if not isinstance(operation, ContractPrimitive):
        return record
    mappings = operation.index_maps_for_input(operation.inputs.index(value))
    if not mappings:
        return record
    return {
        **record,
        "index_maps": tuple(
            {
                "domain_axis": mapping.domain_axis.id,
                "operand_axis": mapping.operand_axis.id,
                "divisor": mapping.divisor,
                "offset": mapping.offset,
                "modulus": mapping.modulus,
            }
            for mapping in mappings
        ),
    }


def _value_relation_record(source: ProgramValue, target: ProgramValue) -> dict[str, object]:
    return {
        "source_axes": tuple(axis.id for axis in source.axes),
        "target_axes": tuple(axis.id for axis in target.axes),
        "source_shape": tuple(axis.extent for axis in source.axes),
        "target_shape": tuple(axis.extent for axis in target.axes),
    }


def _exact_task_dependences(
    program: TensorProgram,
    tasks: tuple[TiledTask, ...],
    *,
    task_for_operation: Mapping[int, int],
) -> tuple[TaskDependence, ...]:
    producer_ordinal = {
        output.name: ordinal
        for ordinal, operation in enumerate(program.operations)
        for output in primitive_outputs(operation)
    }
    pairs_by_tasks: dict[tuple[int, int], set[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(set)
    for consumer_ordinal, operation in enumerate(program.operations):
        consumer_task_id = task_for_operation[consumer_ordinal]
        for value in primitive_inputs(operation):
            source_ordinal = producer_ordinal.get(value.name)
            if source_ordinal is None:
                continue
            producer_task_id = task_for_operation[source_ordinal]
            if producer_task_id == consumer_task_id:
                continue
            producer_task = tasks[producer_task_id]
            consumer_task = tasks[consumer_task_id]
            pairs_by_tasks[(producer_task_id, consumer_task_id)].update(
                _value_task_pairs(producer_task, consumer_task, operation=operation, value=value)
            )
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    return tuple(
        TaskDependence(
            TaskRelation.from_pairs(tasks[source].family, tasks[target].family, tuple(sorted(pairs))),
            visibility,
        )
        for (source, target), pairs in sorted(pairs_by_tasks.items())
    )


def _value_task_pairs(
    producer: TiledTask,
    consumer: TiledTask,
    *,
    operation: TensorPrimitive,
    value: ProgramValue,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    producer_axis_position = {axis.axis_id: index for index, axis in enumerate(producer.structure.owner_axes)}
    consumer_axis_position = {axis.axis_id: index for index, axis in enumerate(consumer.structure.owner_axes)}
    index_maps = ()
    if isinstance(operation, ContractPrimitive):
        index_maps = operation.index_maps_for_input(operation.inputs.index(value))
    mapped_operand_axes = {mapping.operand_axis.id for mapping in index_maps}
    common_axes = (set(producer_axis_position) & set(consumer_axis_position)) - mapped_operand_axes
    return tuple(
        (source, target)
        for source in producer.family.coordinates
        for target in consumer.family.coordinates
        if all(source[producer_axis_position[axis]] == target[consumer_axis_position[axis]] for axis in common_axes)
        and all(
            _mapped_tiles_intersect(
                mapping,
                producer,
                consumer,
                source,
                target,
                producer_axis_position,
                consumer_axis_position,
            )
            for mapping in index_maps
            if mapping.operand_axis.id in producer_axis_position and mapping.domain_axis.id in consumer_axis_position
        )
    )


def _mapped_tiles_intersect(
    mapping: AxisIndexMap,
    producer: TiledTask,
    consumer: TiledTask,
    source: tuple[int, ...],
    target: tuple[int, ...],
    producer_axis_position: Mapping[int, int],
    consumer_axis_position: Mapping[int, int],
) -> bool:
    source_position = producer_axis_position[mapping.operand_axis.id]
    target_position = consumer_axis_position[mapping.domain_axis.id]
    source_axis = producer.structure.owner_axes[source_position]
    target_axis = consumer.structure.owner_axes[target_position]
    source_start = source[source_position] * source_axis.tile_extent
    source_stop = min(source_start + source_axis.tile_extent, source_axis.extent)
    target_start = target[target_position] * target_axis.tile_extent
    target_stop = min(target_start + target_axis.tile_extent, target_axis.extent)
    # Floor division maps a contiguous domain tile onto a contiguous integer
    # interval.  Test that interval directly instead of materializing the full
    # axis map once for every producer/consumer tile pair.
    mapped_start = target_start // mapping.divisor + mapping.offset
    mapped_stop = (target_stop - 1) // mapping.divisor + mapping.offset + 1
    if mapping.modulus is None:
        return mapped_start < source_stop and source_start < mapped_stop

    source_start = max(source_start, 0)
    source_stop = min(source_stop, mapping.modulus)
    if source_start >= source_stop:
        return False
    # Find the first integer in the mapped interval whose residue lies in the
    # operand tile.  This is exact even when the interval wraps modulo N.
    residue = mapped_start % mapping.modulus
    if source_start <= residue < source_stop:
        first_intersection = mapped_start
    elif residue < source_start:
        first_intersection = mapped_start + source_start - residue
    else:
        first_intersection = mapped_start + mapping.modulus - residue + source_start
    return first_intersection < mapped_stop


def _constraint_for_task(task: TiledTask) -> PhysicalTaskConstraint:
    structure = task.structure
    relation_classes = tuple(
        dict.fromkeys(
            (
                *(access.relation_class for access in task.accesses),
                *(edge.relation_class for edge in structure.phase_edges),
            )
        )
    ) or (RelationAccessClass.AFFINE,)
    return PhysicalTaskConstraint(
        principal_kind=structure.principal_kind,
        required_capabilities=tuple(phase.capability for phase in structure.phases),
        supported_capabilities=tuple(dict.fromkeys(phase.capability for phase in structure.phases)),
        required_edges=tuple(
            PhaseEdgeConstraint(
                structure.phases[edge.source_phase].capability,
                structure.phases[edge.target_phase].capability,
                edge.relation_class,
            )
            for edge in structure.phase_edges
        ),
        owner_rank=len(structure.owner_axes),
        streamed_rank=len(structure.streamed_axes),
        reduction_rank=len(structure.reduction_axes),
        contraction_rank=len(structure.contraction_axes),
        supported_relation_classes=relation_classes,
        supported_element_types=structure.element_types,
        supported_layout_classes=structure.layout_classes,
        maximum_resources=structure.resources,
    )
