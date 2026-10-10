# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-independent SM90 lowering for a normalized-exp Contract pullback."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from tile_lifetime.cute_pipeline_artifact import CutePipelineCompileRequest
from tile_lifetime.h100_scalar_ast import factor_uniform_multiplicative_constant
from tile_lifetime.h100_streaming_lowering import (
    LoweredScoreMap,
    Sm90BackwardAccumulatorSchedule,
    Sm90RelationPackedLayout,
    derive_sm90_relation_packed_layout,
    lower_normalized_exp_score_map,
    restrict_normalized_exp_score_vjp,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractPullbackProgram,
    is_identity_post_normalization_pullback,
    verify_normalized_exp_finalizer_vjp,
    verify_normalized_exp_score_map_vjp,
)
from tile_lifetime.sm90_backward_accumulation import (
    Sm90BackwardAccumulationPlan,
    validate_ordered_backward_variant,
)
from tile_lifetime.streaming_pullback_event_schedule import (
    StreamingIntermediateStorage,
    StreamingPullbackInputDescriptor,
    StreamingPullbackInputEventSchedule,
    derive_streaming_pullback_input_event_schedule,
    streaming_pullback_input_event_schedule_record,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractOperandPrecision,
    EvaluationOrder,
    NumericalPolicy,
    ProgramValue,
    ScalarExpression,
    serialize_scalar_expression,
)


@dataclass(frozen=True)
class H100NormalizedExpBackwardLowering:
    """Graph-derived scalar, relation, dimension, and physical schedule data."""

    score_map: LoweredScoreMap
    score_map_vjp_expression: ScalarExpression
    score_map_vjp_codegen_expression: ScalarExpression
    terminal_contract_epilogue_scale: float
    physical_schedule: Sm90BackwardAccumulatorSchedule
    relation_layout: Sm90RelationPackedLayout
    input_event_schedule: StreamingPullbackInputEventSchedule
    compile_request: CutePipelineCompileRequest
    accumulation_plan: Sm90BackwardAccumulationPlan | None


def lower_h100_normalized_exp_backward(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
) -> H100NormalizedExpBackwardLowering:
    """Legalize a shared-region pullback for the current D128 BF16 SM90 implementation."""
    verify_normalized_exp_score_map_vjp(program)
    verify_normalized_exp_finalizer_vjp(program)
    region = program.forward.region
    producer_left, producer_right = region.producer.inputs
    consumer_operand = region.consumer.inputs[1]
    if any(operand.dtype.value != "bf16" for operand in (producer_left, producer_right, consumer_operand)):
        raise ValueError("the current SM90 reverse schedule requires BF16 contraction operands")
    relation_layout = derive_sm90_relation_packed_layout(region)
    scheduled_layout = program.schedule.contract.relation_layout
    if scheduled_layout is not None and scheduled_layout != relation_layout:
        raise ValueError("the selected reverse relation layout does not match the Contract operands")
    physical_schedule = replace(
        program.schedule,
        contract=replace(program.schedule.contract, relation_layout=relation_layout),
    )
    if physical_schedule.contract.tile_m != 64 or physical_schedule.contract.tile_n != 128:
        raise ValueError("the current D128 SM90 reverse schedule requires the certified 64x128 tile")

    dimensions = region.dimensions
    if len(dimensions.producer_features) != 1 or dimensions.producer_features[0].extent != 128:
        raise ValueError("the current SM90 reverse schedule requires a producer feature dimension of 128")
    if len(dimensions.consumer_features) != 1 or dimensions.consumer_features[0].extent != 128:
        raise ValueError("the current SM90 reverse schedule requires a consumer feature dimension of 128")
    score_map = lower_normalized_exp_score_map(region)
    score_map_vjp_expression = restrict_normalized_exp_score_vjp(
        region,
        program.pullback.score_map_vjp.expression,
        cotangent_input=program.pullback.normalized_exp_pullback.output.name,
    )
    score_map_vjp_codegen_expression, terminal_contract_epilogue_scale = factor_uniform_multiplicative_constant(
        score_map_vjp_expression
    )
    accumulation_plan = _accumulation_plan(program, physical_schedule, relation_layout)
    input_event_schedule = _input_event_schedule(program, physical_schedule, relation_layout)
    compile_request = _compile_request(
        program,
        physical_schedule,
        relation_layout,
        score_map_vjp_expression,
        input_event_schedule,
    )
    return H100NormalizedExpBackwardLowering(
        score_map=score_map,
        score_map_vjp_expression=score_map_vjp_expression,
        score_map_vjp_codegen_expression=score_map_vjp_codegen_expression,
        terminal_contract_epilogue_scale=terminal_contract_epilogue_scale,
        physical_schedule=physical_schedule,
        relation_layout=relation_layout,
        input_event_schedule=input_event_schedule,
        compile_request=compile_request,
        accumulation_plan=accumulation_plan,
    )


def _accumulation_plan(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    schedule: Sm90BackwardAccumulatorSchedule,
    layout: Sm90RelationPackedLayout,
) -> Sm90BackwardAccumulationPlan | None:
    if not schedule.deterministic_accumulation:
        return None
    if program.forward.source.numerical_policy is not NumericalPolicy.FAST:
        raise ValueError("ordered SM90 accumulation requires explicit FAST source policy")
    validate_ordered_backward_variant(
        relation_group_size=layout.relation_group_size, layout_transport=False, combined=False
    )
    if schedule.owner_subtile_factor != 1 or schedule.epilogue_cluster_size != 1 or schedule.epilogue_threads != 128:
        raise ValueError("ordered SM90 accumulation requires unpartitioned owner tiles and 128-thread epilogues")
    contracts = (
        program.forward.region.producer,
        program.forward.region.consumer,
        *(gradient.contract for gradient in program.pullback.terminal_gradients),
    )
    if any(contract.order is not EvaluationOrder.ORDER_FREE for contract in contracts):
        raise ValueError("ordered FAST physical accumulation does not implement SOURCE_ORDERED Contracts")
    if any(
        precision is not ContractOperandPrecision.DEFAULT
        for contract in contracts
        for precision in contract.operand_precision
    ):
        raise ValueError("ordered SM90 accumulation only admits DEFAULT operand precision")
    return Sm90BackwardAccumulationPlan(
        owner_extent=layout.owner_extent,
        streamed_extent=layout.streamed_extent,
        owner_tile=schedule.contract.tile_m,
        streamed_tile=schedule.contract.tile_n,
        batch=layout.batch_extent,
        heads=layout.flattened_relation_axis.extent,
        producer_features=layout.producer_feature_extent,
        consumer_features=layout.consumer_feature_extent,
        traversal=program.ownership.traversal,
    )


def _input_event_schedule(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    physical_schedule: Sm90BackwardAccumulatorSchedule,
    relation_layout: Sm90RelationPackedLayout,
) -> StreamingPullbackInputEventSchedule:
    pullback = program.pullback
    identity_pullback = is_identity_post_normalization_pullback(program)
    cotangent_terms = 1 if identity_pullback else physical_schedule.post_normalization_cotangent_expansion_terms
    cotangent_stages = physical_schedule.consumer_output_cotangent_stages if identity_pullback else 1
    owner_tile = physical_schedule.contract.tile_m
    streamed_tile = physical_schedule.contract.tile_n
    producer_bytes = 2
    accumulator_bytes = 4
    return derive_streaming_pullback_input_event_schedule(
        StreamingPullbackInputDescriptor(
            producer_lhs_stage_name="producer-left and Fold-state stage",
            output_cotangent_stage_name="output-cotangent and output-dependent Fold-state stage",
            score_contract_name=program.forward.region.producer.name,
            output_cotangent_contract_name=pullback.consumer_weight_cotangent.name,
            normalized_weight_name=pullback.normalized_weight.name,
            pullback_map_name=pullback.normalized_exp_pullback.name,
            score_vjp_name=pullback.score_map_vjp.name,
            producer_lhs_gradient_name=pullback.terminal_gradients[0].contract.name,
            lhs_gradient_handoff_name="producer-left gradient shared-to-global handoff",
            producer_rhs_gradient_name=pullback.terminal_gradients[1].contract.name,
            consumer_rhs_gradient_name=pullback.terminal_gradients[2].contract.name,
            producer_rhs_epilogue_stage_name="producer-right gradient epilogue stage",
            producer_rhs_epilogue_transfer_name="producer-right gradient epilogue transfer",
            consumer_rhs_epilogue_stage_name="consumer-operand gradient epilogue stage",
            consumer_rhs_epilogue_transfer_name="consumer-operand gradient epilogue transfer",
            owner_extent=relation_layout.owner_extent,
            owner_tile_size=owner_tile,
            streamed_extent=relation_layout.streamed_extent,
            streamed_tile_size=streamed_tile,
            score_tile_coordinates=_score_tile_coordinates(
                program.ownership.traversal,
                owner_extent=relation_layout.owner_extent,
                streamed_extent=relation_layout.streamed_extent,
                owner_tile=owner_tile,
                streamed_tile=streamed_tile,
            ),
            relation_group_extent=relation_layout.relation_group_size,
            deterministic_accumulation=physical_schedule.deterministic_accumulation,
            accumulation_plan=_accumulation_plan(program, physical_schedule, relation_layout),
            producer_lhs_pipeline_depth=physical_schedule.resident_left_stages,
            output_cotangent_pipeline_depth=cotangent_stages,
            score_workspace_depth=physical_schedule.score_cotangent_stages,
            normalized_weight_storage=(
                StreamingIntermediateStorage.REGISTERS
                if physical_schedule.normalized_weight_register_source
                else StreamingIntermediateStorage.SHARED
            ),
            matrix_warpgroups=physical_schedule.contract.threads // 128 - 1,
            lhs_gradient_warpgroups=(
                1 if physical_schedule.left_gradient_single_warpgroup else physical_schedule.contract.threads // 128 - 1
            ),
            producer_lhs_transaction_bytes=(
                owner_tile * relation_layout.producer_feature_extent * producer_bytes + owner_tile * accumulator_bytes
            ),
            output_cotangent_transaction_bytes=(
                cotangent_terms * owner_tile * relation_layout.consumer_feature_extent * producer_bytes
                + owner_tile * accumulator_bytes
            ),
            producer_rhs_resident_transaction_bytes=(
                streamed_tile * relation_layout.producer_feature_extent * producer_bytes
            ),
            consumer_rhs_resident_transaction_bytes=(
                streamed_tile * relation_layout.consumer_feature_extent * producer_bytes
            ),
        )
    )


def _score_tile_coordinates(
    traversal: ContractDomainTraversal,
    *,
    owner_extent: int,
    streamed_extent: int,
    owner_tile: int,
    streamed_tile: int,
) -> tuple[tuple[int, int], ...]:
    """Return exact active score tiles from the selected Contract domain."""
    owner_partition_count = (owner_extent + owner_tile - 1) // owner_tile
    streamed_partition_count = (streamed_extent + streamed_tile - 1) // streamed_tile
    if traversal is ContractDomainTraversal.FULL:
        return tuple(
            (owner_partition, streamed_partition)
            for owner_partition in range(owner_partition_count)
            for streamed_partition in range(streamed_partition_count)
        )
    if traversal is ContractDomainTraversal.LOWER_TRIANGULAR:
        return tuple(
            (owner_partition, streamed_partition)
            for owner_partition in range(owner_partition_count)
            for streamed_partition in range(streamed_partition_count)
            if streamed_partition * streamed_tile < min((owner_partition + 1) * owner_tile, owner_extent)
        )
    raise ValueError(f"the current SM90 reverse Event schedule does not support traversal {traversal.value}")


def _compile_request(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    physical_schedule: Sm90BackwardAccumulatorSchedule,
    relation_layout: Sm90RelationPackedLayout,
    score_map_vjp_expression: ScalarExpression,
    input_event_schedule: StreamingPullbackInputEventSchedule,
) -> CutePipelineCompileRequest:
    region = program.forward.region
    producer_left, producer_right = region.producer.inputs
    consumer_operand = region.consumer.inputs[1]
    output = region.finalizer.output
    index_relations = tuple(
        _relation_record(contract_name, input_index, relation)
        for contract_name, contract in (("producer", region.producer), ("consumer", region.consumer))
        for input_index in range(len(contract.inputs))
        for relation in contract.index_maps_for_input(input_index)
    )
    dimensions = region.dimensions
    return CutePipelineCompileRequest(
        schema_version=3,
        target="sm_90a",
        numerical_policy="fast",
        physical_template="contract_fold_dependent_map_contract_sm90_v1",
        tensors=tuple(
            _tensor_record(role, tensor)
            for role, tensor in (
                ("producer_left", producer_left),
                ("producer_right", producer_right),
                ("consumer_operand", consumer_operand),
                ("final_output", output),
            )
        ),
        scalar_programs={
            "score_map": json.loads(serialize_scalar_expression(region.score_map.expression)),
            "normalized_exp_pullback": json.loads(
                serialize_scalar_expression(program.pullback.normalized_exp_pullback.expression)
            ),
            "score_map_vjp": json.loads(serialize_scalar_expression(score_map_vjp_expression)),
            "finalizer": json.loads(serialize_scalar_expression(region.finalizer.expression)),
            "post_normalization_finalizer": json.loads(
                serialize_scalar_expression(program.pullback.post_normalization.expression)
            ),
            "post_normalization_finalizer_vjp": json.loads(
                serialize_scalar_expression(program.pullback.post_normalization_vjp.expression)
            ),
        },
        index_relations=index_relations,
        contraction_dimensions={
            "producer_features": tuple(_axis_record(axis) for axis in dimensions.producer_features),
            "normalized_axes": tuple(_axis_record(axis) for axis in dimensions.normalized_axes),
            "consumer_features": tuple(_axis_record(axis) for axis in dimensions.consumer_features),
            "shared_relation_axes": tuple(_axis_record(axis) for axis in dimensions.shared_relation_axes),
            "relation_group_axes": tuple(_axis_record(axis) for axis in dimensions.relation_group_axes),
        },
        schedule={
            "family": physical_schedule.contract.family,
            "owner_tile": physical_schedule.contract.tile_m,
            "streamed_tile": physical_schedule.contract.tile_n,
            "relation_layout": _relation_layout_record(relation_layout),
            "domain_traversal": program.ownership.traversal.value,
            "right_owner_fold_order": program.right_owner_fold_order.value,
            "threads": physical_schedule.contract.threads,
            "wgmma_instruction": physical_schedule.contract.wgmma_instruction,
            "accumulator_dtype": physical_schedule.contract.accumulator_dtype.value,
            "resident_left_stages": physical_schedule.resident_left_stages,
            "consumer_output_cotangent_stages": physical_schedule.consumer_output_cotangent_stages,
            "post_normalization_cotangent_expansion_terms": (
                physical_schedule.post_normalization_cotangent_expansion_terms
            ),
            "score_cotangent_stages": physical_schedule.score_cotangent_stages,
            "score_cotangent_swap_ab": physical_schedule.score_cotangent_swap_ab,
            "right_gradient_swap_ab": physical_schedule.right_gradient_swap_ab,
            "left_gradient_swap_ab": physical_schedule.left_gradient_swap_ab,
            "score_atom_layout_m": physical_schedule.score_atom_layout_m,
            "right_gradient_atom_layout_n": physical_schedule.right_gradient_atom_layout_n,
            "left_atom_layout_m": physical_schedule.left_atom_layout_m,
            "resident_consumer_operand_in_registers": physical_schedule.resident_consumer_operand_in_registers,
            "left_gradient_single_warpgroup": physical_schedule.left_gradient_single_warpgroup,
            "deterministic_accumulation": physical_schedule.deterministic_accumulation,
            **(
                {"ordered_accumulation": _accumulation_plan(program, physical_schedule, relation_layout).record}
                if physical_schedule.deterministic_accumulation
                else {}
            ),
            "owner_subtile_factor": physical_schedule.owner_subtile_factor,
            "epilogue_tile_m": physical_schedule.epilogue_tile_m,
            "epilogue_threads": physical_schedule.epilogue_threads,
            "epilogue_cluster_size": physical_schedule.epilogue_cluster_size,
            "use_padded_epilogue_offsets": physical_schedule.use_padded_epilogue_offsets,
            "compute_capability": physical_schedule.compute_capability,
            "mainloop_dynamic_shared_memory_bytes": physical_schedule.mainloop_dynamic_shared_memory_bytes,
            "epilogue_dynamic_shared_memory_bytes": physical_schedule.epilogue_dynamic_shared_memory_bytes,
            "physical_event_schedule": streaming_pullback_input_event_schedule_record(input_event_schedule),
            "post_normalization_pullback": {
                "normalized_output_storage": "fp32",
                "normalized_output_cotangent_storage": "bf16",
                "normalized_output_cotangent_expansion": "iterated_bf16_residual",
                "normalized_output_cotangent_terms": physical_schedule.post_normalization_cotangent_expansion_terms,
                "normalized_output_cotangent_stages_per_term": 1,
                "expanded_cotangent_shared_memory_bytes": (
                    physical_schedule.contract.tile_m
                    * dimensions.consumer_features[0].extent
                    * 2
                    * physical_schedule.post_normalization_cotangent_expansion_terms
                ),
                "feature_fold": "normalized_output_times_cotangent_sum",
            },
        },
        algebra_identity={
            "domain_predicate": None if region.domain_predicate is None else region.domain_predicate.canonical_record,
        },
    )


def _relation_record(contract_name: str, input_index: int, relation: AxisIndexMap) -> dict[str, Any]:
    return {
        "contract": contract_name,
        "input": input_index,
        "domain_axis": _axis_record(relation.domain_axis),
        "operand_axis": _axis_record(relation.operand_axis),
        "divisor": relation.divisor,
        "offset": relation.offset,
        "modulus": relation.modulus,
    }


def _relation_layout_record(layout: Sm90RelationPackedLayout) -> dict[str, Any]:
    return {
        "batch_shape": layout.batch_shape,
        "shared_relation_axis": _axis_record(layout.shared_relation_axis),
        "relation_group_axes": tuple(_axis_record(axis) for axis in layout.relation_group_axes),
        "flattened_relation_axis": _axis_record(layout.flattened_relation_axis),
        "shared_relation_map": _relation_record("physical_layout", 0, layout.shared_relation_map),
        "group_relation_maps": tuple(
            _relation_record("physical_layout", 0, relation) for relation in layout.group_relation_maps
        ),
        "owner_extent": layout.owner_extent,
        "streamed_extent": layout.streamed_extent,
        "producer_feature_extent": layout.producer_feature_extent,
        "consumer_feature_extent": layout.consumer_feature_extent,
        "owner_physical_shape": layout.owner_physical_shape(layout.producer_feature_extent),
        "owner_physical_strides": layout.owner_physical_strides(layout.producer_feature_extent),
        "streamed_producer_physical_shape": layout.streamed_physical_shape(layout.producer_feature_extent),
        "streamed_producer_physical_strides": layout.streamed_physical_strides(layout.producer_feature_extent),
        "streamed_consumer_physical_shape": layout.streamed_physical_shape(layout.consumer_feature_extent),
        "streamed_consumer_physical_strides": layout.streamed_physical_strides(layout.consumer_feature_extent),
    }


def _tensor_record(role: str, tensor: ProgramValue) -> dict[str, Any]:
    return {
        "role": role,
        "dtype": tensor.dtype.value,
        "shape": tensor.shape,
        "axes": tuple(_axis_record(axis) for axis in tensor.axes),
        "alignment": 16,
    }


def _axis_record(axis: Any) -> dict[str, Any]:
    return {"id": axis.id, "label": axis.label, "extent": axis.extent}
