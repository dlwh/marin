# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import tile_lifetime.physical_normalized_exp_contract as physical_normalized_exp_contract
from shuttle.ir import DType
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineExpression,
    CuteAffineExpressionKind,
    CuteEventExecutionPoint,
    CuteEventRealizationKind,
    CuteSourceTaskRole,
    CuteStructuralEventCover,
)
from tile_lifetime.event_dataflow_adapters import streaming_contract_fold_event_descriptor
from tile_lifetime.h100_streaming_lowering import (
    Sm90ContractSchedule,
    Sm90ForwardPipelineSchedule,
    lower_normalized_exp_score_map,
    sm90_relation_packed_layout,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    derive_normalized_exp_contract_pullback,
)
from tile_lifetime.physical_artifact_compiler import (
    GpuPhysicalArtifactRequest,
    ShuttleBlockRecord,
    ShuttleNormalizedExpContractRegionRecord,
    ShuttleOperationRecord,
    ShuttleTypeRecord,
    ShuttleValueRecord,
    decode_gpu_physical_artifact_request,
)
from tile_lifetime.physical_event_cover import (
    affine_expression_from_record,
    derive_lpt_algebraic_refinement_covers,
)
from tile_lifetime.physical_normalized_exp_contract import (
    NormalizedExpContractGeometry,
    PhysicalNormalizedExpContractNotApplicable,
    PhysicalNormalizedExpContractRegion,
    PhysicalNormalizedExpForwardMatch,
    PhysicalNormalizedExpPullbackMatch,
    contract_indexing_dimensions,
    normalized_exp_consumer_dimension_roles,
    normalized_exp_contract_program,
    normalized_exp_sufficient_state_cut_types,
    physical_normalized_exp_contract_match,
    physical_saved_state_reverse_boundary,
    physical_saved_state_reverse_region,
    source_argument_buffers,
    source_bound_buffers,
    source_bound_schedule_tasks,
    source_bound_tasks,
    source_result_buffers,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_event_schedule import (
    canonical_streaming_physical_event_schedule_record,
    derive_streaming_physical_event_schedule,
)
from tile_lifetime.tensor_program import (
    EvaluationOrder,
    NumericalPolicy,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_expression_inputs,
    scalar_input,
    serialize_scalar_expression,
)

_NATIVE_SCALAR_VALUE_AND_GRAD_FIXTURE = (
    Path(__file__).parent / "fixtures/native_schema_v3/jax-bf16-causal-gqa-scalar-value-and-grad-s128-d128-sm90.json"
)


def _operation(
    name: str,
    *,
    attributes: dict[str, object] | None = None,
    operations: tuple[ShuttleOperationRecord, ...] = (),
) -> ShuttleOperationRecord:
    regions = ((ShuttleBlockRecord((), operations),),) if operations else ()
    return ShuttleOperationRecord(name, attributes or {}, (), (), regions)


def _event_lowering_plan() -> ShuttleOperationRecord:
    return _operation(
        "shuttle.event_lowering_plan",
        attributes={
            "schema_version": 1,
            "source_schedule_fingerprint": "c" * 64,
            "physical_launch_recipe_fingerprint": "d" * 64,
            "fingerprint": "e" * 64,
        },
        operations=(_operation("shuttle.event_lowering_plan_yield"),),
    )


def _annotate_forward_capability(request: GpuPhysicalArtifactRequest) -> GpuPhysicalArtifactRequest:
    body = request.region.body()
    operations = tuple(operation for operation in body.operations if operation.results)
    by_value = {
        result.id: (ordinal, operation) for ordinal, operation in enumerate(operations) for result in operation.results
    }
    producer_ordinal, producer = by_value[100]
    consumer_ordinal, consumer = by_value[109]
    normalized_map = by_value[108][1]
    score_value = normalized_map.operands[0]
    maximum_value = 104
    normalizer_value = 105
    normalized_operand = next(index for index, operand in enumerate(consumer.operands) if operand in {108, 135})
    normalized_weight_value = consumer.operands[normalized_operand]
    fold_ordinals = (by_value[maximum_value][0],)
    yielded = body.operations[-1].operands
    score_ordinal, score_operation = by_value[score_value]
    maximum_ordinal, maximum_operation = by_value[maximum_value]
    normalizer_ordinal, normalizer_operation = by_value[normalizer_value]
    normalized_weight_ordinal, normalized_weight_operation = by_value[normalized_weight_value]
    finalizer_ordinal, finalizer_operation = by_value[yielded[0]]
    capability = ShuttleNormalizedExpContractRegionRecord(
        ordinal=0,
        producer_operation=producer_ordinal,
        score_operation=score_ordinal,
        score_result=next(index for index, value in enumerate(score_operation.results) if value.id == score_value),
        fold_operations=fold_ordinals,
        maximum_operation=maximum_ordinal,
        maximum_result=next(index for index, value in enumerate(maximum_operation.results) if value.id == maximum_value),
        normalizer_operation=normalizer_ordinal,
        normalizer_result=next(
            index for index, value in enumerate(normalizer_operation.results) if value.id == normalizer_value
        ),
        normalized_weight_operation=normalized_weight_ordinal,
        normalized_weight_result=next(
            index
            for index, value in enumerate(normalized_weight_operation.results)
            if value.id == normalized_weight_value
        ),
        consumer_operation=consumer_ordinal,
        normalized_operand=normalized_operand,
        finalizer_operation=finalizer_ordinal,
        finalizer_result=next(
            index for index, value in enumerate(finalizer_operation.results) if value.id == yielded[0]
        ),
        output_result=0,
    )
    assert producer.results[0].id == 100
    return replace(request, normalized_exp_contract_regions=(capability,))


def _buffer(
    ordinal: int,
    element_type: str,
    shape: tuple[int, ...],
    *,
    live_in: bool = False,
    live_out: bool = False,
    source_argument: int | None = None,
    producer: int | None = None,
    consumers: tuple[int, ...] = (),
) -> ShuttleOperationRecord:
    attributes: dict[str, object] = {
        "ordinal": ordinal,
        "tensor_type": {"kind": "ranked_tensor", "element_type": element_type, "shape": list(shape)},
        "storage": {
            "kind": "mlir_attribute",
            "value": (
                "#shuttle.materialization_storage<external>"
                if live_in or live_out
                else "#shuttle.materialization_storage<temporary>"
            ),
        },
        "live_in": live_in,
        "live_out": live_out,
        "consumers": list(consumers),
    }
    if live_in:
        attributes["source_argument"] = ordinal if source_argument is None else source_argument
    if producer is not None:
        attributes["producer"] = producer
    return _operation("shuttle.materialization_buffer", attributes=attributes)


def _task(
    ordinal: int,
    *,
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    dependencies: tuple[int, ...],
    reductions: tuple[int, ...],
    owner: tuple[int, ...],
    vector: tuple[int, ...],
    state_inputs: tuple[int, ...],
    state_outputs: tuple[int, ...],
) -> ShuttleOperationRecord:
    attributes: dict[str, object] = {
        "ordinal": ordinal,
        "composition_sources": [],
        "input_buffers": list(inputs),
        "output_buffers": list(outputs),
        "dependencies": list(dependencies),
        "domain_shape": [1, 1, 2, 2, 128, 128],
        "reduction_dimensions": list(reductions),
        "composition_owner_dimensions": list(owner),
        "online_vector_dimensions": list(vector),
        "composition_stream_dimensions": list(reductions),
        "composition_producer_contraction_dimensions": [5],
        "composition_consumer_contraction_dimensions": [5],
    }
    if state_inputs:
        attributes["composition_state_inputs"] = list(state_inputs)
    if state_outputs:
        attributes["composition_state_outputs"] = list(state_outputs)
    return _operation("shuttle.materialization_task", attributes=attributes)


def _schedule_task(task: ShuttleOperationRecord, *, tile: tuple[int, ...], serial_tiles: int) -> ShuttleOperationRecord:
    attributes = task.attributes
    schedule_attributes: dict[str, object] = {
        "ordinal": attributes["ordinal"],
        "source_task": attributes["ordinal"],
        "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<contract>"},
        "domain_shape": attributes["domain_shape"],
        "grid_shape": [1],
        "tile_shape": list(tile),
        "serial_tiles": serial_tiles,
        "workgroup_threads": 384,
        "subgroup_size": 32,
        "scratch_bytes": 198_656,
        "input_buffers": attributes["input_buffers"],
        "output_buffers": attributes["output_buffers"],
        "dependencies": attributes["dependencies"],
        "reduction_axis": attributes["reduction_dimensions"][-1],
        "composition_owner_dimensions": attributes["composition_owner_dimensions"],
        "online_vector_dimensions": attributes["online_vector_dimensions"],
        "composition_stream_dimensions": attributes["composition_stream_dimensions"],
        "composition_producer_contraction_dimensions": attributes["composition_producer_contraction_dimensions"],
        "composition_consumer_contraction_dimensions": attributes["composition_consumer_contraction_dimensions"],
    }
    for name in ("composition_state_inputs", "composition_state_outputs"):
        if name in attributes:
            schedule_attributes[name] = attributes[name]
    return _operation("shuttle.schedule_task", attributes=schedule_attributes)


def _affine_map(*dimensions: int) -> dict[str, object]:
    return {
        "kind": "affine_map",
        "dimensions": 6,
        "symbols": 0,
        "results": [{"kind": "dimension", "position": dimension} for dimension in dimensions],
    }


def _value(identifier: int, type_: ShuttleTypeRecord) -> ShuttleValueRecord:
    return ShuttleValueRecord(identifier, type_)


def _relation(
    identifier: int,
    semantics: str,
    operands: tuple[int, ...],
    type_: ShuttleTypeRecord,
    *,
    parameters: tuple[object, ...] = (),
) -> ShuttleOperationRecord:
    return ShuttleOperationRecord(
        "shuttle.relation",
        {
            "semantics": {
                "kind": "mlir_attribute",
                "value": f"#shuttle.relation_semantics<{semantics}>",
            },
            "parameters": parameters,
        },
        operands,
        (_value(identifier, type_),),
        (),
    )


def _contract(
    identifier: int,
    operands: tuple[int, int],
    type_: ShuttleTypeRecord,
    maps: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> ShuttleOperationRecord:
    return ShuttleOperationRecord(
        "shuttle.contract",
        {"indexing_maps": list(maps)},
        operands,
        (_value(identifier, type_),),
        (),
    )


def _map(identifier: int, operands: tuple[int, ...], type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    return ShuttleOperationRecord("shuttle.map", {}, operands, (_value(identifier, type_),), ())


def _multiply_map(identifier: int, operands: tuple[int, int], type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    arguments = (_value(identifier * 10, scalar_type), _value(identifier * 10 + 1, scalar_type))
    product = _value(identifier * 10 + 2, scalar_type)
    body = ShuttleBlockRecord(
        arguments,
        (
            ShuttleOperationRecord("arith.mulf", {}, tuple(argument.id for argument in arguments), (product,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (product.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, operands, (_value(identifier, type_),), ((body,),))


def _affine_scalar_map(identifier: int, operand: int, type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    argument = _value(identifier * 10, scalar_type)
    scale = _value(identifier * 10 + 1, scalar_type)
    scaled = _value(identifier * 10 + 2, scalar_type)
    offset = _value(identifier * 10 + 3, scalar_type)
    result = _value(identifier * 10 + 4, scalar_type)
    body = ShuttleBlockRecord(
        (argument,),
        (
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": "3f000000"}},
                (),
                (scale,),
                (),
            ),
            ShuttleOperationRecord("arith.mulf", {}, (argument.id, scale.id), (scaled,), ()),
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": "3f800000"}},
                (),
                (offset,),
                (),
            ),
            ShuttleOperationRecord("arith.addf", {}, (scaled.id, offset.id), (result,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (result.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (operand,), (_value(identifier, type_),), ((body,),))


def _scale_scalar_map(identifier: int, operand: int, type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    argument = _value(identifier * 10, scalar_type)
    scale = _value(identifier * 10 + 1, scalar_type)
    result = _value(identifier * 10 + 2, scalar_type)
    body = ShuttleBlockRecord(
        (argument,),
        (
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": "3f000000"}},
                (),
                (scale,),
                (),
            ),
            ShuttleOperationRecord("arith.mulf", {}, (argument.id, scale.id), (result,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (result.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (operand,), (_value(identifier, type_),), ((body,),))


def _scaled_tanh_scalar_map(identifier: int, operand: int, type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    argument = _value(identifier * 10, scalar_type)
    scale = _value(identifier * 10 + 1, scalar_type)
    scaled = _value(identifier * 10 + 2, scalar_type)
    cap = _value(identifier * 10 + 3, scalar_type)
    divided = _value(identifier * 10 + 4, scalar_type)
    activated = _value(identifier * 10 + 5, scalar_type)
    result = _value(identifier * 10 + 6, scalar_type)
    body = ShuttleBlockRecord(
        (argument,),
        (
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": "3E3504F3"}},
                (),
                (scale,),
                (),
            ),
            ShuttleOperationRecord("arith.mulf", {}, (argument.id, scale.id), (scaled,), ()),
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": "41400000"}},
                (),
                (cap,),
                (),
            ),
            ShuttleOperationRecord("arith.divf", {}, (scaled.id, cap.id), (divided,), ()),
            ShuttleOperationRecord("math.tanh", {}, (divided.id,), (activated,), ()),
            ShuttleOperationRecord("arith.mulf", {}, (cap.id, activated.id), (result,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (result.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (operand,), (_value(identifier, type_),), ((body,),))


def _convert_scalar_map(
    identifier: int,
    operand: int,
    input_type: ShuttleTypeRecord,
    output_type: ShuttleTypeRecord,
) -> ShuttleOperationRecord:
    argument = _value(identifier * 10, ShuttleTypeRecord("scalar", input_type.element_type, ()))
    result = _value(identifier * 10 + 1, ShuttleTypeRecord("scalar", output_type.element_type, ()))
    body = ShuttleBlockRecord(
        (argument,),
        (
            ShuttleOperationRecord(
                "shuttle.scalar_convert",
                {
                    "semantics": {
                        "kind": "mlir_attribute",
                        "value": "#shuttle.scalar_convert_semantics<round_nearest_even>",
                    }
                },
                (argument.id,),
                (result,),
                (),
            ),
            ShuttleOperationRecord("shuttle.yield", {}, (result.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (operand,), (_value(identifier, output_type),), ((body,),))


def _normalized_pullback_map(
    identifier: int,
    operands: tuple[int, int, int],
    type_: ShuttleTypeRecord,
) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    arguments = tuple(_value(identifier * 10 + offset, scalar_type) for offset in range(3))
    difference = _value(identifier * 10 + 3, scalar_type)
    result = _value(identifier * 10 + 4, scalar_type)
    body = ShuttleBlockRecord(
        arguments,
        (
            ShuttleOperationRecord("arith.subf", {}, (arguments[1].id, arguments[2].id), (difference,), ()),
            ShuttleOperationRecord("arith.mulf", {}, (arguments[0].id, difference.id), (result,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (result.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, operands, (_value(identifier, type_),), ((body,),))


def _sum_fold(identifier: int, operand: int, type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    combiner = ShuttleBlockRecord((), (_operation("arith.addf"),))
    return ShuttleOperationRecord(
        "shuttle.fold",
        {"reduction_dimensions": [5]},
        (operand,),
        (_value(identifier, type_),),
        ((combiner,),),
    )


def _constant_map(identifier: int, type_: ShuttleTypeRecord, bits: str) -> ShuttleOperationRecord:
    scalar = _value(identifier * 10, ShuttleTypeRecord("scalar", "f32", ()))
    body = ShuttleBlockRecord(
        (),
        (
            ShuttleOperationRecord(
                "arith.constant",
                {"value": {"kind": "float", "element_type": "f32", "bits": bits}},
                (),
                (scalar,),
                (),
            ),
            ShuttleOperationRecord("shuttle.yield", {}, (scalar.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (), (_value(identifier, type_),), ((body,),))


def _identity_map(identifier: int, operand: int, type_: ShuttleTypeRecord) -> ShuttleOperationRecord:
    scalar = _value(identifier * 10, ShuttleTypeRecord("scalar", "f32", ()))
    body = ShuttleBlockRecord(
        (scalar,),
        (ShuttleOperationRecord("shuttle.yield", {}, (scalar.id,), (), ()),),
    )
    return ShuttleOperationRecord("shuttle.map", {}, (operand,), (_value(identifier, type_),), ((body,),))


def _transpose_map(
    identifier: int,
    operand: int,
    type_: ShuttleTypeRecord,
    *,
    input_dimensions: tuple[int, ...],
) -> ShuttleOperationRecord:
    scalar = _value(identifier * 10, ShuttleTypeRecord("scalar", "f32", ()))
    body = ShuttleBlockRecord(
        (scalar,),
        (ShuttleOperationRecord("shuttle.yield", {}, (scalar.id,), (), ()),),
    )
    return ShuttleOperationRecord(
        "shuttle.map",
        {
            "semantics": {"kind": "mlir_attribute", "value": "#shuttle.map_semantics<transpose>"},
            "indexing_maps": [_affine_map(*input_dimensions), _affine_map(*range(len(input_dimensions)))],
        },
        (operand,),
        (_value(identifier, type_),),
        ((body,),),
    )


def _normalized_weight_map(
    identifier: int,
    operands: tuple[int, int, int],
    type_: ShuttleTypeRecord,
) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    score, maximum, normalizer = tuple(_value(identifier * 10 + index, scalar_type) for index in range(3))
    difference = _value(identifier * 10 + 3, scalar_type)
    exponential = _value(identifier * 10 + 4, scalar_type)
    normalized = _value(identifier * 10 + 5, scalar_type)
    body = ShuttleBlockRecord(
        (score, maximum, normalizer),
        (
            ShuttleOperationRecord("arith.subf", {}, (score.id, maximum.id), (difference,), ()),
            ShuttleOperationRecord("math.exp", {}, (difference.id,), (exponential,), ()),
            ShuttleOperationRecord("arith.divf", {}, (exponential.id, normalizer.id), (normalized,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (normalized.id,), (), ()),
        ),
    )
    return ShuttleOperationRecord("shuttle.map", {}, operands, (_value(identifier, type_),), ((body,),))


def _online_normalized_exp_fold(
    maximum_identifier: int,
    normalizer_identifier: int,
    operands: tuple[int, int, int, int],
    type_: ShuttleTypeRecord,
) -> ShuttleOperationRecord:
    scalar_type = ShuttleTypeRecord("scalar", "f32", ())
    score, weight, prior_maximum, prior_normalizer = tuple(_value(2000 + index, scalar_type) for index in range(4))
    maximum = _value(2004, scalar_type)
    score_difference = _value(2005, scalar_type)
    score_exponential = _value(2006, scalar_type)
    score_contribution = _value(2007, scalar_type)
    prior_difference = _value(2008, scalar_type)
    prior_exponential = _value(2009, scalar_type)
    prior_contribution = _value(2010, scalar_type)
    normalizer = _value(2011, scalar_type)
    body = ShuttleBlockRecord(
        (score, weight, prior_maximum, prior_normalizer),
        (
            ShuttleOperationRecord("arith.maximumf", {}, (score.id, prior_maximum.id), (maximum,), ()),
            ShuttleOperationRecord("arith.subf", {}, (score.id, maximum.id), (score_difference,), ()),
            ShuttleOperationRecord("math.exp", {}, (score_difference.id,), (score_exponential,), ()),
            ShuttleOperationRecord("arith.mulf", {}, (score_exponential.id, weight.id), (score_contribution,), ()),
            ShuttleOperationRecord("arith.subf", {}, (prior_maximum.id, maximum.id), (prior_difference,), ()),
            ShuttleOperationRecord("math.exp", {}, (prior_difference.id,), (prior_exponential,), ()),
            ShuttleOperationRecord(
                "arith.mulf", {}, (prior_exponential.id, prior_normalizer.id), (prior_contribution,), ()
            ),
            ShuttleOperationRecord("arith.addf", {}, (score_contribution.id, prior_contribution.id), (normalizer,), ()),
            ShuttleOperationRecord("shuttle.yield", {}, (maximum.id, normalizer.id), (), ()),
        ),
    )
    return ShuttleOperationRecord(
        "shuttle.fold",
        {
            "accumulator_types": [
                {"kind": "scalar", "element_type": "f32"},
                {"kind": "scalar", "element_type": "f32"},
            ],
            "indexing_maps": [
                _affine_map(0, 1, 2, 3, 4),
                _affine_map(0, 1, 2, 3, 4),
                _affine_map(0, 1, 2, 3),
                _affine_map(0, 1, 2, 3),
            ],
            "operandSegmentSizes": [2, 2],
            "order_free": True,
            "reduction_dimensions": [4],
        },
        operands,
        (_value(maximum_identifier, type_), _value(normalizer_identifier, type_)),
        ((body,),),
    )


def _combined_fold_forward_request() -> GpuPhysicalArtifactRequest:
    query_type = ShuttleTypeRecord("ranked_tensor", "bf16", (1, 1, 2, 4, 8))
    key_type = ShuttleTypeRecord("ranked_tensor", "bf16", (1, 1, 4, 8))
    value_type = ShuttleTypeRecord("ranked_tensor", "bf16", (1, 1, 4, 6))
    score_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 1, 2, 4, 4))
    state_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 1, 2, 4))
    output_accumulator_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 1, 2, 4, 6))
    output_type = ShuttleTypeRecord("ranked_tensor", "bf16", output_accumulator_type.shape)
    query, key, value = (_value(index, type_) for index, type_ in enumerate((query_type, key_type, value_type), 1))
    producer = _contract(
        100,
        (query.id, key.id),
        score_type,
        (
            _affine_map(0, 1, 2, 3, 5),
            _affine_map(0, 1, 4, 5),
            _affine_map(0, 1, 2, 3, 4),
        ),
    )
    producer = replace(
        producer,
        attributes=dict(
            producer.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"],
        ),
    )
    weight = _constant_map(101, score_type, "3F800000")
    maximum_seed = _constant_map(102, state_type, "FF800000")
    normalizer_seed = _constant_map(103, state_type, "0")
    fold = _online_normalized_exp_fold(
        104,
        105,
        (producer.results[0].id, weight.results[0].id, maximum_seed.results[0].id, normalizer_seed.results[0].id),
        state_type,
    )
    maximum = _identity_map(106, fold.results[0].id, score_type)
    normalizer = _identity_map(107, fold.results[1].id, score_type)
    normalized = _normalized_weight_map(
        108,
        (producer.results[0].id, maximum.results[0].id, normalizer.results[0].id),
        score_type,
    )
    consumer = _contract(
        109,
        (normalized.results[0].id, value.id),
        output_accumulator_type,
        (
            _affine_map(0, 1, 2, 3, 5),
            _affine_map(0, 1, 5, 4),
            _affine_map(0, 1, 2, 3, 4),
        ),
    )
    consumer = replace(
        consumer,
        attributes=dict(
            consumer.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"],
        ),
    )
    finalizer = _convert_scalar_map(110, consumer.results[0].id, output_accumulator_type, output_type)
    yield_operation = ShuttleOperationRecord("shuttle.yield", {}, (finalizer.results[0].id,), (), ())
    region = ShuttleOperationRecord(
        "shuttle.region",
        {"policy": {"kind": "mlir_attribute", "value": "#shuttle.policy<fast>"}},
        (),
        (),
        (
            (
                ShuttleBlockRecord(
                    (query, key, value),
                    (
                        producer,
                        weight,
                        maximum_seed,
                        normalizer_seed,
                        fold,
                        maximum,
                        normalizer,
                        normalized,
                        consumer,
                        finalizer,
                        yield_operation,
                    ),
                ),
            ),
        ),
    )
    buffers = (
        _buffer(0, "bf16", query_type.shape, live_in=True, source_argument=0, consumers=(0,)),
        _buffer(1, "bf16", key_type.shape, live_in=True, source_argument=1, consumers=(0,)),
        _buffer(2, "bf16", value_type.shape, live_in=True, source_argument=2, consumers=(1,)),
        _buffer(3, "f32", score_type.shape, producer=0, consumers=(1,)),
        _buffer(4, "f32", state_type.shape, producer=0, consumers=(1,)),
        _buffer(5, "f32", state_type.shape, producer=0, consumers=(1,)),
        _buffer(6, "bf16", output_type.shape, live_out=True, producer=1),
    )
    return _annotate_forward_capability(
        GpuPhysicalArtifactRequest(
            target="sm_90",
            fingerprint="b" * 64,
            region=region,
            materialization=_operation("shuttle.materialization_plan", operations=buffers),
            schedule=_operation("shuttle.schedule_plan"),
            event_lowering_plan=_event_lowering_plan(),
            binding_indices=(0, 1, 2, None, None, None, 0),
            region_result_indices=(None, None, None, None, None, None, 0),
            reuse_groups=tuple(range(7)),
        )
    )


def _q_major_combined_fold_forward_request() -> GpuPhysicalArtifactRequest:
    query_type = ShuttleTypeRecord("ranked_tensor", "bf16", (19, 1, 3, 2, 8))
    key_type = ShuttleTypeRecord("ranked_tensor", "bf16", (23, 1, 3, 8))
    value_type = ShuttleTypeRecord("ranked_tensor", "bf16", (23, 1, 3, 6))
    score_type = ShuttleTypeRecord("ranked_tensor", "f32", (19, 23, 1, 3, 2))
    state_type = ShuttleTypeRecord("ranked_tensor", "f32", (19, 1, 3, 2))
    output_accumulator_type = ShuttleTypeRecord("ranked_tensor", "f32", (19, 1, 3, 2, 6))
    output_type = ShuttleTypeRecord("ranked_tensor", "bf16", output_accumulator_type.shape)
    query, key, value = (_value(index, type_) for index, type_ in enumerate((query_type, key_type, value_type), 1))
    producer = _contract(
        100,
        (query.id, key.id),
        score_type,
        (
            _affine_map(0, 2, 3, 4, 5),
            _affine_map(1, 2, 3, 5),
            _affine_map(0, 1, 2, 3, 4),
        ),
    )
    producer = replace(
        producer,
        attributes=dict(
            producer.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"],
        ),
    )
    weight = _constant_map(101, score_type, "3F800000")
    maximum_seed = _constant_map(102, state_type, "FF800000")
    normalizer_seed = _constant_map(103, state_type, "0")
    fold = _online_normalized_exp_fold(
        104,
        105,
        (producer.results[0].id, weight.results[0].id, maximum_seed.results[0].id, normalizer_seed.results[0].id),
        state_type,
    )
    fold = replace(
        fold,
        attributes={
            **fold.attributes,
            "indexing_maps": [
                _affine_map(0, 1, 2, 3, 4),
                _affine_map(0, 1, 2, 3, 4),
                _affine_map(0, 2, 3, 4),
                _affine_map(0, 2, 3, 4),
            ],
            "reduction_dimensions": [1],
        },
    )
    maximum = _identity_map(106, fold.results[0].id, score_type)
    normalizer = _identity_map(107, fold.results[1].id, score_type)
    normalized = _normalized_weight_map(
        108,
        (producer.results[0].id, maximum.results[0].id, normalizer.results[0].id),
        score_type,
    )
    consumer = _contract(
        109,
        (normalized.results[0].id, value.id),
        output_accumulator_type,
        (
            _affine_map(0, 5, 1, 2, 3),
            _affine_map(5, 1, 2, 4),
            _affine_map(0, 1, 2, 3, 4),
        ),
    )
    consumer = replace(
        consumer,
        attributes=dict(
            consumer.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"],
        ),
    )
    finalizer = _convert_scalar_map(110, consumer.results[0].id, output_accumulator_type, output_type)
    yield_operation = ShuttleOperationRecord("shuttle.yield", {}, (finalizer.results[0].id,), (), ())
    region = ShuttleOperationRecord(
        "shuttle.region",
        {"policy": {"kind": "mlir_attribute", "value": "#shuttle.policy<fast>"}},
        (),
        (),
        (
            (
                ShuttleBlockRecord(
                    (query, key, value),
                    (
                        producer,
                        weight,
                        maximum_seed,
                        normalizer_seed,
                        fold,
                        maximum,
                        normalizer,
                        normalized,
                        consumer,
                        finalizer,
                        yield_operation,
                    ),
                ),
            ),
        ),
    )
    buffers = (
        _buffer(0, "bf16", query_type.shape, live_in=True, source_argument=0, consumers=(0,)),
        _buffer(1, "bf16", key_type.shape, live_in=True, source_argument=1, consumers=(0,)),
        _buffer(2, "bf16", value_type.shape, live_in=True, source_argument=2, consumers=(1,)),
        _buffer(3, "f32", score_type.shape, producer=0, consumers=(1,)),
        _buffer(4, "f32", state_type.shape, producer=0, consumers=(1,)),
        _buffer(5, "f32", state_type.shape, producer=0, consumers=(1,)),
        _buffer(6, "bf16", output_type.shape, live_out=True, producer=1),
    )
    return _annotate_forward_capability(
        GpuPhysicalArtifactRequest(
            target="sm_90",
            fingerprint="b" * 64,
            region=region,
            materialization=_operation("shuttle.materialization_plan", operations=buffers),
            schedule=_operation("shuttle.schedule_plan"),
            event_lowering_plan=_event_lowering_plan(),
            binding_indices=(0, 1, 2, None, None, None, 0),
            region_result_indices=(None, None, None, None, None, None, 0),
            reuse_groups=tuple(range(7)),
        )
    )


def _reordered_consumer_output_forward_request() -> GpuPhysicalArtifactRequest:
    request = _q_major_combined_fold_forward_request()
    body = request.region.body()
    immediate_output_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 3, 6, 19, 2))
    public_output_type = ShuttleTypeRecord("ranked_tensor", "bf16", (19, 1, 3, 2, 6))
    operations: list[ShuttleOperationRecord] = []
    for operation in body.operations:
        if operation.name == "shuttle.contract" and operation.results[0].id == 100:
            operation = replace(
                operation,
                attributes={
                    **operation.attributes,
                    "indexing_maps": [
                        _affine_map(3, 0, 1, 4, 5),
                        _affine_map(2, 0, 1, 5),
                        _affine_map(3, 2, 0, 1, 4),
                    ],
                },
            )
        elif operation.name == "shuttle.contract" and operation.results[0].id == 109:
            operation = replace(
                operation,
                attributes={
                    **operation.attributes,
                    "indexing_maps": [
                        _affine_map(3, 2, 0, 1, 4),
                        _affine_map(2, 0, 1, 5),
                        _affine_map(0, 1, 5, 3, 4),
                    ],
                },
                results=(_value(109, immediate_output_type),),
            )
        elif operation.name == "shuttle.map" and operation.results[0].id == 110:
            operation = _convert_scalar_map(110, 109, immediate_output_type, public_output_type)
            operation = replace(
                operation,
                attributes={
                    **operation.attributes,
                    "indexing_maps": [_affine_map(1, 2, 4, 0, 3), _affine_map(0, 1, 2, 3, 4)],
                },
            )
        operations.append(operation)
    return _annotate_forward_capability(
        replace(
            request,
            region=replace(request.region, regions=((replace(body, operations=tuple(operations)),),)),
        )
    )


def _q_major_offset_domain_forward_request() -> GpuPhysicalArtifactRequest:
    request = _q_major_combined_fold_forward_request()
    body = request.region.body()
    producer = next(operation for operation in body.operations if operation.name == "shuttle.contract")
    score_type = producer.results[0].type
    owner_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (19,))
    owner_reshape_type = ShuttleTypeRecord("ranked_tensor", "i32", (19, 1, 1, 1, 1))
    stream_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (23,))
    stream_reshape_type = ShuttleTypeRecord("ranked_tensor", "i32", (1, 23, 1, 1, 1))
    relation_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (19, 23, 1, 1, 1))
    relation_predicate_type = ShuttleTypeRecord("ranked_tensor", "i1", relation_index_type.shape)
    score_predicate_type = ShuttleTypeRecord("ranked_tensor", "i1", score_type.shape)
    scalar_index_type = ShuttleTypeRecord("ranked_tensor", "i32", ())

    owner_iota = _relation(120, "iota", (), owner_index_type, parameters=(0,))
    owner_reshape = _relation(121, "reshape", (120,), owner_reshape_type)
    stream_iota = _relation(122, "iota", (), stream_index_type, parameters=(0,))
    stream_reshape = _relation(123, "reshape", (122,), stream_reshape_type)
    offset = _relation(124, "constant", (), scalar_index_type, parameters=(4,))
    offset_broadcast = _relation(125, "broadcast_in_dim", (124,), owner_reshape_type)
    shifted_owner = _relation(126, "add", (121, 125), owner_reshape_type)
    stream_broadcast = _relation(
        127,
        "broadcast_in_dim",
        (123,),
        relation_index_type,
        parameters=(0, 1, 2, 3, 4),
    )
    owner_broadcast = _relation(
        128,
        "broadcast_in_dim",
        (126,),
        relation_index_type,
        parameters=(0, 1, 2, 3, 4),
    )
    compare = _relation(129, "compare", (127, 128), relation_predicate_type, parameters=("SIGNED", "LE"))
    predicate_broadcast = _relation(
        130,
        "broadcast_in_dim",
        (129,),
        score_predicate_type,
        parameters=(0, 1, 2, 3, 4),
    )
    rejected = _relation(131, "constant", (), ShuttleTypeRecord("ranked_tensor", "f32", ()), parameters=("FF800000",))
    rejected_broadcast = _relation(132, "broadcast_in_dim", (131,), score_type)
    select = _relation(133, "select", (130, producer.results[0].id, 132), score_type)

    replacements: list[ShuttleOperationRecord] = []
    for operation in body.operations:
        if operation is producer:
            replacements.extend(
                (
                    operation,
                    owner_iota,
                    owner_reshape,
                    stream_iota,
                    stream_reshape,
                    offset,
                    offset_broadcast,
                    shifted_owner,
                    stream_broadcast,
                    owner_broadcast,
                    compare,
                    predicate_broadcast,
                    rejected,
                    rejected_broadcast,
                    select,
                )
            )
            continue
        if operation.name == "shuttle.fold":
            operation = replace(operation, operands=(select.results[0].id, *operation.operands[1:]))
        elif operation.name == "shuttle.map" and producer.results[0].id in operation.operands:
            operation = replace(
                operation,
                operands=tuple(
                    select.results[0].id if operand == producer.results[0].id else operand
                    for operand in operation.operands
                ),
            )
        replacements.append(operation)
    region = replace(request.region, regions=((replace(body, operations=tuple(replacements)),),))
    return _annotate_forward_capability(replace(request, region=region))


def _q_major_reversed_scaled_tanh_offset_domain_forward_request() -> GpuPhysicalArtifactRequest:
    request = _q_major_offset_domain_forward_request()
    body = request.region.body()
    producer = next(operation for operation in body.operations if operation.name == "shuttle.contract")
    producer_maps = producer.attributes["indexing_maps"]
    reversed_producer = replace(
        producer,
        operands=tuple(reversed(producer.operands)),
        attributes={**producer.attributes, "indexing_maps": [producer_maps[1], producer_maps[0], producer_maps[2]]},
    )
    score_map = _scaled_tanh_scalar_map(134, producer.results[0].id, producer.results[0].type)
    operations: list[ShuttleOperationRecord] = []
    for operation in body.operations:
        if operation is producer:
            operations.extend((reversed_producer, score_map))
            continue
        if operation.name == "shuttle.relation" and operation.results[0].id == 133:
            operation = replace(
                operation,
                operands=tuple(
                    score_map.results[0].id if operand == producer.results[0].id else operand
                    for operand in operation.operands
                ),
            )
        operations.append(operation)
    return _annotate_forward_capability(
        replace(
            request,
            region=replace(request.region, regions=((replace(body, operations=tuple(operations)),),)),
        )
    )


def _ordinary_jax_relation_chained_forward_request() -> GpuPhysicalArtifactRequest:
    request = _q_major_reversed_scaled_tanh_offset_domain_forward_request()
    body = request.region.body()
    fold = next(operation for operation in body.operations if operation.name == "shuttle.fold")
    score_type = next(
        operation.results[0].type
        for operation in body.operations
        if operation.name == "shuttle.contract" and operation.results[0].type.element_type == "f32"
    )
    maximum = _relation(
        106,
        "broadcast_in_dim",
        (fold.results[0].id,),
        score_type,
        parameters=(0, 2, 3, 4),
    )
    normalizer = _relation(
        107,
        "broadcast_in_dim",
        (fold.results[1].id,),
        score_type,
        parameters=(0, 2, 3, 4),
    )
    normalized_relation = _relation(135, "reshape", (108,), score_type)

    operations: list[ShuttleOperationRecord] = []
    for operation in body.operations:
        if operation.results and operation.results[0].id == maximum.results[0].id:
            operation = maximum
        elif operation.results and operation.results[0].id == normalizer.results[0].id:
            operation = normalizer
        elif operation.name == "shuttle.contract" and 108 in operation.operands:
            operations.append(normalized_relation)
            operation = replace(
                operation,
                operands=tuple(135 if operand == 108 else operand for operand in operation.operands),
            )
        operations.append(operation)
    return _annotate_forward_capability(
        replace(
            request,
            region=replace(request.region, regions=((replace(body, operations=tuple(operations)),),)),
        )
    )


def _ordinary_jax_transposed_score_forward_request() -> GpuPhysicalArtifactRequest:
    request = _ordinary_jax_relation_chained_forward_request()
    body = request.region.body()
    transposed_score_type = ShuttleTypeRecord("ranked_tensor", "f32", (23, 19, 1, 3, 2))
    transpose = _transpose_map(
        136,
        133,
        transposed_score_type,
        input_dimensions=(1, 0, 2, 3, 4),
    )

    operations: list[ShuttleOperationRecord] = []
    for operation in body.operations:
        if operation.results and operation.results[0].id == 133:
            operations.extend((operation, transpose))
            continue
        if operation.name == "shuttle.fold":
            operation = replace(
                operation,
                operands=(transpose.results[0].id, *operation.operands[1:]),
                attributes={
                    **operation.attributes,
                    "indexing_maps": [
                        _affine_map(0, 1, 2, 3, 4),
                        _affine_map(0, 1, 2, 3, 4),
                        _affine_map(1, 2, 3, 4),
                        _affine_map(1, 2, 3, 4),
                    ],
                    "reduction_dimensions": [0],
                },
            )
        elif operation.results and operation.results[0].id in (106, 107):
            operation = replace(
                operation,
                attributes={**operation.attributes, "parameters": (1, 2, 3, 4)},
                results=(_value(operation.results[0].id, transposed_score_type),),
            )
        elif operation.results and operation.results[0].id == 108:
            operation = replace(
                operation,
                operands=(transpose.results[0].id, *operation.operands[1:]),
                results=(_value(108, transposed_score_type),),
            )
        elif operation.results and operation.results[0].id == 135:
            operation = replace(operation, results=(_value(135, transposed_score_type),))
        elif operation.name == "shuttle.contract" and 135 in operation.operands:
            maps = operation.attributes["indexing_maps"]
            operation = replace(
                operation,
                attributes={**operation.attributes, "indexing_maps": [_affine_map(5, 0, 1, 2, 3), *maps[1:]]},
            )
        operations.append(operation)
    return _annotate_forward_capability(
        replace(
            request,
            region=replace(request.region, regions=((replace(body, operations=tuple(operations)),),)),
        )
    )


def _saved_state_request() -> GpuPhysicalArtifactRequest:
    owner_shape = (1, 1, 2, 128)
    data_shape = (1, 1, 2, 2, 128, 128)
    argument_order = (3, 1, 4, 0, 2, 5)
    operations = (
        _buffer(0, "bf16", data_shape, live_in=True, source_argument=argument_order[0], consumers=(0, 1)),
        _buffer(1, "bf16", data_shape, live_in=True, source_argument=argument_order[1], consumers=(0, 1)),
        _buffer(2, "bf16", data_shape, live_in=True, source_argument=argument_order[2], consumers=(0, 1)),
        _buffer(3, "bf16", data_shape, live_in=True, source_argument=argument_order[3], consumers=(0,)),
        _buffer(4, "bf16", data_shape, live_in=True, source_argument=argument_order[4], consumers=(0, 1)),
        _buffer(5, "f32", owner_shape, live_in=True, source_argument=argument_order[5], consumers=(0, 1)),
        _buffer(6, "bf16", data_shape, live_out=True, producer=0),
        _buffer(7, "bf16", data_shape, live_out=True, producer=1),
        _buffer(8, "bf16", data_shape, live_out=True, producer=1),
        _buffer(9, "f32", owner_shape, producer=0, consumers=(1,)),
        _task(
            0,
            inputs=(0, 1, 2, 3, 4, 5),
            outputs=(6, 9),
            dependencies=(),
            reductions=(5,),
            owner=(0, 1, 2, 3),
            vector=(4,),
            state_inputs=(5,),
            state_outputs=(9,),
        ),
        _task(
            1,
            inputs=(0, 1, 2, 4, 5, 9),
            outputs=(7, 8),
            dependencies=(0,),
            reductions=(4, 5),
            owner=(0, 1, 2),
            vector=(3,),
            state_inputs=(5, 9),
            state_outputs=(),
        ),
    )
    materialization = _operation(
        "shuttle.materialization_plan",
        attributes={"stateful_contract_fold_pullback": True},
        operations=operations,
    )
    owner_task = operations[-2]
    streamed_task = operations[-1]
    schedule = _operation(
        "shuttle.schedule_plan",
        operations=(
            _schedule_task(owner_task, tile=(1, 1, 2, 2, 128, 128), serial_tiles=1),
            _schedule_task(streamed_task, tile=(1, 1, 2, 2, 128, 64), serial_tiles=2),
        ),
    )
    data_type = ShuttleTypeRecord("ranked_tensor", "bf16", data_shape)
    state_type = ShuttleTypeRecord("ranked_tensor", "f32", owner_shape)
    argument_types = (
        data_type,
        data_type,
        data_type,
        data_type,
        data_type,
        state_type,
    )
    arguments = tuple(_value(100 + index, type_) for index, type_ in enumerate(argument_types))
    finalized_output, producer_stream, finalizer_cotangent, producer_owner, consumer, saved_state = (
        argument.id for argument in arguments
    )
    score = _contract(
        200,
        (producer_owner, producer_stream),
        data_type,
        (
            _affine_map(0, 1, 2, 3, 4),
            _affine_map(0, 1, 4, 5),
            _affine_map(0, 1, 2, 3, 5),
        ),
    )
    normalized = _map(201, (score.results[0].id, saved_state), data_type)
    consumer_pullback = _contract(
        202,
        (finalizer_cotangent, consumer),
        data_type,
        (
            _affine_map(0, 1, 2, 3, 4),
            _affine_map(0, 1, 4, 5),
            _affine_map(0, 1, 2, 3, 5),
        ),
    )
    score_pullback = _map(203, (normalized.results[0].id, consumer_pullback.results[0].id), data_type)
    output_product = _multiply_map(204, (finalized_output, finalizer_cotangent), data_type)
    output_state = _sum_fold(205, output_product.results[0].id, state_type)
    adjusted_score_pullback = _normalized_pullback_map(
        206,
        (normalized.results[0].id, consumer_pullback.results[0].id, output_state.results[0].id),
        data_type,
    )
    producer_owner_cotangent = _contract(
        207,
        (adjusted_score_pullback.results[0].id, producer_stream),
        data_type,
        (
            _affine_map(0, 1, 2, 3, 5),
            _affine_map(0, 1, 4, 5),
            _affine_map(0, 1, 2, 3, 4),
        ),
    )
    producer_stream_cotangent = _contract(
        208,
        (producer_owner, adjusted_score_pullback.results[0].id),
        data_type,
        (
            _affine_map(0, 1, 3, 4, 5),
            _affine_map(0, 1, 2, 4, 5),
            _affine_map(0, 1, 2, 3),
        ),
    )
    consumer_cotangent = _contract(
        209,
        (normalized.results[0].id, finalizer_cotangent),
        data_type,
        (
            _affine_map(0, 1, 2, 4, 5),
            _affine_map(0, 1, 3, 4, 5),
            _affine_map(0, 1, 2, 3),
        ),
    )
    yield_operation = ShuttleOperationRecord(
        "shuttle.yield",
        {},
        (
            producer_owner_cotangent.results[0].id,
            producer_stream_cotangent.results[0].id,
            consumer_cotangent.results[0].id,
        ),
        (),
        (),
    )
    region = ShuttleOperationRecord(
        "shuttle.region",
        {},
        (),
        (),
        (
            (
                ShuttleBlockRecord(
                    arguments,
                    (
                        score,
                        normalized,
                        consumer_pullback,
                        score_pullback,
                        output_product,
                        output_state,
                        adjusted_score_pullback,
                        producer_owner_cotangent,
                        producer_stream_cotangent,
                        consumer_cotangent,
                        yield_operation,
                    ),
                ),
            ),
        ),
    )
    region_result_indices: list[int | None] = [None] * 10
    region_result_indices[6:9] = [0, 1, 2]
    binding_indices = (*argument_order, 0, 1, 2, None)
    return GpuPhysicalArtifactRequest(
        target="sm_90",
        fingerprint="a" * 64,
        region=region,
        materialization=materialization,
        schedule=schedule,
        event_lowering_plan=_event_lowering_plan(),
        binding_indices=binding_indices,
        region_result_indices=tuple(region_result_indices),
        reuse_groups=tuple(range(10)),
    )


def _replace_buffer_attributes(
    request: GpuPhysicalArtifactRequest,
    ordinal: int,
    attributes: dict[str, object],
) -> GpuPhysicalArtifactRequest:
    body = request.materialization.body()
    operations = tuple(
        replace(operation, attributes=attributes) if index == ordinal else operation
        for index, operation in enumerate(body.operations)
    )
    materialization = replace(
        request.materialization,
        regions=((replace(body, operations=operations),),),
    )
    return replace(request, materialization=materialization)


def _replace_region_operation(
    request: GpuPhysicalArtifactRequest,
    ordinal: int,
    operation: ShuttleOperationRecord,
) -> GpuPhysicalArtifactRequest:
    body = request.region.body()
    operations = (*body.operations[:ordinal], operation, *body.operations[ordinal + 1 :])
    return replace(request, region=replace(request.region, regions=((replace(body, operations=operations),),)))


def test_physical_region_extracts_canonical_two_state_normalized_exp_fold() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(_combined_fold_forward_request())

    assert region.maximum is region.normalizer
    assert len(region.maximum.results) == 2
    assert region.score_expression == scalar_input("contract.producer")
    assert region.finalizer_expression.kind is ScalarExpressionKind.CONVERT
    assert region.geometry.relation_group_shape == (2,)
    assert region.geometry.owner_extent == 4
    assert region.geometry.streamed_extent == 4
    assert region.geometry.producer_feature_extent == 8
    assert region.geometry.consumer_feature_extent == 6
    assert region.geometry.batch_dimensions == (0,)
    assert region.geometry.shared_relation_dimension == 1
    assert region.geometry.relation_group_dimensions == (2,)
    assert region.geometry.owner_dimension == 3
    assert region.geometry.streamed_dimension == 4


def test_physical_region_declines_unrelated_typed_algebra() -> None:
    request = _combined_fold_forward_request()
    body = request.region.body()
    unrelated_region = replace(
        request.region,
        regions=((replace(body, operations=(body.operations[-1],)),),),
    )

    with pytest.raises(ValueError, match="planner-declared normalized-exp source ordinal is stale"):
        physical_normalized_exp_contract.physical_normalized_exp_contract_region(
            replace(request, region=unrelated_region)
        )

    with pytest.raises(ValueError, match="planner-declared normalized-exp source ordinal is stale"):
        physical_normalized_exp_contract_match(replace(request, region=unrelated_region))


def test_physical_match_requires_a_typed_capability_for_each_direction() -> None:
    forward = physical_normalized_exp_contract_match(_combined_fold_forward_request())
    saved_reverse = physical_normalized_exp_contract_match(_saved_state_request())

    assert isinstance(forward, PhysicalNormalizedExpForwardMatch)
    assert saved_reverse is None
    assert forward.region.score_expression == scalar_input("contract.producer")


def test_physical_match_reports_fast_policy_at_physical_boundary() -> None:
    request = _combined_fold_forward_request()
    source_ordered = replace(
        request.region,
        attributes={"policy": {"kind": "mlir_attribute", "value": "#shuttle.policy<source_ordered>"}},
    )

    with pytest.raises(ValueError, match="requires FAST numerical policy") as exc_info:
        physical_normalized_exp_contract_match(replace(request, region=source_ordered))

    assert not isinstance(exc_info.value, PhysicalNormalizedExpContractNotApplicable)


def test_physical_region_derives_owner_and_stream_from_reversed_producer_relations() -> None:
    request = _combined_fold_forward_request()
    body = request.region.body()
    producer_index, producer = next(
        (index, operation)
        for index, operation in enumerate(body.operations)
        if operation.name == "shuttle.contract" and operation.results[0].id == 100
    )
    maps = producer.attributes["indexing_maps"]
    reversed_producer = replace(
        producer,
        operands=tuple(reversed(producer.operands)),
        attributes={**producer.attributes, "indexing_maps": [maps[1], maps[0], maps[2]]},
    )

    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _replace_region_operation(request, producer_index, reversed_producer)
    )

    assert region.producer_left_argument == 0
    assert region.producer_right_argument == 1
    assert region.geometry.batch_shape == (1,)
    assert region.geometry.shared_relation_extent == 1
    assert region.geometry.relation_group_shape == (2,)
    assert region.geometry.owner_extent == 4
    assert region.geometry.streamed_extent == 4
    assert region.geometry.batch_dimensions == (0,)
    assert region.geometry.shared_relation_dimension == 1
    assert region.geometry.relation_group_dimensions == (2,)
    assert region.geometry.owner_dimension == 3
    assert region.geometry.streamed_dimension == 4


def test_physical_region_derives_q_major_owner_from_compact_relation_projection() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _q_major_combined_fold_forward_request()
    )

    assert region.geometry.batch_shape == (1,)
    assert region.geometry.shared_relation_extent == 3
    assert region.geometry.relation_group_shape == (2,)
    assert region.geometry.owner_extent == 19
    assert region.geometry.streamed_extent == 23
    assert region.geometry.batch_dimensions == (2,)
    assert region.geometry.shared_relation_dimension == 3
    assert region.geometry.relation_group_dimensions == (4,)
    assert region.geometry.owner_dimension == 0
    assert region.geometry.streamed_dimension == 1


def test_online_cut_types_preserve_source_fold_axis_relations() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _q_major_combined_fold_forward_request()
    )

    score_type, maximum_type, normalizer_type = normalized_exp_sufficient_state_cut_types(region)

    assert score_type.shape == (19, 23, 1, 3, 2)
    assert maximum_type.shape == (19, 1, 3, 2)
    assert normalizer_type == maximum_type
    assert maximum_type.shape != (
        *region.geometry.batch_shape,
        region.geometry.shared_relation_extent,
        *region.geometry.relation_group_shape,
        region.geometry.owner_extent,
    )


def test_physical_region_uses_finalizer_relation_for_public_output_layout() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _reordered_consumer_output_forward_request()
    )

    assert contract_indexing_dimensions(region.producer) == (
        (3, 0, 1, 4, 5),
        (2, 0, 1, 5),
        (3, 2, 0, 1, 4),
    )
    assert contract_indexing_dimensions(region.consumer)[2] == (0, 1, 5, 3, 4)
    assert region.public_output_dimensions == (3, 0, 1, 4, 5)
    assert region.geometry.batch_dimensions == (0,)
    assert region.geometry.shared_relation_dimension == 1
    assert region.geometry.relation_group_dimensions == (4,)
    assert region.geometry.owner_dimension == 3
    assert region.geometry.streamed_dimension == 2
    assert region.geometry.consumer_feature_dimension == 5
    assert region.finalizer_expression.kind is ScalarExpressionKind.CONVERT


def test_combined_fold_weighted_contract_near_miss_reports_match_diagnostics() -> None:
    request = _combined_fold_forward_request()
    body = request.region.body()
    consumer_index, consumer = next(
        (index, operation)
        for index, operation in enumerate(body.operations)
        if operation.name == "shuttle.contract" and operation.results[0].id == 109
    )
    maps = consumer.attributes["indexing_maps"]
    near_miss = replace(
        consumer,
        attributes={**consumer.attributes, "indexing_maps": [maps[0], _affine_map(0, 1, 4, 5), maps[2]]},
    )

    with pytest.raises(ValueError) as exc_info:
        physical_normalized_exp_contract.physical_normalized_exp_contract_region(
            _replace_region_operation(request, consumer_index, near_miss)
        )

    message = str(exc_info.value)
    assert "producer Contract does not expose owner, stream, and shared relation axes" in message
    assert "producer_dimensions=" in message
    assert "consumer_dimensions=" in message


def test_geometry_near_miss_reports_each_compact_projection_candidate() -> None:
    request = _q_major_combined_fold_forward_request()
    body = request.region.body()
    consumer_index, consumer = next(
        (index, operation)
        for index, operation in enumerate(body.operations)
        if operation.name == "shuttle.contract" and operation.results[0].id == 109
    )
    maps = consumer.attributes["indexing_maps"]
    near_miss = replace(
        consumer,
        attributes={**consumer.attributes, "indexing_maps": [maps[0], maps[1], _affine_map(0, 1, 2, 4, 3)]},
    )

    with pytest.raises(ValueError) as exc_info:
        physical_normalized_exp_contract.physical_normalized_exp_contract_region(
            _replace_region_operation(request, consumer_index, near_miss)
        )

    message = str(exc_info.value)
    assert "producer_dimensions=(operands=((0, 2, 3, 4, 5), (1, 2, 3, 5)),result=(0, 1, 2, 3, 4))" in message
    assert "consumer_dimensions=(normalized=(0, 5, 1, 2, 3),value=(5, 1, 2, 4),result=(0, 1, 2, 4, 3))" in message
    assert "producer_to_consumer_lineage=((0, 0), (1, 5), (2, 1), (3, 2), (4, 3))" in message
    assert "consumer_streamed=(dimension=d5,extent=23)" in message
    assert "owner_operand=0,stream_operand=1,common=(2, 3),owner_only=(0, 4),stream_only=(1,)" in message
    assert "owner=d0,groups=(4,),packed=(3, 4)" in message
    assert "producer_compact=accepted(dimensions=(0, 2, 3, 4, 5),projected=(3, 4))" in message
    assert "public_output_compact=rejected(dimensions=(0, 1, 2, 4, 3),projected=(2, 3))" in message
    assert "owner=d4,groups=(0,),packed=(3, 0)" in message
    assert 'raw={"dimensions":6,"kind":"affine_map"' in message


def test_physical_region_combines_reversed_producer_scalar_map_and_affine_domain() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _q_major_reversed_scaled_tanh_offset_domain_forward_request()
    )

    assert region.producer_left_argument == 0
    assert region.producer_right_argument == 1
    assert region.geometry.owner_dimension == 0
    assert region.geometry.streamed_dimension == 1
    assert region.domain_predicate is not None
    assert region.domain_predicate.predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((1, 1),), "constant": 0},
        "right": {"coefficients": ((0, 1),), "constant": 4},
    }
    assert serialize_scalar_expression(region.score_expression) == (
        '{"kind":"multiply","operands":[{"constant":12.0,"kind":"constant"},'
        '{"kind":"tanh","operands":[{"kind":"divide","operands":['
        '{"kind":"multiply","operands":[{"input":"contract.producer","kind":"input"},'
        '{"constant":0.1767766922712326,"kind":"constant"}]},{"constant":12.0,"kind":"constant"}]}]}]}'
    )


def test_normalized_exp_program_remaps_affine_domain_to_logical_axis_ids() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _q_major_reversed_scaled_tanh_offset_domain_forward_request()
    )

    program = normalized_exp_contract_program(
        region,
        schedule=StreamingTileSchedule(query_tile_size=16, key_value_tile_size=16, pipeline_depth=2),
    )

    assert program.region.domain_predicate is not None
    assert program.region.domain_predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((4, 1),), "constant": 0},
        "right": {"coefficients": ((3, 1),), "constant": 4},
    }


def test_affine_domain_changes_region_identity_without_selecting_a_schedule_or_traversal() -> None:
    schedule = Sm90ForwardPipelineSchedule(Sm90ContractSchedule(128, 128), stages=2)
    unrestricted = normalized_exp_contract_program(
        physical_normalized_exp_contract.physical_normalized_exp_contract_region(
            _q_major_combined_fold_forward_request()
        ),
        schedule=schedule,
    )
    restricted = normalized_exp_contract_program(
        physical_normalized_exp_contract.physical_normalized_exp_contract_region(
            _q_major_offset_domain_forward_request()
        ),
        schedule=schedule,
    )

    unrestricted_score = lower_normalized_exp_score_map(unrestricted.region)
    restricted_score = lower_normalized_exp_score_map(restricted.region)

    assert unrestricted.region_fingerprint != restricted.region_fingerprint
    assert unrestricted.schedule == restricted.schedule
    assert unrestricted_score.domain_traversal is ContractDomainTraversal.FULL
    assert restricted_score.domain_traversal is ContractDomainTraversal.FULL
    unrestricted_events = derive_streaming_physical_event_schedule(
        streaming_contract_fold_event_descriptor(unrestricted)
    )
    restricted_events = derive_streaming_physical_event_schedule(streaming_contract_fold_event_descriptor(restricted))
    assert canonical_streaming_physical_event_schedule_record(
        unrestricted_events
    ) == canonical_streaming_physical_event_schedule_record(restricted_events)


def test_physical_region_finds_weighted_contract_through_scalar_maps_and_index_relations() -> None:
    request = _ordinary_jax_relation_chained_forward_request()
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(request)

    assert 135 in region.consumer.operands
    assert region.exponential.name == "shuttle.relation"
    assert region.exponential.results[0].id == 135
    assert region.producer_left_argument == 0
    assert region.producer_right_argument == 1
    assert region.geometry.owner_dimension == 0
    assert region.geometry.streamed_dimension == 1
    assert region.producer_to_consumer_dimensions == ((0, 0), (1, 5), (2, 1), (3, 2), (4, 3))
    assert normalized_exp_consumer_dimension_roles(region) == ((1,), 2, (3,), 0)
    assert region.domain_predicate is not None
    assert region.domain_predicate.predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((1, 1),), "constant": 0},
        "right": {"coefficients": ((0, 1),), "constant": 4},
    }


def test_physical_region_composes_transposed_score_relation_into_weighted_contract() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _ordinary_jax_transposed_score_forward_request()
    )

    assert region.producer_left_argument == 0
    assert region.producer_right_argument == 1
    assert region.geometry.owner_dimension == 0
    assert region.geometry.streamed_dimension == 1
    assert region.geometry.normalized_dimension == 5
    assert region.geometry.owner_extent == 19
    assert region.geometry.streamed_extent == 23
    assert region.domain_predicate is not None
    assert region.domain_predicate.predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((1, 1),), "constant": 0},
        "right": {"coefficients": ((0, 1),), "constant": 4},
    }
    assert ScalarExpressionKind.TANH in {
        expression.kind
        for expression in (
            region.score_expression,
            *region.score_expression.operands,
            *region.score_expression.operands[1].operands,
        )
    }


def test_consumer_dimension_roles_follow_score_lineage_not_contract_map_position() -> None:
    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _ordinary_jax_relation_chained_forward_request()
    )
    producer = replace(
        region.producer,
        attributes={
            **region.producer.attributes,
            "indexing_maps": [
                _affine_map(2, 0, 1, 5),
                _affine_map(3, 0, 1, 4, 5),
                _affine_map(0, 1, 2, 3, 4),
            ],
        },
    )
    consumer = replace(
        region.consumer,
        attributes={
            **region.consumer.attributes,
            "indexing_maps": [
                _affine_map(3, 5, 0, 1, 4),
                _affine_map(5, 0, 1, 2),
                _affine_map(0, 1, 2, 3, 4),
            ],
        },
    )
    region = replace(
        region,
        producer=producer,
        consumer=consumer,
        producer_to_consumer_dimensions=((0, 0), (1, 1), (2, 5), (3, 3), (4, 4)),
        geometry=replace(
            region.geometry,
            batch_dimensions=(0,),
            shared_relation_dimension=1,
            relation_group_dimensions=(4,),
            owner_dimension=3,
            streamed_dimension=2,
            normalized_dimension=5,
            consumer_feature_dimension=2,
        ),
    )
    producer_score_dimensions = contract_indexing_dimensions(region.producer)[2]
    consumer_normalized_dimensions = contract_indexing_dimensions(region.consumer)[0]
    positional_relation = tuple(zip(producer_score_dimensions, consumer_normalized_dimensions, strict=True))

    assert positional_relation != region.producer_to_consumer_dimensions
    assert normalized_exp_consumer_dimension_roles(region) == ((0,), 1, (4,), 3)
    assert contract_indexing_dimensions(region.consumer)[1] == (5, 0, 1, 2)


def test_sm90_algebraic_event_cover_maps_exact_relation_points_to_common_ctas() -> None:
    geometry = NormalizedExpContractGeometry(
        batch_shape=(1,),
        shared_relation_extent=8,
        relation_group_shape=(4,),
        owner_extent=2048,
        streamed_extent=2048,
        producer_feature_extent=128,
        consumer_feature_extent=128,
        batch_dimensions=(0,),
        shared_relation_dimension=1,
        relation_group_dimensions=(2,),
        owner_dimension=3,
        streamed_dimension=4,
        producer_feature_dimension=5,
        normalized_dimension=4,
        consumer_feature_dimension=5,
    )
    layout = sm90_relation_packed_layout(
        batch_shape=geometry.batch_shape,
        shared_relation_extent=geometry.shared_relation_extent,
        relation_group_shape=geometry.relation_group_shape,
        owner_extent=geometry.owner_extent,
        streamed_extent=geometry.streamed_extent,
        producer_feature_extent=geometry.producer_feature_extent,
        consumer_feature_extent=geometry.consumer_feature_extent,
    )
    schedule = Sm90ForwardPipelineSchedule(
        Sm90ContractSchedule(128, 128, relation_layout=layout),
        stages=2,
    )
    producer_row = {
        "kind": "add",
        "left": {
            "kind": "floor_divide",
            "left": {"kind": "dimension", "position": 0},
            "right": {"kind": "constant", "value": 16},
        },
        "right": {"kind": "dimension", "position": 1},
    }
    cover = CuteStructuralEventCover(
        CuteSourceTaskRole.STATE_ROOT,
        CuteSourceTaskRole.STATE_TERMINAL,
        CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
        CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
        CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
    )

    (physical_cover,) = derive_lpt_algebraic_refinement_covers(
        cover,
        producer_row=affine_expression_from_record(producer_row),
        relation_dimensions=2,
        batch=geometry.batch_size,
        shared_relation=geometry.shared_relation_extent,
        relation_group=geometry.relation_group_size,
        owner_extent=geometry.owner_extent,
        producer_feature_extent=geometry.producer_feature_extent,
        consumer_feature_extent=geometry.consumer_feature_extent,
        owner_tile=schedule.tile_m,
    )

    assert physical_cover.relation_subset is None
    assert physical_cover.producer_owner == physical_cover.consumer_owner
    assert physical_cover.producer_time == physical_cover.consumer_time
    assert physical_cover.producer_owner is not None
    owner_x = physical_cover.producer_owner.results[0]

    def evaluate(expression: CuteAffineExpression, coordinates: tuple[int, ...]) -> int:
        if expression.kind is CuteAffineExpressionKind.DIMENSION:
            return coordinates[expression.payload]
        if expression.kind is CuteAffineExpressionKind.CONSTANT:
            return expression.payload
        left = evaluate(expression.operands[0], coordinates)
        right = evaluate(expression.operands[1], coordinates)
        if expression.kind is CuteAffineExpressionKind.ADD:
            return left + right
        if expression.kind is CuteAffineExpressionKind.MULTIPLY:
            return left * right
        if expression.kind is CuteAffineExpressionKind.MODULO:
            return left % right
        if expression.kind is CuteAffineExpressionKind.FLOOR_DIVIDE:
            return left // right
        if expression.kind is CuteAffineExpressionKind.CEIL_DIVIDE:
            return -(-left // right)
        raise AssertionError(expression.kind)

    for row in (0, 32, 2048, 8192, 65535):
        coordinates = (row * 16, 0)
        query = row % 2048
        relation = (row // 2048) % 4
        shared = (row // 8192) % 8
        packed_owner = (query * 4 + relation) // 128
        expected = 8 * (63 - packed_owner) + shared
        assert evaluate(owner_x, coordinates) == expected


def test_physical_region_consumes_declared_fold_capability_without_formula_rediscovery() -> None:
    request = _combined_fold_forward_request()
    body = request.region.body()
    fold_index, fold = next(
        (index, operation)
        for index, operation in enumerate(body.operations)
        if operation.name == "shuttle.fold" and len(operation.results) == 2
    )
    reducer = fold.body()
    malformed_operations = tuple(
        replace(operation, name="arith.subf") if operation.name == "arith.addf" else operation
        for operation in reducer.operations
    )
    malformed_fold = replace(fold, regions=((replace(reducer, operations=malformed_operations),),))

    region = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _replace_region_operation(request, fold_index, malformed_fold)
    )

    assert region.maximum is region.normalizer


def test_score_domain_preserves_affine_offset_and_source_relation_program() -> None:
    score_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 8, 4, 128, 128))
    index_type = ShuttleTypeRecord("ranked_tensor", "i32", (128,))
    owner_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (1, 1, 1, 128, 1))
    stream_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (1, 1, 1, 1, 128))
    full_index_type = ShuttleTypeRecord("ranked_tensor", "i32", score_type.shape)
    predicate_type = ShuttleTypeRecord("ranked_tensor", "i1", score_type.shape)
    scalar_index_type = ShuttleTypeRecord("ranked_tensor", "i32", ())
    owner_iota = _relation(30, "iota", (), index_type, parameters=(0,))
    owner_reshape = _relation(31, "reshape", (30,), owner_index_type)
    offset = _relation(32, "constant", (), scalar_index_type, parameters=(4,))
    offset_broadcast = _relation(33, "broadcast_in_dim", (32,), owner_index_type)
    shifted_owner = _relation(34, "add", (31, 33), owner_index_type)
    stream_iota = _relation(35, "iota", (), index_type, parameters=(0,))
    stream_reshape = _relation(36, "reshape", (35,), stream_index_type)
    stream_broadcast = _relation(37, "broadcast_in_dim", (36,), full_index_type, parameters=(0, 1, 2, 3, 4))
    owner_broadcast = _relation(38, "broadcast_in_dim", (34,), full_index_type, parameters=(0, 1, 2, 3, 4))
    compare = _relation(39, "compare", (37, 38), predicate_type, parameters=("SIGNED", "LE"))
    select = _relation(42, "select", (39, 26, 40), score_type)
    producer = _contract(
        26,
        (1, 2),
        score_type,
        (_affine_map(0, 1, 2, 3, 5), _affine_map(0, 1, 2, 4, 5), _affine_map(0, 1, 2, 3, 4)),
    )
    operations = (
        owner_iota,
        owner_reshape,
        offset,
        offset_broadcast,
        shifted_owner,
        stream_iota,
        stream_reshape,
        stream_broadcast,
        owner_broadcast,
        compare,
        select,
    )
    producer_by_value = {result.id: operation for operation in operations for result in operation.results}

    domain = physical_normalized_exp_contract._domain_predicate(operations, producer, producer_by_value)

    assert domain is not None
    assert domain.relation is select
    assert domain.predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((4, 1),), "constant": 0},
        "right": {"coefficients": ((3, 1),), "constant": 4},
    }
    source_semantics = tuple(operation.attributes["semantics"]["value"] for operation in domain.source_operations)
    assert len(source_semantics) == 10
    assert source_semantics.count("#shuttle.relation_semantics<iota>") == 2
    assert "#shuttle.relation_semantics<constant>" in source_semantics
    assert "#shuttle.relation_semantics<add>" in source_semantics
    assert source_semantics[-1] == "#shuttle.relation_semantics<compare>"


def test_score_domain_composes_structural_map_lineage_before_interpreting_iotas() -> None:
    raw_score_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 3, 23, 19, 2))
    score_type = ShuttleTypeRecord("ranked_tensor", "f32", (19, 23, 1, 3, 2))
    owner_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (19,))
    owner_reshape_type = ShuttleTypeRecord("ranked_tensor", "i32", (19, 1, 1, 1, 1))
    stream_index_type = ShuttleTypeRecord("ranked_tensor", "i32", (23,))
    stream_reshape_type = ShuttleTypeRecord("ranked_tensor", "i32", (1, 23, 1, 1, 1))
    predicate_type = ShuttleTypeRecord("ranked_tensor", "i1", score_type.shape)
    scalar_type = ShuttleTypeRecord("ranked_tensor", "f32", ())
    producer = _contract(
        26,
        (1, 2),
        raw_score_type,
        (_affine_map(0, 1, 3, 4, 5), _affine_map(0, 1, 2, 5), _affine_map(0, 1, 2, 3, 4)),
    )
    transpose = _transpose_map(
        27,
        producer.results[0].id,
        score_type,
        input_dimensions=(2, 3, 1, 0, 4),
    )
    owner_iota = _relation(30, "iota", (), owner_index_type, parameters=(0,))
    owner_reshape = _relation(31, "reshape", (30,), owner_reshape_type)
    stream_iota = _relation(32, "iota", (), stream_index_type, parameters=(0,))
    stream_reshape = _relation(33, "reshape", (32,), stream_reshape_type)
    owner_broadcast = _relation(34, "broadcast_in_dim", (31,), score_type, parameters=(0, 1, 2, 3, 4))
    stream_broadcast = _relation(35, "broadcast_in_dim", (33,), score_type, parameters=(0, 1, 2, 3, 4))
    compare = _relation(36, "compare", (35, 34), predicate_type, parameters=("SIGNED", "LE"))
    rejected = _relation(37, "constant", (), scalar_type, parameters=("FF800000",))
    rejected_broadcast = _relation(38, "broadcast_in_dim", (37,), score_type)
    select = _relation(39, "select", (36, transpose.results[0].id, 38), score_type)
    operations = (
        producer,
        transpose,
        owner_iota,
        owner_reshape,
        stream_iota,
        stream_reshape,
        owner_broadcast,
        stream_broadcast,
        compare,
        rejected,
        rejected_broadcast,
        select,
    )
    producer_by_value = {result.id: operation for operation in operations for result in operation.results}

    domain = physical_normalized_exp_contract._domain_predicate(operations, producer, producer_by_value)

    assert domain is not None
    assert domain.predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((2, 1),), "constant": 0},
        "right": {"coefficients": ((3, 1),), "constant": 0},
    }
    physical = physical_normalized_exp_contract.physical_normalized_exp_contract_region(
        _ordinary_jax_relation_chained_forward_request()
    )
    physical = replace(
        physical,
        domain_predicate=domain,
        geometry=replace(
            physical.geometry,
            batch_dimensions=(0,),
            shared_relation_dimension=1,
            relation_group_dimensions=(4,),
            owner_dimension=3,
            streamed_dimension=2,
        ),
    )
    forward = normalized_exp_contract_program(
        physical,
        schedule=StreamingTileSchedule(query_tile_size=16, key_value_tile_size=16, pipeline_depth=2),
    )

    assert forward.region.dimensions.normalized_axes[0].id == 4
    assert forward.region.domain_predicate is not None
    assert forward.region.domain_predicate.canonical_record == {
        "comparison": "signed_less_equal",
        "left": {"coefficients": ((4, 1),), "constant": 0},
        "right": {"coefficients": ((3, 1),), "constant": 0},
    }
    assert derive_normalized_exp_contract_pullback(forward.region).region.domain_predicate == (
        forward.region.domain_predicate
    )


def test_scalar_map_extraction_preserves_typed_rounding_conversion() -> None:
    input_type = ShuttleTypeRecord("ranked_tensor", "f32", (1, 8, 4, 128, 128))
    output_type = ShuttleTypeRecord("ranked_tensor", "bf16", input_type.shape)
    finalizer = _convert_scalar_map(102, 101, input_type, output_type)

    expression = physical_normalized_exp_contract._map_scalar_expression(
        finalizer,
        (scalar_input("normalized_output"),),
    )

    assert expression.kind is ScalarExpressionKind.CONVERT
    assert expression.result_dtype is DType.BF16
    assert expression.conversion_semantics is ScalarConversionSemantics.ROUND_NEAREST_EVEN
    assert expression.operands == (scalar_input("normalized_output"),)


def test_saved_state_reverse_boundary_is_admitted_only_from_structural_state_flow() -> None:
    request = _saved_state_request()

    boundary = physical_saved_state_reverse_boundary(request)

    assert tuple(buffer.ordinal for buffer in boundary.inputs) == (0, 1, 2, 3, 4, 5)
    assert tuple(buffer.ordinal for buffer in boundary.outputs) == (6, 7, 8)
    assert boundary.owner_task.state_outputs == (9,)
    assert boundary.owner_task.vector_dimensions == (4,)
    assert boundary.streamed_task.state_inputs == (5, 9)
    assert boundary.streamed_task.vector_dimensions == (3,)
    assert boundary.producer_owner_input.ordinal == 0
    assert boundary.producer_stream_input.ordinal == 1
    assert boundary.consumer_input.ordinal == 2
    assert boundary.finalizer_output_input.ordinal == 3
    assert boundary.finalizer_cotangent_input.ordinal == 4
    assert boundary.saved_state_input.ordinal == 5
    assert boundary.producer_owner_cotangent_output.ordinal == 6
    assert boundary.producer_stream_cotangent_output.ordinal == 7
    assert boundary.consumer_cotangent_output.ordinal == 8
    assert boundary.output_dependent_state_output.ordinal == 9
    assert boundary.producer.results[0].id == 200
    assert boundary.consumer_weight_cotangent.results[0].id == 202
    assert boundary.output_dependent_fold.results[0].id == 205
    assert boundary.score_expression == scalar_input("contract.producer")
    assert boundary.output_product_expression.kind is ScalarExpressionKind.MULTIPLY
    assert boundary.score_map_vjp_expression == scalar_input("cotangent.normalized_exp_contract.mapped_score")

    malformed_plan = replace(
        request.materialization,
        attributes={"stateful_contract_fold_pullback": False},
    )
    with pytest.raises(ValueError, match="not a certified stateful Contract/Fold pullback"):
        physical_saved_state_reverse_boundary(replace(request, materialization=malformed_plan))


def test_physical_region_translates_to_generic_algebra_without_conflating_contract_features() -> None:
    raw_score = scalar_input("contract.producer")
    score_expression = scalar_binary(
        ScalarExpressionKind.ADD,
        scalar_binary(ScalarExpressionKind.MULTIPLY, raw_score, scalar_constant(0.5)),
        scalar_constant(1.0),
    )
    finalizer_expression = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input("contract.consumer_accumulator"),
        scalar_input("contract.normalizer"),
    )
    operation = _operation("shuttle.placeholder")
    physical = PhysicalNormalizedExpContractRegion(
        producer=operation,
        score_program=(operation,),
        maximum=operation,
        exponential=operation,
        normalizer=operation,
        consumer=operation,
        finalizer=operation,
        producer_left_argument=0,
        producer_right_argument=1,
        consumer_argument=2,
        output_result=0,
        public_output_dimensions=(0, 1, 2, 3, 4),
        producer_to_consumer_dimensions=((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)),
        geometry=NormalizedExpContractGeometry(
            batch_shape=(2,),
            shared_relation_extent=3,
            relation_group_shape=(4,),
            owner_extent=64,
            streamed_extent=128,
            producer_feature_extent=128,
            consumer_feature_extent=80,
            batch_dimensions=(0,),
            shared_relation_dimension=1,
            relation_group_dimensions=(2,),
            owner_dimension=3,
            streamed_dimension=4,
            producer_feature_dimension=5,
            normalized_dimension=5,
            consumer_feature_dimension=4,
        ),
        score_expression=score_expression,
        finalizer_expression=finalizer_expression,
        domain_predicate=None,
        buffers=(),
    )
    schedule = StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=3)

    program = normalized_exp_contract_program(physical, schedule=schedule)

    assert program.schedule == schedule
    assert program.source.numerical_policy is NumericalPolicy.FAST
    assert program.region.producer.order is EvaluationOrder.ORDER_FREE
    assert program.region.fold_state.maximum.order is EvaluationOrder.ORDER_FREE
    assert program.region.fold_state.normalizer.order is EvaluationOrder.ORDER_FREE
    assert program.region.consumer.order is EvaluationOrder.ORDER_FREE
    assert program.region.producer.inputs[0].shape == (2, 3, 4, 64, 128)
    assert program.region.producer.inputs[1].shape == (2, 3, 128, 128)
    assert program.region.consumer.inputs[1].shape == (2, 3, 128, 80)
    assert program.region.finalizer.output.shape == (2, 3, 4, 64, 80)
    assert tuple(axis.extent for axis in program.region.dimensions.shared_relation_axes) == (3,)
    assert tuple(axis.extent for axis in program.region.dimensions.relation_group_axes) == (4,)
    assert program.region.producer_index_relations == ((), ())
    assert program.region.consumer_index_relations == ((), ())
    assert program.region.score_map.expression.kind is ScalarExpressionKind.ADD
    assert program.region.finalizer.expression.kind is ScalarExpressionKind.DIVIDE

    direct_finalizer = scalar_convert(
        scalar_input("contract.consumer_accumulator"),
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )
    direct_program = normalized_exp_contract_program(
        replace(physical, finalizer_expression=direct_finalizer),
        schedule=schedule,
    )
    assert direct_program.region.finalizer.expression.kind is ScalarExpressionKind.CONVERT
    normalized_consumer = direct_program.region.finalizer.expression.operands[0]
    assert normalized_consumer.kind is ScalarExpressionKind.DIVIDE
    assert scalar_expression_inputs(normalized_consumer) == {
        "contract.consumer_accumulator",
        "contract.normalizer",
    }


def test_explicit_relation_group_axes_do_not_become_streaming_schedule_axes() -> None:
    operation = _operation("shuttle.placeholder")
    physical = PhysicalNormalizedExpContractRegion(
        producer=operation,
        score_program=(operation,),
        maximum=operation,
        exponential=operation,
        normalizer=operation,
        consumer=operation,
        finalizer=operation,
        producer_left_argument=0,
        producer_right_argument=1,
        consumer_argument=2,
        output_result=0,
        public_output_dimensions=(0, 1, 2, 3, 4),
        producer_to_consumer_dimensions=((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)),
        geometry=NormalizedExpContractGeometry(
            batch_shape=(2,),
            shared_relation_extent=3,
            relation_group_shape=(4,),
            owner_extent=64,
            streamed_extent=128,
            producer_feature_extent=128,
            consumer_feature_extent=80,
            batch_dimensions=(0,),
            shared_relation_dimension=1,
            relation_group_dimensions=(2,),
            owner_dimension=3,
            streamed_dimension=4,
            producer_feature_dimension=5,
            normalized_dimension=5,
            consumer_feature_dimension=4,
        ),
        score_expression=scalar_input("contract.producer"),
        finalizer_expression=scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input("contract.consumer_accumulator"),
            scalar_input("contract.normalizer"),
        ),
        domain_predicate=None,
        buffers=(),
    )
    program = normalized_exp_contract_program(
        physical,
        schedule=Sm90ForwardPipelineSchedule(
            contract=Sm90ContractSchedule(tile_m=64, tile_n=128, threads=256),
            stages=2,
        ),
    )

    descriptor = streaming_contract_fold_event_descriptor(program)

    assert descriptor.fold_extent == 128
    assert descriptor.resident_tile_size == 64
    assert descriptor.streamed_tile_size == 128
    assert descriptor.resident_reduction_dimension == 128
    assert descriptor.output_dimension == 80


def test_source_bound_tasks_reject_overlapping_relation_partitions() -> None:
    request = _saved_state_request()
    body = request.materialization.body()
    owner_task = body.operations[10]
    malformed_task = replace(
        owner_task,
        attributes=dict(owner_task.attributes, online_vector_dimensions=[3, 4]),
    )
    materialization = replace(
        request.materialization,
        regions=((replace(body, operations=(*body.operations[:10], malformed_task, *body.operations[11:])),),),
    )

    with pytest.raises(ValueError, match="disjoint exhaustive partition"):
        source_bound_tasks(replace(request, materialization=materialization))


def test_source_bound_schedule_tasks_retain_exact_task_provenance() -> None:
    schedules = source_bound_schedule_tasks(_saved_state_request())

    assert tuple(schedule.source_task for schedule in schedules) == (0, 1)
    assert schedules[0].owner_dimensions == (0, 1, 2, 3)
    assert schedules[0].vector_dimensions == (4,)
    assert schedules[0].stream_dimensions == (5,)
    assert schedules[0].state_inputs == (5,)
    assert schedules[0].state_outputs == (9,)
    assert schedules[1].tile_shape == (1, 1, 2, 2, 128, 64)
    assert schedules[1].serial_tiles == 2


def test_stateless_generic_tasks_need_no_online_ownership_metadata() -> None:
    request = _saved_state_request()
    materialization_body = request.materialization.body()
    materialization_operations = list(materialization_body.operations)
    for index in (10, 11):
        attributes = dict(materialization_operations[index].attributes)
        for name in (
            "composition_owner_dimensions",
            "online_vector_dimensions",
            "composition_stream_dimensions",
            "composition_producer_contraction_dimensions",
            "composition_consumer_contraction_dimensions",
            "composition_state_inputs",
            "composition_state_outputs",
        ):
            attributes.pop(name, None)
        materialization_operations[index] = replace(materialization_operations[index], attributes=attributes)
    schedule_body = request.schedule.body()
    schedule_operations = []
    for index, operation in enumerate(schedule_body.operations):
        attributes = dict(operation.attributes)
        for name in (
            "composition_owner_dimensions",
            "online_vector_dimensions",
            "composition_stream_dimensions",
            "composition_producer_contraction_dimensions",
            "composition_consumer_contraction_dimensions",
            "composition_state_inputs",
            "composition_state_outputs",
        ):
            attributes.pop(name, None)
        attributes["kind"] = {
            "kind": "mlir_attribute",
            "value": "#shuttle.schedule_task_kind<row_fold>" if index == 0 else "#shuttle.schedule_task_kind<contract>",
        }
        if index == 1:
            attributes["tile_shape"] = [8]
        schedule_operations.append(replace(operation, attributes=attributes))
    request = replace(
        request,
        materialization=replace(
            request.materialization,
            regions=((replace(materialization_body, operations=tuple(materialization_operations)),),),
        ),
        schedule=replace(
            request.schedule,
            regions=((replace(schedule_body, operations=tuple(schedule_operations)),),),
        ),
    )

    schedules = source_bound_schedule_tasks(request)

    assert schedules[0].owner_dimensions == (0, 1, 2, 3, 4)
    assert schedules[0].stream_dimensions == (5,)
    assert schedules[0].vector_dimensions == ()
    assert schedules[1].tile_shape == (8,)


def test_stateful_tasks_require_explicit_online_ownership_metadata() -> None:
    request = _saved_state_request()
    body = request.materialization.body()
    owner = body.operations[10]
    attributes = dict(owner.attributes)
    for name in (
        "composition_owner_dimensions",
        "online_vector_dimensions",
        "composition_stream_dimensions",
        "composition_producer_contraction_dimensions",
        "composition_consumer_contraction_dimensions",
    ):
        attributes.pop(name)
    malformed = replace(owner, attributes=attributes)
    request = replace(
        request,
        materialization=replace(
            request.materialization,
            regions=((replace(body, operations=(*body.operations[:10], malformed, body.operations[11])),),),
        ),
    )

    with pytest.raises(ValueError, match="stateful materialization tasks require an explicit composition partition"):
        source_bound_tasks(request)


def test_source_bound_schedule_tasks_reject_materialization_drift() -> None:
    request = _saved_state_request()
    body = request.schedule.body()
    schedule = body.operations[1]
    malformed = replace(schedule, attributes=dict(schedule.attributes, source_task=0))
    request = replace(
        request,
        schedule=replace(request.schedule, regions=((replace(body, operations=(body.operations[0], malformed)),),)),
    )

    with pytest.raises(ValueError, match="binds materialization task 0, expected 1"):
        source_bound_schedule_tasks(request)


def test_source_argument_bindings_do_not_depend_on_materialization_buffer_order() -> None:
    request = _saved_state_request()
    buffers = source_bound_buffers(request)
    first = dict(request.materialization.body().operations[0].attributes, source_argument=0)
    second = dict(request.materialization.body().operations[3].attributes, source_argument=3)
    request = _replace_buffer_attributes(request, 0, first)
    request = _replace_buffer_attributes(request, 3, second)
    request = replace(request, binding_indices=(0, *request.binding_indices[1:3], 3, *request.binding_indices[4:]))

    by_argument = source_argument_buffers(request, source_bound_buffers(request))

    assert by_argument[0].ordinal == 0
    assert by_argument[3].ordinal == 3
    assert tuple(buffer.source_argument for buffer in buffers[:4]) == (3, 1, 4, 0)


def test_source_result_bindings_do_not_depend_on_materialization_buffer_order() -> None:
    request = _saved_state_request()
    result_indices = list(request.region_result_indices)
    result_indices[6:9] = [2, 0, 1]
    request = replace(request, region_result_indices=tuple(result_indices))
    body = request.region.body()
    yielded = body.operations[-1]
    request = replace(
        request,
        region=replace(
            request.region,
            regions=(
                (
                    replace(
                        body,
                        operations=(
                            *body.operations[:-1],
                            replace(yielded, operands=(yielded.operands[1], yielded.operands[2], yielded.operands[0])),
                        ),
                    ),
                ),
            ),
        ),
    )

    by_result = source_result_buffers(request, source_bound_buffers(request))

    assert {result: buffer.ordinal for result, buffer in by_result.items()} == {0: 7, 1: 8, 2: 6}


def test_saved_state_reverse_operand_roles_follow_ssa_sources_not_buffer_ordinals() -> None:
    request = _saved_state_request()
    producer_owner = dict(request.materialization.body().operations[0].attributes, source_argument=4)
    consumer = dict(request.materialization.body().operations[2].attributes, source_argument=3)
    request = _replace_buffer_attributes(request, 0, producer_owner)
    request = _replace_buffer_attributes(request, 2, consumer)
    request = replace(request, binding_indices=(4, request.binding_indices[1], 3, *request.binding_indices[3:]))

    boundary = physical_saved_state_reverse_boundary(request)

    assert boundary.producer_owner_input.ordinal == 2
    assert boundary.consumer_input.ordinal == 0
    assert boundary.producer_stream_input.ordinal == 1
    assert boundary.finalizer_output_input.ordinal == 3
    assert boundary.finalizer_cotangent_input.ordinal == 4


def test_saved_state_reverse_scalar_vjp_is_derived_from_recomputed_score_map() -> None:
    request = _saved_state_request()
    body = request.region.body()
    score = body.operations[0]
    normalized = body.operations[1]
    affine_score = _affine_scalar_map(211, score.results[0].id, score.results[0].type)
    normalized = replace(normalized, operands=(affine_score.results[0].id, normalized.operands[1]))
    operations = [score, affine_score, normalized, *body.operations[2:-1]]
    adjusted_score_pullback = next(operation for operation in operations if operation.results[0].id == 206)
    extracted_score_vjp = _scale_scalar_map(
        212, adjusted_score_pullback.results[0].id, adjusted_score_pullback.results[0].type
    )
    operations.append(extracted_score_vjp)
    operations = [
        (
            replace(
                operation,
                operands=tuple(
                    extracted_score_vjp.results[0].id if operand == adjusted_score_pullback.results[0].id else operand
                    for operand in operation.operands
                ),
            )
            if operation.name == "shuttle.contract" and operation.results[0].id in (207, 208)
            else operation
        )
        for operation in operations
    ]
    operations.append(body.operations[-1])
    request = replace(request, region=replace(request.region, regions=((replace(body, operations=tuple(operations)),),)))

    boundary = physical_saved_state_reverse_boundary(request)

    assert boundary.score_expression.kind is ScalarExpressionKind.ADD
    assert serialize_scalar_expression(boundary.score_map_vjp_expression) == serialize_scalar_expression(
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("cotangent.normalized_exp_contract.mapped_score"),
            scalar_constant(0.5),
        )
    )


def test_saved_state_reverse_rejects_score_recompute_without_matching_source_vjp() -> None:
    request = _saved_state_request()
    body = request.region.body()
    score = body.operations[0]
    normalized = body.operations[1]
    affine_score = _affine_scalar_map(211, score.results[0].id, score.results[0].type)
    normalized = replace(normalized, operands=(affine_score.results[0].id, normalized.operands[1]))
    operations = (score, affine_score, normalized, *body.operations[2:])
    request = replace(request, region=replace(request.region, regions=((replace(body, operations=operations),),)))

    with pytest.raises(ValueError, match="scalar subgraphs matching the generic score-Map VJP"):
        physical_saved_state_reverse_boundary(request)


def test_saved_state_reverse_region_reconstructs_generic_contract_dimensions() -> None:
    boundary = physical_saved_state_reverse_boundary(_saved_state_request())
    shapes = {
        0: (1, 1, 2, 128, 128),
        1: (1, 1, 128, 128),
        2: (1, 1, 128, 128),
        3: (1, 1, 2, 128, 128),
        4: (1, 1, 2, 128, 128),
        5: (1, 1, 2, 128),
        6: (1, 1, 2, 128, 128),
        7: (1, 1, 128, 128),
        8: (1, 1, 128, 128),
        9: (1, 1, 2, 128),
    }

    def physical_shape(buffer):
        return replace(buffer, type=replace(buffer.type, shape=shapes[buffer.ordinal]))

    by_ordinal = {buffer.ordinal: physical_shape(buffer) for buffer in (*boundary.inputs, *boundary.outputs)}
    by_ordinal[9] = physical_shape(boundary.output_dependent_state_output)
    producer = replace(
        boundary.producer,
        attributes=dict(
            boundary.producer.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"],
        ),
    )
    consumer_weight = replace(
        boundary.consumer_weight_cotangent,
        attributes=dict(
            boundary.consumer_weight_cotangent.attributes,
            iterator_kinds=["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"],
        ),
    )
    boundary = replace(
        boundary,
        inputs=tuple(by_ordinal[buffer.ordinal] for buffer in boundary.inputs),
        outputs=tuple(by_ordinal[buffer.ordinal] for buffer in boundary.outputs),
        producer_owner_input=by_ordinal[0],
        producer_stream_input=by_ordinal[1],
        consumer_input=by_ordinal[2],
        finalizer_output_input=by_ordinal[3],
        finalizer_cotangent_input=by_ordinal[4],
        saved_state_input=by_ordinal[5],
        producer_owner_cotangent_output=by_ordinal[6],
        producer_stream_cotangent_output=by_ordinal[7],
        consumer_cotangent_output=by_ordinal[8],
        output_dependent_state_output=by_ordinal[9],
        producer=producer,
        consumer_weight_cotangent=consumer_weight,
    )

    region = physical_saved_state_reverse_region(boundary)
    program = normalized_exp_contract_program(
        region,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )

    assert region.geometry.owner_extent == 128
    assert region.geometry.streamed_extent == 128
    assert region.geometry.producer_feature_extent == 128
    assert region.geometry.consumer_feature_extent == 128
    assert program.region.producer.inputs[0].shape == (1, 1, 2, 128, 128)
    assert program.region.producer.inputs[1].shape == (1, 1, 128, 128)
    assert program.region.consumer.inputs[1].shape == (1, 1, 128, 128)
    assert program.region.score_map.expression == boundary.score_expression


def test_saved_state_reverse_operand_roles_reject_incompatible_contract_relation() -> None:
    request = _saved_state_request()
    owner_contract = request.region.body().operations[7]
    maps = list(owner_contract.attributes["indexing_maps"])
    maps[1] = _affine_map(0, 1, 4)
    request = _replace_region_operation(
        request,
        7,
        replace(owner_contract, attributes=dict(owner_contract.attributes, indexing_maps=maps)),
    )

    with pytest.raises(ValueError, match="unambiguous direct operand relations"):
        physical_saved_state_reverse_boundary(request)


def test_saved_state_reverse_operand_roles_accept_composed_affine_head_relation() -> None:
    request = _saved_state_request()
    owner_contract = request.region.body().operations[7]
    maps = list(owner_contract.attributes["indexing_maps"])
    direct_map = dict(maps[1])
    results = list(direct_map["results"])
    results[1] = {
        "kind": "floor_divide",
        "left": {"kind": "dimension", "position": 1},
        "right": {"kind": "constant", "value": 4},
    }
    direct_map["results"] = results
    maps[1] = direct_map
    request = _replace_region_operation(
        request,
        7,
        replace(owner_contract, attributes=dict(owner_contract.attributes, indexing_maps=maps)),
    )

    boundary = physical_saved_state_reverse_boundary(request)

    assert boundary.producer_stream_input.ordinal == 1


def test_saved_state_reverse_operand_roles_reject_ambiguous_output_dependent_fold() -> None:
    request = _saved_state_request()
    body = request.region.body()
    output_product = body.operations[4]
    duplicate = _sum_fold(210, output_product.results[0].id, body.operations[5].results[0].type)
    operations = (*body.operations[:-1], duplicate, body.operations[-1])
    request = replace(request, region=replace(request.region, regions=((replace(body, operations=operations),),)))

    with pytest.raises(ValueError, match=r"expected one output-dependent Fold.*found 2"):
        physical_saved_state_reverse_boundary(request)


def test_real_native_scalar_value_and_grad_consumes_typed_pullback_capability() -> None:
    request = decode_gpu_physical_artifact_request(_NATIVE_SCALAR_VALUE_AND_GRAD_FIXTURE.read_text())
    capability = request.normalized_exp_contract_pullbacks[0]
    phase_event = request.event_relations[-1]

    match = physical_normalized_exp_contract_match(request)

    assert isinstance(match, PhysicalNormalizedExpPullbackMatch)
    assert len(request.event_relations) == 7
    assert (phase_event.producer_task, phase_event.consumer_task) == (capability.objective_task,) * 2
    assert (phase_event.producer_phase, phase_event.consumer_phase) == ("partial", "merge")
    assert match.pullback.forward_output_buffer == capability.forward_result_buffer
    assert (
        match.pullback.output_cotangent_argument
        == source_bound_buffers(request)[capability.output_cotangent_buffer].source_argument
    )
    assert (
        match.pullback.query_cotangent_buffer,
        match.pullback.key_cotangent_buffer,
        match.pullback.value_cotangent_buffer,
    ) == capability.terminal_cotangent_buffers
    assert match.pullback.score_map_vjp_expression == scalar_expression_vjp(
        match.pullback.forward.score_expression,
        input_name="contract.producer",
        cotangent_name="cotangent.normalized_exp_contract.mapped_score",
    )


def test_real_native_pullback_rejects_stale_typed_source_bindings() -> None:
    request = decode_gpu_physical_artifact_request(_NATIVE_SCALAR_VALUE_AND_GRAD_FIXTURE.read_text())
    capability = request.normalized_exp_contract_pullbacks[0]

    with pytest.raises(ValueError, match="terminal cotangent buffers disagree"):
        physical_normalized_exp_contract_match(
            replace(
                request,
                normalized_exp_contract_pullbacks=(
                    replace(
                        capability,
                        terminal_cotangent_buffers=(
                            capability.maximum_state_buffer,
                            *capability.terminal_cotangent_buffers[1:],
                        ),
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda request: _replace_buffer_attributes(
                request,
                0,
                {
                    key: value
                    for key, value in request.materialization.body().operations[0].attributes.items()
                    if key != "source_argument"
                },
            ),
            "has no source argument",
        ),
        (
            lambda request: _replace_buffer_attributes(
                request,
                1,
                dict(request.materialization.body().operations[1].attributes, source_argument=3),
            ),
            "bound by multiple source buffers",
        ),
        (
            lambda request: _replace_buffer_attributes(
                request,
                0,
                dict(request.materialization.body().operations[0].attributes, source_argument=99),
            ),
            "outside the Region boundary",
        ),
        (
            lambda request: replace(request, binding_indices=(1, *request.binding_indices[1:])),
            "physical binding disagrees",
        ),
    ),
)
def test_saved_state_reverse_boundary_rejects_corrupt_source_argument_bindings(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        physical_saved_state_reverse_boundary(mutate(_saved_state_request()))
