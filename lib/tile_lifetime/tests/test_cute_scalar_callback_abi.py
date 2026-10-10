# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import struct
from dataclasses import replace

import pytest

pytest.importorskip("cutlass.cute")

from tile_lifetime import (
    StreamingTileSchedule,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.backends.h100.cute_scalar_ast import (
    cute_keyword_score_map_adapter,
    cute_keyword_score_map_vjp_adapter,
    cute_post_normalization_vjp,
    cute_score_map,
    cute_score_map_vjp,
)
from tile_lifetime.backends.h100.cute_streaming_backward_emitter import (
    _forward_launch,
    _order_source_slots,
    _packed_relation_tensor_argument,
    _value_and_grad_kernel_partition,
)
from tile_lifetime.backends.h100.cute_streaming_emitter import build_h100_streaming_template
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotKind,
)
from tile_lifetime.h100_streaming_lowering import Sm90ContractSchedule, Sm90ForwardPipelineSchedule
from tile_lifetime.tensor_program import DType, ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input


def test_score_map_keeps_generic_roles_behind_keyword_callback_adapter() -> None:
    score_map = cute_score_map(scalar_input("raw_score"), {"raw_score": "score"})
    adapter = cute_keyword_score_map_adapter(score_map)

    assert tuple(inspect.signature(score_map).parameters) == (
        "scores",
        "batch_idx",
        "relation_idx",
        "owner_idx",
        "streamed_idx",
        "seqlen_info",
        "aux_tensors",
    )
    assert tuple(inspect.signature(adapter).parameters) == (
        "scores",
        "batch_idx",
        "relation_idx",
        "q_idx",
        "kv_idx",
        "seqlen_info",
        "aux_tensors",
    )


def test_score_map_vjp_keeps_generic_roles_behind_keyword_callback_adapter() -> None:
    score_map_vjp = cute_score_map_vjp(
        scalar_input("mapped_score_cotangent"),
        {"mapped_score_cotangent": "cotangent"},
    )
    adapter = cute_keyword_score_map_vjp_adapter(score_map_vjp)

    assert tuple(inspect.signature(score_map_vjp).parameters) == (
        "gradient",
        "scores",
        "batch_idx",
        "relation_idx",
        "owner_idx",
        "streamed_idx",
        "seqlen_info",
        "aux_tensors",
    )
    assert tuple(inspect.signature(adapter).parameters) == (
        "gradient",
        "scores",
        "batch_idx",
        "relation_idx",
        "q_idx",
        "kv_idx",
        "seqlen_info",
        "aux_tensors",
    )


def test_post_normalization_vjp_callback_uses_primal_and_cotangent_ast_inputs() -> None:
    expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input("output_cotangent"),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(0.25),
            scalar_input("normalized_output"),
        ),
    )

    callback = cute_post_normalization_vjp(
        expression,
        normalized_output_input="normalized_output",
        cotangent_input="output_cotangent",
    )

    assert tuple(inspect.signature(callback).parameters) == ("cotangent", "normalized_output")


def test_linear_post_normalization_vjp_callback_does_not_require_primal_input() -> None:
    expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_constant(0.75),
        scalar_input("output_cotangent"),
    )

    callback = cute_post_normalization_vjp(
        expression,
        normalized_output_input="normalized_output",
        cotangent_input="output_cotangent",
    )

    assert tuple(inspect.signature(callback).parameters) == ("cotangent", "normalized_output")


def test_packed_relation_tensor_argument_preserves_group_and_sequence_strides() -> None:
    argument = _packed_relation_tensor_argument(
        "normalized_output",
        "normalized_output",
        CutePipelineSlotAccess.WRITE,
        (1, 128, 32, 128),
        4,
        logical_strides=(524_288, 128, 16_384, 1),
    )

    assert argument.size == 56
    assert argument.fields[1].data == struct.pack("<4i", 128, 128, 8, 1)
    assert argument.fields[2].data == struct.pack("<4q", 16_384, 128, 65_536, 524_288)


def test_source_slot_ordering_preserves_sparse_materialization_ordinals() -> None:
    slots = (
        CutePipelineSlot("query", CutePipelineElementType.BF16, (8,), CutePipelineSlotKind.PUBLIC_INPUT, 16, 0),
        CutePipelineSlot("output", CutePipelineElementType.BF16, (8,), CutePipelineSlotKind.PUBLIC_OUTPUT, 16, None, 0),
        CutePipelineSlot("scratch", CutePipelineElementType.F32, (8,), CutePipelineSlotKind.SCRATCH, 16),
    )

    ordered = _order_source_slots(slots, 2, {"query": 6, "output": 0})

    assert tuple(slot.name for slot in ordered) == ("output", "query", "scratch")
    with pytest.raises(ValueError, match="uniquely identify nonnegative source buffers"):
        _order_source_slots(slots, 2, {"query": 6, "output": 6})


def test_value_and_grad_kernel_partition_accounts_for_queue_initialization_order() -> None:
    names = ("semaphore", "forward", "preprocess", "map", "zero_k", "zero_v", "backward", "dq", "dk", "dv")

    forward, scalar_map, reverse = _value_and_grad_kernel_partition(
        names,
        has_device_queues=True,
        generic_finalizer=True,
    )

    assert forward == "forward"
    assert scalar_map == "map"
    assert reverse == ("semaphore", "preprocess", "zero_k", "zero_v", "backward", "dq", "dk", "dv")


def test_value_and_grad_kernel_partition_without_optional_kernels_starts_at_forward() -> None:
    names = ("forward", "preprocess", "zero_k", "zero_v", "backward", "dq", "dk", "dv")

    forward, scalar_map, reverse = _value_and_grad_kernel_partition(
        names,
        has_device_queues=False,
        generic_finalizer=False,
    )

    assert forward == "forward"
    assert scalar_map is None
    assert reverse == names[1:]


@pytest.mark.parametrize("access", (CutePipelineSlotAccess.READ, CutePipelineSlotAccess.WRITE))
def test_tiled_forward_launch_preserves_row_state_abi_and_realized_shared_memory(access) -> None:
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=32,
        key_length=32,
        query_heads=4,
        key_value_heads=2,
        key_dimension=128,
        value_dimension=128,
        score_map=scaled_score_map(0.5),
        input_dtype=DType.BF16,
    )
    logical = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )
    template = build_h100_streaming_template(
        replace(
            logical,
            schedule=Sm90ForwardPipelineSchedule(
                contract=Sm90ContractSchedule(tile_m=128, tile_n=128),
                stages=2,
            ),
        )
    )
    launch = _forward_launch(
        template,
        "tiled_kernel",
        writes_log_sum_exp=access is CutePipelineSlotAccess.WRITE,
        dynamic_shared_memory_bytes=197632,
        row_state_access=access,
        row_max_stride=(256, 64, 1),
        row_sum_stride=(128, 32, 1),
    )
    names = tuple(argument.name for argument in launch.arguments)
    row_index = 5 if access is CutePipelineSlotAccess.WRITE else 4
    assert names[row_index : row_index + 3] == ("row_max", "row_sum", "query_tma")
    row_max, row_sum = launch.arguments[row_index : row_index + 2]
    assert row_max.fields[0].access is access
    assert row_sum.fields[0].access is access
    assert row_max.fields[1].data == struct.pack("<3i", 32, 4, 1)
    assert row_sum.fields[1].data == struct.pack("<3i", 32, 4, 1)
    assert row_max.fields[2].data == struct.pack("<2q", 64, 256)
    assert row_sum.fields[2].data == struct.pack("<2q", 32, 128)
    assert launch.grid == (1, 2, 1)
    assert launch.block == (384, 1, 1)
    assert launch.dynamic_shared_memory_bytes == 197632
    assert launch.function_attributes[0].value == 197632
