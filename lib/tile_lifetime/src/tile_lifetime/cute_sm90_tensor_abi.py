# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host parameter encodings for the retained CuTe SM90 tensor layouts."""

import struct

from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineSlotAccess,
    CutePipelineStructuredArgument,
    CutePipelineStructuredImmediateField,
    CutePipelineStructuredSlotAddressField,
)


def sm90_tensor_map_argument(
    name: str,
    slot: str,
    original_shape: tuple[int, ...],
    box_shape: tuple[int, int, int, int],
    access: CutePipelineSlotAccess,
    *,
    logical_strides: tuple[int, ...] | None = None,
) -> CutePipelineStructuredArgument:
    """Encode the ordinary dimension/sequence/head/batch TMA descriptor."""
    batch, sequence, heads, dimension = original_shape
    element_bytes = 2
    strides = logical_strides or (sequence * heads * dimension, heads * dimension, dimension, 1)
    if len(strides) != 4 or strides[3] != 1:
        raise ValueError(f"tensor-map argument {name!r} requires a rank-four unit-minor logical layout")
    batch_stride, sequence_stride, head_stride, _ = (stride * element_bytes for stride in strides)
    contiguous_box = min(box_shape[0], 64)
    return _tensor_map_argument_from_layout(
        name,
        slot,
        contiguous_extent=dimension,
        noncontiguous_extents=(sequence, heads, batch),
        noncontiguous_byte_strides=(sequence_stride, head_stride, batch_stride),
        box_words=((contiguous_box - 1) << 56, box_shape[1] - 1),
        access=access,
    )


def sm90_packed_head_tensor_map_argument(
    name: str,
    slot: str,
    original_shape: tuple[int, ...],
    owner_tile: int,
    group_size: int,
    access: CutePipelineSlotAccess,
    *,
    logical_strides: tuple[int, ...] | None = None,
) -> CutePipelineStructuredArgument:
    """Encode the CuTe descriptor after packing an owner relation group."""
    batch, sequence, heads, dimension = original_shape
    if heads % group_size:
        raise ValueError("packed tensor-map head extent must be divisible by the operand relation group")
    if owner_tile % group_size:
        raise ValueError("packed tensor-map owner tile must be divisible by the operand relation group")
    # CuTe collapses the static singleton group before forming the TMA layout.
    # Its resulting dimensions remain D,S,H,B rather than the packed D,H,S,B.
    if group_size == 1:
        return sm90_tensor_map_argument(
            name,
            slot,
            original_shape,
            (owner_tile, owner_tile, 1, 1),
            access,
            logical_strides=logical_strides,
        )
    element_bytes = 2
    strides = logical_strides or (sequence * heads * dimension, heads * dimension, dimension, 1)
    if len(strides) != 4 or strides[3] != 1:
        raise ValueError(f"tensor-map argument {name!r} requires a rank-four unit-minor logical layout")
    batch_stride, sequence_stride, head_stride, _ = (stride * element_bytes for stride in strides)
    packed_minor = (group_size - 1) | ((owner_tile // group_size - 1) << 8)
    return _tensor_map_argument_from_layout(
        name,
        slot,
        contiguous_extent=dimension,
        noncontiguous_extents=(heads, sequence, batch),
        noncontiguous_byte_strides=(head_stride, sequence_stride, batch_stride),
        box_words=((min(owner_tile, 64) - 1) << 56, packed_minor),
        access=access,
    )


def _tensor_map_argument_from_layout(
    name: str,
    slot: str,
    *,
    contiguous_extent: int,
    noncontiguous_extents: tuple[int, int, int],
    noncontiguous_byte_strides: tuple[int, int, int],
    box_words: tuple[int, int],
    access: CutePipelineSlotAccess,
) -> CutePipelineStructuredArgument:
    element_bytes = 2
    stride0, stride1, stride2 = noncontiguous_byte_strides
    extent0, extent1, extent2 = noncontiguous_extents
    if any(stride % 16 for stride in noncontiguous_byte_strides):
        raise ValueError(f"tensor-map argument {name!r} requires 16-byte-aligned global strides")
    addressable_bytes = contiguous_extent * element_bytes + sum(
        (extent - 1) * stride for extent, stride in zip(noncontiguous_extents, noncontiguous_byte_strides, strict=True)
    )
    descriptor_words = (
        ((addressable_bytes >= 128 * 1024) << 21) | ((stride0 // 16) << 32) | 0x46532,
        (stride1 // 16) | ((stride2 // 16) << 32),
        ((stride0 >> 36) & 0xF) << 32 | ((stride1 >> 36) & 0xF) << 36 | ((stride2 >> 36) & 0xF) << 40,
        (contiguous_extent - 1) | ((extent0 - 1) << 32),
        (extent1 - 1) | ((extent2 - 1) << 32),
        box_words[0],
        box_words[1],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    return CutePipelineStructuredArgument(
        name=name,
        size=128,
        alignment=64,
        fields=(
            CutePipelineStructuredSlotAddressField("address", slot, access, 0, 0),
            CutePipelineStructuredImmediateField("descriptor", 8, struct.pack("<15Q", *descriptor_words)),
        ),
    )


def packed_log_sum_exp_argument(
    name: str, slot: str, *, batch: int, sequence: int, shared_heads: int, group_size: int
) -> CutePipelineStructuredArgument:
    """Encode packed row state with a static group extent and dynamic strides."""
    # pack_gqa_layout uses shape ((group_size, S), H, B). The group extent is
    # compile-time metadata, while its head stride remains a runtime field.
    return CutePipelineStructuredArgument(
        name=name,
        size=48,
        alignment=8,
        fields=(
            CutePipelineStructuredSlotAddressField("data", slot, CutePipelineSlotAccess.WRITE, 0, 0),
            CutePipelineStructuredImmediateField("shape", 8, struct.pack("<3i", sequence, shared_heads, batch)),
            CutePipelineStructuredImmediateField(
                "strides", 24, struct.pack("<3q", sequence, sequence * group_size, sequence * shared_heads * group_size)
            ),
        ),
    )
