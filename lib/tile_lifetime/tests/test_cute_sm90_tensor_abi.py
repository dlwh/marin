# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

import pytest
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir

from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CutePipelineSlotAccess,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
    cute_kernel_argument_size,
)
from tile_lifetime.cute_sm90_launch_abi import (
    relocated_sm90_arguments,
    sm90_launch_function_attributes,
    sm90_launch_parameter_schema,
)
from tile_lifetime.cute_sm90_tensor_abi import packed_log_sum_exp_argument, sm90_packed_head_tensor_map_argument

_CAPTURE_DIRECTORY = Path(__file__).parent / "fixtures/sm90_contract_abi"


@pytest.mark.parametrize("cluster_dimensions", [(0, 0, 0), (1, 1, 1)])
def test_single_block_contract_export_omits_only_inert_cluster_permission(cluster_dimensions) -> None:
    report = json.loads((_CAPTURE_DIRECTORY / "auxiliary.json").read_text())
    measured = report["baseline"][0]["function_attributes"]
    shared = CuteKernelFunctionAttribute(
        CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
        measured["CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"],
    )
    cluster_permission = CuteKernelFunctionAttribute(
        CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
        measured["CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED"],
    )
    # A nondefault carveout is not inert cluster permission and must survive.
    carveout = CuteKernelFunctionAttribute(CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT, 100)
    assert sm90_launch_function_attributes(cluster_dimensions, (shared, cluster_permission, carveout)) == (
        shared,
        carveout,
    )


@pytest.mark.parametrize("cluster_dimensions", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_contract_export_rejects_nontrivial_cluster_even_with_permission(cluster_dimensions) -> None:
    permission = CuteKernelFunctionAttribute(CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
    with pytest.raises(ValueError, match="nontrivial captured cluster"):
        sm90_launch_function_attributes(cluster_dimensions, (permission,))


@pytest.mark.parametrize(
    "case,descriptor_slots",
    [
        ("identity", ((1, "left"), (3, "weight"), (5, "output"))),
        ("auxiliary", ((1, "left"), (3, "weight"), (5, "output"), (7, "scale"))),
    ],
)
def test_retained_contract_abi_reconstructs_independent_device_captures(case, descriptor_slots) -> None:
    report = json.loads((_CAPTURE_DIRECTORY / f"{case}.json").read_text())
    baseline = tuple(bytes.fromhex(value) for value in report["baseline"][0]["parameter_hex"])
    with make_ir_context():
        module = ir.Module.parse((_CAPTURE_DIRECTORY / f"{case}.mlir").read_text())
        schema = sm90_launch_parameter_schema(module)
    relocations = {
        record["slot"]: (
            record["address"],
            tuple(bytes.fromhex(value) for value in record["captures"][0]["parameter_hex"]),
        )
        for record in report["relocations"]
    }
    accesses = {
        slot: CutePipelineSlotAccess.WRITE if slot == "output" else CutePipelineSlotAccess.READ
        for slot in report["slot_addresses"]
    }
    arguments = relocated_sm90_arguments(
        schema,
        baseline,
        report["slot_addresses"],
        relocations,
        accesses,
    )
    assert tuple(cute_kernel_argument_size(argument) for argument in arguments) == tuple(
        size for _, size in report["baseline"][0]["parameter_layout"]
    )
    assert arguments[0].data == b"\0"
    for index, slot in descriptor_slots:
        argument = arguments[index]
        assert isinstance(argument, CutePipelineStructuredArgument)
        (pointer,) = (field for field in argument.fields if isinstance(field, CutePipelineStructuredSlotAddressField))
        assert pointer.slot == slot
        assert pointer.access == accesses[slot]
        assert pointer.destination_byte_offset == 0
        assert argument.zero_filled_template[8:64] == baseline[index][8:64]
        assert argument.zero_filled_template[64:] == bytes(64)

    # Reconstruct every parameter at each independently changed address, as a
    # launch consumer would. The auxiliary capture includes process-local data
    # in its reserved tails, which must never survive into the artifact.
    for changed_slot, (address, capture) in relocations.items():
        addresses = {**report["slot_addresses"], changed_slot: address}
        for index, argument in enumerate(arguments):
            if isinstance(argument, CutePipelineStructuredArgument):
                encoded = bytearray(argument.zero_filled_template)
                for field in argument.fields:
                    if isinstance(field, CutePipelineStructuredSlotAddressField):
                        struct.pack_into("<Q", encoded, field.destination_byte_offset, addresses[field.slot])
                assert bytes(encoded[:64]) == capture[index][:64]
                assert bytes(encoded[64:]) == bytes(64)
            else:
                assert argument.data == capture[index]

    # A changed descriptor bit is metadata, not a second inferred relocation.
    descriptor_index, moved_slot = descriptor_slots[-1]
    moved_address, moved = relocations[moved_slot]
    changed = bytearray(moved[descriptor_index])
    changed[40] ^= 1
    relocations[moved_slot] = moved_address, (*moved[:descriptor_index], bytes(changed), *moved[descriptor_index + 1 :])
    with pytest.raises(ValueError, match="unexplained changes"):
        relocated_sm90_arguments(
            schema,
            baseline,
            report["slot_addresses"],
            relocations,
            accesses,
        )


@pytest.mark.parametrize("case,allocation", [("identity", "%60"), ("auxiliary", "%184")])
def test_contract_abi_rejects_unproved_descriptor_reserved_boundary(case, allocation) -> None:
    source = (_CAPTURE_DIRECTORY / f"{case}.mlir").read_text()
    # Move a real descriptor store into the reserved tail without changing its
    # allocation type or width. A width-only classifier would accept this.
    source = source.replace(f"{allocation}[0, 7]", f"{allocation}[0, 8]")
    with make_ir_context():
        module = ir.Module.parse(source)
        with pytest.raises(ValueError, match="defined/reserved"):
            sm90_launch_parameter_schema(module)


def test_singleton_group_tma_matches_captured_sm90_descriptor() -> None:
    argument = sm90_packed_head_tensor_map_argument(
        "query_tma", "query", (1, 32, 2, 128), 64, 1, CutePipelineSlotAccess.READ
    )
    # Driver-owned graph node captured from the retained CuTe stats kernel.
    # The base pointer is relocated separately; only defined descriptor bytes
    # 8..64 are compared, excluding the runtime's reserved stack-valued tail.
    captured_words = (288050, 32, 16, 1024, 0, 0, 127, 31, 1, 0, 0, 1056964608, 63, 0)
    assert argument.zero_filled_template[8:64] == struct.pack("<14I", *captured_words)


@pytest.mark.parametrize("group_size", [2, 4])
def test_nonsingleton_group_tma_retains_packed_head_then_sequence(group_size: int) -> None:
    argument = sm90_packed_head_tensor_map_argument(
        "query_tma", "query", (1, 32, 2 * group_size, 128), 64, group_size, CutePipelineSlotAccess.READ
    )
    descriptor = argument.zero_filled_template
    assert struct.unpack_from("<3I", descriptor, 36) == (2 * group_size - 1, 31, 0)
    assert struct.unpack_from("<I", descriptor, 56)[0] == (group_size - 1) | ((64 // group_size - 1) << 8)


@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_packed_row_state_omits_static_group_extent(group_size: int) -> None:
    argument = packed_log_sum_exp_argument("lse", "lse", batch=2, sequence=32, shared_heads=3, group_size=group_size)
    descriptor = argument.zero_filled_template
    assert struct.unpack_from("<3i", descriptor, 8) == (32, 3, 2)
    assert struct.unpack_from("<3q", descriptor, 24) == (32, 32 * group_size, 96 * group_size)
    assert descriptor[20:24] == bytes(4)
