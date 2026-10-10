# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover typed SM90 launch parameters from retained host LLVM and captures."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Any

from tile_lifetime.cute_device_ir import _operations
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CutePipelineKernelArgument,
    CutePipelinePackedBytesArgument,
    CutePipelineSlotAccess,
    CutePipelineStructuredArgument,
    CutePipelineStructuredImmediateField,
    CutePipelineStructuredSlotAddressField,
)

_TMA_TYPE = "!llvm.struct<(struct<(array<16 x i64>)>)>"


def sm90_launch_function_attributes(
    cluster_dimensions: tuple[int, int, int],
    captured_attributes: tuple[CuteKernelFunctionAttribute, ...],
) -> tuple[CuteKernelFunctionAttribute, ...]:
    """Omit inert cluster permission only for a proven single-block cluster."""
    if cluster_dimensions not in {(0, 0, 0), (1, 1, 1)}:
        raise ValueError("SM90 export cannot discard nontrivial captured cluster dimensions")
    return tuple(
        attribute
        for attribute in captured_attributes
        if attribute.kind is not CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED
    )


@dataclass(frozen=True)
class Sm90LaunchParameter:
    """A compiler-proven by-value layout; reserved bytes are never captured."""

    size: int
    alignment: int
    defined_bytes: int
    address_offsets: tuple[int, ...] = ()


def sm90_launch_parameter_schema(module: Any) -> tuple[Sm90LaunchParameter, ...]:
    """Read the launch's ordered argument allocas and prove supported layouts."""
    with module.context:
        launches = [
            operation
            for operation in _operations(module.operation)
            if operation.name == "llvm.call" and str(operation.attributes["callee"]) == "@_cudaLaunchKernelEx"
        ]
        if len(launches) != 1:
            raise ValueError("SM90 export requires exactly one retained CUDA launch")
        table = launches[0].operands[2]
        allocation = table.owner
        if allocation.name != "llvm.alloca" or str(allocation.attributes["elem_type"]) != "!llvm.ptr":
            raise ValueError("SM90 launch argument table is not a pointer array")
        count = _constant(allocation.operands[0])
        arguments = {}
        for use in table.uses:
            gep = use.owner
            if gep.name != "llvm.getelementptr":
                continue
            indices = _indices(gep)
            if len(indices) != 1:
                raise ValueError("SM90 launch argument table has an unsupported index")
            pointer = _stored_value(gep.results[0])
            argument_allocation = pointer.owner
            if argument_allocation.name != "llvm.alloca" or _constant(argument_allocation.operands[0]) != 1:
                raise ValueError("SM90 launch argument is not a single typed allocation")
            type_name = str(argument_allocation.attributes["elem_type"])
            if type_name == _TMA_TYPE:
                _prove_tensor_map(_stored_value(pointer))
                schema = Sm90LaunchParameter(128, 64, 64, (0,))
            else:
                size, alignment = _scalar_layout(type_name)
                schema = Sm90LaunchParameter(size, alignment, size)
            if indices[0] in arguments:
                raise ValueError("SM90 launch argument index was initialized more than once")
            arguments[indices[0]] = schema
        if set(arguments) != set(range(count)):
            raise ValueError("SM90 retained launch argument table is incomplete")
        return tuple(arguments[index] for index in range(count))


def relocated_sm90_arguments(
    schema: tuple[Sm90LaunchParameter, ...],
    baseline: tuple[bytes, ...],
    addresses: dict[str, int],
    relocations: dict[str, tuple[int, tuple[bytes, ...]]],
    accesses: dict[str, CutePipelineSlotAccess],
) -> tuple[CutePipelineKernelArgument, ...]:
    """Bind independently relocated tensors at compiler-proven address fields.

    Each experiment changes exactly one tensor. Every defined byte must match
    the reconstructed argument; varying descriptor metadata is never interpreted
    as an additional pointer. Reserved TMA words are omitted and zero-filled.
    """
    if set(addresses) != set(relocations) or set(addresses) != set(accesses):
        raise ValueError("SM90 export requires an independent capture for every live tensor")
    captures = (baseline, *(capture for _, capture in relocations.values()))
    if any(len(capture) != len(schema) for capture in captures):
        raise ValueError("SM90 capture parameter count differs from retained compiler types")
    if any(len(data) != parameter.size for capture in captures for parameter, data in zip(schema, capture, strict=True)):
        raise ValueError("SM90 capture parameter width differs from retained compiler types")
    arguments = []
    bound_slots = set()
    for index, (parameter, data) in enumerate(zip(schema, baseline, strict=True)):
        name = f"argument_{index}"
        fields = []
        bindings = {}
        for offset in parameter.address_offsets:
            candidates = [
                slot
                for slot, address in addresses.items()
                if data[offset : offset + 8] == struct.pack("<Q", address)
                and relocations[slot][0] != address
                and relocations[slot][1][index][offset : offset + 8] == struct.pack("<Q", relocations[slot][0])
            ]
            if len(candidates) != 1:
                raise ValueError(f"SM90 parameter {index} address at {offset} lacks a unique tensor identity")
            slot = candidates[0]
            bindings[offset] = slot
            bound_slots.add(slot)
            fields.append(CutePipelineStructuredSlotAddressField(f"address_{offset}", slot, accesses[slot], 0, offset))
        for changed_slot, (changed_address, capture) in relocations.items():
            expected = bytearray(data[: parameter.defined_bytes])
            for offset, slot in bindings.items():
                if slot == changed_slot:
                    struct.pack_into("<Q", expected, offset, changed_address)
            if bytes(expected) != capture[index][: parameter.defined_bytes]:
                raise ValueError(f"SM90 parameter {index} has unexplained changes while relocating {changed_slot!r}")
        if not bindings:
            arguments.append(CutePipelinePackedBytesArgument(name, data, parameter.alignment))
            continue
        cursor = 0
        for offset in (*sorted(bindings), parameter.defined_bytes):
            if cursor < offset:
                fields.append(CutePipelineStructuredImmediateField(f"bytes_{cursor}", cursor, data[cursor:offset]))
            cursor = offset + 8
        arguments.append(CutePipelineStructuredArgument(name, parameter.size, parameter.alignment, tuple(fields)))
    if bound_slots != set(addresses):
        raise ValueError(f"SM90 launch omits live tensor addresses: {sorted(set(addresses) - bound_slots)}")
    return tuple(arguments)


def _constant(value: Any) -> int:
    operation = value.owner
    if operation.name != "llvm.mlir.constant":
        raise ValueError("SM90 ABI layout requires a constant index")
    return int(operation.attributes["value"].value)


def _indices(gep: Any) -> tuple[int, ...]:
    dynamic = iter(gep.operands[1:])
    return tuple(
        _constant(next(dynamic)) if index == -(2**31) else index for index in gep.attributes["rawConstantIndices"]
    )


def _stored_value(pointer: Any) -> Any:
    stores = [use.owner for use in pointer.uses if use.owner.name == "llvm.store" and use.owner.operands[1] == pointer]
    if len(stores) != 1:
        raise ValueError("SM90 ABI argument requires a unique initialization")
    return stores[0].operands[0]


def _prove_tensor_map(value: Any) -> None:
    insert = value.owner
    if insert.name != "llvm.insertvalue" or tuple(insert.attributes["position"]) != (0,):
        raise ValueError("SM90 tensor-map wrapper is not structurally recognized")
    load = insert.operands[1].owner
    if load.name != "llvm.load" or str(load.results[0].type) != "!llvm.struct<(array<16 x i64>)>":
        raise ValueError("SM90 tensor-map payload is not a typed descriptor load")
    pointer = load.operands[0]
    if pointer.owner.name == "llvm.inttoptr":
        cast = pointer.owner.operands[0].owner
        if cast.name != "llvm.ptrtoint":
            raise ValueError("SM90 tensor-map storage has an unknown address conversion")
        pointer = cast.operands[0]
    allocation = pointer.owner
    if (
        allocation.name != "llvm.alloca"
        or str(allocation.attributes["elem_type"]) != "i64"
        or _constant(allocation.operands[0]) != 16
        or int(allocation.attributes["alignment"].value) != 64
    ):
        raise ValueError("SM90 tensor map requires a 64-byte-aligned sixteen-word allocation")
    words = {}
    for use in pointer.uses:
        gep = use.owner
        if gep.name != "llvm.getelementptr":
            continue
        indices = _indices(gep)
        if str(gep.attributes["elem_type"]) != "!llvm.array<16 x i64>" or len(indices) != 2 or indices[0] != 0:
            raise ValueError("SM90 descriptor storage has an unsupported field access")
        stores = [use.owner for use in gep.results[0].uses if use.owner.name == "llvm.store"]
        if not stores or any(str(store.operands[0].type) != "i64" for store in stores):
            raise ValueError("SM90 descriptor word lacks typed initialization")
        words[indices[1]] = stores
    if set(words) != set(range(8)):
        raise ValueError("SM90 descriptor defined/reserved word boundary is not proved")
    # CuTe's address encoding drops four alignment bits and masks the 57-bit
    # CUDA address. Prove that exact expression before declaring a relocation.
    address_stores = [store for store in words[0] if store.operands[0].owner.name != "llvm.mlir.constant"]
    if len(address_stores) != 1:
        raise ValueError("SM90 descriptor does not have one address expression")
    shift = address_stores[0].operands[0].owner
    if shift.name != "llvm.shl" or _constant(shift.operands[1]) != 4:
        raise ValueError("unsupported SM90 descriptor address encoding")
    mask = shift.operands[0].owner
    if mask.name != "llvm.and" or _constant(mask.operands[1]) != 2**53 - 1:
        raise ValueError("unsupported SM90 descriptor address mask")
    divide = mask.operands[0].owner
    if (
        divide.name != "llvm.udiv"
        or _constant(divide.operands[1]) != 16
        or divide.operands[0].owner.name != "llvm.ptrtoint"
    ):
        raise ValueError("unsupported SM90 descriptor address alignment")


def _scalar_layout(type_name: str) -> tuple[int, int]:
    scalar = re.fullmatch(r"i(1|8|16|32|64)", type_name)
    if scalar:
        size = max(1, int(scalar[1]) // 8)
        return size, size
    type_name = type_name.removeprefix("!llvm.")
    if not type_name.startswith("struct<(") or not type_name.endswith(")>"):
        raise ValueError(f"unsupported SM90 by-value parameter type: {type_name}")
    body = type_name[len("struct<(") : -2]
    depth = 0
    start = 0
    children = []
    for index, character in enumerate(body):
        depth += character == "<"
        depth -= character == ">"
        if character == "," and depth == 0:
            children.append(body[start:index].strip())
            start = index + 1
    if body:
        children.append(body[start:].strip())
    size, alignment = 0, 1
    for child in children:
        child_size, child_alignment = _scalar_layout(child)
        if size % child_alignment:
            raise ValueError("SM90 scalar record has unmodeled padding")
        size += child_size
        alignment = max(alignment, child_alignment)
    if size % alignment:
        raise ValueError("SM90 scalar record has unmodeled tail padding")
    return size, alignment
