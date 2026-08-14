# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent decoder for the closed ABI15 composed H100 transports."""

import hashlib
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from abi15_h100_architecture_contract import load_architecture_build_contract
from gpu_transport import decode_mlir_byte_string

MAGIC = b"SHUTGPU\0"
TRANSPORT_SCHEMA_VERSION = 2
DEVICE_SCHEMA_VERSION = 3
INVOCATION_SCHEMA_VERSION = 3
BUNDLE_SCHEMA_VERSION = 2
MAXIMUM_TRANSPORT_BYTES = 16 * 1024 * 1024
MAXIMUM_RECORDS = 256
MAXIMUM_CODE_BYTES = 8 * 1024 * 1024
MAXIMUM_ENTRY_CODE_BYTES = 512 * 1024
MAXIMUM_SLOT_BYTES = 32 * 1024 * 1024
MAXIMUM_TEMPORARY_BYTES = 256 * 1024 * 1024
MAXIMUM_AGGREGATE_TASK_POSITIONS = 67_129_347
MAXIMUM_DYNAMIC_SHARED_MEMORY_BYTES = 16 * 1024
MAXIMUM_RANK = 8
MAXIMUM_BLOCK_THREADS = 1024
MAXIMUM_I64 = (1 << 63) - 1
TARGET = b"shuttle.gpu.executable_bundle.v2"

_DIGEST = re.compile(rb"[0-9a-f]{64}\Z")
_HEX_ESCAPE = re.compile(r"[0-9A-Fa-f]{2}\Z")
_REGISTER = r"%rd[0-9]+"
_INVALID = "invalid ABI15 Shuttle GPU executable transport"
_Enum = TypeVar("_Enum", bound=StrEnum)


class NumericalPolicy(StrEnum):
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


class CallRole(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


class ElementType(StrEnum):
    BF16 = "bf16"
    F32 = "f32"


class Access(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class Storage(StrEnum):
    EXTERNAL = "external"
    TEMPORARY = "temporary"


class BindingKind(StrEnum):
    NONE = "none"
    OPERAND = "operand"
    RESULT = "result"


class ReductionOrder(StrEnum):
    SERIAL_ASCENDING_FMA_RN_F32 = "serial_ascending_fma_rn_f32"
    LEAF_ORDER_FREE = "leaf_order_free"


@dataclass(frozen=True)
class TransportCoordinate:
    """One policy and custom-call role in the ABI15 composed cell."""

    policy: NumericalPolicy
    call_role: CallRole


@dataclass(frozen=True)
class GpuLaunch:
    """One decoded launch and its exact PTX slice."""

    ordinal: int
    code_offset: int
    code_length: int
    code_digest: str
    code: bytes
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]
    dependencies: tuple[int, ...]
    reduction_order: ReductionOrder | None


@dataclass(frozen=True)
class GpuSlot:
    """One decoded invocation slot."""

    ordinal: int
    element_type: ElementType
    dimensions: tuple[int, ...]
    required_bytes: int
    access: Access
    storage: Storage
    binding_kind: BindingKind
    binding_index: int | None


@dataclass(frozen=True)
class GpuExternalBinding:
    """One external operand or result binding."""

    kind: BindingKind
    index: int
    slot_ordinal: int
    element_type: ElementType
    dimensions: tuple[int, ...]
    required_bytes: int


@dataclass(frozen=True)
class GpuTransport:
    """Closed ABI15 transport projection used by the evidence verifier."""

    coordinate: TransportCoordinate
    schedule_fingerprint: str
    plan_fingerprint: str
    transport_root: str
    device_root: str
    invocation_root: str
    bundle_root: str
    code_digest: str
    launches: tuple[GpuLaunch, ...]
    slots: tuple[GpuSlot, ...]
    external_bindings: tuple[GpuExternalBinding, ...]
    temporary_bytes: int


class _InvalidTransport(Exception):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    @property
    def position(self) -> int:
        return self.offset

    def take(self, length: int) -> bytes:
        if length < 0 or length > len(self.data) - self.offset:
            raise _InvalidTransport
        start = self.offset
        self.offset += length
        return self.data[start : self.offset]

    def unsigned(self, width: int) -> int:
        return int.from_bytes(self.take(width), "little", signed=False)

    def u8(self) -> int:
        return self.unsigned(1)

    def u16(self) -> int:
        return self.unsigned(2)

    def u32(self) -> int:
        return self.unsigned(4)

    def u64(self) -> int:
        return self.unsigned(8)

    def i64(self) -> int:
        return int.from_bytes(self.take(8), "little", signed=True)

    def blob(self) -> bytes:
        return self.take(self.u64())

    def text(self) -> bytes:
        return self.blob()

    def integers(self) -> tuple[int, ...]:
        count = self.u64()
        if count > MAXIMUM_RECORDS or count > (len(self.data) - self.offset) // 8:
            raise _InvalidTransport
        return tuple(self.i64() for _ in range(count))


def _require(condition: bool) -> None:
    if not condition:
        raise _InvalidTransport


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _digest_text(value: bytes) -> str:
    _require(_DIGEST.fullmatch(value) is not None)
    return value.decode("ascii")


def _enum_member(values: tuple[_Enum, ...], value: int) -> _Enum:
    _require(0 <= value < len(values))
    return values[value]


def _checked_product(values: tuple[int, ...], maximum: int) -> int:
    product = 1
    for value in values:
        _require(value > 0 and value <= maximum // product)
        product *= value
    return product


def _unquoted_mlir_syntax(text: str) -> str:
    syntax = list(text)
    in_string = False
    position = 0
    while position < len(text):
        character = text[position]
        if not in_string:
            if character == '"':
                syntax[position] = " "
                in_string = True
            position += 1
            continue
        syntax[position] = " "
        if character == '"':
            in_string = False
            position += 1
            continue
        if character != "\\":
            position += 1
            continue
        escaped = text[position + 1 : position + 3]
        escape_length = 2 if len(escaped) == 2 and _HEX_ESCAPE.fullmatch(escaped) is not None else 1
        for offset in range(1, escape_length + 1):
            if position + offset < len(syntax):
                syntax[position + offset] = " "
        position += escape_length + 1
    if in_string:
        raise ValueError("canonical MLIR contains an unterminated string")
    return "".join(syntax)


def _quoted_attribute(operation: str, syntax: str, name: str) -> bytes:
    matches = list(re.finditer(rf"\b{re.escape(name)}\s*=", syntax))
    if len(matches) != 1:
        raise ValueError(f"canonical MLIR must contain one {name}")
    position = matches[0].end()
    while position < len(operation) and operation[position].isspace():
        position += 1
    if position >= len(operation) or operation[position] != '"':
        raise ValueError(f"canonical MLIR {name} must be a quoted string")
    value, end = decode_mlir_byte_string(operation, position + 1)
    if end > len(operation):
        raise ValueError(f"canonical MLIR {name} is unterminated")
    return value


def extract_gpu_transports(generated_mlir: str) -> tuple[bytes, bytes]:
    """Extract exactly the forward and backward v2 calls in source order."""
    operations = [line for line in generated_mlir.splitlines() if '"stablehlo.custom_call"' in line]
    if len(operations) != 2:
        raise ValueError("ABI15 generated MLIR must contain exactly two StableHLO custom calls")
    bundles = []
    for operation in operations:
        if (
            re.match(
                r'^\s*%[A-Za-z0-9_.$-]+(?::[0-9]+)?\s*=\s*"stablehlo\.custom_call"\s*\(',
                operation,
            )
            is None
        ):
            raise ValueError("ABI15 custom call is not parser-canonical")
        syntax = _unquoted_mlir_syntax(operation)
        if "//" in syntax:
            raise ValueError("ABI15 custom call contains a trailing comment")
        if _quoted_attribute(operation, syntax, "call_target_name") != TARGET:
            raise ValueError("ABI15 generated MLIR contains the wrong custom-call target")
        if re.findall(r"\bapi_version\s*=\s*(-?[0-9]+)\s*:\s*i32\b", syntax) != ["4"]:
            raise ValueError("ABI15 custom call must use API version 4")
        bundles.append(_quoted_attribute(operation, syntax, "bundle_bytes"))
    return bundles[0], bundles[1]


def _expected_counts(coordinate: TransportCoordinate) -> tuple[int, int, int]:
    if coordinate.call_role is CallRole.FORWARD:
        return 2, 4, 3
    return 10, 13, 5


def _decode_launches(reader: _Reader, code: bytes, slot_count: int, launch_count: int) -> tuple[GpuLaunch, ...]:
    _require(reader.u64() == launch_count)
    launches = []
    producers: dict[int, int] = {}
    expected_offset = 0
    aggregate_positions = 0
    for ordinal in range(launch_count):
        _require(reader.i64() == ordinal and reader.i64() == ordinal)
        code_offset = reader.i64()
        code_length = reader.i64()
        inputs = reader.integers()
        outputs = reader.integers()
        _require(code_offset == expected_offset and 0 < code_length <= MAXIMUM_ENTRY_CODE_BYTES)
        input_count = reader.u64()
        _require(input_count == len(inputs) and all(reader.u8() == 0 for _ in range(input_count)))
        output_count = reader.u64()
        _require(output_count == len(outputs) and all(reader.u8() == 1 for _ in range(output_count)))
        dependencies = reader.integers()
        _require(reader.u8() in (0, 1))
        has_reduction = reader.u8()
        _require(has_reduction in (0, 1))
        reduction_order = None
        if has_reduction:
            reduction_order = _enum_member(
                (
                    ReductionOrder.SERIAL_ASCENDING_FMA_RN_F32,
                    ReductionOrder.LEAF_ORDER_FREE,
                ),
                reader.u8(),
            )
        entry_digest = _digest_text(reader.text())
        _require(expected_offset <= len(code) - code_length)
        code_slice = code[expected_offset : expected_offset + code_length]
        _require(entry_digest == _sha256(code_slice))
        expected_offset += code_length
        grid = (reader.u32(), reader.u32(), reader.u32())
        block = (reader.u16(), reader.u16(), reader.u16())
        shared_memory = reader.u32()
        arity = reader.u16()
        _require(_checked_product(block, MAXIMUM_BLOCK_THREADS) <= MAXIMUM_BLOCK_THREADS)
        positions = _checked_product(grid + block, MAXIMUM_AGGREGATE_TASK_POSITIONS)
        _require(positions <= MAXIMUM_AGGREGATE_TASK_POSITIONS - aggregate_positions)
        aggregate_positions += positions
        _require(shared_memory <= MAXIMUM_DYNAMIC_SHARED_MEMORY_BYTES and arity == len(inputs) + len(outputs))
        _require(all(0 <= slot < slot_count for slot in inputs + outputs))
        expected_dependencies = tuple(sorted({producers[slot] for slot in inputs if slot in producers}))
        _require(dependencies == expected_dependencies and len(set(dependencies)) == len(dependencies))
        for output in outputs:
            _require(output not in producers)
            producers[output] = ordinal
        _require(code_slice.endswith(b"\n") and b".visible .entry shuttle_entry(" in code_slice)
        launches.append(
            GpuLaunch(
                ordinal,
                code_offset,
                code_length,
                entry_digest,
                code_slice,
                grid,
                block,
                inputs,
                outputs,
                dependencies,
                reduction_order,
            )
        )
    _require(expected_offset == len(code))
    return tuple(launches)


def _contiguous_strides(element_bytes: int, dimensions: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(dimensions)
    stride = element_bytes
    for axis in range(len(dimensions) - 1, -1, -1):
        strides[axis] = stride
        stride *= dimensions[axis]
    return tuple(strides)


def _decode_slots(reader: _Reader, slot_count: int) -> tuple[tuple[GpuSlot, ...], tuple[GpuExternalBinding, ...], int]:
    _require(reader.u64() == slot_count)
    slots = []
    bindings = []
    temporary_bytes = 0
    for ordinal in range(slot_count):
        _require(reader.i64() == ordinal and reader.i64() == ordinal)
        element_type = _enum_member((ElementType.BF16, ElementType.F32), reader.u8())
        rank = reader.u64()
        _require(rank <= MAXIMUM_RANK)
        element_bytes = 2 if element_type is ElementType.BF16 else 4
        dimensions = tuple(reader.i64() for _ in range(rank))
        _require(all(extent > 0 for extent in dimensions))
        required_bytes = element_bytes
        for extent in dimensions:
            _require(required_bytes <= MAXIMUM_I64 // extent)
            required_bytes *= extent
        encoded_required = reader.i64()
        strides = reader.integers()
        offset = reader.i64()
        alignment = reader.i64()
        address_space = reader.u8()
        access = _enum_member((Access.READ, Access.WRITE, Access.READ_WRITE), reader.u8())
        storage = _enum_member((Storage.EXTERNAL, Storage.TEMPORARY), reader.u8())
        alias_group = reader.i64()
        reuse_group = reader.i64()
        binding_kind = _enum_member((BindingKind.NONE, BindingKind.OPERAND, BindingKind.RESULT), reader.u8())
        has_binding = reader.u8()
        _require(has_binding in (0, 1))
        binding_index = reader.i64() if has_binding else None
        _require(
            encoded_required == required_bytes
            and required_bytes <= MAXIMUM_SLOT_BYTES
            and strides == _contiguous_strides(element_bytes, dimensions)
            and offset == 0
            and alignment == element_bytes
            and address_space == 1
            and alias_group == ordinal
            and reuse_group == ordinal
        )
        if storage is Storage.TEMPORARY:
            _require(binding_kind is BindingKind.NONE and binding_index is None and access is Access.READ_WRITE)
            _require(required_bytes <= MAXIMUM_TEMPORARY_BYTES - temporary_bytes)
            temporary_bytes += required_bytes
        else:
            _require(binding_kind is not BindingKind.NONE and binding_index is not None)
            _require(binding_kind is not BindingKind.OPERAND or access is Access.READ)
            _require(binding_kind is not BindingKind.RESULT or access is Access.WRITE)
            bindings.append(
                GpuExternalBinding(
                    binding_kind,
                    binding_index,
                    ordinal,
                    element_type,
                    dimensions,
                    required_bytes,
                )
            )
        slots.append(
            GpuSlot(
                ordinal,
                element_type,
                dimensions,
                required_bytes,
                access,
                storage,
                binding_kind,
                binding_index,
            )
        )
    return tuple(slots), tuple(bindings), temporary_bytes


def _require_bindings(coordinate: TransportCoordinate, bindings: tuple[GpuExternalBinding, ...]) -> None:
    if coordinate.call_role is CallRole.FORWARD:
        expected = (
            (BindingKind.OPERAND, 0, (33, 2)),
            (BindingKind.OPERAND, 1, (2, 34)),
            (BindingKind.RESULT, 0, (33, 34)),
        )
    else:
        expected = (
            (BindingKind.OPERAND, 0, (33, 34)),
            (BindingKind.OPERAND, 1, (33, 2)),
            (BindingKind.OPERAND, 2, (2, 34)),
            (BindingKind.RESULT, 1, (2, 34)),
            (BindingKind.RESULT, 0, (33, 2)),
        )
    observed = tuple((binding.kind, binding.index, binding.dimensions) for binding in bindings)
    _require(observed == expected and all(binding.element_type is ElementType.BF16 for binding in bindings))


def _address_maps(ptx: bytes, output_shape: tuple[int, int], reduction: bool) -> dict[int, tuple[int, int]]:
    text = ptx.decode("ascii")
    coordinate_registers: dict[str, int] = {}
    for dimension, (stride, extent) in enumerate(((output_shape[1], output_shape[0]), (1, output_shape[1]))):
        pattern = re.compile(rf"div\.u64\s+({_REGISTER}),\s*%rd0,\s*{stride};\s*rem\.u64\s+\1,\s*\1,\s*{extent};")
        match = pattern.search(text)
        _require(match is not None)
        coordinate_registers[match.group(1)] = dimension
    if reduction:
        serial = re.search(rf"mov\.u64\s+({_REGISTER}),\s*0;", text)
        lane = re.search(rf"cvt\.u64\.u32\s+({_REGISTER}),\s*%r2;", text)
        reduction_register = lane or serial
        _require(reduction_register is not None)
        coordinate_registers[reduction_register.group(1)] = 2

    linear_maps: dict[str, list[int]] = {}
    parameter_maps: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        start = re.fullmatch(rf"\s*mov\.u64\s+({_REGISTER}),\s*0;\s*", line)
        if start and start.group(1) not in coordinate_registers:
            linear_maps[start.group(1)] = []
            continue
        step = re.fullmatch(
            rf"\s*mad\.lo\.u64\s+({_REGISTER}),\s*\1,\s*[0-9]+,\s*({_REGISTER});\s*",
            line,
        )
        if step and step.group(1) in linear_maps and step.group(2) in coordinate_registers:
            linear_maps[step.group(1)].append(coordinate_registers[step.group(2)])
            continue
        pointer = re.fullmatch(rf"\s*ld\.param\.u64\s+({_REGISTER}),\s*\[p([0-9]+)\];\s*", line)
        if not pointer:
            continue
        pointer_register, parameter = pointer.group(1), int(pointer.group(2))
        pointer_use = re.search(
            rf"mad\.lo\.u64\s+{re.escape(pointer_register)},\s*({_REGISTER}),\s*[24],\s*{re.escape(pointer_register)};",
            text,
        )
        if pointer_use and len(linear_maps.get(pointer_use.group(1), ())) == 2:
            parameter_maps[parameter] = tuple(linear_maps[pointer_use.group(1)])  # type: ignore[assignment]
    return parameter_maps


def _validate_contract_ptx(ptx: bytes, coordinate: TransportCoordinate, ordinal: int) -> None:
    if coordinate.call_role is CallRole.FORWARD:
        output_shape, expected = (33, 34), {0: (0, 2), 1: (2, 1)}
    elif ordinal == 3:
        output_shape, expected = (34, 2), {0: (2, 0), 1: (2, 1)}
    else:
        output_shape, expected = (33, 2), {0: (0, 2), 1: (1, 2)}
    _require(_address_maps(ptx, output_shape, reduction=True) == expected)
    text = ptx.decode("ascii")
    _require(text.count("fma.rn.f32") == 1 and "atom." not in text and "mma." not in text)
    if coordinate.policy is NumericalPolicy.SOURCE_ORDERED:
        _require("shfl.sync.down.b32" not in text and re.search(r"add\.u64 .*?, 1;", text) is not None)
    else:
        _require(re.search(r"add\.u64 .*?, 32;", text) is not None)
        cursor = text.find("fma.rn.f32")
        for offset in (16, 8, 4, 2, 1):
            shuffle = text.find("shfl.sync.down.b32", cursor)
            _require(shuffle >= 0 and f", {offset}, 31, 0xffffffff;" in text[shuffle:])
            add = text.find("add.rn.f32", shuffle)
            _require(add > shuffle)
            cursor = add + 1


def _validate_map_ptx(ptx: bytes, input_slot: GpuSlot, output_slot: GpuSlot, kind: str) -> None:
    expected_map = (1, 0) if kind == "map_f32_transpose_d1_d0" else (0, 1)
    _require(_address_maps(ptx, output_slot.dimensions, reduction=False) == {0: expected_map})
    text = ptx.decode("ascii")
    _require("fma.rn.f32" not in text and "shfl." not in text and "atom." not in text and "mma." not in text)
    _require(text.count("ld.param.u64") == 2)
    if input_slot.element_type is ElementType.BF16:
        _require(text.count("ld.global.u16") == 1 and text.count("cvt.f32.bf16") == 1)
        _require("ld.global.f32" not in text)
    else:
        _require(text.count("ld.global.f32") == 1 and "ld.global.u16" not in text)
    if output_slot.element_type is ElementType.BF16:
        _require(text.count("cvt.rn.bf16.f32") == 1 and text.count("st.global.u16") == 1)
        _require("st.global.f32" not in text)
    else:
        _require(text.count("st.global.f32") == 1 and "st.global.u16" not in text)


def _validate_architecture(
    coordinate: TransportCoordinate,
    launches: tuple[GpuLaunch, ...],
    slots: tuple[GpuSlot, ...],
) -> None:
    design = load_architecture_build_contract()
    expected = design["coordinates"][coordinate.call_role.value]
    for slot, spec in zip(slots, expected["slots"], strict=True):
        _require(
            slot.ordinal == spec["ordinal"]
            and slot.element_type.value == spec["element_type"]
            and list(slot.dimensions) == spec["dimensions"]
            and slot.access.value == spec["access"]
            and slot.storage.value == spec["storage"]
            and slot.binding_kind.value == spec["binding_kind"]
            and slot.binding_index == spec["binding_index"]
        )
    for launch, spec in zip(launches, expected["launches"], strict=True):
        _require(
            launch.ordinal == spec["ordinal"]
            and list(launch.input_slots) == spec["input_slots"]
            and list(launch.output_slots) == spec["output_slots"]
            and list(launch.dependencies) == spec["dependencies"]
        )
        kind = spec["kind"]
        if kind.startswith("contract_"):
            expected_order = (
                ReductionOrder.SERIAL_ASCENDING_FMA_RN_F32
                if coordinate.policy is NumericalPolicy.SOURCE_ORDERED
                else ReductionOrder.LEAF_ORDER_FREE
            )
            _require(launch.reduction_order is expected_order)
            _validate_contract_ptx(launch.code, coordinate, launch.ordinal)
        else:
            _require(launch.reduction_order is None)
            _validate_map_ptx(launch.code, slots[launch.input_slots[0]], slots[launch.output_slots[0]], kind)


def _validate_ptx(
    coordinate: TransportCoordinate,
    launches: tuple[GpuLaunch, ...],
    slots: tuple[GpuSlot, ...],
) -> None:
    contract_ordinals = (0,) if coordinate.call_role is CallRole.FORWARD else (3, 8)
    for ordinal in contract_ordinals:
        expected_order = (
            ReductionOrder.SERIAL_ASCENDING_FMA_RN_F32
            if coordinate.policy is NumericalPolicy.SOURCE_ORDERED
            else ReductionOrder.LEAF_ORDER_FREE
        )
        _require(launches[ordinal].reduction_order is expected_order)
        _validate_contract_ptx(launches[ordinal].code, coordinate, ordinal)
    if coordinate.call_role is CallRole.BACKWARD:
        transpose = _address_maps(launches[4].code, (2, 34), reduction=False)
        _require(transpose.get(0) == (1, 0))
    _validate_architecture(coordinate, launches, slots)


def _parse_transport(data: bytes, coordinate: TransportCoordinate) -> GpuTransport:
    launch_count, slot_count, binding_count = _expected_counts(coordinate)
    reader = _Reader(data)
    _require(len(data) <= MAXIMUM_TRANSPORT_BYTES)
    _require(reader.take(len(MAGIC)) == MAGIC and reader.u32() == TRANSPORT_SCHEMA_VERSION)
    device_start = reader.position
    _require(reader.i64() == DEVICE_SCHEMA_VERSION and reader.u8() == 2)
    policy = _enum_member((NumericalPolicy.SOURCE_ORDERED, NumericalPolicy.FAST), reader.u8())
    _require(policy is coordinate.policy)
    schedule = _digest_text(reader.text())
    code = reader.blob()
    code_digest = _digest_text(reader.text())
    device_root_offset = reader.position
    device_root = _digest_text(reader.text())
    device_suffix_start = reader.position
    _require(0 < len(code) <= MAXIMUM_CODE_BYTES and code_digest == _sha256(code))
    launches = _decode_launches(reader, code, slot_count, launch_count)
    invocation_start = reader.position
    _require(
        device_root
        == _sha256(
            data[device_start:device_root_offset],
            data[device_suffix_start:invocation_start],
        )
    )
    _require(reader.i64() == INVOCATION_SCHEMA_VERSION)
    plan = _digest_text(reader.text())
    _require(_digest_text(reader.text()) == schedule)
    invocation_root_offset = reader.position
    invocation_root = _digest_text(reader.text())
    invocation_suffix_start = reader.position
    slots, bindings, temporary_bytes = _decode_slots(reader, slot_count)
    bundle_start = reader.position
    _require(
        invocation_root
        == _sha256(
            data[invocation_start:invocation_root_offset],
            data[invocation_suffix_start:bundle_start],
        )
    )
    _require(len(bindings) == binding_count)
    _require_bindings(coordinate, bindings)
    _require(reader.i64() == BUNDLE_SCHEMA_VERSION)
    _require(_digest_text(reader.text()) == schedule)
    _require(_digest_text(reader.text()) == device_root)
    _require(_digest_text(reader.text()) == invocation_root)
    _require(reader.u8() == 1)
    bundle_root_offset = reader.position
    bundle_root = _digest_text(reader.text())
    _require(bundle_root == _sha256(data[:bundle_root_offset]) and reader.position == len(data))
    _validate_ptx(coordinate, launches, slots)
    return GpuTransport(
        coordinate,
        schedule,
        plan,
        _sha256(data),
        device_root,
        invocation_root,
        bundle_root,
        code_digest,
        launches,
        slots,
        bindings,
        temporary_bytes,
    )


def load_gpu_transport(path: Path, coordinate: TransportCoordinate) -> GpuTransport:
    """Load one exact ABI15 policy/call transport and fail closed on drift."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > MAXIMUM_TRANSPORT_BYTES:
            raise ValueError(_INVALID)
        data = stream.read(MAXIMUM_TRANSPORT_BYTES + 1)
    if len(data) != size or len(data) > MAXIMUM_TRANSPORT_BYTES:
        raise ValueError(_INVALID)
    try:
        return _parse_transport(data, coordinate)
    except (UnicodeDecodeError, _InvalidTransport):
        raise ValueError(_INVALID) from None


def gpu_transport_descriptor(transport: GpuTransport) -> dict[str, object]:
    """Return the canonical independent projection stored beside a transport."""
    return {
        "bundle_root": transport.bundle_root,
        "call": transport.coordinate.call_role.value,
        "code_sha256": transport.code_digest,
        "device_root": transport.device_root,
        "external_bindings": [
            {
                "binding_index": binding.index,
                "dimensions": list(binding.dimensions),
                "element_type": binding.element_type.value,
                "kind": binding.kind.value,
                "required_bytes": binding.required_bytes,
                "slot_ordinal": binding.slot_ordinal,
            }
            for binding in transport.external_bindings
        ],
        "invocation_root": transport.invocation_root,
        "launches": [
            {
                "block": list(launch.block),
                "code_length": launch.code_length,
                "code_offset": launch.code_offset,
                "dependencies": list(launch.dependencies),
                "grid": list(launch.grid),
                "input_slots": list(launch.input_slots),
                "ordinal": launch.ordinal,
                "output_slots": list(launch.output_slots),
                "ptx_sha256": launch.code_digest,
                "reduction_order": launch.reduction_order.value if launch.reduction_order is not None else None,
            }
            for launch in transport.launches
        ],
        "plan_fingerprint": transport.plan_fingerprint,
        "policy": transport.coordinate.policy.value,
        "schedule_fingerprint": transport.schedule_fingerprint,
        "schema_version": 1,
        "slots": [
            {
                "access": slot.access.value,
                "binding_index": slot.binding_index,
                "binding_kind": slot.binding_kind.value,
                "dimensions": list(slot.dimensions),
                "element_type": slot.element_type.value,
                "ordinal": slot.ordinal,
                "required_bytes": slot.required_bytes,
                "storage": slot.storage.value,
            }
            for slot in transport.slots
        ],
        "temporary_bytes": transport.temporary_bytes,
        "transport_sha256": transport.transport_root,
    }
