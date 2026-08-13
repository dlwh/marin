# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent decoder for the closed ABI 10 GPU transport schema."""

import hashlib
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

MAGIC = b"SHUTGPU\0"
TRANSPORT_SCHEMA_VERSION = 2
DEVICE_SCHEMA_VERSION = 3
INVOCATION_ABI_SCHEMA_VERSION = 3
BUNDLE_SCHEMA_VERSION = 2
LAUNCH_COUNT: int = 19
SLOT_COUNT: int = 21
TEMPORARY_SLOT_COUNT: int = 18
TEMPORARY_BYTES = 201_416_716

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

_DIGEST = re.compile(rb"[0-9a-f]{64}\Z")
_HEX_ESCAPE = re.compile(r"[0-9A-Fa-f]{2}\Z")
_INVALID = "invalid Shuttle GPU executable transport"
_Enum = TypeVar("_Enum", bound=StrEnum)


class ElementType(StrEnum):
    BF16 = "bf16"
    F32 = "f32"


class CodeFormat(StrEnum):
    CUDA_PTX_SM90_V1 = "cuda_ptx_sm90_v1"


class NumericalPolicy(StrEnum):
    SOURCE_ORDERED = "source_ordered"


class Predication(StrEnum):
    NONE = "none"
    DOMAIN_BOUNDS = "domain_bounds"


class ReductionOrder(StrEnum):
    TREE_ASSOCIATION_FREE_LEAF_ORDER_FIXED = "tree_association_free_leaf_order_fixed"


class Access(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class AddressSpace(StrEnum):
    DEVICE = "device"


class Storage(StrEnum):
    EXTERNAL = "external"
    TEMPORARY = "temporary"


class BindingKind(StrEnum):
    NONE = "none"
    OPERAND = "operand"
    RESULT = "result"


class Completion(StrEnum):
    STREAM_ORDERED = "stream_ordered"


@dataclass(frozen=True)
class GpuLaunch:
    """One launch in source-task order, including its verified PTX slice."""

    ordinal: int
    source_task: int
    code_offset: int
    code_length: int
    code_digest: str
    code: bytes
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]
    input_accesses: tuple[Access, ...]
    output_accesses: tuple[Access, ...]
    dependencies: tuple[int, ...]
    predication: Predication
    reduction_order: ReductionOrder | None


@dataclass(frozen=True)
class GpuSlot:
    """One invocation slot in source-buffer order."""

    ordinal: int
    source_buffer: int
    element_type: ElementType
    dimensions: tuple[int, ...]
    required_bytes: int
    byte_strides: tuple[int, ...]
    byte_offset: int
    alignment: int
    address_space: AddressSpace
    access: Access
    storage: Storage
    alias_group: int
    reuse_group: int
    binding_kind: BindingKind
    binding_index: int | None


@dataclass(frozen=True)
class GpuExternalBinding:
    """One operand or result binding projected from an external slot."""

    kind: BindingKind
    index: int
    slot_ordinal: int
    element_type: ElementType
    dimensions: tuple[int, ...]
    required_bytes: int
    alignment: int


@dataclass(frozen=True)
class GpuTransport:
    """Validated roots and ordered metadata for one ABI 10 GPU transport.

    ``transport_root`` hashes the complete file. The device, invocation, and
    bundle roots are the closed hashes embedded by the production serializer.
    """

    transport_schema_version: int
    device_schema_version: int
    invocation_abi_schema_version: int
    bundle_schema_version: int
    code_format: CodeFormat
    numerical_policy: NumericalPolicy
    completion: Completion
    schedule_fingerprint: str
    plan_fingerprint: str
    transport_root: str
    device_root: str
    invocation_root: str
    bundle_root: str
    transport_size: int
    code_digest: str
    code: bytes
    launches: tuple[GpuLaunch, ...]
    slots: tuple[GpuSlot, ...]
    external_bindings: tuple[GpuExternalBinding, ...]
    temporary_bytes: int


class _InvalidTransport(Exception):
    pass


def decode_mlir_byte_string(text: str, start: int = 0) -> tuple[bytes, int]:
    """Decode one MLIR quoted byte string beginning after its opening quote."""
    decoded = bytearray()
    position = start
    while position < len(text):
        character = text[position]
        if character == '"':
            return bytes(decoded), position + 1
        if character != "\\":
            codepoint = ord(character)
            if codepoint < 0x20 or codepoint > 0x7E:
                raise ValueError("MLIR bundle string contains a non-ASCII literal")
            decoded.append(codepoint)
            position += 1
            continue
        if position + 1 >= len(text):
            raise ValueError("MLIR bundle string ends in an incomplete escape")
        escaped = text[position + 1]
        if escaped in {'"', "\\"}:
            decoded.append(ord(escaped))
            position += 2
            continue
        token = text[position + 1 : position + 3]
        if len(token) != 2 or _HEX_ESCAPE.fullmatch(token) is None:
            raise ValueError("MLIR bundle string contains an unsupported escape")
        decoded.append(int(token, 16))
        position += 3
    raise ValueError("MLIR bundle string is unterminated")


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


def _quoted_mlir_attributes(operation: str, syntax: str, name: str) -> list[bytes]:
    values: list[bytes] = []
    for match in re.finditer(rf"\b{re.escape(name)}\s*=", syntax):
        position = match.end()
        while position < len(operation) and operation[position].isspace():
            position += 1
        if position >= len(operation) or operation[position] != '"':
            raise ValueError(f"canonical MLIR {name} must be a quoted string")
        value, _ = decode_mlir_byte_string(operation, position + 1)
        values.append(value)
    return values


def extract_gpu_transport(generated_mlir: str) -> bytes:
    """Extract the bundle from one parser-canonicalized Shuttle custom call."""
    operation_lines = [line for line in generated_mlir.splitlines() if '"stablehlo.custom_call"' in line]
    if (
        len(operation_lines) != 1
        or re.match(r'^\s*%[A-Za-z0-9_.$-]+\s*=\s*"stablehlo\.custom_call"\s*\(', operation_lines[0]) is None
    ):
        raise ValueError("canonical MLIR must contain exactly one StableHLO custom call")
    operation = operation_lines[0]
    syntax = _unquoted_mlir_syntax(operation)
    if "//" in syntax:
        raise ValueError("canonical MLIR custom call contains a trailing comment")
    targets = _quoted_mlir_attributes(operation, syntax, "call_target_name")
    if targets != [b"shuttle.gpu.executable_bundle.v1"]:
        raise ValueError("canonical MLIR must contain the exact Shuttle GPU target")
    api_versions = re.findall(r"\bapi_version\s*=\s*(-?[0-9]+)\s*:\s*i32\b", syntax)
    if api_versions != ["4"]:
        raise ValueError("canonical MLIR must contain the API-v4 custom call")
    bundles = _quoted_mlir_attributes(operation, syntax, "bundle_bytes")
    if len(bundles) != 1:
        raise ValueError("generated MLIR must contain exactly one bundle_bytes string")
    return bundles[0]


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


def _decode_launches(reader: _Reader, code: bytes) -> tuple[GpuLaunch, ...]:
    _require(reader.u64() == LAUNCH_COUNT)
    launches: list[GpuLaunch] = []
    producers: dict[int, int] = {}
    expected_offset = 0
    aggregate_positions = 0

    for ordinal in range(LAUNCH_COUNT):
        entry_ordinal = reader.i64()
        source_task = reader.i64()
        code_offset = reader.i64()
        code_length = reader.i64()
        inputs = reader.integers()
        outputs = reader.integers()
        input_count = reader.u64()
        _require(
            entry_ordinal == ordinal
            and source_task == ordinal
            and code_offset == expected_offset
            and 0 < code_length <= MAXIMUM_ENTRY_CODE_BYTES
            and input_count == len(inputs)
        )
        _require(all(reader.u8() == 0 for _ in range(input_count)))
        output_count = reader.u64()
        _require(output_count == len(outputs))
        _require(all(reader.u8() == 1 for _ in range(output_count)))

        dependencies = reader.integers()
        predication = _enum_member((Predication.NONE, Predication.DOMAIN_BOUNDS), reader.u8())
        has_reduction = reader.u8()
        _require(has_reduction <= 1)
        reduction_order = None
        if has_reduction:
            reduction_order = _enum_member((ReductionOrder.TREE_ASSOCIATION_FREE_LEAF_ORDER_FIXED,), reader.u8())
        entry_digest = _digest_text(reader.text())

        _require(expected_offset <= len(code) - code_length)
        code_slice = code[expected_offset : expected_offset + code_length]
        _require(entry_digest == _sha256(code_slice))
        expected_offset += code_length

        grid = (reader.u32(), reader.u32(), reader.u32())
        block = (reader.u16(), reader.u16(), reader.u16())
        dynamic_shared_memory_bytes = reader.u32()
        arity = reader.u16()
        _require(_checked_product(block, MAXIMUM_BLOCK_THREADS) <= MAXIMUM_BLOCK_THREADS)
        positions = _checked_product(grid + block, MAXIMUM_AGGREGATE_TASK_POSITIONS)
        _require(positions <= MAXIMUM_AGGREGATE_TASK_POSITIONS - aggregate_positions)
        aggregate_positions += positions
        _require(dynamic_shared_memory_bytes <= MAXIMUM_DYNAMIC_SHARED_MEMORY_BYTES)
        _require(arity == len(inputs) + len(outputs))
        _require(all(0 <= slot < SLOT_COUNT for slot in inputs + outputs))

        expected_dependencies = tuple(sorted({producers[slot] for slot in inputs if slot in producers}))
        _require(
            len(set(dependencies)) == len(dependencies)
            and all(0 <= dependency < ordinal for dependency in dependencies)
            and dependencies == expected_dependencies
        )
        for output in outputs:
            _require(output not in producers)
            producers[output] = ordinal

        _require(code_slice.endswith(b"\n") and b".visible .entry shuttle_entry(" in code_slice)
        launches.append(
            GpuLaunch(
                ordinal=ordinal,
                source_task=source_task,
                code_offset=code_offset,
                code_length=code_length,
                code_digest=entry_digest,
                code=code_slice,
                grid=grid,
                block=block,
                dynamic_shared_memory_bytes=dynamic_shared_memory_bytes,
                input_slots=inputs,
                output_slots=outputs,
                input_accesses=(Access.READ,) * len(inputs),
                output_accesses=(Access.WRITE,) * len(outputs),
                dependencies=dependencies,
                predication=predication,
                reduction_order=reduction_order,
            )
        )

    _require(expected_offset == len(code) and len(producers) == LAUNCH_COUNT)
    return tuple(launches)


def _contiguous_strides(element_bytes: int, dimensions: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(dimensions)
    stride = element_bytes
    for axis in range(len(dimensions) - 1, -1, -1):
        strides[axis] = stride
        stride *= dimensions[axis]
    return tuple(strides)


def _decode_slots(
    reader: _Reader,
) -> tuple[tuple[GpuSlot, ...], tuple[GpuExternalBinding, ...], int, int]:
    _require(reader.u64() == SLOT_COUNT)
    slots: list[GpuSlot] = []
    bindings: list[GpuExternalBinding] = []
    temporary_bytes = 0
    scalar_slots = 0

    for ordinal in range(SLOT_COUNT):
        slot_ordinal = reader.i64()
        source_buffer = reader.i64()
        element_type = _enum_member((ElementType.BF16, ElementType.F32), reader.u8())
        rank = reader.u64()
        _require(slot_ordinal == ordinal and source_buffer == ordinal and rank <= MAXIMUM_RANK)

        element_bytes: int = 2 if element_type is ElementType.BF16 else 4
        dimensions: list[int] = []
        required_bytes: int = element_bytes
        for _ in range(rank):
            extent = reader.i64()
            _require(extent > 0 and required_bytes <= MAXIMUM_I64 // extent)
            dimensions.append(extent)
            required_bytes *= extent
        shape = tuple(dimensions)
        if not shape:
            scalar_slots += 1

        encoded_required_bytes = reader.i64()
        byte_strides = reader.integers()
        byte_offset = reader.i64()
        alignment = reader.i64()
        address_space = reader.u8()
        access = _enum_member((Access.READ, Access.WRITE, Access.READ_WRITE), reader.u8())
        storage = _enum_member((Storage.EXTERNAL, Storage.TEMPORARY), reader.u8())
        alias_group = reader.i64()
        reuse_group = reader.i64()
        binding_kind = _enum_member((BindingKind.NONE, BindingKind.OPERAND, BindingKind.RESULT), reader.u8())
        has_binding_index = reader.u8()
        _require(has_binding_index <= 1)
        binding_index = reader.i64() if has_binding_index else None

        _require(
            encoded_required_bytes == required_bytes
            and required_bytes <= MAXIMUM_SLOT_BYTES
            and byte_strides == _contiguous_strides(element_bytes, shape)
            and byte_offset == 0
            and alignment == element_bytes
            and address_space == 1
            and alias_group == ordinal
            and reuse_group == ordinal
        )

        if storage is Storage.TEMPORARY:
            _require(
                binding_kind is BindingKind.NONE
                and binding_index is None
                and access is Access.READ_WRITE
                and required_bytes <= MAXIMUM_TEMPORARY_BYTES - temporary_bytes
            )
            temporary_bytes += required_bytes
        else:
            _require(
                binding_index is not None
                and binding_kind is not BindingKind.NONE
                and (binding_kind is not BindingKind.OPERAND or access is Access.READ)
                and (binding_kind is not BindingKind.RESULT or access is Access.WRITE)
            )
            bindings.append(
                GpuExternalBinding(
                    kind=binding_kind,
                    index=binding_index,
                    slot_ordinal=ordinal,
                    element_type=element_type,
                    dimensions=shape,
                    required_bytes=required_bytes,
                    alignment=alignment,
                )
            )

        slots.append(
            GpuSlot(
                ordinal=ordinal,
                source_buffer=source_buffer,
                element_type=element_type,
                dimensions=shape,
                required_bytes=required_bytes,
                byte_strides=byte_strides,
                byte_offset=byte_offset,
                alignment=alignment,
                address_space=AddressSpace.DEVICE,
                access=access,
                storage=storage,
                alias_group=alias_group,
                reuse_group=reuse_group,
                binding_kind=binding_kind,
                binding_index=binding_index,
            )
        )

    return tuple(slots), tuple(bindings), temporary_bytes, scalar_slots


def _require_forward_bindings(bindings: tuple[GpuExternalBinding, ...]) -> None:
    expected = (
        (BindingKind.OPERAND, 0, 0, (2048, 4096), 2048 * 4096 * 2),
        (BindingKind.OPERAND, 1, 1, (4096,), 4096 * 2),
        (BindingKind.RESULT, 0, 20, (2048, 4096), 2048 * 4096 * 2),
    )
    _require(len(bindings) == len(expected))
    for binding, (kind, index, slot, shape, required_bytes) in zip(bindings, expected, strict=True):
        _require(
            binding.kind is kind
            and binding.index == index
            and binding.slot_ordinal == slot
            and binding.element_type is ElementType.BF16
            and binding.dimensions == shape
            and binding.required_bytes == required_bytes
            and binding.alignment == 2
        )


def _parse_gpu_transport(data: bytes) -> GpuTransport:
    _require(len(data) <= MAXIMUM_TRANSPORT_BYTES)
    reader = _Reader(data)
    _require(reader.take(len(MAGIC)) == MAGIC and reader.u32() == TRANSPORT_SCHEMA_VERSION)

    device_start = reader.position
    device_schema_version = reader.i64()
    code_format = reader.u8()
    numerical_policy = reader.u8()
    schedule_fingerprint = _digest_text(reader.text())
    code = reader.blob()
    code_digest = _digest_text(reader.text())
    device_root_offset = reader.position
    device_root = _digest_text(reader.text())
    device_suffix_start = reader.position
    _require(
        device_schema_version == DEVICE_SCHEMA_VERSION
        and code_format == 2
        and numerical_policy == 0
        and 0 < len(code) <= MAXIMUM_CODE_BYTES
        and code_digest == _sha256(code)
    )
    launches = _decode_launches(reader, code)
    invocation_start = reader.position
    _require(device_root == _sha256(data[device_start:device_root_offset], data[device_suffix_start:invocation_start]))

    invocation_schema_version = reader.i64()
    plan_fingerprint = _digest_text(reader.text())
    invocation_schedule = _digest_text(reader.text())
    invocation_root_offset = reader.position
    invocation_root = _digest_text(reader.text())
    invocation_suffix_start = reader.position
    _require(invocation_schema_version == INVOCATION_ABI_SCHEMA_VERSION and invocation_schedule == schedule_fingerprint)
    slots, bindings, temporary_bytes, scalar_slots = _decode_slots(reader)
    bundle_start = reader.position
    _require(
        invocation_root
        == _sha256(
            data[invocation_start:invocation_root_offset],
            data[invocation_suffix_start:bundle_start],
        )
    )
    _require_forward_bindings(bindings)
    _require(
        temporary_bytes == TEMPORARY_BYTES and scalar_slots == 3 and len(slots) - len(bindings) == TEMPORARY_SLOT_COUNT
    )

    bundle_schema_version = reader.i64()
    bundle_schedule = _digest_text(reader.text())
    bundle_device_root = _digest_text(reader.text())
    bundle_invocation_root = _digest_text(reader.text())
    completion = reader.u8()
    bundle_root_offset = reader.position
    bundle_root = _digest_text(reader.text())
    _require(
        bundle_schema_version == BUNDLE_SCHEMA_VERSION
        and bundle_schedule == schedule_fingerprint
        and bundle_device_root == device_root
        and bundle_invocation_root == invocation_root
        and completion == 1
        and bundle_root == _sha256(data[:bundle_root_offset])
        and reader.position == len(data)
    )

    return GpuTransport(
        transport_schema_version=TRANSPORT_SCHEMA_VERSION,
        device_schema_version=device_schema_version,
        invocation_abi_schema_version=invocation_schema_version,
        bundle_schema_version=bundle_schema_version,
        code_format=CodeFormat.CUDA_PTX_SM90_V1,
        numerical_policy=NumericalPolicy.SOURCE_ORDERED,
        completion=Completion.STREAM_ORDERED,
        schedule_fingerprint=schedule_fingerprint,
        plan_fingerprint=plan_fingerprint,
        transport_root=_sha256(data),
        device_root=device_root,
        invocation_root=invocation_root,
        bundle_root=bundle_root,
        transport_size=len(data),
        code_digest=code_digest,
        code=code,
        launches=launches,
        slots=slots,
        external_bindings=bindings,
        temporary_bytes=temporary_bytes,
    )


def load_gpu_transport(path: Path) -> GpuTransport:
    """Load and validate one closed schema-2 ABI 10 GPU transport file.

    Args:
        path: Transport file written by the GPU executable bundle serializer.

    Returns:
        Validated roots, slots, bindings, and source-ordered launch metadata.

    Raises:
        ValueError: The file is oversized, truncated, or noncanonical.
    """
    with path.open("rb") as transport_file:
        size = os.fstat(transport_file.fileno()).st_size
        if size > MAXIMUM_TRANSPORT_BYTES:
            raise ValueError(_INVALID)
        data = transport_file.read(MAXIMUM_TRANSPORT_BYTES + 1)
    if len(data) != size or len(data) > MAXIMUM_TRANSPORT_BYTES:
        raise ValueError(_INVALID)
    try:
        return _parse_gpu_transport(data)
    except _InvalidTransport:
        raise ValueError(_INVALID) from None
