# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the independent ABI 10 GPU transport decoder."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from gpu_transport import (
    MAXIMUM_TRANSPORT_BYTES,
    Access,
    AddressSpace,
    BindingKind,
    Completion,
    ElementType,
    Storage,
    load_gpu_transport,
)


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def unsigned(self, value: int, width: int) -> None:
        self.data.extend(value.to_bytes(width, "little", signed=False))

    def u8(self, value: int) -> None:
        self.unsigned(value, 1)

    def u16(self, value: int) -> None:
        self.unsigned(value, 2)

    def u32(self, value: int) -> None:
        self.unsigned(value, 4)

    def u64(self, value: int) -> None:
        self.unsigned(value, 8)

    def i64(self, value: int) -> None:
        self.data.extend(value.to_bytes(8, "little", signed=True))

    def text(self, value: bytes) -> None:
        self.blob(value)

    def blob(self, value: bytes) -> None:
        self.u64(len(value))
        self.data.extend(value)

    def integers(self, values: tuple[int, ...]) -> None:
        self.u64(len(values))
        for value in values:
            self.i64(value)


@dataclass(frozen=True)
class _BuiltTransport:
    data: bytes
    ptx_slices: tuple[bytes, ...]
    device_root: str
    invocation_root: str
    bundle_root: str


def _sha256(*parts: bytes | bytearray) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _write_launch(
    writer: _Writer,
    ordinal: int,
    code_offset: int,
    ptx: bytes,
    *,
    excessive_inputs: bool,
    invalid_code_digest: bool,
    invalid_dependencies: bool,
    oversized_grid: bool,
) -> None:
    inputs = (0,) * 257 if excessive_inputs and ordinal == 0 else ((0, 1) if ordinal == 0 else (ordinal + 1,))
    outputs = (ordinal + 2,)
    dependencies = () if ordinal == 0 else (ordinal - 1,)
    if invalid_dependencies and ordinal == 1:
        dependencies = ()
    writer.i64(ordinal)
    writer.i64(ordinal)
    writer.i64(code_offset)
    writer.i64(len(ptx))
    writer.integers(inputs)
    writer.integers(outputs)
    writer.u64(len(inputs))
    for _ in inputs:
        writer.u8(0)
    writer.u64(len(outputs))
    for _ in outputs:
        writer.u8(1)
    writer.integers(dependencies)
    writer.u8(0)
    writer.u8(1 if ordinal == 9 else 0)
    if ordinal == 9:
        writer.u8(0)
    entry_digest = _sha256(b"wrong ptx") if invalid_code_digest and ordinal == 0 else _sha256(ptx)
    writer.text(entry_digest.encode())
    grid = (1 << 31, 1 << 31, 4) if oversized_grid and ordinal == 0 else (ordinal + 1, 1, 1)
    for extent in grid:
        writer.u32(extent)
    for extent in (32, 1, 1):
        writer.u16(extent)
    writer.u32(1024 if ordinal == 9 else 0)
    writer.u16(len(inputs) + len(outputs))


def _strides(element_bytes: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    values = [0] * len(shape)
    stride = element_bytes
    for axis in range(len(shape) - 1, -1, -1):
        values[axis] = stride
        stride *= shape[axis]
    return tuple(values)


def _write_slot(writer: _Writer, ordinal: int, *, invalid_binding: bool) -> None:
    if ordinal == 0:
        element, shape, access, storage, binding, binding_index = 0, (2048, 4096), 0, 0, 1, 0
    elif ordinal == 1:
        element, shape, access, storage, binding, binding_index = 0, (4096,), 0, 0, 1, 1
    elif 2 <= ordinal <= 4:
        element, shape, access, storage, binding, binding_index = 1, (), 2, 1, 0, None
    elif 5 <= ordinal <= 18:
        element, shape, access, storage, binding, binding_index = 0, (6_291_456,), 2, 1, 0, None
    elif ordinal == 19:
        element, shape, access, storage, binding, binding_index = 0, (12_627_968,), 2, 1, 0, None
    else:
        element, shape, access, storage, binding, binding_index = 0, (2048, 4096), 1, 0, 2, 0
    if invalid_binding and ordinal == 20:
        binding_index = 1

    element_bytes = 2 if element == 0 else 4
    required_bytes = element_bytes
    for extent in shape:
        required_bytes *= extent
    writer.i64(ordinal)
    writer.i64(ordinal)
    writer.u8(element)
    writer.u64(len(shape))
    for extent in shape:
        writer.i64(extent)
    writer.i64(required_bytes)
    writer.integers(_strides(element_bytes, shape))
    writer.i64(0)
    writer.i64(element_bytes)
    writer.u8(1)
    writer.u8(access)
    writer.u8(storage)
    writer.i64(ordinal)
    writer.i64(ordinal)
    writer.u8(binding)
    writer.u8(binding_index is not None)
    if binding_index is not None:
        writer.i64(binding_index)


def _build_transport(
    *,
    excessive_inputs: bool = False,
    invalid_binding: bool = False,
    invalid_code_digest: bool = False,
    invalid_dependencies: bool = False,
    oversized_grid: bool = False,
) -> _BuiltTransport:
    schedule = _sha256(b"source-ordered-schedule").encode()
    plan = _sha256(b"forward-materialization-plan").encode()
    ptx_slices = tuple(
        f"// task {ordinal}\n.visible .entry shuttle_entry() {{ ret; }}\n".encode() for ordinal in range(19)
    )
    code = b"".join(ptx_slices)

    device_prefix = _Writer()
    device_prefix.i64(3)
    device_prefix.u8(2)
    device_prefix.u8(0)
    device_prefix.text(schedule)
    device_prefix.blob(code)
    device_prefix.text(_sha256(code).encode())
    device_suffix = _Writer()
    device_suffix.u64(19)
    offset = 0
    for ordinal, ptx in enumerate(ptx_slices):
        _write_launch(
            device_suffix,
            ordinal,
            offset,
            ptx,
            excessive_inputs=excessive_inputs,
            invalid_code_digest=invalid_code_digest,
            invalid_dependencies=invalid_dependencies,
            oversized_grid=oversized_grid,
        )
        offset += len(ptx)
    device_root = _sha256(device_prefix.data, device_suffix.data)

    invocation_prefix = _Writer()
    invocation_prefix.i64(3)
    invocation_prefix.text(plan)
    invocation_prefix.text(schedule)
    invocation_suffix = _Writer()
    invocation_suffix.u64(21)
    for ordinal in range(21):
        _write_slot(invocation_suffix, ordinal, invalid_binding=invalid_binding)
    invocation_root = _sha256(invocation_prefix.data, invocation_suffix.data)

    writer = _Writer()
    writer.data.extend(b"SHUTGPU\0")
    writer.u32(2)
    writer.data.extend(device_prefix.data)
    writer.text(device_root.encode())
    writer.data.extend(device_suffix.data)
    writer.data.extend(invocation_prefix.data)
    writer.text(invocation_root.encode())
    writer.data.extend(invocation_suffix.data)
    writer.i64(2)
    writer.text(schedule)
    writer.text(device_root.encode())
    writer.text(invocation_root.encode())
    writer.u8(1)
    bundle_root = _sha256(writer.data)
    writer.text(bundle_root.encode())
    return _BuiltTransport(bytes(writer.data), ptx_slices, device_root, invocation_root, bundle_root)


def _write_transport(path: Path, built: _BuiltTransport) -> None:
    path.write_bytes(built.data)


def test_load_gpu_transport_exposes_closed_forward_projection(tmp_path: Path) -> None:
    built = _build_transport()
    path = tmp_path / "forward.shutgpu"
    _write_transport(path, built)

    transport = load_gpu_transport(path)

    assert transport.transport_root == _sha256(built.data)
    assert transport.device_root == built.device_root
    assert transport.invocation_root == built.invocation_root
    assert transport.bundle_root == built.bundle_root
    assert transport.completion is Completion.STREAM_ORDERED
    assert transport.numerical_policy == "source_ordered"
    assert transport.temporary_bytes == 201_416_716
    assert [launch.ordinal for launch in transport.launches] == list(range(19))
    assert [launch.code for launch in transport.launches] == list(built.ptx_slices)
    assert transport.launches[0].input_slots == (0, 1)
    assert transport.launches[0].input_accesses == (Access.READ, Access.READ)
    assert transport.launches[-1].output_slots == (20,)
    assert transport.launches[-1].dependencies == (17,)
    assert [binding.kind for binding in transport.external_bindings] == [
        BindingKind.OPERAND,
        BindingKind.OPERAND,
        BindingKind.RESULT,
    ]
    assert [binding.slot_ordinal for binding in transport.external_bindings] == [0, 1, 20]
    assert transport.slots[0].element_type is ElementType.BF16
    assert transport.slots[0].dimensions == (2048, 4096)
    assert transport.slots[2].storage is Storage.TEMPORARY
    assert transport.slots[2].address_space is AddressSpace.DEVICE
    assert transport.slots[2].access is Access.READ_WRITE


@pytest.mark.parametrize(
    "mutation",
    [
        {"excessive_inputs": True},
        {"invalid_binding": True},
        {"invalid_code_digest": True},
        {"invalid_dependencies": True},
        {"oversized_grid": True},
    ],
)
def test_load_gpu_transport_rejects_bounded_or_noncanonical_projection(
    tmp_path: Path, mutation: dict[str, bool]
) -> None:
    path = tmp_path / "invalid.shutgpu"
    _write_transport(path, _build_transport(**mutation))

    with pytest.raises(ValueError):
        load_gpu_transport(path)


def test_load_gpu_transport_rejects_corrupt_closed_root(tmp_path: Path) -> None:
    built = _build_transport()
    corrupt = bytearray(built.data)
    corrupt[-1] = ord("0") if corrupt[-1] != ord("0") else ord("1")
    path = tmp_path / "corrupt.shutgpu"
    path.write_bytes(corrupt)

    with pytest.raises(ValueError):
        load_gpu_transport(path)


def test_load_gpu_transport_rejects_file_above_transport_bound(tmp_path: Path) -> None:
    path = tmp_path / "oversized.shutgpu"
    with path.open("wb") as transport_file:
        transport_file.truncate(MAXIMUM_TRANSPORT_BYTES + 1)

    with pytest.raises(ValueError):
        load_gpu_transport(path)
