# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the independent ABI15 H100 transport decoder."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from abi15_h100_gpu_transport import (
    CallRole,
    NumericalPolicy,
    TransportCoordinate,
    extract_gpu_transports,
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

    def blob(self, value: bytes) -> None:
        self.u64(len(value))
        self.data.extend(value)

    def text(self, value: str) -> None:
        self.blob(value.encode())

    def integers(self, values: tuple[int, ...]) -> None:
        self.u64(len(values))
        for value in values:
            self.i64(value)


@dataclass(frozen=True)
class _BuiltTransport:
    data: bytes
    roots: tuple[str, str, str]


def _sha256(*parts: bytes | bytearray) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _contract_ptx(policy: NumericalPolicy, maps: tuple[tuple[int, ...], tuple[int, ...]]) -> bytes:
    reduction = 2 if maps == ((0, 2), (2, 1)) else (33 if maps == ((2, 0), (2, 1)) else 34)
    lane = "  cvt.u64.u32 %rd4, %r2;\n" if policy is NumericalPolicy.FAST else "  mov.u64 %rd4, 0;\n"
    step = 32 if policy is NumericalPolicy.FAST else 1
    shuffle = ""
    if policy is NumericalPolicy.FAST:
        shuffle = "".join(
            f"  shfl.sync.down.b32 %r6, %r5, {offset}, 31, 0xffffffff;\n" "  add.rn.f32 %f0, %f0, %f3;\n"
            for offset in (16, 8, 4, 2, 1)
        )
    output_shape = (33, 34) if reduction == 2 else ((34, 2) if reduction == 33 else (33, 2))
    input_shapes = (
        ((33, 2), (2, 34)) if reduction == 2 else (((33, 34), (33, 2)) if reduction == 33 else ((33, 34), (2, 34)))
    )
    lines = [
        ".version 8.0",
        ".target sm_90",
        ".address_size 64",
        ".visible .entry shuttle_entry()",
        "{",
        f"  div.u64 %rd1, %rd0, {output_shape[1]};",
        f"  rem.u64 %rd1, %rd1, {output_shape[0]};",
        "  div.u64 %rd2, %rd0, 1;",
        f"  rem.u64 %rd2, %rd2, {output_shape[1]};",
        lane.rstrip(),
    ]
    coordinate_registers = {0: "%rd1", 1: "%rd2", 2: "%rd4"}
    next_linear = 10
    next_pointer = 20
    for parameter, (shape, affine_map) in enumerate(zip(input_shapes, maps, strict=True)):
        lines.append(f"  mov.u64 %rd{next_linear}, 0;")
        for extent, dimension in zip(shape, affine_map, strict=True):
            lines.append(
                f"  mad.lo.u64 %rd{next_linear}, %rd{next_linear}, {extent}, {coordinate_registers[dimension]};"
            )
        lines.extend(
            [
                f"  ld.param.u64 %rd{next_pointer}, [p{parameter}];",
                f"  mad.lo.u64 %rd{next_pointer}, %rd{next_linear}, 2, %rd{next_pointer};",
            ]
        )
        next_linear += 1
        next_pointer += 1
    lines.extend(
        [
            "  fma.rn.f32 %f0, %f1, %f2, %f0;",
            f"  add.u64 %rd4, %rd4, {step};",
            shuffle.rstrip(),
            f"  setp.ge.u64 %p1, %rd4, {reduction};",
            "  ret;",
            "}",
        ]
    )
    return ("\n".join(line for line in lines if line) + "\n").encode()


def _map_ptx(
    input_shape: tuple[int, int] | None = None,
    affine_map: tuple[int, int] | None = None,
) -> bytes:
    lines = [
        ".version 8.0",
        ".target sm_90",
        ".address_size 64",
        ".visible .entry shuttle_entry()",
        "{",
        "  div.u64 %rd1, %rd0, 34;",
        "  rem.u64 %rd1, %rd1, 2;",
        "  div.u64 %rd2, %rd0, 1;",
        "  rem.u64 %rd2, %rd2, 34;",
    ]
    if input_shape is not None and affine_map is not None:
        registers = {0: "%rd1", 1: "%rd2"}
        lines.append("  mov.u64 %rd10, 0;")
        for extent, dimension in zip(input_shape, affine_map, strict=True):
            lines.append(f"  mad.lo.u64 %rd10, %rd10, {extent}, {registers[dimension]};")
        lines.extend(["  ld.param.u64 %rd20, [p0];", "  mad.lo.u64 %rd20, %rd10, 4, %rd20;"])
    lines.extend(["  cvt.rn.bf16.f32 %h1, %f0;", "  ret;", "}"])
    return ("\n".join(lines) + "\n").encode()


def _ptx_slices(coordinate: TransportCoordinate) -> tuple[bytes, ...]:
    if coordinate.call_role is CallRole.FORWARD:
        return (_contract_ptx(coordinate.policy, ((0, 2), (2, 1))), _map_ptx())
    slices = [_map_ptx() for _ in range(10)]
    slices[3] = _contract_ptx(coordinate.policy, ((2, 0), (2, 1)))
    slices[4] = _map_ptx((34, 2), (1, 0))
    slices[8] = _contract_ptx(coordinate.policy, ((0, 2), (1, 2)))
    return tuple(slices)


def _bindings(
    coordinate: TransportCoordinate,
) -> tuple[tuple[int, tuple[int, ...], int, int, int | None], ...]:
    if coordinate.call_role is CallRole.FORWARD:
        return (
            (0, (33, 2), 0, 1, 0),
            (0, (2, 34), 0, 1, 1),
            (1, (33, 34), 2, 0, None),
            (0, (33, 34), 1, 2, 0),
        )
    values = [
        (0, (33, 34), 0, 1, 0),
        (0, (33, 2), 0, 1, 1),
        (0, (2, 34), 0, 1, 2),
    ]
    values.extend((1, (1,), 2, 0, None) for _ in range(8))
    values.extend(((0, (33, 2), 1, 2, 0), (0, (2, 34), 1, 2, 1)))
    return tuple(values)


def _write_slot(
    writer: _Writer,
    ordinal: int,
    spec: tuple[int, tuple[int, ...], int, int, int | None],
) -> None:
    element, shape, access, binding, binding_index = spec
    storage = 1 if binding == 0 else 0
    element_bytes = 2 if element == 0 else 4
    required = element_bytes
    for extent in shape:
        required *= extent
    strides = []
    stride = element_bytes
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    writer.i64(ordinal)
    writer.i64(ordinal)
    writer.u8(element)
    writer.u64(len(shape))
    for extent in shape:
        writer.i64(extent)
    writer.i64(required)
    writer.integers(tuple(reversed(strides)))
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


def _build_transport(coordinate: TransportCoordinate, *, affine_mutation: bool = False) -> _BuiltTransport:
    policy_byte = 0 if coordinate.policy is NumericalPolicy.SOURCE_ORDERED else 1
    schedule = _sha256(f"{coordinate.policy}:{coordinate.call_role}".encode())
    plan = _sha256(f"plan:{coordinate.call_role}".encode())
    ptx_slices = _ptx_slices(coordinate)
    if affine_mutation:
        mutated = ptx_slices[3].replace(
            b"mad.lo.u64 %rd10, %rd10, 34, %rd1",
            b"mad.lo.u64 %rd10, %rd10, 34, %rd2",
        )
        assert len(mutated) == len(ptx_slices[3]) and mutated != ptx_slices[3]
        ptx_slices = (*ptx_slices[:3], mutated, *ptx_slices[4:])
    code = b"".join(ptx_slices)
    slot_specs = _bindings(coordinate)

    device_prefix = _Writer()
    device_prefix.i64(3)
    device_prefix.u8(2)
    device_prefix.u8(policy_byte)
    device_prefix.text(schedule)
    device_prefix.blob(code)
    device_prefix.text(_sha256(code))
    device_suffix = _Writer()
    device_suffix.u64(len(ptx_slices))
    offset = 0
    producers: dict[int, int] = {}
    for ordinal, ptx in enumerate(ptx_slices):
        inputs = (0, 1) if ordinal == 0 else (min(ordinal + 1, len(slot_specs) - 1),)
        output = min(ordinal + 2, len(slot_specs) - 1)
        dependencies = tuple(sorted({producers[value] for value in inputs if value in producers}))
        device_suffix.i64(ordinal)
        device_suffix.i64(ordinal)
        device_suffix.i64(offset)
        device_suffix.i64(len(ptx))
        device_suffix.integers(inputs)
        device_suffix.integers((output,))
        device_suffix.u64(len(inputs))
        for _ in inputs:
            device_suffix.u8(0)
        device_suffix.u64(1)
        device_suffix.u8(1)
        device_suffix.integers(dependencies)
        device_suffix.u8(0)
        contract = ordinal in ({0} if coordinate.call_role is CallRole.FORWARD else {3, 8})
        device_suffix.u8(contract)
        if contract:
            device_suffix.u8(0 if coordinate.policy is NumericalPolicy.SOURCE_ORDERED else 1)
        device_suffix.text(_sha256(ptx))
        for extent in (1, 1, 1):
            device_suffix.u32(extent)
        for extent in (32, 1, 1):
            device_suffix.u16(extent)
        device_suffix.u32(0)
        device_suffix.u16(len(inputs) + 1)
        producers[output] = ordinal
        offset += len(ptx)
    device_root = _sha256(device_prefix.data, device_suffix.data)

    invocation_prefix = _Writer()
    invocation_prefix.i64(3)
    invocation_prefix.text(plan)
    invocation_prefix.text(schedule)
    invocation_suffix = _Writer()
    invocation_suffix.u64(len(slot_specs))
    for ordinal, spec in enumerate(slot_specs):
        _write_slot(invocation_suffix, ordinal, spec)
    invocation_root = _sha256(invocation_prefix.data, invocation_suffix.data)

    writer = _Writer()
    writer.data.extend(b"SHUTGPU\0")
    writer.u32(2)
    writer.data.extend(device_prefix.data)
    writer.text(device_root)
    writer.data.extend(device_suffix.data)
    writer.data.extend(invocation_prefix.data)
    writer.text(invocation_root)
    writer.data.extend(invocation_suffix.data)
    writer.i64(2)
    writer.text(schedule)
    writer.text(device_root)
    writer.text(invocation_root)
    writer.u8(1)
    bundle_root = _sha256(writer.data)
    writer.text(bundle_root)
    return _BuiltTransport(bytes(writer.data), (device_root, invocation_root, bundle_root))


@pytest.mark.parametrize("policy", list(NumericalPolicy))
def test_extract_gpu_transports_preserves_two_call_source_order(
    policy: NumericalPolicy,
) -> None:
    forward = _build_transport(TransportCoordinate(policy, CallRole.FORWARD)).data
    backward = _build_transport(TransportCoordinate(policy, CallRole.BACKWARD)).data

    def escaped(data: bytes) -> str:
        return "".join(
            (chr(value) if 32 <= value < 127 and value not in (34, 92) else f"\\{value:02X}") for value in data
        )

    target = "shuttle.gpu.executable_bundle.v2"
    generated = (
        '%0 = "stablehlo.custom_call"() <{api_version = 4 : i32, '
        f'bundle_bytes = "{escaped(forward)}", call_target_name = "{target}"}}> : '
        "() -> tensor<1xi1>\n"
        '%1:2 = "stablehlo.custom_call"() <{api_version = 4 : i32, '
        f'bundle_bytes = "{escaped(backward)}", call_target_name = "{target}"}}> : '
        "() -> tuple<tensor<1xi1>, tensor<1xi1>>"
    )

    assert extract_gpu_transports(generated) == (forward, backward)


@pytest.mark.parametrize("policy", list(NumericalPolicy))
@pytest.mark.parametrize("role", list(CallRole))
def test_load_gpu_transport_validates_policy_topology_and_affine_realization(
    tmp_path: Path, policy: NumericalPolicy, role: CallRole
) -> None:
    coordinate = TransportCoordinate(policy, role)
    built = _build_transport(coordinate)
    path = tmp_path / "bundle.bin"
    path.write_bytes(built.data)

    decoded = load_gpu_transport(path, coordinate)

    expected_launches = 2 if role is CallRole.FORWARD else 10
    expected_slots = 4 if role is CallRole.FORWARD else 13
    assert len(decoded.launches) == expected_launches
    assert len(decoded.slots) == expected_slots
    assert len(decoded.external_bindings) == (3 if role is CallRole.FORWARD else 5)
    assert (
        decoded.device_root,
        decoded.invocation_root,
        decoded.bundle_root,
    ) == built.roots
    assert decoded.coordinate == coordinate


@pytest.mark.parametrize("policy", list(NumericalPolicy))
def test_load_gpu_transport_rejects_policy_substitution_with_rehashed_roots(
    tmp_path: Path, policy: NumericalPolicy
) -> None:
    actual = NumericalPolicy.FAST if policy is NumericalPolicy.SOURCE_ORDERED else NumericalPolicy.SOURCE_ORDERED
    path = tmp_path / "bundle.bin"
    path.write_bytes(_build_transport(TransportCoordinate(actual, CallRole.FORWARD)).data)

    with pytest.raises(ValueError):
        load_gpu_transport(path, TransportCoordinate(policy, CallRole.FORWARD))


def test_load_gpu_transport_rejects_affine_map_mutation_with_valid_transport_roots(
    tmp_path: Path,
) -> None:
    coordinate = TransportCoordinate(NumericalPolicy.FAST, CallRole.BACKWARD)
    path = tmp_path / "bundle.bin"
    path.write_bytes(_build_transport(coordinate, affine_mutation=True).data)

    with pytest.raises(ValueError):
        load_gpu_transport(path, coordinate)


def test_extract_gpu_transports_rejects_extra_shuttle_call() -> None:
    operation = (
        '%0 = "stablehlo.custom_call"() <{api_version = 4 : i32, bundle_bytes = "SHUTGPU\\00", '
        'call_target_name = "shuttle.gpu.executable_bundle.v2"}> : () -> tensor<1xi1>'
    )

    with pytest.raises(ValueError):
        extract_gpu_transports("\n".join((operation, operation, operation)))
