# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical transport for graph-derived CuTe pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tile_lifetime.cute_device_ir import CuteDeviceIrSnapshot
from tile_lifetime.event_buffering import EventRealizationKind, erased_event_realization, verify_event_realizations
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventMemoryScope,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventDeviceLegalizationRegion,
    internal_event_device_legalization_region_record,
    internal_event_realization_records,
)

_MAXIMUM_KERNEL_ARGUMENT_BYTES = 4096
_MAXIMUM_STRUCTURED_FIELDS = 256
_MAXIMUM_ARGUMENT_ALIGNMENT = 256
_POINTER_BYTES = 8
_MAXIMUM_AFFINE_RANK = 16
_MAXIMUM_AFFINE_RESULTS = 32
_MAXIMUM_AFFINE_DEPTH = 32
_MAXIMUM_AFFINE_NODES = 4096
_MAXIMUM_EVENT_MECHANISM_CAPACITY = 256 * 1024 * 1024
_MAXIMUM_EVENT_TRANSACTION_BYTES = 32 * 1024 * 1024
_MAXIMUM_EVENT_ARRIVAL_COUNT = 2**32 - 1


class CutePipelineElementType(StrEnum):
    """Element or immediate types understood by the native launch ABI."""

    PRED = "pred"
    U8 = "u8"
    I8 = "i8"
    U16 = "u16"
    I16 = "i16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"


class CutePipelineSlotKind(StrEnum):
    """Storage ownership at the public/native pipeline boundary."""

    PUBLIC_INPUT = "public_input"
    PUBLIC_OUTPUT = "public_output"
    PUBLIC_INPUT_OUTPUT = "public_input_output"
    SCRATCH = "scratch"


class CutePipelineSlotAccess(StrEnum):
    """Access granted to one kernel argument referencing a storage slot."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class CuteTensorMapInterleave(StrEnum):
    """Interleave applied while constructing a CUDA tensor map."""

    NONE = "none"
    BYTES_16 = "bytes_16"
    BYTES_32 = "bytes_32"


class CuteTensorMapSwizzle(StrEnum):
    """Shared-memory swizzle applied by tensor-map transfers."""

    NONE = "none"
    BYTES_32 = "bytes_32"
    BYTES_64 = "bytes_64"
    BYTES_128 = "bytes_128"


class CuteTensorMapL2Promotion(StrEnum):
    """L2 promotion window requested by a CUDA tensor map."""

    NONE = "none"
    BYTES_64 = "bytes_64"
    BYTES_128 = "bytes_128"
    BYTES_256 = "bytes_256"


class CuteTensorMapOobFill(StrEnum):
    """Out-of-bounds fill behavior requested by a CUDA tensor map."""

    NONE = "none"
    NAN_REQUEST_ZERO_FMA = "nan_request_zero_fma"


class CuteKernelFunctionAttributeKind(StrEnum):
    """CUDA function attributes required before a launch."""

    MAX_DYNAMIC_SHARED_MEMORY_BYTES = "max_dynamic_shared_memory_bytes"
    PREFERRED_SHARED_MEMORY_CARVEOUT = "preferred_shared_memory_carveout"
    NON_PORTABLE_CLUSTER_SIZE_ALLOWED = "non_portable_cluster_size_allowed"


class CuteSourceTaskRole(StrEnum):
    """Structural source-task role implemented by one physical launch."""

    STATE_ROOT = "state_root"
    STATE_CARRY = "state_carry"
    STATE_TERMINAL = "state_terminal"


class CuteEventRealizationKind(StrEnum):
    """Physical realization of one exact logical EventTensor."""

    PROGRAM_ORDER = "program_order"
    STREAM_ORDER = "stream_order"
    ALGEBRAIC_REFINEMENT = "algebraic_refinement"
    MBARRIER = "mbarrier"
    PIPELINE = "pipeline"
    QUEUE = "queue"


class CuteEventPhysicalScope(StrEnum):
    """Visibility scope of one allocated physical Event mechanism."""

    THREAD_BLOCK = "thread_block"
    CLUSTER = "cluster"
    DEVICE = "device"
    SYSTEM = "system"


class CuteAffineExpressionKind(StrEnum):
    """Symbol-free affine operations supported by native Event covers."""

    DIMENSION = "dimension"
    CONSTANT = "constant"
    ADD = "add"
    MULTIPLY = "multiply"
    MODULO = "modulo"
    FLOOR_DIVIDE = "floor_divide"
    CEIL_DIVIDE = "ceil_divide"


@dataclass(frozen=True)
class CuteAffineExpression:
    """One bounded symbol-free affine expression."""

    kind: CuteAffineExpressionKind
    payload: int = 0
    operands: tuple[CuteAffineExpression, ...] = ()

    def __post_init__(self) -> None:
        _require_enum(self.kind, CuteAffineExpressionKind, "affine-expression kind")
        if not isinstance(self.payload, int) or isinstance(self.payload, bool):
            raise ValueError("affine-expression payloads must be integers")
        if any(not isinstance(operand, CuteAffineExpression) for operand in self.operands):
            raise ValueError("affine-expression operands must be affine expressions")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "operands": tuple(operand.canonical_record for operand in self.operands),
            "payload": self.payload,
        }


@dataclass(frozen=True)
class CuteAffineMap:
    """A symbol-free affine map over one Event relation domain."""

    dimensions: int
    results: tuple[CuteAffineExpression, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool):
            raise ValueError("affine-map rank must be an integer")
        if not 0 <= self.dimensions <= _MAXIMUM_AFFINE_RANK:
            raise ValueError("affine-map rank exceeds the native Event-cover bound")
        if not self.results or len(self.results) > _MAXIMUM_AFFINE_RESULTS:
            raise ValueError("affine maps require a bounded nonempty result tuple")
        nodes = [0]
        for result in self.results:
            _validate_affine_expression(result, self.dimensions, 0, nodes)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "results": tuple(result.canonical_record for result in self.results),
        }


@dataclass(frozen=True)
class CuteIntegerSet:
    """A conjunction selecting one subset of an Event relation."""

    dimensions: int
    constraints: tuple[CuteAffineExpression, ...]
    equalities: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool):
            raise ValueError("integer-set rank must be an integer")
        if not 0 <= self.dimensions <= _MAXIMUM_AFFINE_RANK:
            raise ValueError("integer-set rank exceeds the native Event-cover bound")
        if not self.constraints or len(self.constraints) > _MAXIMUM_AFFINE_RESULTS:
            raise ValueError("integer sets require a bounded nonempty constraint tuple")
        if len(self.constraints) != len(self.equalities):
            raise ValueError("integer-set constraints and equality flags must align")
        if any(not isinstance(equality, bool) for equality in self.equalities):
            raise ValueError("integer-set equality flags must be booleans")
        nodes = [0]
        for constraint in self.constraints:
            _validate_affine_expression(constraint, self.dimensions, 0, nodes)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "constraints": tuple(constraint.canonical_record for constraint in self.constraints),
            "dimensions": self.dimensions,
            "equalities": self.equalities,
        }


def _validate_affine_expression(
    expression: CuteAffineExpression,
    dimensions: int,
    depth: int,
    nodes: list[int],
) -> None:
    nodes[0] += 1
    if nodes[0] > _MAXIMUM_AFFINE_NODES or depth > _MAXIMUM_AFFINE_DEPTH:
        raise ValueError("affine expression exceeds the native Event-cover bound")
    if expression.kind is CuteAffineExpressionKind.DIMENSION:
        if expression.operands or not 0 <= expression.payload < dimensions:
            raise ValueError("affine dimension is outside its relation domain")
        return
    if expression.kind is CuteAffineExpressionKind.CONSTANT:
        if expression.operands:
            raise ValueError("affine constants must be leaves")
        return
    if expression.payload != 0 or len(expression.operands) != 2:
        raise ValueError("affine binary operations require two operands and a zero payload")
    if expression.kind is CuteAffineExpressionKind.MULTIPLY and not any(
        operand.kind is CuteAffineExpressionKind.CONSTANT for operand in expression.operands
    ):
        raise ValueError("affine multiplication requires one constant operand")
    if expression.kind in {
        CuteAffineExpressionKind.MODULO,
        CuteAffineExpressionKind.FLOOR_DIVIDE,
        CuteAffineExpressionKind.CEIL_DIVIDE,
    } and (expression.operands[1].kind is not CuteAffineExpressionKind.CONSTANT or expression.operands[1].payload <= 0):
        raise ValueError("affine division and modulo require a positive constant divisor")
    for operand in expression.operands:
        _validate_affine_expression(operand, dimensions, depth + 1, nodes)


def affine_dimension(position: int) -> CuteAffineExpression:
    """Construct one affine relation-domain dimension."""
    return CuteAffineExpression(CuteAffineExpressionKind.DIMENSION, position)


def affine_constant(value: int) -> CuteAffineExpression:
    """Construct one affine integer constant."""
    return CuteAffineExpression(CuteAffineExpressionKind.CONSTANT, value)


@dataclass(frozen=True)
class CuteEventExecutionPoint:
    """One owner-local execution phase in a named physical launch."""

    launch: str
    phase: int

    def __post_init__(self) -> None:
        _validate_identifier(self.launch, "Event execution-point launch")
        if not isinstance(self.phase, int) or isinstance(self.phase, bool) or self.phase < 0:
            raise ValueError("Event execution-point phase must be a nonnegative integer")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {"launch": self.launch, "phase": self.phase}


@dataclass(frozen=True)
class CuteEventCoverEndpoint:
    """One physical launch owner and owner-local time map."""

    launch: int
    owner: CuteAffineMap
    local_time: CuteAffineMap

    def __post_init__(self) -> None:
        if not isinstance(self.launch, int) or isinstance(self.launch, bool) or self.launch < 0:
            raise ValueError("Event-cover launch ordinals must be nonnegative integers")
        if self.owner.dimensions != self.local_time.dimensions:
            raise ValueError("Event-cover owner and local-time maps must share a relation domain")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "launch": self.launch,
            "local_time": self.local_time.canonical_record,
            "owner": self.owner.canonical_record,
        }


@dataclass(frozen=True)
class CuteEventCoverPiece:
    """One affine subset of a source Event relation and its physical cover."""

    relation_subset: CuteIntegerSet | None
    producer: CuteEventCoverEndpoint
    consumer: CuteEventCoverEndpoint

    def __post_init__(self) -> None:
        dimensions = self.producer.owner.dimensions
        if self.consumer.owner.dimensions != dimensions:
            raise ValueError("Event-cover endpoints must share a relation domain")
        if self.relation_subset is not None and self.relation_subset.dimensions != dimensions:
            raise ValueError("Event-cover predicates must use the endpoint relation domain")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer.canonical_record,
            "producer": self.producer.canonical_record,
            "relation_subset": self.relation_subset.canonical_record if self.relation_subset is not None else None,
        }


@dataclass(frozen=True)
class CuteEventMechanismAllocation:
    """Bounded storage and generation mapping for one physical Event mechanism."""

    allocation: int
    scope: CuteEventPhysicalScope
    capacity: int
    slot: CuteAffineMap
    generation_phase: CuteAffineMap
    generation_count: int
    arrival_count: int
    transaction_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.allocation, int) or isinstance(self.allocation, bool) or self.allocation < 0:
            raise ValueError("Event mechanism allocation ordinals must be nonnegative integers")
        _require_enum(self.scope, CuteEventPhysicalScope, "Event mechanism scope")
        if (
            not isinstance(self.capacity, int)
            or isinstance(self.capacity, bool)
            or not 0 < self.capacity <= _MAXIMUM_EVENT_MECHANISM_CAPACITY
        ):
            raise ValueError("Event mechanism capacity exceeds the native transport bound")
        if self.slot.dimensions != self.generation_phase.dimensions:
            raise ValueError("Event mechanism maps must share a relation domain")
        if len(self.slot.results) != 1:
            raise ValueError("Event mechanism slot maps must have one result")
        if len(self.generation_phase.results) > 2:
            raise ValueError("Event mechanism generation/phase maps may have at most two results")
        if (
            not isinstance(self.generation_count, int)
            or isinstance(self.generation_count, bool)
            or not 0 < self.generation_count <= _MAXIMUM_EVENT_MECHANISM_CAPACITY
        ):
            raise ValueError("Event mechanism generation count exceeds the native transport bound")
        if (
            not isinstance(self.arrival_count, int)
            or isinstance(self.arrival_count, bool)
            or not 0 < self.arrival_count <= _MAXIMUM_EVENT_ARRIVAL_COUNT
        ):
            raise ValueError("Event mechanism arrival count exceeds the native transport bound")
        if self.transaction_bytes is not None and (
            not isinstance(self.transaction_bytes, int)
            or isinstance(self.transaction_bytes, bool)
            or not 0 < self.transaction_bytes <= _MAXIMUM_EVENT_TRANSACTION_BYTES
        ):
            raise ValueError("Event mechanism transaction bytes exceed the native transport bound")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "allocation": self.allocation,
            "arrival_count": self.arrival_count,
            "capacity": self.capacity,
            "generation_count": self.generation_count,
            "generation_phase": self.generation_phase.canonical_record,
            "scope": self.scope.value,
            "slot": self.slot.canonical_record,
            "transaction_bytes": self.transaction_bytes,
        }


@dataclass(frozen=True)
class CuteStructuralEventCover:
    """Emitter-declared physical cover for one structural task-role edge."""

    producer_role: CuteSourceTaskRole
    consumer_role: CuteSourceTaskRole
    producer: CuteEventExecutionPoint
    consumer: CuteEventExecutionPoint
    kind: CuteEventRealizationKind | None = None
    relation_subset: CuteIntegerSet | None = None
    producer_owner: CuteAffineMap | None = None
    consumer_owner: CuteAffineMap | None = None
    producer_time: CuteAffineMap | None = None
    consumer_time: CuteAffineMap | None = None

    def __post_init__(self) -> None:
        _require_enum(self.producer_role, CuteSourceTaskRole, "Event-cover producer role")
        _require_enum(self.consumer_role, CuteSourceTaskRole, "Event-cover consumer role")
        if not isinstance(self.producer, CuteEventExecutionPoint) or not isinstance(
            self.consumer, CuteEventExecutionPoint
        ):
            raise ValueError("Event covers require producer and consumer execution points")
        if self.kind is not None:
            _require_enum(self.kind, CuteEventRealizationKind, "Event-cover realization kind")
            if self.kind not in {
                CuteEventRealizationKind.PROGRAM_ORDER,
                CuteEventRealizationKind.STREAM_ORDER,
                CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
            }:
                raise ValueError("structural Event covers can declare only erased realization kinds")
        maps = (self.producer_owner, self.consumer_owner, self.producer_time, self.consumer_time)
        if any(mapping is not None for mapping in maps) and any(mapping is None for mapping in maps):
            raise ValueError("structural Event covers must provide all owner/time maps together")
        if self.relation_subset is not None and self.producer_owner is None:
            raise ValueError("structural Event-cover predicates require explicit owner/time maps")
        if self.producer_owner is not None:
            dimensions = self.producer_owner.dimensions
            if any(mapping is None or mapping.dimensions != dimensions for mapping in maps):
                raise ValueError("structural Event-cover maps must share a relation domain")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer.canonical_record,
            "consumer_role": self.consumer_role.value,
            "kind": self.kind.value if self.kind is not None else None,
            "producer": self.producer.canonical_record,
            "producer_role": self.producer_role.value,
            "relation_subset": self.relation_subset.canonical_record if self.relation_subset is not None else None,
            "producer_owner": self.producer_owner.canonical_record if self.producer_owner is not None else None,
            "consumer_owner": self.consumer_owner.canonical_record if self.consumer_owner is not None else None,
            "producer_time": self.producer_time.canonical_record if self.producer_time is not None else None,
            "consumer_time": self.consumer_time.canonical_record if self.consumer_time is not None else None,
        }


@dataclass(frozen=True)
class CuteEventRealization:
    """Source-bound full-relation physical EventTensor realization."""

    event: int
    kind: CuteEventRealizationKind
    covering_launch_order: tuple[int, ...]
    cover_pieces: tuple[CuteEventCoverPiece, ...] = ()
    mechanism: CuteEventMechanismAllocation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, int) or isinstance(self.event, bool) or self.event < 0:
            raise ValueError("Event realization ordinal must be a nonnegative integer")
        _require_enum(self.kind, CuteEventRealizationKind, "Event realization kind")
        erased = self.kind in {
            CuteEventRealizationKind.PROGRAM_ORDER,
            CuteEventRealizationKind.STREAM_ORDER,
            CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
        }
        if erased and (not self.covering_launch_order or len(self.covering_launch_order) % 4):
            raise ValueError("erased Event realization witnesses must contain launch/phase groups of four")
        if not erased and self.covering_launch_order:
            raise ValueError("physical Event mechanisms cannot claim erased-realization witnesses")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.covering_launch_order
        ):
            raise ValueError("Event realization witnesses must be nonnegative integers")
        if not self.cover_pieces or any(not isinstance(piece, CuteEventCoverPiece) for piece in self.cover_pieces):
            raise ValueError("Event realizations require valid affine cover pieces")
        relation_dimensions = self.cover_pieces[0].producer.owner.dimensions
        if any(piece.producer.owner.dimensions != relation_dimensions for piece in self.cover_pieces):
            raise ValueError("Event realization cover pieces must share one relation domain")
        if erased and self.mechanism is not None:
            raise ValueError("erased Event realizations cannot allocate a physical mechanism")
        if not erased and not isinstance(self.mechanism, CuteEventMechanismAllocation):
            raise ValueError("non-erased Event realizations require a physical mechanism allocation")
        if self.mechanism is not None:
            if self.mechanism.slot.dimensions != relation_dimensions:
                raise ValueError("Event mechanism rank must match its exact relation cover")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "covering_launch_order": self.covering_launch_order,
            "event": self.event,
            "kind": self.kind.value,
            "cover_pieces": tuple(piece.canonical_record for piece in self.cover_pieces),
            "mechanism": self.mechanism.canonical_record if self.mechanism is not None else None,
        }


@dataclass(frozen=True)
class CutePipelineSlot:
    """One typed tensor allocation shared by launches in a native pipeline."""

    name: str
    element_type: CutePipelineElementType
    shape: tuple[int, ...]
    kind: CutePipelineSlotKind
    alignment: int
    input_index: int | None = None
    output_index: int | None = None
    reuse_group: int | None = None

    def __post_init__(self) -> None:
        _require_enum(self.element_type, CutePipelineElementType, f"pipeline slot {self.name!r} element type")
        _require_enum(self.kind, CutePipelineSlotKind, f"pipeline slot {self.name!r} kind")
        _validate_identifier(self.name, "pipeline slot")
        if any(not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0 for extent in self.shape):
            raise ValueError(f"pipeline slot {self.name!r} must have positive extents")
        _validate_alignment(self.alignment, f"pipeline slot {self.name!r}")
        has_input = self.kind in (CutePipelineSlotKind.PUBLIC_INPUT, CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT)
        has_output = self.kind in (CutePipelineSlotKind.PUBLIC_OUTPUT, CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT)
        _validate_public_index(self.input_index, required=has_input, label=f"pipeline slot {self.name!r} input")
        _validate_public_index(self.output_index, required=has_output, label=f"pipeline slot {self.name!r} output")
        if self.reuse_group is not None and (type(self.reuse_group) is not int or self.reuse_group < 0):
            raise ValueError(f"pipeline slot {self.name!r} reuse group must be a nonnegative integer")

    @property
    def required_bytes(self) -> int:
        elements = math.prod(self.shape)
        return elements * _element_type_bytes(self.element_type)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "element_type": self.element_type.value,
            "input_index": self.input_index,
            "kind": self.kind.value,
            "name": self.name,
            "output_index": self.output_index,
            "reuse_group": self.reuse_group,
            "shape": self.shape,
        }


@dataclass(frozen=True)
class CutePipelineSlotArgument:
    """One pointer argument into a declared pipeline slot."""

    name: str
    slot: str
    access: CutePipelineSlotAccess
    byte_offset: int = 0

    def __post_init__(self) -> None:
        _require_enum(self.access, CutePipelineSlotAccess, f"kernel argument {self.name!r} access")
        _validate_identifier(self.name, "kernel argument")
        _validate_identifier(self.slot, f"kernel argument {self.name!r} slot")
        if not isinstance(self.byte_offset, int) or isinstance(self.byte_offset, bool) or self.byte_offset < 0:
            raise ValueError(f"kernel argument {self.name!r} byte offset must be a nonnegative integer")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "access": self.access.value,
            "byte_offset": self.byte_offset,
            "kind": "slot",
            "name": self.name,
            "slot": self.slot,
        }


@dataclass(frozen=True)
class CutePipelineScalarArgument:
    """One typed scalar encoded directly in a kernel parameter buffer."""

    name: str
    element_type: CutePipelineElementType
    value: bool | int | float

    def __post_init__(self) -> None:
        _require_enum(self.element_type, CutePipelineElementType, f"kernel argument {self.name!r} element type")
        _validate_identifier(self.name, "kernel argument")
        _validate_scalar(self.element_type, self.value, name=self.name)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "element_type": self.element_type.value,
            "kind": "scalar",
            "name": self.name,
            "value": self.value,
        }


@dataclass(frozen=True)
class CutePipelinePackedBytesArgument:
    """One aligned, workload-independent by-value byte payload."""

    name: str
    data: bytes
    alignment: int

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "kernel argument")
        if not self.data:
            raise ValueError(f"packed kernel argument {self.name!r} must not be empty")
        _validate_alignment(self.alignment, f"packed kernel argument {self.name!r}")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "data_hex": self.data.hex(),
            "kind": "packed_bytes",
            "name": self.name,
        }


@dataclass(frozen=True)
class CutePipelineStructuredImmediateField:
    """Literal bytes copied into a structured kernel argument."""

    name: str
    destination_byte_offset: int
    data: bytes

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "structured argument field")
        _validate_destination_offset(self.destination_byte_offset, name=self.name)
        if not self.data or len(self.data) > _MAXIMUM_KERNEL_ARGUMENT_BYTES:
            raise ValueError(f"structured immediate field {self.name!r} must contain 1..4096 bytes")

    @property
    def width(self) -> int:
        return len(self.data)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "data_hex": self.data.hex(),
            "destination_byte_offset": self.destination_byte_offset,
            "kind": "immediate_bytes",
            "name": self.name,
        }


@dataclass(frozen=True)
class CutePipelineStructuredSlotAddressField:
    """A launch-time slot-address relocation in a structured argument."""

    name: str
    slot: str
    access: CutePipelineSlotAccess
    slot_byte_offset: int
    destination_byte_offset: int

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "structured argument field")
        _validate_identifier(self.slot, f"structured argument field {self.name!r} slot")
        _require_enum(self.access, CutePipelineSlotAccess, f"structured field {self.name!r} access")
        if (
            not isinstance(self.slot_byte_offset, int)
            or isinstance(self.slot_byte_offset, bool)
            or self.slot_byte_offset < 0
        ):
            raise ValueError(f"structured slot-address field {self.name!r} slot byte offset must be nonnegative")
        _validate_destination_offset(self.destination_byte_offset, name=self.name)
        if self.destination_byte_offset % _POINTER_BYTES:
            raise ValueError(f"structured slot-address field {self.name!r} destination must be pointer-aligned")

    @property
    def width(self) -> int:
        return _POINTER_BYTES

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "access": self.access.value,
            "destination_byte_offset": self.destination_byte_offset,
            "kind": "slot_address",
            "name": self.name,
            "slot": self.slot,
            "slot_byte_offset": self.slot_byte_offset,
        }


CutePipelineStructuredField = CutePipelineStructuredImmediateField | CutePipelineStructuredSlotAddressField


@dataclass(frozen=True)
class CutePipelineStructuredArgument:
    """A zero-initialized by-value argument patched from literals and slots."""

    name: str
    size: int
    alignment: int
    fields: tuple[CutePipelineStructuredField, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "kernel argument")
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or not 1 <= self.size <= _MAXIMUM_KERNEL_ARGUMENT_BYTES
        ):
            raise ValueError(f"structured kernel argument {self.name!r} size must be between one and 4096")
        _validate_alignment(self.alignment, f"structured kernel argument {self.name!r}")
        if self.alignment > _MAXIMUM_ARGUMENT_ALIGNMENT:
            raise ValueError(f"structured kernel argument {self.name!r} alignment must not exceed 256")
        if not self.fields or len(self.fields) > _MAXIMUM_STRUCTURED_FIELDS:
            raise ValueError(f"structured kernel argument {self.name!r} requires 1..256 fields")
        if any(
            not isinstance(field, (CutePipelineStructuredImmediateField, CutePipelineStructuredSlotAddressField))
            for field in self.fields
        ):
            raise ValueError(f"structured kernel argument {self.name!r} has an invalid field")
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ValueError(f"structured kernel argument {self.name!r} field names must be unique")
        if any(isinstance(field, CutePipelineStructuredSlotAddressField) for field in self.fields):
            if self.alignment < _POINTER_BYTES:
                raise ValueError(f"structured kernel argument {self.name!r} must provide pointer alignment")
        spans: list[tuple[int, int, str]] = []
        for field in self.fields:
            end = field.destination_byte_offset + field.width
            if end > self.size:
                raise ValueError(f"structured argument field {field.name!r} exceeds argument {self.name!r}")
            for begin, previous_end, previous_name in spans:
                if field.destination_byte_offset < previous_end and begin < end:
                    raise ValueError(f"structured argument fields {previous_name!r} and {field.name!r} overlap")
            spans.append((field.destination_byte_offset, end, field.name))

    @property
    def zero_filled_template(self) -> bytes:
        """Return literal fields with relocation locations and all gaps zeroed."""
        template = bytearray(self.size)
        for field in self.fields:
            if isinstance(field, CutePipelineStructuredImmediateField):
                begin = field.destination_byte_offset
                template[begin : begin + field.width] = field.data
        return bytes(template)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "fields": tuple(
                field.canonical_record for field in sorted(self.fields, key=lambda item: item.destination_byte_offset)
            ),
            "gap_fill": "zero",
            "kind": "structured_bytes",
            "name": self.name,
            "size": self.size,
        }


@dataclass(frozen=True)
class CutePipelineTensorMapArgument:
    """A tensor-map descriptor constructed at launch time from a bound slot."""

    name: str
    slot: str
    access: CutePipelineSlotAccess
    element_type: CutePipelineElementType
    global_shape: tuple[int, ...]
    global_byte_strides: tuple[int, ...]
    box_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    interleave: CuteTensorMapInterleave
    swizzle: CuteTensorMapSwizzle
    l2_promotion: CuteTensorMapL2Promotion
    oob_fill: CuteTensorMapOobFill
    byte_offset: int = 0
    alignment: int = 64

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "kernel argument")
        _validate_identifier(self.slot, f"kernel argument {self.name!r} slot")
        _require_enum(self.access, CutePipelineSlotAccess, f"tensor map {self.name!r} access")
        _require_enum(self.element_type, CutePipelineElementType, f"tensor map {self.name!r} element type")
        _require_enum(self.interleave, CuteTensorMapInterleave, f"tensor map {self.name!r} interleave")
        _require_enum(self.swizzle, CuteTensorMapSwizzle, f"tensor map {self.name!r} swizzle")
        _require_enum(self.l2_promotion, CuteTensorMapL2Promotion, f"tensor map {self.name!r} L2 promotion")
        _require_enum(self.oob_fill, CuteTensorMapOobFill, f"tensor map {self.name!r} out-of-bounds fill")
        if not isinstance(self.byte_offset, int) or isinstance(self.byte_offset, bool) or self.byte_offset < 0:
            raise ValueError(f"tensor map {self.name!r} byte offset must be a nonnegative integer")
        _validate_alignment(self.alignment, f"tensor map {self.name!r}")
        if not 64 <= self.alignment <= _MAXIMUM_ARGUMENT_ALIGNMENT:
            raise ValueError(f"tensor map {self.name!r} descriptor alignment must be between 64 and 256")
        _validate_tensor_map_dimensions(self)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "access": self.access.value,
            "alignment": self.alignment,
            "box_shape": self.box_shape,
            "byte_offset": self.byte_offset,
            "element_strides": self.element_strides,
            "element_type": self.element_type.value,
            "global_byte_strides": self.global_byte_strides,
            "global_shape": self.global_shape,
            "interleave": self.interleave.value,
            "kind": "tensor_map",
            "l2_promotion": self.l2_promotion.value,
            "name": self.name,
            "oob_fill": self.oob_fill.value,
            "slot": self.slot,
            "swizzle": self.swizzle.value,
        }


CutePipelineKernelArgument = (
    CutePipelineSlotArgument
    | CutePipelineScalarArgument
    | CutePipelinePackedBytesArgument
    | CutePipelineStructuredArgument
    | CutePipelineTensorMapArgument
)


def cute_kernel_argument_size(argument: CutePipelineKernelArgument) -> int:
    """Return the by-value byte width passed to the CUDA kernel parameter."""
    if isinstance(argument, CutePipelineSlotArgument):
        return _POINTER_BYTES
    if isinstance(argument, CutePipelineScalarArgument):
        return _element_type_bytes(argument.element_type)
    if isinstance(argument, CutePipelinePackedBytesArgument):
        return len(argument.data)
    if isinstance(argument, CutePipelineStructuredArgument):
        return argument.size
    if isinstance(argument, CutePipelineTensorMapArgument):
        return 128
    raise TypeError(f"unsupported CuTe kernel argument {type(argument).__name__}")


@dataclass(frozen=True)
class CuteKernelFunctionAttribute:
    """One integer-valued CUDA function attribute set before a launch."""

    kind: CuteKernelFunctionAttributeKind
    value: int

    def __post_init__(self) -> None:
        _require_enum(self.kind, CuteKernelFunctionAttributeKind, "CUDA function attribute kind")
        if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 0:
            raise ValueError(f"CUDA function attribute {self.kind.value!r} must be a nonnegative integer")
        if self.kind is CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT and self.value > 100:
            raise ValueError("preferred shared-memory carveout must be between 0 and 100")
        if self.kind is CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED and self.value not in (0, 1):
            raise ValueError("non-portable cluster-size permission must be zero or one")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "value": self.value}


@dataclass(frozen=True)
class CuteKernelLaunch:
    """One concrete kernel launch in pipeline execution order."""

    name: str
    kernel_symbol: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    function_attributes: tuple[CuteKernelFunctionAttribute, ...]
    arguments: tuple[CutePipelineKernelArgument, ...]
    source_task_roles: tuple[CuteSourceTaskRole, ...] = ()
    source_tasks: tuple[int, ...] = ()
    source_phase: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "kernel launch")
        if not self.kernel_symbol or "\0" in self.kernel_symbol:
            raise ValueError(f"kernel launch {self.name!r} must name a device symbol")
        _validate_launch_dimensions(self.grid, self.block, launch_name=self.name)
        if (
            not isinstance(self.dynamic_shared_memory_bytes, int)
            or isinstance(self.dynamic_shared_memory_bytes, bool)
            or self.dynamic_shared_memory_bytes < 0
        ):
            raise ValueError(f"kernel launch {self.name!r} dynamic shared memory must be a nonnegative integer")
        attribute_kinds = tuple(attribute.kind for attribute in self.function_attributes)
        if any(not isinstance(attribute, CuteKernelFunctionAttribute) for attribute in self.function_attributes):
            raise ValueError(f"kernel launch {self.name!r} has an invalid CUDA function attribute")
        if len(set(attribute_kinds)) != len(attribute_kinds):
            raise ValueError(f"kernel launch {self.name!r} repeats a CUDA function attribute")
        if any(
            not isinstance(
                argument,
                (
                    CutePipelineSlotArgument,
                    CutePipelineScalarArgument,
                    CutePipelinePackedBytesArgument,
                    CutePipelineStructuredArgument,
                    CutePipelineTensorMapArgument,
                ),
            )
            for argument in self.arguments
        ):
            raise ValueError(f"kernel launch {self.name!r} has an invalid kernel argument")
        argument_names = tuple(argument.name for argument in self.arguments)
        if len(set(argument_names)) != len(argument_names):
            raise ValueError(f"kernel launch {self.name!r} repeats a kernel argument name")
        if any(not isinstance(role, CuteSourceTaskRole) for role in self.source_task_roles) or len(
            set(self.source_task_roles)
        ) != len(self.source_task_roles):
            raise ValueError(f"kernel launch {self.name!r} requires unique structural source-task roles")
        if any(
            not isinstance(source_task, int) or isinstance(source_task, bool) or source_task < 0
            for source_task in self.source_tasks
        ) or len(set(self.source_tasks)) != len(self.source_tasks):
            raise ValueError(f"kernel launch {self.name!r} requires unique nonnegative source tasks")
        if self.source_phase is not None and (not isinstance(self.source_phase, str) or not self.source_phase):
            raise ValueError(f"kernel launch {self.name!r} source phase must be a non-empty string")
        maximum_dynamic_shared = next(
            (
                attribute.value
                for attribute in self.function_attributes
                if attribute.kind is CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES
            ),
            None,
        )
        if maximum_dynamic_shared is not None and self.dynamic_shared_memory_bytes > maximum_dynamic_shared:
            raise ValueError(
                f"kernel launch {self.name!r} requests more dynamic shared memory than its function attribute permits"
            )

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "arguments": tuple(argument.canonical_record for argument in self.arguments),
            "block": self.block,
            "dynamic_shared_memory_bytes": self.dynamic_shared_memory_bytes,
            "function_attributes": tuple(
                attribute.canonical_record
                for attribute in sorted(self.function_attributes, key=lambda item: item.kind.value)
            ),
            "grid": self.grid,
            "kernel_symbol": self.kernel_symbol,
            "name": self.name,
            "source_task_roles": tuple(role.value for role in self.source_task_roles),
            "source_tasks": self.source_tasks,
            "source_phase": self.source_phase,
        }


@dataclass(frozen=True)
class CutePipelineLaunchRecipe:
    """Workload-agnostic storage and ordered launches for a device pipeline."""

    schema_version: int
    slots: tuple[CutePipelineSlot, ...]
    launches: tuple[CuteKernelLaunch, ...]
    structural_event_covers: tuple[CuteStructuralEventCover, ...] = ()
    event_realizations: tuple[CuteEventRealization, ...] = ()
    physical_event_provenance: Mapping[str, Any] | None = None
    host_call_arguments: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported CuTe launch-recipe schema version: {self.schema_version}")
        if not self.slots or not self.launches:
            raise ValueError("CuTe launch recipes require storage slots and kernel launches")
        if any(not isinstance(slot, CutePipelineSlot) for slot in self.slots):
            raise ValueError("CuTe launch recipes contain an invalid storage slot")
        if any(not isinstance(launch, CuteKernelLaunch) for launch in self.launches):
            raise ValueError("CuTe launch recipes contain an invalid kernel launch")
        if any(not isinstance(cover, CuteStructuralEventCover) for cover in self.structural_event_covers):
            raise ValueError("CuTe launch recipes contain an invalid structural Event cover")
        if any(not isinstance(realization, CuteEventRealization) for realization in self.event_realizations):
            raise ValueError("CuTe launch recipes contain an invalid Event realization")
        if self.physical_event_provenance is not None:
            fingerprint = self.physical_event_provenance.get("fingerprint")
            payload = dict(self.physical_event_provenance)
            payload.pop("fingerprint", None)
            expected_fingerprint = _sha256(_canonical_json(payload).encode())
            if not isinstance(fingerprint, str) or fingerprint != expected_fingerprint:
                raise ValueError("physical Event provenance requires its canonical envelope fingerprint")
            schedule = self.physical_event_provenance.get("schedule")
            schedule_fingerprint = self.physical_event_provenance.get("physical_schedule_fingerprint")
            if (
                self.physical_event_provenance.get("schema_version") != 1
                or not isinstance(schedule, Mapping)
                or not isinstance(schedule_fingerprint, str)
                or schedule.get("fingerprint") != schedule_fingerprint
            ):
                raise ValueError("physical Event provenance must bind its physical schedule identity")
            _canonical_json(self.physical_event_provenance)
        slot_by_name = {slot.name: slot for slot in self.slots}
        if len(slot_by_name) != len(self.slots):
            raise ValueError("CuTe launch-recipe slot names must be unique")
        bound_host_slots = tuple(name for name in self.host_call_arguments if name is not None)
        if len(set(bound_host_slots)) != len(bound_host_slots):
            raise ValueError("CuTe host-call slot arguments must be unique")
        if unknown_host_slots := set(bound_host_slots) - set(slot_by_name):
            raise ValueError(f"CuTe host-call arguments reference unknown slots: {sorted(unknown_host_slots)}")
        launch_names = tuple(launch.name for launch in self.launches)
        if len(set(launch_names)) != len(launch_names):
            raise ValueError("CuTe launch names must be unique")
        launch_ordinals = {name: ordinal for ordinal, name in enumerate(launch_names)}
        for cover in self.structural_event_covers:
            try:
                producer_launch = launch_ordinals[cover.producer.launch]
                consumer_launch = launch_ordinals[cover.consumer.launch]
            except KeyError as error:
                raise ValueError("structural Event cover references an unknown launch") from error
            algebraic_refinement = cover.kind is CuteEventRealizationKind.ALGEBRAIC_REFINEMENT
            if algebraic_refinement and producer_launch != consumer_launch:
                raise ValueError("algebraic Event refinement must fuse both source endpoints into one launch")
            if not algebraic_refinement and (
                producer_launch > consumer_launch
                or (producer_launch == consumer_launch and cover.producer.phase >= cover.consumer.phase)
            ):
                raise ValueError("structural Event cover must order its producer before its consumer")
        if tuple(realization.event for realization in self.event_realizations) != tuple(
            range(len(self.event_realizations))
        ):
            raise ValueError("bound Event realizations must have dense source EventTensor ordinals")
        mechanism_allocations = tuple(
            realization.mechanism.allocation
            for realization in self.event_realizations
            if realization.mechanism is not None
        )
        if mechanism_allocations != tuple(range(len(mechanism_allocations))):
            raise ValueError("physical Event mechanism allocations must have dense ordinals")
        for realization in self.event_realizations:
            crosses_launch = False
            for index in range(0, len(realization.covering_launch_order), 4):
                producer_launch, producer_phase, consumer_launch, consumer_phase = realization.covering_launch_order[
                    index : index + 4
                ]
                if producer_launch >= len(self.launches) or consumer_launch >= len(self.launches):
                    raise ValueError("Event realization witness references an unknown launch")
                if realization.kind is CuteEventRealizationKind.ALGEBRAIC_REFINEMENT:
                    if producer_launch != consumer_launch:
                        raise ValueError("algebraic Event refinement witnesses must share one launch")
                elif producer_launch > consumer_launch or (
                    producer_launch == consumer_launch and producer_phase >= consumer_phase
                ):
                    raise ValueError("Event realization witnesses must order producer before consumer")
                crosses_launch |= producer_launch < consumer_launch
            if realization.kind is CuteEventRealizationKind.PROGRAM_ORDER and crosses_launch:
                raise ValueError("program-order Event witnesses must stay within one launch")
            if realization.kind is CuteEventRealizationKind.STREAM_ORDER and not crosses_launch:
                raise ValueError("stream-order Event witnesses must cross forward in launch order")
            for piece in realization.cover_pieces:
                producer_launch = piece.producer.launch
                consumer_launch = piece.consumer.launch
                if producer_launch >= len(self.launches) or consumer_launch >= len(self.launches):
                    raise ValueError("bound Event cover references an unknown launch")
                if realization.kind is CuteEventRealizationKind.STREAM_ORDER:
                    if producer_launch >= consumer_launch:
                        raise ValueError("stream-order Event covers must cross forward in launch order")
                elif (
                    realization.kind
                    in {
                        CuteEventRealizationKind.PROGRAM_ORDER,
                        CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
                    }
                    and producer_launch != consumer_launch
                ):
                    raise ValueError("same-launch Event covers must fuse both endpoints")
        _validate_contiguous_indexes(self.slots, input_indexes=True)
        _validate_contiguous_indexes(self.slots, input_indexes=False)
        _validate_slot_arguments(self.launches, slot_by_name)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "event_realizations": tuple(realization.canonical_record for realization in self.event_realizations),
            "launches": tuple(launch.canonical_record for launch in self.launches),
            "schema_version": self.schema_version,
            "slots": tuple(slot.canonical_record for slot in sorted(self.slots, key=lambda item: item.name)),
            "structural_event_covers": tuple(cover.canonical_record for cover in self.structural_event_covers),
            "physical_event_provenance": self.physical_event_provenance,
            "host_call_arguments": self.host_call_arguments,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_record)

    @property
    def digest(self) -> str:
        return _sha256(self.canonical_json.encode())


def bind_launch_source_tasks(
    recipe: CutePipelineLaunchRecipe,
    source_tasks_by_role: Mapping[CuteSourceTaskRole, tuple[int, ...]],
) -> CutePipelineLaunchRecipe:
    """Close emitter-declared launch roles over source task ordinals."""
    roles = tuple(source_tasks_by_role)
    if any(not isinstance(role, CuteSourceTaskRole) for role in roles):
        raise ValueError("source-task bindings require structural source-task roles")
    source_tasks = tuple(task for role in roles for task in source_tasks_by_role[role])
    if any(not isinstance(task, int) or isinstance(task, bool) or task < 0 for task in source_tasks) or len(
        set(source_tasks)
    ) != len(source_tasks):
        raise ValueError("structural source-task roles must partition unique nonnegative source tasks")
    launches: list[CuteKernelLaunch] = []
    covered: set[int] = set()
    for launch in recipe.launches:
        if launch.source_tasks:
            raise ValueError(f"kernel launch {launch.name!r} already has bound source-task coverage")
        if not launch.source_task_roles:
            raise ValueError(f"kernel launch {launch.name!r} has no structural source-task role")
        try:
            launch_tasks = tuple(task for role in launch.source_task_roles for task in source_tasks_by_role[role])
        except KeyError as error:
            raise ValueError(f"kernel launch {launch.name!r} references an unbound source-task role") from error
        if not launch_tasks:
            raise ValueError(f"kernel launch {launch.name!r} resolves to no source tasks")
        covered.update(launch_tasks)
        launches.append(replace(launch, source_tasks=launch_tasks))
    missing = set(source_tasks) - covered
    if missing:
        raise ValueError(f"physical launches do not cover source tasks {tuple(sorted(missing))}")
    return replace(recipe, launches=tuple(launches))


def bind_event_realizations(
    recipe: CutePipelineLaunchRecipe,
    *,
    source_events: tuple[tuple[int, int, int, int, str | None, str | None], ...],
    source_task_roles: Mapping[int, CuteSourceTaskRole],
) -> CutePipelineLaunchRecipe:
    """Bind emitter-declared structural covers to exact source EventTensor ordinals."""
    if recipe.event_realizations:
        raise ValueError("launch recipe already has bound Event realizations")
    if tuple(event for event, *_ in source_events) != tuple(range(len(source_events))):
        raise ValueError("source EventTensors must have dense ordinals")
    launch_ordinals = {launch.name: ordinal for ordinal, launch in enumerate(recipe.launches)}
    realizations: list[CuteEventRealization] = []
    used_covers: set[int] = set()
    for event, producer_task, consumer_task, relation_dimensions, producer_phase, consumer_phase in source_events:
        if not isinstance(relation_dimensions, int) or isinstance(relation_dimensions, bool) or relation_dimensions < 0:
            raise ValueError("source EventTensors require a nonnegative relation rank")
        if (producer_phase is None) != (consumer_phase is None):
            raise ValueError("source EventTensor phases must be present together")
        if producer_phase is not None:
            assert consumer_phase is not None
            if producer_task != consumer_task:
                raise ValueError("source phase Events must remain within one source task")
            phase_launches = {
                launch.source_phase: ordinal
                for ordinal, launch in enumerate(recipe.launches)
                if producer_task in launch.source_tasks and launch.source_phase is not None
            }
            if len(phase_launches) != sum(
                producer_task in launch.source_tasks and launch.source_phase is not None for launch in recipe.launches
            ):
                raise ValueError(f"physical recipe repeats a named phase for source task {producer_task}")
            try:
                producer_launch = phase_launches[producer_phase]
                consumer_launch = phase_launches[consumer_phase]
            except KeyError as error:
                raise ValueError(
                    f"physical recipe has no launch for source task {producer_task} phase {error.args[0]!r}"
                ) from error
            if producer_launch >= consumer_launch:
                raise ValueError(f"source task {producer_task} phase Event does not follow physical launch order")
            owner = CuteAffineMap(
                relation_dimensions,
                (affine_constant(0), affine_constant(0), affine_constant(0)),
            )
            producer_time = CuteAffineMap(relation_dimensions, (affine_constant(0),))
            consumer_time = CuteAffineMap(relation_dimensions, (affine_constant(0),))
            realizations.append(
                CuteEventRealization(
                    event,
                    CuteEventRealizationKind.STREAM_ORDER,
                    (producer_launch, 0, consumer_launch, 0),
                    (
                        CuteEventCoverPiece(
                            None,
                            CuteEventCoverEndpoint(producer_launch, owner, producer_time),
                            CuteEventCoverEndpoint(consumer_launch, owner, consumer_time),
                        ),
                    ),
                )
            )
            continue
        try:
            producer_role = source_task_roles[producer_task]
            consumer_role = source_task_roles[consumer_task]
        except KeyError as error:
            raise ValueError("source EventTensor references an unbound structural task") from error
        witnesses: list[int] = []
        pieces: list[CuteEventCoverPiece] = []
        crosses_launch = False
        declared_kinds: set[CuteEventRealizationKind | None] = set()
        for cover_index, cover in enumerate(recipe.structural_event_covers):
            if cover.producer_role is not producer_role or cover.consumer_role is not consumer_role:
                continue
            used_covers.add(cover_index)
            producer_launch = launch_ordinals[cover.producer.launch]
            consumer_launch = launch_ordinals[cover.consumer.launch]
            crosses_launch |= producer_launch < consumer_launch
            declared_kinds.add(cover.kind)
            witnesses.extend((producer_launch, cover.producer.phase, consumer_launch, cover.consumer.phase))
            cover_kind = cover.kind or (
                CuteEventRealizationKind.STREAM_ORDER
                if producer_launch < consumer_launch
                else CuteEventRealizationKind.PROGRAM_ORDER
            )
            if cover.producer_owner is not None:
                assert cover.consumer_owner is not None
                assert cover.producer_time is not None
                assert cover.consumer_time is not None
                if any(
                    mapping.dimensions != relation_dimensions
                    for mapping in (
                        cover.producer_owner,
                        cover.consumer_owner,
                        cover.producer_time,
                        cover.consumer_time,
                    )
                ):
                    raise ValueError("structural Event-cover rank does not match the source relation")
                pieces.append(
                    CuteEventCoverPiece(
                        cover.relation_subset,
                        CuteEventCoverEndpoint(producer_launch, cover.producer_owner, cover.producer_time),
                        CuteEventCoverEndpoint(consumer_launch, cover.consumer_owner, cover.consumer_time),
                    )
                )
            elif cover_kind is CuteEventRealizationKind.STREAM_ORDER:
                global_owner = CuteAffineMap(
                    relation_dimensions,
                    (affine_constant(0), affine_constant(0), affine_constant(0)),
                )
                producer_time = CuteAffineMap(relation_dimensions, (affine_constant(cover.producer.phase),))
                consumer_time = CuteAffineMap(relation_dimensions, (affine_constant(cover.consumer.phase),))
                pieces.append(
                    CuteEventCoverPiece(
                        None,
                        CuteEventCoverEndpoint(producer_launch, global_owner, producer_time),
                        CuteEventCoverEndpoint(consumer_launch, global_owner, consumer_time),
                    )
                )
        if not witnesses:
            raise ValueError(f"physical recipe has no Event cover for source task edge {producer_task}->{consumer_task}")
        if len(declared_kinds) != 1:
            raise ValueError("one source EventTensor cannot mix order and algebraic-refinement covers")
        declared_kind = next(iter(declared_kinds))
        kind = declared_kind or (
            CuteEventRealizationKind.STREAM_ORDER if crosses_launch else CuteEventRealizationKind.PROGRAM_ORDER
        )
        if not pieces:
            raise ValueError(f"physical recipe has no instance-level Event cover for source EventTensor {event}")
        realizations.append(
            CuteEventRealization(
                event,
                kind,
                tuple(witnesses),
                tuple(pieces),
            )
        )
    unused_covers = set(range(len(recipe.structural_event_covers))) - used_covers
    if unused_covers:
        raise ValueError("physical recipe contains a structural Event cover absent from the source schedule")
    return replace(recipe, event_realizations=tuple(realizations))


def bind_physical_event_provenance(
    recipe: CutePipelineLaunchRecipe,
    provenance: Mapping[str, Any],
) -> CutePipelineLaunchRecipe:
    """Bind exact refined internal Event provenance without changing source Events."""
    if recipe.physical_event_provenance is not None:
        raise ValueError("launch recipe already has physical Event provenance")
    schedule_fingerprint = provenance.get("fingerprint")
    if not isinstance(schedule_fingerprint, str) or len(schedule_fingerprint) != 64:
        raise ValueError("physical Event schedule requires a SHA-256 fingerprint")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "physical_schedule_fingerprint": schedule_fingerprint,
        "schedule": dict(provenance),
    }
    payload["fingerprint"] = _sha256(_canonical_json(payload).encode())
    return replace(recipe, physical_event_provenance=payload)


def combine_physical_event_schedules(*schedules: Mapping[str, Any]) -> dict[str, Any]:
    """Combine physical Event schedules while retaining each exact region record."""
    payload: dict[str, Any] = {"schema_version": 1, "regions": tuple(dict(schedule) for schedule in schedules)}
    payload["fingerprint"] = _sha256(_canonical_json(payload).encode())
    return payload


@dataclass(frozen=True)
class CutePipelineCompileRequest:
    """Backend-independent identity of one physical-template instantiation."""

    schema_version: int
    target: str
    numerical_policy: str
    physical_template: str
    tensors: tuple[Mapping[str, Any], ...]
    scalar_programs: Mapping[str, Any]
    index_relations: tuple[Mapping[str, Any], ...]
    contraction_dimensions: Mapping[str, Any]
    schedule: Mapping[str, Any]
    algebra_identity: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @property
    def digest(self) -> str:
        return _sha256(self.canonical_json.encode())

    @property
    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class CutePipelineArtifact:
    """Compiled host object, embedded device image, and sealed provenance."""

    request: CutePipelineCompileRequest
    host_object: bytes
    device_image: bytes
    kernel_names: tuple[str, ...]
    host_entry: str
    compiler_ir: str
    toolchain: Mapping[str, str]
    source_digests: Mapping[str, str]
    launch_recipe: CutePipelineLaunchRecipe | None = None
    structured_device_ir: tuple[CuteDeviceIrSnapshot, ...] = ()
    internal_event_legalizations: tuple[InternalEventDeviceLegalizationRegion, ...] = ()
    compiler_ir_format: str = "mlir"

    def __post_init__(self) -> None:
        if not self.host_object or not self.device_image:
            raise ValueError("CuTe pipeline artifacts require one host object and one device image")
        if not self.kernel_names or len(set(self.kernel_names)) != len(self.kernel_names):
            raise ValueError("CuTe pipeline artifact kernel names must be nonempty and unique")
        if any(not name or "\0" in name for name in self.kernel_names):
            raise ValueError("CuTe pipeline artifact kernel names must name device symbols")
        if not self.host_entry or "\0" in self.host_entry:
            raise ValueError("CuTe pipeline artifact must name its host entry")
        if not self.compiler_ir_format or "\0" in self.compiler_ir_format:
            raise ValueError("CuTe pipeline artifact must name its compiler IR format")
        if self.launch_recipe is not None:
            missing_symbols = {
                launch.kernel_symbol
                for launch in self.launch_recipe.launches
                if launch.kernel_symbol not in self.kernel_names
            }
            if missing_symbols:
                raise ValueError(
                    "CuTe launch recipe references symbols absent from its device image: "
                    + ", ".join(sorted(missing_symbols))
                )
        if tuple(region.region_ordinal for region in self.internal_event_legalizations) != tuple(
            range(len(self.internal_event_legalizations))
        ):
            raise ValueError("internal Event legalization regions must be dense and ordered")

    @property
    def host_object_digest(self) -> str:
        return _sha256(self.host_object)

    @property
    def device_image_digest(self) -> str:
        return _sha256(self.device_image)

    @property
    def compiler_ir_digest(self) -> str:
        return _sha256(self.compiler_ir.encode())

    @property
    def native_cache_identity(self) -> str | None:
        """Identify the native image and launch ABI independently of host glue."""
        if self.launch_recipe is None:
            return None
        return _sha256(
            _canonical_json(
                {
                    "device_image_sha256": self.device_image_digest,
                    "launch_recipe": self.launch_recipe.canonical_record,
                }
            ).encode()
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "request": json.loads(self.request.canonical_json),
            "request_digest": self.request.digest,
            "host_object": {
                "format": "elf64-relocatable",
                "sha256": self.host_object_digest,
                "bytes": len(self.host_object),
                "entry": self.host_entry,
            },
            "device_image": {
                "format": "cuda-fatbin",
                "sha256": self.device_image_digest,
                "bytes": len(self.device_image),
                "kernels": self.kernel_names,
            },
            "compiler_ir": {
                "format": self.compiler_ir_format,
                "sha256": self.compiler_ir_digest,
                "bytes": len(self.compiler_ir.encode()),
            },
            "structured_device_ir": tuple(snapshot.canonical_record for snapshot in self.structured_device_ir),
            "internal_event_legalizations": tuple(
                record
                for region in self.internal_event_legalizations
                for record in internal_event_device_legalization_region_record(region)
            ),
            "toolchain": dict(self.toolchain),
            "source_digests": dict(self.source_digests),
            "launch_recipe": self.launch_recipe.canonical_record if self.launch_recipe is not None else None,
            "native_cache_identity": self.native_cache_identity,
        }

    @property
    def canonical_manifest_json(self) -> str:
        return _canonical_json(self.manifest)

    @property
    def native_transport(self) -> bytes:
        """Serialize the typed recipe and fatbin for Shuttle's native GPU runtime."""
        if self.launch_recipe is None:
            raise ValueError("native CuTe transport requires a launch recipe")
        return _serialize_native_transport(
            self.request, self.launch_recipe, (self,), (0,) * len(self.launch_recipe.launches)
        )

    def write(self, directory: Path, *, stem: str) -> tuple[Path, Path, Path, Path]:
        """Write the sealed manifest and its three content-addressed payloads."""
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / f"{stem}.json"
        object_path = directory / f"{stem}.o"
        device_image_path = directory / f"{stem}.fatbin"
        mlir_path = directory / f"{stem}.mlir"
        manifest_path.write_text(self.canonical_manifest_json + "\n")
        object_path.write_bytes(self.host_object)
        device_image_path.write_bytes(self.device_image)
        mlir_path.write_text(self.compiler_ir)
        return manifest_path, object_path, device_image_path, mlir_path


@dataclass(frozen=True)
class CuteMultiImagePipelineArtifact:
    """One canonical launch recipe whose symbols are scoped to multiple images."""

    request: CutePipelineCompileRequest
    images: tuple[CutePipelineArtifact, ...]
    launch_recipe: CutePipelineLaunchRecipe
    image_by_launch: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("multi-image CuTe artifacts require at least one device image")
        if len(self.image_by_launch) != len(self.launch_recipe.launches):
            raise ValueError("multi-image CuTe artifact must bind every launch to one image")
        for launch, image_index in zip(self.launch_recipe.launches, self.image_by_launch, strict=True):
            if not 0 <= image_index < len(self.images):
                raise ValueError(f"CuTe launch {launch.name!r} references an unavailable image")
            if launch.kernel_symbol not in self.images[image_index].kernel_names:
                raise ValueError(f"CuTe launch {launch.name!r} symbol is absent from its bound image {image_index}")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "request_digest": self.request.digest,
            "launch_recipe_digest": self.launch_recipe.digest,
            "image_by_launch": self.image_by_launch,
            "images": tuple(
                {
                    "request_digest": image.request.digest,
                    "host_object_digest": image.host_object_digest,
                    "device_image_digest": image.device_image_digest,
                    "kernel_names": image.kernel_names,
                    "host_entry": image.host_entry,
                }
                for image in self.images
            ),
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.canonical_record).encode())

    @property
    def native_transport(self) -> bytes:
        """Serialize the complete recipe with image-scoped kernel bindings."""
        return _serialize_native_transport(self.request, self.launch_recipe, self.images, self.image_by_launch)


class _NativeTransportWriter:
    def __init__(self) -> None:
        self.data = bytearray()

    def u8(self, value: int) -> None:
        self.data.extend(struct.pack("<B", value))

    def u16(self, value: int) -> None:
        self.data.extend(struct.pack("<H", value))

    def u32(self, value: int) -> None:
        self.data.extend(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        self.data.extend(struct.pack("<Q", value))

    def i64(self, value: int) -> None:
        self.data.extend(struct.pack("<q", value))

    def text(self, value: str) -> None:
        self.blob(value.encode())

    def blob(self, value: bytes) -> None:
        self.u64(len(value))
        self.data.extend(value)

    def integers(self, values: tuple[int, ...]) -> None:
        self.u64(len(values))
        for value in values:
            self.i64(value)


def _write_native_affine_expression(writer: _NativeTransportWriter, expression: CuteAffineExpression) -> None:
    writer.u8(
        {
            CuteAffineExpressionKind.DIMENSION: 0,
            CuteAffineExpressionKind.CONSTANT: 1,
            CuteAffineExpressionKind.ADD: 2,
            CuteAffineExpressionKind.MULTIPLY: 3,
            CuteAffineExpressionKind.MODULO: 4,
            CuteAffineExpressionKind.FLOOR_DIVIDE: 5,
            CuteAffineExpressionKind.CEIL_DIVIDE: 6,
        }[expression.kind]
    )
    writer.i64(expression.payload)
    writer.u8(len(expression.operands))
    for operand in expression.operands:
        _write_native_affine_expression(writer, operand)


def _write_native_affine_map(writer: _NativeTransportWriter, mapping: CuteAffineMap) -> None:
    writer.u8(mapping.dimensions)
    writer.u8(len(mapping.results))
    for result in mapping.results:
        _write_native_affine_expression(writer, result)


def _write_native_integer_set(writer: _NativeTransportWriter, integer_set: CuteIntegerSet) -> None:
    writer.u8(integer_set.dimensions)
    writer.u8(len(integer_set.constraints))
    for constraint, equality in zip(integer_set.constraints, integer_set.equalities, strict=True):
        writer.u8(int(equality))
        _write_native_affine_expression(writer, constraint)


def _write_native_event_cover_endpoint(
    writer: _NativeTransportWriter,
    endpoint: CuteEventCoverEndpoint,
) -> None:
    writer.i64(endpoint.launch)
    _write_native_affine_map(writer, endpoint.owner)
    _write_native_affine_map(writer, endpoint.local_time)


def _write_native_event_mechanism(
    writer: _NativeTransportWriter,
    mechanism: CuteEventMechanismAllocation,
) -> None:
    writer.i64(mechanism.allocation)
    writer.u8(
        {
            CuteEventPhysicalScope.THREAD_BLOCK: 0,
            CuteEventPhysicalScope.CLUSTER: 1,
            CuteEventPhysicalScope.DEVICE: 2,
            CuteEventPhysicalScope.SYSTEM: 3,
        }[mechanism.scope]
    )
    writer.i64(mechanism.capacity)
    _write_native_affine_map(writer, mechanism.slot)
    _write_native_affine_map(writer, mechanism.generation_phase)
    writer.i64(mechanism.generation_count)
    writer.i64(mechanism.arrival_count)
    writer.u8(mechanism.transaction_bytes is not None)
    if mechanism.transaction_bytes is not None:
        writer.i64(mechanism.transaction_bytes)


def _serialize_native_transport(
    request: CutePipelineCompileRequest,
    recipe: CutePipelineLaunchRecipe,
    images: tuple[CutePipelineArtifact, ...],
    image_by_launch: tuple[int, ...],
) -> bytes:
    if any(not launch.source_tasks for launch in recipe.launches):
        raise ValueError("native CuTe transport requires source-task coverage for every physical launch")
    slot_ordinals = {slot.name: ordinal for ordinal, slot in enumerate(recipe.slots)}
    writer = _NativeTransportWriter()
    writer.data.extend(b"SHUTGPU\0")
    writer.u32(12)
    device_encodings = {
        "sm90": (5, 4),
        "sm90a": (5, 4),
        "sm_90": (5, 4),
        "sm_90a": (5, 4),
        "sm100": (6, 5),
        "sm100a": (6, 5),
        "sm_100": (6, 5),
        "sm_100a": (6, 5),
    }
    try:
        device_schema, code_format = device_encodings[request.target]
    except KeyError as error:
        raise ValueError(f"native CuTe transport does not support target {request.target!r}") from error
    for image in images:
        if device_encodings.get(image.request.target) != (device_schema, code_format):
            raise ValueError("native CuTe transport requires every image to match the pipeline device target")
    writer.i64(device_schema)
    writer.u8(code_format)
    writer.text(recipe.digest)
    writer.text(request.digest)
    writer.u64(len(images))
    for image in images:
        writer.blob(image.device_image)
        writer.text(image.device_image_digest)
    writer.u64(len(recipe.launches))
    slot_access_masks = {name: 0 for name in slot_ordinals}
    last_writers: dict[str, int] = {}
    for ordinal, launch in enumerate(recipe.launches):
        accesses = _launch_slot_accesses(launch)
        inputs = tuple(sorted(slot_ordinals[name] for name, mask in accesses.items() if mask & 1))
        outputs = tuple(sorted(slot_ordinals[name] for name, mask in accesses.items() if mask & 2))
        dependencies = tuple(
            sorted({last_writers[name] for name, mask in accesses.items() if mask & 1 and name in last_writers})
        )
        for name, mask in accesses.items():
            slot_access_masks[name] |= mask
            if mask & 2:
                last_writers[name] = ordinal

        writer.i64(ordinal)
        writer.u64(image_by_launch[ordinal])
        writer.integers(launch.source_tasks)
        writer.text(launch.kernel_symbol)
        for extent in launch.grid:
            writer.u32(extent)
        for extent in launch.block:
            writer.u16(extent)
        writer.u32(launch.dynamic_shared_memory_bytes)
        writer.u16(len(launch.arguments))
        for argument in launch.arguments:
            _write_native_argument(writer, argument, slot_ordinals)
        writer.u16(len(launch.function_attributes))
        for attribute in launch.function_attributes:
            writer.u8(
                {
                    CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES: 0,
                    CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT: 1,
                    CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED: 2,
                }[attribute.kind]
            )
            writer.u64(attribute.value)
        writer.integers(inputs)
        writer.integers(outputs)
        writer.integers(dependencies)

    writer.u64(len(recipe.event_realizations))
    for realization in recipe.event_realizations:
        writer.i64(realization.event)
        writer.u8(
            {
                CuteEventRealizationKind.PROGRAM_ORDER: 0,
                CuteEventRealizationKind.STREAM_ORDER: 1,
                CuteEventRealizationKind.ALGEBRAIC_REFINEMENT: 2,
                CuteEventRealizationKind.MBARRIER: 3,
                CuteEventRealizationKind.PIPELINE: 4,
                CuteEventRealizationKind.QUEUE: 5,
            }[realization.kind]
        )
        writer.integers(realization.covering_launch_order)
        writer.u64(len(realization.cover_pieces))
        for piece in realization.cover_pieces:
            dimensions = piece.producer.owner.dimensions
            writer.u8(dimensions)
            writer.u8(piece.relation_subset is not None)
            if piece.relation_subset is not None:
                _write_native_integer_set(writer, piece.relation_subset)
            _write_native_event_cover_endpoint(writer, piece.producer)
            _write_native_event_cover_endpoint(writer, piece.consumer)
        writer.u8(realization.mechanism is not None)
        if realization.mechanism is not None:
            _write_native_event_mechanism(writer, realization.mechanism)
        # Source Event certificates remain unavailable through opaque
        # transport. Internal Event legalizations use the separate in-process
        # provider result and are never embedded here as trusted records.
        writer.u8(0)

    writer.u8(recipe.physical_event_provenance is not None)
    if recipe.physical_event_provenance is not None:
        canonical_provenance = _canonical_json(recipe.physical_event_provenance)
        writer.text(canonical_provenance)
        writer.text(_sha256(canonical_provenance.encode()))

    snapshots = tuple(snapshot for image in images for snapshot in image.structured_device_ir)
    writer.u64(len(snapshots))
    for snapshot in snapshots:
        writer.text(snapshot.bytecode_sha256)
        writer.u64(len(snapshot.operation_counts))
        for operation_name, count in snapshot.operation_counts:
            writer.text(operation_name)
            writer.u64(count)

    writer.u64(len(recipe.slots))
    storage_groups: dict[tuple[str, int | str], int] = {}
    for ordinal, slot in enumerate(recipe.slots):
        storage_identity = _slot_storage_identity(slot)
        reuse_group = storage_groups.setdefault(storage_identity, len(storage_groups))
        writer.i64(ordinal)
        try:
            element_tag = {
                CutePipelineElementType.BF16: 0,
                CutePipelineElementType.F32: 1,
                CutePipelineElementType.PRED: 2,
                CutePipelineElementType.I32: 3,
            }[slot.element_type]
        except KeyError as error:
            raise ValueError(f"native CuTe transport does not support {slot.element_type.value} tensor slots") from error
        writer.u8(element_tag)
        writer.u64(len(slot.shape))
        for extent in slot.shape:
            writer.i64(extent)
        writer.i64(slot.required_bytes)
        writer.integers(_contiguous_byte_strides(slot))
        writer.i64(0)
        writer.i64(slot.alignment)
        writer.u8(1)
        aggregate_access = slot_access_masks[slot.name]
        if slot.kind is CutePipelineSlotKind.PUBLIC_INPUT:
            abi_access = _access_mask(CutePipelineSlotAccess.READ)
        elif slot.kind is CutePipelineSlotKind.PUBLIC_OUTPUT:
            abi_access = _access_mask(CutePipelineSlotAccess.WRITE)
        else:
            abi_access = aggregate_access
        writer.u8(abi_access - 1)
        temporary = slot.kind is CutePipelineSlotKind.SCRATCH
        writer.u8(1 if temporary else 0)
        writer.i64(ordinal)
        writer.i64(reuse_group)
        if slot.kind in (CutePipelineSlotKind.PUBLIC_INPUT, CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT):
            writer.u8(1)
            writer.i64(slot.input_index if slot.input_index is not None else -1)
        elif slot.kind is CutePipelineSlotKind.PUBLIC_OUTPUT:
            writer.u8(2)
            writer.i64(slot.output_index if slot.output_index is not None else -1)
        else:
            writer.u8(0)
            writer.i64(-1)
    writer.text(hashlib.sha256(writer.data).hexdigest())
    return bytes(writer.data)


def _slot_storage_identity(slot: CutePipelineSlot) -> tuple[str, int | str]:
    # Explicit scratch reuse and independent slots occupy separate namespaces.
    if slot.kind is CutePipelineSlotKind.SCRATCH and slot.reuse_group is not None:
        return ("reuse", slot.reuse_group)
    return ("slot", slot.name)


def verify_materialized_cut_launch_dependencies(
    launches: tuple[CuteKernelLaunch, ...],
    *,
    slots: tuple[CutePipelineSlot, ...],
    cut_slots: tuple[str, ...],
    launch_schedule: Mapping[str, Any],
) -> None:
    """Prove each materialized phase cut is covered by typed launch Events.

    Cut values have one SSA writer and one materialized generation. Every
    reader, including a later component or a structured pointer argument,
    must follow that writer in both executable dependence representations.
    """
    if not cut_slots:
        return
    if launch_schedule.get("schema_version") != 2:
        raise ValueError("materialized phase cuts require a typed schema-2 launch schedule")
    realizations = launch_schedule["realizations"]
    if len(realizations) != 1 or realizations[0]["kind"] != "erased_stream_order":
        raise ValueError("materialized phase cuts require one stream-order Event realization")
    relation = realizations[0]["event"]["required_dependence"]["relation"]
    expected_domain = {
        "kind": "task_family",
        "name": "pipeline_launch",
        "axes": [{"name": "launch", "extent": len(launches)}],
        "placement": "cuda_stream",
    }
    if any(
        _canonical_json(relation[key]) != _canonical_json(expected_domain) for key in ("source_domain", "target_domain")
    ):
        raise ValueError("materialized phase cuts require the actual launch-coordinate domain")
    predecessors = {(tuple(source), tuple(target)) for source, target in launch_schedule["predecessor_pairs"]}
    required = {(tuple(pair["source"]), tuple(pair["target"])) for pair in relation["pairs"]}
    accesses = tuple(_launch_slot_accesses(launch) for launch in launches)
    for name in cut_slots:
        bindings = tuple(slot for slot in slots if slot.name == name)
        if len(bindings) != 1:
            raise ValueError(f"materialized phase cut {name!r} requires one slot binding")
        if bindings[0].reuse_group is not None:
            raise ValueError(f"materialized phase cut {name!r} requires independent SSA storage")
        writers = tuple(index for index, access in enumerate(accesses) if access.get(name, 0) & 2)
        readers = tuple(index for index, access in enumerate(accesses) if access.get(name, 0) & 1)
        if len(writers) != 1 or not readers:
            raise ValueError(f"materialized phase cut {name!r} requires exactly one writer and at least one reader")
        writer = writers[0]
        for reader in readers:
            if writer >= reader:
                raise ValueError(f"materialized phase cut {name!r} reader does not follow its writer")
            edge = ((writer,), (reader,))
            if edge not in predecessors or edge not in required:
                raise ValueError(f"materialized phase cut {name!r} lacks typed launch dependence {writer}->{reader}")


def launch_dependency_event_schedule(
    launches: tuple[CuteKernelLaunch, ...], *, slots: tuple[CutePipelineSlot, ...]
) -> dict[str, Any]:
    """Derive typed stream-order Events from the recipe's actual slot dependencies."""
    task = TaskFamily("pipeline_launch", (TaskAxis("launch", len(launches)),), placement="cuda_stream")
    storage_by_slot = {slot.name: _slot_storage_identity(slot) for slot in slots}
    last_writers: dict[tuple[str, Any], int] = {}
    readers: dict[tuple[str, Any], set[int]] = {}
    pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for ordinal, launch in enumerate(launches):
        accesses: dict[tuple[str, Any], int] = {}
        for name, access in _launch_slot_accesses(launch).items():
            storage = storage_by_slot[name]
            accesses[storage] = accesses.get(storage, 0) | access
        for name, access in accesses.items():
            if name in last_writers:
                pairs.add(((last_writers[name],), (ordinal,)))
            if access & 2:
                pairs.update(((reader,), (ordinal,)) for reader in readers.get(name, set()))
        for name, access in accesses.items():
            if access & 2:
                last_writers[name] = ordinal
                readers[name] = set()
            elif access & 1:
                readers.setdefault(name, set()).add(ordinal)
    relation = TaskRelation.from_pairs(task, task, tuple(sorted(pairs)))
    dependence = TaskDependence(relation, MemoryVisibility(EventMemoryScope.DEVICE))
    plan = derive_event_tensor_plan(dependence, name="pipeline_launch_dependencies")
    program = EventDataflowProgram((task,), (dependence,), (plan,))
    verify_event_dataflow_program(program)
    audit = verify_event_realizations(
        program,
        (
            erased_event_realization(
                plan,
                kind=EventRealizationKind.ERASED_STREAM_ORDER,
                mechanism="ordered_cuda_launch_sequence",
                reason="producer writes precede dependent reads on the same CUDA stream",
                ordering=relation,
            ),
        ),
    )
    return {
        "schema_version": 2,
        "name": "pipeline_launch_dependencies",
        "task_axes": ({"name": "launch", "extent": len(launches)},),
        "predecessor_pairs": tuple(sorted(pairs)),
        "realizations": internal_event_realization_records(audit, {plan.name: plan}, ()),
        "allocations": (),
    }


def _launch_slot_accesses(launch: CuteKernelLaunch) -> dict[str, int]:
    accesses: dict[str, int] = {}
    for argument in launch.arguments:
        if isinstance(argument, (CutePipelineSlotArgument, CutePipelineTensorMapArgument)):
            accesses[argument.slot] = accesses.get(argument.slot, 0) | _access_mask(argument.access)
        elif isinstance(argument, CutePipelineStructuredArgument):
            for field in argument.fields:
                if isinstance(field, CutePipelineStructuredSlotAddressField):
                    accesses[field.slot] = accesses.get(field.slot, 0) | _access_mask(field.access)
    return accesses


def _access_mask(access: CutePipelineSlotAccess) -> int:
    return {
        CutePipelineSlotAccess.READ: 1,
        CutePipelineSlotAccess.WRITE: 2,
        CutePipelineSlotAccess.READ_WRITE: 3,
    }[access]


def _write_native_argument(
    writer: _NativeTransportWriter,
    argument: CutePipelineKernelArgument,
    slot_ordinals: Mapping[str, int],
) -> None:
    if isinstance(argument, CutePipelineSlotArgument):
        writer.u8(0)
        writer.i64(slot_ordinals[argument.slot])
        writer.u64(argument.byte_offset)
        writer.u8(_access_mask(argument.access) - 1)
        return
    if isinstance(argument, CutePipelineTensorMapArgument):
        writer.u8(3)
        writer.i64(slot_ordinals[argument.slot])
        writer.u64(argument.byte_offset)
        writer.u8(_element_type_bytes(argument.element_type))
        writer.u8(len(argument.global_shape))
        for value in argument.global_shape:
            writer.u64(value)
        for value in argument.global_byte_strides:
            writer.u64(value)
        for value in argument.box_shape:
            writer.u32(value)
        for value in argument.element_strides:
            writer.u32(value)
        writer.u8(
            {CuteTensorMapInterleave.NONE: 0, CuteTensorMapInterleave.BYTES_16: 1, CuteTensorMapInterleave.BYTES_32: 2}[
                argument.interleave
            ]
        )
        writer.u8(
            {
                CuteTensorMapSwizzle.NONE: 0,
                CuteTensorMapSwizzle.BYTES_32: 1,
                CuteTensorMapSwizzle.BYTES_64: 2,
                CuteTensorMapSwizzle.BYTES_128: 3,
            }[argument.swizzle]
        )
        writer.u8(
            {
                CuteTensorMapL2Promotion.NONE: 0,
                CuteTensorMapL2Promotion.BYTES_64: 1,
                CuteTensorMapL2Promotion.BYTES_128: 2,
                CuteTensorMapL2Promotion.BYTES_256: 3,
            }[argument.l2_promotion]
        )
        writer.u8({CuteTensorMapOobFill.NONE: 0, CuteTensorMapOobFill.NAN_REQUEST_ZERO_FMA: 1}[argument.oob_fill])
        writer.u16(argument.alignment)
        writer.u8(_access_mask(argument.access) - 1)
        return
    if isinstance(argument, CutePipelineStructuredArgument):
        writer.u8(4)
        writer.u32(argument.size)
        writer.u16(argument.alignment)
        writer.u16(len(argument.fields))
        for field in argument.fields:
            if isinstance(field, CutePipelineStructuredSlotAddressField):
                writer.u8(0)
                writer.u32(field.destination_byte_offset)
                writer.i64(slot_ordinals[field.slot])
                writer.u64(field.slot_byte_offset)
                writer.u8(_access_mask(field.access) - 1)
            else:
                writer.u8(1)
                writer.u32(field.destination_byte_offset)
                writer.blob(field.data)
        return
    if isinstance(argument, CutePipelineScalarArgument):
        if argument.element_type is CutePipelineElementType.I32:
            scalar_type, data = 0, struct.pack("<i", int(argument.value))
        elif argument.element_type is CutePipelineElementType.U32:
            scalar_type, data = 1, struct.pack("<I", int(argument.value))
        elif argument.element_type is CutePipelineElementType.I64:
            scalar_type, data = 2, struct.pack("<q", int(argument.value))
        elif argument.element_type is CutePipelineElementType.U64:
            scalar_type, data = 3, struct.pack("<Q", int(argument.value))
        elif argument.element_type is CutePipelineElementType.F32:
            scalar_type, data = 4, struct.pack("<f", float(argument.value))
        elif argument.element_type is CutePipelineElementType.F64:
            scalar_type, data = 5, struct.pack("<d", float(argument.value))
        else:
            raise ValueError(f"native scalar transport does not support {argument.element_type.value}")
        writer.u8(1)
        writer.u8(scalar_type)
        writer.u16(len(data))
        writer.blob(data)
        return
    assert isinstance(argument, CutePipelinePackedBytesArgument)
    writer.u8(2)
    writer.u8(0)
    writer.u16(argument.alignment)
    writer.blob(argument.data)


def _contiguous_byte_strides(slot: CutePipelineSlot) -> tuple[int, ...]:
    stride = _element_type_bytes(slot.element_type)
    strides = []
    for extent in reversed(slot.shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def elf64_symbol_bytes(object_bytes: bytes, symbol_name: str) -> bytes:
    """Extract one defined data symbol from a little-endian ELF64 object."""
    if object_bytes[:16] != b"\x7fELF\x02\x01\x01" + b"\0" * 9:
        raise ValueError("CuTe host object must be a little-endian ELF64 object")
    if len(object_bytes) < 64:
        raise ValueError("ELF64 object is truncated")
    section_offset = struct.unpack_from("<Q", object_bytes, 40)[0]
    section_entry_size, section_count = struct.unpack_from("<HH", object_bytes, 58)
    if section_entry_size != 64 or section_count == 0:
        raise ValueError("ELF64 object has an unsupported section table")
    section_table_end = section_offset + section_entry_size * section_count
    if section_table_end > len(object_bytes):
        raise ValueError("ELF64 section table is truncated")

    sections = [
        struct.unpack_from("<IIQQQQIIQQ", object_bytes, section_offset + ordinal * section_entry_size)
        for ordinal in range(section_count)
    ]
    for section in sections:
        section_type = section[1]
        if section_type != 2:  # SHT_SYMTAB
            continue
        symbol_offset, symbol_size, string_section_ordinal, symbol_entry_size = (
            section[4],
            section[5],
            section[6],
            section[9],
        )
        if symbol_entry_size != 24 or symbol_size % symbol_entry_size:
            raise ValueError("ELF64 symbol table has an unsupported entry size")
        string_section = sections[string_section_ordinal]
        string_table = _slice(object_bytes, string_section[4], string_section[5], "ELF64 string table")
        symbol_table = _slice(object_bytes, symbol_offset, symbol_size, "ELF64 symbol table")
        for symbol_ordinal in range(symbol_size // symbol_entry_size):
            name_offset, _, _, defining_section, value, size = struct.unpack_from(
                "<IBBHQQ", symbol_table, symbol_ordinal * symbol_entry_size
            )
            name = _elf_string(string_table, name_offset)
            if name != symbol_name:
                continue
            if defining_section == 0 or defining_section >= len(sections):
                raise ValueError(f"ELF64 symbol {symbol_name!r} is not defined in a data section")
            data_section = sections[defining_section]
            section_address, data_offset, data_size = data_section[3], data_section[4], data_section[5]
            relative_offset = value - section_address
            if relative_offset < 0 or relative_offset + size > data_size:
                raise ValueError(f"ELF64 symbol {symbol_name!r} exceeds its defining section")
            return _slice(object_bytes, data_offset + relative_offset, size, f"ELF64 symbol {symbol_name!r}")
    raise ValueError(f"ELF64 object does not define symbol {symbol_name!r}")


def _require_enum(value: Any, enum_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{label} must be a {enum_type.__name__}")


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{label} must be a Python-style identifier: {value!r}")


def _validate_alignment(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value & (value - 1):
        raise ValueError(f"{label} alignment must be a positive power of two")


def _validate_destination_offset(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"structured argument field {name!r} destination byte offset must be nonnegative")


def _validate_public_index(value: int | None, *, required: bool, label: str) -> None:
    if required:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} index must be a nonnegative integer")
        return
    if value is not None:
        raise ValueError(f"{label} index is not valid for this storage kind")


def _element_type_bytes(element_type: CutePipelineElementType) -> int:
    return {
        CutePipelineElementType.PRED: 1,
        CutePipelineElementType.U8: 1,
        CutePipelineElementType.I8: 1,
        CutePipelineElementType.U16: 2,
        CutePipelineElementType.I16: 2,
        CutePipelineElementType.F16: 2,
        CutePipelineElementType.BF16: 2,
        CutePipelineElementType.U32: 4,
        CutePipelineElementType.I32: 4,
        CutePipelineElementType.F32: 4,
        CutePipelineElementType.U64: 8,
        CutePipelineElementType.I64: 8,
        CutePipelineElementType.F64: 8,
    }[element_type]


def _validate_scalar(element_type: CutePipelineElementType, value: bool | int | float, *, name: str) -> None:
    if element_type is CutePipelineElementType.PRED:
        if not isinstance(value, bool):
            raise ValueError(f"predicate kernel argument {name!r} requires a boolean value")
        return
    integer_ranges = {
        CutePipelineElementType.U8: (0, 2**8 - 1),
        CutePipelineElementType.I8: (-(2**7), 2**7 - 1),
        CutePipelineElementType.U16: (0, 2**16 - 1),
        CutePipelineElementType.I16: (-(2**15), 2**15 - 1),
        CutePipelineElementType.U32: (0, 2**32 - 1),
        CutePipelineElementType.I32: (-(2**31), 2**31 - 1),
        CutePipelineElementType.U64: (0, 2**64 - 1),
        CutePipelineElementType.I64: (-(2**63), 2**63 - 1),
    }
    if element_type in integer_ranges:
        lower, upper = integer_ranges[element_type]
        if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
            raise ValueError(f"integer kernel argument {name!r} is outside the range of {element_type.value}")
        return
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"floating-point kernel argument {name!r} requires a finite float value")


def _validate_launch_dimensions(
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    *,
    launch_name: str,
) -> None:
    if len(grid) != 3 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid):
        raise ValueError(f"kernel launch {launch_name!r} grid must contain three positive integers")
    if len(block) != 3 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in block):
        raise ValueError(f"kernel launch {launch_name!r} block must contain three positive integers")
    if grid[0] > 2**31 - 1 or grid[1] > 65535 or grid[2] > 65535:
        raise ValueError(f"kernel launch {launch_name!r} grid exceeds CUDA dimension limits")
    if block[0] > 1024 or block[1] > 1024 or block[2] > 64 or math.prod(block) > 1024:
        raise ValueError(f"kernel launch {launch_name!r} block exceeds CUDA thread limits")


def _validate_tensor_map_dimensions(argument: CutePipelineTensorMapArgument) -> None:
    rank = len(argument.global_shape)
    if not 1 <= rank <= 5:
        raise ValueError(f"tensor map {argument.name!r} rank must be between one and five")
    if len(argument.global_byte_strides) != rank - 1:
        raise ValueError(f"tensor map {argument.name!r} requires one global byte stride per noncontiguous dimension")
    if len(argument.box_shape) != rank or len(argument.element_strides) != rank:
        raise ValueError(f"tensor map {argument.name!r} box shape and element strides must match its rank")
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or not 1 <= extent <= 2**32 - 1
        for extent in argument.global_shape
    ):
        raise ValueError(f"tensor map {argument.name!r} global extents must fit unsigned 32-bit dimensions")
    if any(
        not isinstance(stride, int) or isinstance(stride, bool) or not 16 <= stride < 2**40 or stride % 16
        for stride in argument.global_byte_strides
    ):
        raise ValueError(f"tensor map {argument.name!r} global byte strides must be 16-byte multiples below 2^40")
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or not 1 <= extent <= 256
        for extent in argument.box_shape
    ):
        raise ValueError(f"tensor map {argument.name!r} box extents must be between one and 256")
    if any(
        not isinstance(stride, int) or isinstance(stride, bool) or not 1 <= stride <= 8
        for stride in argument.element_strides
    ):
        raise ValueError(f"tensor map {argument.name!r} element strides must be between one and eight")
    floating_types = {
        CutePipelineElementType.F16,
        CutePipelineElementType.BF16,
        CutePipelineElementType.F32,
        CutePipelineElementType.F64,
    }
    if argument.oob_fill is CuteTensorMapOobFill.NAN_REQUEST_ZERO_FMA and argument.element_type not in floating_types:
        raise ValueError(f"tensor map {argument.name!r} NaN out-of-bounds fill requires a floating-point element type")


def _validate_contiguous_indexes(slots: tuple[CutePipelineSlot, ...], *, input_indexes: bool) -> None:
    indexes = sorted(
        index for slot in slots if (index := slot.input_index if input_indexes else slot.output_index) is not None
    )
    label = "input" if input_indexes else "output"
    if indexes != list(range(len(indexes))):
        raise ValueError(f"CuTe launch-recipe public {label} indexes must be unique and contiguous from zero")


def _validate_slot_arguments(
    launches: tuple[CuteKernelLaunch, ...],
    slot_by_name: Mapping[str, CutePipelineSlot],
) -> None:
    observed_accesses: dict[str, set[CutePipelineSlotAccess]] = {name: set() for name in slot_by_name}

    def observe_slot(
        *,
        launch_name: str,
        argument_name: str,
        slot_name: str,
        byte_offset: int,
        width: int,
        access: CutePipelineSlotAccess,
    ) -> CutePipelineSlot:
        if slot_name not in slot_by_name:
            raise ValueError(
                f"kernel launch {launch_name!r} argument {argument_name!r} references unknown slot {slot_name!r}"
            )
        slot = slot_by_name[slot_name]
        if byte_offset >= slot.required_bytes or width > slot.required_bytes - byte_offset:
            raise ValueError(
                f"kernel launch {launch_name!r} argument {argument_name!r} address range exceeds slot {slot.name!r}"
            )
        if slot.kind is CutePipelineSlotKind.PUBLIC_INPUT and access is not CutePipelineSlotAccess.READ:
            raise ValueError(f"public input slot {slot.name!r} may only be read")
        observed_accesses[slot.name].add(access)
        return slot

    for launch in launches:
        for argument in launch.arguments:
            if isinstance(argument, CutePipelineStructuredArgument):
                for field in argument.fields:
                    if not isinstance(field, CutePipelineStructuredSlotAddressField):
                        continue
                    observe_slot(
                        launch_name=launch.name,
                        argument_name=f"{argument.name}.{field.name}",
                        slot_name=field.slot,
                        byte_offset=field.slot_byte_offset,
                        width=1,
                        access=field.access,
                    )
                continue
            if not isinstance(argument, (CutePipelineSlotArgument, CutePipelineTensorMapArgument)):
                continue
            span_bytes = 1
            if isinstance(argument, CutePipelineTensorMapArgument):
                span_bytes = _element_type_bytes(argument.element_type) * argument.global_shape[0]
                span_bytes += sum(
                    (extent - 1) * stride
                    for extent, stride in zip(
                        argument.global_shape[1:],
                        argument.global_byte_strides,
                        strict=True,
                    )
                )
            slot = observe_slot(
                launch_name=launch.name,
                argument_name=argument.name,
                slot_name=argument.slot,
                byte_offset=argument.byte_offset,
                width=span_bytes,
                access=argument.access,
            )
            element_bytes = _element_type_bytes(slot.element_type)
            if argument.byte_offset % element_bytes:
                raise ValueError(
                    f"kernel launch {launch.name!r} argument {argument.name!r} has a misaligned element offset"
                )
            if isinstance(argument, CutePipelineTensorMapArgument):
                if argument.element_type is not slot.element_type:
                    raise ValueError(f"tensor map {argument.name!r} element type does not match slot {slot.name!r}")

    for slot in slot_by_name.values():
        accesses = observed_accesses[slot.name]
        if not accesses:
            raise ValueError(f"pipeline slot {slot.name!r} is not referenced by any kernel launch")
        reads = bool(accesses & {CutePipelineSlotAccess.READ, CutePipelineSlotAccess.READ_WRITE})
        writes = bool(accesses & {CutePipelineSlotAccess.WRITE, CutePipelineSlotAccess.READ_WRITE})
        if slot.kind is CutePipelineSlotKind.PUBLIC_INPUT and not reads:
            raise ValueError(f"public input slot {slot.name!r} is never read")
        if slot.kind is CutePipelineSlotKind.PUBLIC_OUTPUT and not writes:
            raise ValueError(f"public output slot {slot.name!r} is never written")
        if slot.kind is CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT and not (reads and writes):
            raise ValueError(f"public input-output slot {slot.name!r} must be both read and written")


def _slice(data: bytes, offset: int, size: int, label: str) -> bytes:
    end = offset + size
    if offset < 0 or size < 0 or end > len(data):
        raise ValueError(f"{label} is truncated")
    return data[offset:end]


def _elf_string(table: bytes, offset: int) -> str:
    if offset >= len(table):
        raise ValueError("ELF64 symbol name exceeds its string table")
    end = table.find(b"\0", offset)
    if end < 0:
        raise ValueError("ELF64 symbol name is not NUL terminated")
    return table[offset:end].decode()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
