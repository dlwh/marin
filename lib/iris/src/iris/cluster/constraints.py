# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constraint types and helpers for Iris resource scheduling.

This module is the canonical home for all constraint-related types:

- WellKnownAttribute: canonical string keys for worker metadata
- AttributeValue, ConstraintOp, Constraint: core constraint dataclasses
- DeviceType and device-config helpers (get_device_type, get_device_variant, etc.)
- NormalizedConstraints and extraction functions for demand routing
- Constraint factory functions (preemptible_constraint, region_constraint, etc.)
- constraints_from_resources: auto-generates device constraints from ResourceSpecProto

All production code should reference WellKnownAttribute enum members instead of
raw string literals so that typos are caught at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from typing import Any, ClassVar

from iris.rpc import cluster_pb2, config_pb2


class WellKnownAttribute(StrEnum):
    """Canonical attribute keys for constraint-based scheduling."""

    DEVICE_TYPE = "device-type"
    DEVICE_VARIANT = "device-variant"
    PREEMPTIBLE = "preemptible"
    REGION = "region"
    ZONE = "zone"
    TPU_NAME = "tpu-name"
    TPU_WORKER_ID = "tpu-worker-id"
    TPU_TOPOLOGY = "tpu-topology"
    TPU_VM_COUNT = "tpu-vm-count"
    GPU_VARIANT = "gpu-variant"
    GPU_COUNT = "gpu-count"


# ---------------------------------------------------------------------------
# Step 1 types: core constraint primitives (depend only on cluster_pb2)
# ---------------------------------------------------------------------------


class DeviceType(Enum):
    """Device type for demand routing."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


def get_device_type_enum(device: cluster_pb2.DeviceConfig) -> DeviceType:
    """Extract device type as enum from DeviceConfig."""
    if device.HasField("gpu"):
        return DeviceType.GPU
    if device.HasField("tpu"):
        return DeviceType.TPU
    return DeviceType.CPU


def get_device_type(device: cluster_pb2.DeviceConfig) -> str:
    """Extract device type from DeviceConfig."""
    if device.HasField("cpu"):
        return "cpu"
    if device.HasField("gpu"):
        return "gpu"
    if device.HasField("tpu"):
        return "tpu"
    return "cpu"


def get_device_variant(device: cluster_pb2.DeviceConfig) -> str | None:
    """Extract device variant (e.g., GPU model) from DeviceConfig."""
    if device.HasField("gpu"):
        return device.gpu.variant if device.gpu.variant else None
    if device.HasField("tpu"):
        return device.tpu.variant if device.tpu.variant else None
    return None


@dataclass(frozen=True)
class AttributeValue:
    """Typed attribute value for worker attributes and constraint matching.

    Used for coscheduling and constraint-based worker filtering.
    Values can be strings, integers, or floats.
    """

    value: str | int | float

    def to_proto(self) -> cluster_pb2.AttributeValue:
        """Convert to protobuf representation."""
        proto = cluster_pb2.AttributeValue()
        if isinstance(self.value, str):
            proto.string_value = self.value
        elif isinstance(self.value, int):
            proto.int_value = self.value
        elif isinstance(self.value, float):
            proto.float_value = self.value
        return proto

    @staticmethod
    def from_proto(proto: cluster_pb2.AttributeValue) -> AttributeValue:
        """Convert from protobuf representation."""
        if proto.HasField("string_value"):
            return AttributeValue(proto.string_value)
        elif proto.HasField("int_value"):
            return AttributeValue(proto.int_value)
        elif proto.HasField("float_value"):
            return AttributeValue(proto.float_value)
        # Default to empty string if no value set
        return AttributeValue("")


class ConstraintOp(IntEnum):
    """Constraint operators for worker attribute matching.

    Used to define constraints that filter which workers can run a job.
    Each operator compares a worker attribute against a constraint value.

    Example:
        >>> # Match workers where region equals "us-central1"
        >>> Constraint(key="region", op=ConstraintOp.EQ, value="us-central1")
        >>> # Match workers with memory > 32GB
        >>> Constraint(key="memory_gb", op=ConstraintOp.GT, value=32)
        >>> # Match workers that have the "gpu" attribute set
        >>> Constraint(key="gpu", op=ConstraintOp.EXISTS)
    """

    EQ = 0
    NE = 1
    EXISTS = 2
    NOT_EXISTS = 3
    GT = 4
    GE = 5
    LT = 6
    LE = 7
    IN = 8

    def to_proto(self) -> cluster_pb2.ConstraintOp:
        """Convert to protobuf ConstraintOp enum value."""
        mapping = {
            ConstraintOp.EQ: cluster_pb2.CONSTRAINT_OP_EQ,
            ConstraintOp.NE: cluster_pb2.CONSTRAINT_OP_NE,
            ConstraintOp.EXISTS: cluster_pb2.CONSTRAINT_OP_EXISTS,
            ConstraintOp.NOT_EXISTS: cluster_pb2.CONSTRAINT_OP_NOT_EXISTS,
            ConstraintOp.GT: cluster_pb2.CONSTRAINT_OP_GT,
            ConstraintOp.GE: cluster_pb2.CONSTRAINT_OP_GE,
            ConstraintOp.LT: cluster_pb2.CONSTRAINT_OP_LT,
            ConstraintOp.LE: cluster_pb2.CONSTRAINT_OP_LE,
            ConstraintOp.IN: cluster_pb2.CONSTRAINT_OP_IN,
        }
        return mapping[self]


@dataclass(frozen=True)
class Constraint:
    """Worker constraint for job scheduling.

    Constraints filter which workers are eligible to run a job based on
    worker attributes. Workers must satisfy all constraints to be considered.

    Example:
        >>> # Require a specific TPU pod
        >>> Constraint(key="tpu-name", op=ConstraintOp.EQ, value="my-tpu-pod")
        >>> # Require workers in a specific zone
        >>> Constraint(key="zone", op=ConstraintOp.EQ, value="us-central1-a")
        >>> # Require workers with at least 64GB memory
        >>> Constraint(key="memory_gb", op=ConstraintOp.GE, value=64)
        >>> # Require workers that have a GPU
        >>> Constraint(key="gpu", op=ConstraintOp.EXISTS)
        >>> # Require workers in one of several regions
        >>> Constraint(key="region", op=ConstraintOp.IN, values=("us-central1", "us-central2"))
    """

    key: str
    op: ConstraintOp
    value: str | int | float | None = None
    values: tuple[str | int | float, ...] | None = None

    def to_proto(self) -> cluster_pb2.Constraint:
        """Convert to protobuf representation."""
        proto = cluster_pb2.Constraint(key=self.key, op=self.op.to_proto())
        if self.value is not None:
            proto.value.CopyFrom(AttributeValue(self.value).to_proto())
        if self.values is not None:
            for v in self.values:
                proto.values.append(AttributeValue(v).to_proto())
        return proto

    @staticmethod
    def from_proto(proto: cluster_pb2.Constraint) -> Constraint:
        """Convert from protobuf representation."""
        op = ConstraintOp(proto.op)
        value: str | int | float | None = None
        if proto.HasField("value"):
            value = AttributeValue.from_proto(proto.value).value
        values: tuple[str | int | float, ...] | None = None
        if proto.values:
            values = tuple(AttributeValue.from_proto(v).value for v in proto.values)
        return Constraint(key=proto.key, op=op, value=value, values=values)


# ---------------------------------------------------------------------------
# Step 2 types: constraint helpers (depend on WellKnownAttribute)
# ---------------------------------------------------------------------------


def preemptible_constraint(preemptible: bool = True) -> Constraint:
    """Constraint requiring workers to be preemptible (or not)."""
    return Constraint(key=WellKnownAttribute.PREEMPTIBLE, op=ConstraintOp.EQ, value=str(preemptible).lower())


def zone_constraint(zone: str) -> Constraint:
    """Constraint requiring workers to be in a given zone."""
    if not zone:
        raise ValueError("zone must be non-empty")
    return Constraint(key=WellKnownAttribute.ZONE, op=ConstraintOp.EQ, value=zone)


def region_constraint(regions: list[str]) -> Constraint:
    """Constraint requiring workers to be in one of the given regions.

    Emits an EQ constraint for a single region or an IN constraint for multiple
    regions.

    Args:
        regions: Non-empty list of region strings. Must be a list, not a bare string.

    Raises:
        TypeError: If regions is a string (common mistake — pass [region] instead).
        ValueError: If regions is empty or contains empty strings.
    """
    if isinstance(regions, str):
        raise TypeError("region_constraint() requires a list of strings, not a bare string. Use [region] instead.")
    if not regions:
        raise ValueError("regions must be non-empty")
    for r in regions:
        if not r:
            raise ValueError("region must be non-empty")
    if len(regions) == 1:
        return Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=regions[0])
    return Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=tuple(regions))


def device_variant_constraint(variants: Sequence[str]) -> Constraint:
    """Constraint requiring scheduling on workers with one of the given device variants.

    Args:
        variants: Non-empty sequence of device variant strings (e.g., ["v4-8", "v5p-8"]).

    Raises:
        TypeError: If variants is a string (common mistake — pass [variant] instead).
        ValueError: If variants is empty or contains empty strings.
    """
    if isinstance(variants, str):
        raise TypeError(
            "device_variant_constraint() requires a sequence of strings, not a bare string. Use [variant] instead."
        )
    if not variants:
        raise ValueError("variants must be non-empty")
    for v in variants:
        if not v:
            raise ValueError("variant must be non-empty")
    if len(variants) == 1:
        return Constraint(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.EQ, value=variants[0])
    return Constraint(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.IN, values=tuple(variants))


@dataclass(frozen=True)
class NormalizedConstraints:
    """Normalized canonical placement constraints derived from proto constraints.

    Combines device type, device variant, preemptible preference, and
    region/zone requirements into a single object for demand routing.
    The autoscaler uses this instead of carrying separate fields.
    """

    device_type: DeviceType | None
    device_variants: frozenset[str] | None
    preemptible: bool | None
    required_regions: frozenset[str] | None
    required_zones: frozenset[str] | None

    _KEY_TO_FIELD: ClassVar[dict[str, str]] = {
        "device-type": "device_type",
        "device-variant": "device_variants",
        "preemptible": "preemptible",
        "region": "required_regions",
        "zone": "required_zones",
    }

    def get(self, key: str) -> Any:
        """Look up a routing constraint value by its well-known key."""
        field_name = self._KEY_TO_FIELD.get(key)
        if field_name is None:
            return None
        return getattr(self, field_name)


def preemptible_preference_from_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> bool | None:
    """Extract preemptible preference from constraints.

    Returns:
        True if explicitly required preemptible workers, False if explicitly
        requiring non-preemptible workers, or None if unspecified.

    Raises:
        ValueError: If multiple conflicting preemptible constraints are present
            or an invalid non-boolean value is used.
    """
    values: set[bool] = set()
    for constraint in constraints:
        if constraint.key != WellKnownAttribute.PREEMPTIBLE:
            continue
        if constraint.op != cluster_pb2.CONSTRAINT_OP_EQ:
            raise ValueError("preemptible constraint must use EQ")
        if not constraint.value.HasField("string_value"):
            raise ValueError("preemptible constraint requires string value")
        raw = constraint.value.string_value.strip().lower()
        if raw == "true":
            values.add(True)
        elif raw == "false":
            values.add(False)
        else:
            raise ValueError("preemptible constraint must be 'true' or 'false'")

    if len(values) > 1:
        raise ValueError("conflicting preemptible constraints")
    return next(iter(values)) if values else None


def required_regions_from_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> frozenset[str] | None:
    """Extract required regions from constraints.

    Returns:
        Set of required regions when specified, otherwise None.

    Raises:
        ValueError: If region constraints use invalid operators/values or contain
            conflicting EQ values.
    """
    regions: set[str] = set()
    has_in = False
    for constraint in constraints:
        if constraint.key != WellKnownAttribute.REGION:
            continue
        if constraint.op == cluster_pb2.CONSTRAINT_OP_IN:
            if not constraint.values:
                raise ValueError("IN region constraint requires at least one value")
            for av in constraint.values:
                if not av.HasField("string_value"):
                    raise ValueError("region constraint requires string value")
                region = av.string_value.strip()
                if not region:
                    raise ValueError("region constraint must be non-empty")
                regions.add(region)
            has_in = True
        elif constraint.op == cluster_pb2.CONSTRAINT_OP_EQ:
            if not constraint.value.HasField("string_value"):
                raise ValueError("region constraint requires string value")
            region = constraint.value.string_value.strip()
            if not region:
                raise ValueError("region constraint must be non-empty")
            regions.add(region)
        else:
            raise ValueError(f"region constraint must use EQ or IN, got {constraint.op}")

    if not has_in and len(regions) > 1:
        raise ValueError("conflicting region constraints")
    return frozenset(regions) if regions else None


def required_zones_from_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> frozenset[str] | None:
    """Extract required zones from constraints.

    Returns:
        Set of required zones when specified, otherwise None.

    Raises:
        ValueError: If zone constraints use invalid operators/values or contain
            conflicting EQ values.
    """
    zones: set[str] = set()
    has_in = False
    for constraint in constraints:
        if constraint.key != WellKnownAttribute.ZONE:
            continue
        if constraint.op == cluster_pb2.CONSTRAINT_OP_IN:
            if not constraint.values:
                raise ValueError("IN zone constraint requires at least one value")
            for av in constraint.values:
                if not av.HasField("string_value"):
                    raise ValueError("zone constraint requires string value")
                zone = av.string_value.strip()
                if not zone:
                    raise ValueError("zone constraint must be non-empty")
                zones.add(zone)
            has_in = True
        elif constraint.op == cluster_pb2.CONSTRAINT_OP_EQ:
            if not constraint.value.HasField("string_value"):
                raise ValueError("zone constraint requires string value")
            zone = constraint.value.string_value.strip()
            if not zone:
                raise ValueError("zone constraint must be non-empty")
            zones.add(zone)
        else:
            raise ValueError(f"zone constraint must use EQ or IN, got {constraint.op}")

    if not has_in and len(zones) > 1:
        raise ValueError("conflicting zone constraints")
    return frozenset(zones) if zones else None


def device_type_from_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> DeviceType | None:
    """Extract device type from constraints.

    Returns:
        DeviceType when a device-type constraint is present, otherwise None.

    Raises:
        ValueError: If device-type constraints use invalid operators/values or
            contain conflicting values.
    """
    values: set[str] = set()
    for constraint in constraints:
        if constraint.key != WellKnownAttribute.DEVICE_TYPE:
            continue
        if constraint.op == cluster_pb2.CONSTRAINT_OP_EQ:
            if not constraint.value.HasField("string_value"):
                raise ValueError("device-type constraint requires string value")
            values.add(constraint.value.string_value.strip().lower())
        elif constraint.op == cluster_pb2.CONSTRAINT_OP_IN:
            if not constraint.values:
                raise ValueError("IN device-type constraint requires at least one value")
            for av in constraint.values:
                if not av.HasField("string_value"):
                    raise ValueError("device-type constraint requires string value")
                values.add(av.string_value.strip().lower())
        else:
            raise ValueError(f"device-type constraint must use EQ or IN, got {constraint.op}")

    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"conflicting device-type constraints: {values}")

    raw = next(iter(values))
    try:
        return DeviceType(raw)
    except ValueError as e:
        raise ValueError(f"unknown device type: {raw}") from e


def required_device_variants_from_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> frozenset[str] | None:
    """Extract required device variants from constraints.

    Returns:
        Set of required device variants when specified, otherwise None.

    Raises:
        ValueError: If device-variant constraints use invalid operators/values.
    """
    variants: set[str] = set()
    for constraint in constraints:
        if constraint.key != WellKnownAttribute.DEVICE_VARIANT:
            continue
        if constraint.op == cluster_pb2.CONSTRAINT_OP_IN:
            if not constraint.values:
                raise ValueError("IN device-variant constraint requires at least one value")
            for av in constraint.values:
                if not av.HasField("string_value"):
                    raise ValueError("device-variant constraint requires string value")
                variant = av.string_value.strip()
                if not variant:
                    raise ValueError("device-variant constraint must be non-empty")
                variants.add(variant)
        elif constraint.op == cluster_pb2.CONSTRAINT_OP_EQ:
            if not constraint.value.HasField("string_value"):
                raise ValueError("device-variant constraint requires string value")
            variant = constraint.value.string_value.strip()
            if not variant:
                raise ValueError("device-variant constraint must be non-empty")
            variants.add(variant)
        else:
            raise ValueError(f"device-variant constraint must use EQ or IN, got {constraint.op}")
    return frozenset(variants) if variants else None


def normalize_constraints(constraints: Sequence[cluster_pb2.Constraint]) -> NormalizedConstraints:
    """Normalize canonical placement constraints from protobuf constraints."""
    return NormalizedConstraints(
        device_type=device_type_from_constraints(constraints),
        device_variants=required_device_variants_from_constraints(constraints),
        preemptible=preemptible_preference_from_constraints(constraints),
        required_regions=required_regions_from_constraints(constraints),
        required_zones=required_zones_from_constraints(constraints),
    )


def merge_constraints(parent: Sequence[Constraint], child: Sequence[Constraint]) -> list[Constraint]:
    """Merge parent and child constraints with canonical-key override semantics."""

    merged_by_key: dict[str, list[Constraint]] = {}
    for constraint in parent:
        merged_by_key.setdefault(constraint.key, []).append(constraint)

    _CANONICAL_KEYS = frozenset(d.key for d in CONSTRAINT_REGISTRY.values() if d.canonical)
    for key in _CANONICAL_KEYS:
        child_for_key = [constraint for constraint in child if constraint.key == key]
        if child_for_key:
            merged_by_key[key] = child_for_key

    for constraint in child:
        if constraint.key in _CANONICAL_KEYS:
            continue
        existing = merged_by_key.setdefault(constraint.key, [])
        if constraint not in existing:
            existing.append(constraint)

    result: list[Constraint] = []
    for constraints_for_key in merged_by_key.values():
        result.extend(constraints_for_key)
    return result


# ---------------------------------------------------------------------------
# ConstraintDescriptor registry
# ---------------------------------------------------------------------------


class ConstraintKind(StrEnum):
    """Whether a constraint is a label match or a capacity-deducted resource."""

    TAG = "tag"
    CONSUMABLE = "consumable"


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Single source of truth for a well-known constraint.

    Each well-known attribute gets one descriptor that declares its type,
    allowed operators, and (for routing constraints) how to match a scaling
    group's value against a requested value.
    """

    key: str
    kind: ConstraintKind
    python_type: type
    allowed_ops: frozenset[int]
    canonical: bool
    routing: bool
    extract: Callable[..., Any] | None
    match: Callable[[Any, Any], bool] | None


# --- Match functions for routing descriptors ---


def _match_device_type(group_val: DeviceType, requested: DeviceType) -> bool:
    if requested == DeviceType.CPU:
        return True
    return group_val == requested


def _match_device_variant(group_val: str, requested: frozenset[str]) -> bool:
    if not requested:
        return True
    return group_val.lower() in {v.lower() for v in requested}


def _match_preemptible(group_val: bool, requested: bool) -> bool:
    return group_val == requested


def _match_set_membership(group_val: str, requested: frozenset[str]) -> bool:
    return group_val in requested


# --- Extract functions ---


def _extract_string(constraint: cluster_pb2.Constraint) -> str:
    return constraint.value.string_value.strip()


def _extract_string_lower(constraint: cluster_pb2.Constraint) -> str:
    return constraint.value.string_value.strip().lower()


def _extract_bool_string(constraint: cluster_pb2.Constraint) -> str:
    return constraint.value.string_value.strip().lower()


def _extract_int(constraint: cluster_pb2.Constraint) -> int:
    return constraint.value.int_value


_EQ_IN = frozenset({cluster_pb2.CONSTRAINT_OP_EQ, cluster_pb2.CONSTRAINT_OP_IN})
_EQ_ONLY = frozenset({cluster_pb2.CONSTRAINT_OP_EQ})
_ALL_OPS = frozenset(
    {
        cluster_pb2.CONSTRAINT_OP_EQ,
        cluster_pb2.CONSTRAINT_OP_NE,
        cluster_pb2.CONSTRAINT_OP_EXISTS,
        cluster_pb2.CONSTRAINT_OP_NOT_EXISTS,
        cluster_pb2.CONSTRAINT_OP_GT,
        cluster_pb2.CONSTRAINT_OP_GE,
        cluster_pb2.CONSTRAINT_OP_LT,
        cluster_pb2.CONSTRAINT_OP_LE,
        cluster_pb2.CONSTRAINT_OP_IN,
    }
)


CONSTRAINT_REGISTRY: dict[str, ConstraintDescriptor] = {}


def _register(desc: ConstraintDescriptor) -> ConstraintDescriptor:
    CONSTRAINT_REGISTRY[desc.key] = desc
    return desc


_register(
    ConstraintDescriptor(
        key="device-type",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_EQ_IN,
        canonical=True,
        routing=True,
        extract=_extract_string_lower,
        match=_match_device_type,
    )
)
_register(
    ConstraintDescriptor(
        key="device-variant",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_EQ_IN,
        canonical=True,
        routing=True,
        extract=_extract_string,
        match=_match_device_variant,
    )
)
_register(
    ConstraintDescriptor(
        key="preemptible",
        kind=ConstraintKind.TAG,
        python_type=bool,
        allowed_ops=_EQ_ONLY,
        canonical=True,
        routing=True,
        extract=_extract_bool_string,
        match=_match_preemptible,
    )
)
_register(
    ConstraintDescriptor(
        key="region",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_EQ_IN,
        canonical=True,
        routing=True,
        extract=_extract_string,
        match=_match_set_membership,
    )
)
_register(
    ConstraintDescriptor(
        key="zone",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_EQ_IN,
        canonical=True,
        routing=True,
        extract=_extract_string,
        match=_match_set_membership,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-name",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_string,
        match=None,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-worker-id",
        kind=ConstraintKind.TAG,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_int,
        match=None,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-topology",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_string,
        match=None,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-vm-count",
        kind=ConstraintKind.TAG,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_int,
        match=None,
    )
)
_register(
    ConstraintDescriptor(
        key="gpu-variant",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_string,
        match=None,
    )
)
_register(
    ConstraintDescriptor(
        key="gpu-count",
        kind=ConstraintKind.CONSUMABLE,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
        extract=_extract_int,
        match=None,
    )
)


def routing_descriptors() -> list[ConstraintDescriptor]:
    """Return all routing descriptors in registry insertion order."""
    return [d for d in CONSTRAINT_REGISTRY.values() if d.routing]


# ---------------------------------------------------------------------------
# Resource-derived constraint generation
# ---------------------------------------------------------------------------


def constraints_from_resources(resources: cluster_pb2.ResourceSpecProto) -> list[Constraint]:
    """Auto-generate device constraints from a job's resource spec.

    Produces Constraint objects for device-type and device-variant when the
    resource spec carries a non-CPU device.  CPU jobs get no auto-generated
    device constraints since CPU resources are fungible across all workers.

    The controller merges these with explicit user constraints using
    merge_constraints(), where explicit constraints for canonical keys replace
    auto-generated ones.
    """
    constraints: list[Constraint] = []

    if not resources.HasField("device"):
        return constraints

    device_type = get_device_type(resources.device)
    if device_type != "cpu":
        constraints.append(
            Constraint(
                key=WellKnownAttribute.DEVICE_TYPE,
                op=ConstraintOp.EQ,
                value=device_type,
            )
        )

    variant = get_device_variant(resources.device)
    if variant and variant != "auto":
        constraints.append(
            Constraint(
                key=WellKnownAttribute.DEVICE_VARIANT,
                op=ConstraintOp.EQ,
                value=variant,
            )
        )

    return constraints


def accelerator_type_to_string(accel_type: int) -> str:
    """Convert AcceleratorType proto enum value to a scheduling string."""
    if accel_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
        return "cpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_CPU:
        return "cpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_GPU:
        return "gpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_TPU:
        return "tpu"
    raise ValueError(f"Unknown accelerator type: {accel_type}")


def worker_attributes_from_resources(resources: config_pb2.ScaleGroupResources) -> dict[str, str]:
    """Derive well-known worker attributes from scale group resources config.

    This ensures local workers advertise the same device-type, device-variant,
    and preemptible attributes that constraint matching expects.
    """
    attrs: dict[str, str] = {}
    attrs[WellKnownAttribute.DEVICE_TYPE] = accelerator_type_to_string(resources.device_type)
    if resources.device_variant:
        attrs[WellKnownAttribute.DEVICE_VARIANT] = resources.device_variant
    attrs[WellKnownAttribute.PREEMPTIBLE] = str(resources.preemptible).lower()
    return attrs
