# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical attribute keys and resource-derived constraint generation.

WellKnownAttribute defines the canonical string keys used as attribute names
in worker metadata and constraint matching.  All production code should
reference these enum members instead of raw string literals so that
typos are caught at import time rather than silently producing scheduling
mismatches.

Constraint helper functions (preemptible_constraint, region_constraint, etc.)
live in iris.cluster.types alongside the Constraint dataclass they return.

constraints_from_resources() auto-generates device constraints from a job's
ResourceSpecProto so that callers don't need to manually add device-type
and device-variant constraints that are already implied by the resource spec.
"""

from __future__ import annotations

from enum import StrEnum

from iris.rpc import cluster_pb2


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


def constraints_from_resources(resources: cluster_pb2.ResourceSpecProto) -> list:
    """Auto-generate device constraints from a job's resource spec.

    Produces Constraint objects for device-type and device-variant when the
    resource spec carries a non-CPU device.  CPU jobs get no auto-generated
    device constraints since CPU resources are fungible across all workers.

    The controller merges these with explicit user constraints using
    merge_constraints(), where explicit constraints for canonical keys replace
    auto-generated ones.

    Returns Constraint objects (typed as list to avoid forward-reference issues
    at module level; actual type is list[Constraint]).
    """
    # Late import to avoid circular dependency: types.py imports from this module
    from iris.cluster.types import (
        Constraint,
        ConstraintOp,
        get_device_type,
        get_device_variant,
    )

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
