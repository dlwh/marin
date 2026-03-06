# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ConstraintDescriptor registry and match functions."""

import pytest

from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    DeviceType,
    NormalizedConstraints,
    _match_device_type,
    _match_device_variant,
    _match_set_membership,
    merge_constraints,
    normalize_constraints,
)
from iris.rpc import cluster_pb2


def _eq_constraint(key: str, value: str) -> cluster_pb2.Constraint:
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_EQ)
    c.value.string_value = value
    return c


def _in_constraint(key: str, values: list[str]) -> cluster_pb2.Constraint:
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_IN)
    for v in values:
        av = c.values.add()
        av.string_value = v
    return c


# --- Match functions ---


def test_match_device_type_cpu_matches_all():
    """CPU demand routes to any group — this is a key business rule."""
    assert _match_device_type(DeviceType.GPU, DeviceType.CPU)
    assert _match_device_type(DeviceType.TPU, DeviceType.CPU)
    assert _match_device_type(DeviceType.CPU, DeviceType.CPU)


def test_match_device_type_gpu_requires_gpu():
    assert _match_device_type(DeviceType.GPU, DeviceType.GPU)
    assert not _match_device_type(DeviceType.TPU, DeviceType.GPU)
    assert not _match_device_type(DeviceType.CPU, DeviceType.GPU)


def test_match_device_variant_case_insensitive():
    assert _match_device_variant("H100", frozenset({"h100"}))
    assert _match_device_variant("h100", frozenset({"H100"}))


def test_match_device_variant_empty_matches_all():
    """Empty variant set means no preference — matches any group."""
    assert _match_device_variant("anything", frozenset())


def test_match_set_membership():
    assert _match_set_membership("us-central1", frozenset({"us-central1", "us-east1"}))
    assert not _match_set_membership("eu-west1", frozenset({"us-central1", "us-east1"}))


# --- Normalization: proto constraints → NormalizedConstraints ---


@pytest.mark.parametrize(
    "constraints, expected",
    [
        (
            [_eq_constraint("device-type", "gpu")],
            NormalizedConstraints(DeviceType.GPU, None, None, None, None),
        ),
        (
            [_eq_constraint("preemptible", "true")],
            NormalizedConstraints(None, None, True, None, None),
        ),
        (
            [_eq_constraint("region", "us-central1")],
            NormalizedConstraints(None, None, None, frozenset({"us-central1"}), None),
        ),
        (
            [_in_constraint("zone", ["us-central1-a", "us-central1-b"])],
            NormalizedConstraints(None, None, None, None, frozenset({"us-central1-a", "us-central1-b"})),
        ),
        (
            [
                _eq_constraint("device-type", "tpu"),
                _eq_constraint("device-variant", "v5litepod-16"),
                _eq_constraint("preemptible", "false"),
            ],
            NormalizedConstraints(DeviceType.TPU, frozenset({"v5litepod-16"}), False, None, None),
        ),
    ],
)
def test_normalize_constraints_parameterized(constraints, expected):
    result = normalize_constraints(constraints)
    assert result == expected


# --- merge_constraints canonical override ---


def test_merge_canonical_key_child_overrides_parent():
    parent = [Constraint(key="device-type", op=ConstraintOp.EQ, value="gpu")]
    child = [Constraint(key="device-type", op=ConstraintOp.EQ, value="tpu")]
    merged = merge_constraints(parent, child)
    assert len(merged) == 1
    assert merged[0].value == "tpu"


def test_merge_non_canonical_key_appends():
    parent = [Constraint(key="tpu-name", op=ConstraintOp.EQ, value="pod-a")]
    child = [Constraint(key="tpu-name", op=ConstraintOp.EQ, value="pod-b")]
    merged = merge_constraints(parent, child)
    assert len(merged) == 2
