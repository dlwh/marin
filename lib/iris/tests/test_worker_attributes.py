# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker_attributes_from_resources and local worker attribute propagation."""

import pytest

from iris.cluster.constraints import (
    WellKnownAttribute,
    accelerator_type_to_string,
    worker_attributes_from_resources,
)
from iris.rpc import config_pb2


@pytest.mark.parametrize(
    "accel_type,expected",
    [
        (config_pb2.ACCELERATOR_TYPE_CPU, "cpu"),
        (config_pb2.ACCELERATOR_TYPE_GPU, "gpu"),
        (config_pb2.ACCELERATOR_TYPE_TPU, "tpu"),
        (config_pb2.ACCELERATOR_TYPE_UNSPECIFIED, "cpu"),
    ],
)
def test_accelerator_type_to_string(accel_type: int, expected: str):
    assert accelerator_type_to_string(accel_type) == expected


def test_accelerator_type_to_string_unknown():
    with pytest.raises(ValueError, match="Unknown accelerator type"):
        accelerator_type_to_string(9999)


@pytest.mark.parametrize(
    "device_type,device_variant,preemptible,expected_attrs",
    [
        (
            config_pb2.ACCELERATOR_TYPE_TPU,
            "v4-8",
            True,
            {"device-type": "tpu", "device-variant": "v4-8", "preemptible": "true"},
        ),
        (
            config_pb2.ACCELERATOR_TYPE_GPU,
            "a100",
            False,
            {"device-type": "gpu", "device-variant": "a100", "preemptible": "false"},
        ),
        (
            config_pb2.ACCELERATOR_TYPE_CPU,
            "",
            False,
            {"device-type": "cpu", "preemptible": "false"},
        ),
    ],
    ids=["tpu", "gpu", "cpu"],
)
def test_worker_attributes_from_resources(
    device_type: int,
    device_variant: str,
    preemptible: bool,
    expected_attrs: dict[str, str],
):
    resources = config_pb2.ScaleGroupResources(
        device_type=device_type,
        device_variant=device_variant,
        preemptible=preemptible,
    )
    attrs = worker_attributes_from_resources(resources)
    assert attrs == expected_attrs


def test_local_autoscaler_worker_attributes():
    """Verify create_local_autoscaler populates well-known attributes from resources."""
    from iris.cluster.config import make_local_config
    from iris.cluster.controller.local import create_local_autoscaler
    from iris.cluster.platform.local import LocalPlatform

    config = config_pb2.IrisClusterConfig()
    sg = config_pb2.ScaleGroupConfig(
        name="test-tpu",
        min_slices=0,
        max_slices=1,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            device_type=config_pb2.ACCELERATOR_TYPE_TPU,
            device_variant="v4-8",
            preemptible=True,
        ),
    )
    config.scale_groups["test-tpu"].CopyFrom(sg)
    config = make_local_config(config)

    autoscaler, temp_dir = create_local_autoscaler(config, "http://127.0.0.1:9999")
    try:
        platform = autoscaler._platform
        assert isinstance(platform, LocalPlatform)
        attrs = platform._worker_attributes_by_group
        assert "test-tpu" in attrs
        assert attrs["test-tpu"][WellKnownAttribute.DEVICE_TYPE] == "tpu"
        assert attrs["test-tpu"][WellKnownAttribute.DEVICE_VARIANT] == "v4-8"
        assert attrs["test-tpu"][WellKnownAttribute.PREEMPTIBLE] == "true"
    finally:
        temp_dir.cleanup()
