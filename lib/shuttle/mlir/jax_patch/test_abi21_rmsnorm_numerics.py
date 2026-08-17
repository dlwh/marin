# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from abi15_h100_numerics import bf16_sha256
from abi21_rmsnorm_numerics import (
    FEATURE_COUNT,
    ROW_COUNT,
    architecture_inputs,
    binary64_reference,
    error_metrics,
    schedule_reference,
)
from abi21_rmsnorm_prehardware_contract import load_contract


def test_schedule_reference_recomputes_all_public_roles_from_sealed_inputs() -> None:
    inputs = architecture_inputs()
    outputs = schedule_reference(inputs)
    contract = load_contract()

    assert len(inputs.x) == ROW_COUNT * FEATURE_COUNT
    assert len(inputs.weight) == FEATURE_COUNT
    assert len(inputs.output_cotangent) == ROW_COUNT * FEATURE_COUNT
    assert len(outputs.result) == ROW_COUNT * FEATURE_COUNT
    assert len(outputs.input_cotangent) == ROW_COUNT * FEATURE_COUNT
    assert len(outputs.weight_cotangent) == FEATURE_COUNT
    assert contract.input_digests == {
        "x": bf16_sha256(inputs.x),
        "weight": bf16_sha256(inputs.weight),
        "output_cotangent": bf16_sha256(inputs.output_cotangent),
    }
    assert contract.reference_output_digests == {
        "result": bf16_sha256(outputs.result),
        "input_cotangent": bf16_sha256(outputs.input_cotangent),
        "weight_cotangent": bf16_sha256(outputs.weight_cotangent),
    }


def test_schedule_reference_detects_one_bit_input_drift() -> None:
    inputs = architecture_inputs()
    changed_x = (inputs.x[0] ^ 1, *inputs.x[1:])

    changed = schedule_reference(replace(inputs, x=changed_x))
    baseline = schedule_reference(inputs)

    assert bf16_sha256(changed.result) != bf16_sha256(baseline.result)
    assert bf16_sha256(changed.input_cotangent) != bf16_sha256(baseline.input_cotangent)


def test_schedule_reference_keeps_finite_bf16_public_outputs() -> None:
    outputs = schedule_reference(architecture_inputs())

    for values in (outputs.result, outputs.input_cotangent, outputs.weight_cotangent):
        assert values
        assert all(value & 0x7F80 != 0x7F80 for value in values)


def test_schedule_reference_satisfies_independent_binary64_role_limits() -> None:
    inputs = architecture_inputs()
    schedule = schedule_reference(inputs)
    analytic = binary64_reference(inputs)
    limits = {
        "result": (0.0078125, 0.002, 0.00390625, 1),
        "input_cotangent": (0.0078125, 0.002, 0.00390625, 1),
        "weight_cotangent": (0.03125, 0.015625, 0.00390625, 1),
    }

    for role, expected_limits in limits.items():
        metrics = error_metrics(getattr(schedule, role), getattr(analytic, role))
        assert metrics.maximum_absolute_error <= expected_limits[0]
        assert metrics.mean_absolute_error <= expected_limits[1]
        assert metrics.relative_linf_error <= expected_limits[2]
        assert metrics.maximum_bf16_ulp_error <= expected_limits[3]
