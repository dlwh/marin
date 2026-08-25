# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import jax
import numpy as np
import pytest
import track1_generic_composition_h100_subject as subject
from track1_generic_composition_numerical_policy import (
    CASES,
    POLICY_SHA256,
    SEMANTIC_CONTRACT,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    load_numerical_policy,
)
from track1_map_h100_prototype import validate_generic_composition_result


def _snapshot(mode: subject.Mode, invocation: int) -> tuple[tuple[object, ...], ...]:
    options_sha256, tuning_sha256 = subject.observer_identity(mode)
    membership = (
        "[[#shuttle.source_ref<0, 0, 0, 0>, "
        "#shuttle.source_ref<0, 0, 1, 0>, "
        "#shuttle.source_ref<0, 0, 2, 0>, "
        "#shuttle.source_ref<0, 0, 3, 0>, "
        "#shuttle.source_ref<0, 0, 4, 0>, "
        "#shuttle.source_ref<0, 0, 5, 0>]]"
    )
    manifest = f"selected_regions = {membership}"
    return (
        (
            invocation,
            "algebra_coverage",
            mode.value,
            options_sha256,
            tuning_sha256,
            membership,
            manifest,
            "a" * 64,
            "",
            False,
            "",
            0,
            0,
        ),
        (
            invocation,
            "generated_bundle",
            mode.value,
            options_sha256,
            tuning_sha256,
            membership,
            manifest,
            "a" * 64,
            "b" * 64,
            False,
            "",
            2,
            2,
        ),
        (
            invocation,
            "final_erasure",
            mode.value,
            options_sha256,
            tuning_sha256,
            "",
            "",
            "",
            "c" * 64,
            True,
            "",
            0,
            0,
        ),
    )


def _result_document() -> dict[str, object]:
    results = []
    invocation = 1
    for case in CASES:
        inputs = deterministic_inputs(case)
        actual = binary64_oracle(*inputs).astype(np.float32)
        _, stablehlo_sha256 = subject.stablehlo_source(case, inputs)
        for mode in subject.Mode:
            snapshot = () if mode is subject.Mode.DISABLED else _snapshot(mode, invocation)
            result = subject.validate_coordinate(
                case,
                mode,
                actual,
                actual.copy(),
                snapshot,
                stablehlo_sha256,
            )
            results.append(result)
            if snapshot:
                invocation += 1
    return subject.result_document(tuple(results))


def test_policy_binds_two_ordinary_jax_shapes_and_independent_binary64_oracle() -> None:
    document, cases = load_numerical_policy()

    assert document["semantic_contract"] == SEMANTIC_CONTRACT
    assert [case.case_id for case in cases] == ["rank2-a", "rank3-b"]
    for case in cases:
        inputs = deterministic_inputs(case)
        source, digest = subject.stablehlo_source(case, inputs)
        assert digest == case.stablehlo_sha256
        assert source.count("stablehlo.dot_general") == 2
        assert source.count("stablehlo.reduce") == 1
        assert source.count("stablehlo.multiply") == 1
        actual = np.asarray(jax.jit(subject.composition)(*map(jax.device_put, inputs)))
        validated = subject.validate_coordinate(
            case,
            subject.Mode.DISABLED,
            actual,
            actual.copy(),
            (),
            digest,
        )
        assert validated.result_sha256 == array_sha256(actual)
        assert all(validated.metrics[name] <= limit for name, limit in case.limits.items())


def test_coordinate_rejects_public_abi_repeat_full_write_and_observer_drift() -> None:
    case = CASES[0]
    inputs = deterministic_inputs(case)
    actual = binary64_oracle(*inputs).astype(np.float32)
    _, stablehlo_sha256 = subject.stablehlo_source(case, inputs)
    snapshot = _snapshot(subject.Mode.SOURCE_ORDERED, 7)

    accepted = subject.validate_coordinate(
        case,
        subject.Mode.SOURCE_ORDERED,
        actual,
        actual.copy(),
        snapshot,
        stablehlo_sha256,
    )
    assert accepted.observer is not None
    assert accepted.observer.invocation_id == 7

    wrong_shape = actual.reshape(-1)
    with pytest.raises(RuntimeError, match="public result ABI"):
        subject.validate_coordinate(
            case,
            subject.Mode.SOURCE_ORDERED,
            wrong_shape,
            wrong_shape.copy(),
            snapshot,
            stablehlo_sha256,
        )
    repeated = actual.copy()
    repeated.flat[0] = np.nextafter(repeated.flat[0], np.float32(np.inf))
    with pytest.raises(RuntimeError, match="repeat result drift"):
        subject.validate_coordinate(
            case,
            subject.Mode.SOURCE_ORDERED,
            actual,
            repeated,
            snapshot,
            stablehlo_sha256,
        )
    unwritten = np.zeros_like(actual)
    with pytest.raises(RuntimeError, match="maximum_absolute_error"):
        subject.validate_coordinate(
            case,
            subject.Mode.SOURCE_ORDERED,
            unwritten,
            unwritten.copy(),
            snapshot,
            stablehlo_sha256,
        )
    with pytest.raises(ValueError, match="source attribution"):
        subject.validate_observer_snapshot(
            subject.Mode.SOURCE_ORDERED,
            tuple((*event[:5], "one source", *event[6:]) for event in snapshot),
        )


def test_threshold_failure_reports_bounded_coordinate_before_rejection() -> None:
    case = CASES[0]
    actual = binary64_oracle(*deterministic_inputs(case)).astype(np.float32)
    actual.flat[0] += np.float32(0.01)
    reports: list[dict[str, object]] = []

    with pytest.raises(subject.CoordinateThresholdError, match="maximum_absolute_error"):
        subject.validate_coordinate(
            case,
            subject.Mode.SOURCE_ORDERED,
            actual,
            actual.copy(),
            _snapshot(subject.Mode.SOURCE_ORDERED, 17),
            case.stablehlo_sha256,
            device=subject.DeviceEvidence(backend="gpu", count=1, kind="NVIDIA H100"),
            report_cell=reports.append,
        )

    assert len(reports) == 1
    report = reports[0]
    assert report["schema"] == "track1_generic_composition_h100_cell"
    assert report["schema_version"] == 1
    assert report["case_id"] == "rank2-a"
    assert report["mode"] == "source_ordered"
    assert report["shapes"] == {"a": [2, 3], "b": [3, 5], "init": [], "p": [5, 7], "y": [2, 7]}
    assert report["semantic_groups"] == SEMANTIC_CONTRACT["groups"]
    assert report["private_boundary"] == [
        {
            "consumer": "map_contract",
            "dtype": "f32",
            "name": "h",
            "producer": "contract_fold",
            "shape": [2, 5],
        },
        {
            "consumer": "map_contract",
            "dtype": "f32",
            "name": "s",
            "producer": "contract_fold",
            "shape": [2],
        },
    ]
    assert report["threshold_exceeded"] == ["maximum_absolute_error", "mean_absolute_error"]
    assert set(report["metrics"]) == {
        "masked_bf16_absolute_floor",
        "maximum_absolute_error",
        "maximum_masked_bf16_ulp_error",
        "maximum_raw_bf16_ulp_error",
        "mean_absolute_error",
        "relative_linf_error",
    }
    assert report["first_mismatch"]["index"] == [0, 0]
    assert report["worst_absolute"]["index"] == [0, 0]
    assert set(report["worst_bf16_ulp"]) == set(report["worst_absolute"])
    assert set(report["output_digests"]) == {"actual", "reference_binary64", "reference_f32", "repeat"}
    assert all(len(digest.split(":")) == 8 for digest in report["output_digests"].values())
    assert report["observer"]["status"] == "selected"
    assert report["observer"]["invocation_id"] == 17
    assert report["device"] == {
        "actual_complete": True,
        "backend": "gpu",
        "h100": True,
        "repeat_complete": True,
        "visible_count": 1,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert len(payload) < subject.MAX_CELL_REPORT_BYTES


def test_run_matrix_reports_every_coordinate_before_aggregate_threshold_failure(monkeypatch) -> None:
    observed: list[tuple[str, subject.Mode]] = []
    reports: list[dict[str, object]] = []
    next_invocation = 1

    def execute(case, mode):
        nonlocal next_invocation
        observed.append((case.case_id, mode))
        actual = binary64_oracle(*deterministic_inputs(case)).astype(np.float32)
        if case is CASES[0] and mode is subject.Mode.DISABLED:
            actual.flat[0] += np.float32(0.01)
        snapshot = () if mode is subject.Mode.DISABLED else _snapshot(mode, next_invocation)
        if snapshot:
            next_invocation += 1
        return actual, actual.copy(), snapshot, case.stablehlo_sha256

    monkeypatch.setattr(
        subject,
        "require_h100",
        lambda: subject.DeviceEvidence(backend="gpu", count=1, kind="NVIDIA H100"),
    )
    monkeypatch.setattr(subject, "load_numerical_policy", lambda: None)
    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="rank2-a/disabled:maximum_absolute_error"):
        subject.run_matrix(report_cell=reports.append)

    assert observed == [(case.case_id, mode) for case in CASES for mode in subject.Mode]
    assert [(report["case_id"], report["mode"]) for report in reports] == [
        (case.case_id, mode.value) for case in CASES for mode in subject.Mode
    ]
    assert reports[0]["threshold_exceeded"]
    assert all(not report["threshold_exceeded"] for report in reports[1:])


def test_structural_failure_remains_fail_fast_without_partial_threshold_recovery(monkeypatch) -> None:
    observed: list[tuple[str, subject.Mode]] = []
    reports: list[dict[str, object]] = []

    def execute(case, mode):
        observed.append((case.case_id, mode))
        actual = binary64_oracle(*deterministic_inputs(case)).astype(np.float32)
        repeated = actual.copy()
        repeated.flat[0] = np.nextafter(repeated.flat[0], np.float32(np.inf))
        return actual, repeated, (), case.stablehlo_sha256

    monkeypatch.setattr(
        subject,
        "require_h100",
        lambda: subject.DeviceEvidence(backend="gpu", count=1, kind="NVIDIA H100"),
    )
    monkeypatch.setattr(subject, "load_numerical_policy", lambda: None)
    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="repeat result drift"):
        subject.run_matrix(report_cell=reports.append)

    assert observed == [("rank2-a", subject.Mode.DISABLED)]
    assert reports == []


@pytest.mark.parametrize(
    "mutation",
    [
        "contract",
        "full_write",
        "repeat",
        "stablehlo",
        "metric",
        "observer_id",
        "telemetry_type",
    ],
)
def test_persisted_result_rejects_semantic_and_evidence_mutations(mutation: str) -> None:
    document = _result_document()
    changed = deepcopy(document)
    coordinates = changed["coordinates"]
    assert isinstance(coordinates, list)
    if mutation == "contract":
        changed["semantic_contract"]["groups"].reverse()
    elif mutation == "full_write":
        coordinates[0]["full_write"] = False
    elif mutation == "repeat":
        coordinates[1]["repeat_sha256"] = "f" * 64
    elif mutation == "stablehlo":
        coordinates[2]["stablehlo_sha256"] = "f" * 64
    elif mutation == "metric":
        coordinates[3]["metrics"]["maximum_absolute_error"] = CASES[1].limits["maximum_absolute_error"] + 1.0
    elif mutation == "observer_id":
        coordinates[4]["observer"]["invocation_id"] = coordinates[1]["observer"]["invocation_id"]
    elif mutation == "telemetry_type":
        coordinates[5]["observer"]["launch_count_telemetry"] = 2.0
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="generic composition"):
        validate_generic_composition_result(changed)


def test_persisted_result_is_closed_and_canonicalizable() -> None:
    document = _result_document()

    assert document["numerical_policy_sha256"] == POLICY_SHA256
    assert validate_generic_composition_result(document) == document
