# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import ml_dtypes
import numpy as np
import pytest
import track1_rmsnorm_h100_subject as subject
from track1_rmsnorm_h100_subject import Boundary, Coordinate, Mode


def _observer_snapshot(coordinate: Coordinate) -> tuple[tuple[object, ...], ...]:
    identity = subject.observer_identity(coordinate)
    membership = "[[#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 1, 0>]]"
    manifest = f"selected_regions = {membership}"
    return (
        (
            31,
            "algebra_coverage",
            coordinate.mode.value,
            identity.policy_sha256,
            identity.tuning_sha256,
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
            31,
            "generated_bundle",
            coordinate.mode.value,
            identity.policy_sha256,
            identity.tuning_sha256,
            membership,
            manifest,
            "a" * 64,
            "b" * 64,
            False,
            "",
            subject.expected_bundle_shape(coordinate.boundary)[0],
            subject.expected_bundle_shape(coordinate.boundary)[1],
        ),
        (
            31,
            "final_erasure",
            coordinate.mode.value,
            identity.policy_sha256,
            identity.tuning_sha256,
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


def test_rmsnorm_canary_covers_two_shapes_three_boundaries_and_three_modes() -> None:
    coordinates = Coordinate.all()

    assert len(coordinates) == 18
    assert {(coordinate.shape, coordinate.boundary, coordinate.mode) for coordinate in coordinates} == {
        (shape, boundary, mode) for shape in ((2, 7, 19), (3, 2, 5, 11, 263)) for boundary in Boundary for mode in Mode
    }
    for mode in (Mode.SOURCE_ORDERED, Mode.FAST):
        options = subject.compiler_options_for(mode)
        assert options["xla_shuttle_enable"] is True
        payload = json.loads(options["xla_shuttle_options"])
        assert payload["pipeline_abi_version"] == 22
        assert payload["tuning"]["materialization"] == "prefer_fusion"
    assert subject.compiler_options_for(Mode.DISABLED) is None


def test_rmsnorm_inputs_and_boundary_results_match_independent_shapes() -> None:
    for shape in ((2, 7, 19), (3, 2, 5, 11, 263)):
        x, weight, cotangent = subject.deterministic_inputs(shape)
        assert x.shape == shape
        assert weight.shape == (shape[-1],)
        assert cotangent.shape == shape
        assert x.dtype == weight.dtype == cotangent.dtype == np.dtype(ml_dtypes.bfloat16)
        assert np.all(x != 0)
        for boundary, roles in (
            (Boundary.FORWARD, ("result",)),
            (Boundary.BACKWARD, ("input_cotangent", "weight_cotangent")),
            (Boundary.COMPOSED, ("result", "input_cotangent", "weight_cotangent")),
        ):
            arrays = jax.jit(subject.boundary_function(boundary))(
                jax.device_put(x), jax.device_put(weight), jax.device_put(cotangent)
            )
            assert tuple(np.asarray(value).shape for value in arrays) == tuple(
                shape if role != "weight_cotangent" else (shape[-1],) for role in roles
            )
            assert tuple(np.asarray(value).dtype for value in arrays) == tuple(
                np.dtype(ml_dtypes.bfloat16) for _ in roles
            )


def test_run_matrix_rejects_enabled_result_drift_through_public_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        roles = subject.result_roles(coordinate.boundary)
        arrays = []
        for ordinal, role in enumerate(roles):
            shape = (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape
            value = np.full(shape, ordinal + 1, dtype=ml_dtypes.bfloat16)
            if coordinate == Coordinate((2, 7, 19), Boundary.COMPOSED, Mode.FAST) and role == "input_cotangent":
                value = value.copy()
                value.flat[0] = ml_dtypes.bfloat16(-0.0)
            arrays.append(value)
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="RMSNorm numerical mismatch") as raised:
        subject.run_matrix()

    evidence = json.loads(str(raised.value).split(": ", 1)[1])
    assert evidence["boundary"] == "composed"
    assert evidence["mode"] == "fast"
    assert evidence["role"] == "input_cotangent"
    assert evidence["actual_sha256"] != evidence["expected_sha256"]
    assert len(str(raised.value)) < 1_024


def test_run_matrix_rejects_a_shared_disabled_and_enabled_host_oracle_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = []
        for role in subject.result_roles(coordinate.boundary):
            shape = (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape
            arrays.append(np.zeros(shape, dtype=ml_dtypes.bfloat16))
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="host oracle"):
        subject.run_matrix()


def test_result_document_binds_all_output_hashes_and_observer_invocations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = []
        for ordinal, role in enumerate(subject.result_roles(coordinate.boundary)):
            shape = (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape
            arrays.append(np.full(shape, ordinal + 1, dtype=ml_dtypes.bfloat16))
        return tuple(arrays), (() if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate))

    monkeypatch.setattr(subject, "_execute", execute)
    document = subject.result_document(subject.run_matrix())

    assert document["schema"] == "track1_rmsnorm_h100_result"
    assert len(document["coordinates"]) == 18
    assert sum(item["observer"] is not None for item in document["coordinates"]) == 12
    for item in document["coordinates"]:
        assert list(item["results"]) == list(subject.result_roles(Boundary(item["boundary"])))
        assert all(len(digest) == 64 for digest in item["results"].values())
        if item["observer"] is not None:
            assert item["observer"]["invocation_id"] == 31
            assert item["observer"]["generated_bundle_sha256"] == "b" * 64
            expected = subject.expected_bundle_shape(Boundary(item["boundary"]))
            assert (item["observer"]["launch_count"], item["observer"]["temporary_count"]) == expected
