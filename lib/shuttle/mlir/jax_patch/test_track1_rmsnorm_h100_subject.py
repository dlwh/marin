# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import ml_dtypes
import numpy as np
import pytest
import track1_rmsnorm_h100_subject as subject
from track1_rmsnorm_h100_subject import Boundary, Coordinate, Mode
from track1_rmsnorm_numerical_policy import POLICY_SHA256, binary64_oracle, emitted_tree_oracle


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


def _next_bf16(value: np.ndarray) -> None:
    bits = value.view(np.uint16)
    bits.flat[0] = np.uint16(int(bits.flat[0]) + 1)


def _accepted_arrays(coordinate: Coordinate) -> list[np.ndarray]:
    oracle = emitted_tree_oracle(coordinate.shape)
    return [oracle[role].copy() for role in subject.result_roles(coordinate.boundary)]


def test_run_matrix_accepts_in_contract_enabled_drift_as_a_labeled_disabled_jax_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = _accepted_arrays(coordinate)
        if coordinate.mode is Mode.DISABLED:
            analytic = binary64_oracle(coordinate.shape)
            arrays = [analytic[role].astype(ml_dtypes.bfloat16) for role in subject.result_roles(coordinate.boundary)]
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)
    results = subject.run_matrix()

    changed = next(
        result
        for result in results
        if result.coordinate == Coordinate((3, 2, 5, 11, 263), Boundary.BACKWARD, Mode.SOURCE_ORDERED)
    )
    assert changed.disabled_jax_bitwise_equal == {"input_cotangent": False, "weight_cotangent": True}


def test_run_matrix_rejects_independent_binary64_threshold_exceedance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        roles = subject.result_roles(coordinate.boundary)
        arrays = _accepted_arrays(coordinate)
        for index, role in enumerate(roles):
            if coordinate == Coordinate((2, 7, 19), Boundary.COMPOSED, Mode.FAST) and role == "input_cotangent":
                arrays[index].flat[0] = ml_dtypes.bfloat16(64.0)
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="independent binary64 threshold"):
        subject.run_matrix()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_run_matrix_rejects_nonfinite_public_results(monkeypatch: pytest.MonkeyPatch, nonfinite: float) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = _accepted_arrays(coordinate)
        if coordinate == Coordinate((2, 7, 19), Boundary.FORWARD, Mode.SOURCE_ORDERED):
            arrays[0].flat[0] = ml_dtypes.bfloat16(nonfinite)
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="finite"):
        subject.run_matrix()


def test_run_matrix_rejects_backward_public_result_order_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = _accepted_arrays(coordinate)
        if coordinate == Coordinate((2, 7, 19), Boundary.BACKWARD, Mode.SOURCE_ORDERED):
            arrays.reverse()
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="result type differs from its public boundary"):
        subject.run_matrix()


def test_run_matrix_rejects_repeat_drift_before_accepting_numerics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = _accepted_arrays(coordinate)
        repeated = [value.copy() for value in arrays]
        if coordinate == Coordinate((2, 7, 19), Boundary.FORWARD, Mode.SOURCE_ORDERED):
            _next_bf16(repeated[0])
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(repeated), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="repeat result drift"):
        subject.run_matrix()


def test_run_matrix_rejects_a_shared_disabled_and_enabled_binary64_oracle_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = []
        for role in subject.result_roles(coordinate.boundary):
            shape = (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape
            arrays.append(np.zeros(shape, dtype=ml_dtypes.bfloat16))
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="independent binary64 threshold"):
        subject.run_matrix()


def test_result_document_binds_all_output_hashes_and_observer_invocations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "require_h100", lambda: None)

    def execute(coordinate: Coordinate):
        arrays = _accepted_arrays(coordinate)
        snapshot = () if coordinate.mode is Mode.DISABLED else _observer_snapshot(coordinate)
        return tuple(arrays), tuple(value.copy() for value in arrays), snapshot

    monkeypatch.setattr(subject, "_execute", execute)
    document = subject.result_document(subject.run_matrix())

    assert document["schema"] == "track1_rmsnorm_h100_result"
    assert document["schema_version"] == 2
    assert document["numerical_policy_sha256"] == POLICY_SHA256
    assert len(document["coordinates"]) == 18
    assert sum(item["observer"] is not None for item in document["coordinates"]) == 12
    for item in document["coordinates"]:
        roles = list(subject.result_roles(Boundary(item["boundary"])))
        assert list(item["results"]) == roles
        assert list(item["repeat_results"]) == roles
        assert list(item["analytic_oracle"]) == roles
        assert list(item["emitted_tree_oracle"]) == roles
        assert list(item["disabled_jax"]) == roles
        assert list(item["disabled_jax_bitwise_equal"]) == roles
        assert list(item["numerics"]) == roles
        assert all(len(digest) == 64 for digest in item["results"].values())
        assert item["repeat_results"] == item["results"]
        assert item["disabled_jax_bitwise_equal"] == {
            role: item["results"][role] == item["disabled_jax"][role] for role in roles
        }
        for role in roles:
            assert set(item["numerics"][role]) == {
                "maximum_absolute_error",
                "maximum_bf16_ulp_error",
                "mean_absolute_error",
                "relative_linf_error",
            }
        if item["observer"] is not None:
            assert item["observer"]["invocation_id"] == 31
            assert item["observer"]["generated_bundle_sha256"] == "b" * 64
            expected = subject.expected_bundle_shape(Boundary(item["boundary"]))
            assert (item["observer"]["launch_count"], item["observer"]["temporary_count"]) == expected
