# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
import track1_map_h100_subject as subject
from track1_map_h100_subject import (
    Coordinate,
    Mode,
    cache_attribution,
    compiler_options_for,
    deterministic_input,
    mixed_map_value_and_gradient,
    observer_identity,
    validate_observer_snapshot,
)


def _event(
    phase: str,
    coordinate: Coordinate,
    *,
    membership: str,
    manifest: str,
    unsupported: str,
    fingerprint: str,
    no_shuttle: bool,
) -> tuple[object, ...]:
    identity = observer_identity(coordinate)
    return (
        17,
        phase,
        coordinate.mode.value,
        identity.policy_sha256,
        identity.tuning_sha256,
        membership,
        manifest,
        unsupported,
        fingerprint,
        no_shuttle,
        "",
    )


def _successful_snapshot(coordinate: Coordinate) -> tuple[tuple[object, ...], ...]:
    membership = (
        "[[#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 1, 0>, "
        "#shuttle.source_ref<0, 0, 2, 0>], [#shuttle.source_ref<0, 0, 4, 0>, "
        "#shuttle.source_ref<0, 0, 5, 0>, #shuttle.source_ref<0, 0, 6, 0>, "
        "#shuttle.source_ref<0, 0, 7, 0>]]"
    )
    manifest = (
        "selected_regions = "
        + membership
        + ", excluded = #shuttle.source_ref<0, 0, 3, 0>, reason = unsupported_operation"
    )
    return (
        _event(
            "algebra_coverage",
            coordinate,
            membership=membership,
            manifest=manifest,
            unsupported="a" * 64,
            fingerprint="",
            no_shuttle=False,
        ),
        _event(
            "generated_bundle",
            coordinate,
            membership=membership,
            manifest=manifest,
            unsupported="a" * 64,
            fingerprint="b" * 64,
            no_shuttle=False,
        ),
        _event(
            "final_erasure",
            coordinate,
            membership="",
            manifest="",
            unsupported="",
            fingerprint="c" * 64,
            no_shuttle=True,
        ),
    )


def test_two_static_bf16_fixtures_have_bitwise_primal_and_gradient_oracles() -> None:
    for rank, shape in ((2, (2, 3)), (3, (2, 3, 5))):
        value = deterministic_input(rank)
        assert value.shape == shape
        assert value.dtype == np.dtype(ml_dtypes.bfloat16)
        assert np.all(value != 0)

        primal, gradient = jax.jit(mixed_map_value_and_gradient)(value)
        expected_primal = np.abs(-value.astype(np.float32)).astype(ml_dtypes.bfloat16)
        expected_gradient = np.where(value > 0, 1, -1).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(np.asarray(primal), expected_primal)
        np.testing.assert_array_equal(np.asarray(gradient), expected_gradient)


def test_disabled_source_ordered_and_fast_use_three_distinct_compile_inputs() -> None:
    disabled = compiler_options_for(Mode.DISABLED)
    source_ordered = compiler_options_for(Mode.SOURCE_ORDERED)
    fast = compiler_options_for(Mode.FAST)

    assert disabled is None
    assert source_ordered["xla_shuttle_enable"] is True
    assert fast["xla_shuttle_enable"] is True
    assert source_ordered["xla_shuttle_options"] != fast["xla_shuttle_options"]
    for options in (source_ordered, fast):
        serialized = options["xla_shuttle_options"]
        assert isinstance(serialized, str)
        payload = json.loads(serialized)
        assert payload["pipeline_abi_version"] == 22
        assert payload["tuning"]["materialization"] == "automatic"


def test_observer_contract_attributes_both_map_islands_and_final_erasure() -> None:
    coordinate = Coordinate(rank=2, mode=Mode.SOURCE_ORDERED)
    snapshot = _successful_snapshot(coordinate)

    evidence = validate_observer_snapshot(coordinate, snapshot)

    assert evidence.invocation_id == 17
    assert evidence.generated_bundle_sha256 == "b" * 64
    assert evidence.final_module_sha256 == "c" * 64

    bad_phase = list(snapshot)
    bad_phase[1] = tuple([bad_phase[1][0], "lowered_coverage", *bad_phase[1][2:]])
    with pytest.raises(ValueError, match="phases"):
        validate_observer_snapshot(coordinate, tuple(bad_phase))

    bad_membership = list(snapshot)
    for index in (0, 1):
        bad_membership[index] = tuple(
            [
                *bad_membership[index][:5],
                str(bad_membership[index][5]).replace("source_ref<0, 0, 7, 0>", ""),
                *bad_membership[index][6:],
            ]
        )
    with pytest.raises(ValueError, match="Map source"):
        validate_observer_snapshot(coordinate, tuple(bad_membership))

    with pytest.raises(ValueError, match="disabled"):
        validate_observer_snapshot(Coordinate(2, Mode.DISABLED), snapshot)
    assert validate_observer_snapshot(Coordinate(2, Mode.DISABLED), ()) is None


def test_cache_attribution_requires_six_unique_coordinate_entries() -> None:
    records = {}
    for ordinal, coordinate in enumerate(Coordinate.all()):
        records[coordinate] = (
            f"jit_mixed_map-{ordinal:064x}-cache",
            100 + ordinal,
            hashlib.sha256(str(coordinate).encode()).hexdigest(),
        )

    assert cache_attribution(records) == records

    duplicate = dict(records)
    first, second = Coordinate.all()[:2]
    duplicate[second] = duplicate[first]
    with pytest.raises(ValueError, match="unique"):
        cache_attribution(duplicate)

    missing = dict(records)
    missing.pop(Coordinate.all()[-1])
    with pytest.raises(ValueError, match="six"):
        cache_attribution(missing)


def test_matrix_populates_then_reuses_six_attributed_cache_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    listeners = []

    monkeypatch.setattr(subject, "require_h100", lambda: None)
    monkeypatch.setattr(subject, "configure_cache", lambda directory: None)
    monkeypatch.setattr(
        subject.jax.monitoring,
        "register_event_listener",
        listeners.append,
    )

    def arrays(coordinate: Coordinate) -> tuple[np.ndarray, np.ndarray]:
        value = deterministic_input(coordinate.rank)
        return (
            np.abs(-value.astype(np.float32)).astype(ml_dtypes.bfloat16),
            np.where(value > 0, 1, -1).astype(ml_dtypes.bfloat16),
        )

    def populate(coordinate: Coordinate):
        (cache_directory / f"jit_{coordinate.label}-{coordinate.rank:064x}-cache").write_bytes(coordinate.label.encode())
        snapshot = () if coordinate.mode is Mode.DISABLED else _successful_snapshot(coordinate)
        return arrays(coordinate), snapshot

    monkeypatch.setattr(subject, "_execute", populate)
    populated = subject.run_matrix(cache_directory, reuse=False)
    attribution = {result.coordinate: result.cache_entry for result in populated}

    assert len(populated) == 6
    assert len(set(attribution.values())) == 6
    assert sum(result.observer is not None for result in populated) == 4
    populate_bytes = subject.cache_snapshot(cache_directory)

    def reuse(coordinate: Coordinate):
        listeners[-1](subject.CACHE_HIT_EVENT)
        return arrays(coordinate), ()

    monkeypatch.setattr(subject, "_execute", reuse)
    reused = subject.run_matrix(
        cache_directory,
        reuse=True,
        expected_attribution=attribution,
    )

    assert [result.primal_sha256 for result in reused] == [result.primal_sha256 for result in populated]
    assert [result.gradient_sha256 for result in reused] == [result.gradient_sha256 for result in populated]
    assert all(result.observer is None for result in reused)
    assert subject.cache_snapshot(cache_directory) == populate_bytes


def test_matrix_numerical_failure_reports_one_bounded_role_specific_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    monkeypatch.setattr(subject, "require_h100", lambda: None)
    monkeypatch.setattr(subject, "configure_cache", lambda directory: None)
    monkeypatch.setattr(subject.jax.monitoring, "register_event_listener", lambda listener: None)

    def execute(coordinate: Coordinate):
        value = deterministic_input(coordinate.rank)
        primal = np.abs(-value.astype(np.float32)).astype(ml_dtypes.bfloat16)
        gradient = np.where(value > 0, 1, -1).astype(ml_dtypes.bfloat16)
        if coordinate == Coordinate(2, Mode.SOURCE_ORDERED):
            gradient = gradient.copy()
            gradient[1, 2] = ml_dtypes.bfloat16(0.5)
        cache_directory.joinpath(f"jit_{coordinate.label}-{coordinate.rank:064x}-cache").write_bytes(
            coordinate.label.encode()
        )
        snapshot = () if coordinate.mode is Mode.DISABLED else _successful_snapshot(coordinate)
        return (primal, gradient), snapshot

    monkeypatch.setattr(subject, "_execute", execute)

    with pytest.raises(RuntimeError, match="Track 1 numerical mismatch") as raised:
        subject.run_matrix(cache_directory, reuse=False)

    prefix, raw_evidence = str(raised.value).split(": ", 1)
    assert prefix == "Track 1 numerical mismatch"
    evidence = json.loads(raw_evidence)
    rank_two_value = deterministic_input(2)
    expected_gradient = np.where(rank_two_value > 0, 1, -1).astype(ml_dtypes.bfloat16)
    actual_gradient = expected_gradient.copy()
    actual_gradient[1, 2] = ml_dtypes.bfloat16(0.5)

    def grouped_sha256(value: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
        text = digest.hexdigest()
        return ":".join(text[index : index + 8] for index in range(0, len(text), 8))

    assert evidence == {
        "actual_bits": "0x3f00",
        "actual_dtype": "bfloat16",
        "actual_sha256": grouped_sha256(actual_gradient),
        "actual_shape": [2, 3],
        "actual_value": 0.5,
        "coordinate": "rank2_source_ordered",
        "expected_bits": "0x3f80",
        "expected_dtype": "bfloat16",
        "expected_sha256": grouped_sha256(expected_gradient),
        "expected_shape": [2, 3],
        "expected_value": 1.0,
        "first_index": [1, 2],
        "max_abs": 0.5,
        "mismatch_count": 1,
        "role": "gradient",
        "schema_version": 1,
    }
    for digest in (evidence["actual_sha256"], evidence["expected_sha256"]):
        assert len(digest.split(":")) == 8
        assert all(len(chunk) == 8 for chunk in digest.split(":"))
    assert len(str(raised.value)) < 1_024


def test_reuse_rejects_cache_drift_before_executing_subject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    attribution = {}
    for ordinal, coordinate in enumerate(Coordinate.all()):
        name = f"jit_{coordinate.label}-{ordinal:064x}-cache"
        content = coordinate.label.encode()
        (cache_directory / name).write_bytes(content)
        attribution[coordinate] = (name, len(content), hashlib.sha256(content).hexdigest())
    (cache_directory / attribution[Coordinate.all()[-1]][0]).write_bytes(b"drift")

    monkeypatch.setattr(subject, "require_h100", lambda: None)
    monkeypatch.setattr(subject, "configure_cache", lambda directory: None)
    monkeypatch.setattr(
        subject,
        "_execute",
        lambda coordinate: pytest.fail(f"executed {coordinate.label}"),
    )

    with pytest.raises(RuntimeError, match="bytes differ"):
        subject.run_matrix(
            cache_directory,
            reuse=True,
            expected_attribution=attribution,
        )
