# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import jax
import ml_dtypes
import numpy as np
import pytest
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
