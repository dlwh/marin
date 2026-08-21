# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import ml_dtypes
import numpy as np
import pytest
import track1_rmsnorm_contract_h100_subject as subject
from track1_rmsnorm_contract_h100_subject import Boundary, Coordinate, Mode
from track1_rmsnorm_contract_numerical_policy import SHAPES, binary64_oracle, deterministic_inputs


def _observer_snapshot(coordinate: Coordinate, launches: int, temporaries: int) -> tuple[tuple[object, ...], ...]:
    identity = subject.observer_identity(coordinate)
    membership = "[[#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 1, 0>]]"
    manifest = f"selected_regions = {membership}"
    return (
        (
            41,
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
            41,
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
            launches,
            temporaries,
        ),
        (
            41,
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


def test_subject_covers_two_contract_shapes_two_boundaries_and_three_modes() -> None:
    coordinates = Coordinate.all()

    assert len(coordinates) == 12
    assert {(item.shape, item.boundary, item.mode) for item in coordinates} == {
        (shape, boundary, mode) for shape in SHAPES for boundary in Boundary for mode in Mode
    }
    for mode in (Mode.SOURCE_ORDERED, Mode.FAST):
        options = subject.compiler_options_for(mode)
        assert options["xla_shuttle_enable"] is True
        payload = json.loads(options["xla_shuttle_options"])
        assert payload["pipeline_abi_version"] == 22
        assert payload["tuning"]["materialization"] == "prefer_fusion"
    assert subject.compiler_options_for(Mode.DISABLED) is None


def test_boundary_function_matches_public_primal_and_vjp_role_order_on_host() -> None:
    for shape in SHAPES:
        inputs = deterministic_inputs(shape)
        for boundary in Boundary:
            compiled = jax.jit(subject.boundary_function(boundary))
            arrays = compiled(*map(jax.device_put, inputs))
            actual = tuple(np.asarray(value) for value in arrays)
            repeated = tuple(np.asarray(value) for value in compiled(*map(jax.device_put, inputs)))
            roles = subject.result_roles(boundary)
            expected = binary64_oracle(shape)
            assert len(actual) == len(roles)
            assert tuple(value.shape for value in actual) == tuple(expected[role].shape for role in roles)
            assert all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in actual)
            assert all(np.isfinite(value.astype(np.float32)).all() for value in actual)
            validated = subject.validate_coordinate(Coordinate(shape, boundary, Mode.DISABLED), actual, repeated, ())
            assert tuple(validated.results) == roles


def test_observer_requires_normative_primal_topology_but_records_composed_topology_diagnostically() -> None:
    primal_inline = Coordinate((5, 19, 7), Boundary.PRIMAL, Mode.SOURCE_ORDERED)
    primal_compact = Coordinate((5, 19, 257), Boundary.PRIMAL, Mode.FAST)
    composed = Coordinate((5, 19, 7), Boundary.COMPOSED, Mode.SOURCE_ORDERED)

    inline = subject.validate_observer_snapshot(primal_inline, _observer_snapshot(primal_inline, 1, 0))
    compact = subject.validate_observer_snapshot(primal_compact, _observer_snapshot(primal_compact, 2, 1))
    diagnostic = subject.validate_observer_snapshot(composed, _observer_snapshot(composed, 42, 38))
    assert (inline.launch_count, inline.temporary_count, inline.topology_status) == (1, 0, "normative_primal")
    assert (compact.launch_count, compact.temporary_count, compact.topology_status) == (2, 1, "normative_primal")
    assert (diagnostic.launch_count, diagnostic.temporary_count, diagnostic.topology_status) == (
        42,
        38,
        "diagnostic_non_performant",
    )

    with pytest.raises(ValueError, match="primal bundle topology"):
        subject.validate_observer_snapshot(primal_inline, _observer_snapshot(primal_inline, 2, 1))
    with pytest.raises(ValueError, match="bounded positive topology"):
        subject.validate_observer_snapshot(composed, _observer_snapshot(composed, 0, 0))


def test_validate_coordinate_rejects_repeat_nonfinite_role_and_threshold_drift() -> None:
    coordinate = Coordinate((5, 19, 7), Boundary.COMPOSED, Mode.SOURCE_ORDERED)
    roles = subject.result_roles(coordinate.boundary)
    arrays = tuple(
        value.astype(ml_dtypes.bfloat16) for value in (binary64_oracle(coordinate.shape)[role] for role in roles)
    )
    snapshot = _observer_snapshot(coordinate, 42, 38)
    accepted = subject.validate_coordinate(coordinate, arrays, tuple(value.copy() for value in arrays), snapshot)
    assert tuple(accepted.results) == roles
    assert accepted.repeat_results == accepted.results

    repeated = list(value.copy() for value in arrays)
    repeated[0].view(np.uint16).flat[0] ^= np.uint16(1)
    with pytest.raises(RuntimeError, match="repeat result drift"):
        subject.validate_coordinate(coordinate, arrays, tuple(repeated), snapshot)

    nonfinite = list(value.copy() for value in arrays)
    nonfinite[1].flat[0] = ml_dtypes.bfloat16(float("nan"))
    with pytest.raises(RuntimeError, match="input_cotangent result must be finite"):
        subject.validate_coordinate(coordinate, tuple(nonfinite), arrays, snapshot)

    swapped = (arrays[1], arrays[0], *arrays[2:])
    with pytest.raises(RuntimeError, match="result type differs from its public boundary"):
        subject.validate_coordinate(coordinate, swapped, swapped, snapshot)

    exceeded = list(value.copy() for value in arrays)
    exceeded[2].flat[0] = ml_dtypes.bfloat16(1024.0)
    with pytest.raises(RuntimeError, match="independent binary64 threshold"):
        subject.validate_coordinate(coordinate, tuple(exceeded), tuple(exceeded), snapshot)


def test_disabled_jax_comparison_is_labeled_diagnostic_not_the_acceptance_oracle() -> None:
    coordinate = Coordinate((5, 19, 7), Boundary.PRIMAL, Mode.SOURCE_ORDERED)
    expected = subject.emitted_primal_oracle(coordinate.shape, fast=False)
    disabled = expected.copy()
    disabled.view(np.uint16).flat[0] ^= np.uint16(1)

    accepted = subject.validate_coordinate(
        coordinate,
        (expected,),
        (expected.copy(),),
        _observer_snapshot(coordinate, 1, 0),
        disabled_jax=(disabled,),
    )

    assert accepted.disabled_jax_equal == {"result": False}
    assert accepted.disabled_jax_results["result"] != accepted.results["result"]
