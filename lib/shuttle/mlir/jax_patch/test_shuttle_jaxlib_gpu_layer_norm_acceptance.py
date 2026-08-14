# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from shuttle_jaxlib_gpu_layer_norm_acceptance import (
    BOUNDARIES,
    OBSERVER_FINGERPRINTS,
    PIPELINE_ABI_VERSION,
    POLICIES,
    SHAPES,
    arrays,
    boundary_function,
    boundary_inputs,
    cell_identities,
    expected_identity,
    load_baseline,
    output_names,
    qualify_outputs,
    save_baseline,
    subject_options,
    validate_observer_events,
)

from shuttle import Numerics


def _events(shape_id: str, boundary: str, numerics: Numerics) -> list[dict[str, object]]:
    identity = expected_identity(numerics)
    generated, final = OBSERVER_FINGERPRINTS[(shape_id, boundary, numerics.value)]
    common = {
        "invocation_id": 19,
        "policy": identity.policy,
        "policy_digest": identity.policy_digest,
        "tuning_digest": identity.tuning_digest,
        "failure_pass": "",
    }
    return [
        {
            **common,
            "phase": "algebra_coverage",
            "region_membership": "[[#shuttle.source_ref<0, 0, 0, 0>]]",
            "coverage_manifest": "{complete = true}",
            "unsupported_fingerprint": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
            "normalized_module_fingerprint": "",
            "no_shuttle_semantics": False,
        },
        {
            **common,
            "phase": "generated_bundle",
            "region_membership": "[[#shuttle.source_ref<0, 0, 0, 0>]]",
            "coverage_manifest": "{complete = true}",
            "unsupported_fingerprint": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
            "normalized_module_fingerprint": generated,
            "no_shuttle_semantics": False,
        },
        {
            **common,
            "phase": "final_erasure",
            "region_membership": "",
            "coverage_manifest": "",
            "unsupported_fingerprint": "",
            "normalized_module_fingerprint": final,
            "no_shuttle_semantics": True,
        },
    ]


def test_historical_layer_norm_gpu_driver_rejects_current_abi15_options() -> None:
    assert PIPELINE_ABI_VERSION == 13
    assert BOUNDARIES == ("forward", "backward", "composed")
    assert POLICIES == (Numerics.SOURCE_ORDERED, Numerics.FAST)
    assert len(cell_identities()) == 12
    assert len(set(cell_identities())) == 12
    assert set(OBSERVER_FINGERPRINTS) == set(cell_identities())
    assert len({fingerprint for pair in OBSERVER_FINGERPRINTS.values() for fingerprint in pair}) == 24

    for numerics in POLICIES:
        with pytest.raises(AssertionError, match="requires pipeline ABI 13"):
            subject_options(numerics)
        identity = expected_identity(numerics)
        assert json.loads(identity.canonical_options)["pipeline_abi_version"] == 13
        assert identity.policy == numerics.value


def test_layer_norm_public_jax_result_order_matches_independent_oracle() -> None:
    shape = SHAPES[0]
    values = {}
    assert [len(boundary_inputs(shape, boundary)) for boundary in BOUNDARIES] == [3, 3, 4]
    for boundary in BOUNDARIES:
        assert boundary_function(boundary).__name__ == boundary
        values[boundary] = arrays(boundary_function(boundary)(*boundary_inputs(shape, boundary)))
        assert tuple(value.shape for value in values[boundary]) == tuple(
            {
                "y": (shape.rows, shape.columns),
                "dx": (shape.rows, shape.columns),
                "dgamma": (shape.columns,),
                "dbeta": (shape.columns,),
            }[name]
            for name in output_names(boundary)
        )
    assert values["composed"][0].tobytes() == values["forward"][0].tobytes()
    for composed, backward in zip(values["composed"][1:], values["backward"], strict=True):
        assert composed.tobytes() == backward.tobytes()
    report = qualify_outputs(values["composed"], values["composed"], shape, "composed")
    assert [record["name"] for record in report] == ["y", "dx", "dgamma", "dbeta"]


def test_layer_norm_baseline_roundtrips_bfloat16_bits_and_rejects_shape_drift(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    save_baseline(path)
    values = load_baseline(path, SHAPES[0], "composed")
    assert [value.dtype for value in values] == [np.dtype("bfloat16")] * 4

    stored = {key: value.copy() for key, value in np.load(path).items()}
    stored[f"{SHAPES[0].shape_id}_composed_1"] = stored[f"{SHAPES[0].shape_id}_composed_1"].reshape(13, 7)
    np.savez(path, **stored)
    with pytest.raises(AssertionError, match="baseline bit payload changed"):
        load_baseline(path, SHAPES[0], "composed")


def test_layer_norm_observer_accepts_exact_cell_identity() -> None:
    shape = SHAPES[0]
    identity = expected_identity(Numerics.FAST)
    report = validate_observer_events(_events(shape.shape_id, "backward", Numerics.FAST), shape, "backward", identity)
    assert report["invocation_id"] == 19
    assert report["generated_bundle_fingerprint"] == OBSERVER_FINGERPRINTS[(shape.shape_id, "backward", "fast")][0]


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("phase", "three ordered observer phases"),
        ("invocation", "one invocation ID"),
        ("policy", "policy identity differs"),
        ("coverage", "coverage manifest"),
        ("generated", "generated-bundle fingerprint changed"),
        ("final", "final-erasure evidence changed"),
    ],
)
def test_layer_norm_observer_rejects_cross_cell_or_phase_drift(mutation: str, diagnostic: str) -> None:
    shape = SHAPES[0]
    events = _events(shape.shape_id, "forward", Numerics.SOURCE_ORDERED)
    if mutation == "phase":
        events[0], events[1] = events[1], events[0]
    elif mutation == "invocation":
        events[2]["invocation_id"] = 20
    elif mutation == "policy":
        events[1]["policy"] = "fast"
    elif mutation == "coverage":
        events[1]["coverage_manifest"] = "{different = true}"
    elif mutation == "generated":
        events[1]["normalized_module_fingerprint"] = "0" * 64
    elif mutation == "final":
        events[2]["normalized_module_fingerprint"] = "0" * 64
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    with pytest.raises(AssertionError, match=diagnostic):
        validate_observer_events(events, shape, "forward", expected_identity(Numerics.SOURCE_ORDERED))


def test_layer_norm_numerical_contract_rejects_large_subject_error() -> None:
    shape = SHAPES[0]
    ordinary = arrays(boundary_function("forward")(*boundary_inputs(shape, "forward")))
    corrupt = ordinary[0].copy()
    corrupt[0, 0] = np.dtype("bfloat16").type(8.0)
    with pytest.raises(AssertionError, match="independent-oracle limits"):
        qualify_outputs((corrupt,), ordinary, shape, "forward")
