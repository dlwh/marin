# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the untimed ABI 10 H100 forward evidence contract."""

import copy
import json
from typing import Any

import pytest
from target1_abi10_h100_forward_contract import BACKEND_PAIRS, load_contract, validate_observation

SHA = "a" * 64
REVISION = "b" * 40


def _metrics() -> dict[str, float | int]:
    return {
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_linf_error": 0.0,
        "max_bfloat16_ulp_error": 0,
    }


def _output() -> dict[str, object]:
    return {"array_digest": SHA, "raw_sha256": SHA, "metrics": _metrics()}


def _subject(*, shuttle: bool) -> dict[str, object]:
    identity: dict[str, object] = {
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "jax_revision": "619764c15117fbefc4ba13ab941871cb514c23f6",
        "xla_revision": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
        "compiler_options_sha256": None,
    }
    value: dict[str, object] = {
        "identity": identity,
        "output": _output(),
        "repeatability": {
            "sequential_raw_sha256": [SHA] * 3,
            "concurrent_raw_sha256": [SHA] * 2,
        },
    }
    if shuttle:
        contract = load_contract()
        identity["pipeline_abi_version"] = 10
        identity["compiler_options_sha256"] = contract["compiler"]["canonical_options_sha256"]
        value["architecture"] = {
            "marin_revision": REVISION,
            "registration": "real_cuda_plugin_ffi_handlers_through_pjrt_passed",
            "observer": {
                "phases": ["algebra_coverage", "generated_bundle", "final_erasure"],
                "complete_source_results": 20,
                "excluded_source_results": 0,
                "final_no_shuttle_semantics": True,
                "final_module_sha256": SHA,
                "generated_bundle_artifact_sha256": SHA,
            },
            "custom_call": {
                "target": "shuttle.gpu.executable_bundle.v1",
                "api_version": 4,
                "backend_attribute_count": 8,
                "output_alias_count": 0,
            },
            "bundle": {
                "transport_sha256": SHA,
                "descriptor_sha256": SHA,
                "transport_schema_version": 2,
                "device_schema_version": 3,
                "invocation_abi_schema_version": 3,
                "bundle_schema_version": 2,
                "completion": "stream_ordered",
                "launch_count": 19,
                "slot_count": 21,
                "temporary_slot_count": 18,
                "temporary_bytes": 201_416_716,
                "external_bindings": ["operand:0", "operand:1", "result:0"],
                "alias_count": 0,
            },
            "ptx": {
                "assembler": "ptxas 13.2.78",
                "assembler_sha256": SHA,
                "slice_sha256": [SHA] * 19,
                "all_sm90_assembled": True,
                "spill_bytes_per_slice": [0] * 19,
            },
            "cache": {
                "populate_hits": 0,
                "reuse_hits": 1,
                "entry_key": "jit_forward-" + SHA + "-cache",
                "entry_sha256": SHA,
                "entry_size": 4096,
                "reuse_observer_events": [],
                "bytes_immutable": True,
            },
            "direct_and_jax_bitwise_equal": True,
        }
    return value


def _valid_observation() -> dict[str, object]:
    contract = load_contract()
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "status": "sealed_untimed_hardware_observation",
        "coordinate": contract["coordinate"],
        "provenance": {
            "marin_revision": REVISION,
            "clean_tree": True,
            "hardware_class": "h100",
            "device_model": "NVIDIA H100 80GB HBM3",
            "device_uuid": "GPU-9ac34b34-d6f8-a9b7-5e59-69081a79412d",
            "compute_capability": "9.0",
            "physical_sm_count": 120,
            "driver_version": "595.71.05",
            "cuda_runtime_version": "13.0",
            "cuda_toolkit_version": "13.0",
            "wheel_sha256": SHA,
            "cuda_plugin_sha256": SHA,
        },
        "ordinary_jax": _subject(shuttle=False),
        "shuttle": _subject(shuttle=True),
        "te_results": [
            {
                "backend_pair": list(pair),
                "qualified_against_reference": True,
                "subject_identity_sha256": SHA,
                "output": _output(),
            }
            for pair in BACKEND_PAIRS
        ],
        "performance_status": "not_started",
        "scorecard_effect": contract["scorecard_effect"],
    }


def test_contract_accepts_only_complete_untimed_forward_evidence() -> None:
    observation = _valid_observation()
    validated = validate_observation(observation)
    assert validated["performance_status"] == "not_started"
    assert validated["scorecard_effect"]["on_success"] == {
        "cell_status": "partial",
        "architecture_status": "accepted",
        "numerical_status": "accepted",
        "performance_status": "not_started",
        "blockers": ["performance_evidence_missing"],
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("compiler", "pipeline_abi_version"), 10.0),
        (("compiler", "canonical_options"), "{}"),
        (("inputs", "raw_sha256"), ["c" * 64, "d" * 64]),
        (("reference", "limits", "max_bfloat16_ulp_error"), 1.0),
        (("scorecard_effect", "on_success", "performance_status"), "accepted"),
    ],
)
def test_contract_rejects_self_consistent_coordinate_and_type_drift(
    tmp_path, path: tuple[str, ...], value: object
) -> None:
    document = copy.deepcopy(load_contract())
    destination: Any = document
    for field in path[:-1]:
        destination = destination[field]
    destination[path[-1]] = value
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_contract(contract_path)


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("timing", "fields differ"),
        ("abi", "pipeline_abi_version"),
        ("missing_te", "all four"),
        ("duplicate_te", "duplicate TE"),
        ("output_drift", "bitwise ordinary-JAX"),
        ("repeatability", "not bitwise repeatable"),
        ("observer", "phases"),
        ("bundle", "temporary_bytes"),
        ("cache", "reuse_hits"),
        ("spill", "zero spill bytes"),
        ("reference", "independent-reference limit"),
    ],
)
def test_observation_rejects_acceptance_shortcuts(mutation: str, diagnostic: str) -> None:
    observation = copy.deepcopy(_valid_observation())
    if mutation == "timing":
        observation["latency_ms"] = 1.0
    elif mutation == "abi":
        observation["shuttle"]["identity"]["pipeline_abi_version"] = 9
    elif mutation == "missing_te":
        observation["te_results"].pop()
    elif mutation == "duplicate_te":
        observation["te_results"][-1]["backend_pair"] = observation["te_results"][0]["backend_pair"]
    elif mutation == "output_drift":
        observation["shuttle"]["output"]["raw_sha256"] = "c" * 64
        observation["shuttle"]["repeatability"]["sequential_raw_sha256"] = ["c" * 64] * 3
        observation["shuttle"]["repeatability"]["concurrent_raw_sha256"] = ["c" * 64] * 2
    elif mutation == "repeatability":
        observation["shuttle"]["repeatability"]["sequential_raw_sha256"][0] = "c" * 64
    elif mutation == "observer":
        observation["shuttle"]["architecture"]["observer"]["phases"].pop(1)
    elif mutation == "bundle":
        observation["shuttle"]["architecture"]["bundle"]["temporary_bytes"] -= 1
    elif mutation == "cache":
        observation["shuttle"]["architecture"]["cache"]["reuse_hits"] = 0
    elif mutation == "spill":
        observation["shuttle"]["architecture"]["ptx"]["spill_bytes_per_slice"][3] = 4
    elif mutation == "reference":
        observation["shuttle"]["output"]["metrics"]["max_absolute_error"] = 0.01
    else:
        raise AssertionError(mutation)
    with pytest.raises((AssertionError, ValueError), match=diagnostic):
        validate_observation(observation)
