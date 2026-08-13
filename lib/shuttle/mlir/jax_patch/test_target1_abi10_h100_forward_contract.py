# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the untimed ABI 10 H100 forward evidence contract."""

import copy
import hashlib
import json
from typing import Any

import pytest
from target1_abi10_h100_forward_contract import BACKEND_PAIRS, load_contract, validate_observation

REVISION = "b" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _collector() -> dict[str, object]:
    return {
        "schema_version": 1,
        "marin_revision": REVISION,
        "runner_binary_sha256": _digest("runner"),
        "harness_sha256": _digest("harness"),
        "build_manifest_sha256": _digest("build-manifest"),
        "artifact_manifest_sha256": _digest("artifact-manifest"),
    }


def _te_execution_identity() -> dict[str, object]:
    collector = _collector()
    return {
        "schema_version": 1,
        "hardware_class": "h100",
        "marin_revision": REVISION,
        "harness_sha256": collector["harness_sha256"],
        "build_manifest_sha256": collector["build_manifest_sha256"],
        "runner_binary_sha256": collector["runner_binary_sha256"],
        "transformer_engine": {
            "expert_oracle_contract_id": "target1_rowwise_bf16_te_2_17_expert_oracle_v1",
            "version": "2.17.0",
            "source_tag": "v2.17",
            "source_commit": "2e559f062497bef768dfbe9d7e45548fadeca80a",
            "resolved_library_path": "/sealed/libtransformer_engine.so",
            "library_sha256": _digest("transformer-engine-library"),
            "elf_build_id": "te-build-id",
            "resolved_shared_library_dependencies": [
                {
                    "path": "/sealed/libcudart.so",
                    "sha256": _digest("libcudart"),
                    "elf_build_id": "cudart-build-id",
                }
            ],
        },
        "toolchain": {
            "compiler": "nvcc 13.0",
            "build_flags": ["-O3"],
            "target_architectures": ["sm_90"],
        },
        "cuda": {
            "toolkit_version": "13.0",
            "nvcc_version": "13.0",
            "driver_version": 13000,
            "runtime_version": 13000,
        },
        "cudnn": {
            "compile_time_version": "9.0",
            "runtime_version": "9.0",
            "resolved_library_path": "/sealed/libcudnn.so",
            "library_sha256": _digest("libcudnn"),
            "elf_build_id": "cudnn-build-id",
        },
        "device": {
            "ordinal": 0,
            "model": "NVIDIA H100 80GB HBM3",
            "uuid": "GPU-9ac34b34-d6f8-a9b7-5e59-69081a79412d",
            "compute_capability": "9.0",
            "physical_sm_count": 120,
        },
    }


def _metrics() -> dict[str, float | int]:
    return {
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_linf_error": 0.0,
        "max_bfloat16_ulp_error": 0,
    }


def _output(label: str) -> dict[str, object]:
    return {
        "array_digest": _digest(f"{label}:array"),
        "raw_sha256": _digest(f"{label}:raw"),
        "metrics": _metrics(),
    }


def _subject(*, shuttle: bool) -> dict[str, object]:
    output = _output("candidate")
    identity: dict[str, object] = {
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "jax_revision": "619764c15117fbefc4ba13ab941871cb514c23f6",
        "xla_revision": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
        "compiler_options_sha256": None,
    }
    value: dict[str, object] = {
        "identity": identity,
        "output": output,
        "repeatability": {
            "sequential_raw_sha256": [output["raw_sha256"]] * 3,
            "concurrent_raw_sha256": [output["raw_sha256"]] * 2,
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
                "invocation_id": 7,
                "phases": ["algebra_coverage", "generated_bundle", "final_erasure"],
                "policy": "source_ordered",
                "compiler_options_sha256": contract["compiler"]["canonical_options_sha256"],
                "tuning_sha256": _digest("canonical-tuning"),
                "complete_source_results": 20,
                "excluded_source_results": 0,
                "coverage_manifest_sha256": _digest("coverage-manifest"),
                "phase_coverage_manifest_sha256": [_digest("coverage-manifest")] * 3,
                "unsupported_fingerprint_sha256": _digest("unsupported-empty"),
                "failure_pass": "",
                "final_no_shuttle_semantics": True,
                "final_module_sha256": _digest("final-module"),
                "generated_bundle_artifact_sha256": _digest("generated-bundle"),
            },
            "custom_call": {
                "target": "shuttle.gpu.executable_bundle.v1",
                "api_version": 4,
                "backend_attribute_count": 8,
                "output_alias_count": 0,
            },
            "bundle": {
                "transport_sha256": _digest("transport"),
                "descriptor_sha256": _digest("descriptor"),
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
                "assembler_sha256": _digest("assembler"),
                "slice_sha256": [_digest(f"ptx-slice:{index}") for index in range(19)],
                "all_sm90_assembled": True,
                "spill_bytes_per_slice": [0] * 19,
            },
            "cache": {
                "compiler_options_sha256": contract["compiler"]["canonical_options_sha256"],
                "generated_bundle_artifact_sha256": _digest("generated-bundle"),
                "populate_worker_id": _digest("populate-worker"),
                "reuse_worker_id": _digest("reuse-worker"),
                "populate_process_id": 101,
                "reuse_process_id": 202,
                "populate_before": [],
                "populate_after": [
                    {
                        "key": "jit_forward-" + _digest("cache-key") + "-cache",
                        "sha256": _digest("cache-entry"),
                        "size": 4096,
                    }
                ],
                "reuse_before": [
                    {
                        "key": "jit_forward-" + _digest("cache-key") + "-cache",
                        "sha256": _digest("cache-entry"),
                        "size": 4096,
                    }
                ],
                "reuse_after": [
                    {
                        "key": "jit_forward-" + _digest("cache-key") + "-cache",
                        "sha256": _digest("cache-entry"),
                        "size": 4096,
                    }
                ],
                "populate_hits": 0,
                "reuse_hits": 1,
                "reuse_observer_events": [],
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
        "inputs": contract["inputs"],
        "reference_output_array_digest": contract["reference"]["output_array_digest"],
        "collector": _collector(),
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
            "wheel_sha256": _digest("wheel"),
            "cuda_plugin_sha256": _digest("cuda-plugin"),
        },
        "ordinary_jax": _subject(shuttle=False),
        "shuttle": _subject(shuttle=True),
        "te_results": [_te_result(pair) for pair in BACKEND_PAIRS],
        "performance_status": "not_started",
        "scorecard_effect": contract["scorecard_effect"],
    }


def _te_result(pair: tuple[str, str]) -> dict[str, object]:
    contract = load_contract()
    identity = _te_execution_identity()
    return {
        "backend_pair": list(pair),
        "qualified_against_reference": True,
        "execution_identity_sha256": _canonical_digest(identity),
        "execution_identity": identity,
        "result_artifact_sha256": _digest(f"te-result:{pair[0]}:{pair[1]}"),
        "input_raw_sha256": {
            "x": contract["inputs"]["raw_sha256"][0],
            "gamma": contract["inputs"]["raw_sha256"][1],
        },
        "reference_output_array_digest": contract["reference"]["output_array_digest"],
        "output": _output(f"te-output:{pair[0]}:{pair[1]}"),
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
        (("expert_oracle", "source_commit"), "c" * 40),
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
        ("observer_splice", "phase_coverage_manifest_sha256"),
        ("collector", "runner_binary_sha256"),
        ("te_identity", "runner_binary_sha256"),
        ("bundle", "temporary_bytes"),
        ("cache", "reuse_hits"),
        ("cache_same_worker", "workers must be distinct"),
        ("cache_mutation", "reuse_after"),
        ("spill", "zero spill bytes"),
        ("reference", "independent-reference limit"),
        ("input_identity", "observation.inputs"),
        ("reference_identity", "reference_output_array_digest"),
        ("contract_override", "contract"),
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
    elif mutation == "observer_splice":
        observation["shuttle"]["architecture"]["observer"]["phase_coverage_manifest_sha256"][1] = "c" * 64
    elif mutation == "collector":
        observation["collector"]["runner_binary_sha256"] = "c" * 64
    elif mutation == "te_identity":
        identity = observation["te_results"][0]["execution_identity"]
        identity["runner_binary_sha256"] = "c" * 64
        observation["te_results"][0]["execution_identity_sha256"] = _canonical_digest(identity)
    elif mutation == "bundle":
        observation["shuttle"]["architecture"]["bundle"]["temporary_bytes"] -= 1
    elif mutation == "cache":
        observation["shuttle"]["architecture"]["cache"]["reuse_hits"] = 0
    elif mutation == "cache_same_worker":
        cache = observation["shuttle"]["architecture"]["cache"]
        cache["reuse_worker_id"] = cache["populate_worker_id"]
    elif mutation == "cache_mutation":
        observation["shuttle"]["architecture"]["cache"]["reuse_after"][0]["sha256"] = "c" * 64
    elif mutation == "spill":
        observation["shuttle"]["architecture"]["ptx"]["spill_bytes_per_slice"][3] = 4
    elif mutation == "reference":
        observation["shuttle"]["output"]["metrics"]["max_absolute_error"] = 0.01
    elif mutation == "input_identity":
        observation["inputs"]["raw_sha256"][0] = "c" * 64
    elif mutation == "reference_identity":
        observation["reference_output_array_digest"] = "c" * 64
    elif mutation == "contract_override":
        contract = copy.deepcopy(load_contract())
        contract["coordinate"]["boundary"] = "backward"
        observation["coordinate"]["boundary"] = "backward"
        with pytest.raises((AssertionError, ValueError), match=diagnostic):
            validate_observation(observation, contract)
        return
    else:
        raise AssertionError(mutation)
    with pytest.raises((AssertionError, ValueError), match=diagnostic):
        validate_observation(observation)
