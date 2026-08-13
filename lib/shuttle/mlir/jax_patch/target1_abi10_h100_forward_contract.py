# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed evidence contract for the first ABI 10 H100 Target 1 cell."""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from target1_te_oracle_harness import validate_execution_identity

CONTRACT_PATH = Path(__file__).with_name("target1-rowwise-bf16-abi10-h100-forward-source-ordered-v1.json")
CONTRACT_ID = "target1_rowwise_bf16_abi10_h100_forward_source_ordered_v1"
SCHEMA_VERSION = 1
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
METRICS = (
    "max_absolute_error",
    "mean_absolute_error",
    "relative_linf_error",
    "max_bfloat16_ulp_error",
)
BACKEND_PAIRS = (
    ("transformer_engine", "transformer_engine"),
    ("cudnn", "cudnn"),
    ("transformer_engine", "cudnn"),
    ("cudnn", "transformer_engine"),
)
CANONICAL_OPTIONS = (
    '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
    '"pipeline_abi_version":10,"schema_version":1,"tuning":{"cluster_shape":[],'
    '"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
)
INPUT_ARRAY_DIGESTS = (
    "47c7e1bda25abe1f0a2e0f2c34154f0484d367d1309ae2ddb08450ed9eb4b5df",
    "a69953eeff9555b4a0cf42fbe2c978a4c3559eb76329b437f6bd32cdaa284e7a",
)
INPUT_RAW_SHA256 = (
    "0f3b7cfa2054631d63038001deb298b086d0c2baffeaddd6082c4518df91e13c",
    "e90137a7f830b75f00ee6dae7c89bbd7fba30dfb31ff7951fec129441c170821",
)
REFERENCE_OUTPUT_ARRAY_DIGEST = "fe18afac66306e065ebf824c52cc6d95776e49521ed6209e405b58e40c934bdd"
TRANSFORMER_ENGINE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"
TE_PLAN_ID = "shape_boundary_backend_alternating_v1"
TUNING_SHA256 = "50db16cae6ad71d7d22f104f0477945ae0ec470957496b26904e17bdf0dcd7eb"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
UNSUPPORTED_EMPTY_SHA256 = hashlib.sha256(b"[]").hexdigest()
GENERATED_BUNDLE_SHA256 = "910b4192a0fc9d04986ca941d1508ffb7352430526f615e30b242aadc3ae4bf7"
FINAL_MODULE_SHA256 = "a7a977e017eb62dd277d2fe56db3dff3e2fff6ce6538e64d69821ed0dce23914"
REFERENCE_LIMITS = {
    "max_absolute_error": 0.0078125,
    "mean_absolute_error": 0.0023046755673821195,
    "relative_linf_error": 0.007751937984496124,
    "max_bfloat16_ulp_error": 1,
}
SCORECARD_EFFECT = {
    "on_success": {
        "cell_status": "partial",
        "architecture_status": "accepted",
        "numerical_status": "accepted",
        "performance_status": "not_started",
        "blockers": ["performance_evidence_missing"],
    },
    "target_status": "blocked",
}


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the closed pre-run contract and reject any constant drift."""
    document = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    root = _closed(
        document,
        {
            "schema_version",
            "contract_id",
            "coordinate",
            "compiler",
            "inputs",
            "reference",
            "expert_oracle",
            "repeatability",
            "performance",
            "scorecard_effect",
        },
        "contract",
    )
    _equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _equal(root["contract_id"], CONTRACT_ID, "contract_id")
    coordinate = _closed(
        root["coordinate"], {"target_id", "boundary", "shape", "shape_id", "policy", "hardware_class"}, "coordinate"
    )
    _equal(
        coordinate,
        {
            "target_id": 1,
            "boundary": "forward",
            "shape": [2048, 4096],
            "shape_id": "rmsnorm_bf16_2048x4096",
            "policy": "source_ordered",
            "hardware_class": "h100",
        },
        "coordinate",
    )
    compiler = _closed(
        root["compiler"],
        {
            "pipeline_abi_version",
            "canonical_options",
            "canonical_options_sha256",
            "custom_call_target",
            "custom_call_api_version",
            "transport_schema_version",
            "device_schema_version",
            "invocation_abi_schema_version",
            "bundle_schema_version",
            "completion",
            "launch_count",
            "slot_count",
            "temporary_slot_count",
            "temporary_bytes",
        },
        "compiler",
    )
    _equal(compiler["pipeline_abi_version"], 10, "compiler.pipeline_abi_version")
    _equal(compiler["canonical_options"], CANONICAL_OPTIONS, "compiler.canonical_options")
    _equal(
        hashlib.sha256(_string(compiler["canonical_options"], "compiler.canonical_options").encode()).hexdigest(),
        compiler["canonical_options_sha256"],
        "compiler.canonical_options_sha256",
    )
    expected_compiler = {
        "custom_call_target": "shuttle.gpu.executable_bundle.v1",
        "custom_call_api_version": 4,
        "transport_schema_version": 2,
        "device_schema_version": 3,
        "invocation_abi_schema_version": 3,
        "bundle_schema_version": 2,
        "completion": "stream_ordered",
        "launch_count": 19,
        "slot_count": 21,
        "temporary_slot_count": 18,
        "temporary_bytes": 201_416_716,
    }
    for field, expected in expected_compiler.items():
        _equal(compiler[field], expected, f"compiler.{field}")
    inputs = _closed(root["inputs"], {"generator", "array_digests", "raw_sha256"}, "inputs")
    _equal(inputs["generator"], "target1_numerical_oracle.fixed_inputs", "inputs.generator")
    _equal(inputs["array_digests"], list(INPUT_ARRAY_DIGESTS), "inputs.array_digests")
    _equal(inputs["raw_sha256"], list(INPUT_RAW_SHA256), "inputs.raw_sha256")
    reference = _closed(root["reference"], {"implementation", "output_array_digest", "limits"}, "reference")
    _equal(reference["implementation"], "target1_numerical_oracle.independent_reference", "reference.implementation")
    _equal(reference["output_array_digest"], REFERENCE_OUTPUT_ARRAY_DIGEST, "reference.output_array_digest")
    _equal(reference["limits"], REFERENCE_LIMITS, "reference.limits")
    expert = _closed(
        root["expert_oracle"],
        {"transformer_engine_version", "source_commit", "required_forward_backend_pairs"},
        "expert_oracle",
    )
    _equal(expert["transformer_engine_version"], "2.17.0", "expert_oracle.transformer_engine_version")
    _equal(expert["source_commit"], TRANSFORMER_ENGINE_COMMIT, "expert_oracle.source_commit")
    _equal(
        expert["required_forward_backend_pairs"],
        [list(pair) for pair in BACKEND_PAIRS],
        "expert_oracle.required_forward_backend_pairs",
    )
    _equal(
        root["repeatability"],
        {
            "sequential_invocations": 3,
            "concurrent_invocations": 2,
            "rule": "all_output_bytes_bitwise_equal_to_same_device_ordinary_jax",
        },
        "repeatability",
    )
    _equal(
        root["performance"],
        {
            "status": "not_started",
            "timing_fields_permitted": False,
            "blocker": "performance_repeatability_contract_missing",
        },
        "performance",
    )
    _equal(root["scorecard_effect"], SCORECARD_EFFECT, "scorecard_effect")
    return root


def validate_observation(document: object, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate one sealed, untimed H100 observation for the exact ABI 10 cell."""
    sealed_contract = load_contract()
    if contract is not None:
        _equal(dict(contract), sealed_contract, "contract")
    contract = sealed_contract
    root = _closed(
        document,
        {
            "schema_version",
            "contract_id",
            "status",
            "coordinate",
            "inputs",
            "reference_output_array_digest",
            "collector",
            "provenance",
            "ordinary_jax",
            "shuttle",
            "te_results",
            "performance_status",
            "scorecard_effect",
        },
        "observation",
    )
    _equal(root["schema_version"], SCHEMA_VERSION, "observation.schema_version")
    _equal(root["contract_id"], CONTRACT_ID, "observation.contract_id")
    _equal(root["status"], "sealed_untimed_hardware_observation", "observation.status")
    _equal(root["coordinate"], contract["coordinate"], "observation.coordinate")
    _equal(root["inputs"], contract["inputs"], "observation.inputs")
    _equal(
        root["reference_output_array_digest"],
        contract["reference"]["output_array_digest"],
        "observation.reference_output_array_digest",
    )
    _equal(root["performance_status"], "not_started", "observation.performance_status")
    _equal(root["scorecard_effect"], contract["scorecard_effect"], "observation.scorecard_effect")
    collector = _collector(root["collector"])
    provenance = _provenance(root["provenance"])
    if collector["marin_revision"] != provenance["marin_revision"]:
        raise ValueError("collector revision differs from hardware provenance")
    ordinary = _subject(root["ordinary_jax"], "ordinary_jax", contract)
    shuttle = _subject(root["shuttle"], "shuttle", contract)
    _equal(shuttle["output"]["raw_sha256"], ordinary["output"]["raw_sha256"], "Shuttle bitwise ordinary-JAX parity")
    _equal(shuttle["output"]["array_digest"], ordinary["output"]["array_digest"], "Shuttle array ordinary-JAX parity")
    architecture = _architecture(shuttle["architecture"], contract)
    if architecture["marin_revision"] != provenance["marin_revision"]:
        raise ValueError("Shuttle architecture revision differs from hardware provenance")
    results = _sequence(root["te_results"], "te_results")
    if len(results) != len(BACKEND_PAIRS):
        raise ValueError("te_results must contain all four predeclared forward backend pairs")
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(results):
        result = _te_result(value, index, contract, collector, provenance)
        pair = tuple(result["backend_pair"])
        if pair in seen:
            raise ValueError("duplicate TE forward backend pair")
        seen.add(pair)
        for metric in METRICS:
            candidate = shuttle["output"]["metrics"][metric]
            oracle = result["output"]["metrics"][metric]
            floor = contract["reference"]["limits"][metric]
            if candidate > max(oracle, floor):
                raise AssertionError(f"Shuttle {metric} exceeds a matched TE forward result")
    if seen != set(BACKEND_PAIRS):
        raise ValueError("TE forward backend-pair set is incomplete")
    _equal(
        collector["artifact_manifest_sha256"],
        _canonical_sha256(
            {
                "collector": {key: value for key, value in collector.items() if key != "artifact_manifest_sha256"},
                "te_result_artifacts": [result["result_artifact_sha256"] for result in results],
            }
        ),
        "collector.artifact_manifest_sha256",
    )
    return root


def _subject(value: object, name: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"identity", "output", "repeatability"}
    if name == "shuttle":
        fields.add("architecture")
    subject = _closed(value, fields, name)
    identity_fields = {"jax_version", "jaxlib_version", "jax_revision", "xla_revision", "compiler_options_sha256"}
    if name == "shuttle":
        identity_fields.add("pipeline_abi_version")
    identity = _closed(subject["identity"], identity_fields, f"{name}.identity")
    _equal(identity["jax_version"], "0.10.1", f"{name}.identity.jax_version")
    _equal(identity["jaxlib_version"], "0.10.1", f"{name}.identity.jaxlib_version")
    _equal(identity["jax_revision"], "619764c15117fbefc4ba13ab941871cb514c23f6", f"{name}.identity.jax_revision")
    _equal(identity["xla_revision"], "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69", f"{name}.identity.xla_revision")
    if name == "ordinary_jax":
        _equal(identity["compiler_options_sha256"], None, "ordinary_jax.identity.compiler_options_sha256")
    else:
        _equal(identity["pipeline_abi_version"], 10, "shuttle.identity.pipeline_abi_version")
        _equal(
            identity["compiler_options_sha256"],
            contract["compiler"]["canonical_options_sha256"],
            "shuttle.identity.compiler_options_sha256",
        )
    output = _output(subject["output"], f"{name}.output", contract)
    repeatability = _closed(
        subject["repeatability"], {"sequential_raw_sha256", "concurrent_raw_sha256"}, f"{name}.repeatability"
    )
    sequential = _sha_list(repeatability["sequential_raw_sha256"], 3, f"{name}.repeatability.sequential_raw_sha256")
    concurrent = _sha_list(repeatability["concurrent_raw_sha256"], 2, f"{name}.repeatability.concurrent_raw_sha256")
    if set((*sequential, *concurrent, output["raw_sha256"])) != {output["raw_sha256"]}:
        raise AssertionError(f"{name} output is not bitwise repeatable")
    return subject


def _output(value: object, name: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    output = _closed(value, {"array_digest", "raw_sha256", "metrics"}, name)
    _sha(output["array_digest"], f"{name}.array_digest")
    _sha(output["raw_sha256"], f"{name}.raw_sha256")
    metrics = _metrics(output["metrics"], f"{name}.metrics")
    for metric, limit in contract["reference"]["limits"].items():
        if metrics[metric] > limit:
            raise AssertionError(f"{name}.{metric} exceeds the independent-reference limit")
    return output


def _architecture(value: object, contract: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _closed(
        value,
        {
            "marin_revision",
            "registration",
            "observer",
            "custom_call",
            "bundle",
            "ptx",
            "cache",
            "direct_and_jax_bitwise_equal",
        },
        "shuttle.architecture",
    )
    _revision(architecture["marin_revision"], "shuttle.architecture.marin_revision")
    _equal(
        architecture["registration"],
        "real_cuda_plugin_ffi_handlers_through_pjrt_passed",
        "shuttle.architecture.registration",
    )
    observer = _closed(
        architecture["observer"],
        {
            "invocation_id",
            "phase_invocation_ids",
            "phases",
            "policy",
            "compiler_options_sha256",
            "tuning_sha256",
            "complete_source_results",
            "excluded_source_results",
            "coverage_manifest_sha256",
            "phase_coverage_manifest_sha256",
            "unsupported_fingerprint_sha256",
            "failure_pass",
            "final_no_shuttle_semantics",
            "final_module_sha256",
            "generated_bundle_artifact_sha256",
        },
        "shuttle.architecture.observer",
    )
    if _integer(observer["invocation_id"], "shuttle.architecture.observer.invocation_id") == 0:
        raise ValueError("observer invocation id must be positive")
    _equal(
        observer["phase_invocation_ids"],
        [observer["invocation_id"]] * 3,
        "shuttle.architecture.observer.phase_invocation_ids",
    )
    _equal(
        observer["phases"],
        ["algebra_coverage", "generated_bundle", "final_erasure"],
        "shuttle.architecture.observer.phases",
    )
    _equal(observer["policy"], "source_ordered", "shuttle.architecture.observer.policy")
    _equal(
        observer["compiler_options_sha256"],
        contract["compiler"]["canonical_options_sha256"],
        "shuttle.architecture.observer.compiler_options_sha256",
    )
    _equal(observer["tuning_sha256"], TUNING_SHA256, "shuttle.architecture.observer.tuning_sha256")
    _equal(observer["complete_source_results"], 20, "shuttle.architecture.observer.complete_source_results")
    _equal(observer["excluded_source_results"], 0, "shuttle.architecture.observer.excluded_source_results")
    coverage = _sha(observer["coverage_manifest_sha256"], "shuttle.architecture.observer.coverage_manifest_sha256")
    _equal(
        observer["phase_coverage_manifest_sha256"],
        [coverage, coverage, EMPTY_SHA256],
        "shuttle.architecture.observer.phase_coverage_manifest_sha256",
    )
    _equal(
        observer["unsupported_fingerprint_sha256"],
        [UNSUPPORTED_EMPTY_SHA256, UNSUPPORTED_EMPTY_SHA256, EMPTY_SHA256],
        "shuttle.architecture.observer.unsupported_fingerprint_sha256",
    )
    _equal(observer["failure_pass"], "", "shuttle.architecture.observer.failure_pass")
    _equal(observer["final_no_shuttle_semantics"], True, "shuttle.architecture.observer.final_no_shuttle_semantics")
    _equal(observer["final_module_sha256"], FINAL_MODULE_SHA256, "shuttle.architecture.observer.final_module_sha256")
    _equal(
        observer["generated_bundle_artifact_sha256"],
        GENERATED_BUNDLE_SHA256,
        "shuttle.architecture.observer.generated_bundle_artifact_sha256",
    )
    compiler = contract["compiler"]
    custom_call = _closed(
        architecture["custom_call"],
        {"target", "api_version", "backend_attribute_count", "output_alias_count"},
        "shuttle.architecture.custom_call",
    )
    _equal(
        custom_call,
        {
            "target": compiler["custom_call_target"],
            "api_version": compiler["custom_call_api_version"],
            "backend_attribute_count": 8,
            "output_alias_count": 0,
        },
        "shuttle.architecture.custom_call",
    )
    bundle = _closed(
        architecture["bundle"],
        {
            "transport_sha256",
            "descriptor_sha256",
            "transport_schema_version",
            "device_schema_version",
            "invocation_abi_schema_version",
            "bundle_schema_version",
            "completion",
            "launch_count",
            "slot_count",
            "temporary_slot_count",
            "temporary_bytes",
            "external_bindings",
            "alias_count",
        },
        "shuttle.architecture.bundle",
    )
    _sha(bundle["transport_sha256"], "shuttle.architecture.bundle.transport_sha256")
    _sha(bundle["descriptor_sha256"], "shuttle.architecture.bundle.descriptor_sha256")
    for field in (
        "transport_schema_version",
        "device_schema_version",
        "invocation_abi_schema_version",
        "bundle_schema_version",
        "completion",
        "launch_count",
        "slot_count",
        "temporary_slot_count",
        "temporary_bytes",
    ):
        _equal(bundle[field], compiler[field], f"shuttle.architecture.bundle.{field}")
    _equal(
        bundle["external_bindings"],
        ["operand:0", "operand:1", "result:0"],
        "shuttle.architecture.bundle.external_bindings",
    )
    _equal(bundle["alias_count"], 0, "shuttle.architecture.bundle.alias_count")
    ptx = _closed(
        architecture["ptx"],
        {"assembler", "assembler_sha256", "slice_sha256", "all_sm90_assembled", "spill_bytes_per_slice"},
        "shuttle.architecture.ptx",
    )
    _string(ptx["assembler"], "shuttle.architecture.ptx.assembler")
    _sha(ptx["assembler_sha256"], "shuttle.architecture.ptx.assembler_sha256")
    _sha_list(ptx["slice_sha256"], 19, "shuttle.architecture.ptx.slice_sha256")
    _equal(ptx["all_sm90_assembled"], True, "shuttle.architecture.ptx.all_sm90_assembled")
    spills = _sequence(ptx["spill_bytes_per_slice"], "shuttle.architecture.ptx.spill_bytes_per_slice")
    if len(spills) != 19 or any(_integer(item, "spill bytes") != 0 for item in spills):
        raise AssertionError("every ABI 10 PTX slice must assemble with zero spill bytes")
    cache = _closed(
        architecture["cache"],
        {
            "compiler_options_sha256",
            "generated_bundle_artifact_sha256",
            "populate_worker_id",
            "reuse_worker_id",
            "populate_process_id",
            "reuse_process_id",
            "populate_before",
            "populate_after",
            "reuse_before",
            "reuse_after",
            "cache_trace_artifact_sha256",
            "populate_hits",
            "reuse_hits",
            "reuse_observer_events",
        },
        "shuttle.architecture.cache",
    )
    _equal(
        cache["compiler_options_sha256"],
        contract["compiler"]["canonical_options_sha256"],
        "shuttle.architecture.cache.compiler_options_sha256",
    )
    _equal(
        cache["generated_bundle_artifact_sha256"],
        observer["generated_bundle_artifact_sha256"],
        "shuttle.architecture.cache.generated_bundle_artifact_sha256",
    )
    populate_worker = _sha(cache["populate_worker_id"], "shuttle.architecture.cache.populate_worker_id")
    reuse_worker = _sha(cache["reuse_worker_id"], "shuttle.architecture.cache.reuse_worker_id")
    if populate_worker == reuse_worker:
        raise ValueError("cache populate and reuse workers must be distinct")
    populate_process = _integer(cache["populate_process_id"], "shuttle.architecture.cache.populate_process_id")
    reuse_process = _integer(cache["reuse_process_id"], "shuttle.architecture.cache.reuse_process_id")
    if populate_process == 0 or reuse_process == 0 or populate_process == reuse_process:
        raise ValueError("cache populate and reuse processes must be distinct and positive")
    populate_before = _cache_snapshot(cache["populate_before"], "shuttle.architecture.cache.populate_before")
    populate_after = _cache_snapshot(cache["populate_after"], "shuttle.architecture.cache.populate_after")
    reuse_before = _cache_snapshot(cache["reuse_before"], "shuttle.architecture.cache.reuse_before")
    reuse_after = _cache_snapshot(cache["reuse_after"], "shuttle.architecture.cache.reuse_after")
    if populate_before:
        raise ValueError("cache population must start from an empty cache")
    if len(populate_after) != 1:
        raise ValueError("cache population must add exactly one entry")
    _equal(reuse_before, populate_after, "shuttle.architecture.cache.reuse_before")
    _equal(reuse_after, populate_after, "shuttle.architecture.cache.reuse_after")
    _equal(
        cache["cache_trace_artifact_sha256"],
        _canonical_sha256({key: value for key, value in cache.items() if key != "cache_trace_artifact_sha256"}),
        "shuttle.architecture.cache.cache_trace_artifact_sha256",
    )
    _equal(cache["populate_hits"], 0, "shuttle.architecture.cache.populate_hits")
    _equal(cache["reuse_hits"], 1, "shuttle.architecture.cache.reuse_hits")
    _equal(cache["reuse_observer_events"], [], "shuttle.architecture.cache.reuse_observer_events")
    _equal(architecture["direct_and_jax_bitwise_equal"], True, "shuttle.architecture.direct_and_jax_bitwise_equal")
    return architecture


def _te_result(
    value: object,
    index: int,
    contract: Mapping[str, Any],
    collector: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    name = f"te_results[{index}]"
    result = _closed(
        value,
        {
            "backend_pair",
            "qualified_against_reference",
            "execution_identity_sha256",
            "execution_identity",
            "result_artifact_sha256",
            "sealed_run",
            "input_raw_sha256",
            "reference_output_array_digest",
            "output",
        },
        name,
    )
    pair = _sequence(result["backend_pair"], f"{name}.backend_pair")
    if len(pair) != 2 or tuple(pair) not in BACKEND_PAIRS:
        raise ValueError(f"{name}.backend_pair is not predeclared")
    _equal(result["qualified_against_reference"], True, f"{name}.qualified_against_reference")
    identity = validate_execution_identity(result["execution_identity"])
    _equal(
        result["execution_identity_sha256"],
        _canonical_sha256(identity),
        f"{name}.execution_identity_sha256",
    )
    sealed_run = _closed(
        result["sealed_run"],
        {
            "plan_id",
            "plan_sha256",
            "position",
            "result_path",
            "shape",
            "boundary",
            "forward_backend",
            "backward_backend",
        },
        f"{name}.sealed_run",
    )
    _equal(sealed_run["plan_id"], TE_PLAN_ID, f"{name}.sealed_run.plan_id")
    _sha(sealed_run["plan_sha256"], f"{name}.sealed_run.plan_sha256")
    _equal(sealed_run["position"], index, f"{name}.sealed_run.position")
    _equal(sealed_run["result_path"], f"result-{index:02d}.json", f"{name}.sealed_run.result_path")
    _equal(sealed_run["shape"], [2048, 4096], f"{name}.sealed_run.shape")
    _equal(sealed_run["boundary"], "forward", f"{name}.sealed_run.boundary")
    _equal(sealed_run["forward_backend"], pair[0], f"{name}.sealed_run.forward_backend")
    _equal(sealed_run["backward_backend"], pair[1], f"{name}.sealed_run.backward_backend")
    _equal(
        result["input_raw_sha256"],
        {"x": INPUT_RAW_SHA256[0], "gamma": INPUT_RAW_SHA256[1]},
        f"{name}.input_raw_sha256",
    )
    _equal(
        result["reference_output_array_digest"],
        REFERENCE_OUTPUT_ARRAY_DIGEST,
        f"{name}.reference_output_array_digest",
    )
    _equal(identity["hardware_class"], "h100", f"{name}.execution_identity.hardware_class")
    _equal(identity["marin_revision"], provenance["marin_revision"], f"{name}.execution_identity.marin_revision")
    for field in ("runner_binary_sha256", "harness_sha256", "build_manifest_sha256"):
        _equal(identity[field], collector[field], f"{name}.execution_identity.{field}")
    device = identity["device"]
    _equal(device["model"], provenance["device_model"], f"{name}.execution_identity.device.model")
    _equal(device["uuid"], provenance["device_uuid"], f"{name}.execution_identity.device.uuid")
    _equal(
        device["compute_capability"],
        provenance["compute_capability"],
        f"{name}.execution_identity.device.compute_capability",
    )
    _equal(
        device["physical_sm_count"],
        provenance["physical_sm_count"],
        f"{name}.execution_identity.device.physical_sm_count",
    )
    _output(result["output"], f"{name}.output", contract)
    _equal(
        result["result_artifact_sha256"],
        _canonical_sha256({key: value for key, value in result.items() if key != "result_artifact_sha256"}),
        f"{name}.result_artifact_sha256",
    )
    return result


def _collector(value: object) -> dict[str, Any]:
    collector = _closed(
        value,
        {
            "schema_version",
            "marin_revision",
            "runner_binary_sha256",
            "harness_sha256",
            "build_manifest_sha256",
            "artifact_manifest_sha256",
        },
        "collector",
    )
    _equal(collector["schema_version"], 1, "collector.schema_version")
    _revision(collector["marin_revision"], "collector.marin_revision")
    for field in ("runner_binary_sha256", "harness_sha256", "build_manifest_sha256", "artifact_manifest_sha256"):
        _sha(collector[field], f"collector.{field}")
    return collector


def _cache_snapshot(value: object, name: str) -> list[dict[str, Any]]:
    records = _sequence(value, name)
    keys: set[str] = set()
    for index, value in enumerate(records):
        record = _closed(value, {"key", "sha256", "size"}, f"{name}[{index}]")
        key = _string(record["key"], f"{name}[{index}].key")
        if key in keys:
            raise ValueError(f"{name} contains duplicate keys")
        keys.add(key)
        _sha(record["sha256"], f"{name}[{index}].sha256")
        if _integer(record["size"], f"{name}[{index}].size") == 0:
            raise ValueError(f"{name}[{index}].size must be positive")
    return records


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _provenance(value: object) -> dict[str, Any]:
    provenance = _closed(
        value,
        {
            "marin_revision",
            "clean_tree",
            "hardware_class",
            "device_model",
            "device_uuid",
            "compute_capability",
            "physical_sm_count",
            "driver_version",
            "cuda_runtime_version",
            "cuda_toolkit_version",
            "wheel_sha256",
            "cuda_plugin_sha256",
        },
        "provenance",
    )
    _revision(provenance["marin_revision"], "provenance.marin_revision")
    _equal(provenance["clean_tree"], True, "provenance.clean_tree")
    _equal(provenance["hardware_class"], "h100", "provenance.hardware_class")
    if "H100" not in _string(provenance["device_model"], "provenance.device_model"):
        raise ValueError("provenance.device_model is not H100")
    if not re.fullmatch(r"GPU-[0-9a-f-]+", _string(provenance["device_uuid"], "provenance.device_uuid")):
        raise ValueError("provenance.device_uuid is malformed")
    _equal(provenance["compute_capability"], "9.0", "provenance.compute_capability")
    if _integer(provenance["physical_sm_count"], "provenance.physical_sm_count") <= 0:
        raise ValueError("physical SM count must be positive")
    for field in ("driver_version", "cuda_runtime_version", "cuda_toolkit_version"):
        _string(provenance[field], f"provenance.{field}")
    _sha(provenance["wheel_sha256"], "provenance.wheel_sha256")
    _sha(provenance["cuda_plugin_sha256"], "provenance.cuda_plugin_sha256")
    return provenance


def _metrics(value: object, name: str) -> dict[str, float | int]:
    metrics = _closed(value, set(METRICS), name)
    for field in METRICS[:-1]:
        item = metrics[field]
        if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
            raise ValueError(f"{name}.{field} must be a finite nonnegative number")
    _integer(metrics[METRICS[-1]], f"{name}.{METRICS[-1]}")
    return metrics


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields differ from the closed schema")
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    return value


def _sha_list(value: object, count: int, name: str) -> list[str]:
    items = _sequence(value, name)
    if len(items) != count:
        raise ValueError(f"{name} must contain exactly {count} records")
    for index, item in enumerate(items):
        _sha(item, f"{name}[{index}]")
    return items


def _sha(value: object, name: str) -> str:
    value = _string(value, name)
    if not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _revision(value: object, name: str) -> str:
    value = _string(value, name)
    if not REVISION.fullmatch(value):
        raise ValueError(f"{name} must be a 40-character revision")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{name} differs from the closed contract")
    if type(expected) is dict:
        actual_mapping = cast(dict[Any, Any], actual)
        expected_mapping = cast(dict[Any, Any], expected)
        if set(actual_mapping) != set(expected_mapping):
            raise ValueError(f"{name} differs from the closed contract")
        for key in expected_mapping:
            _equal(actual_mapping[key], expected_mapping[key], f"{name}.{key}")
        return
    if type(expected) is list:
        actual_sequence = cast(list[Any], actual)
        expected_sequence = cast(list[Any], expected)
        if len(actual_sequence) != len(expected_sequence):
            raise ValueError(f"{name} differs from the closed contract")
        for index, (actual_item, expected_item) in enumerate(zip(actual_sequence, expected_sequence, strict=True)):
            _equal(actual_item, expected_item, f"{name}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{name} differs from the closed contract")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")
