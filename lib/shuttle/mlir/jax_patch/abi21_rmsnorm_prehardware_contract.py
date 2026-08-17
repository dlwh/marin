# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed validator for the unsealed ABI21 RMSNorm prehardware boundary."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from abi15_h100_numerics import bf16_sha256
from abi21_rmsnorm_numerics import (
    architecture_inputs,
    binary64_reference,
    error_metrics,
    schedule_reference,
)

CONTRACT_FILENAME = "abi21-rmsnorm-prehardware-v1.json"
ARCHITECTURE_FILENAME = "abi21-rmsnorm-prehardware-architecture-v1.json"
ROLE_MANIFEST_FILENAME = "abi21-rmsnorm-prehardware-role-paths-v1.json"
CONTRACT_SHA256 = "df0b288c54e12f749a438bf917978754825535c618e58bb759459af1c62d11ba"
ARCHITECTURE_SHA256 = "c055064bd84f072bd92eb3865ddc84ec643ff09077c8c5ca7f3030de942d6f89"
ROLE_MANIFEST_SHA256 = "b115df249b8d54053c6c42817d0f887dbdcab290b13e0b5205b497ce07e281bf"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

BOUNDARIES = ("forward", "backward", "composed")
POLICIES = ("source_ordered", "fast")
HARDWARE = ("h100", "b200")
READINESS_BLOCKERS = (
    "representative_numerical_evidence_missing",
    "sealed_build_identity_missing",
    "sealed_image_identity_missing",
    "controller_authority_missing",
    "sm100_codegen_missing",
    "h100_execution_missing",
    "b200_execution_missing",
    "performance_contract_pending",
)
RAW_COUNTS = {"forward": (19, 21), "backward": (48, 51), "composed": (51, 54)}
HOOK_COUNTS = {"forward": (19, 21), "backward": (43, 46), "composed": (46, 49)}
EXTERNAL_BINDINGS = {"forward": 3, "backward": 5, "composed": 6}
TASK_CENSUS = {"forward": (18, 1, 0), "backward": (38, 3, 2), "composed": (41, 3, 2)}
PUBLIC_PROJECTIONS = {"forward": (0,), "backward": (1, 0), "composed": (0, 2, 1)}
OUTPUT_ROLES = {
    "forward": ("result",),
    "backward": ("input_cotangent", "weight_cotangent"),
    "composed": ("result", "input_cotangent", "weight_cotangent"),
}
EXECUTION_RUNS = ("first", "sequential", "thread_0", "thread_1", "reuse")
COMPILE_WITNESS_ROLES = ("algebra", "plan", "schedule", "final_mlir", "descriptor", "ptx_witness")
STATIC_ROLES = {
    "architecture_contract": "contract/architecture.json",
    "contract": "contract/prehardware.json",
    "input_output_cotangent": "inputs/output-cotangent.bf16",
    "input_weight": "inputs/weight.bf16",
    "input_x": "inputs/x.bf16",
    "numerical_reference": "reference/schedule-derived.json",
    "role_manifest": "contract/role-paths.json",
    "source_allowlist": "contract/source-allowlist.json",
}
ANALYTIC_LIMITS = {
    "result": {
        "maximum_absolute_error": 0.0078125,
        "maximum_bf16_ulp_error": 1,
        "mean_absolute_error": 0.002,
        "relative_linf_error": 0.00390625,
    },
    "input_cotangent": {
        "maximum_absolute_error": 0.0078125,
        "maximum_bf16_ulp_error": 1,
        "mean_absolute_error": 0.002,
        "relative_linf_error": 0.00390625,
    },
    "weight_cotangent": {
        "maximum_absolute_error": 0.03125,
        "maximum_bf16_ulp_error": 1,
        "mean_absolute_error": 0.015625,
        "relative_linf_error": 0.00390625,
    },
}


@dataclass(frozen=True)
class PrehardwareContract:
    """Closed fields needed by local builders and readiness checks."""

    sha256: str
    contract_id: str
    status: str
    pipeline_abi_version: int
    architecture_shape: tuple[int, ...]
    metamorphic_shape: tuple[int, ...]
    acceptance_shape: tuple[int, ...]
    boundaries: tuple[str, ...]
    policies: tuple[str, ...]
    acceptance_hardware: tuple[str, ...]
    acceptance_cells: int
    representative_live_temporary_bytes: dict[str, int]
    numerical_policy_status: str
    readiness_blockers: tuple[str, ...]
    input_digests: dict[str, str]
    reference_output_digests: dict[str, str]


@dataclass(frozen=True)
class ArchitectureContract:
    """Independent projections of raw and XLA-hook bundle topology."""

    raw_fixture_counts: dict[str, tuple[int, int]]
    xla_hook_counts: dict[str, tuple[int, int]]
    external_bindings: dict[str, int]
    task_census: dict[str, tuple[int, int, int]]
    public_result_projections: dict[str, tuple[int, ...]]
    executed_ptx_slices: int


@dataclass(frozen=True)
class RoleManifest:
    """Closed generators for raw prehardware artifact paths."""

    boundaries: tuple[str, ...]
    policies: tuple[str, ...]
    calls: dict[str, tuple[int, ...]]
    output_roles: dict[str, tuple[str, ...]]
    execution_runs: tuple[str, ...]
    compile_witness_roles: tuple[str, ...]
    static_roles: dict[str, str]


def contract_path() -> Path:
    """Return the checked-in canonical prehardware contract path."""
    return Path(__file__).with_name(CONTRACT_FILENAME)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_exact(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} byte drift")
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if type(document) is not dict or raw != _canonical_json(document):
        raise ValueError(f"{name} must use canonical JSON")
    return document


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} must use the closed schema")
    return value


def _sha(value: object, name: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _relative(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _topology_record(
    value: object, boundary: str, expected_counts: tuple[int, int], expected_tasks: tuple[int, int, int]
) -> None:
    record = _closed(
        value,
        {
            "external_bindings",
            "launches",
            "public_result_projection",
            "slots",
            "task_census",
            "temporary_slots",
        },
        f"{boundary} topology",
    )
    if tuple(record["public_result_projection"]) != PUBLIC_PROJECTIONS[boundary]:
        raise ValueError(f"{boundary} public result projection drift")
    launches, slots = expected_counts
    if record["launches"] != launches or record["slots"] != slots:
        raise ValueError(f"{boundary} XLA hook or raw topology drift")
    tasks = _closed(record["task_census"], {"column_fold", "map", "row_fold"}, f"{boundary} task census")
    observed_tasks = (tasks["map"], tasks["row_fold"], tasks["column_fold"])
    if observed_tasks != expected_tasks or sum(observed_tasks) != record["launches"]:
        raise ValueError(f"{boundary} task census drift")
    if record["external_bindings"] != EXTERNAL_BINDINGS[boundary]:
        raise ValueError(f"{boundary} external binding drift")
    if record["temporary_slots"] != slots - EXTERNAL_BINDINGS[boundary]:
        raise ValueError(f"{boundary} temporary slot drift")


def validate_architecture_document(document: dict[str, Any]) -> None:
    """Validate raw-fixture, live-hook, projection, and rank-five geometry."""
    root = _closed(
        document,
        {
            "executed_matrix",
            "metamorphic_column_folds",
            "pipeline_abi_version",
            "public_boundaries",
            "raw_fixture_topology",
            "schema_version",
            "schemas",
            "status",
            "target",
            "xla_hook_topology",
        },
        "architecture contract",
    )
    if (
        root["schema_version"] != 1
        or root["pipeline_abi_version"] != 21
        or root["status"] != "local_unsealed_architecture_design"
        or root["target"] != "shuttle.gpu.executable_bundle.v2"
        or root["schemas"] != {"bundle": 2, "device": 3, "invocation": 3, "transport": 2}
    ):
        raise ValueError("architecture contract identity drift")
    if root["public_boundaries"] != {
        "forward": {"inputs": ["x", "weight"], "outputs": ["result"]},
        "backward": {
            "inputs": ["x", "weight", "output_cotangent"],
            "outputs": ["input_cotangent", "weight_cotangent"],
        },
        "composed": {
            "inputs": ["x", "weight", "output_cotangent"],
            "outputs": ["result", "input_cotangent", "weight_cotangent"],
        },
    }:
        raise ValueError("public boundary drift")
    raw = _closed(root["raw_fixture_topology"], set(BOUNDARIES), "raw fixture topology")
    hook = _closed(root["xla_hook_topology"], set(BOUNDARIES), "XLA hook topology")
    raw_tasks = {"forward": (18, 1, 0), "backward": (43, 3, 2), "composed": (46, 3, 2)}
    for boundary in BOUNDARIES:
        _topology_record(raw[boundary], boundary, RAW_COUNTS[boundary], raw_tasks[boundary])
        try:
            _topology_record(hook[boundary], boundary, HOOK_COUNTS[boundary], TASK_CENSUS[boundary])
        except ValueError as error:
            if "projection" in str(error) or "task census" in str(error):
                raise
            raise ValueError(f"{boundary} XLA hook topology drift") from error
    matrix = _closed(root["executed_matrix"], {"boundaries", "policies", "ptx_slices", "shape"}, "executed matrix")
    if matrix != {
        "boundaries": list(BOUNDARIES),
        "policies": list(POLICIES),
        "ptx_slices": 216,
        "shape": [2, 7, 19],
    }:
        raise ValueError("executed PTX matrix drift")
    folds = _closed(root["metamorphic_column_folds"], {"main", "singleton"}, "metamorphic ColumnFold")
    if folds != {
        "main": {
            "block_threads": 256,
            "feature_count": 263,
            "grid": 263,
            "reduction_extent": 330,
            "scratch_bytes": 1024,
            "serial_tiles": 2,
            "tile_shape": [256, 1, 1, 1, 1],
        },
        "singleton": {
            "block_threads": 32,
            "feature_count": 263,
            "grid": 263,
            "reduction_extent": 1,
            "scratch_bytes": 128,
            "serial_tiles": 1,
            "tile_shape": [1, 1, 1, 1, 1],
        },
    }:
        raise ValueError("metamorphic ColumnFold geometry drift")


def load_architecture_contract(path: Path | None = None) -> ArchitectureContract:
    """Load the exact architecture bytes and return their typed projection."""
    selected = Path(__file__).with_name(ARCHITECTURE_FILENAME) if path is None else path
    document = _load_exact(selected, ARCHITECTURE_SHA256, "ABI21 architecture contract")
    validate_architecture_document(document)
    return ArchitectureContract(
        RAW_COUNTS.copy(),
        HOOK_COUNTS.copy(),
        EXTERNAL_BINDINGS.copy(),
        TASK_CENSUS.copy(),
        PUBLIC_PROJECTIONS.copy(),
        216,
    )


def validate_role_manifest_document(document: dict[str, Any]) -> None:
    """Validate the closed role generator without expanding artifact files."""
    root = _closed(
        document,
        {
            "boundaries",
            "calls",
            "compile_witness_roles",
            "execution_runs",
            "output_roles",
            "policies",
            "schema_version",
            "static_roles",
            "status",
        },
        "role manifest",
    )
    if root["schema_version"] != 1 or root["status"] != "raw_subject_artifact_only":
        raise ValueError("role manifest identity drift")
    if tuple(root["boundaries"]) != BOUNDARIES or tuple(root["policies"]) != POLICIES:
        raise ValueError("role manifest coordinate drift")
    if tuple(root["execution_runs"]) != EXECUTION_RUNS or tuple(root["compile_witness_roles"]) != COMPILE_WITNESS_ROLES:
        raise ValueError("role manifest run or witness drift")
    calls = _closed(root["calls"], set(BOUNDARIES), "role manifest calls")
    if any(calls[boundary] != list(range(HOOK_COUNTS[boundary][0])) for boundary in BOUNDARIES):
        raise ValueError("role manifest call ordinal drift")
    outputs = _closed(root["output_roles"], set(BOUNDARIES), "role manifest outputs")
    if any(tuple(outputs[boundary]) != OUTPUT_ROLES[boundary] for boundary in BOUNDARIES):
        raise ValueError("role manifest output role drift")
    if root["static_roles"] != STATIC_ROLES or list(root["static_roles"]) != sorted(STATIC_ROLES):
        raise ValueError("role manifest static role drift")
    paths = [_relative(path, f"static role {role}") for role, path in root["static_roles"].items()]
    if len(paths) != len(set(paths)) or "accepted_bundle" in root["static_roles"]:
        raise ValueError("role manifest aliases a path or claims acceptance")


def load_role_manifest(path: Path | None = None) -> RoleManifest:
    """Load the exact role-manifest bytes and return their typed projection."""
    selected = Path(__file__).with_name(ROLE_MANIFEST_FILENAME) if path is None else path
    document = _load_exact(selected, ROLE_MANIFEST_SHA256, "ABI21 role manifest")
    validate_role_manifest_document(document)
    return RoleManifest(
        BOUNDARIES,
        POLICIES,
        {boundary: tuple(range(HOOK_COUNTS[boundary][0])) for boundary in BOUNDARIES},
        OUTPUT_ROLES.copy(),
        EXECUTION_RUNS,
        COMPILE_WITNESS_ROLES,
        STATIC_ROLES.copy(),
    )


def expected_role_paths(manifest: RoleManifest) -> dict[str, str]:
    """Expand the exact runtime and compile-only role roster."""
    roles = dict(manifest.static_roles)
    for boundary in manifest.boundaries:
        for output in manifest.output_roles[boundary]:
            roles[f"baseline_{boundary}_{output}"] = f"baseline/{boundary}/{output.replace('_', '-')}.bf16"
    for policy in manifest.policies:
        for boundary in manifest.boundaries:
            prefix = f"{policy}_{boundary}"
            roles[f"generated_mlir_{prefix}"] = f"compiler/{policy}/{boundary}/generated.mlir"
            roles[f"final_mlir_{prefix}"] = f"compiler/{policy}/{boundary}/final.mlir"
            roles[f"transport_{prefix}"] = f"compiler/{policy}/{boundary}/transport.bin"
            roles[f"descriptor_{prefix}"] = f"compiler/{policy}/{boundary}/descriptor.json"
            roles[f"observer_populate_{prefix}"] = f"observer/{policy}/{boundary}/populate.jsonl"
            roles[f"observer_reuse_{prefix}"] = f"observer/{policy}/{boundary}/reuse.jsonl"
            roles[f"cache_entry_{prefix}"] = f"cache/entries/{policy}/{boundary}.bin"
            for run in manifest.execution_runs:
                for output in manifest.output_roles[boundary]:
                    roles[f"output_{prefix}_{run}_{output}"] = (
                        f"outputs/{policy}/{boundary}/{run}/{output.replace('_', '-')}.bf16"
                    )
            for ordinal in manifest.calls[boundary]:
                roles[f"ptx_{prefix}_{ordinal:02d}"] = f"assembly/{policy}/{boundary}/entry-{ordinal:02d}.ptx"
                roles[f"cubin_{prefix}_{ordinal:02d}"] = f"assembly/{policy}/{boundary}/entry-{ordinal:02d}.cubin"
            for witness in manifest.compile_witness_roles:
                roles[f"metamorphic_{prefix}_{witness}"] = (
                    f"metamorphic/{policy}/{boundary}/{witness.replace('_', '-')}.json"
                    if witness != "final_mlir"
                    else f"metamorphic/{policy}/{boundary}/final.mlir"
                )
    if "accepted_bundle" in roles or len(roles) != len(set(roles.values())):
        raise ValueError("expanded role manifest aliases a path or claims acceptance")
    return dict(sorted(roles.items()))


def _validate_numerical_oracle(value: object) -> tuple[dict[str, str], dict[str, str]]:
    oracle = _closed(
        value,
        {"analytic_limits", "input_sha256", "reference_output_sha256", "schedule_recurrence"},
        "numerical oracle",
    )
    if oracle["schedule_recurrence"] != {
        "accumulator_type": "f32",
        "epsilon": "1e-6",
        "public_rounding": "round_nearest_even_bf16",
        "reduction_order": "tree_association_free_leaf_order_fixed",
        "rsqrt_realization": "sqrt_rn_f32_then_divide",
    }:
        raise ValueError("schedule-derived recurrence drift")
    inputs = architecture_inputs()
    outputs = schedule_reference(inputs)
    input_digests = {
        "x": bf16_sha256(inputs.x),
        "weight": bf16_sha256(inputs.weight),
        "output_cotangent": bf16_sha256(inputs.output_cotangent),
    }
    output_digests = {
        "result": bf16_sha256(outputs.result),
        "input_cotangent": bf16_sha256(outputs.input_cotangent),
        "weight_cotangent": bf16_sha256(outputs.weight_cotangent),
    }
    if oracle["input_sha256"] != input_digests:
        raise ValueError("deterministic input digest drift")
    if oracle["reference_output_sha256"] != output_digests:
        raise ValueError("schedule-derived output digest drift")
    limits = oracle["analytic_limits"]
    if limits != ANALYTIC_LIMITS:
        raise ValueError("binary64 analytic threshold drift")
    analytic = binary64_reference(inputs)
    for role in ("result", "input_cotangent", "weight_cotangent"):
        metrics = error_metrics(getattr(outputs, role), getattr(analytic, role))
        observed = {
            "maximum_absolute_error": metrics.maximum_absolute_error,
            "maximum_bf16_ulp_error": metrics.maximum_bf16_ulp_error,
            "mean_absolute_error": metrics.mean_absolute_error,
            "relative_linf_error": metrics.relative_linf_error,
        }
        if any(observed[name] > limit for name, limit in limits[role].items()):
            raise ValueError(f"{role} exceeds its binary64 analytic threshold")
    return input_digests, output_digests


def validate_contract_document(document: dict[str, Any]) -> None:
    """Validate the semantic contract without treating caller bytes as authority."""
    root = _closed(
        document,
        {
            "authority",
            "contract_id",
            "dependencies",
            "matrix",
            "numerical_oracle",
            "numerical_policy",
            "pipeline_abi_version",
            "process_contract",
            "readiness_blockers",
            "schema_version",
            "shape_tiers",
            "source_boundary",
            "status",
        },
        "prehardware contract",
    )
    if (
        root["schema_version"] != 1
        or root["contract_id"] != "shuttle_abi21_rmsnorm_prehardware_v1"
        or root["status"] != "local_unsealed_prehardware_design"
        or root["pipeline_abi_version"] != 21
    ):
        raise ValueError("prehardware contract identity drift")
    dependencies = _closed(
        root["dependencies"],
        {"architecture_contract", "architecture_contract_sha256", "role_manifest", "role_manifest_sha256"},
        "dependencies",
    )
    if (
        dependencies["architecture_contract"] != ARCHITECTURE_FILENAME
        or dependencies["architecture_contract_sha256"] != ARCHITECTURE_SHA256
    ):
        raise ValueError("architecture contract dependency drift")
    if (
        dependencies["role_manifest"] != ROLE_MANIFEST_FILENAME
        or dependencies["role_manifest_sha256"] != ROLE_MANIFEST_SHA256
    ):
        raise ValueError("role manifest dependency drift")
    _sha(dependencies["architecture_contract_sha256"], "architecture contract digest")
    _sha(dependencies["role_manifest_sha256"], "role manifest digest")
    tiers = _closed(root["shape_tiers"], {"acceptance", "architecture_execution", "metamorphic_compile"}, "shape tiers")
    if tiers["architecture_execution"] != {
        "dimensions": [2, 7, 19],
        "feature_count": 19,
        "flattened_rows": 14,
        "role": "untimed_runtime_cache_observer_numerical",
        "status": "prehardware_not_executed",
    } or tiers["metamorphic_compile"] != {
        "dimensions": [3, 2, 5, 11, 263],
        "feature_count": 263,
        "flattened_rows": 330,
        "role": "compile_only_rank_witness",
        "status": "local_only",
    }:
        raise ValueError("architecture or metamorphic tier drift")
    acceptance = _closed(
        tiers["acceptance"], {"dimensions", "hardware", "live_temporary_bytes", "role", "status"}, "acceptance tier"
    )
    if acceptance["dimensions"] != [2048, 4096] or acceptance["hardware"] != list(HARDWARE):
        raise ValueError("acceptance tier shape or hardware drift")
    if acceptance["live_temporary_bytes"] != {"backward": 167_804_936, "composed": 167_804_936}:
        raise ValueError("representative live temporary byte drift")
    if acceptance["role"] != "target1_scorecard" or acceptance["status"] != "blocked":
        raise ValueError("acceptance tier status drift")
    matrix = _closed(root["matrix"], {"acceptance_cells", "boundaries", "hardware", "policies"}, "boundary matrix")
    if matrix != {
        "acceptance_cells": 12,
        "boundaries": list(BOUNDARIES),
        "hardware": list(HARDWARE),
        "policies": list(POLICIES),
    }:
        raise ValueError("boundary matrix drift")
    policy = _closed(
        root["numerical_policy"],
        {
            "analytic_thresholds_required",
            "imported_target1_v1",
            "normalized_code_equality_required",
            "ordinary_jax_role",
            "schedule_reference_required",
            "status",
        },
        "numerical policy",
    )
    if policy["imported_target1_v1"] is not False:
        raise ValueError("immutable Target-1 v1 numerical gate must not be imported")
    if policy["ordinary_jax_role"] != "recorded_comparator_optional_stronger_result":
        raise ValueError("ordinary JAX comparator must not become a bitwise acceptance gate")
    if policy["normalized_code_equality_required"] is not True:
        raise ValueError("SOURCE_ORDERED and FAST policy identity proof drift")
    if policy != {
        "analytic_thresholds_required": True,
        "imported_target1_v1": False,
        "normalized_code_equality_required": True,
        "ordinary_jax_role": "recorded_comparator_optional_stronger_result",
        "schedule_reference_required": True,
        "status": "predeclared_unexecuted",
    }:
        raise ValueError("numerical policy drift")
    if root["authority"] != {
        "build_manifest_sha256": None,
        "controller_trust_root_sha256": None,
        "image_digest": None,
        "launch_authorization_sha256": None,
        "source_capsule_sha256": None,
        "status": "not_authorized",
    }:
        raise ValueError("hardware authority must remain explicitly unsealed")
    if tuple(root["readiness_blockers"]) != READINESS_BLOCKERS:
        raise ValueError("readiness blocker drift")
    if root["process_contract"] != {
        "cache_entries_after_populate": 6,
        "observer_events_after_populate": 18,
        "observer_phases": ["algebra_coverage", "generated_bundle", "final_erasure"],
        "populate_hits": 0,
        "processes": ["baseline", "populate", "reuse"],
        "reuse_hits": 6,
        "reuse_observer_events": 0,
    }:
        raise ValueError("process, cache, or observer contract drift")
    if root["source_boundary"] != {
        "accumulator_type": "f32",
        "automatic_differentiation": "jax_owned_vjp",
        "epsilon": "1e-6",
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "public_type": "bf16",
    }:
        raise ValueError("ordinary-JAX source boundary drift")
    _validate_numerical_oracle(root["numerical_oracle"])


def load_contract(path: Path | None = None) -> PrehardwareContract:
    """Load the exact prehardware bytes and recompute every local dependency."""
    selected = contract_path() if path is None else path
    document = _load_exact(selected, CONTRACT_SHA256, "ABI21 RMSNorm prehardware contract")
    validate_contract_document(document)
    architecture_path = Path(__file__).with_name(ARCHITECTURE_FILENAME)
    role_path = Path(__file__).with_name(ROLE_MANIFEST_FILENAME)
    if hashlib.sha256(architecture_path.read_bytes()).hexdigest() != ARCHITECTURE_SHA256:
        raise ValueError("architecture contract byte drift")
    if hashlib.sha256(role_path.read_bytes()).hexdigest() != ROLE_MANIFEST_SHA256:
        raise ValueError("role manifest byte drift")
    load_architecture_contract()
    load_role_manifest()
    tiers = document["shape_tiers"]
    inputs, outputs = _validate_numerical_oracle(document["numerical_oracle"])
    return PrehardwareContract(
        CONTRACT_SHA256,
        document["contract_id"],
        document["status"],
        document["pipeline_abi_version"],
        tuple(tiers["architecture_execution"]["dimensions"]),
        tuple(tiers["metamorphic_compile"]["dimensions"]),
        tuple(tiers["acceptance"]["dimensions"]),
        BOUNDARIES,
        POLICIES,
        HARDWARE,
        document["matrix"]["acceptance_cells"],
        dict(tiers["acceptance"]["live_temporary_bytes"]),
        document["numerical_policy"]["status"],
        READINESS_BLOCKERS,
        inputs,
        outputs,
    )


def readiness_blockers() -> tuple[str, ...]:
    """Return the exact predeclared reasons external execution is forbidden."""
    return load_contract().readiness_blockers
