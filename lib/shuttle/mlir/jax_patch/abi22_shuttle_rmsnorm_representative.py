# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""File-backed representative RMSNorm evidence for the ABI22 GPU runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import ml_dtypes
import numpy as np
from abi22_te_rmsnorm_oracle_v2 import (
    POLICY_V4_ACCEPTANCE_INPUT_DIGESTS,
    POLICY_V4_ACCEPTANCE_LIMITS,
    POLICY_V4_ACCEPTANCE_PATTERN,
    POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS,
    POLICY_V4_PARTITION_COUNTS,
    error_metrics_v4,
    prepare_policy_v4_cases,
    repeat_output_diagnostics_v4,
    validate_policy_v4_evidence,
)

ACCEPTANCE_PATTERN = POLICY_V4_ACCEPTANCE_PATTERN
REPRESENTATIVE_SHAPE = (2048, 4096)
POLICIES = ("source_ordered", "fast")
BOUNDARIES = ("forward", "backward_recompute", "composed")
RESULT_ROLES = {
    "forward": ("y",),
    "backward_recompute": ("dx", "dgamma"),
    "composed": ("y", "dx", "dgamma"),
}
ROLE_LIMITS = {
    role: {
        "absolute_floor": limits["absolute_floor"],
        "below_floor_count": POLICY_V4_PARTITION_COUNTS[ACCEPTANCE_PATTERN][role]["below_floor"],
        "eligible_count": POLICY_V4_PARTITION_COUNTS[ACCEPTANCE_PATTERN][role]["above_floor"],
        "max_absolute_error_below_floor": limits["max_absolute_error_below_floor"],
        "max_absolute_error_above_floor": limits["max_absolute_error_above_floor"],
        "max_mean_absolute_error": limits["max_mean_absolute_error"],
        "max_relative_error_above_floor": limits["max_relative_error_above_absolute_floor"],
        "max_bfloat16_ulp_above_floor": limits["max_bfloat16_ulp_above_absolute_floor"],
    }
    for role, limits in POLICY_V4_ACCEPTANCE_LIMITS.items()
}
_FULL_BYTES = REPRESENTATIVE_SHAPE[0] * REPRESENTATIVE_SHAPE[1] * 2
_FEATURE_BYTES = REPRESENTATIVE_SHAPE[1] * 2
RESOURCE_CAPS = {
    "maximum_launch_count": 256,
    "maximum_slot_count": 256,
    "maximum_temporary_bytes": 256 * 1024 * 1024,
}
_X = {
    "role": "x",
    "runner_index": 0,
    "required_bytes": _FULL_BYTES,
    "tensor_type": "bf16[2048,4096]",
    "alignment": 2,
}
_GAMMA = {
    "role": "gamma",
    "runner_index": 1,
    "required_bytes": _FEATURE_BYTES,
    "tensor_type": "bf16[4096]",
    "alignment": 2,
}
_DY = {
    "role": "dy",
    "runner_index": 2,
    "required_bytes": _FULL_BYTES,
    "tensor_type": "bf16[2048,4096]",
    "alignment": 2,
}
BINDING_CONTRACT = {
    "forward": {
        "operands": (_X, _GAMMA),
        "results": (
            {
                "role": "y",
                "public_result_index": 0,
                "runner_index": 0,
                "required_bytes": _FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
        ),
    },
    "backward_recompute": {
        "operands": (_X, _GAMMA, _DY),
        "results": (
            {
                "role": "dx",
                "public_result_index": 0,
                "runner_index": 0,
                "required_bytes": _FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dgamma",
                "public_result_index": 1,
                "runner_index": 1,
                "required_bytes": _FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
        ),
    },
    "composed": {
        "operands": (_X, _GAMMA, _DY),
        "results": (
            {
                "role": "y",
                "public_result_index": 0,
                "runner_index": 0,
                "required_bytes": _FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dx",
                "public_result_index": 1,
                "runner_index": 1,
                "required_bytes": _FULL_BYTES,
                "tensor_type": "bf16[2048,4096]",
                "alignment": 2,
            },
            {
                "role": "dgamma",
                "public_result_index": 2,
                "runner_index": 2,
                "required_bytes": _FEATURE_BYTES,
                "tensor_type": "bf16[4096]",
                "alignment": 2,
            },
        ),
    },
}
_RUNNER_API = {
    "host": "GpuRunner::Execute_BlockHostUntilDone",
    "cuda_event": "GpuRunner::ExecuteTimed",
}
_CALL_SEQUENCE = {
    "forward": ["shuttle_rmsnorm_forward"],
    "backward_recompute": ["shuttle_rmsnorm_backward_recompute"],
    "composed": ["shuttle_rmsnorm_composed"],
}
_R12_RESULT_SHA256 = "bce080011168c768c70a6aca8dcbae82f8d021c1f4fae7610e57476c2b4b2439"
_R12_POLICY_RAW_SHA256 = "92fbb8ad362b8d28c005c4e2edd34e8690c79016f8a24c5a93d388b13563bcf3"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _shape(role: str) -> tuple[int, ...]:
    return REPRESENTATIVE_SHAPE if role != "dgamma" else (REPRESENTATIVE_SHAPE[1],)


def _read_bf16(path: Path, role: str) -> np.ndarray:
    expected = math.prod(_shape(role))
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != expected:
        raise ValueError(f"BF16 output size drifted: {path}")
    return bits.view(ml_dtypes.bfloat16).reshape(_shape(role))


def _validate_binding(raw: Mapping[str, object], expected: Mapping[str, object], *, kind: str, slot_count: int) -> int:
    expected_fields = {"kind": kind, **expected}
    if kind == "result":
        expected_fields["artifact"] = raw.get("artifact")
    if set(raw) != {*expected_fields, "slot_ordinal"}:
        raise ValueError(f"raw {kind} binding fields drifted")
    for field, value in expected_fields.items():
        if raw.get(field) != value:
            raise ValueError(f"raw {kind} binding {field} drifted")
    slot = raw.get("slot_ordinal")
    if type(slot) is not int or slot < 0 or slot >= slot_count:
        raise ValueError(f"raw {kind} binding slot is invalid")
    return slot


def _validate_resources(resources: object, *, boundary: str, bindings: list[object]) -> dict[str, int]:
    fields = {
        "launch_count",
        "slot_count",
        "external_operand_count",
        "external_result_count",
        "temporary_allocation_count",
        "temporary_bytes",
    }
    if not isinstance(resources, Mapping) or set(resources) != fields:
        raise ValueError("raw runner resource fields drifted")
    if any(type(resources.get(field)) is not int for field in fields):
        raise ValueError("raw runner resources must be integer observations")
    observed = {field: int(resources[field]) for field in fields}
    if not 0 < observed["launch_count"] <= RESOURCE_CAPS["maximum_launch_count"]:
        raise ValueError("raw runner launch count exceeds the generic cap")
    if not 0 < observed["slot_count"] <= RESOURCE_CAPS["maximum_slot_count"]:
        raise ValueError("raw runner slot count exceeds the generic cap")
    if not 0 <= observed["temporary_bytes"] <= RESOURCE_CAPS["maximum_temporary_bytes"]:
        raise ValueError("raw runner temporary bytes exceed the generic cap")
    if observed["temporary_allocation_count"] < 0:
        raise ValueError("raw runner temporary allocation count is negative")
    operand_count = sum(isinstance(binding, Mapping) and binding.get("kind") == "operand" for binding in bindings)
    result_count = sum(isinstance(binding, Mapping) and binding.get("kind") == "result" for binding in bindings)
    if operand_count != len(BINDING_CONTRACT[boundary]["operands"]):
        raise ValueError("raw external operand arity drifted")
    if result_count != len(BINDING_CONTRACT[boundary]["results"]):
        raise ValueError("raw external result arity drifted")
    if observed["external_operand_count"] != operand_count or observed["external_result_count"] != result_count:
        raise ValueError("raw external resource counts drifted")
    if observed["temporary_allocation_count"] == 0 and observed["temporary_bytes"] != 0:
        raise ValueError("raw zero-temporary plan reports temporary bytes")
    if observed["temporary_allocation_count"] > 0 and observed["temporary_bytes"] <= 0:
        raise ValueError("raw temporary plan reports no temporary bytes")
    return observed


def _validate_optimization(
    transport: Mapping[str, object], resources: Mapping[str, int], derived_temporary_traffic: int
) -> None:
    integer_fields = {
        "materialized_baseline_launch_count",
        "materialized_baseline_temporary_traffic_bytes",
        "fused_temporary_traffic_bytes",
    }
    if any(type(transport.get(field)) is not int or int(transport[field]) < 0 for field in integer_fields):
        raise ValueError("raw modeled optimization fields drifted")
    baseline_launches = int(transport["materialized_baseline_launch_count"])
    baseline_traffic = int(transport["materialized_baseline_temporary_traffic_bytes"])
    fused_traffic = int(transport["fused_temporary_traffic_bytes"])
    if resources["launch_count"] > baseline_launches or fused_traffic > baseline_traffic:
        raise ValueError("fused plan regresses modeled launch or temporary traffic")
    if resources["launch_count"] == baseline_launches and fused_traffic == baseline_traffic:
        raise ValueError("fused plan has no strict modeled improvement")
    if fused_traffic != derived_temporary_traffic:
        raise ValueError("fused temporary traffic is inconsistent with task ownership")


def _validate_task_graph(
    value: object,
    temporary_value: object,
    resources: Mapping[str, int],
    operand_slots: set[int],
    result_slots: set[int],
) -> int:
    if not isinstance(value, list) or len(value) != resources["launch_count"]:
        raise ValueError("raw task graph size drifted")
    tasks: dict[int, Mapping[str, object]] = {}
    for task in value:
        if not isinstance(task, Mapping) or set(task) != {
            "task_id",
            "dependencies",
            "input_slots",
            "output_slots",
        }:
            raise ValueError("raw task graph fields drifted")
        task_id = task.get("task_id")
        if type(task_id) is not int or task_id in tasks:
            raise ValueError("raw task identifiers drifted")
        for field in ("dependencies", "input_slots", "output_slots"):
            ordinals = task.get(field)
            if not isinstance(ordinals, list) or any(type(ordinal) is not int for ordinal in ordinals):
                raise ValueError(f"raw task {field} drifted")
            if len(set(ordinals)) != len(ordinals):
                raise ValueError(f"raw task {field} contains duplicates")
        for slot in [*task["input_slots"], *task["output_slots"]]:
            if slot < 0 or slot >= resources["slot_count"]:
                raise ValueError("raw task graph references an invalid slot")
        tasks[task_id] = task
    for task in tasks.values():
        if any(dependency not in tasks for dependency in task["dependencies"]):
            raise ValueError("raw task graph references an unknown dependency")
    if len(operand_slots) != resources["external_operand_count"]:
        raise ValueError("raw task graph external operand ownership count drifted")
    if len(result_slots) != resources["external_result_count"]:
        raise ValueError("raw task graph public result ownership count drifted")
    if operand_slots & result_slots:
        raise ValueError("raw external operand and result slots alias")
    if not isinstance(temporary_value, list):
        raise ValueError("raw temporary slot descriptors drifted")
    temporary_slots: dict[int, tuple[int, int]] = {}
    for temporary in temporary_value:
        if not isinstance(temporary, Mapping) or set(temporary) != {
            "slot_ordinal",
            "required_bytes",
            "reuse_group",
        }:
            raise ValueError("raw temporary slot descriptor fields drifted")
        slot = temporary.get("slot_ordinal")
        required_bytes = temporary.get("required_bytes")
        reuse_group = temporary.get("reuse_group")
        if (
            type(slot) is not int
            or slot < 0
            or slot >= resources["slot_count"]
            or slot in temporary_slots
            or type(required_bytes) is not int
            or required_bytes <= 0
            or required_bytes >= _FULL_BYTES
            or type(reuse_group) is not int
            or reuse_group < 0
        ):
            raise ValueError("raw temporary slot descriptor is invalid")
        temporary_slots[slot] = (required_bytes, reuse_group)
    all_slots = operand_slots | result_slots | set(temporary_slots)
    if all_slots != set(range(resources["slot_count"])):
        raise ValueError("raw logical slot ownership is incomplete")
    allocation_bytes: dict[int, int] = {}
    for required_bytes, reuse_group in temporary_slots.values():
        allocation_bytes[reuse_group] = max(allocation_bytes.get(reuse_group, 0), required_bytes)
    if len(allocation_bytes) != resources["temporary_allocation_count"]:
        raise ValueError("raw temporary reuse-group count drifted")
    if sum(allocation_bytes.values()) != resources["temporary_bytes"]:
        raise ValueError("raw temporary allocation bytes drifted")
    task_positions = {task["task_id"]: position for position, task in enumerate(value)}
    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {slot: [] for slot in range(resources["slot_count"])}
    for task in value:
        task_id = task["task_id"]
        if not task["output_slots"]:
            raise ValueError("raw task has no output ownership")
        for slot in task["output_slots"]:
            if slot in operand_slots:
                raise ValueError("raw task overwrites an external operand")
            if slot in producers:
                raise ValueError("raw slot has multiple task producers")
            producers[slot] = task_id
        for slot in task["input_slots"]:
            consumers[slot].append(task_id)
    if any(slot not in producers for slot in result_slots):
        raise ValueError("raw public result has no task producer")
    if set(producers) != set(temporary_slots) | result_slots:
        raise ValueError("raw task output ownership is incomplete")
    if any(not consumers[slot] for slot in operand_slots):
        raise ValueError("raw external operand is unused")
    if any(not consumers[slot] for slot in temporary_slots):
        raise ValueError("raw temporary has no consumer")
    for task in value:
        task_id = task["task_id"]
        expected_dependencies = {producers[slot] for slot in task["input_slots"] if slot in producers}
        if set(task["dependencies"]) != expected_dependencies:
            raise ValueError("raw task dependencies do not match input ownership")
        if any(task_positions[dependency] >= task_positions[task_id] for dependency in expected_dependencies):
            raise ValueError("raw temporary lifetime is not topologically ordered")
    lifetimes: dict[int, tuple[int, int]] = {}
    for slot in temporary_slots:
        producer_position = task_positions[producers[slot]]
        consumer_positions = [task_positions[consumer] for consumer in consumers[slot]]
        if min(consumer_positions) <= producer_position:
            raise ValueError("raw temporary is consumed before its producer")
        lifetimes[slot] = (producer_position, max(consumer_positions))
    for left, (_left_bytes, left_group) in temporary_slots.items():
        for right, (_right_bytes, right_group) in temporary_slots.items():
            if left >= right or left_group != right_group:
                continue
            left_start, left_end = lifetimes[left]
            right_start, right_end = lifetimes[right]
            if left_start <= right_end and right_start <= left_end:
                raise ValueError("raw temporary reuse lifetimes overlap")
    return sum(
        required_bytes * (1 + len(consumers[slot])) for slot, (required_bytes, _reuse_group) in temporary_slots.items()
    )


def prepare_representative_cases(case_root: Path) -> dict[str, dict[str, str]]:
    """Materialize and verify the sealed v4 inputs and references."""
    expected = {
        "input_digests": dict(POLICY_V4_ACCEPTANCE_INPUT_DIGESTS),
        "reference_digests": dict(POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS),
    }
    expected_files = {
        "x.bf16": (_FULL_BYTES, expected["input_digests"]["x"]),
        "gamma.bf16": (_FEATURE_BYTES, expected["input_digests"]["gamma"]),
        "dy.bf16": (_FULL_BYTES, expected["input_digests"]["dy"]),
        "reference_y.bf16": (_FULL_BYTES, expected["reference_digests"]["y"]),
        "reference_dx.bf16": (_FULL_BYTES, expected["reference_digests"]["dx"]),
        "reference_dgamma.bf16": (_FEATURE_BYTES, expected["reference_digests"]["dgamma"]),
    }
    acceptance_directory = case_root / ACCEPTANCE_PATTERN
    if all(
        (path := acceptance_directory / name).is_file() and path.stat().st_size == size and _sha256_file(path) == digest
        for name, (size, digest) in expected_files.items()
    ):
        return expected
    prepared = prepare_policy_v4_cases(case_root)
    acceptance = prepared.get(ACCEPTANCE_PATTERN)
    if acceptance != expected:
        raise ValueError("v4 acceptance case provenance drifted")
    return expected


def execution_plan(*, runner: Path, gpu_transport_test: Path, case_root: Path, raw_root: Path) -> dict[str, object]:
    """Return the exact fresh-process SO/FAST F/B/C execution matrix."""
    runs = []
    for policy in POLICIES:
        for boundary in BOUNDARIES:
            for repeat in range(2):
                runs.append(
                    {
                        "policy": policy,
                        "boundary": boundary,
                        "repeat": repeat,
                        "argv": [
                            str(runner),
                            str(gpu_transport_test),
                            "--case-root",
                            str(case_root),
                            "--raw-root",
                            str(raw_root),
                            "--policy",
                            policy,
                            "--boundary",
                            boundary,
                            "--repeat",
                            str(repeat),
                        ],
                    }
                )
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_plan_v1",
        "fresh_process_per_coordinate": True,
        "runs": runs,
    }


def validate_output_artifact(value: Mapping[str, object], *, raw_root: Path, case_root: Path) -> dict[str, object]:
    """Validate one regular, non-case-aliased output artifact."""
    if set(value) != {"path", "byte_count", "sha256"}:
        raise ValueError("output artifact fields drifted")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise ValueError("output artifact path drifted")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("output artifact must be a regular file")
    resolved = path.resolve()
    try:
        resolved.relative_to(raw_root.resolve())
    except ValueError as error:
        raise ValueError("output artifact is outside raw output storage") from error
    case_resolved = case_root.resolve()
    try:
        resolved.relative_to(case_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("output artifact aliases case storage")
    for reference in case_root.rglob("*.bf16"):
        if reference.is_file() and os.path.samefile(resolved, reference):
            raise ValueError("output artifact aliases a case file")
    payload = resolved.read_bytes()
    if type(value.get("byte_count")) is not int or value["byte_count"] != len(payload):
        raise ValueError("output artifact byte count drifted")
    digest = _digest(value.get("sha256"), "output artifact digest")
    if digest != _sha256_bytes(payload):
        raise ValueError("output artifact digest drifted")
    return dict(value)


def _positive_samples(value: object, label: str) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 20:
        raise ValueError(f"{label} sample count drifted")
    for sample in value:
        if type(sample) not in (int, float) or not math.isfinite(float(sample)) or sample <= 0:
            raise ValueError(f"{label} samples must be positive and finite")
    return value


def _validate_raw_cell(
    raw: Mapping[str, object],
    *,
    policy: str,
    boundary: str,
    repeat: int,
    raw_root: Path,
    case_root: Path,
) -> dict[str, object]:
    required = {
        "schema",
        "policy",
        "boundary",
        "repeat",
        "runner_api",
        "external_bindings",
        "resources",
        "transport",
        "timing",
    }
    if set(raw) != required or raw.get("schema") != "shuttle_abi22_rmsnorm_representative_raw_cell_v2":
        raise ValueError("raw runner cell schema drifted")
    if (raw.get("policy"), raw.get("boundary"), raw.get("repeat")) != (policy, boundary, repeat):
        raise ValueError("raw runner coordinate drifted")
    if raw.get("runner_api") != _RUNNER_API:
        raise ValueError("raw runner API drifted")
    bindings = raw.get("external_bindings")
    if not isinstance(bindings, list):
        raise ValueError("raw external bindings drifted")
    resources = _validate_resources(raw.get("resources"), boundary=boundary, bindings=bindings)
    expected_operands = BINDING_CONTRACT[boundary]["operands"]
    expected_results = BINDING_CONTRACT[boundary]["results"]
    if len(bindings) != len(expected_operands) + len(expected_results):
        raise ValueError("raw external binding count drifted")
    external_slots = set()
    operand_slots: set[int] = set()
    result_slots: set[int] = set()
    for raw_binding, expected in zip(bindings[: len(expected_operands)], expected_operands, strict=True):
        if not isinstance(raw_binding, Mapping):
            raise ValueError("raw operand binding is not an object")
        operand_slot = _validate_binding(raw_binding, expected, kind="operand", slot_count=resources["slot_count"])
        external_slots.add(operand_slot)
        operand_slots.add(operand_slot)
    outputs = []
    case_directory = case_root / ACCEPTANCE_PATTERN
    for raw_binding, expected in zip(bindings[len(expected_operands) :], expected_results, strict=True):
        artifact = raw_binding.get("artifact") if isinstance(raw_binding, Mapping) else None
        if not isinstance(raw_binding, Mapping) or not isinstance(artifact, Mapping):
            raise ValueError("raw result binding vector drifted")
        result_slot = _validate_binding(raw_binding, expected, kind="result", slot_count=resources["slot_count"])
        external_slots.add(result_slot)
        result_slots.add(result_slot)
        validated_artifact = validate_output_artifact(artifact, raw_root=raw_root, case_root=case_root)
        role = expected["role"]
        actual = _read_bf16(Path(validated_artifact["path"]), role)
        reference_path = case_directory / f"reference_{role}.bf16"
        reference = _read_bf16(reference_path, role)
        outputs.append(
            {
                "role": role,
                "public_result_index": expected["public_result_index"],
                "runner_result_index": expected["runner_index"],
                "slot_ordinal": raw_binding["slot_ordinal"],
                "digest": validated_artifact["sha256"],
                "reference_digest": _sha256_file(reference_path),
                "artifact": validated_artifact,
                **error_metrics_v4(actual, reference, role=role),
            }
        )
    transport = raw.get("transport")
    if not isinstance(transport, Mapping) or set(transport) != {
        "contract_operation_count",
        "code_bytes",
        "executable_digest",
        "module_load_digest",
        "ptx_digests",
        "materialized_baseline_launch_count",
        "materialized_baseline_temporary_traffic_bytes",
        "fused_temporary_traffic_bytes",
        "task_graph",
        "temporary_slots",
    }:
        raise ValueError("raw transport fields drifted")
    if transport.get("contract_operation_count") != 0:
        raise ValueError("representative RMSNorm must not fuse Contract")
    if type(transport.get("code_bytes")) is not int or transport["code_bytes"] <= 0:
        raise ValueError("raw code byte count drifted")
    _digest(transport.get("executable_digest"), "executable digest")
    _digest(transport.get("module_load_digest"), "module load digest")
    ptx_digests = transport.get("ptx_digests")
    if not isinstance(ptx_digests, list) or len(ptx_digests) != resources["launch_count"]:
        raise ValueError("raw PTX digest vector drifted")
    for digest in ptx_digests:
        _digest(digest, "PTX digest")
    derived_temporary_traffic = _validate_task_graph(
        transport.get("task_graph"),
        transport.get("temporary_slots"),
        resources,
        operand_slots,
        result_slots,
    )
    if len(external_slots) != len(bindings):
        raise ValueError("raw external bindings alias one internal slot")
    _validate_optimization(transport, resources, derived_temporary_traffic)
    timing = raw.get("timing")
    if not isinstance(timing, Mapping) or set(timing) != {
        "warmup_iterations",
        "measured_iterations",
        "host_clock",
        "host_samples_ns",
        "cuda_event_status",
        "cuda_event_samples_ns",
    }:
        raise ValueError("raw timing fields drifted")
    if timing.get("warmup_iterations") != 10 or timing.get("measured_iterations") != 20:
        raise ValueError("raw timing iteration count drifted")
    if timing.get("host_clock") != "steady_clock_execute_block_host_until_done_ns":
        raise ValueError("raw host timing clock drifted")
    if timing.get("cuda_event_status") != "supplemental_not_acceptance":
        raise ValueError("raw CUDA-event status drifted")
    _positive_samples(timing.get("host_samples_ns"), "host")
    _positive_samples(timing.get("cuda_event_samples_ns"), "CUDA-event")
    input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_cell_v2",
        "producer": "shuttle_gpu_executable_bundle",
        "input_pattern": ACCEPTANCE_PATTERN,
        "shape": list(REPRESENTATIVE_SHAPE),
        "epsilon": 1e-6,
        "policy": policy,
        "boundary": boundary,
        "repeat": repeat,
        "inputs": {role: POLICY_V4_ACCEPTANCE_INPUT_DIGESTS[role] for role in input_roles},
        "call_sequence": _CALL_SEQUENCE[boundary],
        "outputs": outputs,
        "runner_api": dict(raw["runner_api"]),
        "resources": dict(raw["resources"]),
        "transport": dict(transport),
        "timing": dict(timing),
    }


def _raw_cell_path(raw_root: Path, policy: str, boundary: str, repeat: int) -> Path:
    return raw_root / policy / boundary / f"repeat-{repeat}" / "cell.json"


def _load_cells(raw_root: Path, case_root: Path) -> list[dict[str, object]]:
    cells = []
    for policy in POLICIES:
        for boundary in BOUNDARIES:
            for repeat in range(2):
                path = _raw_cell_path(raw_root, policy, boundary, repeat)
                if not path.is_file():
                    raise ValueError(f"raw runner cell is missing: {path}")
                try:
                    raw = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(f"raw runner cell is unreadable: {path}") from error
                if not isinstance(raw, Mapping):
                    raise ValueError("raw runner cell is not an object")
                cells.append(
                    _validate_raw_cell(
                        raw,
                        policy=policy,
                        boundary=boundary,
                        repeat=repeat,
                        raw_root=raw_root,
                        case_root=case_root,
                    )
                )
    return cells


def _repeat_diagnostics(cells: list[dict[str, object]]) -> dict[str, object]:
    by_coordinate = {(cell["policy"], cell["boundary"], cell["repeat"]): cell for cell in cells}
    coordinates = []
    for policy in POLICIES:
        for boundary in BOUNDARIES:
            for role_index, role in enumerate(RESULT_ROLES[boundary]):
                first = by_coordinate[(policy, boundary, 0)]["outputs"][role_index]
                second = by_coordinate[(policy, boundary, 1)]["outputs"][role_index]
                first_array = _read_bf16(Path(first["artifact"]["path"]), role)
                second_array = _read_bf16(Path(second["artifact"]["path"]), role)
                coordinates.append(
                    {
                        "policy": policy,
                        "boundary": boundary,
                        "role": role,
                        "repeat0_artifact_sha256": first["artifact"]["sha256"],
                        "repeat1_artifact_sha256": second["artifact"]["sha256"],
                        **repeat_output_diagnostics_v4(first_array, second_array),
                    }
                )
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_repeat_diagnostics_v1",
        "coordinates": coordinates,
    }


def _external_evidence(r12_result: Path, r12_policy_raw: Path) -> dict[str, str]:
    result_sha256 = _sha256_file(r12_result)
    raw_sha256 = _sha256_file(r12_policy_raw)
    if result_sha256 != _R12_RESULT_SHA256 or raw_sha256 != _R12_POLICY_RAW_SHA256:
        raise ValueError("sealed r12 evidence identity drifted")
    result = json.loads(r12_result.read_text())
    raw = json.loads(r12_policy_raw.read_text())
    if result.get("schema") != "shuttle_track2_te_h100_batch_result_v4":
        raise ValueError("r12 result schema drifted")
    acceptance = result.get("evidence", {}).get("acceptance", {})
    if (
        acceptance.get("status") != "heldout_splitmix64_v3_numerics_accepted"
        or acceptance.get("accepted_input_pattern") != ACCEPTANCE_PATTERN
        or acceptance.get("accepted_shape") != list(REPRESENTATIVE_SHAPE)
    ):
        raise ValueError("r12 acceptance evidence drifted")
    validate_policy_v4_evidence(raw)
    return {
        "transformer_engine_xla_status": "heldout_splitmix64_v3_numerics_accepted",
        "r12_result_sha256": result_sha256,
        "r12_policy_raw_sha256": raw_sha256,
        "comparison": "same_h100_host_blocking_reported_without_threshold",
    }


def _evidence_document(cells: list[dict[str, object]], *, r12_result: Path, r12_policy_raw: Path) -> dict[str, object]:
    identities = [cell["transport"]["executable_digest"] for cell in cells]
    return {
        "schema": "shuttle_abi22_rmsnorm_representative_evidence_v2",
        "status": "heldout_splitmix64_v3_shuttle_numerics_accepted",
        "producer": "shuttle_gpu_executable_bundle",
        "shape": list(REPRESENTATIVE_SHAPE),
        "epsilon": 1e-6,
        "input_pattern": ACCEPTANCE_PATTERN,
        "compiler": {
            "compiler_identity": _sha256_bytes(_canonical_json(identities).encode()),
        },
        "external_evidence": _external_evidence(r12_result, r12_policy_raw),
        "protocol": {
            "fresh_process_per_coordinate": True,
            "warmup_iterations": 10,
            "measured_iterations": 20,
            "acceptance_clock": "host_blocking",
            "cuda_event_status": "supplemental_not_acceptance",
        },
        "cells": cells,
        "scope": {
            "norm_contract": "excluded",
            "performance": "reported_without_threshold",
            "raw_outputs_persisted_before_validation": True,
        },
    }


def _prevalidation_document(
    cells: list[dict[str, object]], *, r12_result: Path, r12_policy_raw: Path
) -> dict[str, object]:
    document = _evidence_document(cells, r12_result=r12_result, r12_policy_raw=r12_policy_raw)
    document["schema"] = "shuttle_abi22_rmsnorm_representative_prevalidation_raw_v2"
    document["status"] = "unvalidated_numerical_evidence"
    return document


def _validate_metrics(output: Mapping[str, object], role: str) -> None:
    limits = ROLE_LIMITS[role]
    if output.get("all_finite") is not True:
        raise ValueError("Shuttle output contains a nonfinite value")
    if output.get("absolute_floor") != limits["absolute_floor"]:
        raise ValueError("Shuttle absolute floor drifted")
    if (
        output.get("below_floor_count") != limits["below_floor_count"]
        or output.get("eligible_count") != limits["eligible_count"]
    ):
        raise ValueError("Shuttle metric partition count drifted")
    below = output.get("max_absolute_error_below_floor")
    above = output.get("max_absolute_error_above_floor")
    if type(below) not in (int, float) or type(above) not in (int, float):
        raise ValueError("Shuttle partition maximum type drifted")
    if output.get("max_absolute_error") != max(below, above):
        raise ValueError("Shuttle whole/partition maximum drifted")
    gates = {
        "max_absolute_error_below_floor": limits["max_absolute_error_below_floor"],
        "max_absolute_error_above_floor": limits["max_absolute_error_above_floor"],
        "mean_absolute_error": limits["max_mean_absolute_error"],
        "max_relative_error_above_absolute_floor": limits["max_relative_error_above_floor"],
        "max_bfloat16_ulp_above_absolute_floor": limits["max_bfloat16_ulp_above_floor"],
    }
    for field, limit in gates.items():
        observed = output.get(field)
        if type(observed) not in (int, float) or not math.isfinite(float(observed)) or observed > limit:
            raise ValueError(f"Shuttle {role} {field} exceeds the frozen limit")


def _validate_claims(document: Mapping[str, object], *, r12_result: Path, r12_policy_raw: Path) -> None:
    expected_fields = {
        "schema": "shuttle_abi22_rmsnorm_representative_evidence_v2",
        "status": "heldout_splitmix64_v3_shuttle_numerics_accepted",
        "producer": "shuttle_gpu_executable_bundle",
        "shape": list(REPRESENTATIVE_SHAPE),
        "epsilon": 1e-6,
        "input_pattern": ACCEPTANCE_PATTERN,
        "external_evidence": _external_evidence(r12_result, r12_policy_raw),
        "protocol": {
            "fresh_process_per_coordinate": True,
            "warmup_iterations": 10,
            "measured_iterations": 20,
            "acceptance_clock": "host_blocking",
            "cuda_event_status": "supplemental_not_acceptance",
        },
        "scope": {
            "norm_contract": "excluded",
            "performance": "reported_without_threshold",
            "raw_outputs_persisted_before_validation": True,
        },
    }
    if set(document) != {*expected_fields, "compiler", "cells"}:
        raise ValueError("assembled Shuttle evidence fields drifted")
    for field, expected in expected_fields.items():
        if document.get(field) != expected:
            raise ValueError(f"assembled Shuttle {field} drifted")
    compiler = document.get("compiler")
    if not isinstance(compiler, Mapping) or set(compiler) != {"compiler_identity"}:
        raise ValueError("assembled Shuttle compiler identity fields drifted")
    _digest(compiler.get("compiler_identity"), "compiler identity")
    coordinates = [(policy, boundary, repeat) for policy in POLICIES for boundary in BOUNDARIES for repeat in range(2)]
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != len(coordinates):
        raise ValueError("assembled Shuttle cell matrix drifted")
    repeat_digests = {}
    repeat_transports = {}
    repeat_resources = {}
    for cell, (policy, boundary, repeat) in zip(cells, coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("assembled Shuttle cell is not an object")
        input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
        fixed = {
            "schema": "shuttle_abi22_rmsnorm_representative_cell_v2",
            "producer": "shuttle_gpu_executable_bundle",
            "input_pattern": ACCEPTANCE_PATTERN,
            "shape": list(REPRESENTATIVE_SHAPE),
            "epsilon": 1e-6,
            "policy": policy,
            "boundary": boundary,
            "repeat": repeat,
            "inputs": {role: POLICY_V4_ACCEPTANCE_INPUT_DIGESTS[role] for role in input_roles},
            "call_sequence": _CALL_SEQUENCE[boundary],
            "runner_api": _RUNNER_API,
        }
        for field, expected in fixed.items():
            if cell.get(field) != expected:
                raise ValueError(f"assembled Shuttle cell {field} drifted")
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("assembled Shuttle output vector drifted")
        raw_bindings = []
        for binding in BINDING_CONTRACT[boundary]["operands"]:
            raw_bindings.append({"kind": "operand", **binding, "slot_ordinal": len(raw_bindings)})
        for output, binding in zip(outputs, BINDING_CONTRACT[boundary]["results"], strict=True):
            if not isinstance(output, Mapping):
                raise ValueError("assembled Shuttle output is not an object")
            raw_bindings.append(
                {
                    "kind": "result",
                    **binding,
                    "slot_ordinal": output.get("slot_ordinal"),
                    "artifact": output.get("artifact"),
                }
            )
        resources = _validate_resources(cell.get("resources"), boundary=boundary, bindings=raw_bindings)
        timing = cell.get("timing")
        if not isinstance(timing, Mapping):
            raise ValueError("assembled Shuttle timing drifted")
        if (
            timing.get("warmup_iterations") != 10
            or timing.get("measured_iterations") != 20
            or timing.get("host_clock") != "steady_clock_execute_block_host_until_done_ns"
            or timing.get("cuda_event_status") != "supplemental_not_acceptance"
        ):
            raise ValueError("assembled Shuttle timing contract drifted")
        _positive_samples(timing.get("host_samples_ns"), "host")
        _positive_samples(timing.get("cuda_event_samples_ns"), "CUDA-event")
        transport = cell.get("transport")
        if not isinstance(transport, Mapping) or transport.get("contract_operation_count") != 0:
            raise ValueError("assembled Shuttle transport drifted")
        if type(transport.get("code_bytes")) is not int or transport["code_bytes"] <= 0:
            raise ValueError("assembled Shuttle code byte count drifted")
        _digest(transport.get("executable_digest"), "executable digest")
        _digest(transport.get("module_load_digest"), "module load digest")
        ptx_digests = transport.get("ptx_digests")
        if not isinstance(ptx_digests, list) or len(ptx_digests) != resources["launch_count"]:
            raise ValueError("assembled Shuttle PTX digest vector drifted")
        for digest in ptx_digests:
            _digest(digest, "PTX digest")
        result_slots = {
            output.get("slot_ordinal")
            for output in outputs
            if isinstance(output, Mapping) and type(output.get("slot_ordinal")) is int
        }
        temporary_slots = transport.get("temporary_slots")
        temporary_slot_ordinals = (
            {
                temporary.get("slot_ordinal")
                for temporary in temporary_slots
                if isinstance(temporary_slots, list)
                and isinstance(temporary, Mapping)
                and type(temporary.get("slot_ordinal")) is int
            }
            if isinstance(temporary_slots, list)
            else set()
        )
        operand_slots = set(range(resources["slot_count"])) - result_slots - temporary_slot_ordinals
        derived_temporary_traffic = _validate_task_graph(
            transport.get("task_graph"), temporary_slots, resources, operand_slots, result_slots
        )
        _validate_optimization(transport, resources, derived_temporary_traffic)
        bindings = BINDING_CONTRACT[boundary]["results"]
        if not isinstance(outputs, list) or len(outputs) != len(bindings):
            raise ValueError("assembled Shuttle output vector drifted")
        for output, binding in zip(outputs, bindings, strict=True):
            if not isinstance(output, Mapping):
                raise ValueError("assembled Shuttle output is not an object")
            fixed_output = {
                "role": binding["role"],
                "public_result_index": binding["public_result_index"],
                "runner_result_index": binding["runner_index"],
                "reference_digest": POLICY_V4_ACCEPTANCE_REFERENCE_DIGESTS[binding["role"]],
            }
            for field, expected in fixed_output.items():
                if output.get(field) != expected:
                    raise ValueError(f"assembled Shuttle output {field} drifted")
            slot = output.get("slot_ordinal")
            if type(slot) is not int or slot < 0 or slot >= resources["slot_count"]:
                raise ValueError("assembled Shuttle output slot is invalid")
            digest = _digest(output.get("digest"), "output digest")
            artifact = output.get("artifact")
            if not isinstance(artifact, Mapping) or artifact.get("sha256") != digest:
                raise ValueError("assembled Shuttle output artifact digest drifted")
            _validate_metrics(output, binding["role"])
            repeat_digests[(policy, boundary, binding["role"], repeat)] = digest
        repeat_transports[(policy, boundary, repeat)] = dict(transport)
        repeat_resources[(policy, boundary, repeat)] = dict(resources)
    for policy in POLICIES:
        for boundary in BOUNDARIES:
            if repeat_transports[(policy, boundary, 0)] != repeat_transports[(policy, boundary, 1)]:
                raise ValueError("Shuttle repeat transport identity drifted")
            if repeat_resources[(policy, boundary, 0)] != repeat_resources[(policy, boundary, 1)]:
                raise ValueError("Shuttle repeat resource observation drifted")
            for role in RESULT_ROLES[boundary]:
                if repeat_digests[(policy, boundary, role, 0)] != repeat_digests[(policy, boundary, role, 1)]:
                    raise ValueError("Shuttle acceptance repeat output digest drifted")


def validate_shuttle_evidence(
    value: Mapping[str, object],
    *,
    r12_result: Path,
    r12_policy_raw: Path,
    raw_root: Path,
    case_root: Path,
) -> dict[str, object]:
    """Rebuild and validate evidence from persisted runner cells and BF16 outputs."""
    document = dict(value)
    _validate_claims(document, r12_result=r12_result, r12_policy_raw=r12_policy_raw)
    expected_cells = _load_cells(raw_root, case_root)
    expected = _evidence_document(expected_cells, r12_result=r12_result, r12_policy_raw=r12_policy_raw)
    if document != expected:
        raise ValueError("assembled Shuttle evidence differs from persisted runner evidence")
    return document


def execute_representative(
    *,
    runner: Path,
    gpu_transport_test: Path,
    case_root: Path,
    raw_root: Path,
    reports_root: Path,
    r12_result: Path,
    r12_policy_raw: Path,
    invoke: Callable[[list[str]], int],
) -> dict[str, object]:
    """Execute, persist, and validate all representative Shuttle coordinates."""
    prepare_representative_cases(case_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    plan = execution_plan(
        runner=runner,
        gpu_transport_test=gpu_transport_test,
        case_root=case_root,
        raw_root=raw_root,
    )
    cells = []
    for run in plan["runs"]:
        status = invoke(run["argv"])
        if status != 0:
            coordinate = f"{run['policy']}/{run['boundary']}/repeat{run['repeat']}"
            raise ValueError(f"representative Shuttle runner failed for {coordinate}: {status}")
        raw_path = _raw_cell_path(raw_root, run["policy"], run["boundary"], run["repeat"])
        if not raw_path.is_file():
            raise ValueError("representative Shuttle runner did not persist its raw cell")
        try:
            raw = json.loads(raw_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"raw runner cell is unreadable: {raw_path}") from error
        if not isinstance(raw, Mapping):
            raise ValueError("raw runner cell is not an object")
        cells.append(
            _validate_raw_cell(
                raw,
                policy=run["policy"],
                boundary=run["boundary"],
                repeat=run["repeat"],
                raw_root=raw_root,
                case_root=case_root,
            )
        )
    diagnostics = _repeat_diagnostics(cells)
    raw_path = reports_root / "shuttle-policy-raw-v1.json"
    diagnostics_path = reports_root / "shuttle-repeat-diagnostics-v1.json"
    raw_path.write_text(
        _canonical_json(_prevalidation_document(cells, r12_result=r12_result, r12_policy_raw=r12_policy_raw))
    )
    diagnostics_path.write_text(_canonical_json(diagnostics))
    evidence = _evidence_document(cells, r12_result=r12_result, r12_policy_raw=r12_policy_raw)
    return validate_shuttle_evidence(
        evidence,
        r12_result=r12_result,
        r12_policy_raw=r12_policy_raw,
        raw_root=raw_root,
        case_root=case_root,
    )


def main() -> int:
    """Run the sealed representative matrix and write accepted evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--gpu-transport-test", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--r12-result", type=Path, required=True)
    parser.add_argument("--r12-policy-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    def invoke(argv: list[str]) -> int:
        return subprocess.run(argv, check=False).returncode

    evidence = execute_representative(
        runner=arguments.runner,
        gpu_transport_test=arguments.gpu_transport_test,
        case_root=arguments.case_root,
        raw_root=arguments.raw_root,
        reports_root=arguments.reports_root,
        r12_result=arguments.r12_result,
        r12_policy_raw=arguments.r12_policy_raw,
        invoke=invoke,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(_canonical_json(evidence))
    print(_canonical_json(evidence), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
