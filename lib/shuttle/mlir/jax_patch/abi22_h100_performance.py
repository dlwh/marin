# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate ABI22 H100 timing reports and compare external baselines."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import itertools
import json
import math
import os
import stat
import statistics
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CELL_KEYS = (
    ("rmsnorm_forward", "source_ordered"),
    ("rmsnorm_forward", "fast"),
    ("rmsnorm_backward", "source_ordered"),
    ("rmsnorm_backward", "fast"),
    ("rmsnorm_contract_inline", "source_ordered"),
    ("rmsnorm_contract_inline", "fast"),
    ("rmsnorm_contract_compact", "source_ordered"),
    ("rmsnorm_contract_compact", "fast"),
)

_SHAPES = {
    "rmsnorm_forward": [2, 7, 19],
    "rmsnorm_backward": [2, 7, 19],
    "rmsnorm_contract_inline": [5, 19, 7],
    "rmsnorm_contract_compact": [5, 19, 257],
}

_RESULT_COUNTS = {
    "rmsnorm_forward": 1,
    "rmsnorm_backward": 2,
    "rmsnorm_contract_inline": 1,
    "rmsnorm_contract_compact": 1,
}

_SHUTTLE_RESOURCES = {
    "rmsnorm_forward": (1, 3, 2, 1, 0, 0),
    "rmsnorm_backward": (3, 6, 3, 2, 1, 56),
    "rmsnorm_contract_inline": (1, 4, 3, 1, 0, 0),
    "rmsnorm_contract_compact": (2, 5, 3, 1, 1, 20),
}

_RESOURCE_FIELDS = (
    "launch_count",
    "slot_count",
    "external_operand_count",
    "external_result_count",
    "temporary_allocation_count",
    "temporary_bytes",
)

EXPLORATORY_MODE = "exploratory_host_v1"
HOST_CLOCK = "host_block_until_ready"
TE_UNAVAILABLE_BLOCKER = "te_2_17_runner_lacks_required_shapes_epsilon_and_norm_contract"
EXPLORATORY_MAX_BF16_ULP = 4

_LAYERNORM_FBC_CONTRACT_DIGEST = "ec148234b552b2480b23f63a47f533c81a00c58af7abed3359ecbb427f0de420"
_LAYERNORM_FBC_CONTRACT_ID = "abi22_layernorm_fbc_topology_diagnostic_red_v1"
_LAYERNORM_FBC_REPORT_SCHEMA = "shuttle_abi22_layernorm_fbc_observed_result_v1"
_LAYERNORM_PUBLIC_ROLES = {
    "forward": {"inputs": ("x", "gamma", "beta"), "outputs": ("y",)},
    "backward": {"inputs": ("x", "gamma", "dy"), "outputs": ("dx", "dgamma", "dbeta")},
    "composed": {
        "inputs": ("x", "gamma", "beta", "dy"),
        "outputs": ("y", "dx", "dgamma", "dbeta"),
    },
}
_LAYERNORM_SUBJECTS = ("shuttle", "transformer_engine")
_LAYERNORM_VECTOR_ROLES = frozenset({"gamma", "beta", "dgamma", "dbeta"})
_UINT64_LIMIT = 1 << 64


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0) or value < 0:
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{label} must be a {qualifier}integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def workload_digest(case: str, policy: str, shape: Sequence[int], operands: Sequence[bytes]) -> str:
    """Hash a producer-independent benchmark coordinate and its BF16 inputs."""
    digest = hashlib.sha256()

    def update_integer(value: int) -> None:
        digest.update(struct.pack("<Q", value))

    def update_text(value: str) -> None:
        encoded = value.encode()
        update_integer(len(encoded))
        digest.update(encoded)

    update_text("shuttle-h100-performance-workload-v1")
    update_text(case)
    update_text(policy)
    update_integer(len(shape))
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise ValueError("workload shape must contain positive integers")
        update_integer(extent)
    update_integer(len(operands))
    for operand in operands:
        if not isinstance(operand, bytes) or len(operand) % 2 != 0:
            raise ValueError("workload operands must be serialized BF16 bytes")
        update_integer(len(operand) // 2)
        digest.update(operand)
    return digest.hexdigest()


def _validate_resources(value: object, case: str, producer: str) -> dict[str, Any]:
    resources = _mapping(value, f"{case}.resources")
    code_bytes = _integer(resources.get("code_bytes"), f"{case}.resources.code_bytes", positive=True)
    counters = tuple(_integer(resources.get(field), f"{case}.resources.{field}") for field in _RESOURCE_FIELDS)
    if producer == "shuttle" and counters != _SHUTTLE_RESOURCES[case]:
        raise ValueError(f"{case} Shuttle resource counters do not match the sealed launch graph")
    if counters[0] <= 0:
        raise ValueError(f"{case}.resources.launch_count must be positive")
    return {
        "code_bytes": code_bytes,
        **{field: counter for field, counter in zip(_RESOURCE_FIELDS, counters, strict=True)},
    }


def _validate_cell(
    value: object,
    cell_index: int,
    producer: str,
    measured_iterations: int,
) -> dict[str, Any]:
    cell = _mapping(value, f"cells[{cell_index}]")
    expected_case, expected_policy = CELL_KEYS[cell_index]
    if (cell.get("case"), cell.get("policy")) != (expected_case, expected_policy):
        raise ValueError("report must contain the complete case/policy matrix in canonical order")
    if cell.get("shape") != _SHAPES[expected_case]:
        raise ValueError(f"{expected_case} shape does not match the benchmark protocol")

    samples = _list(cell.get("samples_ns"), f"{expected_case}.samples_ns")
    rounds = _list(cell.get("sample_rounds"), f"{expected_case}.sample_rounds")
    positions = _list(cell.get("sample_positions"), f"{expected_case}.sample_positions")
    if (
        len(samples) != measured_iterations
        or len(rounds) != measured_iterations
        or len(positions) != measured_iterations
    ):
        raise ValueError(f"{expected_case} sample arrays must equal measured_iterations")
    for round_index, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
            raise ValueError(f"{expected_case} samples must be positive integer nanoseconds")
        if rounds[round_index] != round_index:
            raise ValueError(f"{expected_case} sample rounds must be contiguous")
        expected_position = cell_index if round_index % 2 == 0 else len(CELL_KEYS) - 1 - cell_index
        if positions[round_index] != expected_position:
            raise ValueError(f"{expected_case} counterbalanced position is invalid")

    resources = _validate_resources(cell.get("resources"), expected_case, producer)
    output = _mapping(cell.get("output"), f"{expected_case}.output")
    digests = _list(output.get("digests"), f"{expected_case}.output.digests")
    if len(digests) != _RESULT_COUNTS[expected_case]:
        raise ValueError(f"{expected_case} output digest count is invalid")
    for result_index, digest in enumerate(digests):
        _digest(digest, f"{expected_case}.output.digests[{result_index}]")
    max_abs_error = _finite(output.get("max_abs_error"), f"{expected_case}.output.max_abs_error")
    max_ulp = _integer(output.get("max_bf16_ulp"), f"{expected_case}.output.max_bf16_ulp")
    maximum_ulp = 1 if expected_policy == "source_ordered" else 4
    if max_ulp > maximum_ulp:
        raise ValueError(f"{expected_case} {expected_policy} BF16 ULP error exceeds {maximum_ulp}")

    identity = _mapping(cell.get("identity"), f"{expected_case}.identity")
    workload_digest = _digest(identity.get("workload_digest"), f"{expected_case}.identity.workload_digest")
    fixture_digest = _digest(identity.get("fixture_digest"), f"{expected_case}.identity.fixture_digest")
    executable_digest = _digest(identity.get("executable_digest"), f"{expected_case}.identity.executable_digest")
    module_load_digest = _digest(identity.get("module_load_digest"), f"{expected_case}.identity.module_load_digest")
    ptx_digests = _list(identity.get("ptx_digests"), f"{expected_case}.identity.ptx_digests")
    if len(ptx_digests) != resources["launch_count"]:
        raise ValueError(f"{expected_case} requires one PTX digest per launch")
    for launch_index, digest in enumerate(ptx_digests):
        _digest(digest, f"{expected_case}.identity.ptx_digests[{launch_index}]")

    return {
        "case": expected_case,
        "policy": expected_policy,
        "shape": list(_SHAPES[expected_case]),
        "samples_ns": list(samples),
        "sample_rounds": list(rounds),
        "sample_positions": list(positions),
        "resources": resources,
        "output": {
            "digests": list(digests),
            "max_abs_error": max_abs_error,
            "max_bf16_ulp": max_ulp,
        },
        "identity": {
            "workload_digest": workload_digest,
            "fixture_digest": fixture_digest,
            "executable_digest": executable_digest,
            "ptx_digests": list(ptx_digests),
            "module_load_digest": module_load_digest,
        },
    }


def validate_report(report: object, *, expected_producer: str | None = None) -> dict[str, Any]:
    """Validate one complete raw H100 timing report."""
    root = _mapping(report, "report")
    if root.get("schema") != "shuttle_h100_performance_v1":
        raise ValueError("report schema is not shuttle_h100_performance_v1")
    producer = root.get("producer")
    if producer not in {"shuttle", "xla", "te"} or (expected_producer and producer != expected_producer):
        raise ValueError(f"report producer must be {expected_producer or 'shuttle, xla, or te'}")
    device = _mapping(root.get("device"), "device")
    if not isinstance(device.get("model"), str) or "H100" not in device["model"]:
        raise ValueError("device.model must identify an H100")
    ordinal = _integer(device.get("ordinal"), "device.ordinal")
    compiler = _mapping(root.get("compiler"), "compiler")
    if not isinstance(compiler.get("identity"), str) or not compiler["identity"]:
        raise ValueError("compiler.identity must be nonempty")
    options_digest = _digest(compiler.get("options_digest"), "compiler.options_digest")
    protocol = _mapping(root.get("protocol"), "protocol")
    if protocol.get("clock") != "cuda_event":
        raise ValueError("protocol.clock must be cuda_event")
    if protocol.get("counterbalance") != "forward_reverse_alternating":
        raise ValueError("protocol.counterbalance must be forward_reverse_alternating")
    warmups = _integer(protocol.get("warmup_iterations"), "protocol.warmup_iterations", positive=True)
    measured = _integer(protocol.get("measured_iterations"), "protocol.measured_iterations", positive=True)
    if measured < 2 or measured % 2 != 0:
        raise ValueError("protocol.measured_iterations must be positive and even to counterbalance both orders")
    cells = _list(root.get("cells"), "cells")
    if len(cells) != len(CELL_KEYS):
        raise ValueError(f"{producer} report must contain the complete case/policy matrix")
    validated_cells = [_validate_cell(cell, index, producer, measured) for index, cell in enumerate(cells)]
    return {
        "schema": "shuttle_h100_performance_v1",
        "producer": producer,
        "device": {"model": device["model"], "ordinal": ordinal},
        "compiler": {"identity": compiler["identity"], "options_digest": options_digest},
        "protocol": {
            "clock": "cuda_event",
            "counterbalance": "forward_reverse_alternating",
            "warmup_iterations": warmups,
            "measured_iterations": measured,
        },
        "cells": validated_cells,
    }


def _validate_host_protocol(value: object) -> dict[str, Any]:
    protocol = _mapping(value, "protocol")
    if protocol.get("mode") != EXPLORATORY_MODE:
        raise ValueError(f"protocol.mode must be {EXPLORATORY_MODE}")
    if protocol.get("clock") != HOST_CLOCK:
        raise ValueError(f"protocol.clock must be {HOST_CLOCK}")
    if protocol.get("counterbalance") != "forward_reverse_alternating":
        raise ValueError("protocol.counterbalance must be forward_reverse_alternating")
    warmups = _integer(protocol.get("warmup_iterations"), "protocol.warmup_iterations", positive=True)
    measured = _integer(protocol.get("measured_iterations"), "protocol.measured_iterations", positive=True)
    if measured < 2 or measured % 2 != 0:
        raise ValueError("protocol.measured_iterations must be positive and even to counterbalance both orders")
    return {
        "mode": EXPLORATORY_MODE,
        "clock": HOST_CLOCK,
        "counterbalance": "forward_reverse_alternating",
        "warmup_iterations": warmups,
        "measured_iterations": measured,
    }


def _validate_host_samples(
    cell: dict[str, Any],
    cell_index: int,
    measured_iterations: int,
    *,
    field: str,
) -> list[int]:
    case, _ = CELL_KEYS[cell_index]
    samples = _list(cell.get(field), f"{case}.{field}")
    if len(samples) != measured_iterations:
        raise ValueError(f"{case}.{field} must equal measured_iterations")
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
            raise ValueError(f"{case}.{field} must contain positive integer nanoseconds")
    return list(samples)


def _shuttle_exploratory_report(report: object) -> dict[str, Any]:
    root = _mapping(report, "Shuttle report")
    accepted = validate_report(root, expected_producer="shuttle")
    protocol = _validate_host_protocol(root.get("host_protocol"))
    if protocol["warmup_iterations"] != accepted["protocol"]["warmup_iterations"]:
        raise ValueError("Shuttle host and CUDA-event warmup counts must match")
    if protocol["measured_iterations"] != accepted["protocol"]["measured_iterations"]:
        raise ValueError("Shuttle host and CUDA-event measured counts must match")
    raw_cells = _list(root.get("cells"), "cells")
    cells = []
    for index, cell in enumerate(accepted["cells"]):
        raw_cell = _mapping(raw_cells[index], f"cells[{index}]")
        rounds = _list(raw_cell.get("host_sample_rounds"), f"{cell['case']}.host_sample_rounds")
        positions = _list(raw_cell.get("host_sample_positions"), f"{cell['case']}.host_sample_positions")
        if len(rounds) != protocol["measured_iterations"] or len(positions) != protocol["measured_iterations"]:
            raise ValueError(f"{cell['case']} host counterbalance arrays must equal measured_iterations")
        for round_index in range(protocol["measured_iterations"]):
            expected_position = index if round_index % 2 == 0 else len(CELL_KEYS) - 1 - index
            if rounds[round_index] != round_index or positions[round_index] != expected_position:
                raise ValueError(f"{cell['case']} host counterbalanced position is invalid")
        cells.append(
            {
                "case": cell["case"],
                "policy": cell["policy"],
                "shape": cell["shape"],
                "samples_ns": _validate_host_samples(
                    raw_cell,
                    index,
                    protocol["measured_iterations"],
                    field="host_samples_ns",
                ),
                "sample_rounds": list(rounds),
                "sample_positions": list(positions),
                "output": cell["output"],
                "identity": {
                    "workload_digest": cell["identity"]["workload_digest"],
                    "executable_digest": cell["identity"]["executable_digest"],
                },
            }
        )
    return {
        "schema": "shuttle_h100_exploratory_host_v1",
        "producer": "shuttle",
        "device": accepted["device"],
        "compiler": accepted["compiler"],
        "protocol": protocol,
        "cells": cells,
    }


def validate_exploratory_report(report: object, *, expected_producer: str) -> dict[str, Any]:
    """Validate one complete host-blocking exploratory timing report."""
    root = _mapping(report, "exploratory report")
    if root.get("schema") != "shuttle_h100_exploratory_host_v1":
        raise ValueError("exploratory report schema is not shuttle_h100_exploratory_host_v1")
    if expected_producer not in {"shuttle", "xla"} or root.get("producer") != expected_producer:
        raise ValueError(f"exploratory report producer must be {expected_producer}")
    device = _mapping(root.get("device"), "device")
    if not isinstance(device.get("model"), str) or "H100" not in device["model"]:
        raise ValueError("device.model must identify an H100")
    ordinal = _integer(device.get("ordinal"), "device.ordinal")
    compiler = _mapping(root.get("compiler"), "compiler")
    if not isinstance(compiler.get("identity"), str) or not compiler["identity"]:
        raise ValueError("compiler.identity must be nonempty")
    options_digest = _digest(compiler.get("options_digest"), "compiler.options_digest")
    protocol = _validate_host_protocol(root.get("protocol"))
    cells = _list(root.get("cells"), "cells")
    if len(cells) != len(CELL_KEYS):
        raise ValueError(f"{expected_producer} exploratory report must contain the complete case/policy matrix")
    validated_cells = []
    for index, value in enumerate(cells):
        cell = _mapping(value, f"cells[{index}]")
        case, policy = CELL_KEYS[index]
        if (cell.get("case"), cell.get("policy")) != (case, policy):
            raise ValueError("exploratory report must contain the complete case/policy matrix in canonical order")
        if cell.get("shape") != _SHAPES[case]:
            raise ValueError(f"{case} shape does not match the benchmark protocol")
        samples = _validate_host_samples(cell, index, protocol["measured_iterations"], field="samples_ns")
        rounds = _list(cell.get("sample_rounds"), f"{case}.sample_rounds")
        positions = _list(cell.get("sample_positions"), f"{case}.sample_positions")
        if len(rounds) != protocol["measured_iterations"] or len(positions) != protocol["measured_iterations"]:
            raise ValueError(f"{case} counterbalance arrays must equal measured_iterations")
        for round_index in range(protocol["measured_iterations"]):
            expected_position = index if round_index % 2 == 0 else len(CELL_KEYS) - 1 - index
            if rounds[round_index] != round_index or positions[round_index] != expected_position:
                raise ValueError(f"{case} counterbalanced position is invalid")
        output = _mapping(cell.get("output"), f"{case}.output")
        digests = _list(output.get("digests"), f"{case}.output.digests")
        if len(digests) != _RESULT_COUNTS[case]:
            raise ValueError(f"{case} output digest count is invalid")
        for result_index, digest in enumerate(digests):
            _digest(digest, f"{case}.output.digests[{result_index}]")
        maximum_absolute_error = _finite(output.get("max_abs_error"), f"{case}.output.max_abs_error")
        maximum_ulp = _integer(output.get("max_bf16_ulp"), f"{case}.output.max_bf16_ulp")
        if maximum_ulp > EXPLORATORY_MAX_BF16_ULP:
            raise ValueError(
                f"{case} {policy} exceeds the exploratory independent-oracle bound "
                f"of {EXPLORATORY_MAX_BF16_ULP} BF16 ULP"
            )
        identity = _mapping(cell.get("identity"), f"{case}.identity")
        workload = _digest(identity.get("workload_digest"), f"{case}.identity.workload_digest")
        executable = _digest(identity.get("executable_digest"), f"{case}.identity.executable_digest")
        validated_cells.append(
            {
                "case": case,
                "policy": policy,
                "shape": list(_SHAPES[case]),
                "samples_ns": samples,
                "sample_rounds": list(rounds),
                "sample_positions": list(positions),
                "output": {
                    "digests": list(digests),
                    "max_abs_error": maximum_absolute_error,
                    "max_bf16_ulp": maximum_ulp,
                },
                "identity": {"workload_digest": workload, "executable_digest": executable},
            }
        )
    return {
        "schema": "shuttle_h100_exploratory_host_v1",
        "producer": expected_producer,
        "device": {"model": device["model"], "ordinal": ordinal},
        "compiler": {"identity": compiler["identity"], "options_digest": options_digest},
        "protocol": protocol,
        "cells": validated_cells,
    }


def _validate_te_unavailable(value: object) -> dict[str, str]:
    blocker = _mapping(value, "TE exploratory blocker")
    expected = {
        "schema": "shuttle_h100_exploratory_unavailable_v1",
        "producer": "te",
        "protocol_mode": EXPLORATORY_MODE,
        "status": "unavailable",
        "blocker": TE_UNAVAILABLE_BLOCKER,
    }
    if blocker != expected:
        raise ValueError("TE exploratory blocker does not match the audited TE 2.17 limitation")
    return expected


def build_exploratory(shuttle: object, xla: object | None, te_status: object | None) -> dict[str, Any]:
    """Compare matched Shuttle and XLA host-blocking reports."""
    if xla is None:
        raise ValueError("xla exploratory report is required")
    if te_status is None:
        raise ValueError("TE exploratory blocker is required")
    reports = {
        "shuttle": validate_exploratory_report(_shuttle_exploratory_report(shuttle), expected_producer="shuttle"),
        "xla": validate_exploratory_report(xla, expected_producer="xla"),
    }
    te = _validate_te_unavailable(te_status)
    reference = reports["shuttle"]
    candidate = reports["xla"]
    if candidate["device"] != reference["device"]:
        raise ValueError("xla exploratory device does not match Shuttle")
    if candidate["protocol"] != reference["protocol"]:
        raise ValueError("xla exploratory timing protocol does not match Shuttle")
    comparisons = []
    for index, (case, policy) in enumerate(CELL_KEYS):
        shuttle_cell = reference["cells"][index]
        xla_cell = candidate["cells"][index]
        if xla_cell["identity"]["workload_digest"] != shuttle_cell["identity"]["workload_digest"]:
            raise ValueError(f"xla {case} {policy} workload does not match Shuttle")
        shuttle_median = statistics.median(shuttle_cell["samples_ns"])
        xla_median = statistics.median(xla_cell["samples_ns"])
        comparisons.append(
            {
                "case": case,
                "policy": policy,
                "shuttle_median_ns": shuttle_median,
                "xla_median_ns": xla_median,
                "speedup_vs_xla": xla_median / shuttle_median,
            }
        )
    return {
        "schema": "shuttle_h100_exploratory_comparison_v1",
        "device": reference["device"],
        "protocol": reference["protocol"],
        "compiler_identities": {producer: report["compiler"] for producer, report in reports.items()},
        "te": te,
        "comparisons": comparisons,
    }


def build_acceptance(
    shuttle: object,
    xla: object | None,
    te: object | None,
) -> dict[str, Any]:
    """Validate matched Shuttle, XLA, and TE reports and derive timing ratios."""
    if xla is None:
        raise ValueError("xla baseline is required in acceptance mode")
    if te is None:
        raise ValueError("te baseline is required in acceptance mode")
    reports = {
        "shuttle": validate_report(shuttle, expected_producer="shuttle"),
        "xla": validate_report(xla, expected_producer="xla"),
        "te": validate_report(te, expected_producer="te"),
    }
    reference = reports["shuttle"]
    for producer in ("xla", "te"):
        candidate = reports[producer]
        if candidate["device"] != reference["device"]:
            raise ValueError(f"{producer} baseline device does not match Shuttle")
        if candidate["protocol"] != reference["protocol"]:
            raise ValueError(f"{producer} baseline timing protocol does not match Shuttle")
        for index, (case, policy) in enumerate(CELL_KEYS):
            reference_digest = reference["cells"][index]["identity"]["workload_digest"]
            candidate_digest = candidate["cells"][index]["identity"]["workload_digest"]
            if candidate_digest != reference_digest:
                raise ValueError(f"{producer} {case} {policy} workload does not match Shuttle")

    comparisons = []
    for index, (case, policy) in enumerate(CELL_KEYS):
        medians = {
            producer: statistics.median(report["cells"][index]["samples_ns"]) for producer, report in reports.items()
        }
        comparisons.append(
            {
                "case": case,
                "policy": policy,
                "shuttle_median_ns": medians["shuttle"],
                "xla_median_ns": medians["xla"],
                "te_median_ns": medians["te"],
                "speedup_vs_xla": medians["xla"] / medians["shuttle"],
                "speedup_vs_te": medians["te"] / medians["shuttle"],
            }
        )
    return {
        "schema": "shuttle_h100_performance_acceptance_v1",
        "device": reference["device"],
        "protocol": reference["protocol"],
        "compiler_identities": {producer: report["compiler"] for producer, report in reports.items()},
        "comparisons": comparisons,
    }


def _layernorm_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are not the closed schema")


def _layernorm_contract(value: object) -> dict[str, Any]:
    contract = _mapping(value, "LayerNorm F/B/C topology diagnostic contract")
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if hashlib.sha256(canonical).hexdigest() != _LAYERNORM_FBC_CONTRACT_DIGEST:
        raise ValueError("LayerNorm F/B/C topology diagnostic contract does not match the frozen v1 contract")
    if contract.get("contract_id") != _LAYERNORM_FBC_CONTRACT_ID:
        raise ValueError("LayerNorm F/B/C topology diagnostic contract id is invalid")
    return contract


def _layernorm_role_bytes(role: str, shape: tuple[int, int]) -> int:
    rows, features = shape
    elements = features if role in _LAYERNORM_VECTOR_ROLES else rows * features
    return elements * 2


def _layernorm_address(value: object, byte_count: int, label: str) -> tuple[int, int]:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{label} must be a canonical hexadecimal address")
    try:
        address = int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical hexadecimal address") from error
    if address <= 0 or hex(address) != value or byte_count <= 0 or address + byte_count > _UINT64_LIMIT:
        raise ValueError(f"{label} span is invalid or overflows uint64")
    return address, address + byte_count


def _layernorm_require_disjoint(spans: list[tuple[int, int, str]]) -> None:
    ordered = sorted(spans)
    for previous, current in itertools.pairwise(ordered):
        if previous[1] > current[0]:
            raise ValueError(f"{previous[2]} overlaps {current[2]}")


def _layernorm_artifact(value: object, label: str) -> tuple[bytes, tuple[int, int]]:
    artifact = _mapping(value, label)
    _layernorm_exact_keys(artifact, {"path", "byte_count", "sha256"}, label)
    path_value = artifact.get("path")
    if not isinstance(path_value, str):
        raise ValueError(f"{label}.path must be a string")
    path = Path(path_value)
    if not path.is_absolute() or os.path.normpath(path_value) != path_value:
        raise ValueError(f"{label}.path must be a normalized absolute path")
    byte_count = _integer(artifact.get("byte_count"), f"{label}.byte_count", positive=True)
    expected_digest = _digest(artifact.get("sha256"), f"{label}.sha256")
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}.path must name a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            chunks = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(f"{label}.path cannot be read safely") from error
    identity = (opened.st_dev, opened.st_ino)
    if identity != (before.st_dev, before.st_ino) or identity != (after.st_dev, after.st_ino):
        raise ValueError(f"{label}.path changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != byte_count or hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"{label} bytes do not match the declared identity")
    return payload, identity


@functools.cache
def _layernorm_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load dependencies that are optional for the legacy RMSNorm protocol paths."""
    return (
        importlib.import_module("abi22_layernorm_bc_numerical_contract"),
        importlib.import_module("abi22_layernorm_forward_numerical_contract"),
        importlib.import_module("ml_dtypes"),
        importlib.import_module("numpy"),
    )


def _layernorm_fixed_inputs(boundary: str, shape: tuple[int, int]) -> dict[str, Any]:
    backward_numerics, forward_numerics, ml_dtypes, np = _layernorm_dependencies()

    if shape == (7, 13) and boundary == "forward":
        x, gamma, beta = forward_numerics.fixed_inputs(*shape, "balanced")
        return {"x": x, "gamma": gamma, "beta": beta}
    if shape == (7, 13):
        values = backward_numerics.fixed_inputs(*shape)
        return {role: values[role] for role in _LAYERNORM_PUBLIC_ROLES[boundary]["inputs"]}
    if shape != (3, 7):
        raise ValueError("LayerNorm F/B/C coordinate shape is outside the frozen matrix")

    def generated(count: int, multiplier: int, offset: int, modulus: int, center: int) -> Any:
        indices = np.arange(count, dtype=np.int64)
        return (((indices * multiplier + offset) % modulus - center).astype(np.float32) / 8).astype(ml_dtypes.bfloat16)

    rows, features = shape
    values = {
        "x": generated(rows * features, 17, 3, 31, 15).reshape(shape),
        "gamma": generated(features, 5, 1, 13, 6),
        "beta": generated(features, 3, 2, 11, 5),
        "dy": generated(rows * features, 11, 5, 29, 14).reshape(shape),
    }
    return {role: values[role] for role in _LAYERNORM_PUBLIC_ROLES[boundary]["inputs"]}


def _layernorm_references(boundary: str, shape: tuple[int, int]) -> dict[str, Any]:
    backward_numerics, forward_numerics, _, np = _layernorm_dependencies()

    inputs = _layernorm_fixed_inputs(boundary, shape)
    if shape == (3, 7):
        x = inputs["x"].astype(np.float64)
        gamma = inputs["gamma"].astype(np.float64)
        beta = inputs.get("beta", np.zeros_like(inputs["gamma"])).astype(np.float64)
        dy = inputs.get("dy", np.zeros_like(inputs["x"])).astype(np.float64)
        features = shape[1]
        mean = np.sum(x, axis=1, dtype=np.float64) / features
        centered = x - mean[:, None]
        variance = np.sum(centered * centered, axis=1, dtype=np.float64) / features
        inverse = 1.0 / np.sqrt(variance + float(backward_numerics.EPSILON))
        normalized = centered * inverse[:, None]
        if boundary == "forward":
            outputs = {"y": normalized * gamma[None, :] + beta[None, :]}
        else:
            weighted = dy * gamma[None, :]
            outputs = {
                "dx": (
                    inverse[:, None]
                    * (
                        weighted
                        - np.sum(weighted, axis=1, dtype=np.float64)[:, None] / features
                        - normalized * (np.sum(weighted * normalized, axis=1, dtype=np.float64)[:, None] / features)
                    )
                ),
                "dgamma": np.sum(dy * normalized, axis=0, dtype=np.float64),
                "dbeta": np.sum(dy, axis=0, dtype=np.float64),
            }
            if boundary == "composed":
                outputs = {"y": normalized * gamma[None, :] + beta[None, :], **outputs}
        return {role: backward_numerics.round_binary64_to_bf16(value) for role, value in outputs.items()}
    if boundary == "forward":
        return {"y": forward_numerics.binary64_reference(inputs["x"], inputs["gamma"], inputs["beta"])}
    return backward_numerics.binary64_reference(inputs, boundary)


def _layernorm_array_digest(boundary: str, value: Any) -> str:
    backward_numerics, forward_numerics, _, _ = _layernorm_dependencies()

    return forward_numerics.array_digest(value) if boundary == "forward" else backward_numerics.array_digest(value)


def _layernorm_metrics(boundary: str, actual: Any, reference: Any) -> dict[str, float | int]:
    backward_numerics, forward_numerics, _, _ = _layernorm_dependencies()

    measured = (
        forward_numerics.error_metrics(actual, reference)
        if boundary == "forward"
        else backward_numerics.error_metrics(actual, reference)
    )
    return measured.__dict__


def _layernorm_bf16_array(payload: bytes, role: str, shape: tuple[int, int]) -> Any:
    _, _, ml_dtypes, np = _layernorm_dependencies()

    expected_shape = (shape[1],) if role in _LAYERNORM_VECTOR_ROLES else shape
    expected_elements = math.prod(expected_shape)
    if len(payload) != expected_elements * 2:
        raise ValueError(f"LayerNorm {role} raw BF16 byte extent is invalid")
    return np.frombuffer(payload, dtype=np.uint16).copy().view(ml_dtypes.bfloat16).reshape(expected_shape)


def _layernorm_validate_runtime(
    value: object,
    boundary: str,
    shape: tuple[int, int],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[int, int, str]]]:
    runtime = _mapping(value, "LayerNorm runtime")
    _layernorm_exact_keys(
        runtime,
        {
            "subjects",
            "inputs",
            "outputs",
            "private_spans",
            "device_limits",
            "bridge_created_streams",
            "bridge_created_allocations",
            "fallback_targets",
            "host_round_trips",
        },
        "LayerNorm runtime",
    )
    subjects = _list(runtime.get("subjects"), "LayerNorm runtime.subjects")
    if len(subjects) != len(_LAYERNORM_SUBJECTS):
        raise ValueError("LayerNorm runtime must contain both subjects")
    subject_records = {}
    for index, raw_subject in enumerate(subjects):
        subject = _mapping(raw_subject, f"LayerNorm runtime.subjects[{index}]")
        _layernorm_exact_keys(subject, {"subject", "device", "context", "stream"}, "LayerNorm runtime subject")
        name = subject.get("subject")
        if name not in _LAYERNORM_SUBJECTS or name in subject_records:
            raise ValueError("LayerNorm runtime subjects are invalid")
        if any(
            not isinstance(subject.get(field), str) or not subject[field] for field in ("device", "context", "stream")
        ):
            raise ValueError("LayerNorm runtime subject identity fields must be nonempty strings")
        subject_records[name] = subject
    if set(subject_records) != set(_LAYERNORM_SUBJECTS):
        raise ValueError("LayerNorm runtime subjects are incomplete")
    identities = {(record["device"], record["context"], record["stream"]) for record in subject_records.values()}
    if len(identities) != 1:
        raise ValueError("LayerNorm subjects must share the device, context, and stream")

    roles = _LAYERNORM_PUBLIC_ROLES[boundary]
    spans: list[tuple[int, int, str]] = []
    inputs = _list(runtime.get("inputs"), "LayerNorm runtime.inputs")
    if len(inputs) != len(roles["inputs"]):
        raise ValueError("LayerNorm runtime public inputs are incomplete")
    for index, (raw_input, role) in enumerate(zip(inputs, roles["inputs"], strict=True)):
        input_record = _mapping(raw_input, f"LayerNorm runtime.inputs[{index}]")
        _layernorm_exact_keys(
            input_record,
            {"role", "dtype", "byte_count", "shuttle_address", "transformer_engine_address"},
            "LayerNorm runtime input",
        )
        byte_count = _integer(input_record.get("byte_count"), f"LayerNorm runtime {role} byte_count", positive=True)
        if input_record.get("role") != role or input_record.get("dtype") != "bf16":
            raise ValueError("LayerNorm runtime public input role or dtype is invalid")
        if byte_count != _layernorm_role_bytes(role, shape):
            raise ValueError("LayerNorm runtime public input byte extent is invalid")
        shuttle_span = _layernorm_address(input_record.get("shuttle_address"), byte_count, f"LayerNorm {role} input")
        te_span = _layernorm_address(
            input_record.get("transformer_engine_address"), byte_count, f"Transformer Engine {role} input"
        )
        if shuttle_span != te_span:
            raise ValueError("LayerNorm subjects must share public input addresses")
        spans.append((*shuttle_span, f"public input {role}"))

    outputs = _list(runtime.get("outputs"), "LayerNorm runtime.outputs")
    expected_outputs = [(subject, role) for subject in _LAYERNORM_SUBJECTS for role in roles["outputs"]]
    if len(outputs) != len(expected_outputs):
        raise ValueError("LayerNorm runtime public outputs are incomplete")
    for index, (raw_output, (subject, role)) in enumerate(zip(outputs, expected_outputs, strict=True)):
        output = _mapping(raw_output, f"LayerNorm runtime.outputs[{index}]")
        _layernorm_exact_keys(output, {"subject", "role", "dtype", "byte_count", "address"}, "LayerNorm runtime output")
        byte_count = _integer(output.get("byte_count"), f"LayerNorm runtime {role} output bytes", positive=True)
        if output.get("subject") != subject or output.get("role") != role or output.get("dtype") != "bf16":
            raise ValueError("LayerNorm runtime public output identity is invalid")
        if byte_count != _layernorm_role_bytes(role, shape):
            raise ValueError("LayerNorm runtime public output byte extent is invalid")
        span = _layernorm_address(output.get("address"), byte_count, f"LayerNorm {subject} {role} output")
        spans.append((*span, f"{subject} output {role}"))

    private_spans = _list(runtime.get("private_spans"), "LayerNorm runtime.private_spans")
    for index, raw_private in enumerate(private_spans):
        private = _mapping(raw_private, f"LayerNorm runtime.private_spans[{index}]")
        _layernorm_exact_keys(private, {"subject", "owner", "address", "byte_count"}, "LayerNorm private span")
        if private.get("subject") not in _LAYERNORM_SUBJECTS or private.get("owner") != "caller":
            raise ValueError("LayerNorm private spans must be caller-owned by a known subject")
        byte_count = _integer(private.get("byte_count"), "LayerNorm private span byte_count", positive=True)
        span = _layernorm_address(private.get("address"), byte_count, f"LayerNorm private span {index}")
        spans.append((*span, f"private span {index}"))

    limits = _mapping(runtime.get("device_limits"), "LayerNorm runtime.device_limits")
    expected_limits = _mapping(contract["execution"]["h100_device_limits"], "LayerNorm contract device limits")
    if limits != expected_limits:
        raise ValueError("LayerNorm runtime device limits do not match the pinned H100 properties")
    if runtime.get("bridge_created_streams") != 0 or runtime.get("bridge_created_allocations") != 0:
        raise ValueError("LayerNorm provider bridge created a stream or allocation")
    if runtime.get("fallback_targets") != [] or runtime.get("host_round_trips") != 0:
        raise ValueError("LayerNorm execution used a fallback or host round trip")
    return runtime, spans


def _layernorm_validate_topology(
    value: object,
    boundary: str,
    shape: tuple[int, int],
    runtime_spans: list[tuple[int, int, str]],
    device_limits: dict[str, Any],
) -> dict[str, Any]:
    topology = _mapping(value, "LayerNorm observed topology")
    _layernorm_exact_keys(
        topology,
        {"launches", "slots", "public_binding_slots", "scratch_allocations"},
        "LayerNorm observed topology",
    )
    roles = _LAYERNORM_PUBLIC_ROLES[boundary]
    slots = _list(topology.get("slots"), "LayerNorm observed topology.slots")
    if not slots:
        raise ValueError("LayerNorm observed topology must contain slots")
    slot_records: dict[int, dict[str, Any]] = {}
    for index, raw_slot in enumerate(slots):
        slot = _mapping(raw_slot, f"LayerNorm slot[{index}]")
        _layernorm_exact_keys(slot, {"slot", "storage", "byte_count", "role"}, "LayerNorm slot")
        ordinal = _integer(slot.get("slot"), "LayerNorm slot ordinal")
        if ordinal in slot_records:
            raise ValueError("LayerNorm slot ordinals must be unique")
        storage = slot.get("storage")
        role = slot.get("role")
        if storage not in {"input", "output", "scratch"}:
            raise ValueError("LayerNorm slot storage is invalid")
        if storage == "scratch":
            if role is not None:
                raise ValueError("LayerNorm scratch slots cannot have public roles")
        elif not isinstance(role, str):
            raise ValueError("LayerNorm public slots must have roles")
        _integer(slot.get("byte_count"), "LayerNorm slot byte_count", positive=True)
        slot_records[ordinal] = slot

    bindings = _mapping(topology.get("public_binding_slots"), "LayerNorm public binding slots")
    _layernorm_exact_keys(bindings, {"inputs", "outputs"}, "LayerNorm public binding slots")
    bound_slots: set[int] = set()
    for kind, storage in (("inputs", "input"), ("outputs", "output")):
        mapping = _mapping(bindings.get(kind), f"LayerNorm public binding {kind}")
        expected_roles = roles[kind]
        if set(mapping) != set(expected_roles):
            raise ValueError(f"LayerNorm public {kind} binding roles are invalid")
        for role in expected_roles:
            ordinal = _integer(mapping.get(role), f"LayerNorm public {role} slot")
            if ordinal in bound_slots or ordinal not in slot_records:
                raise ValueError("LayerNorm public binding slots must form a bijection")
            slot = slot_records[ordinal]
            if (
                slot["storage"] != storage
                or slot["role"] != role
                or slot["byte_count"] != _layernorm_role_bytes(role, shape)
            ):
                raise ValueError("LayerNorm public binding slot descriptor is invalid")
            bound_slots.add(ordinal)
    public_slot_ordinals = {ordinal for ordinal, slot in slot_records.items() if slot["storage"] in {"input", "output"}}
    if bound_slots != public_slot_ordinals:
        raise ValueError("LayerNorm public slots and binding maps do not have exact coverage")

    launches = _list(topology.get("launches"), "LayerNorm observed topology.launches")
    if not launches:
        raise ValueError("LayerNorm observed topology must contain a launch")
    launch_records: dict[int, dict[str, Any]] = {}
    producer: dict[int, int] = {}
    dependency_graph: dict[int, set[int]] = {}
    max_grid = device_limits["max_grid_dimensions"]
    for index, raw_launch in enumerate(launches):
        launch = _mapping(raw_launch, f"LayerNorm launch[{index}]")
        _layernorm_exact_keys(
            launch,
            {
                "ordinal",
                "dependencies",
                "input_slots",
                "output_slots",
                "grid",
                "block",
                "dynamic_shared_bytes",
                "code_sha256",
                "code_artifact",
            },
            "LayerNorm launch",
        )
        ordinal = _integer(launch.get("ordinal"), "LayerNorm launch ordinal")
        if ordinal in launch_records:
            raise ValueError("LayerNorm launch ordinals must be unique")
        dependencies = _list(launch.get("dependencies"), "LayerNorm launch dependencies")
        dependency_set = {_integer(item, "LayerNorm launch dependency") for item in dependencies}
        if len(dependency_set) != len(dependencies) or ordinal in dependency_set:
            raise ValueError("LayerNorm launch dependencies are invalid")
        input_slots = _list(launch.get("input_slots"), "LayerNorm launch inputs")
        output_slots = _list(launch.get("output_slots"), "LayerNorm launch outputs")
        parsed_inputs = [_integer(item, "LayerNorm launch input slot") for item in input_slots]
        parsed_outputs = [_integer(item, "LayerNorm launch output slot") for item in output_slots]
        if len(set(parsed_inputs)) != len(parsed_inputs) or len(set(parsed_outputs)) != len(parsed_outputs):
            raise ValueError("LayerNorm launch slot lists contain duplicates")
        if any(slot not in slot_records for slot in (*parsed_inputs, *parsed_outputs)):
            raise ValueError("LayerNorm launch references an unknown slot")
        for slot in parsed_outputs:
            if slot_records[slot]["storage"] == "input" or slot in producer:
                raise ValueError("LayerNorm slot has an invalid or duplicate producer")
            producer[slot] = ordinal
        grid = _list(launch.get("grid"), "LayerNorm launch grid")
        block = _list(launch.get("block"), "LayerNorm launch block")
        if len(grid) != 3 or len(block) != 3:
            raise ValueError("LayerNorm launch geometry must be three-dimensional")
        parsed_grid = [_integer(value, "LayerNorm launch grid extent", positive=True) for value in grid]
        parsed_block = [_integer(value, "LayerNorm launch block extent", positive=True) for value in block]
        if any(extent > limit for extent, limit in zip(parsed_grid, max_grid, strict=True)):
            raise ValueError("LayerNorm launch grid exceeds the pinned H100 limit")
        if math.prod(parsed_block) > device_limits["max_threads_per_block"]:
            raise ValueError("LayerNorm launch block exceeds the pinned H100 limit")
        shared = _integer(launch.get("dynamic_shared_bytes"), "LayerNorm dynamic shared bytes")
        if shared > device_limits["max_dynamic_shared_bytes"]:
            raise ValueError("LayerNorm launch shared memory exceeds the pinned H100 limit")
        payload, _ = _layernorm_artifact(launch.get("code_artifact"), f"LayerNorm launch {ordinal} code")
        code_digest = _digest(launch.get("code_sha256"), f"LayerNorm launch {ordinal} code_sha256")
        if hashlib.sha256(payload).hexdigest() != code_digest:
            raise ValueError("LayerNorm launch code digest does not match its artifact bytes")
        launch_records[ordinal] = launch
        dependency_graph[ordinal] = dependency_set

    if any(
        dependency not in launch_records for dependencies in dependency_graph.values() for dependency in dependencies
    ):
        raise ValueError("LayerNorm launch dependency references an unknown launch")

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(ordinal: int) -> None:
        if ordinal in visiting:
            raise ValueError("LayerNorm launch dependency graph is cyclic")
        if ordinal in visited:
            return
        visiting.add(ordinal)
        for dependency in dependency_graph[ordinal]:
            visit(dependency)
        visiting.remove(ordinal)
        visited.add(ordinal)

    for ordinal in launch_records:
        visit(ordinal)
    for ordinal, launch in launch_records.items():
        for slot in launch["input_slots"]:
            if slot_records[slot]["storage"] == "input":
                continue
            if slot not in producer or producer[slot] not in dependency_graph[ordinal]:
                raise ValueError("LayerNorm produced input lacks its direct producer dependency")
    for ordinal, slot in slot_records.items():
        if slot["storage"] in {"scratch", "output"} and ordinal not in producer:
            raise ValueError("LayerNorm scratch or public output slot has no producer")

    allocations = _list(topology.get("scratch_allocations"), "LayerNorm scratch allocations")
    scratch_slots = {ordinal for ordinal, slot in slot_records.items() if slot["storage"] == "scratch"}
    sliced_slots: set[int] = set()
    allocation_ids: set[int] = set()
    all_spans = list(runtime_spans)
    for index, raw_allocation in enumerate(allocations):
        allocation = _mapping(raw_allocation, f"LayerNorm scratch allocation[{index}]")
        _layernorm_exact_keys(
            allocation,
            {"allocation", "subject", "owner", "address", "byte_count", "slices"},
            "LayerNorm scratch allocation",
        )
        allocation_id = _integer(allocation.get("allocation"), "LayerNorm scratch allocation ordinal")
        if allocation_id in allocation_ids:
            raise ValueError("LayerNorm scratch allocation ordinals must be unique")
        allocation_ids.add(allocation_id)
        if allocation.get("subject") != "shuttle" or allocation.get("owner") != "caller":
            raise ValueError("LayerNorm scratch allocations must be caller-owned by Shuttle")
        byte_count = _integer(allocation.get("byte_count"), "LayerNorm scratch allocation bytes", positive=True)
        span = _layernorm_address(allocation.get("address"), byte_count, f"LayerNorm scratch allocation {index}")
        all_spans.append((*span, f"scratch allocation {index}"))
        slices = _list(allocation.get("slices"), f"LayerNorm scratch allocation[{index}].slices")
        parsed_slices = []
        for slice_index, raw_slice in enumerate(slices):
            slice_record = _mapping(raw_slice, f"LayerNorm scratch slice[{index}][{slice_index}]")
            _layernorm_exact_keys(slice_record, {"slot", "offset", "byte_count", "lifetime"}, "LayerNorm scratch slice")
            slot = _integer(slice_record.get("slot"), "LayerNorm scratch slice slot")
            offset = _integer(slice_record.get("offset"), "LayerNorm scratch slice offset")
            slice_bytes = _integer(slice_record.get("byte_count"), "LayerNorm scratch slice bytes", positive=True)
            if slot not in scratch_slots or slot in sliced_slots:
                raise ValueError("LayerNorm scratch slices must cover each scratch slot exactly once")
            if slice_record.get("lifetime") != "whole_report" or slot_records[slot]["byte_count"] != slice_bytes:
                raise ValueError("LayerNorm scratch slice lifetime or byte extent is invalid")
            if offset + slice_bytes > byte_count:
                raise ValueError("LayerNorm scratch slice is out of allocation bounds")
            sliced_slots.add(slot)
            parsed_slices.append((offset, offset + slice_bytes))
        parsed_slices.sort()
        cursor = 0
        for start, end in parsed_slices:
            if start != cursor:
                raise ValueError("LayerNorm scratch slices must exactly and disjointly cover the allocation")
            cursor = end
        if cursor != byte_count:
            raise ValueError("LayerNorm scratch slices leave allocation bytes uncovered")
    if sliced_slots != scratch_slots:
        raise ValueError("LayerNorm scratch slots and allocation slices do not have exact coverage")
    _layernorm_require_disjoint(all_spans)
    return topology


def _layernorm_validate_captures(
    value: object,
    boundary: str,
    shape: tuple[int, int],
    contract: dict[str, Any],
) -> None:
    _, _, _, np = _layernorm_dependencies()

    captures = _list(value, "LayerNorm captures")
    expected_coordinates = [(subject, capture) for subject in _LAYERNORM_SUBJECTS for capture in range(2)]
    if len(captures) != len(expected_coordinates):
        raise ValueError("LayerNorm captures must contain two captures per subject")
    references = _layernorm_references(boundary, shape)
    roles = _LAYERNORM_PUBLIC_ROLES[boundary]["outputs"]
    repeat_digests: dict[tuple[str, str], str] = {}
    artifact_identities: set[tuple[int, int]] = set()
    for index, (raw_capture, (subject, capture_index)) in enumerate(zip(captures, expected_coordinates, strict=True)):
        capture = _mapping(raw_capture, f"LayerNorm captures[{index}]")
        _layernorm_exact_keys(capture, {"subject", "capture_index", "outputs"}, "LayerNorm capture")
        if capture.get("subject") != subject or capture.get("capture_index") != capture_index:
            raise ValueError("LayerNorm capture coordinate is invalid")
        outputs = _list(capture.get("outputs"), f"LayerNorm {subject} capture outputs")
        if len(outputs) != len(roles):
            raise ValueError("LayerNorm capture public outputs are incomplete")
        for output_index, (raw_output, role) in enumerate(zip(outputs, roles, strict=True)):
            output = _mapping(raw_output, f"LayerNorm capture output[{output_index}]")
            _layernorm_exact_keys(
                output,
                {
                    "role",
                    "dtype",
                    "byte_count",
                    "fully_written",
                    "unwritten_sentinel_count",
                    "digest",
                    "binary64_reference_digest",
                    "raw_bf16_artifact",
                    "metrics",
                },
                "LayerNorm capture output",
            )
            byte_count = _integer(output.get("byte_count"), f"LayerNorm captured {role} bytes", positive=True)
            if output.get("role") != role or output.get("dtype") != "bf16":
                raise ValueError("LayerNorm captured public output role or dtype is invalid")
            if byte_count != _layernorm_role_bytes(role, shape):
                raise ValueError("LayerNorm captured public output byte extent is invalid")
            if output.get("fully_written") is not True or output.get("unwritten_sentinel_count") != 0:
                raise ValueError("LayerNorm captured public output is not fully written")
            payload, identity = _layernorm_artifact(
                output.get("raw_bf16_artifact"), f"LayerNorm {subject} capture {capture_index} {role}"
            )
            if identity in artifact_identities:
                raise ValueError("LayerNorm capture artifacts must be independent regular files")
            artifact_identities.add(identity)
            actual = _layernorm_bf16_array(payload, role, shape)
            if not np.isfinite(actual.astype(np.float32)).all():
                raise ValueError("LayerNorm captured public output contains a nonfinite value")
            digest = _digest(output.get("digest"), f"LayerNorm captured {role} digest")
            if digest != _layernorm_array_digest(boundary, actual):
                raise ValueError("LayerNorm captured public output digest does not match raw BF16 bytes")
            reference = references[role]
            reference_digest = _digest(
                output.get("binary64_reference_digest"), f"LayerNorm captured {role} binary64 digest"
            )
            if reference_digest != _layernorm_array_digest(boundary, reference):
                raise ValueError("LayerNorm binary64 reference digest is invalid")
            measured = _layernorm_metrics(boundary, actual, reference)
            declared_metrics = _mapping(output.get("metrics"), f"LayerNorm captured {role} metrics")
            _layernorm_exact_keys(declared_metrics, set(measured), f"LayerNorm captured {role} metrics")
            validated_metrics = {
                "max_absolute_error": _finite(
                    declared_metrics.get("max_absolute_error"), f"LayerNorm captured {role} max absolute error"
                ),
                "mean_absolute_error": _finite(
                    declared_metrics.get("mean_absolute_error"), f"LayerNorm captured {role} mean absolute error"
                ),
                "relative_linf_error": _finite(
                    declared_metrics.get("relative_linf_error"), f"LayerNorm captured {role} relative Linf error"
                ),
                "max_bfloat16_ulp_error": _integer(
                    declared_metrics.get("max_bfloat16_ulp_error"), f"LayerNorm captured {role} BF16 ULP error"
                ),
            }
            if validated_metrics != measured:
                raise ValueError("LayerNorm captured output metrics do not match raw BF16 bytes")
            limits = contract["numerical"]["limits"][boundary][role]
            if any(measured[field] > limit for field, limit in limits.items()):
                raise ValueError("LayerNorm captured output exceeds its independent binary64 limits")
            repeat_key = (subject, role)
            if capture_index == 0:
                repeat_digests[repeat_key] = digest
            elif repeat_digests.get(repeat_key) != digest:
                raise ValueError("LayerNorm public output is not exactly repeatable")


def _layernorm_validate_timing(value: object, contract: dict[str, Any]) -> list[Any]:
    timing = _mapping(value, "LayerNorm timing")
    _layernorm_exact_keys(timing, {"clock", "warmup_blocks", "measured_blocks"}, "LayerNorm timing")
    if timing.get("clock") != contract["timing"]["clock"]:
        raise ValueError("LayerNorm timing clock is invalid")
    first_orders = []
    for measured, field, count in (
        (False, "warmup_blocks", contract["timing"]["warmup_blocks"]),
        (True, "measured_blocks", contract["timing"]["measured_blocks"]),
    ):
        blocks = _list(timing.get(field), f"LayerNorm timing.{field}")
        if len(blocks) != count:
            raise ValueError(f"LayerNorm timing.{field} count is invalid")
        first_order = None
        for block_index, raw_block in enumerate(blocks):
            block = _mapping(raw_block, f"LayerNorm timing.{field}[{block_index}]")
            _layernorm_exact_keys(block, {"block_index", "order", "invocations"}, "LayerNorm timing block")
            if block.get("block_index") != block_index:
                raise ValueError("LayerNorm timing block indices must be contiguous")
            order = block.get("order")
            if order not in {"STTS", "TSST"}:
                raise ValueError("LayerNorm timing block order is invalid")
            if first_order is None:
                first_order = order
            expected_order = first_order if block_index % 2 == 0 else ({"STTS": "TSST", "TSST": "STTS"}[first_order])
            if order != expected_order:
                raise ValueError("LayerNorm timing block order does not alternate")
            expected_subjects = {
                "STTS": ("shuttle", "transformer_engine", "transformer_engine", "shuttle"),
                "TSST": ("transformer_engine", "shuttle", "shuttle", "transformer_engine"),
            }[order]
            invocations = _list(block.get("invocations"), "LayerNorm timing block invocations")
            if len(invocations) != 4:
                raise ValueError("LayerNorm timing blocks must contain four balanced invocations")
            subject_ordinals = {subject: 0 for subject in _LAYERNORM_SUBJECTS}
            for position, (raw_invocation, subject) in enumerate(zip(invocations, expected_subjects, strict=True)):
                invocation = _mapping(raw_invocation, "LayerNorm timing invocation")
                expected_keys = {"position", "subject", "subject_invocation"}
                if measured:
                    expected_keys |= {"duration_nanoseconds", "timer"}
                _layernorm_exact_keys(invocation, expected_keys, "LayerNorm timing invocation")
                if (
                    invocation.get("position") != position
                    or invocation.get("subject") != subject
                    or invocation.get("subject_invocation") != subject_ordinals[subject]
                ):
                    raise ValueError("LayerNorm timing invocation order or ordinal is invalid")
                subject_ordinals[subject] += 1
                if measured:
                    _integer(invocation.get("duration_nanoseconds"), "LayerNorm duration", positive=True)
                    timer = _mapping(invocation.get("timer"), "LayerNorm invocation timer")
                    _layernorm_exact_keys(
                        timer,
                        {"source", "use_delay_kernel", "cuda_event_handle_count", "timer_layer_count"},
                        "LayerNorm invocation timer",
                    )
                    expected_source = (
                        "gpu_runner_execute_timed" if subject == "shuttle" else "stream_executor_event_based_timer"
                    )
                    if (
                        timer.get("source") != expected_source
                        or timer.get("use_delay_kernel") is not False
                        or timer.get("cuda_event_handle_count") != 2
                        or timer.get("timer_layer_count") != 1
                    ):
                        raise ValueError("LayerNorm invocation timer mechanism is invalid")
        first_orders.append(first_order)
    if first_orders[0] != first_orders[1]:
        raise ValueError("LayerNorm warmup and measured blocks must share the same initial order")
    return timing["measured_blocks"]


def validate_layernorm_fbc_observed_result(report: object, contract: object) -> dict[str, Any]:
    """Validate one topology-neutral LayerNorm F/B/C observed timing coordinate."""
    accepted_contract = _layernorm_contract(contract)
    root = _mapping(report, "LayerNorm F/B/C observed result")
    _layernorm_exact_keys(
        root,
        {
            "schema",
            "contract_id",
            "coordinate",
            "runtime",
            "observed_topology",
            "observed_schedule_digest",
            "captures",
            "timing",
        },
        "LayerNorm F/B/C observed result",
    )
    if root.get("schema") != _LAYERNORM_FBC_REPORT_SCHEMA or root.get("contract_id") != _LAYERNORM_FBC_CONTRACT_ID:
        raise ValueError("LayerNorm F/B/C observed result identity is invalid")
    coordinate = _mapping(root.get("coordinate"), "LayerNorm coordinate")
    _layernorm_exact_keys(coordinate, {"boundary", "shape", "shuttle_policy"}, "LayerNorm coordinate")
    boundary = coordinate.get("boundary")
    if boundary not in _LAYERNORM_PUBLIC_ROLES:
        raise ValueError("LayerNorm coordinate boundary is invalid")
    raw_shape = coordinate.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 2:
        raise ValueError("LayerNorm coordinate shape is invalid")
    shape = tuple(_integer(extent, "LayerNorm shape extent", positive=True) for extent in raw_shape)
    matrix = accepted_contract["matrix"]
    if list(shape) not in matrix["shapes"] or coordinate.get("shuttle_policy") != matrix["shuttle_policy"]:
        raise ValueError("LayerNorm coordinate is outside the frozen matrix")
    runtime, runtime_spans = _layernorm_validate_runtime(root.get("runtime"), boundary, shape, accepted_contract)
    topology = _layernorm_validate_topology(
        root.get("observed_topology"), boundary, shape, runtime_spans, runtime["device_limits"]
    )
    schedule_digest = _digest(root.get("observed_schedule_digest"), "LayerNorm observed schedule digest")
    _layernorm_validate_captures(root.get("captures"), boundary, shape, accepted_contract)
    timing_blocks = _layernorm_validate_timing(root.get("timing"), accepted_contract)
    return {
        "verdict": "pass",
        "topology_admission_role": "diagnostic_only",
        "observed_topology": topology,
        "observed_schedule_digest": schedule_digest,
        "timing_blocks": timing_blocks,
    }


def _load(path: Path) -> object:
    if not path.is_file():
        raise ValueError(f"performance report is missing: {path}")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuttle", type=Path, required=True)
    parser.add_argument("--xla", type=Path)
    parser.add_argument("--te", type=Path)
    parser.add_argument("--te-status", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    shuttle = _load(arguments.shuttle)
    if arguments.acceptance and arguments.exploratory:
        parser.error("--acceptance and --exploratory are mutually exclusive")
    if arguments.acceptance:
        result = build_acceptance(
            shuttle,
            _load(arguments.xla) if arguments.xla else None,
            _load(arguments.te) if arguments.te else None,
        )
    elif arguments.exploratory:
        result = build_exploratory(
            shuttle,
            _load(arguments.xla) if arguments.xla else None,
            _load(arguments.te_status) if arguments.te_status else None,
        )
    else:
        result = validate_report(shuttle, expected_producer="shuttle")
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
