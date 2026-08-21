# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate ABI22 H100 timing reports and compare external baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
