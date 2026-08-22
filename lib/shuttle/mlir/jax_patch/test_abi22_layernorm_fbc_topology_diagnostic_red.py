# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import abi22_h100_performance as performance
import abi22_layernorm_bc_numerical_contract as numerics
import abi22_layernorm_forward_numerical_contract as forward_numerics
import ml_dtypes
import numpy as np

CONTRACT_PATH = Path(__file__).with_name("abi22-layernorm-fbc-topology-diagnostic-red-v1.json")
BF16 = "bf16"
PUBLIC_ROLES = {
    "forward": {"inputs": ("x", "gamma", "beta"), "outputs": ("y",)},
    "backward": {"inputs": ("x", "gamma", "dy"), "outputs": ("dx", "dgamma", "dbeta")},
    "composed": {
        "inputs": ("x", "gamma", "beta", "dy"),
        "outputs": ("y", "dx", "dgamma", "dbeta"),
    },
}
PRIMARY_SHAPE = (7, 13)
DISPATCH_SHAPE = (3, 7)


def _role_bytes(role: str, shape: tuple[int, int]) -> int:
    rows, features = shape
    return features * 2 if role in {"gamma", "beta", "dgamma", "dbeta"} else rows * features * 2


def _launch(
    ordinal: int,
    *,
    dependencies: list[int],
    inputs: list[int],
    outputs: list[int],
    grid: int,
    block: int,
    shared: int,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "dependencies": dependencies,
        "input_slots": inputs,
        "output_slots": outputs,
        "grid": [grid, 1, 1],
        "block": [block, 1, 1],
        "dynamic_shared_bytes": shared,
    }


def _slot(slot: int, storage: str, byte_count: int, role: str | None = None) -> dict[str, Any]:
    return {"slot": slot, "storage": storage, "byte_count": byte_count, "role": role}


def _one_launch_topology(boundary: str, shape: tuple[int, int] = PRIMARY_SHAPE) -> dict[str, Any]:
    roles = PUBLIC_ROLES[boundary]
    output_slots = {role: index for index, role in enumerate(roles["outputs"])}
    input_slots = {role: len(output_slots) + index for index, role in enumerate(reversed(roles["inputs"]))}
    slots = [
        *[_slot(slot, "output", _role_bytes(role, shape), role) for role, slot in output_slots.items()],
        *[_slot(slot, "input", _role_bytes(role, shape), role) for role, slot in input_slots.items()],
    ]
    slots.sort(key=lambda value: value["slot"])
    return {
        "launches": [
            _launch(
                0,
                dependencies=[],
                inputs=list(input_slots.values()),
                outputs=list(output_slots.values()),
                grid=4,
                block=64,
                shared=256,
            )
        ],
        "slots": slots,
        "public_binding_slots": {"inputs": input_slots, "outputs": output_slots},
        "scratch_allocations": [],
    }


def _two_launch_topology(boundary: str, shape: tuple[int, int] = PRIMARY_SHAPE) -> dict[str, Any]:
    rows, features = shape
    if boundary == "forward":
        return {
            "launches": [
                _launch(
                    0,
                    dependencies=[],
                    inputs=[0],
                    outputs=[3, 4],
                    grid=rows,
                    block=32,
                    shared=128,
                ),
                _launch(
                    1,
                    dependencies=[0],
                    inputs=[1, 2, 3, 4],
                    outputs=[5],
                    grid=4,
                    block=64,
                    shared=256,
                ),
            ],
            "slots": [
                _slot(0, "input", _role_bytes("x", shape), "x"),
                _slot(1, "input", _role_bytes("gamma", shape), "gamma"),
                _slot(2, "input", _role_bytes("beta", shape), "beta"),
                _slot(3, "scratch", _role_bytes("x", shape)),
                _slot(4, "scratch", rows * 4),
                _slot(5, "output", _role_bytes("y", shape), "y"),
            ],
            "public_binding_slots": {
                "inputs": {"x": 0, "gamma": 1, "beta": 2},
                "outputs": {"y": 5},
            },
            "scratch_allocations": [
                {
                    "allocation": 0,
                    "subject": "shuttle",
                    "owner": "caller",
                    "address": "0x600000",
                    "byte_count": _role_bytes("x", shape),
                    "slices": [
                        {
                            "slot": 3,
                            "offset": 0,
                            "byte_count": _role_bytes("x", shape),
                            "lifetime": "whole_report",
                        }
                    ],
                },
                {
                    "allocation": 1,
                    "subject": "shuttle",
                    "owner": "caller",
                    "address": "0x610000",
                    "byte_count": rows * 4,
                    "slices": [{"slot": 4, "offset": 0, "byte_count": rows * 4, "lifetime": "whole_report"}],
                },
            ],
        }
    if boundary == "backward":
        launch_outputs = ([3, 4, 7], [5, 6])
        launch_inputs = ([0, 1, 2], [0, 2, 3, 4])
        slots = [
            _slot(0, "input", _role_bytes("x", shape), "x"),
            _slot(1, "input", _role_bytes("gamma", shape), "gamma"),
            _slot(2, "input", _role_bytes("dy", shape), "dy"),
            _slot(3, "scratch", rows * 4),
            _slot(4, "scratch", rows * 4),
            _slot(5, "output", _role_bytes("dbeta", shape), "dbeta"),
            _slot(6, "output", _role_bytes("dgamma", shape), "dgamma"),
            _slot(7, "output", _role_bytes("dx", shape), "dx"),
        ]
        public = {"inputs": {"x": 0, "gamma": 1, "dy": 2}, "outputs": {"dx": 7, "dgamma": 6, "dbeta": 5}}
    else:
        launch_outputs = ([4, 5, 6, 9], [7, 8])
        launch_inputs = ([0, 1, 2, 3], [0, 3, 4, 5])
        slots = [
            _slot(0, "input", _role_bytes("x", shape), "x"),
            _slot(1, "input", _role_bytes("gamma", shape), "gamma"),
            _slot(2, "input", _role_bytes("beta", shape), "beta"),
            _slot(3, "input", _role_bytes("dy", shape), "dy"),
            _slot(4, "scratch", rows * 4),
            _slot(5, "scratch", rows * 4),
            _slot(6, "output", _role_bytes("y", shape), "y"),
            _slot(7, "output", _role_bytes("dbeta", shape), "dbeta"),
            _slot(8, "output", _role_bytes("dgamma", shape), "dgamma"),
            _slot(9, "output", _role_bytes("dx", shape), "dx"),
        ]
        public = {
            "inputs": {"x": 0, "gamma": 1, "beta": 2, "dy": 3},
            "outputs": {"y": 6, "dx": 9, "dgamma": 8, "dbeta": 7},
        }
    return {
        "launches": [
            _launch(
                0,
                dependencies=[],
                inputs=list(launch_inputs[0]),
                outputs=list(launch_outputs[0]),
                grid=rows,
                block=32,
                shared=128,
            ),
            _launch(
                1,
                dependencies=[0],
                inputs=list(launch_inputs[1]),
                outputs=list(launch_outputs[1]),
                grid=features,
                block=32,
                shared=128,
            ),
        ],
        "slots": slots,
        "public_binding_slots": public,
        "scratch_allocations": [
            {
                "allocation": 0,
                "subject": "shuttle",
                "owner": "caller",
                "address": "0x600000",
                "byte_count": rows * 4,
                "slices": [
                    {
                        "slot": 3 if boundary == "backward" else 4,
                        "offset": 0,
                        "byte_count": rows * 4,
                        "lifetime": "whole_report",
                    }
                ],
            },
            {
                "allocation": 1,
                "subject": "shuttle",
                "owner": "caller",
                "address": "0x610000",
                "byte_count": rows * 4,
                "slices": [
                    {
                        "slot": 4 if boundary == "backward" else 5,
                        "offset": 0,
                        "byte_count": rows * 4,
                        "lifetime": "whole_report",
                    }
                ],
            },
        ],
    }


def _three_launch_topology(shape: tuple[int, int] = PRIMARY_SHAPE) -> dict[str, Any]:
    rows, _ = shape
    return {
        "launches": [
            _launch(
                0,
                dependencies=[],
                inputs=[2, 4, 6, 7],
                outputs=[0, 1],
                grid=4,
                block=64,
                shared=256,
            ),
            _launch(
                1,
                dependencies=[0],
                inputs=[2, 4, 6, 7, 0, 1],
                outputs=[3, 5],
                grid=2,
                block=128,
                shared=512,
            ),
            _launch(
                2,
                dependencies=[0],
                inputs=[2, 7, 0, 1],
                outputs=[8, 9],
                grid=rows,
                block=32,
                shared=128,
            ),
        ],
        "slots": [
            _slot(0, "scratch", rows * 4),
            _slot(1, "scratch", rows * 4),
            _slot(2, "input", _role_bytes("x", shape), "x"),
            _slot(3, "output", _role_bytes("y", shape), "y"),
            _slot(4, "input", _role_bytes("gamma", shape), "gamma"),
            _slot(5, "output", _role_bytes("dx", shape), "dx"),
            _slot(6, "input", _role_bytes("beta", shape), "beta"),
            _slot(7, "input", _role_bytes("dy", shape), "dy"),
            _slot(8, "output", _role_bytes("dgamma", shape), "dgamma"),
            _slot(9, "output", _role_bytes("dbeta", shape), "dbeta"),
        ],
        "public_binding_slots": {
            "inputs": {"x": 2, "gamma": 4, "beta": 6, "dy": 7},
            "outputs": {"y": 3, "dx": 5, "dgamma": 8, "dbeta": 9},
        },
        "scratch_allocations": [
            {
                "allocation": 0,
                "subject": "shuttle",
                "owner": "caller",
                "address": "0x600000",
                "byte_count": rows * 8,
                "slices": [
                    {"slot": 0, "offset": 0, "byte_count": rows * 4, "lifetime": "whole_report"},
                    {"slot": 1, "offset": rows * 4, "byte_count": rows * 4, "lifetime": "whole_report"},
                ],
            }
        ],
    }


def _invocations(order: str, block_index: int, measured: bool) -> list[dict[str, Any]]:
    subjects = {
        "STTS": ("shuttle", "transformer_engine", "transformer_engine", "shuttle"),
        "TSST": ("transformer_engine", "shuttle", "shuttle", "transformer_engine"),
    }[order]
    subject_ordinals = {"shuttle": 0, "transformer_engine": 0}
    invocations = []
    for position, subject in enumerate(subjects):
        invocation = {
            "position": position,
            "subject": subject,
            "subject_invocation": subject_ordinals[subject],
        }
        subject_ordinals[subject] += 1
        if measured:
            invocation["duration_nanoseconds"] = 100_000 + block_index * 100 + position
            invocation["timer"] = {
                "source": "gpu_runner_execute_timed" if subject == "shuttle" else "stream_executor_event_based_timer",
                "use_delay_kernel": False,
                "cuda_event_handle_count": 2,
                "timer_layer_count": 1,
            }
        invocations.append(invocation)
    return invocations


def _timing(first_order: str = "STTS") -> dict[str, Any]:
    if first_order not in {"STTS", "TSST"}:
        raise ValueError("first order must be STTS or TSST")

    def blocks(count: int, measured: bool) -> list[dict[str, Any]]:
        result = []
        for index in range(count):
            order = first_order if index % 2 == 0 else ({"STTS": "TSST", "TSST": "STTS"}[first_order])
            result.append(
                {
                    "block_index": index,
                    "order": order,
                    "invocations": _invocations(order, index, measured),
                }
            )
        return result

    return {
        "clock": "cuda_event_same_stream",
        "warmup_blocks": blocks(10, False),
        "measured_blocks": blocks(50, True),
    }


def _dispatch_inputs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, features = shape

    def values(count: int, multiplier: int, offset: int, modulus: int, center: int) -> np.ndarray:
        indices = np.arange(count, dtype=np.int64)
        return (((indices * multiplier + offset) % modulus - center).astype(np.float32) / 8).astype(ml_dtypes.bfloat16)

    return {
        "x": values(rows * features, 17, 3, 31, 15).reshape(shape),
        "gamma": values(features, 5, 1, 13, 6),
        "beta": values(features, 3, 2, 11, 5),
        "dy": values(rows * features, 11, 5, 29, 14).reshape(shape),
    }


def _dispatch_binary64_reference(inputs: dict[str, np.ndarray], boundary: str) -> dict[str, np.ndarray]:
    x = inputs["x"].astype(np.float64)
    gamma = inputs["gamma"].astype(np.float64)
    beta = inputs.get("beta", np.zeros_like(inputs["gamma"])).astype(np.float64)
    dy = inputs.get("dy", np.zeros_like(inputs["x"])).astype(np.float64)
    features = x.shape[1]
    mean = np.sum(x, axis=1, dtype=np.float64) / features
    centered = x - mean[:, None]
    variance = np.sum(centered * centered, axis=1, dtype=np.float64) / features
    inverse = 1.0 / np.sqrt(variance + float(numerics.EPSILON))
    normalized = centered * inverse[:, None]
    if boundary == "forward":
        return {"y": numerics.round_binary64_to_bf16(normalized * gamma[None, :] + beta[None, :])}
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
    return {role: numerics.round_binary64_to_bf16(value) for role, value in outputs.items()}


def _references(
    boundary: str,
    shape: tuple[int, int],
    alternate_output: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if shape == PRIMARY_SHAPE and boundary == "forward":
        x, gamma, beta = forward_numerics.fixed_inputs(*shape, "balanced")
        public_inputs = {"x": x, "gamma": gamma, "beta": beta}
        references = {"y": forward_numerics.binary64_reference(x, gamma, beta)}
    elif shape == PRIMARY_SHAPE:
        all_inputs = numerics.fixed_inputs(*shape)
        public_inputs = {role: all_inputs[role] for role in PUBLIC_ROLES[boundary]["inputs"]}
        references = numerics.binary64_reference(public_inputs, boundary)
    else:
        all_inputs = _dispatch_inputs(shape)
        public_inputs = {role: all_inputs[role] for role in PUBLIC_ROLES[boundary]["inputs"]}
        references = _dispatch_binary64_reference(public_inputs, boundary)
    actual = {role: value.copy() for role, value in references.items()}
    if alternate_output:
        role = "y" if boundary == "forward" else "dbeta"
        bits = actual[role].view(np.uint16)
        bits.flat[0] = np.uint16(int(bits.flat[0]) + 1)
        metrics = (
            forward_numerics.error_metrics(actual[role], references[role])
            if boundary == "forward"
            else numerics.error_metrics(actual[role], references[role])
        )
        if boundary == "forward":
            forward_numerics.require_accepted(metrics, "balanced")
        else:
            numerics.require_accepted(metrics, role)
        if shape == PRIMARY_SHAPE:
            if boundary == "forward":
                scheduled_role = forward_numerics.scheduled_reference(
                    public_inputs["x"], public_inputs["gamma"], public_inputs["beta"]
                )
            else:
                scheduled_role = numerics.scheduled_reference(public_inputs, boundary)[role]
            assert _array_digest(boundary, actual[role]) != _array_digest(boundary, scheduled_role)
    return references, actual


def _require_exact_contract_epsilon(contract: dict[str, Any]) -> None:
    declared = contract["numerical"]["epsilon_f32"]
    bits = int(declared["hex_bits"], 16)
    decoded = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
    if decoded != float(declared["exact_decimal"]) or decoded != float(numerics.EPSILON):
        raise AssertionError("contract epsilon must encode the exact public f32 value")


def _array_digest(boundary: str, value: np.ndarray) -> str:
    return forward_numerics.array_digest(value) if boundary == "forward" else numerics.array_digest(value)


def _metrics(boundary: str, actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    measured = (
        forward_numerics.error_metrics(actual, reference)
        if boundary == "forward"
        else numerics.error_metrics(actual, reference)
    )
    return measured.__dict__


def _report(
    boundary: str,
    shape: tuple[int, int],
    topology: dict[str, Any],
    artifact_directory: Path,
    *,
    alternate_output: bool = False,
) -> dict[str, Any]:
    topology = copy.deepcopy(topology)
    observed_schedule_digest = hashlib.sha256(
        json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for launch in topology["launches"]:
        payload = f"sealed observed PTX {boundary} {len(topology['launches'])} {launch['ordinal']}\n".encode()
        artifact = artifact_directory / (
            f"ptx-{boundary}-{shape[0]}x{shape[1]}-{len(topology['launches'])}-{launch['ordinal']}.ptx"
        )
        artifact.write_bytes(payload)
        launch["code_sha256"] = hashlib.sha256(payload).hexdigest()
        launch["code_artifact"] = {
            "path": str(artifact),
            "byte_count": len(payload),
            "sha256": launch["code_sha256"],
        }
    roles = PUBLIC_ROLES[boundary]
    input_addresses = {role: hex(0x100000 + index * 0x10000) for index, role in enumerate(roles["inputs"])}
    references, actual_outputs = _references(boundary, shape, alternate_output)
    outputs = []
    for subject_index, subject in enumerate(("shuttle", "transformer_engine")):
        for role_index, role in enumerate(roles["outputs"]):
            outputs.append(
                {
                    "subject": subject,
                    "role": role,
                    "dtype": BF16,
                    "byte_count": _role_bytes(role, shape),
                    "address": hex(0x200000 + subject_index * 0x100000 + role_index * 0x10000),
                }
            )
    captures = []
    for subject in ("shuttle", "transformer_engine"):
        for capture_index in range(2):
            capture_outputs = []
            for role in roles["outputs"]:
                value = actual_outputs[role]
                reference = references[role]
                payload = value.tobytes(order="C")
                artifact = artifact_directory / (
                    f"{boundary}-{shape[0]}x{shape[1]}-{len(topology['launches'])}-{subject}-{capture_index}-{role}.bf16"
                )
                artifact.write_bytes(payload)
                digest = _array_digest(boundary, value)
                capture_outputs.append(
                    {
                        "role": role,
                        "dtype": BF16,
                        "byte_count": _role_bytes(role, shape),
                        "fully_written": True,
                        "unwritten_sentinel_count": 0,
                        "digest": digest,
                        "binary64_reference_digest": _array_digest(boundary, reference),
                        "raw_bf16_artifact": {
                            "path": str(artifact),
                            "byte_count": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                        "metrics": _metrics(boundary, value, reference),
                    }
                )
            captures.append({"subject": subject, "capture_index": capture_index, "outputs": capture_outputs})
    return {
        "schema": "shuttle_abi22_layernorm_fbc_observed_result_v1",
        "contract_id": "abi22_layernorm_fbc_topology_diagnostic_red_v1",
        "coordinate": {"boundary": boundary, "shape": list(shape), "shuttle_policy": "fast"},
        "runtime": {
            "subjects": [
                {"subject": "shuttle", "device": "H100:0", "context": "context-0", "stream": "stream-0"},
                {
                    "subject": "transformer_engine",
                    "device": "H100:0",
                    "context": "context-0",
                    "stream": "stream-0",
                },
            ],
            "inputs": [
                {
                    "role": role,
                    "dtype": BF16,
                    "byte_count": _role_bytes(role, shape),
                    "shuttle_address": input_addresses[role],
                    "transformer_engine_address": input_addresses[role],
                }
                for role in roles["inputs"]
            ],
            "outputs": outputs,
            "private_spans": [
                {
                    "subject": "transformer_engine",
                    "owner": "caller",
                    "address": "0x500000",
                    "byte_count": 4096,
                }
            ],
            "device_limits": {
                "max_threads_per_block": 1024,
                "max_block_dimensions": [1024, 1024, 64],
                "max_grid_dimensions": [2147483647, 65535, 65535],
                "max_dynamic_shared_bytes": 227328,
            },
            "bridge_created_streams": 0,
            "bridge_created_allocations": 0,
            "fallback_targets": [],
            "host_round_trips": 0,
        },
        "observed_topology": topology,
        "observed_schedule_digest": observed_schedule_digest,
        "captures": captures,
        "timing": _timing(),
    }


def _capture_output(report: dict[str, Any], subject: str, capture_index: int, role: str) -> dict[str, Any]:
    capture = next(
        capture
        for capture in report["captures"]
        if capture["subject"] == subject and capture["capture_index"] == capture_index
    )
    return next(output for output in capture["outputs"] if output["role"] == role)


def _topology_slot(report: dict[str, Any], slot: int) -> dict[str, Any]:
    return next(record for record in report["observed_topology"]["slots"] if record["slot"] == slot)


def _reseal_raw_output(
    output: dict[str, Any],
    *,
    boundary: str,
    actual: np.ndarray,
    reference: np.ndarray,
    declared_metrics: dict[str, float | int] | None = None,
) -> None:
    payload = actual.tobytes(order="C")
    source_artifact = Path(output["raw_bf16_artifact"]["path"])
    artifact = source_artifact.with_name(
        f"{source_artifact.stem}-mutation-{hashlib.sha256(payload).hexdigest()[:12]}{source_artifact.suffix}"
    )
    artifact.write_bytes(payload)
    output["raw_bf16_artifact"] = {
        "path": str(artifact),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    output["digest"] = _array_digest(boundary, actual)
    output["binary64_reference_digest"] = _array_digest(boundary, reference)
    output["metrics"] = declared_metrics if declared_metrics is not None else _metrics(boundary, actual, reference)


def _variants() -> list[tuple[str, tuple[int, int], str, dict[str, Any], bool]]:
    return [
        ("forward", PRIMARY_SHAPE, "fused_one_launch_no_scratch", _one_launch_topology("forward"), False),
        (
            "forward",
            PRIMARY_SHAPE,
            "two_launch_full_and_compact_split_scratch",
            _two_launch_topology("forward"),
            False,
        ),
        ("backward", PRIMARY_SHAPE, "fused_one_launch_no_scratch", _one_launch_topology("backward"), True),
        (
            "backward",
            PRIMARY_SHAPE,
            "current_two_launch_split_scratch",
            _two_launch_topology("backward"),
            False,
        ),
        (
            "composed",
            PRIMARY_SHAPE,
            "current_two_launch_split_scratch",
            _two_launch_topology("composed"),
            False,
        ),
        (
            "composed",
            PRIMARY_SHAPE,
            "split_three_launch_packed_scratch",
            _three_launch_topology(),
            True,
        ),
        (
            "forward",
            DISPATCH_SHAPE,
            "held_out_shape_fused_one_launch",
            _one_launch_topology("forward", DISPATCH_SHAPE),
            False,
        ),
        (
            "backward",
            DISPATCH_SHAPE,
            "held_out_shape_two_launch",
            _two_launch_topology("backward", DISPATCH_SHAPE),
            False,
        ),
        (
            "composed",
            DISPATCH_SHAPE,
            "held_out_shape_three_launch",
            _three_launch_topology(DISPATCH_SHAPE),
            False,
        ),
    ]


class LayerNormFbcTopologyDiagnosticRedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(CONTRACT_PATH.read_text())
        _require_exact_contract_epsilon(self.contract)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.artifact_directory = Path(self.temporary_directory.name)

    def test_accepts_one_two_and_three_launch_telemetry(self) -> None:
        reports = [
            (
                boundary,
                shape,
                name,
                _report(
                    boundary,
                    shape,
                    topology,
                    self.artifact_directory,
                    alternate_output=alternate_output,
                ),
            )
            for boundary, shape, name, topology, alternate_output in _variants()
        ]
        forward_digest_drift = list(copy.deepcopy(reports[0]))
        forward_digest_drift[2] = "same_forward_schedule_with_diagnostic_digest_drift"
        forward_digest_drift[3]["observed_schedule_digest"] = "f" * 64
        reports.append(tuple(forward_digest_drift))
        tsst_first = list(copy.deepcopy(reports[5]))
        tsst_first[2] = "same_composed_topology_with_tsst_first"
        tsst_first[3]["timing"] = _timing("TSST")
        reports.append(tuple(tsst_first))
        validator = performance.validate_layernorm_fbc_observed_result
        for boundary, shape, name, report in reports:
            with self.subTest(boundary=boundary, shape=shape, name=name):
                self.assertFalse(self.contract["numerical"]["schedule_digest_is_admission"])
                accepted = validator(report, self.contract)
                self.assertEqual(accepted["verdict"], "pass")
                self.assertEqual(accepted["topology_admission_role"], "diagnostic_only")
                self.assertEqual(accepted["observed_topology"], report["observed_topology"])
                self.assertEqual(accepted["observed_schedule_digest"], report["observed_schedule_digest"])
                self.assertEqual(accepted["timing_blocks"], report["timing"]["measured_blocks"])
                report["observed_topology"]["launches"][0]["block"][0] = 0
                report["timing"]["measured_blocks"][0]["invocations"][0]["duration_nanoseconds"] = 0
                self.assertNotEqual(accepted["observed_topology"], report["observed_topology"])
                self.assertNotEqual(accepted["timing_blocks"], report["timing"]["measured_blocks"])

    def test_rejects_public_numerical_and_repeatability_failures(self) -> None:
        baseline = _report(
            "composed",
            PRIMARY_SHAPE,
            _one_launch_topology("composed"),
            self.artifact_directory,
        )
        references, alternate = _references("composed", PRIMARY_SHAPE, True)
        validator = performance.validate_layernorm_fbc_observed_result
        mutations = []
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][0]["role"] = "dbeta"
        mutations.append(("runtime duplicate and missing public role", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["inputs"][0]["dtype"] = "f32"
        mutations.append(("public dtype", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["inputs"][0]["byte_count"] -= 2
        mutations.append(("public byte extent", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][0]["dtype"] = "f32"
        mutations.append(("public output dtype", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][0]["byte_count"] -= 2
        mutations.append(("public output byte extent", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][0]["role"] = "dbeta"
        mutations.append(("capture duplicate and missing public role", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][0]["dtype"] = "f32"
        mutations.append(("capture public dtype", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][0]["byte_count"] -= 2
        mutations.append(("capture public byte extent", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][0]["fully_written"] = False
        changed["captures"][0]["outputs"][0]["unwritten_sentinel_count"] = 1
        mutations.append(("full write", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["capture_index"] = False
        mutations.append(("capture index boolean", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][0]["unwritten_sentinel_count"] = 0.0
        mutations.append(("sentinel count float", changed))
        changed = copy.deepcopy(baseline)
        changed["captures"][0]["outputs"][1]["metrics"]["max_absolute_error"] = 1.0
        mutations.append(("binary64 limit", changed))
        changed = copy.deepcopy(baseline)
        gross = np.full_like(references["y"], 256.0)
        _reseal_raw_output(
            _capture_output(changed, "shuttle", 0, "y"),
            boundary="composed",
            actual=gross,
            reference=references["y"],
            declared_metrics={
                "max_absolute_error": 0.0,
                "mean_absolute_error": 0.0,
                "relative_linf_error": 0.0,
                "max_bfloat16_ulp_error": 0,
            },
        )
        mutations.append(("recomputed raw BF16 numerical limits", changed))
        changed = copy.deepcopy(baseline)
        _reseal_raw_output(
            _capture_output(changed, "shuttle", 1, "dbeta"),
            boundary="composed",
            actual=alternate["dbeta"],
            reference=references["dbeta"],
        )
        mutations.append(("repeat raw BF16 output", changed))
        for name, changed in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validator(changed, self.contract)
        contract_mutations = []
        changed_contract = copy.deepcopy(self.contract)
        changed_contract["numerical"]["epsilon_f32"]["hex_bits"] = "0x3727c5ad"
        contract_mutations.append(("epsilon hex bits drift", changed_contract))
        changed_contract = copy.deepcopy(self.contract)
        changed_contract["numerical"]["epsilon_f32"]["exact_decimal"] = "1.0000000656873453e-5"
        contract_mutations.append(("epsilon exact decimal drift", changed_contract))
        changed_contract = copy.deepcopy(self.contract)
        changed_contract["numerical"]["epsilon_f32"] = {
            "hex_bits": "0x3727c5ad",
            "exact_decimal": "1.0000000656873453e-5",
        }
        contract_mutations.append(("epsilon coherent value drift", changed_contract))
        for name, changed_contract in contract_mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validator(baseline, changed_contract)

    def test_rejects_runtime_fairness_fallback_and_graph_failures(self) -> None:
        baseline = _report("composed", PRIMARY_SHAPE, _three_launch_topology(), self.artifact_directory)
        split_scratch_baseline = _report(
            "composed",
            PRIMARY_SHAPE,
            _two_launch_topology("composed"),
            self.artifact_directory,
        )
        validator = performance.validate_layernorm_fbc_observed_result
        mutations = []
        changed = copy.deepcopy(baseline)
        changed["runtime"]["inputs"][0]["transformer_engine_address"] = "0xdeadbeef"
        mutations.append(("shared input", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["inputs"][0]["role"] = "dy"
        mutations.append(("runtime duplicate and missing input role", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["subjects"][1]["device"] = "H100:1"
        mutations.append(("same device", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["subjects"][1]["context"] = "context-1"
        mutations.append(("same context", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["subjects"][1]["stream"] = "stream-1"
        mutations.append(("same stream", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["fallback_targets"] = ["xla_gpu"]
        mutations.append(("fallback", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["host_round_trips"] = 1
        mutations.append(("host round trip", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][1]["address"] = changed["runtime"]["outputs"][0]["address"]
        mutations.append(("output alias", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][1]["address"] = hex(int(changed["runtime"]["outputs"][0]["address"], 16) + 64)
        mutations.append(("partial output overlap", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["private_spans"][0]["address"] = hex(
            int(changed["runtime"]["outputs"][0]["address"], 16) + 64
        )
        mutations.append(("output private partial overlap", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["outputs"][0]["address"] = hex((1 << 64) - 64)
        mutations.append(("output span address overflow", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["bridge_created_streams"] = 1
        mutations.append(("bridge stream", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["bridge_created_allocations"] = 1
        mutations.append(("bridge allocation", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["bridge_created_streams"] = False
        mutations.append(("bridge stream boolean", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["host_round_trips"] = 0.0
        mutations.append(("host round trip float", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["device_limits"]["max_threads_per_block"] = 1024.0
        mutations.append(("device limit float", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["scratch_allocations"][0]["address"] = changed["runtime"]["outputs"][0]["address"]
        mutations.append(("scratch output alias", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["scratch_allocations"][0]["slices"][1]["offset"] = 0
        mutations.append(("overlapping scratch slice offsets", changed))
        changed = copy.deepcopy(baseline)
        allocation = changed["observed_topology"]["scratch_allocations"][0]
        allocation["slices"][1]["offset"] = allocation["byte_count"]
        mutations.append(("out of bounds scratch slice", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["scratch_allocations"][0]["slices"].pop()
        mutations.append(("missing scratch slice", changed))
        changed = copy.deepcopy(baseline)
        slices = changed["observed_topology"]["scratch_allocations"][0]["slices"]
        slices.append(copy.deepcopy(slices[0]))
        mutations.append(("duplicate scratch slice", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["scratch_allocations"][0]["slices"][0]["lifetime"] = "launch_0"
        mutations.append(("scratch slice lifetime", changed))
        changed = copy.deepcopy(baseline)
        allocation = changed["observed_topology"]["scratch_allocations"][0]
        allocation["slices"][0]["byte_count"] -= 1
        allocation["slices"][1]["offset"] -= 1
        allocation["slices"][1]["byte_count"] += 1
        mutations.append(("scratch slice byte count differs from slot", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["scratch_allocations"][0]["byte_count"] += 1
        mutations.append(("scratch allocation has uncovered byte", changed))
        changed = copy.deepcopy(split_scratch_baseline)
        allocations = changed["observed_topology"]["scratch_allocations"]
        allocations[1]["address"] = hex(int(allocations[0]["address"], 16) + 8)
        mutations.append(("distinct scratch allocations partially overlap", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["dependencies"] = [2]
        mutations.append(("cycle", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][1]["input_slots"].append(99)
        mutations.append(("unproduced read", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][1]["dependencies"] = []
        mutations.append(("missing producer dependency", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][2]["output_slots"].append(5)
        mutations.append(("duplicate public writer", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][1]["code_sha256"] = "0" * 64
        mutations.append(("unsealed launch code", changed))
        changed = copy.deepcopy(baseline)
        code_artifact = changed["observed_topology"]["launches"][1]["code_artifact"]
        code_bytes = Path(code_artifact["path"]).read_bytes()
        tampered_code = bytes([code_bytes[0] ^ 1]) + code_bytes[1:]
        tampered_path = self.artifact_directory / "tampered-same-length.ptx"
        tampered_path.write_bytes(tampered_code)
        code_artifact["path"] = str(tampered_path)
        mutations.append(("tampered launch artifact with unchanged hashes", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["block"][0] = 0
        mutations.append(("illegal geometry", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["block"][0] = 1025
        mutations.append(("block exceeds H100 limit", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["block"] = [1, 1, 65]
        mutations.append(("block z dimension exceeds H100 limit", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["grid"][0] = 2147483648
        mutations.append(("grid exceeds H100 limit", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["launches"][0]["dynamic_shared_bytes"] = 227329
        mutations.append(("shared memory exceeds H100 limit", changed))
        changed = copy.deepcopy(baseline)
        changed["runtime"]["device_limits"]["max_threads_per_block"] = 2048
        changed["observed_topology"]["launches"][0]["block"][0] = 1536
        mutations.append(("forged coherent device limit and geometry", changed))
        changed = copy.deepcopy(baseline)
        del changed["observed_topology"]["public_binding_slots"]["outputs"]["dbeta"]
        mutations.append(("missing public binding role", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["public_binding_slots"]["outputs"]["dbeta"] = changed["observed_topology"][
            "public_binding_slots"
        ]["outputs"]["dgamma"]
        mutations.append(("wrong public role to slot", changed))
        changed = copy.deepcopy(baseline)
        del changed["observed_topology"]["public_binding_slots"]["inputs"]["beta"]
        mutations.append(("missing public input binding role", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["public_binding_slots"]["inputs"]["beta"] = changed["observed_topology"][
            "public_binding_slots"
        ]["inputs"]["gamma"]
        mutations.append(("wrong public input role to slot", changed))
        public_slot = baseline["observed_topology"]["public_binding_slots"]["outputs"]["dbeta"]
        changed = copy.deepcopy(baseline)
        _topology_slot(changed, public_slot)["role"] = "dgamma"
        mutations.append(("mapped public slot wrong role", changed))
        changed = copy.deepcopy(baseline)
        _topology_slot(changed, public_slot)["storage"] = "scratch"
        mutations.append(("mapped public slot wrong storage", changed))
        changed = copy.deepcopy(baseline)
        _topology_slot(changed, public_slot)["byte_count"] -= 2
        mutations.append(("mapped public slot wrong byte count", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["slots"].append(copy.deepcopy(_topology_slot(changed, public_slot)))
        mutations.append(("duplicate slot record", changed))
        changed = copy.deepcopy(baseline)
        changed["observed_topology"]["slots"] = [
            record for record in changed["observed_topology"]["slots"] if record["slot"] != public_slot
        ]
        mutations.append(("missing slot record", changed))
        for name, changed in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validator(changed, self.contract)

        race_report = copy.deepcopy(baseline)
        code_artifact = race_report["observed_topology"]["launches"][0]["code_artifact"]
        code_path = Path(code_artifact["path"])
        original_bytes = code_path.read_bytes()
        replacement = bytes([original_bytes[0] ^ 1]) + original_bytes[1:]
        target_identity = (code_path.stat().st_dev, code_path.stat().st_ino)
        original_read = performance.os.read
        rewritten = False

        def rewrite_after_eof(descriptor: int, byte_count: int) -> bytes:
            nonlocal rewritten
            payload = original_read(descriptor, byte_count)
            opened = performance.os.fstat(descriptor)
            if not payload and not rewritten and (opened.st_dev, opened.st_ino) == target_identity:
                code_path.write_bytes(replacement)
                rewritten = True
            return payload

        try:
            with mock.patch.object(performance.os, "read", side_effect=rewrite_after_eof), self.assertRaises(ValueError):
                validator(race_report, self.contract)
        finally:
            code_path.write_bytes(original_bytes)

    def test_rejects_timing_protocol_failures(self) -> None:
        baseline = _report(
            "backward",
            PRIMARY_SHAPE,
            _two_launch_topology("backward"),
            self.artifact_directory,
        )
        validator = performance.validate_layernorm_fbc_observed_result
        mutations = []
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"].pop()
        mutations.append(("warmup count", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][9]["order"] = "STTS"
        mutations.append(("late warmup order", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][9]["invocations"][0]["subject"] = "shuttle"
        mutations.append(("late warmup subject order", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][9]["invocations"][3]["subject_invocation"] = 0
        mutations.append(("late warmup subject invocation ordinal", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][0]["block_index"] = 0.0
        mutations.append(("warmup block index float", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][0]["invocations"][0]["position"] = False
        mutations.append(("warmup position boolean", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["warmup_blocks"][0]["invocations"][0]["subject_invocation"] = 0.0
        mutations.append(("warmup subject ordinal float", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"].pop()
        mutations.append(("measured count", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][1]["order"] = "STTS"
        mutations.append(("balanced order", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][0]["subject"] = "shuttle"
        mutations.append(("subject order", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["subject_invocation"] = 0
        mutations.append(("subject invocation ordinal", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["duration_nanoseconds"] = 0
        mutations.append(("positive duration", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["clock"] = "host_clock"
        mutations.append(("CUDA event clock", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["timer"]["source"] = "host_clock"
        mutations.append(("timer source", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["timer"]["use_delay_kernel"] = True
        mutations.append(("delay kernel", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["timer"]["cuda_event_handle_count"] = 4
        mutations.append(("event handle count", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][0]["invocations"][0]["timer"]["cuda_event_handle_count"] = 2.0
        mutations.append(("event handle count float", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][49]["invocations"][3]["timer"]["timer_layer_count"] = 2
        mutations.append(("timer layer count", changed))
        changed = copy.deepcopy(baseline)
        changed["timing"]["measured_blocks"][0]["invocations"][0]["timer"]["timer_layer_count"] = True
        mutations.append(("timer layer count boolean", changed))
        for name, changed in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validator(changed, self.contract)


if __name__ == "__main__":
    unittest.main()
