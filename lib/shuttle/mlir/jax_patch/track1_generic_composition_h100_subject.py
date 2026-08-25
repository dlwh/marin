# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX H100 subject for anonymous Contract/Fold composition."""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jaxlib import _jax
from track1_generic_composition_numerical_policy import (
    CASES,
    MAX_CELL_REPORT_BYTES,
    POLICY_SHA256,
    SEMANTIC_CONTRACT,
    Case,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    load_numerical_policy,
)

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options


class Mode(StrEnum):
    DISABLED = "disabled"
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True)
class ObserverEvidence:
    invocation_id: int
    generated_bundle_sha256: str
    final_module_sha256: str
    membership_sha256: str
    launch_count: int
    temporary_count: int


@dataclass(frozen=True)
class DeviceEvidence:
    backend: str
    count: int
    kind: str


@dataclass(frozen=True)
class CoordinateResult:
    case: Case
    mode: Mode
    result_sha256: str
    repeat_sha256: str
    stablehlo_sha256: str
    metrics: dict[str, float]
    observer: ObserverEvidence | None


class CoordinateThresholdError(RuntimeError):
    """One coordinate exceeded its unchanged prospective numerical limits."""

    def __init__(self, result: CoordinateResult, exceeded: tuple[str, ...]):
        self.result = result
        self.exceeded = exceeded
        super().__init__(f"{result.case.case_id}/{result.mode.value} exceeds {','.join(exceeded)}")


def tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.PREFER_FUSION,
        maximum_candidates=1,
    )


def compiler_options_for(mode: Mode) -> dict[str, bool | str] | None:
    if mode is Mode.DISABLED:
        return None
    numerics = Numerics.SOURCE_ORDERED if mode is Mode.SOURCE_ORDERED else Numerics.FAST
    return compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=numerics,
        tuning=tuning(),
    )


def composition(
    a: jax.Array,
    b: jax.Array,
    p: jax.Array,
    init: jax.Array,
) -> jax.Array:
    """Evaluate h=Contract0; s=Fold(h); y=Contract1(Map(h,s),p)."""
    h = jnp.matmul(a, b, preferred_element_type=jnp.float32)
    s = jax.lax.reduce(h, init, jax.lax.add, dimensions=(h.ndim - 1,))
    projected = jax.lax.broadcast_in_dim(s, h.shape, tuple(range(s.ndim)))
    z = h * projected
    return jnp.matmul(z, p, preferred_element_type=jnp.float32)


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib lacks the test-only Shuttle observer bridge")
    return bridge


def require_h100() -> DeviceEvidence:
    if jax.default_backend() != "gpu":
        raise RuntimeError("generic composition subject requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("generic composition subject requires one visible H100")
    return DeviceEvidence(backend="gpu", count=1, kind=devices[0].device_kind)


def _sha(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"observer {name} must be a lowercase SHA-256")
    return value


def stablehlo_source(case: Case, inputs: tuple[np.ndarray, ...]) -> tuple[str, str]:
    lowered = jax.jit(composition).lower(*map(jnp.asarray, inputs))
    text = str(lowered.compiler_ir("stablehlo"))
    operations = (
        "stablehlo.dot_general",
        "stablehlo.reduce",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
        "stablehlo.dot_general",
    )
    offset = 0
    for operation in operations:
        position = text.find(operation, offset)
        if position < 0:
            raise RuntimeError("ordinary JAX composition source operation sequence differs")
        offset = position + len(operation)
    if text.count("stablehlo.dot_general") != 2 or text.count("stablehlo.reduce") != 1:
        raise RuntimeError("ordinary JAX composition source closure differs")
    if text.count("stablehlo.multiply") != 1 or text.count("stablehlo.broadcast_in_dim") != 1:
        raise RuntimeError("ordinary JAX dependent Map source closure differs")
    result_type = f"tensor<{'x'.join(map(str, case.result_shape))}xf32>"
    if "%arg3" not in text or result_type not in text:
        raise RuntimeError("ordinary JAX composition public ABI differs")
    return text, hashlib.sha256(text.encode()).hexdigest()


def observer_identity(mode: Mode) -> tuple[str, str]:
    options = compiler_options_for(mode)
    if options is None:
        raise ValueError("disabled compilation has no Shuttle observer identity")
    canonical = options["xla_shuttle_options"]
    if type(canonical) is not str:
        raise TypeError("Shuttle options must use canonical JSON text")
    payload = json.loads(canonical)
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), hashlib.sha256(canonical_tuning.encode()).hexdigest()


def validate_observer_snapshot(
    mode: Mode,
    snapshot: tuple[tuple[object, ...], ...],
) -> ObserverEvidence | None:
    """Validate one unique registered-transform observation."""
    if mode is Mode.DISABLED:
        if snapshot:
            raise ValueError("disabled generic composition emitted observer events")
        return None
    if len(snapshot) != 3 or any(type(event) is not tuple or len(event) != 13 for event in snapshot):
        raise ValueError("enabled generic composition must emit three observer records")
    if tuple(event[1] for event in snapshot) != ("algebra_coverage", "generated_bundle", "final_erasure"):
        raise ValueError("generic composition observer phase order differs")
    invocation = snapshot[0][0]
    if type(invocation) is not int or invocation <= 0 or any(event[0] != invocation for event in snapshot):
        raise ValueError("generic composition observer invocation is not unique and positive")
    policy_sha256, tuning_sha256 = observer_identity(mode)
    if any(event[2:5] != (mode.value, policy_sha256, tuning_sha256) for event in snapshot):
        raise ValueError("generic composition observer identity differs")
    membership = snapshot[0][5]
    manifest = snapshot[0][6]
    if (
        type(membership) is not str
        or membership.count("#shuttle.source_ref<") != 6
        or membership != snapshot[1][5]
        or type(manifest) is not str
        or not manifest
        or manifest != snapshot[1][6]
    ):
        raise ValueError("generic composition source attribution changed before bundle generation")
    if snapshot[0][9] is not False or snapshot[1][9] is not False:
        raise ValueError("generic composition semantics disappeared before final erasure")
    if snapshot[2][5:8] != ("", "", "") or snapshot[2][9] is not True:
        raise ValueError("generic composition final phase did not erase Shuttle semantics")
    if any(event[10] != "" for event in snapshot):
        raise ValueError("generic composition observer carries a failure pass")
    if snapshot[0][11:13] != (0, 0) or snapshot[2][11:13] != (0, 0):
        raise ValueError("generic composition observer emitted resource telemetry outside bundle generation")
    launches, temporaries = snapshot[1][11:13]
    if type(launches) is not int or launches <= 0 or type(temporaries) is not int or temporaries < 0:
        raise ValueError("generic composition resource telemetry is invalid")
    return ObserverEvidence(
        invocation_id=invocation,
        generated_bundle_sha256=_sha(snapshot[1][8], "generated bundle fingerprint"),
        final_module_sha256=_sha(snapshot[2][8], "final module fingerprint"),
        membership_sha256=hashlib.sha256(membership.encode()).hexdigest(),
        launch_count=launches,
        temporary_count=temporaries,
    )


def _execute(
    case: Case,
    mode: Mode,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[object, ...], ...], str]:
    inputs = deterministic_inputs(case)
    _, stablehlo_sha256 = stablehlo_source(case, inputs)
    if stablehlo_sha256 != case.stablehlo_sha256:
        raise RuntimeError("ordinary JAX composition source digest differs from policy")
    options = compiler_options_for(mode)
    compiled = jax.jit(composition) if options is None else jax.jit(composition, compiler_options=options)
    with _bridge().subscribe() as capture:
        actual = np.asarray(compiled(*map(jnp.asarray, inputs)).block_until_ready())
        repeated = np.asarray(compiled(*map(jnp.asarray, inputs)).block_until_ready())
        snapshot = tuple(capture.snapshot_with_bundle_shape())
    return actual, repeated, snapshot, stablehlo_sha256


def _grouped_sha256(value: np.ndarray) -> str:
    digest = array_sha256(value)
    return ":".join(digest[offset : offset + 8] for offset in range(0, len(digest), 8))


def _grouped_digest(digest: str) -> str:
    return ":".join(digest[offset : offset + 8] for offset in range(0, len(digest), 8))


def _ordered_bf16(bits: np.ndarray) -> np.ndarray:
    return np.where(
        bits & np.uint16(0x8000),
        bits ^ np.uint16(0xFFFF),
        bits ^ np.uint16(0x8000),
    ).astype(np.int32)


def _index_document(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    return [int(index) for index in np.unravel_index(flat_index, shape)]


def _scalar_diagnostic(
    actual: np.ndarray,
    reference: np.ndarray,
    reference_f32: np.ndarray,
    actual_bf16: np.ndarray,
    reference_bf16: np.ndarray,
    *,
    flat_index: int,
) -> dict[str, object]:
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    reference_f32_flat = reference_f32.reshape(-1)
    actual_f32_bits = int(actual_flat.view(np.uint32)[flat_index])
    reference_f32_bits = int(reference_f32_flat.view(np.uint32)[flat_index])
    actual_bf16_bits = int(actual_bf16.reshape(-1).view(np.uint16)[flat_index])
    reference_bf16_bits = int(reference_bf16.reshape(-1).view(np.uint16)[flat_index])
    actual_ordered = int(_ordered_bf16(np.asarray([actual_bf16_bits], dtype=np.uint16))[0])
    reference_ordered = int(_ordered_bf16(np.asarray([reference_bf16_bits], dtype=np.uint16))[0])
    return {
        "absolute_error": abs(float(actual_flat[flat_index]) - float(reference_flat[flat_index])),
        "actual_bf16_bits": f"0x{actual_bf16_bits:04x}",
        "actual_f32_bits": f"0x{actual_f32_bits:08x}",
        "actual_value": float(actual_flat[flat_index]),
        "index": _index_document(flat_index, actual.shape),
        "ordered_bf16_ulp_error": abs(actual_ordered - reference_ordered),
        "reference_bf16_bits": f"0x{reference_bf16_bits:04x}",
        "reference_f32_bits": f"0x{reference_f32_bits:08x}",
        "reference_f32_value": float(reference_f32_flat[flat_index]),
        "reference_value": float(reference_flat[flat_index]),
    }


def _observer_document(observer: ObserverEvidence | None) -> dict[str, object]:
    if observer is None:
        return {
            "event_count": 0,
            "final_erasure": False,
            "final_module_sha256": None,
            "generated_bundle_sha256": None,
            "invocation_id": None,
            "membership_sha256": None,
            "status": "disabled",
        }
    return {
        "event_count": 3,
        "final_erasure": True,
        "final_module_sha256": _grouped_digest(observer.final_module_sha256),
        "generated_bundle_sha256": _grouped_digest(observer.generated_bundle_sha256),
        "invocation_id": observer.invocation_id,
        "membership_sha256": _grouped_digest(observer.membership_sha256),
        "status": "selected",
    }


def _coordinate_report(
    result: CoordinateResult,
    actual: np.ndarray,
    repeated: np.ndarray,
    reference: np.ndarray,
    device: DeviceEvidence,
) -> dict[str, object]:
    reference_f32 = reference.astype(np.float32)
    actual_bf16 = actual.astype(ml_dtypes.bfloat16)
    reference_bf16 = reference.astype(ml_dtypes.bfloat16)
    actual_bf16_bits = actual_bf16.view(np.uint16)
    reference_bf16_bits = reference_bf16.view(np.uint16)
    ordered_distance = np.abs(_ordered_bf16(actual_bf16_bits) - _ordered_bf16(reference_bf16_bits))
    absolute = np.abs(actual.astype(np.float64) - reference)
    mismatches = np.flatnonzero(actual.view(np.uint32).reshape(-1) != reference_f32.view(np.uint32).reshape(-1))
    first_mismatch = None
    if mismatches.size:
        first_mismatch = _scalar_diagnostic(
            actual,
            reference,
            reference_f32,
            actual_bf16,
            reference_bf16,
            flat_index=int(mismatches[0]),
        )
    floor = result.case.limits["maximum_absolute_error"]
    mask = np.abs(reference) >= floor
    masked_distance = ordered_distance[mask]
    exceeded = tuple(name for name, limit in result.case.limits.items() if result.metrics[name] > limit)
    h_shape = (*result.case.a_shape[:-1], result.case.b_shape[-1])
    if device.backend != "gpu" or device.count != 1 or "H100" not in device.kind.upper():
        raise RuntimeError("generic composition diagnostic device evidence differs")
    report = {
        "case_id": result.case.case_id,
        "device": {
            "actual_complete": True,
            "backend": device.backend,
            "h100": True,
            "repeat_complete": True,
            "visible_count": device.count,
        },
        "first_mismatch": first_mismatch,
        "full_write": True,
        "metrics": {
            **result.metrics,
            "masked_bf16_absolute_floor": floor,
            "maximum_masked_bf16_ulp_error": int(np.max(masked_distance, initial=0)),
            "maximum_raw_bf16_ulp_error": int(np.max(ordered_distance, initial=0)),
        },
        "mode": result.mode.value,
        "numerical_policy_sha256": _grouped_digest(POLICY_SHA256),
        "observer": _observer_document(result.observer),
        "output_digests": {
            "actual": _grouped_sha256(actual),
            "reference_binary64": _grouped_sha256(reference),
            "reference_f32": _grouped_sha256(reference_f32),
            "repeat": _grouped_sha256(repeated),
        },
        "private_boundary": [
            {
                "consumer": "map_contract",
                "dtype": "f32",
                "name": "h",
                "producer": "contract_fold",
                "shape": list(h_shape),
            },
            {
                "consumer": "map_contract",
                "dtype": "f32",
                "name": "s",
                "producer": "contract_fold",
                "shape": list(h_shape[:-1]),
            },
        ],
        "repeat_bitwise_equal": True,
        "schema": "track1_generic_composition_h100_cell",
        "schema_version": 1,
        "semantic_groups": SEMANTIC_CONTRACT["groups"],
        "shapes": {
            "a": list(result.case.a_shape),
            "b": list(result.case.b_shape),
            "init": [],
            "p": list(result.case.p_shape),
            "y": list(result.case.result_shape),
        },
        "stablehlo_sha256": _grouped_digest(result.stablehlo_sha256),
        "threshold_exceeded": list(exceeded),
        "thresholds": result.case.limits,
        "worst_absolute": _scalar_diagnostic(
            actual,
            reference,
            reference_f32,
            actual_bf16,
            reference_bf16,
            flat_index=int(np.argmax(absolute)),
        ),
        "worst_bf16_ulp": _scalar_diagnostic(
            actual,
            reference,
            reference_f32,
            actual_bf16,
            reference_bf16,
            flat_index=int(np.argmax(ordered_distance)),
        ),
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) >= MAX_CELL_REPORT_BYTES:
        raise RuntimeError("generic composition cell report exceeds its bounded diagnostic size")
    return report


def validate_coordinate(
    case: Case,
    mode: Mode,
    actual: np.ndarray,
    repeated: np.ndarray,
    snapshot: tuple[tuple[object, ...], ...],
    stablehlo_sha256: str,
    *,
    device: DeviceEvidence | None = None,
    report_cell: Callable[[dict[str, object]], None] | None = None,
) -> CoordinateResult:
    """Validate one complete public f32 output against the independent oracle."""
    observer = validate_observer_snapshot(mode, snapshot)
    for value in (actual, repeated):
        if value.dtype != np.dtype(np.float32) or value.shape != case.result_shape:
            raise RuntimeError("generic composition public result ABI differs")
        if not np.all(np.isfinite(value)):
            raise RuntimeError("generic composition public result is not finite")
    if actual.tobytes(order="C") != repeated.tobytes(order="C"):
        raise RuntimeError("generic composition repeat result drift")
    oracle = binary64_oracle(*deterministic_inputs(case))
    absolute = np.abs(actual.astype(np.float64) - oracle)
    maximum = float(np.max(absolute))
    mean = float(np.mean(absolute))
    denominator = max(float(np.max(np.abs(oracle))), np.finfo(np.float64).tiny)
    relative = maximum / denominator
    metrics = {
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "relative_linf_error": relative,
    }
    if np.any(oracle.astype(np.float32) == 0):
        raise RuntimeError("generic composition full-write oracle contains zero")
    result = CoordinateResult(
        case=case,
        mode=mode,
        result_sha256=array_sha256(actual),
        repeat_sha256=array_sha256(repeated),
        stablehlo_sha256=stablehlo_sha256,
        metrics=metrics,
        observer=observer,
    )
    if report_cell is not None:
        if device is None:
            raise RuntimeError("generic composition diagnostic requires device evidence")
        report_cell(_coordinate_report(result, actual, repeated, oracle, device))
    exceeded = tuple(name for name, limit in case.limits.items() if metrics[name] > limit)
    if exceeded:
        raise CoordinateThresholdError(result, exceeded)
    return result


def run_matrix(
    report_cell: Callable[[dict[str, object]], None] | None = None,
) -> tuple[CoordinateResult, ...]:
    """Execute all six ordinary-JAX coordinates on one H100."""
    device = require_h100()
    load_numerical_policy()
    results = []
    invocation_ids = set()
    threshold_failures: list[str] = []
    for case in CASES:
        for mode in Mode:
            actual, repeated, snapshot, stablehlo_sha256 = _execute(case, mode)
            try:
                result = validate_coordinate(
                    case,
                    mode,
                    actual,
                    repeated,
                    snapshot,
                    stablehlo_sha256,
                    device=device,
                    report_cell=report_cell,
                )
            except CoordinateThresholdError as error:
                result = error.result
                threshold_failures.append(f"{case.case_id}/{mode.value}:{','.join(error.exceeded)}")
            if result.observer is not None:
                if result.observer.invocation_id in invocation_ids:
                    raise RuntimeError("generic composition observer invocation was reused")
                invocation_ids.add(result.observer.invocation_id)
            results.append(result)
    if threshold_failures:
        raise RuntimeError("generic composition campaign exceeded unchanged thresholds: " + ";".join(threshold_failures))
    return tuple(results)


def result_document(results: tuple[CoordinateResult, ...]) -> dict[str, object]:
    """Return the closed canonical H100 result document."""
    return {
        "coordinates": [
            {
                "case_id": result.case.case_id,
                "full_write": True,
                "metrics": result.metrics,
                "mode": result.mode.value,
                "observer": (
                    None
                    if result.observer is None
                    else {
                        "final_module_sha256": result.observer.final_module_sha256,
                        "generated_bundle_sha256": result.observer.generated_bundle_sha256,
                        "invocation_id": result.observer.invocation_id,
                        "launch_count_telemetry": result.observer.launch_count,
                        "membership_sha256": result.observer.membership_sha256,
                        "temporary_count_telemetry": result.observer.temporary_count,
                    }
                ),
                "repeat_sha256": result.repeat_sha256,
                "result_sha256": result.result_sha256,
                "stablehlo_sha256": result.stablehlo_sha256,
            }
            for result in results
        ],
        "numerical_policy_sha256": POLICY_SHA256,
        "pid": os.getpid(),
        "schema": "track1_generic_composition_h100_result",
        "schema_version": 1,
        "semantic_contract": SEMANTIC_CONTRACT,
    }
