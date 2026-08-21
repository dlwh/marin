# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX H100 subject for the ABI22 RMSNorm-to-Contract path."""

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
from track1_rmsnorm_contract_numerical_policy import (
    LIMITS,
    MASKED_ULP_ABSOLUTE_FLOORS,
    POLICY_SHA256,
    SHAPES,
    NumericalMetrics,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    emitted_primal_oracle,
    load_numerical_policy,
    numerical_metrics,
)

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options

MAX_CELL_REPORT_BYTES = 4096


class Boundary(StrEnum):
    PRIMAL = "primal"
    COMPOSED = "composed"


class Mode(StrEnum):
    DISABLED = "disabled"
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True, order=True)
class Coordinate:
    shape: tuple[int, int, int]
    boundary: Boundary
    mode: Mode

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError("RMSNorm Contract shape is outside the canary matrix")

    @classmethod
    def all(cls) -> tuple["Coordinate", ...]:
        return tuple(cls(shape, boundary, mode) for boundary in Boundary for shape in SHAPES for mode in Mode)


@dataclass(frozen=True)
class ObserverIdentity:
    policy_sha256: str
    tuning_sha256: str


@dataclass(frozen=True)
class ObserverEvidence:
    invocation_id: int
    generated_bundle_sha256: str
    final_module_sha256: str
    launch_count: int
    temporary_count: int
    topology_status: str


@dataclass(frozen=True)
class CoordinateResult:
    coordinate: Coordinate
    results: dict[str, str]
    repeat_results: dict[str, str]
    disabled_jax_results: dict[str, str]
    disabled_jax_equal: dict[str, bool]
    analytic_oracle: dict[str, str]
    numerics: dict[str, dict[str, float | int]]
    observer: ObserverEvidence | None


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


def observer_identity(coordinate: Coordinate) -> ObserverIdentity:
    options = compiler_options_for(coordinate.mode)
    if options is None:
        raise ValueError("disabled compilation has no Shuttle observer identity")
    canonical = options["xla_shuttle_options"]
    if type(canonical) is not str:
        raise TypeError("Shuttle options must use canonical JSON text")
    payload = json.loads(canonical)
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        hashlib.sha256(canonical.encode()).hexdigest(),
        hashlib.sha256(canonical_tuning.encode()).hexdigest(),
    )


def rmsnorm_contract(x: jax.Array, weight: jax.Array, projection: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    mean_square = jnp.mean(local * local, axis=-1, keepdims=True)
    inverse_root = jax.lax.rsqrt(mean_square + jnp.float32(1e-6))
    normalized = (local * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    result = jnp.matmul(normalized, projection, preferred_element_type=jnp.float32)
    return result.astype(jnp.bfloat16)


def result_roles(boundary: Boundary) -> tuple[str, ...]:
    if boundary is Boundary.PRIMAL:
        return ("result",)
    return ("result", "input_cotangent", "weight_cotangent", "projection_cotangent")


def boundary_function(
    boundary: Boundary,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    def subject(
        x: jax.Array,
        weight: jax.Array,
        projection: jax.Array,
        cotangent: jax.Array,
    ) -> tuple[jax.Array, ...]:
        result, pullback = jax.vjp(rmsnorm_contract, x, weight, projection)
        if boundary is Boundary.PRIMAL:
            return (result,)
        return (result, *pullback(cotangent))

    return subject


def _sha(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"observer {name} must be a lowercase SHA-256")
    return value


def validate_observer_snapshot(
    coordinate: Coordinate,
    snapshot: tuple[tuple[object, ...], ...],
) -> ObserverEvidence | None:
    if coordinate.mode is Mode.DISABLED:
        if snapshot:
            raise ValueError("disabled RMSNorm Contract compilation emitted observer events")
        return None
    if len(snapshot) != 3 or any(type(event) is not tuple or len(event) != 13 for event in snapshot):
        raise ValueError("enabled RMSNorm Contract compilation must emit three observer records")
    if tuple(event[1] for event in snapshot) != ("algebra_coverage", "generated_bundle", "final_erasure"):
        raise ValueError("enabled RMSNorm Contract compilation emitted the wrong observer phases")
    invocation_ids = {event[0] for event in snapshot}
    if len(invocation_ids) != 1 or type(snapshot[0][0]) is not int or snapshot[0][0] <= 0:
        raise ValueError("RMSNorm Contract observer phases do not share one positive invocation")
    identity = observer_identity(coordinate)
    for event in snapshot:
        if event[2:5] != (coordinate.mode.value, identity.policy_sha256, identity.tuning_sha256):
            raise ValueError("RMSNorm Contract observer identity differs from compiler options")
        if event[10] != "":
            raise ValueError("successful RMSNorm Contract observer event carries a failure pass")
    membership = snapshot[0][5]
    manifest = snapshot[0][6]
    if (
        type(membership) is not str
        or "#shuttle.source_ref<" not in membership
        or membership != snapshot[1][5]
        or type(manifest) is not str
        or not manifest
        or manifest != snapshot[1][6]
    ):
        raise ValueError("RMSNorm Contract selection attribution changed before bundle generation")
    if snapshot[0][9] is not False or snapshot[1][9] is not False:
        raise ValueError("RMSNorm Contract semantics disappeared before final erasure")
    if snapshot[2][5:8] != ("", "", "") or snapshot[2][9] is not True:
        raise ValueError("RMSNorm Contract final phase did not erase Shuttle semantics")
    if snapshot[0][11:13] != (0, 0) or snapshot[2][11:13] != (0, 0):
        raise ValueError("RMSNorm Contract observer reported topology outside bundle generation")
    launches, temporaries = snapshot[1][11:13]
    if type(launches) is not int or type(temporaries) is not int:
        raise ValueError("RMSNorm Contract observer topology must use integers")
    topology_status = "diagnostic_non_performant"
    if coordinate.boundary is Boundary.PRIMAL:
        expected = (1, 0) if coordinate.shape[-1] == 7 else (2, 1)
        if (launches, temporaries) != expected:
            raise ValueError("RMSNorm Contract primal bundle topology differs")
        topology_status = "normative_primal"
    elif launches <= 0 or temporaries <= 0 or launches > 128 or temporaries > 128:
        raise ValueError("RMSNorm Contract composed bundle lacks bounded positive topology")
    return ObserverEvidence(
        invocation_id=snapshot[0][0],
        generated_bundle_sha256=_sha(snapshot[1][8], "generated bundle fingerprint"),
        final_module_sha256=_sha(snapshot[2][8], "final module fingerprint"),
        launch_count=launches,
        temporary_count=temporaries,
        topology_status=topology_status,
    )


def _metrics_document(metrics: NumericalMetrics) -> dict[str, float | int]:
    return {
        "maximum_absolute_error": metrics.maximum_absolute_error,
        "maximum_bf16_ulp_error": metrics.maximum_bf16_ulp_error,
        "maximum_masked_bf16_ulp_error": metrics.maximum_masked_bf16_ulp_error,
        "mean_absolute_error": metrics.mean_absolute_error,
        "relative_linf_error": metrics.relative_linf_error,
    }


def _grouped_sha256(value: np.ndarray) -> str:
    digest = array_sha256(value)
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
    analytic: np.ndarray,
    rounded: np.ndarray,
    *,
    flat_index: int,
) -> dict[str, float | int | str | list[int]]:
    actual_flat = actual.reshape(-1)
    analytic_flat = analytic.reshape(-1)
    rounded_flat = rounded.reshape(-1)
    actual_bits = int(actual_flat.view(np.uint16)[flat_index])
    rounded_bits = int(rounded_flat.view(np.uint16)[flat_index])
    actual_value = float(actual_flat[flat_index])
    analytic_value = float(analytic_flat[flat_index])
    rounded_value = float(rounded_flat[flat_index])
    actual_ordered = int(_ordered_bf16(np.asarray([actual_bits], dtype=np.uint16))[0])
    rounded_ordered = int(_ordered_bf16(np.asarray([rounded_bits], dtype=np.uint16))[0])
    return {
        "absolute_error": abs(actual_value - analytic_value),
        "actual_bits": f"0x{actual_bits:04x}",
        "actual_value": actual_value,
        "analytic_rounded_bits": f"0x{rounded_bits:04x}",
        "analytic_rounded_value": rounded_value,
        "analytic_value": analytic_value,
        "index": _index_document(flat_index, actual.shape),
        "masked_bf16_code_distance": abs((actual_bits & 0x7FFF) - (rounded_bits & 0x7FFF)),
        "ordered_bf16_ulp_error": abs(actual_ordered - rounded_ordered),
        "raw_bf16_code_distance": abs(actual_bits - rounded_bits),
    }


def _cell_report(
    coordinate: Coordinate,
    role: str,
    actual: np.ndarray,
    repeated: np.ndarray,
    diagnostic: np.ndarray,
    analytic: np.ndarray,
    metrics: NumericalMetrics,
) -> dict[str, object]:
    rounded = analytic.astype(ml_dtypes.bfloat16)
    actual_bits = actual.view(np.uint16)
    rounded_bits = rounded.view(np.uint16)
    raw_distance = np.abs(actual_bits.astype(np.int32) - rounded_bits.astype(np.int32))
    magnitude_distance = np.abs(
        (actual_bits & np.uint16(0x7FFF)).astype(np.int32) - (rounded_bits & np.uint16(0x7FFF)).astype(np.int32)
    )
    ordered_distance = np.abs(_ordered_bf16(actual_bits) - _ordered_bf16(rounded_bits))
    absolute = np.abs(actual.astype(np.float64) - analytic.astype(np.float64))
    mismatches = np.flatnonzero(actual_bits.reshape(-1) != rounded_bits.reshape(-1))
    first_mismatch = None
    if mismatches.size:
        first_mismatch = _scalar_diagnostic(
            actual,
            analytic,
            rounded,
            flat_index=int(mismatches[0]),
        )
    observed = _metrics_document(metrics)
    report = {
        "boundary": coordinate.boundary.value,
        "first_mismatch": first_mismatch,
        "max_bf16_masked_code_distance": int(np.max(magnitude_distance, initial=0)),
        "maximum_raw_bf16_code_distance": int(np.max(raw_distance, initial=0)),
        "metrics": observed,
        "mode": coordinate.mode.value,
        "output_digests": {
            "actual": _grouped_sha256(actual),
            "analytic_rounded": _grouped_sha256(rounded),
            "disabled_jax": _grouped_sha256(diagnostic),
            "repeat": _grouped_sha256(repeated),
        },
        "role": role,
        "schema": "rmsnorm_contract_cell_v2",
        "schema_version": 2,
        "shape": list(coordinate.shape),
        "sign_crossing_count": int(np.count_nonzero(((actual_bits ^ rounded_bits) & np.uint16(0x8000)) != np.uint16(0))),
        "threshold_exceeded": sorted(name for name, limit in LIMITS[role].items() if observed[name] > limit),
        "thresholds": LIMITS[role],
        "worst_absolute": _scalar_diagnostic(
            actual,
            analytic,
            rounded,
            flat_index=int(np.argmax(absolute)),
        ),
        "worst_bf16_ulp": _scalar_diagnostic(
            actual,
            analytic,
            rounded,
            flat_index=int(np.argmax(ordered_distance)),
        ),
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) >= MAX_CELL_REPORT_BYTES:
        raise RuntimeError("RMSNorm Contract cell report exceeds its bounded diagnostic size")
    return report


def _array_bytes_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C")
    )


def _expected_shape(shape: tuple[int, int, int], role: str) -> tuple[int, ...]:
    rows, reduction, columns = shape
    return {
        "result": (rows, columns),
        "input_cotangent": (rows, reduction),
        "weight_cotangent": (reduction,),
        "projection_cotangent": (reduction, columns),
    }[role]


def validate_coordinate(
    coordinate: Coordinate,
    arrays: tuple[np.ndarray, ...],
    repeat_arrays: tuple[np.ndarray, ...],
    snapshot: tuple[tuple[object, ...], ...],
    *,
    disabled_jax: tuple[np.ndarray, ...] | None = None,
    report_cell: Callable[[dict[str, object]], None] | None = None,
) -> CoordinateResult:
    """Validate one public result tuple and return its canonical evidence."""
    observer = validate_observer_snapshot(coordinate, snapshot)
    roles = result_roles(coordinate.boundary)
    if len(arrays) != len(roles) or len(repeat_arrays) != len(roles):
        raise RuntimeError("RMSNorm Contract result count differs from its public boundary")
    for observed_arrays in (arrays, repeat_arrays):
        for role, array in zip(roles, observed_arrays, strict=True):
            if array.shape != _expected_shape(coordinate.shape, role) or array.dtype != np.dtype(ml_dtypes.bfloat16):
                raise RuntimeError(f"RMSNorm Contract {role} result type differs from its public boundary")
            if not np.isfinite(array.astype(np.float32)).all():
                raise RuntimeError(f"RMSNorm Contract {role} result must be finite")
    for role, actual, repeated in zip(roles, arrays, repeat_arrays, strict=True):
        if not _array_bytes_equal(actual, repeated):
            raise RuntimeError(f"RMSNorm Contract {role} repeat result drift")
    diagnostic = arrays if disabled_jax is None else disabled_jax
    if len(diagnostic) != len(roles):
        raise RuntimeError("disabled JAX diagnostic result count differs from the public boundary")
    for role, value in zip(roles, diagnostic, strict=True):
        if value.shape != _expected_shape(coordinate.shape, role) or value.dtype != np.dtype(ml_dtypes.bfloat16):
            raise RuntimeError(f"disabled JAX {role} diagnostic type differs from the public boundary")
    analytic = binary64_oracle(coordinate.shape)
    role_metrics = {
        role: numerical_metrics(
            actual,
            analytic[role],
            masked_ulp_absolute_floor=MASKED_ULP_ABSOLUTE_FLOORS[role],
        )
        for role, actual in zip(roles, arrays, strict=True)
    }
    if report_cell is not None:
        for role, actual, repeated, reference in zip(roles, arrays, repeat_arrays, diagnostic, strict=True):
            report_cell(_cell_report(coordinate, role, actual, repeated, reference, analytic[role], role_metrics[role]))
    if coordinate.mode is not Mode.DISABLED:
        for role, metrics in role_metrics.items():
            observed = _metrics_document(metrics)
            for name, limit in LIMITS[role].items():
                if observed[name] <= limit:
                    continue
                if name == "maximum_masked_bf16_ulp_error":
                    raise RuntimeError(f"RMSNorm Contract {role} exceeds its masked BF16 ULP threshold")
                raise RuntimeError(f"RMSNorm Contract {role} exceeds its independent binary64 threshold")
    if coordinate.mode is not Mode.DISABLED and coordinate.boundary is Boundary.PRIMAL:
        expected = emitted_primal_oracle(coordinate.shape, fast=coordinate.mode is Mode.FAST)
        if not _array_bytes_equal(arrays[0], expected):
            raise RuntimeError("RMSNorm Contract primal differs from its emitted recurrence oracle")
    return CoordinateResult(
        coordinate=coordinate,
        results={role: array_sha256(value) for role, value in zip(roles, arrays, strict=True)},
        repeat_results={role: array_sha256(value) for role, value in zip(roles, repeat_arrays, strict=True)},
        disabled_jax_results={role: array_sha256(value) for role, value in zip(roles, diagnostic, strict=True)},
        disabled_jax_equal={
            role: _array_bytes_equal(actual, reference)
            for role, actual, reference in zip(roles, arrays, diagnostic, strict=True)
        },
        analytic_oracle={role: array_sha256(analytic[role].astype(ml_dtypes.bfloat16)) for role in roles},
        numerics={role: _metrics_document(metrics) for role, metrics in role_metrics.items()},
        observer=observer,
    )


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib lacks the test-only Shuttle observer bridge")
    return bridge


def require_h100() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("RMSNorm Contract subject requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("RMSNorm Contract subject requires one visible H100")


def _execute(
    coordinate: Coordinate,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[tuple[object, ...], ...]]:
    inputs = deterministic_inputs(coordinate.shape)
    options = compiler_options_for(coordinate.mode)
    function = boundary_function(coordinate.boundary)
    compiled = jax.jit(function) if options is None else jax.jit(function, compiler_options=options)
    with _bridge().subscribe() as capture:
        outputs = compiled(*inputs)
        arrays = tuple(np.asarray(output.block_until_ready()) for output in outputs)
        repeated = compiled(*inputs)
        repeat_arrays = tuple(np.asarray(output.block_until_ready()) for output in repeated)
        snapshot = tuple(capture.snapshot_with_bundle_shape())
    return arrays, repeat_arrays, snapshot


def run_matrix(
    *,
    report_cell: Callable[[dict[str, object]], None] | None = None,
) -> tuple[CoordinateResult, ...]:
    """Execute all 12 coordinates on one H100."""
    require_h100()
    load_numerical_policy()
    results = []
    disabled_results: dict[tuple[tuple[int, int, int], Boundary], tuple[np.ndarray, ...]] = {}
    for coordinate in Coordinate.all():
        arrays, repeated, snapshot = _execute(coordinate)
        key = (coordinate.shape, coordinate.boundary)
        if coordinate.mode is Mode.DISABLED:
            disabled_results[key] = arrays
        baseline = disabled_results.get(key)
        if baseline is None:
            raise RuntimeError("enabled RMSNorm Contract coordinate ran before its disabled JAX diagnostic")
        results.append(
            validate_coordinate(
                coordinate,
                arrays,
                repeated,
                snapshot,
                disabled_jax=baseline,
                report_cell=report_cell,
            )
        )
    return tuple(results)


def result_document(results: tuple[CoordinateResult, ...]) -> dict[str, object]:
    return {
        "coordinates": [
            {
                "analytic_oracle": result.analytic_oracle,
                "boundary": result.coordinate.boundary.value,
                "disabled_jax_equal": result.disabled_jax_equal,
                "disabled_jax_results": result.disabled_jax_results,
                "mode": result.coordinate.mode.value,
                "numerics": result.numerics,
                "observer": (
                    None
                    if result.observer is None
                    else {
                        "final_module_sha256": result.observer.final_module_sha256,
                        "generated_bundle_sha256": result.observer.generated_bundle_sha256,
                        "invocation_id": result.observer.invocation_id,
                        "launch_count": result.observer.launch_count,
                        "temporary_count": result.observer.temporary_count,
                        "topology_status": result.observer.topology_status,
                    }
                ),
                "repeat_results": result.repeat_results,
                "results": result.results,
                "shape": list(result.coordinate.shape),
            }
            for result in results
        ],
        "numerical_policy_sha256": POLICY_SHA256,
        "pid": os.getpid(),
        "schema": "track1_rmsnorm_contract_h100_result",
        "schema_version": 2,
    }
