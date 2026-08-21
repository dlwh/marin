# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX H100 subject for the generic ABI22 RMSNorm Fold regions."""

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
from track1_rmsnorm_numerical_policy import (
    POLICY_SHA256,
    SHAPES,
    NumericalMetrics,
    array_sha256,
    binary64_oracle,
    deterministic_inputs,
    load_numerical_policy,
    numerical_metrics,
)

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options


class Boundary(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"
    COMPOSED = "composed"


class Mode(StrEnum):
    DISABLED = "disabled"
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True, order=True)
class Coordinate:
    shape: tuple[int, ...]
    boundary: Boundary
    mode: Mode

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError("Track 1 RMSNorm shape is outside the canary matrix")

    @classmethod
    def all(cls) -> tuple["Coordinate", ...]:
        return tuple(cls(shape, boundary, mode) for shape in SHAPES for boundary in Boundary for mode in Mode)


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


@dataclass(frozen=True)
class CoordinateResult:
    coordinate: Coordinate
    results: dict[str, str]
    repeat_results: dict[str, str]
    analytic_oracle: dict[str, str]
    emitted_tree_oracle: dict[str, str]
    numerics: dict[str, dict[str, float | int]]
    disabled_jax: dict[str, str]
    disabled_jax_bitwise_equal: dict[str, bool]
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


def rms_norm_forward(x: jax.Array, weight: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    mean_square = jnp.mean(local * local, axis=-1, keepdims=True)
    inverse_root = jax.lax.rsqrt(mean_square + jnp.float32(1e-6))
    return (local * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def result_roles(boundary: Boundary) -> tuple[str, ...]:
    if boundary is Boundary.FORWARD:
        return ("result",)
    if boundary is Boundary.BACKWARD:
        return ("input_cotangent", "weight_cotangent")
    return ("result", "input_cotangent", "weight_cotangent")


def expected_bundle_shape(boundary: Boundary) -> tuple[int, int]:
    return {
        Boundary.FORWARD: (1, 0),
        Boundary.BACKWARD: (3, 1),
        Boundary.COMPOSED: (4, 1),
    }[boundary]


def boundary_function(boundary: Boundary) -> Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    def subject(x: jax.Array, weight: jax.Array, cotangent: jax.Array) -> tuple[jax.Array, ...]:
        result, pullback = jax.vjp(rms_norm_forward, x, weight)
        if boundary is Boundary.FORWARD:
            return (result,)
        gradients = pullback(cotangent)
        if boundary is Boundary.BACKWARD:
            return gradients
        return (result, *gradients)

    return subject


def _metrics_document(metrics: NumericalMetrics) -> dict[str, float | int]:
    return {
        "maximum_absolute_error": metrics.maximum_absolute_error,
        "maximum_bf16_ulp_error": metrics.maximum_bf16_ulp_error,
        "mean_absolute_error": metrics.mean_absolute_error,
        "relative_linf_error": metrics.relative_linf_error,
    }


def _require_accepted_numerics(
    coordinate: Coordinate,
    role: str,
    metrics: NumericalMetrics,
    limits: dict[str, float | int],
) -> None:
    observed = _metrics_document(metrics)
    exceeded = tuple(name for name, limit in limits.items() if observed[name] > limit)
    if not exceeded:
        return
    evidence = {
        "boundary": coordinate.boundary.value,
        "exceeded": list(exceeded),
        "limits": limits,
        "metrics": observed,
        "mode": coordinate.mode.value,
        "role": role,
        "schema_version": 1,
        "shape": list(coordinate.shape),
    }
    raise RuntimeError(
        "Track 1 RMSNorm result exceeds its independent binary64 threshold: "
        + json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _sha_array(value: np.ndarray) -> str:
    return array_sha256(value)


def _array_bytes_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C")
    )


def _grouped(digest: str) -> str:
    return ":".join(digest[index : index + 8] for index in range(0, len(digest), 8))


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
            raise ValueError("disabled RMSNorm compilation emitted Shuttle observer events")
        return None
    if len(snapshot) != 3 or any(type(event) is not tuple or len(event) != 13 for event in snapshot):
        raise ValueError("enabled RMSNorm compilation must emit three closed observer records")
    if tuple(event[1] for event in snapshot) != ("algebra_coverage", "generated_bundle", "final_erasure"):
        raise ValueError("enabled RMSNorm compilation emitted the wrong observer phases")
    invocation_ids = {event[0] for event in snapshot}
    if len(invocation_ids) != 1 or type(snapshot[0][0]) is not int or snapshot[0][0] <= 0:
        raise ValueError("RMSNorm observer phases do not share one positive invocation")
    identity = observer_identity(coordinate)
    for event in snapshot:
        if event[2:5] != (coordinate.mode.value, identity.policy_sha256, identity.tuning_sha256):
            raise ValueError("RMSNorm observer policy identity differs from compiler options")
        if event[10] != "":
            raise ValueError("successful RMSNorm observer event carries a failure pass")
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
        raise ValueError("RMSNorm observer selection attribution changed before bundle generation")
    if snapshot[0][9] is not False or snapshot[1][9] is not False:
        raise ValueError("RMSNorm observer erased Shuttle semantics before the final phase")
    if snapshot[2][5:8] != ("", "", "") or snapshot[2][9] is not True:
        raise ValueError("RMSNorm observer final phase did not erase Shuttle semantics")
    expected_launches, expected_temporaries = expected_bundle_shape(coordinate.boundary)
    if snapshot[0][11:13] != (0, 0) or snapshot[2][11:13] != (0, 0):
        raise ValueError("RMSNorm observer reported bundle shape outside bundle generation")
    if snapshot[1][11:13] != (expected_launches, expected_temporaries):
        raise ValueError("RMSNorm generated bundle launch or temporary count differs")
    return ObserverEvidence(
        invocation_id=snapshot[0][0],
        generated_bundle_sha256=_sha(snapshot[1][8], "generated bundle fingerprint"),
        final_module_sha256=_sha(snapshot[2][8], "final module fingerprint"),
        launch_count=expected_launches,
        temporary_count=expected_temporaries,
    )


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib lacks the test-only Shuttle observer bridge")
    return bridge


def require_h100() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("Track 1 RMSNorm subject requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("Track 1 RMSNorm subject requires one visible H100")


def _execute(
    coordinate: Coordinate,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[tuple[object, ...], ...]]:
    x, weight, cotangent = deterministic_inputs(coordinate.shape)
    options = compiler_options_for(coordinate.mode)
    function = boundary_function(coordinate.boundary)
    compiled = jax.jit(function) if options is None else jax.jit(function, compiler_options=options)
    with _bridge().subscribe() as capture:
        outputs = compiled(x, weight, cotangent)
        arrays = tuple(np.asarray(output.block_until_ready()) for output in outputs)
        repeated = compiled(x, weight, cotangent)
        repeat_arrays = tuple(np.asarray(output.block_until_ready()) for output in repeated)
        snapshot = tuple(capture.snapshot_with_bundle_shape())
    return arrays, repeat_arrays, snapshot


def run_matrix() -> tuple[CoordinateResult, ...]:
    """Execute the full ordinary-JAX RMSNorm canary matrix on one H100."""
    require_h100()
    policy = load_numerical_policy()
    acceptance = policy["acceptance"]
    shape_policies = policy["shapes"]
    if type(acceptance) is not dict or type(shape_policies) is not dict:
        raise RuntimeError("Track 1 RMSNorm numerical policy is malformed")
    thresholds = acceptance["independent_binary64_thresholds"]
    if type(thresholds) is not dict:
        raise RuntimeError("Track 1 RMSNorm numerical thresholds are malformed")
    baselines: dict[tuple[tuple[int, ...], Boundary], tuple[np.ndarray, ...]] = {}
    results = []
    for coordinate in Coordinate.all():
        arrays, repeat_arrays, snapshot = _execute(coordinate)
        observer = validate_observer_snapshot(coordinate, snapshot)
        roles = result_roles(coordinate.boundary)
        expected_shapes = tuple(
            (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape for role in roles
        )
        if len(arrays) != len(roles) or len(repeat_arrays) != len(roles):
            raise RuntimeError("Track 1 RMSNorm result count differs from its public boundary")
        for observed_arrays in (arrays, repeat_arrays):
            for role, array, expected_shape in zip(roles, observed_arrays, expected_shapes, strict=True):
                if array.shape != expected_shape or array.dtype != np.dtype(ml_dtypes.bfloat16):
                    raise RuntimeError(f"Track 1 RMSNorm {role} result type differs from its public boundary")
                if not np.isfinite(array.astype(np.float32)).all():
                    raise RuntimeError(f"Track 1 RMSNorm {role} result must be finite")
        for role, actual, repeated in zip(roles, arrays, repeat_arrays, strict=True):
            if not _array_bytes_equal(actual, repeated):
                raise RuntimeError(
                    f"Track 1 RMSNorm {role} repeat result drift: "
                    f"first={_grouped(_sha_array(actual))},repeat={_grouped(_sha_array(repeated))}"
                )
        shape_id = "x".join(map(str, coordinate.shape))
        shape_policy = shape_policies[shape_id]
        if type(shape_policy) is not dict:
            raise RuntimeError("Track 1 RMSNorm shape policy is malformed")
        analytic_by_role = binary64_oracle(coordinate.shape)
        oracle = tuple(analytic_by_role[role] for role in roles)
        role_metrics = {
            role: numerical_metrics(actual, expected)
            for role, actual, expected in zip(roles, arrays, oracle, strict=True)
        }
        for role, metrics in role_metrics.items():
            limits = thresholds[role]
            if type(limits) is not dict:
                raise RuntimeError("Track 1 RMSNorm role thresholds are malformed")
            _require_accepted_numerics(coordinate, role, metrics, limits)
            expected_schedule_sha256 = shape_policy["emitted_tree_outputs"][role]
            if coordinate.mode is not Mode.DISABLED and _sha_array(arrays[roles.index(role)]) != (
                expected_schedule_sha256
            ):
                raise RuntimeError(f"Track 1 RMSNorm {role} differs from its predeclared emitted-tree oracle")
        key = (coordinate.shape, coordinate.boundary)
        if coordinate.mode is Mode.DISABLED:
            baselines[key] = arrays
        disabled_arrays = baselines[key]
        results.append(
            CoordinateResult(
                coordinate=coordinate,
                results={role: _sha_array(array) for role, array in zip(roles, arrays, strict=True)},
                repeat_results={role: _sha_array(array) for role, array in zip(roles, repeat_arrays, strict=True)},
                analytic_oracle={role: shape_policy["binary64_rounded_outputs"][role] for role in roles},
                emitted_tree_oracle={role: shape_policy["emitted_tree_outputs"][role] for role in roles},
                numerics={role: _metrics_document(metrics) for role, metrics in role_metrics.items()},
                disabled_jax={role: _sha_array(array) for role, array in zip(roles, disabled_arrays, strict=True)},
                disabled_jax_bitwise_equal={
                    role: _array_bytes_equal(array, baseline)
                    for role, array, baseline in zip(roles, arrays, disabled_arrays, strict=True)
                },
                observer=observer,
            )
        )
    return tuple(results)


def result_document(results: tuple[CoordinateResult, ...]) -> dict[str, object]:
    return {
        "coordinates": [
            {
                "analytic_oracle": result.analytic_oracle,
                "boundary": result.coordinate.boundary.value,
                "disabled_jax": result.disabled_jax,
                "disabled_jax_bitwise_equal": result.disabled_jax_bitwise_equal,
                "emitted_tree_oracle": result.emitted_tree_oracle,
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
        "schema": "track1_rmsnorm_h100_result",
        "schema_version": 2,
    }
