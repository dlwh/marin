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

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options

SHAPES = ((2, 7, 19), (3, 2, 5, 11, 263))


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
    host_oracle: dict[str, str]
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


def deterministic_inputs(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape not in SHAPES:
        raise ValueError("Track 1 RMSNorm shape is outside the canary matrix")
    size = int(np.prod(shape))
    ordinal = np.arange(size, dtype=np.float32)
    magnitude = 0.25 + (ordinal % 29) / 8.0
    sign = np.where((ordinal.astype(np.int64) % 2) == 0, 1.0, -1.0)
    x = (magnitude * sign).astype(ml_dtypes.bfloat16).reshape(shape)
    weight = np.linspace(0.5, 1.5, shape[-1], dtype=np.float32).astype(ml_dtypes.bfloat16)
    cotangent = (0.375 + (ordinal % 17) / 7.0).astype(ml_dtypes.bfloat16).reshape(shape)
    return x, weight, cotangent


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


def host_oracle(shape: tuple[int, ...], boundary: Boundary) -> tuple[np.ndarray, ...]:
    """Evaluate an independently derived float32 RMSNorm/VJP oracle on the host."""
    x_bf16, weight_bf16, cotangent_bf16 = deterministic_inputs(shape)
    x = x_bf16.astype(np.float32)
    weight = weight_bf16.astype(np.float32)
    cotangent = cotangent_bf16.astype(np.float32)
    mean_square = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    inverse_root = np.float32(1.0) / np.sqrt(mean_square + np.float32(1e-6))
    result = (x * inverse_root * weight).astype(ml_dtypes.bfloat16)
    weighted_cotangent = cotangent * weight
    projection = np.mean(weighted_cotangent * x, axis=-1, keepdims=True, dtype=np.float32)
    input_cotangent = np.asarray(
        weighted_cotangent * inverse_root - x * inverse_root * inverse_root * inverse_root * projection,
        dtype=ml_dtypes.bfloat16,
    )
    prefix_axes = tuple(range(len(shape) - 1))
    weight_cotangent = np.asarray(
        np.sum(cotangent * x * inverse_root, axis=prefix_axes, dtype=np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    if boundary is Boundary.FORWARD:
        return (result,)
    if boundary is Boundary.BACKWARD:
        return input_cotangent, weight_cotangent
    return result, input_cotangent, weight_cotangent


def _sha_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_bytes_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C")
    )


def _grouped(digest: str) -> str:
    return ":".join(digest[index : index + 8] for index in range(0, len(digest), 8))


def _numerical_mismatch(
    coordinate: Coordinate,
    role: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> RuntimeError:
    evidence = {
        "actual_dtype": str(actual.dtype),
        "actual_sha256": _grouped(_sha_array(actual)),
        "actual_shape": list(actual.shape),
        "boundary": coordinate.boundary.value,
        "expected_dtype": str(expected.dtype),
        "expected_sha256": _grouped(_sha_array(expected)),
        "expected_shape": list(expected.shape),
        "mode": coordinate.mode.value,
        "role": role,
        "schema_version": 1,
        "shape": list(coordinate.shape),
    }
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload) >= 960:
        raise RuntimeError("Track 1 RMSNorm mismatch evidence exceeds its bound")
    return RuntimeError(f"Track 1 RMSNorm numerical mismatch: {payload}")


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


def _execute(coordinate: Coordinate) -> tuple[tuple[np.ndarray, ...], tuple[tuple[object, ...], ...]]:
    x, weight, cotangent = deterministic_inputs(coordinate.shape)
    options = compiler_options_for(coordinate.mode)
    function = boundary_function(coordinate.boundary)
    compiled = jax.jit(function) if options is None else jax.jit(function, compiler_options=options)
    with _bridge().subscribe() as capture:
        outputs = compiled(x, weight, cotangent)
        arrays = tuple(np.asarray(output.block_until_ready()) for output in outputs)
        snapshot = tuple(capture.snapshot_with_bundle_shape())
    return arrays, snapshot


def run_matrix() -> tuple[CoordinateResult, ...]:
    """Execute the full ordinary-JAX RMSNorm canary matrix on one H100."""
    require_h100()
    baselines: dict[tuple[tuple[int, ...], Boundary], tuple[np.ndarray, ...]] = {}
    results = []
    for coordinate in Coordinate.all():
        arrays, snapshot = _execute(coordinate)
        observer = validate_observer_snapshot(coordinate, snapshot)
        roles = result_roles(coordinate.boundary)
        expected_shapes = tuple(
            (coordinate.shape[-1],) if role == "weight_cotangent" else coordinate.shape for role in roles
        )
        if len(arrays) != len(roles):
            raise RuntimeError("Track 1 RMSNorm result count differs from its public boundary")
        for role, array, expected_shape in zip(roles, arrays, expected_shapes, strict=True):
            if array.shape != expected_shape or array.dtype != np.dtype(ml_dtypes.bfloat16):
                raise RuntimeError(f"Track 1 RMSNorm {role} result type differs from its public boundary")
        oracle = host_oracle(coordinate.shape, coordinate.boundary)
        key = (coordinate.shape, coordinate.boundary)
        if coordinate.mode is Mode.DISABLED:
            for role, actual, expected in zip(roles, arrays, oracle, strict=True):
                if not np.allclose(
                    actual.astype(np.float32),
                    expected.astype(np.float32),
                    rtol=0.02,
                    atol=0.02,
                    equal_nan=False,
                ):
                    raise RuntimeError(f"Track 1 RMSNorm {role} differs from the independent host oracle")
            baselines[key] = arrays
        else:
            expected_arrays = baselines[key]
            for role, actual, expected in zip(roles, arrays, expected_arrays, strict=True):
                if not _array_bytes_equal(actual, expected):
                    raise _numerical_mismatch(coordinate, role, actual, expected)
        results.append(
            CoordinateResult(
                coordinate=coordinate,
                results={role: _sha_array(array) for role, array in zip(roles, arrays, strict=True)},
                host_oracle={role: _sha_array(array) for role, array in zip(roles, oracle, strict=True)},
                observer=observer,
            )
        )
    return tuple(results)


def result_document(results: tuple[CoordinateResult, ...]) -> dict[str, object]:
    return {
        "coordinates": [
            {
                "boundary": result.coordinate.boundary.value,
                "host_oracle": result.host_oracle,
                "mode": result.coordinate.mode.value,
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
                "results": result.results,
                "shape": list(result.coordinate.shape),
            }
            for result in results
        ],
        "pid": os.getpid(),
        "schema": "track1_rmsnorm_h100_result",
        "schema_version": 1,
    }
