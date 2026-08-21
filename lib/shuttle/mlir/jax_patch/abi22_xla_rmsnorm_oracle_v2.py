# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX/XLA cell producer for the matched RMSNorm v2 protocol."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
from abi22_te_rmsnorm_oracle_v2 import (
    BOUNDARIES,
    CALL_SEQUENCES,
    EPSILON,
    JAX_REVISION,
    MEASURED_INVOCATIONS,
    OUTPUT_ROLES,
    SHAPES,
    WARMUP_INVOCATIONS,
    error_metrics,
)
from jaxlib import version as jaxlib_version


def _observed_runtime() -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("ordinary-JAX RMSNorm evidence requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("ordinary-JAX RMSNorm evidence requires one visible H100")
    if jax.__version__ != "0.10.1" or jaxlib.__version__ != "0.10.1":
        raise RuntimeError("ordinary-JAX RMSNorm evidence requires JAX and jaxlib 0.10.1")
    if jaxlib_version._git_hash != JAX_REVISION:
        raise RuntimeError("ordinary-JAX RMSNorm evidence JAX revision differs")
    device = devices[0]
    return {
        "backend": "gpu",
        "device_count": 1,
        "device_id": device.id,
        "device_kind": device.device_kind,
        "jax_git_commit": jaxlib_version._git_hash,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform_version": device.client.platform_version,
    }


def _read_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype=np.uint16)
    if value.size != int(np.prod(shape)):
        raise ValueError(f"{path} has the wrong element count")
    return value.view(ml_dtypes.bfloat16).reshape(shape)


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(value.view(np.uint16).tobytes(order="C")).hexdigest()


def _rmsnorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(local * local, axis=-1, keepdims=True) + EPSILON)
    return (local * inverse * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def _function(boundary: str) -> Any:
    if boundary == "forward":
        return _rmsnorm
    if boundary == "backward_recompute":

        def backward(x: jax.Array, gamma: jax.Array, dy: jax.Array) -> tuple[jax.Array, jax.Array]:
            _, pullback = jax.vjp(_rmsnorm, x, gamma)
            return pullback(dy)

        return backward
    if boundary == "composed":

        def composed(x: jax.Array, gamma: jax.Array, dy: jax.Array) -> tuple[jax.Array, ...]:
            y, pullback = jax.vjp(_rmsnorm, x, gamma)
            dx, dgamma = pullback(dy)
            return y, dx, dgamma

        return composed
    raise ValueError("boundary is outside the RMSNorm v2 contract")


def _ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _arrays(value: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(leaf) for leaf in jax.tree.leaves(value))


def run_cell(boundary: str, shape: tuple[int, int], case_directory: Path) -> dict[str, Any]:
    """Run one XLA cell with host-blocking samples and BF16 output evidence."""
    if boundary not in BOUNDARIES or shape not in SHAPES:
        raise ValueError("cell is outside the RMSNorm v2 matrix")
    observed_runtime = _observed_runtime()
    _, features = shape
    operands = [
        _read_bf16(case_directory / "x.bf16", shape),
        _read_bf16(case_directory / "gamma.bf16", (features,)),
    ]
    if boundary != "forward":
        operands.append(_read_bf16(case_directory / "dy.bf16", shape))
    input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    input_digests = {role: _digest(operand) for role, operand in zip(input_roles, operands, strict=True)}
    device_operands = tuple(jax.device_put(operand) for operand in operands)
    compiled = jax.jit(_function(boundary)).lower(*device_operands).compile()
    for _ in range(WARMUP_INVOCATIONS):
        _ready(compiled(*device_operands))
    samples = []
    result = None
    for _ in range(MEASURED_INVOCATIONS):
        started = time.perf_counter_ns()
        result = _ready(compiled(*device_operands))
        samples.append(time.perf_counter_ns() - started)
    if result is None:
        raise AssertionError("measurement loop did not execute")
    outputs = []
    for role, actual in zip(OUTPUT_ROLES[boundary], _arrays(result), strict=True):
        reference_shape = shape if role != "dgamma" else (features,)
        reference = _read_bf16(case_directory / f"reference_{role}.bf16", reference_shape)
        outputs.append(
            {
                "role": role,
                "digest": _digest(actual),
                "reference_digest": _digest(reference),
                **error_metrics(actual, reference, role=role),
            }
        )
    return {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == SHAPES[0] else "representative_performance_qualification",
        "boundary": boundary,
        "input_digests": input_digests,
        "execution": {
            "call_sequence": list(CALL_SEQUENCES[boundary]),
            "rsigma_state": "invocation_local_not_public",
        },
        "outputs": outputs,
        "observed_runtime": observed_runtime,
        "timing": {
            "warmup_invocations": WARMUP_INVOCATIONS,
            "raw_host_blocking_ns": samples,
            "median_host_blocking_ns": statistics.median(samples),
            "raw_cuda_event_ns": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", choices=BOUNDARIES, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.epsilon != EPSILON:
        parser.error("epsilon must be exactly 1e-6")
    document = run_cell(arguments.boundary, (arguments.rows, arguments.features), arguments.case_directory)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
