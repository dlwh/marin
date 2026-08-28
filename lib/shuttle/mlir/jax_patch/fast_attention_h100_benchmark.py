# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exploratory matched H100 timing for ordinary XLA, Shuttle, and Grug FA4."""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import importlib
import json
import math
import statistics
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib import _jax

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options


def _load_grug_boundary() -> tuple[Callable[..., jax.Array], Callable[..., Any]]:
    attention_path = next(
        candidate for entry in sys.path if (candidate := Path(entry) / "levanter/grug/attention").is_dir()
    )
    package_paths = {
        "levanter": attention_path.parents[1],
        "levanter.grug": attention_path.parent,
        "levanter.grug.attention": attention_path,
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    cache_boundary = types.ModuleType("levanter.cutlass_kernel_cache")

    @functools.cache
    def cutlass_call(launcher: Any, **kwargs: Any) -> Any:
        return importlib.import_module("cutlass.jax").cutlass_call(launcher, **kwargs)

    def cute_launcher_factory(build: Callable[..., Any]) -> Callable[..., Any]:
        return functools.lru_cache(maxsize=None)(build)

    cache_boundary.cutlass_call = cutlass_call
    cache_boundary.cute_launcher_factory = cute_launcher_factory
    sys.modules[cache_boundary.__name__] = cache_boundary
    from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward  # noqa: PLC0415
    from levanter.grug.attention._fa4_cute_config import flash4_cute_kernel_config  # noqa: PLC0415

    return fa4_cute_attention_forward, flash4_cute_kernel_config


fa4_cute_attention_forward, flash4_cute_kernel_config = _load_grug_boundary()


WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 51


@dataclass(frozen=True)
class Shape:
    batch: int
    sequence: int
    query_heads: int
    key_value_heads: int
    dimension: int


class CudaEvents:
    def __init__(self) -> None:
        runtime = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
        if not runtime.is_file():
            raise RuntimeError(f"pinned CUDA runtime is missing: {runtime}")
        self.library = ctypes.CDLL(str(runtime))
        self.library.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.cudaEventDestroy.argtypes = [ctypes.c_void_p]

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {result}")

    def elapsed_ns(self, stream: int, function: Callable[[], jax.Array]) -> tuple[int, jax.Array]:
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self._check(self.library.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate(start)")
        self._check(self.library.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate(stop)")
        try:
            self._check(self.library.cudaEventRecord(start, ctypes.c_void_p(stream)), "cudaEventRecord(start)")
            result = function()
            ready_capsule = result.__dlpack__(stream=stream)
            self._check(self.library.cudaEventRecord(stop, ctypes.c_void_p(stream)), "cudaEventRecord(stop)")
            self._check(self.library.cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")
            milliseconds = ctypes.c_float()
            self._check(
                self.library.cudaEventElapsedTime(ctypes.byref(milliseconds), start, stop),
                "cudaEventElapsedTime",
            )
            del ready_capsule
            return round(float(milliseconds.value) * 1_000_000), result
        finally:
            self.library.cudaEventDestroy(start)
            self.library.cudaEventDestroy(stop)


def _stream(device: jax.Device) -> int:
    stream = device.get_stream_for_external_ready_events()
    if type(stream) is not int or stream <= 2:
        raise RuntimeError(f"JAX device does not expose a concrete CUDA event stream: {stream!r}")
    return stream


def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q.astype(jnp.float32), k.astype(jnp.float32))
    scores *= np.float32(1.0 / math.sqrt(q.shape[-1]))
    sequence = q.shape[-2]
    query_index = jnp.arange(sequence, dtype=jnp.int32)[:, None]
    key_index = jnp.arange(sequence, dtype=jnp.int32)[None, :]
    predicate = jnp.broadcast_to(key_index <= query_index, scores.shape)
    scores = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    exponentials = jnp.exp(scores - maximum)
    normalized = exponentials / jnp.sum(exponentials, axis=-1, keepdims=True)
    return jnp.einsum("bhgqk,bhkv->bhgqv", normalized, v.astype(jnp.float32)).astype(jnp.bfloat16)


def grug_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    batch, key_value_heads, group, sequence, dimension = q.shape
    q_bshd = q.transpose(0, 3, 1, 2, 4).reshape(batch, sequence, key_value_heads * group, dimension)
    k_bshd = k.transpose(0, 2, 1, 3)
    v_bshd = v.transpose(0, 2, 1, 3)
    lower_bounds = jnp.zeros((batch, sequence), dtype=jnp.int32)
    valid = jnp.ones((batch, sequence), dtype=jnp.bool_)
    output = fa4_cute_attention_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        lower_bounds,
        valid,
        sm_scale=1.0 / math.sqrt(dimension),
        kernel_config=flash4_cute_kernel_config(dimension, arch=90),
    )
    return output.reshape(batch, sequence, key_value_heads, group, dimension).transpose(0, 2, 3, 1, 4)


def offset_softcap_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    scores = jnp.einsum("qbhgd,kbhd->qkbhg", q.astype(jnp.float32), k.astype(jnp.float32))
    scores *= np.float32(0.1767766952966369)
    scores = np.float32(12.0) * jnp.tanh(scores / np.float32(12.0))
    query = jnp.arange(q.shape[0], dtype=jnp.int32).reshape(q.shape[0], 1, 1, 1, 1)
    key = jnp.arange(k.shape[0], dtype=jnp.int32).reshape(1, k.shape[0], 1, 1, 1)
    predicate = jnp.broadcast_to(key <= query + jnp.int32(4), scores.shape)
    scores = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(scores, axis=1, keepdims=True)
    exponentials = jnp.exp(scores - maximum)
    normalized = exponentials / jnp.sum(exponentials, axis=1, keepdims=True)
    return jnp.einsum("qkbhg,kbhv->qbhgv", normalized, v.astype(jnp.float32)).astype(jnp.bfloat16)


def _shuttle_options() -> dict[str, bool | str]:
    return compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.FAST,
        tuning=Tuning(
            tile_sizes=(),
            cluster_shape=(),
            pipeline_stages=1,
            materialization=Materialization.PREFER_FUSION,
            maximum_candidates=1,
        ),
    )


def _inputs(shape: Shape, device: jax.Device) -> tuple[jax.Array, jax.Array, jax.Array]:
    generator = np.random.default_rng(20260827)
    arrays = (
        generator.normal(
            size=(
                shape.batch,
                shape.key_value_heads,
                shape.query_heads // shape.key_value_heads,
                shape.sequence,
                shape.dimension,
            )
        ),
        generator.normal(size=(shape.batch, shape.key_value_heads, shape.sequence, shape.dimension)),
        generator.normal(size=(shape.batch, shape.key_value_heads, shape.sequence, shape.dimension)),
    )
    return tuple(jax.device_put(value.astype(jnp.bfloat16), device) for value in arrays)


def _compile(
    shape: Shape, operands: tuple[jax.Array, ...]
) -> tuple[dict[str, Callable[[], jax.Array]], tuple[tuple[object, ...], ...], dict[str, str | int]]:
    ordinary = jax.jit(attention).lower(*operands).compile()
    observer = getattr(_jax, "_shuttle_test_observer", None)
    if observer is None:
        raise RuntimeError("jaxlib lacks the Shuttle observer bridge")
    with observer.subscribe() as capture:
        shuttle = jax.jit(attention, compiler_options=_shuttle_options()).lower(*operands).compile()
        snapshot = tuple(capture.snapshot())
    grug = jax.jit(grug_attention).lower(*operands).compile()
    cutlass_compile = importlib.import_module("cutlass.jax.compile")
    compile_results = tuple(cutlass_compile._CUTLASS_COMPILE_CACHE.values())
    if len(compile_results) != 1:
        raise RuntimeError(f"Grug compilation produced {len(compile_results)} CUTLASS modules")
    module = bytes(compile_results[0].module)
    artifact_directory = Path("/tmp/shuttle-matched/artifacts")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    module_path = artifact_directory / "grug-fa4-cutlass.o"
    module_path.write_bytes(module)
    cutlass_identity = {
        "bytes": len(module),
        "fingerprint": bytes(compile_results[0].fingerprint).hex(),
        "sha256": hashlib.sha256(module).hexdigest(),
    }
    return (
        {
            "ordinary_xla": lambda: ordinary(*operands),
            "shuttle_fast": lambda: shuttle(*operands),
            "grug_fa4": lambda: grug(*operands),
        },
        snapshot,
        cutlass_identity,
    )


def _softcap_canary(device: jax.Device) -> dict[str, float | bool]:
    generator = np.random.default_rng(20260828)
    shapes = ((19, 1, 3, 2, 128), (23, 1, 3, 128), (23, 1, 3, 128))
    operands = tuple(jax.device_put(generator.normal(size=shape).astype(jnp.bfloat16), device) for shape in shapes)
    ordinary = jax.jit(offset_softcap_attention).lower(*operands).compile()
    shuttle = jax.jit(offset_softcap_attention, compiler_options=_shuttle_options()).lower(*operands).compile()
    reference = np.asarray(ordinary(*operands).block_until_ready(), dtype=np.float32)
    actual = np.asarray(shuttle(*operands).block_until_ready(), dtype=np.float32)
    absolute = np.abs(actual - reference)
    maximum = float(np.max(absolute, initial=0.0))
    mean = float(np.mean(absolute))
    finite = bool(np.all(np.isfinite(actual)))
    passed = finite and maximum <= 0.015625 and mean <= 0.001
    if not passed:
        raise RuntimeError(f"softcap canary failed: finite={finite}, maximum={maximum}, mean={mean}")
    return {"finite": finite, "maximum_absolute_error_vs_xla": maximum, "mean_absolute_error_vs_xla": mean}


def run(shape: Shape) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("benchmark requires the JAX GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("benchmark requires exactly one visible H100")
    operands = _inputs(shape, devices[0])
    functions, observer_snapshot, cutlass_identity = _compile(shape, operands)
    outputs: dict[str, jax.Array] = {}
    for name, function in functions.items():
        output = function()
        output.block_until_ready()
        outputs[name] = output
        for _ in range(WARMUP_ITERATIONS - 1):
            function().block_until_ready()
    stream = _stream(devices[0])
    timer = CudaEvents()
    samples = {name: [] for name in functions}
    names = tuple(functions)
    for round_index in range(MEASURED_ITERATIONS):
        order = names if round_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            elapsed, output = timer.elapsed_ns(stream, functions[name])
            samples[name].append(elapsed)
            outputs[name] = output
    reference = np.asarray(outputs["ordinary_xla"], dtype=np.float32)
    errors = {
        name: float(np.max(np.abs(np.asarray(output, dtype=np.float32) - reference), initial=0.0))
        for name, output in outputs.items()
    }
    if any(not np.all(np.isfinite(np.asarray(output, dtype=np.float32))) for output in outputs.values()):
        raise RuntimeError("matched benchmark produced a non-finite output")
    if errors["shuttle_fast"] > 0.015625 or errors["grug_fa4"] > 0.015625:
        raise RuntimeError(f"matched benchmark numerical error exceeded BF16 policy: {errors}")
    bundle_events = [event for event in observer_snapshot if event[1] == "generated_bundle"]
    if len(bundle_events) != 1:
        raise RuntimeError("Shuttle compilation did not emit one generated-bundle observation")
    return {
        "schema": "fast_attention_h100_matched_benchmark",
        "shape": shape.__dict__,
        "dtype": "bf16",
        "device": devices[0].device_kind,
        "stream": stream,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "samples_ns": samples,
        "median_ns": {name: statistics.median(values) for name, values in samples.items()},
        "maximum_absolute_error_vs_xla": errors,
        "shuttle_generated_bundle_sha256": bundle_events[0][8],
        "softcap_axis1_canary": _softcap_canary(devices[0]),
        "threshold_applied": True,
        "jax_version": jax.__version__,
        "grug_cutlass_object": cutlass_identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=128)
    arguments = parser.parse_args()
    report = run(Shape(1, arguments.sequence, 32, 8, arguments.dimension))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
