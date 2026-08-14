# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed ordinary-JAX ABI 11 GPU LayerNorm acceptance driver."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from acceptance_contract import ObserverIdentity, decode_native_snapshot
from jaxlib import _jax
from shuttle_jaxlib_target1_acceptance import (
    CacheHitCounter,
    acceptance_tuning,
    cache_snapshot,
    configure_cache,
)

from shuttle import ExecutionMode, Numerics, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PIPELINE_ABI_VERSION = 11
BOUNDARIES = ("forward", "backward", "composed")
POLICIES = (Numerics.SOURCE_ORDERED, Numerics.FAST)
WORKERS = ("baseline", "populate", "reuse")
OBSERVER_PHASES = ("algebra_coverage", "generated_bundle", "final_erasure")
UNSUPPORTED_FINGERPRINT = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


@dataclass(frozen=True)
class Shape:
    rows: int
    columns: int
    shape_id: str


SHAPES = (
    Shape(7, 13, "81928ab3539c0f03"),
    Shape(2048, 4096, "44d152ecc3e9ff18"),
)

# (generated-bundle fingerprint, final-erasure fingerprint), derived locally
# from the exact checked-in StableHLO fixtures and pinned XLA pipeline.
OBSERVER_FINGERPRINTS = {
    ("81928ab3539c0f03", "forward", "source_ordered"): (
        "9709ec5748d1782201c66b757b277e581c4b6ba7d22e4f7c0dcca6d8671e5a73",
        "1c3f73a65db2df46631437f30f781de0b7e94e0c04d0cea91285178b681cf0bf",
    ),
    ("81928ab3539c0f03", "forward", "fast"): (
        "38f9c6b58ea83d426a297e0e2e499e547f1ffc6e2af3eac4acdb30fa2f5d6cd3",
        "64a17a580ab728cfc3ff5eb381508112f9c322a69681c545d288bc7bdb0437fe",
    ),
    ("81928ab3539c0f03", "backward", "source_ordered"): (
        "ddc1f20c8db96585d2ae722a13fcdbdb764b3d965a52a43ebe7c62d6de9d5b76",
        "6aa2dd0cd4a8bfdfb9e530ff2e7d0ed8c05b1a8fdb924bc91538272f8b062fb1",
    ),
    ("81928ab3539c0f03", "backward", "fast"): (
        "139587df1e0fbe5e9c000533c1b1395a4a447c895b82c306831c76639a1dcb3f",
        "ae00099b7fea645587f925f49e98ba2f344f60e137c20a7941e4fd012e3df000",
    ),
    ("81928ab3539c0f03", "composed", "source_ordered"): (
        "6b59d6a04b3e8366571b08e5ed6278ebc8594eba66d5a7dc52749ae6fdd023d4",
        "966722fb8e7ed4f2fcb8c3747b7e3eb678a71d7fe8a7f3d04658ab89872d8019",
    ),
    ("81928ab3539c0f03", "composed", "fast"): (
        "539ccc57b4cc30004f6ba3b756c3c09aaf29bfea4d2ab2a9f1d1bfb8a91af551",
        "e9657c7c406ccb0037bc990287fe5f261f17c994e1e02d78c03b8086cc9b0038",
    ),
    ("44d152ecc3e9ff18", "forward", "source_ordered"): (
        "51a675acd6bf1254176267c8a0d01ccd04357710f564233d35a70a14a780fbb5",
        "4eceb5ea70cd2e919c9823af01be735a5937b3d25b539be050dc81083691a2cd",
    ),
    ("44d152ecc3e9ff18", "forward", "fast"): (
        "b03a5d84794682f2eca360820fbc937656882f547cc9725395e75b7c8b944d76",
        "055c5a4bd1e22e71037003cbe85ac2baa2a3589c53be7237acc6c10545d58858",
    ),
    ("44d152ecc3e9ff18", "backward", "source_ordered"): (
        "3b2f78ad26b3cad2758d78d5adefab0c16756838979469af41733870245bb55d",
        "fa32168a74fc038ba7a11a768f168eea57d19ae70ac0ac23c6c22a08d02ba9b7",
    ),
    ("44d152ecc3e9ff18", "backward", "fast"): (
        "91c44297721ebcb838332a9826e3fa93c07e55c5506fe16aa30da5e622012597",
        "18e412711edc31de7fe46375b94ea95692495ae43202e9f257c264e63b04319d",
    ),
    ("44d152ecc3e9ff18", "composed", "source_ordered"): (
        "8c60d2debd1344bffcd251517f9c05c9b3a886f2780c97377d93ae63be86c865",
        "537b96019fa2cdc021ff70252f588c8e1eddaabaaa754076e98a4fed6e708a94",
    ),
    ("44d152ecc3e9ff18", "composed", "fast"): (
        "a6fea4448ad3a907dcec72e94013132d370ee5c9fc58966912856eac5a8db954",
        "74a606047a0a88279adae7efa80d4eaca9c614957fc806e75eb63601277e8961",
    ),
}

NUMERICAL_LIMITS = {
    "81928ab3539c0f03": {
        "y": (0.0078125, 0.001),
        "dx": (0.0078125, 0.001),
        "dgamma": (0.015625, 0.004),
        "dbeta": (0.015625, 0.004),
    },
    "44d152ecc3e9ff18": {
        "y": (0.0078125, 0.001),
        "dx": (1.0, 0.1),
        "dgamma": (0.125, 0.05),
        "dbeta": (1.0, 0.5),
    },
}


def layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    mean = jnp.mean(local, axis=-1, keepdims=True)
    centered = local - mean
    inverse = jax.lax.rsqrt(jnp.mean(centered * centered, axis=-1, keepdims=True) + jnp.float32(1e-5))
    return (centered * inverse * gamma.astype(jnp.float32) + beta.astype(jnp.float32)).astype(jnp.bfloat16)


def boundary_function(boundary: str):
    if boundary == "forward":
        return layer_norm
    if boundary == "backward":

        def backward(x, gamma, dy):
            beta = jnp.zeros_like(gamma)
            _, pullback = jax.vjp(layer_norm, x, gamma, beta)
            return pullback(dy)

        return backward
    if boundary == "composed":

        def composed(x, gamma, beta, dy):
            y, pullback = jax.vjp(layer_norm, x, gamma, beta)
            dx, dgamma, dbeta = pullback(dy)
            return y, dx, dgamma, dbeta

        return composed
    raise ValueError(f"unknown LayerNorm boundary: {boundary}")


def fixed_inputs(shape: Shape) -> tuple[np.ndarray, ...]:
    bfloat16 = np.dtype("bfloat16")
    x = np.linspace(-0.75, 0.875, shape.rows * shape.columns, dtype=np.float32).reshape(shape.rows, shape.columns)
    gamma = np.linspace(-0.625, 1.0, shape.columns, dtype=np.float32)
    beta = np.linspace(-0.25, 0.375, shape.columns, dtype=np.float32)
    dy = np.linspace(-0.5, 0.75, shape.rows * shape.columns, dtype=np.float32).reshape(shape.rows, shape.columns)
    return tuple(value.astype(bfloat16) for value in (x, gamma, beta, dy))


def independent_reference(shape: Shape) -> tuple[np.ndarray, ...]:
    x, gamma, beta, dy = (value.astype(np.float64) for value in fixed_inputs(shape))
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    inverse = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inverse
    y = normalized * gamma + beta
    scaled_cotangent = dy * gamma
    dx = (
        scaled_cotangent
        - np.mean(scaled_cotangent, axis=-1, keepdims=True)
        - normalized * np.mean(scaled_cotangent * normalized, axis=-1, keepdims=True)
    ) * inverse
    dgamma = np.sum(dy * normalized, axis=0)
    dbeta = np.sum(dy, axis=0)
    return y, dx, dgamma, dbeta


def boundary_inputs(shape: Shape, boundary: str) -> tuple[np.ndarray, ...]:
    values = fixed_inputs(shape)
    if boundary == "forward":
        return values[:3]
    if boundary == "backward":
        return values[0], values[1], values[3]
    return values


def arrays(value: Any) -> tuple[np.ndarray, ...]:
    values = value if isinstance(value, tuple) else (value,)
    return tuple(np.asarray(item.block_until_ready()) for item in values)


def output_names(boundary: str) -> tuple[str, ...]:
    return {
        "forward": ("y",),
        "backward": ("dx", "dgamma", "dbeta"),
        "composed": ("y", "dx", "dgamma", "dbeta"),
    }[boundary]


def cell_identities() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (shape.shape_id, boundary, numerics.value)
        for shape in SHAPES
        for numerics in POLICIES
        for boundary in BOUNDARIES
    )


def subject_options(numerics: Numerics) -> dict[str, object]:
    options = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=numerics,
        tuning=acceptance_tuning(),
    )
    encoded = options.get("xla_shuttle_options")
    if not isinstance(encoded, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(encoded)
    if payload.get("pipeline_abi_version") != PIPELINE_ABI_VERSION:
        raise AssertionError("LayerNorm GPU proof requires pipeline ABI 11")
    if payload.get("numerics") != numerics.value:
        raise AssertionError("LayerNorm GPU proof lost its numerical policy")
    return options


def expected_identity(numerics: Numerics) -> ObserverIdentity:
    canonical = subject_options(numerics)["xla_shuttle_options"]
    if not isinstance(canonical, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(canonical)
    tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        policy=numerics.value,
        policy_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(tuning.encode()).hexdigest(),
        canonical_options=canonical,
        canonical_tuning=tuning,
    )


def validate_observer_events(
    events: Sequence[Mapping[str, Any]], shape: Shape, boundary: str, identity: ObserverIdentity
) -> dict[str, object]:
    if len(events) != len(OBSERVER_PHASES) or tuple(event["phase"] for event in events) != OBSERVER_PHASES:
        raise AssertionError("GPU compilation must emit exactly three ordered observer phases")
    if len({event["invocation_id"] for event in events}) != 1:
        raise AssertionError("GPU observer phases do not share one invocation ID")
    for event in events:
        if event["policy"] != identity.policy or event["policy_digest"] != identity.policy_digest:
            raise AssertionError("GPU observer policy identity differs from compiler options")
        if event["tuning_digest"] != identity.tuning_digest or event["failure_pass"] != "":
            raise AssertionError("GPU observer tuning or success identity changed")
    algebra, generated, final = events
    if not algebra["coverage_manifest"] or algebra["coverage_manifest"] != generated["coverage_manifest"]:
        raise AssertionError("GPU coverage manifest is missing or changed before bundle generation")
    if algebra["region_membership"] != generated["region_membership"]:
        raise AssertionError("GPU selected-region membership changed before bundle generation")
    if algebra["unsupported_fingerprint"] != UNSUPPORTED_FINGERPRINT:
        raise AssertionError("GPU algebra phase contains unsupported structural islands")
    if generated["unsupported_fingerprint"] != UNSUPPORTED_FINGERPRINT:
        raise AssertionError("GPU generated phase contains unsupported structural islands")
    expected_generated, expected_final = OBSERVER_FINGERPRINTS[(shape.shape_id, boundary, identity.policy)]
    if generated["normalized_module_fingerprint"] != expected_generated:
        raise AssertionError("GPU generated-bundle fingerprint changed")
    if generated["no_shuttle_semantics"] is not False:
        raise AssertionError("GPU generated phase erased the bundle early")
    if (
        final["region_membership"] != ""
        or final["coverage_manifest"] != ""
        or final["unsupported_fingerprint"] != ""
        or final["normalized_module_fingerprint"] != expected_final
        or final["no_shuttle_semantics"] is not True
    ):
        raise AssertionError("GPU final-erasure evidence changed")
    return {
        "invocation_id": algebra["invocation_id"],
        "coverage_manifest_sha256": hashlib.sha256(algebra["coverage_manifest"].encode()).hexdigest(),
        "generated_bundle_fingerprint": expected_generated,
        "final_fingerprint": expected_final,
    }


def qualify_outputs(
    actual: Sequence[np.ndarray], ordinary: Sequence[np.ndarray], shape: Shape, boundary: str
) -> list[dict[str, object]]:
    references = independent_reference(shape)
    reference_by_name = dict(zip(("y", "dx", "dgamma", "dbeta"), references, strict=True))
    reports = []
    for name, value, baseline in zip(output_names(boundary), actual, ordinary, strict=True):
        if value.dtype != np.dtype("bfloat16") or value.shape != baseline.shape:
            raise AssertionError(f"{name}: Shuttle output shape or dtype changed")
        reference = reference_by_name[name]
        absolute = np.abs(value.astype(np.float64) - reference)
        ordinary_absolute = np.abs(baseline.astype(np.float64) - reference)
        maximum_limit, mean_limit = NUMERICAL_LIMITS[shape.shape_id][name]
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(baseline)):
            raise AssertionError(f"{name}: output contains NaN or infinity")
        if float(np.max(absolute)) > maximum_limit or float(np.mean(absolute)) > mean_limit:
            raise AssertionError(f"{name}: Shuttle output exceeds the independent-oracle limits")
        if float(np.max(ordinary_absolute)) > maximum_limit or float(np.mean(ordinary_absolute)) > mean_limit:
            raise AssertionError(f"{name}: ordinary JAX exceeds the independent-oracle limits")
        reports.append(
            {
                "name": name,
                "shape": list(value.shape),
                "raw_bfloat16_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "ordinary_jax_raw_bfloat16_sha256": hashlib.sha256(baseline.tobytes(order="C")).hexdigest(),
                "maximum_absolute_error": float(np.max(absolute)),
                "mean_absolute_error": float(np.mean(absolute)),
                "shuttle_vs_ordinary_maximum_absolute_error": float(
                    np.max(np.abs(value.astype(np.float64) - baseline.astype(np.float64)))
                ),
            }
        )
    return reports


def _baseline_key(shape: Shape, boundary: str, result: int) -> str:
    return f"{shape.shape_id}_{boundary}_{result}"


def save_baseline(path: Path) -> dict[str, object]:
    configure_cache(None)
    stored = {}
    for shape in SHAPES:
        for boundary in BOUNDARIES:
            values = arrays(jax.jit(boundary_function(boundary))(*boundary_inputs(shape, boundary)))
            for result, value in enumerate(values):
                if value.dtype != np.dtype("bfloat16"):
                    raise AssertionError(f"{shape.shape_id}/{boundary}: ordinary JAX dtype changed")
                stored[_baseline_key(shape, boundary, result)] = value.view(np.uint16)
    np.savez(path, **stored)
    return {"worker": "baseline", "pid": os.getpid(), "cell_count": len(SHAPES) * len(BOUNDARIES)}


def load_baseline(path: Path, shape: Shape, boundary: str) -> tuple[np.ndarray, ...]:
    names = output_names(boundary)
    expected_shapes = {
        "y": (shape.rows, shape.columns),
        "dx": (shape.rows, shape.columns),
        "dgamma": (shape.columns,),
        "dbeta": (shape.columns,),
    }
    with np.load(path) as stored:
        bits = tuple(stored[_baseline_key(shape, boundary, result)].copy() for result in range(len(names)))
    values = []
    for name, value in zip(names, bits, strict=True):
        if value.dtype != np.uint16 or value.shape != expected_shapes[name]:
            raise AssertionError(f"{shape.shape_id}/{boundary}: baseline bit payload changed")
        values.append(value.view(np.dtype("bfloat16")))
    return tuple(values)


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib was not built with the Shuttle observer bridge")
    return bridge


def run_subject(worker: str, baseline_path: Path, cache_directory: Path, key_map_path: Path) -> dict[str, object]:
    configure_cache(cache_directory)
    initial = cache_snapshot(cache_directory)
    if worker == "populate" and initial:
        raise AssertionError("ABI 11 populate cache must start empty")
    expected_keys = json.loads(key_map_path.read_text()) if worker == "reuse" else {}
    expected_labels = {f"{shape_id}_{boundary}_{policy}" for shape_id, boundary, policy in cell_identities()}
    if worker == "reuse" and set(expected_keys) != expected_labels:
        raise AssertionError("ABI 11 reuse key map does not cover the twelve exact cells")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    reports = []
    keys = {}
    for shape in SHAPES:
        for numerics in POLICIES:
            for boundary in BOUNDARIES:
                label = f"{shape.shape_id}_{boundary}_{numerics.value}"
                before = cache_snapshot(cache_directory)
                hits = counter.count
                compiled = jax.jit(boundary_function(boundary), compiler_options=subject_options(numerics))
                with _bridge().subscribe() as capture:
                    actual = arrays(compiled(*boundary_inputs(shape, boundary)))
                    events = decode_native_snapshot(capture.snapshot())
                ordinary = load_baseline(baseline_path, shape, boundary)
                numerical = qualify_outputs(actual, ordinary, shape, boundary)
                after = cache_snapshot(cache_directory)
                added = set(after) - set(before)
                if worker == "populate":
                    if len(added) != 1 or counter.count != hits:
                        raise AssertionError(f"{label}: expected one ABI 11 cache miss")
                    keys[label] = added.pop()
                    observer = validate_observer_events(events, shape, boundary, expected_identity(numerics))
                elif after != before or counter.count != hits + 1:
                    raise AssertionError(f"{label}: expected one immutable ABI 11 cache hit")
                else:
                    if events:
                        raise AssertionError(f"{label}: persistent-cache hit invoked Shuttle")
                    keys[label] = expected_keys.get(label)
                    if keys[label] not in after:
                        raise AssertionError(f"{label}: attributed cache entry is missing")
                    observer = None
                reports.append(
                    {
                        "shape_id": shape.shape_id,
                        "boundary": boundary,
                        "numerics": numerics.value,
                        "cache_file": keys[label],
                        "observer": observer,
                        "outputs": numerical,
                    }
                )
    final = cache_snapshot(cache_directory)
    if len(final) != len(cell_identities()) or len(set(keys.values())) != len(cell_identities()):
        raise AssertionError("ABI 11 proof requires twelve distinct persistent-cache entries")
    if worker == "populate":
        key_map_path.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n")
    elif final != initial:
        raise AssertionError("ABI 11 reuse changed persistent-cache bytes")
    return {
        "worker": worker,
        "pid": os.getpid(),
        "pipeline_abi_version": PIPELINE_ABI_VERSION,
        "cache_hits": counter.count,
        "cache_files": sorted(final),
        "cell_to_cache_file": keys,
        "cells": reports,
    }


def run_worker(arguments: argparse.Namespace) -> dict[str, object]:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError("LayerNorm GPU proof requires exact JAX/jaxlib 0.10.1")
    if jax.default_backend() != "gpu":
        raise RuntimeError("LayerNorm GPU proof requires the CUDA backend")
    if arguments.worker == "baseline":
        return save_baseline(arguments.baseline)
    return run_subject(arguments.worker, arguments.baseline, arguments.cache_directory, arguments.key_map)


def run_orchestrator(work_directory: Path) -> dict[str, object]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise RuntimeError("LayerNorm GPU work directory must start empty")
    baseline = work_directory / "disabled-baseline.npz"
    cache_directory = work_directory / "persistent-cache"
    key_map = work_directory / "cache-keys.json"
    reports = {}
    for worker in WORKERS:
        report = work_directory / f"{worker}.json"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                worker,
                "--baseline",
                str(baseline),
                "--cache-directory",
                str(cache_directory),
                "--key-map",
                str(key_map),
                "--report",
                str(report),
            ],
            check=True,
            env=environment,
        )
        reports[worker] = json.loads(report.read_text())
    if len({reports[worker]["pid"] for worker in WORKERS}) != len(WORKERS):
        raise AssertionError("baseline, populate, and reuse must run in distinct processes")
    if reports["populate"]["cell_to_cache_file"] != reports["reuse"]["cell_to_cache_file"]:
        raise AssertionError("reuse process changed ABI 11 cache-key attribution")
    if reports["reuse"]["cache_hits"] != len(cell_identities()):
        raise AssertionError("reuse process omitted an ABI 11 cache hit")
    return {
        "classification": "untimed_generic_gpu_layer_norm_correctness_and_cache_only",
        "timing_collected": False,
        "scorecard_status_changed": False,
        "workers": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--key-map", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.baseline is None:
            parser.error("worker mode requires --baseline")
        if arguments.worker != "baseline" and (arguments.cache_directory is None or arguments.key_map is None):
            parser.error("cache workers require --cache-directory and --key-map")
        result = run_worker(arguments)
    else:
        if arguments.work_directory is None:
            parser.error("orchestrator mode requires --work-directory")
        result = run_orchestrator(arguments.work_directory)
    arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
