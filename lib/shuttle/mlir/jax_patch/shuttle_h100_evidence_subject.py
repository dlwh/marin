# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect untimed ordinary-JAX ABI 10 H100 forward evidence as raw files."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from concurrent import futures
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from acceptance_contract import decode_native_snapshot
from jaxlib import _jax
from shuttle_jaxlib_target1_acceptance import CacheHitCounter, acceptance_tuning, cache_snapshot, configure_cache
from target1_abi10_h100_evidence import load_authorization, write_artifact_manifest
from target1_abi10_h100_forward_contract import CANONICAL_OPTIONS
from target1_numerical_oracle import fixed_inputs, independent_reference

from shuttle import ExecutionMode, Numerics, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
WORKERS = ("baseline", "populate", "reuse")


def row_fold_scale(x: jax.Array, gamma: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(local * local, axis=-1, keepdims=True) + 1e-5)
    return (local * inverse * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def subject_options() -> dict[str, object]:
    options = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=acceptance_tuning(),
    )
    if options.get("xla_shuttle_options") != CANONICAL_OPTIONS:
        raise AssertionError("GPU subject compiler options differ from the sealed ABI 10 identity")
    return options


def _ready(value: Any) -> np.ndarray:
    return np.asarray(value.block_until_ready())


def _write_bfloat16(path: Path, value: np.ndarray, name: str) -> None:
    if value.dtype != np.dtype("bfloat16") or value.shape != (2048, 4096):
        raise AssertionError(f"{name} shape or dtype differs from the closed forward cell")
    path.write_bytes(value.tobytes(order="C"))


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib was not built with the Shuttle observer bridge")
    return bridge


def _report(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_files() -> set[Path]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("H100 evidence requires Linux /proc/self/maps")
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)")).resolve()
            if path.is_file() and not path.is_symlink():
                paths.add(path)
    return paths


def _one_mapped(mapped: set[Path], name: str) -> Path:
    matches = [path for path in mapped if path.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one mapped {name}")
    return matches[0]


def _distribution_record(distribution: importlib.metadata.Distribution) -> Path:
    files = distribution.files
    if files is None:
        raise RuntimeError("CUDA plugin distribution file roster is missing")
    records = [file for file in files if file.name == "RECORD" and ".dist-info" in file.as_posix()]
    if len(records) != 1:
        raise RuntimeError("CUDA plugin distribution has no unique RECORD")
    return Path(distribution.locate_file(records[0])).resolve()


def _runtime_binary_identity() -> dict[str, str]:
    mapped = _mapped_files()
    extension = Path(_jax.__file__).resolve()
    if extension not in mapped:
        raise RuntimeError("jaxlib._jax module file is not mapped")
    plugin_package = sys.modules.get("jax_plugins.cuda")
    plugin_module = getattr(plugin_package, "cuda_plugin_extension", None)
    plugin_file = getattr(plugin_module, "__file__", None)
    if type(plugin_file) is not str:
        raise RuntimeError("could not identify the loaded CUDA plugin extension")
    plugin = Path(plugin_file).resolve()
    if plugin.is_symlink() or not plugin.is_file():
        raise RuntimeError("loaded CUDA plugin extension is not a regular file")
    if plugin not in mapped:
        raise RuntimeError("CUDA plugin extension module file is not mapped")
    common = _one_mapped(mapped, "libjax_common.so")
    pjrt = _one_mapped(mapped, "xla_cuda_plugin.so")
    plugin_distribution = importlib.metadata.distribution("jax-cuda13-plugin")
    pjrt_distribution = importlib.metadata.distribution("jax-cuda13-pjrt")
    if plugin_distribution.version != JAXLIB_VERSION or pjrt_distribution.version != JAXLIB_VERSION:
        raise RuntimeError("CUDA plugin distributions differ from exact JAX 0.10.1")
    plugin_record = _distribution_record(plugin_distribution)
    pjrt_record = _distribution_record(pjrt_distribution)
    if not plugin_record.is_file() or not pjrt_record.is_file():
        raise RuntimeError("CUDA plugin distribution RECORD is missing")
    return {
        "jax_extension_sha256": _sha256(extension),
        "jax_common_sha256": _sha256(common),
        "cuda_plugin_extension_sha256": _sha256(plugin),
        "cuda_pjrt_plugin_sha256": _sha256(pjrt),
        "cuda_plugin_record_sha256": _sha256(plugin_record),
        "cuda_pjrt_record_sha256": _sha256(pjrt_record),
    }


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    values = fixed_inputs(2048, 4096, "forward")
    if len(values) != 2:
        raise AssertionError("closed forward input arity changed")
    return values


def run_baseline(output: Path) -> dict[str, object]:
    configure_cache(None)
    x, gamma = _inputs()
    reference = independent_reference("forward", (x, gamma))[0]
    ordinary = _ready(jax.jit(row_fold_scale)(x, gamma))
    _write_bfloat16(output / "input_x", x, "input_x")
    if gamma.dtype != np.dtype("bfloat16") or gamma.shape != (4096,):
        raise AssertionError("input_gamma shape or dtype changed")
    (output / "input_gamma").write_bytes(gamma.tobytes(order="C"))
    _write_bfloat16(output / "reference_output", reference, "reference_output")
    _write_bfloat16(output / "ordinary_jax_output", ordinary, "ordinary_jax_output")
    return {
        "worker": "baseline",
        "pid": os.getpid(),
        "observer_events": 0,
        "runtime_binaries": _runtime_binary_identity(),
    }


def _compile_and_run(x: np.ndarray, gamma: np.ndarray):
    compiled = jax.jit(row_fold_scale, compiler_options=subject_options())
    return compiled, _ready(compiled(x, gamma))


def run_populate(output: Path, cache_directory: Path) -> dict[str, object]:
    configure_cache(cache_directory)
    if cache_snapshot(cache_directory):
        raise AssertionError("GPU cache population must start empty")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    x, gamma = _inputs()
    bridge = _bridge()
    with bridge.subscribe() as capture:
        compiled, first = _compile_and_run(x, gamma)
        sequential = [first, _ready(compiled(x, gamma)), _ready(compiled(x, gamma))]
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_outputs = list(executor.map(lambda _: _ready(compiled(x, gamma)), range(2)))
        snapshot = capture.snapshot()
        events = decode_native_snapshot(snapshot)
    after = cache_snapshot(cache_directory)
    if counter.count != 0 or len(after) != 1:
        raise AssertionError("GPU population must compile once and add one public cache entry")
    for index, value in enumerate(sequential):
        _write_bfloat16(output / f"shuttle_sequential_{index}", value, f"sequential {index}")
    for index, value in enumerate(concurrent_outputs):
        _write_bfloat16(output / f"shuttle_concurrent_{index}", value, f"concurrent {index}")
    _report(output / "observer_populate", list(snapshot))
    cache_name, (cache_size, cache_sha256) = next(iter(after.items()))
    shutil.copyfile(cache_directory / cache_name, output / "cache_entry")
    return {
        "worker": "populate",
        "pid": os.getpid(),
        "cache_hits": counter.count,
        "cache_name": cache_name,
        "cache_size": cache_size,
        "cache_sha256": cache_sha256,
        "observer_event_count": len(events),
        "runtime_binaries": _runtime_binary_identity(),
    }


def run_reuse(output: Path, cache_directory: Path) -> dict[str, object]:
    configure_cache(cache_directory)
    before = cache_snapshot(cache_directory)
    if len(before) != 1:
        raise AssertionError("GPU reuse requires exactly one populated cache entry")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    x, gamma = _inputs()
    with _bridge().subscribe() as capture:
        _, value = _compile_and_run(x, gamma)
        snapshot = capture.snapshot()
        events = decode_native_snapshot(snapshot)
    after = cache_snapshot(cache_directory)
    if counter.count != 1 or events or after != before:
        raise AssertionError("GPU reuse must hit once without observer events or cache mutation")
    _write_bfloat16(output / "shuttle_reuse_check", value, "reuse output")
    _report(output / "observer_reuse", list(snapshot))
    cache_name, (cache_size, cache_sha256) = next(iter(after.items()))
    return {
        "worker": "reuse",
        "pid": os.getpid(),
        "cache_hits": counter.count,
        "cache_name": cache_name,
        "cache_size": cache_size,
        "cache_sha256": cache_sha256,
        "observer_event_count": len(events),
        "runtime_binaries": _runtime_binary_identity(),
    }


def run_worker(worker: str, output: Path, cache_directory: Path) -> dict[str, object]:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError("H100 evidence requires exact JAX/jaxlib 0.10.1")
    if jax.default_backend() != "gpu":
        raise RuntimeError("H100 evidence worker requires the CUDA backend")
    if worker == "baseline":
        return run_baseline(output)
    if worker == "populate":
        return run_populate(output, cache_directory)
    if worker == "reuse":
        return run_reuse(output, cache_directory)
    raise ValueError(f"unknown H100 evidence worker: {worker}")


def worker_command(
    worker: str,
    output: Path,
    cache_directory: Path,
    report_path: Path,
    authorization_sha256: str,
    challenge: str,
) -> list[str]:
    """Return the exact authorization-bound child-process command."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--authorization-sha256",
        authorization_sha256,
        "--challenge",
        challenge,
        "--worker",
        worker,
        "--output",
        str(output),
        "--cache-directory",
        str(cache_directory),
        "--report",
        str(report_path),
    ]


def run_orchestrator(
    output: Path,
    collector_job_id: str,
    role_paths_path: Path,
    authorization_sha256: str,
    challenge: str,
) -> dict[str, object]:
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError("subject output must be a prepopulated real directory")
    cache_directory = output / "persistent-cache"
    reports = {}
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    for worker in WORKERS:
        report_path = output / (f"cache_{worker}_report" if worker != "baseline" else "baseline_report")
        subprocess.run(
            worker_command(worker, output, cache_directory, report_path, authorization_sha256, challenge),
            check=True,
            env=environment,
        )
        reports[worker] = json.loads(report_path.read_text())
    if len({reports[worker]["pid"] for worker in WORKERS}) != len(WORKERS):
        raise AssertionError("baseline, populate, and reuse must run in distinct processes")
    if reports["populate"]["cache_sha256"] != reports["reuse"]["cache_sha256"]:
        raise AssertionError("reuse process changed the attributed cache entry")
    shutil.rmtree(cache_directory)
    role_paths = json.loads(role_paths_path.read_text())
    sealed = write_artifact_manifest(output, collector_job_id, role_paths, load_authorization())
    return {
        "schema_version": 1,
        "workers": reports,
        "timing_collected": False,
        "artifact_manifest_sha256": sealed.manifest_sha256,
        "external_upload_status": "not_started",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--collector-job-id")
    parser.add_argument("--role-paths", type=Path)
    arguments = parser.parse_args()
    authorization = load_authorization()
    if (
        arguments.authorization_sha256 != authorization.sha256
        or arguments.challenge != authorization.decoded()["challenge"]
    ):
        raise RuntimeError("subject argv differs from the sealed authorization")
    if arguments.worker is None:
        if arguments.collector_job_id is None or arguments.role_paths is None:
            parser.error("orchestrator mode requires --collector-job-id and --role-paths")
        result = run_orchestrator(
            arguments.output,
            arguments.collector_job_id,
            arguments.role_paths,
            arguments.authorization_sha256,
            arguments.challenge,
        )
    else:
        if arguments.cache_directory is None or arguments.report is None:
            parser.error("worker mode requires --cache-directory and --report")
        result = run_worker(arguments.worker, arguments.output, arguments.cache_directory)
    if arguments.report is not None:
        _report(arguments.report, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
