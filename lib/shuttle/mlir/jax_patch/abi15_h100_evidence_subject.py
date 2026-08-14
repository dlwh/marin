# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect raw untimed ABI15 ordinary-JAX H100 evidence."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent import futures
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from abi15_h100_composed_contract import load_contract
from abi15_h100_evidence import (
    ProcessReport,
    expected_role_paths,
    load_role_manifest,
    require_sealed_hardware_authority,
    seal_raw_artifact,
    validate_process_reports,
)
from abi15_h100_numerics import (
    COLUMNS,
    REDUCTION,
    ROWS,
    Inputs,
    ReductionPolicy,
    dense_inputs,
    reference,
    sparse_policy_inputs,
)
from jaxlib import _jax
from shuttle_jaxlib_target1_acceptance import (
    CacheHitCounter,
    cache_snapshot,
    configure_cache,
)

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options

JAXLIB_VERSION = "0.10.1"
WORKERS = ("baseline", "populate", "reuse")
DATASETS = {"dense": dense_inputs, "sparse_policy_discriminator": sparse_policy_inputs}
OUTPUT_ROLES = ("primal", "lhs_cotangent", "rhs_cotangent")


def composed(lhs: jax.Array, rhs: jax.Array, output_cotangent: jax.Array) -> tuple[jax.Array, ...]:
    """Match the pinned ordinary-JAX composed fixture boundary."""

    def matmul(left: jax.Array, right: jax.Array) -> jax.Array:
        return (left.astype(jnp.float32) @ right.astype(jnp.float32)).astype(jnp.bfloat16)

    primal, pullback = jax.vjp(matmul, lhs, rhs)
    lhs_cotangent, rhs_cotangent = pullback(output_cotangent)
    return primal, lhs_cotangent, rhs_cotangent


def _tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def _options(policy: str) -> dict[str, object]:
    numerics = Numerics.SOURCE_ORDERED if policy == "source_ordered" else Numerics.FAST
    options = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=numerics,
        tuning=_tuning(),
    )
    contract = load_contract()
    expected = contract.source_ordered.options_sha256 if policy == "source_ordered" else contract.fast.options_sha256
    canonical = options.get("xla_shuttle_options")
    if type(canonical) is not str:
        raise RuntimeError("ABI15 compiler options are not canonical text")
    if hashlib.sha256(canonical.encode()).hexdigest() != expected:
        raise RuntimeError("ABI15 subject options differ from the immutable contract")
    return options


def _arrays(inputs: Inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def array(values: tuple[int, ...], shape: tuple[int, ...]) -> np.ndarray:
        raw = struct.pack(f"<{len(values)}H", *values)
        return np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(shape)

    return (
        array(inputs.lhs, (ROWS, REDUCTION)),
        array(inputs.rhs, (REDUCTION, COLUMNS)),
        array(inputs.output_cotangent, (ROWS, COLUMNS)),
    )


def _ready(value: tuple[jax.Array, ...]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(item.block_until_ready()) for item in value)


def _write_arrays(output: Path, roles: dict[str, str], prefix: str, values: tuple[np.ndarray, ...]) -> None:
    for role, value in zip(OUTPUT_ROLES, values, strict=True):
        if value.dtype != np.dtype(ml_dtypes.bfloat16):
            raise RuntimeError("ABI15 subject output dtype drift")
        path = output / roles[f"{prefix}_{role}"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.tobytes(order="C"))


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib lacks the Shuttle observer bridge")
    return bridge


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_observer(path: Path, snapshot: tuple[dict[str, object], ...] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in snapshot))


def _cache_record(snapshot: dict[str, tuple[int, str]]) -> list[dict[str, object]]:
    return [{"filename": name, "sha256": digest, "size": size} for name, (size, digest) in sorted(snapshot.items())]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_files() -> dict[Path, tuple[tuple[int, int, str, int], ...]]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("ABI15 H100 evidence requires Linux /proc/self/maps")
    paths: dict[Path, list[tuple[int, int, str, int]]] = {}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)")).resolve()
            if path.is_file() and not path.is_symlink():
                start_text, end_text = fields[0].split("-", maxsplit=1)
                paths.setdefault(path, []).append((int(start_text, 16), int(end_text, 16), fields[3], int(fields[4])))
    return {path: tuple(ranges) for path, ranges in paths.items()}


def _one_mapped(mapped: dict[Path, tuple[tuple[int, int, str, int], ...]], name: str) -> Path:
    matches = [path for path in mapped if path.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one mapped {name}")
    return matches[0]


def _mapping_record(
    role: str,
    path: Path,
    ranges: tuple[tuple[int, int, str, int], ...],
) -> tuple[str, str, str, int, int, int, str]:
    if path.is_symlink() or not path.is_file() or not ranges:
        raise RuntimeError(f"loaded {role} mapping is not one existing regular file")
    devices = {device for _, _, device, _ in ranges}
    inodes = {inode for _, _, _, inode in ranges}
    file_stat = path.stat()
    expected_device = f"{os.major(file_stat.st_dev):02x}:{os.minor(file_stat.st_dev):02x}"
    if devices != {expected_device} or inodes != {file_stat.st_ino}:
        raise RuntimeError(f"mapped device/inode identity differs from loaded {role} bytes")
    return (
        role,
        str(path),
        expected_device,
        file_stat.st_ino,
        min(start for start, _, _, _ in ranges),
        max(end for _, end, _, _ in ranges),
        _sha256_file(path),
    )


def _distribution_record(distribution: importlib.metadata.Distribution) -> Path:
    files = distribution.files
    if files is None:
        raise RuntimeError("CUDA plugin distribution file roster is missing")
    records = [file for file in files if file.name == "RECORD" and ".dist-info" in file.as_posix()]
    if len(records) != 1:
        raise RuntimeError("CUDA plugin distribution has no unique RECORD")
    return Path(distribution.locate_file(records[0])).resolve()


def _runtime_binary_identity() -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str, str, int, int, int, str], ...],
]:
    mapped = _mapped_files()
    extension = Path(_jax.__file__).resolve()
    plugin_package = sys.modules.get("jax_plugins.cuda")
    plugin_module = getattr(plugin_package, "cuda_plugin_extension", None)
    plugin_file = getattr(plugin_module, "__file__", None)
    if extension not in mapped or type(plugin_file) is not str:
        raise RuntimeError("could not identify exact mapped JAX/CUDA extension bytes")
    plugin = Path(plugin_file).resolve()
    if plugin.is_symlink() or not plugin.is_file() or plugin not in mapped:
        raise RuntimeError("loaded CUDA plugin extension is not one mapped regular file")
    plugin_distribution = importlib.metadata.distribution("jax-cuda13-plugin")
    pjrt_distribution = importlib.metadata.distribution("jax-cuda13-pjrt")
    if plugin_distribution.version != JAXLIB_VERSION or pjrt_distribution.version != JAXLIB_VERSION:
        raise RuntimeError("CUDA plugin distributions differ from exact JAX 0.10.1")
    paths = {
        "cuda_pjrt_plugin_sha256": _one_mapped(mapped, "xla_cuda_plugin.so"),
        "cuda_pjrt_record_sha256": _distribution_record(pjrt_distribution),
        "cuda_plugin_extension_sha256": plugin,
        "cuda_plugin_record_sha256": _distribution_record(plugin_distribution),
        "jax_common_sha256": _one_mapped(mapped, "libjax_common.so"),
        "jax_extension_sha256": extension,
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise RuntimeError("loaded runtime evidence contains a symlink or missing file")
    digests = tuple((name, _sha256_file(path)) for name, path in sorted(paths.items()))
    mapping_paths = {
        "cuda_pjrt_plugin": paths["cuda_pjrt_plugin_sha256"],
        "cuda_plugin_extension": paths["cuda_plugin_extension_sha256"],
        "jax_common": paths["jax_common_sha256"],
        "jax_extension": paths["jax_extension_sha256"],
    }
    mappings = []
    for role, path in sorted(mapping_paths.items()):
        mappings.append(_mapping_record(role, path, mapped[path]))
    return digests, tuple(mappings)


def _compile(policy: str) -> Any:
    return jax.jit(composed, compiler_options=_options(policy))


def _run_repeats(compiled: Any, arrays: tuple[np.ndarray, ...]) -> dict[str, tuple[np.ndarray, ...]]:
    first = _ready(compiled(*arrays))
    sequential = _ready(compiled(*arrays))
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        threaded = list(executor.map(lambda _: _ready(compiled(*arrays)), range(2)))
    return {
        "first": first,
        "sequential": sequential,
        "thread_0": threaded[0],
        "thread_1": threaded[1],
    }


def run_baseline(output: Path) -> ProcessReport:
    configure_cache(None)
    roles = expected_role_paths(load_role_manifest())
    compiled = jax.jit(composed)
    for dataset, factory in DATASETS.items():
        values = _ready(compiled(*_arrays(factory())))
        _write_arrays(output, roles, f"baseline_{dataset}", values)
    binaries, mappings = _runtime_binary_identity()
    return ProcessReport("baseline", os.getpid(), 0, 0, (), (), binaries, mappings)


def run_populate(output: Path, cache_directory: Path) -> ProcessReport:
    configure_cache(cache_directory)
    if cache_snapshot(cache_directory):
        raise RuntimeError("ABI15 populate cache must start empty")
    roles = expected_role_paths(load_role_manifest())
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    policy_entries = {}
    observer_events = 0
    for policy in ("source_ordered", "fast"):
        before = cache_snapshot(cache_directory)
        with _bridge().subscribe() as capture:
            compiled = _compile(policy)
            for dataset, factory in DATASETS.items():
                for run, values in _run_repeats(compiled, _arrays(factory())).items():
                    _write_arrays(output, roles, f"output_{policy}_{dataset}_{run}", values)
            snapshot = tuple(capture.snapshot())
        _write_observer(output / roles[f"observer_populate_{policy}"], snapshot)
        observer_events += len(snapshot)
        after = cache_snapshot(cache_directory)
        new_entries = set(after) - set(before)
        if len(new_entries) != 1:
            raise RuntimeError("each ABI15 policy must add one cache entry")
        name = new_entries.pop()
        size, digest = after[name]
        policy_entries[policy] = (name, size, digest)
        cache_output = output / roles[f"cache_entry_{policy}"]
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_directory / name, cache_output)
    after = cache_snapshot(cache_directory)
    if counter.count != 0 or len(after) != 2:
        raise RuntimeError("ABI15 populate must miss twice and create two entries")
    _write_json(output / roles["cache_populate_before"], [])
    _write_json(output / roles["cache_populate_after"], _cache_record(after))
    binaries, mappings = _runtime_binary_identity()
    return ProcessReport(
        "populate",
        os.getpid(),
        counter.count,
        observer_events,
        (),
        tuple((policy, *record) for policy, record in sorted(policy_entries.items())),
        binaries,
        mappings,
    )


def run_reuse(output: Path, cache_directory: Path) -> ProcessReport:
    configure_cache(cache_directory)
    before = cache_snapshot(cache_directory)
    if len(before) != 2:
        raise RuntimeError("ABI15 reuse requires exactly two cache entries")
    roles = expected_role_paths(load_role_manifest())
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    observed = 0
    for policy in ("source_ordered", "fast"):
        with _bridge().subscribe() as capture:
            compiled = _compile(policy)
            for dataset, factory in DATASETS.items():
                _write_arrays(
                    output,
                    roles,
                    f"output_{policy}_{dataset}_reuse",
                    _ready(compiled(*_arrays(factory()))),
                )
            snapshot = tuple(capture.snapshot())
        observed += len(snapshot)
        _write_observer(output / roles[f"observer_reuse_{policy}"], snapshot)
    after = cache_snapshot(cache_directory)
    if counter.count != 2 or observed != 0 or after != before:
        raise RuntimeError("ABI15 reuse must hit twice without observer or cache mutation")
    _write_json(output / roles["cache_reuse_before"], _cache_record(before))
    _write_json(output / roles["cache_reuse_after"], _cache_record(after))
    copied = _policy_cache_digests(output, roles)
    snapshot_by_digest = {digest: (name, size) for name, (size, digest) in before.items()}
    if len(snapshot_by_digest) != len(before) or set(copied.values()) != set(snapshot_by_digest):
        raise RuntimeError("ABI15 reuse cache bytes do not map uniquely to both policy entries")
    entries = tuple(
        sorted(
            (policy, snapshot_by_digest[digest][0], snapshot_by_digest[digest][1], digest)
            for policy, digest in copied.items()
        )
    )
    binaries, mappings = _runtime_binary_identity()
    return ProcessReport(
        "reuse",
        os.getpid(),
        counter.count,
        observed,
        entries,
        entries,
        binaries,
        mappings,
    )


def _policy_cache_digests(output: Path, roles: dict[str, str]) -> dict[str, str]:
    return {
        policy: hashlib.sha256((output / roles[f"cache_entry_{policy}"]).read_bytes()).hexdigest()
        for policy in ("source_ordered", "fast")
    }


def _report_value(report: ProcessReport) -> dict[str, object]:
    return {
        "cache_after": list(report.cache_after),
        "cache_before": list(report.cache_before),
        "cache_hits": report.cache_hits,
        "observer_events": report.observer_events,
        "pid": report.pid,
        "process": report.process,
        "runtime_binaries": dict(report.runtime_binaries),
        "runtime_mappings": [
            {
                "device": device,
                "end": end,
                "inode": inode,
                "path": path,
                "role": role,
                "sha256": sha256,
                "start": start,
            }
            for role, path, device, inode, start, end, sha256 in report.runtime_mappings
        ],
    }


def worker_command(
    worker: str,
    output: Path,
    cache_directory: Path,
    report: Path,
    authorization_sha256: str,
) -> list[str]:
    """Return the exact fresh-process worker argv."""
    launcher = Path(__file__).with_name("abi15_h100_evidence_launcher.py")
    return [
        sys.executable,
        str(launcher.resolve()),
        "--authorization-sha256",
        authorization_sha256,
        "--cache-directory",
        str(cache_directory),
        "--output",
        str(output),
        "--report",
        str(report),
        "--worker",
        worker,
    ]


def run_orchestrator(
    output: Path,
    build_artifact: Path,
    *,
    authorization_sha256: str,
    harness_commit: str,
    harness_tree: str,
) -> str:
    """Run three fresh workers and close one raw, never accepted, artifact."""
    require_sealed_hardware_authority()
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("ABI15 subject output must be a fresh real directory")
    if build_artifact.is_symlink() or not build_artifact.is_dir():
        raise RuntimeError("ABI15 build artifact must be a real directory")
    shutil.copytree(build_artifact, output, dirs_exist_ok=True)
    roles = expected_role_paths(load_role_manifest())
    contract_path = (
        Path(__file__).parents[4] / ".agents/projects/tile_lifetime_compiler/shuttle_abi15_h100_composed_prerun_v1.json"
    )
    fixture_path = Path(__file__).parents[1] / "test/Inputs/jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir"
    architecture_path = Path(__file__).with_name("abi15-h100-architecture-build-contract-v1.json")
    for role, source in (
        ("architecture_build_contract", architecture_path),
        ("prerun_contract", contract_path),
        ("fixture", fixture_path),
    ):
        destination = output / roles[role]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for dataset, factory in DATASETS.items():
        inputs = factory()
        for role, values in (
            ("lhs", inputs.lhs),
            ("rhs", inputs.rhs),
            ("output_cotangent", inputs.output_cotangent),
        ):
            path = output / roles[f"input_{dataset}_{role}"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(struct.pack(f"<{len(values)}H", *values))
        for policy in ReductionPolicy:
            expected = reference(inputs, policy)
            for role, values in zip(
                OUTPUT_ROLES,
                (expected.primal, expected.lhs_cotangent, expected.rhs_cotangent),
                strict=True,
            ):
                path = output / roles[f"reference_{policy.value}_{dataset}_{role}"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(struct.pack(f"<{len(values)}H", *values))
    cache_directory = output / ".persistent-cache"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    reports = []
    for worker in WORKERS:
        report_path = output / roles[f"{worker}_report"]
        subprocess.run(
            worker_command(worker, output, cache_directory, report_path, authorization_sha256),
            check=True,
            env=environment,
        )
        value = json.loads(report_path.read_text())
        reports.append(
            ProcessReport(
                value["process"],
                value["pid"],
                value["cache_hits"],
                value["observer_events"],
                tuple(tuple(item) for item in value["cache_before"]),
                tuple(tuple(item) for item in value["cache_after"]),
                tuple(sorted(value["runtime_binaries"].items())),
                tuple(
                    (
                        item["role"],
                        item["path"],
                        item["device"],
                        item["inode"],
                        item["start"],
                        item["end"],
                        item["sha256"],
                    )
                    for item in value["runtime_mappings"]
                ),
            )
        )
    validate_process_reports(tuple(reports))
    shutil.rmtree(cache_directory)
    if (output / "accepted_bundle.json").exists():
        raise RuntimeError("task subject must never create accepted_bundle.json")
    return seal_raw_artifact(
        output,
        roles,
        harness_commit=harness_commit,
        harness_tree=harness_tree,
    )
