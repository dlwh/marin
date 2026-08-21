# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the versioned Transformer Engine/XLA RMSNorm oracle on one H100."""

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abi22_te_rmsnorm_oracle_v2 import (
    BOUNDARIES,
    HELDOUT_INPUT_DIGESTS,
    HELDOUT_PATTERN,
    INPUT_DIGESTS,
    JAX_REVISION,
    NUMERICAL_POLICY,
    PRODUCERS,
    QUALIFICATION_PATTERN,
    REPRESENTATIVE_SHAPE,
    SHAPES,
    XLA_REVISION,
    assemble_report,
    build_matched_acceptance,
    execute_plan,
    execute_qualification_plan,
    execution_plan,
    load_te_raw_cell,
    prepare_cases,
    prepare_heldout_case,
    qualification_plan,
    validate_build_manifest,
    validate_qualification_evidence,
    validate_runtime_manifest,
)
from track1_map_h100_prototype import BASE_IMAGE, canonical_json

SOURCE_MANIFEST_SCHEMA = "shuttle_track2_te_h100_source_manifest_v1"
SOURCE_MANIFEST_NAME = ".shuttle-track2-te-source-manifest.json"
RESULT_SCHEMA = "shuttle_track2_te_h100_batch_result_v1"
TE_SOURCE_URL = "https://github.com/NVIDIA/TransformerEngine.git"
TE_SOURCE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"
TE_SOURCE_TREE = "c722c6cd46cc59a0dbabadd846af3f45b15c1a95"
TE_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/c3/21/"
    "5e0ec562539798ffd375c218b8bd7612b387455ccc79ae14576f411c9609/"
    "transformer_engine_cu13-2.17.0-py3-none-manylinux_2_28_x86_64.whl"
)
TE_WHEEL_SHA256 = "1c8e6627cf02358201f513ab902c7b4f82f179214bd775edc97ce0f2001d82da"
TE_WHEEL_LIBRARY_MEMBER = "transformer_engine/wheel_lib/libtransformer_engine.so"
SHUTTLE_BLOCKER = (
    "reviewed_dbff57e3_rmsnorm_canary_uses_epsilon_1e-6_but_does_not_admit_"
    "representative_2048x4096_without_new_harness_authorization"
)
RUNTIME_PYTHON = Path("/opt/h100-evidence-runtime/bin/python")
RUNTIME_SITE_PACKAGES = Path("/opt/h100-evidence-runtime/lib/python3.12/site-packages")
CUDA_ROOT = Path("/usr/local/cuda-13.2")
NVIDIA_SITE_PACKAGES = RUNTIME_SITE_PACKAGES / "nvidia"
NVIDIA_CU13_ROOT = NVIDIA_SITE_PACKAGES / "cu13"
NVIDIA_CU13_LIB = NVIDIA_CU13_ROOT / "lib"
NVIDIA_CUDNN_ROOT = NVIDIA_SITE_PACKAGES / "cudnn"
NVIDIA_NCCL_ROOT = NVIDIA_SITE_PACKAGES / "nccl"
NVTE_CUDA_INCLUDE_DIR = NVIDIA_CU13_ROOT / "include"
RUNTIME_LIBRARY_PATHS = {
    "cublas": NVIDIA_CU13_LIB / "libcublas.so.13",
    "cublas_lt": NVIDIA_CU13_LIB / "libcublasLt.so.13",
    "cudnn": NVIDIA_CUDNN_ROOT / "lib/libcudnn.so.9",
    "nccl": NVIDIA_NCCL_ROOT / "lib/libnccl.so.2",
    "cudart": NVIDIA_CU13_LIB / "libcudart.so.13",
    "nvrtc": NVIDIA_CU13_LIB / "libnvrtc.so.13",
}
RUNNER_SOURCE = Path("lib/shuttle/mlir/jax_patch/abi22_te_rmsnorm_oracle_runner_v2.cpp")
XLA_RUNNER = Path("lib/shuttle/mlir/jax_patch/abi22_xla_rmsnorm_oracle_v2.py")
BUILD_MANIFEST = Path("lib/shuttle/mlir/jax_patch/abi22-te-rmsnorm-oracle-runner-build-v2.json")
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BatchPaths:
    workspace: Path
    te_source: Path
    te_wheel: Path
    te_runtime: Path
    te_library: Path
    te_runner: Path
    cases: Path
    raw: Path
    reports: Path
    logs: Path
    te_report: Path
    xla_report: Path
    acceptance: Path
    runtime_manifest: Path
    result: Path

    @classmethod
    def under(cls, workspace: Path) -> "BatchPaths":
        runtime = workspace / "te-runtime"
        reports = workspace / "reports"
        return cls(
            workspace=workspace,
            te_source=workspace / "transformer-engine",
            te_wheel=workspace / "tools/transformer_engine_cu13-2.17.0.whl",
            te_runtime=runtime,
            te_library=runtime / TE_WHEEL_LIBRARY_MEMBER,
            te_runner=workspace / "abi22_te_rmsnorm_oracle_runner_v2",
            cases=workspace / "cases",
            raw=workspace / "raw",
            reports=reports,
            logs=workspace / "logs",
            te_report=reports / "transformer-engine-2.17.json",
            xla_report=reports / "ordinary-jax-xla.json",
            acceptance=reports / "matched-acceptance.json",
            runtime_manifest=reports / "te-runtime-manifest.json",
            result=reports / "result.json",
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one regular file")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one JSON object")
    return value


def validate_source_manifest(runner_repository: Path, manifest_path: Path) -> dict[str, str]:
    """Validate every reviewed source byte extracted by Iris."""
    raw = manifest_path.read_bytes()
    manifest = _read_json(manifest_path, "source manifest")
    if set(manifest) != {"commit", "entries", "schema", "tree"}:
        raise RuntimeError("source manifest fields differ")
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise RuntimeError("source manifest schema differs")
    commit = manifest.get("commit")
    tree = manifest.get("tree")
    entries = manifest.get("entries")
    if type(commit) is not str or GIT_COMMIT.fullmatch(commit) is None:
        raise RuntimeError("source manifest commit is invalid")
    if type(tree) is not str or GIT_COMMIT.fullmatch(tree) is None:
        raise RuntimeError("source manifest tree is invalid")
    if type(entries) is not list or not entries:
        raise RuntimeError("source manifest entries are invalid")
    seen = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256", "size"}:
            raise RuntimeError("source manifest entry is invalid")
        relative = entry["path"]
        digest = entry["sha256"]
        size = entry["size"]
        if (
            type(relative) is not str
            or relative in seen
            or relative.startswith("/")
            or any(part in ("", ".", "..") for part in Path(relative).parts)
            or type(digest) is not str
            or SHA256.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError("source manifest entry is invalid")
        path = runner_repository / relative
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RuntimeError(f"source differs from its manifest: {relative}")
        if metadata.st_size != size or _file_sha256(path) != digest:
            raise RuntimeError(f"source differs from its manifest: {relative}")
        seen.add(relative)
    return {
        "commit": commit,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "tree": tree,
    }


def batch_plan(*, workspace: Path, runner_repository: Path) -> dict[str, Any]:
    """Return the exact external phase plan without executing it."""
    paths = BatchPaths.under(workspace)
    library_directory = paths.te_library.parent
    compile_argv = [
        "/usr/bin/g++",
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-isystem",
        str(paths.te_source / "transformer_engine/common/include"),
        f"-I{CUDA_ROOT / 'include'}",
        str(runner_repository / RUNNER_SOURCE),
        f"-L{library_directory}",
        f"-Wl,-rpath,{library_directory}",
        "-ltransformer_engine",
        f"-L{CUDA_ROOT / 'lib64'}",
        f"-Wl,-rpath,{CUDA_ROOT / 'lib64'}",
        "-lcudart",
        "-Wl,--no-as-needed",
        *map(str, RUNTIME_LIBRARY_PATHS.values()),
        "-Wl,--as-needed",
        "-Wl,--allow-shlib-undefined",
        "-o",
        str(paths.te_runner),
    ]
    return {
        "base_image": BASE_IMAGE,
        "hardware": {"gpu_count": 1, "gpu_model": "H100"},
        "oracle": {
            "schema": "shuttle_abi22_te_xla_rmsnorm_plan_v2",
            "epsilon": 1e-6,
            "shapes": [list(shape) for shape in SHAPES],
            "representative_shape": [2048, 4096],
            "boundaries": list(BOUNDARIES),
            "producer_order": ["transformer_engine_2_17", "ordinary_jax_xla"],
            "run_count": len(SHAPES) * len(BOUNDARIES) * 2,
        },
        "qualification": {**NUMERICAL_POLICY, "run_count": 24},
        "phases": [
            {
                "name": "device",
                "argv": [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,compute_cap,driver_version",
                    "--format=csv,noheader",
                ],
            },
            {"name": "te_init", "argv": ["git", "init", "--quiet"]},
            {"name": "te_remote", "argv": ["git", "remote", "add", "origin", TE_SOURCE_URL]},
            {"name": "te_fetch", "argv": ["git", "fetch", "--depth=1", "origin", TE_SOURCE_COMMIT]},
            {"name": "te_checkout", "argv": ["git", "checkout", "--detach", "FETCH_HEAD"]},
            {
                "name": "te_wheel_fetch",
                "argv": [
                    "curl",
                    "--fail",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--output",
                    str(paths.te_wheel),
                    TE_WHEEL_URL,
                ],
            },
            {
                "name": "te_wheel_extract",
                "argv": [str(RUNTIME_PYTHON), "-m", "zipfile", "-e", str(paths.te_wheel), str(paths.te_runtime)],
            },
            {"name": "te_runner_build", "argv": compile_argv},
            {"name": "te_runner_link_check", "argv": ["ldd", str(paths.te_runner)]},
        ],
        "schema": "shuttle_track2_te_h100_batch_plan_v1",
        "runner_repository": str(runner_repository),
        "shuttle": {"status": "pending", "blocker": SHUTTLE_BLOCKER},
        "workspace": str(workspace),
    }


def _run(
    argv: list[str], *, cwd: Path | None, log: Path, phase: str, environment: Mapping[str, str] | None = None
) -> None:
    print(f"TRACK2_TE_H100_PHASE={phase}", flush=True)
    with log.open("wb") as stream:
        result = subprocess.run(argv, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode == 0:
        return
    tail = log.read_bytes()[-32 * 1024 :].decode(errors="replace")
    raise RuntimeError(f"phase {phase} failed with exit {result.returncode}\n{tail}")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _observed_te_source(source: Path) -> dict[str, str]:
    return {
        "commit": _git(source, "rev-parse", "HEAD^{commit}"),
        "tree": _git(source, "rev-parse", "HEAD^{tree}"),
        "status": _git(source, "status", "--porcelain", "--untracked-files=all"),
    }


def validate_te_hydration(paths: BatchPaths) -> dict[str, str]:
    """Bind the checkout, official wheel, and extracted shared library bytes."""
    observed = _observed_te_source(paths.te_source)
    if observed != {"commit": TE_SOURCE_COMMIT, "tree": TE_SOURCE_TREE, "status": ""}:
        raise RuntimeError("Transformer Engine source identity differs")
    version_path = paths.te_source / "build_tools/VERSION.txt"
    if not version_path.is_file() or version_path.read_text().strip() != "2.17.0":
        raise RuntimeError("Transformer Engine source version differs")
    if not paths.te_wheel.is_file() or paths.te_wheel.is_symlink():
        raise RuntimeError("Transformer Engine wheel is not a regular file")
    if _file_sha256(paths.te_wheel) != TE_WHEEL_SHA256:
        raise RuntimeError("Transformer Engine wheel SHA-256 differs")
    if not paths.te_library.is_file() or paths.te_library.is_symlink():
        raise RuntimeError("extracted Transformer Engine library is not a regular file")
    try:
        with zipfile.ZipFile(paths.te_wheel) as archive:
            members = [info for info in archive.infolist() if info.filename == TE_WHEEL_LIBRARY_MEMBER]
            if len(members) != 1 or members[0].is_dir():
                raise RuntimeError("official wheel lacks one Transformer Engine library member")
            wheel_library = archive.read(members[0])
    except zipfile.BadZipFile as error:
        raise RuntimeError("Transformer Engine wheel is not a ZIP archive") from error
    library_bytes = paths.te_library.read_bytes()
    if wheel_library != library_bytes:
        raise RuntimeError("extracted Transformer Engine library differs from the pinned wheel")
    return {
        "source_commit": TE_SOURCE_COMMIT,
        "source_tree": TE_SOURCE_TREE,
        "version": "2.17.0",
        "wheel_sha256": TE_WHEEL_SHA256,
        "library_path": str(paths.te_library),
        "library_sha256": hashlib.sha256(library_bytes).hexdigest(),
    }


def runtime_environment(paths: BatchPaths) -> dict[str, str]:
    """Return the runner environment with the reviewed TE library first."""
    search = [paths.te_library.parent, *(path.parent for path in RUNTIME_LIBRARY_PATHS.values())]
    nvidia = RUNTIME_SITE_PACKAGES / "nvidia"
    if nvidia.is_dir():
        search.extend(sorted(path for path in nvidia.glob("*/lib") if path.is_dir()))
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        search.extend(Path(value) for value in existing.split(":") if value)
    unique = list(dict.fromkeys(str(path) for path in search))
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("LD_BIND_NOW", None)
    environment["LD_LIBRARY_PATH"] = ":".join(unique)
    environment["NVTE_CUDA_INCLUDE_DIR"] = str(NVTE_CUDA_INCLUDE_DIR)
    return environment


def _elf_soname(path: Path) -> str:
    result = subprocess.run(["readelf", "-d", path], check=True, capture_output=True, text=True)
    matches = re.findall(r"\(SONAME\).*Library soname: \[([^]]+)\]", result.stdout)
    if len(matches) != 1:
        raise RuntimeError(f"runtime library has no unique ELF SONAME: {path}")
    return matches[0]


def _validated_runtime_libraries(
    library_paths: Mapping[str, Path],
    *,
    approved_roots: tuple[Path, ...],
    soname_reader: Callable[[Path], str],
) -> dict[str, dict[str, str]]:
    identities = {}
    resolved_roots = tuple(root.resolve(strict=True) for root in approved_roots)
    for role, requested in library_paths.items():
        try:
            resolved = requested.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"runtime library is missing: {role}") from error
        if not resolved.is_file() or resolved.is_symlink():
            raise RuntimeError(f"runtime library is not one regular file: {role}")
        if not any(resolved.is_relative_to(root) for root in resolved_roots):
            raise RuntimeError(f"runtime library is outside an approved runtime root: {role}")
        soname = soname_reader(resolved)
        if not soname or "/" in soname:
            raise RuntimeError(f"runtime library SONAME is invalid: {role}")
        identities[role] = {
            "requested_path": str(requested),
            "resolved_path": str(resolved),
            "sha256": _file_sha256(resolved),
            "soname": soname,
        }
    return identities


def _validated_cuda_include(include_directory: Path, *, approved_root: Path) -> dict[str, str]:
    try:
        resolved_directory = include_directory.resolve(strict=True)
        resolved_root = approved_root.resolve(strict=True)
        header = (resolved_directory / "cuda_runtime_api.h").resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError("NVRTC CUDA runtime headers are missing") from error
    if not resolved_directory.is_dir() or not resolved_directory.is_relative_to(resolved_root):
        raise RuntimeError("NVRTC CUDA include directory is outside the approved runtime root")
    if not header.is_file() or header.is_symlink() or not header.is_relative_to(resolved_root):
        raise RuntimeError("NVRTC CUDA header is outside the approved runtime root")
    return {
        "include_directory": str(include_directory),
        "header_path": str(header),
        "header_sha256": _file_sha256(header),
    }


def _validate_link_closure(log: Path, libraries: Mapping[str, Mapping[str, str]]) -> None:
    output = log.read_text()
    failed = "not found" in output or "libtransformer_engine.so" not in output
    for identity in libraries.values():
        soname = identity["soname"]
        matches = re.findall(rf"^\s*{re.escape(soname)}\s+=>\s+(\S+)\s+\(", output, flags=re.MULTILINE)
        if len(matches) != 1:
            failed = True
            continue
        loaded = Path(matches[0]).resolve()
        if str(loaded) != identity["resolved_path"] or _file_sha256(loaded) != identity["sha256"]:
            failed = True
    if failed:
        raise RuntimeError(f"Transformer Engine runtime dependency closure is incomplete\n{output[-32 * 1024 :]}")


def _invoke_oracle(paths: BatchPaths, argv: list[str]) -> int:
    return subprocess.run(argv, cwd=paths.workspace, env=runtime_environment(paths), check=False).returncode


def _device_identity(log: Path) -> dict[str, Any]:
    rows = [row.strip() for row in log.read_text().splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("batch requires exactly one visible H100")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or "H100" not in fields[0]:
        raise RuntimeError("batch requires exactly one visible H100")
    model, uuid, compute_capability, driver = fields
    return {
        "device": {"model": model, "uuid": uuid, "compute_capability": compute_capability},
        "driver": driver,
    }


def _execute_hydration(paths: BatchPaths, plan: dict[str, Any]) -> dict[str, Any]:
    phases = plan["phases"]
    device_log = paths.logs / "device.log"
    _run(phases[0]["argv"], cwd=None, log=device_log, phase="device")
    observed = _device_identity(device_log)
    paths.te_source.mkdir()
    for phase in phases[1:5]:
        _run(phase["argv"], cwd=paths.te_source, log=paths.logs / f"{phase['name']}.log", phase=phase["name"])
    source = _observed_te_source(paths.te_source)
    if source != {"commit": TE_SOURCE_COMMIT, "tree": TE_SOURCE_TREE, "status": ""}:
        raise RuntimeError("Transformer Engine source identity differs")
    paths.te_wheel.parent.mkdir()
    _run(phases[5]["argv"], cwd=None, log=paths.logs / "te_wheel_fetch.log", phase="te_wheel_fetch")
    if _file_sha256(paths.te_wheel) != TE_WHEEL_SHA256:
        raise RuntimeError("Transformer Engine wheel SHA-256 differs")
    paths.te_runtime.mkdir()
    _run(phases[6]["argv"], cwd=None, log=paths.logs / "te_wheel_extract.log", phase="te_wheel_extract")
    runner_repository = Path(plan["runner_repository"])
    validate_build_manifest(runner_repository / BUILD_MANIFEST, runner_repository / RUNNER_SOURCE)
    cuda_runtime_headers = _validated_cuda_include(NVTE_CUDA_INCLUDE_DIR, approved_root=NVIDIA_CU13_ROOT)
    runtime_libraries = _validated_runtime_libraries(
        RUNTIME_LIBRARY_PATHS,
        approved_roots=(NVIDIA_CU13_ROOT, NVIDIA_CUDNN_ROOT, NVIDIA_NCCL_ROOT),
        soname_reader=_elf_soname,
    )
    _run(phases[7]["argv"], cwd=None, log=paths.logs / "te_runner_build.log", phase="te_runner_build")
    link_log = paths.logs / "te_runner_link_check.log"
    _run(
        phases[8]["argv"],
        cwd=None,
        log=link_log,
        phase="te_runner_link_check",
        environment=runtime_environment(paths),
    )
    _validate_link_closure(link_log, runtime_libraries)
    observed["transformer_engine"] = validate_te_hydration(paths)
    observed["runtime_libraries"] = runtime_libraries
    observed["cuda_runtime_headers"] = cuda_runtime_headers
    return observed


def _artifact_identities(paths: BatchPaths) -> dict[str, dict[str, int | str]]:
    files = [paths.te_report, paths.xla_report, paths.acceptance, paths.runtime_manifest]
    files.extend(sorted(path for path in paths.raw.iterdir() if path.is_file() and not path.is_symlink()))
    return {
        str(path.relative_to(paths.workspace)): {"sha256": _file_sha256(path), "size": path.stat().st_size}
        for path in files
    }


def _validated_xla_raw_cell(
    raw: dict[str, Any], *, expected_device: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell = dict(raw)
    runtime = cell.pop("observed_runtime", None)
    fields = {
        "backend",
        "device_count",
        "device_id",
        "device_kind",
        "jax_git_commit",
        "jax_version",
        "jaxlib_version",
        "platform_version",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != fields:
        raise ValueError("ordinary-JAX observed runtime fields drifted")
    if (
        runtime["backend"] != "gpu"
        or runtime["device_count"] != 1
        or runtime["device_id"] != 0
        or not isinstance(runtime["device_kind"], str)
        or "H100" not in runtime["device_kind"].upper()
        or "H100" not in expected_device.get("model", "").upper()
    ):
        raise ValueError("ordinary-JAX observed runtime is not the one visible H100")
    if (
        runtime["jax_version"] != "0.10.1"
        or runtime["jaxlib_version"] != "0.10.1"
        or runtime["jax_git_commit"] != JAX_REVISION
        or not isinstance(runtime["platform_version"], str)
        or not runtime["platform_version"]
    ):
        raise ValueError("ordinary-JAX observed runtime identity drifted")
    return cell, dict(runtime)


def execute_oracle(
    *,
    paths: BatchPaths,
    device: Mapping[str, str],
    invoke: Callable[[list[str]], int] | None = None,
    runner_repository: Path | None = None,
) -> dict[str, Any]:
    """Execute and validate the closed v2 producer matrix."""
    prepare_cases(paths.cases)
    prepare_heldout_case(paths.cases)
    runtime_manifest = {
        "schema": "shuttle_abi22_te_rmsnorm_runtime_v2",
        "runner_sha256": _file_sha256(paths.te_runner),
        "transformer_engine_library": {
            "path": str(paths.te_library),
            "sha256": _file_sha256(paths.te_library),
        },
    }
    paths.runtime_manifest.write_bytes(canonical_json(runtime_manifest))
    runtime_identity = validate_runtime_manifest(runtime_manifest, paths.te_runner)
    source_root = paths.workspace.parent if runner_repository is None else runner_repository
    plan = execution_plan(
        str(paths.te_runner),
        str(source_root / XLA_RUNNER),
        str(RUNTIME_PYTHON),
        paths.cases,
    )
    runner = invoke if invoke is not None else lambda argv: _invoke_oracle(paths, argv)
    execute_plan(plan, runner)
    policy_plan = qualification_plan(
        str(paths.te_runner),
        str(source_root / XLA_RUNNER),
        str(RUNTIME_PYTHON),
        paths.cases,
    )
    execute_qualification_plan(policy_plan, runner)
    te_cells = []
    xla_cells = []
    observed_xla_runtime = None
    for shape in SHAPES:
        rows, features = shape
        case = paths.cases / f"{rows}x{features}"
        for boundary in BOUNDARIES:
            te_raw = paths.raw / f"transformer_engine_2_17-{rows}x{features}-{boundary}.json"
            xla_raw = paths.raw / f"ordinary_jax_xla-{rows}x{features}-{boundary}.json"
            te_cells.append(load_te_raw_cell(te_raw, case))
            xla_cell, runtime = _validated_xla_raw_cell(
                _read_json(xla_raw, "ordinary-JAX/XLA raw cell"), expected_device=device
            )
            if observed_xla_runtime is None:
                observed_xla_runtime = runtime
            elif observed_xla_runtime != runtime:
                raise ValueError("ordinary-JAX observed runtime differs across cells")
            xla_cells.append(xla_cell)
    if observed_xla_runtime is None:
        raise AssertionError("ordinary-JAX matrix was empty")
    policy_cells: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    rows, features = REPRESENTATIVE_SHAPE
    for run in policy_plan["runs"]:
        producer = run["producer"]
        pattern = run["input_pattern"]
        boundary = run["boundary"]
        repeat = run["repeat"]
        case = paths.cases / (f"{rows}x{features}" if pattern == QUALIFICATION_PATTERN else HELDOUT_PATTERN)
        raw = paths.raw / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json"
        if producer == "transformer_engine_2_17":
            cell = load_te_raw_cell(
                raw,
                case,
                expected_input_digests=HELDOUT_INPUT_DIGESTS if pattern == HELDOUT_PATTERN else None,
            )
        else:
            cell, runtime = _validated_xla_raw_cell(
                _read_json(raw, "ordinary-JAX/XLA policy cell"), expected_device=device
            )
            if observed_xla_runtime != runtime:
                raise ValueError("ordinary-JAX observed runtime differs in policy qualification")
        input_digests = INPUT_DIGESTS[REPRESENTATIVE_SHAPE]
        if pattern == HELDOUT_PATTERN:
            input_digests = HELDOUT_INPUT_DIGESTS
        roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
        if cell["input_digests"] != {role: input_digests[role] for role in roles}:
            raise ValueError("policy input bytes differ from the predeclared pattern")
        cell.update(
            producer=producer,
            input_pattern=pattern,
            repeat=repeat,
            scope="representative_policy_qualification",
        )
        policy_cells[(producer, pattern, boundary, repeat)] = cell
    qualification = validate_qualification_evidence(
        {
            "schema": "shuttle_abi22_rmsnorm_policy_evidence_v1",
            "cells": [
                policy_cells[(producer, pattern, boundary, repeat)]
                for producer in PRODUCERS
                for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN)
                for boundary in BOUNDARIES
                for repeat in range(2)
            ],
        }
    )
    print(f"TRACK2_TE_POLICY_DIAGNOSTICS={canonical_json(qualification).decode()}")
    diagnostics = {
        producer: [{"shape": cell["shape"], "boundary": cell["boundary"], "outputs": cell["outputs"]} for cell in cells]
        for producer, cells in (
            ("transformer_engine_2_17", te_cells),
            ("ordinary_jax_xla", xla_cells),
        )
    }
    print(f"TRACK2_TE_NUMERICAL_DIAGNOSTICS={canonical_json(diagnostics).decode()}")
    te_report = assemble_report("transformer_engine_2_17", te_cells, runtime_identity, device)
    xla_identity = {
        "jax_version": observed_xla_runtime["jax_version"],
        "jaxlib_version": observed_xla_runtime["jaxlib_version"],
        "xla_revision": XLA_REVISION,
    }
    xla_report = assemble_report("ordinary_jax_xla", xla_cells, xla_identity, device)
    acceptance = build_matched_acceptance(
        te_report,
        xla_report,
        qualification=qualification,
        expected_te_identity=runtime_identity,
    )
    paths.te_report.write_bytes(canonical_json(te_report))
    paths.xla_report.write_bytes(canonical_json(xla_report))
    paths.acceptance.write_bytes(canonical_json(acceptance))
    return {
        "acceptance": acceptance,
        "artifacts": _artifact_identities(paths),
        "oracle_plan": {"base": dict(plan), "qualification": dict(policy_plan)},
        "observed_xla_runtime": observed_xla_runtime,
        "reports": {
            "transformer_engine_2_17": te_report,
            "ordinary_jax_xla": xla_report,
        },
        "runtime_identity": dict(runtime_identity),
        "shuttle": {"status": "pending", "blocker": SHUTTLE_BLOCKER},
    }


def execute_batch(
    *,
    workspace: Path,
    runner_repository: Path,
    manifest_path: Path,
    observed_base_image: str,
) -> dict[str, Any]:
    """Hydrate exact dependencies and execute both producers sequentially."""
    if observed_base_image != BASE_IMAGE:
        raise RuntimeError("observed base image differs from the sealed batch request")
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("batch workspace must be fresh")
    if platform.system() != "Linux":
        raise RuntimeError("Track 2 TE H100 batch requires Linux")
    source = validate_source_manifest(runner_repository, manifest_path)
    workspace.mkdir(mode=0o700)
    paths = BatchPaths.under(workspace)
    paths.logs.mkdir()
    paths.reports.mkdir()
    paths.raw.mkdir()
    plan = batch_plan(workspace=workspace, runner_repository=runner_repository)
    hydration = _execute_hydration(paths, plan)
    evidence = execute_oracle(paths=paths, device=hydration["device"], runner_repository=runner_repository)
    logs = {
        path.name: {"sha256": _file_sha256(path), "size": path.stat().st_size} for path in sorted(paths.logs.iterdir())
    }
    result = {
        "base_image": BASE_IMAGE,
        "device": {**hydration["device"], "driver": hydration["driver"]},
        "evidence": evidence,
        "schema": RESULT_SCHEMA,
        "source": source,
        "tools": {
            "cuda_runtime_headers": hydration["cuda_runtime_headers"],
            "logs": logs,
            "runner_binary_sha256": _file_sha256(paths.te_runner),
            "runtime_libraries": hydration["runtime_libraries"],
            "transformer_engine": hydration["transformer_engine"],
        },
    }
    paths.result.write_bytes(canonical_json(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--runner-repository", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--base-image", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.plan:
        validate_source_manifest(arguments.runner_repository, arguments.source_manifest)
        result = batch_plan(workspace=arguments.workspace, runner_repository=arguments.runner_repository)
    else:
        result = execute_batch(
            workspace=arguments.workspace,
            runner_repository=arguments.runner_repository,
            manifest_path=arguments.source_manifest,
            observed_base_image=arguments.base_image,
        )
    print("TRACK2_TE_H100_RESULT=" + canonical_json(result).decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
