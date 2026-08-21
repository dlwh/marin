# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the ABI22 Shuttle and ordinary-JAX H100 harnesses on one GPU."""

import argparse
import hashlib
import json
import platform
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from abi22_shuttle_rmsnorm_representative import validate_shuttle_evidence
from track1_map_h100_build import (
    HERMETIC_CUDA_COMPUTE_CAPABILITIES,
    XLA_COMMIT,
)
from track1_map_h100_prototype import (
    BASE_IMAGE,
    BAZEL_SHA256,
    BAZEL_URL,
    BAZEL_VERSION,
    PHYSICAL_CUDA_TOOLS_ROOT,
    canonical_json,
)

SOURCE_MANIFEST_SCHEMA = "shuttle_track2_h100_source_manifest_v1"
SOURCE_MANIFEST_NAME = ".shuttle-track2-source-manifest.json"
RESULT_SCHEMA = "shuttle_track2_h100_batch_result_v1"
REPRESENTATIVE_RESULT_SCHEMA = "shuttle_track2_rmsnorm_representative_batch_result_v1"
TRACK2_HERMETIC_CUDA_VERSION = "12.9.1"
XLA_WORKSPACE_PATCH = Path("lib/shuttle/mlir/jax_patch/track2_xla_workspace.patch")
RUNTIME_PYTHON = Path("/opt/h100-evidence-runtime/bin/python")
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BatchPaths:
    workspace: Path
    xla: Path
    bazel: Path
    bazel_output: Path
    reports: Path
    logs: Path
    xla_report: Path
    te_status: Path
    shuttle_report: Path
    shuttle_validated: Path
    exploratory_comparison: Path
    result: Path
    cases: Path
    raw: Path
    representative_report: Path

    @classmethod
    def under(cls, workspace: Path) -> "BatchPaths":
        reports = workspace / "reports"
        return cls(
            workspace=workspace,
            xla=workspace / "xla",
            bazel=workspace / "tools" / f"bazel-{BAZEL_VERSION}",
            bazel_output=workspace / "bazel-output",
            reports=reports,
            logs=workspace / "logs",
            xla_report=reports / "xla-host.json",
            te_status=reports / "te-status.json",
            shuttle_report=reports / "shuttle.json",
            shuttle_validated=reports / "shuttle.validated.json",
            exploratory_comparison=reports / "exploratory-host.json",
            result=reports / "result.json",
            cases=workspace / "cases",
            raw=workspace / "raw",
            representative_report=reports / "shuttle-representative.json",
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be one regular file")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not JSON") from error
    if type(value) is not dict:
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def validate_source_manifest(runner_repository: Path, manifest_path: Path) -> dict[str, str]:
    """Validate the complete reviewed source inventory extracted by Iris."""
    raw = manifest_path.read_bytes()
    manifest = _read_json(manifest_path, "source manifest")
    if set(manifest) != {"commit", "entries", "schema", "tree"} or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
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


def _common_bazel(paths: BatchPaths, runner_repository: Path, *, command: str = "test") -> list[str]:
    common = [
        str(paths.bazel),
        f"--output_user_root={paths.bazel_output}",
        command,
        f"--override_repository=shuttle_mlir={runner_repository / 'lib/shuttle/mlir'}",
        "--config=cuda",
        f"--repo_env=HERMETIC_CUDA_VERSION={TRACK2_HERMETIC_CUDA_VERSION}",
        "--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=" + ",".join(HERMETIC_CUDA_COMPUTE_CAPABILITIES),
        "--define=SHUTTLE_TEST_OBSERVER=1",
    ]
    if command == "test":
        common.extend(["--nocache_test_results", "--test_output=streamed", "--test_timeout=3600"])
    return common


def _test_env(name: str, value: object) -> str:
    return f"--test_env={name}={value}"


def _compiler_identity(source: dict[str, str]) -> str:
    return f"shuttle-abi22:{source['commit']}:{source['tree']}:{source['manifest_sha256']}"


def batch_plan(*, workspace: Path, runner_repository: Path, compiler_identity: str) -> dict[str, object]:
    """Return the exact same-device phase plan without executing it."""
    if not compiler_identity.startswith("shuttle-abi22:") or not compiler_identity.strip():
        raise ValueError("compiler identity is invalid")
    paths = BatchPaths.under(workspace)
    workspace_patch = runner_repository / XLA_WORKSPACE_PATCH
    wrappers = [
        runner_repository / "lib/shuttle/mlir/test/run-h100-xla-baseline.sh",
        runner_repository / "lib/shuttle/mlir/test/run-h100-performance.sh",
    ]
    common = _common_bazel(paths, runner_repository)
    xla = [
        *common,
        "@shuttle_mlir//:abi22_h100_xla_baseline_test",
        _test_env("SHUTTLE_RUN_H100_XLA_BASELINE", 1),
        _test_env("SHUTTLE_H100_BASELINE_PYTHON", RUNTIME_PYTHON),
        _test_env("SHUTTLE_H100_XLA_HOST_REPORT", paths.xla_report),
        _test_env("SHUTTLE_H100_TE_STATUS", paths.te_status),
    ]
    shuttle = [
        *common,
        "@shuttle_mlir//:gpu_transport_v2_h100_performance_test",
        _test_env("SHUTTLE_RUN_H100_PERFORMANCE", 1),
        _test_env("SHUTTLE_H100_PERF_EXPLORATORY", 1),
        _test_env("SHUTTLE_H100_PERF_REPORT", paths.shuttle_report),
        _test_env("SHUTTLE_H100_PERF_VALIDATED_REPORT", paths.shuttle_validated),
        _test_env("SHUTTLE_H100_XLA_HOST_REPORT", paths.xla_report),
        _test_env("SHUTTLE_H100_TE_STATUS", paths.te_status),
        _test_env("SHUTTLE_H100_PERF_EXPLORATORY_REPORT", paths.exploratory_comparison),
        _test_env("SHUTTLE_COMPILER_IDENTITY", compiler_identity),
    ]
    return {
        "base_image": BASE_IMAGE,
        "hardware": {"gpu_count": 1, "gpu_model": "H100"},
        "phases": [
            {
                "name": "device",
                "argv": ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            },
            {"name": "source_permissions", "argv": ["chmod", "755", *(str(path) for path in wrappers)]},
            {"name": "xla_init", "argv": ["git", "init", "--quiet"]},
            {
                "name": "xla_remote",
                "argv": ["git", "remote", "add", "origin", "https://github.com/openxla/xla.git"],
            },
            {"name": "xla_fetch", "argv": ["git", "fetch", "--depth=1", "origin", XLA_COMMIT]},
            {"name": "xla_checkout", "argv": ["git", "checkout", "--detach", "FETCH_HEAD"]},
            {"name": "xla_workspace_check", "argv": ["git", "apply", "--check", str(workspace_patch)]},
            {"name": "xla_workspace_apply", "argv": ["git", "apply", str(workspace_patch)]},
            {
                "name": "bazel_fetch",
                "argv": [
                    "curl",
                    "--fail",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--output",
                    str(paths.bazel),
                    BAZEL_URL,
                ],
            },
            {"name": "xla", "argv": xla},
            {"name": "shuttle", "argv": shuttle},
        ],
        "schema": "shuttle_track2_h100_batch_plan_v1",
        "workspace": str(workspace),
    }


def representative_batch_plan(*, workspace: Path, runner_repository: Path) -> dict[str, object]:
    """Return the exact one-H100 representative Shuttle execution plan."""
    paths = BatchPaths.under(workspace)
    hydration = batch_plan(
        workspace=workspace,
        runner_repository=runner_repository,
        compiler_identity=f"shuttle-abi22:{'0' * 40}:{'0' * 40}:{'0' * 64}",
    )["phases"][:9]
    hydration[1] = {
        "name": "source_permissions",
        "argv": [
            "chmod",
            "755",
            str(runner_repository / "lib/shuttle/mlir/test/run-h100-shuttle-rmsnorm-representative.sh"),
            str(runner_repository / "lib/shuttle/mlir/test/run-h100-shuttle-rmsnorm-representative-matrix.sh"),
        ],
    }
    build = [
        *_common_bazel(paths, runner_repository, command="build"),
        "@shuttle_mlir//:run_h100_shuttle_rmsnorm_representative_matrix",
    ]
    execute = [
        *_common_bazel(paths, runner_repository, command="run"),
        "@shuttle_mlir//:run_h100_shuttle_rmsnorm_representative_matrix",
        "--",
        "--python",
        str(RUNTIME_PYTHON),
        "--case-root",
        str(paths.cases),
        "--raw-root",
        str(paths.raw),
        "--reports-root",
        str(paths.reports),
        "--output",
        str(paths.representative_report),
    ]
    return {
        "base_image": BASE_IMAGE,
        "hardware": {"gpu_count": 1, "gpu_model": "H100"},
        "phases": [
            *hydration,
            {"name": "shuttle_build", "argv": build},
            {"name": "shuttle_execute", "argv": execute},
        ],
        "schema": "shuttle_track2_rmsnorm_representative_batch_plan_v1",
        "workspace": str(workspace),
    }


def _run(argv: list[str], *, cwd: Path | None, log: Path, phase: str) -> None:
    print(f"TRACK2_H100_PHASE={phase}", flush=True)
    with log.open("wb") as stream:
        result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode == 0:
        return
    tail = log.read_bytes()[-32 * 1024 :].decode(errors="replace")
    raise RuntimeError(f"phase {phase} failed with exit {result.returncode}\n{tail}")


def _fetch_xla(paths: BatchPaths, phases: list[dict[str, object]]) -> None:
    paths.xla.mkdir()
    for phase in phases:
        name = str(phase["name"])
        _run(phase["argv"], cwd=paths.xla, log=paths.logs / f"{name}.log", phase=name)
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.xla, check=True, capture_output=True, text=True
    ).stdout.strip()
    if observed != XLA_COMMIT:
        raise RuntimeError("hydrated XLA commit differs")


def _device_identity(paths: BatchPaths, phase: dict[str, object]) -> dict[str, object]:
    log = paths.logs / "device.log"
    _run(phase["argv"], cwd=None, log=log, phase=str(phase["name"]))
    rows = [row.strip() for row in log.read_text().splitlines() if row.strip()]
    if len(rows) != 1 or "H100" not in rows[0]:
        raise RuntimeError("batch requires exactly one visible H100")
    name, uuid, driver = (field.strip() for field in rows[0].split(",", maxsplit=2))
    return {"driver": driver, "model": name, "uuid": uuid}


def execute_batch(
    *,
    workspace: Path,
    runner_repository: Path,
    manifest_path: Path,
    observed_base_image: str,
) -> dict[str, object]:
    """Build and execute both timing producers sequentially on one H100."""
    if observed_base_image != BASE_IMAGE:
        raise RuntimeError("observed base image differs from the sealed batch request")
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("batch workspace must be fresh")
    if platform.system() != "Linux":
        raise RuntimeError("Track 2 H100 batch requires Linux")
    source = validate_source_manifest(runner_repository, manifest_path)
    workspace.mkdir(mode=0o700)
    paths = BatchPaths.under(workspace)
    paths.logs.mkdir()
    paths.reports.mkdir()
    paths.bazel.parent.mkdir()
    plan = batch_plan(
        workspace=workspace,
        runner_repository=runner_repository,
        compiler_identity=_compiler_identity(source),
    )
    phases = plan["phases"]
    device = _device_identity(paths, phases[0])
    _run(phases[1]["argv"], cwd=None, log=paths.logs / "source_permissions.log", phase="source_permissions")
    _fetch_xla(paths, phases[2:8])
    _run(phases[8]["argv"], cwd=None, log=paths.logs / "bazel_fetch.log", phase="bazel_fetch")
    if _file_sha256(paths.bazel) != BAZEL_SHA256:
        raise RuntimeError("hydrated Bazel SHA-256 differs")
    paths.bazel.chmod(0o755)
    _run(phases[9]["argv"], cwd=paths.xla, log=paths.logs / "xla.log", phase="xla")
    _run(phases[10]["argv"], cwd=paths.xla, log=paths.logs / "shuttle.log", phase="shuttle")
    reports = {
        "exploratory_comparison": _read_json(paths.exploratory_comparison, "exploratory comparison"),
        "shuttle": _read_json(paths.shuttle_report, "Shuttle report"),
        "shuttle_validated": _read_json(paths.shuttle_validated, "validated Shuttle report"),
        "te_status": _read_json(paths.te_status, "TE status"),
        "xla": _read_json(paths.xla_report, "XLA report"),
    }
    log_identities = {
        path.name: {"sha256": _file_sha256(path), "size": path.stat().st_size} for path in sorted(paths.logs.iterdir())
    }
    result = {
        "base_image": BASE_IMAGE,
        "device": device,
        "reports": reports,
        "schema": RESULT_SCHEMA,
        "source": source,
        "tools": {
            "bazel": {"sha256": BAZEL_SHA256, "version": BAZEL_VERSION},
            "cuda": {
                "build_version": TRACK2_HERMETIC_CUDA_VERSION,
                "compute_capabilities": list(HERMETIC_CUDA_COMPUTE_CAPABILITIES),
                "physical_root": str(PHYSICAL_CUDA_TOOLS_ROOT),
            },
            "logs": log_identities,
            "xla_commit": XLA_COMMIT,
            "xla_workspace_patch_sha256": _file_sha256(runner_repository / XLA_WORKSPACE_PATCH),
        },
    }
    paths.result.write_bytes(canonical_json(result))
    return result


def execute_representative_batch(
    *,
    workspace: Path,
    runner_repository: Path,
    manifest_path: Path,
    observed_base_image: str,
) -> dict[str, object]:
    """Build and execute the reviewed representative Shuttle matrix."""
    if observed_base_image != BASE_IMAGE:
        raise RuntimeError("observed base image differs from the sealed batch request")
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("batch workspace must be fresh")
    if platform.system() != "Linux":
        raise RuntimeError("Track 2 H100 batch requires Linux")
    source = validate_source_manifest(runner_repository, manifest_path)
    workspace.mkdir(mode=0o700)
    paths = BatchPaths.under(workspace)
    paths.logs.mkdir()
    paths.reports.mkdir()
    paths.bazel.parent.mkdir()
    phases = representative_batch_plan(workspace=workspace, runner_repository=runner_repository)["phases"]
    device = _device_identity(paths, phases[0])
    _run(phases[1]["argv"], cwd=None, log=paths.logs / "source_permissions.log", phase="source_permissions")
    _fetch_xla(paths, phases[2:8])
    _run(phases[8]["argv"], cwd=None, log=paths.logs / "bazel_fetch.log", phase="bazel_fetch")
    if _file_sha256(paths.bazel) != BAZEL_SHA256:
        raise RuntimeError("hydrated Bazel SHA-256 differs")
    paths.bazel.chmod(0o755)
    _run(phases[9]["argv"], cwd=paths.xla, log=paths.logs / "shuttle_build.log", phase="shuttle_build")
    _run(phases[10]["argv"], cwd=paths.xla, log=paths.logs / "shuttle_execute.log", phase="shuttle_execute")
    evidence = _read_json(paths.representative_report, "representative Shuttle report")
    r12_root = runner_repository / "lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r12-success-20260821"
    validate_shuttle_evidence(
        evidence,
        r12_result=r12_root / "result-v4.json",
        r12_policy_raw=r12_root / "policy-raw-v4.json",
        raw_root=paths.raw,
        case_root=paths.cases,
    )
    log_identities = {
        path.name: {"sha256": _file_sha256(path), "size": path.stat().st_size} for path in sorted(paths.logs.iterdir())
    }
    report_identities = {
        path.name: {"sha256": _file_sha256(path), "size": path.stat().st_size}
        for path in sorted(paths.reports.iterdir())
        if path != paths.result
    }
    result = {
        "base_image": BASE_IMAGE,
        "device": device,
        "evidence": evidence,
        "reports": report_identities,
        "schema": REPRESENTATIVE_RESULT_SCHEMA,
        "source": source,
        "tools": {
            "bazel": {"sha256": BAZEL_SHA256, "version": BAZEL_VERSION},
            "cuda": {
                "build_version": TRACK2_HERMETIC_CUDA_VERSION,
                "compute_capabilities": list(HERMETIC_CUDA_COMPUTE_CAPABILITIES),
                "physical_root": str(PHYSICAL_CUDA_TOOLS_ROOT),
            },
            "logs": log_identities,
            "xla_commit": XLA_COMMIT,
            "xla_workspace_patch_sha256": _file_sha256(runner_repository / XLA_WORKSPACE_PATCH),
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
    action.add_argument("--plan-representative", action="store_true")
    action.add_argument("--execute-representative", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.plan_representative:
        result = representative_batch_plan(
            workspace=arguments.workspace,
            runner_repository=arguments.runner_repository,
        )
    elif arguments.execute_representative:
        result = execute_representative_batch(
            workspace=arguments.workspace,
            runner_repository=arguments.runner_repository,
            manifest_path=arguments.source_manifest,
            observed_base_image=arguments.base_image,
        )
    elif arguments.plan:
        source = validate_source_manifest(arguments.runner_repository, arguments.source_manifest)
        result = batch_plan(
            workspace=arguments.workspace,
            runner_repository=arguments.runner_repository,
            compiler_identity=_compiler_identity(source),
        )
    else:
        result = execute_batch(
            workspace=arguments.workspace,
            runner_repository=arguments.runner_repository,
            manifest_path=arguments.source_manifest,
            observed_base_image=arguments.base_image,
        )
    print("TRACK2_H100_RESULT=" + canonical_json(result).decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
