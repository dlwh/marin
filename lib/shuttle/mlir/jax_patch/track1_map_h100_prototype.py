# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorization-bound single-job prototype for the Track 1 Map subject."""

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_string
from track1_map_h100_build import (
    JAX_COMMIT,
    JAX_PATCH_SHA256,
    JAX_TREE,
    XLA_COMMIT,
    XLA_PATCH_SHA256,
    BuildInputs,
    BuildToolDigests,
    SourceCheckoutsIdentity,
    apply_source_patches,
    inspect_build_inputs,
    offline_wheel_command,
    require_offline_build_inputs,
    require_source_checkouts_unchanged,
    source_checkouts_identity,
)

INTEGRATION_COMMIT = "41502f2079385df29fccbf1137bce553aa090297"
INTEGRATION_TREE = "a1c4babf2b5822ac8f1a96b0d6a133b61426d72a"
BASE_IMAGE = (
    "ghcr.io/marin-community/iris-task-h100-evidence:"
    "5238c39e7a506b919fc803046de4a9dc2c29d02f"
    "@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef"
)
BAZEL_VERSION = "7.7.0"
BAZEL_SHA256 = "fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf"
BAZEL_URL = f"https://github.com/bazelbuild/bazel/releases/download/{BAZEL_VERSION}/bazel-{BAZEL_VERSION}-linux-x86_64"
JAX_URL = "https://github.com/jax-ml/jax.git"
XLA_URL = "https://github.com/openxla/xla.git"
AUTHORIZATION_SCHEMA = "track1_map_h100_prototype_authorization"
MAX_AUTHORIZATION_BYTES = 32 * 1024
LOG_TAIL_READ_BYTES = 64 * 1024
LOG_TAIL_MAX_CHARACTERS = 8 * 1024
LOG_TAIL_MAX_LINES = 80
WHEEL_PREFIXES = ("jaxlib-", "jax_cuda13_plugin-", "jax_cuda13_pjrt-")
WORKER_MODES = ("disabled", "source_ordered", "fast")
SOURCE_MANIFEST_RELATIVE = Path("scripts/shuttle/track1_map_h100_source_manifest.json")
SOURCE_ROOTS = (Path("lib/shuttle/mlir"), Path("lib/shuttle/src/shuttle"))
IGNORED_SOURCE_PARTS = frozenset({"__pycache__", ".pytest_cache"})
AUTHORIZATION_LOG_VALUE = re.compile(r"(?im)\b((?:proxy[-_])?authorization\s*[:=]\s*)[^\r\n]*")
SENSITIVE_LOG_VALUE = re.compile(
    r"""(?i)(?<![A-Za-z0-9_-])(["']?([A-Za-z][A-Za-z0-9_-]{0,127})["']?\s*[:=]\s*)"""
    r"""((?:bearer|basic)\s+\S+|"[^"\r\n]*"|'[^'\r\n]*'|\S+)"""
)


@dataclass(frozen=True)
class Authorization:
    document: dict[str, object]
    sha256: str


@dataclass(frozen=True)
class PrototypePaths:
    workspace: Path
    jax: Path
    xla: Path
    bazel: Path
    bazel_output: Path
    hydration_wheels: Path
    wheels: Path
    runtime: Path
    cache: Path
    populate: Path
    reuse: Path
    logs: Path
    image_identity: Path
    hydration_identity: Path

    @classmethod
    def under(cls, workspace: Path) -> "PrototypePaths":
        return cls(
            workspace=workspace,
            jax=workspace / "jax",
            xla=workspace / "xla",
            bazel=workspace / "tools" / f"bazel-{BAZEL_VERSION}",
            bazel_output=workspace / "bazel-output",
            hydration_wheels=workspace / "hydration-wheels",
            wheels=workspace / "wheels",
            runtime=workspace / "runtime",
            cache=workspace / "cache",
            populate=workspace / "populate.json",
            reuse=workspace / "reuse.json",
            logs=workspace / "logs",
            image_identity=workspace / "base-image.identity",
            hydration_identity=workspace / "hydration-identities.json",
        )


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"prototype input must be one regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_authorization(source_manifest_sha256: str) -> dict[str, object]:
    return {
        "attempts": 1,
        "base_image": BASE_IMAGE,
        "hardware": {"gpu_count": 1, "gpu_model": "H100"},
        "integration": {
            "commit": INTEGRATION_COMMIT,
            "source_manifest_sha256": source_manifest_sha256,
            "tree": INTEGRATION_TREE,
        },
        "permissions": {"build": True, "network_hydration": True, "run": True},
        "schema": AUTHORIZATION_SCHEMA,
        "schema_version": 1,
        "sources": {
            "jax": {
                "commit": JAX_COMMIT,
                "patch_sha256": list(JAX_PATCH_SHA256),
                "tree": JAX_TREE,
                "url": JAX_URL,
            },
            "xla": {
                "commit": XLA_COMMIT,
                "patch_sha256": list(XLA_PATCH_SHA256),
                "url": XLA_URL,
            },
        },
        "toolchain": {
            "bazel": {"sha256": BAZEL_SHA256, "url": BAZEL_URL, "version": BAZEL_VERSION},
            "cuda_root": "/usr/local/cuda-13.2",
            "python": "3.12",
        },
    }


def load_authorization(path: Path, approved_sha256: str) -> Authorization:
    """Load an authorization bound to a separately approved SHA-256."""
    if not re.fullmatch(r"[0-9a-f]{64}", approved_sha256):
        raise ValueError("approved authorization SHA-256 must use 64 lowercase hexadecimal characters")
    if path.is_symlink() or not path.is_file():
        raise ValueError("authorization must be one regular file")
    raw = path.read_bytes()
    if len(raw) > MAX_AUTHORIZATION_BYTES:
        raise ValueError("authorization exceeds the prototype size limit")
    if _sha256_bytes(raw) != approved_sha256:
        raise ValueError("authorization SHA-256 differs from the external approval binding")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authorization must contain one JSON object") from error
    if type(document) is not dict or raw != canonical_json(document):
        raise ValueError("authorization must use canonical JSON")
    integration = document.get("integration")
    if type(integration) is not dict:
        raise ValueError("authorization integration scope is missing")
    source_manifest_sha256 = integration.get("source_manifest_sha256")
    if type(source_manifest_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("authorization source manifest SHA-256 is invalid")
    if document != _expected_authorization(source_manifest_sha256):
        raise ValueError("authorization differs from the closed prototype scope")
    return Authorization(document=document, sha256=approved_sha256)


def _build_commands(paths: PrototypePaths, runner_repository: Path) -> tuple[list[str], list[str]]:
    offline = offline_wheel_command(
        jax_repository=paths.jax,
        marin_repository=runner_repository,
        bazel=paths.bazel,
        bazel_output_root=paths.bazel_output,
        cuda_root=Path("/usr/local/cuda-13.2"),
        wheel_output=paths.wheels,
        xla_repository=paths.xla,
    )
    hydrate = [argument for argument in offline if argument != "--bazel_options=--repository_disable_download"]
    hydrate[hydrate.index(f"--output_path={paths.wheels}")] = f"--output_path={paths.hydration_wheels}"
    return hydrate, offline


def _worker_command(paths: PrototypePaths, runner_repository: Path, role: str) -> list[str]:
    command = [
        str(paths.runtime / "bin/python"),
        str(runner_repository / "lib/shuttle/mlir/jax_patch/track1_map_h100_worker.py"),
        "--role",
        role,
        "--cache-directory",
        str(paths.cache),
        "--output",
        str(paths.populate if role == "populate" else paths.reuse),
    ]
    if role == "reuse":
        command.extend(("--populate-result", str(paths.populate)))
    return command


def prototype_plan(
    authorization: Authorization,
    *,
    workspace: Path,
    runner_repository: Path,
    python: Path,
    cuda_root: Path,
    observed_base_image: str,
) -> dict[str, object]:
    """Render the closed command plan without touching the workspace."""
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("prototype requires a fresh workspace path")
    if observed_base_image != BASE_IMAGE:
        raise RuntimeError("observed base image differs from authorization")
    if cuda_root != Path("/usr/local/cuda-13.2"):
        raise RuntimeError("CUDA root differs from authorization")
    paths = PrototypePaths.under(workspace)
    hydrate, offline = _build_commands(paths, runner_repository)
    install_patterns = [str(paths.wheels / f"{prefix}*.whl") for prefix in WHEEL_PREFIXES]
    phases = [
        {
            "name": "fetch_jax",
            "argv": ["git", "fetch", "--depth=1", JAX_URL, JAX_COMMIT],
            "commands": [
                ["git", "init", "--quiet"],
                ["git", "remote", "add", "origin", JAX_URL],
                ["git", "fetch", "--depth=1", "origin", JAX_COMMIT],
                ["git", "checkout", "--detach", "FETCH_HEAD"],
            ],
            "cwd": str(paths.jax),
        },
        {
            "name": "fetch_xla",
            "argv": ["git", "fetch", "--depth=1", XLA_URL, XLA_COMMIT],
            "commands": [
                ["git", "init", "--quiet"],
                ["git", "remote", "add", "origin", XLA_URL],
                ["git", "fetch", "--depth=1", "origin", XLA_COMMIT],
                ["git", "checkout", "--detach", "FETCH_HEAD"],
            ],
            "cwd": str(paths.xla),
        },
        {
            "name": "fetch_bazel",
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
        {"name": "hydrate_bazel_repositories", "argv": hydrate, "cwd": str(paths.jax)},
        {"name": "offline_wheel_build", "argv": offline, "cwd": str(paths.jax)},
        {
            "name": "create_runtime",
            "argv": [str(python), "-m", "venv", "--system-site-packages", str(paths.runtime)],
        },
        {
            "name": "install_wheels",
            "argv": [
                str(paths.runtime / "bin/python"),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                *install_patterns,
            ],
        },
        {"name": "populate", "argv": _worker_command(paths, runner_repository, "populate")},
        {"name": "reuse", "argv": _worker_command(paths, runner_repository, "reuse")},
    ]
    return {
        "authorization_sha256": authorization.sha256,
        "base_image": BASE_IMAGE,
        "phases": phases,
        "schema_version": 1,
        "workspace": str(workspace),
    }


def wheel_set(directory: Path) -> tuple[Path, Path, Path]:
    """Return exactly one regular wheel for each authorized distribution."""
    selected = []
    for prefix in WHEEL_PREFIXES:
        matches = sorted(directory.glob(f"{prefix}*.whl"))
        if len(matches) != 1:
            raise RuntimeError(f"prototype wheel output must contain exactly one {prefix} wheel")
        wheel = matches[0]
        if wheel.is_symlink() or not wheel.is_file():
            raise RuntimeError("prototype wheels must be regular files")
        selected.append(wheel)
    all_wheels = sorted(directory.glob("*.whl"))
    if all_wheels != sorted(selected):
        raise RuntimeError("prototype wheel output contains an unauthorized wheel")
    return selected[0], selected[1], selected[2]


def _sha(value: object, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"worker result {name} must be a lowercase SHA-256")
    return value


def source_bundle_document(bundle_root: Path) -> dict[str, object]:
    """Return the canonical content closure used by the external build and workers."""
    entries = []
    for relative_root in SOURCE_ROOTS:
        root = bundle_root / relative_root
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"source bundle root is missing or not a directory: {relative_root}")
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(bundle_root)
            if any(part in IGNORED_SOURCE_PARTS for part in relative.parts) or path.suffix == ".pyc":
                continue
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"source bundle contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"source bundle contains a special file: {relative}")
            entries.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256_file(path),
                    "size": metadata.st_size,
                }
            )
    return {"entries": entries, "roots": [root.as_posix() for root in SOURCE_ROOTS], "schema_version": 1}


def require_source_bundle(bundle_root: Path, authorization: Authorization) -> dict[str, object]:
    """Match a live Iris source bundle to the authorized canonical manifest."""
    manifest_path = bundle_root / SOURCE_MANIFEST_RELATIVE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("source bundle manifest must be one regular file")
    raw = manifest_path.read_bytes()
    try:
        expected = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("source bundle manifest is not JSON") from error
    if type(expected) is not dict or raw != canonical_json(expected):
        raise RuntimeError("source bundle manifest must use canonical JSON")
    integration = authorization.document["integration"]
    assert type(integration) is dict
    if _sha256_bytes(raw) != integration["source_manifest_sha256"]:
        raise RuntimeError("source bundle manifest SHA-256 differs from authorization")
    observed = source_bundle_document(bundle_root)
    if observed != expected:
        raise RuntimeError("live source bundle differs from the authorized manifest")
    return expected


def validate_worker_result(document: object, role: str) -> dict[str, object]:
    """Validate one canonical JSON-ready worker result."""
    if role not in ("populate", "reuse"):
        raise ValueError("worker result validator role is invalid")
    if type(document) is not dict or set(document) != {"coordinates", "pid", "schema_version", "worker"}:
        raise ValueError("worker result has the wrong root schema")
    if document["schema_version"] != 1 or document["worker"] != role:
        raise ValueError("worker result has the wrong role or schema version")
    if type(document["pid"]) is not int or document["pid"] <= 0:
        raise ValueError("worker result PID must be positive")
    coordinates = document["coordinates"]
    if type(coordinates) is not list or len(coordinates) != 6:
        raise ValueError("worker result must cover six coordinates")
    expected = [(rank, mode) for rank in (2, 3) for mode in WORKER_MODES]
    cache_entries = []
    for index, coordinate in enumerate(coordinates):
        if type(coordinate) is not dict or set(coordinate) != {
            "cache_entry",
            "gradient_sha256",
            "mode",
            "observer",
            "primal_sha256",
            "rank",
        }:
            raise ValueError("worker result coordinate has the wrong schema")
        if (coordinate["rank"], coordinate["mode"]) != expected[index]:
            raise ValueError("worker result coordinate order differs")
        _sha(coordinate["primal_sha256"], "primal")
        _sha(coordinate["gradient_sha256"], "gradient")
        entry = coordinate["cache_entry"]
        if (
            type(entry) is not list
            or len(entry) != 3
            or type(entry[0]) is not str
            or not entry[0].endswith("-cache")
            or type(entry[1]) is not int
            or entry[1] <= 0
        ):
            raise ValueError("worker result cache entry is invalid")
        _sha(entry[2], "cache entry")
        cache_entries.append(tuple(entry))
        observer = coordinate["observer"]
        if role == "reuse" or coordinate["mode"] == "disabled":
            if observer is not None:
                raise ValueError("worker result has an observer in an excluded phase")
        elif (
            type(observer) is not dict
            or set(observer) != {"final_module_sha256", "generated_bundle_sha256", "invocation_id"}
            or type(observer["invocation_id"]) is not int
            or observer["invocation_id"] <= 0
        ):
            raise ValueError("worker result enabled observer is invalid")
        else:
            _sha(observer["final_module_sha256"], "final module")
            _sha(observer["generated_bundle_sha256"], "generated bundle")
    if len(set(cache_entries)) != 6:
        raise ValueError("worker result cache attribution is not unique")
    return document


def validate_result_pair(populate: object, reuse: object) -> dict[str, object]:
    """Validate fresh-process cache reuse and result identity."""
    populated = validate_worker_result(populate, "populate")
    reused = validate_worker_result(reuse, "reuse")
    if populated["pid"] == reused["pid"]:
        raise ValueError("worker result processes must have distinct PIDs")
    for first, second in zip(populated["coordinates"], reused["coordinates"], strict=True):
        if any(
            first[field] != second[field]
            for field in ("rank", "mode", "primal_sha256", "gradient_sha256", "cache_entry")
        ):
            raise ValueError("worker result reuse differs from populate output or cache")
    return {"populate": populated, "reuse": reused}


def _load_canonical_document(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be one regular file")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not JSON") from error
    if type(document) is not dict or raw != canonical_json(document):
        raise RuntimeError(f"{name} must use canonical JSON")
    return document


def _sanitized_log_tail(log: Path) -> str:
    with log.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        start = max(0, stream.tell() - LOG_TAIL_READ_BYTES)
        stream.seek(start)
        raw = stream.read()
    truncated = start > 0
    if start:
        _, newline, suffix = raw.partition(b"\n")
        if newline:
            raw = suffix
    tail = redact_string(raw.decode(errors="replace"))
    tail = AUTHORIZATION_LOG_VALUE.sub(rf"\1{REDACTED_VALUE}", tail)

    def redact_sensitive_value(match: re.Match[str]) -> str:
        if is_sensitive_key_name(match.group(2)):
            return f"{match.group(1)}{REDACTED_VALUE}"
        return match.group(0)

    tail = SENSITIVE_LOG_VALUE.sub(redact_sensitive_value, tail)
    tail = re.sub(r"(?i)\b(bearer\s+)\S+", rf"\1{REDACTED_VALUE}", tail)
    tail = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[REDACTED]@", tail)
    tail = re.sub(
        r"(?i)([?&](?:api[_-]?key|authorization|credential|password|secret|token)=)[^&\s]+",
        r"\1[REDACTED]",
        tail,
    )
    tail = "".join(character if character in "\n\t" or character.isprintable() else "�" for character in tail)
    lines = tail.splitlines()
    if len(lines) > LOG_TAIL_MAX_LINES:
        truncated = True
        lines = lines[-LOG_TAIL_MAX_LINES:]
    tail = "\n".join(lines)
    if len(tail) > LOG_TAIL_MAX_CHARACTERS:
        truncated = True
        tail = tail[-LOG_TAIL_MAX_CHARACTERS:]
    if truncated:
        tail = "[tail truncated]\n" + tail
    return tail or "[no output]"


def _run(
    command: list[str],
    *,
    cwd: Path | None,
    log: Path,
    phase: str,
    environment: dict[str, str] | None = None,
) -> None:
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(
            f"prototype phase {phase} exited with status {result.returncode}; sanitized log tail:\n"
            f"{_sanitized_log_tail(log)}"
        )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _require_source(repository: Path, commit: str, *, tree: str | None = None) -> None:
    if _git(repository, "rev-parse", "HEAD") != commit:
        raise RuntimeError("hydrated source commit differs from authorization")
    if tree is not None and _git(repository, "rev-parse", "HEAD^{tree}") != tree:
        raise RuntimeError("hydrated source tree differs from authorization")
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("hydrated source checkout is not clean")


def _identity(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def _hydration_identity(
    paths: PrototypePaths,
    cuda_root: Path,
    inspected: BuildInputs,
    source_identity: SourceCheckoutsIdentity,
    source_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "base_image": BASE_IMAGE,
        "patches": {
            "jax_sha256": list(inspected.jax_patch_sha256),
            "xla_sha256": list(inspected.xla_patch_sha256),
        },
        "schema_version": 1,
        "sources": {
            "integration_commit": INTEGRATION_COMMIT,
            "integration_source_manifest_sha256": source_manifest_sha256,
            "integration_tree": INTEGRATION_TREE,
            "patched_checkouts": asdict(source_identity),
        },
        "tools": {
            "bazel": {"identity": _identity([str(paths.bazel), "--version"]), "sha256": _sha256_file(paths.bazel)},
            "nvcc": {
                "identity": _identity([str(cuda_root / "bin/nvcc"), "--version"]),
                "sha256": _sha256_file(cuda_root / "bin/nvcc"),
            },
            "ptxas": {
                "identity": _identity([str(cuda_root / "bin/ptxas"), "--version"]),
                "sha256": _sha256_file(cuda_root / "bin/ptxas"),
            },
        },
    }


def _initialize_repository(path: Path, url: str, commit: str, log: Path) -> None:
    path.mkdir()
    source = path.name
    _run(["git", "init", "--quiet"], cwd=path, log=log, phase=f"fetch_{source}_init")
    _run(
        ["git", "remote", "add", "origin", url],
        cwd=path,
        log=log.with_suffix(".remote.log"),
        phase=f"fetch_{source}_remote",
    )
    _run(
        ["git", "fetch", "--depth=1", "origin", commit],
        cwd=path,
        log=log.with_suffix(".fetch.log"),
        phase=f"fetch_{source}",
    )
    _run(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=path,
        log=log.with_suffix(".checkout.log"),
        phase=f"fetch_{source}_checkout",
    )


def _runtime_environment(paths: PrototypePaths, runner_repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = (
        runner_repository / "lib/shuttle/src",
        runner_repository / "lib/shuttle/mlir/jax_patch",
    )
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_paths))
    return environment


def execute_prototype(
    authorization: Authorization,
    *,
    workspace: Path,
    runner_repository: Path,
    python: Path,
    cuda_root: Path,
    observed_base_image: str,
) -> dict[str, object]:
    """Hydrate once, build offline, and run populate/reuse in fresh processes."""
    prototype_plan(
        authorization,
        workspace=workspace,
        runner_repository=runner_repository,
        python=python,
        cuda_root=cuda_root,
        observed_base_image=observed_base_image,
    )
    source_manifest = require_source_bundle(runner_repository, authorization)
    source_manifest_sha256 = _sha256_bytes(canonical_json(source_manifest))
    if platform.system().lower() != "linux":
        raise RuntimeError("Track 1 external prototype requires Linux")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("prototype Python 3.12 must resolve to one executable file")
    if not _identity([str(python), "--version"]).startswith("Python 3.12."):
        raise RuntimeError("prototype requires Python 3.12")
    for tool in (cuda_root / "bin/nvcc", cuda_root / "bin/ptxas"):
        if tool.is_symlink() or not tool.is_file() or not os.access(tool, os.X_OK):
            raise RuntimeError(f"prototype CUDA tool must be one executable regular file: {tool}")
    paths = PrototypePaths.under(workspace)
    workspace.mkdir(mode=0o700)
    paths.logs.mkdir()
    paths.bazel.parent.mkdir()
    _initialize_repository(paths.jax, JAX_URL, JAX_COMMIT, paths.logs / "jax-init.log")
    _initialize_repository(paths.xla, XLA_URL, XLA_COMMIT, paths.logs / "xla-init.log")
    _require_source(paths.jax, JAX_COMMIT, tree=JAX_TREE)
    _require_source(paths.xla, XLA_COMMIT)
    _run(
        [
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
        cwd=None,
        log=paths.logs / "bazel-fetch.log",
        phase="fetch_bazel",
    )
    observed_bazel_sha256 = _sha256_file(paths.bazel)
    if observed_bazel_sha256 != BAZEL_SHA256:
        raise RuntimeError(f"hydrated Bazel SHA-256 differs: expected {BAZEL_SHA256}, observed {observed_bazel_sha256}")
    paths.bazel.chmod(0o755)
    paths.image_identity.write_text(BASE_IMAGE + "\n")

    inspected = inspect_build_inputs(
        marin_repository=runner_repository,
        jax_repository=paths.jax,
        bazel=paths.bazel,
        cuda_root=cuda_root,
        image_archive=paths.image_identity,
        xla_repository=paths.xla,
    )
    tools = BuildToolDigests(
        bazel=BAZEL_SHA256,
        image=_sha256_file(paths.image_identity),
        nvcc=_sha256_file(cuda_root / "bin/nvcc"),
        ptxas=_sha256_file(cuda_root / "bin/ptxas"),
    )
    require_offline_build_inputs(inspected, expected_tool_sha256=tools)
    apply_source_patches(inspected)
    patched_sources = source_checkouts_identity(jax_repository=paths.jax, xla_repository=paths.xla)
    hydration_identity = _hydration_identity(paths, cuda_root, inspected, patched_sources, source_manifest_sha256)
    paths.hydration_identity.write_bytes(canonical_json(hydration_identity))
    hydrate, offline = _build_commands(paths, runner_repository)
    _run(
        hydrate,
        cwd=paths.jax,
        log=paths.logs / "hydrate-build.log",
        phase="hydrate_bazel_repositories",
    )
    require_source_checkouts_unchanged(
        patched_sources,
        jax_repository=paths.jax,
        xla_repository=paths.xla,
    )
    if _hydration_identity(paths, cuda_root, inspected, patched_sources, source_manifest_sha256) != hydration_identity:
        raise RuntimeError("hydrated tool identities changed before the offline build")
    require_source_checkouts_unchanged(
        patched_sources,
        jax_repository=paths.jax,
        xla_repository=paths.xla,
    )
    _run(offline, cwd=paths.jax, log=paths.logs / "offline-build.log", phase="offline_wheel_build")
    wheels = wheel_set(paths.wheels)

    _run(
        [str(python), "-m", "venv", "--system-site-packages", str(paths.runtime)],
        cwd=None,
        log=paths.logs / "runtime.log",
        phase="create_runtime",
    )
    _run(
        [
            str(paths.runtime / "bin/python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            *map(str, wheels),
        ],
        cwd=None,
        log=paths.logs / "wheel-install.log",
        phase="install_wheels",
    )
    environment = _runtime_environment(paths, runner_repository)
    _run(
        _worker_command(paths, runner_repository, "populate"),
        cwd=runner_repository,
        log=paths.logs / "populate.log",
        phase="populate",
        environment=environment,
    )
    _run(
        _worker_command(paths, runner_repository, "reuse"),
        cwd=runner_repository,
        log=paths.logs / "reuse.log",
        phase="reuse",
        environment=environment,
    )
    workers = validate_result_pair(
        _load_canonical_document(paths.populate, "populate result"),
        _load_canonical_document(paths.reuse, "reuse result"),
    )
    return {
        "attempt": 1,
        "authorization_sha256": authorization.sha256,
        "base_image": BASE_IMAGE,
        "integration": {"commit": INTEGRATION_COMMIT, "tree": INTEGRATION_TREE},
        "schema": "track1_map_h100_prototype_result",
        "schema_version": 1,
        "hydration": hydration_identity,
        "wheels": [
            {"name": wheel.name, "sha256": _sha256_file(wheel), "size": wheel.stat().st_size} for wheel in wheels
        ],
        "workers": workers,
    }
