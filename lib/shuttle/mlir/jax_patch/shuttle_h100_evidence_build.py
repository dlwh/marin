# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize bounded ABI 10 GPU compiler and assembly evidence.

This collector builds the authorized JAX wheels and Shuttle compiler, extracts
the serialized transport from parser-canonical MLIR, validates it with the
independent decoder, and assembles every decoded PTX slice with the authorized
ptxas binary. It does not upload artifacts or execute GPU kernels.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

from gpu_transport import GpuTransport, extract_gpu_transport, load_gpu_transport
from target1_abi10_h100_evidence import gpu_transport_descriptor, load_authorization

_REGISTER_COUNT = re.compile(r"Used ([0-9]+) registers")
_SPILL_COUNTS = re.compile(r"([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads")
_MAXIMUM_GENERATED_MLIR_BYTES = 32 * 1024 * 1024
_JAX_WHEELS = "jaxlib,jax-cuda-plugin,jax-cuda-pjrt"
_FORWARD_FIXTURE = "lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def parse_ptxas_resources(stderr: str) -> tuple[int, int]:
    """Return register and spill-byte counts from one verbose ptxas result."""
    registers = [int(value) for value in _REGISTER_COUNT.findall(stderr)]
    spills = [(int(stores), int(loads)) for stores, loads in _SPILL_COUNTS.findall(stderr)]
    if len(registers) != 1 or len(spills) != 1:
        raise ValueError("ptxas verbose output lacks one closed resource record")
    spill_bytes = spills[0][0] + spills[0][1]
    if registers[0] <= 0 or spill_bytes != 0:
        raise ValueError("ptxas resource record reports invalid registers or spills")
    return registers[0], spill_bytes


def _write_new(path: Path, data: bytes) -> None:
    if (
        path.exists()
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.parent.is_dir()
    ):
        raise RuntimeError(f"refusing to overwrite evidence artifact: {path}")
    path.write_bytes(data)


def _ptxas_version(ptxas: Path) -> str:
    result = subprocess.run([str(ptxas), "--version"], check=True, capture_output=True, text=True)
    match = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", result.stdout + result.stderr)
    if match is None:
        raise RuntimeError("could not parse the exact ptxas version")
    return match.group(1)


def _bazel_version(bazel: Path) -> str:
    result = subprocess.run([str(bazel), "--version"], check=True, capture_output=True, text=True)
    match = re.fullmatch(r"bazel ([0-9]+\.[0-9]+\.[0-9]+)\s*", result.stdout + result.stderr)
    if match is None:
        raise RuntimeError("could not parse the exact Bazel version")
    return match.group(1)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _prepare_directories(output: Path) -> None:
    for name in ("assembly", "build", "compiler"):
        path = output / name
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"evidence output directory already exists: {path}")
        path.mkdir()


def _require_fresh_output(path: Path, name: str) -> None:
    if (
        path.exists()
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.parent.is_dir()
    ):
        raise RuntimeError(f"{name} must be a new path under a real existing parent")


def _apply_jax_patches(jax_repository: Path, document: dict[str, object], marin_repository: Path) -> None:
    source = document["source"]
    if _git(jax_repository, "rev-parse", "HEAD") != source["jax_commit"]:
        raise RuntimeError("JAX checkout differs from the sealed commit")
    if _git(jax_repository, "rev-parse", "HEAD^{tree}") != source["jax_tree"]:
        raise RuntimeError("JAX checkout differs from the sealed tree")
    if _git(jax_repository, "status", "--porcelain"):
        raise RuntimeError("JAX checkout must be clean before applying the sealed patches")
    patches = sorted((marin_repository / "lib/shuttle/mlir/jax_patch").glob("000[1-3]-*.patch"))
    if [_sha256(path) for path in patches] != source["jax_patch_sha256"]:
        raise RuntimeError("JAX patch bytes differ from the sealed authorization")
    subprocess.run(["git", "apply", "--check", *map(str, patches)], cwd=jax_repository, check=True)
    subprocess.run(["git", "apply", *map(str, patches)], cwd=jax_repository, check=True)


def _build_outputs(
    jax_repository: Path,
    marin_repository: Path,
    bazel: Path,
    bazel_output_root: Path,
    wheel_output: Path,
    authorized_commands: list[str],
) -> tuple[list[str], Path]:
    command = [
        sys.executable,
        "build/build.py",
        "build",
        f"--wheels={_JAX_WHEELS}",
        f"--bazel_path={bazel}",
        f"--output_path={wheel_output}",
        "--cuda_version=13.0.0",
        "--cuda_compute_capabilities=9.0",
        "--disable_nccl",
        "--bazel_options=--repo_env=ML_WHEEL_TYPE=release",
        f"--bazel_startup_options=--output_user_root={bazel_output_root}",
        f"--bazel_options=--override_repository=shuttle_mlir={marin_repository / 'lib/shuttle/mlir'}",
        "--bazel_options=--local_resources=host_cpus=32",
    ]
    command_text = shlex.join(command)
    shuttle_command = [
        str(bazel),
        f"--output_user_root={bazel_output_root}",
        "build",
        f"--override_repository=shuttle_mlir={marin_repository / 'lib/shuttle/mlir'}",
        "@shuttle_mlir//:shuttle-test-opt",
        "--local_resources=host_cpus=32",
    ]
    shuttle_command_text = shlex.join(shuttle_command)
    if authorized_commands[:2] != [command_text, shuttle_command_text]:
        raise RuntimeError("build argv differs from the sealed authorization")
    subprocess.run(command, cwd=jax_repository, check=True)
    subprocess.run(shuttle_command, cwd=jax_repository, check=True)
    query = subprocess.run(
        [
            str(bazel),
            f"--output_user_root={bazel_output_root}",
            "cquery",
            f"--override_repository=shuttle_mlir={marin_repository / 'lib/shuttle/mlir'}",
            "@shuttle_mlir//:shuttle-test-opt",
            "--output=files",
        ],
        cwd=jax_repository,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [Path(line) for line in query.stdout.splitlines() if line]
    if len(paths) != 1:
        raise RuntimeError("Bazel did not identify exactly one shuttle-test-opt output")
    shuttle_test_opt = paths[0] if paths[0].is_absolute() else jax_repository / paths[0]
    return [command_text, shuttle_command_text], shuttle_test_opt


def _one_wheel(directory: Path, prefix: str) -> Path:
    matches = sorted(directory.glob(f"{prefix}-*.whl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one built {prefix} wheel")
    return matches[0]


def _install_wheels(wheels: dict[str, Path], executed_commands: list[str], authorized_commands: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        *map(str, wheels.values()),
    ]
    command_text = shlex.join(command)
    if authorized_commands != [*executed_commands, command_text]:
        raise RuntimeError("wheel install argv differs from the sealed authorization")
    subprocess.run(command, check=True)
    return [*executed_commands, command_text]


def _wheel_member(wheel: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(wheel) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"wheel does not contain exactly one {suffix}")
        return archive.read(matches[0])


def _generate_mlir(shuttle_test_opt: Path, fixture: Path) -> str:
    result = subprocess.run(
        [
            str(shuttle_test_opt),
            str(fixture),
            "--shuttle-gpu-executable-bundle-pipeline",
            "--mlir-print-op-generic",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if len(result.stdout.encode()) > _MAXIMUM_GENERATED_MLIR_BYTES:
        raise RuntimeError("compiled MLIR exceeds the bounded evidence input size")
    return result.stdout


def assemble_transport(output: Path, transport: GpuTransport, ptxas: Path) -> dict[str, object]:
    """Write and assemble every decoded PTX slice in launch-ordinal order."""
    records = []
    for launch in transport.launches:
        ptx = output / "assembly" / f"entry-{launch.ordinal:02d}.ptx"
        cubin = output / "assembly" / f"entry-{launch.ordinal:02d}.cubin"
        _write_new(ptx, launch.code)
        if cubin.exists() or cubin.is_symlink():
            raise RuntimeError(f"refusing to overwrite evidence artifact: {cubin}")
        result = subprocess.run(
            [str(ptxas), "-arch=sm_90", "-v", "-o", str(cubin), str(ptx)],
            check=True,
            capture_output=True,
            text=True,
        )
        if cubin.is_symlink() or not cubin.is_file() or cubin.stat().st_size == 0:
            raise RuntimeError(f"ptxas did not create a nonempty cubin for ordinal {launch.ordinal}")
        register_count, spill_bytes = parse_ptxas_resources(result.stdout + result.stderr)
        records.append(
            {
                "ordinal": launch.ordinal,
                "ptx_sha256": _sha256(ptx),
                "cubin_sha256": _sha256(cubin),
                "exit_code": result.returncode,
                "spill_bytes": spill_bytes,
                "register_count": register_count,
            }
        )
    return {"entries": records}


def materialize(arguments: argparse.Namespace) -> None:
    authorization = load_authorization()
    document = authorization.decoded()
    if arguments.authorization_sha256 != authorization.sha256 or arguments.challenge != document["challenge"]:
        raise RuntimeError("builder argv differs from the sealed authorization")
    if (
        not arguments.output.is_dir()
        or arguments.output.is_symlink()
        or any(parent.is_symlink() for parent in arguments.output.parents)
    ):
        raise RuntimeError("builder output must be an existing real directory")
    _prepare_directories(arguments.output)
    if (
        arguments.jax_repository.is_symlink()
        or any(parent.is_symlink() for parent in arguments.jax_repository.parents)
        or not arguments.jax_repository.is_dir()
    ):
        raise RuntimeError("JAX repository must be a real directory")
    _require_fresh_output(arguments.bazel_output_root, "Bazel output root")
    _require_fresh_output(arguments.wheel_output, "wheel output")
    if arguments.ptxas.is_symlink() or not arguments.ptxas.is_file():
        raise RuntimeError("ptxas must be a regular file")
    if arguments.bazel.is_symlink() or not arguments.bazel.is_file():
        raise RuntimeError("Bazel must be a regular file")
    if _sha256(arguments.bazel) != document["toolchain"]["bazel_sha256"]:
        raise RuntimeError("Bazel bytes differ from the sealed authorization")
    if _bazel_version(arguments.bazel) != document["toolchain"]["bazel_version"]:
        raise RuntimeError("Bazel version differs from the sealed authorization")
    if _sha256(arguments.ptxas) != document["toolchain"]["ptxas_sha256"]:
        raise RuntimeError("ptxas bytes differ from the sealed authorization")
    if _ptxas_version(arguments.ptxas) != document["toolchain"]["ptxas_version"]:
        raise RuntimeError("ptxas version differs from the sealed authorization")

    marin_repository = Path(__file__).resolve().parents[4]
    _apply_jax_patches(arguments.jax_repository, document, marin_repository)
    commands, shuttle_test_opt = _build_outputs(
        arguments.jax_repository,
        marin_repository,
        arguments.bazel,
        arguments.bazel_output_root,
        arguments.wheel_output,
        document["iris"]["build_commands"],
    )
    generated_mlir = _generate_mlir(shuttle_test_opt, marin_repository / _FORWARD_FIXTURE)
    _write_new(arguments.output / "compiler" / "generated.mlir", generated_mlir.encode())
    transport_bytes = extract_gpu_transport(generated_mlir)
    transport_path = arguments.output / "compiler" / "gpu-transport.bin"
    _write_new(transport_path, transport_bytes)
    transport = load_gpu_transport(transport_path)
    expected = document["generated_bundle"]
    if transport.transport_root != expected["transport_sha256"]:
        raise RuntimeError("compiled GPU transport differs from the sealed authorization")
    descriptor = _canonical_json(gpu_transport_descriptor(transport)).encode()
    _write_new(arguments.output / "compiler" / "decoded-descriptor.json", descriptor)

    assembly = assemble_transport(arguments.output, transport, arguments.ptxas)
    records = {
        "schema_version": 1,
        "ptxas_version": document["toolchain"]["ptxas_version"],
        "ptxas_sha256": document["toolchain"]["ptxas_sha256"],
        "target": document["toolchain"]["target"],
        **assembly,
    }
    _write_new(arguments.output / "assembly" / "ptxas-records.json", _canonical_json(records).encode())

    wheels = {
        "jaxlib": _one_wheel(arguments.wheel_output, "jaxlib"),
        "cuda_plugin": _one_wheel(arguments.wheel_output, "jax_cuda13_plugin"),
        "cuda_pjrt": _one_wheel(arguments.wheel_output, "jax_cuda13_pjrt"),
    }
    commands = _install_wheels(wheels, commands, document["iris"]["build_commands"])
    for key, name in (
        ("jaxlib", "jaxlib.whl"),
        ("cuda_plugin", "jax-cuda13-plugin.whl"),
        ("cuda_pjrt", "jax-cuda13-pjrt.whl"),
    ):
        _write_new(arguments.output / "build" / name, wheels[key].read_bytes())
    extracted = {
        "jax-extension.so": _wheel_member(wheels["jaxlib"], "/_jax.so"),
        "libjax-common.so": _wheel_member(wheels["jaxlib"], "/libjax_common.so"),
        "cuda-plugin-extension.so": _wheel_member(wheels["cuda_plugin"], "/cuda_plugin_extension.so"),
        "xla-cuda-plugin.so": _wheel_member(wheels["cuda_pjrt"], "/xla_cuda_plugin.so"),
        "cuda-plugin-RECORD": _wheel_member(wheels["cuda_plugin"], ".dist-info/RECORD"),
        "cuda-pjrt-RECORD": _wheel_member(wheels["cuda_pjrt"], ".dist-info/RECORD"),
    }
    for name, data in extracted.items():
        _write_new(arguments.output / "build" / name, data)
    transcript_path = arguments.output / "build" / "registration-transcript.json"
    subprocess.run(
        [
            sys.executable,
            str(marin_repository / "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_gpu_ffi_registration.py"),
            "--plugin-extension",
            str(arguments.output / "build" / "cuda-plugin-extension.so"),
            "--output",
            str(transcript_path),
        ],
        check=True,
    )

    manifest = {
        "schema_version": 1,
        "jax_commit": document["source"]["jax_commit"],
        "jax_tree": document["source"]["jax_tree"],
        "xla_commit": document["source"]["xla_commit"],
        "jax_patch_sha256": document["source"]["jax_patch_sha256"],
        "bazel_version": document["toolchain"]["bazel_version"],
        "bazel_sha256": document["toolchain"]["bazel_sha256"],
        "ptxas_version": document["toolchain"]["ptxas_version"],
        "ptxas_sha256": document["toolchain"]["ptxas_sha256"],
        "jax_extension_sha256": _sha256(arguments.output / "build" / "jax-extension.so"),
        "jax_common_sha256": _sha256(arguments.output / "build" / "libjax-common.so"),
        "cuda_plugin_extension_sha256": _sha256(arguments.output / "build" / "cuda-plugin-extension.so"),
        "cuda_pjrt_plugin_sha256": _sha256(arguments.output / "build" / "xla-cuda-plugin.so"),
        "jaxlib_wheel_sha256": _sha256(arguments.output / "build" / "jaxlib.whl"),
        "cuda_plugin_wheel_sha256": _sha256(arguments.output / "build" / "jax-cuda13-plugin.whl"),
        "cuda_pjrt_wheel_sha256": _sha256(arguments.output / "build" / "jax-cuda13-pjrt.whl"),
        "commands": commands,
    }
    _write_new(arguments.output / "build" / "gpu-build-manifest.json", _canonical_json(manifest).encode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--bazel", type=Path, required=True)
    parser.add_argument("--jax-repository", type=Path, required=True)
    parser.add_argument("--bazel-output-root", type=Path, required=True)
    parser.add_argument("--wheel-output", type=Path, required=True)
    materialize(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
