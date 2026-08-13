# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize bounded ABI 10 GPU compiler and assembly evidence.

This collector does not build JAX, upload artifacts, or execute GPU kernels. It
extracts the serialized transport from already-generated MLIR, validates it
with the independent decoder, assembles every decoded PTX slice with the
authorized ptxas binary, and records the exact supplied build artifacts.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from gpu_transport import GpuTransport, load_gpu_transport
from target1_abi10_h100_evidence import gpu_transport_descriptor, load_authorization

_BUNDLE_MARKER = 'bundle_bytes = "'
_HEX_ESCAPE = re.compile(r"[0-9A-Fa-f]{2}\Z")
_REGISTER_COUNT = re.compile(r"Used ([0-9]+) registers")
_SPILL_COUNTS = re.compile(r"([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads")
_MAXIMUM_GENERATED_MLIR_BYTES = 32 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def decode_mlir_byte_string(text: str, start: int = 0) -> tuple[bytes, int]:
    """Decode one MLIR quoted byte string beginning after its opening quote."""
    decoded = bytearray()
    position = start
    while position < len(text):
        character = text[position]
        if character == '"':
            return bytes(decoded), position + 1
        if character != "\\":
            codepoint = ord(character)
            if codepoint < 0x20 or codepoint > 0x7E:
                raise ValueError("MLIR bundle string contains a non-ASCII literal")
            decoded.append(codepoint)
            position += 1
            continue
        if position + 1 >= len(text):
            raise ValueError("MLIR bundle string ends in an incomplete escape")
        escaped = text[position + 1]
        if escaped in {'"', "\\"}:
            decoded.append(ord(escaped))
            position += 2
            continue
        token = text[position + 1 : position + 3]
        if len(token) != 2 or _HEX_ESCAPE.fullmatch(token) is None:
            raise ValueError("MLIR bundle string contains an unsupported escape")
        decoded.append(int(token, 16))
        position += 3
    raise ValueError("MLIR bundle string is unterminated")


def extract_gpu_transport(generated_mlir: str) -> bytes:
    """Extract the unique serialized bundle from replacement-pipeline MLIR."""
    starts: list[int] = []
    position = 0
    while True:
        found = generated_mlir.find(_BUNDLE_MARKER, position)
        if found < 0:
            break
        starts.append(found + len(_BUNDLE_MARKER))
        position = starts[-1]
    if len(starts) != 1:
        raise ValueError("generated MLIR must contain exactly one bundle_bytes string")
    transport, end = decode_mlir_byte_string(generated_mlir, starts[0])
    if generated_mlir.find(_BUNDLE_MARKER, end) >= 0:
        raise ValueError("generated MLIR contains multiple bundle_bytes strings")
    return transport


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
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"build artifact is not a regular file: {source}")
    _write_new(destination, source.read_bytes())


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
    if not arguments.output.is_dir() or arguments.output.is_symlink():
        raise RuntimeError("builder output must be an existing real directory")
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

    if arguments.compiled_mlir.is_symlink() or not arguments.compiled_mlir.is_file():
        raise RuntimeError("compiled MLIR must be a regular file")
    if arguments.compiled_mlir.stat().st_size > _MAXIMUM_GENERATED_MLIR_BYTES:
        raise RuntimeError("compiled MLIR exceeds the bounded evidence input size")
    generated_mlir = arguments.compiled_mlir.read_text()
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

    transcript = json.loads(arguments.registration_transcript.read_text())
    if type(transcript) is not dict or transcript.get("plugin_extension_sha256") != _sha256(
        arguments.cuda_plugin_extension
    ):
        raise RuntimeError("registration transcript is not bound to the supplied CUDA plugin")
    build_files = {
        "jax_extension": (arguments.jax_extension, "jax-extension.so"),
        "jax_common": (arguments.jax_common, "libjax-common.so"),
        "cuda_plugin_extension": (arguments.cuda_plugin_extension, "cuda-plugin-extension.so"),
    }
    for source, destination in build_files.values():
        _copy_new(source, arguments.output / "build" / destination)
    _copy_new(arguments.registration_transcript, arguments.output / "build" / "registration-transcript.json")

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
        "commands": document["iris"]["build_commands"],
    }
    _write_new(arguments.output / "build" / "gpu-build-manifest.json", _canonical_json(manifest).encode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled-mlir", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--bazel", type=Path, required=True)
    parser.add_argument("--jax-extension", type=Path, required=True)
    parser.add_argument("--jax-common", type=Path, required=True)
    parser.add_argument("--cuda-plugin-extension", type=Path, required=True)
    parser.add_argument("--registration-transcript", type=Path, required=True)
    materialize(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
