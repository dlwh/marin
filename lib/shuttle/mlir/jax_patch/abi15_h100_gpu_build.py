# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-hardware ABI15 build and SM90 assembly boundary."""

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from abi15_h100_architecture_contract import load_architecture_build_contract
from abi15_h100_evidence import require_sealed_hardware_authority
from abi15_h100_gpu_transport import GpuTransport, TransportCoordinate

_REGISTER_COUNT = re.compile(r"Used ([0-9]+) registers")
_SPILL_COUNTS = re.compile(r"([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} must use the closed schema")
    return value


def _sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _revision(value: object, name: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full Git SHA-1")
    return value


def parse_ptxas_resources(output: str) -> tuple[int, int]:
    """Return the unique register count and total spill bytes."""
    registers = _REGISTER_COUNT.findall(output)
    spills = _SPILL_COUNTS.findall(output)
    if len(registers) != 1 or len(spills) != 1:
        raise ValueError("ptxas output lacks one bounded resource record")
    register_count = int(registers[0])
    spill_stores, spill_loads = map(int, spills[0])
    if register_count <= 0:
        raise ValueError("ptxas reported no registers")
    return register_count, spill_stores + spill_loads


def assemble_transports(
    output: Path,
    transports: Mapping[TransportCoordinate, GpuTransport],
    ptxas: Path,
) -> list[dict[str, object]]:
    """Assemble the exact four transports without loading or executing a GPU."""
    require_sealed_hardware_authority()
    expected = {
        ("source_ordered", "forward"),
        ("source_ordered", "backward"),
        ("fast", "forward"),
        ("fast", "backward"),
    }
    if {(coordinate.policy.value, coordinate.call_role.value) for coordinate in transports} != expected:
        raise ValueError("assembly requires the four ABI15 policy/call transports")
    records = []
    for coordinate in sorted(transports, key=lambda item: (item.policy.value, item.call_role.value)):
        transport = transports[coordinate]
        for launch in transport.launches:
            directory = output / coordinate.policy.value / coordinate.call_role.value
            directory.mkdir(parents=True, exist_ok=True)
            ptx = directory / f"entry-{launch.ordinal:02d}.ptx"
            cubin = directory / f"entry-{launch.ordinal:02d}.cubin"
            if ptx.exists() or ptx.is_symlink() or cubin.exists() or cubin.is_symlink():
                raise ValueError("assembly output paths must be fresh")
            ptx.write_bytes(launch.code)
            command = [str(ptxas), "-arch=sm_90", "-v", "-o", str(cubin), str(ptx)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if cubin.is_symlink() or not cubin.is_file() or cubin.stat().st_size == 0:
                raise ValueError("ptxas did not create a nonempty regular cubin")
            registers, spills = parse_ptxas_resources(result.stdout + result.stderr)
            records.append(
                {
                    "call": coordinate.call_role.value,
                    "cubin_sha256": _sha256(cubin),
                    "ordinal": launch.ordinal,
                    "policy": coordinate.policy.value,
                    "ptx_sha256": _sha256(ptx),
                    "registers": registers,
                    "spill_bytes": spills,
                }
            )
    if len(records) != 24:
        raise ValueError("ABI15 assembly must produce 24 records")
    return records


def validate_build_candidate(value: object) -> dict[str, Any]:
    """Validate an unsealed pre-hardware build candidate without accepting it."""
    policy = load_architecture_build_contract()["build_policy"]
    root = _closed(
        value,
        {
            "commands",
            "registration_transcript_sha256",
            "schema_version",
            "source",
            "status",
            "toolchain",
            "transports",
            "wheels",
        },
        "build candidate",
    )
    if root["schema_version"] != 1 or root["status"] != "pre_hardware_build_candidate_not_authority":
        raise ValueError("build candidate must remain explicitly non-authoritative")
    source = _closed(
        root["source"],
        {
            "harness_commit",
            "harness_tree",
            "jax_commit",
            "jax_patch_sha256",
            "jax_tree",
            "production_commit",
            "production_tree",
            "task_entrypoint",
            "xla_commit",
        },
        "build source",
    )
    for field in (
        "harness_commit",
        "harness_tree",
        "jax_commit",
        "jax_tree",
        "xla_commit",
        "production_commit",
        "production_tree",
    ):
        _revision(source[field], f"source.{field}")
    expected_source = policy["source"]
    if any(
        source[field] != expected_source[field]
        for field in ("jax_commit", "jax_tree", "production_commit", "production_tree", "task_entrypoint", "xla_commit")
    ):
        raise ValueError("build candidate differs from the pinned local source policy")
    patches = source["jax_patch_sha256"]
    expected_patches = [record["sha256"] for record in expected_source["jax_patches"]]
    if patches != expected_patches:
        raise ValueError("build source differs from the three ordered JAX patches")
    toolchain = _closed(
        root["toolchain"],
        {"bazel_sha256", "bazel_version", "ptxas_sha256", "ptxas_version", "target"},
        "toolchain",
    )
    constraints = policy["tool_constraints"]
    if (
        toolchain["target"] != constraints["ptx_target"]
        or toolchain["bazel_version"] != constraints["bazel_version"]
        or toolchain["ptxas_version"] != constraints["ptxas_version"]
    ):
        raise ValueError("build toolchain version or target drift")
    _sha(toolchain["bazel_sha256"], "Bazel digest")
    _sha(toolchain["ptxas_sha256"], "ptxas digest")
    commands = root["commands"]
    if commands != policy["commands"]:
        raise ValueError("build candidate command templates differ from the pinned local policy")
    wheels = _closed(root["wheels"], {"cuda_plugin", "cuda_pjrt", "jaxlib"}, "wheels")
    for name, wheel_value in wheels.items():
        wheel = _closed(wheel_value, {"sha256", "size"}, f"wheel {name}")
        _sha(wheel["sha256"], f"wheel {name} digest")
        if type(wheel["size"]) is not int or wheel["size"] <= 0:
            raise ValueError(f"wheel {name} must be nonempty")
    _sha(root["registration_transcript_sha256"], "registration transcript digest")
    transports = root["transports"]
    if type(transports) is not list or len(transports) != 4:
        raise ValueError("build candidate must contain four transports")
    expected_coordinates = [
        ("source_ordered", "forward", 2),
        ("source_ordered", "backward", 10),
        ("fast", "forward", 2),
        ("fast", "backward", 10),
    ]
    observed_coordinates = []
    roots = {
        "transport": set(),
        "device": set(),
        "invocation": set(),
        "bundle": set(),
        "code": set(),
    }
    total_slices = 0
    for index, transport_value in enumerate(transports):
        transport = _closed(
            transport_value,
            {
                "bundle_root",
                "call",
                "code_sha256",
                "device_root",
                "invocation_root",
                "launches",
                "policy",
                "ptx_slices",
                "transport_sha256",
            },
            f"transport[{index}]",
        )
        observed_coordinates.append((transport["policy"], transport["call"], transport["launches"]))
        for name, field in (
            ("transport", "transport_sha256"),
            ("device", "device_root"),
            ("invocation", "invocation_root"),
            ("bundle", "bundle_root"),
            ("code", "code_sha256"),
        ):
            roots[name].add(_sha(transport[field], f"transport[{index}].{field}"))
        slices = transport["ptx_slices"]
        if type(slices) is not list or len(slices) != transport["launches"]:
            raise ValueError("transport slice count differs from launch count")
        for ordinal, slice_value in enumerate(slices):
            record = _closed(
                slice_value,
                {"cubin_sha256", "ordinal", "ptx_sha256", "registers", "spill_bytes"},
                f"transport[{index}].ptx_slices[{ordinal}]",
            )
            if record["ordinal"] != ordinal or type(record["registers"]) is not int or record["registers"] <= 0:
                raise ValueError("PTX assembly ordinal or register count drift")
            if record["spill_bytes"] != 0:
                raise ValueError("PTX assembly contains spills")
            _sha(record["ptx_sha256"], "PTX digest")
            _sha(record["cubin_sha256"], "cubin digest")
        total_slices += len(slices)
    if observed_coordinates != expected_coordinates or total_slices != 24:
        raise ValueError("build candidate topology differs from the ABI15 cell")
    if any(len(values) != 4 for values in roots.values()):
        raise ValueError("policy/call build identities collapsed")
    return root
