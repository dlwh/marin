# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the rebuilt CUDA plugin's ABI15 Shuttle v2 registration."""

import argparse
import json
import os
from pathlib import Path
from types import ModuleType

from abi15_h100_evidence import require_sealed_hardware_authority
from shuttle_jaxlib_gpu_ffi_registration import file_sha256, load_extension
from shuttle_jaxlib_gpu_ffi_registration_abi11 import (
    register_through_pjrt,
    validated_shuttle_bundle,
)

TARGET = "shuttle.gpu.executable_bundle.v2"


def registration_transcript(
    module: ModuleType,
    extension_path: Path,
    plugin_mappings: tuple[tuple[int, int, str, int], ...],
) -> dict[str, object]:
    """Return the ABI15 transcript while reusing the reviewed v2 PJRT checks."""
    extension_path = extension_path.resolve()
    extension_stat = extension_path.stat()
    expected_device = f"{os.major(extension_stat.st_dev):02x}:{os.minor(extension_stat.st_dev):02x}"
    if any(device != expected_device or inode != extension_stat.st_ino for _, _, device, inode in plugin_mappings):
        raise RuntimeError("plugin mapping device/inode differs from loaded extension")
    handlers = module.ffi_handlers()
    bundle = validated_shuttle_bundle(handlers)
    registration = register_through_pjrt(module, bundle)
    return {
        "exported_targets": sorted(handlers),
        "pipeline_abi_version": 15,
        "plugin_extension_sha256": file_sha256(extension_path),
        "plugin_mappings": [
            {
                "device": device,
                "end": end,
                "inode": inode,
                "path": str(extension_path),
                "sha256": file_sha256(extension_path),
                "start": start,
            }
            for start, end, device, inode in plugin_mappings
        ],
        "registration": registration,
        "schema_version": 1,
        "upstream_target_count": len(handlers) - 1,
    }


def _extension_mappings(extension_path: Path) -> tuple[tuple[int, int, str, int], ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("ABI15 registration evidence requires Linux /proc/self/maps")
    ranges = []
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped = Path(fields[5].removesuffix(" (deleted)")).resolve()
        if mapped != extension_path:
            continue
        start, end = fields[0].split("-", maxsplit=1)
        ranges.append((int(start, 16), int(end, 16), fields[3], int(fields[4])))
    if not ranges:
        raise RuntimeError("loaded CUDA plugin extension has no process mapping")
    return tuple(ranges)


def main() -> int:
    require_sealed_hardware_authority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    extension_path = arguments.plugin_extension.resolve()
    module = load_extension(extension_path)
    transcript = registration_transcript(module, extension_path, _extension_mappings(extension_path))
    arguments.output.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
