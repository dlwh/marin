# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the rebuilt CUDA plugin's ABI15 Shuttle v2 registration."""

import argparse
import json
from pathlib import Path
from types import ModuleType

from shuttle_jaxlib_gpu_ffi_registration import file_sha256, load_extension
from shuttle_jaxlib_gpu_ffi_registration_abi11 import (
    register_through_pjrt,
    validated_shuttle_bundle,
)

TARGET = "shuttle.gpu.executable_bundle.v2"


def registration_transcript(module: ModuleType, extension_path: Path) -> dict[str, object]:
    """Return the ABI15 transcript while reusing the reviewed v2 PJRT checks."""
    handlers = module.ffi_handlers()
    bundle = validated_shuttle_bundle(handlers)
    registration = register_through_pjrt(module, bundle)
    return {
        "exported_targets": sorted(handlers),
        "pipeline_abi_version": 15,
        "plugin_extension_sha256": file_sha256(extension_path),
        "registration": registration,
        "schema_version": 1,
        "upstream_target_count": len(handlers) - 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    extension_path = arguments.plugin_extension.resolve()
    transcript = registration_transcript(load_extension(extension_path), extension_path)
    arguments.output.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
