# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the rebuilt CUDA plugin's ABI 11 Shuttle typed-FFI bundle."""

import argparse
import ctypes
import json
from pathlib import Path
from types import ModuleType

from shuttle_jaxlib_gpu_ffi_registration import (
    GPU_CUSTOM_CALL_EXTENSION,
    STAGES,
    PjrtApiPrefix,
    PjrtExtensionBase,
    PjrtGpuCustomCall,
    PjrtGpuRegisterCustomCallArgs,
    RegisterCustomCall,
    capsule,
    capsule_pointer,
    file_sha256,
    load_extension,
)

TARGET = "shuttle.gpu.executable_bundle.v2"


def validated_shuttle_bundle(handlers: object) -> dict[str, object]:
    """Return the one complete ABI 11 Shuttle handler bundle."""
    if type(handlers) is not dict or TARGET not in handlers:
        targets = sorted(handlers) if type(handlers) is dict else []
        raise RuntimeError(f"missing ABI 11 Shuttle CUDA typed-FFI target: {targets}")
    bundle = handlers[TARGET]
    if type(bundle) is not dict or set(bundle) != {*STAGES, "api_version", "traits"}:
        raise RuntimeError("CUDA plugin exported an unknown Shuttle handler field")
    if bundle["api_version"] != 1 or bundle["traits"] != 0:
        raise RuntimeError("CUDA plugin exported the wrong Shuttle typed-FFI metadata")
    if any(bundle[stage] is None for stage in STAGES):
        raise RuntimeError("CUDA plugin did not export one complete Shuttle handler bundle")
    return bundle


def register_through_pjrt(module: ModuleType, bundle: dict[str, object]) -> dict[str, object]:
    """Route the exported bundle through the plugin's real PJRT entry point."""
    observations: list[tuple[str, int, tuple[int, ...]]] = []

    @RegisterCustomCall
    def register(args_pointer: ctypes.POINTER(PjrtGpuRegisterCustomCallArgs)) -> None:
        args = args_pointer.contents
        target = ctypes.string_at(args.function_name, args.function_name_size).decode()
        observations.append(
            (
                target,
                args.api_version,
                tuple(
                    int(pointer)
                    for pointer in (
                        args.handler_instantiate,
                        args.handler_prepare,
                        args.handler_initialize,
                        args.handler_execute,
                    )
                ),
            )
        )
        return None

    extension = PjrtGpuCustomCall(
        base=PjrtExtensionBase(
            struct_size=ctypes.sizeof(PjrtGpuCustomCall),
            type=GPU_CUSTOM_CALL_EXTENSION,
            next=ctypes.POINTER(PjrtExtensionBase)(),
        ),
        custom_call=register,
    )
    api = PjrtApiPrefix(
        struct_size=ctypes.sizeof(PjrtApiPrefix),
        extension_start=ctypes.pointer(extension.base),
    )
    module.register_custom_call_target(
        capsule(ctypes.addressof(api)),
        TARGET,
        bundle,
        "CUDA",
        api_version=1,
        traits=0,
    )

    if len(observations) != 1:
        raise RuntimeError("CUDA plugin did not make one PJRT registration call")
    target, api_version, pointers = observations[0]
    exported_pointers = tuple(capsule_pointer(bundle[stage]) for stage in STAGES)
    if target != TARGET or api_version != 1 or pointers != exported_pointers:
        raise RuntimeError("CUDA plugin registered the wrong Shuttle typed-FFI bundle")
    if len(set(pointers)) != len(STAGES) or any(pointer == 0 for pointer in pointers):
        raise RuntimeError("CUDA plugin exported duplicate or null Shuttle stage pointers")
    return {
        "target": target,
        "api_version": api_version,
        "traits": 0,
        "stage_pointers": dict(zip(STAGES, pointers, strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    extension_path = args.plugin_extension.resolve()
    module = load_extension(extension_path)
    handlers = module.ffi_handlers()
    bundle = validated_shuttle_bundle(handlers)
    registration = register_through_pjrt(module, bundle)
    result = {
        "schema_version": 1,
        "pipeline_abi_version": 11,
        "plugin_extension_sha256": file_sha256(extension_path),
        "exported_targets": sorted(handlers),
        "upstream_target_count": len(handlers) - 1,
        "registration": registration,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
