# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from shuttle_jaxlib_gpu_ffi_registration import (
    STAGES,
    PjrtApiPrefix,
    PjrtGpuCustomCall,
    PjrtGpuRegisterCustomCallArgs,
    capsule,
    capsule_pointer,
)
from shuttle_jaxlib_gpu_ffi_registration_abi15 import TARGET, main, registration_transcript


def _bundle() -> dict[str, object]:
    return {
        **{stage: capsule(0x1000 + index * 0x100) for index, stage in enumerate(STAGES)},
        "api_version": 1,
        "traits": 0,
    }


def _plugin() -> SimpleNamespace:
    def register_custom_call_target(
        api_capsule: object,
        target: str,
        bundle: dict[str, object],
        platform: str,
        *,
        api_version: int,
        traits: int,
    ) -> None:
        assert platform == "CUDA"
        api = ctypes.cast(capsule_pointer(api_capsule), ctypes.POINTER(PjrtApiPrefix)).contents
        extension = ctypes.cast(api.extension_start, ctypes.POINTER(PjrtGpuCustomCall)).contents
        pointers = [capsule_pointer(bundle[stage]) for stage in STAGES]
        encoded_target = target.encode()
        arguments = PjrtGpuRegisterCustomCallArgs(
            struct_size=ctypes.sizeof(PjrtGpuRegisterCustomCallArgs),
            function_name=encoded_target,
            function_name_size=len(encoded_target),
            api_version=api_version,
            handler_instantiate=pointers[0],
            handler_prepare=pointers[1],
            handler_initialize=pointers[2],
            handler_execute=pointers[3],
        )
        extension.custom_call(ctypes.pointer(arguments))

    return SimpleNamespace(
        ffi_handlers=lambda: {"upstream": {}, TARGET: _bundle()},
        register_custom_call_target=register_custom_call_target,
    )


def test_abi15_registration_transcript_rotates_pipeline_identity_and_binds_plugin(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "cuda_plugin_extension.so"
    plugin.write_bytes(b"plugin")

    plugin_stat = plugin.stat()
    device = f"{os.major(plugin_stat.st_dev):02x}:{os.minor(plugin_stat.st_dev):02x}"
    transcript = registration_transcript(_plugin(), plugin, ((0x1000, 0x5000, device, plugin_stat.st_ino),))

    assert transcript["pipeline_abi_version"] == 15
    assert transcript["plugin_extension_sha256"] == "5e689e2b01672bf33996e75d5e372ff60c536ce1599a1458e867cd8f4bef5160"
    assert transcript["exported_targets"] == [TARGET, "upstream"]
    assert transcript["upstream_target_count"] == 1
    assert transcript["registration"]["target"] == TARGET
    assert transcript["plugin_mappings"] == [
        {
            "device": device,
            "end": 0x5000,
            "inode": plugin_stat.st_ino,
            "path": str(plugin),
            "sha256": transcript["plugin_extension_sha256"],
            "start": 0x1000,
        }
    ]


def test_abi15_registration_main_stops_before_loading_or_writing_while_unsealed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        main()

    assert not any(tmp_path.iterdir())
