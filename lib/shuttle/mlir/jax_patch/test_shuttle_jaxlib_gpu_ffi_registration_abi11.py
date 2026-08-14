# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ctypes
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
from shuttle_jaxlib_gpu_ffi_registration_abi11 import (
    TARGET,
    register_through_pjrt,
    validated_shuttle_bundle,
)


def _bundle() -> dict[str, object]:
    return {
        **{stage: capsule(0x1000 + index * 0x100) for index, stage in enumerate(STAGES)},
        "api_version": 1,
        "traits": 0,
    }


def _plugin(*, target_override: str | None = None, swap_stages: bool = False) -> SimpleNamespace:
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
        assert traits == 0
        api_pointer = capsule_pointer(api_capsule)
        api = ctypes.cast(api_pointer, ctypes.POINTER(PjrtApiPrefix)).contents
        extension = ctypes.cast(api.extension_start, ctypes.POINTER(PjrtGpuCustomCall)).contents
        pointers = [capsule_pointer(bundle[stage]) for stage in STAGES]
        if swap_stages:
            pointers[0], pointers[1] = pointers[1], pointers[0]
        registered_target = (target_override or target).encode()
        args = PjrtGpuRegisterCustomCallArgs(
            struct_size=ctypes.sizeof(PjrtGpuRegisterCustomCallArgs),
            function_name=registered_target,
            function_name_size=len(registered_target),
            api_version=api_version,
            handler_instantiate=pointers[0],
            handler_prepare=pointers[1],
            handler_initialize=pointers[2],
            handler_execute=pointers[3],
        )
        extension.custom_call(ctypes.pointer(args))

    return SimpleNamespace(register_custom_call_target=register_custom_call_target)


def test_abi11_bundle_registers_exact_v2_target_and_four_stages() -> None:
    bundle = validated_shuttle_bundle({"upstream": {}, TARGET: _bundle()})

    registration = register_through_pjrt(_plugin(), bundle)

    assert registration["target"] == TARGET
    assert registration["api_version"] == 1
    assert registration["traits"] == 0
    assert set(registration["stage_pointers"]) == set(STAGES)


@pytest.mark.parametrize(
    ("handlers", "message"),
    [
        ({}, "missing ABI 11"),
        ({TARGET: {**_bundle(), "unknown": 1}}, "unknown Shuttle handler field"),
        ({TARGET: {**_bundle(), "api_version": 0}}, "wrong Shuttle typed-FFI metadata"),
        ({TARGET: {**_bundle(), "execute": None}}, "complete Shuttle handler bundle"),
        ({TARGET: _bundle(), "shuttle.gpu.executable_bundle.v1": _bundle()}, "another Shuttle"),
        ({TARGET: _bundle(), "shuttle.gpu.executable_bundle.v3": _bundle()}, "another Shuttle"),
    ],
)
def test_abi11_bundle_rejects_missing_or_corrupt_exports(handlers: object, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        validated_shuttle_bundle(handlers)


@pytest.mark.parametrize(
    "plugin",
    [
        _plugin(target_override="shuttle.gpu.executable_bundle.v1"),
        _plugin(swap_stages=True),
    ],
)
def test_abi11_registration_rejects_target_or_stage_drift(plugin: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="registered the wrong"):
        register_through_pjrt(plugin, _bundle())
