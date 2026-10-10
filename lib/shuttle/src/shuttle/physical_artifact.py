# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover generic GPU physical compilers for Shuttle's JAX boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points
from threading import RLock

GPU_PHYSICAL_ARTIFACT_COMPILER_GROUP = "shuttle.gpu_physical_artifact_compilers"

GpuInternalEventLegalizationRecords = Sequence[Mapping[str, object]]
GpuPhysicalArtifactResult = bytes | tuple[bytes, GpuInternalEventLegalizationRecords] | None
GpuPhysicalArtifactProvider = Callable[[str], GpuPhysicalArtifactResult]

_LOCK = RLock()
_PROVIDERS: list[GpuPhysicalArtifactProvider] = []
_ENTRY_POINTS_LOADED = False
_BRIDGE_BOUND = False


def register_gpu_physical_artifact_provider(provider: GpuPhysicalArtifactProvider) -> None:
    """Add a generic serialized-request provider to the process-wide chain."""
    if not callable(provider):
        raise TypeError("GPU physical artifact provider must be callable")
    with _LOCK:
        if provider not in _PROVIDERS:
            _PROVIDERS.append(provider)
    _bind_jaxlib_bridge()


def activate_gpu_physical_artifact_providers(*, discover: bool = True) -> None:
    """Discover installed providers and bind their composite JAX callback."""
    global _ENTRY_POINTS_LOADED

    if discover:
        with _LOCK:
            if not _ENTRY_POINTS_LOADED:
                discovered = sorted(
                    entry_points(group=GPU_PHYSICAL_ARTIFACT_COMPILER_GROUP),
                    key=lambda entry_point: (entry_point.name, entry_point.value),
                )
                for entry_point in discovered:
                    provider = entry_point.load()
                    if not callable(provider):
                        raise TypeError(
                            f"Shuttle GPU physical artifact entry point {entry_point.name!r} must be callable"
                        )
                    if provider not in _PROVIDERS:
                        _PROVIDERS.append(provider)
                _ENTRY_POINTS_LOADED = True
    _bind_jaxlib_bridge()


def _bind_jaxlib_bridge() -> None:
    global _BRIDGE_BOUND

    with _LOCK:
        if _BRIDGE_BOUND or not _PROVIDERS:
            return
        from jaxlib import _jax  # noqa: PLC0415

        bridge = getattr(_jax, "_shuttle_physical_artifact", None)
        if bridge is None:
            return
        bridge.register(_compile_gpu_physical_artifact)
        _BRIDGE_BOUND = True


def _compile_gpu_physical_artifact(serialized: str) -> GpuPhysicalArtifactResult:
    with _LOCK:
        providers = tuple(_PROVIDERS)
    for provider in providers:
        result = provider(serialized)
        if result is None:
            continue
        if type(result) is bytes:
            return result
        if type(result) is tuple and len(result) == 2 and type(result[0]) is bytes and type(result[1]) in (list, tuple):
            return result
        raise TypeError(
            "GPU physical artifact provider must return bytes, " "(bytes, internal_event_legalizations), or None"
        )
    return None
