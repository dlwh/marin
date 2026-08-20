# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal ordinary-JAX H100 subject for Track 1 mixed Map islands."""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jaxlib import _jax

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options

CACHE_HIT_EVENT = "/jax/compilation_cache/cache_hits"
SHAPES = {2: (2, 3), 3: (2, 3, 5)}
MAP_SOURCE_OPERATIONS = (0, 1, 2, 4, 5, 6, 7)


class Mode(StrEnum):
    DISABLED = "disabled"
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True, order=True)
class Coordinate:
    rank: int
    mode: Mode

    def __post_init__(self) -> None:
        if self.rank not in SHAPES:
            raise ValueError("Track 1 Map rank must be 2 or 3")

    @classmethod
    def all(cls) -> tuple["Coordinate", ...]:
        return tuple(cls(rank, mode) for rank in SHAPES for mode in Mode)

    @property
    def label(self) -> str:
        return f"rank{self.rank}_{self.mode.value}"


@dataclass(frozen=True)
class ObserverIdentity:
    policy_sha256: str
    tuning_sha256: str


@dataclass(frozen=True)
class ObserverEvidence:
    invocation_id: int
    generated_bundle_sha256: str
    final_module_sha256: str


@dataclass(frozen=True)
class CoordinateResult:
    coordinate: Coordinate
    primal_sha256: str
    gradient_sha256: str
    cache_entry: tuple[str, int, str]
    observer: ObserverEvidence | None


class CacheHitCounter:
    def __init__(self) -> None:
        self.count = 0

    def observe(self, event: str, **metadata: str | int) -> None:
        del metadata
        if event == CACHE_HIT_EVENT:
            self.count += 1


def tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def compiler_options_for(mode: Mode) -> dict[str, bool | str] | None:
    if mode is Mode.DISABLED:
        return None
    numerics = Numerics.SOURCE_ORDERED if mode is Mode.SOURCE_ORDERED else Numerics.FAST
    return compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=numerics,
        tuning=tuning(),
    )


def observer_identity(coordinate: Coordinate) -> ObserverIdentity:
    options = compiler_options_for(coordinate.mode)
    if options is None:
        raise ValueError("disabled compilation has no Shuttle observer identity")
    canonical = options["xla_shuttle_options"]
    if type(canonical) is not str:
        raise TypeError("Shuttle options must use canonical JSON text")
    payload = json.loads(canonical)
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        hashlib.sha256(canonical.encode()).hexdigest(),
        hashlib.sha256(canonical_tuning.encode()).hexdigest(),
    )


def deterministic_input(rank: int) -> np.ndarray:
    shape = SHAPES.get(rank)
    if shape is None:
        raise ValueError("Track 1 Map rank must be 2 or 3")
    size = int(np.prod(shape))
    return np.linspace(-3.0, 3.0, size, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(shape)


def mixed_map_value_and_gradient(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the mixed Map value and its analytic nonzero-input gradient."""
    value = jnp.abs((-x.astype(jnp.float32)).astype(jnp.bfloat16))
    gradient = (value.astype(jnp.float32) / x.astype(jnp.float32)).astype(jnp.bfloat16)
    return value, gradient


def _sha_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"observer {name} must be a lowercase SHA-256")
    return value


def validate_observer_snapshot(
    coordinate: Coordinate,
    snapshot: tuple[tuple[object, ...], ...],
) -> ObserverEvidence | None:
    """Validate selected/excluded attribution and final erasure for one compile."""
    if coordinate.mode is Mode.DISABLED:
        if snapshot:
            raise ValueError("disabled compilation emitted Shuttle observer events")
        return None
    if len(snapshot) != 3 or any(type(event) is not tuple or len(event) != 11 for event in snapshot):
        raise ValueError("enabled compilation must emit three closed observer records")
    if tuple(event[1] for event in snapshot) != ("algebra_coverage", "generated_bundle", "final_erasure"):
        raise ValueError("enabled compilation emitted the wrong observer phases")
    invocation_ids = {event[0] for event in snapshot}
    if len(invocation_ids) != 1 or type(snapshot[0][0]) is not int or snapshot[0][0] <= 0:
        raise ValueError("observer phases do not share one positive invocation")
    identity = observer_identity(coordinate)
    for event in snapshot:
        if event[2:5] != (coordinate.mode.value, identity.policy_sha256, identity.tuning_sha256):
            raise ValueError("observer policy identity differs from compiler options")
        if event[10] != "":
            raise ValueError("successful observer event carries a failure pass")
    membership = snapshot[0][5]
    if type(membership) is not str or membership != snapshot[1][5]:
        raise ValueError("observer Map region membership changed before bundle generation")
    for operation in MAP_SOURCE_OPERATIONS:
        if f"#shuttle.source_ref<0, 0, {operation}, 0>" not in membership:
            raise ValueError("observer omitted a selected Map source")
    if "#shuttle.source_ref<0, 0, 3, 0>" in membership:
        raise ValueError("observer selected the unsupported abs source")
    manifest = snapshot[0][6]
    if (
        type(manifest) is not str
        or manifest != snapshot[1][6]
        or "#shuttle.source_ref<0, 0, 3, 0>" not in manifest
        or "unsupported_operation" not in manifest
    ):
        raise ValueError("observer did not attribute abs to XLA")
    unsupported = _sha(snapshot[0][7], "unsupported fingerprint")
    if snapshot[1][7] != unsupported:
        raise ValueError("observer unsupported fingerprint changed")
    if snapshot[0][9] is not False or snapshot[1][9] is not False:
        raise ValueError("observer erased Shuttle semantics before the final phase")
    if snapshot[2][5:8] != ("", "", "") or snapshot[2][9] is not True:
        raise ValueError("observer final phase did not erase Shuttle semantics")
    return ObserverEvidence(
        invocation_id=snapshot[0][0],
        generated_bundle_sha256=_sha(snapshot[1][8], "generated bundle fingerprint"),
        final_module_sha256=_sha(snapshot[2][8], "final module fingerprint"),
    )


def cache_attribution(
    records: dict[Coordinate, tuple[str, int, str]],
) -> dict[Coordinate, tuple[str, int, str]]:
    if set(records) != set(Coordinate.all()) or len(records) != 6:
        raise ValueError("cache attribution must cover all six Track 1 coordinates")
    values = tuple(records.values())
    if len(set(values)) != len(values):
        raise ValueError("cache attribution entries must be unique")
    for name, size, digest in values:
        if not name.endswith("-cache") or type(size) is not int or size <= 0:
            raise ValueError("cache attribution contains an invalid entry")
        _sha(digest, "cache entry")
    return records


def cache_snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    snapshot = {}
    for path in directory.glob("*-cache"):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("persistent cache contains a non-regular entry")
        snapshot[path.name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return snapshot


def configure_cache(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compilation_cache_max_size", -1)


def _bridge() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib lacks the test-only Shuttle observer bridge")
    return bridge


def require_h100() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("Track 1 subject requires the GPU backend")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError("Track 1 subject requires one visible H100")


def _execute(coordinate: Coordinate) -> tuple[tuple[np.ndarray, np.ndarray], tuple[tuple[object, ...], ...]]:
    value = deterministic_input(coordinate.rank)
    options = compiler_options_for(coordinate.mode)
    compiled = (
        jax.jit(mixed_map_value_and_gradient)
        if options is None
        else jax.jit(mixed_map_value_and_gradient, compiler_options=options)
    )
    with _bridge().subscribe() as capture:
        primal, gradient = compiled(value)
        result = (np.asarray(primal.block_until_ready()), np.asarray(gradient.block_until_ready()))
        snapshot = tuple(capture.snapshot())
    return result, snapshot


def run_matrix(
    cache_directory: Path,
    *,
    reuse: bool,
    expected_attribution: dict[Coordinate, tuple[str, int, str]] | None = None,
) -> tuple[CoordinateResult, ...]:
    """Execute six real GPU compilations in one populate or reuse process."""
    require_h100()
    configure_cache(cache_directory)
    initial = cache_snapshot(cache_directory)
    if (reuse and len(initial) != 6) or (not reuse and initial):
        raise RuntimeError("persistent cache does not match the requested worker role")
    if reuse:
        if expected_attribution is None:
            raise RuntimeError("cache reuse requires populate attribution")
        cache_attribution(expected_attribution)
        if {name: (size, digest) for name, size, digest in expected_attribution.values()} != initial:
            raise RuntimeError("cache reuse bytes differ from populate attribution")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    baselines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    attributed: dict[Coordinate, tuple[str, int, str]] = {}
    results = []
    for coordinate in Coordinate.all():
        before = cache_snapshot(cache_directory)
        arrays, snapshot = _execute(coordinate)
        if reuse:
            if snapshot:
                raise RuntimeError("cache reuse unexpectedly reran the Shuttle transform")
            observer = None
        else:
            observer = validate_observer_snapshot(coordinate, snapshot)
        if coordinate.mode is Mode.DISABLED:
            baselines[coordinate.rank] = arrays
        else:
            for actual, expected in zip(arrays, baselines[coordinate.rank], strict=True):
                if not np.array_equal(actual, expected):
                    raise RuntimeError("enabled output or gradient differs from disabled JAX")
        after = cache_snapshot(cache_directory)
        if reuse:
            if after != before:
                raise RuntimeError("cache reuse mutated persistent entries")
            assert expected_attribution is not None
            cache_entry = expected_attribution[coordinate]
        else:
            new_names = set(after) - set(before)
            if len(new_names) != 1:
                raise RuntimeError("each Track 1 coordinate must create one cache entry")
            name = new_names.pop()
            size, digest = after[name]
            cache_entry = (name, size, digest)
            attributed[coordinate] = cache_entry
        results.append(
            CoordinateResult(
                coordinate,
                _sha_array(arrays[0]),
                _sha_array(arrays[1]),
                cache_entry,
                observer,
            )
        )
    final = cache_snapshot(cache_directory)
    if len(final) != 6:
        raise RuntimeError("Track 1 matrix must own exactly six cache entries")
    if reuse:
        if counter.count != 6:
            raise RuntimeError("Track 1 reuse must report six persistent-cache hits")
    else:
        if counter.count != 0:
            raise RuntimeError("Track 1 populate unexpectedly hit persistent cache")
        cache_attribution(attributed)
    return tuple(results)


def result_document(results: tuple[CoordinateResult, ...], *, reuse: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "worker": "reuse" if reuse else "populate",
        "pid": os.getpid(),
        "coordinates": [
            {
                "rank": result.coordinate.rank,
                "mode": result.coordinate.mode.value,
                "primal_sha256": result.primal_sha256,
                "gradient_sha256": result.gradient_sha256,
                "cache_entry": result.cache_entry,
                "observer": (
                    None
                    if result.observer is None
                    else {
                        "invocation_id": result.observer.invocation_id,
                        "generated_bundle_sha256": result.observer.generated_bundle_sha256,
                        "final_module_sha256": result.observer.final_module_sha256,
                    }
                ),
            }
            for result in results
        ],
    }
