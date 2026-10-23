# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native normal-entrypoint training against a full Trainer checkpoint resume.

Runs two updates uninterrupted and one update followed by a restore and a second
update. Checkpoints go only to the explicit checkpoint root; local logs and
per-process shard manifests go to the output directory. FP32/highest precision
and forced-final-only checkpoint saves keep this a bounded correctness smoke.
"""

import dataclasses
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from jax.experimental import multihost_utils
from rigging.filesystem.storage_path import StoragePath

import levanter.config
from levanter.checkpoint import (
    CheckpointCandidate,
    CheckpointerConfig,
    discover_checkpoint_candidates,
    load_checkpoint,
)
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.main import train_lm
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.optim.config import AdamConfig, ConstantLrSchedule
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.jax_utils import parameter_count
from levanter.utils.mesh import MeshConfig


logger = logging.getLogger(__name__)
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HOST_SHARD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class NativeCheckpointSmokeConfig:
    checkpoint_root: str
    output_dir: str
    expected_process_count: int
    expected_global_device_count: int
    expected_local_device_count: int
    expected_backend: str
    model: NativeLlamaLmConfig
    mesh: MeshConfig
    vocab_size: int
    batch_size: int
    seq_len: int
    distributed: DistributedConfig = DistributedConfig()
    seed: int = 42
    per_device_parallelism: int = 1
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class ShardDigest:
    device_id: int
    index: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class LeafDigest:
    path: str
    dtype: str
    shape: tuple[int, ...]
    sharding: str
    shards: tuple[ShardDigest, ...]


@dataclass
class SyntheticDataset(ListAsyncDataset[GrugLmExample]):
    """Reconstructible in-memory examples; configuration logging serializes only these integers."""

    vocab_size: int
    seq_len: int
    example_count: int

    def __post_init__(self):
        # Use the enabled backend, then retain only host arrays for the loader.
        with jax.default_device(jax.local_devices()[0]):
            examples = [
                jax.device_get(
                    GrugLmExample.causal((jnp.arange(self.seq_len, dtype=jnp.int32) + index) % self.vocab_size)
                )
                for index in range(self.example_count)
            ]
        super().__init__(examples)


def _data_config(config: NativeCheckpointSmokeConfig) -> LmDataConfig:
    examples = SyntheticDataset(config.vocab_size, config.seq_len, 4 * config.batch_size)
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=config.vocab_size,
        shuffle=False,
        components={"synthetic": DirectDatasetComponent(datasets={"train": examples, "validation": examples})},
    )


def _train_config(
    config: NativeCheckpointSmokeConfig, data: LmDataConfig, name: str, steps: int, *, resume: bool = False
) -> train_lm.TrainLmConfig:
    return train_lm.TrainLmConfig(
        model=config.model,
        data=data,
        data_seed=config.seed,
        train_seq_len=config.seq_len,
        optimizer=AdamConfig(
            learning_rate=config.learning_rate, warmup=0, lr_schedule=ConstantLrSchedule(), weight_decay=0.0
        ),
        trainer=TrainerConfig(
            id=name,
            seed=config.seed,
            tracker=NoopConfig(),
            log_dir=Path(config.output_dir) / f"process-{jax.process_index()}" / "logs",
            load_checkpoint=resume,
            num_train_steps=steps,
            train_batch_size=config.batch_size,
            per_device_parallelism=config.per_device_parallelism,
            per_device_eval_parallelism=config.per_device_parallelism,
            mp=jmp.get_policy("f32"),
            max_eval_batches=1,
            steps_per_eval=1,
            mesh=config.mesh,
            use_explicit_mesh_axes=True,
            require_accelerator=config.expected_backend != "cpu",
            distributed=DistributedConfig(initialize_jax_distributed=False),
            checkpointer=CheckpointerConfig(
                base_path=str(StoragePath(config.checkpoint_root) / name),
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
                delete_old_temp_checkpoints=False,
            ),
            jax_config={"jax_default_matmul_precision": "highest"},
            log_xla_hlo=False,
            log_jaxprs=False,
        ),
    )


def _state_template(config: NativeCheckpointSmokeConfig, full: train_lm.TrainLmConfig) -> TrainerState:
    sharding = train_lm._native_llama_sharding(full.trainer)
    return eqx.filter_eval_shape(
        lambda: TrainerState.init(
            full.optimizer.build(2),
            config.model.build(Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(0), sharding=sharding),
            key=jax.random.PRNGKey(0),
            mp=full.trainer.mp,
        )
    )


def _shard_sha256(array: jax.Array) -> str:
    if array.nbytes > _MAX_HOST_SHARD_BYTES:
        raise ValueError(f"Addressable shard exceeds the {_MAX_HOST_SHARD_BYTES}-byte host copy limit")
    digest = hashlib.sha256()
    # Transfer one capped addressable shard, never a global tensor. Chunk only
    # host views so different offsets do not compile separate device slices.
    flat = np.asarray(array).reshape(-1)
    chunk_elements = max(1, _HASH_CHUNK_BYTES // array.dtype.itemsize)
    for offset in range(0, flat.size, chunk_elements):
        host_chunk = flat[offset : offset + chunk_elements]
        if np.issubdtype(host_chunk.dtype, np.floating) and not np.all(np.isfinite(host_chunk)):
            raise ValueError("Checkpoint contains nonfinite floating-point state")
        digest.update(memoryview(host_chunk.view(np.uint8)))
    return digest.hexdigest()


def _state_digests(state: TrainerState) -> tuple[LeafDigest, ...]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(state, eqx.is_array))
    records = []
    for path, array in leaves:
        shards = tuple(
            ShardDigest(
                device_id=shard.device.id,
                index=repr(shard.index),
                shape=tuple(shard.data.shape),
                sha256=_shard_sha256(shard.data),
            )
            for shard in sorted(array.addressable_shards, key=lambda shard: shard.device.id)
        )
        records.append(
            LeafDigest(
                path=jax.tree_util.keystr(path),
                dtype=str(array.dtype),
                shape=tuple(array.shape),
                sharding=str(array.sharding),
                shards=shards,
            )
        )
    return tuple(records)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _checkpoint_digests(
    config: NativeCheckpointSmokeConfig,
    template: TrainerState,
    mesh: jax.sharding.Mesh,
    checkpoint: CheckpointCandidate,
) -> tuple[LeafDigest, ...]:
    if config.model.checkpoint_vocab_size(checkpoint.metadata) != config.vocab_size:
        raise ValueError("Smoke checkpoint must retain the configured model vocabulary and semantic metadata")
    with jax.set_mesh(mesh):
        state = load_checkpoint(template, checkpoint.path, mesh=mesh)
    optimizer_counts = [
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_flatten_with_path(eqx.filter(state.opt_state, eqx.is_array))[0]
        if np.issubdtype(value.dtype, np.integer)
    ]
    if not optimizer_counts:
        raise ValueError("Adam checkpoint has no optimizer counters")
    for name, counter in [("step", state.step), *optimizer_counts]:
        if any(not np.all(np.asarray(shard.data) == 2) for shard in counter.addressable_shards):
            raise ValueError(f"Expected step and Adam counters equal to two, got {name} from {checkpoint.path}")
    records = _state_digests(state)
    # Only Python digests leave this function; the restored device state can be
    # released before the other checkpoint is loaded.
    return records


def main(config: NativeCheckpointSmokeConfig):
    if StoragePath(config.output_dir).scheme is not None:
        raise ValueError("output_dir must be a plain local filesystem path, without a URL scheme")
    output = Path(config.output_dir).expanduser().resolve()
    checkpoint_root = StoragePath(config.checkpoint_root)
    if checkpoint_root.scheme == "file":
        raise ValueError("Use a plain local filesystem checkpoint_root instead of a file:// URL")
    if not checkpoint_root.is_remote:
        checkpoint_path = Path(config.checkpoint_root).expanduser().resolve()
        if output.is_relative_to(checkpoint_path) or checkpoint_path.is_relative_to(output):
            raise ValueError("output_dir and checkpoint_root must be separate, non-overlapping directories")
    if config.expected_process_count * config.expected_local_device_count != config.expected_global_device_count:
        raise ValueError("Expected process/local/global device counts are inconsistent")
    jax.config.update("jax_default_matmul_precision", "highest")
    config.distributed.initialize()
    observed = {
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "backend": jax.default_backend(),
    }
    expected = {
        "process_count": config.expected_process_count,
        "global_device_count": config.expected_global_device_count,
        "local_device_count": config.expected_local_device_count,
        "backend": config.expected_backend,
    }
    if observed != expected:
        raise ValueError(f"Topology mismatch: expected {expected}, observed {observed}")
    if StoragePath(config.checkpoint_root).exists():
        raise FileExistsError(f"checkpoint_root must be a fresh dedicated prefix: {config.checkpoint_root}")
    multihost_utils.sync_global_devices("native-checkpoint-smoke-fresh-root")
    output.mkdir(parents=True, exist_ok=True)
    rank = jax.process_index()
    result_path = output / f"result-process-{rank}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite a previous smoke result: {result_path}")
    (output / f"configuration-process-{rank}.yaml").write_text(draccus.dump(config))

    data = _data_config(config)
    full = _train_config(config, data, "full", 2)
    mesh = full.trainer.device_mesh
    sharding = train_lm._native_llama_sharding(full.trainer)
    if sharding.model_axis is not None and config.vocab_size % mesh.shape[sharding.model_axis]:
        raise ValueError("Smoke vocabulary must be divisible by the model mesh size; no vocabulary padding is allowed")
    with jax.set_mesh(mesh):
        template = _state_template(config, full)
    template_arrays = [
        leaf for leaf in jax.tree.leaves(template.saveable_state) if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    planned_max_shard_bytes = max(
        math.prod(leaf.sharding.shard_shape(leaf.shape) if leaf.sharding is not None else leaf.shape)
        * np.dtype(leaf.dtype).itemsize
        for leaf in template_arrays
    )
    if planned_max_shard_bytes > _MAX_HOST_SHARD_BYTES:
        raise ValueError("Planned checkpoint shard exceeds the bounded host-copy limit")
    checkpoint_bytes = sum(
        math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
        for leaf in jax.tree.leaves(template.saveable_state)
        if isinstance(leaf, jax.ShapeDtypeStruct)
    )
    provenance = {
        **observed,
        "process_index": rank,
        "devices": [
            {
                "id": device.id,
                "process_index": device.process_index,
                "kind": device.device_kind,
                "platform": device.platform,
            }
            for device in jax.devices()
        ],
        "jax_version": jax.__version__,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "parameter_count": parameter_count(template.model),
        "checkpoint_array_bytes": checkpoint_bytes,
        "expected_checkpoint_writes": 3,
        "planned_max_shard_bytes": planned_max_shard_bytes,
        "host_shard_copy_limit_bytes": _MAX_HOST_SHARD_BYTES,
        "hash_chunk_bytes": _HASH_CHUNK_BYTES,
        "expected_total_checkpoint_array_bytes": 3 * checkpoint_bytes,
    }
    _write_json(output / f"preflight-process-{rank}.json", provenance)
    logger.info("Native checkpoint smoke preflight: %s", provenance)

    # Fail here rather than letting the normal tracker silently lose its config artifact.
    (output / f"training-configuration-process-{rank}.yaml").write_text(draccus.dump(full))
    train_lm.main(full)
    train_lm.main(_train_config(config, data, "resumed", 1))
    train_lm.main(_train_config(config, data, "resumed", 2, resume=True))
    full_candidates = discover_checkpoint_candidates(str(StoragePath(config.checkpoint_root) / "full"))
    resumed_candidates = discover_checkpoint_candidates(str(StoragePath(config.checkpoint_root) / "resumed"))
    if [candidate.step for candidate in full_candidates] != [1] or [
        candidate.step for candidate in resumed_candidates
    ] != [0, 1]:
        raise ValueError("Expected exactly three completed checkpoints: full step-1 and resumed steps 0/1")
    expected_records = _checkpoint_digests(config, template, mesh, full_candidates[-1])
    actual_records = _checkpoint_digests(config, template, mesh, resumed_candidates[-1])
    expected_json = [dataclasses.asdict(record) for record in expected_records]
    actual_json = [dataclasses.asdict(record) for record in actual_records]
    _write_json(output / f"full-shards-process-{rank}.json", expected_json)
    _write_json(output / f"resumed-shards-process-{rank}.json", actual_json)
    matches = expected_records == actual_records
    all_match = bool(np.all(multihost_utils.process_allgather(np.asarray(matches))))
    report = {
        **provenance,
        "checkpoint_paths": [candidate.path for candidate in (*full_candidates, *resumed_candidates)],
        "checkpoint_count": len(full_candidates) + len(resumed_candidates),
        "local_shards_compared": sum(len(record.shards) for record in actual_records),
        "observed_max_shard_bytes": max(
            math.prod(shard.shape) * np.dtype(record.dtype).itemsize
            for record in actual_records
            for shard in record.shards
        ),
        "local_full_state_sha256": hashlib.sha256(json.dumps(expected_json, sort_keys=True).encode()).hexdigest(),
        "local_resumed_state_sha256": hashlib.sha256(json.dumps(actual_json, sort_keys=True).encode()).hexdigest(),
        "exact_state_match": matches,
        "all_processes_match": all_match,
    }
    _write_json(result_path, report)
    if not all_match:
        raise ValueError(
            "Resumed training differs from uninterrupted training; inspect the per-process shard manifests"
        )
    multihost_utils.sync_global_devices("native-checkpoint-smoke-complete")


if __name__ == "__main__":
    levanter.config.main(main)()
