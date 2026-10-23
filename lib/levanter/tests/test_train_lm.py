# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import dataclasses
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import equinox as eqx
import draccus
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
import safetensors.numpy
from chex import assert_trees_all_close

from haliax import Axis
from haliax.state_dict import to_torch_compatible_state_dict
from haliax.quantization import QuantizationConfig
from jax.sharding import PartitionSpec as P

import levanter.main.train_lm as train_lm
from levanter.testing import tiny_corpus
from levanter.adaptor import LoraAdaptorConfig
from levanter.checkpoint import (
    CheckpointerConfig,
    discover_checkpoint_candidates,
    latest_checkpoint_path,
    load_checkpoint,
)
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.models.llama_native import NativeLlamaSharding
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.tokenizers import load_tokenizer
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.optim.config import AdamConfig, ConstantLrSchedule
from levanter.tracker import NoopConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer_state import TrainerState, trainables_only
from levanter.testing.helpers import arrays_only, use_test_mesh
from levanter.utils.mesh import MeshConfig


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(arrays_only(tree))


def _assert_training_recorded(output_path: str) -> dict:
    """Load the JsonFileTracker record and assert training produced finite metrics.

    The smoke tests run ``train_lm.main`` end to end; the only stable observable
    effect is the metrics persisted by the tracker on ``finish()``. Asserting on
    them catches silent no-ops (no step ever logged) and NaN/inf loss blowups.
    """
    with open(os.path.join(output_path, "eval_results.json")) as f:
        metrics = json.load(f)
    assert metrics["parameter_count"] > 0
    assert "train/loss" in metrics, "per-step logging hook never fired"
    assert math.isfinite(metrics["train/loss"])
    return metrics


@pytest.mark.tpu
def test_train_lm():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,  # use default for platform
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_train_lm_fp8():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,  # use default for platform
            ),
            trainer=train_lm.TrainerConfig(
                quantization=QuantizationConfig(fp8=True),
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_train_lm_with_lora_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            adapter=LoraAdaptorConfig(r=4),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_restore_lm_model_from_partial_checkpoint_recovers_base_model():
    config = train_lm.LlamaConfig(
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
        hidden_dim=16,
        attn_backend=None,
    )
    Vocab = Axis("vocab", 32)
    base_key, wrong_base_key, adapter_key, wrong_adapter_key = jrandom.split(jrandom.PRNGKey(0), 4)

    adapter = LoraAdaptorConfig(r=4, a_init_mode="random")
    trained_model = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    wrong_resume_skeleton = adapter.apply(config.build(Vocab, key=wrong_base_key), key=wrong_adapter_key)
    correct_source_skeleton = adapter.apply(config.build(Vocab, key=base_key), key=wrong_adapter_key)
    trainable_filter = adapter.trainable_filter(trained_model)

    checkpointed_trainables = trainables_only(trained_model, trainable_filter)
    wrong_resumed_model = eqx.combine(checkpointed_trainables, wrong_resume_skeleton)
    restored_model = train_lm._restore_lm_model_from_partial_checkpoint(
        wrong_resumed_model,
        correct_source_skeleton,
        trainable_filter,
    )

    assert_trees_all_close(_array_leaves(restored_model), _array_leaves(trained_model))


def test_train_lm_direct_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 128
        seq_len = 64
        data = []
        for i in range(8):
            tokens = jnp.full((seq_len,), i % vocab_size, dtype=jnp.int32)
            data.append(GrugLmExample.causal(tokens))
        dataset = ListAsyncDataset(data)

        component = DirectDatasetComponent(datasets={"train": dataset})
        data_config = LmDataConfig(components={"direct": component}, vocab_size=vocab_size, tokenizer="passthrough")

        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=seq_len,
                hidden_dim=32,
                attn_backend=None,
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


@pytest.mark.tpu
@pytest.mark.timeout(120)
def test_native_train_lm_validates_and_resumes_identical_checkpoint_state(tmp_path):
    parsed = draccus.decode(
        train_lm.TrainLmConfig,
        {
            "model": {
                "type": "llama_native",
                "hidden_dim": 16,
                "intermediate_dim": 24,
                "num_layers": 4,
                "num_heads": 4,
                "num_kv_heads": 2,
                "max_seq_len": 16,
                "gradient_checkpointing": {"save_carries": False, "save_inputs": True, "nested": 2},
            }
        },
    )
    examples = ListAsyncDataset(
        [
            GrugLmExample.causal(
                (jnp.arange(8) + index) % 31,
                segment_ids=jnp.array([0, 0, 0, 0, 1, 1, 1, 1]),
                sliding_window=3,
            )
            for index in range(16)
        ]
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=31,
        shuffle=False,
        components={"direct": DirectDatasetComponent(datasets={"train": examples, "validation": examples})},
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    data_size = len(jax.devices()) // model_size
    batch_size = 2 * data_size
    optimizer = AdamConfig(learning_rate=1e-3, warmup=0, lr_schedule=ConstantLrSchedule(), weight_decay=0.0)

    def configuration(name, steps, *, resume=False):
        run_dir = tmp_path / name
        return dataclasses.replace(
            parsed,
            data=data,
            data_seed=7,
            train_seq_len=8,
            optimizer=optimizer,
            trainer=train_lm.TrainerConfig(
                id="native",
                seed=42,
                num_train_steps=steps,
                train_batch_size=batch_size,
                per_device_parallelism=1,
                per_device_eval_parallelism=1,
                mp=jmp.get_policy("f32"),
                max_eval_batches=1,
                steps_per_eval=1,
                mesh=MeshConfig(axes={"data": data_size, "model": model_size}, batch_axis_name="examples"),
                tracker=JsonFileTrackerConfig(output_path=str(run_dir / "tracker")),
                checkpointer=CheckpointerConfig(base_path=str(run_dir / "checkpoints"), keep=[{"every": 1}]),
                load_checkpoint=resume,
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_xla_hlo=False,
                log_jaxprs=False,
            ),
        )

    full = configuration("full", 3)
    train_lm.main(full)
    train_lm.main(configuration("resumed", 2))
    resumed = configuration("resumed", 3, resume=True)
    train_lm.main(resumed)
    for name in ("full", "resumed"):
        metrics = _assert_training_recorded(str(tmp_path / name / "tracker"))
        assert math.isfinite(metrics["eval/loss"])
    mesh = dataclasses.replace(full.trainer, use_explicit_mesh_axes=True).device_mesh
    with jax.set_mesh(mesh):
        sharding = NativeLlamaSharding(
            batch_axes=("replica_dcn", "replica", "data"), fsdp_axis="data", model_axis="model"
        )
        assert isinstance(full.model, NativeLlamaLmConfig)
        model = full.model.build(
            Axis("vocab", math.ceil(31 / model_size) * model_size), key=jrandom.PRNGKey(0), sharding=sharding
        )
        template = TrainerState.init(optimizer.build(3), model, key=jrandom.PRNGKey(0), mp=full.trainer.mp)
        expected = load_checkpoint(template, latest_checkpoint_path(tmp_path / "full/checkpoints/native"), mesh=mesh)
        actual = load_checkpoint(template, latest_checkpoint_path(tmp_path / "resumed/checkpoints/native"), mesh=mesh)
    assert int(actual.step) == 3
    for actual_leaf, expected_leaf in zip(_array_leaves(actual), _array_leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    assert actual.model.native.token_embeddings.sharding.spec == sharding.column_storage
    assert actual.model.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")

    weights_only = dataclasses.replace(
        configuration("weights-only", 1),
        optimizer=dataclasses.replace(optimizer, learning_rate=0.0),
        initialize_model_from_checkpoint_path=str(tmp_path / "full/checkpoints/native"),
    )
    train_lm.main(weights_only)
    with jax.set_mesh(mesh):
        initialized = load_checkpoint(
            template, latest_checkpoint_path(tmp_path / "weights-only/checkpoints/native"), mesh=mesh
        )
    assert int(initialized.step) == 1
    for actual_leaf, expected_leaf in zip(
        _array_leaves(initialized.model), _array_leaves(expected.model), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@dataclasses.dataclass(frozen=True)
class LocalLlamaCheckpoint:
    path: Path
    config: train_lm.LlamaConfig
    weights: dict[str, np.ndarray]


def _write_llama_checkpoint(source, config, hf_config):
    source.mkdir()
    with use_test_mesh():
        model = LlamaLMHeadModel.init(Axis("vocab", 32), config, key=jrandom.PRNGKey(13))
        weights = to_torch_compatible_state_dict(model)
    hf_config.save_pretrained(source)
    safetensors.numpy.save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    return LocalLlamaCheckpoint(source, config, weights)


@pytest.fixture(params=[False, True], ids=["untied", "tied"])
def native_hf_source(tmp_path, request):
    config = train_lm.LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        tie_word_embeddings=request.param,
    )
    return _write_llama_checkpoint(tmp_path / "hf-source", config, config.to_hf_config(32))


@pytest.mark.tpu
@pytest.mark.timeout(180)
def test_native_train_lm_hf_preserves_source_vocab_weights_and_resume(tmp_path, native_hf_source):
    source = native_hf_source.path
    source_config = native_hf_source.config
    weights = native_hf_source.weights
    model_config = NativeLlamaLmConfig.from_hf_config(source_config.to_hf_config(32))
    # Context length is a runtime limit, not a different weight architecture.
    model_config = dataclasses.replace(
        model_config, max_seq_len=32, reference_checkpoint=str(tmp_path / "unused-default-reference")
    )
    examples = ListAsyncDataset([GrugLmExample.causal((jnp.arange(8) + i) % 31) for i in range(16)])
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=31,
        shuffle=False,
        components={"direct": DirectDatasetComponent(datasets={"train": examples, "validation": examples})},
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    data_size = len(jax.devices()) // model_size
    optimizer = AdamConfig(learning_rate=1e-3, warmup=0, lr_schedule=ConstantLrSchedule(), weight_decay=0.0)

    def configuration(name, steps, *, resume=False, learning_rate=1e-3):
        run_dir = tmp_path / name
        return train_lm.TrainLmConfig(
            model=model_config,
            data=data,
            data_seed=7,
            train_seq_len=8,
            initialize_from_hf=str(source),
            optimizer=dataclasses.replace(optimizer, learning_rate=learning_rate),
            trainer=train_lm.TrainerConfig(
                id="native-hf",
                seed=42,
                num_train_steps=steps,
                train_batch_size=2 * data_size,
                per_device_parallelism=1,
                per_device_eval_parallelism=1,
                mp=jmp.get_policy("f32"),
                max_eval_batches=1,
                steps_per_eval=1,
                mesh=MeshConfig(axes={"data": data_size, "model": model_size}),
                tracker=JsonFileTrackerConfig(output_path=str(run_dir / "tracker")),
                checkpointer=CheckpointerConfig(base_path=str(run_dir / "checkpoints"), keep=[{"every": 1}]),
                load_checkpoint=resume,
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_xla_hlo=False,
                log_jaxprs=False,
            ),
        )

    frozen = configuration("frozen", 1, learning_rate=0.0)
    frozen = dataclasses.replace(
        frozen, trainer=dataclasses.replace(frozen.trainer, mp=jmp.get_policy("p=f32,c=bf16"))
    )
    train_lm.main(frozen)
    train_lm.main(configuration("full", 3))
    train_lm.main(configuration("resumed", 1))
    checkpoint_root = tmp_path / "resumed/checkpoints/native-hf"
    candidate = discover_checkpoint_candidates(checkpoint_root)[-1]
    identity = candidate.metadata["levanter_model"]
    assert identity["vocab_size"] == 32
    assert identity["architecture"]["head_dim"] == 4
    assert "reference_checkpoint" not in identity["architecture"]

    # Older checkpoints still reconstruct padded vocabulary from the explicit HF source.
    metadata_file = Path(candidate.path) / "metadata.json"
    old_metadata = json.loads(metadata_file.read_text())
    del old_metadata["levanter_model"]
    metadata_file.write_text(json.dumps(old_metadata))
    train_lm.main(configuration("resumed", 2, resume=True))

    # New metadata permits a full restore after the original pretrained source is removed.
    shutil.rmtree(source)
    resume = dataclasses.replace(
        configuration("resumed", 3, resume=True),
        model=dataclasses.replace(model_config, max_seq_len=64, head_dim=4),
    )
    candidate = discover_checkpoint_candidates(checkpoint_root)[-1]
    metadata_file = Path(candidate.path) / "metadata.json"
    current_metadata = json.loads(metadata_file.read_text())
    unknown_metadata = {**current_metadata, "levanter_model": {**current_metadata["levanter_model"], "version": 99}}
    metadata_file.write_text(json.dumps(unknown_metadata))
    with pytest.raises(ValueError, match="Unsupported checkpoint model metadata"):
        train_lm.main(resume)
    malformed_metadata = {
        **current_metadata,
        "levanter_model": {
            **current_metadata["levanter_model"],
            "architecture": {**current_metadata["levanter_model"]["architecture"], "rope_factor": True},
        },
    }
    metadata_file.write_text(json.dumps(malformed_metadata))
    with pytest.raises(ValueError, match="checkpoint architecture"):
        train_lm.main(resume)
    metadata_file.write_text(json.dumps(current_metadata))
    for incompatible in (
        dataclasses.replace(model_config, num_heads=8, num_kv_heads=4, head_dim=2),
        dataclasses.replace(model_config, rope=DefaultRotaryEmbeddingsConfig(factor=8.0)),
    ):
        with pytest.raises(ValueError, match="checkpoint architecture"):
            train_lm.main(dataclasses.replace(resume, model=incompatible))
    train_lm.main(resume)
    # A real resume wins over unused initialization paths, including missing sources.
    for field in ("initialize_model_from_checkpoint_path", "initialize_from_checkpoint_path"):
        train_lm.main(dataclasses.replace(resume, initialize_from_hf=False, **{field: str(tmp_path / "missing")}))

    trained_checkpoint = str(tmp_path / "full/checkpoints/native-hf")
    for name, field, incompatible in (
        (
            "weights-init",
            "initialize_model_from_checkpoint_path",
            dataclasses.replace(model_config, rope=DefaultRotaryEmbeddingsConfig(factor=8.0)),
        ),
        (
            "state-init",
            "initialize_from_checkpoint_path",
            dataclasses.replace(model_config, num_heads=8, num_kv_heads=4, head_dim=2),
        ),
    ):
        initialize = dataclasses.replace(
            configuration(name, 1, learning_rate=0.0),
            initialize_from_hf=False,
            **{field: trained_checkpoint},
        )
        with pytest.raises(ValueError, match="checkpoint architecture"):
            train_lm.main(dataclasses.replace(initialize, model=incompatible))
        train_lm.main(initialize)
    for name in ("frozen", "full", "resumed", "weights-init", "state-init"):
        metrics = _assert_training_recorded(str(tmp_path / name / "tracker"))
        assert math.isfinite(metrics["eval/loss"])

    mesh = dataclasses.replace(frozen.trainer, use_explicit_mesh_axes=True).device_mesh
    sharding = NativeLlamaSharding(batch_axes=("replica_dcn", "replica", "data"), fsdp_axis="data", model_axis="model")
    with jax.set_mesh(mesh):
        model = model_config.build(Axis("vocab", 32), key=jrandom.PRNGKey(0), sharding=sharding)
        template = TrainerState.init(optimizer.build(2), model, key=jrandom.PRNGKey(0), mp=frozen.trainer.mp)

        def restored(name):
            return load_checkpoint(
                template, latest_checkpoint_path(tmp_path / name / "checkpoints/native-hf"), mesh=mesh
            )

        initialized, full, resumed, weights_initialized, state_initialized = (
            restored(name) for name in ("frozen", "full", "resumed", "weights-init", "state-init")
        )

    # HF contains weights only: a new Trainer starts at zero, takes one step,
    # and preserves every source row when its learning rate is zero.
    assert int(initialized.step) == 1
    counters = [leaf for leaf in _array_leaves(initialized.opt_state) if jnp.issubdtype(leaf.dtype, jnp.integer)]
    assert counters
    for count in counters:
        np.testing.assert_array_equal(count, 1)
    assert initialized.model.Vocab.size == 32
    for name, actual in initialized.model.to_state_dict().items():
        np.testing.assert_array_equal(actual, weights[name])
    assert initialized.model.native.token_embeddings.sharding.spec == sharding.column_storage
    assert initialized.model.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")
    assert int(resumed.step) == 3
    for actual, expected in zip(_array_leaves(resumed), _array_leaves(full), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert any(not np.array_equal(value, weights[name]) for name, value in full.model.to_state_dict().items())
    for initialized_state, expected_count in ((weights_initialized, 1), (state_initialized, 4)):
        assert int(initialized_state.step) == 1
        assert initialized_state.model.Vocab.size == 32
        for actual, expected in zip(_array_leaves(initialized_state.model), _array_leaves(full.model), strict=True):
            np.testing.assert_array_equal(actual, expected)
        counts = [
            leaf for leaf in _array_leaves(initialized_state.opt_state) if jnp.issubdtype(leaf.dtype, jnp.integer)
        ]
        assert counts
        for count in counts:
            np.testing.assert_array_equal(count, expected_count)
    fresh_key = jrandom.split(jrandom.PRNGKey(42), 4)[3]
    np.testing.assert_array_equal(weights_initialized.training_key, jrandom.split(fresh_key)[1])
    np.testing.assert_array_equal(state_initialized.training_key, jrandom.split(full.training_key)[1])


def test_native_hf_rejects_equal_shape_different_attention_architecture(native_hf_source):
    source = native_hf_source.path
    source_config = native_hf_source.config
    # Both architectures have q/o size 16x16 and k/v size 8x16, but the
    # alternate interpretation splits them into different heads and RoPE pairs.
    wrong = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(source_config.to_hf_config(32)), num_heads=2, num_kv_heads=1
    )
    config = train_lm.TrainLmConfig(
        model=wrong,
        initialize_from_hf=str(source),
        data=LmDataConfig(tokenizer="passthrough", vocab_size=31),
        trainer=train_lm.TrainerConfig(
            id="incompatible-native-hf",
            tracker=NoopConfig(),
            load_checkpoint=False,
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            checkpointer=CheckpointerConfig(base_path=str(source.parent / "checkpoints")),
            mesh=MeshConfig(axes={"data": len(jax.devices()), "model": 1}),
        ),
    )
    with pytest.raises(ValueError, match="num_attention_heads mismatch"):
        train_lm.main(config)


def test_native_hf_rejects_default_rope_with_ignored_scaling_factor(native_hf_source):
    source = native_hf_source.path
    config_path = source / "config.json"
    hf_config = json.loads(config_path.read_text())
    # HF default RoPE ignores factor, while Levanter's linear RoPE applies it.
    hf_config["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0, "factor": 8.0}
    config_path.write_text(json.dumps(hf_config))
    model = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(native_hf_source.config.to_hf_config(32)),
        rope=DefaultRotaryEmbeddingsConfig(factor=8.0),
    )
    config = train_lm.TrainLmConfig(
        model=model,
        initialize_from_hf=str(source),
        data=LmDataConfig(tokenizer="passthrough", vocab_size=31),
        trainer=train_lm.TrainerConfig(
            id="incompatible-native-hf",
            tracker=NoopConfig(),
            load_checkpoint=False,
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            checkpointer=CheckpointerConfig(base_path=str(source.parent / "checkpoints")),
            mesh=MeshConfig(axes={"data": len(jax.devices()), "model": 1}),
        ),
    )
    with pytest.raises(ValueError, match="scaled RoPE must declare linear"):
        train_lm.main(config)


@pytest.mark.tpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "tied,pretrained,save_dtype", [(False, True, None), (True, True, "bfloat16"), (False, False, None)]
)
def test_native_train_lm_exports_portable_local_weights(
    tmp_path, tiny_hf_tokenizer_path, tied, pretrained, save_dtype
):
    model_config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        tie_word_embeddings=tied,
        rope=DefaultRotaryEmbeddingsConfig(factor=8.0),
        reference_checkpoint=str(tmp_path / "unused-default-reference"),
    )
    named_config = train_lm.LlamaConfig(
        **{field.name: getattr(model_config, field.name) for field in dataclasses.fields(train_lm.LlamaConfig)}
    )
    source = _write_llama_checkpoint(tmp_path / "source", named_config, model_config.to_hf_config(32))
    examples = ListAsyncDataset([GrugLmExample.causal((jnp.arange(8) + i) % 31) for i in range(16)])
    model_size = 2 if len(jax.devices()) >= 4 else 1
    data_size = len(jax.devices()) // model_size
    config = train_lm.TrainLmConfig(
        model=model_config,
        initialize_from_hf=str(source.path) if pretrained else False,
        hf_save_path=str(tmp_path / "export"),
        hf_save_steps=1,
        hf_save_dtype=save_dtype,
        train_seq_len=8,
        data=LmDataConfig(
            tokenizer=tiny_hf_tokenizer_path,
            shuffle=False,
            components={"direct": DirectDatasetComponent(datasets={"train": examples, "validation": examples})},
        ),
        optimizer=AdamConfig(learning_rate=1e-3, warmup=0, lr_schedule=ConstantLrSchedule()),
        trainer=train_lm.TrainerConfig(
            id="native-export",
            num_train_steps=2,
            train_batch_size=2 * data_size,
            per_device_parallelism=1,
            per_device_eval_parallelism=1,
            mp=jmp.get_policy("f32"),
            max_eval_batches=1,
            steps_per_eval=1,
            mesh=MeshConfig(axes={"data": data_size, "model": model_size}),
            tracker=JsonFileTrackerConfig(output_path=str(tmp_path / "tracker")),
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints"), keep=[{"every": 1}]),
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_xla_hlo=False,
            log_jaxprs=False,
        ),
    )
    train_lm.main(config)
    assert math.isfinite(_assert_training_recorded(str(tmp_path / "tracker"))["eval/loss"])
    exported = tmp_path / "export/native-export/step-1"
    fields = json.loads((exported / "config.json").read_text())
    expected_vocab = 32 if pretrained or model_size == 2 else 31
    assert fields["vocab_size"] == expected_vocab
    assert fields["architectures"] == ["LlamaForCausalLM"]
    assert fields["head_dim"] == 8
    assert fields["rope_parameters"]["rope_type"] == "linear"
    assert fields["rope_scaling"]["factor"] == 8.0
    tokenizer = config.data.the_tokenizer
    exported_tokenizer = load_tokenizer(str(exported))
    assert exported_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert exported_tokenizer.encode("token3 token4") == tokenizer.encode("token3 token4")
    converter = model_config.hf_checkpoint_converter(str(exported), tokenizer=tokenizer)
    reloaded_config = NativeLlamaLmConfig.from_hf_config(converter.default_hf_config)
    mesh = dataclasses.replace(config.trainer, use_explicit_mesh_axes=True).device_mesh
    sharding = NativeLlamaSharding(batch_axes=("replica_dcn", "replica", "data"), fsdp_axis="data", model_axis="model")
    with jax.set_mesh(mesh):
        reloaded = converter.load_pretrained(
            reloaded_config.model_type,
            config=reloaded_config,
            model_init=lambda vocab: reloaded_config.build(vocab, key=jrandom.PRNGKey(0), sharding=sharding),
        )
        saved = load_checkpoint(
            reloaded,
            latest_checkpoint_path(tmp_path / "checkpoints/native-export"),
            subpath="model",
            mesh=mesh,
        )
    saved_weights = saved.to_state_dict()
    for name, actual in reloaded.to_state_dict().items():
        expected = saved_weights[name]
        if save_dtype is not None:
            expected = expected.astype(jnp.dtype(save_dtype)).astype(expected.dtype)
        np.testing.assert_array_equal(actual, expected)
    if pretrained:
        assert any(not np.array_equal(value, source.weights[name]) for name, value in saved_weights.items())
    assert reloaded.native.token_embeddings.sharding.spec == sharding.column_storage
    assert reloaded.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")


def _frozen_lm_config(tmpdir, data_config, *, out, seed, init_from=None, ckpt_base=None):
    """A tiny one-step ``TrainLmConfig`` with LR 0 (weights frozen) and a fixed data order.

    Freezing the weights (LR 0) and pinning ``data_seed`` makes the single logged ``train/loss`` a
    deterministic function of the initial model and the data alone — so two runs log the same loss
    iff they start from the same weights, regardless of the trainer ``seed`` that would otherwise
    change the random init.
    """
    # Always base the checkpointer under tmpdir (never the default relative "checkpoints/", which would
    # pollute the cwd). The step policy saves every step so the run leaves a discoverable checkpoint;
    # append_run_id_to_base_path=False keeps it at a stable <base>/step-0 address for init_from.
    checkpointer = CheckpointerConfig(
        base_path=ckpt_base or os.path.join(out, "ckpts"),
        save_interval=datetime.timedelta(minutes=30),
        keep=[dict(every=1)],
        append_run_id_to_base_path=False,
    )
    return train_lm.TrainLmConfig(
        data=data_config,
        model=train_lm.LlamaConfig(
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32, attn_backend=None
        ),
        optimizer=AdamConfig(learning_rate=0.0),
        data_seed=0,
        initialize_model_from_checkpoint_path=init_from,
        trainer=train_lm.TrainerConfig(
            num_train_steps=1,
            train_batch_size=len(jax.devices()),
            max_eval_batches=1,
            tracker=JsonFileTrackerConfig(output_path=out),
            require_accelerator=False,
            seed=seed,
            checkpointer=checkpointer,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
    )


def _logged_loss(output_path: str) -> float:
    with open(os.path.join(output_path, "eval_results.json")) as f:
        return json.load(f)["train/loss"]


def test_train_lm_initialize_model_from_checkpoint():
    """initialize_model_from_checkpoint_path restores the model weights (fresh optimizer, step 0).

    Run once from scratch (seed 0) saving a checkpoint, then again initializing weights-only from it
    with a different trainer seed. With LR 0 and a fixed data order, the second run logs the first
    run's loss iff the weights were actually loaded; a control run from scratch with the second seed
    logs a different loss, proving the assertion is sensitive to which weights the model started from.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)

        ref_base = os.path.join(tmpdir, "ref_ckpts")
        out_ref = os.path.join(tmpdir, "ref")
        train_lm.main(_frozen_lm_config(tmpdir, data_config, out=out_ref, seed=0, ckpt_base=ref_base))
        loss_ref = _logged_loss(out_ref)

        out_init = os.path.join(tmpdir, "init")
        train_lm.main(
            _frozen_lm_config(tmpdir, data_config, out=out_init, seed=999, init_from=latest_checkpoint_path(ref_base))
        )
        loss_init = _logged_loss(out_init)

        out_ctrl = os.path.join(tmpdir, "ctrl")
        train_lm.main(_frozen_lm_config(tmpdir, data_config, out=out_ctrl, seed=999))
        loss_ctrl = _logged_loss(out_ctrl)

        assert loss_init == pytest.approx(loss_ref, abs=1e-4), "weights were not restored from the checkpoint"
        assert abs(loss_ctrl - loss_ref) > 1e-4, "control run coincidentally matched; test is not sensitive"


def test_train_lm_rejects_multiple_init_sources():
    """The three weight-init sources are mutually exclusive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32),
            initialize_from_checkpoint_path="/some/full/checkpoint",
            initialize_model_from_checkpoint_path="/some/model/checkpoint",
            trainer=train_lm.TrainerConfig(
                num_train_steps=1,
                train_batch_size=len(jax.devices()),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        with pytest.raises(ValueError, match="at most one of"):
            train_lm.main(config)


def test_train_lm_rejects_weights_only_init_with_trainer_initialize_from():
    """trainer.initialize_from is a full-state resume; combined with the weights-only field it would
    restore step > 0 and silently skip the weights-only init, so reject the combination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32),
            initialize_model_from_checkpoint_path="/some/model/checkpoint",
            trainer=train_lm.TrainerConfig(
                num_train_steps=1,
                train_batch_size=len(jax.devices()),
                initialize_from="/some/full/checkpoint",
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        with pytest.raises(ValueError, match="trainer.initialize_from"):
            train_lm.main(config)
