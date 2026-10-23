# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import dataclasses
import json
import math
import os
import tempfile

import equinox as eqx
import draccus
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
from chex import assert_trees_all_close

from haliax import Axis
from haliax.quantization import QuantizationConfig
from jax.sharding import PartitionSpec as P

import levanter.main.train_lm as train_lm
from levanter.testing import tiny_corpus
from levanter.adaptor import LoraAdaptorConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path, load_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.models.llama_native import NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.optim.config import AdamConfig, ConstantLrSchedule
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer_state import TrainerState, trainables_only
from levanter.testing.helpers import arrays_only
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
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 2,
                "max_seq_len": 16,
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
