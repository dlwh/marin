# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import dataclasses
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import equinox as eqx
import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import optax
import pytest
import safetensors.numpy
from transformers import LlamaConfig as HfLlamaConfig
from chex import assert_trees_all_close

from haliax import Axis
from haliax.state_dict import to_torch_compatible_state_dict
from haliax.quantization import QuantizationConfig
from jax.sharding import PartitionSpec as P

import levanter.main.train_lm as train_lm
import levanter.main.eval_lm as eval_lm
import levanter.main.viz_logprobs as viz_logprobs
from levanter.testing import tiny_corpus
from levanter.adaptor import LoraAdaptorConfig
from levanter.adaptor.lora import lora_state_dict, merge_lora_modules, unwrap_lora_modules
from levanter.checkpoint import (
    CheckpointerConfig,
    discover_checkpoint_candidates,
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.compat.hf_checkpoints import RepoRef
from levanter.distributed import DistributedConfig
from levanter.models.llama_native import NativeLlamaSharding
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.tokenizers import load_tokenizer
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.llama_native_lm import NativeLlamaLmConfig, native_llama_sharding
from levanter.utils.activation import ActivationFunctionEnum
from levanter.model_loading import load_hf_checkpoint, load_levanter_checkpoint
from levanter.optim.config import AdamConfig, ConstantLrSchedule
from levanter.tracker import NoopConfig, NoopTracker, current_tracker
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer_state import TrainerState, trainables_only
from levanter.quantization import cast_model_preserving_quantization_state
from levanter.testing.helpers import arrays_only, use_test_mesh
from levanter.testing.hf_checkpoints import write_llama_checkpoint
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode


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
@pytest.mark.parametrize("activation_function", ["silu", "gelu_new"])
def test_native_train_lm_validates_and_resumes_identical_checkpoint_state(tmp_path, activation_function):
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
                "activation_function": activation_function,
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
    incompatible = dataclasses.replace(
        resumed,
        model=dataclasses.replace(
            resumed.model,
            activation_function=(
                ActivationFunctionEnum.gelu_new if activation_function == "silu" else ActivationFunctionEnum.silu
            ),
        ),
    )
    with pytest.raises(ValueError, match="checkpoint architecture"):
        train_lm.main(incompatible)
    if activation_function != "silu":
        checkpoint = discover_checkpoint_candidates(tmp_path / "resumed/checkpoints/native")[-1]
        metadata_path = Path(checkpoint.path) / "metadata.json"
        original_metadata = metadata_path.read_text()
        metadata = json.loads(original_metadata)
        del metadata["levanter_model"]["architecture"]["activation_function"]
        metadata_path.write_text(json.dumps(metadata))
        try:
            with pytest.raises(ValueError, match="checkpoint architecture"):
                train_lm.main(resumed)
        finally:
            metadata_path.write_text(original_metadata)
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
    return write_llama_checkpoint(tmp_path / "hf-source", config, config.to_hf_config(32), key=jrandom.PRNGKey(13))


@pytest.mark.tpu
@pytest.mark.parametrize("activation_function", [ActivationFunctionEnum.silu, ActivationFunctionEnum.gelu_new])
def test_native_shared_loaders_preserve_master_weights_vocab_and_semantics(
    tmp_path, native_hf_source, tiny_hf_tokenizer_path, activation_function
):
    source_config = dataclasses.replace(native_hf_source.config, activation_function=activation_function)
    source_config.to_hf_config(32).save_pretrained(native_hf_source.path)
    model_config = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(source_config.to_hf_config(32)),
        max_seq_len=32,
        reference_checkpoint=str(tmp_path / "unused-default-reference"),
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    trainer = train_lm.TrainerConfig(
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": len(jax.devices()) // model_size, "model": model_size}),
    )
    tokenizer = load_tokenizer(tiny_hf_tokenizer_path)
    loader_args = dict(
        axis_mapping=trainer.parameter_axis_mapping,
        compute_axis_mapping=trainer.compute_axis_mapping,
        batch_axis_name=trainer.batch_axis_name,
    )
    batch, position = Axis("batch", 2 * trainer.device_mesh.shape["data"]), Axis("position", 4)
    tokens = hax.named(
        (jnp.arange(batch.size * position.size).reshape(batch.size, position.size) * 3) % 31, (batch, position)
    )
    optimizer_config = AdamConfig(
        learning_rate=0.1, weight_decay=0.2, warmup=0, lr_schedule="constant", max_grad_norm=None
    )
    with current_tracker(NoopTracker()), use_test_mesh(), hax.axis_mapping({}):
        named = source_config.build(Axis("vocab", 32), key=jrandom.PRNGKey(13))
        for name, value in to_torch_compatible_state_dict(named).items():
            np.testing.assert_array_equal(value, native_hf_source.weights[name])
        expected_logits = np.asarray(eqx.filter_jit(lambda m: m(tokens, AttentionMask.causal()).array)(named))
        optimizer = optimizer_config.build(4)
        named_updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, named), optimizer.init(named), params=named)
        expected_updates = to_torch_compatible_state_dict(named_updates)
    checkpoint_root = tmp_path / "native-checkpoints"
    with current_tracker(NoopTracker()), trainer.use_device_mesh():
        model = load_hf_checkpoint(
            model_config, str(native_hf_source.path), tokenizer=tokenizer, compute_dtype=jnp.bfloat16, **loader_args
        )
        assert model.Vocab.size == 32
        assert isinstance(model.config, NativeLlamaLmConfig)
        assert model.config.max_seq_len == 32
        assert model.native.token_embeddings.sharding.spec == P("model", "data")
        actual = to_torch_compatible_state_dict(model)
        for name, expected in native_hf_source.weights.items():
            np.testing.assert_array_equal(actual[name], expected)
            assert actual[name].dtype == np.float32
        np.testing.assert_allclose(
            eqx.filter_jit(lambda m: m(tokens, AttentionMask.causal()).array)(model),
            expected_logits,
            atol=1e-5,
            rtol=1e-5,
        )
        optimizer = optimizer_config.build(4)
        # Real zero-gradient updates isolate default decay on the source-loaded tree.
        updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, model), optimizer.init(model), params=model)
        for name, update in to_torch_compatible_state_dict(updates).items():
            np.testing.assert_allclose(update, expected_updates[name], atol=1e-5, rtol=1e-5, err_msg=name)
        save_checkpoint({"model": model}, 1, str(checkpoint_root / "step-1"), metadata=model.checkpoint_metadata())
        shutil.rmtree(native_hf_source.path)
        restored = load_levanter_checkpoint(
            model_config,
            str(checkpoint_root),
            Vocab=Axis("vocab", len(tokenizer)),
            key=jrandom.PRNGKey(0),
            **loader_args,
        )
        for name, expected in actual.items():
            np.testing.assert_array_equal(to_torch_compatible_state_dict(restored)[name], expected)
        assert restored.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")
        np.testing.assert_allclose(
            eqx.filter_jit(lambda m: m(tokens, AttentionMask.causal()).array)(restored),
            expected_logits,
            atol=1e-5,
            rtol=1e-5,
        )
        # Equal q/k/v/out tensor shapes do not imply equal attention geometry.
        different_heads = dataclasses.replace(model_config, num_heads=2, num_kv_heads=1, head_dim=8)
        for incompatible in (
            different_heads,
            dataclasses.replace(model_config, rope=DefaultRotaryEmbeddingsConfig(factor=8)),
            dataclasses.replace(
                model_config,
                activation_function=(
                    ActivationFunctionEnum.gelu_new
                    if activation_function == ActivationFunctionEnum.silu
                    else ActivationFunctionEnum.silu
                ),
            ),
        ):
            with pytest.raises(ValueError, match="architecture"):
                load_levanter_checkpoint(
                    incompatible,
                    str(checkpoint_root),
                    Vocab=Axis("vocab", len(tokenizer)),
                    key=jrandom.PRNGKey(0),
                    **loader_args,
                )


@pytest.mark.tpu
@pytest.mark.parametrize("bookkeeping", ["missing-step", "invalid-step", "invalid-timestamp"])
def test_native_shared_checkpoint_validates_identity_without_discovery_bookkeeping(tmp_path, bookkeeping):
    model_config = NativeLlamaLmConfig(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        activation_function=ActivationFunctionEnum.gelu_new,
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    trainer = train_lm.TrainerConfig(
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": len(jax.devices()) // model_size, "model": model_size}),
    )
    loader_args = dict(
        Vocab=Axis("vocab", 32),
        key=jrandom.PRNGKey(1),
        axis_mapping=trainer.parameter_axis_mapping,
        compute_axis_mapping=trainer.compute_axis_mapping,
        batch_axis_name=trainer.batch_axis_name,
    )
    checkpoint = tmp_path / "checkpoint"
    with trainer.use_device_mesh():
        sharding = native_llama_sharding(
            trainer.parameter_axis_mapping, trainer.compute_axis_mapping, trainer.batch_axis_name
        )
        model = model_config.build(Axis("vocab", 32), key=jrandom.PRNGKey(0), sharding=sharding)
        expected = to_torch_compatible_state_dict(model)
        batch, position = Axis("batch", 2 * trainer.device_mesh.shape["data"]), Axis("position", 4)
        tokens = hax.named(
            jnp.arange(batch.size * position.size).reshape(batch.size, position.size) % 31, (batch, position)
        )
        expected_logits = np.asarray(eqx.filter_jit(lambda m: m(tokens, AttentionMask.causal()).array)(model))
        save_checkpoint({"model": model}, 1, str(checkpoint), metadata=model.checkpoint_metadata())
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if bookkeeping == "missing-step":
            del metadata["step"]
        elif bookkeeping == "invalid-step":
            metadata["step"] = "invalid"
        else:
            metadata["timestamp"] = "invalid"
        metadata_path.write_text(json.dumps(metadata))
        assert discover_checkpoint_candidates(checkpoint) == []
        with pytest.raises(ValueError, match="architecture"):
            load_levanter_checkpoint(
                dataclasses.replace(model_config, activation_function=ActivationFunctionEnum.silu),
                str(checkpoint),
                **loader_args,
            )
        invalid_version = {**metadata, "levanter_model": {**metadata["levanter_model"], "version": 99}}
        metadata_path.write_text(json.dumps(invalid_version))
        with pytest.raises(ValueError, match="Unsupported checkpoint model metadata"):
            load_levanter_checkpoint(model_config, str(checkpoint), **loader_args)
        metadata_path.write_text(json.dumps(metadata))
        restored = load_levanter_checkpoint(
            model_config, str(checkpoint), **{**loader_args, "Vocab": Axis("vocab", 31)}
        )
        assert restored.Vocab.size == 32
        # Missing semantic identity deliberately trusts the ORIGINAL nondefault
        # configuration and complete vocabulary; it does not infer SiLU.
        without_identity = {key: value for key, value in metadata.items() if key != "levanter_model"}
        metadata_path.write_text(json.dumps(without_identity))
        legacy = load_levanter_checkpoint(model_config, str(checkpoint), **loader_args)
        for loaded in (restored, legacy):
            for name, value in to_torch_compatible_state_dict(loaded).items():
                np.testing.assert_array_equal(value, expected[name])
            np.testing.assert_allclose(
                eqx.filter_jit(lambda m: m(tokens, AttentionMask.causal()).array)(loaded),
                expected_logits,
                atol=1e-5,
                rtol=1e-5,
            )
        metadata_path.write_text("{")
        with pytest.raises(json.JSONDecodeError):
            load_levanter_checkpoint(model_config, str(checkpoint), **loader_args)


def test_named_shared_hf_loader_retains_source_config_reconstruction(native_hf_source, tiny_hf_tokenizer_path):
    tokenizer = load_tokenizer(tiny_hf_tokenizer_path)
    supplied_config = dataclasses.replace(
        native_hf_source.config,
        hidden_dim=32,
        num_layers=1,
        reference_checkpoint=tiny_hf_tokenizer_path,
    )
    with use_test_mesh():
        model = load_hf_checkpoint(
            supplied_config,
            str(native_hf_source.path),
            tokenizer=tokenizer,
            compute_dtype=jnp.float32,
            axis_mapping={},
        )
        actual = to_torch_compatible_state_dict(model)
    assert isinstance(model, LlamaLMHeadModel)
    assert model.config.hidden_dim != supplied_config.hidden_dim
    for name, expected in native_hf_source.weights.items():
        np.testing.assert_array_equal(actual[name], expected)


@pytest.mark.tpu
@pytest.mark.parametrize("source_kind", ["hf", "levanter"])
def test_native_eval_and_visualization_use_shared_loaders(
    tmp_path, native_hf_source, tiny_hf_tokenizer_path, source_kind
):
    model_config = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(native_hf_source.config.to_hf_config(32)),
        reference_checkpoint=str(tmp_path / "unused-default-reference"),
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    data_size = len(jax.devices()) // model_size
    trainer = train_lm.TrainerConfig(
        id="native-eval",
        per_device_eval_parallelism=1,
        max_eval_batches=1,
        mesh=MeshConfig(axes={"data": data_size, "model": model_size}, batch_axis_name="examples"),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path / "eval")),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_xla_hlo=False,
        log_jaxprs=False,
    )
    examples = ListAsyncDataset([GrugLmExample.causal((jnp.arange(8) + i) % 31) for i in range(data_size)])
    data = LmDataConfig(
        tokenizer=tiny_hf_tokenizer_path,
        shuffle=False,
        components={"direct": DirectDatasetComponent(datasets={"validation": examples})},
    )
    source = str(native_hf_source.path)
    if source_kind == "levanter":
        with dataclasses.replace(trainer, use_explicit_mesh_axes=True).use_device_mesh():
            model = load_hf_checkpoint(
                model_config,
                source,
                tokenizer=data.the_tokenizer,
                compute_dtype=jnp.float32,
                axis_mapping=trainer.parameter_axis_mapping,
                compute_axis_mapping=trainer.compute_axis_mapping,
                batch_axis_name=trainer.batch_axis_name,
            )
            source = str(tmp_path / "checkpoint")
            save_checkpoint({"model": model}, 1, source, metadata=model.checkpoint_metadata())
        shutil.rmtree(native_hf_source.path)
    eval_lm.main(
        eval_lm.EvalLmConfig(
            model=model_config,
            data=data,
            trainer=trainer,
            max_eval_length=8,
            hf_checkpoint=RepoRef.from_string(source) if source_kind == "hf" else None,
            checkpoint_path=source if source_kind == "levanter" else None,
        )
    )
    metrics = json.loads((tmp_path / "eval/eval_results.json").read_text())
    assert math.isfinite(metrics["eval/loss"])
    viz_logprobs.main(
        viz_logprobs.VizLmConfig(
            model=model_config,
            data=data,
            trainer=dataclasses.replace(trainer, tracker=NoopConfig()),
            max_eval_length=8,
            checkpoint_path=source,
            checkpoint_is_hf=source_kind == "hf",
            comparison_model_path=source,
            comparison_is_hf=source_kind == "hf",
            path=str(tmp_path / "visualization"),
            num_docs=data_size,
        )
    )
    html = (tmp_path / "visualization/direct.html").read_text()
    diff = (tmp_path / "visualization/direct_diff.html").read_text()
    log_probs = [float(value) for value in re.findall("data-lp='([^']+)'", html)]
    differences = [float(value) for value in re.findall("data-diff='([^']+)'", diff)]
    assert len(log_probs) == 8 * data_size
    assert np.isfinite(log_probs).all()
    np.testing.assert_array_equal(differences, np.zeros(8 * data_size))


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

    # Inferred full-state resumes need HF metadata, but not its pretrained weights.
    inferred_model = NativeLlamaLmConfig(
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        reference_checkpoint=str(source),
    )

    def inferred_configuration(name, steps, *, resume=False):
        return dataclasses.replace(
            configuration(name, steps, resume=resume),
            model=inferred_model,
            initialize_from_hf=True,
            use_hf_model_config=True,
        )

    inferred_full = train_lm.main(inferred_configuration("inferred-full", 3))
    train_lm.main(inferred_configuration("inferred-resumed", 2))
    (source / "model.safetensors").unlink()
    inferred_resumed = train_lm.main(inferred_configuration("inferred-resumed", 3, resume=True))
    np.testing.assert_array_equal(inferred_full.loss, inferred_resumed.loss)
    for actual, expected in zip(
        _array_leaves(inferred_resumed.state), _array_leaves(inferred_full.state), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

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
        for name in ("inferred-full", "inferred-resumed"):
            saved = restored(name)
            for actual, expected in zip(_array_leaves(saved), _array_leaves(inferred_full.state), strict=True):
                np.testing.assert_array_equal(actual, expected)

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
    source = write_llama_checkpoint(
        tmp_path / "source", named_config, model_config.to_hf_config(32), key=jrandom.PRNGKey(13)
    )
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


@pytest.mark.tpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dropout", [0.0, 0.25])
@pytest.mark.parametrize("hf_config_source", ["explicit", "inferred_string", "inferred_bool"])
def test_native_train_lm_lora_hf_update_resume_and_local_exports(
    tmp_path, native_hf_source, tiny_hf_tokenizer_path, hf_config_source, dropout
):
    source_checkpoint = native_hf_source
    inferred = hf_config_source != "explicit"
    if inferred:
        source_config = dataclasses.replace(
            native_hf_source.config,
            head_dim=8,
            rope=DefaultRotaryEmbeddingsConfig(theta=50000, factor=4),
        )
        source_hf_config = HfLlamaConfig(
            **{
                **native_hf_source.config.to_hf_config(32).to_dict(),
                "head_dim": 8,
                "max_position_embeddings": 16,
                "rope_theta": 50000,
                "rope_scaling": {"rope_type": "linear", "factor": 4},
                "rope_parameters": {"rope_type": "linear", "rope_theta": 50000, "factor": 4},
            }
        )
        source_checkpoint = write_llama_checkpoint(
            tmp_path / "hf-inferred-source", source_config, source_hf_config, key=jrandom.PRNGKey(13)
        )
    source = source_checkpoint.path
    model_config = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(HfLlamaConfig.from_pretrained(source)),
        reference_checkpoint=str(tmp_path / "unused-default-reference"),
    )
    initial_model_config = model_config
    if inferred:
        model_config = dataclasses.replace(
            model_config, attn_backend=AttentionBackend.VANILLA, gradient_checkpointing="save_all"
        )
        initial_model_config = NativeLlamaLmConfig(
            hidden_dim=8,
            intermediate_dim=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=4,
            attn_backend=model_config.attn_backend,
            gradient_checkpointing=model_config.gradient_checkpointing,
            reference_checkpoint=(
                str(source) if hf_config_source == "inferred_bool" else model_config.reference_checkpoint
            ),
            tokenizer=tiny_hf_tokenizer_path,
        )
    examples = ListAsyncDataset([GrugLmExample.causal((jnp.arange(8) + i) % 31) for i in range(16)])
    data = LmDataConfig(
        tokenizer=tiny_hf_tokenizer_path,
        shuffle=False,
        components={"direct": DirectDatasetComponent(datasets={"train": examples})},
    )
    model_size = 2 if jax.device_count() >= 4 else 1
    data_size = jax.device_count() // model_size
    optimizer = AdamConfig(learning_rate=1e-3, warmup=0, lr_schedule=ConstantLrSchedule(), weight_decay=0.0)
    adapter = LoraAdaptorConfig(r=3, alpha=6, dropout=dropout, a_init_mode="random")

    def configuration(name, steps, resume=False):
        directory = tmp_path / name
        return train_lm.TrainLmConfig(
            model=initial_model_config,
            data=data,
            data_seed=7,
            train_seq_len=8,
            initialize_from_hf=True if hf_config_source == "inferred_bool" else str(source),
            use_hf_model_config=inferred,
            adapter=adapter,
            optimizer=optimizer,
            peft_save_path=str(directory / "peft"),
            merged_hf_save_path=str(directory / "merged"),
            hf_save_steps=1,
            trainer=train_lm.TrainerConfig(
                id="native-lora",
                seed=42,
                num_train_steps=steps,
                train_batch_size=2 * data_size,
                per_device_parallelism=1,
                mp=jmp.get_policy("f32"),
                max_eval_batches=0,
                mesh=MeshConfig(axes={"data": data_size, "model": model_size}),
                tracker=JsonFileTrackerConfig(output_path=str(directory / "tracker")),
                checkpointer=CheckpointerConfig(base_path=str(directory / "checkpoints"), keep=[{"every": 1}]),
                load_checkpoint=resume,
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_xla_hlo=False,
                log_jaxprs=False,
            ),
        )

    full_info = train_lm.main(configuration("full", 3))
    train_lm.main(configuration("resumed", 2))
    if inferred:
        # Same-shaped weights must not be reinterpreted through changed HF metadata on resume.
        metadata_path = source / "config.json"
        original_metadata = metadata_path.read_text()
        incompatible = json.loads(original_metadata)
        incompatible.update(num_attention_heads=8, num_key_value_heads=4, head_dim=4)
        metadata_path.write_text(json.dumps(incompatible))
        try:
            with pytest.raises(ValueError, match="checkpoint architecture"):
                train_lm.main(configuration("resumed", 3, resume=True))
        finally:
            metadata_path.write_text(original_metadata)
    resumed_info = train_lm.main(configuration("resumed", 3, resume=True))
    np.testing.assert_array_equal(full_info.loss, resumed_info.loss)
    for actual, expected in zip(_array_leaves(resumed_info.state), _array_leaves(full_info.state), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for info in (full_info, resumed_info):
        actual_model = info.state.model
        assert isinstance(actual_model.config, NativeLlamaLmConfig)
        assert actual_model.Vocab.size == 32 > len(data.the_tokenizer)
        assert actual_model.config.head_dim == model_config.head_dim
        assert actual_model.config.max_seq_len == source_checkpoint.config.max_seq_len
        assert actual_model.config.rope == source_checkpoint.config.rope
        if inferred:
            assert actual_model.native.config.head_dim == 8
            assert actual_model.native.layers.stacked.self_attn.q_proj.wrapped.weight.shape == (2, 32, 16)
            assert actual_model.config.max_seq_len == 16
            assert actual_model.config.attn_backend == AttentionBackend.VANILLA
            assert actual_model.config.gradient_checkpointing == "save_all"
            assert actual_model.config.reference_checkpoint == str(source)
            assert actual_model.config.tokenizer == tiny_hf_tokenizer_path
        for name, value in unwrap_lora_modules(actual_model).to_state_dict().items():
            np.testing.assert_array_equal(value, source_checkpoint.weights[name])
    mesh = dataclasses.replace(configuration("full", 3).trainer, use_explicit_mesh_axes=True).device_mesh
    sharding = native_llama_sharding(
        configuration("full", 3).trainer.parameter_axis_mapping,
        configuration("full", 3).trainer.compute_axis_mapping,
        "batch",
    )
    tokenizer = data.the_tokenizer
    with jax.set_mesh(mesh):
        base = load_hf_checkpoint(
            model_config,
            str(source),
            tokenizer=tokenizer,
            compute_dtype=jnp.float32,
            axis_mapping=configuration("full", 3).trainer.parameter_axis_mapping,
            compute_axis_mapping=configuration("full", 3).trainer.compute_axis_mapping,
            batch_axis_name="batch",
        )
        skeleton = adapter.apply(base, key=jrandom.PRNGKey(0))
        template = TrainerState.init(
            optimizer.build(3),
            skeleton,
            key=jrandom.PRNGKey(0),
            mp=jmp.get_policy("f32"),
            is_trainable=adapter.trainable_filter(skeleton),
        )

        def restored(name):
            return load_checkpoint(
                template.saveable_state, latest_checkpoint_path(tmp_path / name / "checkpoints/native-lora"), mesh=mesh
            )

        full, resumed = restored("full"), restored("resumed")
        assert int(full.step) == int(resumed.step) == 3
        for actual, expected in zip(_array_leaves(full), _array_leaves(resumed), strict=True):
            np.testing.assert_array_equal(actual, expected)
        counts = [v for v in _array_leaves(full.opt_state) if jnp.issubdtype(v.dtype, jnp.integer)]
        assert counts
        for count in counts:
            np.testing.assert_array_equal(count, 3)
        adam_states = jax.tree.leaves(full.opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
        adam_states = [node for node in adam_states if isinstance(node, optax.ScaleByAdamState)]
        assert adam_states
        for adam in adam_states:
            for moments in (adam.mu, adam.nu):
                factors = moments.native.layers.stacked.self_attn.q_proj.lora
                assert np.any(factors.lora_A.weight != 0)
                assert np.any(factors.lora_B.weight != 0)
        trained = inference_mode(
            train_lm._restore_lm_model_from_partial_checkpoint(
                full.model, skeleton, adapter.trainable_filter(skeleton)
            ),
            True,
        )
        for name, actual in unwrap_lora_modules(trained).to_state_dict().items():
            np.testing.assert_array_equal(actual, source_checkpoint.weights[name])
        assert any(np.any(value != 0) for key, value in lora_state_dict(trained).items() if ".lora_A." in key)
        assert trained.native.layers.stacked.self_attn.q_proj.lora.lora_A.weight.sharding.spec == P(None, None, "data")
        wrong_base = model_config.build(Axis("vocab", 32), key=jrandom.PRNGKey(999), sharding=sharding)
        wrong_skeleton = adapter.apply(wrong_base, key=jrandom.PRNGKey(0))
        wrong = inference_mode(
            train_lm._restore_lm_model_from_partial_checkpoint(
                full.model, wrong_skeleton, adapter.trainable_filter(wrong_skeleton)
            ),
            True,
        )
        tokens = hax.named(
            jnp.arange(8 * data_size).reshape(2 * data_size, 4) % 31,
            (Axis("batch", 2 * data_size), Axis("position", 4)),
        )
        assert not np.allclose(
            trained(tokens, AttentionMask.causal()).array, wrong(tokens, AttentionMask.causal()).array
        )
        merged = merge_lora_modules(trained)
        peft_path = tmp_path / "full/peft/native-lora/step-2"
        peft_config = json.loads((peft_path / "adapter_config.json").read_text())
        assert peft_config["base_model_name_or_path"] == str(source)
        assert peft_config["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        exported = safetensors.numpy.load_file(peft_path / "adapter_model.safetensors")
        for key, value in lora_state_dict(trained).items():
            np.testing.assert_array_equal(exported[key], value)
        converter = model_config.hf_checkpoint_converter(
            str(tmp_path / "full/merged/native-lora/step-2"), tokenizer=tokenizer
        )
        reloaded = converter.load_pretrained(
            model_config.model_type,
            config=model_config,
            model_init=lambda vocab: model_config.build(vocab, key=jrandom.PRNGKey(1), sharding=sharding),
        )
        for name, value in reloaded.to_state_dict().items():
            np.testing.assert_array_equal(value, merged.to_state_dict()[name])
        tokens = hax.named(
            jnp.arange(8 * data_size).reshape(2 * data_size, 4) % 31,
            (Axis("batch", 2 * data_size), Axis("position", 4)),
        )
        np.testing.assert_allclose(
            trained(tokens, AttentionMask.causal()).array,
            reloaded(tokens, AttentionMask.causal()).array,
            atol=1e-5,
            rtol=1e-5,
        )
    for name in ("full", "resumed"):
        _assert_training_recorded(str(tmp_path / name / "tracker"))


@pytest.mark.timeout(180)
@pytest.mark.parametrize("policy", ["f32", "params=float32,compute=bfloat16,output=float32"])
@pytest.mark.parametrize("microbatch", [False, True])
def test_native_train_lm_fp8_checkpoint_resume_preserves_next_update(tmp_path, policy, microbatch):
    examples = ListAsyncDataset(
        [GrugLmExample.causal((jnp.arange(4) + index) % 16) for index in range(8 * len(jax.devices()))]
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
        components={"direct": DirectDatasetComponent(datasets={"train": examples})},
    )
    parallel_size = 2 if len(jax.devices()) % 4 == 0 else 1
    optimizer = AdamConfig(learning_rate=1e-3, warmup=0, lr_schedule=ConstantLrSchedule(), weight_decay=0.0)
    mp = jmp.get_policy(policy)

    def configuration(name, steps, *, resume=False):
        return train_lm.TrainLmConfig(
            model=NativeLlamaLmConfig(
                hidden_dim=8,
                intermediate_dim=16,
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=4,
                gradient_checkpointing="nested",
            ),
            data=data,
            data_seed=7,
            train_seq_len=4,
            optimizer=optimizer,
            trainer=train_lm.TrainerConfig(
                id="native-fp8",
                seed=42,
                num_train_steps=steps,
                train_batch_size=4 * len(jax.devices()) // parallel_size,
                per_device_parallelism=2 if microbatch else -1,
                mp=mp,
                quantization=QuantizationConfig(fp8=True, amax_history_length=4),
                mesh=MeshConfig(
                    axes={"replica": -1, "data": parallel_size, "model": parallel_size}, batch_axis_name="examples"
                ),
                tracker=NoopConfig(),
                checkpointer=CheckpointerConfig(base_path=str(tmp_path / name), keep=[{"every": 1}]),
                load_checkpoint=resume,
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_xla_hlo=False,
                log_jaxprs=False,
            ),
        )

    full = configuration("full", 2)
    train_lm.main(full)
    train_lm.main(configuration("resumed", 1))
    train_lm.main(configuration("resumed", 2, resume=True))
    mesh = dataclasses.replace(full.trainer, use_explicit_mesh_axes=True).device_mesh
    sharding = native_llama_sharding(
        full.trainer.parameter_axis_mapping, full.trainer.compute_axis_mapping, full.trainer.batch_axis_name
    )
    with jax.set_mesh(mesh):
        model = full.model.build(Axis("vocab", 16), key=jrandom.PRNGKey(0), sharding=sharding)
        template = TrainerState.init(
            optimizer.build(2), model, key=jrandom.PRNGKey(0), mp=mp, quantization=full.trainer.quantization
        )
        expected = load_checkpoint(template, latest_checkpoint_path(tmp_path / "full/native-fp8"), mesh=mesh)
        actual = load_checkpoint(template, latest_checkpoint_path(tmp_path / "resumed/native-fp8"), mesh=mesh)
        first = load_checkpoint(
            template, discover_checkpoint_candidates(tmp_path / "resumed/native-fp8")[0].path, mesh=mesh
        )
    assert int(actual.step) == 2
    for actual_leaf, expected_leaf in zip(_array_leaves(actual), _array_leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    assert actual.model.native.layers.stacked.self_attn.q_proj.weight.dtype == jnp.float32
    assert not np.array_equal(
        actual.model.native.layers.stacked.self_attn.q_proj.weight,
        first.model.native.layers.stacked.self_attn.q_proj.weight,
    )
    operation = actual.model.native.layers.stacked.self_attn.q_proj.dot_general
    for name in ("input_scale", "kernel_scale", "output_grad_scale"):
        assert getattr(operation, name).shape == (2, 1)
        assert getattr(operation, name).dtype == jnp.float32
    for name in ("input_amax_history", "kernel_amax_history", "output_grad_amax_history"):
        value = getattr(operation, name)
        assert value.shape == (2, 4)
        assert value.dtype == jnp.float32
        assert np.any(value != 0)
    # The production fused CE takes raw head weights, so attaching its operation
    # preserves logits quantization without inventing a quantized training GEMM.
    head_state = actual.model.native.lm_head.dot_general
    for name in ("input_scale", "kernel_scale", "output_grad_scale"):
        np.testing.assert_array_equal(getattr(head_state, name), np.ones(1, np.float32))
    for name in ("input_amax_history", "kernel_amax_history", "output_grad_amax_history"):
        np.testing.assert_array_equal(getattr(head_state, name), np.zeros(4, np.float32))
    with jax.set_mesh(mesh):
        compute_model = cast_model_preserving_quantization_state(actual.model, mp.cast_to_compute)
        tokens = jnp.arange(4 * len(jax.devices())).reshape(len(jax.devices()), 4) % 16
        hidden, logits = eqx.filter_jit(lambda model: (model.native.activations(tokens), model.native.logits(tokens)))(
            compute_model
        )
    # The unused head keeps unit scales. Its independent reference quantizes
    # operands with NumPy and rounds the ordinary matmul to the compute dtype.
    quantized_hidden = np.asarray(hidden).astype(jnp.float8_e4m3fn).astype(np.float32)
    quantized_weight = np.asarray(compute_model.native.lm_head.weight).astype(jnp.float8_e4m3fn).astype(np.float32)
    expected_logits = (quantized_hidden @ quantized_weight.T).astype(mp.compute_dtype).astype(np.float32)
    assert np.all(np.isfinite(logits))
    assert np.any(logits != 0)
    np.testing.assert_allclose(np.asarray(logits).astype(np.float32), expected_logits, atol=1e-5, rtol=1e-5)
    adam_states = jax.tree.leaves(actual.opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    adam_states = [node for node in adam_states if isinstance(node, optax.ScaleByAdamState)]
    assert adam_states
    for adam in adam_states:
        for moments in (adam.mu, adam.nu):
            assert moments.native.layers.stacked.self_attn.q_proj.dot_general is None
            assert np.any(moments.native.layers.stacked.self_attn.q_proj.weight != 0)
