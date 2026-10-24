# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from haliax.state_dict import to_torch_compatible_state_dict
from jax.sharding import PartitionSpec as P

from levanter.checkpoint import CheckpointerConfig, load_checkpoint, save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.main.convert_llama_checkpoint import convert_named_llama_state
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmHeadModel, native_llama_sharding
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig


def _loss(model, example, *, key):
    return model.compute_next_token_loss(example, key=key)


def _example(index):
    Batch, Pos = hax.Axis("batch", 4), hax.Axis("position", 8)
    tokens = (np.arange(32, dtype=np.int32).reshape(4, 8) * (index + 1) + 3 * index + 1) % 32
    return LmExample(
        hax.named(jnp.asarray(tokens), (Batch, Pos)),
        LmExample.causal_loss_mask(Pos).astype(jnp.float32).broadcast_to((Batch, Pos)),
    )


def _canonical(model):
    values = (
        model.to_state_dict() if isinstance(model, NativeLlamaLmHeadModel) else to_torch_compatible_state_dict(model)
    )
    return {name: np.array(value, copy=True) for name, value in values.items()}


def _retained_arrays(state):
    return {
        jax.tree_util.keystr(path): np.array(value, copy=True)
        for path, value in jax.tree_util.tree_flatten_with_path(state)[0]
        if eqx.is_array(value)
        and ".model." not in jax.tree_util.keystr(path)
        and ".mu." not in jax.tree_util.keystr(path)
        and ".nu." not in jax.tree_util.keystr(path)
    }


def _named_state(tmp_path, tied, compute_dtype):
    config = LlamaConfig(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        gradient_checkpointing=False,
        attn_backend="vanilla",
        scan_layers=True,
        tie_word_embeddings=tied,
    )
    optimizer = AdamConfig(
        learning_rate=0.003,
        weight_decay=0.07,
        lr_schedule="linear",
        warmup=0,
        min_lr_ratio=0.2,
        max_grad_norm=1.0,
    )
    trainer_config = TrainerConfig(
        id="convert-llama",
        num_train_steps=8,
        train_batch_size=4,
        per_device_parallelism=-1,
        tracker=NoopConfig(),
        require_accelerator=False,
        load_checkpoint=False,
        log_dir=tmp_path / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "unused"), save_interval=None, keep=None),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mesh=MeshConfig(axes={"data": len(jax.devices()), "model": 1}),
        mp=jmp.Policy(param_dtype=jnp.float32, compute_dtype=compute_dtype, output_dtype=jnp.float32),
    )
    with Trainer(trainer_config, optimizer.build(8), _loss, add_default_hooks=False) as trainer:
        state = trainer.initial_state(
            jax.random.PRNGKey(29), model=config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(11))
        )
        for index in range(2):
            state = trainer.train_step(state, _example(index)).state
    return state, optimizer, trainer_config


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.timeout(180)
def test_named_full_state_conversion_preserves_moments_and_next_trainer_update(tmp_path, tied, compute_dtype):
    source, optimizer, trainer_config = _named_state(tmp_path, tied, compute_dtype)
    source_path = tmp_path / "named"
    with trainer_config.use_device_mesh():
        save_checkpoint(
            source.saveable_state, 1, source_path, is_temporary=False, metadata={"recipe": "standard-adam"}
        )
        source = load_checkpoint(
            source.saveable_state,
            source_path,
            allow_partial=False,
            mesh=trainer_config.device_mesh,
            axis_mapping=trainer_config.parameter_axis_mapping,
        )
        expected = {"model": _canonical(source.model)}
        for name in ("mu", "nu"):
            expected[name] = _canonical(optax.tree_utils.tree_get(source.opt_state, name))
            assert sum(np.count_nonzero(value) for value in expected[name].values()) > 0
        retained = _retained_arrays(source)
        assert int(source.step) == 2
        assert float(source.opt_state.hyperparams["learning_rate"]) == pytest.approx(0.0027)
        assert float(source.opt_state.hyperparams["learning_rate"]) != float(optimizer.lr_scheduler(8)(source.step))
        assert any(
            np.any(value != np.asarray(jnp.asarray(value, dtype=jnp.bfloat16), dtype=np.float32))
            for value in expected["model"].values()
        )
        independent = jax.tree.map(lambda value: jnp.copy(value) if eqx.is_array(value) else value, source)

    native_config = dataclasses.replace(trainer_config, use_explicit_mesh_axes=True)
    sharding = native_llama_sharding(native_config.parameter_axis_mapping, native_config.compute_axis_mapping)
    with native_config.use_device_mesh():
        converted = convert_named_llama_state(source, optimizer_config=optimizer, num_train_steps=8, sharding=sharding)
        assert converted.training_key.dtype == source.training_key.dtype
        assert jax.random.key_impl(converted.training_key) == jax.random.key_impl(source.training_key)
        np.testing.assert_array_equal(
            jax.random.key_data(converted.training_key), jax.random.key_data(source.training_key)
        )
        for name, model in [
            ("model", converted.model),
            ("mu", optax.tree_utils.tree_get(converted.opt_state, "mu")),
            ("nu", optax.tree_utils.tree_get(converted.opt_state, "nu")),
        ]:
            actual = _canonical(model)
            assert actual.keys() == expected[name].keys()
            for key, value in actual.items():
                assert value.dtype == expected[name][key].dtype
                np.testing.assert_array_equal(value, expected[name][key], err_msg=f"{name}/{key}")
            assert model.native.token_embeddings.sharding.spec == sharding.column_storage
            assert model.native.layers.stacked.self_attn.q_proj.weight.sharding.spec == P(
                None, sharding.model_axis, sharding.fsdp_axis
            )
        for name, value in _retained_arrays(converted).items():
            np.testing.assert_array_equal(value, retained[name], err_msg=name)
        assert converted.step.sharding.spec == P()
        assert converted.training_key.sharding.spec == P()
        assert optax.tree_utils.tree_get(converted.opt_state, "ScaleByAdamState").count.sharding.spec == P()
        assert all(
            value.sharding.spec == P()
            for value in jax.tree.leaves(
                converted.opt_state, is_leaf=lambda value: isinstance(value, optax.ScaleByAdamState)
            )
            if eqx.is_array(value)
        )
        # A new checkpoint has a fresh write timestamp. Keep the exact original
        # completion marker separately, and its completed-step index independently
        # of TrainerState.step and the optimizer schedule counters.
        metadata = json.loads((source_path / "metadata.json").read_text())
        native_path = tmp_path / "native"
        save_checkpoint(
            converted.saveable_state,
            metadata["step"],
            native_path,
            is_temporary=metadata["is_temporary"],
            metadata={
                "recipe": metadata["recipe"],
                "source_checkpoint_metadata": metadata,
                **converted.model.checkpoint_metadata(),
            },
        )
        output_metadata = json.loads((native_path / "metadata.json").read_text())
        assert output_metadata["source_checkpoint_metadata"] == metadata
        assert output_metadata["step"] == 1
        assert output_metadata["recipe"] == metadata["recipe"]
        assert converted.model.config.checkpoint_vocab_size(output_metadata) == 32
        restored = load_checkpoint(
            converted.saveable_state, native_path, allow_partial=False, mesh=native_config.device_mesh
        )
        for actual, expected_leaf in zip(jax.tree.leaves(restored), jax.tree.leaves(converted)):
            np.testing.assert_array_equal(actual, expected_leaf)
        fresh = dataclasses.replace(
            restored, opt_state=restored.optimizer.init(eqx.filter(restored.model, eqx.is_inexact_array))
        )
        fresh = jax.tree.map(lambda value: jnp.copy(value) if eqx.is_array(value) else value, fresh)

    with Trainer(trainer_config, optimizer.build(8), _loss, add_default_hooks=False) as trainer:
        next_named = trainer.train_step(independent, _example(2))
        expected_next = {
            "model": _canonical(next_named.state.model),
            **{name: _canonical(optax.tree_utils.tree_get(next_named.state.opt_state, name)) for name in ("mu", "nu")},
        }
        next_retained = _retained_arrays(next_named.state)
        expected_loss = float(next_named.loss)
    layout = ExplicitTrainingLayout(native_config.device_mesh, P(sharding.batch, None))
    with Trainer(native_config, optimizer.build(8), _loss, explicit_layout=layout, add_default_hooks=False) as trainer:
        next_native = trainer.train_step(restored, _example(2))
        assert float(next_native.loss) == pytest.approx(expected_loss, abs=1e-4, rel=1e-4)
        for name, model in [
            ("model", next_native.state.model),
            ("mu", optax.tree_utils.tree_get(next_native.state.opt_state, "mu")),
            ("nu", optax.tree_utils.tree_get(next_native.state.opt_state, "nu")),
        ]:
            for key, value in _canonical(model).items():
                np.testing.assert_allclose(
                    value, expected_next[name][key], atol=1e-4, rtol=1e-4, err_msg=f"{name}/{key}"
                )
        for name, value in _retained_arrays(next_native.state).items():
            np.testing.assert_array_equal(value, next_retained[name], err_msg=name)
        # Reset moments/counters is a different continuation, even from identical
        # master weights: exercise the ordinary Trainer to distinguish it.
        reset_next = trainer.train_step(fresh, _example(2))
        reset_values = _canonical(reset_next.state.model)
        assert any(
            not np.allclose(value, expected_next["model"][key], atol=1e-4, rtol=1e-4)
            for key, value in reset_values.items()
        )


def test_named_conversion_preserves_int64_step_or_rejects_disabled_x64(tmp_path):
    source, optimizer, trainer_config = _named_state(tmp_path, False, jnp.float32)
    native_config = dataclasses.replace(trainer_config, use_explicit_mesh_axes=True)
    sharding = native_llama_sharding(native_config.parameter_axis_mapping, native_config.compute_axis_mapping)
    with trainer_config.use_device_mesh(), jax.enable_x64():
        source = dataclasses.replace(source, step=jnp.asarray(source.step, dtype=jnp.int64))
    with native_config.use_device_mesh(), jax.enable_x64():
        converted = convert_named_llama_state(source, optimizer_config=optimizer, num_train_steps=8, sharding=sharding)
        assert converted.step.dtype == source.step.dtype == jnp.int64
        np.testing.assert_array_equal(converted.step, source.step)
    with native_config.use_device_mesh(), jax.enable_x64(False):
        with pytest.raises(ValueError, match="Retained state.*dtype"):
            convert_named_llama_state(source, optimizer_config=optimizer, num_train_steps=8, sharding=sharding)


def test_named_conversion_rejects_missing_moment_and_same_size_geometry_corruption(tmp_path):
    source, optimizer, trainer_config = _named_state(tmp_path, False, jnp.float32)
    adam = optax.tree_utils.tree_get(source.opt_state, "ScaleByAdamState")
    broken_mu = eqx.tree_at(lambda model: model.embeddings.token_embeddings.weight.array, adam.mu, None)
    broken = dataclasses.replace(source, opt_state=optax.tree_utils.tree_set(source.opt_state, mu=broken_mu))
    native_config = dataclasses.replace(trainer_config, use_explicit_mesh_axes=True)
    sharding = native_llama_sharding(native_config.parameter_axis_mapping, native_config.compute_axis_mapping)
    with native_config.use_device_mesh():
        typed_source = dataclasses.replace(
            source,
            training_key=jax.random.wrap_key_data(
                jax.random.key_data(source.training_key), impl=jax.random.key_impl(source.training_key)
            ),
        )
        typed = convert_named_llama_state(
            typed_source, optimizer_config=optimizer, num_train_steps=8, sharding=sharding
        )
        assert typed.training_key.dtype == typed_source.training_key.dtype
        assert jax.random.key_impl(typed.training_key) == jax.random.key_impl(typed_source.training_key)
        np.testing.assert_array_equal(
            jax.random.key_data(typed.training_key), jax.random.key_data(typed_source.training_key)
        )
        with pytest.raises(ValueError, match="Source optimizer"):
            convert_named_llama_state(broken, optimizer_config=optimizer, num_train_steps=8, sharding=sharding)
        wrong_config = dataclasses.replace(source.model.config, num_heads=4, num_kv_heads=2, head_dim=4)
        wrong_model = dataclasses.replace(
            source.model, transformer=dataclasses.replace(source.model.transformer, config=wrong_config)
        )
        wrong = dataclasses.replace(source, model=wrong_model)
        with pytest.raises(ValueError, match="Source model"):
            convert_named_llama_state(wrong, optimizer_config=optimizer, num_train_steps=8, sharding=sharding)
    with trainer_config.use_device_mesh():
        save_checkpoint(broken.saveable_state, 1, tmp_path / "missing-moment", is_temporary=False)
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            load_checkpoint(
                source.saveable_state,
                tmp_path / "missing-moment",
                allow_partial=False,
                mesh=trainer_config.device_mesh,
                axis_mapping=trainer_config.parameter_axis_mapping,
            )
