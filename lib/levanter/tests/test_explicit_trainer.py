# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grad_accum import microbatched_arrays
from levanter.checkpoint import save_checkpoint
from levanter.metrics import Metric, ReductionType
from levanter.optim.model_averaging import EmaModelAveragingConfig
from levanter.tracker import NoopConfig
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.testing.cpu_devices import run_on_cpu_devices


class _Model(eqx.Module):
    weight: jax.Array
    bias: jax.Array


def _loss(model, batch, *, key):
    del key
    error = batch["x"] * model.weight + model.bias - batch["target"]
    return jnp.mean(jnp.square(error).astype(jnp.float32))


@pytest.mark.parametrize("policy", ["f32", "params=float32,compute=bfloat16,output=float32"])
@pytest.mark.parametrize("microbatch", [False, True])
def test_explicit_trainer_matches_adamw_with_frozen_params_ema_and_reusable_batch(policy, microbatch):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    mp = jmp.get_policy(policy)
    config = TrainerConfig(
        id="explicit-trainer",
        tracker=NoopConfig(),
        mp=mp,
        load_checkpoint=False,
        train_batch_size=4,
        per_device_parallelism=2 if microbatch else -1,
        log_jaxprs=False,
        log_xla_hlo=False,
        model_averaging=EmaModelAveragingConfig(beta=0.8),
    )
    optimizer = optax.adamw(0.01, weight_decay=0.1)
    trainer = Trainer(
        config, optimizer, _loss, add_default_hooks=False, explicit_layout=ExplicitTrainingLayout(mesh, P("data"))
    )
    completed_steps = []
    trainer.add_hook(lambda info: completed_steps.append(info.step))
    with trainer:
        model = _Model(jnp.array([0.2, 0.4, 0.6]), jnp.array([0.1, -0.2, 0.3]))
        trainable = eqx.tree_at(lambda value: value.bias, jax.tree.map(lambda _: True, model), False)
        state = trainer.initial_state(jax.random.key(7), model=model, is_trainable=trainable)
        batch = {
            "x": jax.device_put(
                jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12, NamedSharding(mesh, P("data", None))
            ),
            "target": jax.device_put(jnp.ones((4, 3)), NamedSharding(mesh, P("data", None))),
        }
        reference_weight = state.model.weight.copy()
        fixed_bias = state.model.bias.copy()
        reference_opt = optimizer.init(_Model(reference_weight, None))
        reference_ema = reference_weight
        reference_key = state.training_key.copy()
        for step in range(2):
            loss_key, reference_key = jax.random.split(reference_key)

            def objective(weight):
                return _loss(mp.cast_to_compute(_Model(weight, fixed_bias)), batch, key=loss_key)

            expected_loss, gradient = jax.value_and_grad(objective)(reference_weight)
            updates, reference_opt = optimizer.update(
                _Model(gradient, None), reference_opt, _Model(reference_weight, None)
            )
            reference_weight = reference_weight + updates.weight
            reference_ema = 0.8 * reference_ema + 0.2 * reference_weight
            result = trainer.train_step(state, batch)
            state = result.state
            np.testing.assert_allclose(result.loss, expected_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(state.model.weight, reference_weight, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(state.model.bias, fixed_bias)
            np.testing.assert_allclose(state.model_averaging.model.weight, reference_ema, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(jax.random.key_data(state.training_key), jax.random.key_data(reference_key))
            assert int(state.step) == step + 1
            # Donation applies to the trainer state, so the same batch remains usable.
            np.testing.assert_array_equal(batch["target"], jnp.ones((4, 3)))
    assert completed_steps == [0, 1]


def test_native_microbatch_metrics_use_their_reduction_and_split_rng():
    model = jnp.array([0.25, 0.5])
    batch = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 16
    key = jax.random.key(11)

    def loss(weight, inputs, *, key):
        value = jnp.mean(inputs * weight)
        return value, {
            "minimum": Metric.from_value(jnp.min(inputs), ReductionType.MIN),
            "maximum": Metric.from_value(-jnp.min(inputs), ReductionType.MAX),
            "total": Metric.from_value(inputs.sum(), ReductionType.SUM),
            "average": Metric.from_value(value, ReductionType.MEAN),
            "weighted": Metric(jnp.square(inputs).sum(), inputs.sum(), ReductionType.MEAN),
            "last": Metric.from_value(jax.random.uniform(key), ReductionType.LAST),
        }

    gradient = eqx.filter_value_and_grad(loss, has_aux=True)
    accumulated = microbatched_arrays(lambda inputs, **kwargs: gradient(model, inputs, **kwargs), 2)
    (actual_loss, metrics), actual_gradient = eqx.filter_jit(accumulated)(batch, key=key)
    (expected_loss, _), expected_gradient = gradient(model, batch, key=key)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-6)
    np.testing.assert_allclose(metrics["minimum"].value(), batch.min())
    np.testing.assert_allclose(metrics["maximum"].value(), -batch.min())
    np.testing.assert_allclose(metrics["total"].value(), batch.sum())
    np.testing.assert_allclose(metrics["average"].value(), expected_loss)
    np.testing.assert_allclose(metrics["weighted"].value(), jnp.square(batch).sum() / batch.sum())
    np.testing.assert_array_equal(metrics["last"].value(), jax.random.uniform(jax.random.split(key, 4)[-1]))


def test_explicit_trainer_checkpoint_resumes_same_next_step(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    layout = ExplicitTrainingLayout(mesh, P("data"))
    config = TrainerConfig(
        id="native-checkpoint",
        tracker=NoopConfig(),
        load_checkpoint=False,
        load_checkpoint_path=str(tmp_path),
        train_batch_size=4,
        per_device_parallelism=2,
        log_jaxprs=False,
        log_xla_hlo=False,
        model_averaging=EmaModelAveragingConfig(beta=0.8),
    )
    optimizer = optax.adamw(0.01)
    trainer = Trainer(config, optimizer, _loss, add_default_hooks=False, explicit_layout=layout)

    def initialize_model():
        return _Model(jnp.array([0.2, 0.4, 0.6]), jnp.array([0.1, -0.2, 0.3]))

    batch = {"x": jnp.ones((4, 3)), "target": jnp.zeros((4, 3))}
    with trainer:
        state = trainer.initial_state(jax.random.PRNGKey(12), model_init=initialize_model)
        state = trainer.train_step(state, batch).state
        save_checkpoint(state.saveable_state, step=1, checkpoint_path=tmp_path / "step-1")
        expected = trainer.train_step(state, batch)

    restored_trainer = Trainer(
        dataclasses.replace(config, load_checkpoint=True),
        optimizer,
        _loss,
        add_default_hooks=False,
        explicit_layout=layout,
    )
    with restored_trainer:
        restored = restored_trainer.initial_state(jax.random.PRNGKey(99), model_init=initialize_model)
        actual = restored_trainer.train_step(restored, batch)
    np.testing.assert_allclose(actual.loss, expected.loss, atol=1e-6, rtol=1e-6)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual.state), jax.tree.leaves(expected.state), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=1e-6, rtol=1e-6)


def test_explicit_trainer_microbatch_preserves_data_and_parameter_sharding():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.callbacks import JitCallback
        from levanter.tracker import NoopConfig
        from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig

        class Model(eqx.Module):
            weight: jax.Array

        class CaptureGradients(JitCallback):
            def inside_step(self, state, inside_info):
                return inside_info.grads.weight

            def on_step(self, info, gradients):
                self.gradients = gradients

        def loss(model, batch, *, key):
            return jnp.mean(jnp.square(batch * model.weight))

        mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("replica_dcn", "context"), axis_types=(AxisType.Explicit,) * 2)
        layout = ExplicitTrainingLayout(mesh, P("replica_dcn", "context"))
        config = TrainerConfig(id="sharded-native", tracker=NoopConfig(), load_checkpoint=False,
                               train_batch_size=8, per_device_parallelism=2,
                               log_jaxprs=False, log_xla_hlo=False)
        trainer = Trainer(config, optax.adamw(0.01), loss, add_default_hooks=False, explicit_layout=layout)
        capture = CaptureGradients()
        trainer.add_hook(capture)
        with trainer:
            weight = jax.device_put(jnp.arange(4, dtype=jnp.float32)/4, NamedSharding(mesh, P("context")))
            batch = jax.device_put(jnp.arange(32, dtype=jnp.float32).reshape(8, 4)/32,
                                   NamedSharding(mesh, P("replica_dcn", "context")))
            model = Model(weight)
            expected_loss, expected_gradient = jax.jit(jax.value_and_grad(lambda w: loss(Model(w), batch, key=None)))(weight)
            state = trainer.initial_state(jax.random.PRNGKey(7), model=model)
            parameter_sharding = weight.sharding
            result = trainer.train_step(state, batch)
            np.testing.assert_allclose(result.loss, expected_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(capture.gradients, expected_gradient, atol=1e-5, rtol=1e-5)
            assert capture.gradients.sharding == parameter_sharding
            assert result.state.model.weight.sharding == parameter_sharding
            assert result.state.opt_state[0].mu.weight.sharding == parameter_sharding
            np.testing.assert_array_equal(batch, jnp.arange(32, dtype=jnp.float32).reshape(8, 4)/32)
        """,
        device_count=4,
    )
