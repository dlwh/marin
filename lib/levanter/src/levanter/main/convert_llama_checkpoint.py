# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time conversion of standard named Llama state to native Llama state."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.nn.mup import LinearStandardParam
from haliax.state_dict import to_torch_compatible_state_dict
from jax.sharding import NamedSharding, PartitionSpec as P

from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native import NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig, NativeLlamaLmHeadModel
from levanter.optim.config import AdamConfig
from levanter.trainer_state import TrainerState


def _validate_tree(actual, template, label: str) -> None:
    if jax.tree.structure(actual) != jax.tree.structure(template):
        raise ValueError(f"{label} does not match the original recipe's tree/axes")
    for value, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(template)):
        if not eqx.is_array(value) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"{label} has incompatible array shape or dtype")


def _convert_model(named: LlamaLMHeadModel, native: NativeLlamaLmHeadModel) -> NativeLlamaLmHeadModel:
    # Export on the host before named flattening encounters the destination's
    # Explicit mesh: the source arrays still carry their old Auto mesh.
    source = to_torch_compatible_state_dict(jax.device_get(named))
    target = native.to_state_dict()
    if source.keys() != target.keys():
        raise ValueError("Named/native canonical parameter keys do not match")
    for name, value in source.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"Canonical parameter {name} has incompatible geometry or dtype")
    return native.from_state_dict(source)


def convert_named_llama_state(
    source: TrainerState[LlamaLMHeadModel],
    *,
    optimizer_config: AdamConfig,
    num_train_steps: int,
    sharding: NativeLlamaSharding,
) -> TrainerState[NativeLlamaLmHeadModel]:
    """Convert a strictly restored named state using its explicit original Adam recipe.

    This offline, single-host CPU conversion materializes canonical tensors on
    the host. Call under the destination explicit mesh, using the complete stored
    vocabulary rather than a tokenizer vocabulary. The caller must supply the original
    optimizer configuration and schedule horizon: checkpoint arrays do not encode
    that static recipe. All parameters must be trainable FP32 masters, using
    standard Linear parameterization and no averaging, adapters or quantization.
    The first conversion contract uses the default embedding/norm decay mask.

    Model parameters and both nonzero Adam moment trees use the same canonical
    serialization. Every other dynamic optimizer leaf, Trainer step and training
    key is retained rather than reconstructed from the step. Checkpoint metadata
    is separate from TrainerState: callers must preserve its completed-step index
    and other metadata when saving an ordinary native checkpoint.
    """
    if type(source) is not TrainerState or type(source.model) is not LlamaLMHeadModel:
        raise ValueError("Migration requires an unextended named Llama TrainerState")
    if jax.process_count() != 1 or jax.default_backend() != "cpu":
        raise ValueError("Migration currently requires an offline single-host CPU restore")
    if type(source.model.config) is not LlamaConfig:
        raise ValueError("Migration requires the original standard Llama configuration")
    if source.is_trainable is not True or source.model_averaging is not None:
        raise ValueError("Migration requires all parameters trainable and no model averaging")
    if source.mp.param_dtype != jnp.float32 or source.mp.compute_dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Migration supports FP32 masters with FP32 or BF16 computation")
    if type(optimizer_config) is not AdamConfig or (
        optimizer_config.skip_bad_steps
        or optimizer_config.clip_update_norm is not None
        or optimizer_config.update_rms_clipping is not None
        or optimizer_config.adamc_weight_decay
        or optimizer_config.weight_decay_modules is not None
        or optimizer_config.default_weight_decay_mask is False
    ):
        raise ValueError("Migration requires maintained Adam without history transforms or custom decay masks")
    for leaf in jax.tree.leaves(source.model, is_leaf=lambda value: isinstance(value, hax.nn.Linear)):
        if isinstance(leaf, hax.nn.Linear) and type(leaf.reparam) is not LinearStandardParam:
            raise ValueError("Migration does not support muP or other Linear reparameterizations")

    # Match TrainerState.init's master-parameter precision rather than the
    # runtime's default floating dtype (which can differ when x64 is enabled).
    expected_model = eqx.filter_eval_shape(
        lambda: source.mp.cast_to_param(source.model.config.build(source.model.Vocab, key=jax.random.PRNGKey(0)))
    )
    _validate_tree(source.model, expected_model, "Source model")
    optimizer = optimizer_config.build(num_train_steps)
    expected_opt_state = optimizer.init(eqx.filter(source.model, eqx.is_inexact_array))
    _validate_tree(source.opt_state, expected_opt_state, "Source optimizer")
    source_adam = optax.tree_utils.tree_get(source.opt_state, "ScaleByAdamState")

    config = NativeLlamaLmConfig(
        **{field.name: getattr(source.model.config, field.name) for field in dataclasses.fields(source.model.config)}
    )
    config.validate_sharding(sharding, jax.sharding.get_mesh(), source.model.Vocab.size)
    native = config.build(source.model.Vocab, key=jax.random.PRNGKey(0), sharding=sharding)
    template = TrainerState.init(optimizer, native, key=source.training_key, mp=source.mp)
    target_adam = optax.tree_utils.tree_get(template.opt_state, "ScaleByAdamState")

    mask = optimizer_config.build_weight_decay_mask()
    assert mask is not None  # The supported recipe requires the default decay mask.
    named_mask = jax.tree.map(
        lambda value, decays: hax.full(value.axes, decays, dtype=value.dtype),
        source.model,
        mask(source.model),
        is_leaf=hax.is_named_array,
    )
    native_mask = jax.tree.map(lambda value, decays: jnp.full_like(value, decays), native, mask(native))
    expected_mask = to_torch_compatible_state_dict(named_mask)
    actual_mask = native_mask.to_state_dict()
    if expected_mask.keys() != actual_mask.keys() or any(
        not np.array_equal(value, actual_mask[name]) for name, value in expected_mask.items()
    ):
        raise ValueError("Named/native canonical weight-decay masks differ")

    model = _convert_model(source.model, template.model)
    mu = _convert_model(source_adam.mu, target_adam.mu)
    nu = _convert_model(source_adam.nu, target_adam.nu)
    replicated = NamedSharding(jax.sharding.get_mesh(), P())

    def retain(value):
        if isinstance(value, optax.ScaleByAdamState):
            return value._replace(count=retain(value.count), mu=mu, nu=nu)
        host = jax.device_get(value)
        copied = jax.device_put(host, replicated)
        _validate_tree(copied, value, "Retained state")
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            if jax.random.key_impl(copied) != jax.random.key_impl(value):
                raise ValueError("Retained training key changed PRNG implementation")
            host = jax.device_get(jax.random.key_data(value))
            actual = jax.device_get(jax.random.key_data(copied))
        else:
            actual = jax.device_get(copied)
        if not np.array_equal(host, actual):
            raise ValueError("Retained state changed array values during placement")
        return copied

    opt_state = jax.tree.map(
        retain,
        source.opt_state,
        is_leaf=lambda value: isinstance(value, optax.ScaleByAdamState),
    )
    _validate_tree(opt_state, template.opt_state, "Converted optimizer")
    return dataclasses.replace(
        template,
        model=model,
        opt_state=opt_state,
        step=retain(source.step),
        training_key=retain(source.training_key),
    )
