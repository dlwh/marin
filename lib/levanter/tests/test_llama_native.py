# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.layers.attention import AttentionBackend, AttentionMask as NamedAttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"num_kv_heads": 1, "tie_word_embeddings": True, "use_bias": True},
        {"use_qk_norm": True, "hybrid_norm": True, "input_embedding_norm": True},
        {"head_dim": 6, "use_layer_norm_weight": False, "rope_factor": 2.0},
    ],
)
def test_native_llama_matches_named_logits_loss_and_gradients(options):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    config = dataclasses.replace(
        NativeLlamaConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=24,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=8,
        ),
        **options,
    )
    legacy_config = LlamaConfig(
        **{
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if field.name
            in {
                "hidden_dim",
                "intermediate_dim",
                "num_layers",
                "num_heads",
                "num_kv_heads",
                "head_dim",
                "max_seq_len",
                "layer_norm_epsilon",
                "tie_word_embeddings",
                "use_bias",
                "use_layer_norm_weight",
                "input_embedding_norm",
                "hybrid_norm",
                "use_qk_norm",
                "upcast_attn",
            }
        },
        attn_backend=AttentionBackend.VANILLA,
        rope=DefaultRotaryEmbeddingsConfig(theta=config.rope_theta, factor=config.rope_factor),
    )
    Batch, Pos = hax.Axis("batch", 2), hax.Axis("position", 8)
    tokens = (jnp.arange(16).reshape(2, 8) * 3) % config.vocab_size
    segments = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]])
    positions = jnp.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 0, 1, 2, 0, 1, 2]])
    weights = jnp.ones((2, 8)).at[:, -1].set(0).at[0, 2].set(0.5)
    named_tokens = hax.named(tokens, (Batch, Pos))
    named_positions = hax.named(positions, (Batch, Pos))
    named_mask = NamedAttentionMask.causal().with_segment_ids(hax.named(segments, (Batch, Pos)))
    native_mask = AttentionMask.causal().with_segment_ids(segments)
    with jax.set_mesh(mesh):
        legacy = LlamaLMHeadModel.init(hax.Axis("vocab", config.vocab_size), legacy_config, key=jax.random.key(0))
        source = legacy.to_state_dict()
        source = {key: value + 0.1 if key.endswith(".bias") else value for key, value in source.items()}
        legacy = legacy.from_state_dict(source)
        native = NativeLlamaModel.init(config, key=jax.random.key(1)).from_state_dict(source)
        expected_logits = (
            legacy(named_tokens, named_mask, pos_ids=named_positions).rearrange((Batch, Pos, legacy.Vocab)).array
        )
        actual_logits = jax.jit(lambda model: model(tokens, native_mask, pos_ids=positions))(native)
        np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)

        def legacy_loss(model):
            hidden = (
                model.activations(named_tokens, named_mask, pos_ids=named_positions)
                .rearrange((Batch, Pos, model.Embed))
                .array
            )
            head = model.get_lm_head().rearrange((model.Embed, model.Vocab)).array
            return fused_linear_softmax_cross_entropy_loss(
                hidden, head, jnp.roll(tokens, -1, axis=-1), weight=weights, implementation="xla"
            )

        def native_loss(model):
            return model.next_token_loss(
                tokens, weights, mask=native_mask, pos_ids=positions, loss_implementation="xla"
            )

        expected_loss, expected_grads = eqx.filter_value_and_grad(legacy_loss)(legacy)
        actual_loss, actual_grads = eqx.filter_jit(eqx.filter_value_and_grad(native_loss))(native)
        np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-5, rtol=1e-5)
        expected_state = expected_grads.to_state_dict()
        actual_state = actual_grads.to_state_dict()
        assert set(actual_state) == set(expected_state)
        for key in actual_state:
            np.testing.assert_allclose(
                actual_state[key],
                expected_state[key].reshape(actual_state[key].shape),
                atol=1e-5,
                rtol=1e-5,
                err_msg=key,
            )
        exported = native.to_state_dict()
        for key in exported:
            np.testing.assert_array_equal(exported[key], source[key].reshape(exported[key].shape))


@pytest.mark.tpu
def test_native_llama_explicit_fsdp_tensor_parallel_forward_and_gradient_layout():
    device_count = len(jax.devices())
    if device_count < 4:
        pytest.skip("Requires at least four CPU devices; run with JAX_NUM_CPU_DEVICES=4")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    config = NativeLlamaConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
    )
    with jax.set_mesh(mesh):
        reference = config.build(key=jax.random.key(0))
        layout = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
        model = config.build(key=jax.random.key(1), sharding=layout).from_state_dict(reference.to_state_dict())
        tokens = jnp.arange(32).reshape(4, 8) % config.vocab_size
        weights = jnp.ones_like(tokens, dtype=jnp.float32)
        reference_logits = eqx.filter_jit(lambda m: m(tokens))(reference)
        sharded_logits = eqx.filter_jit(lambda m: m(tokens))(model)
        np.testing.assert_allclose(sharded_logits, reference_logits, atol=1e-5, rtol=1e-5)
        assert sharded_logits.sharding.spec == layout.logits
        assert model.layers.stacked.self_attn.q_proj.weight.sharding.spec == P(None, *layout.column_storage)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights, loss_implementation="xla"))
        )(model)
        reference_loss, reference_grads = eqx.filter_value_and_grad(
            lambda m: m.next_token_loss(tokens, weights, loss_implementation="xla")
        )(reference)
        np.testing.assert_allclose(loss, reference_loss, atol=1e-5, rtol=1e-5)
        for actual, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(reference_grads), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        for parameter, gradient in zip(jax.tree.leaves(model), jax.tree.leaves(grads), strict=True):
            assert parameter.sharding == gradient.sharding
