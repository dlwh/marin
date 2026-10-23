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
from levanter.layers.rotary_native import RotaryConfig, apply_rotary
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding


@pytest.mark.parametrize(
    ("options", "compute_dtype"),
    [
        ({}, jnp.float32),
        ({"num_kv_heads": 1, "tie_word_embeddings": True, "use_bias": True}, jnp.float32),
        ({"use_qk_norm": True, "hybrid_norm": True, "input_embedding_norm": True}, jnp.float32),
        ({"head_dim": 6, "use_layer_norm_weight": False, "rope_factor": 2.0}, jnp.float32),
        pytest.param({}, jnp.bfloat16, id="bfloat16-compute-float32-parameters"),
    ],
)
def test_native_llama_matches_named_logits_loss_and_gradients(options, compute_dtype):
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

        def to_compute(model):
            return jax.tree.map(
                lambda value: value.astype(compute_dtype) if eqx.is_inexact_array(value) else value, model
            )

        expected_logits = eqx.filter_jit(
            lambda model: to_compute(model)(named_tokens, named_mask, pos_ids=named_positions)
            .rearrange((Batch, Pos, legacy.Vocab))
            .array
        )(legacy)
        actual_logits = eqx.filter_jit(lambda model: to_compute(model)(tokens, native_mask, pos_ids=positions))(native)
        np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)

        def legacy_loss(model):
            model = to_compute(model)
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
            return to_compute(model).next_token_loss(
                tokens, weights, mask=native_mask, pos_ids=positions, loss_implementation="xla"
            )

        expected_loss, expected_grads = eqx.filter_jit(eqx.filter_value_and_grad(legacy_loss))(legacy)
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


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("tie_word_embeddings,head_dim,use_bias", [(False, None, False), (True, 6, True)])
def test_native_llama_initialization_matches_named_parameters_and_forward(
    num_layers, tie_word_embeddings, head_dim, use_bias
):
    parameters = dict(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        max_seq_len=8,
        tie_word_embeddings=tie_word_embeddings,
        use_bias=use_bias,
    )
    config = NativeLlamaConfig(vocab_size=32, **parameters)
    legacy_config = LlamaConfig(**parameters, attn_backend=AttentionBackend.VANILLA)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    Batch, Pos = hax.Axis("batch", 2), hax.Axis("position", 8)
    tokens = (jnp.arange(16).reshape(2, 8) * 3) % config.vocab_size

    with jax.set_mesh(mesh):
        key = jax.random.key(7)
        legacy = LlamaLMHeadModel.init(hax.Axis("vocab", config.vocab_size), legacy_config, key=key)
        native = config.build(key=key)
        expected_state = legacy.to_state_dict()
        actual_state = native.to_state_dict()
        assert set(actual_state) == set(expected_state)
        for name, value in actual_state.items():
            np.testing.assert_array_equal(value, expected_state[name].reshape(value.shape), err_msg=name)

        expected = eqx.filter_jit(
            lambda model: model(hax.named(tokens, (Batch, Pos)), NamedAttentionMask.causal())
            .rearrange((Batch, Pos, model.Vocab))
            .array
        )(legacy)
        actual = eqx.filter_jit(lambda model: model(tokens))(native)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.tpu
@pytest.mark.parametrize("factor", [1.0, 8.0], ids=["default", "linear-scaled"])
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_native_rotary_long_positions_match_named_values_and_input_gradients(factor, compute_dtype):
    # Match the sequence length/head width that exposed TPU device-power error.
    shape = (8, 512, 16, 64)
    axes = tuple(hax.Axis(name, size) for name, size in zip(("batch", "position", "heads", "head_size"), shape))
    rng = np.random.default_rng(173)
    host_x = rng.standard_normal(shape).astype(np.float32)
    host_cotangent = rng.standard_normal(shape).astype(np.float32)
    host_positions = np.broadcast_to(np.arange(shape[1], dtype=np.int32), shape[:2]).copy()
    legacy = DefaultRotaryEmbeddingsConfig(theta=10000.0, factor=factor).build(axes[-1])
    native_config = RotaryConfig(theta=10000.0, factor=factor)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))

    def expected_forward(master, positions):
        return (
            legacy(hax.named(master.astype(compute_dtype), axes), hax.named(positions, axes[:2])).rearrange(axes).array
        )

    def actual_forward(master, positions):
        return apply_rotary(master.astype(compute_dtype), positions, native_config)

    def objective(forward, master, positions, cotangent):
        return jnp.sum(forward(master, positions).astype(jnp.float32) * cotangent)

    with jax.set_mesh(mesh):
        x = jnp.asarray(host_x)
        cotangent = jnp.asarray(host_cotangent)
        positions = jnp.asarray(host_positions)
        # Positions remain runtime arguments so the test exercises the compiled
        # frequency construction rather than constant-folding a fixed output.
        expected = jax.jit(expected_forward)(x, positions)
        actual = jax.jit(actual_forward)(x, positions)
        expected_grad = jax.jit(jax.grad(lambda master, pos, ct: objective(expected_forward, master, pos, ct)))(
            x, positions, cotangent
        )
        actual_grad = jax.jit(jax.grad(lambda master, pos, ct: objective(actual_forward, master, pos, ct)))(
            x, positions, cotangent
        )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
