# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.quantization import Fp8DotGeneralOp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.layers.attention import AttentionBackend, AttentionMask as NamedAttentionMask
from levanter.layers.linear import Linear
from levanter.layers.rotary import (
    DefaultRotaryEmbeddingsConfig,
    Llama3RotaryEmbeddingsConfig,
    YarnRotaryEmbeddingsConfig,
)
from levanter.layers.rotary_native import Llama3RotaryConfig, RotaryConfig, YarnRotaryConfig, apply_rotary
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.llama_config import LlamaConfig
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopTracker, current_tracker
from levanter.testing.helpers import skip_if_no_torch
from levanter.tokenizers import load_tokenizer
from levanter.utils.activation import ActivationFunctionEnum


def _native_fp8_operation():
    return dataclasses.replace(
        Fp8DotGeneralOp.init(amax_history_length=4),
        input_scale=jnp.array([7], jnp.float32),
        kernel_scale=jnp.array([3], jnp.float32),
        output_grad_scale=jnp.array([9], jnp.float32),
        input_amax_history=jnp.array([896, 448, 224, 112], jnp.float32),
        kernel_amax_history=jnp.array([224, 112, 56, 28], jnp.float32),
        output_grad_amax_history=jnp.array([229376, 114688, 57344, 28672], jnp.float32),
    )


def _assert_native_fp8_state(operation, update, inputs, weight, cotangent):
    expected = {
        "input_scale": [2],
        "kernel_scale": [0.5],
        "output_grad_scale": [4],
        "input_amax_history": [np.max(np.abs(inputs)), 224, 112, 896],
        "kernel_amax_history": [np.max(np.abs(weight)), 56, 28, 224],
        "output_grad_amax_history": [np.max(np.abs(cotangent)), 57344, 28672, 229376],
    }
    for name, values in expected.items():
        actual = getattr(update, name)
        np.testing.assert_array_equal(actual, np.asarray(values, np.float32), err_msg=name)
        source = getattr(operation, name)
        assert actual.sharding.is_equivalent_to(source.sharding, actual.ndim), name


@pytest.mark.tpu
@pytest.mark.parametrize("projection", ["column", "row"])
def test_native_linear_fp8_values_gradients_state_and_storage_layout(projection):
    model_size = 2 if len(jax.devices()) >= 4 else 1
    devices = jax.devices()[: 2 * model_size if model_size == 2 else 1]
    mesh = Mesh(
        np.asarray(devices).reshape(-1, model_size),
        ("data", "model"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    layout = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
    column = projection == "column"
    storage = layout.column_storage if column else layout.row_storage
    compute = layout.column_compute if column else layout.row_compute
    input_spec = layout.residual if column else layout.intermediate
    output_spec = layout.intermediate if column else layout.residual
    # Exact FP8 values make NumPy's ordinary matmul an independent forward/VJP oracle.
    inputs = ((np.arange(64).reshape(4, 2, 8) % 9) - 4).astype(np.float32) / 2
    weight = ((np.arange(64).reshape(8, 8) % 7) - 3).astype(np.float32) / 8
    bias = np.arange(8, dtype=np.float32) / 16
    cotangent = ((np.arange(64).reshape(4, 2, 8) % 5) - 2).astype(np.float32) / 4
    expected_output = inputs @ weight.T + bias
    expected_input_grad = cotangent @ weight
    expected_weight_grad = cotangent.reshape(-1, 8).T @ inputs.reshape(-1, 8)
    with jax.set_mesh(mesh):
        operation = jax.tree.map(
            lambda value: jax.device_put(value, NamedSharding(mesh, P())), _native_fp8_operation()
        )
        linear = Linear(
            jax.device_put(weight, NamedSharding(mesh, storage)),
            jax.device_put(bias, NamedSharding(mesh, P(storage[0]))),
            compute,
            operation,
        )
        x = jax.device_put(inputs, NamedSharding(mesh, input_spec))
        ct = jax.device_put(cotangent, NamedSharding(mesh, output_spec))

        def objective(parameters, output_cotangent):
            model, inputs = parameters
            output = model(inputs, out_sharding=output_spec)
            return jnp.sum(output * output_cotangent), output

        (_, output), (gradient, input_gradient) = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))(
            (linear, x), ct
        )
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(input_gradient, expected_input_grad)
        np.testing.assert_array_equal(gradient.weight, expected_weight_grad)
        np.testing.assert_array_equal(gradient.bias, cotangent.sum(axis=(0, 1)))
        assert output.sharding == NamedSharding(mesh, output_spec)
        assert input_gradient.sharding == x.sharding
        assert gradient.weight.sharding == linear.weight.sharding
        assert gradient.bias.sharding == linear.bias.sharding
        _assert_native_fp8_state(operation, gradient.dot_general, inputs, weight, cotangent)


@pytest.mark.tpu
@pytest.mark.parametrize("tied", [False, True], ids=["untied-fp8", "tied"])
def test_native_llama_logits_and_unembed_preserve_quantized_and_tied_head_behavior(tied):
    model_size = 2 if len(jax.devices()) >= 4 else 1
    devices = jax.devices()[: 2 * model_size if model_size == 2 else 1]
    mesh = Mesh(
        np.asarray(devices).reshape(-1, model_size),
        ("data", "model"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    layout = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
    config = NativeLlamaConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        tie_word_embeddings=tied,
    )
    with jax.set_mesh(mesh):
        model = config.build(key=jax.random.key(0), sharding=layout)
        if not tied:
            operation = jax.tree.map(
                lambda value: jax.device_put(value, NamedSharding(mesh, P())), _native_fp8_operation()
            )
            model = eqx.tree_at(lambda value: value.lm_head.dot_general, model, operation, is_leaf=lambda x: x is None)
        tokens = jax.device_put(np.arange(16, dtype=np.int32).reshape(4, 4), NamedSharding(mesh, layout.tokens))
        hidden = eqx.filter_jit(lambda value: value.activations(tokens))(model)
        weight = np.asarray(model.get_lm_head())
        if tied:
            expected_logits = np.asarray(hidden) @ weight.T
        else:
            # Delayed histories select scales 2 and .5; NumPy performs only the
            # elementwise FP8 casts and an ordinary, dequantized reference matmul.
            quantized_hidden = (np.asarray(hidden) / 2).astype(jnp.float8_e4m3fn).astype(np.float32) * 2
            quantized_weight = (weight / 0.5).astype(jnp.float8_e4m3fn).astype(np.float32) * 0.5
            expected_logits = quantized_hidden @ quantized_weight.T
            assert np.max(np.abs(expected_logits - np.asarray(hidden) @ weight.T)) > 1e-5
        logits = eqx.filter_jit(lambda value: value.logits(tokens))(model)
        np.testing.assert_allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)
        assert logits.sharding == NamedSharding(mesh, layout.logits)

        # Override compute placement separately from the model's configured
        # tensor parallelism, as callers of the shared unembed method may do.
        override = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data")
        standalone_hidden = jax.device_put(np.full((4, 2, 8), 0.3, np.float32), NamedSharding(mesh, override.residual))

        def objective(value):
            output = value.unembed(standalone_hidden, sharding=override)
            return output.sum(), output

        (_, output), gradient = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))(model)
        if tied:
            expected_output = np.asarray(standalone_hidden) @ weight.T
            assert gradient.lm_head is None
            np.testing.assert_allclose(gradient.token_embeddings, np.full_like(weight, 8 * 0.3), atol=1e-5, rtol=1e-5)
        else:
            expected_output = np.full((4, 2, 8), 0.3125, np.float32) @ quantized_weight.T
            _assert_native_fp8_state(
                operation, gradient.lm_head.dot_general, np.asarray(standalone_hidden), weight, np.ones((4, 2, 16))
            )
            assert gradient.lm_head.weight.sharding == model.lm_head.weight.sharding
        np.testing.assert_allclose(output, expected_output, atol=1e-5, rtol=1e-5)
        assert output.sharding == NamedSharding(mesh, override.logits)


@pytest.mark.tpu
@pytest.mark.parametrize("save_dtype", [None, jnp.bfloat16], ids=["master", "bf16"])
@pytest.mark.parametrize(
    ("rope", "activation_function"),
    [
        (DefaultRotaryEmbeddingsConfig(), ActivationFunctionEnum.silu),
        (Llama3RotaryEmbeddingsConfig(), ActivationFunctionEnum.silu),
        (YarnRotaryEmbeddingsConfig(factor=8.0), ActivationFunctionEnum.silu),
        (DefaultRotaryEmbeddingsConfig(), ActivationFunctionEnum.gelu_new),
    ],
    ids=["default", "llama3", "yarn", "gelu-new"],
)
def test_native_hf_export_sharded_dtype_and_layout_roundtrip(
    tmp_path, tiny_hf_tokenizer_path, save_dtype, rope, activation_function
):
    config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        rope=rope,
        activation_function=activation_function,
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(-1, model_size),
        ("data", "model"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    sharding = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
    converter = config.hf_export_converter(load_tokenizer(tiny_hf_tokenizer_path))
    output = tmp_path / "export"
    with jax.set_mesh(mesh):
        model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(0), sharding=sharding)
        converter.save_pretrained(model, str(output), max_shard_size=512, dtype=save_dtype)
        assert len(list(output.glob("*.safetensors"))) > 1
        assert (output / "model.safetensors.index.json").exists()
        converter = converter.replaced(reference_checkpoint=str(output))
        restored_config = NativeLlamaLmConfig.from_hf_config(converter.default_hf_config)
        restored_config.validate_hf_config(converter.default_hf_config)
        restored = converter.load_pretrained(
            restored_config.model_type,
            config=restored_config,
            model_init=lambda vocab: restored_config.build(vocab, key=jax.random.PRNGKey(1), sharding=sharding),
        )
        expected_state = {}
        for name, value in model.to_state_dict().items():
            expected = value if save_dtype is None else value.astype(save_dtype).astype(value.dtype)
            expected_state[name] = expected
            np.testing.assert_array_equal(restored.to_state_dict()[name], expected)
        batch_size = max(2, mesh.shape["data"])
        tokens = hax.named(
            jnp.arange(batch_size * 4).reshape(batch_size, 4) % model.Vocab.size,
            (hax.Axis("batch", batch_size), hax.Axis("position", 4)),
        )
        expected_model = model.from_state_dict(expected_state)
        np.testing.assert_allclose(
            restored(tokens, NamedAttentionMask.causal()).array,
            expected_model(tokens, NamedAttentionMask.causal()).array,
            atol=1e-5,
            rtol=1e-5,
        )
        assert restored.Vocab.size == 32
        assert restored.config.actual_head_size == 8
        assert restored.native.token_embeddings.sharding.spec == sharding.column_storage
        assert restored.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")


@skip_if_no_torch
@pytest.mark.parametrize(
    "rope",
    [
        DefaultRotaryEmbeddingsConfig(),
        DefaultRotaryEmbeddingsConfig(factor=8.0),
        DefaultRotaryEmbeddingsConfig(theta=500000.0),
        Llama3RotaryEmbeddingsConfig(),
        YarnRotaryEmbeddingsConfig(factor=8.0),
        YarnRotaryEmbeddingsConfig(factor=8.0, mscale=1.5),
        YarnRotaryEmbeddingsConfig(factor=8.0, original_max_position_embeddings=131072),
    ],
    ids=["default", "linear", "custom-base", "llama3", "yarn", "yarn-mscale", "yarn-long-context"],
)
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_native_hf_export_transformers_local_logits(tmp_path, tiny_hf_tokenizer_path, rope, tied):
    import torch  # noqa: PLC0415  # optional dependency
    from transformers import AutoModelForCausalLM  # noqa: PLC0415  # optional Torch model

    config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        tie_word_embeddings=tied,
        rope=rope,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    ids = np.arange(16, dtype=np.int32).reshape(2, 8)
    converter = config.hf_export_converter(load_tokenizer(tiny_hf_tokenizer_path))
    output = tmp_path / "export"
    with jax.set_mesh(mesh):
        model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(3))
        converter.save_pretrained(model, str(output))
        tokens = hax.named(jnp.asarray(ids), (hax.Axis("batch", 2), hax.Axis("position", 8)))
        expected = np.asarray(
            model(tokens, NamedAttentionMask.causal()).rearrange(("batch", "position", "vocab")).array
        )
    reference = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        actual = reference(input_ids=torch.from_numpy(ids).long()).logits.numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "mask_kind", ["causal", "dense", "dense_float", "explicit_int", "bidirectional", "offset", "none"]
)
@pytest.mark.parametrize("unbatched", [False, True])
def test_native_lm_boundary_preserves_short_positions_masks_loss_and_gradients(mask_kind, unbatched):
    config = draccus.decode(
        NativeLlamaLmConfig,
        {
            "hidden_dim": 16,
            "intermediate_dim": 24,
            "num_layers": 4,
            "num_heads": 4,
            "num_kv_heads": 2,
            "max_seq_len": 16,
            "attn_backend": "vanilla",
            "gradient_checkpointing": {"save_carries": False, "save_inputs": True, "nested": 2},
        },
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    Batch, Pos, Vocab = hax.Axis("batch", 2), hax.Axis("position", 8), hax.Axis("vocab", 32)
    tokens = hax.named((jnp.arange(16).reshape(2, 8) * 3) % Vocab.size, (Batch, Pos))
    positions = hax.named(jnp.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 0, 1, 2, 0, 1, 2]]), (Batch, Pos))
    segments = hax.named(jnp.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]]), (Batch, Pos))
    weights = hax.named(jnp.ones((2, 8)).at[0, 2].set(0.5).at[:, -1].set(0), (Batch, Pos))
    if unbatched:
        tokens, positions, segments, weights = (value[Batch, 0] for value in (tokens, positions, segments, weights))
    else:
        positions = positions.rearrange((Pos, Batch))
        segments = segments.rearrange((Pos, Batch))
    mask = NamedAttentionMask.causal().with_segment_ids(segments).with_sliding_window(3)
    if mask_kind in ("dense", "dense_float", "explicit_int"):
        mask = mask.materialize(Pos, Pos.alias("key_position"))
        if mask_kind == "dense_float":
            mask = mask.astype(jnp.float32)
        elif mask_kind == "explicit_int":
            mask = NamedAttentionMask.explicit(mask.astype(jnp.int32))
    elif mask_kind == "bidirectional":
        mask = NamedAttentionMask.bidirectional_sliding_window(2).with_segment_ids(segments)
    elif mask_kind == "offset":
        mask = NamedAttentionMask.causal(offset=hax.named(jnp.array(1), ())).with_segment_ids(segments)
    elif mask_kind == "none":
        mask = None
    output_axes = (Pos, Vocab) if unbatched else (Batch, Pos, Vocab)
    with jax.set_mesh(mesh):
        model = config.build(Vocab, key=jax.random.key(0))
        named_config = dataclasses.replace(
            LlamaConfig(), **{field.name: getattr(config, field.name) for field in dataclasses.fields(LlamaConfig)}
        )
        expected = LlamaLMHeadModel.init(Vocab, named_config, key=jax.random.key(1))
        shapes = expected.to_state_dict()
        expected = expected.from_state_dict(
            {name: value.reshape(shapes[name].shape) for name, value in model.to_state_dict().items()}
        )
        actual_logits = eqx.filter_jit(
            lambda m, attention_mask: m(tokens, attention_mask, pos_ids=positions).rearrange(output_axes).array
        )(model, mask)
        expected_logits = eqx.filter_jit(
            lambda m, attention_mask: m(tokens, attention_mask, pos_ids=positions).rearrange(output_axes).array
        )(expected, mask)
        np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)

        example = LmExample(tokens, weights, NamedAttentionMask() if mask is None else mask)
        with hax.axis_mapping({}):
            expected_loss, expected_grads = eqx.filter_value_and_grad(
                lambda m: m.compute_next_token_loss(example, logsumexp_weight=0.01).scalar()
            )(expected)
        actual_loss, actual_grads = eqx.filter_value_and_grad(
            lambda m: m.compute_next_token_loss(example, logsumexp_weight=0.01).scalar()
        )(model)
        np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-5, rtol=1e-5)
        for name, actual in actual_grads.to_state_dict().items():
            expected_array = expected_grads.to_state_dict()[name]
            np.testing.assert_allclose(
                actual, expected_array.reshape(actual.shape), atol=1e-5, rtol=1e-5, err_msg=name
            )


@pytest.mark.tpu
@pytest.mark.parametrize("tied", [False, True])
def test_native_lm_hf_config_and_vocab_resize_preserve_model_values_and_placement(tied):
    config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        tie_word_embeddings=tied,
    )
    recovered = NativeLlamaLmConfig.from_hf_config(config.to_hf_config(vocab_size=32))
    if len(jax.devices()) >= 4:
        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"), axis_types=(AxisType.Explicit,) * 2
        )
        sharding = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
    else:
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
        sharding = NativeLlamaSharding()
    with jax.set_mesh(mesh):
        model = recovered.build(hax.Axis("vocab", 32), key=jax.random.key(0), sharding=sharding)
        original = model.to_state_dict()
        grown = model.resize_vocab(40, key=jax.random.key(1))
        np.testing.assert_array_equal(np.asarray(grown.native.token_embeddings)[:32], model.native.token_embeddings)
        np.testing.assert_array_equal(np.asarray(grown.native.get_lm_head())[:32], model.native.get_lm_head())
        assert grown.native.token_embeddings.sharding == model.native.token_embeddings.sharding
        assert grown.native.get_lm_head().sharding == model.native.get_lm_head().sharding
        shrunk = grown.resize_vocab(32)
        for name, value in shrunk.to_state_dict().items():
            np.testing.assert_array_equal(value, original[name])
        tokens = hax.named(jnp.arange(8), hax.Axis("position", 8))
        np.testing.assert_allclose(shrunk(tokens).array, model(tokens).array, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("options", "compute_dtype"),
    [
        ({}, jnp.float32),
        ({"num_kv_heads": 1, "tie_word_embeddings": True, "use_bias": True}, jnp.float32),
        ({"use_qk_norm": True, "hybrid_norm": True, "input_embedding_norm": True}, jnp.float32),
        ({"head_dim": 6, "use_layer_norm_weight": False, "rope": RotaryConfig(factor=2.0)}, jnp.float32),
        ({"rope": Llama3RotaryConfig()}, jnp.float32),
        ({"rope": YarnRotaryConfig(factor=8.0, mscale=1.5)}, jnp.float32),
        pytest.param({}, jnp.bfloat16, id="bfloat16-compute-float32-parameters"),
        *[
            pytest.param({"activation_function": activation}, jnp.float32, id=activation.value)
            for activation in (
                ActivationFunctionEnum.relu,
                ActivationFunctionEnum.relu2,
                ActivationFunctionEnum.swish,
                ActivationFunctionEnum.gelu,
                ActivationFunctionEnum.gelu_new,
                ActivationFunctionEnum.quick_gelu,
                ActivationFunctionEnum.tanh,
            )
        ],
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
    if isinstance(config.rope, Llama3RotaryConfig):
        named_rope = Llama3RotaryEmbeddingsConfig(**dataclasses.asdict(config.rope))
    elif isinstance(config.rope, YarnRotaryConfig):
        named_rope = YarnRotaryEmbeddingsConfig(**dataclasses.asdict(config.rope))
    else:
        named_rope = DefaultRotaryEmbeddingsConfig(**dataclasses.asdict(config.rope))
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
                "activation_function",
            }
        },
        attn_backend=AttentionBackend.VANILLA,
        rope=named_rope,
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
@pytest.mark.parametrize(
    "activation_function", [ActivationFunctionEnum.silu, ActivationFunctionEnum.relu2, ActivationFunctionEnum.gelu_new]
)
def test_native_llama_explicit_fsdp_tensor_parallel_forward_and_gradient_layout(activation_function):
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
        activation_function=activation_function,
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


def _assert_rotary_values_and_gradients_match_named(named_config, native_config, compute_dtype, shape, host_positions):
    axes = tuple(hax.Axis(name, size) for name, size in zip(("batch", "position", "heads", "head_size"), shape))
    rng = np.random.default_rng(173)
    host_x = rng.standard_normal(shape).astype(np.float32)
    host_cotangent = rng.standard_normal(shape).astype(np.float32)
    named = named_config.build(axes[-1])
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))

    def expected_forward(master, positions):
        return (
            named(hax.named(master.astype(compute_dtype), axes), hax.named(positions, axes[:2])).rearrange(axes).array
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


@pytest.mark.tpu
@pytest.mark.parametrize("factor", [1.0, 8.0], ids=["default", "linear-scaled"])
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_native_rotary_long_positions_match_named_values_and_input_gradients(factor, compute_dtype):
    # Match the sequence length/head width that exposed TPU device-power error.
    shape = (8, 512, 16, 64)
    host_positions = np.broadcast_to(np.arange(shape[1], dtype=np.int32), shape[:2]).copy()
    _assert_rotary_values_and_gradients_match_named(
        DefaultRotaryEmbeddingsConfig(theta=10000.0, factor=factor),
        RotaryConfig(theta=10000.0, factor=factor),
        compute_dtype,
        shape,
        host_positions,
    )


@pytest.mark.parametrize(
    "named_config,native_config",
    [
        (Llama3RotaryEmbeddingsConfig(), Llama3RotaryConfig()),
        (
            Llama3RotaryEmbeddingsConfig(factor=32.0, original_max_position_embeddings=4096),
            Llama3RotaryConfig(factor=32.0, original_max_position_embeddings=4096),
        ),
        (YarnRotaryEmbeddingsConfig(), YarnRotaryConfig()),
        (YarnRotaryEmbeddingsConfig(factor=8.0), YarnRotaryConfig(factor=8.0)),
        (
            YarnRotaryEmbeddingsConfig(factor=8.0, original_max_position_embeddings=131072),
            YarnRotaryConfig(factor=8.0, original_max_position_embeddings=131072),
        ),
        (
            YarnRotaryEmbeddingsConfig(factor=8.0, mscale=1.5, beta_fast=16.0, beta_slow=2.0),
            YarnRotaryConfig(factor=8.0, mscale=1.5, beta_fast=16.0, beta_slow=2.0),
        ),
    ],
    ids=["llama3", "llama3-context", "yarn-unscaled", "yarn", "yarn-long-context", "yarn-temperature-ramp"],
)
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_native_scaled_rotary_long_positions_match_named_values_and_gradients(
    named_config, native_config, compute_dtype
):
    shape = (2, 9, 4, 64)
    host_positions = np.broadcast_to(
        np.asarray([0, 1, 4095, 8191, 8192, 65536, 131071, 131072, 262143], dtype=np.int32), shape[:2]
    ).copy()
    _assert_rotary_values_and_gradients_match_named(named_config, native_config, compute_dtype, shape, host_positions)


@pytest.mark.parametrize(
    "rope,parameters,message",
    [
        (YarnRotaryEmbeddingsConfig(factor=8.0, mscale=1.5), {}, "attention factor"),
        (YarnRotaryEmbeddingsConfig(factor=8.0), {"mscale_all_dim": 1.0}, "attention factor"),
        (YarnRotaryEmbeddingsConfig(factor=8.0), {"truncate": False}, "integer correction ramp"),
    ],
    ids=["implicit-mscale", "mscale-ratio", "floating-ramp"],
)
def test_native_hf_yarn_rejects_different_scaling_semantics(rope, parameters, message):
    config = NativeLlamaLmConfig(rope=rope)
    hf_config = config.to_hf_config(32)
    # External HF checkpoints can infer a different multiplier from identical
    # mscale fields, or use a floating rather than integer correction ramp.
    hf_config.rope_parameters.pop("attention_factor")
    hf_config.rope_parameters.update(parameters)
    hf_config.rope_scaling = hf_config.rope_parameters
    with pytest.raises(ValueError, match=message):
        config.validate_hf_config(hf_config)
    with pytest.raises(ValueError, match=message):
        NativeLlamaLmConfig.from_hf_config(hf_config)


@pytest.mark.parametrize(
    "mask_config,embedding_decays",
    [
        ({}, False),
        ({"weight_decay_modules": ["token_embeddings"], "default_weight_decay_mask": True}, True),
        ({"weight_decay_modules": r".*token_embeddings.*", "default_weight_decay_mask": True}, True),
        ({"default_weight_decay_mask": False}, True),
    ],
    ids=["default", "embedding-list-override", "embedding-regex-override", "unmasked"],
)
def test_native_default_adam_weight_decay_matches_named(mask_config, embedding_decays):
    config = LlamaConfig(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        use_bias=True,
    )
    optimizer_config = AdamConfig(
        learning_rate=0.1,
        weight_decay=0.2,
        warmup=0,
        lr_schedule="constant",
        max_grad_norm=None,
        **mask_config,
    )
    devices = np.asarray(jax.devices()[:1])
    named_mesh = Mesh(devices, ("data",), axis_types=(AxisType.Auto,))
    native_mesh = Mesh(devices, ("data",), axis_types=(AxisType.Explicit,))
    with current_tracker(NoopTracker()), jax.set_mesh(named_mesh), hax.axis_mapping({}):
        named = config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(0))
        # Nonzero biases make an accidental change to their decay policy visible.
        named = jax.tree.map(lambda value: value + 0.25, named)
        source = jax.tree.map(np.asarray, named.to_state_dict())
        optimizer = optimizer_config.build(4)
        named_updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, named), optimizer.init(named), params=named)
        expected = jax.tree.map(np.asarray, named_updates.to_state_dict())
        mask = optimizer_config.build_weight_decay_mask()
        if mask is not None:
            assert mask(named).embeddings.token_embeddings.weight == embedding_decays

    native_config = NativeLlamaLmConfig(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    )
    with current_tracker(NoopTracker()), jax.set_mesh(native_mesh):
        native = native_config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1)).from_state_dict(source)
        optimizer = optimizer_config.build(4)
        # Zero gradients isolate the real optimizer's decoupled weight-decay contribution.
        native_updates, _ = optimizer.update(
            jax.tree.map(jnp.zeros_like, native), optimizer.init(native), params=native
        )
        actual = native_updates.to_state_dict()
        for name, update in actual.items():
            np.testing.assert_allclose(
                update, expected[name].reshape(update.shape), atol=1e-5, rtol=1e-5, err_msg=name
            )
        if mask is not None:
            assert mask(native).native.token_embeddings == embedding_decays

    embedding_update = expected["model.embed_tokens.weight"]
    if embedding_decays:
        np.testing.assert_allclose(embedding_update, -0.02 * source["model.embed_tokens.weight"], atol=1e-5, rtol=1e-5)
    else:
        np.testing.assert_array_equal(embedding_update, np.zeros_like(embedding_update))
    assert np.max(np.abs(expected["model.layers.0.mlp.gate_proj.weight"])) > 0
    if mask_config.get("default_weight_decay_mask") is not False:
        for name in ("model.norm.weight", "model.layers.0.mlp.gate_proj.bias"):
            np.testing.assert_array_equal(expected[name], np.zeros_like(expected[name]))
