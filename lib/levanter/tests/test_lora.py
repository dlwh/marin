# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import tempfile

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import numpy as np
import optax
from safetensors.numpy import load_file
from chex import assert_trees_all_close
from haliax.quantization import DefaultDotGeneralOp, DotGeneralOp
from safetensors import safe_open
from levanter.testing.helpers import (
    skip_if_hf_model_not_accessible,
    skip_if_module_missing,
    skip_if_no_torch,
    use_test_mesh,
)
from transformers import AutoModelForCausalLM

from levanter.adaptor.lora import (
    LoraConfig,
    lora_state_dict,
    unwrap_lora_modules,
    LoraLinear,
    lora_trainable_params_filter,
    loraize,
    merge_lora_modules,
    save_merged_hf_model,
    save_peft_pretrained,
)
from levanter.callbacks import StepInfo
from levanter.checkpoint import Checkpointer
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionMask
from levanter.layers.linear import Linear as ArrayLinear
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.models.lm_model import LmExample
from levanter.inference.page_table import PageTable
from levanter.inference.jit_scheduler import SequenceTable
from levanter.tokenizers import load_tokenizer
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode

In = hax.Axis("In", 10)
Mid = hax.Axis("Mid", 20)
Out = hax.Axis("Out", 5)


def test_loraize_simple():
    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

    module = Module(first=hnn.Linear.init(In, Mid, key=k0), second=hnn.Linear.init(Mid, Out, key=k1))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"], a_init_mode="random"), key=k0)
    assert isinstance(loraized.first, LoraLinear)
    assert isinstance(loraized.second, hnn.Linear)

    loraized = loraize(module, LoraConfig(r=8, target_modules=["second"], a_init_mode="random"), key=k0)
    assert isinstance(loraized.first, hnn.Linear)
    assert isinstance(loraized.second, LoraLinear)

    input = hax.random.normal(k0, (In,))
    assert not hax.all(hax.isclose(module(input), loraized(input)))


def test_lora_scan_layers():
    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

        @staticmethod
        def init(*, key):
            k1, k2 = jax.random.split(key)
            first = hnn.Linear.init(In, Mid, key=k1, out_first=True)
            second = hnn.Linear.init(Mid, In, key=k2, out_first=True)
            return Module(first, second)

    Layers = hax.Axis("Layers", 3)

    k0 = jax.random.PRNGKey(0)
    module: hnn.Stacked[Module] = hnn.Stacked.init(Layers, Module)(key=jax.random.split(k0, 3))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"], a_init_mode="random"), key=k0)
    assert isinstance(loraized, hnn.Stacked)
    assert isinstance(loraized.stacked.first, LoraLinear)
    assert isinstance(loraized.stacked.second, hnn.Linear)

    assert loraized.stacked.first.lora.lora_A.weight.axes == (Layers, hax.Axis("LORA_R", 8), In)
    assert loraized.stacked.first.lora.lora_B.weight.axes == (Layers, Mid, hax.Axis("LORA_R", 8))

    assert loraized.stacked.second.weight.axes == (Layers, In, Mid)
    input = hax.random.normal(k0, (In,))
    assert not hax.all(hax.isclose(module.fold(input), loraized.fold(input)))


def test_merge_lora():
    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

        @staticmethod
        def init(*, key):
            k1, k2 = jax.random.split(key)
            first = hnn.Linear.init(In, Mid, key=k1, init_scale=0.02)
            second = hnn.Linear.init(Mid, In, key=k2, init_scale=0.02)
            return Module(first, second)

    Layers = hax.Axis("Layers", 2)

    # tpu matmuls are very imprecise, so we force higher precision
    class PreciseDotGeneralOp(DotGeneralOp):
        def __call__(self, lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kwargs):
            return jax.lax.dot_general(
                lhs,
                rhs,
                dimension_numbers,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=preferred_element_type,
                **kwargs,
            )

    k0 = jax.random.PRNGKey(0)
    module: hnn.Stacked[Module] = hnn.Stacked.init(Layers, Module)(key=jax.random.split(k0, Layers.size))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["second"], a_init_mode="random"), key=k0)
    assert isinstance(loraized, hnn.Stacked)

    merged = merge_lora_modules(loraized)

    assert isinstance(merged, hnn.Stacked)
    assert merged.stacked.second.weight.axes == module.stacked.second.weight.axes

    def replace_dot_general(x):
        if isinstance(x, DefaultDotGeneralOp):
            return PreciseDotGeneralOp()
        return x

    merged = jax.tree.map(replace_dot_general, merged, is_leaf=lambda x: isinstance(x, DefaultDotGeneralOp))
    loraized = jax.tree.map(replace_dot_general, loraized, is_leaf=lambda x: isinstance(x, DefaultDotGeneralOp))

    input = hax.random.normal(k0, (In,))
    # light tolerances for TPU
    assert_trees_all_close(merged.fold(input), loraized.fold(input), rtol=1e-3, atol=3e-3)


@skip_if_module_missing("peft")
@skip_if_no_torch
def test_lora_load_in_peft(local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415  # optional dep: torch

    # Local tokenizer keeps converter construction off the Hub; the base model is
    # saved/reloaded from a temp dir so the tokenizer is incidental here.
    converter: HFCheckpointConverter = Gpt2Config(tokenizer=local_gpt2_tokenizer_path).hf_checkpoint_converter()
    config = Gpt2Config(max_seq_len=128, hidden_dim=128, num_layers=2, num_heads=2)
    Vocab = converter.Vocab

    model = Gpt2LMHeadModel.init(Vocab, config=config, key=jax.random.PRNGKey(0))
    model = inference_mode(model, True)

    input = hax.random.randint(jax.random.PRNGKey(0), config.max_Pos, 0, Vocab.size)
    torch_input = torch.tensor(np.array(input.array), dtype=torch.long).reshape((1, -1))

    causal_mask = AttentionMask.causal()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        from peft import PeftConfig, PeftModel  # noqa: PLC0415  # optional dep: peft

        converter.save_pretrained(model, f"{tmpdir}/model", save_tokenizer=False)

        lora_config = LoraConfig(r=8, target_modules=["c_attn"], a_init_mode="random")
        loraized = loraize(model, lora_config, key=jax.random.PRNGKey(0))
        save_peft_pretrained(loraized, lora_config, f"{tmpdir}/model", f"{tmpdir}/loraized")
        peft_config = PeftConfig.from_pretrained(f"{tmpdir}/loraized")

        hf_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path).cpu()
        hf_model.eval()
        hf_out = hf_model(torch_input)
        hf_out = hf_out.logits.detach().numpy()

        lev_out = model(input, attn_mask=causal_mask)
        lev_out = np.array(lev_out.array)
        assert np.allclose(lev_out, hf_out, atol=1e-4)

        # load with peft
        hf_lora_model = PeftModel.from_pretrained(hf_model, f"{tmpdir}/loraized").cpu()

        lev_lora_out = loraized(input, attn_mask=causal_mask)
        lev_lora_out = np.array(lev_lora_out.array)

        hf_lora_model.eval()
        hf_lora_out = hf_lora_model(torch_input)
        hf_lora_out = hf_lora_out.logits.detach().numpy()

        assert np.allclose(lev_lora_out, hf_lora_out, atol=1e-4)
        assert not np.allclose(lev_lora_out, hf_out, atol=1e-4)


@skip_if_module_missing("peft")
@skip_if_no_torch
def test_lora_merged_load_in_hf(local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415  # optional dep: torch

    # Local tokenizer keeps converter construction off the Hub; the base model is
    # saved/reloaded from a temp dir so the tokenizer is incidental here.
    converter: HFCheckpointConverter = Gpt2Config(tokenizer=local_gpt2_tokenizer_path).hf_checkpoint_converter()
    config = Gpt2Config(max_seq_len=128, hidden_dim=128, num_layers=2, num_heads=2)
    Vocab = converter.Vocab

    model = Gpt2LMHeadModel.init(Vocab, config=config, key=jax.random.PRNGKey(0))
    model = inference_mode(model, True)

    input = hax.random.randint(jax.random.PRNGKey(0), config.max_Pos, 0, Vocab.size)
    torch_input = torch.tensor(np.array(input.array), dtype=torch.long).reshape((1, -1))

    causal_mask = AttentionMask.causal()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        converter.save_pretrained(model, f"{tmpdir}/model", save_tokenizer=False)

        lora_config = LoraConfig(r=8, target_modules=["c_attn"], a_init_mode="random")
        loraized = loraize(model, lora_config, key=jax.random.PRNGKey(0))
        save_merged_hf_model(loraized, converter, f"{tmpdir}/loraized", save_tokenizer=False)

        hf_model = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/model").cpu()
        hf_model.eval()
        hf_out = hf_model(torch_input)
        hf_out = hf_out.logits.detach().numpy()

        lev_out = model(input, attn_mask=causal_mask)
        lev_out = np.array(lev_out.array)
        assert np.allclose(lev_out, hf_out, atol=1e-4)

        # load merged model with hf
        hf_lora_model = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/loraized").cpu()

        lev_lora_out = loraized(input, attn_mask=causal_mask)
        lev_lora_out = np.array(lev_lora_out.array)

        hf_lora_model.eval()
        hf_lora_out = hf_lora_model(torch_input)
        hf_lora_out = hf_lora_out.logits.detach().numpy()

        assert np.allclose(lev_lora_out, hf_lora_out, atol=1e-4)
        assert not np.allclose(lev_lora_out, hf_out, atol=1e-4)


@skip_if_no_torch
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_lora_merged_load_in_hf_llama():
    import torch  # noqa: PLC0415  # optional dep: torch

    config = LlamaConfig(
        max_seq_len=32,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        gradient_checkpointing=False,
        scan_layers=False,
    )
    converter: HFCheckpointConverter = config.hf_checkpoint_converter()
    vocab = hax.Axis("vocab", 128)

    model = LlamaLMHeadModel.init(vocab, config=config, key=jax.random.PRNGKey(0))
    model = inference_mode(model, True)
    lora_config = LoraConfig(
        r=8,
        alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        a_init_mode="random",
    )

    input_ids = hax.random.randint(jax.random.PRNGKey(1), config.max_Pos.resize(16), 0, vocab.size)
    torch_input = torch.tensor(np.array(input_ids.array), dtype=torch.long).reshape((1, -1))
    attn_mask = AttentionMask.causal()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        loraized = loraize(model, lora_config, key=jax.random.PRNGKey(2))
        merged = merge_lora_modules(loraized)

        lev_lora_out = np.array(loraized(input_ids, attn_mask=attn_mask).array)
        lev_merged_out = np.array(merged(input_ids, attn_mask=attn_mask).array)
        np.testing.assert_allclose(lev_lora_out, lev_merged_out, rtol=1e-4, atol=1e-4)

        save_merged_hf_model(
            loraized,
            converter,
            f"{tmpdir}/loraized",
            save_reference_code=False,
            save_tokenizer=False,
        )

        hf_lora_model = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/loraized").cpu()
        hf_lora_model.eval()
        hf_lora_out = hf_lora_model(torch_input).logits[0].detach().numpy()

        np.testing.assert_allclose(lev_lora_out, hf_lora_out, rtol=1e-4, atol=1e-4)


def test_lora_peft_export_uses_hf_llama_keys_and_target_modules(tmp_path):
    config = LlamaConfig(
        max_seq_len=32,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        gradient_checkpointing=False,
        scan_layers=False,
    )
    vocab = hax.Axis("vocab", 128)
    model = LlamaLMHeadModel.init(vocab, config=config, key=jax.random.PRNGKey(0))
    lora_config = LoraConfig(r=8)

    loraized = loraize(model, lora_config, key=jax.random.PRNGKey(1))
    save_peft_pretrained(loraized, lora_config, "marin-community/marin-8b-base", str(tmp_path))

    with open(tmp_path / "adapter_config.json") as f:
        adapter_config = json.load(f)

    assert adapter_config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    with safe_open(tmp_path / "adapter_model.safetensors", framework="numpy") as tensors:
        keys = list(tensors.keys())

    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in keys
    assert all("base_model.model.transformer." not in key for key in keys)


def test_lora_peft_export_does_not_use_generic_numpy_state_dict(tmp_path, monkeypatch):
    config = LlamaConfig(
        max_seq_len=32,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        gradient_checkpointing=False,
        scan_layers=False,
    )
    vocab = hax.Axis("vocab", 128)
    model = LlamaLMHeadModel.init(vocab, config=config, key=jax.random.PRNGKey(0))
    lora_config = LoraConfig(r=8)

    loraized = loraize(model, lora_config, key=jax.random.PRNGKey(1))

    def fail_generic_helper(*args, **kwargs):
        raise AssertionError("PEFT export should use its mesh-aware state-dict path")

    monkeypatch.setattr("haliax.state_dict.to_numpy_state_dict", fail_generic_helper)
    save_peft_pretrained(loraized, lora_config, "marin-community/marin-8b-base", str(tmp_path))

    with safe_open(tmp_path / "adapter_model.safetensors", framework="numpy") as tensors:
        assert tensors.keys()


def test_lora_works_with_checkpointer():
    with tempfile.TemporaryDirectory() as tempdir:
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)

        class Module(eqx.Module):
            first: hnn.Linear
            second: hnn.Linear

            def __call__(self, x):
                return self.second(self.first(x))

        module = Module(first=hnn.Linear.init(In, Mid, key=k0), second=hnn.Linear.init(Mid, Out, key=k1))

        loraized = loraize(module, LoraConfig(r=8, target_modules=["first"], a_init_mode="random"), key=k0)
        lora_filter = lora_trainable_params_filter(loraized)

        optimizer = optax.adam(1e-3)

        trainer_state = TrainerState.init(optimizer, loraized, key=k0, is_trainable=lora_filter)
        info = StepInfo(trainer_state, 0.0, 0.0)

        checkpointer = Checkpointer(tempdir, None, [])
        checkpointer.save_checkpoint(
            tree=info.state.saveable_state,
            step=info.step,
            destination="loraized",
        )
        checkpointer.wait_until_finished()


def test_loraize_a_init_mode_zero():
    """A=0 init: lora_A all zeros, lora_B has non-zero entries from default Linear.init."""
    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

    module = Module(first=hnn.Linear.init(In, Mid, key=k0), second=hnn.Linear.init(Mid, Out, key=k1))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"], a_init_mode="zero"), key=k0)
    assert isinstance(loraized.first, LoraLinear)
    assert hax.all(loraized.first.lora.lora_A.weight == 0.0)
    assert not hax.all(loraized.first.lora.lora_B.weight == 0.0)

    # B @ A = 0 at init, so the loraized output matches the base output.
    input = hax.random.normal(k0, (In,))
    base_first_out = module.first(input)
    loraized_first_out = loraized.first(input)
    assert hax.all(hax.isclose(base_first_out, loraized_first_out))


def test_loraize_a_zero_and_b_zero_degenerate():
    """A=0 and B=0 together: degenerate identity init (no learning), but must not crash."""
    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

    module = Module(first=hnn.Linear.init(In, Mid, key=k0), second=hnn.Linear.init(Mid, Out, key=k1))

    loraized = loraize(
        module,
        LoraConfig(r=8, target_modules=["first"], a_init_mode="zero", zero_init_b=True),
        key=k0,
    )
    assert hax.all(loraized.first.lora.lora_A.weight == 0.0)
    assert hax.all(loraized.first.lora.lora_B.weight == 0.0)


@pytest.mark.tpu
@pytest.mark.parametrize("row", [False, True], ids=["column", "row"])
@pytest.mark.parametrize("dropout", [0.0, 0.25, 1.0])
def test_array_lora_matches_independent_values_gradients_and_placements(row, dropout):
    model_size = 2 if jax.device_count() >= 4 else 1
    mesh = Mesh(
        np.array(jax.devices()).reshape(-1, model_size), ("data", "model"), axis_types=(AxisType.Explicit,) * 2
    )
    storage, compute = (P("data", "model"), P(None, "model")) if row else (P("model", "data"), P("model", None))
    x_host = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    g_host = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    with jax.set_mesh(mesh):
        base = ArrayLinear.init(
            8,
            8,
            key=jax.random.PRNGKey(1),
            use_bias=True,
            initializer_std=0.2,
            storage_spec=storage,
            compute_spec=compute,
        )
        config = LoraConfig(r=3, alpha=2, a_init_mode="random", dropout=dropout)
        module = loraize(base, config, key=jax.random.PRNGKey(2))
        x = reshard(jnp.array(x_host), P("data", "model" if row else None))
        output_spec = P("data", None if row else "model")
        cotangent = reshard(jnp.array(g_host), output_spec)
        value, (gradient, dx) = eqx.filter_value_and_grad(
            lambda args: jnp.sum(args[0](args[1], out_sharding=output_spec, key=jax.random.key(7)) * cotangent)
        )((module, x))
        w, a, b = (np.asarray(v) for v in (base.weight, module.lora.lora_A.weight, module.lora.lora_B.weight))
        scale = config.alpha / config.r
        # The wrapper's second key drives only the adapter input, not the base branch.
        mask = np.asarray(jax.random.bernoulli(jax.random.split(jax.random.key(7))[1], 1 - dropout, x_host.shape))
        multiplier = mask / (1 - dropout) if dropout < 1 else np.zeros_like(x_host)
        dropped = x_host * multiplier
        expected = x_host @ w.T + scale * dropped @ a.T @ b.T + np.asarray(base.bias)
        np.testing.assert_allclose(
            module(x, out_sharding=output_spec, key=jax.random.key(7)), expected, atol=1e-5, rtol=1e-5
        )
        assert module(x, out_sharding=output_spec, key=jax.random.key(7)).sharding.spec == output_spec
        np.testing.assert_allclose(value, (expected * g_host).sum(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(dx, g_host @ w + scale * (g_host @ b @ a) * multiplier, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(gradient.wrapped.weight, g_host.T @ x_host, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(gradient.lora.lora_A.weight, scale * b.T @ g_host.T @ dropped, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(gradient.lora.lora_B.weight, scale * g_host.T @ dropped @ a.T, atol=1e-5, rtol=1e-5)
        assert gradient.lora.lora_A.weight.sharding == module.lora.lora_A.weight.sharding
        assert gradient.lora.lora_B.weight.sharding == module.lora.lora_B.weight.sharding
        if dropout:
            with pytest.raises(RuntimeError, match="requires a key"):
                module(x, out_sharding=output_spec)
        inference = inference_mode(module, True)
        merged = module.merge()(x, out_sharding=output_spec)
        np.testing.assert_allclose(inference(x, out_sharding=output_spec), merged, atol=1e-5, rtol=1e-5)


def test_array_lora_multilayer_canonical_export_and_base_view():
    mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    config = NativeLlamaConfig(
        vocab_size=32, hidden_dim=16, intermediate_dim=24, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=8
    )
    with jax.set_mesh(mesh):
        base = config.build(key=jax.random.PRNGKey(1))
        adapted = loraize(base, LoraConfig(r=3), key=jax.random.PRNGKey(2))
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
        np.testing.assert_array_equal(base(tokens), adapted(tokens))
        for actual, expected in zip(jax.tree.leaves(unwrap_lora_modules(adapted)), jax.tree.leaves(base), strict=True):
            np.testing.assert_array_equal(actual, expected)
        factors = adapted.layers.stacked.self_attn.q_proj.lora
        assert not np.array_equal(factors.lora_B.weight[0], factors.lora_B.weight[1])
        adapted = loraize(base, LoraConfig(r=3, a_init_mode="random"), key=jax.random.PRNGKey(2))
        merged = merge_lora_modules(adapted)
        np.testing.assert_allclose(adapted(tokens), merged(tokens), atol=1e-5, rtol=1e-5)
        exported = lora_state_dict(adapted)
        assert len(exported) == 2 * 7 * 2
        for index, layer in enumerate(adapted.layers.unstacked()):
            for category, names in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                for name in names:
                    projection = getattr(getattr(layer, category), name)
                    for factor in ("lora_A", "lora_B"):
                        np.testing.assert_array_equal(
                            exported[f"base_model.model.model.layers.{index}.{category}.{name}.{factor}.weight"],
                            getattr(projection.lora, factor).weight,
                        )
        with pytest.raises(ValueError, match="did not select"):
            loraize(base, LoraConfig(target_modules=["missing"]), key=jax.random.PRNGKey(2))


@skip_if_no_torch
def test_array_lora_local_peft_and_merged_exports_match_independent_torch(tmp_path, tiny_hf_tokenizer_path):
    import torch  # noqa: PLC0415  # optional dependency

    mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    config = NativeLlamaLmConfig(
        hidden_dim=16, intermediate_dim=24, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=8
    )
    adapter = LoraConfig(r=3, alpha=6, a_init_mode="random", target_modules=["q_proj"])
    converter = config.hf_export_converter(load_tokenizer(tiny_hf_tokenizer_path))
    with jax.set_mesh(mesh):
        base = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(1))
        adapted = loraize(base, adapter, key=jax.random.PRNGKey(2))
        converter.save_pretrained(base, str(tmp_path / "base"))
        save_peft_pretrained(adapted, adapter, str(tmp_path / "base"), str(tmp_path / "adapter"))
        save_merged_hf_model(adapted, converter, str(tmp_path / "merged"))
        tokens = hax.named(jnp.arange(8).reshape(2, 4), (hax.Axis("batch", 2), hax.Axis("position", 4)))
        expected = np.array(adapted(tokens, AttentionMask.causal()).array)
        assert not np.array_equal(expected, base(tokens, AttentionMask.causal()).array)
    fields = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    factors = load_file(tmp_path / "adapter/adapter_model.safetensors")
    assert len(factors) == 4
    independent = AutoModelForCausalLM.from_pretrained(tmp_path / "base", local_files_only=True).eval()
    state = independent.state_dict()
    for key, a in factors.items():
        if ".lora_A." not in key:
            continue
        b = factors[key.replace("lora_A", "lora_B")]
        weight_key = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        state[weight_key] += torch.from_numpy(b @ a) * (fields["lora_alpha"] / fields["r"])
    independent.load_state_dict(state, strict=True)
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", local_files_only=True).eval()
    with torch.no_grad():
        torch_tokens = torch.arange(8).reshape(2, 4)
        for model in (independent, merged):
            np.testing.assert_allclose(model(torch_tokens).logits.numpy(), expected, atol=1e-4, rtol=1e-4)


def _native_with_named_lora_factors(native, named_factors):
    layers = native.native.layers.stacked
    locations = []
    values = []
    for category, names in (
        ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj", "down_proj")),
    ):
        for name in names:
            projection = getattr(getattr(layers, category), name)
            for factor in ("lora_A", "lora_B"):
                template = getattr(projection.lora, factor).weight
                arrays = [
                    named_factors[f"base_model.model.model.layers.{index}.{category}.{name}.{factor}.weight"]
                    for index in range(native.config.num_layers)
                ]
                values.append(reshard(jnp.asarray(np.stack(arrays)).reshape(template.shape), template.sharding.spec))
                locations.append((category, name, factor))
    return eqx.tree_at(
        lambda model: tuple(
            getattr(getattr(getattr(model.native.layers.stacked, category), name).lora, factor).weight
            for category, name, factor in locations
        ),
        native,
        tuple(values),
    )


@pytest.mark.tpu
@pytest.mark.timeout(120)
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("input_order", ["batch-first", "position-first", "unbatched"])
def test_native_lora_dropout_matches_named_loss_gradients_and_logits(num_layers, input_order):
    unbatched = input_order == "unbatched"
    config = LlamaConfig(
        hidden_dim=8,
        intermediate_dim=12,
        num_layers=num_layers,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        attn_backend="vanilla",
        gradient_checkpointing=True,
    )
    adapter = LoraConfig(r=2, alpha=2, dropout=0.25, a_init_mode="random")
    tokens_host = np.arange(4 if unbatched else 4 * jax.device_count()).reshape((-1, 4)) % 16
    if unbatched:
        tokens_host = tokens_host[0]
    axes = (hax.Axis("position", 4),) if unbatched else (hax.Axis("batch", len(tokens_host)), hax.Axis("position", 4))
    if input_order == "position-first":
        tokens_host = tokens_host.T
        axes = axes[::-1]
    named_mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Auto,))
    key = jax.random.key(17)
    with jax.set_mesh(named_mesh), hax.axis_mapping({}):
        named = LlamaLMHeadModel.init(hax.Axis("vocab", 16), config, key=jax.random.key(0))
        source = jax.tree.map(np.asarray, named.to_state_dict())
        named = loraize(named, adapter, key=jax.random.key(2))
        factors = jax.tree.map(np.asarray, lora_state_dict(named))
        example = LmExample(
            hax.named(jnp.asarray(tokens_host), axes),
            hax.named(jnp.array([1, 1, 1, 0], dtype=jnp.float32), (hax.Axis("position", 4),)),
        )
        loss, gradient = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.compute_next_token_loss(example, key=key).array)
        )(named)
        expected_loss = np.asarray(loss)
        expected_gradient = jax.tree.map(np.asarray, lora_state_dict(gradient))
        expected_logits = np.asarray(named(example.tokens, example.attn_mask, key=key).array)
        expected_inference = np.asarray(inference_mode(named, True)(example.tokens, example.attn_mask).array)
    native_mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(native_mesh):
        native_config = NativeLlamaLmConfig(
            **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
        )
        native = native_config.build(
            hax.Axis("vocab", 16),
            key=jax.random.key(1),
            sharding=NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data"),
        ).from_state_dict(source)
        native = _native_with_named_lora_factors(loraize(native, adapter, key=jax.random.key(3)), factors)
        example = LmExample(
            hax.named(jnp.asarray(tokens_host), axes),
            hax.named(jnp.array([1, 1, 1, 0], dtype=jnp.float32), (hax.Axis("position", 4),)),
        )
        loss, gradient = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.compute_next_token_loss(example, key=key).array)
        )(native)
        np.testing.assert_allclose(loss, expected_loss, atol=1e-4, rtol=1e-4)
        actual_gradient = lora_state_dict(gradient)
        for name, value in actual_gradient.items():
            np.testing.assert_allclose(
                value, expected_gradient[name].reshape(value.shape), atol=1e-4, rtol=1e-4, err_msg=name
            )
        assert any(np.any(value != 0) for value in actual_gradient.values())
        logits = eqx.filter_jit(lambda m: m(example.tokens, example.attn_mask, key=key).array)(native)
        np.testing.assert_allclose(logits, expected_logits, atol=1e-4, rtol=1e-4)
        if input_order == "batch-first":
            np.testing.assert_allclose(
                native.native(example.tokens.array, key=key), expected_logits, atol=1e-4, rtol=1e-4
            )
        other_logits = native(example.tokens, example.attn_mask, key=jax.random.key(18)).array
        assert not np.allclose(logits, other_logits, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            inference_mode(native, True)(example.tokens, example.attn_mask).array,
            expected_inference,
            atol=1e-4,
            rtol=1e-4,
        )
        with pytest.raises(RuntimeError, match="requires a key"):
            native.compute_next_token_loss(example)


def _lora_paged_outputs(model, key):
    table = PageTable.init(max_pages=4, max_seqs=1, page_size=2, max_pages_per_seq=2)
    state = SequenceTable.init(table.max_seqs, table.pages_per_seq, table.page_size)
    state, _ = state.reserve_slot(0)
    cache = model.initial_cache(table.spec(), dtype=jnp.float32)
    outputs = []
    for values, positions in (([1, 2], [0, 1]), ([3, 4], [2, 3])):
        tokens = hax.named(jnp.array(values, dtype=jnp.int32), "position")
        pos = hax.named(jnp.array(positions, dtype=jnp.int32), "position")
        slots = hax.named(jnp.zeros(2, dtype=jnp.int32), "position")
        state, table, info = state.allocate_for_seq(table, slots, pos)
        output, cache = eqx.filter_jit(lambda m, c: m.decode(tokens, c, info, pos, key=key))(model, cache)
        outputs.append(np.asarray(output.array))
    return np.concatenate(outputs), [np.asarray(layer.kv_pages.array) for layer in cache]


@pytest.mark.tpu
@pytest.mark.timeout(120)
def test_native_lora_dropout_paged_prefill_continuation_matches_named():
    options = dict(hidden_dim=8, intermediate_dim=12, num_layers=2, num_heads=2, num_kv_heads=1, max_seq_len=4)
    adapter = LoraConfig(r=2, alpha=2, dropout=0.25, a_init_mode="random")
    key = jax.random.key(17)
    named_mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(named_mesh), hax.axis_mapping({}):
        named = LlamaConfig(**options, attn_backend="vanilla").build(hax.Axis("vocab", 16), key=jax.random.key(0))
        source = jax.tree.map(np.asarray, named.to_state_dict())
        named = loraize(named, adapter, key=jax.random.key(2))
        factors = jax.tree.map(np.asarray, lora_state_dict(named))
        expected, expected_cache = _lora_paged_outputs(named, key)
        expected_inference, _ = _lora_paged_outputs(inference_mode(named, True), None)
    native_mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(native_mesh):
        native = (
            NativeLlamaLmConfig(**options)
            .build(hax.Axis("vocab", 16), key=jax.random.key(1), sharding=NativeLlamaSharding(fsdp_axis="data"))
            .from_state_dict(source)
        )
        native = _native_with_named_lora_factors(loraize(native, adapter, key=jax.random.key(3)), factors)
        actual, actual_cache = _lora_paged_outputs(native, key)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        for actual_layer, expected_layer in zip(actual_cache, expected_cache, strict=True):
            np.testing.assert_allclose(actual_layer, expected_layer, atol=1e-4, rtol=1e-4)
        inferred, _ = _lora_paged_outputs(inference_mode(native, True), None)
        np.testing.assert_allclose(inferred, expected_inference, atol=1e-4, rtol=1e-4)
        assert not np.allclose(actual, inferred, atol=1e-5, rtol=1e-5)
        with pytest.raises(RuntimeError, match="requires a key"):
            _lora_paged_outputs(native, None)
