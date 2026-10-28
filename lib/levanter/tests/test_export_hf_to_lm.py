# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.state_dict import to_torch_compatible_state_dict
from transformers import AutoTokenizer, LlamaConfig as HfLlamaConfig

import haliax

import levanter.main.export_hf_to_lm as export_hf_to_lm
import levanter.main.export_lm_to_hf as export_lm_to_hf
from levanter.testing import tiny_corpus
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.layers.attention import AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.llama_config import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.testing.hf_checkpoints import write_llama_checkpoint
from levanter.testing.helpers import use_test_mesh
from levanter.trainer import TrainerConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.jax_utils import is_inexact_arrayish, local_cpu_mesh
from levanter.utils.mesh import MeshConfig

# Pad the model embedding past the tokenizer vocab (as Qwen does for TPU efficiency), so the
# emit_padded_tokenizer path has something to pad.
_VOCAB_PAD = 8


def test_import_hf_to_lm_saves_model_under_subpath_and_emits_padded_tokenizer():
    """levanter -> HF -> levanter round-trip through export_hf_to_lm.

    The model is saved under a ``model`` subtree (so it loads with ``subpath="model"``, matching a
    TrainerState's model field) and a tokenizer padded to the model vocab is emitted at the output root.
    """
    model_config = Gpt2Config(num_layers=2, num_heads=2, max_seq_len=32, hidden_dim=32, use_flash_attention=True)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        tokenizer = tiny_corpus.tiny_corpus_config(tmpdir).the_tokenizer
        vocab_size = len(tokenizer) + _VOCAB_PAD
        Vocab = haliax.Axis("vocab", vocab_size)
        model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

        # levanter -> HF: write config.json + safetensors + the (unpadded) tokenizer. No torch needed.
        levanter_ckpt = os.path.join(tmpdir, "levanter_ckpt")
        trainable, _ = eqx.partition(model, is_inexact_arrayish)
        save_checkpoint({"model": trainable}, 0, levanter_ckpt)
        hf_dir = os.path.join(tmpdir, "hf")
        export_lm_to_hf.main(
            export_lm_to_hf.ConvertLmConfig(
                trainer=SimpleNamespace(device_mesh=contextlib.nullcontext(), parameter_axis_mapping={}),
                checkpoint_path=levanter_ckpt,
                checkpoint_subpath="model",
                output_dir=hf_dir,
                model=model_config,
                override_vocab_size=vocab_size,  # the embedding is padded past the tokenizer
                use_cpu=True,
            )
        )

        # HF -> levanter: the model as a `model/` subtree + a padded tokenizer at the output root.
        native_dir = os.path.join(tmpdir, "native")
        export_hf_to_lm.main(
            export_hf_to_lm.ImportHfConfig(
                hf_checkpoint=hf_dir,
                output_path=native_dir,
                model=model_config,
                use_hf_model_config=False,
                dtype="float32",
                resize_vocab_to_match_tokenizer=False,
                subpath="model",
                emit_padded_tokenizer=True,
            )
        )

        # The checkpoint root holds the metadata; the model is a `model` subtree inside it (an OCDBT
        # key prefix, not an on-disk dir), reachable with subpath="model" below. The tokenizer is at the root.
        assert os.path.exists(os.path.join(native_dir, "metadata.json"))
        # The emitted tokenizer is padded up to the model vocab, so a downstream Vocab axis built from
        # len(tokenizer) matches the checkpoint embedding.
        padded_tokenizer = load_tokenizer(native_dir)
        assert len(padded_tokenizer) == vocab_size
        assert len(padded_tokenizer) == len(tokenizer) + _VOCAB_PAD

        # The model loads back from the `model` subtree with weights identical to the original.
        reloaded = load_checkpoint(model, native_dir, subpath="model")
        for original_leaf, reloaded_leaf in zip(
            jax.tree_util.tree_leaves(eqx.filter(model, is_inexact_arrayish)),
            jax.tree_util.tree_leaves(eqx.filter(reloaded, is_inexact_arrayish)),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(original_leaf), np.asarray(reloaded_leaf), rtol=1e-5, atol=1e-5)


@pytest.fixture
def native_import_source(tmp_path, tiny_hf_tokenizer_path):
    native_config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        activation_function=ActivationFunctionEnum.gelu_new,
        rope=DefaultRotaryEmbeddingsConfig(theta=50000, factor=2),
    )
    named_config = LlamaConfig(
        **{field.name: getattr(native_config, field.name) for field in dataclasses.fields(LlamaConfig)}
    )
    source = write_llama_checkpoint(
        tmp_path / "hf-source", named_config, native_config.to_hf_config(32), key=jax.random.PRNGKey(17)
    )
    AutoTokenizer.from_pretrained(tiny_hf_tokenizer_path).save_pretrained(source.path)
    return source


@pytest.mark.tpu
@pytest.mark.parametrize(
    "infer_config, dtype, resize_vocab, emit_padded_tokenizer",
    [
        pytest.param(False, "float32", False, True, id="explicit-full"),
        pytest.param(True, "bfloat16", True, False, id="inferred-resized-bf16"),
    ],
)
def test_native_import_preserves_source_semantics_vocab_and_storage_dtype(
    native_import_source, tmp_path, infer_config, dtype, resize_vocab, emit_padded_tokenizer
):
    source = native_import_source
    tokenizer = load_tokenizer(str(source.path))
    native_config = NativeLlamaLmConfig.from_hf_config(HfLlamaConfig.from_pretrained(source.path))
    supplied_config = dataclasses.replace(
        native_config,
        reference_checkpoint=str(tmp_path / "unavailable-reference"),
        tokenizer=None,
        activation_function=ActivationFunctionEnum.silu if infer_config else native_config.activation_function,
    )
    output = tmp_path / "imported"
    export_hf_to_lm.main(
        export_hf_to_lm.ImportHfConfig(
            hf_checkpoint=str(source.path),
            output_path=str(output),
            model=supplied_config,
            use_hf_model_config=infer_config,
            tokenizer=str(source.path),
            dtype=dtype,
            resize_vocab_to_match_tokenizer=resize_vocab,
            subpath="model",
            emit_padded_tokenizer=emit_padded_tokenizer,
        )
    )
    stored_vocab = len(tokenizer) if resize_vocab else 32
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["levanter_model"]["vocab_size"] == stored_vocab
    if emit_padded_tokenizer:
        padded_tokenizer = load_tokenizer(str(output))
        assert len(padded_tokenizer) == stored_vocab
        assert padded_tokenizer.encode("hello world") == tokenizer.encode("hello world")

    model_size = 2 if len(jax.devices()) >= 4 and not resize_vocab else 1
    trainer = TrainerConfig(
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": len(jax.devices()) // model_size, "model": model_size}),
    )
    batch = haliax.Axis("batch", 2 * trainer.device_mesh.shape["data"])
    position = haliax.Axis("position", 4)
    token_ids = np.arange(batch.size * position.size).reshape(batch.size, position.size) % stored_vocab
    token_ids[-1, -1] = stored_vocab - 1
    storage_dtype = jnp.dtype(dtype)
    with local_cpu_mesh():
        named = source.config.build(haliax.Axis("vocab", 32), key=jax.random.PRNGKey(17))
        for name, value in to_torch_compatible_state_dict(named).items():
            np.testing.assert_array_equal(value, source.weights[name])
        # Compare source semantics in FP32 after the requested storage rounding.
        expected = jax.tree.map(
            lambda value: value.astype(storage_dtype).astype(jnp.float32) if eqx.is_inexact_array(value) else value,
            named,
        ).resize_vocab(stored_vocab)
        expected_weights = to_torch_compatible_state_dict(expected)
        tokens = haliax.named(jnp.asarray(token_ids), (batch, position))
        expected_logits = np.asarray(
            eqx.filter_jit(lambda m, ids: m(ids, AttentionMask.causal()).array)(expected, tokens)
        )

    shutil.rmtree(source.path)
    with trainer.use_device_mesh():
        loader_args = dict(
            Vocab=haliax.Axis("vocab", len(tokenizer)),
            axis_mapping=trainer.parameter_axis_mapping,
            compute_axis_mapping=trainer.compute_axis_mapping,
            key=jax.random.PRNGKey(0),
        )
        restored = load_levanter_checkpoint(native_config, str(output), **loader_args)
        assert restored.Vocab.size == stored_vocab
        actual_weights = to_torch_compatible_state_dict(restored)
        assert actual_weights.keys() == expected_weights.keys()
        for name, value in actual_weights.items():
            assert value.dtype == storage_dtype
            np.testing.assert_array_equal(value, np.asarray(expected_weights[name]).astype(storage_dtype))
        restored_fp32 = jax.tree.map(
            lambda value: value.astype(jnp.float32) if eqx.is_inexact_array(value) else value, restored
        )
        tokens = haliax.named(jnp.asarray(token_ids), (batch, position))
        actual_logits = eqx.filter_jit(lambda m, ids: m(ids, AttentionMask.causal()).array)(restored_fp32, tokens)
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)
        with pytest.raises(ValueError, match="architecture"):
            load_levanter_checkpoint(
                dataclasses.replace(native_config, activation_function=ActivationFunctionEnum.silu),
                str(output),
                **loader_args,
            )


@pytest.mark.parametrize("semantic", ["activation", "rope"])
def test_native_import_rejects_same_shape_source_mismatch_before_writing(native_import_source, tmp_path, semantic):
    source = native_import_source
    native_config = NativeLlamaLmConfig.from_hf_config(HfLlamaConfig.from_pretrained(source.path))
    native_config = dataclasses.replace(native_config, reference_checkpoint=str(source.path))
    if semantic == "activation":
        native_config = dataclasses.replace(native_config, activation_function=ActivationFunctionEnum.silu)
    else:
        native_config = dataclasses.replace(native_config, rope=dataclasses.replace(native_config.rope, factor=1))
    output = tmp_path / "invalid-import"
    with pytest.raises(ValueError, match="hidden_act|RoPE"):
        export_hf_to_lm.main(
            export_hf_to_lm.ImportHfConfig(
                hf_checkpoint=str(source.path),
                output_path=str(output),
                model=native_config,
                use_hf_model_config=False,
                tokenizer=str(source.path),
                dtype="float32",
                subpath="model",
            )
        )
    assert not output.exists()
