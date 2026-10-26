# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import shutil

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax.state_dict import to_torch_compatible_state_dict
from jax.sharding import AxisType, Mesh, PartitionSpec as P
from transformers import AutoTokenizer

from levanter.checkpoint import save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.layers.attention import AttentionMask
from levanter.main import inference_repl
from levanter.model_loading import load_hf_checkpoint
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.testing.hf_checkpoints import write_llama_checkpoint
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig


@pytest.fixture
def repl_source(tmp_path, tiny_hf_tokenizer_path):
    config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
    )
    source = write_llama_checkpoint(
        tmp_path / "hf-source", config, config.to_hf_config(32), key=jax.random.PRNGKey(13)
    )
    AutoTokenizer.from_pretrained(tiny_hf_tokenizer_path).save_pretrained(source.path)
    return source


@pytest.mark.tpu
@pytest.mark.parametrize(
    "native,source_kind,dtype,cache_mapping",
    [
        pytest.param(True, "hf", jnp.float32, "explicit", id="native-hf-f32"),
        pytest.param(True, "hf", jnp.bfloat16, "explicit", id="native-hf-bf16"),
        pytest.param(True, "checkpoint", jnp.float32, "explicit", id="native-checkpoint-f32"),
        pytest.param(True, "checkpoint", jnp.bfloat16, "explicit", id="native-checkpoint-bf16"),
        pytest.param(False, "checkpoint", jnp.float32, "ordinary", id="named-checkpoint-source-absent"),
        pytest.param(True, "hf", jnp.float32, "ordinary", id="native-hf-f32-ordinary-cache"),
    ],
)
def test_repl_main_local_source_completion_preserves_precision_and_matches_named_oracle(
    repl_source, tmp_path, monkeypatch, record_property, native, source_kind, dtype, cache_mapping
):
    source = repl_source
    tokenizer_path = tmp_path / "outer-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(source.path)
    if not native:
        # Named checkpoint templates retain their tokenizer-sized vocabulary contract.
        tokenizer.add_tokens(["token31"])
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer = load_tokenizer(str(tokenizer_path))
    assert len(tokenizer) == (31 if native else 32)
    policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=dtype, output_dtype=dtype)
    with hax.partitioning.set_mesh(Mesh(np.asarray(jax.devices()), ("data",))):
        oracle = source.config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(13))
        for name, weights in to_torch_compatible_state_dict(oracle).items():
            np.testing.assert_array_equal(weights, source.weights[name])
        oracle = policy.cast_to_compute(oracle)
        sequence = [3, 4, 5]
        for _ in range(3):
            tokens = hax.named(jnp.asarray(sequence, dtype=jnp.int32), "position")
            logits = eqx.filter_jit(lambda model, inputs: model(inputs, AttentionMask.causal()).array)(oracle, tokens)[
                -1
            ]
            sequence.append(int(jnp.argmax(logits)))
    model_size = 2 if len(jax.devices()) >= 4 else 1
    trainer = TrainerConfig(
        mp=policy,
        mesh=MeshConfig(
            axes={"data": len(jax.devices()) // model_size, "model": model_size},
            compute_mapping={"kv_head": "model"} if cache_mapping == "explicit" else {},
        ),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        tracker=NoopConfig(),
        log_dir=tmp_path / "logs",
    )
    model_type = NativeLlamaLmConfig if native else LlamaConfig
    model_config = model_type.from_hf_config(source.config.to_hf_config(32))
    path = source.path
    if source_kind == "checkpoint":
        model_config = dataclasses.replace(
            model_config, reference_checkpoint=str(source.path), tokenizer=str(tokenizer_path)
        )
        with dataclasses.replace(trainer, use_explicit_mesh_axes=native).use_device_mesh():
            model = load_hf_checkpoint(
                model_config,
                str(source.path),
                tokenizer=tokenizer,
                compute_dtype=jnp.float32,
                axis_mapping=trainer.parameter_axis_mapping,
                compute_axis_mapping=trainer.compute_axis_mapping,
                batch_axis_name=trainer.batch_axis_name,
            )
            path = tmp_path / "levanter" / "step-1"
            save_checkpoint({"model": model}, 1, str(path), metadata=model.checkpoint_metadata() if native else None)
        model_config = dataclasses.replace(model_config, reference_checkpoint=str(source.path), tokenizer=None)
        shutil.rmtree(source.path)
        assert not source.path.exists()
    else:
        # HF architecture inference must replace this deliberately different template.
        model_config = dataclasses.replace(model_config, hidden_dim=32, num_layers=2)
    config = inference_repl.InferenceReplConfig(
        checkpoint=str(path),
        tokenizer=None if source_kind == "hf" and dtype == jnp.float32 else str(tokenizer_path),
        model=model_config,
        trainer=trainer,
        server=InferenceServerConfig(
            temperature=0.0,
            service=InferenceEngineConfig(
                max_seq_len=16,
                max_seqs=2,
                max_pages=24,
                page_size=2,
                max_queued_tokens=16,
                max_seqs_in_prefill=1,
                max_prefill_size=8,
                max_rounds=4,
                max_tokens_per_round=2,
            ),
        ),
        command="complete",
        args="token3 token4 token5",
        max_tokens=3,
    )
    responses = []
    contexts = []
    original_print = inference_repl.ReplContext._print_completion_response

    def capture_terminal_response(context, response):
        # Observe terminal output only; loading, queues, threads and decoding remain real.
        contexts.append(context)
        responses.append(response)
        original_print(context, response)

    monkeypatch.setattr(inference_repl.ReplContext, "_print_completion_response", capture_terminal_response)
    try:
        inference_repl.main(config)
        assert len(responses) == 1
        response = responses[0]
        assert tokenizer.encode(response.choices[0].text, add_special_tokens=False) == sequence[3:]
        assert response.usage.prompt_tokens == 3
        assert response.usage.completion_tokens == 3
        context = contexts[0].server.inference_context
        loaded = context.model
        assert loaded.Vocab.size == 32
        assert len(context.tokenizer) == len(tokenizer)
        assert context.tokenizer.encode(config.args, add_special_tokens=False) == [3, 4, 5]
        actual_weights = to_torch_compatible_state_dict(loaded)
        assert set(actual_weights) == set(source.weights)
        for name, expected in source.weights.items():
            np.testing.assert_array_equal(actual_weights[name], np.asarray(expected).astype(dtype))
        arrays = [value for value in jax.tree.leaves(loaded) if eqx.is_inexact_array(value)]
        assert all(value.dtype == dtype for value in arrays)
        axis_type = AxisType.Explicit if native else AxisType.Auto
        for value in arrays:
            assert value.sharding.mesh.axis_types == (axis_type,) * len(value.sharding.mesh.axis_names)
            assert value.sharding.mesh.shape["data"] == len(jax.devices()) // model_size
            assert value.sharding.mesh.shape["model"] == model_size
        if native:
            model = loaded.native
            layers = model.layers.stacked
            expected_shardings = [
                (model.token_embeddings, P("model", "data")),
                (model.lm_head.weight, P("model", "data")),
                (model.norm.weight, P("data")),
                (layers.input_layernorm.weight, P(None, "data")),
                (layers.post_attention_layernorm.weight, P(None, "data")),
                (layers.self_attn.q_proj.weight, P(None, "model", "data")),
                (layers.self_attn.k_proj.weight, P(None, "model", "data")),
                (layers.self_attn.v_proj.weight, P(None, "model", "data")),
                (layers.self_attn.o_proj.weight, P(None, "data", "model")),
                (layers.mlp.up_proj.weight, P(None, "model", "data")),
                (layers.mlp.gate_proj.weight, P(None, "model", "data")),
                (layers.mlp.down_proj.weight, P(None, "data", "model")),
            ]
            assert {id(value) for value in arrays} == {id(value) for value, _ in expected_shardings}
            for value, expected in expected_shardings:
                assert value.sharding.spec == expected
            for cache in context.engine.gen_state.cache:
                assert cache.kv_pages.array.sharding.spec == P(None, None, "model", None)
                assert cache.kv_pages.array.dtype == dtype
        record_property(
            "repl_runtime",
            json.dumps(
                {
                    "native": native,
                    "source_kind": source_kind,
                    "compute_dtype": str(dtype),
                    "source_vocab": 32,
                    "tokenizer_vocab": len(context.tokenizer),
                    "generated_tokens": sequence[3:],
                    "mesh": dict(arrays[0].sharding.mesh.shape),
                    "axis_types": [str(t) for t in arrays[0].sharding.mesh.axis_types],
                    "parameter_specs": [str(value.sharding.spec) for value in arrays],
                },
                sort_keys=True,
            ),
        )
    finally:
        for context in contexts:
            context.server.shutdown()
