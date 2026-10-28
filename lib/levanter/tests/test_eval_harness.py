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
import pytest
from levanter.testing.helpers import skip_if_module_missing
from transformers import AutoTokenizer

from levanter.checkpoint import save_checkpoint
from levanter.data.packing import PromptCompletion
from levanter.distributed import DistributedConfig
from levanter.eval_harness import (
    EvalHarnessMainConfig,
    LmEvalHarnessConfig,
    _call_with_retry,
    _iterate_tokenized_requests,
    run_eval_harness_main,
)
from levanter.eval_harness_config import TaskConfig
from levanter.layers.attention import AttentionMask
from levanter.models.llama_config import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig, native_llama_sharding
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig


@skip_if_module_missing("lm_eval")
@pytest.mark.tpu
@pytest.mark.parametrize("model_kind", ["named", "native"])
@pytest.mark.parametrize("checkpoint_format", ["levanter", "hf"])
def test_eval_harness_main_local_checkpoint_matches_full_forward(
    tmp_path, tiny_hf_tokenizer_path, monkeypatch, model_kind, checkpoint_format
):
    monkeypatch.chdir(tmp_path)
    model_type = NativeLlamaLmConfig if model_kind == "native" else LlamaConfig
    model_config = model_type(
        hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=2, max_seq_len=16
    )
    model_size = 2 if len(jax.devices()) >= 4 else 1
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mp=jmp.get_policy("p=f32,c=f32"),
        per_device_eval_parallelism=2,
        mesh=MeshConfig(
            axes={"data": len(jax.devices()) // model_size, "model": model_size},
            batch_axis_name="examples",
            shared_mapping={"kv_head": "model"},
            compute_mapping={"examples": ["data"]},
        ),
    )
    tokenizer = load_tokenizer(tiny_hf_tokenizer_path)
    build_trainer = dataclasses.replace(trainer, use_explicit_mesh_axes=model_config.requires_explicit_mesh_axes)
    checkpoint = tmp_path / "checkpoint"
    contexts = [[3, 4], [6]]
    targets = [5, 7]
    logprobs = []
    with build_trainer.use_device_mesh():
        key = jax.random.PRNGKey(23)
        # Native checkpoint metadata must recover the padded row absent from the tokenizer.
        vocab = hax.Axis("vocab", 32) if model_kind == "native" else hax.Axis("vocab", len(tokenizer))
        if isinstance(model_config, NativeLlamaLmConfig):
            sharding = native_llama_sharding(
                trainer.parameter_axis_mapping, trainer.compute_axis_mapping, trainer.batch_axis_name
            )
            model = model_config.build(vocab, key=key, sharding=sharding)
            metadata = model.checkpoint_metadata()
        else:
            model = model_config.build(vocab, key=key)
            metadata = None
        for context, target in zip(contexts, targets, strict=True):
            tokens = hax.named(
                jnp.broadcast_to(jnp.asarray(context, dtype=jnp.int32), (len(jax.devices()), len(context))),
                (hax.Axis("examples", len(jax.devices())), "position"),
            )
            logits = eqx.filter_jit(lambda m, t: m(t, attn_mask=AttentionMask.causal()))(model, tokens)
            last_logits = np.asarray(logits.array)[0, -1]
            logprobs.append(float(last_logits[target] - np.logaddexp.reduce(last_logits)))
        if checkpoint_format == "hf":
            if isinstance(model_config, NativeLlamaLmConfig):
                converter = model_config.hf_export_converter(tokenizer)
            else:
                converter = dataclasses.replace(
                    model_config.hf_checkpoint_converter(tiny_hf_tokenizer_path),
                    reference_checkpoint=None,
                    tokenizer=tokenizer,
                )
            converter.save_pretrained(model, str(checkpoint), max_shard_size=512)
            assert json.loads((checkpoint / "config.json").read_text())["vocab_size"] == vocab.size
        else:
            save_checkpoint({"model": model}, 0, checkpoint, metadata=metadata)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "test.jsonl").write_text(
        "".join(
            json.dumps({"context": " ".join(f"token{token}" for token in context), "target": f" token{target}"}) + "\n"
            for context, target in zip(contexts, targets, strict=True)
        )
    )
    report = run_eval_harness_main(
        EvalHarnessMainConfig(
            model=model_config,
            trainer=trainer,
            tokenizer=tiny_hf_tokenizer_path,
            checkpoint_path=str(checkpoint),
            checkpoint_is_hf=checkpoint_format == "hf",
            eval_harness=LmEvalHarnessConfig(
                task_spec=[
                    TaskConfig(
                        task="lambada_openai",
                        task_alias="local_likelihood",
                        dataset_path=str(dataset),
                        test_split="test",
                        output_type="loglikelihood",
                        doc_to_text="context",
                        doc_to_target="target",
                        metric_list=[{"metric": "perplexity", "aggregation": "perplexity", "higher_is_better": False}],
                    )
                ],
                max_length=16,
            ),
        )
    )
    expected = np.exp(-np.mean(logprobs))
    np.testing.assert_allclose(report["results"]["local_likelihood"]["perplexity,none"], expected, rtol=1e-5)
    saved = json.loads((tmp_path / "lm_eval_harness_results.json").read_text())
    np.testing.assert_allclose(saved["results"]["local_likelihood"]["perplexity,none"], expected, rtol=1e-5)


@skip_if_module_missing("lm_eval")
def test_iterate_tokenized_requests_with_chat_template():
    """Test the chat template functionality in _iterate_tokenized_requests"""
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    # Load a tokenizer with chat template - Llama 3 has one
    hf_tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Create chat-like requests with contexts formatted via the chat template, matching harness usage
    raw_contexts = [
        "What's the best way to learn Python?",
        "Explain quantum computing.",
    ]
    completions = [
        " Practice by building small projects.",
        " Quantum computing uses quantum bits or qubits.",
    ]

    formatted_contexts = [
        hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": context}], tokenize=False, add_generation_prompt=True
        )
        for context in raw_contexts
    ]

    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(formatted_contexts[i], completions[i]),
            idx=i,
            metadata=("test_task", i, None),
        )
        for i in range(len(raw_contexts))
    ]

    # Parameters
    max_len = 100  # Larger max_len to accommodate chat template additions
    batch_size = 2

    # Run with chat template
    results = list(_iterate_tokenized_requests(requests, hf_tokenizer, max_len, batch_size))

    # Verify the results
    assert len(results) == len(requests)

    # Check each result
    for i, result in enumerate(results):
        # Should be a PromptCompletion object
        assert isinstance(result, PromptCompletion)
        assert result.segment_id == i

        # The context should have been transformed by the chat template
        context, completion = requests[i].args

        # Check completion is preserved
        completion_text = completion.strip()
        decoded = hf_tokenizer.decode(result.ids)
        assert completion_text in decoded, f"Completion not found for example {i}"

        # Verify prompt_length is correctly set
        assert result.prompt_length < len(
            result.ids
        ), f"Example {i} has invalid prompt_length ({result.prompt_length} >= {len(result.ids)})"

        # Verify that prompt_length approximately matches the expected chat context length
        # It might not match exactly due to truncation or other processing
        prompt_tokens = result.ids[: result.prompt_length]
        prompt_decoded = hf_tokenizer.decode(prompt_tokens)

        # The prompt should contain key parts of the chat context generated by the template
        if "<|start_header_id|>" in context:
            assert (
                "<|start_header_id|>" in prompt_decoded
            ), f"Chat template formatting not found in prompt for example {i}"

        # Alternative check: the prompt should be longer than the original context
        # due to the chat template adding formatting
        original_context_tokens = hf_tokenizer(raw_contexts[i], truncation=False, padding=False)["input_ids"]
        assert result.prompt_length > len(
            original_context_tokens
        ), f"Prompt length not increased by chat template for example {i}"


@skip_if_module_missing("lm_eval")
def test_iterate_tokenized_requests():
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    hf_tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},  # Empty dict as it's not used in the function
            arguments=("What is the capital of France?", " Paris"),
            idx=0,
            metadata=("test_task", 0, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("The quick brown fox", " jumps over the lazy dog"),
            idx=1,
            metadata=("test_task", 1, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("To be or not to be,", " that is the question"),
            idx=2,
            metadata=("test_task", 2, None),
        ),
    ]

    # Parameters
    max_len = 50
    batch_size = 2

    # Run the function
    results = list(_iterate_tokenized_requests(requests, hf_tokenizer, max_len, batch_size))

    # Verify results
    assert len(results) == 3  # One result per request

    for i, result in enumerate(results):
        # Check basic structure
        assert isinstance(result, PromptCompletion)
        assert result.segment_id == i

        # Check token content
        context, completion = requests[i].args
        decoded = hf_tokenizer.decode(result.ids)

        # The decoded tokens should contain the original text
        assert context in decoded or decoded.startswith(context.strip())
        assert completion in decoded or decoded.endswith(completion.strip())

        # Verify prompt_length
        prompt_tokens = result.ids[: result.prompt_length]
        decoded_prompt = hf_tokenizer.decode(prompt_tokens)

        # The decoded prompt should match or contain the context
        assert context.strip() in decoded_prompt

        # Sequence length should be within max_len
        assert len(result.ids) <= max_len


@skip_if_module_missing("lm_eval")
def test_task_config():
    task_spec = [
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_10shot",
            num_fewshot=10,
        ),
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_5shot",
            num_fewshot=5,
        ),
        "lambada_openai",
    ]

    config = LmEvalHarnessConfig(
        task_spec=task_spec,
    )

    q = config.to_task_dict()

    assert len(q) == 3


def test_call_with_retry_does_not_sleep_after_the_last_attempt(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("levanter.eval_harness.time.sleep", sleeps.append)

    attempts = 0

    def always_fails():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
        _call_with_retry(always_fails, max_retries=3, base_delay=1.0)

    assert attempts == 3
    assert len(sleeps) == 2
