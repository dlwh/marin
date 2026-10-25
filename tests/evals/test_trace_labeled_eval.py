# Copyright The Marin Authors
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
from levanter.checkpoint import save_checkpoint
from levanter.data.sharded_datasource import FirstRowsShardedDataSource, ShardedDataSource
from levanter.data.text.datasets import UrlDatasetSourceConfig
from levanter.data.text.trace_chat import TraceChatEvaluationFormat
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig, native_llama_sharding
from levanter.testing.hf_checkpoints import write_llama_checkpoint
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.trace_labeled_eval import (
    DEFAULT_OUTCOME_JUDGE_PROMPT,
    TraceLabeledEvalConfig,
    TraceLabeledEvalDatasetConfig,
    TraceRowAdapterConfig,
    _source_for_dataset,
    trace_labeled_eval,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


class FailingAfterLimitSource(ShardedDataSource[int]):
    def __init__(self):
        self.rows_read = 0

    @property
    def shard_names(self):
        return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int):
        if shard_name != "data":
            raise ValueError(f"Unknown shard {shard_name!r}")

        for value in range(row, 3):
            if value >= 2:
                raise RuntimeError("read past requested rows")
            self.rows_read += 1
            yield value


def _write_local_tokenizer(path, corpus: list[str]) -> None:
    tokens = ["[UNK]"]
    seen = set(tokens)
    for text in corpus:
        for token in text.split():
            if token not in seen:
                seen.add(token)
                tokens.append(token)

    tokenizer = Tokenizer(WordLevel(vocab={token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"unk_token": "[UNK]"}))


def test_first_rows_source_does_not_pull_one_extra_row():
    source = FailingAfterLimitSource()
    limited_source = FirstRowsShardedDataSource(source, max_rows=2)

    assert list(limited_source.open_shard("data")) == [0, 1]
    assert source.rows_read == 2


@pytest.fixture
def trace_dataset(tmp_path):
    row = {
        "trajectory": [
            {"role": "system", "system_prompt": "You are a coding agent."},
            {"role": "user", "content": [{"type": "text", "text": "Fix the bug."}]},
            {"role": "ai", "text": "I will inspect the repo.", "thinking": "Need to inspect failing tests."},
        ],
        "test_result": {"git_patch": "diff --git a/a.py b/a.py"},
        "resolved": False,
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")

    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    _write_local_tokenizer(
        tokenizer_path,
        [
            "You are a coding agent.",
            "Fix the bug.",
            "I will inspect the repo.",
            "Need to inspect failing tests.",
            "Final Patch:\ndiff --git a/a.py b/a.py",
            DEFAULT_OUTCOME_JUDGE_PROMPT,
            "INCORRECT",
            "INCORRECT<|eot_id|>",
        ],
    )
    trace_format = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=MARIN_CHAT_TEMPLATE,
        loss_tags=("patch", "outcome"),
    )
    return TraceLabeledEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(path)]),
        split="train",
        trace_format=trace_format,
        row_adapter=TraceRowAdapterConfig(
            input_messages_field="trajectory",
            patch_field="test_result.git_patch",
            outcome_field="resolved",
        ),
    )


def test_trace_row_adapter_adds_patch_and_outcome_targets(tmp_path, trace_dataset):
    tokenizer = load_tokenizer(str(tmp_path / "tokenizer"))
    dataset_config = trace_dataset
    trace_format = dataset_config.trace_format
    adapted_row = next(iter(_source_for_dataset(dataset_config)))
    assert adapted_row["trace_outcome_label"] == "INCORRECT"
    assert adapted_row["messages"][0]["content"] == "You are a coding agent."
    assert adapted_row["messages"][1]["content"] == "Fix the bug."
    assert adapted_row["messages"][2]["role"] == "assistant"
    assert adapted_row["messages"][2]["reasoning_content"] == "Need to inspect failing tests."
    assert adapted_row["messages"][-2] == {
        "role": "user",
        "content": DEFAULT_OUTCOME_JUDGE_PROMPT,
    }
    assert adapted_row["messages"][-1] == {
        "role": "assistant",
        "content": "INCORRECT",
        "loss_tags": ["outcome"],
    }

    processed = trace_format.build_preprocessor(tokenizer)([adapted_row])[0]
    input_ids = processed["input_ids"]
    labels = processed["loss_labels"]
    patch_text = tokenizer.decode(input_ids[labels == 4].tolist(), skip_special_tokens=False)
    outcome_text = tokenizer.decode(input_ids[labels == 5].tolist(), skip_special_tokens=False)

    assert "Final Patch:" in patch_text
    assert "diff --git" in patch_text
    assert "INCORRECT" in outcome_text


@pytest.mark.tpu
@pytest.mark.parametrize("model_source", ["named-hf", "native-hf", "native-local"])
def test_trace_labeled_eval_matches_named_labeled_loss_and_preserves_config(tmp_path, trace_dataset, model_source):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = load_tokenizer(str(tokenizer_path))
    named_config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=256,
        attn_backend="vanilla",
        reference_checkpoint=str(tokenizer_path),
        tokenizer=str(tokenizer_path),
    )
    source_vocab = len(tokenizer) + 1
    source = write_llama_checkpoint(
        tmp_path / "hf-source", named_config, named_config.to_hf_config(source_vocab), key=jax.random.PRNGKey(13)
    )
    adapted_row = next(iter(_source_for_dataset(trace_dataset)))
    processed = trace_dataset.trace_format.build_preprocessor(tokenizer)([adapted_row])[0]
    token_ids = processed["input_ids"]
    with local_cpu_mesh():
        named = named_config.build(hax.Axis("vocab", source_vocab), key=jax.random.PRNGKey(13))
        for name, value in to_torch_compatible_state_dict(named).items():
            np.testing.assert_array_equal(value, source.weights[name])
        tokens = hax.named(jnp.asarray(token_ids), "position")
        logits = eqx.filter_jit(lambda m, ids: m(ids, AttentionMask.causal()).array)(named, tokens)
        log_probs = np.asarray(jax.nn.log_softmax(logits, axis=-1))[:-1]
    token_losses = -np.take_along_axis(log_probs, token_ids[1:, None], axis=-1).squeeze(-1)
    label_spec = trace_dataset.trace_format.loss_label_spec()
    expected_losses = {}
    expected_counts = {}
    for label, label_ids in label_spec.aggregates.items():
        selected = np.isin(processed["loss_labels"][1:], label_ids)
        expected_counts[label] = int(selected.sum())
        expected_losses[label] = float(token_losses[selected].mean())
    assert expected_counts == {"patch": 7, "outcome": 1}

    if model_source == "named-hf":
        mesh = MeshConfig(batch_axis_name="examples")
        data_size = len(jax.devices())
    else:
        model_size = 2 if len(jax.devices()) >= 4 else 1
        data_size = len(jax.devices()) // model_size
        mesh = MeshConfig(
            axes={"data": data_size, "vocab": model_size},
            batch_axis_name="examples",
            shared_mapping={"heads": "vocab", "mlp": "vocab"},
            compute_mapping={"examples": ("replica_dcn", "replica", "data"), "vocab": "vocab"},
            param_mapping={"embed": "data", "vocab": "vocab"},
        )
    trainer = TrainerConfig(
        mesh=mesh,
        mp=jmp.get_policy("f32"),
        train_batch_size=data_size,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        tracker=NoopConfig(),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        shutdown_at_exit=False,
        log_dir=tmp_path / "logs",
        id=model_source,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    native_config = NativeLlamaLmConfig(
        **{field.name: getattr(named_config, field.name) for field in dataclasses.fields(named_config)}
    )
    model_config = named_config if model_source == "named-hf" else native_config
    checkpoint_path = source.path
    if model_source == "native-local":
        checkpoint_path = tmp_path / "native-checkpoint"
        checkpoint_trainer = dataclasses.replace(trainer, use_explicit_mesh_axes=True)
        with checkpoint_trainer.use_device_mesh():
            sharding = native_llama_sharding(
                trainer.parameter_axis_mapping, trainer.compute_axis_mapping, trainer.batch_axis_name
            )
            native = native_config.build(hax.Axis("vocab", source_vocab), key=jax.random.PRNGKey(0), sharding=sharding)
            native = native.from_state_dict(source.weights)
            for name, value in native.to_state_dict().items():
                np.testing.assert_array_equal(value, source.weights[name])
            save_checkpoint(
                {"model": native},
                0,
                str(checkpoint_path),
                is_temporary=False,
                metadata=native.checkpoint_metadata(),
            )
    output = tmp_path / "results"
    config = TraceLabeledEvalConfig(
        name=model_source,
        checkpoint_path=str(checkpoint_path),
        checkpoint_is_hf=model_source != "native-local",
        tokenizer=str(tokenizer_path),
        model=model_config,
        datasets={"tiny": trace_dataset},
        trainer=trainer,
        max_eval_length=named_config.max_seq_len,
        output_path=str(output),
        dataset_eval_max_attempts=1,
    )
    original_config = dataclasses.asdict(config)
    trace_labeled_eval(config)
    assert dataclasses.asdict(config) == original_config
    results = json.loads((output / "results.json").read_text())
    assert results["status"] == "completed"
    assert results["completed_datasets"] == 1
    metrics = results["datasets"]["tiny"]["metrics"]
    for label, expected in expected_losses.items():
        np.testing.assert_allclose(metrics[f"trace_labeled_eval/tiny/{label}/loss"], expected, atol=1e-5, rtol=1e-5)
        assert metrics[f"trace_labeled_eval/tiny/{label}/tokens"] == expected_counts[label]
    assert (output / "cache" / "tiny").exists()

    if model_source == "native-local":
        shutil.rmtree(output / "cache")
        (tmp_path / "traces.jsonl").unlink()
        trace_labeled_eval(config)
        resumed = json.loads((output / "results.json").read_text())
        assert resumed["status"] == "completed"
        assert resumed["datasets"] == results["datasets"]
        assert not (output / "cache").exists()
        assert dataclasses.asdict(config) == original_config
