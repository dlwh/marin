# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
import safetensors.numpy
from transformers import LlamaConfig as HfLlamaConfig
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, set_mesh
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict
from haliax.quantization import apply_updates, partition_for_grad_overwrite
from haliax.jax_utils import is_jax_array_like
from jax.sharding import Mesh, PartitionSpec as P
from safetensors import safe_open

from levanter.data.loader import DataLoader
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.data.text.preference import (
    DpoExample,
    PreferenceChatLmDatasetFormat,
    PreferenceChatProcessor,
    PreferenceLmDataConfig,
    PreferencePairDataset,
)
from levanter.adaptor import AdaptorExportConfig, LoraAdaptorConfig, NoAdaptorConfig
from levanter.adaptor.lora import lora_state_dict, merge_lora_modules, unwrap_lora_modules
from levanter.callbacks import StepInfo
from levanter.dpo import (
    CachedDpoExample,
    DpoModel,
    ReferenceEvalCacheConfig,
    ValidationDatasetSpec,
    build_or_load_reference_eval_cache,
    dpo_loss,
    dpo_loss_from_logps,
    load_reference_eval_cache,
    logp_sum,
    reference_eval_cache_metadata,
    reference_eval_cache_path,
)
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path, load_checkpoint, save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.main.model_init import prepare_model_init_context
from levanter.model_loading import load_hf_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.lora_dpo import LoraDpoConfig, _translate_legacy_lora_dpo_config
from levanter.main.train_dpo import (
    AdapterBaseReferenceConfig,
    SeparateReferenceConfig,
    TrainDpoConfig,
    _build_dpo_dataset,
    _build_model,
    _initialize_policy_model_from_source,
    _load_separate_reference_model,
    main as train_dpo_main,
    _derive_training_keys,
    _install_separate_reference_export_hooks,
    _periodic_eval_callback,
    _restore_policy_model_from_partial_checkpoint,
    _scheduled_eval_callback,
    _validate_dpo_config,
)
from levanter.metrics import Metric
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.lm_model import LmExample
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig, native_llama_sharding
from levanter.optim.config import AdamConfig
from levanter.optim.model_averaging import ModelAveraging
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from levanter.tokenizers import MarinTokenizer, load_tokenizer as load_marin_tokenizer
from levanter.trainer_state import TrainerState, saveable_training_mask, trainables_only
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig, TrainerHooks
from levanter.tracker import NoopTracker, current_tracker
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.utils.mesh import MeshConfig
from levanter.testing.helpers import use_test_mesh
from levanter.testing.hf_checkpoints import write_llama_checkpoint
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.tree_utils import inference_mode


MODEL_NAME = "marin-community/marin-tokenizer"


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tok = load_marin_tokenizer(MODEL_NAME)
        return Path(tok.name_or_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {exc}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def _load_tokenizer(tokenizer_path: Path) -> MarinTokenizer:
    return load_marin_tokenizer(str(tokenizer_path))


def _tiny_gpt2_config() -> Gpt2Config:
    return Gpt2Config(
        max_seq_len=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )


def _build_policy_model(config: Gpt2Config):
    Vocab = hax.Axis("vocab", 32)
    return config.build(Vocab, key=jrandom.PRNGKey(0))


def _make_dpo_example(*, seq_len: int = 8, vocab_size: int = 32, key=jrandom.PRNGKey(0)) -> DpoExample:
    Pos = hax.Axis("position", seq_len)
    chosen_key, rejected_key = jrandom.split(key)
    chosen = LmExample.causal(hax.random.randint(chosen_key, Pos, 0, vocab_size))
    rejected = LmExample.causal(hax.random.randint(rejected_key, Pos, 0, vocab_size))
    return DpoExample(chosen=chosen, rejected=rejected)


def test_trainer_state_init_with_policy_model():
    config = _tiny_gpt2_config()
    model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=True,
        mp=jmp.get_policy("f32"),
    )

    assert isinstance(state.model, Gpt2LMHeadModel)


def test_separate_reference_state_marks_reference_as_non_saveable():
    config = _tiny_gpt2_config()
    policy_model = _build_policy_model(config)
    reference_model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        DpoModel(policy=policy_model, reference=reference_model),
        key=jrandom.PRNGKey(0),
        is_trainable=DpoModel(policy=True, reference=False),
        mp=jmp.get_policy("f32"),
    )

    assert isinstance(state.model, DpoModel)
    assert isinstance(state.model.reference, Gpt2LMHeadModel)
    mask = saveable_training_mask(state, DpoModel(policy=True, reference=False))
    assert mask.model.reference is False


def test_adapter_base_reference_requires_non_none_adapter():
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=NoAdaptorConfig(),
    )

    with pytest.raises(ValueError, match="requires a non-none adapter"):
        _validate_dpo_config(config)


def test_lora_adapter_base_reference_requires_zero_initial_delta():
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=LoraAdaptorConfig(zero_init_b=False, a_init_mode="random"),
    )

    with pytest.raises(ValueError, match="requires zero adapter delta at init"):
        _validate_dpo_config(config)


@pytest.mark.parametrize(
    "zero_init_b,a_init_mode",
    [(True, "random"), (False, "zero")],
)
def test_lora_adapter_base_reference_accepts_one_zero_factor(zero_init_b, a_init_mode):
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=LoraAdaptorConfig(zero_init_b=zero_init_b, a_init_mode=a_init_mode),
    )

    _validate_dpo_config(config)


def test_lora_adapter_base_reference_rejects_both_factors_zero():
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=LoraAdaptorConfig(zero_init_b=True, a_init_mode="zero"),
    )

    with pytest.raises(ValueError, match="exactly one zero LoRA factor"):
        _validate_dpo_config(config)


def test_training_keys_preserve_legacy_full_dpo_model_key_path():
    seed = 123
    legacy_data_key, _legacy_loader_key, legacy_model_key, legacy_training_key = jrandom.split(
        jrandom.PRNGKey(seed), 4
    )
    legacy_policy_key, _legacy_reference_key = jrandom.split(legacy_model_key)

    data_key, model_key, policy_key, _adapter_key, training_key = _derive_training_keys(seed)

    assert np.array_equal(np.asarray(data_key), np.asarray(legacy_data_key))
    assert np.array_equal(np.asarray(model_key), np.asarray(legacy_model_key))
    assert np.array_equal(np.asarray(policy_key), np.asarray(legacy_policy_key))
    assert np.array_equal(np.asarray(training_key), np.asarray(legacy_training_key))


def test_periodic_eval_callback_dedupes_same_step():
    calls: list[int] = []
    callback = _periodic_eval_callback(lambda info: calls.append(info.step))

    callback(SimpleNamespace(step=0), force=True)
    callback(SimpleNamespace(step=0))
    callback(SimpleNamespace(step=1))
    callback(SimpleNamespace(step=1), force=True)

    assert calls == [0, 1]


def test_scheduled_eval_callback_runs_only_on_scheduled_steps():
    calls: list[int] = []
    callback = _scheduled_eval_callback(lambda info: calls.append(info.step), {5, 10})

    callback(SimpleNamespace(step=0), force=True)
    callback(SimpleNamespace(step=4))
    callback(SimpleNamespace(step=5))
    callback(SimpleNamespace(step=5), force=True)
    callback(SimpleNamespace(step=10))

    assert calls == [0, 5, 10]


def test_canonical_dpo_config_parses_from_yaml(tmp_path: Path):
    config_path = tmp_path / "canonical_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: gpt2",
                "adapter:",
                "  type: none",
                "reference:",
                "  type: separate",
                "  model_path: meta-llama/Llama-3.1-8B-Instruct",
                "  is_hf: true",
                "reference_eval_cache:",
                "  mode: build_or_load",
                "  cache_dir: gs://example-bucket/reference-eval",
            ]
        )
    )

    config = draccus.parse(TrainDpoConfig, str(config_path), args=[])

    assert isinstance(config.adapter, NoAdaptorConfig)
    assert isinstance(config.reference, SeparateReferenceConfig)
    assert config.reference_eval_cache == ReferenceEvalCacheConfig(
        mode="build_or_load",
        cache_dir="gs://example-bucket/reference-eval",
    )


def test_canonical_lora_dpo_config_parses_from_yaml(tmp_path: Path):
    config_path = tmp_path / "canonical_lora_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: gpt2",
                "adapter:",
                "  type: lora",
                "  r: 8",
                "  alpha: 8.0",
                "reference:",
                "  type: adapter_base",
            ]
        )
    )

    config = draccus.parse(TrainDpoConfig, str(config_path), args=[])

    assert isinstance(config.adapter, LoraAdaptorConfig)
    assert isinstance(config.reference, AdapterBaseReferenceConfig)


def test_legacy_lora_dpo_config_translates_to_canonical(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "legacy_lora_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "initialize_from_hf: test-model",
                "lora:",
                "  r: 8",
                "  alpha: 8.0",
                "  zero_init_b: false",
                "  a_init_mode: zero",
                "hf_save_steps: 100",
            ]
        )
    )

    legacy_config = draccus.parse(LoraDpoConfig, str(config_path), args=[])
    monkeypatch.setattr(
        "levanter.main.lora_dpo.HFCheckpointConverter.from_hf",
        lambda *args, **kwargs: SimpleNamespace(default_config=Gpt2Config()),
    )

    translated = _translate_legacy_lora_dpo_config(legacy_config)

    assert isinstance(translated.adapter, LoraAdaptorConfig)
    assert isinstance(translated.reference, AdapterBaseReferenceConfig)
    assert translated.adapter.zero_init_b is False
    assert translated.adapter.a_init_mode == "zero"


def test_preference_chat_processor_skips_invalid_rows(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {"chosen": [], "rejected": []},
        {
            "chosen": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "No."},
            ],
        },
    ]

    result = processor(batch)
    assert len(result) == 1


def test_preference_lm_data_config_rejects_non_preference_format():
    with pytest.raises(ValueError, match="requires preference_chat"):
        PreferenceLmDataConfig(components={"train": DatasetComponent(format=TextLmDatasetFormat())})


def test_preference_lm_data_config_rejects_packed_preference_format():
    with pytest.raises(ValueError, match="Packed preference_chat datasets are not supported"):
        PreferenceLmDataConfig(
            components={
                "train": DatasetComponent(format=PreferenceChatLmDatasetFormat(pack=True)),
            }
        )


def test_preference_lm_data_config_rejects_non_raise_slice_strategy():
    with pytest.raises(ValueError, match="slice_strategy must be 'raise'"):
        PreferenceLmDataConfig(
            components={
                "train": DatasetComponent(format=PreferenceChatLmDatasetFormat(slice_strategy="left")),
            }
        )


def test_build_dpo_dataset_with_strict_preference_config(monkeypatch):
    config = PreferenceLmDataConfig(
        components={
            "pref": DatasetComponent(format=PreferenceChatLmDatasetFormat()),
        },
        shuffle=False,
    )

    monkeypatch.setattr(
        PreferenceLmDataConfig,
        "build_caches",
        lambda self, split: {"pref": object()},
    )

    monkeypatch.setattr(
        "levanter.main.train_dpo.dataset_for_preference_format",
        lambda format, Pos, cache: ListAsyncDataset([0]),
    )

    dataset = _build_dpo_dataset(config, hax.Axis("position", 8), key=jrandom.PRNGKey(0))
    assert isinstance(dataset, MixtureDataset)
    assert set(dataset.datasets.keys()) == {"pref"}


def test_dpo_loss_decreases_with_margin():
    Batch = hax.Axis("batch", 2)
    delta_ref = hax.zeros(Batch)

    delta_small = hax.named(jnp.array([0.1, 0.2], dtype=jnp.float32), Batch)
    delta_large = hax.named(jnp.array([1.0, 1.2], dtype=jnp.float32), Batch)

    loss_small, metrics_small = dpo_loss_from_logps(delta_small, delta_ref, beta=1.0)
    loss_large, metrics_large = dpo_loss_from_logps(delta_large, delta_ref, beta=1.0)

    assert float(loss_large) < float(loss_small)
    assert float(metrics_large["dpo_accuracy"]) >= float(metrics_small["dpo_accuracy"])


def test_dpo_metrics_are_explicit_metrics():
    Batch = hax.Axis("batch", 2)
    delta_ref = hax.zeros(Batch)
    delta_pi = hax.named(jnp.array([0.4, 0.8], dtype=jnp.float32), Batch)

    _, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=1.0)

    assert isinstance(metrics["dpo_loss"], Metric)
    assert isinstance(metrics["dpo_margin_policy"], Metric)
    assert isinstance(metrics["dpo_margin_ref"], Metric)
    assert isinstance(metrics["dpo_accuracy"], Metric)


def test_dpo_loss_matches_cached_reference_values():
    config = _tiny_gpt2_config()
    policy_model = _build_policy_model(config)

    Vocab = hax.Axis("vocab", 32)
    reference_model = config.build(Vocab, key=jrandom.PRNGKey(1))
    reference_model = inference_mode(reference_model, True)

    example = _make_dpo_example(key=jrandom.PRNGKey(2))

    logp_ref_chosen = logp_sum(reference_model, example.chosen, key=None)
    logp_ref_rejected = logp_sum(reference_model, example.rejected, key=None)
    cached_example = CachedDpoExample(
        chosen=example.chosen,
        rejected=example.rejected,
        logp_ref_chosen=float(np.asarray(logp_ref_chosen.array)),
        logp_ref_rejected=float(np.asarray(logp_ref_rejected.array)),
    )

    uncached_loss, uncached_metrics = dpo_loss(policy_model, reference_model, example, beta=0.1)
    cached_loss, cached_metrics = dpo_loss(policy_model, None, cached_example, beta=0.1)

    assert np.isclose(float(uncached_loss), float(cached_loss))
    assert np.isclose(float(uncached_metrics["dpo_loss"]), float(cached_metrics["dpo_loss"]))
    assert np.isclose(float(uncached_metrics["dpo_accuracy"]), float(cached_metrics["dpo_accuracy"]))


def test_reference_eval_cache_path_changes_with_slice_bounds():
    dataset = ListAsyncDataset([_make_dpo_example()])
    base_spec = ValidationDatasetSpec(
        name="val",
        dataset=dataset,
        source_cache_path="gs://bucket/tokenized/run/validation",
        source_split="validation",
    )
    split_spec = dataclasses.replace(base_spec, source_split="train", slice_start=10, slice_end=20)
    reference_identity = {"reference_type": "separate", "reference": {"model_path": "hf/model"}}

    base_path = reference_eval_cache_path(base_spec, reference_identity=reference_identity, seq_len=32, cache_dir=None)
    split_path = reference_eval_cache_path(
        split_spec, reference_identity=reference_identity, seq_len=32, cache_dir=None
    )

    assert base_path != split_path


def test_load_reference_eval_cache_rejects_metadata_mismatch(tmp_path: Path):
    cache_dir = tmp_path / "reference_eval_cache"
    spec = ValidationDatasetSpec(
        name="val",
        dataset=ListAsyncDataset([_make_dpo_example()]),
        source_cache_path=str(tmp_path / "validation"),
        source_split="validation",
    )
    metadata = reference_eval_cache_metadata(
        spec,
        reference_identity={"reference_type": "separate", "reference": {"model_path": "hf/model-a"}},
        seq_len=8,
    )

    with SerialCacheWriter(
        str(cache_dir),
        {"logp_ref_chosen": np.zeros((), dtype=np.float32), "logp_ref_rejected": np.zeros((), dtype=np.float32)},
        metadata=metadata,
    ) as writer:
        writer.write_batch(
            {
                "logp_ref_chosen": np.array([1.0], dtype=np.float32),
                "logp_ref_rejected": np.array([2.0], dtype=np.float32),
            }
        )

    mismatched_metadata = reference_eval_cache_metadata(
        spec,
        reference_identity={"reference_type": "separate", "reference": {"model_path": "hf/model-b"}},
        seq_len=8,
    )

    with pytest.raises(FileNotFoundError, match="metadata mismatch"):
        load_reference_eval_cache(str(cache_dir), metadata=mismatched_metadata)


def test_build_or_load_reference_eval_cache_matches_direct_logps(tmp_path: Path):
    config = _tiny_gpt2_config()
    reference_model = inference_mode(_build_policy_model(config), True)

    examples = [_make_dpo_example(key=jrandom.PRNGKey(i)) for i in range(3)]
    dataset = ListAsyncDataset(examples)
    spec = ValidationDatasetSpec(
        name="val",
        dataset=dataset,
        source_cache_path=str(tmp_path / "validation"),
        source_split="validation",
    )
    reference_identity = {"reference_type": "separate", "reference": {"model_path": "hf/model"}}
    cache_dir = reference_eval_cache_path(
        spec, reference_identity=reference_identity, seq_len=8, cache_dir=str(tmp_path / "cache")
    )
    metadata = reference_eval_cache_metadata(spec, reference_identity=reference_identity, seq_len=8)

    with local_cpu_mesh() as mesh, hax.axis_mapping({"batch": ResourceAxis.DATA}):
        loader = DataLoader(
            dataset,
            batch_size=1,
            mesh=mesh,
            axis_resources=None,
            max_buffered_batches=1,
            fetch_batch_size=1,
            allow_nondivisible_batch_size=True,
        )
        chosen, rejected = build_or_load_reference_eval_cache(
            reference_model=reference_model,
            dataset=dataset,
            eval_loader=loader,
            compute_axis_mapping=None,
            mp=jmp.get_policy("f32"),
            cache_dir=cache_dir,
            metadata=metadata,
        )

    expected_chosen = np.asarray(
        [float(np.asarray(logp_sum(reference_model, ex.chosen, key=None).array)) for ex in examples], dtype=np.float32
    )
    expected_rejected = np.asarray(
        [float(np.asarray(logp_sum(reference_model, ex.rejected, key=None).array)) for ex in examples],
        dtype=np.float32,
    )

    assert np.allclose(chosen, expected_chosen)
    assert np.allclose(rejected, expected_rejected)

    loaded_chosen, loaded_rejected = load_reference_eval_cache(cache_dir, metadata=metadata)
    assert np.allclose(loaded_chosen, expected_chosen)
    assert np.allclose(loaded_rejected, expected_rejected)


def test_logp_sum_passes_key_for_dropout():
    config = Gpt2Config(max_seq_len=8, hidden_dim=16, num_layers=1, num_heads=2, embed_pdrop=0.1)
    Vocab = hax.Axis("vocab", 32)
    model = config.build(Vocab, key=jrandom.PRNGKey(0))
    model = inference_mode(model, False)

    Pos = hax.Axis("position", 8)
    tokens = hax.named(jnp.arange(Pos.size, dtype=jnp.int32) % Vocab.size, Pos)
    example = LmExample.causal(tokens)

    out = logp_sum(model, example, key=jrandom.PRNGKey(1))
    assert isinstance(out, hax.NamedArray)


def test_preference_chat_processor_outputs_masks(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "No."},
            ],
        }
    ]

    result = processor(batch)
    assert len(result) == 1

    row = result[0]
    assert row["chosen_input_ids"].shape == row["chosen_assistant_masks"].shape
    assert row["rejected_input_ids"].shape == row["rejected_assistant_masks"].shape
    assert row["chosen_assistant_masks"].sum() > 0
    assert row["rejected_assistant_masks"].sum() > 0


def test_preference_pair_dataset_builds_example(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Nope."},
            ],
        }
    ]

    processed = processor(batch)
    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, processor.output_exemplar) as writer:
            writer.write_batch(processed)

        cache = writer.result()
        Pos = hax.Axis("position", 128)
        dataset = PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy="raise")
        example = dataset.as_sync_dataset()[0]

        assert isinstance(example, DpoExample)
        assert example.chosen.tokens.axes == (Pos,)
        assert example.rejected.tokens.axes == (Pos,)
        assert example.chosen.loss_weight.array.sum() > 0
        assert example.rejected.loss_weight.array.sum() > 0


def test_apply_updates_handles_none_updates_for_namedarrays():
    Axis = hax.Axis("axis", 4)
    model = hax.ones(Axis)

    updated = apply_updates(model, None, None)

    assert hax.all(updated == model).scalar()


def test_partition_for_grad_overwrite_preserves_namedarrays():
    Axis = hax.Axis("axis", 4)
    model = hax.ones(Axis)

    overwrites, _ = partition_for_grad_overwrite(model)
    updated = apply_updates(model, None, overwrites)

    assert is_jax_array_like(updated.array)
    assert hax.all(updated == model).scalar()


class _DummyModelAveraging(ModelAveraging[Gpt2LMHeadModel]):
    model: Gpt2LMHeadModel

    def update(self, model, step):
        return self

    @property
    def model_params(self) -> Gpt2LMHeadModel:
        return self.model


def test_eval_model_fills_missing_namedarrays_from_model():
    config = _tiny_gpt2_config()
    model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=True,
        mp=jmp.get_policy("f32"),
    )

    missing = eqx.filter(model, lambda _: False)
    state = dataclasses.replace(state, model_averaging=_DummyModelAveraging(model=missing))

    eval_model = state.eval_model
    weight = eval_model.embeddings.token_embeddings.weight

    assert is_jax_array_like(weight.array)


def test_restore_policy_model_from_partial_checkpoint_recovers_base_model():
    config = _tiny_gpt2_config()
    Vocab = Axis("vocab", 32)
    base_key, wrong_base_key, adapter_key, wrong_adapter_key, example_key = jrandom.split(jrandom.PRNGKey(0), 5)

    adapter = LoraAdaptorConfig(r=4, zero_init_b=False, a_init_mode="random")
    trained_policy = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    wrong_resume_skeleton = adapter.apply(config.build(Vocab, key=wrong_base_key), key=wrong_adapter_key)
    correct_source_skeleton = adapter.apply(config.build(Vocab, key=base_key), key=wrong_adapter_key)
    trainable_filter = adapter.trainable_filter(trained_policy)

    checkpointed_trainables = trainables_only(trained_policy, trainable_filter)
    wrong_resumed_policy = eqx.combine(checkpointed_trainables, wrong_resume_skeleton)
    restored_policy = _restore_policy_model_from_partial_checkpoint(
        wrong_resumed_policy,
        correct_source_skeleton,
        trainable_filter,
    )

    example = _make_dpo_example(seq_len=8, vocab_size=Vocab.size, key=example_key)
    trained_logp = logp_sum(inference_mode(trained_policy, True), example.chosen).scalar()
    wrong_logp = logp_sum(inference_mode(wrong_resumed_policy, True), example.chosen).scalar()
    restored_logp = logp_sum(inference_mode(restored_policy, True), example.chosen).scalar()

    assert wrong_logp != pytest.approx(trained_logp)
    assert restored_logp == pytest.approx(trained_logp)


class _CapturingConverter:
    reference_checkpoint = "base-model"

    def __init__(self):
        self.calls = []

    def save_pretrained(self, model, path, **kwargs):
        self.calls.append((model, path, kwargs))


class _CapturingTrainer:
    run_id = "test-run"
    config = SimpleNamespace(checkpointer=None)

    def __init__(self):
        self.hooks = []

    def add_hook(self, hook, *, every: int):
        self.hooks.append((hook, every))


def test_separate_reference_lora_merged_export_saves_policy_model():
    config = _tiny_gpt2_config()
    Vocab = Axis("vocab", 32)
    base_key, reference_key, adapter_key = jrandom.split(jrandom.PRNGKey(0), 3)

    adapter = LoraAdaptorConfig(r=4)
    policy = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    reference = config.build(Vocab, key=reference_key)
    trainer = _CapturingTrainer()
    converter = _CapturingConverter()

    adapter.install_export_hooks(
        trainer=trainer,
        converter=converter,
        tokenizer=None,
        export=AdaptorExportConfig(
            hf_save_steps=1,
            merged_hf_save_path="/tmp/export",
        ),
    )

    hook, _ = trainer.hooks[0]
    hook(SimpleNamespace(step=1, eval_model=DpoModel(policy=policy, reference=reference)))

    assert len(converter.calls) == 1
    saved_model, _, _ = converter.calls[0]
    assert isinstance(saved_model, Gpt2LMHeadModel)


def test_separate_reference_lora_peft_export_saves_policy_adapter(tmp_path):
    config = _tiny_gpt2_config()
    Vocab = Axis("vocab", 32)
    base_key, reference_key, adapter_key = jrandom.split(jrandom.PRNGKey(0), 3)

    adapter = LoraAdaptorConfig(r=4)
    policy = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    reference = config.build(Vocab, key=reference_key)
    trainer = _CapturingTrainer()

    adapter.install_export_hooks(
        trainer=trainer,
        converter=_CapturingConverter(),
        tokenizer=None,
        export=AdaptorExportConfig(
            hf_save_steps=1,
            peft_save_path=str(tmp_path),
        ),
    )

    hook, _ = trainer.hooks[0]
    hook(SimpleNamespace(step=1, eval_model=DpoModel(policy=policy, reference=reference)))

    with safe_open(tmp_path / "step-1" / "adapter_model.safetensors", framework="numpy") as tensors:
        keys = list(tensors.keys())

    assert keys
    assert all(".policy." not in key for key in keys)
    assert all(".reference." not in key for key in keys)


def test_separate_reference_hf_export_passes_generation_config():
    trainer = _CapturingTrainer()
    converter = _CapturingConverter()
    generation_config = {"eos_token_id": [2, 50]}

    _install_separate_reference_export_hooks(
        trainer=trainer,
        converter=converter,
        export=AdaptorExportConfig(
            hf_save_path="/tmp/export",
            hf_save_steps=1,
            generation_config=generation_config,
        ),
    )

    hook, _ = trainer.hooks[0]
    model = object()
    hook(SimpleNamespace(step=1, next_step=2, eval_model=model))

    assert len(converter.calls) == 1
    saved_model, _, saved_kwargs = converter.calls[0]
    assert saved_model is model
    assert saved_kwargs["generation_config"] == generation_config


@pytest.mark.parametrize("updates", [6, 7])
def test_dpo_exports_completed_update_intervals_and_final_checkpoint(tmp_path, updates):
    class ExportTrainer:
        config = SimpleNamespace(checkpointer=None)
        run_id = "export-cadence"
        add_hook = Trainer.add_hook

        def __init__(self):
            self.hooks = TrainerHooks()

    class FileConverter:
        def save_pretrained(self, model, path, **_kwargs):
            # A second export of the same update must fail, including the forced final hook.
            destination = Path(path)
            destination.mkdir()
            (destination / "policy.txt").write_text(str(model))

    trainer = ExportTrainer()
    _install_separate_reference_export_hooks(
        trainer=trainer,
        converter=FileConverter(),
        export=AdaptorExportConfig(hf_save_path=str(tmp_path), hf_save_steps=2),
    )
    for completed in range(1, updates + 1):
        state = SimpleNamespace(step=completed, model=completed, eval_model=completed, opt_state=None)
        info = StepInfo(state, 0.0, 0.0)
        trainer.hooks.run_hooks(info)
    trainer.hooks.run_hooks(info, force=True)

    expected = {"step-1": "2", "step-3": "4", "step-5": "6"}
    if updates == 7:
        expected["step-6"] = "7"
    assert {path.name: (path / "policy.txt").read_text() for path in tmp_path.iterdir()} == expected


def test_vmapped_init_with_sharding_handles_layer_axis():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(f"requires at least 4 JAX devices; found {len(devices)}")

    Embed = Axis("embed", 32)
    Qkv = Axis("qkv", 3)
    Heads = Axis("heads", 4)
    HeadSize = Axis("head_size", 8)
    Layers = Axis("layers", 5)

    def init_fn(key):
        return hax.random.truncated_normal(key, (Embed, Qkv, Heads, HeadSize), -3, 3)

    mesh = Mesh(np.array(devices[:4]).reshape((4, 1)), ("data", "model"))
    with set_mesh(mesh), hax.axis_mapping({"embed": "data", "heads": "model"}):
        keys = jax.random.split(jax.random.PRNGKey(0), Layers.size)
        out = hax.vmap(init_fn, Layers)(keys)
        assert out.axes[0].name == "layers"


@pytest.fixture
def dpo_llama_source(tmp_path, tiny_hf_tokenizer_path):
    config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        tokenizer=tiny_hf_tokenizer_path,
        reference_checkpoint=str(tmp_path / "hf-source"),
    )
    return write_llama_checkpoint(tmp_path / "hf-source", config, config.to_hf_config(32), key=jrandom.PRNGKey(13))


@pytest.fixture
def dpo_separate_reference_source(tmp_path, dpo_llama_source):
    source = tmp_path / "hf-reference"
    config = dataclasses.replace(dpo_llama_source.config, reference_checkpoint=str(source))
    return write_llama_checkpoint(source, config, config.to_hf_config(32), key=jrandom.PRNGKey(97))


def _dpo_trainer_config(tmp_path, *, explicit=False, steps=2):
    model_size = 2 if len(jax.devices()) >= 4 else 1
    data_size = len(jax.devices()) // model_size
    return TrainerConfig(
        id="dpo",
        mesh=MeshConfig(axes={"data": data_size, "model": model_size}),
        use_explicit_mesh_axes=explicit,
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints"), save_interval=None),
        load_checkpoint=False,
        train_batch_size=max(4, 2 * data_size),
        per_device_parallelism=2,
        num_train_steps=steps,
        max_eval_batches=0,
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
    )


def _native_dpo_configuration(tmp_path, source, tokenizer_path):
    model = dataclasses.replace(
        NativeLlamaLmConfig.from_hf_config(HfLlamaConfig.from_pretrained(source.path)),
        reference_checkpoint=str(source.path),
        tokenizer=tokenizer_path,
    )
    return TrainDpoConfig(
        model=model,
        trainer=_dpo_trainer_config(tmp_path, explicit=True),
        initialize_from_hf=str(source.path),
        reference=SeparateReferenceConfig(model_path=str(source.path), is_hf=True),
        optimizer=AdamConfig(learning_rate=0.01, warmup=0, weight_decay=0, lr_schedule="constant"),
        validation_split_fraction=None,
    )


def _separate_dpo_loss(model, example, *, key):
    return dpo_loss(model.policy, model.reference, example, beta=0.1)


@pytest.mark.tpu
def test_native_dpo_policy_updates_frozen_reference_and_checkpoint_next_step(
    tmp_path, dpo_llama_source, tiny_hf_tokenizer_path
):
    config = _native_dpo_configuration(tmp_path, dpo_llama_source, tiny_hf_tokenizer_path)
    tokenizer = load_marin_tokenizer(tiny_hf_tokenizer_path)
    context = prepare_model_init_context(
        config.model, tokenizer=tokenizer, initialize_from_hf=config.initialize_from_hf, use_hf_model_config=False
    )
    sharding = native_llama_sharding(config.trainer.parameter_axis_mapping, config.trainer.compute_axis_mapping)
    layout = ExplicitTrainingLayout(config.trainer.device_mesh, P(sharding.batch, None))
    optimizer = config.optimizer.build(2)
    with (
        current_tracker(NoopTracker()),
        Trainer(
            config.trainer, optimizer, _separate_dpo_loss, add_default_hooks=False, explicit_layout=layout
        ) as trainer,
    ):
        Vocab = Axis("vocab", 32)
        policy = _initialize_policy_model_from_source(
            config=config,
            model_context=context,
            Vocab=Vocab,
            policy_key=jrandom.PRNGKey(2),
            adapter_key=jrandom.PRNGKey(3),
            parameter_axis_mapping=trainer.parameter_axis_mapping,
            trainer=trainer,
        )
        reference = _load_separate_reference_model(
            config,
            reference_model_key=jrandom.PRNGKey(4),
            model_context=context,
            Vocab=Vocab,
            parameter_axis_mapping=trainer.parameter_axis_mapping,
            trainer=trainer,
        )
        for model in (policy, reference):
            assert model.Vocab.size == 32
            assert model.native.token_embeddings.sharding.spec == sharding.column_storage
            assert model.native.layers.stacked.mlp.down_proj.weight.sharding.spec == P(None, "data", "model")
            for name, value in model.to_state_dict().items():
                np.testing.assert_array_equal(value, dpo_llama_source.weights[name])
        axes = config.trainer.TrainBatch, config.model.max_Pos
        token_ids = np.arange(axes[0].size * axes[1].size, dtype=np.int32).reshape(axes[0].size, axes[1].size)
        chosen = hax.named(jnp.asarray(token_ids % 31), axes)
        rejected = hax.named(jnp.asarray((token_ids * 3 + 2) % 31), axes)
        batch = DpoExample(
            chosen=hax.vmap(LmExample.causal, axes[0])(chosen),
            rejected=hax.vmap(LmExample.causal, axes[0])(rejected),
        )
        # Copy before donation; the reference is excluded from trainable/checkpointed state.
        frozen_reference = {name: np.array(value, copy=True) for name, value in reference.to_state_dict().items()}
        initial_policy = {name: np.array(value, copy=True) for name, value in policy.to_state_dict().items()}
        model = DpoModel(policy=policy, reference=reference)
        state = trainer.initial_state(
            jrandom.PRNGKey(5), model=model, is_trainable=DpoModel(policy=True, reference=False)
        )
        first = trainer.train_step(state, batch)
        assert int(first.state.step) == 1
        assert any(
            not np.array_equal(value, initial_policy[name])
            for name, value in first.state.model.policy.to_state_dict().items()
        )
        for name, value in first.state.model.reference.to_state_dict().items():
            np.testing.assert_array_equal(value, frozen_reference[name])
        # Compare a nontrivial policy (after its first real DPO update) with the
        # independent named implementation at exactly those canonical weights.
        updated_weights = {
            name: np.array(value, copy=True) for name, value in first.state.model.policy.to_state_dict().items()
        }
        actual_loss, actual_grad = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda policy: dpo_loss(policy, first.state.model.reference, batch, beta=0.1)[0])
        )(first.state.model.policy)
        actual_grad_weights = {name: np.array(value, copy=True) for name, value in actual_grad.to_state_dict().items()}
        with use_test_mesh(), hax.axis_mapping({}):
            named_template = dpo_llama_source.config.build(Vocab, key=jrandom.PRNGKey(13))
            named_policy = from_torch_compatible_state_dict(
                named_template, {name: jnp.asarray(value) for name, value in updated_weights.items()}
            )
            named_reference = from_torch_compatible_state_dict(
                named_template, {name: jnp.asarray(value) for name, value in frozen_reference.items()}
            )
            named_batch = DpoExample(
                chosen=hax.vmap(LmExample.causal, axes[0])(hax.named(jnp.asarray(token_ids % 31), axes)),
                rejected=hax.vmap(LmExample.causal, axes[0])(hax.named(jnp.asarray((token_ids * 3 + 2) % 31), axes)),
            )
            expected_loss, expected_grad = eqx.filter_jit(
                eqx.filter_value_and_grad(lambda policy: dpo_loss(policy, named_reference, named_batch, beta=0.1)[0])
            )(named_policy)
            np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-5, rtol=1e-4)
            assert float(expected_loss) != float(first.loss)
            expected_grad_weights = to_torch_compatible_state_dict(expected_grad)
            assert any(np.any(value != 0) for value in expected_grad_weights.values())
            for name, value in actual_grad_weights.items():
                np.testing.assert_allclose(value, expected_grad_weights[name], atol=1e-5, rtol=1e-4)
        save_checkpoint(
            first.state.saveable_state, 1, tmp_path / "step-1", metadata=first.state.model.policy.checkpoint_metadata()
        )
        expected = trainer.train_step(first.state, batch)
        assert float(expected.loss) != float(first.loss)
        expected_arrays = [
            np.array(value, copy=True) for value in jax.tree.leaves(eqx.filter(expected.state, eqx.is_array))
        ]
    resumed_config = dataclasses.replace(
        config.trainer, load_checkpoint=True, load_checkpoint_path=str(tmp_path / "step-1")
    )
    with (
        current_tracker(NoopTracker()),
        Trainer(
            resumed_config, optimizer, _separate_dpo_loss, add_default_hooks=False, explicit_layout=layout
        ) as resumed,
    ):
        # Reconstruct the omitted frozen reference from its configured source.
        policy = _build_model(config.model, Vocab, jrandom.PRNGKey(7), resumed)
        reference = _load_separate_reference_model(
            config,
            reference_model_key=jrandom.PRNGKey(8),
            model_context=context,
            Vocab=Vocab,
            parameter_axis_mapping=resumed.parameter_axis_mapping,
            trainer=resumed,
        )
        state = resumed.initial_state(
            jrandom.PRNGKey(9),
            model=DpoModel(policy=policy, reference=reference),
            is_trainable=DpoModel(policy=True, reference=False),
        )
        assert int(state.step) == 1
        actual = resumed.train_step(state, batch)
        np.testing.assert_array_equal(actual.loss, expected.loss)
        for value, expected_value in zip(
            jax.tree.leaves(eqx.filter(actual.state, eqx.is_array)), expected_arrays, strict=True
        ):
            np.testing.assert_array_equal(value, expected_value)


@pytest.mark.tpu
@pytest.mark.timeout(120)
def test_native_dpo_main_preserves_source_vocab_and_resumes_same_checkpoint(
    tmp_path, dpo_llama_source, tiny_hf_tokenizer_path, dpo_preference_data
):
    config = _native_dpo_configuration(tmp_path / "full", dpo_llama_source, tiny_hf_tokenizer_path)
    config = dataclasses.replace(config, data=dpo_preference_data)
    train_dpo_main(config)
    split = dataclasses.replace(config, trainer=_dpo_trainer_config(tmp_path / "split", steps=1))
    train_dpo_main(split)
    checkpoint = latest_checkpoint_path(tmp_path / "split/checkpoints/dpo")
    continued = dataclasses.replace(
        config,
        initialize_from_hf=False,
        trainer=dataclasses.replace(_dpo_trainer_config(tmp_path / "continued"), initialize_from=checkpoint),
    )
    train_dpo_main(continued)
    with dataclasses.replace(config.trainer, use_explicit_mesh_axes=True).use_device_mesh():
        source_model = _build_model(
            config.model,
            Axis("vocab", 32),
            jrandom.PRNGKey(0),
            Trainer(
                dataclasses.replace(config.trainer, use_explicit_mesh_axes=True),
                config.optimizer.build(2),
                _separate_dpo_loss,
                add_default_hooks=False,
            ),
        )
        template = TrainerState.init(
            config.optimizer.build(2),
            DpoModel(policy=source_model, reference=source_model),
            key=jrandom.PRNGKey(0),
            is_trainable=DpoModel(policy=True, reference=False),
            mp=config.trainer.mp,
        )
        template = template.saveable_state
        full = load_checkpoint(
            template,
            latest_checkpoint_path(tmp_path / "full/checkpoints/dpo"),
            mesh=dataclasses.replace(config.trainer, use_explicit_mesh_axes=True).device_mesh,
        )
        resumed = load_checkpoint(
            template,
            latest_checkpoint_path(tmp_path / "continued/checkpoints/dpo"),
            mesh=dataclasses.replace(config.trainer, use_explicit_mesh_axes=True).device_mesh,
        )
    assert int(full.step) == int(resumed.step) == 2
    assert full.model.policy.Vocab.size == resumed.model.policy.Vocab.size == 32
    for value, expected in zip(
        jax.tree.leaves(eqx.filter(resumed, eqx.is_array)),
        jax.tree.leaves(eqx.filter(full, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(value, expected)
    metrics = json.loads((tmp_path / "continued/eval_results.json").read_text())
    assert np.isfinite(metrics["train/loss"])


@pytest.fixture
def dpo_preference_data(tmp_path, tiny_hf_tokenizer_path):
    batch_size = _dpo_trainer_config(tmp_path).train_batch_size
    tokenizer = load_marin_tokenizer(tiny_hf_tokenizer_path)
    format = PreferenceChatLmDatasetFormat(chat_template="{{ messages }}", mask_user_turns=False)
    processor = format.build_preprocessor(tokenizer)
    cache_path = tmp_path / "preference-cache"
    example_count = 3 * batch_size
    with SerialCacheWriter(
        str(cache_path), processor.output_exemplar, metadata=CacheMetadata(preprocessor_metadata=processor.metadata)
    ) as writer:
        writer.write_batch(
            {
                "chosen_input_ids": [((np.arange(8) + i) % 31).astype(np.int32) for i in range(example_count)],
                "rejected_input_ids": [
                    ((np.arange(8) * 3 + i + 2) % 31).astype(np.int32) for i in range(example_count)
                ],
                "chosen_assistant_masks": [np.ones(8, dtype=np.int32) for _ in range(example_count)],
                "rejected_assistant_masks": [np.ones(8, dtype=np.int32) for _ in range(example_count)],
            }
        )
    return PreferenceLmDataConfig(
        tokenizer=tiny_hf_tokenizer_path,
        shuffle=False,
        mixture_block_size=batch_size,
        components={"preference": DatasetComponent(cache_dir=str(cache_path), flat_cache=True, format=format)},
    )


@pytest.mark.tpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("reference_type", ["separate", "adapter_base"])
@pytest.mark.parametrize("hf_config_source", ["explicit", "inferred_string", "inferred_bool"])
def test_native_lora_dpo_main_updates_only_factors_and_resumes_exact_state(
    tmp_path,
    dpo_llama_source,
    dpo_separate_reference_source,
    tiny_hf_tokenizer_path,
    dpo_preference_data,
    reference_type,
    hf_config_source,
):
    inferred = hf_config_source != "explicit"
    if inferred:
        source_config = dataclasses.replace(
            dpo_llama_source.config,
            head_dim=8,
            max_seq_len=16,
            rope=DefaultRotaryEmbeddingsConfig(theta=50000, factor=4),
        )
        source_hf_config = HfLlamaConfig(
            **{
                **dpo_llama_source.config.to_hf_config(32).to_dict(),
                "head_dim": 8,
                "max_position_embeddings": 16,
                "rope_theta": 50000,
                "rope_scaling": {"rope_type": "linear", "factor": 4},
                "rope_parameters": {"rope_type": "linear", "rope_theta": 50000, "factor": 4},
            }
        )
        dpo_llama_source = write_llama_checkpoint(
            tmp_path / "hf-inferred-source", source_config, source_hf_config, key=jrandom.PRNGKey(13)
        )
        dpo_separate_reference_source = write_llama_checkpoint(
            tmp_path / "hf-inferred-reference", source_config, source_hf_config, key=jrandom.PRNGKey(97)
        )
    config = _native_dpo_configuration(tmp_path / "full", dpo_llama_source, tiny_hf_tokenizer_path)
    effective_model = config.model
    if inferred:
        effective_model = dataclasses.replace(
            effective_model, attn_backend=AttentionBackend.VANILLA, gradient_checkpointing="save_all"
        )
        placeholder = NativeLlamaLmConfig(
            hidden_dim=8,
            intermediate_dim=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=4,
            attn_backend=effective_model.attn_backend,
            gradient_checkpointing=effective_model.gradient_checkpointing,
            reference_checkpoint=(
                str(dpo_llama_source.path)
                if hf_config_source == "inferred_bool"
                else str(tmp_path / "unused-default-reference")
            ),
            tokenizer=tiny_hf_tokenizer_path,
        )
        config = dataclasses.replace(
            config,
            model=placeholder,
            initialize_from_hf=True if hf_config_source == "inferred_bool" else str(dpo_llama_source.path),
            use_hf_model_config=True,
        )
    config = dataclasses.replace(
        config,
        data=dpo_preference_data,
        train_seq_len=8,
        adapter=LoraAdaptorConfig(r=3, alpha=6, dropout=0),
        reference=(
            SeparateReferenceConfig(model_path=str(dpo_separate_reference_source.path))
            if reference_type == "separate"
            else AdapterBaseReferenceConfig()
        ),
        trainer=dataclasses.replace(config.trainer, num_train_steps=3, mp=jmp.get_policy("f32")),
        peft_save_path=str(tmp_path / "full/peft"),
        merged_hf_save_path=str(tmp_path / "full/merged"),
        hf_save_steps=1,
    )
    full = train_dpo_main(config)
    split_config = dataclasses.replace(
        config,
        trainer=dataclasses.replace(
            config.trainer,
            tracker=JsonFileTrackerConfig(output_path=str(tmp_path / "resumed")),
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "resumed/checkpoints"), save_interval=None),
            num_train_steps=2,
        ),
        peft_save_path=None,
        merged_hf_save_path=None,
    )
    split = train_dpo_main(split_config)
    assert int(split.state.step) == 2
    resumed = train_dpo_main(
        dataclasses.replace(
            split_config, trainer=dataclasses.replace(split_config.trainer, num_train_steps=3, load_checkpoint=True)
        )
    )
    assert int(full.state.step) == int(resumed.state.step) == 3
    np.testing.assert_array_equal(full.loss, resumed.loss)
    for actual, expected in zip(
        jax.tree.leaves(eqx.filter(resumed.state, eqx.is_array)),
        jax.tree.leaves(eqx.filter(full.state, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    mesh_config = dataclasses.replace(config.trainer, use_explicit_mesh_axes=True)
    with mesh_config.use_device_mesh():
        # Read the persisted partial state independently of main's returned ordinary state.
        for name in ("full", "resumed"):
            saved = load_checkpoint(
                full.state.saveable_state,
                latest_checkpoint_path(tmp_path / name / "checkpoints/dpo"),
                mesh=mesh_config.device_mesh,
            )
            for actual, expected in zip(
                jax.tree.leaves(eqx.filter(saved, eqx.is_array)),
                jax.tree.leaves(eqx.filter(full.state.saveable_state, eqx.is_array)),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)

        policy = full.state.model.policy if isinstance(full.state.model, DpoModel) else full.state.model
        assert policy.Vocab.size == 32 > len(dpo_preference_data.the_tokenizer)
        for info in (full, resumed):
            actual_policy = info.state.model.policy if isinstance(info.state.model, DpoModel) else info.state.model
            assert isinstance(actual_policy.config, NativeLlamaLmConfig)
            assert actual_policy.config.head_dim == effective_model.head_dim
            assert actual_policy.config.max_seq_len == dpo_llama_source.config.max_seq_len
            assert actual_policy.config.rope == dpo_llama_source.config.rope
            if inferred:
                assert actual_policy.native.config.head_dim == 8
                assert actual_policy.native.layers.stacked.self_attn.q_proj.wrapped.weight.shape == (1, 32, 16)
                assert actual_policy.config.max_seq_len == 16
                assert actual_policy.config.attn_backend == AttentionBackend.VANILLA
                assert actual_policy.config.gradient_checkpointing == "save_all"
                assert actual_policy.config.reference_checkpoint == str(dpo_llama_source.path)
                assert actual_policy.config.tokenizer == tiny_hf_tokenizer_path
            for name, value in unwrap_lora_modules(actual_policy).to_state_dict().items():
                np.testing.assert_array_equal(value, dpo_llama_source.weights[name])
            reference = (
                info.state.model.reference
                if isinstance(info.state.model, DpoModel)
                else config.adapter.base_model_view(actual_policy)
            )
            expected_reference = dpo_separate_reference_source if reference_type == "separate" else dpo_llama_source
            assert reference.Vocab.size == 32
            for name, value in reference.to_state_dict().items():
                np.testing.assert_array_equal(value, expected_reference.weights[name])

        base = load_hf_checkpoint(
            effective_model,
            str(dpo_llama_source.path),
            tokenizer=dpo_preference_data.the_tokenizer,
            axis_mapping=config.trainer.parameter_axis_mapping,
            compute_axis_mapping=config.trainer.compute_axis_mapping,
            compute_dtype=jnp.float32,
        )
        _, _, _, adapter_key, _ = _derive_training_keys(config.trainer.seed)
        initial_factors = lora_state_dict(config.adapter.apply(base, key=adapter_key))
        factors = lora_state_dict(policy)
        for factor in ("lora_A", "lora_B"):
            assert any(
                not np.array_equal(value, initial_factors[name])
                for name, value in factors.items()
                if f".{factor}." in name
            )
        exported_factors = safetensors.numpy.load_file(tmp_path / "full/peft/dpo/step-2/adapter_model.safetensors")
        assert exported_factors.keys() == factors.keys()
        for name, value in factors.items():
            np.testing.assert_array_equal(exported_factors[name], value)
        peft_config = json.loads((tmp_path / "full/peft/dpo/step-2/adapter_config.json").read_text())
        assert peft_config["base_model_name_or_path"] == str(dpo_llama_source.path)
        merged = merge_lora_modules(policy)
        reloaded = load_hf_checkpoint(
            effective_model,
            str(tmp_path / "full/merged/dpo/step-2"),
            tokenizer=dpo_preference_data.the_tokenizer,
            axis_mapping=config.trainer.parameter_axis_mapping,
            compute_axis_mapping=config.trainer.compute_axis_mapping,
            compute_dtype=jnp.float32,
        )
        for name, value in reloaded.to_state_dict().items():
            np.testing.assert_array_equal(value, merged.to_state_dict()[name])
        # Probe batches follow the data mesh, including CPU8's data4/model2 layout.
        data_size = mesh_config.device_mesh.shape["data"]
        axes = Axis("batch", 2 * data_size), Axis("position", 4)
        tokens = hax.named(jnp.arange(8 * data_size).reshape(axes[0].size, axes[1].size) % 31, axes)
        np.testing.assert_allclose(
            policy(tokens, AttentionMask.causal()).array,
            reloaded(tokens, AttentionMask.causal()).array,
            atol=1e-5,
            rtol=1e-5,
        )
