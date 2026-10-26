# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax.state_dict import to_torch_compatible_state_dict
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.distributed import DistributedConfig
from levanter.main import sample_lm
from levanter.models.llama import LlamaConfig
from levanter.model_loading import load_hf_checkpoint
from levanter.testing.hf_checkpoints import LocalLlamaCheckpoint, write_llama_checkpoint
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.eval_harness import LevanterHarnessLM, _LmEvalHarnessWorker

from levanter.grug.attention import AttentionMask
import levanter.inference.engine as engine_module
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams, SequenceTable
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID
from levanter.layers.attention import AttentionMask as NamedAttentionMask
from levanter.layers.rotary import (
    DefaultRotaryEmbeddingsConfig,
    Llama3RotaryEmbeddingsConfig,
    YarnRotaryEmbeddingsConfig,
)
from levanter.models.llama_native import NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.testing.helpers import skip_if_module_missing
from levanter.tokenizers import load_tokenizer
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.jax_utils import sharded_tree_size


def _mesh():
    model_size = 2 if len(jax.devices()) >= 4 else 1
    return Mesh(
        np.asarray(jax.devices()).reshape(-1, model_size),
        ("data", "model"),
        axis_types=(AxisType.Explicit,) * 2,
    )


def _model(config):
    return config.build(
        hax.Axis("vocab", 32),
        key=jax.random.key(0),
        sharding=NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model"),
    )


def _full_logits(model, tokens):
    batch = jax.sharding.get_abstract_mesh().shape["data"]
    values = eqx.filter_jit(lambda m, t: m.native(t, AttentionMask.causal()))(
        model, jnp.asarray([tokens] * batch, dtype=jnp.int32)
    )
    return np.asarray(values)[0]


@eqx.filter_jit(donate="all-except-first")
def _decode(operands, cache):
    model, tokens, positions, info = operands
    return model.decode(tokens, cache, info, positions)


@pytest.mark.tpu
@pytest.mark.parametrize(
    "options",
    [
        {"num_kv_heads": 4},
        {"tie_word_embeddings": True, "rope": Llama3RotaryEmbeddingsConfig()},
        {
            "use_bias": True,
            "hybrid_norm": True,
            "input_embedding_norm": True,
            "use_qk_norm": True,
            "rope": YarnRotaryEmbeddingsConfig(factor=8.0),
        },
        {"rope": DefaultRotaryEmbeddingsConfig(factor=2.0)},
        {"activation_function": ActivationFunctionEnum.relu2},
        {"activation_function": ActivationFunctionEnum.gelu_new, "rope": YarnRotaryEmbeddingsConfig(factor=8.0)},
    ],
    ids=["mha", "gqa-tied-llama3", "gqa-norms-yarn", "linear-rope", "relu2", "gelu-new-yarn"],
)
def test_native_paged_prefill_continuation_clone_and_reset(options):
    config = dataclasses.replace(
        NativeLlamaLmConfig(
            hidden_dim=16, intermediate_dim=24, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=16
        ),
        **options,
    )
    with jax.set_mesh(_mesh()):
        model = _model(config)
        if config.use_bias:
            model = model.from_state_dict(
                {
                    name: value + 0.1 if name.endswith(".bias") else value
                    for name, value in model.to_state_dict().items()
                }
            )
        sequences = {2: [3, 4, 5, 6, 11, 12], 0: [7, 8, 9, 13, 14]}
        expected = {slot: _full_logits(model, tokens) for slot, tokens in sequences.items()}
        table = PageTable.init(max_pages=12, max_seqs=3, page_size=2, max_pages_per_seq=8)
        state = SequenceTable.init(table.max_seqs, table.pages_per_seq, table.page_size)
        for slot in [2, 0]:
            state, _ = state.reserve_slot(slot)
        cache = model.initial_cache(table.spec(), dtype=jnp.float32)
        assert cache[0].kv_pages.array.sharding.spec == P(None, None, "model", None)
        assert model.native.layers.stacked.self_attn.q_proj.weight.sharding.spec == P(None, "model", "data")

        # Slot order differs from slot number, and the continuation changes the
        # order. Padding is excluded from both page writes and result checks.
        for values, slots, positions in [
            ([3, 4, 7], [2, 2, 0], [0, 1, 0]),
            ([8, 9, 5, 6], [0, 0, 2, 2], [1, 2, 2, 3]),
            ([11, 13], [2, 0], [4, 3]),
            ([14, 12], [0, 2], [4, 5]),
        ]:
            count = len(values)
            tokens = hax.named(jnp.asarray(values + [INVALID], dtype=jnp.int32), "position")
            pos = hax.named(jnp.asarray(positions + [INVALID], dtype=jnp.int32), "position")
            slot_ids = hax.named(jnp.asarray(slots + [INVALID], dtype=jnp.int32), "position")
            state, table, info = state.allocate_for_seq(table, slot_ids, pos)
            logits, cache = _decode((model, tokens, pos, info), cache)
            np.testing.assert_allclose(
                np.asarray(logits.array)[:count],
                np.stack([expected[slot][position] for slot, position in zip(slots, positions, strict=True)]),
                atol=1e-4,
                rtol=1e-4,
            )
            assert logits.array.sharding.spec == P(None, "model")
            for layer_cache in cache:
                pages = np.asarray(layer_cache.kv_pages.array)
                used_pages = np.asarray(table.page_ref_counts.array) > 0
                np.testing.assert_array_equal(pages[~used_pages], 0)

        # Clone an actual partial page, then change only the child's next token.
        state, _ = state.reserve_slot(1)
        source_page = int(np.asarray(state.page_indices.array)[0, 2])
        state, table = state.clone_pages_from(table, 0, 1)
        dest_page = int(np.asarray(state.page_indices.array)[1, 2])
        assert source_page != dest_page
        cache = cache.copy_page(source_page, dest_page)
        parent_pages = np.asarray(state.page_indices.array)[0, :3]
        parent_before = [np.asarray(c.kv_pages.array)[parent_pages].copy() for c in cache]
        tokens = hax.named(jnp.asarray([10], dtype=jnp.int32), "position")
        pos = hax.named(jnp.asarray([5], dtype=jnp.int32), "position")
        state, table, info = state.allocate_for_seq(table, hax.named(jnp.asarray([1]), "position"), pos)
        logits, cache = _decode((model, tokens, pos, info), cache)
        np.testing.assert_allclose(logits.array, _full_logits(model, [7, 8, 9, 13, 14, 10])[5:6], atol=1e-4, rtol=1e-4)
        for before, layer_cache in zip(parent_before, cache, strict=True):
            np.testing.assert_array_equal(np.asarray(layer_cache.kv_pages.array)[parent_pages], before)
        reset = eqx.filter_jit(lambda c: c.reset(), donate="all")(cache)
        for layer_cache in reset:
            np.testing.assert_array_equal(layer_cache.kv_pages.array, 0)
            assert layer_cache.kv_pages.array.sharding.spec == P(None, None, "model", None)


def _greedy_tokens(model, prompt, count):
    sequence = list(prompt)
    for _ in range(count):
        sequence.append(int(np.argmax(_full_logits(model, sequence)[-1])))
    return sequence[len(prompt) :]


def _generation_requests(prompts, count):
    return [
        Request(
            prompt_tokens=prompt,
            request_id=index,
            n_generations=1,
            decode_params=SeqDecodingParams(
                max_num_tokens=jnp.asarray(len(prompt) + count, dtype=jnp.int32),
                temperature=jnp.asarray(0.0),
                top_p=jnp.asarray(1.0),
                stop_tokens=None,
                key=jax.random.PRNGKey(index),
            ),
        )
        for index, prompt in enumerate(prompts)
    ]


@pytest.mark.tpu
@pytest.mark.parametrize(
    "dtype,cache_mapping",
    [
        pytest.param(jnp.float32, "ordinary", id="ordinary-fp32"),
        pytest.param(jnp.bfloat16, "ordinary", id="ordinary-bf16-greedy-only"),
        pytest.param(jnp.float32, "explicit", id="explicit-fp32"),
        pytest.param(jnp.bfloat16, "explicit", id="explicit-bf16-greedy-only"),
    ],
)
def test_native_engine_prefill_incremental_generation_and_reuse(dtype, cache_mapping):
    with jax.set_mesh(_mesh()):
        model = _model(
            NativeLlamaLmConfig(
                hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=2, max_seq_len=16
            )
        )
        model = jax.tree.map(lambda value: value.astype(dtype) if eqx.is_inexact_array(value) else value, model)
        prompts = [[3, 4, 5], [7]]
        expected = [_greedy_tokens(model, prompt, 3) for prompt in prompts]
        engine = InferenceEngine.from_model_with_config(
            model,
            None,
            InferenceEngineConfig(
                max_seq_len=16,
                max_seqs=3,
                max_pages=24,
                page_size=2,
                max_queued_tokens=16,
                max_seqs_in_prefill=2,
                max_prefill_size=8,
                max_rounds=4,
                max_tokens_per_round=2,
                compute_dtype=dtype,
            ),
            axis_resources=(
                MeshConfig().resolved_compute_mapping if cache_mapping == "ordinary" else {"kv_head": "model"}
            ),
        )
        for cache in engine.gen_state.cache:
            assert cache.kv_pages.array.sharding.spec == P(None, None, "model", None)
            assert cache.kv_pages.dtype == dtype
            np.testing.assert_array_equal(cache.kv_pages.array, 0)
        requests = _generation_requests(prompts, 3)
        for _ in range(2):
            result = engine.generate(requests)
            actual = [result.tokens[index] for index in range(len(prompts))]
            assert actual == expected
        assert engine.gen_state.cache[0].kv_pages.array.sharding.spec == P(None, None, "model", None)


@pytest.mark.tpu
def test_named_gqa_engine_keeps_kv_replicated_when_tensor_parallel_exceeds_kv_heads():
    if len(jax.devices()) < 4:
        pytest.skip("Requires four devices for data2/model2")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"))
    mapping = MeshConfig().resolved_compute_mapping
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(mapping):
        model = LlamaConfig(
            hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=1, max_seq_len=16
        ).build(hax.Axis("vocab", 32), key=jax.random.key(0))
        prompt = [3, 4, 5]
        sequence = list(prompt)
        for _ in range(3):
            tokens = hax.named(jnp.asarray(sequence, dtype=jnp.int32), "position")
            logits = eqx.filter_jit(lambda m, t: m(t, NamedAttentionMask.causal()))(model, tokens)
            sequence.append(int(jnp.argmax(logits.array[-1])))
        engine = InferenceEngine.from_model_with_config(
            model,
            None,
            InferenceEngineConfig(
                max_seq_len=16,
                max_seqs=2,
                max_pages=8,
                page_size=2,
                max_queued_tokens=8,
                max_seqs_in_prefill=1,
                max_rounds=4,
                compute_dtype=jnp.float32,
            ),
            axis_resources=mapping,
        )
        assert all(cache.kv_pages.array.sharding.is_fully_replicated for cache in engine.gen_state.cache)
        result = engine.generate(_generation_requests([prompt], 3))
        assert result.tokens[0] == sequence[len(prompt) :]
        assert all(cache.kv_pages.array.sharding.is_fully_replicated for cache in engine.gen_state.cache)


@pytest.mark.tpu
def test_named_engine_cache_mapping_changes_are_part_of_the_compilation_key():
    if len(jax.devices()) < 4:
        pytest.skip("Requires four devices for data2/model2")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"))
    ordinary = MeshConfig().resolved_compute_mapping
    sharded = {**ordinary, "kv_head": "model"}
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_seqs=2,
        max_pages=8,
        page_size=2,
        max_queued_tokens=8,
        max_seqs_in_prefill=1,
        compute_dtype=jnp.float32,
    )
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(ordinary):
        model = LlamaConfig(
            hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=2, max_seq_len=8
        ).build(hax.Axis("vocab", 32), key=jax.random.key(0))
        for mapping in (ordinary, sharded, ordinary):
            engine = InferenceEngine.from_model_with_config(model, None, config, axis_resources=mapping)
            expected = NamedSharding(mesh, P(None, None, "model" if "kv_head" in mapping else None, None))
            for cache in engine.gen_state.cache:
                assert cache.kv_pages.array.sharding.is_equivalent_to(expected, ndim=4)
                np.testing.assert_array_equal(cache.kv_pages.array, 0)
        for mapping in (ordinary, sharded, ordinary, {}):
            with hax.axis_mapping(mapping, merge=False):
                engine = InferenceEngine.from_model_with_config(model, None, config, axis_resources=None)
            expected = NamedSharding(mesh, P(None, None, "model" if "kv_head" in mapping else None, None))
            for cache in engine.gen_state.cache:
                assert cache.kv_pages.array.sharding.is_equivalent_to(expected, ndim=4)
                np.testing.assert_array_equal(cache.kv_pages.array, 0)
    # No ambient mapping must not inherit the previous sharded invocation.
    assert hax.partitioning.current_thread_local_mapping() is None
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(model, None, config, axis_resources=None)
        assert all(cache.kv_pages.array.sharding.is_fully_replicated for cache in engine.gen_state.cache)


@pytest.mark.tpu
def test_native_ordinary_cache_auto_sizing_stays_within_the_device_budget(monkeypatch):
    if len(jax.devices()) < 4:
        pytest.skip("Requires four devices for data2/model2")
    budget = 256
    monkeypatch.setattr(engine_module, "estimated_free_device_memory", lambda _device: budget / 1024**3)
    mapping = MeshConfig().resolved_compute_mapping
    with jax.set_mesh(_mesh()), hax.axis_mapping(mapping):
        model = _model(
            NativeLlamaLmConfig(
                hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=2, max_seq_len=8
            )
        )
        config = InferenceEngineConfig(
            max_seq_len=8,
            max_seqs=2,
            page_size=2,
            hbm_utilization=1.0,
            max_queued_tokens=8,
            max_seqs_in_prefill=1,
            compute_dtype=jnp.float32,
        )
        engine = InferenceEngine.from_model_with_config(model, None, config, axis_resources=mapping)
        cache = engine.gen_state.cache
        estimate = sharded_tree_size(cache, mapping=mapping)
        actual = sum(layer.kv_pages.array.addressable_shards[0].data.nbytes for layer in cache)
        assert 0 < actual <= estimate <= budget
        assert all(layer.kv_pages.array.sharding.spec == P(None, None, "model", None) for layer in cache)


@pytest.mark.tpu
def test_native_bf16_paged_chunks_preserve_prefill_logits_and_cache():
    # This checks temporal cache semantics. Full-forward BF16 numerical parity
    # remains separate: the shared paged reference has different accumulation.
    with jax.set_mesh(_mesh()):
        model = _model(
            NativeLlamaLmConfig(
                hidden_dim=16, intermediate_dim=24, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=16
            )
        )
        model = jax.tree.map(lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value, model)
        outputs = []
        for chunks in [
            [([3, 4, 5, 6, 7, 8, 9], [2, 2, 2, 2, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2])],
            [([3, 4, 7], [2, 2, 0], [0, 1, 0]), ([5, 6, 8, 9], [2, 2, 0, 0], [2, 3, 1, 2])],
        ]:
            table = PageTable.init(8, 3, 2, 8)
            state = SequenceTable.init(3, 8, 2)
            for slot in [2, 0]:
                state, _ = state.reserve_slot(slot)
            cache = model.initial_cache(table.spec(), dtype=jnp.bfloat16)
            logits_by_position = {}
            for values, slots, positions in chunks:
                tokens = hax.named(jnp.asarray(values, dtype=jnp.int32), "position")
                pos = hax.named(jnp.asarray(positions, dtype=jnp.int32), "position")
                state, table, info = state.allocate_for_seq(
                    table, hax.named(jnp.asarray(slots, dtype=jnp.int32), "position"), pos
                )
                logits, cache = _decode((model, tokens, pos, info), cache)
                for row, slot, position in zip(np.asarray(logits.array), slots, positions, strict=True):
                    logits_by_position[slot, position] = row
            outputs.append((logits_by_position, cache, state))
        for coordinate, logits in outputs[0][0].items():
            np.testing.assert_array_equal(outputs[1][0][coordinate], logits)
        for reference, actual in zip(outputs[0][1], outputs[1][1], strict=True):
            assert actual.kv_pages.dtype == jnp.bfloat16
            # Allocation order changes physical page numbers; compare the
            # sequence-addressed cache, not incidental allocator page ids.
            for slot, count in [(2, 4), (0, 3)]:
                ref_pages = np.asarray(outputs[0][2].page_indices.array)[slot, : (count + 1) // 2]
                out_pages = np.asarray(outputs[1][2].page_indices.array)[slot, : (count + 1) // 2]
                ref = np.asarray(reference.kv_pages.array)[ref_pages].reshape(-1, *reference.kv_pages.array.shape[2:])
                out = np.asarray(actual.kv_pages.array)[out_pages].reshape(-1, *actual.kv_pages.array.shape[2:])
                np.testing.assert_array_equal(out[:count], ref[:count])


@pytest.fixture
def native_generation_harness(tmp_path, tiny_hf_tokenizer_path):
    hf_tokenizer = AutoTokenizer.from_pretrained(tiny_hf_tokenizer_path, model_max_length=4, truncation_side="left")
    hf_tokenizer.add_tokens(["token31"])
    hf_tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    destination = tmp_path / "generation-tokenizer"
    hf_tokenizer.save_pretrained(destination)
    tokenizer = load_tokenizer(str(destination))
    with jax.set_mesh(_mesh()):
        model = _model(
            NativeLlamaLmConfig(
                hidden_dim=16, intermediate_dim=24, num_layers=1, num_heads=4, num_kv_heads=2, max_seq_len=64
            )
        )
        model = jax.tree.map(lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value, model)
        worker = _LmEvalHarnessWorker(
            hax.Axis("batch", 2),
            hax.Axis("position", 64),
            model,
            {"kv_head": "model", "heads": "model", "mlp": "model", "vocab": "model"},
            tokenizer,
            None,
            2,
            generation_kwargs={"max_gen_toks": 3, "temperature": 0.0, "n": 1, "seed": 0},
        )
        yield LevanterHarnessLM(worker)


def test_harness_marin_tokenization_preserves_specials_and_explicit_hf_truncation(native_generation_harness):
    harness = native_generation_harness
    text = "token3 token4 token5 token6 token7"
    assert harness.tok_encode(text) == [3, 4, 5, 6, 7]
    assert harness.tok_encode(text, add_special_tokens=True) == [1, 3, 4, 5, 6, 7]
    assert harness.tok_encode([text, "token8"], add_special_tokens=True) == [[1, 3, 4, 5, 6, 7], [1, 8]]
    assert harness.tok_encode(text, left_truncate_len=2, add_special_tokens=True) == [6, 7]
    assert harness.tok_encode([text, "token8"], left_truncate_len=2) == [[6, 7], [8]]
    assert harness.tok_encode(text, truncation=True, add_special_tokens=True) == [1, 5, 6, 7]
    assert harness.tok_encode([text, "token8"], truncation=True, add_special_tokens=True, left_truncate_len=3) == [
        [5, 6, 7],
        [1, 8],
    ]


@pytest.mark.tpu
@skip_if_module_missing("lm_eval")
def test_native_generate_until_uses_real_prefill_incremental_cache_and_stop(native_generation_harness):
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    harness = native_generation_harness
    prompts = ["token3 token4 token5", "token7"]
    tokens = harness.tokenizer.encode_batch(prompts)
    expected_ids = [_greedy_tokens(harness.leader.model, prompt, 3) for prompt in tokens]
    stop = harness.tokenizer.decode([expected_ids[1][-1]])
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, {"until": ["token31"] if index == 0 else [stop], "max_gen_toks": 3}),
            idx=index,
            metadata=("native-local", index, None),
        )
        for index, prompt in enumerate(prompts)
    ]
    actual = harness.generate_until(requests)
    expected = [harness.tokenizer.decode(ids, skip_special_tokens=True) for ids in expected_ids]
    expected[1] = expected[1].split(stop)[0]
    assert actual == expected
    assert actual[0]
    assert actual[1].strip() == harness.tokenizer.decode(expected_ids[1][:-1], skip_special_tokens=True)


@dataclasses.dataclass(frozen=True)
class SamplingSource:
    checkpoint: LocalLlamaCheckpoint
    kind: str
    tokenizer_path: str


@pytest.fixture(params=["hf", "levanter"])
def sampling_source(tmp_path, tiny_hf_tokenizer_path, request) -> SamplingSource:
    named_config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        activation_function=ActivationFunctionEnum.gelu_new,
    )
    source = write_llama_checkpoint(
        tmp_path / "sampling-source", named_config, named_config.to_hf_config(32), key=jax.random.PRNGKey(13)
    )
    return SamplingSource(checkpoint=source, kind=request.param, tokenizer_path=tiny_hf_tokenizer_path)


def _sampling_config(sampling_source, tmp_path, native, dtype):
    source = sampling_source.checkpoint
    source_kind = sampling_source.kind
    tokenizer_path = sampling_source.tokenizer_path
    config_type = NativeLlamaLmConfig if native else LlamaConfig
    model_config = config_type.from_hf_config(source.config.to_hf_config(32))
    model_size = 2 if len(jax.devices()) >= 4 else 1
    trainer = TrainerConfig(
        mp=jmp.Policy(param_dtype=jnp.float32, compute_dtype=dtype, output_dtype=dtype),
        mesh=MeshConfig(axes={"data": len(jax.devices()) // model_size, "model": model_size}),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        tracker=NoopConfig(),
        log_dir=tmp_path / "logs",
    )
    path = str(source.path)
    if source_kind == "levanter":
        with dataclasses.replace(trainer, use_explicit_mesh_axes=native).use_device_mesh():
            model = load_hf_checkpoint(
                model_config,
                path,
                tokenizer=load_tokenizer(tokenizer_path),
                compute_dtype=jnp.float32,
                axis_mapping=trainer.parameter_axis_mapping,
                compute_axis_mapping=trainer.compute_axis_mapping,
                batch_axis_name=trainer.batch_axis_name,
            )
            path = str(tmp_path / "sampling-checkpoint")
            metadata = model.checkpoint_metadata() if native else None
            save_checkpoint({"model": model}, 1, path, metadata=metadata)
        if not native:
            # Named checkpoints retain their existing tokenizer-sized template contract.
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.add_tokens(["token31"])
            tokenizer_path = str(tmp_path / "named-tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            # Local checkpoint sampling must only need the outer tokenizer.
            model_config = dataclasses.replace(
                model_config, tokenizer=None, reference_checkpoint=str(tmp_path / "unavailable-hf-reference")
            )
    else:
        # Sampling has always inferred the source architecture from HF. A placeholder
        # config must not become an accidental architecture override for native loads.
        model_config = dataclasses.replace(model_config, hidden_dim=32, num_layers=2)
    return sample_lm.SampleLmConfig(
        model=model_config,
        trainer=trainer,
        tokenizer=tokenizer_path,
        hf_checkpoint=RepoRef.from_string(path) if source_kind == "hf" else None,
        checkpoint_path=path if source_kind == "levanter" else None,
        prompts=["token3 token4 token5", "token7"],
        stop_sequence=None,
        n_rounds=1,
        n_generations=1,
        max_new_tokens=3,
        temperature=0.0,
        engine=InferenceEngineConfig(
            max_seq_len=16,
            max_seqs=3,
            max_pages=24,
            page_size=2,
            max_queued_tokens=16,
            max_seqs_in_prefill=2,
            max_prefill_size=8,
            max_rounds=4,
            max_tokens_per_round=2,
            compute_dtype=dtype,
        ),
    )


@pytest.mark.tpu
@pytest.mark.parametrize("native", [False, True], ids=["named", "native"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_sampling_source_load_preserves_full_vocabulary_weights_and_compute_precision(
    sampling_source, tmp_path, native, dtype
):
    config = _sampling_config(sampling_source, tmp_path, native, dtype)
    tokenizer = load_tokenizer(config.tokenizer)
    trainer = dataclasses.replace(config.trainer, use_explicit_mesh_axes=native)
    source = sampling_source.checkpoint
    with hax.partitioning.set_mesh(Mesh(np.asarray(jax.devices()), ("data",))):
        oracle = source.config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(13))
        tokens = hax.named(jnp.asarray([3, 4, 5, 6], dtype=jnp.int32), "position")
        expected_logits = np.asarray(
            eqx.filter_jit(lambda m, t: m(t, NamedAttentionMask.causal()).array)(oracle, tokens)
        )
    with trainer.use_device_mesh():
        loaded = sample_lm._load_model(
            config, hax.Axis("vocab", len(tokenizer)), tokenizer=tokenizer, key=jax.random.PRNGKey(0)
        )
        actual = to_torch_compatible_state_dict(loaded)
        assert loaded.Vocab.size == 32
        if native:
            assert len(tokenizer) == 31
        for name, expected in source.weights.items():
            np.testing.assert_array_equal(actual[name], np.asarray(expected).astype(dtype))
        assert all(value.dtype == dtype for value in jax.tree.leaves(loaded) if eqx.is_inexact_array(value))
        if dtype == jnp.float32:
            tokens = hax.named(jnp.asarray([3, 4, 5, 6], dtype=jnp.int32), "position")
            logits = eqx.filter_jit(lambda m, t: m(t, NamedAttentionMask.causal()).array)(loaded, tokens)
            np.testing.assert_allclose(logits, expected_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.tpu
@pytest.mark.parametrize("native", [False, True], ids=["named", "native"])
def test_sampling_main_checkpoint_generation_matches_independent_named_oracle(sampling_source, tmp_path, native):
    config = _sampling_config(sampling_source, tmp_path, native, jnp.float32)
    source = sampling_source.checkpoint
    with hax.partitioning.set_mesh(Mesh(np.asarray(jax.devices()), ("data",))):
        oracle = source.config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(13))
        for name, weights in to_torch_compatible_state_dict(oracle).items():
            np.testing.assert_array_equal(weights, source.weights[name])
        expected_tokens, expected_logprobs = [], []
        for prompt in [[3, 4, 5], [7]]:
            sequence = list(prompt)
            logprobs = []
            for _ in range(3):
                tokens = hax.named(jnp.asarray(sequence, dtype=jnp.int32), "position")
                logits = eqx.filter_jit(lambda m, t: m(t, NamedAttentionMask.causal()).array)(oracle, tokens)[-1]
                token = int(jnp.argmax(logits))
                logprobs.append(float(jax.nn.log_softmax(logits)[token]))
                sequence.append(token)
            expected_tokens.append(sequence[len(prompt) :])
            expected_logprobs.append(logprobs)
    result = sample_lm.main(config)
    assert result is not None
    assert result.tokens == expected_tokens
    assert result.total_generated == 6
    np.testing.assert_allclose(result.logprobs, expected_logprobs, atol=1e-4, rtol=1e-4)
