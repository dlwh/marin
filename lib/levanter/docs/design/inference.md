# Draft Design Document for Inference

> The server sections preserve an earlier design and implementation snapshot.
> Verify HTTP endpoint and streaming support against the current server.
> The native model and sampling sections describe the current model/caller APIs.

## References

* [JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/blob/b282713880943cebe7183815918fb7dd60922b14/llama3/llama3_jax/model.py) <-- pretty basic kv cache, but easy to follow
* [Fancier JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/pull/22/files)

### MaxText

* [Page Manager](https://github.com/AI-Hypercomputer/maxtext/blob/eac885edb371e6141a2bb784f9060f816ce17b23/MaxText/inference/page_manager.py)
* [Paged Attention](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/inference/paged_attention.py#L298)
* [Jetstream](https://github.com/AI-Hypercomputer/JetStream) <-- session management and such
* [MaxEngine](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/maxengine.py) <-- maxtext impl of jetstream protocol

### EasyDel

* [VSurge](https://github.com/erfanzar/EasyDeL/tree/main/easydel/inference/vsurge)

### JAX Repo

- [Ragged Paged Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py)

### Nano-VLLM

torch, not much interesting here given jetstream. focused on batch inference I think?

* https://github.com/GeeeekExplorer/nano-vllm/

### SGLang

* [SGLang](https://github.com/sgl-project/sglang)

## Architecture Overview

**Goal**: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

**Core Architecture**:
- Reuse `run_generation_loop`, `_one_round`, and `GenState` patterns from `src/levanter/main/sample_lm.py` but extract into a reusable service.
- Maintain decode state in a long-lived service object that owns: `JitScheduler`, `PageTable`, `KvPageCache`, `DecodeState`, `Sampler`, `tokenizer`, and the `LlamaLMHeadModel` (or other `LmHeadModel`).
- Requests enqueue prompt tokens into the service; a background loop performs prefill+decode rounds and streams tokens back to callers.
- Keep JIT-safety: no Python control flow in jitted sections; use existing named-jit and Equinox modules.

## Technical Design

### Service Architecture
The inference server is built around a `GenerationService` that encapsulates:
- Model loading and initialization via `HFCheckpointConverter`
- Tokenizer management
- JAX device mesh configuration via `TrainerConfig`
- KV cache management via `PageTable` and `KvPageCache`
- Generation loop orchestration via `JitScheduler` and `DecodeState`
- Sampling via `Sampler`

### API Design
- **OpenAI Compatibility**: Follows OpenAI API v1 specification for `/v1/completions` and `/v1/chat/completions`
- **Request Schema**: Supports `prompt`, `max_tokens`, `temperature`, `stop`, `seed` parameters
- **Response Schema**: Returns structured responses with `choices`, `usage`, and metadata
- **Error Handling**: Proper HTTP status codes and error messages

### Performance Considerations
- **JIT Compilation**: Warmup generation on startup to trigger JIT compilation
- **Memory Management**: Efficient KV cache page allocation and deallocation
- **Batching**: Support for concurrent requests with `JitScheduler`
- **Streaming**: Server-Sent Events (SSE) for real-time token streaming

## File/Code References
- `src/levanter/main/sample_lm.py`: `SampleLmConfig`, `_load_model`, `GenState`, `run_generation_loop`, `_one_round`, `extract_outputs`.
- `src/levanter/inference/jit_scheduler.py`: `JitScheduler`, `DecodeState`, `SeqDecodingParams`.
- `levanter.inference.page_table.PageTable`, `levanter.layers.kv_cache.KvPageCache`.
- `levanter.layers.sampler.Sampler`.
- `levanter.models.llama.LlamaLMHeadModel` and `levanter.models.lm_model.LmHeadModel`.
- `levanter.compat.hf_checkpoints.HFCheckpointConverter`, `load_tokenizer`.
- `levanter.trainer.TrainerConfig` and `levanter.utils.jax_utils.use_cpu_device`.

## Implementation Notes

### Current Status
- Basic FastAPI server with `/v1/completions` endpoint implemented
- `GenerationService` with single-sequence generation working
- Warmup JIT compilation on startup implemented
- Health check endpoint (`/healthz`) functional
- Accurate token counting using actual tokenizer instead of rough estimates
- End-to-end testing completed successfully with tiny HF model on CPU

### Key Design Decisions
- **Optional Dependencies**: FastAPI and Uvicorn are optional `serve` dependencies
- **Configuration**: Uses `draccus` for configuration management, consistent with other Levanter components
- **Error Handling**: Comprehensive error reporting with proper HTTP status codes
- **Logging**: Structured logging with configurable verbosity levels

## Native Llama paged model boundary

`NativeLlamaLmHeadModel.initial_cache(spec, dtype=...)` allocates the existing
`ListCache[KvPageCache]`, one cache per layer. `decode(tokens, cache, batch_info,
pos_ids)` serves both packed prefill and incremental decoding through the existing
`InferenceEngine`. The model math remains array-native; the named boundary reuses
`PageBatchInfo`, KV updates and ragged paged attention. Ordinary forward and
cached decoding share the configured stateless MLP activation, including
non-default activations.

Tokens and absolute per-sequence position IDs have one packed `position` axis.
Sequence tokens must be contiguous, and padding must follow the valid prefix in
`batch_info`. The page table supplies cached-prefix lengths and causal sequence
boundaries. This API does not accept arbitrary attention masks or implement
sequence-parallel or multihost serving.

The packed stream and page table are replicated across data axes. Stored weights
retain their FSDP layout, while computation gathers that storage dimension and
partitions attention/cache heads over the model axis. The KV head count must be divisible by the
model-axis size so each shard retains whole interleaved K/V pairs. Cache arrays
use `[page, slot, 2 * kv_heads, head_size]`; page and slot axes are replicated.
Output projection retains vocabulary shards. The engine selects the rows to
sample before the shared sampler gathers their full vocabulary for sampling,
selected-token log-probabilities and optional nucleus sorting. This also applies
when the nucleus cutoff is omitted; other axis mappings remain in place. This
layout does not imply a throughput improvement over the existing decoder.

Cache dtype is the explicit `initial_cache` argument; engine `compute_dtype`
controls cache allocation and does not cast model parameters. KV updates cast to
that cache dtype. BF16 paged reference accumulation can differ from ordinary
full-sequence attention. Cache/chunk consistency and a matching greedy example
are narrower checks than full-forward numerical parity. TPU ragged attention
may fall back to its reference implementation, so a dispatcher result alone does
not validate the accelerator kernel or mixed query/cache dtype support.

The native methods consume and return the existing donated cache state. Callers
must retain the returned cache and use existing page-copy/reset behavior for
clones and subsequent requests. Canonical model selection remains a separate
migration step. All source vocabulary rows are retained; deciding which rows
are valid tokenizer outputs remains an entrypoint policy.

### Sampling entrypoint

`levanter.main.sample_lm` supports native sampling through the opt-in
`model.type: llama_native`; its default remains `llama`. The caller uses the
shared model initialization and source loaders, enables explicit mesh axes for
native models, and passes the existing engine its configured computation mapping.
HF sources determine the effective architecture before native layout validation.
For local sampling, `checkpoint_path` names a run directory containing `step-*`
checkpoints or a concrete checkpoint directory containing the `model/` subtree.
Set the top-level `tokenizer` to its local tokenizer path and keep `model` set to
the original architecture. This path does not resolve an HF reference.
Native Levanter model-subtree checkpoints retain their saved architecture identity
and full vocabulary through checkpoint metadata validation. For metadata-free native checkpoints, sampling derives the vocabulary size from
the tokenizer and rounds it for the configured computation mapping. That size
must equal the checkpoint embedding row count; this entrypoint has no separate
vocabulary-size override. Use a compatible tokenizer or a metadata-bearing
checkpoint when the stored model has extra rows.

Both source paths cast loaded parameters to `trainer.mp.compute_dtype` for
sampling. `engine.compute_dtype` separately controls the KV cache; set both when
choosing computation precision. HF sources and metadata-bearing native checkpoints retain stored vocabulary
rows beyond the tokenizer in logits and sampling. Such rows may decode to empty text, so token IDs
and log probabilities are available from the final `GenerationResult` returned by
`sample_lm.main`; zero sampling rounds return `None`. Printed per-round output is
unchanged. This caller support does not establish accelerator kernel correctness,
performance superiority, or larger-model training acceptance.
