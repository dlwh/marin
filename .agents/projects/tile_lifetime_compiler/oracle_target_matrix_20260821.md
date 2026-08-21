# Shuttle target and oracle matrix

- Date: 2026-08-21
- Research effort: medium
- Stop rule: internal artifacts and current primary repositories no longer
  changed the oracle ranking or the next compiler experiment.

## Summary

Every target needs two independent controls: a natural implementation for
semantics and gradients, and the strongest implementation with the same public
boundary for performance. An expert kernel must not be the only correctness
source. Shuttle may leave operations it does not yet own in XLA GPU MLIR; the
accepted path must still record which operations Shuttle owns and must not call
the complete performance oracle from generated code.

The strongest immediate sequence is normalization, dense attention, local MoE,
distributed MoE backward, GDN, dense-block backward, MSA tie closure, a frozen
Grug layer, and finally an optimizer step. The largest unsealed areas are
backward execution: dense block, distributed MoE, GDN under a safe FLA/Triton
pin, and B200 GQA attention. MSA is currently a forward/decode target.

Hardware results are not interchangeable. Keep H100, single B200/GB200, and
four-GB200 claims in separate rows.

## Evaluation rule

For each target:

1. Freeze the public inputs, outputs, dtypes, shapes, numerical policy, hardware,
   warmup, timing method, and software pins before tuning.
2. Compare values and gradients with natural JAX, NumPy, or an explicitly
   source-ordered recurrence. Include adversarial cases for ties, gates,
   segments, empty routes, and padding where they apply.
3. Compare latency with the strongest matched oracle listed below. A component
   benchmark is not a full-boundary denominator.
4. Report total boundary time. Separately report preparation, transport, and
   command-buffer costs when they explain the result.
5. Record the generic Shuttle operations and static FLOPs owned by the accepted
   path. Calling the complete oracle kernel is not Shuttle ownership.
6. Keep an XLA fallback for operations outside the current ownership boundary.

## Measured internal evidence

These are results already recorded in the
[TLTC logbook](../../logbooks/tile_lifetime_compiler.md) and checked-in
artifacts. They are evidence, not proposed acceptance criteria.

- **RMSNorm and LayerNorm backward.** On GB200, a coalesced 32-column Fold
  measured 0.039704 ms for RMSNorm against 0.051835 ms for the matched
  `torch.compile` algebra, and 0.041109 ms against 0.050499 ms for centered
  LayerNorm. Maximum `dx` error was `1.90735e-6`; maximum scale-gradient error
  was `7.62939e-5`. A corrected generic/XLA full replay measured
  0.107179/0.121351 ms, or 0.883215 times XLA. These are not yet TE 2.18 or
  QuACK-matched denominators. See the
  [GB200 normalization artifacts](../../../lib/tile_lifetime/benchmarks/artifacts/generated_row_normalization_backward_gb200_v0/).
- **Dense attention.** Direct official FA3/FA4 H100 measurements at batch 1,
  sequence 2048/4096, 32 query heads, 8 KV heads, and head dimension 128 were
  0.0672/0.2100 ms for FA3 and 0.0708/0.2275 ms for FA4; maximum difference was
  0.000977. The typed-FFI attention reverse measured 1.070353 times XLA on H100
  and 1.046338 times XLA on GB200 with correct deterministic outputs. The
  checked-in JAX FA4 THD wrapper failed in ragged TMA while the direct Torch
  CuTe path ran, so wrapper failure is not kernel evidence.
- **MSA.** The pinned oracle is MSA commit
  `80434d7f67877c6570ca19cac444b84bc9855dac` with CUTLASS
  `eb61c911471867a5fd2466bfd8f29306cea6ebf8`. At the symmetric 16K matched
  boundary, generated/oracle time was 3.823488/3.191376 ms, or 1.198069 times.
  Six route differences came from one exact tie, so exact source-order parity is
  still open. See the [MSA artifact](../../../lib/tile_lifetime/benchmarks/artifacts/msa_generic_tiled_fold_sm100_v1/README.md).
- **GDN.** With FLA commit `9c8e42e762fce087c27b673af4922795d9edb85e`
  on H100, recurrent/chunk times were 0.084960/0.515104 ms at `T=64`,
  0.321792/0.532176 ms at `T=256`, and 3.940768/0.510624 ms at `T=2048`.
  Chunk time at `T=8192` was 0.703536 ms. The generated chunk executor alone
  measured 0.340032 ms, but factor preparation raised the total to 0.984496 ms,
  or 1.928 times FLA, and materialized an 84,410,368-byte summary. This exposes
  a preparation/consumption fusion problem rather than an executor problem.
- **Dense block forward.** On H100 at sequence 2048, source-ordered generated,
  delayed generated, and oracle times were 1.705818/1.650502/1.523838 ms. At
  sequence 4096 they were 3.478322/3.390837/3.253411 ms. The accepted ratios
  were therefore 1.119422/1.083122 and 1.069131/1.042240. Backward and B200 are
  open. See the [counterbalanced artifact](../../../lib/tile_lifetime/benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1/pooled-summary.json).
- **Local MoE forward.** At 2048 tokens, hidden size 7168, intermediate size
  3072, 96 local experts, top 6, and 256-row padding, the owner-local
  pre-combine composition measured 3.454944 ms against 6.747360 ms for Torch.
  It used standalone MoK grouped GEMM plus generated packing, SwiGLU, and merge.
- **Distributed MoE forward.** On four GB200s with 2048 tokens per rank, hidden
  size 7168, intermediate size 3072, 384 global/96 local experts, and top 6,
  Shuttle measured 4.137120 ms against 3.645056 ms for MoK, or 1.134995 times.
  The result was replicated and a clean Map/Fold replay measured 1.137204 times.
  Backward is open. See the [clean replay artifact](../../../lib/tile_lifetime/benchmarks/artifacts/gb200_moe_clean_map_fold_v1/).
- **Bounded Grug reverse region.** The accepted ten-call H100 artifact measured
  0.689586 ms against 0.585042 ms for XLA, or 1.178695 times. It returned 53
  leaves, with maximum error `9.760261e-7`, mean error `7.977959e-11`, 38
  bitwise-equal leaves, and 84.1% ownership of static dot FLOPs. The twelve-call
  weighted reverse remained unaccepted at 1.241446 times. This small H100
  fixture is not a representative four-GB200 Grug MoE or training step. See the
  [accepted artifact](../../../lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_992a7467_v0/README.md)
  and [unaccepted extension](../../../lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_unaccepted_2732ef51_v0/README.md).

## Target matrix

### RMSNorm and LayerNorm

- **Semantic oracle:** natural JAX/NumPy forward and VJP, with source-ordered
  reductions for deterministic checks.
- **Performance oracle:** [Transformer Engine 2.18 normalization C API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/normalization.html)
  and [QuACK](https://github.com/Dao-AILab/quack); use the faster valid result.
  For normalization into a Contract, retain the executable legacy CODA pin and
  a materialized normalization-plus-GEMM control.
- **Boundary:** `x[N,H]`, `gamma[H]`, optional `beta[H]`, epsilon; output
  `y[N,H]`, saved mean/inverse standard deviation where required, and complete
  backward outputs.
- **Recommended first acceptance:** BF16 `N=2048`, `H=4096`, epsilon `1e-6`,
  with a consumer output width of 6144; H100 and B200 in separate rows.
- **Compiler hypothesis:** one generic full-prefix Fold plus final Map/column
  Fold covers RMS and centered LayerNorm. Attach scale and centering to the
  consumer Contract without adding a normalization-specific IR operation.

### Dense attention against FA4

- **Semantic oracle:** natural JAX attention and JAX VJP, including causal and
  segment masks.
- **Performance oracle:** official
  [FlashAttention-4 CuTe DSL](https://github.com/Dao-AILab/flash-attention).
  Preserve the dense direct API as the performance boundary; use the natural
  Grug segmented computation as the semantic boundary.
- **Boundary:** BSHD `q`, `k`, `v`; causal and segment metadata; output and log
  sum exp; complete `dq`, `dk`, `dv`.
- **Acceptance shapes:** BF16 batch 1, sequence 2048 and 4096, 32 query heads,
  8 KV heads, dimension 128; H100 and B200 separately.
- **Constraint:** FA4 forward support does not imply B200 grouped-query
  backward support. Track the upstream
  [B200 GQA backward limitation](https://github.com/Dao-AILab/flash-attention/issues/1989)
  and use the strongest honest denominator available when the row is frozen.
- **Compiler hypothesis:** normalized-exponential Fold plus QK and PV Contracts
  forms a generic training-attention family; segment metadata should change the
  relation, not introduce an attention-specific primitive.

### MSA

- **Semantic oracle:** natural max-score, stable top-k with lower-index tie
  breaking, and sparse payload execution.
- **Performance oracle:** pinned [MiniMax MSA](https://github.com/MiniMax-AI/MSA)
  plan, `sparse_topk_select`, and sparse attention implementation.
- **Boundary:** dense proxy-score production, sparse top-k selection, then
  sparse payload attention. Keep block size and selected-block count explicit.
- **Acceptance shapes:** upstream `S=16384`, `Hq/Hkv=64/4`, `D=128`, top 16;
  matched row `Q=256`, `K=16384`, `Hq/Hkv=32/8`, `D=128`, block 128, top 16.
  MSA requires SM100.
- **Compiler hypothesis:** encode selection and stable lower-index tie breaking
  generically. Validate an asymmetric natural route before accepting the
  symmetric 16K performance row.

### GDN

- **Semantic oracle:** an independent source-ordered recurrent implementation
  and its natural VJP.
- **Performance oracle:** [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
  fused recurrent and chunk gated-delta-rule APIs.
- **Boundary:** `q,k[B,T,Hq,K]`, `v[B,T,Hv,V]`, `g,beta[B,T,Hv]`, and
  `state[B,Hv,K,V]`; return output, final state, and all requested gradients.
- **Acceptance shapes:** BF16 with FP32 state/accumulation, batch 1, 16 query
  heads, 32 value heads, key/value dimensions 128, sequence 64/256/2048/8192,
  and chunk 64; H100 before B200. Include decode at `T=1`, batch 1/4/16.
- **Constraint:** do not reuse the unsafe Triton 3.4 backward environment.
  Re-pin a validated FLA/Triton combination and include adversarial gate
  regimes. See the upstream [chunk-64 backward issue](https://github.com/fla-org/flash-linear-attention/issues/984)
  and [gate-regime coverage issue](https://github.com/fla-org/flash-linear-attention/issues/891).
- **Compiler hypothesis:** fuse factor preparation into the ordered chunk
  consumer so the large summary never becomes a boundary tensor.

### Dense transformer block

- **Semantic oracle:** natural JAX forward and VJP for the full block.
- **Performance oracle:** a hand-composed official-kernel control using FA4 and
  the executable CODA/QuACK/TE components that match the frozen boundary.
- **Boundary:** normalization, QKV projection, dense causal attention, output
  projection, SwiGLU MLP, residuals, and their complete backward values.
- **Acceptance shapes:** BF16 batch 1, hidden 4096, intermediate 14336, 32 query
  heads, 8 KV heads, dimension 128, sequence 2048/4096; H100 and B200.
- **Compiler hypothesis:** join adjacent generic normalization, attention,
  Contract, and Map regions under one command-buffer-compatible boundary.
  First close backward at the already accepted H100 forward shapes.

### Local MoE

- **Semantic oracle:** natural JAX gather/expert/scatter and VJP with supplied
  routing, including empty experts, repeated routes, and padding.
- **Performance oracle:** [SonicMoE](https://github.com/Dao-AILab/sonic-moe)
  forward/backward on Hopper and Blackwell. Use
  [Mixture of Kittens](https://github.com/cursor/mixture-of-kittens) and QuACK
  as component controls on Blackwell.
- **Boundary:** `x[T,H]`, expert indices and router weights `[T,K]`, concatenated
  `w13`, `w2`, padding policy, output, and complete input/weight/router
  gradients. Routing is an input, not part of this target.
- **Recommended first acceptance:** owner-local `T=2048`, `H=7168`, `I=3072`,
  `E=96`, `K=6`, 256-row padding. Add a Grug-native `H=1024`, `E=256`, `K=4`
  row only after its token count is frozen.
- **Compiler hypothesis:** relation grouping, concatenated W13 Contract,
  generated SwiGLU/W2, and a fixed-slot Fold should cover local execution
  without an MoE-specific operation.

### Distributed MoE

- **Semantic oracle:** natural sharded JAX with an explicit dispatch and combine
  relation; check forward and VJP independently.
- **Performance oracle:** MoK functional forward/backward. Use
  [DeepEP](https://github.com/deepseek-ai/DeepEP) as a secondary transport
  control, not as the semantic combine oracle.
- **Boundary:** the local-MoE values plus source/destination rank, slot, capacity,
  dispatch, inverse relation, and transport completion.
- **Acceptance shape:** four GB200s, 2048 input tokens per rank, hidden 7168,
  intermediate 3072, 384 global/96 local experts, top 6, BF16, 256-row padding.
- **Compiler hypothesis:** derive backward tasks from the same relation and its
  inverse; keep Transport generic and permit XLA collectives until that boundary
  is owned.

### Grug MoE layer

- **Semantic and performance oracle:** the exact natural Grug JAX layer and VJP
  compiled by stock XLA, with production FA4/SonicMoE/DeepEP controls where
  their boundaries match. No external library implements the complete
  QB/GatedNorm/XSA semantics.
- **Boundary:** freeze parameters, QB/GatedNorm state, segment metadata, routing,
  collectives, residuals, output, updated state, and complete gradients.
- **Recommended representative shape:** one layer, batch 1, sequence 2048,
  hidden 1024, routed intermediate 512, shared intermediate 1024, 256 experts,
  top 4, 8 query heads, 2 KV heads, dimension 128, then scale to four GB200s.
  This is a recommendation, not a frozen specification row.
- **Compiler hypothesis:** compose generic relation, normalization, attention,
  gated product, Map, Fold, and Contract regions. Leave collectives in XLA until
  generic Transport is ready.

### Full Grug training step

- **Semantic and performance oracle:** the exact natural JAX
  `value_and_grad` plus optimizer step compiled by stock XLA.
- **Boundary:** freeze model and optimizer parameters, batch, loss and z-loss,
  QB state, updated parameters/state, requested metrics, and sharding. Accept on
  four GB200s.
- **Acceptance phases:** use SGD only as a debug row. Accept the production row
  with the actual grouped MuonH/Adam optimizer and rematerialization policy.
- **Compiler hypothesis:** first own one adjacent generic
  Contract/Map/Fold command-buffer region, then attach its GEMM reverse. XLA
  fallback is allowed elsewhere, but the owned region and FLOP share must be
  explicit.

## Ranked experiments

1. Seal RMSNorm and LayerNorm forward/backward against TE 2.18 and QuACK on
   H100, then B200, using the exact same boundary and timing scope.
2. Seal dense causal attention forward/backward against official FA4 on H100;
   identify and document the honest B200 grouped-query backward denominator.
3. Seal local MoE forward/backward against SonicMoE on H100 and B200.
4. Extend the accepted four-GB200 distributed forward row to complete backward
   against MoK, including relation inversion and transport timing.
5. Re-pin FLA/Triton and close GDN forward/backward across sequence and
   adversarial gate regimes; test fused preparation/consumption first.
6. Complete dense-block backward at the accepted H100 forward shapes, then add
   B200.
7. Resolve MSA stable-tie and asymmetric-route parity. Treat it as forward-only
   unless backward is explicitly added to the target.
8. Freeze the representative Grug layer shape and accept its forward/backward on
   four GB200s.
9. Attempt the full production optimizer step only after its component rows are
   sealed.

## Negative leads

- The limited TE 2.17 runner is not a matched normalization-to-Contract oracle.
- FLA with Triton 3.4 is not a safe chunk-64 backward oracle.
- FA4 B200 forward does not establish grouped-query backward support.
- The current MSA symmetric result is not exact while the stable-tie route
  differs.
- Generated Shuttle code must not invoke a complete FA4, MSA, FLA, SonicMoE,
  MoK, DeepEP, CODA, or TE oracle. Low-level CUTLASS/CuTe/Triton lineage is
  acceptable when the generated operation is generic and provenance is
  recorded.
- The accepted ten-call H100 Grug reverse fixture is not a full Grug-layer or
  training-step result.
- MoK, DeepEP, and SonicMoE must not supply the complete event graph or semantic
  combine for the generated path.
- Component latency must not replace full-boundary latency; the GDN factor
  preparation result is the concrete counterexample.

## Source ledger

| Source | Role and interface | Hardware/software | License and provenance |
| --- | --- | --- | --- |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | LayerNorm/RMSNorm C APIs; `x[N,H]`, per-feature affine values, saved statistics, F/B | Hopper, Ada, Blackwell; use 2.18 API | Apache-2.0 |
| [QuACK](https://github.com/Dao-AILab/quack) | Optimized RMSNorm and LayerNorm F/B | H100, B200/B300, RTX 50; CUDA 12.9+, Python 3.12 | Apache-2.0 |
| [CODA](https://github.com/open-lm-engine/coda-kernels) | Fused `linear_swiglu`, `linear_cross_entropy`, and `linear_qknorm_rope`; retain the known executable legacy pin for comparison | H100, CuTe DSL | No obvious repository license file in the reviewed revision; oracle-only and provenance-sensitive. [Paper](https://arxiv.org/abs/2605.19269) |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FA4 dense attention CuTe DSL, output/LSE and F/B where supported | H100 and B200 | BSD-3-Clause |
| [MiniMax MSA](https://github.com/MiniMax-AI/MSA) | Plan, FMHA, top-k selection, sparse attention | SM100; public example uses block 128/top 16 | MIT; bundled CUTLASS is BSD-3-Clause and FlashInfer/TensorRT portions are Apache-2.0. [Paper](https://arxiv.org/abs/2606.13392) |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | Fused recurrent and chunk gated-delta rule | Current package requires PyTorch 2.7+ and Triton 3.3+; pin exact validated versions | MIT |
| [SonicMoE](https://github.com/Dao-AILab/sonic-moe) | Supplied-route fused MoE F/B; public example `E=128`, `K=8`, `H=4096`, `I=1536`, `x[32768,4096]` | SM90, SM100, SM120; CUDA 12.9+, Python 3.12+, PyTorch 2.7+ | Apache-2.0; DeepEP layout integration remains a separate concern ([issue](https://github.com/Dao-AILab/sonic-moe/issues/68)) |
| [Mixture of Kittens](https://github.com/cursor/mixture-of-kittens) | `build_schedule`, `forward`, `backward` for supplied routes | SM100/SM103; Python 3.12+, PyTorch 2.10+, CUDA 13 | Apache-2.0 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Dispatch/combine transport control | NVIDIA GPU distributed execution | MIT |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | Grouped GEMM and low-level physical building blocks | NVIDIA GPU | BSD-3-Clause |

## Open decisions

- Freeze the representative Grug layer token count and confirm whether the
  proposed compact shape is representative enough for the four-GB200 row.
- Choose the production optimizer and rematerialization policy before defining
  the full-train acceptance boundary.
- Decide whether MSA backward is a target. The current public oracle and the
  checked-in evidence are forward/decode-oriented.
- Freeze exact safe FLA, Triton, FlashAttention, QuACK, SonicMoE, MoK, CUDA, and
  compiler pins in each benchmark manifest rather than inheriting moving heads.
