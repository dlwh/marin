# Shuttle ABI 13 FAST Contract GPU Plan

## Objective

Add one policy-bound FAST realization for the existing anonymous rank-two
BF16 Contract followed by an explicit FP32-to-BF16 result Map. The compiler
selects this path from verified algebra, types, indexing maps, tuning, and
numerical policy. It does not dispatch on function name, fixture identity,
shape identity, or digest.

This checkpoint is a local structural implementation. It does not claim PTX
assembly, CUDA execution, numerical acceptance, cache acceptance, or
performance.

## Closed boundary

The admitted algebra and public ABI remain the ABI 12 Contract/Map boundary:

1. one static rank-two BF16 x BF16 Contract with one contracting dimension,
   no batching, default precision, normalized indexing maps, and an internal
   FP32 result; and
2. one identity-indexed Map that converts that FP32 result to BF16 with
   round-to-nearest-even.

The Contract result has no other use or live-out. Non-default tuning, extra
algebra, FP32 public buffers, batched or multiple contracting dimensions, and
non-default precision remain rejected. SOURCE_ORDERED keeps its ABI 12 serial
ascending-FMA realization unchanged.

## FAST realization

- One warp owns each output element. A block contains up to eight output
  warps, so the launch uses at most 256 threads.
- Lane `l` accumulates `K` positions `l, l + 32, ...` in FP32 with
  `fma.rn.f32`.
- The warp combines lane partials with fixed `shfl.sync.down.b32` offsets
  `16, 8, 4, 2, 1` and `add.rn.f32`. Lane zero writes the FP32 temporary.
- The following Map preserves the explicit RNE BF16 storage boundary.
- The policy-bound schedule records `leaf_order_free`, subgroup size 32,
  reduction axis 2, zero scratch, and `ceil(K / 32)` serial tiles.

The implementation uses no shared memory, atomics, or tensor-core instruction.
It is deterministic for a fixed launch and code image but intentionally does
not preserve SOURCE_ORDERED association.

## Version and wire identity

Pipeline ABI advances from 12 to 13 because FAST Contract now has executable
meaning. Canonical options, observer identity, and persistent-cache identity
therefore rotate and reject ABI 12 and ABI 14.

Materialization and schedule schemas stay at version 2 for Contract plans.
Transport/device/invocation/bundle schemas remain `2/3/3/2`, and the CUDA-only
FFI target remains `shuttle.gpu.executable_bundle.v2`. Contract reduction order
is sealed through the schedule fingerprint and rederived PTX; it is not added
to the unchanged device transport grammar.

## Local gates

- Anonymous shapes cover `K = 16, 32, 33, 65`, partial and full warps, and
  single- and multi-block output grids.
- Generated PTX must contain one lane-strided FMA loop and the exact five-step
  shuffle/add tree. The result Map must retain its BF16 RNE conversion.
- Independent transport decoding must observe distinct SOURCE_ORDERED and FAST
  policy bytes, code, device roots, bundle roots, and bundle digests.
- Rehashed schedule/code mutations, adjacent ABI values, unsupported shapes,
  precision changes, and every non-default tuning field fail closed.
- Existing ABI 9 CPU, ABI 13 Map/Fold, SOURCE_ORDERED Contract, XLA parser,
  observer, fake lifecycle, and typed-FFI gates remain green.

## Evidence boundary and next step

The local gates prove compiler derivation, transport closure, policy/cache
separation, and fake lifecycle behavior. There is no local `ptxas`, so they do
not prove that the new shuffle PTX assembles. They also do not establish GPU
numeric error or speed.

The next acceptance step needs a separately reviewed and authorized CUDA SM90
run that assembles every emitted slice, executes anonymous Contract/Map cells
for both policies against independent references, and proves fresh-process
cache reuse. The direct v34 Contract benchmark remains useful backend evidence
but is architecture-nonconforming because it bypasses this ordinary-JAX MLIR
path.

## Exclusions

- No tensor-core, MMA, WGMMA, tiled shared-memory, or autotuned Contract.
- No multiple Contract, arbitrary nonlinear Map, backward, composed, or VJP
  graph.
- No batching, mixed public dtypes, non-rank-two Contract, or non-default
  tuning.
- No hardware reservation, external upload, timing, scorecard promotion,
  push, or integration in this checkpoint.
