# FAST attention numerical debug log

## Goal

Make the generic FAST online vector attention lowering numerically match the
independent host oracle on H100. Keep the investigation scoped to source/affine
mapping, causal predicate semantics, score transforms, and the online
`{m, l, o[Dv]}` recurrence/finalization. This checkpoint makes no performance
claim.

## Initial state

- Isolated worktree: `/private/tmp/shuttle-attention-state-machine-red`
- Accepted base: `9170712f71455378faab7af9008b60ad536b64f2`
- H100 fixture: ordinary-JAX causal GQA with terminal output epilogue
- First H100 attempt stopped in public transport validation. The emitted entry
  carries `leaf_order_free` (wire value 1), while the runtime parser accepted
  only wire value 0. A one-line runtime change accepting values 0 and 1 is kept
  isolated from all numerical changes.
- With that transport change, the kernel launched and completed on one H100 but
  failed the independent host oracle:
  - maximum absolute error: `0.4152771383523941` (limit `0.003`)
  - mean absolute error: `0.054036289276926927` (limit `0.0005`)
- Exact log: `/private/tmp/shuttle-fast-attention-h100-interactive-test.log`
- Log SHA-256:
  `be567c930166abcc7b842521ac5e2c16526a8a2b218eec5c94d26c0a41fe80eb`
- The reservation was released after the run; no GPU is currently held.

## Hypotheses

1. **Affine addressing/source maps.** A Q, K, V, or output domain coordinate is
   flattened with the wrong map or stride. The largest errors and a causal row
   with a single valid key should distinguish this from softmax instability.
2. **Causal Relation coordinates.** Query/key iotas or compare direction are
   swapped or offset incorrectly, admitting the wrong keys.
3. **Score transform.** Scale or softcap is omitted, duplicated, or applied in
   the wrong order.
4. **Online recurrence/finalization.** Masked leaves are not the identity, or
   `m`, `l`, vector `o`, normalization, or the terminal epilogue is incorrect.

## Evidence and experiments

### Static lowering

- The focused axis fixture reaches materialization, schedule, executable bundle,
  and executable verification with complete output dependency coverage.
- Emitted artifact: `/private/tmp/fast-attention-emitted.mlir`
- Logical source domain: `[1, 2, 2, 17, 7, 17]`
- Source-derived output-vector dimension: `[4]` (`Dv = 7`)
- Reduction dimension: `[5]` (`K = 17`)
- Projected launch covers 68 logical rows. Physical launch choices are not part
  of the correctness contract.

### Root cause 1: causal key coordinate was projected away

- In the failing PTX, score K and V addressing used the active reduction
  induction register `%r24`, but the Relation key iota used `%r41 = 0`.
  The emitted comparison was therefore equivalent to `0 <= query` on every K
  iteration, admitting all 17 keys for every query row.
- The Select emitter evaluated its predicate from freshly reconstructed launch
  coordinates. Those coordinates intentionally project away both the Dv vector
  dimension and the K reduction dimension, so they cannot represent a
  score-domain Relation predicate inside the K loop.
- The fix evaluates the predicate at the same score-domain coordinates as the
  selected score. Regenerated PTX now binds the Relation key iota to `%r24` and
  the query iota to `%r21`, yielding `%r24 <= %r21`.
- Local build passed for `shuttle-test-opt` and `gpu_transport_v2_test` with
  Bazel 7.4.1 and the installed macOS 15.5 SDK. The complete focused lowering,
  source coverage, sufficient-state, schedule, and executable verification
  pipeline also passed.

### Failure diagnostic

Add deterministic failure output for selected `(Hkv, G, Q)` rows. For each Dv
component record actual, reference, and error; independently record expected
`m`, `l`, and pre-normalization `o`. Query row 0 is especially useful because
only key 0 is causally valid, so the normalized output before the epilogue must
equal that key's V vector.

## Results

The first concrete semantic fault is fixed and the focused H100 numerical
canary passes:

- Test: `GpuTransportV2Test.H100RunnerMatchesIndependentFastAttentionOracle`
- Result: 1/1 passed; test body completed in 2809 ms
- Numerical contract: maximum absolute error at most `0.003`, mean absolute
  error at most `0.0005`, and every output finite
- Exact copied test log:
  `/private/tmp/shuttle-fast-attention-h100-fix1-test.log`
- Log SHA-256:
  `dbe6e76d44e2084705b0fc68dcf4eaf0624ef8e172fd0027a21d0b0e047c518b`
- Hardware: one visible NVIDIA H100 80GB HBM3 (`CUDA_VISIBLE_DEVICES=0`)
- Build driver: Bazel 7.7.0 with verified SHA-256
  `fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf`
- XLA source: exact
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`, with the checked Track 2
  workspace patch that registers the local Shuttle repository
- CUDA configuration: hermetic 12.9.1, compute capability 9.0

The failure-only row diagnostic did not fire on the passing run. Its purpose is
to preserve actionable `m/l/o[Dv]` evidence if this canary regresses.

The whole-node Iris reservation was released immediately after copying the
log. Local reservation state is absent and the backing pod reached terminal
`Completed` (`0/3` ready), so it no longer holds GPU capacity.

The final production sources also pass a clean Bazel 7.7.0 build of
`shuttle-opt`, `shuttle-test-opt`, and `gpu_transport_v2_test`; four focused
ordinary-JAX FAST attention lowering smokes; the runfiles-backed native
transport test; and 59/59 proportional lit tests. The three prospective
attention `*-red.mlir` contracts remain intentionally excluded because the
reduced checkpoint does not claim their test-only certificate protocols.

## Future work

- Keep the deterministic row diagnostic for future failures.
- Add structurally distinct hardware fixtures only as a later behavior
  checkpoint; this debug session establishes the focused causal-GQA epilogue
  canary, not broad coverage or performance.
