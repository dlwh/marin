# Shuttle ABI 12 Contract/Map GPU Plan

## Objective

Extend the ABI 11 descriptor-driven GPU path with one structural rank-two
`Contract` followed by an explicit result `Map`. The compiler admits static
BF16 matrix inputs, an internal FP32 accumulator matrix, and a final BF16
result. Selection is based only on verified algebra, types, indexing maps, and
numerical policy; production code must not recognize a fixture, function name,
shape, or digest.

This is a local, no-hardware implementation slice. It does not assemble PTX,
load a CUDA plugin, execute a GPU kernel, qualify numerical output, or promote
an evaluation cell.

## Closed algebra boundary

The admitted region contains exactly:

1. one static rank-two BF16 x BF16 `shuttle.contract` with no batching, one
   contracting dimension, normalized maps
   `lhs(d0,d2)`, `rhs(d2,d1)`, and `result(d0,d1)`, default precision, and an
   FP32 accumulator; and
2. one identity-indexed scalar `shuttle.map` that converts the FP32 Contract
   result to BF16 with round-to-nearest-even.

The Contract must be live only through that Map. FP32 external inputs or
results, batched or multiple contracting dimensions, non-default precision,
non-default tuning, FAST policy, and additional algebra are rejected. Existing
ABI 12 Map/Fold SOURCE_ORDERED and FAST regions retain their prior behavior.

## Physical realization

- Materialization and schedule schemas use version 2 only when a Contract is
  present. Existing Map/Fold plans remain schema 1.
- The Contract schedule domain is `[M, N, K]`. Output elements are assigned in
  row-major order, with each thread computing one or more complete outputs.
- Accumulation starts at positive FP32 zero and visits `K` in exact ascending
  order using `fma.rn.f32`. The schedule records this as
  `serial_ascending_fma_rn_f32` on axis 2.
- The Contract writes one `tensor<MxNxf32>` temporary. The following Map
  performs the explicit BF16 conversion and writes the public result.
- Launch geometry, slot sizes, dependencies, and launch-domain positions are
  derived from checked static extents. The implementation retains the existing
  caps and checked arithmetic.

The new schedule reduction order is compiler-internal. Contract device entries
serialize no reduction-order value, so transport, device, invocation, and
bundle schemas remain `2/3/3/2`. Their byte grammar is unchanged. The generic
typed-FFI target remains `shuttle.gpu.executable_bundle.v2`.

## Versioning and cache identity

Pipeline ABI advances from 11 to 12 because the GPU compiler now gives
Contract-containing algebra a production meaning. Canonical compiler options,
observer identity, and persistent-cache keys therefore reject ABI 11 and bind
ABI 12. The FFI target and transport schemas do not change because the decoded
projection, slot, launch, and lifecycle contracts are unchanged.

## Red-first gates

1. Five anonymous `(M,K,N)` fixtures cross square, rectangular, and
   multi-block output geometry without production shape dispatch.
2. Conversion, materialization, and scheduling prove exactly one connected
   Contract and result Map, schema 2, an FP32 temporary of `M*N*4` bytes, and
   the exact two-launch dependency.
3. Generated PTX is inspected for the canonical positive-zero accumulator,
   ascending `K` loop, `K`-derived LHS/RHS addresses, `fma.rn.f32`, unit
   induction, and backedge. A recurrence permutation must fail the test.
4. Public serialization and an independent decoder agree on all roots, slots,
   bindings, code slices, launch geometry, and dependencies. Rehashed
   launch-geometry and future-produced-input mutations fail closed; direct code
   or root corruption breaks the integrity seal. The loader does not claim
   semantic validation of a fully recomputed self-consistent PTX rewrite.
5. The existing fake pinned-XLA lifecycle loads and launches both entries from
   decoded state; malformed projection, allocation, load, and launch behavior
   propagates failure.
6. Mutations reject FAST Contract, batching, multiple contracting dimensions,
   HIGH precision, FP32 external boundaries, live FP32 output, non-default
   tuning one field at a time, and wrong schedule order.
7. Historical schema-1 Map/Fold SOURCE_ORDERED and FAST gates remain green.

## Evidence boundary

Local native and Python gates establish structural conversion, fail-closed
verification, serialization, and fake lifecycle behavior only. They do not
show that the emitted PTX assembles or computes correctly on CUDA SM90. A later
reviewed and authorized run must rebuild the real plugin, assemble every PTX
slice, compare GPU results with an independent oracle, and prove fresh-process
cache reuse before any architecture or numerical acceptance claim.

## Scope exclusions

- No Contract FAST policy, batching, multiple contracting dimensions, mixed
  external dtypes, fused result algebra, backward/composed Contract graph, or
  non-rank-two Contract.
- No transport schema, device schema, invocation ABI, bundle schema, or FFI
  target change.
- No H100/GB200 reservation, PTXAS claim, GPU execution, timing, profiling,
  upload, scorecard promotion, push, or integration.
