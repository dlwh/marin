# Shuttle ABI 11 Generic Map/Fold GPU Plan

## Objective

Extend the reviewed ABI 10 H100 path from one closed RMSNorm forward descriptor
to a descriptor-driven GPU Map/Fold boundary.  The first acceptance fixtures
are ordinary-JAX LayerNorm forward, JAX-owned backward, and their composed
boundary at `BF16[2048,4096]`, plus `BF16[7,13]` structural mutations.  No
production branch may select LayerNorm, an exported function name, a fixture
digest, or a fixed launch/slot count.

This is a no-hardware implementation slice.  It must preserve the reviewed ABI
10 branch and must not promote any evaluation cell before rebuilt-plugin H100
evidence is separately authorized and collected.

## Entry contract

The ordinary JAX forward is the explicit two-pass computation:

```python
def layer_norm(x, gamma, beta):
    x32 = x.astype(jax.numpy.float32)
    mean = jax.numpy.mean(x32, axis=-1, keepdims=True)
    centered = x32 - mean
    variance = jax.numpy.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(variance + 1e-5)
    return (normalized * gamma.astype(jax.numpy.float32)
            + beta.astype(jax.numpy.float32)).astype(jax.numpy.bfloat16)
```

JAX owns tracing and later AD.  The exported StableHLO is accepted only through
the existing typed conversion into a connected `shuttle.map`/`shuttle.fold`
region.  The compiler must not recognize the formula as LayerNorm.

## Production boundary

### Generic physical derivation

- Retain the existing SSA-derived materialization and schedule records.
- Derive launch count, slot count, dependencies, external binding ordinals, and
  temporary allocation from those records.
- Admit static row and column additive Folds by axis and extents.  Launch
  geometry and scratch must be computed from the schedule record; production
  code must not branch on `2048`, `4096`, `19`, `21`, or fixture identities.
- Preserve checked limits for launch count, slot count, task work, per-slot
  bytes, aggregate temporary bytes, grid/block size, and dynamic shared memory.
- Add scalar subtraction to the closed Map scalar expression emitter.
- Assign temporary reuse groups with a deterministic interval-coloring pass over
  materialization lifetimes.  Buffers may share a group only when lifetimes do
  not overlap and the group allocation satisfies the largest byte/alignment
  requirement.  This is required to keep the representative graphs within the
  existing bounded memory envelope without inventing a workload-specific cap.

### Runtime projection and lifecycle

- Validate any contiguous BF16 operand/result projection carried by the
  decoded invocation ABI; do not hard-code two operands, one result, or slot
  ordinals.
- Allocate every decoded temporary, load every decoded PTX slice, and enqueue
  every decoded launch in dependency order.  Allocate once per decoded reuse
  group and map each slot to its checked group allocation.  Counts come from
  the immutable `GpuExecutable` state.
- Compare FFI view dtype, rank, shape, byte length, alignment, pointer presence,
  address overflow, and pairwise non-aliasing against decoded bindings.
- Retain per-execution temporary ownership and stream-ordered completion.

### Versioning

The existing v1 target is intentionally a closed two-input/one-result RMSNorm
handler.  General projection/count semantics therefore require:

- pipeline ABI `10 -> 11`;
- FFI target `shuttle.gpu.executable_bundle.v2`, with no v1 registration in the
  new pipeline;
- transport/device/invocation/bundle schemas remain `2/3/3/2`.  Their byte
  grammar already carries variable entry, slot, dependency, projection, and
  reuse-group records.  ABI 11 changes which structurally valid graphs the
  compiler may emit and how the v2 handler consumes them, not the wire grammar.

Any later wire-field addition requires a separate schema bump and decoder; ABI
11 must not reinterpret an existing field.

The frozen materialization graphs provide the initial resource boundary:

| Shape/boundary | Tasks | Slots | Temporary bytes without reuse | Peak live bytes |
|---|---:|---:|---:|---:|
| 2048x4096 forward | 31 | 34 | 335,699,988 | 100,663,296 |
| 2048x4096 backward | 73 | 76 | 704,979,008 | 167,772,160 |
| 2048x4096 composed | 80 | 84 | 839,229,504 | 167,804,928 |
| 7x13 forward | 31 | 34 | 4,176 | 1,092 |
| 7x13 backward | 73 | 76 | 8,824 | 1,820 |
| 7x13 composed | 80 | 84 | 10,384 | 1,928 |

The ABI must validate actual reuse groups and total allocated group bytes; it
must not accept the table values as workload-selection constants.

## Red-first gates

1. Freeze six natural JAX LayerNorm StableHLO fixtures: two shapes crossed with
   forward, backward, and composed boundaries.  Pin source and normalized
   fingerprints, input/output layouts, and public result order.
2. Conversion proves only connected Map/Fold algebra, explicit subtraction and
   negation, complete source coverage, and final source erasure.  Forward has
   two row Folds; backward/composed also require feature-axis Folds.
3. Build/verify produces one generic GPU bundle and one v2 typed custom call.
   Assert counts and resource census derived independently from the frozen
   fixture, without copying production constants.
4. Existing ABI 10 RMSNorm transport and v1 lifecycle remain byte-for-byte and
   behaviorally unchanged on their historical branch; the ABI 11 pipeline
   rejects stale ABI 10 options and v1 target selection.
5. Public transport Load and an independent decoder agree on every launch,
   slot, dependency, code slice, external binding, and root.  Mutations cover
   count drift, projection reorder, Fold axis/geometry, resource overflow,
   source-task replay, code substitution, and crossed roots.
6. Pinned-XLA fake lifecycle uses at least two structurally different
   self-consistent transports through the same v2 handler.  It checks distinct
   Prepare states, all decoded allocations, all PTX loads, exact ordered launch
   arguments, failure propagation, and concurrent execution.
7. Rebuilt ordinary JAX is a later required gate: disabled reference, populate,
   and fresh-process reuse; bitwise SOURCE_ORDERED output; one attributed miss,
   one public hit, no reuse observer events, and immutable cache bytes.

## Scope exclusions

- No GPU FAST claim.  FAST requires a separate policy-legality and numerical
  contract, even if its initial code stream is identical.
- No Contract, Scan, Relation, Transport, collective, or EventTensor codegen.
- No H100/GB200 reservation, timing, profiling, upload, scorecard promotion,
  push, or integration.

## Follow-on sequence

1. General multi-axis/multi-result Map/Fold LayerNorm forward, backward, and
   composed boundaries, ABI 11, local and rebuilt-plugin gates.
2. GPU FAST policy with an independently reviewed error contract.
3. Native Contract scheduling/codegen and the four structural Contract/Map
   sweep cases.
4. Authority-separated H100 evidence, then reviewed timing; GB200/B200 follows
   only after the H100 architecture and numerical gates are accepted.
