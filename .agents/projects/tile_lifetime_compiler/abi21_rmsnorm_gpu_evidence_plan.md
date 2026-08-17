# Shuttle ABI21 RMSNorm GPU evidence plan

## Status

Local prehardware design. ABI21 compiles the frozen ordinary-JAX RMSNorm
forward, backward, and composed graphs through the generic GPU bundle pipeline.
No ABI21 wheel or image has been built, no PTX has been assembled with the
authorized CUDA toolchain, and no Shuttle kernel has run on H100 or B200.

This plan is not launch authority. Image construction, external dependency
transfer, GPU reservation, artifact upload, and controller signing require a
separate explicit authorization bound to the final source and artifact hashes.

## Objective

Produce architecture-conforming evidence for ordinary JAX 0.10.1 RMSNorm:

```text
ordinary JAX function
  -> JAX-owned VJP
  -> StableHLO
  -> generic Shuttle Map/Fold regions
  -> materialization and SIMT32 schedules
  -> shuttle.gpu.executable_bundle.v2
  -> installed JAX CUDA FFI runtime
```

The evidence must prove that no workload name, exact evaluation shape, source
digest, operation count, or whole-graph signature selects the GPU lowering.
SOURCE_ORDERED and FAST retain distinct option, bundle, and cache identities.
The evidence contract may name RMSNorm and freeze shapes; production admission
and code generation may not consume those identities.

## Shape tiers

The artifact has three shape roles. Results never promote between roles.

### Architecture execution

Execute logical shape `[2,7,19]`, with flattened row count 14 and feature count
19, for forward, recompute-backward, and composed boundaries under both
policies. This six-coordinate matrix is the bounded runtime, cache, observer,
transport, assembly, and numerical checkpoint.

Use exact source-visible FP32 epsilon `1e-6`. Public inputs and outputs are
BF16; reductions and intermediate arithmetic are FP32.

### Metamorphic compile witness

Compile `[3,2,5,11,263]` for the same three boundaries and both policies. This
rank-five sibling must establish generic rank flattening, projected parameter
addressing, and full-prefix Fold geometry. It is not executed and does not add
hardware, numerical, or performance status.

### Acceptance scorecard

The declared Target-1 shape remains exact rank-two `[2048,4096]`. The eventual
matrix is forward, recompute-backward, and composed under both policies on H100
and B200: 12 hardware cells.

Current ABI21 locally constructs all six representative GPU bundles. This
removes the historical compilation/cap claim as a blocker. The old backward
and composed totals of 537,083,940 and 604,192,804 bytes counted every logical
temporary without reuse. ABI21's lifetime-derived reuse groups require exactly
167,804,936 live temporary bytes at the actual XLA hook boundary for either
backward or composed, about 160.03 MiB and below the unchanged 256 MiB cap.
Successful bundle serialization and public reload prove the local resource
gate, but not device allocation, arithmetic, installed-wheel reachability,
SM100 code generation, or timing.

The raw checked-in StableHLO fixtures contain 48/51 backward and 51/54 composed
tasks/slots. The actual XLA hook, after the pinned StableHLO complex-math
expansion, contains 43/46 and 46/49. Evidence must bind the hook-boundary
topology and may record the raw fixture only as a source-regeneration witness.

## Public boundaries

The ordinary-JAX definitions are fixed before GPU execution:

```python
def rmsnorm(x, weight):
    variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    normalized = x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)
    return (normalized * weight.astype(jnp.float32)).astype(jnp.bfloat16)

def backward(x, weight, output_cotangent):
    _, pullback = jax.vjp(rmsnorm, x, weight)
    return pullback(output_cotangent)  # input_cotangent, weight_cotangent

def composed(x, weight, output_cotangent):
    result, pullback = jax.vjp(rmsnorm, x, weight)
    input_cotangent, weight_cotangent = pullback(output_cotangent)
    return result, input_cotangent, weight_cotangent
```

Roles are:

- forward: `(x, weight) -> result`;
- recompute-backward: `(x, weight, output_cotangent) ->
  (input_cotangent, weight_cotangent)`;
- composed: `(x, weight, output_cotangent) ->
  (result, input_cotangent, weight_cotangent)`.

The internal bundle result order is not the public order. Backward projects
custom-call results `(#1,#0)` to `(input_cotangent,weight_cotangent)` and
composed projects `(#0,#2,#1)` to
`(result,input_cotangent,weight_cotangent)`. The independent decoder must bind
these projections rather than assuming producer order.

## Frozen architecture topology

For `[2,7,19]`, each policy has exactly:

| Boundary | Launches | Slots | External bindings | Task census |
|---|---:|---:|---:|---|
| forward | 19 | 21 | 3 | 18 Map, 1 RowFold |
| backward | 48 | 51 | 5 | 43 Map, 3 RowFold, 2 ColumnFold |
| composed | 51 | 54 | 6 | 46 Map, 3 RowFold, 2 ColumnFold |

The executed six-coordinate matrix therefore contains 236 PTX/cubin slices.
The rank-five sibling must reproduce the same task counts without adding
assemblies. Its main full-prefix ColumnFold is `R=330,H=263`, grid 263, block
256, scratch 1024 bytes, and two serial tiles. The singleton-prefix Fold is
`R=1,H=263`, grid 263, block 32, scratch 128 bytes.

All materialization, schedule, target, transport, device, invocation, and
bundle schema versions remain the versions emitted by ABI21. The contract
records them from an independently decoded artifact; it does not widen them.

## Numerical oracles

The architecture tier uses deterministic, presealed inputs and three distinct
comparators:

1. disabled ordinary JAX/XLA, recorded as the matched framework baseline;
2. an independent high-precision RMSNorm and VJP implementation, which defines
   public-role numerical limits;
3. a new Transformer Engine 2.17 anonymous-shape probe using zero-copy rank-N
   to rank-two views `[14,19]` and `[330,263]`, exact epsilon `1e-6`, and the
   existing forward, backward-recompute, and composed API ordering.

The existing Target-1 TE contract is immutable: it is pinned to epsilon `1e-5`
and only `[7,13]` and `[2048,4096]`. It cannot be cited for the architecture
tier. TE is required for representative scorecard evidence, not for declaring
the small compiler architecture locally closed.

SOURCE_ORDERED and FAST currently select identical RMSNorm code realization
after policy metadata is removed. The prehardware verifier must compare decoded
tasks, schedules, normalized PTX, and launch descriptors pairwise. If they
differ, FAST stops until a separate numerical policy contract is reviewed.

ABI21 does not inherit the ABI5/Target-1 bitwise ordinary-JAX gate. StableHLO
permits implementation-defined binary-tree association while preserving
ascending lexicographic leaf order, and ABI21 records that contract as
`tree_association_free_leaf_order_fixed`. The new versioned contract therefore
requires bit-exact agreement with an independent software model of the emitted
Fold tree plus predeclared per-role independent-reference thresholds. Ordinary
JAX bitwise equality is recorded as an optional stronger diagnostic.

The contract must still report the historical 815-element BF16 `dx`
disagreement if reproduced. The current balanced, leaf-order-preserving
all-five-Fold realization matches `dgamma` but differs from the historical CPU
ordinary-JAX result in 815 of 8,388,608 BF16 `dx` elements; the best historical
alternate tree differed in 745 elements. That observation is not ABI21 GPU
evidence and cannot establish threshold acceptance. No threshold or tree-model
change is permitted after GPU output is observed.

## Compiler, process, and cache state

Bind the exact compiler commit and tree, JAX/XLA revisions, all three integration
patches, canonical SOURCE_ORDERED/FAST ABI21 options, tuning digest, fixture
bytes, JAXPR/StableHLO hashes, and generated/final MLIR hashes.

Run three fresh process classes:

1. baseline: Shuttle disabled, ordinary JAX only, isolated cache;
2. populate: empty cache, compile the six architecture coordinates;
3. reuse: new process, reuse the exact six entries without compilation.

Populate requires six misses, six unique entry keys/files, and 18 observer
events in six ordered triplets:

```text
algebra_coverage -> generated_bundle -> final_erasure
```

Each triplet has one positive invocation ID; invocation IDs are distinct across
coordinates. Reuse requires six hits, zero observer events, no added files,
and byte-identical cache filenames, sizes, and SHA-256 digests.

Observer verification binds the full source membership and result anchors,
empty unsupported set, policy/options/tuning identity, generated bundle root,
and final transformed module. Final erasure must contain neither StableHLO nor
Shuttle semantic operations outside the custom call.

## Transport and PTX closure

Split the existing ABI15 decoder into a workload-neutral schema-v2 decoder and
an ABI21 RMSNorm semantic verifier. For every executed launch, join:

- launch ordinal, dependencies, inputs, outputs, code offset, length, and SHA;
- slot rank, dtype, shape, strides, bytes, access, storage, binding, lifetime,
  and reuse group;
- grid, block, dynamic shared memory, numerical order, and task fingerprint;
- descriptor, device, invocation, ABI, bundle, and transport roots.

The RMSNorm verifier independently validates:

- direct/projected Map indexing, including weight floordiv expansion;
- FP32 arithmetic and exact BF16 round-nearest-even public conversions;
- rsqrt as the admitted `sqrt.rn.f32` plus divide dataflow;
- RowFold final-axis addressing;
- ColumnFold flattened-prefix addressing `(leaf*H+feature)*4`;
- initializer, tree, predicates, accumulator, shared-memory, and store flow;
- first, last, and tile-boundary address witnesses;
- absence of unresolved externs and workload/oracle symbols.

The validator must consume a closed normalized executable instruction sequence,
not selected regex matches. Pointer, value, accumulator, predicate, branch,
shuffle, conversion, and store redefinitions are mutation-tested after all
roots are recomputed.

Assemble all 236 slices with a pinned ptxas executable, SHA, version, and exact
`-arch=sm_90 -v` arguments. Each record requires nonempty cubin bytes, exact
PTX/cubin SHA-256, positive register reporting, and zero spills. B200 evidence
requires a separate SM100 build and artifact identity; SM90 cubins cannot be
reused as Blackwell evidence.

## Artifact roles and closure

Add only versioned `abi21_rmsnorm_*` files. Preserve ABI15 contracts, fixtures,
receipts, and accepted history byte-for-byte.

Static roles include contracts, source allowlist and capsule, ordinary-JAX
fixtures, generator, numerical inputs/references, evaluation manifest, TE
contract, build policy, image policy, and tool identities. Dynamic roles are
keyed by shape, boundary, and policy and include generated/final MLIR,
transport bytes and independent descriptor, cache entry, process/observer
records, inputs/outputs/repeats, and every PTX/cubin/resource record.

The manifest enumerates every regular file and implied directory, rejects
symlinks and special files, and joins size and SHA-256. The raw subject archive
contains no `accepted_bundle`, controller ACK, terminal receipt, or rewritten
authority role.

Reuse the ABI15 source-capsule, raw-export, runtime-wheel/RECORD/mapping,
registration-pointer, cache-snapshot, and ptxas parsing mechanics only after
extracting them into workload-neutral modules. Do not rename matmul-specific
topology or numerical validators.

## Red-first implementation sequence

1. Add canonical ABI21 RMSNorm contract, architecture contract, role manifest,
   and unsealed build policy with exact byte-drift tests.
2. Add a workload-neutral transport decoder extracted from ABI15 and prove the
   old ABI15 decoder produces identical structural records.
3. Add ABI21 Map/RowFold/ColumnFold semantic validation and mutation tests.
4. Add the independent numerical oracle and deterministic architecture-tier
   inputs; seal their exact raw-byte outputs before GPU work.
5. Add source capsule, artifact closure, baseline/populate/reuse subject, and
   verifier. The stdlib launcher must reject before JAX import while authority
   remains unsealed.
6. Add a prehardware builder that reconstructs the exact compiler/wheels,
   extracts all six transports, assembles 236 slices, and emits a raw candidate.
7. Run the complete local contract suite, native ABI21 suite, Python suite,
   changed-files precommit, and an independent adversarial review.
8. Only after a clean reviewed preimage, request separate image/build/H100
   authority. Treat the first H100 run as operational/numerical evidence.
9. Add counterbalanced performance and B200 only after the operational artifact
   is accepted and a separate timing/SM100 contract is reviewed.

Representative and performance work must not block completing the small local
prehardware contract, and success of the small contract must not promote the
representative scorecard.

## Required mutation sensitivity

At minimum reject, after recomputing every affected digest:

- missing, duplicate, or substituted shape/boundary/policy coordinates;
- swapped public result projections or role names;
- collapsed SOURCE_ORDERED/FAST option, cache, bundle, or transport identity;
- source membership, result-anchor, coverage, observer phase, invocation, or
  final-erasure substitutions;
- slot dtype/shape/stride/access/storage/binding/lifetime/reuse mutations;
- launch dependency, ordinal, Fold axis/order/geometry, or Map divisor changes;
- pointer/value/accumulator/predicate/branch/shuffle/store PTX clobbers;
- PTX/cubin omission, reordering, truncation, replacement, or resource drift;
- cache snapshot filename, size, policy attribution, hit/miss, or byte drift;
- runtime wheel, RECORD, loaded mapping, inode/device/range, registration target,
  or handler-pointer substitution;
- one-bit input, output, reference, or repeatability drift;
- any extra file, directory, symlink, FIFO, socket, or device in the artifact.

## Performance protocol

Predeclare representative performance before execution: three independent
counterbalanced runs, 50 CUDA-event samples per boundary/backend/run, compile,
tuning, warmup, and cache population excluded. Compare Shuttle against every
qualified Transformer Engine backend pair and ordinary JAX. Every paired-run
Shuttle/expert ratio must be at most 1.20; 1.10 is the stretch target. Do not
choose the fastest comparator or policy after observing results.

The first architecture artifact is untimed. TE timings without corresponding
Shuttle samples cannot satisfy the scorecard.

## Design stops and launch stops

The prehardware checkpoint remains explicitly unsealed until all of these are
closed:

- hardware allocation and execution of the locally admitted representative
  167,804,936-byte live temporary plan;
- representative agreement with the sealed schedule-tree oracle and
  independent-reference thresholds for both reduction policies;
- installed-wheel ordinary-JAX CUDA-plugin transform reachability;
- exact build/image/wheel/tool/PTX/cubin identities;
- SM100 code generation and assembly for B200;
- controller signer, receipt, task-success, and terminal sequencing if receipts
  are required for the chosen run;
- H100 and B200 execution, retrieval, independent verification, and timing.

Stop locally on any test failure, dirty source, capsule mismatch, open schema,
unmanifested byte, noncanonical archive, or failed mutation. Stop before
external action on missing authority or mutable image identity. On hardware,
seal exactly one failure and do not retry or publish partial headline records.

## Separate architecture debt

ABI21's GPU path is workload-generic. Its still-public legacy CPU executable
bundle path contains exact `7x13`/`2048x4096` and Fold-count dispatch. Retire
that CPU proof from production in a separate change. Do not weaken GPU identity
or provenance checks, which authenticate derived artifacts rather than select a
workload.
