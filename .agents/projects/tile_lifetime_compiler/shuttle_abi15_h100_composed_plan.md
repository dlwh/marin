# Shuttle ABI 15 composed matmul H100 evidence plan

## Status

Pre-run contract design. No image build, upload, PTX assembly, CUDA execution,
cache observation, numerical evidence, profiling, or timing has occurred.

## Objective

Produce the first architecture-conforming hardware evidence for ABI 15's
ordinary-JAX multi-region GPU path. One `jax.jit` composed function must lower
through JAX AD and StableHLO into two independently derived Shuttle custom
calls: a forward Contract/Map region and a backward multi-Contract/Map region.
Both SOURCE_ORDERED and FAST must run through the same public compiler entry
point and retain distinct option, device, schedule, observer, and persistent
cache identities.

This is an untimed assembly, numerical, runtime-identity, and fresh-process
cache checkpoint. It does not change a performance scorecard.

## Frozen cell

- Pinned JAX and jaxlib: `0.10.1`.
- Shape `(M,K,N) = (33,2,34)` with BF16 lhs, rhs, output cotangent, primal,
  lhs cotangent, and rhs cotangent.
- Boundary: ordinary `jax.vjp` composed forward plus pullback.
- Policies: SOURCE_ORDERED and FAST with exact default tuning.
- The dimensions are pairwise distinct. Both backward reductions exceed one
  warp: dRhs reduces 33 rows and dLhs reduces 34 columns. The transpose remains
  nondegenerate because `K=2`.
- Compiler production commit
  `d31a88cb9a93e591469fa8edfcf9dacada90c7b4`, tree
  `66175cbd89dcac0857ee2fbc456f7715170d239c`. Test hardening commit
  `c11c81ff1b24e6ba4c0e3dfc7d843333fcb43919` is additive and does not alter
  production code.

## Expected compiled topology

Each policy produces exactly one persistent JAX executable cache entry and two
Shuttle custom calls inside that executable.

1. Forward call: two launches, four slots, three external bindings, one BF16
   primal result.
2. Backward call: ten launches, thirteen slots, five external bindings, two
   BF16 cotangent results. Contract launch ordinals are 3 and 8; the transpose
   Map is ordinal 4.

Per-call materialization/schedule/transport/device/invocation/bundle schemas
remain `2/2/2/3/3/2`; the target is
`shuttle.gpu.executable_bundle.v2`. The generated observer fingerprint must
bind both tuples in source order.

The backward affine maps must realize:

- dRhs pre-transpose: cotangent `(d2,d0)`, lhs `(d2,d1)`, result `34x2`,
  reduction 33;
- transpose: input `(d1,d0)`, `34x2 -> 2x34`;
- dLhs: cotangent `(d0,d2)`, rhs `(d1,d2)`, result `33x2`, reduction 34.

## Independent numerical inputs

Both datasets run through each already-compiled policy without adding cache
entries.

The dense dataset uses exact BF16 dyadic values:

```text
lhs[i,k] = (((17*i + 29*k + 3) mod 31) - 15) / 8
rhs[k,j] = (((23*k + 11*j + 5) mod 29) - 14) / 8
dy[i,j]  = (((13*i + 7*j + 1) mod 27) - 13) / 8
```

The sparse policy discriminator uses raw BF16 `+2^25`, `-2^25`, `1`, and `0`
bits. It places the large cancellation terms at reduction indices 0 and 32
and the unit at index 1. SOURCE_ORDERED must produce zero while FAST must
produce one at selected dLhs and dRhs coordinates. This exercises the FAST
lane-`+32` iteration and fixed shuffle tree.

References are independent of JAX and the Shuttle emitter:

1. a software IEEE binary32 fused-multiply-add and BF16-RNE interpreter for
   exact raw-bit SOURCE_ORDERED and FAST results;
2. an exact-rational algebra oracle for affine-map and aggregate error reports.

Ordinary JAX/XLA is recorded as a baseline, not treated as the independent
oracle.

## Process and cache state machine

One H100 task launches three fresh subprocesses:

1. `baseline`: Shuttle disabled, isolated cache, ordinary JAX/XLA on both
   datasets;
2. `populate`: fresh empty persistent cache, compile SO and FAST, zero hits,
   exactly two new entries, three ordered observer phases per policy, then
   sequential and two-thread repeatability on both datasets;
3. `reuse`: new PID, exactly two hits, zero observer events, unchanged cache
   roster and bytes, identical outputs and runtime binary mappings.

Every populate invocation records one invocation ID across
`algebra_coverage`, `generated_bundle`, and `final_erasure`, exact policy and
tuning digests, empty unsupported set, complete coverage, and exact generated
and final fingerprints. The cache record binds the public options, compiled
executable, both generated bundle tuples, entry bytes, and the populate/reuse
process identities.

## PTX and runtime joins

Assemble all 24 launch slices: 12 per policy, including structurally repeated
Map kernels. Every ordinal records PTX offset/length/digest, ptxas binary and
argv identity, nonempty cubin digest, registers, zero spills, launch geometry,
and exact transport root. SOURCE_ORDERED Contract slices require serial
`fma.rn.f32` without shuffles. FAST requires stride 32 and the fixed five-step
shuffle/add tree. All slices must preserve their independently decoded affine
address expressions.

The installed runtime must join wheel archive members, installed files, and
loaded mappings for `_jax`, `libjax_common`, `cuda_plugin_extension`, and the
CUDA PJRT plugin. Registration must expose exactly one Shuttle v2 target with
API version 1, traits 0, and four distinct non-null lifecycle pointers while
preserving upstream handlers.

## Authority and retrieval

Use a two-stage seal. A reviewed pre-run manifest pins this contract, compiler
and harness sources, fixture and input bytes, expected oracle outputs,
transport roots, observer fingerprints, build commands, immutable image
digest, exact one-H100 resource request, and zero retries. The job runs that
manifest's exact capsule.

Reuse the reviewed text-only READY/ACK protocol: canonical ASCII JSON and
base64 chunks no larger than 256 KiB raw cross Iris exec while the task remains
alive. The controller reconstructs the archive, reruns an independent semantic
verifier, and writes an ACK bound to the exact archive and export manifest.
The task exits only after validating that ACK.

No self-hash or caller-editable local manifest is an authority root. The
pre-run manifest must be committed and independently reviewed before any image
upload or reservation. A post-run accepted bundle cannot rewrite it.

## Stop conditions

Stop before launch on any dirty or mismatched source, fixture drift, failed
mutation test, noncanonical capsule, mutable image reference, missing authority
seal, wrong toolchain, or incomplete independent decoder/oracle.

On hardware, produce no accepted bundle and do not retry if device identity,
plugin registration, any of 24 PTX assemblies, transport/root closure,
observer phase closure, numerical bits, repeatability, cache transition,
runtime mapping, artifact roster, retrieval reconstruction, or ACK differs.

Even a green result remains
`untimed_cuda_sm90_abi15_composed_matmul_multiregion_assembly_numerical_cache`
with `performance_status=not_started` and no scorecard promotion.
