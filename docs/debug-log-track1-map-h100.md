# Debugging log for Track 1 Map H100

The r5 goal is to make the reviewed Shuttle StableHLO transform available through
the compiled `jaxlib` runtime boundary before the first GPU compilation. The fix
must preserve the generic registration contract and remain observable through a
real module-transform invocation.

## Initial status

Iris job `/dlwh/shuttle-track1-map-h100-prototype-r4` passed source hydration,
offline wheel build, isolated compiled-wheel installation, pinned JAX `0.10.1`
source loading, and runtime import verification. Its first populate compilation
failed after 14 minutes 58.63 seconds with:

```text
FAILED_PRECONDITION: StableHLO module transforms were enabled, but none are registered
```

The error comes from the patched XLA registry's fail-closed path. The runtime
reached XLA with transform options enabled, but the Shuttle adapter's static
initializer had not populated that process's registry.

## Root cause

`ShuttleXlaRegistryAdapter` is linked only into JAX's nanobind `_jax` shared
object. The GPU compile crosses PJRT's C API: `_jax` serializes StableHLO and
compile options, then `pjrt_c_api_gpu_plugin.so` converts that serialized module
to HLO. Each shared object owns a separate process-local transform registry.
The adapter and Python observer therefore live in `_jax`, while the registry
consulted by the GPU plugin is empty. `alwayslink` cannot bridge that DSO-local
state.

The transform must run on the C API client side before program and options are
serialized into the plugin. The client consumes only the enable override so the
plugin does not attempt a second transform; it preserves the canonical options
in both debug options and environment overrides so SO/FAST and ABI policy bytes
remain part of downstream cache identity. Explicit `_jax`
module initialization must register the adapter in the same DSO as the Python
observer, avoiding static-initializer retention and load-order ambiguity.

## Changes to make

- Inspect the Shuttle adapter target, JAX patch dependencies, final `_jax` link
  boundary, and available wheel/symbol evidence.
- Add a behavior regression that crosses the public JAX extension/module-transform
  path and fails when the adapter is absent; do not assert registry membership.
- Replace static-initializer-only registration with the smallest explicit,
  idempotent registration entry point and call it from `_jax` initialization.

## Red regression

The native adapter test now calls the missing explicit registration API through
the real observed transform pipeline. It fails to compile at the old tip because
that API does not exist.

The patched XLA C API client test registers a behavior-bearing transform that
adds a module attribute, compiles through the public `PjRtCApiClient`, and
intercepts `PJRT_Client_Compile`. It requires the serialized module to contain
the attribute and the forwarded options to have consumed both Shuttle keys.
The old boundary forwards the untransformed module and enabled options, so this
test is red independently of registry-membership inspection.

The local native adapter target produced the expected compile failure because
`registerShuttleXlaTransform` is absent. The C API test patch applies cleanly to
the exact patched XLA checkout. A local build of that larger target stopped in
analysis because the offline cache lacks the `shardy` repository; no download
was allowed. The serialized-boundary expectations remain behavior-bearing and
will be compiled by the pinned online hydration followed by offline wheel build.

## Repair

The adapter now exposes an idempotent registration function. The pinned JAX
patch calls it during `_jax` module initialization, which gives the linker a
direct symbol reference and registers in the DSO that also owns the Python
observer.

The XLA client applies enabled StableHLO transforms to MLIR programs before
serializing `PJRT_Client_Compile` or `PJRT_Compile` arguments. On success it
clears the debug enable bit and removes only the enable override. The canonical
options string remains in the debug options and environment overrides, so
numerics and ABI policy remain part of downstream compile/cache identity. The
direct MLIR-to-HLO path uses the same helper; a second call observes the cleared
enable bit and cannot double-transform.

Disabled compilation returns without materializing a `debug_options` field, so
ordinary XLA request bytes remain unchanged. The XLA regression serializes
disabled options before and after preprocessing and requires byte equality.

## r5 patch-stack failure

r5 stopped before build or GPU execution after 16.44 seconds. The hydrated JAX
checkout matched commit `619764c15117fbefc4ba13ab941871cb514c23f6`, but the
ordered patch gate reported:

```text
error: patch failed: jaxlib/jax.cc:135
error: jaxlib/jax.cc: patch does not apply
```

The fourth JAX patch had been checked against a synthetic post-observer file.
Its module-initialization hunk incorrectly included `NB_MODULE` as adjacent
context even though the exact pinned source has intervening initialization.
The previous test mocked subprocess dispatch and therefore verified command
shape without applying the complete stack to pinned source bytes.

## r6 patch-stack repair

Checked source capsules now contain every exact JAX and XLA file touched by the
ordered patch rosters. Their manifests bind the upstream commit and tree plus
each Git blob identity, SHA-256, size, and path. The public build boundary test
materializes both capsules and applies both complete stacks; this reproduces
the r5 failure before the repair.

Patch validation now uses an isolated temporary Git index. Each patch is
checked and applied to that index in roster order, allowing later patches to
depend on earlier changes without touching either source checkout. Both stacks
must validate before either live checkout is modified. A late fourth-XLA-patch
mutation proves that neither live checkout changes on validation failure. The
fourth JAX patch now anchors its registration call to the exact post-observer
function context.

## r6 observer failure

r6 built the patched wheels, passed runtime import and GPU preflight, and
entered populate. The disabled coordinate completed first. The first enabled
coordinate then compiled and blocked on both the primal and analytic-gradient
GPU results before observer validation failed with:

```text
ValueError: observer phases do not share one positive invocation
```

The validator had already established that the snapshot contained exactly the
three ordered `algebra_coverage`, `generated_bundle`, and `final_erasure`
records. The native value-and-gradient fixture reproduces the same module: one
JIT compilation contains two selected Map regions, becomes two custom calls,
and emits three module-level observer records rather than records per region or
per result. All three records share invocation ID zero.

The producer initialized its atomic invocation counter to zero and returned the
pre-increment value. Disabled compilation emits no Shuttle records, so the
first enabled transform in a fresh process necessarily received zero. This is
why local tests that checked only equality between IDs passed while the public
subject's intentional positive-ID contract rejected the snapshot.

## r7 observer repair

Observer IDs now begin at one. A wrapped counter returns zero only after the
entire positive 64-bit space has been consumed; that state terminates with a
stable exhaustion diagnostic rather than reusing an ID. The public registered
transform regression now requires a positive ID and equality across all three
phases. It fails on the first transform at the old tip and covers the same
multi-custom-call value-and-gradient fixture used by the H100 subject without
relaxing subject attribution.

## Future work

- [ ] Record final wheel/DSO evidence from r7 or a later authorized H100 run.
