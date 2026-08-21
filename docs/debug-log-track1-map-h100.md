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

## Future work

- [ ] Record final wheel/DSO evidence from r5 or a later authorized H100 run.
