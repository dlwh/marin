# Pinned JAX jaxlib composition overlay

These patches apply to JAX 0.10.1 commit
`619764c15117fbefc4ba13ab941871cb514c23f6`. That release pins XLA commit
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`.

Patch `0001` adds the always-linked
`@shuttle_mlir//:ShuttleXlaRegistryAdapter` and
`@shuttle_mlir//:ShuttlePythonPhysicalArtifactBridge` libraries to the final
`//jaxlib:_jax` extension. Patch `0004` registers both at module initialization.
The adapter depends on XLA's public generic StableHLO transform registry. XLA's
registry and `mlir_to_hlo` targets do not depend on Shuttle.

The production patch also links `ShuttlePythonPhysicalArtifactBridge`. When
`shuttle.compiler_options` selects GPU executable output, Shuttle discovers
installed providers from the `shuttle.gpu_physical_artifact_compilers` Python
entry-point group and binds their composite callback to the single-assignment
`_jax._shuttle_physical_artifact.register` hook. Applications do not import or
register a workload compiler. Each provider receives a canonical structured
record of generic Shuttle algebra plus its verified materialization and
schedule plans, and returns native fatbin transport bytes or `None` when it has
no legal physical schedule. Native code re-parses and source-validates that
transport before it can replace the algebra region. Provider discovery is
lazy: CUDA compiler dependencies are imported only after XLA submits a physical
artifact request.

The second patch contains an acceptance-only configuration selected with
`--define=SHUTTLE_TEST_OBSERVER=1`. It compiles a guarded private
`_jax._shuttle_test_observer` binding into the final `_jax` DSO and links the
external observer bridge. Without that define, the binding and bridge target
are absent. The registry adapter remains linked in both configurations.

Apply the XLA patches from `../xla_patch` to an exact pinned XLA checkout. Apply
both production JAX patches, then build in JAX's default WORKSPACE mode with
the local repositories injected at the command line:

```bash
test "$(git rev-parse HEAD)" = 619764c15117fbefc4ba13ab941871cb514c23f6
git apply --check /path/to/0001-link-shuttle-xla-registry-adapter.patch
git apply /path/to/0001-link-shuttle-xla-registry-adapter.patch
git apply --check /path/to/0004-register-shuttle-at-module-init.patch
git apply /path/to/0004-register-shuttle-at-module-init.patch
bazel build \
  --override_repository=xla=/path/to/patched/xla \
  --override_repository=shuttle_mlir=/path/to/marin/lib/shuttle/mlir \
  //jaxlib:_jax
```

Apply `0002-add-acceptance-observer-bridge.patch` after `0001` only for the
acceptance wheel. The source/query proof expects both patches and checks that
the test bridge is selected only with its build define.

The CUDA-only ABI 13 proof additionally applies
`0003-link-shuttle-cuda-ffi.patch` to the exact pinned JAX checkout. That patch
links `ShuttleGpuFfi` only into `cuda_plugin_extension` and exposes the single
four-stage `shuttle.gpu.executable_bundle.v2` bundle from the plugin's existing
`ffi_handlers()` entry point. It does not register the target in `_jax` or the
Host adapter. The checked-in registration driver routes that returned bundle
through the plugin's real PJRT registration entry point using a fake extension;
it does not load kernels or launch GPU work.

The Track 1 single-job prototype is authorized by the canonical
`scripts/shuttle/track1_map_h100_prototype_authorization.json` document. The
reviewed Iris job authorization must supply that document's approved SHA-256 as
an independent literal; the job must not compute it from the candidate file.
This external binding avoids a circular trust chain because the authorization
hashes the source manifest and the manifest hashes the launcher. The launcher
remains fail-closed without both inputs. Render the network hydration, offline
build, isolated wheel install, and two worker commands without executing them:

```bash
python -B lib/shuttle/mlir/jax_patch/track1_map_h100_launcher.py \
  --authorization-file scripts/shuttle/track1_map_h100_prototype_authorization.json \
  --approved-authorization-sha256 IRIS_JOB_APPROVED_AUTHORIZATION_SHA256 \
  --workspace /tmp/shuttle-track1-prototype \
  --runner-repository /app \
  --python /opt/h100-evidence-runtime/bin/python \
  --cuda-root /usr/local/cuda-13.2 \
  --base-image ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef \
  --dry-run
```

Use `--execute` only inside the authorized one-attempt `H100x1` job. The
workspace must not exist. Before network access, the executor validates the
Iris-synced Shuttle and Track 1 source files against the authorization's
canonical source manifest. It then fetches the pinned JAX, XLA, and Bazel
inputs once, checks their identities and patch bytes, warms Bazel's
repository state—including the pinned hermetic CUDA 13.0.0 packages—and then
repeats the three-wheel build with repository downloads disabled. The physical
CUDA 13.2 compiler tools are identity evidence only; neither build phase sets
`LOCAL_CUDA_PATH`. The isolated runtime installs the three compiled wheels with
dependency and index access disabled, adds the exact pinned JAX checkout as its
pure-Python source, reuses dependency packages from the immutable base-image
runtime behind that source, and verifies the JAX origin and base version before
populate. The child environment does not expose system site packages. Populate
and reuse run as separate Python processes. Build
logs and wheels remain under the ephemeral workspace; stdout contains one
canonical result document.

For the reviewed CPU acceptance wheel, first run the guarded source/query
proof, then build a release-versioned jaxlib wheel:

```bash
git apply --check /path/to/0002-add-acceptance-observer-bridge.patch
git apply /path/to/0002-add-acceptance-observer-bridge.patch
python /path/to/verify_acceptance_patch.py \
  --bazel /path/to/bazel-7.7.0 \
  --jax-source /path/to/jax \
  --xla-source /path/to/patched/xla \
  --shuttle-mlir /path/to/marin/lib/shuttle/mlir \
  --output-user-root /path/to/bazel-output
python build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --bazel_path=/path/to/bazel-7.7.0 \
  --output_path=/path/to/dist \
  --bazel_options=--override_repository=xla=/path/to/patched/xla \
  --bazel_options=--override_repository=shuttle_mlir=/path/to/marin/lib/shuttle/mlir \
  --bazel_options=--define=SHUTTLE_TEST_OBSERVER=1 \
  --bazel_options=--repo_env=ML_WHEEL_TYPE=release
```

The expected output is one `jaxlib-0.10.1-cp312-*.whl`. Install that wheel,
the matching JAX 0.10.1 source, and `lib/shuttle` into an isolated environment,
then run `shuttle_jaxlib_acceptance.py` with an empty `--work-directory` and a
`--report` path. The checked-in driver owns the two fresh persistent-cache
workers, the cache-disabled concurrency/lifetime worker, and a cache-disabled
context-manager worker that exercises normal and exceptional native capture
teardown. The Iris runner must not replace those semantics with inline Python.

Before building, run the pinned fixture-contract regression suite explicitly:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run pytest -q \
    lib/shuttle/mlir/jax_patch/test_acceptance_contract.py \
    lib/shuttle/mlir/jax_patch/test_verify_acceptance_patch.py \
    lib/shuttle/mlir/jax_patch/test_fixture_audit_gate.py \
    lib/shuttle/mlir/jax_patch/test_bf16_composed_fixture_oracle.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/verify_acceptance_fixture_oracles.py
```

The remote preflight runs the default no-write audit through the checked-in
gate, which builds and resolves the test-only normalizer target directly:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
    --bazel /path/to/bazel-7.7.0 \
    --xla-source /path/to/patched/xla \
    --output-user-root /path/to/bazel-output \
    --repository-cache /path/to/repository-cache \
    --jobs 24 \
    --ram-mb 65536 \
    --python /path/to/jax-0.10.1-python \
    --generator lib/shuttle/mlir/test/Inputs/regenerate-jax-fixtures.py
```

The gate requires the single executable output from
`@shuttle_mlir//:shuttle-test-opt`. Production `shuttle-opt` does not link the
test-only fingerprint pass and is rejected before the fixture audit starts.
The generator includes the failed tool argv, exit code, and bounded stdout and
stderr when a normalizer subprocess fails.

The four composed BF16 primal/VJP fixtures are an inventory boundary, not an
acceptance claim. Audit them separately with the same built normalizer and the
independent hook-boundary verifier:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
    --bazel /path/to/bazel-7.7.0 \
    --xla-source /path/to/patched/xla \
    --output-user-root /path/to/bazel-output \
    --repository-cache /path/to/repository-cache \
    --jobs 24 \
    --ram-mb 65536 \
    --python /path/to/jax-0.10.1-python \
    --generator lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-composed-fixtures.py \
    --verifier lib/shuttle/mlir/jax_patch/verify_bf16_composed_fixture_oracle.py
```

The six BF16 row Fold fixtures cover forward, JAX-owned backward, and composed
boundaries at `2048x4096` and the structural `7x13` mutation. They are fixture
inventory only. Their oracle manifest explicitly records that the evaluation
oracle is unpinned and that no hardware evidence exists. Audit the corpus with
JAX and jaxlib 0.10.1:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python \
    lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-row-fold-scale-fixtures.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python \
    lib/shuttle/mlir/jax_patch/verify_bf16_row_fold_scale_fixture_oracle.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run pytest -q \
    lib/shuttle/mlir/jax_patch/test_bf16_row_fold_scale_fixture_oracle.py
```

To refresh an intentional toolchain or source change, run the generator with
`--write`, then the verifier with `--write`, and rerun the test.

The independent verifier hashes a canonical graph inventory instead of MLIR
SSA names. The inventory includes every top-level operand, attribute, result
type, source-result reference, output anchor, and each reducer block's scalar
`stablehlo.add` result and `stablehlo.return` operation reference.

`shuttle_jaxlib_target1_acceptance.py` is the installed-wheel ABI 9 CPU
contract for those six boundaries. It runs each boundary under
`source_ordered` and `fast`, for twelve distinct compiler/cache identities. A
cache-disabled process saves disabled ordinary-JAX BF16 baselines. Separate
populate and reuse processes require exact bitwise `y`, `dx`, and `dgamma`
parity, total observer source coverage, final Shuttle erasure, twelve uniquely
attributed cache entries, and twelve public second-process cache hits. The
current pipelines round-trip these operations, so bitwise parity is required.
A future non-bitwise `fast` rewrite needs a separately reviewed tolerance
revision before any timing run.

The driver is checked locally without claiming wheel execution:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch:lib/shuttle/src \
  uv run --project lib/shuttle --group test pytest -q \
    lib/shuttle/mlir/jax_patch/test_target1_acceptance_contract.py \
    lib/shuttle/mlir/jax_patch/test_shuttle_jaxlib_target1_acceptance.py
```

Run the installed-wheel contract only inside a reviewed Linux CPU runner:

```bash
JAX_PLATFORMS=cpu python \
  lib/shuttle/mlir/jax_patch/shuttle_jaxlib_target1_acceptance.py \
  --work-directory /fresh/target1-acceptance \
  --report /fresh/target1-acceptance.json
```

The contract is audited from the checked-in pinned forward and JAX-owned VJP
StableHLO fixtures at XLA's module-transform hook boundary. The fixture audit
reapplies the pinned StableHLO complex-math expander and rejects a changed
hook-boundary digest. This matters for the VJP: XLA moves its scalar constant
ahead of the first dot before Shuttle assigns source ordinals. The contract
requires exact normalized selected regions, the complete algebra/lowered
manifest, the VJP constant/broadcast/subtract exclusion island, phase-specific
provenance erasure, and the final normalized module fingerprint. The validator
does not select behavior from callable names or workload keys.

The repository overrides avoid checked-in machine-specific paths. A Bazel
dependency query against the exact JAX release and patched XLA revision proves
the final `_jax` target reaches the always-linked adapter. This checkpoint only
claims the CPU jaxlib composition path. Dynamically loaded GPU PJRT plugins
require their own explicit adapter linkage and registration proof.

## Standalone native CUDA runner

`build_native_gpu_runner.py` builds
`@shuttle_mlir//:gpu_fatbin_h100_test` against the pinned XLA revision above.
This executable exercises the native `GpuRunner` directly. It does not require
a rebuilt JAX wheel or CUDA PJRT plugin.

Run from the Marin repository root on Linux x86-64 with uv, Python 3.12, Git,
Curl, a host C++ compiler, and `ldd`. The directory passed to `--workspace` must
not exist. Outside Iris, replace `/iris/outputs` with a writable output path. Build on a
CPU worker; the validated resource allocation was 32 CPUs, 128 GB memory, and
100 GB scratch with a 60-minute job limit.

```bash
uv run --no-project --python 3.12 python \
  lib/shuttle/mlir/jax_patch/build_native_gpu_runner.py \
  --workspace /tmp/shuttle-native-v12 \
  --output /iris/outputs \
  --bazel-url https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64 \
  --bazel-sha256 fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf \
  --cuda-version 12.9.1 \
  --jobs 32 \
  --ram-mb 98304
```

The helper verifies Bazel's digest and checks out XLA at
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`. It applies
`track2_xla_workspace.patch`, overrides the Shuttle repository with the current
source tree, and explicitly selects hermetic Python 3.12, CUDA 12.9.1, and
compute capability 9.0. Bazel analysis downloads the pinned toolchain before
compilation. The helper then compiles two independent fatbins from
`../test/gpu-fatbin-duplicate-symbol.cu` before building the native executable.

NVCC host preprocessing uses the sysroot selected by XLA's
`sysroot_linux_x86_64` repository. The validated build selected
`sysroot_linux_x86_64_glibc_2_27`. Passing that sysroot through
`--compiler-options=--sysroot=...` avoids mixing CUDA 12.9 headers with newer
host glibc declarations of `cospi`, `sinpi`, `cospif`, and `sinpif`. The helper
records the selected repository and its alias hash. It does not modify CUDA
or system headers.

The output directory contains the executable, `image0.fatbin`,
`image1.fatbin`, required workspace shared libraries under `lib/`, `ldd.txt`,
phase logs, `build-plan.json`, and `build-result.json`. The result records
commands, XLA and Bazel identities, source hashes, NVCC and image hashes, and
the executable hash. `status: built` is written after all these steps pass.
On a CPU worker, only unresolved `libcuda.so.1` is permitted: the manifest
records that the GPU worker must provide this driver library. Other unresolved
libraries fail packaging. Preserve the output directory together when moving
the executable to a GPU worker in the same region.

On an H100 worker with a driver supporting CUDA 12.9.1, set `native_output`
to the extracted build output directory:

```bash
native_output=/path/to/native-build-output
export LD_LIBRARY_PATH="$native_output/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SHUTTLE_TEST_FIRST_IMAGE="$native_output/image0.fatbin" \
SHUTTLE_TEST_SECOND_IMAGE="$native_output/image1.fatbin" \
  "$native_output/gpu_fatbin_h100_test" \
  --gtest_filter=GpuFatbinH100Test.ResolvesDuplicateKernelNamesPerImageWithScalarResult
```

Both images export `shuttle_shared_symbol`. Their launches compute `(3 + 1) * 2`
and must return the rank-zero scalar `8`. The test checks image-specific symbol
resolution through the public native runner.

The dense benchmark exporter produces fixtures for scalar loss and all
parameter/input gradients. For example, export a pair using the package's
CUDA benchmark dependencies on an H100 worker:

```bash
python lib/tile_lifetime/benchmarks/setup_h100_composition_runtime.py \
  --directory /tmp/shuttle-supported-runtime \
  --manifest /path/to/runtime-manifest.json
/tmp/shuttle-supported-runtime/venv/bin/python \
  lib/tile_lifetime/benchmarks/h100_dense_composition_training.py \
  --same-coordinate-ssa \
  --native-fixture-directory /path/to/native-fixtures \
  --output /path/to/dense-export.json
```

Here `baseline` uses recursive fused-expression evaluation and `generated`
uses SSA values for shared expressions at the same output coordinate. The
exporter retains source/input identities and comparison results in the output
JSON. Fixture export follows numerical acceptance. Set `fixture_root` to the
exported directory, then run each variant separately:

```bash
fixture_root=/path/to/native-fixtures
SHUTTLE_TEST_COMPOSED_FIXTURE_DIRECTORY="$fixture_root/baseline" \
  "$native_output/gpu_fatbin_h100_test" \
  --gtest_filter=GpuFatbinH100Test.ExecutesExportedComposedPublicBoundary
SHUTTLE_TEST_COMPOSED_FIXTURE_DIRECTORY="$fixture_root/generated" \
  "$native_output/gpu_fatbin_h100_test" \
  --gtest_filter=GpuFatbinH100Test.ExecutesExportedComposedPublicBoundary
```

Each directory contains `manifest.json`, `bundle.bin`, and raw input and
expected-output files. Manifest `inputs` and `expected_outputs` arrays list
filenames in public binding-index order; `bundle` names the transport file,
and `max_absolute_error` supplies the numerical gate. The runner derives shapes,
dtypes, and byte counts from the bundle and checks every public BF16/F32 output,
including scalar loss and gradients. Nonfinite values fail acceptance. The
dense fixture gate is `6.103515625e-05`; expected outputs are default JAX outputs
cast to each declared output dtype.

`GpuRunner` builds its FFI call frame once during creation. Execution validates
the current public buffers and rebinds their addresses with XLA's
`CallFrame::UpdateWithBuffers` under the runner mutex. Bundle bytes, their digest,
attributes, and tensor shapes remain owned by the frame. A failed completion
fence still prevents another invocation until `AwaitCompletion` succeeds.

After numerical acceptance, the harness retains 30 raw host and stream timing
samples after 10 warmup calls. `ExecuteProfiled` separately measures host frame
setup, FFI enqueue, and the remaining completion wait without inserting device
timer events. Enqueue includes both argument preparation and kernel submission.
Its stages exclude time waiting for the runner mutex; the ordinary host samples
cover the full synchronous call. The `ExecuteTimed` stream interval can include
host submission gaps and is not summed kernel time. On the September 17 paired
H100 replay, frame reuse reduces SSA native host median from 5.218 to 0.457 ms,
with unchanged numerical errors. The
[dispatch artifact](../../../../.agents/artifacts/shuttle-native-dispatch-20260917-v0/README.md)
retains the raw measurements and comparison protocol.

For an export discrepancy, add
`--first-forward-fixture-directory /path/to/forward-fixture`. This writes the
first physical component's inputs and canonical outputs and captures its actual
CUDA graph kernel arguments. Used alone, the option stops before timing; used
with `--native-fixture-directory`, it also produces the full benchmark fixtures.
The isolated fixture lives in the `forward` subdirectory and uses Python's
intermediate component outputs as its reference, rather than full-program JAX
outputs. Run it through the same native fixture test.

The captured argument report compares parameter contents and launch dimensions.
Public pointer bindings are checked; unknown temporary addresses are inferred
consistently and do not prove storage identity. Raw differences can include
uninitialized padding or opaque descriptor bytes, so inspect the affected fields
before treating a byte mismatch as a numerical cause. Parameter count and width
checks alone cannot detect a swapped TMA dimension or an incorrect shape field.
See the [native export incident](https://marina.oa.dev/echo/wiki/344).

The successful CPU build is recorded in
[the source and toolchain manifest](../../../../.agents/artifacts/shuttle-native-runtime-20260905-v0/native-r3-build-result.json)
and [Iris job shuttle-native-multiimage-build-0905-r3](https://iris.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-multiimage-build-0905-r3).
The manifest's executable SHA-256 is
`bf5bfb9f0d958beebe3089b5de268b742c876411a2dbe3edaf4ec8f2383a9b3c`.

The [CPU build incident](https://marina.oa.dev/echo/wiki/345) records the
sysroot and driver-packaging failures that led to this procedure.
