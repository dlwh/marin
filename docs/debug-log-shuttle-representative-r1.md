# Debugging log for Shuttle representative RMSNorm r1

Run the sealed `[2048,4096]` RMSNorm F/B/C matrix through the standalone
GpuRunner on one H100 and preserve exact numerical and timing evidence.

## Initial status

The reviewed r1 request reached `shuttle_execute` after a successful hermetic
build, then failed before the first kernel launch. The inner shell wrapper
reported `unknown argument: /tmp/shuttle-track2-h100/cases`. Iris recorded task
exit 1, zero preemptions, and no retry.

## Hypothesis 1

The matrix launches the `run_h100_shuttle_rmsnorm_representative` shell target
as a subprocess. Bazel supplies that target's default GPU-test argument only
when the target itself is run by Bazel; the default argument is not embedded in
the executable file referenced from another target's runfiles. The Python plan
therefore invoked the shell wrapper with `--case-root` as its first argument.
The wrapper consumed that flag as the GPU-test executable, shifted it away, and
then rejected the bare case-directory value.

## Changes to make

Pass the GPU transport test path explicitly in every per-coordinate runner argv,
including the runfiles-closed matrix boundary. Add a behavior regression that
executes the real shell wrapper with a fake GPU-test process and observes the
forwarded coordinate through its environment.

## Results

The new behavior test failed before production accepted a
`gpu_transport_test` argument. After the repair, it executed the real shell
wrapper and observed the exact case root, raw root, policy, boundary, repeat,
and gtest arguments in the fake GPU-test process. The complete six-test Python
contract, the Bazel test target, the matrix sh_binary build, and a Bazel-run
runfiles/help probe passed. No numerical policy, compiler, PTX, resource, or
timing behavior changed.

## Future work

- [ ] Seal the r2 terminal evidence after the reviewed run.

## Hypothesis 2

The r2 argv repair reached the H100 gtest, but `fixtureSource` returned no
StableHLO source at the first coordinate. The helper uses
`Runfiles::CreateForTest`, which reads `RUNFILES_MANIFEST_FILE` or
`TEST_SRCDIR`. A nested `sh_binary` launched by `bazel run` supplies
`RUNFILES_DIR`; it does not supply the test-only directory variable. The
referenced binary and fixture are present in the matrix runfiles, but the C++
test helper cannot discover that tree.

## Changes to make

At the coordinate shell boundary, derive `TEST_SRCDIR` from `RUNFILES_DIR` only
when the test variable is absent. Add a subprocess regression whose fake GPU
test resolves the representative StableHLO fixture through `TEST_SRCDIR`.

## Results

The local regression reproduced the missing fixture before the repair and
passes after the shell exports the validated `RUNFILES_DIR` as `TEST_SRCDIR`.
The focused Python behavior/request suite passes 9/9. The pinned Bazel 7.7.0
build compiles the matrix target, and a direct `bazel run` resolves the module,
coordinate wrapper, GPU test, and sealed r12 report paths from runfiles before
printing the representative producer help. No compiler admission, schedule,
or numerical tolerance changed.
