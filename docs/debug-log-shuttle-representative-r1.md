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
