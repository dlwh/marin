# Track 2 Shuttle RMSNorm r2 runfiles failure

Track 2 Shuttle representative r2 reached the filtered H100 gtest and failed
before source compilation. `fixtureSource` returned no source for the first
SOURCE_ORDERED forward coordinate because the nested `bazel run` environment
provided `RUNFILES_DIR`, while the test-only runfiles helper reads
`TEST_SRCDIR` or a manifest.

The sealed Iris job `/dlwh/shuttle-track2-rmsnorm-h100-r2` ran on
`cw-rno2a` from source commit
`c166a957628950f9fc55f4d76ef63e67aee30e3f`, tree
`163908c0a3e30695d7489b1ed9b3517253ab6afe`, and reviewed request SHA-256
`e499cd37b943f6cdb7fbbefda0de7f11d780975d02fdfce85681eddf3da34b24`.
The request used one H100, batch priority, one replica, fail-if-exists, and
zero failure, preemption, and task-failure budgets. Iris recorded task exit 1,
one task failure, and zero preemptions.

This artifact contains the submission receipt, terminal job records, and exact
compressed controller log. The 19,258,667-byte request is retained locally at
`/private/tmp/shuttle-representative-request-r2.TaLnkT/request.pb`; it is not
recursively embedded in the next source bundle. The request is reproducible
from the exact source commit with `render-representative --run 2`, and its
identity, size, bundle SHA, and source tree are sealed in `receipt.json`.

r2 proves the explicit GPU-test argv repair reached the intended gtest. It
provides no compiled module, kernel launch, numerical, resource, or performance
evidence.
