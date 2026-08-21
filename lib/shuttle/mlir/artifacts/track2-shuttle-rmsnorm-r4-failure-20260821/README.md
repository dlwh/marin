# Track 2 Shuttle representative r4 runfiles failure

The east migration allocated immediately and completed the pinned device,
source, XLA, Bazel, and Shuttle build phases. The first SOURCE_ORDERED forward
coordinate then failed before source compilation or GPU launch because the
nested gtest had neither `RUNFILES_DIR` nor `TEST_SRCDIR`. The r2 bridge
required an existing `RUNFILES_DIR`, so it could not establish the fixture
lookup root.

The sealed Iris job `/dlwh/shuttle-track2-rmsnorm-h100-r4` ran on
`cw-us-east-02a` from source commit
`0a212699e8bf080a2c6f5b6f508294435c3d792e`, tree
`cf71f6366eddc99c5b760db5d2d1a36227a60e81`, and reviewed request SHA-256
`77eea45a62132b4452b11800fb7feb497749c51345b112eba31b88da272a6a26`.
The request used one H100, batch priority, one replica, fail-if-exists, and
zero failure, preemption, and task-failure budgets.

Iris reports task exit 1, one failure, zero preemptions, and duration 218.69
seconds. The filtered gtest itself failed in 0 ms at `fixtureSource`, before a
source module was compiled or a kernel launched. This run therefore contains
no compiler, numerical, resource, or performance evidence.

This directory preserves the submission receipt, terminal summary, and exact
controller task lines in deterministic `gzip -n` form. The archive SHA-256 is
`2fbe297e64eafddd5c32f179eb7861f70acfb49ee1c22a4de22d2538f20aecc8`;
decompression recovers bytes with SHA-256
`802402e64b874d5adf68b42d53591e052de310d6d71bff38e559d9963826b4f8`.
The 19,274,497-byte request remains locally at
`/private/tmp/shuttle-representative-request-east-r4.5UCrMF/request.pb`; it is
not recursively embedded in the next source bundle. No resubmission was
performed.
