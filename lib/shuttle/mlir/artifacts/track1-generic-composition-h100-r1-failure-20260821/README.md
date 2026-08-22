# Ordinary-JAX generic composition H100 r1 failure

The first anonymous `Contract -> Fold / Map -> Contract` plugin canary reached
the H100 subject and failed its prospectively sealed numerical gate. The worker
reported `generic composition result exceeds maximum_absolute_error`; the r1
reporting path did not retain the measured error or output values. The sealed
source order localizes the failure to the first `rank2-a/disabled` coordinate.
No enabled Shuttle generic-composition coordinate ran, so r1 does not identify
a generated composition codegen failure.

## Identity

- Job: `/dlwh/shuttle-track1-generic-composition-h100-canary-r1`
- Cluster: `cw-us-east-02a`
- Reviewed source commit: `977e678edb2c9fccb4b1c00f3b38b4fec1d9946e`
- Reviewed source tree: `221e99f0943eba6ebd7cbd42ba1658a428b53e4a`
- Source manifest SHA-256: `59fa929094d9f9b8563d2dd613d400ded559be0f0e0d3e137e3f170d0f5c8ca0`
- Authorization SHA-256: `02375f1148b537b7672636f9bafe261167a6c215cb8d1252e8aed228535cec2c`
- Numerical policy SHA-256: `202413163a8c78067b88043db7207eafdeb5b58116341e186398572280466a21`
- Request SHA-256: `d5c2dab82ffd2e3b28859c0572b0b0c510c50219f61907f5bf53712a4059ec90`
- Request size: 20,464,855 bytes
- Bundle SHA-256: `f65cd22ddc26fd9c12d682ef11968504403dee336286c900508cb6a4444a3e2b`
- Bundle size: 20,463,844 bytes

The reviewed request protobuf remains external at
`/private/tmp/shuttle-generic-composition-r1-final.ZCFPQ7/launch.pb`. Its hash
and size are sufficient to identify those bytes; the artifact does not claim
that this temporary path is durable.

## Result

Iris recorded one failed task, zero preemptions, and a 15 minute 40.43 second
task duration. The task reached `validate_coordinate` in the generic
composition worker and rejected `maximum_absolute_error`. The prior reporting
path wrote its canonical result only after all six coordinates passed, so no
coordinate result, output digest, measured error, or wheel identity survived the
failure boundary. Before the threshold check, both disabled-JAX executions had
already passed the f32 public ABI, shape, finiteness, bitwise-repeatability, and
empty-observer checks.

The source contract covered rank-2 and rank-3 cases in disabled,
SOURCE_ORDERED, and FAST modes. The nested source order is case then mode, and
validation aborts immediately, which identifies `rank2-a/disabled` as the
failure. It does not accept numerical correctness, establish an enabled
topology result, or justify a compiler change. A diagnostic-only successor must
emit one bounded canonical coordinate record before applying the unchanged
threshold.

Verify the preserved files with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-generic-composition-h100-r1-failure-20260821 && shasum -a 256 -c SHA256SUMS)
```
