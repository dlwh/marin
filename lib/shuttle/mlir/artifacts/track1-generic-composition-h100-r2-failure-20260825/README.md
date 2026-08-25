# Ordinary-JAX generic composition H100 r2 qualification failure

The diagnostic r2 canary executed all six rank-2/rank-3 and
disabled/SOURCE_ORDERED/FAST coordinates before applying the unchanged v1
numerical thresholds. Four enabled Shuttle coordinates passed. The
`rank3-b/disabled` ordinary-XLA coordinate exceeded only the maximum and mean
absolute-error limits, so the campaign terminated with one failure and no
preemption. This is a qualification failure of the v1 campaign policy, not a
generic-composition compiler failure.

## Identity

- Job: `/dlwh/shuttle-track1-generic-composition-h100-canary-r2`
- Cluster: `cw-us-east-02a`
- Executed source commit: `d56ed83856b99bed6e4da79fa42e5c2f42ef263f`
- Executed source tree: `ffb7ade45a3680fdaee988c2890afb2c629f1a60`
- Source manifest SHA-256: `cf8e39a495b76ea350cb3a50cdaf305d6ee381681c51a84537b572e5ea7b5dd6`
- Authorization SHA-256: `4dcecb73e706c7c36c318809c01edb42d696cfbd2374ea97b256d9bc6fa92af2`
- Numerical policy SHA-256: `202413163a8c78067b88043db7207eafdeb5b58116341e186398572280466a21`
- Request SHA-256: `1ae4f0034dba969126c8a5d187dc9dceeef8f9b07ed38363ea15569f49aca454`
- Request size: 20,672,154 bytes
- Bundle SHA-256: `ad72fbada96f1534350a9116e18e3a9033760fba5f8c19cc71a825a0b5b2a2fe`
- Bundle size: 20,671,143 bytes

The reviewed request protobuf remains external at
`/private/tmp/shuttle-generic-composition-r2-final.LoPX3F/launch.pb`. The
artifact identifies those bytes by SHA-256 and size; it does not claim that
the temporary path is durable.

## Result

Iris recorded one failed task, zero preemptions, and 983.18 seconds runtime.
All six canonical diagnostic records were fsynced before aggregate threshold
validation. Every coordinate passed the f32 public ABI, exact shape,
finiteness, and bitwise-repeat checks. The `full_write` record field is a
harness assertion derived from the sealed nonzero oracle and whole-output
comparison, not an independent device-write measurement. Disabled coordinates
had no observer events. Enabled observer invocation IDs were 1 through 4. Each record
retained the policy-declared `contract_fold` and `map_contract` semantic groups;
those record fields are harness assertions, not an independent device topology
measurement. No launch-count, temporary-count, or wheel identity survived in
the r2 result.

`rank3-b/disabled` measured maximum absolute error
`0.0006204024596327962` against limit `0.0005` and mean absolute error
`0.00011096225522005224` against limit `0.00005`. Its relative Linf error
`0.00021940363857290523` remained below `0.0005`; raw and masked BF16 ULP
distance were both 1. All enabled SOURCE_ORDERED and FAST cells were within
the three v1 limits and had raw/masked BF16 ULP distance zero.

The v1 diagnostic field `reference_binary64` is platform-specific because the
independent oracle uses NumPy matrix multiplication backed by the host BLAS.
Linux and macOS can differ in low binary64 bits. The prospectively sealed
rounded-f32 oracle digests matched the v1 policy on both platforms. The exact
Linux records are preserved without rewriting this field.

r2 also supersedes the r1 artifact's coordinate inference. The r1 exception did
not include a case or mode; fail-fast ordering could not distinguish a failure
in the first coordinate from a later failure after earlier coordinates passed.
r1 therefore remains unlocalized. r2 independently localizes its own failure to
`rank3-b/disabled`.

Verify the preserved files with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-generic-composition-h100-r2-failure-20260825 && shasum -a 256 -c SHA256SUMS)
```
