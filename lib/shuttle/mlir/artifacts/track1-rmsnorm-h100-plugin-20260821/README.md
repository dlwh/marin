# Ordinary-JAX RMSNorm H100 plugin evidence

This directory preserves the successful `r3` canary for the generic ABI22
RMSNorm Fold-region path through the ordinary JAX/XLA plugin. The job ran one
batch-priority H100 in `cw-rno2a` and completed all 18 coordinates:

- rank 3 `[2,7,19]` and rank 5 `[3,2,5,11,263]`;
- forward, backward, and composed public boundaries;
- disabled JAX, SOURCE_ORDERED, and FAST modes.

The enabled coordinates were accepted against the predeclared ABI22 emitted
tree and independent binary64 per-role thresholds. The policy binds the actual
ABI22 positive-zero lane initialization, per-lane adjacent-pair accumulation,
shared adjacent-pair tree, final Fold initializer add, and `sqrt.rn.f32` then
`div.rn.f32` reciprocal-square-root realization. It does not import ABI21's
schedule-exact identity.

Disabled JAX is retained as a diagnostic reference. Rank-5 backward and
composed `input_cotangent` outputs differ bitwise from disabled JAX in both
enabled modes, while matching the predeclared emitted-tree digest and remaining
inside every independent threshold. This evidence therefore makes no
bitwise-JAX-equivalence claim.

## Identity

- Job: `/dlwh/shuttle-track1-rmsnorm-h100-canary-r3`
- Cluster: `cw-rno2a`
- Reviewed run source: `97b1fe38ce2c2f0f75868f6f96280d97f8667425`
- Source manifest SHA-256: `39339a33778c67d5aca853cc5de11efcc0950304ed43ad1757e4cffd9b09138a`
- Authorization SHA-256: `67d4352b187e8d149bc49f3ffe6701815c4327ed8cd9712c64c360dc1d054e30`
- Numerical policy SHA-256: `4f3b354d8908101072c35b94fc308d500a64ebee22b9b88f475ac3137cdf7df9`
- Request SHA-256: `1473f85738b6765559adf4158505eb3a3c5a3c55f13aa4c7df056703291e0eef`
- Request size: 6,373,340 bytes
- Bundle SHA-256: `44dab7b08a79c0314c78dd9cae62c3ee66cb1504af1cf48ea87986e16c5d81f4`
- Bundle size: 6,372,341 bytes
- Canonical result SHA-256: `4bd11ab0c6c89faa0a4ed2c2bf0063734f5cb56fcfadadf9ecd2c9e2ce0f38cb`
- Canonical result size: 38,152 bytes, including its final newline

The request protobuf remains external because it is 6.37 MB. Its digest and
size are preserved in `request.json` and `receipt.json`. The source and policy
commits are replayed on current integration base
`5ddfeb12b568ea42499ec457f508e94ba2843315` in this branch, so the canary source
is reachable without embedding a generated source patch in the artifact.

## Result

Iris reports one successful task, exit code zero, no failure, no preemption,
and a duration of 15 minutes 12.34 seconds. All enabled coordinates repeated
bitwise within the same process. Observer invocation IDs are exactly 1 through
12, with launch/temporary topology 1/0 for forward, 3/1 for backward, and 4/1
for composed.

The worst observed metrics across all coordinates were:

| Metric | Observed | Applicable limit |
| --- | ---: | ---: |
| Maximum BF16 ULP error | 1 | 1 |
| Maximum absolute error | 0.02949355642030227 | 0.03125 for weight cotangent |
| Mean absolute error | 0.006515962406162605 | 0.015625 for weight cotangent |
| Relative Linf error | 0.003138256088840048 | 0.00390625 |

`result.json` is the canonical JSON emitted by the reviewed task and validates
through both `validate_result_pair` and `validate_rmsnorm_result`. Check the
preserved files with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-rmsnorm-h100-plugin-20260821 && shasum -a 256 -c SHA256SUMS)
```

This is one-device functional evidence for the listed shapes and boundaries.
It is not a representative-shape sweep, a performance result, TE parity, or a
claim about other Fold schedules or GPU architectures.
