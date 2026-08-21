# Track 2 TE RMSNorm r10 failure

Track 2 r10 rejected the predeclared `heldout_splitmix64_v1` RMSNorm policy.
Both Transformer Engine 2.17 and ordinary JAX/XLA produced a whole-output
maximum absolute error of `0.015625` for `y`, above the frozen `0.0078125`
limit, in forward and composed execution for both repeats. The one-task H100
job failed with exit 1, one task failure, and zero preemptions. It is
qualification evidence, not acceptance evidence.

## Preserved evidence

[`r10-raw-policy.json`](r10-raw-policy.json) contains all 36 raw policy cells
printed and flushed before validation: two providers, three input patterns,
three boundaries, and two repeats. Each cell includes input and reference
digests, call sequence, output digests, finite status, error metrics, eligible
mask cardinality, timing samples, and provider identity.

[`r10-failure-summary.json`](r10-failure-summary.json) records the reviewed
request, source, job, and failure identities. The first failing cell retained
`max_absolute_error=0.015625`, `mean_absolute_error=3.4313416108489037e-08`,
masked relative error `0.007751937955617905`, and masked/raw BF16 distance of
one ULP. Eight cells violated the same `y` absolute limit.

Ordinary JAX/XLA composed `y` and `dx` output digests differed between repeats
despite identical input/reference digests and call provenance. The raw report
did not persist XLA output arrays, so exact mismatch counts, pairwise repeat
ULP, and pairwise repeat maximum absolute error are unrecoverable. Exact
repeatability remains a required gate; r10 does not justify weakening it.

[`r10-submission-receipt.json`](r10-submission-receipt.json) preserves the
create-only submission receipt. [`SHA256SUMS`](SHA256SUMS) seals the tracked
payloads.

## Decision

Do not widen the v2 limits or rerun this SplitMix coordinate. r7, r9, and r10
are qualification evidence only. A future policy must protect the
below-floor tail separately from above-floor ULP geometry, predeclare a fourth
untouched deterministic input, and pass independent review before hardware.

The complete 6.7 MB request remains at
`/private/tmp/shuttle-track2-te-request-r10.HsEbOp/request.pb`; it is not copied
into Git because it embeds the complete source bundle. Its exact size and
SHA-256 are recorded in the failure summary and receipt.
