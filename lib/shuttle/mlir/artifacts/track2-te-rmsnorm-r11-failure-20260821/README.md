# Track 2 TE RMSNorm r11 qualification failure

Track 2 r11 rejected the reviewed v3 RMSNorm policy because one required
qualification coordinate was not bit-for-bit repeatable. Ordinary JAX/XLA
composed `dx` on `qualification_linspace_v2` differed at three BF16 elements
between its two repeats. The largest pairwise difference was one BF16 ULP, or
`2.384185791015625e-7`, at flat index `4380678`. The task failed with exit 1,
one task failure, and zero preemptions.

This is qualification failure evidence, not numerical acceptance. The v3
contract deliberately requires exact repeat digests for every qualification
and acceptance coordinate. No limit was widened and no replacement run was
submitted.

## Preserved evidence

[`r11-raw-policy-v3.json`](r11-raw-policy-v3.json) contains all 48 raw cells
printed and flushed before validation: two providers, four input patterns,
three boundaries, and two repeats. It includes canonical input/reference
digests, call provenance, per-role numerical metrics, output digests, raw
timing samples, and scopes.

[`r11-repeat-diagnostics-v3.json`](r11-repeat-diagnostics-v3.json) contains
the complete 48-coordinate repeat audit computed from the worker's persisted
BF16 files. It records both digests, mismatch count, first and worst mismatch
indices and values, pairwise maximum absolute error, and pairwise BF16 ULP.
The worker files themselves were ephemeral, but the diagnostics were printed
before validation and recovered from the controller log.

[`r11-failure-summary.json`](r11-failure-summary.json) records the job,
request, source, failure, and evidence identities.
[`controller.log.gz`](controller.log.gz) is the complete compressed 34-line
controller log, including the canonical raw JSON records and traceback.
[`job-summary.txt`](job-summary.txt) and
[`r11-submission-receipt.json`](r11-submission-receipt.json) preserve the
terminal and submission records. [`SHA256SUMS`](SHA256SUMS) seals every
tracked payload.

## Interpretation

The two offending cells have identical input digests, reference digest, shape,
scope, and composed call sequence. Repeat 0 produced digest
`b09429e74cce38c20cc84b56a7d99aa51ded89a0e3d1a942e96eb43390965249`;
repeat 1 produced
`502041f57708e86411c5e2fafe675aba5208a89194cc82b3f8db43e0b5f6ace5`.
At the first and worst differing element, the values were
`5.3882598876953125e-5` and `5.412101745605469e-5`.

The untouched `heldout_splitmix64_v2` pattern is bit-identical across repeats
for all 12 provider/boundary/role coordinates, and every individual metric is
within the frozen v3 role limits. That does not override the failed
qualification repeat gate and does not support an acceptance claim. Timing
samples remain available only as raw qualification measurements; no
performance comparison is promoted.

The complete reviewed request remains at
`/private/tmp/shuttle-track2-te-request-r11.13CRYM/request.pb`; it is not
copied into Git because it embeds the complete source bundle. Its exact size
and SHA-256 are recorded in the failure summary and receipt.
