# Track 2 TE RMSNorm r9 failure

Track 2 r9 rejected the predeclared reversed-input RMSNorm policy. Transformer
Engine 2.17 produced a maximum ordered-BF16 distance of 283 ULP for
`backward_recompute` `dx`; the frozen role limit was 64. The one-task H100 job
failed with exit 1, one task failure, and zero preemptions. It is qualification
evidence, not acceptance evidence.

## Recoverable evidence

The r9 validator checked whole-output maximum absolute error before ULP. The
`dx` tensor therefore passed `max_absolute_error <= 0.0625` before failing the
283-ULP check. Exact maximum absolute error, mean error, relative error,
offending index, reference value, and actual value are unknown. The batch wrote
raw outputs and reports to an ephemeral worker workspace, then raised before it
printed or uploaded them. A read-only task-exec attempt after termination
returned `container not found ("task")`.

[`analysis.json`](analysis.json) records the remaining mathematical inference.
Exhaustive enumeration of the deterministic held-out BF16 reference shows that
every pair compatible with both known facts is same-sign and has reference and
actual magnitude at most `0.07861328125`. No compatible pair crosses zero. The
2,055,325 compatible reference positions are 24.5014% of the tensor, but this
is a feasible-set size, not an observed error frequency. The exact r9 pair may
still have a large relative error.

[`r7-qualification-metrics.json`](r7-qualification-metrics.json) preserves the
representative metrics printed by r7. r7 and r9 are qualification inputs only.
Neither run supports an accepted TE/XLA threshold.

## Proposed next policy

[`proposed-role-policy-v2.json`](proposed-role-policy-v2.json) freezes the
reviewed proposal before another hardware run. It keeps maximum absolute error
over every element. Ordered-BF16 ULP is gated only where
`abs(reference) >= 2 * atol`; passing the absolute gate then guarantees the
selected actual value has the same nonzero sign and at most 50% pointwise
relative absolute error. Raw maximum ULP, mean absolute error, and relative
error remain diagnostics.

The proposal also freezes `heldout_splitmix64_v1`, an untouched deterministic
`[2048,4096]` input. Its counter mapping, seeds, ranges, BF16 input digests, and
binary64-derived reference digests were fixed after r9 and before any execution
of that pattern. Only this pattern may produce a future v2 acceptance result.
The prior linear and reversed patterns remain diagnostic and repeatability
coordinates.

The policy is reviewed and predeclared but not implemented. It does not claim
accuracy for other shapes, input distributions, hardware, providers, small
values below the floor, Shuttle, or norm-to-Contract fusion.

## Reproduction

Run from the repository root:

```bash
uv run python \
  lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r9-failure-20260821/reproduce_analysis.py
```

The script regenerates the compatibility envelope and all six SplitMix64 input
and reference digests without using r9 output bytes.

The complete local r9 request, receipt, controller log, terminal summary, and
verified checksums remain under
`/private/tmp/shuttle-track2-te-request-r9.8fZIjF/`. They are intentionally not
copied into Git: the 6.4 MB request contains the complete source bundle, while
the durable files here preserve its reviewed identities and the recoverable
failure.
