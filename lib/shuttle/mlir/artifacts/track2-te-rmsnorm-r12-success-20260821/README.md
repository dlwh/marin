# Track 2 TE RMSNorm r12 acceptance

Track 2 r12 accepted the reviewed v4 RMSNorm policy for the predeclared
`heldout_splitmix64_v3` input at shape `[2048,4096]` and epsilon `1e-6`.
Transformer Engine 2.17 and ordinary JAX/XLA completed forward, backward, and
composed execution twice on one H100. All 12 provider, boundary, and role
repeat coordinates were bit-identical, and all 24 output observations passed
the frozen finite, partition, absolute, mean, relative, and masked-BF16-ULP
gates.

The sealed Iris job `/dlwh/shuttle-track2-te-h100-r12` completed one task with
exit 0, no failures, and no preemptions. The request used batch priority, one
H100, one replica, and zero failure, preemption, and task-failure budgets. Its
exact SHA-256 is
`4c3cb77f327f3a632009c313bf7c51f7201a15dc67bbb06644c054b4a3d9b9ac`.

## Preserved evidence

[`request.pb`](request.pb) is the exact reviewed launch request, including its
source bundle. [`receipt.json`](receipt.json),
[`job-summary.txt`](job-summary.txt), and [`task.log.gz`](task.log.gz) preserve
the submission, terminal state, and complete canonical worker records. The
compressed log expands to the exact worker log with SHA-256
`251b2f0ca3f877114e781a60f8214734990627b9239d6b4bb0786c6a0c6e0825`.

[`base-reports-v4.json`](base-reports-v4.json) contains two provider reports
with twelve base cells. [`policy-raw-v4.json`](policy-raw-v4.json) contains all
60 policy cells across two providers, five patterns, three boundaries, and two
repeats. [`repeat-diagnostics-v4.json`](repeat-diagnostics-v4.json),
[`policy-diagnostics-v4.json`](policy-diagnostics-v4.json), and
[`numerical-diagnostics.json`](numerical-diagnostics.json) preserve the repeat
and numerical audits. [`result-v4.json`](result-v4.json) is the canonical
accepted result. [`SHA256SUMS`](SHA256SUMS) seals every payload.

The ephemeral BF16 output files were not copied from the worker. The complete
derived reports and their cross-document identities are reproducible from the
preserved records, but the pointwise numerical metrics cannot be recomputed
from raw output bytes. This limits the artifact claim to the reviewed v4
validator and preserved report identities; it is not a raw-array archive.

## Result boundary

The accepted held-out observations had the following worst values across both
providers and repeats:

- `y`: above-floor maximum absolute error `0.015625`, mean absolute error
  `3.4458935260772705e-8`, masked relative error `0.007692307699471712`, and
  masked BF16 distance 1 ULP.
- `dx`: below-floor maximum absolute error `0.00048828125`, above-floor maximum
  absolute error `0.015625`, mean absolute error
  `4.787287366525561e-8`, masked relative error `0.0078125`, and masked BF16
  distance 1 ULP.
- `dgamma`: below-floor maximum absolute error `0.0001220703125`, mean absolute
  error `2.9802322387695312e-8`, and zero above-floor relative or ULP error.

Two ordinary-JAX/XLA composed-`dx` repeat drifts occurred on old qualification
patterns. The v4 contract reports those coordinates without allowing them to
reject or promote acceptance. The host-blocking performance ratios use the
qualification linspace only and have status `reported_without_threshold`.

This evidence does not include Shuttle at `[2048,4096]`; that representative
producer remains pending. Transformer Engine norm-to-Contract is unavailable,
and no norm-to-Contract claim is made.
