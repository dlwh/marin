---
topic: shuttle-track2-te-rmsnorm
description: Transformer Engine 2.17 and ordinary-JAX/XLA RMSNorm numerical qualification
author: dlwh
---

# Shuttle Track 2 TE RMSNorm: task logbook

## Scope

- Goal: qualify a versioned Transformer Engine 2.17 and ordinary-JAX/XLA
  RMSNorm F/B/C oracle at epsilon `1e-6` and shape `[2048,4096]` before using it
  as a Shuttle comparison lane.
- Primary metrics: whole-output maximum absolute error, interpretable BF16 ULP
  distance, exact per-coordinate repeatability, and matched host-blocking time.
- Constraints: r7 and r9 are qualification evidence only; no post-hoc promotion,
  no Shuttle or norm-to-Contract claim, and no new H100 execution before a third
  deterministic input and metric policy are reviewed and frozen.

## Current TL;DR

- r7 measured the original linear input. Representative TE `dx` reached
  `0.03125` maximum absolute error and 35 raw BF16 ULP; ordinary JAX/XLA reached
  `0.0078125` and 24 ULP.
- r9 rejected the predeclared reversed input: TE backward-recompute `dx` reached
  283 raw BF16 ULP against a limit of 64. The whole tensor first passed the
  `0.0625` maximum-absolute-error gate.
- r9 did not preserve raw outputs. Exact index, values, mean error, and relative
  error are unrecoverable. Enumeration bounds the offending pair to same-sign
  magnitudes no larger than `0.07861328125`; this does not prove that the
  relative error was small.
- A reviewed v2 proposal keeps whole-output absolute error and applies ULP only
  where `abs(reference) >= 2 * atol`. The untouched
  `heldout_splitmix64_v1` bytes and digests were frozen before r10.
- r10 rejected that SplitMix input: both providers produced `y` maximum
  absolute error `0.015625` against the frozen `0.0078125` limit. Its complete
  36-cell raw policy report is now durable qualification evidence.
- r11 rejected the v3 run on exact repeatability. Ordinary-JAX/XLA composed
  `dx` on the qualification linspace changed at three BF16 elements between
  repeats. The untouched v3 acceptance pattern was repeatable and within its
  numerical limits, but the required global qualification gate failed.

## Negative results index

- `T2-TE-002`: r9 rejected the reversed held-out input and exposed missing raw
  artifact durability. Evidence:
  [`track2-te-rmsnorm-r9-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r9-failure-20260821/README.md).
- `T2-TE-004`: r10 rejected the SplitMix v2 acceptance input and exposed
  ordinary-JAX composed repeat nondeterminism in three role coordinates.
  Evidence:
  [`track2-te-rmsnorm-r10-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r10-failure-20260821/README.md).
- `T2-TE-006`: r11 rejected the v3 matrix on one qualification repeat
  coordinate. Evidence:
  [`track2-te-rmsnorm-r11-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r11-failure-20260821/README.md).

## Entry log

### 2026-08-21 07:43 - T2-TE-001 r7 qualification metrics

- Hypothesis: role-specific power-of-two absolute and raw-ULP limits derived
  from the original representative input would generalize to an untouched
  deterministic input.
- Commit Hash: source `3e9e330461d149377ac93d05525ec1dc52c2e5e1`, tree
  `62349ffad557e361a83b3177bf002db38d96297e`.
- Command: sealed Iris job `/dlwh/shuttle-track2-te-h100-r7`, request SHA-256
  `15a16a2584e7550ba9b9161d9fcd4fb731def31ca986c2a712d05ca65292cad5`.
- Config: one H100, batch priority, one replica, zero failure/preemption/task
  retry budgets; Transformer Engine 2.17 and ordinary JAX/XLA; RMSNorm
  `[2048,4096]`; epsilon `1e-6`; F/B/C.
- Result: representative TE metrics were `y=(0.00390625,1 ULP)`,
  `dx=(0.03125,35 ULP)`, and `dgamma=(0,0)`; ordinary JAX/XLA metrics were
  `y=(0.00390625,1 ULP)`, `dx=(0.0078125,24 ULP)`, and `dgamma=(0,0)`.
  Architecture `[7,13]` outputs were exact. r7 was retained as qualification
  evidence and did not produce acceptance.
- Interpretation: the data supported a candidate role policy, but it did not
  test a second input distribution.
- Next action: predeclare and execute the reversed linear input with two repeats.

### 2026-08-21 08:15 - T2-TE-002 r9 held-out rejection

- Hypothesis: the r7 role limits would hold for
  `heldout_reversed_linspace_v1`.
- Commit Hash: source `c2fd5aeaa54e07cd78dc6433b83f364fac428c58`, tree
  `92ae0f808ab73d06a562277ca3e8b2bdcc6f1c9f`.
- Command: sealed Iris job `/dlwh/shuttle-track2-te-h100-r9`, request SHA-256
  `92b56d65ab43d12e9b5d2b97170805d825eb0183cbbc3d33fd2f2efcc95e3766`.
- Config: one H100, batch priority, one replica, zero failure/preemption/task
  retry budgets; exact r7 providers, shape, epsilon, and boundaries; two repeats
  per provider and input pattern.
- Result: hypothesis rejected. TE backward-recompute `dx`, repeat 0, measured
  283 raw BF16 ULP against the frozen limit of 64. The task failed with exit 1,
  one task failure, and zero preemptions. Because maximum absolute error is
  validated first, the whole `dx` tensor was within `0.0625` absolute error.
- Interpretation: raw ULP is unstable in the low-magnitude tail. The batch also
  had an artifact bug: it wrote raw arrays and reports only to ephemeral worker
  storage and raised before printing or uploading them. The exact offending
  index, values, maximum absolute error, mean error, and relative error cannot
  be recovered. Compatible-pair enumeration rules out a sign crossing and
  bounds both magnitudes by `0.07861328125`; the 24.5014% compatible-reference
  fraction is a feasible-set count, not observed prevalence.
- Next action: do not rerun or widen the old policy. Preserve the failure and
  freeze a scale-aware v2 proposal plus a third untouched input.
- Artifact:
  [`track2-te-rmsnorm-r9-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r9-failure-20260821/README.md).

### 2026-08-21 08:30 - T2-TE-003 v2 policy and third input predeclared

- Hypothesis: whole-output absolute error plus ULP above a role-specific
  absolute floor distinguishes substantive BF16 drift from exponent-sensitive
  low-magnitude drift.
- Commit Hash: audited integration base
  `3b1108c191d9327d423fb84c103352295d0611fe`; this entry's containing revision
  records the proposal and reproducibility script.
- Command: `uv run python
  lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r9-failure-20260821/reproduce_analysis.py`.
- Config: floor `2 * atol` by role: `y=0.015625`, `dx=0.125`, and
  `dgamma=0.03125`; masked ULP limits remain `2`, `64`, and `8`. The proposed
  acceptance input is `heldout_splitmix64_v1`, generated from fixed uint64
  counter-hash constants and seeds at `[2048,4096]`.
- Result: design review approved the metric and claim boundary. Passing the
  whole-output absolute gate guarantees same-sign actual/reference values and
  at most 50% relative absolute error in the selected region. The six SplitMix
  input/reference digests are frozen in `proposed-role-policy-v2.json`; the
  pattern has not run on hardware.
- Interpretation: r7 and r9 remain diagnostic and repeatability inputs. Only
  the untouched SplitMix pattern may support a future v2 acceptance result.
- Next action: implement the v2 schema red-first, persist raw outputs before
  validation, review the exact request, then run one fresh batch-priority H100
  job. No hardware run is authorized by this entry.

### 2026-08-21 09:14 - T2-TE-004 r10 SplitMix rejection

- Hypothesis: the reviewed four-metric v2 policy would accept the untouched
  `heldout_splitmix64_v1` input while preserving exact per-coordinate repeat
  digests.
- Commit Hash: source `721e1edca6d205c92df6101f542b93972525b4ac`, tree
  `ebf6d299fdad77eef235bcbb5fbac53e4a2adf4c`.
- Command: sealed Iris job `/dlwh/shuttle-track2-te-h100-r10`, request SHA-256
  `c967d0905613c2527b71f0dffaeae727688ff9b9c3449b80f326ec71f62faa75`.
- Config: one H100, batch priority, one replica, zero failure/preemption/task
  retry budgets; exact TE 2.17 and ordinary-JAX/XLA providers; RMSNorm
  `[2048,4096]`; epsilon `1e-6`; F/B/C; two repeats over three patterns.
- Result: hypothesis rejected. Both providers measured `y`
  `max_absolute_error=0.015625` against the frozen `0.0078125` limit in
  forward and composed execution for both repeats. The task failed with exit
  1, one task failure, and zero preemptions. All 36 raw cells were printed and
  recovered.
- Interpretation: above-floor ULP geometry cannot replace below-floor absolute
  protection. Ordinary-JAX composed output digests also differed across
  repeats for qualification-linspace `dx` and SplitMix `y` and `dx`, with
  identical inputs, references, and call provenance. Element mismatch counts
  and pairwise repeat errors are unrecoverable because XLA arrays were not
  persisted. Repeatability remains a required gate.
- Next action: retain r7, r9, and r10 as qualification-only evidence. Review a
  scale-consistent v3 policy with a distinct low-tail gate and freeze a fourth
  untouched input before any hardware execution.
- Artifact:
  [`track2-te-rmsnorm-r10-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r10-failure-20260821/README.md).

### 2026-08-21 10:19 - T2-TE-006 r11 v3 repeatability rejection

- Hypothesis: the reviewed piecewise v3 policy and untouched
  `heldout_splitmix64_v2` input would pass all numerical and exact-repeat gates.
- Commit Hash: source `e887fd2a726bb8ea46eaf93474d41485495d96af`, tree
  `0ca4071807bf9ae607dfae8fda49e622691ddb4f`.
- Command: sealed Iris job `/dlwh/shuttle-track2-te-h100-r11`, request SHA-256
  `b616757fb0e3d4102f6b788d874341c4a8a09cf9f834c15afe650accc733c658`.
- Config: one H100, batch priority, one replica, zero failure/preemption/task
  retry budgets; exact TE 2.17 and ordinary-JAX/XLA providers; RMSNorm
  `[2048,4096]`; epsilon `1e-6`; F/B/C; two repeats over four patterns.
- Result: hypothesis rejected. Ordinary-JAX/XLA composed `dx` on
  `qualification_linspace_v2` differed at three BF16 elements. The largest
  repeat difference was one ULP (`2.384185791015625e-7`) at flat index
  `4380678`. The task failed with exit 1, one task failure, and zero
  preemptions. Complete raw policy and repeat diagnostics were recovered from
  the pre-validation log records.
- Interpretation: the two offending cells have identical input/reference
  provenance and call sequence, so the failure is output nondeterminism rather
  than input drift. All 12 repeat coordinates for the untouched SplitMix v2
  acceptance pattern matched exactly and its individual numerical metrics were
  within the frozen role limits. The global repeat requirement still rejects
  r11, so neither numerical acceptance nor performance is promoted.
- Next action: preserve r11 as qualification evidence. Do not widen the policy
  or submit a replacement run without a separately reviewed change to the
  repeatability hypothesis.
- Artifact:
  [`track2-te-rmsnorm-r11-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r11-failure-20260821/README.md).

### 2026-08-21 12:03 - T2-TE-007 r12 v4 held-out acceptance

- Hypothesis: after separating qualification diagnostics from acceptance, the
  reviewed piecewise v4 policy and untouched `heldout_splitmix64_v3` input
  would pass numerical and exact-repeat gates for Transformer Engine 2.17 and
  ordinary JAX/XLA RMSNorm.
- Commit Hash: source `597850f2ff5e4f613d8d2f5e7ec13d4e897f5488`, tree
  `d55449389e8ba53f22326b407287ecdc555007a8`.
- Command: sealed Iris job `/dlwh/shuttle-track2-te-h100-r12`, request SHA-256
  `4c3cb77f327f3a632009c313bf7c51f7201a15dc67bbb06644c054b4a3d9b9ac`.
- Config: one H100, batch priority, one replica, zero failure, preemption, and
  task-failure budgets; RMSNorm `[2048,4096]`; epsilon `1e-6`; F/B/C; two
  repeats over five patterns. Only `heldout_splitmix64_v3` was acceptance
  evidence; the prior four patterns remained diagnostic.
- Result: hypothesis accepted for the exact v4 scope. All 12 acceptance repeat
  coordinates were bit-identical. Every one of the 24 output observations
  passed the frozen finite, partition, absolute, mean, relative, and masked-ULP
  gates. The task completed with exit 0, no failures, and no preemptions.
- Interpretation: the result qualifies the exact TE 2.17 and ordinary-JAX/XLA
  providers, input, H100, shape, epsilon, boundaries, and v4 hybrid numerical
  guarantee. Two ordinary-JAX/XLA composed-`dx` repeat drifts on old patterns
  remain reported qualification diagnostics. Performance is host-blocking
  qualification data without a threshold. Raw BF16 worker files were not
  retained, so pointwise metrics cannot be independently recomputed from this
  archive.
- Next action: integrate the accepted v4 lineage and evidence, then freeze and
  review a Shuttle producer for the same held-out input before any new H100 run.
  No norm-to-Contract claim is in scope.
- Artifact:
  [`track2-te-rmsnorm-r12-success-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r12-success-20260821/README.md).
