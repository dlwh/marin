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
  `heldout_splitmix64_v1` bytes and digests are frozen but unexecuted.

## Negative results index

- `T2-TE-002`: r9 rejected the reversed held-out input and exposed missing raw
  artifact durability. Evidence:
  [`track2-te-rmsnorm-r9-failure-20260821`](../../lib/shuttle/mlir/artifacts/track2-te-rmsnorm-r9-failure-20260821/README.md).

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
