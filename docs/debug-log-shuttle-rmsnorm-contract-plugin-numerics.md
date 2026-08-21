# Debugging log for Shuttle RMSNorm Contract plugin numerics

Localize the first ordinary-JAX RMSNorm-to-Contract H100 numerical gate failure without changing the compiler or the predeclared numerical policy.

## Initial status

The reviewed batch job `/dlwh/shuttle-track1-rmsnorm-contract-h100-canary-r1` ran source `a686fa10d4f4987d99e7d3cf8f40c24a30a201b8` on one H100 in `cw-rno2a`. It used authorization `09bbb5dc84976bc0b9be0bc0c473fffb1b93f82615e038d6289523a1883e5187` and request `1da2f308e5045e595171c4c2cb1adfdc29e7680faeb6f692d173944dc947a318`. The task failed after 16 minutes and 4.83 seconds with one failed attempt and no preemption:

```text
RuntimeError: RMSNorm Contract input_cotangent exceeds its independent binary64 threshold
```

The failure carried no coordinate, metrics, values, or output identities. No retry was submitted.

## Hypothesis 1: the generic composed Shuttle path exceeds the numerical contract

The failure role exists only on the composed boundary. However, the subject orders each shape as primal disabled/source-ordered/fast followed by composed disabled/source-ordered/fast. Validation checks the composed roles in public order: result, input cotangent, weight cotangent, then projection cotangent. The first cell capable of producing the recorded failure is therefore shape `5x19x7`, composed boundary, disabled mode, input cotangent. That cell has no Shuttle compiler options or observer events. The r1 signal consequently does not implicate either the enabled 42-launch generic VJP path or the primal special region.

## Changes to make

Before applying any acceptance threshold, emit one bounded canonical diagnostic for every role in the current coordinate. Record the coordinate, role, output identities, independent metrics, and first/worst scalar evidence. Flush each record into the worker phase log so a rejected run survives the ephemeral workspace through Iris's bounded sanitized failure tail.

Keep the numerical policy, thresholds, planner, emitter, and runtime unchanged.

## Results

The reviewed diagnostic-only r2 job used source
`7551a7efba485109f92c80bcf029502cecbea094`, authorization
`d89e66d92d2e7fc735574a842ca665e43db92bc1cda3c3446b6b8af223b1c468`,
and request
`c61babe6b7953df9dc236781e4ad152f3d8e418c9dbc06a78f28030d57a2e6f4`.
It failed after 16 minutes and 28.31 seconds with one failed attempt and no
preemption.

The flushed evidence localizes the rejection to shape `[5, 19, 7]`, composed
boundary, disabled mode. The result matched the binary64/BF16 oracle exactly.
The three cotangents passed every maximum absolute, mean absolute, and relative
L-infinity limit, but exceeded the BF16 ULP limit of 8:

| Role | Maximum absolute | Mean absolute | Relative L-infinity | Maximum BF16 ULP |
| --- | ---: | ---: | ---: | ---: |
| input cotangent | 0.0031720557784296055 | 0.0005903688112135287 | 0.0016359067312875996 | 11 |
| weight cotangent | 0.012177731792752944 | 0.002662972541751278 | 0.0038163375227427978 | 11 |
| projection cotangent | 0.0078125 | 0.0015166433233963815 | 0.0026595744680851063 | 131 |

The worst ULP sites were input cotangent `[3, 1]` (`0.0079345703125` versus
binary64 `0.007529460725709383`, rounded BF16 `0.007537841796875`), weight
cotangent `[5]` (`0.1083984375` versus binary64 `0.10307984482951678`, rounded
BF16 `0.10302734375`), and projection cotangent `[10, 1]`
(`-0.00122833251953125` versus binary64 and rounded BF16
`-0.00250244140625`). No sign crossing occurred.

This coordinate runs ordinary disabled JAX before either enabled policy. The
failure therefore occurs before the generic 42-launch Shuttle VJP path, and it
does not exercise the primal special plan. It is a frozen ULP-policy rejection
of the disabled baseline, not evidence of a Shuttle compiler numerical defect.
No threshold or compiler behavior changed, and no successor was submitted.

The complete canonical four-cell record is retained outside the repository at
`/Users/dlwh/src/marin/scratch/20260821-172849_rmsnorm_contract_r2_failure.json`
(SHA-256
`f8118e26165f2f701b8bdc2154e2ec3c277d393b2a89790370c592e32ed4b81d`,
7,853 bytes).

## Prospective v2 campaign

The v1 result qualifies only the geometry of a later metric: a full-range BF16
ULP maximum is unstable near zero even when whole-array absolute and relative
errors are small. It does not qualify new thresholds, inputs, or outputs.

The v2 policy uses a new deterministic input, weight, projection, and cotangent
set that has not run on hardware. Each role retains whole-array maximum
absolute, mean absolute, and relative L-infinity limits. Ordered finite BF16
ULP is an acceptance metric only where the binary64 reference magnitude meets
a predeclared role-specific absolute floor; the full-range ULP remains a
diagnostic. Type, shape, public role order, finiteness, and repeat bytes remain
mandatory.

Disabled JAX is diagnostic-only and cannot reject the v2 campaign. The matrix
runs all six primal coordinates before any composed coordinate, so the
normative 1-launch and 2-launch special plans complete first. The worker then
fsyncs one bounded canonical checkpoint containing all six primal result and
oracle identities, metrics, repeat status, and observer topology. If composed
work later fails, the prototype validates that checkpoint and places it last in
the surfaced error, after the bounded phase-log tail. The compiler, planner,
emitter, and runtime are unchanged.

## Future work

- [ ] Decide whether a later campaign should gate disabled JAX against the independent oracle or retain it as diagnostic-only, without retroactively changing r2.
- [ ] Audit a generic multi-Fold/multi-Contract composed plan only after the semantic gate is localized.
