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

Pending the diagnostic-only r2 run.

## Future work

- [ ] Use r2 evidence to determine whether the independent oracle or the frozen threshold is inconsistent with ordinary disabled JAX.
- [ ] Audit a generic multi-Fold/multi-Contract composed plan only after the semantic gate is localized.
