# Debugging log for generic composition H100 numerical failure

The anonymous ordinary-JAX `Contract -> Fold / Map -> Contract` canary must
identify the r1 numerical failure without changing the compiler, deterministic
inputs, binary64 oracle, thresholds, or topology acceptance.

## Initial status

r1 ran `/dlwh/shuttle-track1-generic-composition-h100-canary-r1` on one batch
H100 in `cw-us-east-02a`. Iris recorded one failed task, zero preemptions, and a
15 minute 40.43 second duration. The worker reached `validate_coordinate` and
rejected `maximum_absolute_error`. The output path serialized only after every
coordinate passed, so the metric value and output values were lost. The sealed
case-then-mode order and immediate validation abort identify the coordinate as
`rank2-a/disabled`. Both disabled-JAX executions had already passed the f32 ABI,
shape, finiteness, bitwise repeat, and empty-observer checks. No enabled generic
composition coordinate ran.

## Hypothesis 1

A bounded canonical record emitted before threshold validation will distinguish
disabled-JAX numerical drift from enabled composition code generation, and will
retain observer and device execution evidence for an enabled failure.

## Changes to make

Keep the r1 numerical policy and source computation unchanged. Emit one
sanitizer-safe record per coordinate containing source identity, semantic group
and private-boundary expectations, numerical measurements, first and worst
mismatches, output digests, repeat status, observer attribution, and device
status. Flush each record before applying the existing thresholds and prove the
failing record survives the prototype log-tail boundary.

## Results

r1 is preserved under
`lib/shuttle/mlir/artifacts/track1-generic-composition-h100-r1-failure-20260821/`.
The exact mechanism and error values remain unknown. The failing coordinate is
`rank2-a/disabled`, so r1 does not implicate generated composition codegen.

## Future work

- [ ] Run one reviewed diagnostic-only r2 canary.
- [ ] Change compiler semantics only after r2 localizes a contract violation.
