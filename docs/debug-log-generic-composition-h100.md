# Debugging log for generic composition H100 numerical failure

The anonymous ordinary-JAX `Contract -> Fold / Map -> Contract` canary must
identify the r1 numerical failure without changing the compiler, deterministic
inputs, binary64 oracle, thresholds, or topology acceptance.

## Initial status

r1 ran `/dlwh/shuttle-track1-generic-composition-h100-canary-r1` on one batch
H100 in `cw-us-east-02a`. Iris recorded one failed task, zero preemptions, and a
15 minute 40.43 second duration. The worker reached `validate_coordinate` and
rejected `maximum_absolute_error`. The output path serialized only after every
coordinate passed, so the case, mode, metric value, and output values were lost.
The initial attribution to `rank2-a/disabled` was invalid: fail-fast ordering
does not prove that the first coordinate failed. r1 remains unlocalized.

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
The exact coordinate, mechanism, and error values remain unknown, so r1 does
not implicate generated composition codegen.

## Diagnostic successor

r2 retains the r1 compiler, corpus, binary64 oracle, thresholds, and semantic
topology contract. Each structurally valid coordinate now writes one closed,
canonical NDJSON record to a create-only file and calls `fsync` before applying
the unchanged numerical thresholds. A threshold miss is recorded and deferred
until all six coordinates have run; ABI, finiteness, repeatability, source,
observer, and device failures remain immediate.

The prototype validates the complete six-record file, including coordinate
order, policy and source hashes, public shapes, declared `{h,s}` ownership,
observer uniqueness, numerical metrics, diagnostic BF16 ULP distances, and
record size. On a failed phase it appends those validated records after the
bounded sanitized log tail. On success it carries them in the canonical job
result. This avoids both the 8 KiB tail limit and redaction of ordinary SHA-256
text without allowing arbitrary worker log bytes past the sanitizer.

## Future work

- [x] Review and run one diagnostic-only r2 canary.
- [ ] Replace the platform-specific raw binary64 digest with a stable oracle
  identity before another run.

## r2 result

r2 ran all six coordinates before aggregate validation. Four enabled Shuttle
coordinates passed the unchanged v1 maximum, mean, and relative-error limits;
their raw and masked BF16 ULP distances were zero. Observer invocation IDs were
1 through 4. The only threshold failure was `rank3-b/disabled`: maximum
absolute error was `0.0006204024596327962` against `0.0005`, and mean absolute
error was `0.00011096225522005224` against `0.00005`. Its relative Linf error
and BF16 ULP diagnostics remained in range. Iris recorded one failed task, zero
preemptions, and 983.18 seconds runtime.

The diagnostic schema exposed a platform-specific identity. NumPy's binary64
matrix multiplication uses the platform BLAS, so the raw `reference_binary64`
digest differs between the Linux worker and local macOS validation. The
prospectively sealed input and rounded-f32 oracle digests match on both. The
exact Linux records remain preserved under
`lib/shuttle/mlir/artifacts/track1-generic-composition-h100-r2-failure-20260825/`.

r2 corrects the earlier r1 attribution. The r1 exception carried no case or
mode, so nested execution order did not prove that the first coordinate failed;
it only bounded the failure to the first coordinate that did fail. r1 remains
unlocalized. r2 localizes its own distinct failure to `rank3-b/disabled`.

## Hypothesis 2

Ordinary XLA is useful as a required device diagnostic but is not the Shuttle
numerical acceptance source. A prospective v2 policy should require every
disabled coordinate to satisfy source, ABI, finiteness, repeatability, and
empty-observer checks without letting its numerical threshold classification
reject otherwise valid enabled SOURCE_ORDERED and FAST coordinates. Enabled
coordinates retain the existing thresholds and observer/semantic contract.
