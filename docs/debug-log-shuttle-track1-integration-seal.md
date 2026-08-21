# Debugging log for the Track 1 integration seal

Keep the authorization-bound Track 1 launcher fail-closed after merging reviewed
H100 evidence from independent integration branches.

## Initial status

On merge commit `03521e3b5224a38cfda300af6328ff2819144a1a`, the focused
Track 1 suite passed 86 tests and failed 25. Twenty-four request and closure
tests rejected the live source bundle because the checked-in manifest predated
the LayerNorm B/C evidence commit. The stdlib-only launcher test also failed
because importing the prototype imported `ml_dtypes` before the explicit
authorization rejection.

## Hypothesis 1: the fail-closed launcher imports optional dependencies too early

The launcher can parse `--require-launch-ready` and reject an unsealed launch
using only the standard library. Importing the prototype before argument
parsing unnecessarily loads the numerical-policy dependency graph.

## Changes to make

Import the prototype only after the explicit fail-closed boundary. Keep the
normal authorization-bound dry-run and execution paths unchanged.

## Results

The three launcher tests pass, including the `python -S` subprocess and the
authorization-bound dry run. The source-closure failures remain expected until
the merged integration tip receives a new current manifest and authorization;
historical request and result artifacts must not change.

## Future work

- [x] Rotate the live Track 1 source manifest and authorization on the merged
  integration tip. Independent exact-tip review remains the final handoff gate.

## Hypothesis 2: the merged source closure needs a current seal

The launcher repair advanced the integration tip to
`22b31a106fb80a81df3bfdc5b77d23c396af8fda`. Relative to the reviewed
770-entry manifest, the live closure has ten additional LayerNorm B/C evidence
files and a changed launcher. The existing fail-closed checks correctly reject
this 780-entry tree rather than silently constructing a request from unbound
bytes.

## Changes to make

Regenerate only the current source manifest from the exact merged tree and bind
its SHA-256 in the current authorization. Leave every historical request,
result, receipt, and evidence checksum byte unchanged.

## Results

Before resealing, the exact focused gate passed the three launcher tests and
one fail-before-bundle request test, while 24 Iris request and source-closure
tests failed with `live source bundle differs from the authorized manifest`.
The regenerated manifest contains 780 entries and has SHA-256
`6aca1293b9d2cd6f0799763adc0dfc4f1df6e738d5d1c763a668ec02b5efddbb`.
The bound canonical authorization has SHA-256
`1bd43a8ba0a90632f1e7ac7c26ed125b067c4766c6aefd88b8afdd27922e3e5f`.
The same focused gate now passes all 28 tests. The diff contains only this log,
the current manifest, and the current authorization; historical artifact paths
remain byte-identical to the integration base. The proportional Track 1
prototype, Iris request, and launcher suite passes all 78 tests, and the scoped
pre-commit gate passes.
