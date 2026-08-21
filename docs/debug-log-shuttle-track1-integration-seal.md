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

- [ ] Rotate and independently review the live Track 1 source manifest and
  authorization on the merged integration tip.
