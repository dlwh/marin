# Debugging log for the Shuttle RMSNorm Contract hybrid plan

Replace the ordinary-JAX composed RMSNorm-to-Contract plan's duplicated primal
work with the existing generic special plan while preserving all public VJP
results and structural fallback.

## Initial status

The exact integration base
`d11e58557d19ba3ff6c5741418637a0b6f2735d4` admits the rank-3
`[5,19]x[19,7]` and `[5,19]x[19,257]` composed fixtures through the ABI22
plugin. The public subject is semantically accepted on H100, but the composed
path remains a diagnostic 42-launch generic plan. It recomputes the forward
RMSNorm ancestry rather than materializing the normalized BF16 boundary once
for the reverse consumers.

## Hypothesis 1: substitute the closed primal slice into the generic plan

One public result has a unique closed backward slice matching the existing
RMSNorm-to-Contract planner. The other public results remain ordinary generic
Map, Fold, and Contract dataflow. A hybrid plan can therefore replace only the
generic BF16-boundary, primal Contract, and primal epilogue tasks with the
special plan. The special Contract can additionally materialize its normalized
BF16 boundary for the retained reverse consumer.

This remains structural: selection is based on one unique yielded closed
subgraph, source operations, types, indexing maps, and buffer ownership. It
does not inspect fixture names, workload names, or exact shape constants.

## Localization

The first post-substitution plan failed the existing unused-private-buffer
invariant. Source operation 21, the f32 Map immediately before the normalized
BF16 boundary, was still owned by a generic task even though the special task
now owned source operations 21 through 24. Its result had a producer and no
consumer.

The invariant was correct. The plan needed a reverse live-out slice after task
substitution, not an exception for an unused buffer. Marking every public
live-out, then following producer inputs, removes the displaced generic task
and any other unreachable private storage. Producer, consumer, dependency, and
recomputation declarations are rebuilt from the retained task order.

Independent review then exposed two closure failures masked by the original
rank-2, all-role fixture. First, the materialized normalized output schema
flattened the activation prefix to `{rows, reduction}` even though the planner
admits activation ranks 2 through 8. Second, recomputation declarations were
rebuilt in one pass: a shared source was annotated only after an earlier task
had made the repetition visible. A compact projection-only VJP, with no
retained reverse activation task, consequently declared only the later
Contract occurrence.

The schema now derives the normalized output from the Contract domain prefix
followed by the reduction extent and separately verifies that the checked
prefix product equals the planned row count. Recomputation rebuilding first
counts every retained task-source occurrence, then annotates every Fold or
Contract region occurrence of a repeated source. Exact-source verification
uses the same occurrence model and is independent of task order.

The higher-rank regression uses a rank-3 normalized-boundary consumer rather
than a full rank-3 VJP. The latter contains a reverse Contract with two
reduction dimensions, which the current closed projected-Contract admission
does not support. This change intentionally does not broaden Contract
admission. It also recomputes the retained rank-N Fold property after pruning:
a Fold fully absorbed by the special region does not block the hybrid, while a
retained legacy rank-N Fold still does.

## Changes

The standalone RMSNorm-to-Contract derivation now also accepts a closed yielded
subplan. After generic Map fusion, the hybrid derivation finds exactly one
specializable public result, substitutes the existing inline or compact
RMSNorm-to-Contract tasks, retains the generic reverse tasks, and applies the
reverse live-out slice. Failure at any structural or ownership check returns
the unchanged generic plan.

The special Contract may have a private normalized BF16 output followed by the
public primal BF16 result. The schema fixes that order, shape, type, storage,
and lifetime. PTX writes each normalized element once from the same computed
boundary value used by the primal Contract, then writes the primal result.
Transport retains the same ordered two-slot projection. A public normalized
live-out does not take the hybrid path.

The red contract covers both shapes and both numerical policies through the
executable bundle path. It requires one special Contract region, the compact
Fold region only for 257 columns, exact source coverage, one private normalized
boundary consumed by one reverse Contract, fewer tasks than the prior 42-launch
diagnostic plan, two ordered BF16 PTX stores, and matching ordered transport
outputs. Mutations reject swapped outputs, public lifetime, and external
storage. The test deliberately does not freeze the remaining composed launch
count as the target architecture.

## Results

The final uncached combined native gate passes 63 targets: the ABI22 Map-fusion
Contract suite, registered Track 1 wiring, XLA registration and adapter tests,
the pipeline observer, and all 58 MLIR tests. It includes the rank-2 composed,
rank-3 boundary, and compact projection-only regressions added for the review
repairs.

No GPU job ran for this change. The result establishes a generic compiler,
schema, PTX, and transport plan boundary; it does not establish H100 numerical
or performance behavior.

## Future work

- [ ] Run an independently reviewed ordinary-JAX H100 canary before accepting
  the hybrid composed path as hardware evidence.
- [ ] Measure the remaining generic reverse path before freezing a target
  launch topology.
