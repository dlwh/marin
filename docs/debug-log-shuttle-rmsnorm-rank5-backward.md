# Debugging log for Shuttle rank-5 RMSNorm backward

Identify the exact ABI22 GPU bundle mechanism causing the source-ordered
`input_cotangent` mismatch for shape `[3, 2, 5, 11, 263]` without modifying
production code or tests.

## Initial status

The ordinary-JAX H100 canary reports an enabled-versus-disabled mismatch only
for the rank-5 backward source-ordered `input_cotangent`. The review starts at
clean compiler/canary tip `dbff57e3bc6fda7d88234c8d28428c93ad3ef65e`.

## Hypothesis 1

The rank-263 primary row Fold or the compact primary Fold temporary is indexed
with an incorrect affine linearization when expanded into the backward
`input_cotangent` task.

## Changes to make

Read the structural plan, Fold-region PTX emitter, fixture SSA, and native
plan/PTX assertions. Do not alter production code or tests.

## Preserved r2 evidence

The reviewed r1 request was preempted before subject output. The unchanged r2
request ran on one H100 in `cw-rno2a` and failed after 15 minutes 29 seconds.
Iris reported one task failure, exit status 1, and no preemption.

The first failing coordinate was shape `[3, 2, 5, 11, 263]`, backward boundary,
SOURCE_ORDERED mode, role `input_cotangent`. Both values were BF16 with shape
`[3, 2, 5, 11, 263]`. The actual result digest was
`b7b633ae:8f64da15:b63b74be:5c8582c1:8ec9c8d3:a978bac3:edb64616:8ea97b31`;
the disabled ordinary-JAX baseline digest was
`a1e8d5a1:a707f1e2:171fc39f:184f3d89:a8c6bb14:e38ba3ea:ab839f16:a6d0d056`.

The subject iterates shapes, then forward/backward/composed boundaries, then
disabled/SOURCE_ORDERED/FAST modes. Therefore all nine rank-3 coordinates, all
three rank-5 forward coordinates, and rank-5 backward disabled completed their
checks before the failure. The failing enabled coordinate completed GPU
execution, observer validation, public result count/order/type/shape checks,
and the expected three-launch/one-temporary bundle check before exact byte
comparison failed on the first backward result.

## Hypothesis 1 results

An offline float32 interpreter of the emitted rank-5 backward kernel reproduces
the r2 actual digest exactly:
`b7b633ae8f64da15b63b74be5c8582c18ec9c8d3a978bac3edb646168ea97b31`.
The interpreter uses the canary's deterministic BF16 inputs, the emitted
256-thread by two-contiguous-leaf reduction, adjacent-pair shared-memory
compaction, `sqrt.rn.f32` followed by `div.rn.f32`, and the StableHLO operation
order. This rules out result projection, a missing 263 tail element, temporary
shape expansion, and public role ordering as explanations for the observed
digest.

Serial ascending and 256-strided reduction variants produce the same final BF16
digest for this input. The disabled-XLA digest remains different.

## Hypothesis 2

XLA GPU and Shuttle use different reciprocal-square-root arithmetic. Inspect
the exact pinned XLA lowering for `stablehlo.rsqrt` and compare its instruction
contract with Shuttle's `sqrt.rn.f32` plus `div.rn.f32`. A repair must remain
generic and preserve the explicit ABI22 numerical policy; matching one fixture
digest is not sufficient.

## Hypothesis 2 results

The pinned XLA GPU MLIR path does not use the older elemental LLVM
`sqrt`-then-divide helper. It maps HLO `rsqrt` through `mhlo::RsqrtOp` and
`math::RsqrtOp`; the CUDA conversion lowers f32 `math.rsqrt` to
`__nv_rsqrtf`. Shuttle instead emits two separately rounded PTX instructions,
`sqrt.rn.f32` followed by `div.rn.f32`.

The pinned XLA row-reduction emitter also uses 32 reduction lanes for extent
263. Each lane accumulates the fixed leaf sequence `lane + 32 * k`, then a
shuffle-down tree combines distances 16, 8, 4, 2, and 1. Shuttle uses 256
lanes, two contiguous leaves per lane, and an adjacent-pair shared-memory
tree. Both preserve leaf order but choose different permitted associations.

An offline float32 interpreter separates these two differences. Its grouped
SHA-256 results for the rank-5 backward input cotangent are:

- Shuttle association plus Shuttle sqrt/div: r2 actual
  `b7b633ae:8f64da15:b63b74be:5c8582c1:8ec9c8d3:a978bac3:edb64616:8ea97b31`.
- XLA association plus Shuttle sqrt/div:
  `67b8b217:1ab28014:9b751d96:ab23300c:d59d302e:08406f61:1eb806db:a302c83a`.
- Shuttle association plus a once-rounded reciprocal square root:
  `7f2eb783:afa5a861:a790d844:40f6b475:c84ff3e0:b64be7e0:e12ec2a8:59c13249`.
- XLA association plus a once-rounded reciprocal square root: r2 disabled
  expected exactly
  `a1e8d5a1:a707f1e2:171fc39f:184f3d89:a8c6bb14:e38ba3ea:ab839f16:a6d0d056`.

Both arithmetic choices are therefore necessary to explain the preserved r2
evidence. The failure is not caused by rank-5 affine addressing, the 263 tail,
compact temporary expansion, public result ordering, or FFI projection. No
production repair has been made yet.

## Acceptance-policy audit

The disabled ordinary-JAX GPU result is not the approved source semantic. The
ABI21 RMSNorm numerical policy, inherited by this ABI22 Fold-region prototype,
declares the ordinary-JAX result an optional stronger comparator. Its Fold
schedule fixes leaf order but permits tree reassociation, and it explicitly
binds the reciprocal square root to rounded f32 square root followed by rounded
f32 division. The current Shuttle schedule and PTX satisfy those declarations;
matching XLA's different association and libdevice reciprocal square root would
change the declared recurrence.

The deterministic simulator that reproduces the r2 actual digest was compared
with an independent binary64 RMSNorm/VJP equation evaluated from the exact
public BF16 inputs. For `input_cotangent`, it reports:

| Metric | Observed | Predeclared limit |
| --- | ---: | ---: |
| Maximum absolute error | 0.003906274174185986 | 0.0078125 |
| Mean absolute error | 0.0009506748356342832 | 0.002 |
| Relative L-infinity error | 0.0022029898759516916 | 0.00390625 |
| Maximum BF16 ULP error | 1 | 1 |

Of 86,790 public values, 86,789 equal the correctly rounded binary64 oracle and
one differs by one BF16 ULP. The current output therefore passes every
predeclared per-role acceptance threshold. The r2 stop is a canary-policy false
negative caused by treating the diagnostic disabled-JAX bitwise comparison as
an acceptance gate, not evidence of unsafe admission or incorrect generated
GPU code.
