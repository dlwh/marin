# Debugging log for Shuttle LayerNorm forward near-zero numerics

Determine whether the ABI22 LayerNorm-forward H100 acceptance gate correctly
classifies a near-zero sign crossing without changing the compiler emitter.

## Initial status

The reviewed request at source commit
`f8b4ef44d55c46fc4181927697e1a642449841a3` ran
`@shuttle_mlir//:gpu_transport_v2_h100_layernorm_forward_test` on one H100 in
Iris job `/dlwh/shuttle-layernorm-forward-h100-r3`. The 7x13 BF16 fixture passed
source-ordered and fast policy. The 2048x4096 fixture failed both policies at
element 171197 (row 41, feature 3261): the GPU returned
`+2.2351741790771484e-08`, while the serial-f32 host oracle returned
`-2.6077032089233398e-08`. Their absolute difference is approximately
`4.842877388e-08`, but the signed BF16 ordering reports 26016 ULP because the
values straddle zero. The permitted limits were 1 ULP for source-ordered and 4
ULP for fast. The build completed 8056 actions and the exact test executed; this
is not an infrastructure, binding, nonfinite, poisoned-output, or request error.

The sealed request SHA-256 was
`9534f59f22cdd4f4dd28112de1185ffe40d72618a8933bb6ba5f3edbae73271b`.
The Iris terminal summary is preserved in
`scratch/20260821-0709_monitoring_state.json`.

## Read-only audit conclusion for r3

Raw BF16 ULP is ill-conditioned across zero: a small perturbation can traverse
the signed subnormal encoding interval. That fact does not qualify r3 for
acceptance. The run did not record sealed whole-tensor maximum, mean, and
relative L-infinity metrics against an independent reference, and its generated
inputs and serial-f32 reference are distinct from the ABI11 contract. ABI11 or
Target 1 limits therefore cannot be inherited.

An independent binary64 reconstruction from the public BF16 inputs gives, for
the reported element, mean `0.13217626209370792`, variance
`0.2589315130032364`, and unrounded output `-5.0995900904848845e-09`. The GPU
value is `2.745133188125637e-08` from that analytic result; the existing serial
f32 oracle is `2.0977441998748514e-08` from it. The analytic value lies between
the two observations. This explains the reported element, but it does not
establish whole-tensor accuracy or justify a near-zero waiver. r3 remains a
qualification-only diagnostic and cannot promote a compiler configuration.

An initial host test proposing a near-zero escape was removed before it compiled
or ran. No acceptance behavior or emitter code changed.

## Hypothesis 2

A prospective contract can avoid post-hoc treatment by using new deterministic
inputs, modeling the declared emitted schedule exactly, and fixing limits before
hardware. The schedule model is:

1. BF16 inputs expand exactly to f32.
2. A row fold assigns each of 256 threads 16 consecutive elements for 4096
   features (32 threads and one element per thread for 13 features).
3. Each thread accumulates its contiguous leaf interval with `add.rn.f32` from
   positive zero.
4. Shared-memory partials reduce adjacent pairs with `add.rn.f32` until one
   value remains.
5. Mean and variance scaling use `div.rn.f32`; the variance phase repeats the
   same leaf and tree schedule using the scheduled mean.
6. `rsqrt` is realized as `sqrt.rn.f32` followed by `div.rn.f32` from 1.0.
7. Centering, affine multiplication, and beta addition are individual RNE f32
   operations before `cvt.rn.bf16.f32`.

Two new held-out affine cases use a counter-mixed integer generator and exact
power-of-two BF16 values. `balanced` includes realistic sign crossings;
`positive_offset` keeps output away from zero so raw BF16 ULP remains a strict
independent axis. The existing r3 formula is not used.

Before hardware, the exact schedule model versus binary64 oracle gives these
candidate metrics for 2048x4096:

| case | max absolute | mean absolute | relative L-inf | raw max BF16 ULP |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.015625 | 3.1493970653606596e-08 | 0.004975124378109453 | 9 |
| positive_offset | 0.0625 | 3.725290298461914e-08 | 0.005376344086021506 | 1 |

Both 7x13 cases are bitwise equal between the schedule model and binary64
oracle. Candidate prospective limits are case-specific and are not derived from
hardware: balanced `(0.015625, 1e-6, 0.0078125, 16)` and positive-offset
`(0.0625, 1e-6, 0.0078125, 1)` for maximum absolute, mean absolute, relative
L-inf, and raw maximum BF16 ULP respectively. There is no near-zero escape;
every finite output participates in the raw ULP metric.

## Red-first contract result

The first version of the prospective model used a serial row fold. Its sealed
2048x4096 balanced schedule digest was
`9126627880838eb8205d217555045de1c9c5621e10f5bd74623b9d5801f558ac`,
while the serial implementation produced
`177891fbc6719ea7a4e5826c933df16c38b9c54ac8f39b59fc5fc1517674d902`.
The contract test failed before the exact contiguous-leaf and adjacent-pair
tree was implemented. No compiler source was changed.

The green prospective contract now seals the generator, all input and output
digests, exact software schedule, binary64 oracle, case-specific limits, both
shapes, both policies, and two-run repeatability. Mutation tests reject serial
fold order, missing mean, missing variance, rotated affine indexing, a gross
near-zero error, NaN, infinity, threshold drift, and topology drift.

The H100 test is versioned separately from the r3 diagnostic. It verifies exact
public bindings and resource counts, regenerates and hashes every held-out
input, requires the GPU result to be bitwise equal to the declared schedule
model, measures every finite BF16 element against binary64, and requires two
bitwise-identical runs for every shape, case, and policy. The existing compiler
emitter is unchanged.

## Local verification

- Direct Python contract: 7 tests passed in 0.808 seconds.
- Fresh-root Bazel contract target: passed in 2.1 seconds after 21 actions.
- Fresh-root Bazel GPU transport target: the sealed host model passed and the
  H100-only test skipped locally; compilation and execution completed after
  3804 actions in 231.756 seconds.
- Full GPU transport binary: 48 tests ran in 2.061 seconds; 42 host tests
  passed and the six hardware-only tests skipped.
- Full non-manual native census from a fresh Bazel root: 69 of 70 targets
  passed after 5558 actions. The sole failure was the established ABI19
  libdevice red in
  `DefaultExpProducesDeterministicClosedPtxForBothPolicies`; its other three
  tests passed.

The contract still requires independent review before any r4 request is
rendered or submitted.

## Future work

- [ ] Record full aggregate H100 metrics for both cases, shapes, and policies.
- [ ] Diagnose the reduction-order source only if the aggregate binary64 gate
  rejects the observed implementation.
