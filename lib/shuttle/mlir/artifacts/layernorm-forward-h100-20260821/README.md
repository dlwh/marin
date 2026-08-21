# LayerNorm-forward H100 contract evidence

This directory preserves the successful `r4` qualification of the prospective
ABI22 LayerNorm-forward numerical contract on one H100. The reviewed job ran
all eight coordinates formed by two shapes, two held-out cases, and
SOURCE_ORDERED/FAST. Each coordinate executed twice, matched the declared Fold
schedule bitwise, and satisfied the independent binary64 limits.

This is forward-only functional evidence for direct Shuttle GPU execution. It
does not cover backward, a composed norm-to-Contract boundary, ordinary-JAX
plugin admission, representative production shapes, or performance. The
2,636 ms GoogleTest time and 212.609 s Bazel elapsed time include test and build
overhead and are not kernel benchmarks.

## Identity

- Job: `/dlwh/shuttle-layernorm-forward-h100-r4`
- Cluster and priority: `cw-rno2a`, BATCH
- Source commit: `3b1108c191d9327d423fb84c103352295d0611fe`
- Source tree: `e763d21e9f44cd811afa4471db26db8b93293113`
- Source manifest SHA-256:
  `2897a10650d190421f14404c83708b0bbd06496ce5f33815f526012d262d2a0f`
- Contract: `abi22_layernorm_forward_h100_v1`, SHA-256
  `35dddbabe2e8a074ef33f7cc625583a5e89ba687c450d13b0a3570a9ffe26233`
- Request SHA-256:
  `0d203bb61b5734dc57513671048854c1956572adab9f13954a41a41788c6ee93`
  (6,436,497 bytes)
- Bundle SHA-256:
  `fb75ac0581d0c174fc51bc760e22b23afa44eec234e1ad7b0b48f7c547835975`
  (6,432,533 bytes, 701 tracked Shuttle entries)
- Task result-log SHA-256:
  `2938df61816b8e156901aaee6cd84f395a4e0d412390cbb7719a02b552aa1638`
  (28,523 bytes)

The 6.44 MB request protobuf is intentionally omitted. Its exact reviewed
digest, size, resource request, retry policy, sealed source identities, and
permission closure are retained in `request-metadata.json` and `receipt.json`.
The request was independently decoded and rerendered byte-identically before
submission; review found no P0-P3 issue.

## Numerical result

The declared model is
`simt32_contiguous_leaf_then_adjacent_pair_tree_v1`, with
`sqrt.rn.f32` followed by `div.rn.f32`, the declared f32 epilogue, no near-zero
exception, and raw ordered finite-BF16 ULP distance over every element.

| Shape | Case | Policy | Max abs | Mean abs | Relative Linf | Max BF16 ULP | Schedule/repeat differences |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `7x13` | balanced | SOURCE_ORDERED | 0 | 0 | 0 | 0 | 0 / 0 |
| `7x13` | balanced | FAST | 0 | 0 | 0 | 0 | 0 / 0 |
| `7x13` | positive offset | SOURCE_ORDERED | 0 | 0 | 0 | 0 | 0 / 0 |
| `7x13` | positive offset | FAST | 0 | 0 | 0 | 0 | 0 / 0 |
| `2048x4096` | balanced | SOURCE_ORDERED | 0.015625 | 3.1493970653606596e-8 | 0.004975124378109453 | 9 | 0 / 0 |
| `2048x4096` | balanced | FAST | 0.015625 | 3.1493970653606596e-8 | 0.004975124378109453 | 9 | 0 / 0 |
| `2048x4096` | positive offset | SOURCE_ORDERED | 0.0625 | 3.725290298461914e-8 | 0.005376344086021506 | 1 | 0 / 0 |
| `2048x4096` | positive offset | FAST | 0.0625 | 3.725290298461914e-8 | 0.005376344086021506 | 1 | 0 / 0 |

For every coordinate, the runner reported one launch, three external operands,
one external result, zero temporary allocations, and zero temporary bytes.
`cells.json` preserves the exact input, independent binary64, declared
schedule, and inferred GPU-output digests together with limits and resources.
The GPU-output digest and observed metrics are inferred without approximation:
the passing hardware test requires zero BF16 differences from the sealed
schedule, recomputes the contract digests, and applies the same binary64 metric
routine and limits. The provenance is recorded in that file rather than
claiming the values were printed individually by GoogleTest.

## Terminal result

Iris reports one successful task, exit code zero, no failure or preemption,
and a duration of 3 minutes 53.99 seconds. The H100 was an NVIDIA H100 80GB
HBM3 with driver `595.71.05`. Bazel completed 8,077 actions and both the host
contract target and manual H100 target passed.

`result.log.gz` deterministically compresses the exact task-owned Bazel log
reconstructed by removing Iris line prefixes. Its uncompressed size and
SHA-256 reproduce the values in the canonical `result.json`. Verify every
preserved file with:

```bash
(cd lib/shuttle/mlir/artifacts/layernorm-forward-h100-20260821 && shasum -a 256 -c SHA256SUMS)
gzip -dc lib/shuttle/mlir/artifacts/layernorm-forward-h100-20260821/result.log.gz | shasum -a 256
```
