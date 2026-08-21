# LayerNorm backward/composed H100 contract evidence

This directory preserves the successful `r2` qualification of the prospective
ABI22 LayerNorm backward/composed numerical contract on one H100. The reviewed
job ran all eight coordinates formed by backward/composed boundaries, `7x13`
and `2048x4096` shapes, and SOURCE_ORDERED/FAST. Each coordinate executed twice.
All 28 public-role observations matched the declared schedule bitwise, repeated
bitwise, and satisfied the independent binary64 limits.

This is functional evidence for direct Shuttle GPU execution. It is not an
ordinary-JAX plugin qualification or a performance result. The 4 minute
5.82 second task duration includes source hydration, compilation, and tests;
it is not a kernel benchmark. LayerNorm-forward evidence is preserved
[separately](../layernorm-forward-h100-20260821/README.md).

## Identity

- Job: `/dlwh/shuttle-layernorm-bc-h100-r2`
- Cluster and priority: `cw-rno2a`, BATCH
- Source commit: `481a91743f8122b7c0996822e94e3dfcf0d3bced`
- Source tree: `060b9eb9cbbb8da1f75b40ec42e6cfd28fe7e680`
- Source manifest SHA-256:
  `b8336bb1e682d45254923848705becad617ee1de690985f659835356b6e9e65b`
- Contract: `shuttle_abi22_layernorm_bc_numerical_contract_v1`, SHA-256
  `c28aa8657fa9b854e2015c9fe734da23092c8b0897076cc0d15a780cd13415aa`
- Request SHA-256:
  `f191c8fb5067d48c2d03adcb491b9aa0c798519ab739501fc35ce57fb9024d04`
  (7,032,069 bytes)
- Bundle SHA-256:
  `7b83d36042a3aa361b2da38289cb8604a5690a37739137964fb4759c529d0091`
  (7,025,390 bytes)
- Exact Iris task-log SHA-256:
  `77c5714e2a6b24088f80e3ad572ea16e648e11948fec9bd8e48aa5b8f07466a0`
  (56,539 bytes before deterministic gzip compression)

The request protobuf is intentionally omitted. `request-metadata.json` and
`receipt.json` retain its digest, size, exact source and contract identities,
resource request, retry policy, targets, and clean independent-review verdict.
The reviewed request rerendered byte-identically before submission.

## Numerical result

The H100 output matched the sealed software schedule bitwise for every public
role, so the exact observed analytic metrics are the schedule metrics in
`contract.json`. SOURCE_ORDERED and FAST produced the same values in this
matrix. Both runs of every coordinate had zero schedule and repeatability
differences.

| Public role | Worst shape | Max abs | Mean abs | Relative Linf | Max BF16 ULP |
| --- | --- | ---: | ---: | ---: | ---: |
| `y` | `2048x4096` composed | 0.015625 | 6.057504975398587e-8 | 0.004291845493562232 | 67 |
| `dx` | `2048x4096` backward/composed | 0.0078125 | 3.725514735597457e-8 | 3.838689072327044e-7 | 4 |
| `dgamma` | all | 0 | 0 | 0 | 0 |
| `dbeta` | `2048x4096` backward/composed | 1 | 0.000244140625 | 0.0064516129032258064 | 1 |

All `7x13` role metrics were zero. `metrics.json` preserves the exact 28
GoogleTest properties emitted before result validation. `result.json` preserves
the canonical successful result, including the H100 identity, contract and
source hashes, tool pins, targets, and hashes of the task-internal Bazel log,
metrics artifact, and test XML. `contract.json` is byte-identical to the
executed tracked contract and retains every held-out input, scheduled-output,
and binary64-output digest.

## Public order and resources

Backward binds public inputs `[x,gamma,dy]` and results
`[dx,dgamma,dbeta]`; its physical result slots are `[7,6,5]`. Composed binds
`[x,gamma,beta,dy]` and `[y,dx,dgamma,dbeta]`; its physical result slots are
`[6,9,8,7]`. This projection restores public result order even though the
feature task physically produces `dbeta` before `dgamma`.

Every coordinate used two launches and two compact row-stat temporaries. The
second launch depends on the first. Temporary slots are `[3,4]` for backward
and `[4,5]` for composed. Temporary storage was 56 bytes at `7x13` and 16,384
bytes at `2048x4096`. The contract seals each launch's ordered input/output
slots, dependency, row/feature grid, 32- or 256-thread block, and 128- or
1,024-byte dynamic shared-memory allocation.

## Terminal result

Iris reports one successful task, exit code zero, no failure or preemption, and
a duration of 4 minutes 5.82 seconds. The device was an NVIDIA H100 80GB HBM3
with driver `595.71.05`. Both the host contract target and manual H100 target
passed under Bazel 7.7.0, XLA
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`, and CUDA 12.9.1 for `sm_90`.

`task.log.gz` deterministically compresses the exact Iris task log, including
the canonical metrics and result markers. Verify every preserved file with:

```bash
(cd lib/shuttle/mlir/artifacts/layernorm-bc-h100-20260821 && shasum -a 256 -c SHA256SUMS)
gzip -dc lib/shuttle/mlir/artifacts/layernorm-bc-h100-20260821/task.log.gz | shasum -a 256
```
