# Ordinary-JAX RMSNorm-to-Contract H100 plugin evidence

This directory preserves the successful `r3` canary for generic ABI22
RMSNorm-to-Contract regions through the registered ordinary-JAX/XLA plugin.
The job used one batch-priority H100 in `cw-rno2a`. Its RMSNorm-to-Contract
result contains 12 coordinates:

- shapes `[5,19,7]` and `[5,19,257]` at epsilon `1e-6`;
- standalone primal and composed value-plus-VJP public boundaries;
- disabled JAX, SOURCE_ORDERED, and FAST modes.

The six primal coordinates ran before the composed coordinates. The enabled
primal topology is normative: one launch and no temporary for output extent 7,
and two launches with one 20-byte temporary for output extent 257. The enabled
composed path executed correctly with 42 launches and 38 temporaries, but that
topology remains `diagnostic_non_performant`; this evidence does not accept it
as the target architecture.

## Identity

- Job: `/dlwh/shuttle-track1-rmsnorm-contract-h100-canary-r3`
- Cluster: `cw-rno2a`
- Reviewed run source: `d7f55340f4f08a84c3d7304e88129755bdb890ae`
- Source manifest SHA-256: `3a8a331db045798aa73ccecb41ac247f44d5e440c3d4b19639c4592ca0bf9415`
- Authorization SHA-256: `d0c03b7f72dbe3e7164308b853b0f43a1550979e920afac4f2da96292156c9ff`
- Numerical policy SHA-256: `9fd9e00c6ef90b40d12916f671eeb5b73b385875d8e1ee485e0d029ab272996a`
- Request SHA-256: `ddb2933b25446b0f7cbf26ae87cfe7dbbfa0a7d9c1511f359954209708e86626`
- Request size: 6,758,570 bytes
- Bundle SHA-256: `ed22190110c1f94ba0f2cc461cbc2fce05bdb0841734c30578e19ae59c7ba2ca`
- Bundle size: 6,757,562 bytes
- Canonical result SHA-256: `32af1df017f5790c52b3515c72d8843763852d19d492d3e55baad03ea227a9a5`
- Canonical result size: 58,841 bytes, including its final newline

The exact authorization, source manifest, and numerical policy are preserved
beside the result. The 6.76 MB request protobuf remains external; its digest and
size are recorded in `request.json` and `receipt.json`. `source-lineage.json`
maps the reviewed 24-commit subject, policy, diagnostic, and durability history
onto integration base `bc9e782b25e58b1b44f4ade55bc76e3f6650eebb`.
All replayed run-source files are byte-identical to the reviewed originals
except `BUILD.bazel`, which also retains the newer integration targets.

## Result

Iris reports one successful task, exit code zero, no failure, no preemption,
and 16 minutes 2.2 seconds runtime. All 12 coordinates repeated byte-for-byte.
Enabled observer invocation IDs were exactly 1 through 8. The enabled topology
was 1/0 for extent-7 primal, 2/1 for extent-257 primal, and 42/38 for both
composed shapes, written as launches/temporaries.

The enabled worst numerical metrics against the independent binary64 oracle
were:

| Role | Maximum absolute | Mean absolute | Relative Linf | Masked BF16 ULP |
| --- | ---: | ---: | ---: | ---: |
| Result | 0 | 0 | 0 | 0 |
| Input cotangent | 0.23216684002008492 | 0.0662817995777676 | 0.0023686916414149177 | 0 |
| Weight cotangent | 0.40334577692291873 | 0.14983168789504084 | 0.0031414334546365657 | 0 |
| Projection cotangent | 0 | 0 | 0 | 0 |

SOURCE_ORDERED and FAST matched the predeclared rounded-oracle digests in all
enabled cells. Their composed `input_cotangent`, `weight_cotangent`, and
`projection_cotangent` digests differ from disabled JAX for both shapes; the
primal `result` digest matches. Disabled JAX remains a labeled diagnostic and
is not the acceptance oracle.

`result.json` is the canonical line emitted by the task. Its complete prototype
payload also contains the earlier Map and RMSNorm worker results; the acceptance
claim here is limited to its validated `rmsnorm_contract` block. Check the
artifact with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-rmsnorm-contract-h100-plugin-20260821 && shasum -a 256 -c SHA256SUMS)
```

This is one-device functional evidence for the listed small shapes. It is not
representative-shape coverage, a performance result, Transformer Engine parity,
bitwise-JAX parity for composed cotangents, or acceptance of the 42-launch VJP
topology.
