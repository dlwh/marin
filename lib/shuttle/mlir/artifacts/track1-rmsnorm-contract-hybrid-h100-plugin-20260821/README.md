# Ordinary-JAX RMSNorm-to-Contract hybrid H100 evidence

This directory preserves the successful `r5` canary for the generic ABI22
RMSNorm-to-Contract hybrid composed plan through the registered ordinary-JAX
plugin. The job used one batch-priority H100 in `cw-us-east-02a`. Its 12
coordinates cover:

- shapes `[5,19,7]` and `[5,19,257]` at epsilon `1e-6`;
- standalone primal and composed value-plus-VJP boundaries;
- disabled JAX, SOURCE_ORDERED, and FAST modes.

The hybrid plan materializes the normalized BF16 boundary once, reuses the
specialized primal task, and leaves the remaining VJP Contract and Fold
consumers with the generic planner. The expected topology was sealed before
the run and checked both in the serialized native bundle and in the runtime
observer.

## Identity

- Job: `/dlwh/shuttle-track1-rmsnorm-contract-h100-canary-r5`
- Cluster: `cw-us-east-02a`
- Integration implementation: `406287b97305f6229ea1b7f97a6f51f3ae31c5b6`
- Reviewed run source: `ae1db7865780ccf046c2e8ee49681afb054fa4bd`
- Source manifest SHA-256: `c3dbb267be48d93d4c16debc52cb86cb05e89669af97f671f8c1771f3498d47a`
- Authorization SHA-256: `d9c603e5419a03e046c10907987c79e89ad7e4435e1bcbeec944580bb2cbb62f`
- Numerical policy SHA-256: `9fd9e00c6ef90b40d12916f671eeb5b73b385875d8e1ee485e0d029ab272996a`
- Request SHA-256: `d56f626fc3b30588940a7b7bd28f1b927f6b1b5e28944505d5529902db25fbdd`
- Request size: 7,516,903 bytes
- Bundle SHA-256: `38e887bc11d41cb9a6ba8f3c5cda21f35561133782e704ccf55e1564cfe9b9f2`
- Bundle size: 7,515,895 bytes
- Canonical result SHA-256: `e02da4a840ac1032a2d959a3687178932c93610d1b6d37c6a27f6a7ffbda50c5`
- Canonical result size: 58,825 bytes, including the final newline

The exact run authorization, source manifest, and numerical policy are
preserved beside the result. The request protobuf remains external; its digest,
size, and submission path are recorded in `request.json`. The source lineage
is a direct six-commit descendant of the integration implementation, so no
replay mapping is needed.

## Result

Iris reports one successful task, exit code zero, no failure, no preemption,
and 16 minutes 42.15 seconds runtime. All 12 coordinates repeated
byte-for-byte. Enabled observer invocation IDs were exactly 1 through 8, with
eight distinct generated-bundle and final-module digests.

| Shape | Boundary | Launches | Temporaries | Temporary bytes | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `[5,19,7]` | primal | 1 | 0 | 0 | normative primal |
| `[5,19,257]` | primal | 2 | 1 | 20 | normative primal |
| `[5,19,7]` | composed | 39 | 36 | 6,996 | reviewed hybrid exact |
| `[5,19,257]` | composed | 40 | 37 | 79,016 | reviewed hybrid exact |

SOURCE_ORDERED and FAST matched the same topology at each shape. Runtime
observer records prove launch and temporary counts. The temporary-byte values
come from the pre-run native serialized-bundle contract.

The enabled worst numerical metrics against the independent binary64 oracle
were:

| Role | Maximum absolute | Mean absolute | Relative Linf | Masked BF16 ULP |
| --- | ---: | ---: | ---: | ---: |
| Result | 0 | 0 | 0 | 0 |
| Input cotangent | 0.23216684002008492 | 0.0662817995777676 | 0.0023686916414149177 | 0 |
| Weight cotangent | 0.40334577692291873 | 0.14983168789504084 | 0.0031414334546365657 | 0 |
| Projection cotangent | 0 | 0 | 0 | 0 |

SOURCE_ORDERED and FAST matched their predeclared rounded-oracle digests. Their
composed input, weight, and projection cotangents differ bitwise from disabled
JAX at both shapes; the result matches. Disabled JAX is a labeled diagnostic,
not the acceptance oracle.

The first reviewed request, `r4`, waited for `cw-rno2a` capacity and was killed
under the authorized migration before allocation. Its task retained an empty
worker ID, attempt `-1`, no logs, and zero failures or preemptions. Only the
reviewed `r5` request was submitted in `cw-us-east-02a`.

`result.json` is the complete canonical task result. Its prototype payload also
contains prior Map and RMSNorm worker results; the claim here is limited to the
validated `rmsnorm_contract` block. `facts.json` records the bounded topology,
observer, diagnostic-equality, and worst-metric summary. Check the artifact
with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-rmsnorm-contract-hybrid-h100-plugin-20260821 && shasum -a 256 -c SHA256SUMS)
```

This is one-device functional evidence for the listed small shapes. It is not
representative-shape coverage, a performance result, Transformer Engine parity,
or bitwise-JAX parity for composed cotangents.
