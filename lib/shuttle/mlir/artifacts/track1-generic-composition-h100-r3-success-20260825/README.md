# Ordinary-JAX generic composition H100 r3 success

r3 executed all six rank-2/rank-3 and disabled/SOURCE_ORDERED/FAST
coordinates under the prospective v2 numerical policy. Both enabled policies
passed the unchanged maximum, mean, and relative-error gates. The known
`rank3-b/disabled` ordinary-XLA miss remained a required diagnostic and did not
participate in Shuttle acceptance.

## Identity

- Job: `/dlwh/shuttle-track1-generic-composition-h100-canary-r3`
- Cluster: `cw-us-east-02a`
- Executed source commit: `a2553850f6b1aa6c4b0b20e006305bfb6d6d27d0`
- Executed source tree: `3c5cbcaee6cf468ab942ebd40cb501e0c89d5184`
- Source manifest SHA-256: `c5dba478034a6174ff3d7b4615e7126e3752e60c490e2550731eb4c335bfb5a1`
- Authorization SHA-256: `388c7458ab88f75f6f533443df3f13c399ea3cdb335f6e13754b1f08466ccd2d`
- Numerical policy SHA-256: `78578508ee2106dccc205d6338a6ca94b0ee4d4562766b8a2bf11eae51e11984`
- Request SHA-256: `cd0fdcb01019f6dd38cc9062393bfab7689cb285c30867709adbea466cbf2b75`
- Request size: 20,915,530 bytes
- Bundle SHA-256: `fbfe32effd21a94b8d4b1c7f2a9fd6eee5e179241c9a17c113e781d0bd505778`
- Bundle size: 20,914,519 bytes
- Canonical result SHA-256: `eb4782ba2eee9d69e29e7abaf616ab483104953fa44527d572da0846da8b0cdb`
- Canonical result size: 84,478 bytes

The exact request protobuf is preserved locally at
`/Users/dlwh/src/marin/scratch/20260825-1720_generic_composition_r3_launch.pb`.
It is not checked in because its embedded source bundle is 20.9 MB and exceeds
the repository's 500 KiB new-file gate. `request-metadata.json` binds those
bytes by SHA-256 and size; the reviewed source and authorization rebuild them
deterministically.

## Result

Iris recorded one successful task, exit code 0, zero failures, zero
preemptions, and 958.38 seconds runtime. All six result and diagnostic records
are canonical and pass the production validators. Every coordinate passed f32
ABI, exact shape, finiteness, bitwise repeatability, source identity, and the
mode-specific observer contract. Enabled observer invocation IDs were 1
through 4.

| Case | Mode | Maximum absolute | Mean absolute | Relative Linf | Acceptance failures |
| --- | --- | ---: | ---: | ---: | --- |
| rank2-a | disabled | 4.6509672675665303e-10 | 1.21668790178024e-10 | 5.029568796942033e-7 | none |
| rank2-a | SOURCE_ORDERED | 6.801429590971159e-11 | 2.5995844000467204e-11 | 7.355084668924177e-8 | none |
| rank2-a | FAST | 6.801429590971159e-11 | 2.4981979878570606e-11 | 7.355084668924177e-8 | none |
| rank3-b | disabled | 0.0006204024596327962 | 0.00011096225522005224 | 0.00021940363857290523 | diagnostic only |
| rank3-b | SOURCE_ORDERED | 9.150818462977384e-7 | 9.081548102228527e-8 | 3.236162003428692e-7 | none |
| rank3-b | FAST | 1.5652610008132228e-7 | 2.7371283869245187e-8 | 5.53550285886929e-8 | none |

The rank3 disabled coordinate exceeded the v2 limit-comparison fields for
maximum and mean absolute error, matching r2. Disabled ordinary XLA is a
required diagnostic, not the Shuttle numerical acceptance source. The four
enabled cells passed all three unchanged gates and had zero raw and masked
BF16 ULP distance.

The result observed two launches and two temporaries for each enabled cell.
Those counts are telemetry only. The semantic-group and private `{h,s}`
boundary fields are policy/harness assertions, not independent device topology
measurements. This single run does not establish performance, representative
shape behavior, or a preferred decomposition.

Verify the preserved files with:

```bash
(cd lib/shuttle/mlir/artifacts/track1-generic-composition-h100-r3-success-20260825 && shasum -a 256 -c SHA256SUMS)
```
