# Track 2 H100 artifact preservation

This directory preserves the exact files from the Track 2 r11 handoff except
for `request.pb`. The original `README.md`, `manifest.json`, reports, receipt,
job summary, cancellation proof, and controller log are byte-for-byte copies.
`SHA256SUMS` binds the preserved copies.

The omitted request was 5,990,891 bytes with SHA-256
`412ea5c918495808a509adafb9b5f96a53a2ad9ae8c37872c0e29740832d77ad`.
The exact request identity also appears in `manifest.json`, `result.json`, and
`submission-receipt.json`. Its source bundle SHA-256 was
`d3d58a497cd02218e57a2d1d1809009e2012c450f37003bdf42dfb200ce685cc`.

The job executed commit `5047e923c9b90055feed9c07e29ecec69f578984`, tree
`1532b805a904f4d7c1ffa953d56a1985ef640df4`, with source-manifest SHA-256
`0fdcec09940656ba706e3bfd1fcd3e95c3af2a53d68ce7ba4f719ac7e4183cb7`.
Unlike Track 1, the executed Track 2 source is the preservation base itself.

This is standalone backend evidence. Shuttle consumes ordinary-JAX-generated
StableHLO fixtures and executes generated ABI22 Fold/Contract bundles through
`GpuRunner`; it does not traverse the ordinary `jax.jit` plugin seam. The XLA
comparison uses the same GPU and host-blocking clock but remains exploratory.
Transformer Engine 2.17 was unavailable for these shapes, epsilon, and
norm-to-Contract boundaries, so this artifact cannot establish matched-oracle
performance acceptance.
