# Track 1 Map-only H100 prototype

These three JSON files preserve the exact extracted result bytes from the
successful batch jobs:

- `r9-result.json`: `/dlwh/shuttle-track1-map-h100-prototype-r9`, exit 0,
  zero failures and preemptions, 15 minutes 16.84 seconds.
- `r10-result.json`: `/dlwh/shuttle-track1-map-h100-prototype-r10`, exit 0,
  zero failures and preemptions, 29 minutes 17.49 seconds.
- `r11-result.json`: `/dlwh/shuttle-track1-map-h100-prototype-r11`, exit 0,
  zero failures and preemptions, 28 minutes 48.79 seconds.

Each result records the source that executed inside the sealed job:
integration commit `41502f2079385df29fccbf1137bce553aa090297`, tree
`a1c4babf2b5822ac8f1a96b0d6a133b61426d72a`. The reviewed evidence closure
continued through commit `715dfe1a27db38f16a911cc40820a4ed1c14e12d`.
Both are ancestors of the preservation base commit
`5047e923c9b90055feed9c07e29ecec69f578984`, tree
`1532b805a904f4d7c1ffa953d56a1985ef640df4`. Do not report the preservation
base as the code executed by the Track 1 jobs.

All three runs used authorization SHA-256
`cb7231dfbd997ccbb23ce923d06a0670bc2133316a7a3f88984bcecbdcf0b75b`
and bundle SHA-256
`2c88c14732d42fdb2d8a8cc8d116b6fa4a1b05e9b704aa83e664c0c210fb3e55`.
Their request SHA-256 values were, respectively,
`2afc795c99a73d5c3ab77ca999d42b3b4143f3bdf24a41509a8ca04b8d8820df`,
`6457368b2b0fcf20ecb2da0fb5362866562c414e61ee58a474d0831226410fad`,
and `e5605432fe1aa2b3ff620557dac956dc5912dcd6aa12424fb3a97e2b45cf8044`.
The request protobufs are not preserved here.

The files include the newline added when the canonical job output was
extracted. `r9-result.json` therefore has file SHA-256
`025251f00697e5fa591a891f49485c2fe94594aec581a9772aa9cd7e6680c78f`,
while the 9,346 canonical JSON bytes before that newline have SHA-256
`2d863b350ee287b76dfcbdab88700cb62834f0ad7d66116566e3e222e9779e19`.
`SHA256SUMS` binds the preserved file bytes. Independent wheel builds had
different wheel hashes across r9, r10, and r11 even though output, observer,
and persistent-cache evidence matched. That is a remaining build
reproducibility issue, not evidence that the runtime behavior differed.

The subject is an ordinary `jax.jit` function that returns a mixed-Map primal
and an analytic gradient. It proves the ordinary JAX/XLA module-transform,
typed-FFI, generated-PTX, and fallback wiring. It does not prove that
JAX-owned AD was transformed, and it contains no performance measurement.
