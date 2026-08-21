# Track 2 Shuttle RMSNorm r5 acceptance

Track 2 Shuttle r5 accepted the reviewed representative RMSNorm policy for the
predeclared `heldout_splitmix64_v3` input at shape `[2048,4096]` and epsilon
`1e-6`. Standalone Shuttle IR-to-CUDA execution completed forward, backward,
and composed execution twice under both SOURCE_ORDERED and FAST. Every
policy, boundary, and role repeat coordinate was bit-identical, and all 24
output observations passed the frozen finite, partition, absolute, mean,
relative, and masked-BF16-ULP gates.

The sealed Iris job `/dlwh/shuttle-track2-rmsnorm-h100-r5` completed one task
with exit 0 in 4 minutes 7.37 seconds, with no failures or preemptions. It ran
on `cw-us-east-02a` with batch priority, one H100, one replica, and zero
failure, preemption, and task-failure budgets. The exact request SHA-256 is
`647c399405f1a7c0df2cbd693536e3ec65d8c4e1de827065376d4a72b23e2092`;
the bundle SHA-256 is
`306649fbf0bcd40ee29bbcdad0593d6adee8d52525680c6ceada8bf3a50d436d`.
The executed source was commit
`e916b99f0f35d0fc32604fd5fa59296e6fd53bf0`, tree
`a50f12f97555efd2e569bc3cca2cd96865877d1d`.

## Numerical and resource result

The output digests are invariant across policy, boundary, and repeat for each
logical role:

- `y`: `67cd51f435f0dfa1da859ee1672ac59c9d69fb654013e7213e170b88bb0530a3`;
  maximum absolute error `0.015625`, mean absolute error
  `2.8492650017142296e-8`, masked relative error
  `0.007575757801532745`, masked BF16 distance 1 ULP.
- `dx`: `a10e4dd45e3b8324c3fd6e3e29340d32ec9a791f5fd548edfb9b370ca573714a`;
  maximum absolute error `0.015625`, below-floor maximum
  `0.00048828125`, mean absolute error `4.962455690815659e-8`, masked
  relative error `0.0078125`, masked BF16 distance 1 ULP.
- `dgamma`: `28d69308ae4d26785592deb62547d5db17ba50c20c8a21b6e4108349929f55f6`;
  exact match to the independent reference.

Resources match the frozen plan under both policies: forward is 1 launch, 3
slots, and no temporary; backward is 3 launches, 6 slots, and one 8,192-byte
temporary; composed is 4 launches, 7 slots, and one 8,192-byte temporary.
No Contract operation was executed.

## Timing and external comparison

The following medians combine the two 20-sample repeats for each Shuttle
coordinate. CUDA-event samples are supplemental; host-blocking is the declared
comparison clock.

| Boundary | Policy | Shuttle host ns | Shuttle CUDA-event ns | r12 TE host ns | r12 XLA host ns | Shuttle / TE | Shuttle / XLA |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward | SOURCE_ORDERED | 410865.5 | 379520 | 15254.5 | 149578 | 26.9341 | 2.74683 |
| forward | FAST | 409944 | 378976 | 15254.5 | 149578 | 26.8736 | 2.74067 |
| backward | SOURCE_ORDERED | 870498 | 791280 | 49709 | 157552 | 17.5119 | 5.52515 |
| backward | FAST | 870520.5 | 790544 | 49709 | 157552 | 17.5123 | 5.52529 |
| composed | SOURCE_ORDERED | 1270132.5 | 1166880 | 50155 | 274613 | 25.3241 | 4.62517 |
| composed | FAST | 1271039 | 1167248 | 50155 | 274613 | 25.3422 | 4.62847 |

This is exploratory, threshold-free comparison evidence. r12 and r5 used the
same H100 80GB model and driver `595.71.05`, but not the same physical GPU:
r12 used `GPU-ead480da-98e4-4844-ce40-ae9448088fc1`, while r5 used
`GPU-d9908860-321d-9964-565b-d527fb57580a`. The inherited result label
`same_h100_host_blocking_reported_without_threshold` is therefore interpreted
as matched H100 model and host-blocking protocol, not same-device evidence.
No performance threshold is accepted.

## Preserved evidence

`batch-result.json` and `task.log.gz` preserve the canonical worker result
and controller record. `shuttle-representative.json.gz` and
`shuttle-policy-raw-v1.json.gz` decompress to the exact report bytes whose
SHA-256 and sizes are recorded by the worker. `receipt.json`, `job-summary.txt`,
`terminal-summary.json`, and `comparison.json` preserve submission,
terminal, and derived comparison identities. `SHA256SUMS` seals every
payload.

The ephemeral BF16 output arrays and the worker's repeat-diagnostics file were
not copied from the worker. Their exact output digests and the diagnostics
file's SHA-256/size remain in the canonical result, but pointwise metrics
cannot now be recomputed from raw output bytes. The 19,290,236-byte request
remains locally at
`/private/tmp/shuttle-representative-request-east-r5.7goepG/request.pb`;
it is not embedded recursively in this source bundle.

The accepted claim is limited to Shuttle RMSNorm F/B/C at the exact held-out
input, shape, epsilon, policies, toolchain, and H100 model. Norm-to-Contract is
excluded.
