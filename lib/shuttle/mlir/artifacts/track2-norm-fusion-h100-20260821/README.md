# Track 2 H100 r11 handoff

Job `/dlwh/shuttle-track2-h100-r11` succeeded on `cw-rno2a` with exit 0,
zero failures, zero preemptions, and 239.77 seconds of task runtime. The device
was an NVIDIA H100 80GB HBM3, UUID
`GPU-cacbf168-b226-0ed4-0320-032bd2f43cc2`, driver `595.71.05`.

The sealed request SHA-256 is
`412ea5c918495808a509adafb9b5f96a53a2ad9ae8c37872c0e29740832d77ad`.
It binds source commit `5047e923c9b90055feed9c07e29ecec69f578984`, tree
`1532b805a904f4d7c1ffa953d56a1985ef640df4`, and source-manifest SHA-256
`0fdcec09940656ba706e3bfd1fcd3e95c3af2a53d68ce7ba4f719ac7e4183cb7`.

All eight Shuttle cells have zero maximum absolute error and zero BF16 ULP
error against the independent oracle.

## Shuttle CUDA-event measurements

Each median is over 20 measured samples after three excluded warmups. Units
are nanoseconds.

| Case | Policy | Median | Launches | Slots | Temporaries | Temporary bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RMSNorm forward | SOURCE_ORDERED | 8,592 | 1 | 3 | 0 | 0 |
| RMSNorm forward | FAST | 8,752 | 1 | 3 | 0 | 0 |
| RMSNorm backward | SOURCE_ORDERED | 20,032 | 3 | 6 | 1 | 56 |
| RMSNorm backward | FAST | 20,176 | 3 | 6 | 1 | 56 |
| Norm→Contract inline | SOURCE_ORDERED | 30,000 | 1 | 4 | 0 | 0 |
| Norm→Contract inline | FAST | 9,712 | 1 | 4 | 0 | 0 |
| Norm→Contract compact | SOURCE_ORDERED | 36,832 | 2 | 5 | 1 | 20 |
| Norm→Contract compact | FAST | 13,200 | 2 | 5 | 1 | 20 |

The full raw CUDA-event samples, counterbalance positions, PTX digests,
module-load digests, output digests, and resource counters are in
`shuttle-report.json` and `shuttle-validated.json`.

## Exploratory same-GPU host comparison

This is protocol `exploratory_host_v1`: host wall-clock timing with
host-blocking completion for both Shuttle and ordinary JAX/XLA, 20 measured
samples after three warmups, counterbalanced forward/reverse order. It does not
mix CUDA-event and host clocks.

| Case | Policy | Shuttle median ns | XLA median ns | Shuttle speedup |
| --- | --- | ---: | ---: | ---: |
| RMSNorm forward | SOURCE_ORDERED | 35,696.5 | 65,893.0 | 1.846× |
| RMSNorm forward | FAST | 35,615.0 | 65,443.5 | 1.838× |
| RMSNorm backward | SOURCE_ORDERED | 87,035.0 | 79,178.0 | 0.910× |
| RMSNorm backward | FAST | 87,144.0 | 78,344.0 | 0.899× |
| Norm→Contract inline | SOURCE_ORDERED | 63,725.5 | 71,202.0 | 1.117× |
| Norm→Contract inline | FAST | 45,419.5 | 71,657.5 | 1.578× |
| Norm→Contract compact | SOURCE_ORDERED | 70,118.0 | 72,206.5 | 1.030× |
| Norm→Contract compact | FAST | 48,138.0 | 72,248.5 | 1.501× |

TE 2.17 is unavailable because its runner lacks the required shapes, epsilon,
and norm→Contract workloads. Therefore this run does not satisfy or claim the
accepted XLA+TE CUDA-event threshold.

Backward output digests use Shuttle executable ABI order `{dw, dx}` but JAX
public result order `{dx, dw}`. The labeled digest sets and numerical results
match exactly; consumers must not compare those unlabeled arrays positionally.
The XLA SOURCE_ORDERED/FAST labels pair the same ordinary-JAX behavior with the
corresponding Shuttle cells; they do not claim XLA implements Shuttle's two
reduction policies.

## Files

- `controller.log`: raw Iris controller log containing the canonical result.
- `result.json`: parsed canonical `TRACK2_H100_RESULT`.
- `shuttle-report.json`: Shuttle raw CUDA-event and host samples.
- `shuttle-validated.json`: validator-normalized Shuttle report.
- `xla-report.json`: ordinary-JAX/XLA raw host-blocking samples.
- `exploratory-comparison.json`: validated same-clock comparison.
- `te-status.json`: explicit TE blocker.
- `request.pb` and `submission-receipt.json`: sealed request and submission identity.
- `job-summary.txt`: terminal Iris task summary.
- `east-cancellations.txt`: proof r9/r10 were Kueue-gated, had no worker or
  runtime duration, and were cancelled after r11 succeeded.
- `manifest.json`: SHA-256 and size for every preserved artifact except this
  human-readable handoff.
