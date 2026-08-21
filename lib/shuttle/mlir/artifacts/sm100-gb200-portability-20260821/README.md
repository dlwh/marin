# SM100 GB200 portability gate

Iris job `/dlwh/shuttle-gb200-portability-r4` succeeded on one batch-priority
NVIDIA GB200 with exit 0, zero failures, and zero preemptions. The task ran for
269.97 seconds. The device reported compute capability 10.0 and driver
`595.71.05`.

The job executed production commit
`43b52d73d2eb13b28c765bda2bfd5ad28af9ceab`. Gate commit
`f3342153b37f12dbed2f81eb39d7648898975b89`, tree
`abb539d85b53ba38b3e297b7756a95327574ae3d`, contains the reviewed hardware
fixture and sealed-request builder. The executed production source remained
the commit above. The source manifest SHA-256 was
`c2571612301c6346cd9bef34635b29fc8eec81a5d8e6635ebe6db0f03e1b74ea`.

The ABI23 `prefer_fusion` pipeline generated schema 4
`cuda_ptx_sm100_v1` artifacts with PTX 8.7 and target `sm_100`. The
ordinary-JAX Map fixture performed three launches and had maximum absolute
error zero. The RMSNorm `[2,7,19]` forward fixture performed one launch and
had maximum BF16 ULP error zero. `result.json` records the exact policy,
tuning, PTX, transport, XLA patch, image, Bazel, CUDA, source, and runtime
identities.

The sealed `request.pb` is omitted because it is 6,363,268 bytes. Its
SHA-256 is
`746f0f4516311ffce7d5cb1e4f6caba06c94a2e380d74f0760a1cfb06b45718c`.
`request-metadata.json` preserves this identity and omission explicitly.
The result file includes the newline added during extraction; its file
SHA-256 is
`f4fc30b9feff78516ca87de827512414ecfa9d9ffd740749f41f24a3e9a96773`.
The 3,989 canonical JSON bytes before that newline have SHA-256
`5b3902ae4d04fe3fddd794d545c251bb0a7f767cd5f87111df03c1348e23a534`.

`controller-sanitized.log` retains only the canonical result marker. It
drops local CLI warnings, tunnel addresses, and log-client setup lines from
the captured controller output. `job-summary.txt` and
`submission-receipt.json` preserve the terminal state and submitted request
identity. `SHA256SUMS` binds every preserved evidence file except itself.

This is a one-device correctness and portability result for SOURCE_ORDERED,
one structurally eligible BF16-boundary Map fixture, and one small RMSNorm
forward fixture. It is not FAST-policy, backward, performance, compile-cost,
persistent-cache-reuse, representative-shape, resource-usage, or SASS
evidence. It does not establish SM100-only optimization quality.
