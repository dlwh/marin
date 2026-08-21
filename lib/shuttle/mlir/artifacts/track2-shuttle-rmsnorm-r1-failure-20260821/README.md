# Track 2 Shuttle RMSNorm r1 harness failure

Track 2 Shuttle representative r1 failed before the first H100 kernel launch.
The hermetic compiler and matrix targets built successfully. During
`shuttle_execute`, the inner shell wrapper received `--case-root` where it
expected the GPU transport test executable. It shifted that flag away and
rejected the bare `/tmp/shuttle-track2-h100/cases` argument.

The sealed Iris job `/dlwh/shuttle-track2-rmsnorm-h100-r1` ran on
`cw-rno2a` from source commit
`280a712d8572fcdda1abddf13c2bbae3e575eae8`, tree
`e258a93d4e164aeb5e57c47c140421a14a68bdf6`, and reviewed request SHA-256
`29e8be6a35d29d10904169fbce77446a4916402c572b995df8b688c2054e988d`.
The request used one H100, batch priority, one replica, fail-if-exists, and
zero failure, preemption, and task-failure budgets. Iris recorded task exit 1,
one task failure, and zero preemptions.

This artifact contains the exact reviewed request, submission receipt, terminal
job records, and compressed controller log. The log proves the build completed
and the first coordinate failed during argument parsing. r1 provides no GPU
numerical, kernel-resource, or performance evidence.

The reproduction is fixed by explicitly passing the GPU transport executable
in every per-coordinate argv. Referencing a Bazel `sh_binary` executable from
another target does not embed that target's default `args`.
