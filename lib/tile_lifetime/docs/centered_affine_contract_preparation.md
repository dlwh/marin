# Centered affine Contract preparation

This checkpoint closes the generated-code and executable tensor-core boundary
for generic source-ordered Map preparation feeding an SM90 Contract.

The frontend and region planner erase centered affine normalization into two
row Folds followed by this consumer preparation:

```text
FP32(source) - row_center
  -> * row_scale
  -> * feature_scale
  -> + feature_bias
  -> BF16 round-to-nearest-even
  -> Contract mainloop
```

The three ordinary binary operations now carry an explicit right-operand
delivery domain. `row` means one FP32 value per M coordinate and `feature`
means one FP32 value per Contract K coordinate. `scale_row` retains its
existing row-scalar contract. These delivery domains are schedule data; the
generator does not inspect normalization or model names.

`cuda_prepared_contract_codegen.py` is the bounded executable specification.
It emits one generic CUDA kernel with separate M-row and K-feature inputs,
performs every preparation operation with explicit FP32 round-to-nearest
arithmetic, converts the prepared value with `__float2bfloat16_rn`, and only
then enters the ordered Contract accumulation. The scalar loop is intended for
correctness and code-ownership validation, not throughput.

The same generator accepts a mutation that changes the final feature-bias add
to subtraction. The operand family and loop structure remain unchanged while
the semantic and source digests change. Epsilon mutations remain upstream in
the generic Fold finalizer and feed a different row-scale value into this same
physical family.

## QuACK boundary

The generated QuACK A-transform source represents compact values in both
delivery domains and both physical dtypes:

- `mvec_bf16` and `mvec_fp32` for one value per Contract M coordinate;
- `kvec_bf16` and `kvec_fp32` for one value per Contract K coordinate.

The pinned patch applies cleanly to QuACK revision
`84ef91df9bec87c7e4938517234fafb07ef844dd`. It replaces the single scale-factor
slot with an ordered tuple of independently typed auxiliary tensors. The
extension owns:

1. a tuple of auxiliary TMA tensors and per-stage shared-memory layouts;
2. descriptor, barrier-byte, and pipeline accounting for every tuple member;
3. a stable mapping from each transform argument to its staged register view;
4. matching runtime and trace-time host views;
5. a cache key containing the operand delivery kinds and dtypes.

The compact host views alias the original `(M,)` and `(K,)` tensors; they do not
repeat a vector per tile or materialize the mapped activation. One shared
arrival barrier covers all auxiliary copies, with its expected byte count
derived from the complete tuple. Each staged register value retains its source
dtype until the generated scalar AST applies an explicit conversion.

This is not specific to centered normalization. The adapter receives only the
scalar AST, explicit Contract axes, leaf dtypes, and leaf delivery axes. The
SM90 schedule fingerprint is independent of the AST.

## Delayed algebraic candidate

The delayed alternative remains a separate real-algebra-equivalent plan:

```text
row_scale * Contract(source * feature_scale, W)
  - (row_center * row_scale) * Contract(feature_scale, W)
  + Contract(feature_bias, W)
```

It retains two parameter-vector projections and changes the BF16 cast and
reduction order. It is not labeled source-ordered and is legal only under the
rounding-reassociation policy. The new preparation generator does not rewrite
or collapse this candidate.

## Source lineage and status

The bounded CUDA source is generated entirely from `GemmProgram` tile
operations and generic operand-delivery attributes. It calls no external
semantic kernel and contains no normalization, Transformer, or workload name.

The tensor-core path derives from the pinned QuACK A-transform and GEMM
mainloop. It retains generic WGMMA, TMA, strip-layout, and pipeline machinery.
No named normalization callback or model control flow is retained.

CPU tests compare the generated preparation boundary and complete bounded
Contract exactly with natural JAX BF16 semantics. Static audits verify explicit
FP32 Fold-state delivery, source-ordered BF16 conversions, and typed auxiliary
bindings.

The H100 executable check evaluates
`bf16(bf16(activation * rsqrt(fold_state / K + epsilon)) * feature_scale)`
without mapped-activation or auxiliary-strip materialization. Against the full
FP32 Fold-state reference, maximum and mean absolute errors are zero at both
tested shapes. The generated median is 0.021208 ms for `256x512x512` and
0.140734 ms for `2048x4096x4096`, respectively 0.263x and 0.713x the eager
Map-then-Contract time. The generated kernel is 1.93x and 1.58x bare GEMM;
those ratios expose the actual transform cost rather than hiding it behind a
semantic special case. Raw samples, generated sources, requests, hashes, and
environment versions are preserved in
`benchmarks/artifacts/h100_typed_multi_aux_map_load_contract_20260903/`.
