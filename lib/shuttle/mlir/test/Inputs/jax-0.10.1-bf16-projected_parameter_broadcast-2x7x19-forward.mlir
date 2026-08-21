// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-projected-parameter-broadcast-fixtures.py
// Generator source SHA-256: FE23C38D3CD3EBC98DA50BC5BCF664278CB922B9161C198693D2000979B42FBA
// Source: jax.jit(ordinary JAX RMS normalization with projected BF16
// weight plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: projected_parameter_broadcast_d607f43a0b672c7e
// Structural fields: {"boundary":"forward","dimensions":[2,7,19],"weight_dimensions":[19]}
// Inputs: x=(2, 7, 19):bfloat16, weight=(19,):bfloat16
// Outputs: result=(2, 7, 19):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: BEECCE1450322AB3A2E1995C25CCCFA9BBE87AB550DF819C305F4ED1F13E79F1
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: C90D1BD79B754C0C362F3B0E73B3D962BF26E4CBE87EA41856AD8BBA9EDDF6A3
module @jit_rms_norm_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x7x19xbf16>, %arg1: tensor<19xbf16>) -> (tensor<2x7x19xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x7x19xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %cst_0 = stablehlo.constant dense<1.900000e+01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x7x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<2x7x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x7x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<2x7x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<2x7x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<2x7x19xf32>
    %11 = stablehlo.convert %arg1 : (tensor<19xbf16>) -> tensor<19xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<2x7x19xf32>
    %15 = stablehlo.convert %14 : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %15 : tensor<2x7x19xbf16>
  }
}
