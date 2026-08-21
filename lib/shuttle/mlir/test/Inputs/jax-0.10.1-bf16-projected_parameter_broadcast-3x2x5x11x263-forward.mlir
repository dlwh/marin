// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-projected-parameter-broadcast-fixtures.py
// Generator source SHA-256: FE23C38D3CD3EBC98DA50BC5BCF664278CB922B9161C198693D2000979B42FBA
// Source: jax.jit(ordinary JAX RMS normalization with projected BF16
// weight plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: projected_parameter_broadcast_a3b7dd4c87851f78
// Structural fields: {"boundary":"forward","dimensions":[3,2,5,11,263],"weight_dimensions":[263]}
// Inputs: x=(3, 2, 5, 11, 263):bfloat16, weight=(263,):bfloat16
// Outputs: result=(3, 2, 5, 11, 263):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 55F142CDB23EA617FE5908634553950A00613F2BF7285F113E3913062E9B549D
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 20E74792DA96BD9DAA327BCAF803274181825F9EB3D95F415D77494028A62D1C
module @jit_rms_norm_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x5x11x263xbf16>, %arg1: tensor<263xbf16>) -> (tensor<3x2x5x11x263xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<3x2x5x11x263xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %cst_0 = stablehlo.constant dense<2.630000e+02> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<3x2x5x11x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<3x2x5x11x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<3x2x5x11x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<3x2x5x11x263xf32>
    %11 = stablehlo.convert %arg1 : (tensor<263xbf16>) -> tensor<263xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [4] : (tensor<263xf32>) -> tensor<1x1x1x1x263xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x263xf32>) -> tensor<3x2x5x11x263xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<3x2x5x11x263xf32>
    %15 = stablehlo.convert %14 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    return %15 : tensor<3x2x5x11x263xbf16>
  }
}
