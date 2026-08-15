// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-projected-parameter-broadcast-fixtures.py
// Generator source SHA-256: 6CD255DE5BC6E9663FD4340955F6E7950050F29B501798AED66DBBEE3A4ADFB9
// Source: jax.jit(ordinary JAX RMS normalization with projected BF16
// weight plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: projected_parameter_broadcast_a3b7dd4c87851f78
// Structural fields: {"boundary":"composed","dimensions":[3,2,5,11,263],"weight_dimensions":[263]}
// Inputs: x=(3, 2, 5, 11, 263):bfloat16, weight=(263,):bfloat16, output_cotangent=(3, 2, 5, 11, 263):bfloat16
// Outputs: result=(3, 2, 5, 11, 263):bfloat16, input_cotangent=(3, 2, 5, 11, 263):bfloat16, weight_cotangent=(263,):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: AA322F1D20234D6D30F8F9F25926E801B6041B989C2C5B7AE72605E4801AE265
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 7FB7DD8C19F490E752C3B5691A48C16A5D8F6033E1A36A4728953B01AFA21BB5
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x5x11x263xbf16>, %arg1: tensor<263xbf16>, %arg2: tensor<3x2x5x11x263xbf16>) -> (tensor<3x2x5x11x263xbf16> {jax.result_info = "result[0]"}, tensor<3x2x5x11x263xbf16> {jax.result_info = "result[1]"}, tensor<263xbf16> {jax.result_info = "result[2]"}) {
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
    %9 = stablehlo.divide %8, %7 : tensor<3x2x5x11x1xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<3x2x5x11x1xf32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %13 = stablehlo.multiply %0, %12 : tensor<3x2x5x11x263xf32>
    %14 = stablehlo.convert %arg1 : (tensor<263xbf16>) -> tensor<263xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [4] : (tensor<263xf32>) -> tensor<1x1x1x1x263xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x263xf32>) -> tensor<3x2x5x11x263xf32>
    %17 = stablehlo.multiply %13, %16 : tensor<3x2x5x11x263xf32>
    %18 = stablehlo.convert %17 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    %19 = stablehlo.convert %arg2 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %20 = stablehlo.multiply %13, %19 : tensor<3x2x5x11x263xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_3) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<263xf32>
    %22 = stablehlo.reshape %21 : (tensor<263xf32>) -> tensor<1x1x1x1x263xf32>
    %23 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x263xf32>) -> tensor<3x2x5x11x263xf32>
    %24 = stablehlo.multiply %19, %23 : tensor<3x2x5x11x263xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%22 init: %cst_4) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x1x1x263xf32>, tensor<f32>) -> tensor<263xf32>
    %26 = stablehlo.convert %25 : (tensor<263xf32>) -> tensor<263xbf16>
    %27 = stablehlo.multiply %0, %24 : tensor<3x2x5x11x263xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.reduce(%27 init: %cst_5) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %29 = stablehlo.reshape %28 : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %30 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<3x2x5x11x263xf32>
    %32 = stablehlo.multiply %29, %11 : tensor<3x2x5x11x1xf32>
    %cst_6 = stablehlo.constant dense<2.630000e+02> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %34 = stablehlo.divide %32, %33 : tensor<3x2x5x11x1xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.reduce(%34 init: %cst_7) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x1xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x263xf32>
    %37 = stablehlo.multiply %0, %36 : tensor<3x2x5x11x263xf32>
    %38 = stablehlo.add %31, %37 : tensor<3x2x5x11x263xf32>
    %39 = stablehlo.multiply %36, %0 : tensor<3x2x5x11x263xf32>
    %40 = stablehlo.add %38, %39 : tensor<3x2x5x11x263xf32>
    %41 = stablehlo.convert %40 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    return %18, %41, %26 : tensor<3x2x5x11x263xbf16>, tensor<3x2x5x11x263xbf16>, tensor<263xbf16>
  }
}
