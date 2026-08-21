// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-projected-parameter-broadcast-fixtures.py
// Generator source SHA-256: FE23C38D3CD3EBC98DA50BC5BCF664278CB922B9161C198693D2000979B42FBA
// Source: jax.jit(ordinary JAX RMS normalization with projected BF16
// weight plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: projected_parameter_broadcast_764e986533883970
// Structural fields: {"boundary":"backward","dimensions":[2048,4096],"weight_dimensions":[4096]}
// Inputs: x=(2048, 4096):bfloat16, weight=(4096,):bfloat16, output_cotangent=(2048, 4096):bfloat16
// Outputs: input_cotangent=(2048, 4096):bfloat16, weight_cotangent=(4096,):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 16B50B7EE606000F7938ECF6119D8DB16EDD02B149CA5A596A0D339A47564D33
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: DC313DC00BC66CA6568FD4246CD2C4B4D598DE0B87091144EC249D85C97664AA
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x4096xbf16>, %arg1: tensor<4096xbf16>, %arg2: tensor<2048x4096xbf16>) -> (tensor<2048x4096xbf16> {jax.result_info = "result[0]"}, tensor<4096xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2048x4096xbf16>) -> tensor<2048x4096xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2048x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %cst_0 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<2048x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<2048x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<2048x1xf32>
    %9 = stablehlo.divide %8, %7 : tensor<2048x1xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<2048x1xf32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %13 = stablehlo.multiply %0, %12 : tensor<2048x4096xf32>
    %14 = stablehlo.convert %arg1 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %16 = stablehlo.convert %arg2 : (tensor<2048x4096xbf16>) -> tensor<2048x4096xf32>
    %17 = stablehlo.multiply %13, %16 : tensor<2048x4096xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%17 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %19 = stablehlo.reshape %18 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %20 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %21 = stablehlo.multiply %16, %20 : tensor<2048x4096xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%19 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %23 = stablehlo.convert %22 : (tensor<4096xf32>) -> tensor<4096xbf16>
    %24 = stablehlo.multiply %0, %21 : tensor<2048x4096xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%24 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %26 = stablehlo.reshape %25 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %27 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %28 = stablehlo.multiply %21, %27 : tensor<2048x4096xf32>
    %29 = stablehlo.multiply %26, %11 : tensor<2048x1xf32>
    %cst_6 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %31 = stablehlo.divide %29, %30 : tensor<2048x1xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%31 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<2048x1xf32>, tensor<f32>) -> tensor<2048xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<2048xf32>) -> tensor<2048x4096xf32>
    %34 = stablehlo.multiply %0, %33 : tensor<2048x4096xf32>
    %35 = stablehlo.add %28, %34 : tensor<2048x4096xf32>
    %36 = stablehlo.multiply %33, %0 : tensor<2048x4096xf32>
    %37 = stablehlo.add %35, %36 : tensor<2048x4096xf32>
    %38 = stablehlo.convert %37 : (tensor<2048x4096xf32>) -> tensor<2048x4096xbf16>
    return %38, %23 : tensor<2048x4096xbf16>, tensor<4096xbf16>
  }
}
