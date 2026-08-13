// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-layer-norm-fixtures.py
// Generator source SHA-256: E95957B1E27FB302E1AF70041353D4AE91CDEDAD3A693A4EAEE4EBF6D704F58A
// Source: jax.jit(ordinary JAX LayerNorm plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: layer_norm_44d152ecc3e9ff18
// Structural fields: {"boundary":"composed","epsilon":1e-05,"features":4096,"rows":2048,"shape_role":"target2_representative_shape"}
// Inputs: x=(2048, 4096):bfloat16, gamma=(4096,):bfloat16, beta=(4096,):bfloat16, dy=(2048, 4096):bfloat16
// Outputs: y=(2048, 4096):bfloat16, dx=(2048, 4096):bfloat16, dgamma=(4096,):bfloat16, dbeta=(4096,):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 4FB6D417CAF0C551EBA590AC69E2D85BE7E12B3282B768ADB8F75780D52E7EC7
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 2895432EBA01D18C409D14A14F0C650D1521C42B0AAD8783575BD409E3611745
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x4096xbf16>, %arg1: tensor<4096xbf16>, %arg2: tensor<4096xbf16>, %arg3: tensor<2048x4096xbf16>) -> (tensor<2048x4096xbf16> {jax.result_info = "result[0]"}, tensor<2048x4096xbf16> {jax.result_info = "result[1]"}, tensor<4096xbf16> {jax.result_info = "result[2]"}, tensor<4096xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2048x4096xbf16>) -> tensor<2048x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %cst_0 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<2048x1xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<2048x4096xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<2048x4096xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %cst_2 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %11 = stablehlo.divide %9, %10 : tensor<2048x1xf32>
    %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %13 = stablehlo.add %11, %12 : tensor<2048x1xf32>
    %14 = stablehlo.rsqrt %13 : tensor<2048x1xf32>
    %15 = stablehlo.divide %14, %13 : tensor<2048x1xf32>
    %cst_4 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %17 = stablehlo.multiply %16, %15 : tensor<2048x1xf32>
    %18 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %19 = stablehlo.multiply %6, %18 : tensor<2048x4096xf32>
    %20 = stablehlo.convert %arg1 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %23 = stablehlo.multiply %19, %22 : tensor<2048x4096xf32>
    %24 = stablehlo.convert %arg2 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %27 = stablehlo.add %23, %26 : tensor<2048x4096xf32>
    %28 = stablehlo.convert %27 : (tensor<2048x4096xf32>) -> tensor<2048x4096xbf16>
    %29 = stablehlo.convert %arg3 : (tensor<2048x4096xbf16>) -> tensor<2048x4096xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.reduce(%29 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %31 = stablehlo.reshape %30 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%31 init: %cst_6) applies stablehlo.add across dimensions = [0] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %33 = stablehlo.convert %32 : (tensor<4096xf32>) -> tensor<4096xbf16>
    %34 = stablehlo.multiply %19, %29 : tensor<2048x4096xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.reduce(%34 init: %cst_7) applies stablehlo.add across dimensions = [0] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %36 = stablehlo.reshape %35 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %37 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %38 = stablehlo.multiply %29, %37 : tensor<2048x4096xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %39 = stablehlo.reduce(%36 init: %cst_8) applies stablehlo.add across dimensions = [0] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %40 = stablehlo.convert %39 : (tensor<4096xf32>) -> tensor<4096xbf16>
    %41 = stablehlo.multiply %6, %38 : tensor<2048x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.reduce(%41 init: %cst_9) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %43 = stablehlo.reshape %42 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %44 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %45 = stablehlo.multiply %38, %44 : tensor<2048x4096xf32>
    %46 = stablehlo.multiply %43, %17 : tensor<2048x1xf32>
    %cst_10 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %48 = stablehlo.divide %46, %47 : tensor<2048x1xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.reduce(%48 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<2048x1xf32>, tensor<f32>) -> tensor<2048xf32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0] : (tensor<2048xf32>) -> tensor<2048x4096xf32>
    %51 = stablehlo.multiply %6, %50 : tensor<2048x4096xf32>
    %52 = stablehlo.add %45, %51 : tensor<2048x4096xf32>
    %53 = stablehlo.multiply %50, %6 : tensor<2048x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<2048x4096xf32>
    %55 = stablehlo.negate %54 : tensor<2048x4096xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %56 = stablehlo.reduce(%55 init: %cst_12) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %57 = stablehlo.reshape %56 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %cst_13 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %59 = stablehlo.divide %57, %58 : tensor<2048x1xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_14) applies stablehlo.add across dimensions = [1] : (tensor<2048x1xf32>, tensor<f32>) -> tensor<2048xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0] : (tensor<2048xf32>) -> tensor<2048x4096xf32>
    %62 = stablehlo.add %54, %61 : tensor<2048x4096xf32>
    %63 = stablehlo.convert %62 : (tensor<2048x4096xf32>) -> tensor<2048x4096xbf16>
    return %28, %63, %40, %33 : tensor<2048x4096xbf16>, tensor<2048x4096xbf16>, tensor<4096xbf16>, tensor<4096xbf16>
  }
}
