// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-layer-norm-fixtures.py
// Generator source SHA-256: E95957B1E27FB302E1AF70041353D4AE91CDEDAD3A693A4EAEE4EBF6D704F58A
// Source: jax.jit(ordinary JAX LayerNorm plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: layer_norm_44d152ecc3e9ff18
// Structural fields: {"boundary":"forward","epsilon":1e-05,"features":4096,"rows":2048,"shape_role":"target2_representative_shape"}
// Inputs: x=(2048, 4096):bfloat16, gamma=(4096,):bfloat16, beta=(4096,):bfloat16
// Outputs: y=(2048, 4096):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: E53EBEDF91FEB3345CFF3B21D8A28AA1232928C923ED785076FC3DEB9E5E99AD
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 595AC4F743612B824F15CAD15C417B84CFE0A2027A58D45CF9311703A26DD25E
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x4096xbf16>, %arg1: tensor<4096xbf16>, %arg2: tensor<4096xbf16>) -> (tensor<2048x4096xbf16> {jax.result_info = "result"}) {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<2048x4096xf32>
    %17 = stablehlo.convert %arg1 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %20 = stablehlo.multiply %16, %19 : tensor<2048x4096xf32>
    %21 = stablehlo.convert %arg2 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %24 = stablehlo.add %20, %23 : tensor<2048x4096xf32>
    %25 = stablehlo.convert %24 : (tensor<2048x4096xf32>) -> tensor<2048x4096xbf16>
    return %25 : tensor<2048x4096xbf16>
  }
}
