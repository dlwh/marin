// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-layer-norm-fixtures.py
// Generator source SHA-256: E95957B1E27FB302E1AF70041353D4AE91CDEDAD3A693A4EAEE4EBF6D704F58A
// Source: jax.jit(ordinary JAX LayerNorm plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: layer_norm_81928ab3539c0f03
// Structural fields: {"boundary":"forward","epsilon":1e-05,"features":13,"rows":7,"shape_role":"structural_shape_mutation"}
// Inputs: x=(7, 13):bfloat16, gamma=(13,):bfloat16, beta=(13,):bfloat16
// Outputs: y=(7, 13):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 38DD84F2E582476C943CF7F2FCA30D7094BB57B433978FA64CC88D87F6DD44CC
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 64BBB24A43B085AA4DEF092B25A17F6005B712EE0E51472FA59DE901329C09BE
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<7x13xbf16>, %arg1: tensor<13xbf16>, %arg2: tensor<13xbf16>) -> (tensor<7x13xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<7x13xbf16>) -> tensor<7x13xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<7xf32>) -> tensor<7x1xf32>
    %cst_0 = stablehlo.constant dense<1.300000e+01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<7x1xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<7x13xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<7x13xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<7xf32>) -> tensor<7x1xf32>
    %cst_2 = stablehlo.constant dense<1.300000e+01> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %11 = stablehlo.divide %9, %10 : tensor<7x1xf32>
    %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %13 = stablehlo.add %11, %12 : tensor<7x1xf32>
    %14 = stablehlo.rsqrt %13 : tensor<7x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<7x13xf32>
    %17 = stablehlo.convert %arg1 : (tensor<13xbf16>) -> tensor<13xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<13xf32>) -> tensor<1x13xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x13xf32>) -> tensor<7x13xf32>
    %20 = stablehlo.multiply %16, %19 : tensor<7x13xf32>
    %21 = stablehlo.convert %arg2 : (tensor<13xbf16>) -> tensor<13xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<13xf32>) -> tensor<1x13xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x13xf32>) -> tensor<7x13xf32>
    %24 = stablehlo.add %20, %23 : tensor<7x13xf32>
    %25 = stablehlo.convert %24 : (tensor<7x13xf32>) -> tensor<7x13xbf16>
    return %25 : tensor<7x13xbf16>
  }
}
