// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rank-n-fold-map-fixtures.py
// Generator source SHA-256: CDD8A69852B59F7AA4CDD20599E50EFF0A6435601F0BAF2328F6AD8DE294E064
// Source: jax.jit(ordinary JAX terminal-axis Fold/Map plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: rank_n_fold_map_3dad9ca7fbf09d1a
// Structural fields: {"boundary":"backward","dimensions":[2,5,17]}
// Inputs: x=(2, 5, 17):bfloat16, output_cotangent=(2, 5, 17):bfloat16
// Outputs: input_cotangent=(2, 5, 17):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 053780F56B90BF6A93A84B4DD7AB4F8A7B328AEA3F50E3C95BF6D41DE1392A6F
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: C0991D2854375FBA728840F5E314CFEE87CED704CEB2ED85478216CE3353CE43
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x5x17xbf16>, %arg1: tensor<2x5x17xbf16>) -> (tensor<2x5x17xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %3 = stablehlo.convert %arg1 : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<2x5x17xf32>
    %5 = stablehlo.multiply %3, %2 : tensor<2x5x17xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %8 = stablehlo.add %5, %7 : tensor<2x5x17xf32>
    %9 = stablehlo.convert %8 : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %9 : tensor<2x5x17xbf16>
  }
}
