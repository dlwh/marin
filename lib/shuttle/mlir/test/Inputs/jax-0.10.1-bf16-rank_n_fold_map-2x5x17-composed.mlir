// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rank-n-fold-map-fixtures.py
// Generator source SHA-256: CDD8A69852B59F7AA4CDD20599E50EFF0A6435601F0BAF2328F6AD8DE294E064
// Source: jax.jit(ordinary JAX terminal-axis Fold/Map plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: rank_n_fold_map_3dad9ca7fbf09d1a
// Structural fields: {"boundary":"composed","dimensions":[2,5,17]}
// Inputs: x=(2, 5, 17):bfloat16, output_cotangent=(2, 5, 17):bfloat16
// Outputs: result=(2, 5, 17):bfloat16, input_cotangent=(2, 5, 17):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 1F9B192E08CBD3F6A2010C9BBB39FD402CCF5041081E6E26597951F6A5C48CE6
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 1083EECB15397C4778442C4ED041AB11192D73AF001D329B69670D62064BA5CD
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x5x17xbf16>, %arg1: tensor<2x5x17xbf16>) -> (tensor<2x5x17xbf16> {jax.result_info = "result[0]"}, tensor<2x5x17xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<2x5x17xf32>
    %4 = stablehlo.convert %3 : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    %5 = stablehlo.convert %arg1 : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<2x5x17xf32>
    %7 = stablehlo.multiply %5, %2 : tensor<2x5x17xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %10 = stablehlo.add %7, %9 : tensor<2x5x17xf32>
    %11 = stablehlo.convert %10 : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %4, %11 : tensor<2x5x17xbf16>, tensor<2x5x17xbf16>
  }
}
