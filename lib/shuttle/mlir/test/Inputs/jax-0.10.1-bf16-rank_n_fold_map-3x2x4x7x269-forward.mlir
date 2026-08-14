// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rank-n-fold-map-fixtures.py
// Generator source SHA-256: CDD8A69852B59F7AA4CDD20599E50EFF0A6435601F0BAF2328F6AD8DE294E064
// Source: jax.jit(ordinary JAX terminal-axis Fold/Map plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: rank_n_fold_map_ab45c3d1ac958a22
// Structural fields: {"boundary":"forward","dimensions":[3,2,4,7,269]}
// Inputs: x=(3, 2, 4, 7, 269):bfloat16
// Outputs: result=(3, 2, 4, 7, 269):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 6DFA79F378922E466BF4A15782B35CE7B361BBCA27F41EC63A5D9331EF3E8417
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: D72C5C4F903D17CC20F0A14990947CD820E22A29E34DD4EF63AF7A2AF4A0C341
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x4x7x269xbf16>) -> (tensor<3x2x4x7x269xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x2x4x7x269xbf16>) -> tensor<3x2x4x7x269xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [4] : (tensor<3x2x4x7x269xf32>, tensor<f32>) -> tensor<3x2x4x7xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<3x2x4x7xf32>) -> tensor<3x2x4x7x269xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3x2x4x7x269xf32>
    %4 = stablehlo.convert %3 : (tensor<3x2x4x7x269xf32>) -> tensor<3x2x4x7x269xbf16>
    return %4 : tensor<3x2x4x7x269xbf16>
  }
}
