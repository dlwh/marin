// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-attention-relation-fixtures.py (jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir)
// Export: jax.jit(fixture.function).lower(f32_shape_struct).compiler_ir(dialect="stablehlo")
// Expression: semantic mutation of the axis-1 fixture: scale 0.25 and softcap 7
// Input: (19, 23, 2, 6):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: AF63ED6BE34037C61A1142F97ED55B1069720E9EC4CEC51892DCDCF55AB24EA9
// Fixture StableHLO SHA-256: 1398ED9E4F90D5C5C61CC5A4765ABF13FA224371E9CF9A41CF9771535D799D33
// Textual SSA alpha-renaming: true

module @jit_offset_causal_score_relation_mutation attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%renamed_scores: tensor<19x23x2x6xf32>) -> (tensor<19x2x6xf32> {jax.result_info = "result[0]"}, tensor<19x1x1xf32> {jax.result_info = "result[1]"}) {
    %renamed_value_0 = stablehlo.iota dim = 0 : tensor<19xi32>
    %renamed_value_1 = stablehlo.reshape %renamed_value_0 : (tensor<19xi32>) -> tensor<19x1x1x1xi32>
    %renamed_value_2 = stablehlo.iota dim = 0 : tensor<23xi32>
    %renamed_value_3 = stablehlo.reshape %renamed_value_2 : (tensor<23xi32>) -> tensor<1x23x1x1xi32>
    %renamed_value_4 = stablehlo.constant dense<4> : tensor<i32>
    %renamed_value_5 = stablehlo.broadcast_in_dim %renamed_value_4, dims = [] : (tensor<i32>) -> tensor<19x1x1x1xi32>
    %renamed_value_6 = stablehlo.add %renamed_value_1, %renamed_value_5 : tensor<19x1x1x1xi32>
    %renamed_value_7 = stablehlo.broadcast_in_dim %renamed_value_3, dims = [0, 1, 2, 3] : (tensor<1x23x1x1xi32>) -> tensor<19x23x1x1xi32>
    %renamed_value_8 = stablehlo.broadcast_in_dim %renamed_value_6, dims = [0, 1, 2, 3] : (tensor<19x1x1x1xi32>) -> tensor<19x23x1x1xi32>
    %renamed_value_9 = stablehlo.compare LE, %renamed_value_7, %renamed_value_8, SIGNED : (tensor<19x23x1x1xi32>, tensor<19x23x1x1xi32>) -> tensor<19x23x1x1xi1>
    %renamed_value_10 = stablehlo.convert %renamed_value_9 : (tensor<19x23x1x1xi1>) -> tensor<19x23x1x1xf32>
    %renamed_value_11 = stablehlo.broadcast_in_dim %renamed_value_9, dims = [0, 1, 2, 3] : (tensor<19x23x1x1xi1>) -> tensor<19x23x2x6xi1>
    %renamed_value_12 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %renamed_value_13 = stablehlo.broadcast_in_dim %renamed_value_12, dims = [] : (tensor<f32>) -> tensor<19x23x2x6xf32>
    %renamed_value_14 = stablehlo.multiply %renamed_scores, %renamed_value_13 : tensor<19x23x2x6xf32>
    %renamed_value_15 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %renamed_value_16 = stablehlo.broadcast_in_dim %renamed_value_15, dims = [] : (tensor<f32>) -> tensor<19x23x2x6xf32>
    %renamed_value_17 = stablehlo.divide %renamed_value_14, %renamed_value_16 : tensor<19x23x2x6xf32>
    %renamed_value_18 = stablehlo.tanh %renamed_value_17 : tensor<19x23x2x6xf32>
    %renamed_value_19 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %renamed_value_20 = stablehlo.broadcast_in_dim %renamed_value_19, dims = [] : (tensor<f32>) -> tensor<19x23x2x6xf32>
    %renamed_value_21 = stablehlo.multiply %renamed_value_20, %renamed_value_18 : tensor<19x23x2x6xf32>
    %renamed_value_22 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %renamed_value_23 = stablehlo.broadcast_in_dim %renamed_value_22, dims = [] : (tensor<f32>) -> tensor<19x23x2x6xf32>
    %renamed_value_24 = stablehlo.select %renamed_value_11, %renamed_value_21, %renamed_value_23 : tensor<19x23x2x6xi1>, tensor<19x23x2x6xf32>
    %renamed_value_25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %renamed_value_26 = stablehlo.reduce(%renamed_value_24 init: %renamed_value_25) applies stablehlo.maximum across dimensions = [1] : (tensor<19x23x2x6xf32>, tensor<f32>) -> tensor<19x2x6xf32>
    %renamed_value_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %renamed_value_28 = stablehlo.reduce(%renamed_value_10 init: %renamed_value_27) applies stablehlo.add across dimensions = [1] : (tensor<19x23x1x1xf32>, tensor<f32>) -> tensor<19x1x1xf32>
    return %renamed_value_26, %renamed_value_28 : tensor<19x2x6xf32>, tensor<19x1x1xf32>
  }
}
