// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-f32-offset-softcap-gqa-online-axis1-mutation.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: semantic scale/softcap mutation of the axis-1 fixture
// Inputs: (19, 2, 3, 2, 5), (23, 2, 3, 5), (23, 2, 3, 11):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 6A7717096D1F3C5A2490848E17D8A3937479799DB052A32F890DAD3BC0FC79A4
// Fixture StableHLO SHA-256: 57BFC9ABD1BFEA0090571B717501C19BFA5DAEFCB0424112BCDD4580C5310273
// Textual SSA alpha-renaming: true
module @jit_offset_softcap_gqa_mutation attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%layout_b_input_0: tensor<19x2x3x2x5xf32>, %layout_b_input_1: tensor<23x2x3x5xf32>, %layout_b_input_2: tensor<23x2x3x11xf32>) -> (tensor<19x2x3x2x11xf32> {jax.result_info = "result"}) {
    %layout_b_value_0 = stablehlo.dot_general %layout_b_input_1, %layout_b_input_0, batching_dims = [1, 2] x [1, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<23x2x3x5xf32>, tensor<19x2x3x2x5xf32>) -> tensor<2x3x23x19x2xf32>
    %layout_b_value_1 = stablehlo.transpose %layout_b_value_0, dims = [3, 2, 0, 1, 4] : (tensor<2x3x23x19x2xf32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_2 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %layout_b_value_3 = stablehlo.broadcast_in_dim %layout_b_value_2, dims = [] : (tensor<f32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_4 = stablehlo.multiply %layout_b_value_1, %layout_b_value_3 : tensor<19x23x2x3x2xf32>
    %layout_b_value_5 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %layout_b_value_6 = stablehlo.broadcast_in_dim %layout_b_value_5, dims = [] : (tensor<f32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_7 = stablehlo.divide %layout_b_value_4, %layout_b_value_6 : tensor<19x23x2x3x2xf32>
    %layout_b_value_8 = stablehlo.tanh %layout_b_value_7 : tensor<19x23x2x3x2xf32>
    %layout_b_value_9 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %layout_b_value_10 = stablehlo.broadcast_in_dim %layout_b_value_9, dims = [] : (tensor<f32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_11 = stablehlo.multiply %layout_b_value_10, %layout_b_value_8 : tensor<19x23x2x3x2xf32>
    %layout_b_value_12 = stablehlo.iota dim = 0 : tensor<19xi32>
    %layout_b_value_13 = stablehlo.reshape %layout_b_value_12 : (tensor<19xi32>) -> tensor<19x1x1x1x1xi32>
    %layout_b_value_14 = stablehlo.iota dim = 0 : tensor<23xi32>
    %layout_b_value_15 = stablehlo.reshape %layout_b_value_14 : (tensor<23xi32>) -> tensor<1x23x1x1x1xi32>
    %layout_b_value_16 = stablehlo.constant dense<4> : tensor<i32>
    %layout_b_value_17 = stablehlo.broadcast_in_dim %layout_b_value_16, dims = [] : (tensor<i32>) -> tensor<19x1x1x1x1xi32>
    %layout_b_value_18 = stablehlo.add %layout_b_value_13, %layout_b_value_17 : tensor<19x1x1x1x1xi32>
    %layout_b_value_19 = stablehlo.broadcast_in_dim %layout_b_value_15, dims = [0, 1, 2, 3, 4] : (tensor<1x23x1x1x1xi32>) -> tensor<19x23x1x1x1xi32>
    %layout_b_value_20 = stablehlo.broadcast_in_dim %layout_b_value_18, dims = [0, 1, 2, 3, 4] : (tensor<19x1x1x1x1xi32>) -> tensor<19x23x1x1x1xi32>
    %layout_b_value_21 = stablehlo.compare LE, %layout_b_value_19, %layout_b_value_20, SIGNED : (tensor<19x23x1x1x1xi32>, tensor<19x23x1x1x1xi32>) -> tensor<19x23x1x1x1xi1>
    %layout_b_value_22 = stablehlo.broadcast_in_dim %layout_b_value_21, dims = [0, 1, 2, 3, 4] : (tensor<19x23x1x1x1xi1>) -> tensor<19x23x2x3x2xi1>
    %layout_b_value_23 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %layout_b_value_24 = stablehlo.broadcast_in_dim %layout_b_value_23, dims = [] : (tensor<f32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_25 = stablehlo.select %layout_b_value_22, %layout_b_value_11, %layout_b_value_24 : tensor<19x23x2x3x2xi1>, tensor<19x23x2x3x2xf32>
    %layout_b_value_26 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %layout_b_value_27 = stablehlo.reduce(%layout_b_value_25 init: %layout_b_value_26) applies stablehlo.maximum across dimensions = [1] : (tensor<19x23x2x3x2xf32>, tensor<f32>) -> tensor<19x2x3x2xf32>
    %layout_b_value_28 = stablehlo.broadcast_in_dim %layout_b_value_27, dims = [0, 2, 3, 4] : (tensor<19x2x3x2xf32>) -> tensor<19x1x2x3x2xf32>
    %layout_b_value_29 = stablehlo.broadcast_in_dim %layout_b_value_28, dims = [0, 1, 2, 3, 4] : (tensor<19x1x2x3x2xf32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_30 = stablehlo.subtract %layout_b_value_25, %layout_b_value_29 : tensor<19x23x2x3x2xf32>
    %layout_b_value_31 = stablehlo.exponential %layout_b_value_30 : tensor<19x23x2x3x2xf32>
    %layout_b_value_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %layout_b_value_33 = stablehlo.reduce(%layout_b_value_31 init: %layout_b_value_32) applies stablehlo.add across dimensions = [1] : (tensor<19x23x2x3x2xf32>, tensor<f32>) -> tensor<19x2x3x2xf32>
    %layout_b_value_34 = stablehlo.broadcast_in_dim %layout_b_value_33, dims = [0, 2, 3, 4] : (tensor<19x2x3x2xf32>) -> tensor<19x1x2x3x2xf32>
    %layout_b_value_35 = stablehlo.broadcast_in_dim %layout_b_value_34, dims = [0, 1, 2, 3, 4] : (tensor<19x1x2x3x2xf32>) -> tensor<19x23x2x3x2xf32>
    %layout_b_value_36 = stablehlo.divide %layout_b_value_31, %layout_b_value_35 : tensor<19x23x2x3x2xf32>
    %layout_b_value_37 = stablehlo.dot_general %layout_b_input_2, %layout_b_value_36, batching_dims = [1, 2] x [2, 3], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<23x2x3x11xf32>, tensor<19x23x2x3x2xf32>) -> tensor<2x3x11x19x2xf32>
    %layout_b_value_38 = stablehlo.transpose %layout_b_value_37, dims = [3, 0, 1, 4, 2] : (tensor<2x3x11x19x2xf32>) -> tensor<19x2x3x2x11xf32>
    return %layout_b_value_38 : tensor<19x2x3x2x11xf32>
  }
}
