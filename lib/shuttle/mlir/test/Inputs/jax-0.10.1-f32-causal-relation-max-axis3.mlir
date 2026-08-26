// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-attention-relation-fixtures.py (jax-0.10.1-f32-causal-relation-max-axis3.mlir)
// Export: jax.jit(fixture.function).lower(f32_shape_struct).compiler_ir(dialect="stablehlo")
// Expression: causal mask over [batch, query_head, query, key], max/sum on axis 3
// Input: (1, 4, 17, 17):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: A9C9D7D3F25F6FFA9AE962ED302BAC6099B707C03750AB8F8A8060B18F35BA4E
// Fixture StableHLO SHA-256: A9C9D7D3F25F6FFA9AE962ED302BAC6099B707C03750AB8F8A8060B18F35BA4E
// Textual SSA alpha-renaming: false

module @jit_causal_score_relation attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x4x17x17xf32>) -> (tensor<1x4x17xf32> {jax.result_info = "result[0]"}, tensor<1x1x17xf32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<17xi32>
    %1 = stablehlo.reshape %0 : (tensor<17xi32>) -> tensor<1x1x17x1xi32>
    %2 = stablehlo.iota dim = 0 : tensor<17xi32>
    %3 = stablehlo.reshape %2 : (tensor<17xi32>) -> tensor<1x1x1x17xi32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3] : (tensor<1x1x1x17xi32>) -> tensor<1x1x17x17xi32>
    %5 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x17x1xi32>) -> tensor<1x1x17x17xi32>
    %6 = stablehlo.compare LE, %4, %5, SIGNED : (tensor<1x1x17x17xi32>, tensor<1x1x17x17xi32>) -> tensor<1x1x17x17xi1>
    %7 = stablehlo.convert %6 : (tensor<1x1x17x17xi1>) -> tensor<1x1x17x17xf32>
    %8 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x1x17x17xi1>) -> tensor<1x4x17x17xi1>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x17x17xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x4x17x17xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x17x17xf32>
    %12 = stablehlo.select %8, %10, %11 : tensor<1x4x17x17xi1>, tensor<1x4x17x17xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x17x17xf32>, tensor<f32>) -> tensor<1x4x17xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%7 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x1x17x17xf32>, tensor<f32>) -> tensor<1x1x17xf32>
    return %13, %14 : tensor<1x4x17xf32>, tensor<1x1x17xf32>
  }
}
