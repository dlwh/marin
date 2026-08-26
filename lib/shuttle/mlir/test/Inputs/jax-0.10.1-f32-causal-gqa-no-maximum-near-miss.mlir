// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-f32-causal-gqa-no-maximum-near-miss.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: near miss without maximum sufficient-state semantics
// Inputs: (1, 2, 2, 17, 8), (1, 2, 17, 8), (1, 2, 17, 7):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: E9A8689115DFA284414C12F0AD160E0970CE1562DFE0392FEE7E212490DD750D
// Fixture StableHLO SHA-256: E9A8689115DFA284414C12F0AD160E0970CE1562DFE0392FEE7E212490DD750D
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_without_maximum attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x2x2x17x8xf32>, %arg1: tensor<1x2x17x8xf32>, %arg2: tensor<1x2x17x7xf32>) -> (tensor<1x2x2x17x7xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x2x2x17x8xf32>, tensor<1x2x17x8xf32>) -> tensor<1x2x2x17x17xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x2x2x17x17xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x2x2x17x17xf32>
    %3 = stablehlo.iota dim = 0 : tensor<17xi32>
    %4 = stablehlo.reshape %3 : (tensor<17xi32>) -> tensor<1x1x1x17x1xi32>
    %5 = stablehlo.iota dim = 0 : tensor<17xi32>
    %6 = stablehlo.reshape %5 : (tensor<17xi32>) -> tensor<1x1x1x1x17xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x17xi32>) -> tensor<1x1x1x17x17xi32>
    %8 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x17x1xi32>) -> tensor<1x1x1x17x17xi32>
    %9 = stablehlo.compare LE, %7, %8, SIGNED : (tensor<1x1x1x17x17xi32>, tensor<1x1x1x17x17xi32>) -> tensor<1x1x1x17x17xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x17x17xi1>) -> tensor<1x2x2x17x17xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2x2x17x17xf32>
    %12 = stablehlo.select %10, %2, %11 : tensor<1x2x2x17x17xi1>, tensor<1x2x2x17x17xf32>
    %13 = stablehlo.exponential %12 : tensor<1x2x2x17x17xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_1) applies stablehlo.add across dimensions = [4] : (tensor<1x2x2x17x17xf32>, tensor<f32>) -> tensor<1x2x2x17xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x2x2x17xf32>) -> tensor<1x2x2x17x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x17x1xf32>) -> tensor<1x2x2x17x17xf32>
    %17 = stablehlo.divide %13, %16 : tensor<1x2x2x17x17xf32>
    %18 = stablehlo.dot_general %17, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x2x2x17x17xf32>, tensor<1x2x17x7xf32>) -> tensor<1x2x2x17x7xf32>
    return %18 : tensor<1x2x2x17x7xf32>
  }
}
