// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-f32-causal-gqa-alternate-score-contract-near-miss.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: near miss whose representative score has two Contract ancestors
// Inputs: (1, 2, 2, 17, 8), (1, 2, 17, 8), (1, 2, 17, 7):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 1E9ADBAA1275A8D75724F846B8DA0580508ADC6E611519CEDF0FC8F07DEA2307
// Fixture StableHLO SHA-256: 1E9ADBAA1275A8D75724F846B8DA0580508ADC6E611519CEDF0FC8F07DEA2307
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_with_alternate_score_contract attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x2x2x17x8xf32>, %arg1: tensor<1x2x17x8xf32>, %arg2: tensor<1x2x17x7xf32>) -> (tensor<1x2x2x17x7xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x2x2x17x8xf32>, tensor<1x2x17x8xf32>) -> tensor<1x2x2x17x17xf32>
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x2x17x8xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<1x2x17x8xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x2x2x17x8xf32>, tensor<1x2x17x8xf32>) -> tensor<1x2x2x17x17xf32>
    %4 = stablehlo.add %0, %3 : tensor<1x2x2x17x17xf32>
    %cst_0 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2x2x17x17xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1x2x2x17x17xf32>
    %7 = stablehlo.iota dim = 0 : tensor<17xi32>
    %8 = stablehlo.reshape %7 : (tensor<17xi32>) -> tensor<1x1x1x17x1xi32>
    %9 = stablehlo.iota dim = 0 : tensor<17xi32>
    %10 = stablehlo.reshape %9 : (tensor<17xi32>) -> tensor<1x1x1x1x17xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x17xi32>) -> tensor<1x1x1x17x17xi32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x17x1xi32>) -> tensor<1x1x1x17x17xi32>
    %13 = stablehlo.compare LE, %11, %12, SIGNED : (tensor<1x1x1x17x17xi32>, tensor<1x1x1x17x17xi32>) -> tensor<1x1x1x17x17xi1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x17x17xi1>) -> tensor<1x2x2x17x17xi1>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x2x2x17x17xf32>
    %16 = stablehlo.select %14, %6, %15 : tensor<1x2x2x17x17xi1>, tensor<1x2x2x17x17xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst_2) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x2x17x17xf32>, tensor<f32>) -> tensor<1x2x2x17xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x2x2x17xf32>) -> tensor<1x2x2x17x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x17x1xf32>) -> tensor<1x2x2x17x17xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<1x2x2x17x17xf32>
    %21 = stablehlo.exponential %20 : tensor<1x2x2x17x17xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%21 init: %cst_3) applies stablehlo.add across dimensions = [4] : (tensor<1x2x2x17x17xf32>, tensor<f32>) -> tensor<1x2x2x17xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x2x2x17xf32>) -> tensor<1x2x2x17x1xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x17x1xf32>) -> tensor<1x2x2x17x17xf32>
    %25 = stablehlo.divide %21, %24 : tensor<1x2x2x17x17xf32>
    %26 = stablehlo.dot_general %25, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x2x2x17x17xf32>, tensor<1x2x17x7xf32>) -> tensor<1x2x2x17x7xf32>
    return %26 : tensor<1x2x2x17x7xf32>
  }
}
