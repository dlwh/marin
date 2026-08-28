// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-f32-causal-gqa-online-benchmark-s2048-d128.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: representative causal GQA B1 Hq32 Hkv8 S2048 D=Dv128
// Inputs: (1, 8, 4, 2048, 128), (1, 8, 2048, 128), (1, 8, 2048, 128):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 8937B0C35B0090EAAF1A25BA5C861CB94AE39C385074C7797D4D37C1BA37614B
// Fixture StableHLO SHA-256: 8937B0C35B0090EAAF1A25BA5C861CB94AE39C385074C7797D4D37C1BA37614B
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_benchmark attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x2048x128xf32>, %arg1: tensor<1x8x2048x128xf32>, %arg2: tensor<1x8x2048x128xf32>) -> (tensor<1x8x4x2048x128xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x128xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x2048xf32>
    %cst = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x8x4x2048x2048xf32>
    %3 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %4 = stablehlo.reshape %3 : (tensor<2048xi32>) -> tensor<1x1x1x2048x1xi32>
    %5 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %6 = stablehlo.reshape %5 : (tensor<2048xi32>) -> tensor<1x1x1x1x2048xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x2048xi32>) -> tensor<1x1x1x2048x2048xi32>
    %8 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x2048x1xi32>) -> tensor<1x1x1x2048x2048xi32>
    %9 = stablehlo.compare LE, %7, %8, SIGNED : (tensor<1x1x1x2048x2048xi32>, tensor<1x1x1x2048x2048xi32>) -> tensor<1x1x1x2048x2048xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x2048x2048xi1>) -> tensor<1x8x4x2048x2048xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %12 = stablehlo.select %10, %2, %11 : tensor<1x8x4x2048x2048xi1>, tensor<1x8x4x2048x2048xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_1) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %16 = stablehlo.subtract %12, %15 : tensor<1x8x4x2048x2048xf32>
    %17 = stablehlo.exponential %16 : tensor<1x8x4x2048x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%17 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %21 = stablehlo.divide %17, %20 : tensor<1x8x4x2048x2048xf32>
    %22 = stablehlo.dot_general %21, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x128xf32>
    return %22 : tensor<1x8x4x2048x128xf32>
  }
}
