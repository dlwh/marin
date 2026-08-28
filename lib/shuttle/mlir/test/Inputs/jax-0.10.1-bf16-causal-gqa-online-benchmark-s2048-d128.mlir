// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-bf16-causal-gqa-online-benchmark-s2048-d128.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: BF16 causal GQA B1 Hq32 Hkv8 S2048 D=Dv128 with FP32 state
// Inputs: (1, 8, 4, 2048, 128), (1, 8, 2048, 128), (1, 8, 2048, 128):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 48D545197651532EFE3CD0D3EED92C3C61BDA7F04E14C5296FA9FD18DCB53181
// Fixture StableHLO SHA-256: 48D545197651532EFE3CD0D3EED92C3C61BDA7F04E14C5296FA9FD18DCB53181
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_bf16_benchmark attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x2048x128xbf16>, %arg1: tensor<1x8x2048x128xbf16>, %arg2: tensor<1x8x2048x128xbf16>) -> (tensor<1x8x4x2048x128xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1x8x4x2048x128xbf16>) -> tensor<1x8x4x2048x128xf32>
    %1 = stablehlo.convert %arg1 : (tensor<1x8x2048x128xbf16>) -> tensor<1x8x2048x128xf32>
    %2 = stablehlo.convert %arg2 : (tensor<1x8x2048x128xbf16>) -> tensor<1x8x2048x128xf32>
    %3 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x128xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x2048xf32>
    %cst = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x8x4x2048x2048xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %7 = stablehlo.reshape %6 : (tensor<2048xi32>) -> tensor<1x1x1x2048x1xi32>
    %8 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %9 = stablehlo.reshape %8 : (tensor<2048xi32>) -> tensor<1x1x1x1x2048xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x2048xi32>) -> tensor<1x1x1x2048x2048xi32>
    %11 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x2048x1xi32>) -> tensor<1x1x1x2048x2048xi32>
    %12 = stablehlo.compare LE, %10, %11, SIGNED : (tensor<1x1x1x2048x2048xi32>, tensor<1x1x1x2048x2048xi32>) -> tensor<1x1x1x2048x2048xi1>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x2048x2048xi1>) -> tensor<1x8x4x2048x2048xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %15 = stablehlo.select %13, %5, %14 : tensor<1x8x4x2048x2048xi1>, tensor<1x8x4x2048x2048xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_1) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %19 = stablehlo.subtract %15, %18 : tensor<1x8x4x2048x2048xf32>
    %20 = stablehlo.exponential %19 : tensor<1x8x4x2048x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %24 = stablehlo.divide %20, %23 : tensor<1x8x4x2048x2048xf32>
    %25 = stablehlo.dot_general %24, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x128xf32>
    %26 = stablehlo.convert %25 : (tensor<1x8x4x2048x128xf32>) -> tensor<1x8x4x2048x128xbf16>
    return %26 : tensor<1x8x4x2048x128xbf16>
  }
}
