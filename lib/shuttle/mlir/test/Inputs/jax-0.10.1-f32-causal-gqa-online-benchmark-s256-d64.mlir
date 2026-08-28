// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-f32-causal-gqa-online-benchmark-s256-d64.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: representative causal GQA B1 Hq32 Hkv8 S256 D=Dv64
// Inputs: (1, 8, 4, 256, 64), (1, 8, 256, 64), (1, 8, 256, 64):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: D719ECD7664F3CD740E5D5B06D484E8A94AB716E8F06A1C49253B0255C137929
// Fixture StableHLO SHA-256: D719ECD7664F3CD740E5D5B06D484E8A94AB716E8F06A1C49253B0255C137929
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_benchmark attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x256x64xf32>, %arg1: tensor<1x8x256x64xf32>, %arg2: tensor<1x8x256x64xf32>) -> (tensor<1x8x4x256x64xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256x64xf32>, tensor<1x8x256x64xf32>) -> tensor<1x8x4x256x256xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x4x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x8x4x256x256xf32>
    %3 = stablehlo.iota dim = 0 : tensor<256xi32>
    %4 = stablehlo.reshape %3 : (tensor<256xi32>) -> tensor<1x1x1x256x1xi32>
    %5 = stablehlo.iota dim = 0 : tensor<256xi32>
    %6 = stablehlo.reshape %5 : (tensor<256xi32>) -> tensor<1x1x1x1x256xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x256xi32>) -> tensor<1x1x1x256x256xi32>
    %8 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x256x1xi32>) -> tensor<1x1x1x256x256xi32>
    %9 = stablehlo.compare LE, %7, %8, SIGNED : (tensor<1x1x1x256x256xi32>, tensor<1x1x1x256x256xi32>) -> tensor<1x1x1x256x256xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x256x256xi1>) -> tensor<1x8x4x256x256xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x4x256x256xf32>
    %12 = stablehlo.select %10, %2, %11 : tensor<1x8x4x256x256xi1>, tensor<1x8x4x256x256xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_1) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x4x256x256xf32>, tensor<f32>) -> tensor<1x8x4x256xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x256x1xf32>) -> tensor<1x8x4x256x256xf32>
    %16 = stablehlo.subtract %12, %15 : tensor<1x8x4x256x256xf32>
    %17 = stablehlo.exponential %16 : tensor<1x8x4x256x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%17 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x256x256xf32>, tensor<f32>) -> tensor<1x8x4x256xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x256x1xf32>) -> tensor<1x8x4x256x256xf32>
    %21 = stablehlo.divide %17, %20 : tensor<1x8x4x256x256xf32>
    %22 = stablehlo.dot_general %21, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256x256xf32>, tensor<1x8x256x64xf32>) -> tensor<1x8x4x256x64xf32>
    return %22 : tensor<1x8x4x256x64xf32>
  }
}
