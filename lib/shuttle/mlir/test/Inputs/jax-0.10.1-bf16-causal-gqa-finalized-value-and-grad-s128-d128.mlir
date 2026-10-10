// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-bf16-causal-gqa-finalized-value-and-grad-s128-d128.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: ordinary-JAX BF16 causal GQA value and VJP with tanh(U) + 0.25 * U finalizer
// Inputs: (1, 8, 4, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128), (1, 8, 4, 128, 128):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 324E17F68821CC74A7BAAB177DBCAAF150204B30F8607182703C6C59E992316D
// Fixture StableHLO SHA-256: 324E17F68821CC74A7BAAB177DBCAAF150204B30F8607182703C6C59E992316D
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_bf16_finalized_value_and_grad attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x128x128xbf16>, %arg1: tensor<1x8x128x128xbf16>, %arg2: tensor<1x8x128x128xbf16>, %arg3: tensor<1x8x4x128x128xbf16>) -> (tensor<1x8x4x128x128xbf16> {jax.result_info = "result[0]"}, tensor<1x8x4x128x128xbf16> {jax.result_info = "result[1]"}, tensor<1x8x128x128xbf16> {jax.result_info = "result[2]"}, tensor<1x8x128x128xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1x8x4x128x128xbf16>) -> tensor<1x8x4x128x128xf32>
    %1 = stablehlo.convert %arg1 : (tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xf32>
    %2 = stablehlo.convert %arg2 : (tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xf32>
    %3 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %cst = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x8x4x128x128xf32>
    %6 = stablehlo.iota dim = 0 : tensor<128xi32>
    %7 = stablehlo.reshape %6 : (tensor<128xi32>) -> tensor<1x1x1x128x1xi32>
    %8 = stablehlo.iota dim = 0 : tensor<128xi32>
    %9 = stablehlo.reshape %8 : (tensor<128xi32>) -> tensor<1x1x1x1x128xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x128xi32>) -> tensor<1x1x1x128x128xi32>
    %11 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x128x1xi32>) -> tensor<1x1x1x128x128xi32>
    %12 = stablehlo.compare LE, %10, %11, SIGNED : (tensor<1x1x1x128x128xi32>, tensor<1x1x1x128x128xi32>) -> tensor<1x1x1x128x128xi1>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x128x128xi1>) -> tensor<1x8x4x128x128xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %15 = stablehlo.select %13, %5, %14 : tensor<1x8x4x128x128xi1>, tensor<1x8x4x128x128xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_1) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %19 = stablehlo.compare EQ, %15, %18, FLOAT : (tensor<1x8x4x128x128xf32>, tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xi1>
    %20 = stablehlo.convert %19 : (tensor<1x8x4x128x128xi1>) -> tensor<1x8x4x128x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %22 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %24 = stablehlo.subtract %15, %23 : tensor<1x8x4x128x128xf32>
    %25 = stablehlo.exponential %24 : tensor<1x8x4x128x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.reduce(%25 init: %cst_3) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %29 = stablehlo.divide %25, %28 : tensor<1x8x4x128x128xf32>
    %30 = stablehlo.multiply %27, %27 : tensor<1x8x4x128x1xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x1xf32>
    %32 = stablehlo.divide %31, %30 : tensor<1x8x4x128x1xf32>
    %33 = stablehlo.dot_general %29, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %34 = stablehlo.tanh %33 : tensor<1x8x4x128x128xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %36 = stablehlo.subtract %35, %34 : tensor<1x8x4x128x128xf32>
    %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %38 = stablehlo.multiply %37, %33 : tensor<1x8x4x128x128xf32>
    %39 = stablehlo.add %34, %38 : tensor<1x8x4x128x128xf32>
    %40 = stablehlo.convert %39 : (tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xbf16>
    %41 = stablehlo.convert %arg3 : (tensor<1x8x4x128x128xbf16>) -> tensor<1x8x4x128x128xf32>
    %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %43 = stablehlo.multiply %42, %41 : tensor<1x8x4x128x128xf32>
    %44 = stablehlo.multiply %41, %36 : tensor<1x8x4x128x128xf32>
    %45 = stablehlo.add %43, %44 : tensor<1x8x4x128x128xf32>
    %46 = stablehlo.multiply %44, %34 : tensor<1x8x4x128x128xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x8x4x128x128xf32>
    %48 = stablehlo.dot_general %47, %29, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x4x128x128xf32>) -> tensor<1x8x128x128xf32>
    %49 = stablehlo.transpose %48, dims = [0, 1, 3, 2] : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %50 = stablehlo.dot_general %47, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %51 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %52 = stablehlo.multiply %50, %51 : tensor<1x8x4x128x128xf32>
    %53 = stablehlo.multiply %52, %25 : tensor<1x8x4x128x128xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %54 = stablehlo.reduce(%53 init: %cst_8) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %56 = stablehlo.negate %55 : tensor<1x8x4x128x1xf32>
    %57 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %58 = stablehlo.divide %50, %57 : tensor<1x8x4x128x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.reduce(%56 init: %cst_9) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x128x1xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [1, 2, 3] : (tensor<8x4x128xf32>) -> tensor<1x8x4x128xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x128xf32>
    %62 = stablehlo.add %58, %61 : tensor<1x8x4x128x128xf32>
    %63 = stablehlo.multiply %62, %25 : tensor<1x8x4x128x128xf32>
    %64 = stablehlo.negate %63 : tensor<1x8x4x128x128xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %65 = stablehlo.reduce(%64 init: %cst_10) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %67 = stablehlo.reduce(%66 init: %cst_11) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x128x1xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [1, 2, 3] : (tensor<8x4x128xf32>) -> tensor<1x8x4x128xf32>
    %69 = stablehlo.divide %68, %21 : tensor<1x8x4x128xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x128xf32>
    %71 = stablehlo.multiply %70, %20 : tensor<1x8x4x128x128xf32>
    %72 = stablehlo.add %63, %71 : tensor<1x8x4x128x128xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %74 = stablehlo.select %13, %72, %73 : tensor<1x8x4x128x128xi1>, tensor<1x8x4x128x128xf32>
    %cst_13 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %76 = stablehlo.multiply %74, %75 : tensor<1x8x4x128x128xf32>
    %77 = stablehlo.dot_general %76, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x4x128x128xf32>) -> tensor<1x8x128x128xf32>
    %78 = stablehlo.dot_general %76, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %79 = stablehlo.convert %49 : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xbf16>
    %80 = stablehlo.convert %77 : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xbf16>
    %81 = stablehlo.convert %78 : (tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xbf16>
    return %40, %81, %80, %79 : tensor<1x8x4x128x128xbf16>, tensor<1x8x4x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>
  }
}
