// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-bf16-causal-gqa-scalar-value-and-grad-s128-d128.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: ordinary-JAX scalar inner-product objective and gradients, B1 Hq32 Hkv8 S128 D=Dv128
// Inputs: (1, 8, 4, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128), (1, 8, 4, 128, 128):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: 491CB29CE2D73E71C57E714742E777006FD9C367EE53414A66D1A07887A86C34
// Fixture StableHLO SHA-256: 491CB29CE2D73E71C57E714742E777006FD9C367EE53414A66D1A07887A86C34
// Textual SSA alpha-renaming: false
module @jit_causal_gqa_bf16_scalar_value_and_grad attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x128x128xbf16>, %arg1: tensor<1x8x128x128xbf16>, %arg2: tensor<1x8x128x128xbf16>, %arg3: tensor<1x8x4x128x128xbf16>) -> (tensor<f32> {jax.result_info = "result[0]"}, tensor<1x8x4x128x128xbf16> {jax.result_info = "result[1]"}, tensor<1x8x128x128xbf16> {jax.result_info = "result[2]"}, tensor<1x8x128x128xbf16> {jax.result_info = "result[3]"}) {
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
    %34 = stablehlo.convert %33 : (tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xbf16>
    %35 = stablehlo.convert %34 : (tensor<1x8x4x128x128xbf16>) -> tensor<1x8x4x128x128xf32>
    %36 = stablehlo.convert %arg3 : (tensor<1x8x4x128x128xbf16>) -> tensor<1x8x4x128x128xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<1x8x4x128x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.reduce(%37 init: %cst_5) applies stablehlo.add across dimensions = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %40 = stablehlo.multiply %39, %36 : tensor<1x8x4x128x128xf32>
    %41 = stablehlo.convert %40 : (tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xbf16>
    %42 = stablehlo.convert %41 : (tensor<1x8x4x128x128xbf16>) -> tensor<1x8x4x128x128xf32>
    %43 = stablehlo.dot_general %42, %29, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x4x128x128xf32>) -> tensor<1x8x128x128xf32>
    %44 = stablehlo.transpose %43, dims = [0, 1, 3, 2] : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %45 = stablehlo.dot_general %42, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %46 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x8x4x128x128xf32>
    %48 = stablehlo.multiply %47, %25 : tensor<1x8x4x128x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.reduce(%48 init: %cst_7) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %50 = stablehlo.reshape %49 : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %51 = stablehlo.negate %50 : tensor<1x8x4x128x1xf32>
    %52 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x128x1xf32>) -> tensor<1x8x4x128x128xf32>
    %53 = stablehlo.divide %45, %52 : tensor<1x8x4x128x128xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %54 = stablehlo.reduce(%51 init: %cst_8) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x128x1xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [1, 2, 3] : (tensor<8x4x128xf32>) -> tensor<1x8x4x128xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x128xf32>
    %57 = stablehlo.add %53, %56 : tensor<1x8x4x128x128xf32>
    %58 = stablehlo.multiply %57, %25 : tensor<1x8x4x128x128xf32>
    %59 = stablehlo.negate %58 : tensor<1x8x4x128x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_9) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x128x128xf32>, tensor<f32>) -> tensor<1x8x4x128xf32>
    %61 = stablehlo.reshape %60 : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x1xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.reduce(%61 init: %cst_10) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x128x1xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [1, 2, 3] : (tensor<8x4x128xf32>) -> tensor<1x8x4x128xf32>
    %64 = stablehlo.divide %63, %21 : tensor<1x8x4x128xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128x128xf32>
    %66 = stablehlo.multiply %65, %20 : tensor<1x8x4x128x128xf32>
    %67 = stablehlo.add %58, %66 : tensor<1x8x4x128x128xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %69 = stablehlo.select %13, %67, %68 : tensor<1x8x4x128x128xi1>, tensor<1x8x4x128x128xf32>
    %cst_12 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x8x4x128x128xf32>
    %71 = stablehlo.multiply %69, %70 : tensor<1x8x4x128x128xf32>
    %72 = stablehlo.dot_general %71, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x4x128x128xf32>) -> tensor<1x8x128x128xf32>
    %73 = stablehlo.dot_general %71, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x4x128x128xf32>
    %74 = stablehlo.convert %44 : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xbf16>
    %75 = stablehlo.convert %72 : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xbf16>
    %76 = stablehlo.convert %73 : (tensor<1x8x4x128x128xf32>) -> tensor<1x8x4x128x128xbf16>
    return %38, %76, %75, %74 : tensor<f32>, tensor<1x8x4x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>
  }
}
