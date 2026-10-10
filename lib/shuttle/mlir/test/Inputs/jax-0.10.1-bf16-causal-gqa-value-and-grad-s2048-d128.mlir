// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX production export retained from
// .agents/artifacts/shuttle-production-attention-20260829/s2048-planner-failure/.
// Artifact payload SHA-256 before this provenance header:
// 6C97840D7A834E737A2D5745A8AEA54344ADE1858F7294181F011C30FD7C9683

module @jit_value_and_grad_identity attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8x4x2048x128xbf16>, %arg1: tensor<1x8x2048x128xbf16>, %arg2: tensor<1x8x2048x128xbf16>, %arg3: tensor<1x8x4x2048x128xbf16>) -> (tensor<1x8x4x2048x128xbf16> {jax.result_info = "result[0]"}, tensor<1x8x4x2048x128xbf16> {jax.result_info = "result[1]"}, tensor<1x8x2048x128xbf16> {jax.result_info = "result[2]"}, tensor<1x8x2048x128xbf16> {jax.result_info = "result[3]"}) {
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
    %17 = stablehlo.reshape %16 : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %19 = stablehlo.compare EQ, %15, %18, FLOAT : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x4x2048x2048xf32>) -> tensor<1x8x4x2048x2048xi1>
    %20 = stablehlo.convert %19 : (tensor<1x8x4x2048x2048xi1>) -> tensor<1x8x4x2048x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %22 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %24 = stablehlo.subtract %15, %23 : tensor<1x8x4x2048x2048xf32>
    %25 = stablehlo.exponential %24 : tensor<1x8x4x2048x2048xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.reduce(%25 init: %cst_3) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %29 = stablehlo.divide %25, %28 : tensor<1x8x4x2048x2048xf32>
    %30 = stablehlo.multiply %27, %27 : tensor<1x8x4x2048x1xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x1xf32>
    %32 = stablehlo.divide %31, %30 : tensor<1x8x4x2048x1xf32>
    %33 = stablehlo.dot_general %29, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x128xf32>
    %34 = stablehlo.convert %33 : (tensor<1x8x4x2048x128xf32>) -> tensor<1x8x4x2048x128xbf16>
    %35 = stablehlo.convert %arg3 : (tensor<1x8x4x2048x128xbf16>) -> tensor<1x8x4x2048x128xf32>
    %36 = stablehlo.dot_general %35, %29, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x128xf32>, tensor<1x8x4x2048x2048xf32>) -> tensor<1x8x128x2048xf32>
    %37 = stablehlo.transpose %36, dims = [0, 1, 3, 2] : (tensor<1x8x128x2048xf32>) -> tensor<1x8x2048x128xf32>
    %38 = stablehlo.dot_general %35, %2, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x128xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x2048xf32>
    %39 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<1x8x4x2048x2048xf32>
    %41 = stablehlo.multiply %40, %25 : tensor<1x8x4x2048x2048xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.reduce(%41 init: %cst_5) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %43 = stablehlo.reshape %42 : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %44 = stablehlo.negate %43 : tensor<1x8x4x2048x1xf32>
    %45 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3, 4] : (tensor<1x8x4x2048x1xf32>) -> tensor<1x8x4x2048x2048xf32>
    %46 = stablehlo.divide %38, %45 : tensor<1x8x4x2048x2048xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.reduce(%44 init: %cst_6) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x2048x1xf32>, tensor<f32>) -> tensor<8x4x2048xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [1, 2, 3] : (tensor<8x4x2048xf32>) -> tensor<1x8x4x2048xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x2048xf32>
    %50 = stablehlo.add %46, %49 : tensor<1x8x4x2048x2048xf32>
    %51 = stablehlo.multiply %50, %25 : tensor<1x8x4x2048x2048xf32>
    %52 = stablehlo.negate %51 : tensor<1x8x4x2048x2048xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53 = stablehlo.reduce(%52 init: %cst_7) applies stablehlo.add across dimensions = [4] : (tensor<1x8x4x2048x2048xf32>, tensor<f32>) -> tensor<1x8x4x2048xf32>
    %54 = stablehlo.reshape %53 : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x1xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %55 = stablehlo.reduce(%54 init: %cst_8) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x8x4x2048x1xf32>, tensor<f32>) -> tensor<8x4x2048xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [1, 2, 3] : (tensor<8x4x2048xf32>) -> tensor<1x8x4x2048xf32>
    %57 = stablehlo.divide %56, %21 : tensor<1x8x4x2048xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x8x4x2048xf32>) -> tensor<1x8x4x2048x2048xf32>
    %59 = stablehlo.multiply %58, %20 : tensor<1x8x4x2048x2048xf32>
    %60 = stablehlo.add %51, %59 : tensor<1x8x4x2048x2048xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %62 = stablehlo.select %13, %60, %61 : tensor<1x8x4x2048x2048xi1>, tensor<1x8x4x2048x2048xf32>
    %cst_10 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x8x4x2048x2048xf32>
    %64 = stablehlo.multiply %62, %63 : tensor<1x8x4x2048x2048xf32>
    %65 = stablehlo.dot_general %64, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x4x2048x128xf32>) -> tensor<1x8x2048x128xf32>
    %66 = stablehlo.dot_general %64, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x2048x128xf32>) -> tensor<1x8x4x2048x128xf32>
    %67 = stablehlo.convert %37 : (tensor<1x8x2048x128xf32>) -> tensor<1x8x2048x128xbf16>
    %68 = stablehlo.convert %65 : (tensor<1x8x2048x128xf32>) -> tensor<1x8x2048x128xbf16>
    %69 = stablehlo.convert %66 : (tensor<1x8x4x2048x128xf32>) -> tensor<1x8x4x2048x128xbf16>
    return %34, %69, %68, %67 : tensor<1x8x4x2048x128xbf16>, tensor<1x8x4x2048x128xbf16>, tensor<1x8x2048x128xbf16>, tensor<1x8x2048x128xbf16>
  }
}
