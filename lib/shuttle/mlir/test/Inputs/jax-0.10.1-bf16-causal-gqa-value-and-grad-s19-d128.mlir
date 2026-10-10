// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-online-normalized-contract-fixtures.py (jax-0.10.1-bf16-causal-gqa-value-and-grad-s19-d128.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: exact H100 canary BF16 normalized contraction value and VJP, B1 Hq6 Hkv3 S19 D=Dv128
// Inputs: (1, 3, 2, 19, 128), (1, 3, 19, 128), (1, 3, 19, 128), (1, 3, 2, 19, 128):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw ordinary-JAX StableHLO SHA-256: A06D0FA589F25B30FF9E9176F6E6BA62779E3770FBA0D9A0996D6E7CCE22DFED
// Fixture StableHLO SHA-256: A06D0FA589F25B30FF9E9176F6E6BA62779E3770FBA0D9A0996D6E7CCE22DFED
// Textual SSA alpha-renaming: false
module @jit_attention_value_and_grad attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x3x2x19x128xbf16>, %arg1: tensor<1x3x19x128xbf16>, %arg2: tensor<1x3x19x128xbf16>, %arg3: tensor<1x3x2x19x128xbf16>) -> (tensor<1x3x2x19x128xbf16> {jax.result_info = "result[0]"}, tensor<1x3x2x19x128xbf16> {jax.result_info = "result[1]"}, tensor<1x3x19x128xbf16> {jax.result_info = "result[2]"}, tensor<1x3x19x128xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x2x19x128xbf16>) -> tensor<1x3x2x19x128xf32>
    %1 = stablehlo.convert %arg1 : (tensor<1x3x19x128xbf16>) -> tensor<1x3x19x128xf32>
    %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x128xf32>, tensor<1x3x19x128xf32>) -> tensor<1x3x2x19x19xf32>
    %cst = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x3x2x19x19xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x3x2x19x19xf32>
    %5 = stablehlo.iota dim = 0 : tensor<19xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<19xi32>) -> tensor<19x1xi32>
    %7 = stablehlo.iota dim = 0 : tensor<19xi32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<19xi32>) -> tensor<1x19xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x19xi32>) -> tensor<19x19xi32>
    %10 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<19x1xi32>) -> tensor<19x19xi32>
    %11 = stablehlo.compare LE, %9, %10, SIGNED : (tensor<19x19xi32>, tensor<19x19xi32>) -> tensor<19x19xi1>
    %12 = stablehlo.broadcast_in_dim %11, dims = [3, 4] : (tensor<19x19xi1>) -> tensor<1x3x2x19x19xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x3x2x19x19xf32>
    %14 = stablehlo.select %12, %4, %13 : tensor<1x3x2x19x19xi1>, tensor<1x3x2x19x19xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = stablehlo.reduce(%14 init: %cst_1) applies stablehlo.maximum across dimensions = [4] : (tensor<1x3x2x19x19xf32>, tensor<f32>) -> tensor<1x3x2x19xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3, 4] : (tensor<1x3x2x19x1xf32>) -> tensor<1x3x2x19x19xf32>
    %18 = stablehlo.compare EQ, %14, %17, FLOAT : (tensor<1x3x2x19x19xf32>, tensor<1x3x2x19x19xf32>) -> tensor<1x3x2x19x19xi1>
    %19 = stablehlo.convert %18 : (tensor<1x3x2x19x19xi1>) -> tensor<1x3x2x19x19xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<1x3x2x19x19xf32>, tensor<f32>) -> tensor<1x3x2x19xf32>
    %21 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3] : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3, 4] : (tensor<1x3x2x19x1xf32>) -> tensor<1x3x2x19x19xf32>
    %23 = stablehlo.subtract %14, %22 : tensor<1x3x2x19x19xf32>
    %24 = stablehlo.exponential %23 : tensor<1x3x2x19x19xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%24 init: %cst_3) applies stablehlo.add across dimensions = [4] : (tensor<1x3x2x19x19xf32>, tensor<f32>) -> tensor<1x3x2x19xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3, 4] : (tensor<1x3x2x19x1xf32>) -> tensor<1x3x2x19x19xf32>
    %28 = stablehlo.divide %24, %27 : tensor<1x3x2x19x19xf32>
    %29 = stablehlo.multiply %26, %26 : tensor<1x3x2x19x1xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x3x2x19x1xf32>
    %31 = stablehlo.divide %30, %29 : tensor<1x3x2x19x1xf32>
    %32 = stablehlo.convert %arg2 : (tensor<1x3x19x128xbf16>) -> tensor<1x3x19x128xf32>
    %33 = stablehlo.dot_general %28, %32, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x19xf32>, tensor<1x3x19x128xf32>) -> tensor<1x3x2x19x128xf32>
    %34 = stablehlo.convert %33 : (tensor<1x3x2x19x128xf32>) -> tensor<1x3x2x19x128xbf16>
    %35 = stablehlo.convert %arg3 : (tensor<1x3x2x19x128xbf16>) -> tensor<1x3x2x19x128xf32>
    %36 = stablehlo.dot_general %35, %28, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x128xf32>, tensor<1x3x2x19x19xf32>) -> tensor<1x3x128x19xf32>
    %37 = stablehlo.transpose %36, dims = [0, 1, 3, 2] : (tensor<1x3x128x19xf32>) -> tensor<1x3x19x128xf32>
    %38 = stablehlo.dot_general %35, %32, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x128xf32>, tensor<1x3x19x128xf32>) -> tensor<1x3x2x19x19xf32>
    %39 = stablehlo.convert %37 : (tensor<1x3x19x128xf32>) -> tensor<1x3x19x128xbf16>
    %40 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3, 4] : (tensor<1x3x2x19x1xf32>) -> tensor<1x3x2x19x19xf32>
    %41 = stablehlo.multiply %38, %40 : tensor<1x3x2x19x19xf32>
    %42 = stablehlo.multiply %41, %24 : tensor<1x3x2x19x19xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%42 init: %cst_5) applies stablehlo.add across dimensions = [4] : (tensor<1x3x2x19x19xf32>, tensor<f32>) -> tensor<1x3x2x19xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x1xf32>
    %45 = stablehlo.negate %44 : tensor<1x3x2x19x1xf32>
    %46 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3, 4] : (tensor<1x3x2x19x1xf32>) -> tensor<1x3x2x19x19xf32>
    %47 = stablehlo.divide %38, %46 : tensor<1x3x2x19x19xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %48 = stablehlo.reduce(%45 init: %cst_6) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x3x2x19x1xf32>, tensor<f32>) -> tensor<3x2x19xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [1, 2, 3] : (tensor<3x2x19xf32>) -> tensor<1x3x2x19xf32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2, 3] : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x19xf32>
    %51 = stablehlo.add %47, %50 : tensor<1x3x2x19x19xf32>
    %52 = stablehlo.multiply %51, %24 : tensor<1x3x2x19x19xf32>
    %53 = stablehlo.negate %52 : tensor<1x3x2x19x19xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %54 = stablehlo.reduce(%53 init: %cst_7) applies stablehlo.add across dimensions = [4] : (tensor<1x3x2x19x19xf32>, tensor<f32>) -> tensor<1x3x2x19xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x1xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %56 = stablehlo.reduce(%55 init: %cst_8) applies stablehlo.add across dimensions = [0, 4] : (tensor<1x3x2x19x1xf32>, tensor<f32>) -> tensor<3x2x19xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [1, 2, 3] : (tensor<3x2x19xf32>) -> tensor<1x3x2x19xf32>
    %58 = stablehlo.divide %57, %20 : tensor<1x3x2x19xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x3x2x19xf32>) -> tensor<1x3x2x19x19xf32>
    %60 = stablehlo.multiply %59, %19 : tensor<1x3x2x19x19xf32>
    %61 = stablehlo.add %52, %60 : tensor<1x3x2x19x19xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x3x2x19x19xf32>
    %63 = stablehlo.select %12, %61, %62 : tensor<1x3x2x19x19xi1>, tensor<1x3x2x19x19xf32>
    %cst_10 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x3x2x19x19xf32>
    %65 = stablehlo.multiply %63, %64 : tensor<1x3x2x19x19xf32>
    %66 = stablehlo.dot_general %65, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x19xf32>, tensor<1x3x2x19x128xf32>) -> tensor<1x3x19x128xf32>
    %67 = stablehlo.dot_general %65, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2x19x19xf32>, tensor<1x3x19x128xf32>) -> tensor<1x3x2x19x128xf32>
    %68 = stablehlo.convert %66 : (tensor<1x3x19x128xf32>) -> tensor<1x3x19x128xbf16>
    %69 = stablehlo.convert %67 : (tensor<1x3x2x19x128xf32>) -> tensor<1x3x2x19x128xbf16>
    return %34, %69, %68, %39 : tensor<1x3x2x19x128xbf16>, tensor<1x3x2x19x128xbf16>, tensor<1x3x19x128xbf16>, tensor<1x3x19x128xbf16>
  }
}
