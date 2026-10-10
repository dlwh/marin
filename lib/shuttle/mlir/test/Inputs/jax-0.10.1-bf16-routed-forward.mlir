// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX export of the output-only shared-plus-routed algebra fixture.
module @jit_output_only attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>, %arg2: tensor<2x2xbf16>, %arg3: tensor<2x2xbf16>, %arg4: tensor<2x2xbf16>, %arg5: tensor<2x2x2xbf16>, %arg6: tensor<2x2x2xbf16>, %arg7: tensor<2x2x2xbf16>) -> (tensor<2x2xbf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
    %1 = stablehlo.convert %0 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    %2:2 = stablehlo.composite "chlo.top_k" %1 {composite_attributes = {k = 1 : i64}, decomposition = @chlo.top_k.impl, version = 1 : i32} : (tensor<2x2xbf16>) -> (tensor<2x1xbf16>, tensor<2x1xi32>)
    %3 = stablehlo.convert %2#0 : (tensor<2x1xbf16>) -> tensor<2x1xf32>
    %4 = stablehlo.reduce(%3 init: %cst_2) applies stablehlo.maximum across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %6 = stablehlo.subtract %3, %5 : tensor<2x1xf32>
    %7 = stablehlo.exponential %6 : tensor<2x1xf32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %10 = stablehlo.divide %7, %9 : tensor<2x1xf32>
    %11 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [1] : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
    %12 = stablehlo.convert %11 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    %13 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [1] : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
    %14 = stablehlo.convert %13 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    %15 = stablehlo.negate %12 : tensor<2x2xbf16>
    %16 = stablehlo.exponential %15 : tensor<2x2xbf16>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2xbf16>
    %18 = stablehlo.add %17, %16 : tensor<2x2xbf16>
    %19 = stablehlo.divide %12, %18 : tensor<2x2xbf16>
    %20 = stablehlo.multiply %19, %14 : tensor<2x2xbf16>
    %21 = stablehlo.dot_general %20, %arg4, contracting_dims = [1] x [1] : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
    %22 = stablehlo.convert %21 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    %23 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %24 = stablehlo.compare LT, %2#1, %23, SIGNED : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi1>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %26 = stablehlo.add %2#1, %25 : tensor<2x1xi32>
    %27 = stablehlo.select %24, %26, %2#1 : tensor<2x1xi1>, tensor<2x1xi32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x1x1xi32>
    %29 = "stablehlo.gather"(%arg5, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 2>}> : (tensor<2x2x2xbf16>, tensor<2x1x1xi32>) -> tensor<2x1x2x2xbf16>
    %30 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %31 = stablehlo.compare LT, %2#1, %30, SIGNED : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi1>
    %32 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %33 = stablehlo.add %2#1, %32 : tensor<2x1xi32>
    %34 = stablehlo.select %31, %33, %2#1 : tensor<2x1xi1>, tensor<2x1xi32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x1x1xi32>
    %36 = "stablehlo.gather"(%arg6, %35) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 2>}> : (tensor<2x2x2xbf16>, tensor<2x1x1xi32>) -> tensor<2x1x2x2xbf16>
    %37 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %38 = stablehlo.compare LT, %2#1, %37, SIGNED : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi1>
    %39 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %40 = stablehlo.add %2#1, %39 : tensor<2x1xi32>
    %41 = stablehlo.select %38, %40, %2#1 : tensor<2x1xi1>, tensor<2x1xi32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x1x1xi32>
    %43 = "stablehlo.gather"(%arg7, %42) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 2>}> : (tensor<2x2x2xbf16>, tensor<2x1x1xi32>) -> tensor<2x1x2x2xbf16>
    %44 = stablehlo.dot_general %arg0, %29, batching_dims = [0] x [0], contracting_dims = [1] x [3] : (tensor<2x2xbf16>, tensor<2x1x2x2xbf16>) -> tensor<2x1x2xf32>
    %45 = stablehlo.convert %44 : (tensor<2x1x2xf32>) -> tensor<2x1x2xbf16>
    %46 = stablehlo.dot_general %arg0, %36, batching_dims = [0] x [0], contracting_dims = [1] x [3] : (tensor<2x2xbf16>, tensor<2x1x2x2xbf16>) -> tensor<2x1x2xf32>
    %47 = stablehlo.convert %46 : (tensor<2x1x2xf32>) -> tensor<2x1x2xbf16>
    %48 = stablehlo.negate %45 : tensor<2x1x2xbf16>
    %49 = stablehlo.exponential %48 : tensor<2x1x2xbf16>
    %50 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x1x2xbf16>
    %51 = stablehlo.add %50, %49 : tensor<2x1x2xbf16>
    %52 = stablehlo.divide %45, %51 : tensor<2x1x2xbf16>
    %53 = stablehlo.multiply %52, %47 : tensor<2x1x2xbf16>
    %54 = stablehlo.dot_general %53, %43, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [3] : (tensor<2x1x2xbf16>, tensor<2x1x2x2xbf16>) -> tensor<2x1x2xf32>
    %55 = stablehlo.convert %54 : (tensor<2x1x2xf32>) -> tensor<2x1x2xbf16>
    %56 = stablehlo.convert %55 : (tensor<2x1x2xbf16>) -> tensor<2x1x2xf32>
    %57 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x2xf32>
    %59 = stablehlo.multiply %56, %58 : tensor<2x1x2xf32>
    %60 = stablehlo.reduce(%59 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<2x1x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    %61 = stablehlo.convert %60 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    %62 = stablehlo.add %22, %61 : tensor<2x2xbf16>
    return %62 : tensor<2x2xbf16>
  }
  func.func private @chlo.top_k.impl(%arg0: tensor<2x2xbf16>) -> (tensor<2x1xbf16>, tensor<2x1xi32>) {
    %0 = stablehlo.iota dim = 1 : tensor<2x2xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %4 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) : (tensor<2x2xbf16>, tensor<2x2xi32>) -> (tensor<2x2xbf16>, tensor<2x2xi32>)
    %2 = stablehlo.slice %1#0 [0:2, 0:1] : (tensor<2x2xbf16>) -> tensor<2x1xbf16>
    %3 = stablehlo.slice %1#1 [0:2, 0:1] : (tensor<2x2xi32>) -> tensor<2x1xi32>
    return %2, %3 : tensor<2x1xbf16>, tensor<2x1xi32>
  }
}
