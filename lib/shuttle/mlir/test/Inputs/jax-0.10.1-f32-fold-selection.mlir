// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @jit_generic_selection {
  func.func public @main(%arg0: tensor<2x4x3xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xi32>) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %scores = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum
        across dimensions = [2]
        : (tensor<2x4x3xf32>, tensor<f32>) -> tensor<2x4xf32>
    %selected:2 = stablehlo.composite "chlo.top_k" %scores
        {composite_attributes = {k = 2 : i64},
         decomposition = @chlo.top_k.impl, version = 1 : i32}
        : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    return %selected#0, %selected#1 : tensor<2x2xf32>, tensor<2x2xi32>
  }

  func.func private @chlo.top_k.impl(%arg0: tensor<2x4xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xi32>) {
    %indices = stablehlo.iota dim = 1 : tensor<2x4xi32>
    %sorted:2 = "stablehlo.sort"(%arg0, %indices)
        <{dimension = 1 : i64, is_stable = true}> ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>,
          %lhs_index: tensor<i32>, %rhs_index: tensor<i32>):
        %greater = stablehlo.compare GT, %lhs, %rhs, TOTALORDER
            : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %greater : tensor<i1>
    }) : (tensor<2x4xf32>, tensor<2x4xi32>)
        -> (tensor<2x4xf32>, tensor<2x4xi32>)
    %values = stablehlo.slice %sorted#0 [0:2, 0:2]
        : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %selected_indices = stablehlo.slice %sorted#1 [0:2, 0:2]
        : (tensor<2x4xi32>) -> tensor<2x2xi32>
    return %values, %selected_indices : tensor<2x2xf32>, tensor<2x2xi32>
  }
}
