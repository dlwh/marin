// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @all_axis_fold_rank1 {
  func.func @main(%input: tensor<513xf32>, %init: tensor<f32>) -> tensor<f32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %sum = stablehlo.add %element, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 0>} : (tensor<513xf32>, tensor<f32>) -> tensor<f32>
    return %result : tensor<f32>
  }
}
