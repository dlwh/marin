// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Local algebra emitted by JAX's VJP for a maximum Fold. Floating equality is
// pointwise value algebra, rather than an index-domain Relation.
module @maximum_vjp_tie_map {
  func.func public @main(%score: tensor<1x2x4x8xf32>, %maximum: tensor<1x2x4x8xf32>) -> tensor<1x2x4xf32> {
    %equal = stablehlo.compare EQ, %score, %maximum, FLOAT : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xi1>
    %weights = stablehlo.convert %equal : (tensor<1x2x4x8xi1>) -> tensor<1x2x4x8xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %count = stablehlo.reduce(%weights init: %zero) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    return %count : tensor<1x2x4xf32>
  }
}
