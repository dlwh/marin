// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// The unused supported scalar separates one dataflow component into two
// source-contiguous candidates. The expansion must not traverse the earlier
// candidate to claim its Fold as an ancestor.
module @fold_ancestry_out_of_component {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %separator = stablehlo.constant dense<1.0> : tensor<f32>
    %expanded = stablehlo.broadcast_in_dim %p, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %scaled = stablehlo.multiply %f, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %scaled : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}
