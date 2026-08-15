// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Anonymous mutation fixture for the ABI21 full-prefix Fold boundary.
module @anonymous {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}
