// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Non-final rank-N reduction.
module @nonfinal {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x17xf32> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [1] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x17xf32>
    return %r : tensor<2x17xf32>
  }
}

// -----

// Multiple reduction dimensions.
module @multiple {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2xf32> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [1, 2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
}

// -----

// Maximum is not additive Fold semantics.
module @maximum {
  func.func public @main(%x: tensor<2x5x17xf32>) -> tensor<2x5xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%x init: %z) across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %m = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %m : tensor<f32>
      }
    return %r : tensor<2x5xf32>
  }
}

// -----

// The initializer must be positive zero exactly.
module @negative_zero {
  func.func public @main(%x: tensor<2x5x17xf32>) -> tensor<2x5xf32> {
    %z = stablehlo.constant dense<-0.0> : tensor<f32>
    %r = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    return %r : tensor<2x5xf32>
  }
}

// -----

// Public and internal element types outside the BF16/f32 boundary.
module @public_f32 {
  func.func public @main(%x: tensor<2x5x17xf32>) -> tensor<2x5xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    return %r : tensor<2x5xf32>
  }
}

// -----

// Rank nine exceeds the public and descriptor rank cap.
module @rank_nine {
  func.func public @main(%x: tensor<1x1x1x1x1x1x1x2x17xbf16>) -> tensor<1x1x1x1x1x1x1x2xf32> {
    %f = stablehlo.convert %x : (tensor<1x1x1x1x1x1x1x2x17xbf16>) -> tensor<1x1x1x1x1x1x1x2x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [8] : (tensor<1x1x1x1x1x1x1x2x17xf32>, tensor<f32>) -> tensor<1x1x1x1x1x1x1x2xf32>
    return %r : tensor<1x1x1x1x1x1x1x2xf32>
  }
}

// -----

// Dynamic extents have no closed descriptor geometry.
module @dynamic {
  func.func public @main(%x: tensor<2x?x17xbf16>) -> tensor<2x?xf32> {
    %f = stablehlo.convert %x : (tensor<2x?x17xbf16>) -> tensor<2x?x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x?x17xf32>, tensor<f32>) -> tensor<2x?xf32>
    return %r : tensor<2x?xf32>
  }
}

// -----

// Mixed Contract/Fold regions are intentionally deferred.
module @mixed_contract_fold {
  func.func public @main(%lhs: tensor<2x5x7xbf16>, %rhs: tensor<2x7x17xbf16>) -> tensor<2x5xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x5x7xbf16>, tensor<2x7x17xbf16>) -> tensor<2x5x17xbf16>
    %f = stablehlo.convert %dot : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    return %r : tensor<2x5xf32>
  }
}
