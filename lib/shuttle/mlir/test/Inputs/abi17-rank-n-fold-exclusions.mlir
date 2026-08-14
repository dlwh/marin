// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Non-final rank-N reduction.
module @nonfinal {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [1] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x17xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 2] : (tensor<2x17xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// Multiple reduction dimensions.
module @multiple {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [1, 2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0] : (tensor<2xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// Maximum is not additive Fold semantics.
module @maximum {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %v = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %v : tensor<f32>
      }
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// The initializer must be positive zero exactly.
module @negative_zero {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<-0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// A nonzero initializer is outside additive Fold semantics.
module @nonzero {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<1.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// Public f32 is outside the BF16/f32 boundary.
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
  func.func public @main(%x: tensor<1x1x1x1x1x1x1x2x17xbf16>) -> tensor<1x1x1x1x1x1x1x2x17xbf16> {
    %f = stablehlo.convert %x : (tensor<1x1x1x1x1x1x1x2x17xbf16>) -> tensor<1x1x1x1x1x1x1x2x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [8] : (tensor<1x1x1x1x1x1x1x2x17xf32>, tensor<f32>) -> tensor<1x1x1x1x1x1x1x2xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1, 2, 3, 4, 5, 6, 7] : (tensor<1x1x1x1x1x1x1x2xf32>) -> tensor<1x1x1x1x1x1x1x2x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<1x1x1x1x1x1x1x2x17xf32>
    %o = stablehlo.convert %m : (tensor<1x1x1x1x1x1x1x2x17xf32>) -> tensor<1x1x1x1x1x1x1x2x17xbf16>
    return %o : tensor<1x1x1x1x1x1x1x2x17xbf16>
  }
}

// -----

// Dynamic extents have no closed descriptor geometry.
module @dynamic {
  func.func public @main(%x: tensor<2x?x17xbf16>) -> tensor<2x?x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x?x17xbf16>) -> tensor<2x?x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x?x17xf32>, tensor<f32>) -> tensor<2x?xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x?xf32>) -> tensor<2x?x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x?x17xf32>
    %o = stablehlo.convert %m : (tensor<2x?x17xf32>) -> tensor<2x?x17xbf16>
    return %o : tensor<2x?x17xbf16>
  }
}

// -----

// Mixed Contract/Fold regions are intentionally deferred.
module @mixed_contract_fold {
  func.func public @main(%lhs: tensor<2x5x7xbf16>, %rhs: tensor<2x7x17xbf16>) -> tensor<2x5x17xbf16> {
    %dot = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x5x7xbf16>, tensor<2x7x17xbf16>) -> tensor<2x5x17xbf16>
    %f = stablehlo.convert %dot : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// A terminal Fold with a non-add reducer is intentionally deferred.
module @multiply {
  func.func public @main(%x: tensor<2x5x17xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5x17xbf16>) -> tensor<2x5x17xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) across dimensions = [2] : (tensor<2x5x17xf32>, tensor<f32>) -> tensor<2x5xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %v = stablehlo.multiply %a, %b : tensor<f32>
        stablehlo.return %v : tensor<f32>
      }
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x5x17xf32>
    %o = stablehlo.convert %m : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}

// -----

// A projected broadcast without a driving Fold is outside this slice.
module @map_only_projected_broadcast {
  func.func public @main(%x: tensor<2x5xbf16>) -> tensor<2x5x17xbf16> {
    %f = stablehlo.convert %x : (tensor<2x5xbf16>) -> tensor<2x5xf32>
    %b = stablehlo.broadcast_in_dim %f, dims = [0, 1] : (tensor<2x5xf32>) -> tensor<2x5x17xf32>
    %o = stablehlo.convert %b : (tensor<2x5x17xf32>) -> tensor<2x5x17xbf16>
    return %o : tensor<2x5x17xbf16>
  }
}
