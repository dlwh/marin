// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Two expanding singleton axes exceed the exactly-one Map boundary.
module @two_expansions {
  func.func public @main(%x: tensor<2x1x1xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x1x1xbf16>) -> tensor<2x1x1xf32>
    %b = stablehlo.broadcast_in_dim %f, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%b init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %b, %p : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Maximum remains outside additive Fold semantics. The wrapper otherwise has
// the ABI18 singleton insertion and expansion Maps and a BF16 public boundary.
module @maximum {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %v = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %v : tensor<f32>
      }
    %s = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %s, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Internal i1 compare/select/conversion Maps are deferred. The additive Fold
// and singleton expansion otherwise match ABI18 exactly.
module @i1_maps {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %s = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %s, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %c = stablehlo.compare  GT, %f, %b,  FLOAT : (tensor<2x7x19xf32>, tensor<2x7x19xf32>) -> tensor<2x7x19xi1>
    %one = stablehlo.constant dense<1.0> : tensor<2x7x19xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<2x7x19xf32>
    %selected = stablehlo.select %c, %one, %zero : tensor<2x7x19xi1>, tensor<2x7x19xf32>
    %o = stablehlo.convert %selected : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// A singleton expansion without a driving Fold or Contract is Map-only.
module @map_only {
  func.func public @main(%x: tensor<2x7x1xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x1xbf16>) -> tensor<2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %f, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %o = stablehlo.convert %b : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Contract and rank-N Fold remain separate closed region families.
module @mixed_contract_fold {
  func.func public @main(%lhs: tensor<2x7x5xbf16>, %rhs: tensor<2x5x19xbf16>) -> tensor<2x7x19xbf16> {
    %dot = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x7x5xbf16>, tensor<2x5x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%dot init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %s = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %s, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %dot, %b : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Public rank nine exceeds both the semantic and descriptor boundaries.
module @rank_nine {
  func.func public @main(%x: tensor<1x1x1x1x1x1x2x7x19xbf16>) -> tensor<1x1x1x1x1x1x2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<1x1x1x1x1x1x2x7x19xbf16>) -> tensor<1x1x1x1x1x1x2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [8] : (tensor<1x1x1x1x1x1x2x7x19xf32>, tensor<f32>) -> tensor<1x1x1x1x1x1x2x7xf32>
    %s = stablehlo.broadcast_in_dim %r, dims = [0, 1, 2, 3, 4, 5, 6, 7] : (tensor<1x1x1x1x1x1x2x7xf32>) -> tensor<1x1x1x1x1x1x2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %s, dims = [0, 1, 2, 3, 4, 5, 6, 7, 8] : (tensor<1x1x1x1x1x1x2x7x1xf32>) -> tensor<1x1x1x1x1x1x2x7x19xf32>
    %m = stablehlo.multiply %f, %b : tensor<1x1x1x1x1x1x2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<1x1x1x1x1x1x2x7x19xf32>) -> tensor<1x1x1x1x1x1x2x7x19xbf16>
    return %o : tensor<1x1x1x1x1x1x2x7x19xbf16>
  }
}

// -----

// Public f32 is isolated as the sole invalid property; the full terminal Fold,
// singleton insertion, expansion, and output shape remain present.
module @public_f32 {
  func.func public @main(%x: tensor<2x7x19xf32>) -> tensor<2x7x19xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %s = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %b = stablehlo.broadcast_in_dim %s, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %o = stablehlo.multiply %x, %b : tensor<2x7x19xf32>
    return %o : tensor<2x7x19xf32>
  }
}
