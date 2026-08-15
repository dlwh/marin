// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Omitting a prefix dimension is not a full-prefix Fold.
module @missing_prefix {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<7x19xf32>
    %o = stablehlo.convert %r : (tensor<7x19xf32>) -> tensor<7x19xbf16>
    return %o : tensor<7x19xbf16>
  }
}

// -----

// A discontiguous reduction is not a full prefix.
module @discontiguous_prefix {
  func.func public @main(%x: tensor<2x7x5x19xbf16>) -> tensor<7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x5x19xbf16>) -> tensor<2x7x5x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 2] : (tensor<2x7x5x19xf32>, tensor<f32>) -> tensor<7x19xf32>
    %o = stablehlo.convert %r : (tensor<7x19xf32>) -> tensor<7x19xbf16>
    return %o : tensor<7x19xbf16>
  }
}

// -----

// Reducing the feature axis does not leave the final feature vector.
module @extra_feature_axis {
  func.func public @main(%x: tensor<2x7x19x5xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19x5xbf16>) -> tensor<2x7x19x5xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1, 3] : (tensor<2x7x19x5xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Positive zero is the only admitted initializer.
module @nonzero_init {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<1.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Negative zero has different IEEE-754 signed-zero semantics.
module @negative_zero_init {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<-0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Maximum is not the exact additive Fold body.
module @maximum_body {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %v = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %v : tensor<f32>
      }
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Multiplication is not the exact additive Fold body.
module @nonadd_body {
  func.func public @main(%x: tensor<2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
      reducer(%a: tensor<f32>, %b: tensor<f32>)  {
        %v = stablehlo.multiply %a, %b : tensor<f32>
        stablehlo.return %v : tensor<f32>
      }
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// A rank-one public output must be the converted full-prefix Fold result.
module @unrelated_rank_one_output {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<19xbf16>) -> (tensor<2x7x19xbf16>, tensor<19xbf16>) {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %b = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %f, %b : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o, %weight : tensor<2x7x19xbf16>, tensor<19xbf16>
  }
}

// -----

// Dynamic prefix extents have no closed reduction product.
module @dynamic_prefix {
  func.func public @main(%x: tensor<2x?x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x?x19xbf16>) -> tensor<2x?x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x?x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Empty prefix extents are outside the positive-static boundary.
module @empty_prefix {
  func.func public @main(%x: tensor<2x0x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x0x19xbf16>) -> tensor<2x0x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x0x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Rank nine exceeds the descriptor rank cap.
module @rank_nine {
  func.func public @main(%x: tensor<1x1x1x1x1x1x2x7x19xbf16>) -> tensor<19xbf16> {
    %f = stablehlo.convert %x : (tensor<1x1x1x1x1x1x2x7x19xbf16>) -> tensor<1x1x1x1x1x1x2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1, 2, 3, 4, 5, 6, 7] : (tensor<1x1x1x1x1x1x2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// Contract/Fold regions remain outside this slice.
module @mixed_contract_fold {
  func.func public @main(%lhs: tensor<2x7x5xbf16>, %rhs: tensor<2x5x19xbf16>) -> tensor<19xbf16> {
    %dot = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x7x5xbf16>, tensor<2x5x19xbf16>) -> tensor<2x7x19xbf16>
    %f = stablehlo.convert %dot : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %o = stablehlo.convert %r : (tensor<19xf32>) -> tensor<19xbf16>
    return %o : tensor<19xbf16>
  }
}

// -----

// The adjacent 2048x4096 cell remains an explicit unreviewed stop. This
// 2049x4096 mutation crosses the unchanged 32 MiB per-slot cap.
module @slot_cap_overflow {
  func.func public @main(%x: tensor<1x2049x4096xbf16>) -> tensor<4096xbf16> {
    %f = stablehlo.convert %x : (tensor<1x2049x4096xbf16>) -> tensor<1x2049x4096xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x2049x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %o = stablehlo.convert %r : (tensor<4096xf32>) -> tensor<4096xbf16>
    return %o : tensor<4096xbf16>
  }
}
