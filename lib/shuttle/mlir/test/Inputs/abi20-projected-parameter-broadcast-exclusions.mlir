// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// A same-rank public singleton is not a projected lower-rank parameter.
module @same_rank_public {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<1x1x19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %w = stablehlo.convert %weight : (tensor<1x1x19xbf16>) -> tensor<1x1x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %b = stablehlo.broadcast_in_dim %w, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %b : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}
// -----

// Pointwise arithmetic breaks the exact live-in conversion/structural path.
module @pointwise_parameter_driver {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %w = stablehlo.convert %weight : (tensor<19xbf16>) -> tensor<19xf32>
    %bad = stablehlo.multiply %w, %w : tensor<19xf32>
    %insert = stablehlo.broadcast_in_dim %bad, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %expanded = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Public f32 weight is the sole invalid public-boundary property.
module @public_f32_weight {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<19xf32>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %insert = stablehlo.broadcast_in_dim %weight, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %expanded = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// A scalar public value is below the rank-one parameter boundary.
module @public_scalar_weight {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<bf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %w = stablehlo.convert %weight : (tensor<bf16>) -> tensor<f32>
    %expanded = stablehlo.broadcast_in_dim %w, dims = [] : (tensor<f32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Map-only projected parameter expansion remains outside the anchored region.
module @map_only_parameter {
  func.func public @main(%weight: tensor<19xbf16>) -> tensor<2x7x19xbf16> {
    %w = stablehlo.convert %weight : (tensor<19xbf16>) -> tensor<19xf32>
    %insert = stablehlo.broadcast_in_dim %w, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %expanded = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %o = stablehlo.convert %expanded : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Non-final reduction remains outside the terminal Fold boundary.
module @nonfinal_fold {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %w = stablehlo.convert %weight : (tensor<19xbf16>) -> tensor<19xf32>
    %insert = stablehlo.broadcast_in_dim %w, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %expanded = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x19xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 2] : (tensor<2x19xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// Multiple reduction dimensions remain outside the one-final-axis boundary.
module @multiple_fold_dimensions {
  func.func public @main(%x: tensor<2x7x19xbf16>, %weight: tensor<19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %w = stablehlo.convert %weight : (tensor<19xbf16>) -> tensor<19xf32>
    %insert = stablehlo.broadcast_in_dim %w, dims = [2] : (tensor<19xf32>) -> tensor<1x1x19xf32>
    %expanded = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<1x1x19xf32>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [2] : (tensor<19xf32>) -> tensor<2x7x19xf32>
    %m = stablehlo.multiply %p, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %m : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// A Fold-derived expansion may not acquire a second public tensor leaf through
// otherwise closed direct pointwise Maps.
module @fold_ancestry_external_leaf {
  func.func public @main(%x: tensor<2x7x19xbf16>, %bias: tensor<2x7x1xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %bias_f32 = stablehlo.convert %bias : (tensor<2x7x1xbf16>) -> tensor<2x7x1xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %p = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %merged = stablehlo.add %p, %bias_f32 : tensor<2x7x1xf32>
    %expanded = stablehlo.broadcast_in_dim %merged, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %scaled = stablehlo.multiply %f, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %scaled : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// The direct-Map producer graph of one expansion may contain only one
// terminal additive Fold.
module @fold_ancestry_second_fold {
  func.func public @main(%x: tensor<2x7x19xbf16>, %y: tensor<2x7x19xbf16>) -> tensor<2x7x19xbf16> {
    %xf = stablehlo.convert %x : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %yf = stablehlo.convert %y : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %rx = stablehlo.reduce(%xf init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %ry = stablehlo.reduce(%yf init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %px = stablehlo.broadcast_in_dim %rx, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %py = stablehlo.broadcast_in_dim %ry, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %merged = stablehlo.add %px, %py : tensor<2x7x1xf32>
    %expanded = stablehlo.broadcast_in_dim %merged, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %scaled = stablehlo.multiply %xf, %expanded : tensor<2x7x19xf32>
    %o = stablehlo.convert %scaled : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}

// -----

// An expanding singleton Map is not a closed direct-Map step in another
// expansion's Fold ancestry.
module @fold_ancestry_nested_expansion {
  func.func public @main(%x: tensor<2x1x19xbf16>) -> tensor<2x7x19xbf16> {
    %f = stablehlo.convert %x : (tensor<2x1x19xbf16>) -> tensor<2x1x19xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%f init: %z) applies stablehlo.add across dimensions = [2] : (tensor<2x1x19xf32>, tensor<f32>) -> tensor<2x1xf32>
    %insert = stablehlo.broadcast_in_dim %r, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %nested = stablehlo.broadcast_in_dim %insert, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x7x1xf32>
    %expanded = stablehlo.broadcast_in_dim %nested, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %o = stablehlo.convert %expanded : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %o : tensor<2x7x19xbf16>
  }
}
