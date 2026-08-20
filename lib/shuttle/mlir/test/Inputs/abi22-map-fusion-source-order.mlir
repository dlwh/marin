// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#id = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>

module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%weight: tensor<1x4xf32>, %input: tensor<2x4xf32>,
      %init: tensor<f32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%weight, %input, %init) ({
    ^bb0(%region_weight: tensor<1x4xf32>,
         %region_input: tensor<2x4xf32>, %region_init: tensor<f32>):
      %broadcast = "shuttle.map"(%region_weight) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#projected, #id], semantics = #shuttle.map_semantics<broadcast_in_dim>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<1x4xf32>) -> tensor<2x4xf32>
      %normalized = "shuttle.map"(%region_input, %broadcast) ({
      ^bb0(%left: f32, %right: f32):
        %difference = arith.subf %left, %right : f32
        %quotient = arith.divf %difference, %right : f32
        "shuttle.yield"(%quotient) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%normalized, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
