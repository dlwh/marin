// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Anonymous algebra fixture for h=Contract0; s=Fold(h); z=Map(h,s);
// y=Contract1(z). Names and dimensions are not compiler selectors.

#contract0_lhs = affine_map<(m, n, k) -> (m, k)>
#contract0_rhs = affine_map<(m, n, k) -> (k, n)>
#contract0_out = affine_map<(m, n, k) -> (m, n)>
#identity = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>
#contract1_lhs = affine_map<(m, o, n) -> (m, n)>
#contract1_rhs = affine_map<(m, o, n) -> (n, o)>
#contract1_out = affine_map<(m, o, n) -> (m, o)>

module @anonymous_graph_a attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @entry(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>,
      %p: tensor<5x7xf32>, %init: tensor<f32>) -> tensor<2x7xf32> {
    %result = "shuttle.region"(%a, %b, %p, %init) ({
    ^bb0(%region_a: tensor<2x3xf32>, %region_b: tensor<3x5xf32>,
         %region_p: tensor<5x7xf32>, %region_init: tensor<f32>):
      %h = "shuttle.contract"(%region_a, %region_b) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract0_lhs, #contract0_rhs, #contract0_out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x5xf32>) -> tensor<2x5xf32>
      %s = "shuttle.fold"(%h, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>, order_free = true,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 1>
      } : (tensor<2x5xf32>, tensor<f32>) -> tensor<2xf32>
      %z = "shuttle.map"(%h, %s) ({
      ^bb0(%element: f32, %scale: f32):
        %scaled = arith.mulf %element, %scale : f32
        "shuttle.yield"(%scaled) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row, #identity],
        source = #shuttle.source_ref<0, 0, 0, 2>
      } : (tensor<2x5xf32>, tensor<2xf32>) -> tensor<2x5xf32>
      %y = "shuttle.contract"(%z, %region_p) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract1_lhs, #contract1_rhs, #contract1_out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 3>
      } : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
      "shuttle.yield"(%y) : (tensor<2x7xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 0, 1>,
                     #shuttle.source_ref<0, 0, 0, 2>,
                     #shuttle.source_ref<0, 0, 0, 3>]
    } : (tensor<2x3xf32>, tensor<3x5xf32>, tensor<5x7xf32>, tensor<f32>)
        -> tensor<2x7xf32>
    return %result : tensor<2x7xf32>
  }
}
