// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>
#map_input = affine_map<(m, k) -> (m, k)>
#map_output = affine_map<(m, n) -> (m, n)>

module @generic_map_contract_map_component attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @entry(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>)
      -> (tensor<2x3xf32>, tensor<2x4xf32>) {
    %mapped_input, %mapped_output = "shuttle.region"(%a, %b) ({
    ^bb0(%region_a: tensor<2x3xf32>, %region_b: tensor<3x4xf32>):
      %negated_input = "shuttle.map"(%region_a) ({
      ^bb0(%element: f32):
        %value = arith.negf %element : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#map_input, #map_input],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>) -> tensor<2x3xf32>
      %contracted = "shuttle.contract"(%negated_input, %region_b) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#lhs, #rhs, #out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 1, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      %negated_output = "shuttle.map"(%contracted) ({
      ^bb0(%element: f32):
        %value = arith.negf %element : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#map_output, #map_output],
        source = #shuttle.source_ref<0, 0, 2, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%negated_input, %negated_output)
          : (tensor<2x3xf32>, tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 1, 0>,
                     #shuttle.source_ref<0, 0, 2, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>)
        -> (tensor<2x3xf32>, tensor<2x4xf32>)
    return %mapped_input, %mapped_output
        : tensor<2x3xf32>, tensor<2x4xf32>
  }
}
