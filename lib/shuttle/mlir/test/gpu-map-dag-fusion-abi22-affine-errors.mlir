// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @symbol(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%input) ({
    ^bb0(%region_input: tensor<2x4xf32>):
      // expected-error @+1 {{map indexing maps must not contain affine symbols}}
      %mapped = "shuttle.map"(%region_input) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, d1)>, affine_map<(d0, d1)[s0] -> (d0, d1)>], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @unclosed_composition(%input: tensor<1x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%input) ({
    ^bb0(%region_input: tensor<1x4xf32>):
      // expected-error @+1 {{bounded-zero floordiv input indexing is reserved for typed static singleton broadcasts}}
      %mapped = "shuttle.map"(%region_input) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [affine_map<(d0, d1) -> (d0 floordiv 3, d1)>, affine_map<(d0, d1) -> (d0, d1)>], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<1x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<1x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
