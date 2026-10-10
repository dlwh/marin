// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @unknown_form(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xbf16>, %region_rhs: tensor<3x4xbf16>):
      // expected-error @+1 {{source precision form must be omitted or empty with DEFAULT precision for every input}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [bf16],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source_precision_form = "default",
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
      "shuttle.yield"(%contract) : (tensor<2x4xbf16>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %result : tensor<2x4xbf16>
  }
}

// -----

module {
  func.func @empty_nondefault(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xbf16>, %region_rhs: tensor<3x4xbf16>):
      // expected-error @+1 {{source precision form must be omitted or empty with DEFAULT precision for every input}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [bf16],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["HIGH", "HIGH"],
        source_precision_form = "empty",
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
      "shuttle.yield"(%contract) : (tensor<2x4xbf16>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %result : tensor<2x4xbf16>
  }
}

// -----

module {
  func.func @omitted_nondefault(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xbf16>, %region_rhs: tensor<3x4xbf16>):
      // expected-error @+1 {{source precision form must be omitted or empty with DEFAULT precision for every input}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [bf16],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "HIGHEST"],
        source_precision_form = "omitted",
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
      "shuttle.yield"(%contract) : (tensor<2x4xbf16>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %result : tensor<2x4xbf16>
  }
}
