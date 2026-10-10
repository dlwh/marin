// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @bad_count(%arg0: tensor<2x4xf32>) {
    %region:2 = "shuttle.region"(%arg0) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 0, 1>]
    }> ({
    ^bb0(%input: tensor<2x4xf32>):
    // expected-error @+1 {{requires an in-range selection dimension and a positive count no larger than that dimension}}
    %selected:2 = "shuttle.selection"(%input) <{
      count = 5 : i64, dimension = 1 : i64,
      comparison = #shuttle.selection_comparison<total_order>,
      order = #shuttle.selection_order<descending>, stable = true,
      tie_break = #shuttle.selection_tie_break<source_index_ascending>,
      sources = [#shuttle.source_ref<0, 0, 0, 0>,
                 #shuttle.source_ref<0, 0, 0, 1>]
    }> : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    "shuttle.yield"(%selected#0, %selected#1)
        : (tensor<2x2xf32>, tensor<2x2xi32>) -> ()
    }) : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    return
  }
}

// -----

module {
  func.func @bad_result_sources(%arg0: tensor<2x4xf32>) {
    %region:2 = "shuttle.region"(%arg0) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 0, 1>]
    }> ({
    ^bb0(%input: tensor<2x4xf32>):
    // expected-error @+1 {{requires ordered source references matching result ordinals}}
    %selected:2 = "shuttle.selection"(%input) <{
      count = 2 : i64, dimension = 1 : i64,
      comparison = #shuttle.selection_comparison<total_order>,
      order = #shuttle.selection_order<descending>, stable = true,
      tie_break = #shuttle.selection_tie_break<source_index_ascending>,
      sources = [#shuttle.source_ref<0, 0, 0, 0>,
                 #shuttle.source_ref<0, 0, 0, 0>]
    }> : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    "shuttle.yield"(%selected#0, %selected#1)
        : (tensor<2x2xf32>, tensor<2x2xi32>) -> ()
    }) : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    return
  }
}

// -----

module {
  func.func @bad_result_shape(%arg0: tensor<2x4xf32>) {
    %region:2 = "shuttle.region"(%arg0) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 0, 1>]
    }> ({
    ^bb0(%input: tensor<2x4xf32>):
    // expected-error @+1 {{requires result shapes equal to the input shape with the selection dimension replaced by count}}
    %selected:2 = "shuttle.selection"(%input) <{
      count = 2 : i64, dimension = 1 : i64,
      comparison = #shuttle.selection_comparison<total_order>,
      order = #shuttle.selection_order<descending>, stable = true,
      tie_break = #shuttle.selection_tie_break<source_index_ascending>,
      sources = [#shuttle.source_ref<0, 0, 0, 0>,
                 #shuttle.source_ref<0, 0, 0, 1>]
    }> : (tensor<2x4xf32>) -> (tensor<2x3xf32>, tensor<2x3xi32>)
    "shuttle.yield"(%selected#0, %selected#1)
        : (tensor<2x3xf32>, tensor<2x3xi32>) -> ()
    }) : (tensor<2x4xf32>) -> (tensor<2x3xf32>, tensor<2x3xi32>)
    return
  }
}

// -----

module {
  func.func @unstable_tie_break(%arg0: tensor<2x4xf32>) {
    %region:2 = "shuttle.region"(%arg0) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 0, 1>]
    }> ({
    ^bb0(%input: tensor<2x4xf32>):
    // expected-error @+1 {{requires stable source-index-ascending tie breaking}}
    %selected:2 = "shuttle.selection"(%input) <{
      count = 2 : i64, dimension = 1 : i64,
      comparison = #shuttle.selection_comparison<total_order>,
      order = #shuttle.selection_order<descending>, stable = false,
      tie_break = #shuttle.selection_tie_break<source_index_ascending>,
      sources = [#shuttle.source_ref<0, 0, 0, 0>,
                 #shuttle.source_ref<0, 0, 0, 1>]
    }> : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    "shuttle.yield"(%selected#0, %selected#1)
        : (tensor<2x2xf32>, tensor<2x2xi32>) -> ()
    }) : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
    return
  }
}
