// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --split-input-file --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %s | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>

module @eligible_projected_input attributes {
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
      %scaled = "shuttle.map"(%region_input, %broadcast) ({
      ^bb0(%left: f32, %right: f32):
        %product = arith.mulf %left, %right : f32
        "shuttle.yield"(%product) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%scaled, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "eligible_projected_input"
// CHECK: "shuttle.materialization_task"()
// CHECK-SAME: composed_input_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0 floordiv 2, d1)>]
// CHECK-SAME: fused_sources = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>]
// CHECK: "shuttle.materialization_task"()
// CHECK-SAME: kind = #shuttle.materialization_task_kind<fold>
// CHECK-NOT: "shuttle.materialization_task"()

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>

module @fanout_boundary attributes {
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
      %left = "shuttle.map"(%region_input, %broadcast) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %product = arith.mulf %lhs, %rhs : f32
        "shuttle.yield"(%product) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %right = "shuttle.map"(%region_input, %broadcast) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %difference = arith.subf %lhs, %rhs : f32
        "shuttle.yield"(%difference) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %joined = "shuttle.map"(%left, %right) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 3>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%joined, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 4>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>, #shuttle.source_ref<0, 0, 0, 4>]} : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "fanout_boundary"
// CHECK: "shuttle.materialization_task"()
// CHECK-SAME: source = #shuttle.source_ref<0, 0, 0, 0>
// CHECK-NOT: fused_sources
// CHECK: "shuttle.materialization_task"()
// CHECK-SAME: fused_sources = [#shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>]
// CHECK: "shuttle.materialization_task"()
// CHECK-SAME: kind = #shuttle.materialization_task_kind<fold>
// CHECK-NOT: "shuttle.materialization_task"()

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>

module @duplicate_use_boundary attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%weight: tensor<1x4xf32>, %init: tensor<f32>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%weight, %init) ({
    ^bb0(%region_weight: tensor<1x4xf32>, %region_init: tensor<f32>):
      %broadcast = "shuttle.map"(%region_weight) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#projected, #id], semantics = #shuttle.map_semantics<broadcast_in_dim>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<1x4xf32>) -> tensor<2x4xf32>
      %squared = "shuttle.map"(%broadcast, %broadcast) ({
      ^bb0(%left: f32, %right: f32):
        %product = arith.mulf %left, %right : f32
        "shuttle.yield"(%product) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%squared, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<1x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "duplicate_use_boundary"
// CHECK-COUNT-3: "shuttle.materialization_task"
// CHECK-NOT: fused_sources

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>

module @public_result_boundary attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%weight: tensor<1x4xf32>, %input: tensor<2x4xf32>,
      %init: tensor<f32>) -> (tensor<2x4xf32>, tensor<2xf32>) {
    %broadcast, %fold = "shuttle.region"(%weight, %input, %init) ({
    ^bb0(%region_weight: tensor<1x4xf32>,
         %region_input: tensor<2x4xf32>, %region_init: tensor<f32>):
      %expanded = "shuttle.map"(%region_weight) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#projected, #id], semantics = #shuttle.map_semantics<broadcast_in_dim>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<1x4xf32>) -> tensor<2x4xf32>
      %scaled = "shuttle.map"(%region_input, %expanded) ({
      ^bb0(%left: f32, %right: f32):
        %product = arith.mulf %left, %right : f32
        "shuttle.yield"(%product) : (f32) -> ()
      }) {indexing_maps = [#id, #id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%scaled, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%expanded, %folded) : (tensor<2x4xf32>, tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<f32>) -> (tensor<2x4xf32>, tensor<2xf32>)
    return %broadcast, %fold : tensor<2x4xf32>, tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "public_result_boundary"
// CHECK-COUNT-3: "shuttle.materialization_task"
// CHECK-NOT: fused_sources

// -----

#small = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d0 floordiv 2, d1)>
#large = affine_map<(d0, d1) -> (d0, d1)>

module @nonidentity_cross_domain_boundary attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<1x4xf32>, %init: tensor<f32>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%input, %init) ({
    ^bb0(%region_input: tensor<1x4xf32>, %region_init: tensor<f32>):
      %negative = "shuttle.map"(%region_input) ({
      ^bb0(%element: f32):
        %value = arith.negf %element : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#small, #small], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
      %broadcast = "shuttle.map"(%negative) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#projected, #large], semantics = #shuttle.map_semantics<broadcast_in_dim>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<1x4xf32>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%broadcast, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<1x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "nonidentity_cross_domain_boundary"
// CHECK-COUNT-3: "shuttle.materialization_task"
// CHECK-NOT: fused_sources

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>

module @bf16_boundary attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x4xf32>, %init: tensor<f32>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%input, %init) ({
    ^bb0(%region_input: tensor<2x4xf32>, %region_init: tensor<f32>):
      %rounded = "shuttle.map"(%region_input) ({
      ^bb0(%element: f32):
        %value = "shuttle.scalar_convert"(%element) {semantics = #shuttle.scalar_convert_semantics<round_nearest_even>} : (f32) -> bf16
        "shuttle.yield"(%value) : (bf16) -> ()
      }) {indexing_maps = [#id, #id], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>) -> tensor<2x4xbf16>
      %expanded = "shuttle.map"(%rounded) ({
      ^bb0(%element: bf16):
        %value = "shuttle.scalar_convert"(%element) {semantics = #shuttle.scalar_convert_semantics<exact>} : (bf16) -> f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xbf16>) -> tensor<2x4xf32>
      %folded = "shuttle.fold"(%expanded, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-LABEL: sym_name = "bf16_boundary"
// CHECK-COUNT-3: "shuttle.materialization_task"
// CHECK-NOT: fused_sources
