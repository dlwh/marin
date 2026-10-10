// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-fold-selection.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-selection-dead-result.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=MATERIALIZATION
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-selection-dead-result.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-fold-selection.mlir --shuttle-stablehlo-source-ordered-pipeline | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-fold-selection-unstable.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra | FileCheck %s --check-prefix=UNSTABLE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-selection.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=BF16
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-selection.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --mlir-print-op-generic | FileCheck %s --check-prefix=BF16-MAT
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-selection.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --mlir-print-op-generic | FileCheck %s --check-prefix=BF16-SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-selection.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering '--shuttle-build-gpu-executable-bundle=target=sm_90' --shuttle-verify-gpu-executable-bundle | FileCheck %s --check-prefix=BF16-GPU

// ALGEBRA: "shuttle.fold"
// ALGEBRA: "shuttle.selection"
// ALGEBRA-SAME: comparison = #shuttle.selection_comparison<total_order>
// ALGEBRA-SAME: count = 2 : i64
// ALGEBRA-SAME: dimension = 1 : i64
// ALGEBRA-SAME: order = #shuttle.selection_order<descending>
// ALGEBRA-SAME: sources = [#shuttle.source_ref<{{.*}}, 0>, #shuttle.source_ref<{{.*}}, 1>]
// ALGEBRA-SAME: stable = true
// ALGEBRA-SAME: tie_break = #shuttle.selection_tie_break<source_index_ascending>
// ALGEBRA-NOT: "stablehlo.composite"
// ALGEBRA-NOT: chlo.top_k
// ALGEBRA: shuttle.coverage_manifest

// LOWERED-NOT: shuttle.
// LOWERED-NOT: stablehlo.composite
// LOWERED-NOT: chlo.top_k
// LOWERED: %[[IOTA:.*]] = stablehlo.iota dim = 1 : tensor<2x4xi32>
// LOWERED: %[[SORT:.*]]:2 = "stablehlo.sort"({{.*}}, %[[IOTA]]) <{dimension = 1 : i64, is_stable = true}> ({

// BF16-GPU: "shuttle.device_module"
// BF16-GPU: "shuttle.invocation_slot"() <{{.*}}required_bytes = 16 : i64{{.*}}tensor_type = tensor<2x4xbf16>
// BF16-GPU: "shuttle.invocation_slot"() <{{.*}}required_bytes = 8 : i64{{.*}}tensor_type = tensor<2x2xbf16>
// BF16-GPU: "shuttle.invocation_slot"() <{{.*}}required_bytes = 16 : i64{{.*}}tensor_type = tensor<2x2xi32>
// BF16-GPU: "shuttle.executable_bundle"
// LOWERED: stablehlo.compare GT, {{.*}}, {{.*}}, TOTALORDER
// LOWERED: stablehlo.return
// LOWERED: }) : (tensor<2x4xf32>, tensor<2x4xi32>) -> (tensor<2x4xf32>, tensor<2x4xi32>)
// LOWERED: stablehlo.slice %[[SORT]]#0 [0:2, 0:2]
// LOWERED: stablehlo.slice %[[SORT]]#1 [0:2, 0:2]

// UNSTABLE: stablehlo.composite "chlo.top_k"

// BF16: "shuttle.selection"
// BF16-SAME: comparison = #shuttle.selection_comparison<total_order>
// BF16-SAME: stable = true
// BF16-SAME: tie_break = #shuttle.selection_tie_break<source_index_ascending>
// BF16-SAME: (tensor<2x4xbf16>) -> (tensor<2x2xbf16>, tensor<2x2xi32>)

// BF16-MAT: "shuttle.materialization_buffer"() <{{.*}}live_out = true{{.*}}storage = #shuttle.materialization_storage<external>{{.*}}tensor_type = tensor<2x2xbf16>
// BF16-MAT: "shuttle.materialization_buffer"() <{{.*}}live_out = true{{.*}}storage = #shuttle.materialization_storage<external>{{.*}}tensor_type = tensor<2x2xi32>

// BF16-SCHEDULE: "shuttle.schedule_task"() <{{.*}}kind = #shuttle.schedule_task_kind<row_fold>
// BF16-SCHEDULE-SAME: selection_comparison = #shuttle.selection_comparison<total_order>
// BF16-SCHEDULE-SAME: selection_count = 2 : i64
// BF16-SCHEDULE-SAME: selection_stable = true
// BF16-SCHEDULE-SAME: selection_tie_break = #shuttle.selection_tie_break<source_index_ascending>
// BF16-SCHEDULE: "shuttle.event_lowering_plan"
// UNSTABLE-NOT: shuttle.selection

// MATERIALIZATION: "shuttle.materialization_buffer"() <{{.*}}consumers = array<i64>{{.*}}lifetime_end = 0 : i64, lifetime_start = 0 : i64{{.*}}live_out = false{{.*}}producer = 0 : i64{{.*}}tensor_type = tensor<2x2xf32>
// MATERIALIZATION: "shuttle.materialization_task"() <{{.*}}domain_shape = array<i64: 2, 4>{{.*}}kind = #shuttle.materialization_task_kind<selection>
// MATERIALIZATION-SAME: output_buffers = array<i64: {{[0-9]+}}, {{[0-9]+}}>
// MATERIALIZATION-SAME: reduction_dimensions = array<i64: 1>
// MATERIALIZATION-SAME: source_group = [#shuttle.source_ref<{{.*}}, 0>, #shuttle.source_ref<{{.*}}, 1>]
// MATERIALIZATION-COUNT-2: kind = #shuttle.materialization_task_kind<relation>

// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 2, 4>
// SCHEDULE-SAME: kind = #shuttle.schedule_task_kind<row_fold>
// SCHEDULE-SAME: reduction_axis = 1 : i64
// SCHEDULE-SAME: reduction_order = #shuttle.schedule_reduction_order<stable_total_order_leaf_order_fixed>
// SCHEDULE-SAME: selection_comparison = #shuttle.selection_comparison<total_order>
// SCHEDULE-SAME: selection_count = 2 : i64
// SCHEDULE-SAME: selection_dimension = 1 : i64
// SCHEDULE-SAME: selection_order = #shuttle.selection_order<descending>
// SCHEDULE-SAME: selection_stable = true
// SCHEDULE-SAME: selection_tie_break = #shuttle.selection_tie_break<source_index_ascending>
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 2, 2, 1>
// SCHEDULE-SAME: kind = #shuttle.schedule_task_kind<elementwise>
// SCHEDULE-SAME: relation_parameters = []
// SCHEDULE-SAME: relation_semantics = #shuttle.relation_semantics<reshape>
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 2, 2, 3>
// SCHEDULE-SAME: kind = #shuttle.schedule_task_kind<elementwise>
// SCHEDULE-SAME: relation_parameters = [{{.*}}index_vector_dim = 2 : i64{{.*}}]
// SCHEDULE-SAME: relation_semantics = #shuttle.relation_semantics<gather>
// SCHEDULE: "shuttle.event_tensor"() <{{.*}}consumer_task = 1 : i64{{.*}}producer_task = 0 : i64{{.*}}relation_domain_shape = array<i64: 2>{{.*}}storage_capacity = 2 : i64
// SCHEDULE: "shuttle.event_tensor"() <{{.*}}consumer_task = 2 : i64{{.*}}producer_task = 1 : i64{{.*}}relation_domain_shape = array<i64: 1>{{.*}}storage_capacity = 1 : i64
// SCHEDULE-COUNT-2: "shuttle.event_realization"() <{{.*}}kind = #shuttle.event_realization_kind<imperative_wait_notify>
