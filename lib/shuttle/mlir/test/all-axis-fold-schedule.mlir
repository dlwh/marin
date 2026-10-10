// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// A Fold with no surviving owner axes is linearized independently of its
// source rank. Its complete reduction-axis set remains explicit in the
// schedule identity.

// RUN: shuttle-test-opt %S/Inputs/f32-reduce-add-all-axis-rank1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=RANK1
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-scalar-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --mlir-print-op-generic | FileCheck %s --check-prefix=RANK5
// RUN: not shuttle-test-opt %S/Inputs/f32-reduce-add-all-axis-rank1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-fold-schedule=kind=all-axis-dimensions' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=DIMENSIONS-ERROR
// RUN: not shuttle-test-opt %S/Inputs/f32-reduce-add-all-axis-rank1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-fold-schedule=kind=all-axis-axis' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=AXIS-ERROR
// RUN: not shuttle-test-opt %S/Inputs/f32-reduce-add-all-axis-rank1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-fold-schedule=kind=all-axis-merge-grid' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY-ERROR
// RUN: not shuttle-test-opt %S/Inputs/f32-reduce-add-all-axis-rank1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-fold-schedule=kind=all-axis-phase-order' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=PHASE-ERROR

// RANK1: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 513>
// RANK1-SAME: grid_shape = array<i64: 1>
// RANK1-SAME: kind = #shuttle.schedule_task_kind<all_axis_fold>
// RANK1-SAME: reduction_dimensions = array<i64: 0>
// RANK1-SAME: reduction_merge_grid_shape = array<i64: 1>
// RANK1-SAME: reduction_merge_serial_tiles = 1 : i64
// RANK1-SAME: reduction_merge_tile_shape = array<i64: 256>
// RANK1-SAME: reduction_partial_buffer_shape = array<i64: 1>
// RANK1-SAME: serial_tiles = 3 : i64
// RANK1-SAME: tile_shape = array<i64: 4096>
// RANK1: "shuttle.event_tensor"() <{{.*}}consumer_phase = "merge"{{.*}}consumer_task = 0 : i64{{.*}}producer_phase = "partial"{{.*}}producer_task = 0 : i64{{.*}}storage_capacity = 1 : i64

// RANK5: "shuttle.schedule_task"() <{dependencies = array<i64: 0>, domain_shape = array<i64: 1, 8, 4, 128, 128>
// RANK5-SAME: grid_shape = array<i64: 128>
// RANK5-SAME: kind = #shuttle.schedule_task_kind<all_axis_fold>
// RANK5-SAME: reduction_dimensions = array<i64: 0, 1, 2, 3, 4>
// RANK5-SAME: reduction_merge_grid_shape = array<i64: 1>
// RANK5-SAME: reduction_merge_serial_tiles = 1 : i64
// RANK5-SAME: reduction_merge_tile_shape = array<i64: 256>
// RANK5-SAME: reduction_partial_buffer_shape = array<i64: 128>
// RANK5-SAME: serial_tiles = 16 : i64
// RANK5-SAME: tile_shape = array<i64: 4096>
// RANK5: "shuttle.event_tensor"() <{{.*}}consumer_task = 1 : i64{{.*}}producer_task = 0 : i64
// RANK5: "shuttle.event_tensor"() <{{.*}}consumer_phase = "merge"{{.*}}consumer_task = 1 : i64{{.*}}producer_phase = "partial"{{.*}}producer_task = 1 : i64{{.*}}storage_capacity = 128 : i64
// RANK5: "shuttle.imperative_event_wait"() <{{.*}}consumer_task = 1 : i64{{.*}}producer_task = 0 : i64
// RANK5: "shuttle.imperative_event_wait"() <{{.*}}consumer_phase = "merge"{{.*}}consumer_task = 1 : i64{{.*}}producer_phase = "partial"{{.*}}producer_task = 1 : i64{{.*}}storage_capacity = 128 : i64

// DIMENSIONS-ERROR: all-axis Fold must reduce every domain dimension exactly once
// AXIS-ERROR: all-axis Fold must reduce every domain dimension exactly once
// GEOMETRY-ERROR: schedule geometry must equal the SIMT32 derivation
// PHASE-ERROR: must uniquely eventize one ordered schedule-task dependency
