// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-A,FORWARD-ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-B,FORWARD-ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-A,BACKWARD-ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-B,COMPOSED-ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward-renamed
// RUN: diff %t.forward %t.forward-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed-renamed
// RUN: diff %t.composed %t.composed-renamed

// This is an anonymous, shape-generic boundary. Function names, fixture hashes,
// workload labels, and exact extents are not admissible dispatch inputs.

// ALGEBRA-A-DAG: #[[IDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// ALGEBRA-A-DAG: #[[PROJA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// ALGEBRA-B-DAG: #[[IDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// ALGEBRA-B-DAG: #[[PROJB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// FORWARD-ALGEBRA-COUNT-1: value = 0.000000e+00 : f32
// FORWARD-ALGEBRA-COUNT-1: "shuttle.fold"
// FORWARD-ALGEBRA-SAME: reduction_dimensions = array<i64: {{2|4}}>
// FORWARD-ALGEBRA: "arith.addf"
// FORWARD-ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// FORWARD-ALGEBRA: semantics = #shuttle.map_semantics<broadcast_in_dim>
// FORWARD-ALGEBRA-NOT: stablehlo.reduce
// BACKWARD-ALGEBRA: value = 0.000000e+00 : f32
// BACKWARD-ALGEBRA: "shuttle.fold"
// BACKWARD-ALGEBRA: "arith.addf"
// BACKWARD-ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// BACKWARD-ALGEBRA: "shuttle.map"
// BACKWARD-ALGEBRA-SAME: indexing_maps = [#[[PROJA]], #[[IDA]]]
// BACKWARD-ALGEBRA-SAME: semantics = #shuttle.map_semantics<broadcast_in_dim>
// BACKWARD-ALGEBRA: value = 0.000000e+00 : f32
// BACKWARD-ALGEBRA: "shuttle.fold"
// BACKWARD-ALGEBRA: "arith.addf"
// BACKWARD-ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// BACKWARD-ALGEBRA: "shuttle.map"
// BACKWARD-ALGEBRA-SAME: indexing_maps = [#[[PROJA]], #[[IDA]]]
// BACKWARD-ALGEBRA-SAME: semantics = #shuttle.map_semantics<broadcast_in_dim>
// COMPOSED-ALGEBRA: value = 0.000000e+00 : f32
// COMPOSED-ALGEBRA: "shuttle.fold"
// COMPOSED-ALGEBRA: "arith.addf"
// COMPOSED-ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// COMPOSED-ALGEBRA: "shuttle.map"
// COMPOSED-ALGEBRA-SAME: indexing_maps = [#[[PROJB]], #[[IDB]]]
// COMPOSED-ALGEBRA-SAME: semantics = #shuttle.map_semantics<broadcast_in_dim>
// COMPOSED-ALGEBRA: value = 0.000000e+00 : f32
// COMPOSED-ALGEBRA: "shuttle.fold"
// COMPOSED-ALGEBRA: "arith.addf"
// COMPOSED-ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// COMPOSED-ALGEBRA: "shuttle.map"
// COMPOSED-ALGEBRA-SAME: indexing_maps = [#[[PROJB]], #[[IDB]]]
// COMPOSED-ALGEBRA-SAME: semantics = #shuttle.map_semantics<broadcast_in_dim>

// PLAN-A: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-A: tensor_type = tensor<2x5x17xf32>
// PLAN-A: tensor_type = tensor<2x5xf32>
// PLAN-A: domain_shape = array<i64: 2, 5, 17>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}reduction_dimensions = array<i64: 2>
// PLAN-B: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-B: tensor_type = tensor<3x2x4x7x269xf32>
// PLAN-B: tensor_type = tensor<3x2x4x7xf32>
// PLAN-B: domain_shape = array<i64: 3, 2, 4, 7, 269>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}reduction_dimensions = array<i64: 4>

// SCHEDULE-A: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// SCHEDULE-A: domain_shape = array<i64: 2, 5, 17>{{.*}}grid_shape = array<i64: 10>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 2{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 128{{.*}}serial_tiles = 1{{.*}}subgroup_size = 32{{.*}}tile_shape = array<i64: 1, 1, 17>{{.*}}workgroup_threads = 32
// SCHEDULE-B: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 4, 7, 269>{{.*}}grid_shape = array<i64: 168>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 1024{{.*}}serial_tiles = 2{{.*}}subgroup_size = 32{{.*}}tile_shape = array<i64: 1, 1, 1, 1, 256>{{.*}}workgroup_threads = 256

// FORWARD: %[[CALL:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FORWARD-SAME: bundle_schema_version = 2 : i64
// FORWARD-SAME: device_schema_version = 3 : i64
// FORWARD-SAME: invocation_abi_schema_version = 3 : i64
// FORWARD-SAME: transport_schema_version = 2 : i64
// FORWARD: return %[[CALL]]
// BACKWARD: %[[CALL:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BACKWARD: return %[[CALL]]
// COMPOSED: %[[CALL:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMPOSED: return %[[CALL]]#0, %[[CALL]]#1
// FORWARD-NOT: shuttle.
// BACKWARD-NOT: shuttle.
// COMPOSED-NOT: shuttle.
