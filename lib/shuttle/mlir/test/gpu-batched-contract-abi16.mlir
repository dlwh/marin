// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA-COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SOURCE-SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=FAST-SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SOURCE-SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=FAST-SCHEDULE-B
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-mutate-contract-materialization-shape --shuttle-verify-materialization-plan 2>&1 | FileCheck %s --check-prefix=MATERIALIZATION-MUTATION
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=grid' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-MUTATION
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=output' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-MUTATION
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=grid' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-MUTATION
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=output' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-MUTATION
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BACKWARD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-batched_contract-3x4x2x13x6-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMPOSED

// ALGEBRA-FORWARD-DAG: #[[FWD_LHS:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// ALGEBRA-FORWARD-DAG: #[[FWD_RHS:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// ALGEBRA-FORWARD-DAG: #[[FWD_OUT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// ALGEBRA-FORWARD: "shuttle.contract"
// ALGEBRA-FORWARD-SAME: indexing_maps = [#[[FWD_LHS]], #[[FWD_RHS]], #[[FWD_OUT]]]
// ALGEBRA-FORWARD-SAME: iterator_kinds = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// ALGEBRA-FORWARD-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-FORWARD-NOT: stablehlo.dot_general

// ALGEBRA-BACKWARD-DAG: #[[DRHS_DY:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// ALGEBRA-BACKWARD-DAG: #[[BACKWARD_WEIGHT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// ALGEBRA-BACKWARD-DAG: #[[BACKWARD_OUT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// ALGEBRA-BACKWARD-DAG: #[[DLHS_DY:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// ALGEBRA-BACKWARD: "shuttle.contract"
// ALGEBRA-BACKWARD-SAME: indexing_maps = [#[[DRHS_DY]], #[[BACKWARD_WEIGHT]], #[[BACKWARD_OUT]]]
// ALGEBRA-BACKWARD: "shuttle.contract"
// ALGEBRA-BACKWARD-SAME: indexing_maps = [#[[DLHS_DY]], #[[BACKWARD_WEIGHT]], #[[BACKWARD_OUT]]]
// ALGEBRA-BACKWARD-NOT: stablehlo.dot_general

// ALGEBRA-COMPOSED-DAG: #[[COMPOSED_FWD_LHS:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// ALGEBRA-COMPOSED-DAG: #[[COMPOSED_FWD_RHS:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// ALGEBRA-COMPOSED-DAG: #[[COMPOSED_OUT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// ALGEBRA-COMPOSED-DAG: #[[COMPOSED_DRHS_DY:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// ALGEBRA-COMPOSED-DAG: #[[COMPOSED_BACKWARD_WEIGHT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// ALGEBRA-COMPOSED: "shuttle.contract"
// ALGEBRA-COMPOSED-SAME: indexing_maps = [#[[COMPOSED_FWD_LHS]], #[[COMPOSED_FWD_RHS]], #[[COMPOSED_OUT]]]
// ALGEBRA-COMPOSED: "shuttle.contract"
// ALGEBRA-COMPOSED-SAME: indexing_maps = [#[[COMPOSED_DRHS_DY]], #[[COMPOSED_BACKWARD_WEIGHT]], #[[COMPOSED_OUT]]]
// ALGEBRA-COMPOSED: "shuttle.contract"
// ALGEBRA-COMPOSED-SAME: indexing_maps = [#[[COMPOSED_FWD_LHS]], #[[COMPOSED_BACKWARD_WEIGHT]], #[[COMPOSED_OUT]]]
// ALGEBRA-COMPOSED-NOT: stablehlo.dot_general

// PLAN-A: "shuttle.materialization_plan"() <{{.*}}schema_version = 2
// PLAN-A-COUNT-8: "shuttle.materialization_buffer"
// PLAN-A-COUNT-4: "shuttle.materialization_task"
// PLAN-A: domain_shape = array<i64: 2, 3, 5, 11, 7>{{.*}}kind = #shuttle.materialization_task_kind<contract>{{.*}}reduction_dimensions = array<i64: 4>
// PLAN-A-COUNT-1: "shuttle.materialization_task"

// PLAN-B: "shuttle.materialization_plan"() <{{.*}}schema_version = 2
// PLAN-B: domain_shape = array<i64: 3, 2, 4, 6, 13>{{.*}}kind = #shuttle.materialization_task_kind<contract>{{.*}}reduction_dimensions = array<i64: 4>

// SOURCE-SCHEDULE: domain_shape = array<i64: 2, 3, 5, 11, 7>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<serial_ascending_fma_rn_f32>{{.*}}serial_tiles = 7{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// FAST-SCHEDULE: domain_shape = array<i64: 2, 3, 5, 11, 7>{{.*}}grid_shape = array<i64: 42>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<leaf_order_free>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 8>{{.*}}workgroup_threads = 256
// SOURCE-SCHEDULE-B: domain_shape = array<i64: 3, 2, 4, 6, 13>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<serial_ascending_fma_rn_f32>{{.*}}serial_tiles = 13{{.*}}tile_shape = array<i64: 144>{{.*}}workgroup_threads = 160
// FAST-SCHEDULE-B: domain_shape = array<i64: 3, 2, 4, 6, 13>{{.*}}grid_shape = array<i64: 18>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<leaf_order_free>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 8>{{.*}}workgroup_threads = 256
// MATERIALIZATION-MUTATION: Contract buffers must bind four distinct parallel axes and one reduction axis with equal bf16 or f32 input types
// SCHEDULE-MUTATION: schedule geometry must equal the SIMT32 derivation

// FORWARD: %[[CALL:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FORWARD-SAME: api_version = 4 : i32
// FORWARD-SAME: bundle_schema_version = 2 : i64
// FORWARD-SAME: device_schema_version = 3 : i64
// FORWARD-SAME: invocation_abi_schema_version = 3 : i64
// FORWARD-SAME: transport_schema_version = 2 : i64
// FORWARD: return %[[CALL]]
// FORWARD-NOT: shuttle.
// FORWARD-NOT: stablehlo.dot_general

// BACKWARD: %[[CALL:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BACKWARD: return %[[CALL]]#1, %[[CALL]]#0
// BACKWARD-NOT: shuttle.
// BACKWARD-NOT: stablehlo.dot_general

// COMPOSED: %[[CALL:[0-9]+]]:3 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMPOSED: return %[[CALL]]#0, %[[CALL]]#2, %[[CALL]]#1
// COMPOSED-NOT: shuttle.
// COMPOSED-NOT: stablehlo.dot_general
