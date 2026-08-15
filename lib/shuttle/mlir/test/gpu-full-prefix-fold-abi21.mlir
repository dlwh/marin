// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic > %t.a-bwd-algebra
// RUN: FileCheck %s --input-file=%t.a-bwd-algebra --check-prefix=ALGEBRA-A
// RUN: FileCheck %s --input-file=%t.a-bwd-algebra --check-prefix=BWD-MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-algebra --check-prefix=FOLD-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-algebra --check-prefix=ZERO-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-algebra --check-prefix=ADD-CENSUS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic > %t.a-comp-algebra
// RUN: FileCheck %s --input-file=%t.a-comp-algebra --check-prefix=ALGEBRA-A
// RUN: FileCheck %s --input-file=%t.a-comp-algebra --check-prefix=COMP-MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-algebra --check-prefix=FOLD-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-algebra --check-prefix=ZERO-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-algebra --check-prefix=ADD-CENSUS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic > %t.b-bwd-algebra
// RUN: FileCheck %s --input-file=%t.b-bwd-algebra --check-prefix=ALGEBRA-B
// RUN: FileCheck %s --input-file=%t.b-bwd-algebra --check-prefix=BWD-MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-algebra --check-prefix=FOLD-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-algebra --check-prefix=ZERO-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-algebra --check-prefix=ADD-CENSUS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic > %t.b-comp-algebra
// RUN: FileCheck %s --input-file=%t.b-comp-algebra --check-prefix=ALGEBRA-B
// RUN: FileCheck %s --input-file=%t.b-comp-algebra --check-prefix=COMP-MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-algebra --check-prefix=FOLD-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-algebra --check-prefix=ZERO-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-algebra --check-prefix=ADD-CENSUS

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.a-bwd-so
// RUN: FileCheck %s --input-file=%t.a-bwd-so --check-prefix=PLAN-BWD
// RUN: FileCheck %s --input-file=%t.a-bwd-so --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-so --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-so --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.a-bwd-fast
// RUN: FileCheck %s --input-file=%t.a-bwd-fast --check-prefix=PLAN-BWD
// RUN: FileCheck %s --input-file=%t.a-bwd-fast --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-fast --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.a-bwd-fast --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.a-comp-so
// RUN: FileCheck %s --input-file=%t.a-comp-so --check-prefix=PLAN-COMP
// RUN: FileCheck %s --input-file=%t.a-comp-so --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-so --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-so --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.a-comp-fast
// RUN: FileCheck %s --input-file=%t.a-comp-fast --check-prefix=PLAN-COMP
// RUN: FileCheck %s --input-file=%t.a-comp-fast --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-fast --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.a-comp-fast --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.b-bwd-so
// RUN: FileCheck %s --input-file=%t.b-bwd-so --check-prefix=PLAN-BWD
// RUN: FileCheck %s --input-file=%t.b-bwd-so --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-so --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-so --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.b-bwd-fast
// RUN: FileCheck %s --input-file=%t.b-bwd-fast --check-prefix=PLAN-BWD
// RUN: FileCheck %s --input-file=%t.b-bwd-fast --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-fast --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.b-bwd-fast --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.b-comp-so
// RUN: FileCheck %s --input-file=%t.b-comp-so --check-prefix=PLAN-COMP
// RUN: FileCheck %s --input-file=%t.b-comp-so --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-so --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-so --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.b-comp-fast
// RUN: FileCheck %s --input-file=%t.b-comp-fast --check-prefix=PLAN-COMP
// RUN: FileCheck %s --input-file=%t.b-comp-fast --check-prefix=COLUMN-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-fast --check-prefix=ROW-CENSUS
// RUN: FileCheck %s --input-file=%t.b-comp-fast --check-prefix=SCHEDULE-B

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMP
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMP
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMP
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMP

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.bwd
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.bwd-renamed
// RUN: diff %t.bwd %t.bwd-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.comp-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.comp
// RUN: diff %t.comp %t.comp-renamed

// Anonymous structural fixtures: names, exact extents, hashes, and workload
// labels are never compiler dispatch inputs.

// BWD-MAP-CENSUS-COUNT-43: "shuttle.map"
// BWD-MAP-CENSUS-NOT: "shuttle.map"
// COMP-MAP-CENSUS-COUNT-46: "shuttle.map"
// COMP-MAP-CENSUS-NOT: "shuttle.map"
// FOLD-CENSUS-COUNT-5: "shuttle.fold"
// FOLD-CENSUS-NOT: "shuttle.fold"
// ZERO-CENSUS-COUNT-5: value = 0.000000e+00 : f32
// ADD-CENSUS-COUNT-5: "arith.addf"{{.*}}fastmath = #arith.fastmath<none>

// ALGEBRA-A: "shuttle.fold"{{.*}}reduction_dimensions = array<i64: 0, 1>
// ALGEBRA-A: (tensor<2x7x19xf32>, tensor<f32>) -> tensor<19xf32>
// ALGEBRA-A: "shuttle.fold"{{.*}}reduction_dimensions = array<i64: 0, 1>
// ALGEBRA-A: (tensor<1x1x19xf32>, tensor<f32>) -> tensor<19xf32>
// ALGEBRA-A: "shuttle.scalar_convert"
// ALGEBRA-A-SAME: semantics = #shuttle.scalar_convert_semantics<round_nearest_even>
// ALGEBRA-A-SAME: (f32) -> bf16
// ALGEBRA-B: "shuttle.fold"{{.*}}reduction_dimensions = array<i64: 0, 1, 2, 3>
// ALGEBRA-B: (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<263xf32>
// ALGEBRA-B: "shuttle.fold"{{.*}}reduction_dimensions = array<i64: 0, 1, 2, 3>
// ALGEBRA-B: (tensor<1x1x1x1x263xf32>, tensor<f32>) -> tensor<263xf32>
// ALGEBRA-B: "shuttle.scalar_convert"
// ALGEBRA-B-SAME: semantics = #shuttle.scalar_convert_semantics<round_nearest_even>
// ALGEBRA-B-SAME: (f32) -> bf16

// PLAN-BWD: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-BWD-COUNT-51: "shuttle.materialization_buffer"
// PLAN-BWD-COUNT-48: "shuttle.materialization_task"
// PLAN-BWD: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// PLAN-COMP: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-COMP-COUNT-54: "shuttle.materialization_buffer"
// PLAN-COMP-COUNT-51: "shuttle.materialization_task"
// PLAN-COMP: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// COLUMN-CENSUS-COUNT-2: kind = #shuttle.schedule_task_kind<column_fold>
// COLUMN-CENSUS-NOT: kind = #shuttle.schedule_task_kind<column_fold>
// ROW-CENSUS-COUNT-3: kind = #shuttle.schedule_task_kind<row_fold>
// ROW-CENSUS-NOT: kind = #shuttle.schedule_task_kind<row_fold>
// SCHEDULE-A: domain_shape = array<i64: 2, 7, 19>{{.*}}grid_shape = array<i64: 19>{{.*}}kind = #shuttle.schedule_task_kind<column_fold>{{.*}}reduction_axis = 0{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 128{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 14, 1, 1>{{.*}}workgroup_threads = 32
// SCHEDULE-A: domain_shape = array<i64: 1, 1, 19>{{.*}}grid_shape = array<i64: 19>{{.*}}kind = #shuttle.schedule_task_kind<column_fold>{{.*}}scratch_bytes = 128{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 1, 1, 1>{{.*}}workgroup_threads = 32
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}grid_shape = array<i64: 263>{{.*}}kind = #shuttle.schedule_task_kind<column_fold>{{.*}}reduction_axis = 0{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 1024{{.*}}serial_tiles = 2{{.*}}tile_shape = array<i64: 256, 1, 1, 1, 1>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 1, 1, 1, 1, 263>{{.*}}grid_shape = array<i64: 263>{{.*}}kind = #shuttle.schedule_task_kind<column_fold>{{.*}}scratch_bytes = 128{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 1, 1, 1, 1, 1>{{.*}}workgroup_threads = 32

// BWD: %[[BWD:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BWD-SAME: bundle_schema_version = 2 : i64
// BWD-SAME: device_schema_version = 3 : i64
// BWD-SAME: invocation_abi_schema_version = 3 : i64
// BWD-SAME: transport_schema_version = 2 : i64
// BWD: return %[[BWD]]#0, %[[BWD]]#1
// BWD-NOT: shuttle.
// COMP: %[[COMP:[0-9]+]]:3 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMP-SAME: bundle_schema_version = 2 : i64
// COMP-SAME: device_schema_version = 3 : i64
// COMP-SAME: invocation_abi_schema_version = 3 : i64
// COMP-SAME: transport_schema_version = 2 : i64
// COMP: return %[[COMP]]#0, %[[COMP]]#1, %[[COMP]]#2
// COMP-NOT: shuttle.
