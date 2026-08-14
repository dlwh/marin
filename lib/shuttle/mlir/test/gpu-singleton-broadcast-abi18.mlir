// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-A,FWD-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-A,BWD-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-A,COMPOSED-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-B,FWD-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-B,BWD-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s --check-prefixes=ALGEBRA-B,COMPOSED-B

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefixes=PLAN-A,SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefixes=PLAN-A,SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefixes=PLAN-B,SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefixes=PLAN-B,SCHEDULE-B

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMPOSED-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=BWD-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=COMPOSED-CALL

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward-so
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward-so-renamed
// RUN: diff %t.forward-so %t.forward-so-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward-fast
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.forward-fast-renamed
// RUN: diff %t.forward-fast %t.forward-fast-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed-so
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed-so-renamed
// RUN: diff %t.composed-so %t.composed-so-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed-fast
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.composed-fast-renamed
// RUN: diff %t.composed-fast %t.composed-fast-renamed

// The cases are anonymous and structurally derived. Symbols, workload labels,
// fixture digests, and exact extents are not admissible dispatch inputs.

// ALGEBRA-A-DAG: #[[EXPANDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 19)>
// ALGEBRA-A-DAG: #[[IDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// ALGEBRA-B-DAG: #[[EXPANDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 floordiv 263)>
// ALGEBRA-B-DAG: #[[IDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// FWD-A-COUNT-1: indexing_maps = [#[[EXPANDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// BWD-A-COUNT-1: indexing_maps = [#[[EXPANDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// COMPOSED-A-COUNT-2: indexing_maps = [#[[EXPANDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// FWD-B-COUNT-1: indexing_maps = [#[[EXPANDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// BWD-B-COUNT-1: indexing_maps = [#[[EXPANDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// COMPOSED-B-COUNT-2: indexing_maps = [#[[EXPANDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// FWD-A-NOT: stablehlo.broadcast_in_dim
// FWD-B-NOT: stablehlo.broadcast_in_dim

// PLAN-A: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-A: domain_shape = array<i64: 2, 7, 19>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-A: domain_shape = array<i64: 2, 7, 19>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}reduction_dimensions = array<i64: 2>
// PLAN-A: domain_shape = array<i64: 2, 7, 1>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-A-COUNT-3: domain_shape = array<i64: 2, 7, 19>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}reduction_dimensions = array<i64: 4>
// PLAN-B: domain_shape = array<i64: 3, 2, 5, 11, 1>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B-COUNT-3: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}kind = #shuttle.materialization_task_kind<map>

// SCHEDULE-A: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// SCHEDULE-A: domain_shape = array<i64: 2, 7, 19>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-A: domain_shape = array<i64: 2, 7, 19>{{.*}}grid_shape = array<i64: 14>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 2{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 128{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 1, 1, 19>{{.*}}workgroup_threads = 32
// SCHEDULE-A: domain_shape = array<i64: 2, 7, 1>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 14>{{.*}}workgroup_threads = 32
// SCHEDULE-A-COUNT-3: domain_shape = array<i64: 2, 7, 19>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: "shuttle.schedule_plan"() <{{.*}}schema_version = 1
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}grid_shape = array<i64: 340>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}grid_shape = array<i64: 330>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 4{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 1024{{.*}}serial_tiles = 2{{.*}}tile_shape = array<i64: 1, 1, 1, 1, 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 5, 11, 1>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B-COUNT-3: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}grid_shape = array<i64: 340>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256

// FWD-CALL: %[[CALL:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FWD-CALL-SAME: bundle_schema_version = 2 : i64
// FWD-CALL-SAME: device_schema_version = 3 : i64
// FWD-CALL-SAME: invocation_abi_schema_version = 3 : i64
// FWD-CALL-SAME: transport_schema_version = 2 : i64
// FWD-CALL: return %[[CALL]]
// BWD-CALL: %[[CALL:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BWD-CALL: return %[[CALL]]
// COMPOSED-CALL: %[[CALL:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMPOSED-CALL: return %[[CALL]]#0, %[[CALL]]#1
// FWD-CALL-NOT: shuttle.
// BWD-CALL-NOT: shuttle.
// COMPOSED-CALL-NOT: shuttle.
