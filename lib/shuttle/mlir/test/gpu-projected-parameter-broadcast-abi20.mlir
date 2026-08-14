// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic > %t.algebra-a
// RUN: FileCheck %s --input-file=%t.algebra-a --check-prefix=ALGEBRA
// RUN: FileCheck %s --input-file=%t.algebra-a --check-prefix=ALGEBRA-A
// RUN: FileCheck %s --input-file=%t.algebra-a --check-prefix=MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.algebra-a --check-prefix=FOLD-CENSUS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic > %t.algebra-b
// RUN: FileCheck %s --input-file=%t.algebra-b --check-prefix=ALGEBRA
// RUN: FileCheck %s --input-file=%t.algebra-b --check-prefix=ALGEBRA-B
// RUN: FileCheck %s --input-file=%t.algebra-b --check-prefix=MAP-CENSUS
// RUN: FileCheck %s --input-file=%t.algebra-b --check-prefix=FOLD-CENSUS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.plan-a-so
// RUN: FileCheck %s --input-file=%t.plan-a-so --check-prefix=PLAN-A-CENSUS
// RUN: FileCheck %s --input-file=%t.plan-a-so --check-prefix=PLAN-A-BUFFERS
// RUN: FileCheck %s --input-file=%t.plan-a-so --check-prefix=PLAN-A-TASKS
// RUN: FileCheck %s --input-file=%t.plan-a-so --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.plan-a-fast
// RUN: FileCheck %s --input-file=%t.plan-a-fast --check-prefix=PLAN-A-CENSUS
// RUN: FileCheck %s --input-file=%t.plan-a-fast --check-prefix=PLAN-A-BUFFERS
// RUN: FileCheck %s --input-file=%t.plan-a-fast --check-prefix=PLAN-A-TASKS
// RUN: FileCheck %s --input-file=%t.plan-a-fast --check-prefix=SCHEDULE-A
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.plan-b-so
// RUN: FileCheck %s --input-file=%t.plan-b-so --check-prefix=PLAN-B-CENSUS
// RUN: FileCheck %s --input-file=%t.plan-b-so --check-prefix=PLAN-B-BUFFERS
// RUN: FileCheck %s --input-file=%t.plan-b-so --check-prefix=PLAN-B-TASKS
// RUN: FileCheck %s --input-file=%t.plan-b-so --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic > %t.plan-b-fast
// RUN: FileCheck %s --input-file=%t.plan-b-fast --check-prefix=PLAN-B-CENSUS
// RUN: FileCheck %s --input-file=%t.plan-b-fast --check-prefix=PLAN-B-BUFFERS
// RUN: FileCheck %s --input-file=%t.plan-b-fast --check-prefix=PLAN-B-TASKS
// RUN: FileCheck %s --input-file=%t.plan-b-fast --check-prefix=SCHEDULE-B
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.a
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.a-renamed
// RUN: diff %t.a %t.a-renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.b
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-projected_parameter_broadcast-3x2x5x11x263-forward.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.b-renamed
// RUN: diff %t.b %t.b-renamed

// Symbols, fixture digests, workload labels, and exact extents are not dispatch
// inputs. Both ranks must traverse the same structural lowering.

// MAP-CENSUS-COUNT-18: "shuttle.map"
// MAP-CENSUS-NOT: "shuttle.map"
// FOLD-CENSUS-COUNT-1: "shuttle.fold"
// FOLD-CENSUS-NOT: "shuttle.fold"
// ALGEBRA: value = 0.000000e+00 : f32
// ALGEBRA: "arith.addf"
// ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// ALGEBRA: "math.rsqrt"
// ALGEBRA-SAME: fastmath = #arith.fastmath<none>
// ALGEBRA-NOT: stablehlo.broadcast_in_dim
// ALGEBRA-NOT: stablehlo.reduce

// ALGEBRA-A-DAG: #[[IDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// ALGEBRA-A-DAG: #[[FOLDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// ALGEBRA-A-DAG: #[[FOLDEXPANDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 19)>
// ALGEBRA-A-DAG: #[[WEIGHTA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d2)>
// ALGEBRA-A-DAG: #[[EXPANDA:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0 floordiv 2, d1 floordiv 7, d2)>
// ALGEBRA-A-DAG: indexing_maps = [#[[FOLDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-A-DAG: indexing_maps = [#[[FOLDEXPANDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-A-DAG: indexing_maps = [#[[WEIGHTA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-A-DAG: indexing_maps = [#[[EXPANDA]], #[[IDA]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>

// ALGEBRA-B-DAG: #[[IDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// ALGEBRA-B-DAG: #[[FOLDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// ALGEBRA-B-DAG: #[[FOLDEXPANDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 floordiv 263)>
// ALGEBRA-B-DAG: #[[WEIGHTB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// ALGEBRA-B-DAG: #[[EXPANDB:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 floordiv 3, d1 floordiv 2, d2 floordiv 5, d3 floordiv 11, d4)>
// ALGEBRA-B-DAG: indexing_maps = [#[[FOLDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-B-DAG: indexing_maps = [#[[FOLDEXPANDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-B-DAG: indexing_maps = [#[[WEIGHTB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA-B-DAG: indexing_maps = [#[[EXPANDB]], #[[IDB]]]{{.*}}semantics = #shuttle.map_semantics<broadcast_in_dim>

// PLAN-A-CENSUS: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-A-CENSUS-COUNT-21: "shuttle.materialization_buffer"
// PLAN-A-CENSUS-COUNT-19: "shuttle.materialization_task"
// PLAN-A-BUFFERS: tensor_type = tensor<19xbf16>
// PLAN-A-BUFFERS: tensor_type = tensor<19xf32>
// PLAN-A-BUFFERS: tensor_type = tensor<1x1x19xf32>
// PLAN-A-TASKS: domain_shape = array<i64: 19>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-A-TASKS: domain_shape = array<i64: 1, 1, 19>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-A-TASKS: domain_shape = array<i64: 2, 7, 19>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B-CENSUS: "shuttle.materialization_plan"() <{{.*}}schema_version = 1
// PLAN-B-CENSUS-COUNT-21: "shuttle.materialization_buffer"
// PLAN-B-CENSUS-COUNT-19: "shuttle.materialization_task"
// PLAN-B-BUFFERS: tensor_type = tensor<263xbf16>
// PLAN-B-BUFFERS: tensor_type = tensor<263xf32>
// PLAN-B-BUFFERS: tensor_type = tensor<1x1x1x1x263xf32>
// PLAN-B-TASKS: domain_shape = array<i64: 263>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B-TASKS: domain_shape = array<i64: 1, 1, 1, 1, 263>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN-B-TASKS: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}kind = #shuttle.materialization_task_kind<map>

// SCHEDULE-A: domain_shape = array<i64: 19>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 19>{{.*}}workgroup_threads = 32
// SCHEDULE-A: domain_shape = array<i64: 1, 1, 19>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 19>{{.*}}workgroup_threads = 32
// SCHEDULE-A: domain_shape = array<i64: 2, 7, 19>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 263>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 1, 1, 1, 1, 263>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256
// SCHEDULE-B: domain_shape = array<i64: 3, 2, 5, 11, 263>{{.*}}grid_shape = array<i64: 340>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}tile_shape = array<i64: 256>{{.*}}workgroup_threads = 256

// CALL: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
