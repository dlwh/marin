// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=MATERIALIZATION
// RUN: shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=DESCRIPTOR
// RUN: shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-test-rename-symbols --shuttle-gpu-executable-bundle-fast-pipeline --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.renamed
// RUN: shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-gpu-executable-bundle-fast-pipeline --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.original
// RUN: diff %t.original %t.renamed
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=grid' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=tile' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=serial' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=threads' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=subgroup' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule '--shuttle-test-mutate-contract-schedule=kind=axis' --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=GEOMETRY
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-schedule-order --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: not shuttle-test-opt %S/Inputs/abi13-bf16-contract-map-fast-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle '--shuttle-test-corrupt-executable=kind=contract-body' --shuttle-verify-gpu-executable-bundle 2>&1 | FileCheck %s --check-prefix=PTX-MUTATION

// MATERIALIZATION-LABEL: module @anonymous_a
// MATERIALIZATION: "shuttle.materialization_plan"() <{{.*}}policy = #shuttle.policy<fast>{{.*}}schema_version = 2
// MATERIALIZATION: "shuttle.materialization_task"() <{{.*}}domain_shape = array<i64: 1, 1, 16>{{.*}}kind = #shuttle.materialization_task_kind<contract>{{.*}}reduction_dimensions = array<i64: 2>
// MATERIALIZATION: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64: 0>{{.*}}kind = #shuttle.materialization_task_kind<map>
// MATERIALIZATION-LABEL: module @anonymous_b
// MATERIALIZATION: domain_shape = array<i64: 1, 9, 32>
// MATERIALIZATION-LABEL: module @anonymous_c
// MATERIALIZATION: domain_shape = array<i64: 2, 3, 33>
// MATERIALIZATION-LABEL: module @anonymous_d
// MATERIALIZATION: domain_shape = array<i64: 17, 19, 65>

// SCHEDULE-LABEL: module @anonymous_a
// SCHEDULE: "shuttle.schedule_plan"() <{{.*}}policy = #shuttle.policy<fast>{{.*}}schema_version = 2
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 1, 16>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 2{{.*}}reduction_order = #shuttle.schedule_reduction_order<leaf_order_free>{{.*}}serial_tiles = 1{{.*}}subgroup_size = 32{{.*}}tile_shape = array<i64: 1>{{.*}}workgroup_threads = 32
// SCHEDULE-LABEL: module @anonymous_b
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 9, 32>{{.*}}grid_shape = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 8>{{.*}}workgroup_threads = 256
// SCHEDULE-LABEL: module @anonymous_c
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 2, 3, 33>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}serial_tiles = 2{{.*}}tile_shape = array<i64: 6>{{.*}}workgroup_threads = 192
// SCHEDULE-LABEL: module @anonymous_d
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 17, 19, 65>{{.*}}grid_shape = array<i64: 41>{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}serial_tiles = 3{{.*}}tile_shape = array<i64: 8>{{.*}}workgroup_threads = 256

// DESCRIPTOR-LABEL: module @anonymous_a
// DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// DESCRIPTOR-SAME: bundle_schema_version = 2 : i64
// DESCRIPTOR-SAME: device_schema_version = 3 : i64
// DESCRIPTOR-SAME: invocation_abi_schema_version = 3 : i64
// DESCRIPTOR-SAME: transport_schema_version = 2 : i64
// DESCRIPTOR-LABEL: module @anonymous_b
// DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// DESCRIPTOR-LABEL: module @anonymous_c
// DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// DESCRIPTOR-LABEL: module @anonymous_d
// DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// DESCRIPTOR-NOT: shuttle.region
// DESCRIPTOR-NOT: shuttle.contract

// GEOMETRY: schedule geometry must equal the SIMT32 derivation
// ORDER: Contract reduction order
// PTX-MUTATION: GPU executable bundle no longer matches Shuttle algebra and schedule
