// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=MATERIALIZATION
// RUN: shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=DESCRIPTOR
// RUN: shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.original
// RUN: shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.renamed
// RUN: diff %t.original %t.renamed
// RUN: not shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-gpu-executable-bundle-fast-pipeline 2>&1 | FileCheck %s --check-prefix=FAST
// RUN: not shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-delete-materialization-task --shuttle-verify-materialization-plan 2>&1 | FileCheck %s --check-prefix=MATERIALIZATION-MUTATION
// RUN: not shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-schedule-dependency --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-DEPENDENCY
// RUN: not shuttle-test-opt %S/Inputs/abi12-bf16-contract-map-sweep.mlir --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-schedule-order --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=SCHEDULE-ORDER

// MATERIALIZATION-LABEL: module @anonymous_a
// MATERIALIZATION: "shuttle.materialization_plan"() <{{.*}}policy = #shuttle.policy<source_ordered>{{.*}}schema_version = 2
// MATERIALIZATION: "shuttle.materialization_task"() <{{.*}}domain_shape = array<i64: 2, 5, 3>{{.*}}kind = #shuttle.materialization_task_kind<contract>{{.*}}reduction_dimensions = array<i64: 2>
// MATERIALIZATION: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64: 0>{{.*}}kind = #shuttle.materialization_task_kind<map>
// MATERIALIZATION-LABEL: module @anonymous_b
// MATERIALIZATION: domain_shape = array<i64: 3, 2, 5>
// MATERIALIZATION-LABEL: module @anonymous_c
// MATERIALIZATION: domain_shape = array<i64: 7, 3, 13>
// MATERIALIZATION-LABEL: module @anonymous_d
// MATERIALIZATION: domain_shape = array<i64: 11, 17, 3>

// SCHEDULE-LABEL: module @anonymous_a
// SCHEDULE: "shuttle.schedule_plan"() <{{.*}}policy = #shuttle.policy<source_ordered>{{.*}}schema_version = 2
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}kind = #shuttle.schedule_task_kind<contract>{{.*}}reduction_axis = 2{{.*}}reduction_order = #shuttle.schedule_reduction_order<serial_ascending_fma_rn_f32>
// SCHEDULE: "shuttle.schedule_task"() <{{.*}}dependencies = array<i64: 0>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>
// SCHEDULE-LABEL: module @anonymous_b
// SCHEDULE: #shuttle.schedule_task_kind<contract>
// SCHEDULE-LABEL: module @anonymous_c
// SCHEDULE: #shuttle.schedule_task_kind<contract>
// SCHEDULE-LABEL: module @anonymous_d
// SCHEDULE: #shuttle.schedule_task_kind<contract>

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

// FAST: ABI12 GPU Contract requires source_ordered numerics
// MATERIALIZATION-MUTATION: has an invalid task lifetime interval
// SCHEDULE-DEPENDENCY: schedule dependencies must equal the materialization task
// SCHEDULE-ORDER: Contract reduction order must equal serial ascending fma.rn.f32 semantics
