// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Exact task dependencies lower through Event Tensors into an executable
// bundle. This validates the generic Contract/Fold capability rather than a
// workload-specific online state-machine certificate.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --mlir-print-op-generic | FileCheck %s

// CHECK: normalized_exp_contract_regions = [
// CHECK: "shuttle.schedule_task"()
// CHECK: "shuttle.event_tensor"()
// CHECK: "shuttle.device_entry"()
// CHECK: "shuttle.event_realization"()
// CHECK-SAME: kind = #shuttle.event_realization_kind<stream_order_erased>
// CHECK: "shuttle.invocation_abi"()
// CHECK: "shuttle.executable_bundle"()
// CHECK-NOT: online_

module {}
