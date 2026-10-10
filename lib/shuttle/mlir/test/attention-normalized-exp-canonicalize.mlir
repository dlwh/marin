// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// FAST rewrites the local max/exp/sum/div slice into an explicit associative
// two-result Fold.  The surrounding producer and consumer Contracts remain
// ordinary algebra operations; this test deliberately does not name a
// workload or constrain a physical schedule.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA

// The same generic scalar DAG and multi-result Fold have an authoritative
// StableHLO representation.  This is a FAST rewrite, so it need not reproduce
// the source operation topology.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-stablehlo-fast-pipeline --mlir-print-op-generic | FileCheck %s --check-prefix=ROUNDTRIP

// Generic physical planning partitions the live SSA cut, not a recovered
// attention topology.  Kernel 0 returns the full producer value and both Fold
// states; kernel 1 consumes those values and the weighted-Contract operand.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN

// Identity, tanh composition, and a third affine score program have the same
// Contract/Fold ownership cut and SIMT32 geometry. Their semantic and device
// identities remain distinct because the score Map AST is compiled into PTX.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --mlir-print-op-generic | FileCheck %s --check-prefixes=PHYSICAL,IDENTITY
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-softcap-score-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --mlir-print-op-generic | FileCheck %s --check-prefixes=PHYSICAL,SOFTCAP
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-affine-score-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --mlir-print-op-generic | FileCheck %s --check-prefixes=PHYSICAL,AFFINE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-report-event-cardinality -o /dev/null | FileCheck %s --check-prefix=PHYSICAL-CARDINALITY
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.identity
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-softcap-score-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.softcap
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-affine-score-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.affine
// RUN: not diff %t.identity %t.softcap
// RUN: not diff %t.identity %t.affine
// RUN: not diff %t.softcap %t.affine

// The representative S=2048 plan uses one event coordinate per consumer
// block. It does not materialize the Cartesian product of producer rows and
// consumer blocks; the subsequent FA4 physical lowering erases these logical
// coordinates altogether.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-online-benchmark-s2048-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=S2048-FORWARD

// A non-trailing logical Fold axis and a post-Contract transpose use the same
// generic composition.  The schedule moves only the physical traversal axis;
// algebra indexing maps continue to define every public and temporary layout.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS1
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-event-relation --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=EVENT-RELATION-ERROR
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-event-bounds --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=EVENT-BOUNDS-ERROR

// ALGEBRA: output_dependency_coverage = complete
// ALGEBRA: %[[STATE:[0-9]+]]:2 = "shuttle.fold"
// ALGEBRA-SAME: accumulator_types = [f32, f32]
// ALGEBRA-SAME: order_free = true
// ALGEBRA-SAME: reduction_dimensions = array<i64: 4>
// ALGEBRA: "arith.maximumf"
// ALGEBRA: "math.exp"
// ALGEBRA: "arith.mulf"
// ALGEBRA: "math.exp"
// ALGEBRA: "arith.mulf"
// ALGEBRA: "arith.addf"
// ALGEBRA: "shuttle.yield"
// ALGEBRA-SAME: shuttle.operation_ref_groups
// ALGEBRA: "shuttle.map"(%[[STATE]]#0)
// ALGEBRA: "shuttle.map"(%[[STATE]]#1)
// ALGEBRA: "shuttle.map"
// ALGEBRA: "arith.subf"
// ALGEBRA: "math.exp"
// ALGEBRA: "arith.divf"
// ALGEBRA-NOT: "shuttle.fold"

// AXIS1: #[[AXIS1_CONSUMER:map[0-9]*]] = affine_map<(d0, d1) -> (d0)>
// AXIS1: #[[AXIS1_PRODUCER:map[0-9]*]] = affine_map<(d0, d1) -> ((d0 * 8) floordiv 11 + d1)>
// AXIS1: #[[AXIS1_SLOT:map[0-9]*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// AXIS1: #[[AXIS1_RELATION:set[0-9]*]] = affine_set<(d0, d1) : (((d0 * 8) floordiv 11 + d1) * 11 - d0 * 8 + 10 >= 0, d0 * 8 - ((d0 * 8) floordiv 11 + d1) * 11 + 7 >= 0, -((d0 * 8) floordiv 11 + d1) + 227 >= 0)>
// AXIS1: composition_kind = "map_contract"
// AXIS1-SAME: source = #shuttle.source_ref<0, 0, 38, 0>
// AXIS1: "shuttle.schedule_task"()
// AXIS1-SAME: domain_shape = array<i64: 19, 23, 2, 3, 2>
// AXIS1-SAME: grid_shape = array<i64: 228>
// AXIS1-SAME: kind = #shuttle.schedule_task_kind<row_fold>
// AXIS1-SAME: reduction_axis = 1 : i64
// AXIS1-SAME: reduction_state_bytes_per_thread = 8 : i64
// AXIS1-SAME: scratch_bytes = 256 : i64
// AXIS1-SAME: tile_shape = array<i64: 1, 23, 1, 1, 1>
// AXIS1: "shuttle.schedule_task"()
// AXIS1-SAME: dependencies = array<i64: 0>
// AXIS1-SAME: kind = #shuttle.schedule_task_kind<contract>
// AXIS1-SAME: output_buffers = array<i64: 6>
// AXIS1: "shuttle.event_tensor"()
// AXIS1-SAME: acquire = true
// AXIS1-SAME: consumer_indices = #[[AXIS1_CONSUMER]]
// AXIS1-SAME: consumer_task = 1 : i64
// AXIS1-SAME: generation_index = #[[GENERATION_INDEX:map[0-9]*]]
// AXIS1-SAME: generations = 1 : i64
// AXIS1-SAME: notify_count = 1 : i64
// AXIS1-SAME: producer_indices = #[[AXIS1_PRODUCER]]
// AXIS1-SAME: producer_task = 0 : i64
// AXIS1-SAME: relation_domain_shape = array<i64: 314, 2>
// AXIS1-SAME: relation_predicate = #[[AXIS1_RELATION]]
// AXIS1-SAME: release = true
// AXIS1-SAME: scope = #shuttle.event_scope<device>
// AXIS1-SAME: storage_capacity = 628 : i64
// AXIS1-SAME: storage_slot = #[[AXIS1_SLOT]]
// AXIS1-COUNT-2: "shuttle.device_entry"()
// AXIS1: "shuttle.event_realization"()
// AXIS1-SAME: covering_launch_order = array<i64: 0, 0, 1, 0>
// AXIS1-SAME: event = 0 : i64
// AXIS1-SAME: kind = #shuttle.event_realization_kind<stream_order_erased>

// EVENT-RELATION-ERROR: must equal the structural tiled task dependence
// EVENT-BOUNDS-ERROR: producer index map escapes the producer grid

// S2048-FORWARD: "shuttle.event_tensor"()
// S2048-FORWARD-SAME: relation_domain_shape = array<i64: 1048576, 1>
// S2048-FORWARD-SAME: relation_predicate = #[[S2048_RELATION:set[0-9]*]]
// S2048-FORWARD-SAME: storage_capacity = 1048576 : i64

// ROUNDTRIP: %[[REDUCE:[0-9]+]]:2 = "stablehlo.reduce"
// ROUNDTRIP: "stablehlo.maximum"
// ROUNDTRIP: "stablehlo.exponential"
// ROUNDTRIP: "stablehlo.add"
// ROUNDTRIP: "stablehlo.map"
// ROUNDTRIP: "stablehlo.subtract"
// ROUNDTRIP: "stablehlo.exponential"
// ROUNDTRIP: "stablehlo.divide"
// ROUNDTRIP-NOT: "stablehlo.reduce"

// PLAN: "shuttle.materialization_plan"
// PLAN-COUNT-7: "shuttle.materialization_buffer"
// PLAN: "shuttle.materialization_task"() <{{.*}}composition_kind = "contract_fold"
// PLAN-SAME: input_buffers = array<i64: 0, 1>
// PLAN-SAME: output_buffers = array<i64: 3, 4, 5>
// PLAN: "shuttle.materialization_task"() <{{.*}}composition_kind = "map_contract"
// PLAN-SAME: dependencies = array<i64: 0>
// PLAN-SAME: input_buffers = array<i64: 3, 4, 5, 2>
// PLAN-SAME: output_buffers = array<i64: 6>
// PLAN-NOT: "shuttle.materialization_task"
// PLAN-NOT: online_

// PHYSICAL: #[[PHYSICAL_RELATION:set[0-9]*]] = affine_set<(d0, d1) : (((d0 * 8) floordiv 7 + d1) * 7 - d0 * 8 + 6 >= 0, d0 * 8 - ((d0 * 8) floordiv 7 + d1) * 7 + 7 >= 0, -((d0 * 8) floordiv 7 + d1) + 67 >= 0)>
// PHYSICAL: "shuttle.materialization_plan"() <{{.*}}normalized_exp_contract_regions = [
// PHYSICAL: "shuttle.materialization_task"() <{{.*}}composition_kind = "contract_fold"
// PHYSICAL-SAME: domain_shape = array<i64: 1, 2, 2, 17, 17>
// PHYSICAL-SAME: input_buffers = array<i64: 0, 1>
// PHYSICAL-SAME: output_buffers = array<i64: 3, 4, 5>
// PHYSICAL-NEXT: "shuttle.materialization_task"() <{{.*}}composition_kind = "map_contract"
// PHYSICAL-SAME: dependencies = array<i64: 0>
// PHYSICAL-SAME: input_buffers = array<i64: 3, 4, 5, 2>
// PHYSICAL-SAME: output_buffers = array<i64: 6>
// PHYSICAL: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 2, 2, 17, 17>
// PHYSICAL-SAME: grid_shape = array<i64: 68>
// PHYSICAL-SAME: kind = #shuttle.schedule_task_kind<row_fold>
// PHYSICAL-SAME: reduction_axis = 4 : i64
// PHYSICAL-SAME: reduction_state_bytes_per_thread = 8 : i64
// PHYSICAL-SAME: scratch_bytes = 256 : i64
// PHYSICAL-SAME: tile_shape = array<i64: 1, 1, 1, 1, 17>
// PHYSICAL-NEXT: "shuttle.schedule_task"() <{{.*}}dependencies = array<i64: 0>
// PHYSICAL-SAME: domain_shape = array<i64: 1, 2, 2, 17, 7, 17>
// PHYSICAL-SAME: grid_shape = array<i64: 60>
// PHYSICAL-SAME: kind = #shuttle.schedule_task_kind<contract>
// PHYSICAL-SAME: tile_shape = array<i64: 8>
// PHYSICAL: "shuttle.event_tensor"()
// PHYSICAL-SAME: relation_domain_shape = array<i64: 60, 2>
// PHYSICAL-SAME: relation_predicate = #[[PHYSICAL_RELATION]]
// PHYSICAL-SAME: storage_capacity = 120 : i64
// PHYSICAL-CARDINALITY: event 0 exact_pair_cardinality = 119
// ASCII bytes for `tanh.approx.f32` in the generated PTX payload.
// IDENTITY-NOT: 116, 97, 110, 104, 46, 97, 112, 112, 114, 111, 120, 46, 102, 51, 50
// SOFTCAP: 116, 97, 110, 104, 46, 97, 112, 112, 114, 111, 120, 46, 102, 51, 50
// AFFINE-NOT: 116, 97, 110, 104, 46, 97, 112, 112, 114, 111, 120, 46, 102, 51, 50
// PHYSICAL: "shuttle.device_entry"() <{{.*}}kernel_arity = 5 : i64
// PHYSICAL-NEXT: "shuttle.device_entry"() <{{.*}}kernel_arity = 5 : i64
// PHYSICAL: "shuttle.event_realization"()
// PHYSICAL-SAME: covering_launch_order = array<i64: 0, 0, 1, 0>
// PHYSICAL-SAME: kind = #shuttle.event_realization_kind<stream_order_erased>
// PHYSICAL: "shuttle.invocation_abi"()
// PHYSICAL-COUNT-7: "shuttle.invocation_slot"()
// PHYSICAL-NOT: "shuttle.invocation_slot"
// PHYSICAL-NOT: online_
