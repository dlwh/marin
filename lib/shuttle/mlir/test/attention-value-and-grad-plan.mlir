// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// JAX owns the VJP. Shuttle must derive the fused reverse task family from
// generic Contract/Fold/Map relations without using argument shapes to guess
// which differential operand is the output cotangent.

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s2048-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=S2048
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s2048-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-report-event-cardinality -o /dev/null | FileCheck %s --check-prefix=S2048-CARDINALITY
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-finalized-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=FINALIZED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-scalar-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=SCALAR
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-event-predicate --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=EVENT-PREDICATE-ERROR
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-test-mutate-schedule-index-domain-provenance --shuttle-verify-simt32-row-fold-schedule 2>&1 | FileCheck %s --check-prefix=INDEX-PROVENANCE-ERROR
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s128-d128.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-test-mutate-materialization-index-domain-provenance --shuttle-verify-materialization-plan 2>&1 | FileCheck %s --check-prefix=MATERIALIZATION-PROVENANCE-ERROR

// CHECK: #set = affine_set<(d0, d1, d2, d3) : (d2 * 64 - d3 * 128 + 63 >= 0, d3 * -128 + 127 >= 0)>
// S2048: #set = affine_set<(d0, d1, d2, d3) : (d2 * 64 - d3 * 128 + 63 >= 0, d3 * -128 + 2047 >= 0)>
// EVENT-PREDICATE-ERROR: must equal the structural tiled task dependence
// INDEX-PROVENANCE-ERROR: structural composition metadata must equal materialization
// MATERIALIZATION-PROVENANCE-ERROR: generic composition must equal structural SSA derivation

// A scalar objective is a generic Map/Fold residual. Its escaped cotangent
// value is an ordinary SSA boundary consumed by both reverse Contract tasks.
// SCALAR: "shuttle.materialization_plan"() <{fingerprint =
// SCALAR-SAME: normalized_exp_contract_pullbacks = [{consumer_weight_cotangent_operation = 53 : i64
// SCALAR-SAME: forward_result_buffer = 11 : i64, forward_task = 0 : i64
// SCALAR-SAME: maximum_state_buffer = 8 : i64
// SCALAR-SAME: normalizer_state_buffer = 9 : i64
// SCALAR-SAME: objective_task = 1 : i64
// SCALAR-SAME: output_cotangent_buffer = 3 : i64, output_dependent_fold_operation = 58 : i64, output_dependent_state_buffer = 10 : i64
// SCALAR-SAME: owner_pullback_task = 2 : i64
// SCALAR-SAME: streamed_pullback_task = 3 : i64, terminal_contract_operations = array<i64: 87, 86, 51>, terminal_cotangent_buffers = array<i64: 7, 6, 5>}]
// SCALAR-SAME: normalized_exp_contract_regions = [{{.*}}ordinal = 0 : i64{{.*}}]
// SCALAR-SAME: stateful_contract_fold_pullback = true
// SCALAR: "shuttle.materialization_task"() <{{.*}}ordinal = 0 : i64
// SCALAR-SAME: output_buffers = array<i64: 11, 8, 9>
// SCALAR-NEXT: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64: 0>
// SCALAR-SAME: fold_region_sources = [#shuttle.source_ref<0, 0, 41, 0>, #shuttle.source_ref<0, 0, 42, 0>, #shuttle.source_ref<0, 0, 43, 0>, #shuttle.source_ref<0, 0, 44, 0>, #shuttle.source_ref<0, 0, 45, 0>]
// SCALAR-SAME: ordinal = 1 : i64
// SCALAR-SAME: output_buffers = array<i64: 4, 12>
// SCALAR-NEXT: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64: 0, 1>
// SCALAR-SAME: input_buffers = array<i64: 0, 1, 2, 3, 8, 9, 12>
// SCALAR-SAME: ordinal = 2 : i64
// SCALAR-NEXT: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64: 0, 1, 2>
// SCALAR-SAME: input_buffers = array<i64: 0, 1, 2, 3, 12, 8, 9, 10>
// SCALAR-SAME: ordinal = 3 : i64
// SCALAR-NEXT: "shuttle.materialization_plan_yield"

// CHECK: "shuttle.materialization_plan"() <{{.*}}stateful_contract_fold_pullback = true
// CHECK-COUNT-11: "shuttle.materialization_buffer"

// The forward task excludes source argument 3 (dO), exports the public value,
// and saves the two normalized-exponential Fold states.
// CHECK: "shuttle.materialization_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2, 3>
// CHECK-SAME: composition_state_outputs = array<i64: 8, 9>
// CHECK-SAME: composition_stream_dimensions = array<i64: 5>
// CHECK-SAME: input_buffers = array<i64: 0, 1, 2>
// CHECK-SAME: online_vector_dimensions = array<i64: 4>
// CHECK-SAME: ordinal = 0 : i64
// CHECK-SAME: output_buffers = array<i64: 4, 8, 9>
// CHECK-SAME: reduction_dimensions = array<i64: 5>

// The owner-adjoint task consumes dO and the forward output/state, emits the
// owner-shaped gradient, and exports the centered pullback state.
// CHECK-NEXT: "shuttle.materialization_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2, 3>
// CHECK-SAME: composition_state_inputs = array<i64: 8, 9>
// CHECK-SAME: composition_state_outputs = array<i64: 10>
// CHECK-SAME: composition_stream_dimensions = array<i64: 5>
// CHECK-SAME: input_buffers = array<i64: 0, 1, 2, 3, 4, 8, 9>
// CHECK-SAME: online_vector_dimensions = array<i64: 4>
// CHECK-SAME: ordinal = 1 : i64
// CHECK-SAME: output_buffers = array<i64: 7, 10>
// CHECK-SAME: reduction_dimensions = array<i64: 5>

// The streamed-adjoint task owns the remaining two terminal Contracts. Its
// owner/vector/stream partition comes from their affine indexing maps.
// CHECK-NEXT: "shuttle.materialization_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2>
// CHECK-SAME: composition_state_inputs = array<i64: 8, 9, 10>
// CHECK-SAME: composition_stream_dimensions = array<i64: 4, 5>
// CHECK-SAME: input_buffers = array<i64: 0, 1, 2, 3, 8, 9, 10>
// CHECK-SAME: online_vector_dimensions = array<i64: 3>
// CHECK-SAME: ordinal = 2 : i64
// CHECK-SAME: output_buffers = array<i64: 5, 6>
// CHECK-SAME: reduction_dimensions = array<i64: 4, 5>
// CHECK-SAME: source = #shuttle.source_ref<0, 0, 77, 0>
// CHECK-NEXT: "shuttle.materialization_plan_yield"

// CHECK: "shuttle.schedule_plan"()
// CHECK-COUNT-11: "shuttle.schedule_buffer"
// CHECK: "shuttle.schedule_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2, 3>
// CHECK-SAME: grid_shape = array<i64: 64>
// CHECK-SAME: online_vector_dimensions = array<i64: 4>
// CHECK-SAME: ordinal = 0 : i64
// CHECK-SAME: tile_shape = array<i64: 1, 1, 1, 64, 128, 128>
// CHECK-NEXT: "shuttle.schedule_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2, 3>
// CHECK-SAME: grid_shape = array<i64: 64>
// CHECK-SAME: online_vector_dimensions = array<i64: 4>
// CHECK-SAME: ordinal = 1 : i64
// CHECK-SAME: tile_shape = array<i64: 1, 1, 1, 64, 128, 128>
// CHECK-NEXT: "shuttle.schedule_task"() <{{.*}}composition_owner_dimensions = array<i64: 0, 1, 2>
// CHECK-SAME: grid_shape = array<i64: 8>
// CHECK-SAME: online_vector_dimensions = array<i64: 3>
// CHECK-SAME: ordinal = 2 : i64
// CHECK-SAME: serial_tiles = 2 : i64
// CHECK-SAME: tile_shape = array<i64: 1, 1, 128, 128, 4, 64>

// The logical Event Tensor relation is exact even though this physical plan
// erases it to ordered launches. Root and carry have a diagonal owner relation;
// the terminal task shares the batch/head prefix and crosses only the distinct
// producer and consumer owner-tile suffixes.
// CHECK-NEXT: "shuttle.event_tensor"() <{{.*}}consumer_task = 1 : i64
// CHECK-SAME: ordinal = 0 : i64
// CHECK-SAME: producer_task = 0 : i64
// CHECK-SAME: relation_domain_shape = array<i64: 64>
// CHECK-SAME: storage_capacity = 64 : i64
// CHECK-NEXT: "shuttle.event_tensor"() <{{.*}}consumer_task = 2 : i64
// CHECK-SAME: ordinal = 1 : i64
// CHECK-SAME: producer_task = 0 : i64
// CHECK-SAME: relation_domain_shape = array<i64: 8, 4, 2, 1>
// CHECK-SAME: relation_predicate = #set
// CHECK-SAME: storage_capacity = 64 : i64
// CHECK-NEXT: "shuttle.event_tensor"() <{{.*}}consumer_task = 2 : i64
// CHECK-SAME: ordinal = 2 : i64
// CHECK-SAME: producer_task = 1 : i64
// CHECK-SAME: relation_domain_shape = array<i64: 8, 4, 2, 1>
// CHECK-SAME: relation_predicate = #set
// CHECK-SAME: storage_capacity = 64 : i64

// The S=2048 graph distinguishes the sequence and feature extents. The two
// streamed terminal Contracts differ by a projected domain permutation, but
// the task anchor retains the producer-stream Contract's native axes.
// S2048: "shuttle.materialization_plan"() <{{.*}}stateful_contract_fold_pullback = true
// S2048-COUNT-2: "shuttle.materialization_task"
// S2048: "shuttle.materialization_task"() <{{.*}}composition_consumer_contraction_dimensions = array<i64: 5>
// S2048-SAME: composition_producer_contraction_dimensions = array<i64: 5>
// S2048-SAME: domain_shape = array<i64: 1, 8, 2048, 128, 4, 2048>
// S2048-SAME: ordinal = 2 : i64
// S2048-SAME: output_buffers = array<i64: 5, 6>
// S2048-SAME: reduction_dimensions = array<i64: 4, 5>
// S2048-SAME: source = #shuttle.source_ref<0, 0, 77, 0>
// S2048: "shuttle.event_tensor"() <{{.*}}consumer_task = 1 : i64
// S2048-SAME: ordinal = 0 : i64
// S2048-SAME: producer_task = 0 : i64
// S2048-SAME: relation_domain_shape = array<i64: 1024>
// S2048-SAME: storage_capacity = 1024 : i64
// S2048-NEXT: "shuttle.event_tensor"() <{{.*}}consumer_task = 2 : i64
// S2048-SAME: ordinal = 1 : i64
// S2048-SAME: producer_task = 0 : i64
// S2048-SAME: relation_domain_shape = array<i64: 8, 4, 32, 16>
// S2048-SAME: relation_predicate = #set
// S2048-SAME: storage_capacity = 16384 : i64
// S2048-CARDINALITY: event 0 exact_pair_cardinality = 1024
// S2048-CARDINALITY-NEXT: event 1 exact_pair_cardinality = 8704
// S2048-CARDINALITY-NEXT: event 2 exact_pair_cardinality = 8704
// S2048-NEXT: "shuttle.event_tensor"() <{{.*}}consumer_task = 2 : i64
// S2048-SAME: ordinal = 2 : i64
// S2048-SAME: producer_task = 1 : i64
// S2048-SAME: relation_domain_shape = array<i64: 8, 4, 32, 16>
// S2048-SAME: relation_predicate = #set
// S2048-SAME: storage_capacity = 16384 : i64

// A branching h(U) = tanh(U) + 0.25 * U scalar Map and its JAX-owned VJP
// retain the same Contract/Fold ownership and schedule family. The planner
// derives the output cotangent and terminal gradient ownership from SSA and
// Contract operand relations, rather than transitive dependence on V.
// FINALIZED: "shuttle.materialization_plan"() <{{.*}}stateful_contract_fold_pullback = true
// FINALIZED-COUNT-11: "shuttle.materialization_buffer"
// FINALIZED: "shuttle.materialization_task"() <{{.*}}input_buffers = array<i64: 0, 1, 2>
// FINALIZED-SAME: ordinal = 0 : i64
// FINALIZED-SAME: output_buffers = array<i64: 4, 8, 9>
// FINALIZED-NEXT: "shuttle.materialization_task"() <{{.*}}input_buffers = array<i64: 0, 1, 2, 3, 4, 8, 9>
// FINALIZED-SAME: ordinal = 1 : i64
// FINALIZED-SAME: output_buffers = array<i64: 7, 10>
// FINALIZED-NEXT: "shuttle.materialization_task"() <{{.*}}input_buffers = array<i64: 0, 1, 2, 3, 8, 9, 10>
// FINALIZED-SAME: ordinal = 2 : i64
// FINALIZED-SAME: output_buffers = array<i64: 5, 6>
// FINALIZED: "shuttle.schedule_plan"()
// FINALIZED-COUNT-11: "shuttle.schedule_buffer"
// FINALIZED: "shuttle.schedule_task"() <{{.*}}ordinal = 0 : i64
// FINALIZED-SAME: tile_shape = array<i64: 1, 1, 1, 64, 128, 128>
// FINALIZED-NEXT: "shuttle.schedule_task"() <{{.*}}ordinal = 1 : i64
// FINALIZED-SAME: tile_shape = array<i64: 1, 1, 1, 64, 128, 128>
// FINALIZED-NEXT: "shuttle.schedule_task"() <{{.*}}ordinal = 2 : i64
// FINALIZED-SAME: tile_shape = array<i64: 1, 1, 128, 128, 4, 64>
