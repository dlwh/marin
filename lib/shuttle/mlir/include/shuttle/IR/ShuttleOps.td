// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IR_SHUTTLEOPS_TD_
#define SHUTTLE_IR_SHUTTLEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "shuttle/IR/ShuttleAttrs.td"
include "shuttle/IR/ShuttleDialect.td"

class Shuttle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Shuttle_Dialect, mnemonic, traits>;

def Shuttle_RegionOp : Shuttle_Op<"region", [RecursiveMemoryEffects,
                                             SingleBlockImplicitTerminator<
                                                 "::mlir::shuttle::YieldOp">]> {
  let summary = "Explicit SSA boundary for selected structural algebra";
  let arguments = (ins Variadic<AnyType>:$inputs, Shuttle_PolicyAttr:$policy,
      ArrayAttr:$source_refs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_ScanOp
    : Shuttle_Op<
          "scan", [ParentOneOf<["::mlir::shuttle::RegionOp"]>, Pure,
                   SingleBlockImplicitTerminator<"::mlir::shuttle::YieldOp">]> {
  let summary = "Ordered state recurrence with an algebraic body";
  let arguments = (ins Variadic<AnyType>:$inputs,
      I64Attr:$axis, I64Attr:$extent, Shuttle_ScanDirectionAttr:$direction,
      DenseI64ArrayAttr:$state_operands,
      DenseI64ArrayAttr:$state_body_arguments,
      DenseI64ArrayAttr:$state_body_results,
      DenseI64ArrayAttr:$state_results,
      DenseI64ArrayAttr:$invariant_operands,
      DenseI64ArrayAttr:$invariant_body_arguments,
      DenseI64ArrayAttr:$scanned_operands,
      DenseI64ArrayAttr:$scanned_body_arguments,
      DenseI64ArrayAttr:$scanned_dimensions,
      DenseI64ArrayAttr:$emitted_body_results,
      DenseI64ArrayAttr:$emitted_results,
      DenseI64ArrayAttr:$emitted_dimensions,
      ArrayAttr:$sources,
      OptionalAttr<ArrayAttr>:$result_sources,
      OptionalAttr<ArrayAttr>:$erased_terminator_records);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_MapOp
    : Shuttle_Op<
          "map", [ParentOneOf<["::mlir::shuttle::RegionOp",
                               "::mlir::shuttle::ScanOp"]>, Pure,
                  SingleBlockImplicitTerminator<"::mlir::shuttle::YieldOp">]> {
  let summary = "Pointwise scalar computation with explicit indexing maps";
  let arguments = (ins Variadic<AnyType>:$inputs,
      AffineMapArrayAttr:$indexing_maps,
      DefaultValuedAttr<Shuttle_MapSemanticsAttr,
                        "::mlir::shuttle::MapSemantics::Pointwise">:$semantics,
      Shuttle_SourceRefAttr:$source,
      OptionalAttr<DictionaryAttr>:$source_group);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_ContractOp
    : Shuttle_Op<"contract", [ParentOneOf<["::mlir::shuttle::RegionOp",
                                            "::mlir::shuttle::ScanOp"]>,
                              Pure]> {
  let summary = "Generic contraction with normalized indexing metadata";
  let arguments = (ins Variadic<AnyType>:$inputs,
      AffineMapArrayAttr:$indexing_maps, ArrayAttr:$iterator_kinds,
      TypeArrayAttr:$accumulator_types, ArrayAttr:$precision,
      StrAttr:$algorithm, Shuttle_SourceRefAttr:$source,
      OptionalAttr<DictionaryAttr>:$source_group,
      OptionalAttr<StrAttr>:$source_precision_form);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def Shuttle_RelationOp
    : Shuttle_Op<"relation", [ParentOneOf<["::mlir::shuttle::RegionOp",
                                            "::mlir::shuttle::ScanOp"]>,
                               Pure]> {
  let summary = "Lossless index-domain relation and predicate operation";
  let arguments = (ins Variadic<AnyType>:$inputs,
      Shuttle_RelationSemanticsAttr:$semantics,
      ArrayAttr:$parameters, Shuttle_SourceRefAttr:$source,
      OptionalAttr<DictionaryAttr>:$source_group);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def Shuttle_SelectionOp
    : Shuttle_Op<"selection", [ParentOneOf<["::mlir::shuttle::RegionOp",
                                             "::mlir::shuttle::ScanOp"]>,
                                Pure]> {
  let summary = "Select ordered values and their source-axis identities";
  let arguments = (ins AnyType:$input, I64Attr:$dimension,
      I64Attr:$count, Shuttle_SelectionOrderAttr:$order,
      Shuttle_SelectionComparisonAttr:$comparison, BoolAttr:$stable,
      Shuttle_SelectionTieBreakAttr:$tie_break, ArrayAttr:$sources);
  let results = (outs AnyType:$selected_values, AnyType:$selected_indices);
  let hasVerifier = 1;
}

def Shuttle_ScalarConvertOp : Shuttle_Op<"scalar_convert", [Pure]> {
  let summary = "Scalar conversion with explicit rounding provenance";
  let arguments = (ins AnyType:$input,
      Shuttle_ScalarConvertSemanticsAttr:$semantics);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Shuttle_FoldOp
    : Shuttle_Op<
          "fold", [AttrSizedOperandSegments,
                   ParentOneOf<["::mlir::shuttle::RegionOp",
                                "::mlir::shuttle::ScanOp"]>, Pure,
                   SingleBlockImplicitTerminator<"::mlir::shuttle::YieldOp">]> {
  let summary = "Reduction with explicit accumulator and ordering contract";
  let arguments = (ins Variadic<AnyType>:$inputs,
      Variadic<AnyType>:$initializers, AffineMapArrayAttr:$indexing_maps,
      DenseI64ArrayAttr:$reduction_dimensions,
      TypeArrayAttr:$accumulator_types, BoolAttr:$order_free,
      Shuttle_SourceRefAttr:$source,
      OptionalAttr<DictionaryAttr>:$source_group);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
  let hasRegionVerifier = 1;
}

def Shuttle_YieldOp
    : Shuttle_Op<"yield", [ParentOneOf<["::mlir::shuttle::RegionOp",
                                        "::mlir::shuttle::ScanOp",
                                        "::mlir::shuttle::MapOp",
                                        "::mlir::shuttle::FoldOp"]>,
                           Pure, Terminator]> {
  let summary = "Yield values from a Shuttle algebra region";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_MaterializationPlanOp
    : Shuttle_Op<
          "materialization_plan", [IsolatedFromAbove,
                            SingleBlockImplicitTerminator<
                                "::mlir::shuttle::MaterializationPlanYieldOp">]> {
  let summary = "Non-executable task, buffer, and lifetime plan";
  let arguments = (ins I64Attr:$schema_version, Shuttle_PolicyAttr:$policy,
      StrAttr:$fingerprint,
      OptionalAttr<BoolAttr>:$componentized,
      OptionalAttr<BoolAttr>:$stateful_contract_fold_pullback,
      OptionalAttr<ArrayAttr>:$normalized_exp_contract_regions,
      OptionalAttr<ArrayAttr>:$normalized_exp_contract_pullbacks);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_MaterializationBufferOp
    : Shuttle_Op<"materialization_buffer",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>]> {
  let summary = "One explicitly materialized logical tensor buffer";
  let arguments = (ins I64Attr:$ordinal, TypeAttr:$tensor_type,
      Shuttle_MaterializationStorageAttr:$storage, BoolAttr:$live_in,
      BoolAttr:$live_out, OptionalAttr<I64Attr>:$producer,
      OptionalAttr<I64Attr>:$source_argument,
      DenseI64ArrayAttr:$consumers, I64Attr:$lifetime_start,
      I64Attr:$lifetime_end);
}

def Shuttle_MaterializationTaskOp
    : Shuttle_Op<"materialization_task",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>]> {
  let summary = "One generic algebra task family";
  let arguments = (ins I64Attr:$ordinal,
      Shuttle_MaterializationTaskKindAttr:$kind,
      DenseI64ArrayAttr:$domain_shape, DenseI64ArrayAttr:$reduction_dimensions,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      DenseI64ArrayAttr:$dependencies, OptionalAttr<BoolAttr>:$order_free,
      StrAttr:$semantic_fingerprint, Shuttle_SourceRefAttr:$source,
      OptionalAttr<ArrayAttr>:$source_group,
      OptionalAttr<DictionaryAttr>:$source_expansion,
      OptionalAttr<ArrayAttr>:$source_expansion_group,
      OptionalAttr<ArrayAttr>:$fused_sources,
      OptionalAttr<AffineMapArrayAttr>:$composed_input_maps,
      OptionalAttr<ArrayAttr>:$fused_output_sources,
      OptionalAttr<ArrayAttr>:$fold_epilogue_sources,
      OptionalAttr<AffineMapArrayAttr>:$epilogue_input_maps,
      OptionalAttr<ArrayAttr>:$fold_region_sources,
      OptionalAttr<ArrayAttr>:$fold_region_output_sources,
      OptionalAttr<ArrayAttr>:$fold_region_phases,
      OptionalAttr<ArrayAttr>:$fold_region_boundary_uses,
      OptionalAttr<ArrayAttr>:$fold_region_recomputed_sources,
      OptionalAttr<ArrayAttr>:$contract_region_sources,
      OptionalAttr<StrAttr>:$contract_region_strategy,
      OptionalAttr<Shuttle_SourceRefAttr>:$contract_region_fold_source,
      OptionalAttr<Shuttle_SourceRefAttr>:$contract_region_contract_source,
      OptionalAttr<Shuttle_SourceRefAttr>:$contract_region_bf16_boundary_source,
      OptionalAttr<ArrayAttr>:$contract_region_epilogue_sources,
      OptionalAttr<I64Attr>:$contract_region_normalized_operand,
      OptionalAttr<DictionaryAttr>:$contract_region_cost,
      OptionalAttr<ArrayAttr>:$contract_region_recomputed_sources,
      OptionalAttr<StrAttr>:$composition_kind,
      OptionalAttr<ArrayAttr>:$composition_sources,
      OptionalAttr<ArrayAttr>:$composition_phases,
      OptionalAttr<ArrayAttr>:$composition_phase_edges,
      OptionalAttr<ArrayAttr>:$composition_result_bindings,
      OptionalAttr<ArrayAttr>:$composition_recomputed_sources,
      OptionalAttr<DictionaryAttr>:$composition_cost,
      OptionalAttr<DenseI64ArrayAttr>:$composition_owner_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_stream_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_inputs,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_outputs,
      OptionalAttr<DenseI64ArrayAttr>:$composition_producer_contraction_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_consumer_contraction_dimensions,
      OptionalAttr<Shuttle_SourceRefAttr>:$composition_index_domain_source,
      OptionalAttr<DenseI64ArrayAttr>:$composition_index_domain_shape,
      OptionalAttr<AffineMapAttr>:$composition_index_domain_map,
      OptionalAttr<Shuttle_SourceRefAttr>:$composition_ordered_index_source,
      OptionalAttr<DenseI64ArrayAttr>:$composition_ordered_index_dimensions,
      OptionalAttr<BoolAttr>:$composition_ordered_index_inclusive,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_access_buffers,
      OptionalAttr<AffineMapArrayAttr>:$composition_state_access_maps,
      OptionalAttr<DenseI64ArrayAttr>:$online_vector_dimensions,
      OptionalAttr<BoolAttr>:$online_mma);
}

def Shuttle_MaterializationPlanYieldOp
    : Shuttle_Op<"materialization_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>,
                  Terminator]> {
  let summary = "Terminate a non-executable materialization plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_SchedulePlanOp
    : Shuttle_Op<
          "schedule_plan", [IsolatedFromAbove,
                            SingleBlockImplicitTerminator<
                                "::mlir::shuttle::SchedulePlanYieldOp">]> {
  let summary = "Non-executable target schedule and logical indexing contract";
  let arguments = (ins I64Attr:$schema_version, Shuttle_ScheduleTargetAttr:$target,
      Shuttle_PolicyAttr:$policy, StrAttr:$source_plan_fingerprint,
      StrAttr:$fingerprint, OptionalAttr<BoolAttr>:$eventized,
      OptionalAttr<ArrayAttr>:$parametric_dependencies);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_ScheduleBufferOp
    : Shuttle_Op<"schedule_buffer",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>]> {
  let summary = "Logical iteration contract for one planned buffer";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_buffer,
      TypeAttr:$tensor_type, Shuttle_ScheduleBufferIndexingAttr:$indexing,
      DenseI64ArrayAttr:$iteration_order, I64Attr:$lifetime_start,
      I64Attr:$lifetime_end);
}

def Shuttle_ScheduleTaskOp
    : Shuttle_Op<"schedule_task",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>]> {
  let summary = "One bounded logical SIMT32 schedule candidate";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_task,
      Shuttle_ScheduleTaskKindAttr:$kind, DenseI64ArrayAttr:$domain_shape,
      DenseI64ArrayAttr:$grid_shape, DenseI64ArrayAttr:$tile_shape,
      I64Attr:$serial_tiles, I64Attr:$workgroup_threads,
      I64Attr:$subgroup_size, I64Attr:$scratch_bytes,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      DenseI64ArrayAttr:$dependencies, OptionalAttr<I64Attr>:$reduction_axis,
      OptionalAttr<DenseI64ArrayAttr>:$reduction_dimensions,
      OptionalAttr<Shuttle_ScheduleReductionOrderAttr>:$reduction_order,
      OptionalAttr<I64Attr>:$reduction_state_bytes_per_thread,
      OptionalAttr<DenseI64ArrayAttr>:$reduction_merge_grid_shape,
      OptionalAttr<DenseI64ArrayAttr>:$reduction_merge_tile_shape,
      OptionalAttr<I64Attr>:$reduction_merge_serial_tiles,
      OptionalAttr<DenseI64ArrayAttr>:$reduction_partial_buffer_shape,
      StrAttr:$semantic_fingerprint,
      OptionalAttr<ArrayAttr>:$composition_phase_schedules,
      OptionalAttr<DenseI64ArrayAttr>:$composition_owner_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_stream_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_inputs,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_outputs,
      OptionalAttr<DenseI64ArrayAttr>:$composition_producer_contraction_dimensions,
      OptionalAttr<DenseI64ArrayAttr>:$composition_consumer_contraction_dimensions,
      OptionalAttr<Shuttle_SourceRefAttr>:$composition_index_domain_source,
      OptionalAttr<DenseI64ArrayAttr>:$composition_index_domain_shape,
      OptionalAttr<AffineMapAttr>:$composition_index_domain_map,
      OptionalAttr<Shuttle_SourceRefAttr>:$composition_ordered_index_source,
      OptionalAttr<DenseI64ArrayAttr>:$composition_ordered_index_dimensions,
      OptionalAttr<BoolAttr>:$composition_ordered_index_inclusive,
      OptionalAttr<DenseI64ArrayAttr>:$composition_state_access_buffers,
      OptionalAttr<AffineMapArrayAttr>:$composition_state_access_maps,
      OptionalAttr<DenseI64ArrayAttr>:$online_vector_dimensions,
      OptionalAttr<BoolAttr>:$online_mma,
      OptionalAttr<StrAttr>:$contract_region_strategy,
      OptionalAttr<DictionaryAttr>:$contract_region_cost,
      OptionalAttr<Shuttle_RelationSemanticsAttr>:$relation_semantics,
      OptionalAttr<ArrayAttr>:$relation_parameters,
      OptionalAttr<I64Attr>:$selection_dimension,
      OptionalAttr<I64Attr>:$selection_count,
      OptionalAttr<Shuttle_SelectionOrderAttr>:$selection_order,
      OptionalAttr<Shuttle_SelectionComparisonAttr>:$selection_comparison,
      OptionalAttr<BoolAttr>:$selection_stable,
      OptionalAttr<Shuttle_SelectionTieBreakAttr>:$selection_tie_break);
}

def Shuttle_EventTensorOp
    : Shuttle_Op<"event_tensor",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>]> {
  let summary = "Generation-aware event over one scheduled task-instance relation";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$producer_task,
      I64Attr:$consumer_task, DenseI64ArrayAttr:$relation_domain_shape,
      OptionalAttr<IntegerSetAttr>:$relation_predicate,
      AffineMapAttr:$producer_indices, AffineMapAttr:$consumer_indices,
      I64Attr:$storage_capacity, AffineMapAttr:$storage_slot,
      AffineMapAttr:$generation_index, I64Attr:$generations,
      I64Attr:$notify_count,
      Shuttle_EventScopeAttr:$scope, BoolAttr:$release, BoolAttr:$acquire,
      OptionalAttr<StrAttr>:$producer_phase,
      OptionalAttr<StrAttr>:$consumer_phase);
}

def Shuttle_EventRealizationOp
    : Shuttle_Op<"event_realization",
                 [ParentOneOf<["::mlir::shuttle::EventLoweringPlanOp"]>]> {
  let summary = "Erasure proof or imperative realization of one logical event";
  let arguments = (ins I64Attr:$event,
      Shuttle_EventRealizationKindAttr:$kind,
      DenseI64ArrayAttr:$covering_launch_order);
}

def Shuttle_ImperativeEventNotifyOp
    : Shuttle_Op<"imperative_event_notify",
                 [ParentOneOf<["::mlir::shuttle::EventLoweringPlanOp"]>]> {
  let summary = "Notify one exact bounded-buffer event relation";
  let arguments = (ins I64Attr:$event, I64Attr:$producer_task,
      I64Attr:$consumer_task, DenseI64ArrayAttr:$relation_domain_shape,
      OptionalAttr<IntegerSetAttr>:$relation_predicate,
      AffineMapAttr:$producer_indices, AffineMapAttr:$consumer_indices,
      I64Attr:$storage_capacity, AffineMapAttr:$storage_slot,
      AffineMapAttr:$generation_index, I64Attr:$generations,
      I64Attr:$notify_count, Shuttle_EventScopeAttr:$scope,
      BoolAttr:$release, OptionalAttr<StrAttr>:$producer_phase,
      OptionalAttr<StrAttr>:$consumer_phase);
}

def Shuttle_ImperativeEventWaitOp
    : Shuttle_Op<"imperative_event_wait",
                 [ParentOneOf<["::mlir::shuttle::EventLoweringPlanOp"]>]> {
  let summary = "Wait for one exact bounded-buffer event relation";
  let arguments = (ins I64Attr:$event, I64Attr:$producer_task,
      I64Attr:$consumer_task, DenseI64ArrayAttr:$relation_domain_shape,
      OptionalAttr<IntegerSetAttr>:$relation_predicate,
      AffineMapAttr:$producer_indices, AffineMapAttr:$consumer_indices,
      I64Attr:$storage_capacity, AffineMapAttr:$storage_slot,
      AffineMapAttr:$generation_index, I64Attr:$generations,
      I64Attr:$notify_count, Shuttle_EventScopeAttr:$scope,
      BoolAttr:$acquire, OptionalAttr<StrAttr>:$producer_phase,
      OptionalAttr<StrAttr>:$consumer_phase);
}

def Shuttle_SchedulePlanYieldOp
    : Shuttle_Op<"schedule_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>,
                  Terminator]> {
  let summary = "Terminate a non-executable schedule plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_EventLoweringPlanOp
    : Shuttle_Op<
          "event_lowering_plan", [IsolatedFromAbove,
                                  SingleBlockImplicitTerminator<
                                      "::mlir::shuttle::EventLoweringPlanYieldOp">]> {
  let summary = "Source-bound requirements or physical realizations of scheduled logical events";
  let arguments = (ins I64Attr:$schema_version,
      StrAttr:$source_schedule_fingerprint,
      OptionalAttr<StrAttr>:$physical_launch_recipe_fingerprint,
      StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_EventLoweringPlanYieldOp
    : Shuttle_Op<"event_lowering_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::EventLoweringPlanOp"]>,
                  Terminator]> {
  let summary = "Terminate an Event Tensor lowering plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_InternalEventPlanOp
    : Shuttle_Op<"internal_event_plan", [IsolatedFromAbove,
                                          SingleBlockImplicitTerminator<
                                              "::mlir::shuttle::InternalEventPlanYieldOp">]> {
  let summary = "Typed physically bound plan for refined internal Event dependencies";
  let arguments = (ins I64Attr:$schema_version,
      StrAttr:$physical_schedule_fingerprint,
      StrAttr:$physical_launch_recipe_fingerprint,
      StrAttr:$canonical_schedule,
      StrAttr:$canonical_schedule_digest,
      StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_InternalEventTaskOp
    : Shuttle_Op<"internal_event_task",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "One exact task family in a refined internal Event plan";
  let arguments = (ins I64Attr:$region, I64Attr:$ordinal, StrAttr:$name,
      ArrayAttr:$axes, OptionalAttr<StrAttr>:$placement);
}

def Shuttle_InternalEventTensorOp
    : Shuttle_Op<"internal_event_tensor",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "One exact refined internal EventTensor relation";
  let arguments = (ins I64Attr:$region, I64Attr:$ordinal, StrAttr:$name,
      I64Attr:$producer_task, I64Attr:$consumer_task,
      ArrayAttr:$required_pairs, ArrayAttr:$event_axes,
      ArrayAttr:$notify_pairs, ArrayAttr:$trigger_pairs,
      ArrayAttr:$initial_counts, StrAttr:$initial_count_provenance,
      StrAttr:$memory_scope, StrAttr:$generation_policy,
      StrAttr:$visibility_scope, BoolAttr:$release, BoolAttr:$acquire,
      OptionalAttr<StrAttr>:$scheduling_mode);
}

def Shuttle_InternalEventAllocationOp
    : Shuttle_Op<"internal_event_allocation",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "One bounded owner-relative allocation for internal Events";
  let arguments = (ins I64Attr:$region, I64Attr:$ordinal,
      StrAttr:$allocation_id, DenseI64ArrayAttr:$covered_events,
      StrAttr:$legalization,
      StrAttr:$scope, BoolAttr:$release, BoolAttr:$acquire,
      I64Attr:$capacity, ArrayAttr:$mechanism_coordinates,
      ArrayAttr:$allocation_owners, DenseI64ArrayAttr:$slots,
      DenseI64ArrayAttr:$generations, I64Attr:$generation_count,
      ArrayAttr:$notify_relations, ArrayAttr:$wait_relations,
      DenseI64ArrayAttr:$initial_counts,
      OptionalAttr<DenseI64ArrayAttr>:$transaction_bytes_by_coordinate);
}

def Shuttle_InternalEventRealizationOp
    : Shuttle_Op<"internal_event_realization",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "Erasure or allocated realization of one internal Event";
  let arguments = (ins I64Attr:$region, I64Attr:$event, StrAttr:$kind,
      StrAttr:$mechanism, DenseI64ArrayAttr:$allocations);
}

def Shuttle_InternalEventNotifyOp
    : Shuttle_Op<"internal_event_notify",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "One exact imperative notification of an internal Event allocation";
  let arguments = (ins I64Attr:$region, I64Attr:$allocation,
      I64Attr:$relation, I64Attr:$task, DenseI64ArrayAttr:$task_coordinate,
      I64Attr:$mechanism_coordinate_ordinal,
      DenseI64ArrayAttr:$mechanism_coordinate,
      DenseI64ArrayAttr:$allocation_owner, I64Attr:$slot,
      I64Attr:$generation, I64Attr:$generation_count,
      I64Attr:$arrival_count, I64Attr:$initial_count, StrAttr:$scope,
      BoolAttr:$release, StrAttr:$legalization,
      OptionalAttr<I64Attr>:$transaction_bytes);
}

def Shuttle_InternalEventWaitOp
    : Shuttle_Op<"internal_event_wait",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>]> {
  let summary = "One exact imperative wait on an internal Event allocation";
  let arguments = (ins I64Attr:$region, I64Attr:$allocation,
      I64Attr:$relation, I64Attr:$task, DenseI64ArrayAttr:$task_coordinate,
      I64Attr:$mechanism_coordinate_ordinal,
      DenseI64ArrayAttr:$mechanism_coordinate,
      DenseI64ArrayAttr:$allocation_owner, I64Attr:$slot,
      I64Attr:$generation, I64Attr:$generation_count,
      I64Attr:$required_count, StrAttr:$scope, BoolAttr:$acquire,
      StrAttr:$legalization);
}

def Shuttle_InternalEventPlanYieldOp
    : Shuttle_Op<"internal_event_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::InternalEventPlanOp"]>,
                  Terminator]> {
  let summary = "Terminate a refined internal Event plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_DeviceModuleOp
    : Shuttle_Op<"device_module", [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<
                             "::mlir::shuttle::DeviceModuleYieldOp">]> {
  let summary = "Closed generated device code and entrypoint bindings";
  let arguments = (ins I64Attr:$schema_version,
      Shuttle_ExecutableCodeFormatAttr:$code_format, Shuttle_PolicyAttr:$policy,
      StrAttr:$source_schedule_fingerprint, DenseI8ArrayAttr:$code,
      StrAttr:$code_digest, StrAttr:$fingerprint,
      OptionalAttr<DenseI8ArrayAttr>:$native_transport,
      OptionalAttr<StrAttr>:$physical_request_fingerprint,
      OptionalAttr<StrAttr>:$physical_compile_request_fingerprint,
      OptionalAttr<StrAttr>:$physical_launch_recipe_fingerprint,
      OptionalAttr<ArrayAttr>:$trusted_source_event_certificates,
      OptionalAttr<ArrayAttr>:$trusted_internal_event_certificates,
      OptionalAttr<ArrayAttr>:$device_ir_snapshot_digests);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_DeviceEntryOp
    : Shuttle_Op<"device_entry",
                 [ParentOneOf<["::mlir::shuttle::DeviceModuleOp"]>]> {
  let summary = "One bytecode entry bound to a scheduled task";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_task,
      I64Attr:$code_offset, I64Attr:$code_length,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      ArrayAttr:$input_accesses, ArrayAttr:$output_accesses,
      DenseI64ArrayAttr:$dependencies,
      Shuttle_ExecutablePredicationAttr:$predication,
      OptionalAttr<Shuttle_ScheduleReductionOrderAttr>:$reduction_order,
      StrAttr:$code_digest,
      OptionalAttr<DenseI64ArrayAttr>:$grid,
      OptionalAttr<DenseI64ArrayAttr>:$block,
      OptionalAttr<I64Attr>:$dynamic_shared_bytes,
      OptionalAttr<I64Attr>:$kernel_arity,
      OptionalAttr<StrAttr>:$online_state_protocol,
      OptionalAttr<DenseI64ArrayAttr>:$online_vector_dimensions,
      OptionalAttr<ArrayAttr>:$logical_coordinate_maps);
}

def Shuttle_DeviceModuleYieldOp
    : Shuttle_Op<"device_module_yield",
                 [ParentOneOf<["::mlir::shuttle::DeviceModuleOp"]>,
                  Terminator]> {
  let summary = "Terminate a generated device module";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_InvocationAbiOp
    : Shuttle_Op<"invocation_abi", [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<
                             "::mlir::shuttle::InvocationAbiYieldOp">]> {
  let summary = "Closed static buffer ABI for a local executable consumer";
  let arguments = (ins I64Attr:$schema_version, StrAttr:$source_plan_fingerprint,
      StrAttr:$source_schedule_fingerprint, StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_InvocationSlotOp
    : Shuttle_Op<"invocation_slot",
                 [ParentOneOf<["::mlir::shuttle::InvocationAbiOp"]>]> {
  let summary = "One statically bound runtime byte buffer";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_buffer,
      TypeAttr:$tensor_type, I64Attr:$required_bytes,
      DenseI64ArrayAttr:$strides, I64Attr:$offset, I64Attr:$alignment,
      Shuttle_ExecutableAddressSpaceAttr:$address_space,
      Shuttle_ExecutableAccessAttr:$access,
      Shuttle_MaterializationStorageAttr:$storage, I64Attr:$alias_group,
      I64Attr:$reuse_group, Shuttle_ExecutableBindingKindAttr:$binding,
      OptionalAttr<I64Attr>:$binding_index);
}

def Shuttle_InvocationAbiYieldOp
    : Shuttle_Op<"invocation_abi_yield",
                 [ParentOneOf<["::mlir::shuttle::InvocationAbiOp"]>,
                  Terminator]> {
  let summary = "Terminate a static invocation ABI";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_ExecutableBundleOp : Shuttle_Op<"executable_bundle"> {
  let summary = "Closed root binding code, ABI, schedule, and completion";
  let arguments = (ins I64Attr:$schema_version,
      StrAttr:$source_schedule_fingerprint, StrAttr:$device_module_fingerprint,
      StrAttr:$invocation_abi_fingerprint,
      OptionalAttr<StrAttr>:$event_lowering_plan_fingerprint,
      Shuttle_ExecutableCompletionAttr:$completion, StrAttr:$fingerprint);
  let hasVerifier = 1;
}

#endif // SHUTTLE_IR_SHUTTLEOPS_TD_
