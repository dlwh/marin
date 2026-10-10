# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research compiler for Transformer tile-lifetime execution plans."""

from tile_lifetime.attention import (
    AttentionPartial,
    NormalizedAttentionPartial,
    finalize_attention_partial,
    finalize_normalized_attention_partial,
    merge_attention_partials,
    merge_normalized_attention_partials,
    normalize_attention_partial,
    summarize_attention_partial,
)
from tile_lifetime.autodiff import (
    BackwardTensorProgram,
    DifferentiatedTensorProgram,
    differentiate_scalar_expression,
    differentiate_tensor_program,
    extract_backward_tensor_program,
    scalar_expression_vjp,
)
from tile_lifetime.compiler import (
    compile_erased_dense_program,
)
from tile_lifetime.contract_map_chain import (
    BoundCastScalarMap,
    ContractMapChainValue,
    RankTwoContractShape,
    TwoContractMapForwardResult,
    TwoContractMapReverseResult,
    TwoContractMapTrainingProgram,
    execute_two_contract_map_forward,
    execute_two_contract_map_reverse,
    form_two_contract_map_training_program,
)
from tile_lifetime.cuda_contract_map_chain_codegen import (
    ContractMapChainSourceAudit,
    GeneratedCudaContractMapChainFfi,
    audit_cuda_contract_map_chain_source,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    GeneratedCudaNormalizedExpContractForwardFfi,
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    GeneratedCudaNormalizedExpContractReverseFfi,
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.cuda_prepared_contract_codegen import (
    GeneratedCudaPreparedContract,
    PreparedContractOperand,
    PreparedContractOperandDelivery,
    PreparedContractSourceAudit,
    audit_cuda_prepared_contract_source,
    generate_cuda_prepared_contract,
)
from tile_lifetime.dense_algebra import DenseSemanticErasureError, erase_dense_semantics
from tile_lifetime.dense_flow import (
    DenseFlowOperation,
    ErasedDenseFlowProgram,
    FlowContract,
    FlowDomainRestriction,
    FlowFold,
    FlowMap,
    FlowMapIteration,
    FlowValue,
    dense_flow_scheduling_keys,
    erase_dense_transformer_semantics,
    pairwise_product_expression,
    pairwise_silu_product_expression,
    validate_erased_dense_flow,
)
from tile_lifetime.expert_parallel import (
    ExpertParallelConfig,
)
from tile_lifetime.expert_parallel_plan import (
    ExpertMaterializationSchedule,
    ExpertOverlapPolicy,
    ExpertParallelPlan,
    ExpertParallelStageKind,
    GateUpPhysicalLayout,
    MapFoldSemantics,
    ReadinessGranularity,
    TileStorage,
    TransportSelection,
    TransportSemantics,
)
from tile_lifetime.expert_parallel_training import (
    ExpertParallelTrainingPlan,
    ExpertParallelTrainingStage,
    ExpertParallelTrainingStageKind,
    derive_expert_parallel_training_plan,
)
from tile_lifetime.expert_parallel_training_runtime import (
    BackwardBufferContract,
    DistributedExpertBackwardABI,
    DistributedExpertBackwardResult,
    ExpertBackwardRankABI,
    derive_distributed_expert_backward_abi,
    verify_distributed_expert_backward_abi,
)
from tile_lifetime.ffi_command_buffer import DirectLaunchFfiPhysicalCandidate
from tile_lifetime.fold_map_contract_training_provider import (
    AxisFoldRegionBinding,
    AxisFoldStageBinding,
    AxisFoldValueBinding,
    FoldMapContractTrainingProvider,
    provide_fold_map_contract_training,
)
from tile_lifetime.fold_placement import (
    FoldAttachment,
    FoldAttachmentSite,
    FoldResultDisposition,
    OwnerTileAvailability,
    attach_fold_to_owner_preparation,
    verify_owner_preparation_fold_attachment,
)
from tile_lifetime.gemm_program import (
    GENERIC_H100_GEMM_BACKEND,
    GemmProgram,
    compile_gemm_program,
)
from tile_lifetime.h100_normalized_exp_backward_lowering import (
    H100NormalizedExpBackwardLowering,
    lower_h100_normalized_exp_backward,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90BackwardAccumulatorSchedule,
    Sm90ContractSchedule,
    Sm90ForwardPipelineSchedule,
    lower_normalized_exp_score_map,
)
from tile_lifetime.jax_collective_transport import (
    JaxCollectiveExecutionPlan,
    build_jax_collective_execution_plan,
    execute_jax_collective_completion,
)
from tile_lifetime.jax_contract_map_chain_ffi import (
    call_cuda_contract_map_chain_forward_ffi,
    call_cuda_contract_map_chain_reverse_ffi,
    register_cuda_contract_map_chain_ffi,
)
from tile_lifetime.jax_fold_map_contract_training import (
    execute_fold_map_contract_training_jax,
    execute_jax_tensor_program_values,
)
from tile_lifetime.jax_normalized_exp_contract_forward_ffi import (
    call_cuda_normalized_exp_contract_forward_ffi,
    register_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.jax_normalized_exp_contract_reverse_ffi import (
    call_cuda_normalized_exp_contract_reverse_ffi,
    register_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.jax_sm90_function import (
    CompiledJaxSm90Function,
    CompleteRequestArtifact,
    CompleteRequestArtifactCompiler,
    JaxSm90Toolchain,
    PreparedJaxSm90Function,
    compile_prepared_jax_sm90_function,
    prepare_jax_sm90_function,
)
from tile_lifetime.linear_pair_map import (
    LinearPairMapTrainingProgram,
    PairMapSavePolicy,
    PairMapVjpProgram,
    build_linear_pair_map_program,
    compile_linear_pair_map_training,
    pair_silu_product_expression,
    pair_tanh_product_expression,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    ForwardOwnershipPlan,
    NormalizedExpContractionDimensions,
    NormalizedExpContractProgram,
    NormalizedExpContractPullback,
    NormalizedExpContractPullbackProgram,
    NormalizedExpContractRegion,
    NormalizedExpFoldState,
    ReverseContractFoldOrder,
    ReverseOwnershipPlan,
    TerminalGradientContract,
    derive_forward_ownership_plan,
    derive_normalized_exp_contract_pullback,
    derive_reverse_ownership_plan,
    normalized_exp_contract_pullback_program,
    normalized_exp_contract_region,
    normalized_exp_finalizer_scale,
    normalized_exp_post_finalizer_expression,
    relation_group_size,
    verify_normalized_exp_finalizer_vjp,
    verify_normalized_exp_score_map_vjp,
)
from tile_lifetime.normalized_exp_contract_training import (
    IndexedFoldSelection,
    NormalizedExpContractTrainingExecution,
    NormalizedExpContractTrainingProgram,
    build_normalized_exp_contract_training_program,
    execute_normalized_exp_contract_training,
    tanh_soft_cap_score_expression,
)
from tile_lifetime.physical_artifact_compiler import (
    GpuPhysicalArtifactCompiler,
    GpuPhysicalArtifactRequest,
    ShuttleBlockRecord,
    ShuttleExactEventRequirement,
    ShuttleOperationRecord,
    ShuttleTypeRecord,
    ShuttleValueRecord,
    decode_gpu_physical_artifact_request,
    register_gpu_physical_artifact_compiler,
)
from tile_lifetime.physical_normalized_exp_contract import (
    NormalizedExpContractGeometry,
    PhysicalNormalizedExpContractRegion,
    SourceBoundBuffer,
    physical_normalized_exp_contract_region,
    source_bound_buffers,
)
from tile_lifetime.plan import (
    AttachmentSite,
    ChunkSummaryRepresentation,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    ScanNumericalContract,
    SemanticErasureReport,
    SemanticLoweringStep,
    StatefulScanExecutionForm,
    StatefulScanSkeleton,
    StateTransitionStructure,
    TransformSkeleton,
)
from tile_lifetime.relation import (
    OrderedRelationFoldProgram,
    RelationPlacement,
    RelationPlan,
    RelationPlanError,
    RelationProjectionAxis,
    RelationProjectionSpec,
    RelationRowProjection,
    build_expert_parallel_relation_plan,
    build_partitioned_merge_rows,
    build_relation_plan,
    compile_ordered_relation_fold,
    project_relation_rows,
    uniform_relation_placement,
)
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
    RelationSelectionProgram,
    SelectionOutputOrder,
    SelectionResult,
    SelectionSemantics,
    SelectionTieBreak,
    UnderfilledSelectionPolicy,
    build_grouped_routed_attention_relation,
    build_routed_attention_relation,
    execute_kv_major_attention,
    execute_projected_block_selection,
    execute_query_major_attention,
    execute_relation_selection,
    make_causal_block_relation,
)
from tile_lifetime.routed_attention_plan import (
    BoundedKVReusePlan,
    BoundedKVReuseWave,
    QueryMajorBlockIndexPlan,
    RoutedAttentionOrientation,
    RoutedAttentionPhysicalPlan,
    RoutedAttentionPlanConfig,
    RoutedStreamingAttentionCompilation,
    bounded_kv_reuse_plan,
    compile_bounded_kv_major_candidate,
    compile_routed_attention_candidates,
    compile_routed_streaming_attention_candidates,
    query_major_block_index_plan,
)
from tile_lifetime.row_normalization_training import (
    FoldMapContractRegion,
    FoldMapContractTrainingPlan,
    GeneratedContractSkeleton,
    GeneratedFoldSkeleton,
    GeneratedMapSkeleton,
    GeneratedRelationSkeleton,
    ReverseMaterializationPolicy,
    build_one_fold_contract_fixture,
    build_two_fold_contract_fixture,
    compile_fold_map_contract_training,
    derive_fold_map_contract_region,
)
from tile_lifetime.semantic_erasure import (
    ErasedTensorProgram,
    SemanticErasureError,
    build_tensor_erasure_report,
    semantic_erasure_errors,
    tensor_program_scheduling_keys,
    validate_erased_tensor_program,
)
from tile_lifetime.shared_reverse_fusion import (
    OwnerComputeComponent,
    OwnerComputeTraversal,
    SharedReverseFusionDisposition,
    SharedReverseFusionPlan,
    plan_shared_producer_reverse_fusion,
)
from tile_lifetime.sm100_routed_lowering import (
    SM100RelationOrientation,
    SM100RoutedSchedule,
    SM100RoutedStreamingLowering,
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    SM100SelectionSchedule,
    SM100SelectionStrategy,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)
from tile_lifetime.stateful_scan import (
    AffineChunkSummary,
    AffineStateTransform,
    ChunkAlgebra,
    LogicalAxis,
    ScanPrimitive,
    ScanPrimitiveKind,
    ScanValue,
    ScanValueRole,
    StatefulScan,
    TensorExpression,
    TensorExpressionKind,
    apply_affine_transform,
    binary_expression,
    compose_affine_transforms,
    contract_expression,
    explain_stateful_scan,
    input_expression,
    summarize_affine_sequence,
    unary_expression,
)
from tile_lifetime.stateful_scan_planner import compile_affine_scan_candidates
from tile_lifetime.stateful_scan_recovery import (
    AffineTensorExpression,
    AppliedLinearMap,
    FactoredAffineChunkSummary,
    RecoveredAffineStateUpdate,
    StateLinearTerm,
    apply_factored_affine_chunk,
    execute_recurrent_factored_affine,
    recover_affine_state_update,
    solve_factored_affine_chunk,
    summarize_factored_affine_chunk,
)
from tile_lifetime.streaming_attention import (
    AttentionScoreAxis,
    ScoreInputSpec,
    ScoreMapSpec,
    StreamingAttentionExecution,
    StreamingAttentionStage,
    StreamingTileSchedule,
    add_score_bias,
    apply_arbitrary_score_mask,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    execute_streaming_attention,
    execute_streaming_attention_with_state,
    execute_tensor_program,
    scaled_score_map,
    streaming_attention_from_semantic_operation,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardExecution,
    StreamingAttentionBackwardFoldOrder,
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardProvenance,
    StreamingAttentionBackwardReassociation,
    StreamingAttentionBackwardStage,
    StreamingAttentionBackwardTileSchedule,
    StreamingAttentionBackwardWorkEstimate,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_fusion_plan,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    execute_streaming_attention_backward,
    verify_streaming_attention_backward_score_map_vjp,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
)
from tile_lifetime.tile_program import (
    TileOp,
    TilePrimitive,
    TileProgram,
    TileProgramError,
    TileProgramStage,
    ValueLifetime,
    optimize_tile_program,
)
from tile_lifetime.tiled_fold_finalize import (
    FoldDenominatorPolicy,
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPartialOrder,
    FoldPhysicalAxis,
    FoldReassociationPolicy,
    FoldScalarReduction,
    TiledFoldAxes,
    TiledFoldFinalizeProgram,
    TiledFoldFinalizeSchedule,
    TiledFoldFinalizeSemantics,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
    evaluate_tiled_fold_finalize,
    normalized_exponential_fold_program,
)
