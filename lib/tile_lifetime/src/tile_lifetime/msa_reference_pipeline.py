# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable generic-algebra reference for projected sparse selection."""

from dataclasses import dataclass

import numpy as np

from shuttle.ir import DType
from tile_lifetime.event_buffering import physical_event_realization, verify_event_realizations
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    EventTensorRuntimeInputs,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    verify_event_dataflow_program,
)
from tile_lifetime.msa_frontend import MSA_INPUT_NAMES, MSADebugConfig
from tile_lifetime.physical_task_contract import (
    BoundedTaskCoordinate,
    CanonicalTaskPlan,
    DependenceRuntimeBinding,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    ParametricEventDerivation,
    ParametricEventTensorPlan,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskDependenceProgram,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccess,
    TaskValueAccessKind,
    TiledTask,
    physical_phase_iteration_space,
)
from tile_lifetime.relation import RelationPlan
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
    SelectionResult,
    build_grouped_routed_attention_relation,
    execute_projected_block_selection,
)
from tile_lifetime.streaming_attention import (
    apply_arbitrary_score_mask,
    build_attention_tensor_program,
    scaled_score_map,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    SelectionPrimitive,
    TensorAxis,
    TensorProgram,
    evaluate_scalar_expression,
    index_axis,
    index_binary,
    index_compare,
    index_constant,
    index_tensor_element,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_select,
    scalar_unary,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class MSAReferenceOracle:
    """Independent target adapter used to validate the canonical algebra path."""

    projected_selection: ProjectedBlockSelectionProgram
    gathered_normalized_exp: TensorProgram
    canonical_program: TensorProgram

    @property
    def operation_kinds(self) -> tuple[str, ...]:
        """Expose the generic algebra sequence consumed by planning."""
        return tuple(
            type(operation).__name__.removesuffix("Primitive") for operation in self.canonical_program.operations
        )


@dataclass(frozen=True)
class MSAReferenceExecution:
    """Observable result of executing the generic algebra reference."""

    output: np.ndarray
    selection: SelectionResult
    relation: RelationPlan


@dataclass(frozen=True)
class MSAReferenceDataflow:
    """Target adapter around generic tasks and parametric exact Event plans."""

    program: EventDataflowProgram
    event_runtime_inputs: tuple[EventTensorRuntimeInputs, ...]
    task_plan: CanonicalTaskPlan
    event_allocations: tuple[EventTensorAllocation, ...]
    event_programs: tuple[ParametricEventTensorPlan, ...]
    numerical_policy: PhysicalNumericalPolicy

    @property
    def event_storage_capacity(self) -> int:
        """Number of statically bounded event slots in one invocation."""
        return sum(len(inputs.event_storage_slots) for inputs in self.event_runtime_inputs)


def build_msa_reference_oracle(config: MSADebugConfig) -> MSAReferenceOracle:
    """Build an executable oracle from existing projected-selection pieces."""
    projected_selection = ProjectedBlockSelectionProgram(
        source_input="query_hidden",
        right_source_input="key_value_hidden",
        left_weight_input="left_index_weight",
        right_weight_input="right_index_weight",
        source_count=config.query_length,
        source_feature_count=config.hidden_dimension,
        right_count=config.key_value_length,
        right_source_feature_count=config.hidden_dimension,
        group_count=config.key_value_heads,
        relation_feature_count=config.index_dimension,
        right_block_size=config.block_size,
        selected_count=config.selected_blocks,
        score_scale=config.index_scale,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=True,
        projection_output_dtype="bf16",
        left_position_offset=config.query_position_offset,
    )
    gathered_normalized_exp = build_attention_tensor_program(
        batch_size=1,
        query_length=config.query_length,
        key_length=config.selected_blocks * config.block_size,
        query_heads=config.query_heads,
        key_value_heads=config.key_value_heads,
        key_dimension=config.head_dimension,
        value_dimension=config.head_dimension,
        score_map=apply_arbitrary_score_mask(scaled_score_map(config.attention_scale)),
        input_dtype=DType.BF16,
    )
    return MSAReferenceOracle(
        projected_selection=projected_selection,
        gathered_normalized_exp=gathered_normalized_exp,
        canonical_program=_build_msa_tensor_program(config),
    )


def execute_msa_reference_oracle(
    oracle: MSAReferenceOracle,
    inputs: tuple[np.ndarray, ...],
) -> MSAReferenceExecution:
    """Execute projected Selection, Relation gathers, and normalized-exp Contracts."""
    if len(inputs) != len(MSA_INPUT_NAMES):
        raise ValueError(f"expected {len(MSA_INPUT_NAMES)} inputs, got {len(inputs)}")
    named_inputs = dict(zip(MSA_INPUT_NAMES, inputs, strict=True))
    selection = execute_projected_block_selection(oracle.projected_selection, named_inputs)
    relation = build_grouped_routed_attention_relation(selection.indices, edge_valid=selection.valid)
    output = _execute_selected_normalized_exp(oracle, named_inputs, selection)
    return MSAReferenceExecution(output=output, selection=selection, relation=relation)


def derive_msa_reference_dataflow(
    oracle: MSAReferenceOracle,
    *,
    numerical_policy: PhysicalNumericalPolicy = PhysicalNumericalPolicy.SOURCE_ORDERED,
) -> MSAReferenceDataflow:
    """Derive tiled ownership, exact dependencies, and bounded Events."""
    semantic = oracle.projected_selection
    query_heads = oracle.gathered_normalized_exp.inputs[0].shape[2]
    heads_per_group = query_heads // semantic.group_count
    query = TaskAxis("query", semantic.source_count)
    group = TaskAxis("group", semantic.group_count)
    block = TaskAxis("block", semantic.right_block_count)
    slot = TaskAxis("selected_slot", semantic.selected_count)
    head = TaskAxis("head", query_heads)
    local_head = TaskAxis("local_head", heads_per_group)

    query_projection = TaskFamily("query_projection", (query, head), placement="matrix_workers")
    key_projection = TaskFamily("key_projection", (block, group), placement="matrix_workers")
    value_projection = TaskFamily("value_projection", (block, group), placement="matrix_workers")
    left_index_projection = TaskFamily("left_index_projection", (query, group), placement="matrix_workers")
    right_index_projection = TaskFamily("right_index_projection", (block,), placement="matrix_workers")
    relation_score_contract = TaskFamily("relation_score_contract", (query, group, block), placement="matrix_workers")
    block_max_fold = TaskFamily("block_max_fold", (query, group, block), placement="reduction_workers")
    bounded_selection = TaskFamily("bounded_selection", (query, group), placement="reduction_workers")
    relation_apply = TaskFamily("relation_apply", (query, group, slot), placement="index_workers")
    score_contract = TaskFamily("score_contract", (query, group, local_head, slot), placement="matrix_workers")
    normalized_exp_fold = TaskFamily("normalized_exp_fold", (query, group, local_head), placement="reduction_workers")
    weighted_contract = TaskFamily("weighted_contract", (query, group, local_head, slot), placement="matrix_workers")
    finalizer = TaskFamily("finalizer", (query, group, local_head), placement="reduction_workers")
    families = (
        query_projection,
        key_projection,
        value_projection,
        left_index_projection,
        right_index_projection,
        relation_score_contract,
        block_max_fold,
        bounded_selection,
        relation_apply,
        score_contract,
        normalized_exp_fold,
        weighted_contract,
        finalizer,
    )
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    dependences: list[TaskDependence] = []

    def add(source: TaskFamily, target: TaskFamily, pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]):
        dependences.append(TaskDependence(TaskRelation.from_pairs(source, target, pairs), visibility))

    query_group_block = tuple(
        (query_index, group_index, block_index)
        for query_index in range(semantic.source_count)
        for group_index in range(semantic.group_count)
        for block_index in range(semantic.right_block_count)
    )
    add(
        left_index_projection,
        relation_score_contract,
        tuple(((q, g), (q, g, b)) for q, g, b in query_group_block),
    )
    add(
        right_index_projection,
        relation_score_contract,
        tuple(((b,), (q, g, b)) for q, g, b in query_group_block),
    )
    add(
        relation_score_contract,
        block_max_fold,
        tuple(((q, g, b), (q, g, b)) for q, g, b in query_group_block),
    )
    add(
        block_max_fold,
        bounded_selection,
        tuple(((q, g, b), (q, g)) for q, g, b in query_group_block),
    )
    query_group_slot = tuple(
        (query_index, group_index, slot_index)
        for query_index in range(semantic.source_count)
        for group_index in range(semantic.group_count)
        for slot_index in range(semantic.selected_count)
    )
    add(
        bounded_selection,
        relation_apply,
        tuple(((q, g), (q, g, s)) for q, g, s in query_group_slot),
    )
    score_coordinates = tuple(
        (query_index, group_index, local_head_index, slot_index)
        for query_index in range(semantic.source_count)
        for group_index in range(semantic.group_count)
        for local_head_index in range(heads_per_group)
        for slot_index in range(semantic.selected_count)
    )
    add(
        query_projection,
        score_contract,
        tuple(((q, g * heads_per_group + lh), (q, g, lh, s)) for q, g, lh, s in score_coordinates),
    )
    add(
        relation_apply,
        score_contract,
        tuple(((q, g, s), (q, g, lh, s)) for q, g, lh, s in score_coordinates),
    )
    add(
        score_contract,
        normalized_exp_fold,
        tuple(((q, g, lh, s), (q, g, lh)) for q, g, lh, s in score_coordinates),
    )
    add(
        relation_apply,
        weighted_contract,
        tuple(((q, g, s), (q, g, lh, s)) for q, g, lh, s in score_coordinates),
    )
    add(
        score_contract,
        weighted_contract,
        tuple(((q, g, lh, s), (q, g, lh, s)) for q, g, lh, s in score_coordinates),
    )
    add(
        normalized_exp_fold,
        weighted_contract,
        tuple(((q, g, lh), (q, g, lh, s)) for q, g, lh, s in score_coordinates),
    )
    add(
        weighted_contract,
        finalizer,
        tuple(((q, g, lh, s), (q, g, lh)) for q, g, lh, s in score_coordinates),
    )

    plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
        )
        for dependence in dependences
    )
    program = EventDataflowProgram(families, tuple(dependences), plans)
    verify_event_dataflow_program(program)
    runtime_inputs = []
    event_allocations = []
    for plan in plans:
        event_count = len(plan.domain.coordinates)
        binding = EventStorageBinding(
            plan.domain,
            tuple(range(event_count)),
            (0,) * event_count,
        )
        runtime_inputs.append(event_tensor_runtime_inputs(plan, storage_binding=binding))
        event_allocations.append(EventTensorAllocation(binding, event_count, 1))
    task_plan = _msa_task_plan(
        oracle.canonical_program,
        families,
        tuple(dependences),
        numerical_policy=numerical_policy,
    )
    dependence_programs = _routed_dependence_programs(oracle.canonical_program, task_plan)
    task_plan = CanonicalTaskPlan(task_plan.tasks, task_plan.dependences, dependence_programs=dependence_programs)
    event_programs = []
    for index, dependence_program in enumerate(dependence_programs):
        target = task_plan.tasks[dependence_program.target_task_id].family
        event_count = int(np.prod([axis.extent for axis in target.axes]))
        event_programs.append(
            ParametricEventTensorPlan(
                dependence_program_index=index,
                derivation=ParametricEventDerivation.EXACT_CONSUMER,
                capacity=event_count,
                slots=tuple(range(event_count)),
                generations=(0,) * event_count,
                generation_count=1,
                memory_scope=EventMemoryScope.DEVICE,
                generation_policy=EventGenerationPolicy.PER_INVOCATION,
                scheduling_mode=EventSchedulingMode.DYNAMIC,
            )
        )
    return MSAReferenceDataflow(
        program=program,
        event_runtime_inputs=tuple(runtime_inputs),
        task_plan=task_plan,
        event_allocations=tuple(event_allocations),
        event_programs=tuple(event_programs),
        numerical_policy=numerical_policy,
    )


def msa_reference_physical_request(
    oracle: MSAReferenceOracle,
    dataflow: MSAReferenceDataflow,
    *,
    numerical_policy: PhysicalNumericalPolicy = PhysicalNumericalPolicy.SOURCE_ORDERED,
) -> ExecutablePhysicalRequest:
    """Build the canonical generic request handed to shared physical lowering."""
    if dataflow.numerical_policy is not numerical_policy:
        raise ValueError(
            "MSA physical request policy does not match its derived task plan: "
            f"request={numerical_policy.value}, tasks={dataflow.numerical_policy.value}"
        )
    schedules = tuple(_reference_schedule(task) for task in dataflow.task_plan.tasks)
    realizations = tuple(
        physical_event_realization(
            plan,
            mechanism="generation_counted_event",
            reason="the reference multi-task schedule does not prove a stronger execution order",
        )
        for plan in dataflow.program.event_plans
    )
    audit = verify_event_realizations(dataflow.program, realizations)
    return ExecutablePhysicalRequest(
        algebra_record=oracle.canonical_program.canonical_record,
        algebra_fingerprint=oracle.canonical_program.semantic_fingerprint,
        task_plan=dataflow.task_plan,
        event_plans=dataflow.program.event_plans,
        event_allocations=dataflow.event_allocations,
        target=PhysicalTarget.SM100,
        numerical_policy=numerical_policy,
        schedules=schedules,
        event_realizations=EventRealizationProof(audit, ()),
        provenance={
            "source_boundary": "ordinary_jax",
            "automatic_differentiation": "jax",
            "source_numerical_policy": NumericalPolicy.SOURCE_ORDERED.value,
        },
        event_programs=dataflow.event_programs,
    )


def _execute_selected_normalized_exp(
    oracle: MSAReferenceOracle,
    inputs: dict[str, np.ndarray],
    selection: SelectionResult,
) -> np.ndarray:
    program = oracle.projected_selection
    query_heads = oracle.gathered_normalized_exp.inputs[0].shape[2]
    head_dimension = oracle.gathered_normalized_exp.inputs[0].shape[3]
    heads_per_group = query_heads // program.group_count

    query_hidden = np.asarray(inputs["query_hidden"], dtype=np.float32)
    key_value_hidden = np.asarray(inputs["key_value_hidden"], dtype=np.float32)
    query = _bf16_round(query_hidden @ np.asarray(inputs["query_weight"], dtype=np.float32)).reshape(
        program.source_count,
        query_heads,
        head_dimension,
    )
    key = _bf16_round(key_value_hidden @ np.asarray(inputs["key_weight"], dtype=np.float32)).reshape(
        program.resolved_right_count,
        program.group_count,
        head_dimension,
    )
    value = _bf16_round(key_value_hidden @ np.asarray(inputs["value_weight"], dtype=np.float32)).reshape(
        program.resolved_right_count,
        program.group_count,
        head_dimension,
    )

    output = np.empty((program.source_count, query_heads, head_dimension), dtype=np.float32)
    for query_index in range(program.source_count):
        query_position = program.left_position_offset + query_index
        for group in range(program.group_count):
            valid_blocks = selection.indices[query_index, group, selection.valid[query_index, group]]
            key_positions = (
                valid_blocks[:, None] * program.right_block_size
                + np.arange(program.right_block_size, dtype=np.int32)[None, :]
            ).reshape(-1)
            key_positions = key_positions[key_positions + program.right_position_offset <= query_position]
            for local_head in range(heads_per_group):
                query_head = group * heads_per_group + local_head
                scores = query[query_index, query_head] @ key[key_positions, group].T
                score_map = oracle.gathered_normalized_exp.operations[1]
                mapped_inputs = {
                    value.name: scores.astype(np.float32) if value.name == "score.raw" else True
                    for value in score_map.inputs
                }
                scores = np.asarray(
                    evaluate_scalar_expression(score_map.expression, mapped_inputs),
                    dtype=np.float32,
                )
                weights = np.exp(scores - np.max(scores))
                weights /= np.sum(weights)
                output[query_index, query_head] = weights @ value[key_positions, group]
    return _bf16_round(output)


def _msa_task_plan(
    program: TensorProgram,
    families: tuple[TaskFamily, ...],
    dependences: tuple[TaskDependence, ...],
    *,
    numerical_policy: PhysicalNumericalPolicy,
) -> CanonicalTaskPlan:
    family_by_name = {family.name: family for family in families}
    axes = {axis.label: axis for value in program.inputs for axis in value.axes if axis.label is not None}
    for operation in program.operations:
        for value in primitive_outputs(operation):
            axes.update({axis.label: axis for axis in value.axes if axis.label is not None})

    def tile(label: str, tile_extent: int | None = None) -> TaskAxisTile:
        axis = axes[label]
        return TaskAxisTile(axis.id, axis.extent, axis.extent if tile_extent is None else tile_extent)

    specifications = (
        (
            "query_projection",
            (0, 4, 5),
            TaskOperationKind.CONTRACT,
            (tile("query", 1), tile("query_head", 1)),
            (),
            (),
            (tile("hidden"),),
        ),
        (
            "key_projection",
            (1, 6, 7),
            TaskOperationKind.CONTRACT,
            (tile("key", 4), tile("group", 1)),
            (),
            (),
            (tile("hidden"),),
        ),
        (
            "value_projection",
            (2, 8, 9),
            TaskOperationKind.CONTRACT,
            (tile("key", 4), tile("group", 1)),
            (),
            (),
            (tile("hidden"),),
        ),
        (
            "left_index_projection",
            (3, 10, 12, 14),
            TaskOperationKind.CONTRACT,
            (tile("query", 1), tile("group", 1)),
            (),
            (),
            (tile("hidden"),),
        ),
        (
            "right_index_projection",
            (11, 13, 15),
            TaskOperationKind.CONTRACT,
            (tile("key", 4),),
            (),
            (),
            (tile("hidden"),),
        ),
        (
            "relation_score_contract",
            (16, 17, 18, 19),
            TaskOperationKind.CONTRACT,
            (tile("query", 1), tile("group", 1)),
            (tile("block", 1),),
            (),
            (tile("index_feature"),),
        ),
        (
            "block_max_fold",
            (20,),
            TaskOperationKind.FOLD,
            (tile("query", 1), tile("group", 1), tile("block", 1)),
            (),
            (tile("block_token"),),
            (),
        ),
        (
            "bounded_selection",
            (21, 22, 23, 24),
            TaskOperationKind.SELECTION,
            (tile("query", 1), tile("group", 1), tile("selected_slot", 1)),
            (),
            (tile("block"),),
            (),
        ),
        (
            "relation_apply",
            (25, 26),
            TaskOperationKind.RELATION,
            (tile("query", 1), tile("group", 1), tile("selected_slot", 1)),
            (),
            (),
            (),
        ),
        (
            "score_contract",
            (27, 28, 29, 30),
            TaskOperationKind.CONTRACT,
            (tile("query", 1), tile("group", 1), tile("local_head", 1)),
            (tile("selected_slot", 1),),
            (),
            (tile("feature"),),
        ),
        (
            "normalized_exp_fold",
            (31, 32, 33, 34),
            TaskOperationKind.FOLD,
            (tile("query", 1), tile("group", 1), tile("local_head", 1)),
            (tile("selected_slot", 1),),
            (tile("block_token"),),
            (),
        ),
        (
            "weighted_contract",
            (35,),
            TaskOperationKind.CONTRACT,
            (tile("query", 1), tile("group", 1), tile("local_head", 1), tile("feature", 1)),
            (tile("selected_slot", 1),),
            (),
            (tile("block_token"),),
        ),
        (
            "finalizer",
            (36, 37, 38),
            TaskOperationKind.MAP,
            (tile("query", 1), tile("group", 1), tile("local_head", 1)),
            (),
            (),
            (),
        ),
    )
    value_ordinals = tensor_program_value_ordinals(program)
    producer = {
        value: ordinal for ordinal, operation in enumerate(program.operations) for value in primitive_outputs(operation)
    }
    task_by_operation = {
        ordinal: task_id for task_id, specification in enumerate(specifications) for ordinal in specification[1]
    }
    tasks = []
    canonical_operations = program.canonical_record["operations"]
    for task_id, (family_name, ordinals, principal, owner, streamed, reduced, contracted) in enumerate(specifications):
        phases = tuple(
            PhysicalPhase(
                index,
                _task_capability(program.operations[ordinal]),
                (ordinal,),
                *physical_phase_iteration_space(program.operations[ordinal]),
            )
            for index, ordinal in enumerate(ordinals)
        )
        phase_edges = tuple(
            PhysicalPhaseEdge(index, index + 1, RelationAccessClass.IDENTITY) for index in range(len(phases) - 1)
        )
        accesses = []
        for phase_id, ordinal in enumerate(ordinals):
            operation = program.operations[ordinal]
            relation_class = (
                RelationAccessClass.BOUNDED_DYNAMIC
                if family_name == "relation_apply"
                else (
                    RelationAccessClass.AFFINE
                    if isinstance(operation, RelationPrimitive)
                    else RelationAccessClass.IDENTITY
                )
            )
            for value in primitive_inputs(operation):
                if producer.get(value) in ordinals:
                    continue
                accesses.append(
                    TaskValueAccess(
                        value_ordinals[value],
                        TaskValueAccessKind.LIVE_IN,
                        phase_id,
                        relation_class,
                        {
                            "algebra_operation": canonical_operations[ordinal],
                            "operand": primitive_inputs(operation).index(value),
                        },
                    )
                )
            for value in primitive_outputs(operation):
                consumers = tuple(
                    other_ordinal
                    for other_ordinal, other in enumerate(program.operations)
                    if value in primitive_inputs(other)
                )
                live_out = (
                    any(task_by_operation.get(consumer) != task_id for consumer in consumers) or value in program.outputs
                )
                accesses.append(
                    TaskValueAccess(
                        value_ordinals[value],
                        TaskValueAccessKind.LIVE_OUT if live_out else TaskValueAccessKind.TEMPORARY,
                        phase_id,
                        relation_class,
                        {
                            "algebra_operation": canonical_operations[ordinal],
                            "result": primitive_outputs(operation).index(value),
                        },
                    )
                )
        element_types = tuple(
            sorted(
                {
                    value.dtype.value
                    for ordinal in ordinals
                    for value in (
                        *primitive_inputs(program.operations[ordinal]),
                        *primitive_outputs(program.operations[ordinal]),
                    )
                }
            )
        )
        tasks.append(
            TiledTask(
                task_id,
                family_by_name[family_name],
                ordinals,
                tuple(_task_operation_kind(program.operations[ordinal]) for ordinal in ordinals),
                PhysicalTaskStructure(
                    principal,
                    phases,
                    phase_edges,
                    owner,
                    streamed,
                    reduced,
                    contracted,
                    (
                        TaskTraversal.SOURCE_ORDERED
                        if numerical_policy is PhysicalNumericalPolicy.SOURCE_ORDERED
                        else TaskTraversal.UNORDERED
                    ),
                    element_types,
                    ("row_major",),
                    TaskResourceEstimate(64, 0, 0),
                ),
                tuple(accesses),
            )
        )
    return CanonicalTaskPlan(tuple(tasks), dependences)


def _routed_dependence_programs(
    algebra: TensorProgram,
    task_plan: CanonicalTaskPlan,
) -> tuple[TaskDependenceProgram, ...]:
    """Build exact K/V producer relations from the runtime Selection results."""
    selection = next(operation for operation in algebra.operations if isinstance(operation, SelectionPrimitive))
    if selection.output_validity is None:
        raise ValueError("routed dependencies require explicit Selection validity")
    selected_indices = selection.indices
    selected_valid = selection.output_validity
    query_count, group_count, slot_count = selected_indices.shape
    score_task = next(task for task in task_plan.tasks if task.family.name == "score_contract")
    heads_per_group = score_task.family.axes[2].extent

    query = TensorAxis(0, query_count)
    group = TensorAxis(1, group_count)
    local_head = TensorAxis(2, heads_per_group)
    slot = TensorAxis(3, slot_count)
    relation_domain = (query, group, local_head, slot)
    route = ProgramValue("selected_index", (query, group, slot), DType.INT32)
    valid = ProgramValue("selected_valid", (query, group, slot), DType.BOOL)
    source_block = ProgramValue("source_block", relation_domain, DType.INT32)
    source_group = ProgramValue("source_group", relation_domain, DType.INT32)
    target_query = ProgramValue("target_query", relation_domain, DType.INT32)
    target_group = ProgramValue("target_group", relation_domain, DType.INT32)
    target_local_head = ProgramValue("target_local_head", relation_domain, DType.INT32)
    target_slot = ProgramValue("target_slot", relation_domain, DType.INT32)
    edge_valid = ProgramValue("edge_valid", relation_domain, DType.BOOL)
    route_indexing = OperandIndexing(
        route,
        relation_domain,
        (index_axis(query), index_axis(group), index_axis(slot)),
    )
    valid_indexing = OperandIndexing(
        valid,
        relation_domain,
        (index_axis(query), index_axis(group), index_axis(slot)),
    )
    relation = RelationPrimitive(
        "selected producer relation",
        (route, valid),
        (source_block, source_group, target_query, target_group, target_local_head, target_slot, edge_valid),
        relation_domain,
        (route_indexing, valid_indexing),
        result_expressions=(
            None,
            index_axis(group),
            index_axis(query),
            index_axis(group),
            index_axis(local_head),
            index_axis(slot),
            None,
        ),
        result_sources=(route, None, None, None, None, None, valid),
        order=EvaluationOrder.SOURCE_ORDERED,
    )
    relation_program = TensorProgram(
        (route, valid),
        (relation,),
        (source_block, source_group, target_query, target_group, target_local_head, target_slot, edge_valid),
        NumericalPolicy.SOURCE_ORDERED,
    )
    value_ordinals = tensor_program_value_ordinals(algebra)
    bindings = (
        DependenceRuntimeBinding(route, value_ordinals[selected_indices]),
        DependenceRuntimeBinding(valid, value_ordinals[selected_valid]),
    )
    task_ids = {task.family.name: task.id for task in task_plan.tasks}
    block_count = next(task for task in task_plan.tasks if task.family.name == "key_projection").family.axes[0].extent

    def dependence(source_name: str, target_name: str) -> TaskDependenceProgram:
        return TaskDependenceProgram(
            task_ids[source_name],
            task_ids[target_name],
            relation_program,
            bindings,
            (
                BoundedTaskCoordinate(source_block, 0, block_count),
                BoundedTaskCoordinate(source_group, 0, group_count),
            ),
            (
                BoundedTaskCoordinate(target_query, 0, query_count),
                BoundedTaskCoordinate(target_group, 0, group_count),
                BoundedTaskCoordinate(target_local_head, 0, heads_per_group),
                BoundedTaskCoordinate(target_slot, 0, slot_count),
            ),
            edge_valid,
            MemoryVisibility(EventMemoryScope.DEVICE),
        )

    return (
        dependence("key_projection", "score_contract"),
        dependence("value_projection", "weighted_contract"),
    )


def _task_operation_kind(operation) -> TaskOperationKind:
    if isinstance(operation, ContractPrimitive):
        return TaskOperationKind.CONTRACT
    if isinstance(operation, FoldPrimitive):
        return TaskOperationKind.FOLD
    if isinstance(operation, MapPrimitive):
        return TaskOperationKind.MAP
    if isinstance(operation, SelectionPrimitive):
        return TaskOperationKind.SELECTION
    assert isinstance(operation, RelationPrimitive)
    return TaskOperationKind.RELATION


def _task_capability(operation) -> TaskCapability:
    return {
        TaskOperationKind.CONTRACT: TaskCapability.MATRIX_CONTRACT,
        TaskOperationKind.FOLD: TaskCapability.ASSOCIATIVE_FOLD,
        TaskOperationKind.MAP: TaskCapability.SCALAR_MAP,
        TaskOperationKind.SELECTION: TaskCapability.BOUNDED_SELECTION,
        TaskOperationKind.RELATION: TaskCapability.INDEX_RELATION,
    }[_task_operation_kind(operation)]


def _reference_schedule(task: TiledTask) -> PhysicalSchedule:
    structure = task.structure
    relation_classes = tuple(sorted({access.relation_class for access in task.accesses}, key=lambda item: item.value))
    if not relation_classes:
        relation_classes = (RelationAccessClass.IDENTITY,)
    constraint = PhysicalTaskConstraint(
        principal_kind=structure.principal_kind,
        required_capabilities=tuple(phase.capability for phase in structure.phases),
        supported_capabilities=tuple(dict.fromkeys(phase.capability for phase in structure.phases)),
        required_edges=tuple(
            PhaseEdgeConstraint(
                structure.phases[edge.source_phase].capability,
                structure.phases[edge.target_phase].capability,
                edge.relation_class,
            )
            for edge in structure.phase_edges
        ),
        owner_rank=len(structure.owner_axes),
        streamed_rank=len(structure.streamed_axes),
        reduction_rank=len(structure.reduction_axes),
        contraction_rank=len(structure.contraction_axes),
        supported_relation_classes=relation_classes,
        supported_element_types=structure.element_types,
        supported_layout_classes=structure.layout_classes,
        maximum_resources=structure.resources,
    )
    capabilities = "_".join(phase.capability.value for phase in structure.phases)
    family = PhysicalScheduleFamily(
        name=f"{structure.principal_kind.value}_{capabilities}",
        target=PhysicalTarget.SM100,
        tasks=(constraint,),
    )
    return PhysicalSchedule(
        family=family,
        task_ids=(task.id,),
        parameters={
            "selection_policy": "algebra_derived",
            "tile_policy": "algebra_derived",
            "traversal": structure.traversal.value,
        },
    )


def _build_msa_tensor_program(config: MSADebugConfig) -> TensorProgram:
    """Express the complete forward as generic tensor algebra."""
    query = TensorAxis(0, config.query_length, "query")
    key = TensorAxis(1, config.key_value_length, "key")
    hidden = TensorAxis(2, config.hidden_dimension, "hidden")
    query_head = TensorAxis(3, config.query_heads, "query_head")
    group = TensorAxis(4, config.key_value_heads, "group")
    feature = TensorAxis(5, config.head_dimension, "feature")
    index_feature = TensorAxis(6, config.index_dimension, "index_feature")
    block = TensorAxis(7, config.block_count, "block")
    block_token = TensorAxis(8, config.block_size, "block_token")
    selected_slot = TensorAxis(9, config.selected_blocks, "selected_slot")
    local_head = TensorAxis(10, config.query_heads // config.key_value_heads, "local_head")
    query_weight_flat = TensorAxis(11, config.query_heads * config.head_dimension, "query_weight_flat")
    group_weight_flat = TensorAxis(12, config.key_value_heads * config.head_dimension, "group_weight_flat")
    index_weight_flat = TensorAxis(13, config.key_value_heads * config.index_dimension, "index_weight_flat")

    query_hidden = ProgramValue("query_hidden", (query, hidden), DType.BF16)
    key_value_hidden = ProgramValue("key_value_hidden", (key, hidden), DType.BF16)
    query_weight = ProgramValue("query_weight", (hidden, query_weight_flat), DType.BF16)
    key_weight = ProgramValue("key_weight", (hidden, group_weight_flat), DType.BF16)
    value_weight = ProgramValue("value_weight", (hidden, group_weight_flat), DType.BF16)
    left_weight = ProgramValue("left_index_weight", (hidden, index_weight_flat), DType.FP32)
    right_weight = ProgramValue("right_index_weight", (hidden, index_feature), DType.FP32)
    inputs = (
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_weight,
        right_weight,
    )
    operations = []

    def reshape_weight(
        source: ProgramValue,
        destination: ProgramValue,
        outer: TensorAxis,
        inner: TensorAxis,
    ) -> None:
        flattened = index_binary(
            IndexExpressionKind.ADD,
            index_binary(IndexExpressionKind.MULTIPLY, index_axis(outer), index_constant(inner.extent)),
            index_axis(inner),
        )
        operations.append(
            RelationPrimitive(
                f"reshape_{source.name}",
                (source,),
                (destination,),
                destination.axes,
                (
                    OperandIndexing(
                        source,
                        destination.axes,
                        (index_axis(hidden), flattened),
                    ),
                ),
            )
        )

    query_weight_shaped = ProgramValue("query_weight_shaped", (hidden, query_head, feature), DType.BF16)
    key_weight_shaped = ProgramValue("key_weight_shaped", (hidden, group, feature), DType.BF16)
    value_weight_shaped = ProgramValue("value_weight_shaped", (hidden, group, feature), DType.BF16)
    left_weight_shaped = ProgramValue("left_weight_shaped", (hidden, group, index_feature), DType.FP32)
    reshape_weight(query_weight, query_weight_shaped, query_head, feature)
    reshape_weight(key_weight, key_weight_shaped, group, feature)
    reshape_weight(value_weight, value_weight_shaped, group, feature)
    reshape_weight(left_weight, left_weight_shaped, group, index_feature)

    projected_values = []
    for name, source, weight, axes in (
        ("query", query_hidden, query_weight_shaped, (query, query_head, feature)),
        ("key", key_value_hidden, key_weight_shaped, (key, group, feature)),
        ("value", key_value_hidden, value_weight_shaped, (key, group, feature)),
    ):
        accumulator = ProgramValue(f"{name}_fp32", axes, DType.FP32)
        projected = ProgramValue(name, axes, DType.BF16)
        operations.append(ContractPrimitive(f"{name}_projection", (source, weight), accumulator, (hidden,), DType.FP32))
        operations.append(
            MapPrimitive(
                f"{name}_convert",
                (accumulator,),
                projected,
                scalar_convert(
                    scalar_input(accumulator.name),
                    result_dtype=DType.BF16,
                    semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
                ),
            )
        )
        projected_values.append(projected)
    projected_query, projected_key, projected_value = projected_values

    query_hidden_fp32 = ProgramValue("query_hidden_fp32", query_hidden.axes, DType.FP32)
    key_value_hidden_fp32 = ProgramValue("key_value_hidden_fp32", key_value_hidden.axes, DType.FP32)
    for source, destination in (
        (query_hidden, query_hidden_fp32),
        (key_value_hidden, key_value_hidden_fp32),
    ):
        operations.append(
            MapPrimitive(
                f"{source.name}_convert",
                (source,),
                destination,
                scalar_convert(
                    scalar_input(source.name),
                    result_dtype=DType.FP32,
                    semantics=ScalarConversionSemantics.EXACT,
                ),
            )
        )
    left_fp32 = ProgramValue("left_index_fp32", (query, group, index_feature), DType.FP32)
    right_fp32 = ProgramValue("right_index_fp32", (key, index_feature), DType.FP32)
    operations.extend(
        (
            ContractPrimitive(
                "left_index_projection",
                (query_hidden_fp32, left_weight_shaped),
                left_fp32,
                (hidden,),
                DType.FP32,
            ),
            ContractPrimitive(
                "right_index_projection",
                (key_value_hidden_fp32, right_weight),
                right_fp32,
                (hidden,),
                DType.FP32,
            ),
        )
    )
    left_index = ProgramValue("left_index", left_fp32.axes, DType.BF16)
    right_index = ProgramValue("right_index", right_fp32.axes, DType.BF16)
    for source, destination in ((left_fp32, left_index), (right_fp32, right_index)):
        operations.append(
            MapPrimitive(
                f"{destination.name}_convert",
                (source,),
                destination,
                scalar_convert(
                    scalar_input(source.name),
                    result_dtype=DType.BF16,
                    semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
                ),
            )
        )

    token_score = ProgramValue("token_score", (query, group, key), DType.FP32)
    operations.append(
        ContractPrimitive(
            "relation_score_contract", (left_index, right_index), token_score, (index_feature,), DType.FP32
        )
    )
    blocked_score = ProgramValue("blocked_score", (query, group, block, block_token), DType.FP32)
    flattened_key = index_binary(
        IndexExpressionKind.ADD,
        index_binary(IndexExpressionKind.MULTIPLY, index_axis(block), index_constant(config.block_size)),
        index_axis(block_token),
    )
    operations.append(
        RelationPrimitive(
            "block_score_relation",
            (token_score,),
            (blocked_score,),
            blocked_score.axes,
            (
                OperandIndexing(
                    token_score,
                    blocked_score.axes,
                    (index_axis(query), index_axis(group), flattened_key),
                ),
            ),
        )
    )
    absolute_query = index_binary(
        IndexExpressionKind.ADD, index_axis(query), index_constant(config.query_position_offset)
    )
    token_valid = ProgramValue("token_valid", (query, block, block_token), DType.BOOL)
    operations.append(
        RelationPrimitive(
            "token_valid_relation",
            (),
            (token_valid,),
            token_valid.axes,
            (),
            result_expressions=(index_compare(IndexPredicateKind.LESS_EQUAL, flattened_key, absolute_query),),
        )
    )
    masked_score = ProgramValue("masked_token_score", blocked_score.axes, DType.FP32)
    operations.append(
        MapPrimitive(
            "mask_token_score",
            (blocked_score, token_valid),
            masked_score,
            scalar_select(
                scalar_input(token_valid.name),
                scalar_binary(
                    ScalarExpressionKind.MULTIPLY,
                    scalar_input(blocked_score.name),
                    scalar_constant(config.index_scale),
                ),
                scalar_constant(float("-inf")),
            ),
        )
    )
    block_score = ProgramValue("block_score", (query, group, block), DType.FP32)
    operations.append(
        FoldPrimitive("block_max_fold", masked_score, block_score, (block_token,), FoldReducer.MAXIMUM, DType.FP32)
    )
    local_block = index_binary(IndexExpressionKind.FLOOR_DIVIDE, absolute_query, index_constant(config.block_size))
    forced_local = ProgramValue("forced_local", (query, block), DType.BOOL)
    block_valid = ProgramValue("block_valid", (query, group, block), DType.BOOL)
    operations.extend(
        (
            RelationPrimitive(
                "forced_local_relation",
                (),
                (forced_local,),
                forced_local.axes,
                (),
                result_expressions=(index_compare(IndexPredicateKind.EQUAL, index_axis(block), local_block),),
            ),
            RelationPrimitive(
                "block_valid_relation",
                (),
                (block_valid,),
                block_valid.axes,
                (),
                result_expressions=(
                    index_compare(
                        IndexPredicateKind.LESS_EQUAL,
                        index_binary(
                            IndexExpressionKind.MULTIPLY,
                            index_axis(block),
                            index_constant(config.block_size),
                        ),
                        absolute_query,
                    ),
                ),
            ),
        )
    )
    ranked_score = ProgramValue("ranked_score", block_score.axes, DType.FP32)
    operations.append(
        MapPrimitive(
            "ranked_score_map",
            (block_score, forced_local, block_valid),
            ranked_score,
            scalar_select(
                scalar_input(block_valid.name),
                scalar_select(
                    scalar_input(forced_local.name),
                    scalar_constant(float("inf")),
                    scalar_input(block_score.name),
                ),
                scalar_constant(float("-inf")),
            ),
        )
    )
    selection_axes = (query, group, selected_slot)
    selected_values = ProgramValue("selected_values", selection_axes, DType.FP32)
    selected_indices = ProgramValue("selected_indices", selection_axes, DType.INT32)
    selected_valid = ProgramValue("selected_valid", selection_axes, DType.BOOL)
    operations.append(
        SelectionPrimitive(
            "bounded_selection",
            ranked_score,
            selected_values,
            selected_indices,
            block,
            config.selected_blocks,
            validity=block_valid,
            output_validity=selected_valid,
        )
    )

    selected_key = ProgramValue("selected_key", (query, group, selected_slot, block_token, feature), DType.BF16)
    selected_value = ProgramValue("selected_value", (query, group, selected_slot, block_token, feature), DType.BF16)
    selected_block = index_tensor_element(
        selected_indices,
        index_axis(query),
        index_axis(group),
        index_axis(selected_slot),
    )
    gathered_key_index = index_binary(
        IndexExpressionKind.ADD,
        index_binary(IndexExpressionKind.MULTIPLY, selected_block, index_constant(config.block_size)),
        index_axis(block_token),
    )
    for name, source, destination in (
        ("gather_key", projected_key, selected_key),
        ("gather_value", projected_value, selected_value),
    ):
        domain = destination.axes
        operations.append(
            RelationPrimitive(
                name,
                (source, selected_indices, selected_valid),
                (destination,),
                domain,
                (
                    OperandIndexing(source, domain, (gathered_key_index, index_axis(group), index_axis(feature))),
                    OperandIndexing(
                        selected_indices,
                        domain,
                        (index_axis(query), index_axis(group), index_axis(selected_slot)),
                    ),
                    OperandIndexing(
                        selected_valid,
                        domain,
                        (index_axis(query), index_axis(group), index_axis(selected_slot)),
                    ),
                ),
                result_sources=(source,),
                predicate=IndexPredicate(
                    IndexPredicateKind.IS_TRUE,
                    expressions=(
                        index_tensor_element(
                            selected_valid,
                            index_axis(query),
                            index_axis(group),
                            index_axis(selected_slot),
                        ),
                    ),
                ),
                order=EvaluationOrder.SOURCE_ORDERED,
            )
        )

    grouped_query = ProgramValue("grouped_query", (query, group, local_head, feature), DType.BF16)
    flattened_head = index_binary(
        IndexExpressionKind.ADD,
        index_binary(IndexExpressionKind.MULTIPLY, index_axis(group), index_constant(local_head.extent)),
        index_axis(local_head),
    )
    operations.append(
        RelationPrimitive(
            "group_query_relation",
            (projected_query,),
            (grouped_query,),
            grouped_query.axes,
            (
                OperandIndexing(
                    projected_query,
                    grouped_query.axes,
                    (index_axis(query), flattened_head, index_axis(feature)),
                ),
            ),
        )
    )
    score_axes = (query, group, local_head, selected_slot, block_token)
    raw_score = ProgramValue("score_raw", score_axes, DType.FP32)
    operations.append(
        ContractPrimitive("score_contract", (grouped_query, selected_key), raw_score, (feature,), DType.FP32)
    )
    score_valid = ProgramValue("score_valid", (query, group, selected_slot, block_token), DType.BOOL)
    valid_lookup = index_tensor_element(
        selected_valid,
        index_axis(query),
        index_axis(group),
        index_axis(selected_slot),
    )
    absolute_selected_key = index_binary(
        IndexExpressionKind.ADD,
        gathered_key_index,
        index_constant(0),
    )
    operations.append(
        RelationPrimitive(
            "score_valid_relation",
            (selected_indices, selected_valid),
            (score_valid,),
            score_valid.axes,
            (
                OperandIndexing(
                    selected_indices,
                    score_valid.axes,
                    (index_axis(query), index_axis(group), index_axis(selected_slot)),
                ),
                OperandIndexing(
                    selected_valid,
                    score_valid.axes,
                    (index_axis(query), index_axis(group), index_axis(selected_slot)),
                ),
            ),
            result_expressions=(
                IndexPredicate(
                    IndexPredicateKind.AND,
                    predicates=(
                        IndexPredicate(IndexPredicateKind.IS_TRUE, expressions=(valid_lookup,)),
                        index_compare(IndexPredicateKind.LESS_EQUAL, absolute_selected_key, absolute_query),
                    ),
                ),
            ),
        )
    )
    mapped_score = ProgramValue("score_mapped", score_axes, DType.FP32)
    operations.append(
        MapPrimitive(
            "score_map",
            (raw_score, score_valid),
            mapped_score,
            scalar_select(
                scalar_input(score_valid.name),
                scalar_binary(
                    ScalarExpressionKind.MULTIPLY,
                    scalar_input(raw_score.name),
                    scalar_constant(config.attention_scale),
                ),
                scalar_constant(float("-inf")),
            ),
        )
    )
    row_axes = (query, group, local_head)
    row_max = ProgramValue("row_max", row_axes, DType.FP32)
    centered = ProgramValue("centered", score_axes, DType.FP32)
    exponentials = ProgramValue("exponentials", score_axes, DType.FP32)
    row_sum = ProgramValue("row_sum", row_axes, DType.FP32)
    operations.extend(
        (
            FoldPrimitive(
                "normalized_max_fold",
                mapped_score,
                row_max,
                (selected_slot, block_token),
                FoldReducer.MAXIMUM,
                DType.FP32,
            ),
            MapPrimitive(
                "center_map",
                (mapped_score, row_max),
                centered,
                scalar_binary(
                    ScalarExpressionKind.SUBTRACT,
                    scalar_input(mapped_score.name),
                    scalar_input(row_max.name),
                ),
            ),
            MapPrimitive(
                "exp_map",
                (centered,),
                exponentials,
                scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered.name)),
            ),
            FoldPrimitive(
                "normalized_sum_fold",
                exponentials,
                row_sum,
                (selected_slot, block_token),
                FoldReducer.SUM,
                DType.FP32,
            ),
        )
    )
    weighted = ProgramValue("weighted", (query, group, local_head, feature), DType.FP32)
    normalized = ProgramValue("normalized", weighted.axes, DType.FP32)
    normalized_bf16 = ProgramValue("normalized_bf16", weighted.axes, DType.BF16)
    output = ProgramValue("output", (query, query_head, feature), DType.BF16)
    operations.extend(
        (
            ContractPrimitive(
                "weighted_contract",
                (exponentials, selected_value),
                weighted,
                (selected_slot, block_token),
                DType.FP32,
            ),
            MapPrimitive(
                "normalize_map",
                (weighted, row_sum),
                normalized,
                scalar_binary(
                    ScalarExpressionKind.DIVIDE,
                    scalar_input(weighted.name),
                    scalar_input(row_sum.name),
                ),
            ),
            MapPrimitive(
                "output_convert",
                (normalized,),
                normalized_bf16,
                scalar_convert(
                    scalar_input(normalized.name),
                    result_dtype=DType.BF16,
                    semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
                ),
            ),
            RelationPrimitive(
                "ungroup_output_relation",
                (normalized_bf16,),
                (output,),
                output.axes,
                (
                    OperandIndexing(
                        normalized_bf16,
                        output.axes,
                        (
                            index_axis(query),
                            index_binary(
                                IndexExpressionKind.FLOOR_DIVIDE,
                                index_axis(query_head),
                                index_constant(local_head.extent),
                            ),
                            index_binary(
                                IndexExpressionKind.MODULO,
                                index_axis(query_head),
                                index_constant(local_head.extent),
                            ),
                            index_axis(feature),
                        ),
                    ),
                ),
            ),
        )
    )
    return TensorProgram(inputs, tuple(operations), (output,), NumericalPolicy.SOURCE_ORDERED)


def _bf16_round(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)
