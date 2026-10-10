# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.backends.h100.generic_codegen_admission import require_generic_codegen_admission
from tile_lifetime.canonical_kernel_composition import (
    CuteKernelComponentFragment,
    compose_cute_pipeline_launch_recipe,
    dependency_convex_contract_phase_schedule,
    dependency_convex_reduction_phase_schedule,
    lower_canonical_kernel_composition,
)
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    verify_materialized_cut_launch_dependencies,
)
from tile_lifetime.dense_flow_planner import (
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
    task_value_accesses,
)
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    PhysicalNumericalPolicy,
    PhysicalPhaseEdge,
    PhysicalTarget,
    RelationAccessClass,
)
from tile_lifetime.sm90_contract_binding import sm90_contract_component_binding
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    derive_streaming_physical_event_schedule,
    streaming_physical_event_schedule_record,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractOperandPrecision,
    ContractPrimitive,
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    execute_tensor_program,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_input,
    tensor_program_value_ordinals,
)


def _producer_program(*, multiple_k=False, fold=True, nonadjacent_reader=False):
    row, column = TensorAxis(0, 3), TensorAxis(1, 16)
    reductions = (TensorAxis(2, 2), TensorAxis(3, 4)) if multiple_k else (TensorAxis(2, 8),)
    activation = ProgramValue("activation", (row, *reductions), DType.BF16)
    weight = ProgramValue("weight", (*reductions, column), DType.BF16)
    residual = ProgramValue("residual", (row, column), DType.BF16)
    product = ProgramValue("product", residual.axes, DType.BF16)
    added = ProgramValue("added", residual.axes, DType.BF16)
    squared = ProgramValue("squared", residual.axes, DType.BF16)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    operations = (
        ContractPrimitive("contract", (activation, weight), product, reductions, DType.FP32),
        MapPrimitive(
            "add",
            (product, residual),
            added,
            scalar_binary(ScalarExpressionKind.ADD, scalar_input(product.name), scalar_input(residual.name)),
        ),
        MapPrimitive(
            "square",
            (added,),
            squared,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(added.name), scalar_input(added.name)),
        ),
    )
    outputs = (added, squared)
    if fold:
        operations += (
            FoldPrimitive(
                "sum", product if nonadjacent_reader else squared, reduced, (column,), FoldReducer.SUM, DType.FP32
            ),
        )
        outputs = (*outputs, reduced) if nonadjacent_reader else (added, reduced)
    return TensorProgram((activation, weight, residual), operations, outputs, NumericalPolicy.FAST)


def _composition(program, phase_edges=None):
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    if phase_edges is not None:
        # Explicit valid task fixture: selection of a fused task is a planner
        # policy; the phase splitter must also handle independently supplied tasks.
        tasks = request.task_plan.tasks
        phases = {phase.operation_ordinals[0]: phase for task in tasks for phase in task.structure.phases}
        kinds = {
            ordinal: kind
            for task in tasks
            for ordinal, kind in zip(task.operation_ordinals, task.operation_kinds, strict=True)
        }
        ordinals = tuple(range(len(program.operations)))
        task = replace(
            tasks[0],
            operation_ordinals=ordinals,
            operation_kinds=tuple(kinds[i] for i in ordinals),
            structure=replace(
                tasks[0].structure, phases=tuple(replace(phases[i], id=i) for i in ordinals), phase_edges=phase_edges
            ),
            accesses=task_value_accesses(program, ordinals),
        )
        plan = CanonicalTaskPlan((task,), ())
        request = physical_request_for_composed_tasks(
            algebra_record=program.canonical_record,
            algebra_fingerprint=program.semantic_fingerprint,
            task_plan=plan,
            event_program=eventize_composed_task_plan(plan),
            target=PhysicalTarget.SM90,
            numerical_policy=PhysicalNumericalPolicy(program.numerical_policy.value),
        )
    return lower_canonical_kernel_composition(request)


def _phase_recipe(program, composition, cut, nested_schedules=()):
    ordinals = tensor_program_value_ordinals(program)
    values = {ordinal: value for value, ordinal in ordinals.items()}
    used = {
        value_id
        for phase in cut.phases
        for value_id in (*phase.component.live_input_value_ids, *phase.component.live_output_value_ids)
    }
    slots = {
        value_id: CutePipelineSlot(
            f"local_{value_id}",
            CutePipelineElementType.BF16 if values[value_id].dtype is DType.BF16 else CutePipelineElementType.F32,
            values[value_id].shape or (1,),
            CutePipelineSlotKind.SCRATCH,
            16,
        )
        for value_id in used
    }
    component = composition.components[0]
    launches = tuple(
        CuteKernelLaunch(
            f"phase_{phase.id}",
            f"kernel_{phase.id}",
            (1, 1, 1),
            (128, 1, 1),
            0,
            (),
            tuple(
                CutePipelineSlotArgument(f"read_{value_id}", slots[value_id].name, CutePipelineSlotAccess.READ)
                for value_id in phase.component.live_input_value_ids
            )
            + tuple(
                CutePipelineSlotArgument(f"write_{value_id}", slots[value_id].name, CutePipelineSlotAccess.WRITE)
                for value_id in phase.component.live_output_value_ids
            ),
            source_tasks=component.task_ids,
        )
        for phase in cut.phases
    )
    return compose_cute_pipeline_launch_recipe(
        composition,
        (
            CuteKernelComponentFragment(
                component.id,
                component.schedule_fingerprint,
                slots,
                launches,
                materialized_cut_value_ids=cut.cut_value_ids,
                physical_event_schedules=nested_schedules,
            ),
        ),
    )


@pytest.mark.parametrize("multiple_k", (False, True))
@pytest.mark.parametrize("fold", (False, True))
def test_initial_contract_cut_preserves_source_phases_live_outputs_and_typed_rounding(multiple_k, fold):
    program = _producer_program(multiple_k=multiple_k, fold=fold)
    composition = _composition(
        program,
        (
            None
            if fold
            else (
                PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),
                PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
            )
        ),
    )
    assert len(composition.components) == 1
    component = composition.components[0]
    cut = dependency_convex_contract_phase_schedule(program, composition, component)
    assert cut is not None
    assert tuple(phase.component.operation_ordinals for phase in cut.phases) == (
        ((0,), (1, 2), (3,)) if fold else ((0,), (1, 2))
    )
    binding = sm90_contract_component_binding(program, cut.phases[0].component)
    assert binding is not None
    assert (binding.rows, binding.reduction, binding.features) == (3, 8, 16)
    assert binding.output is program.operations[0].output
    task = composition.scheduled_request.task_plan.tasks[component.task_ids[0]]
    assert task.structure is not None
    assert tuple(phase for group in cut.phases for phase in group.physical_phases) == task.structure.phases
    assert cut.crossing_phase_edges == tuple(
        edge for edge in task.structure.phase_edges if (edge.source_phase, edge.target_phase) != (1, 2)
    )
    assert all(any(edge is original for original in task.structure.phase_edges) for edge in cut.crossing_phase_edges)
    assert (
        tuple(ordinal for group in cut.phases for ordinal in group.component.operation_ordinals)
        == component.operation_ordinals
    )
    ordinals = tensor_program_value_ordinals(program)
    assert set(component.live_output_value_ids) <= {
        value_id for phase in cut.phases for value_id in phase.component.live_output_value_ids
    }
    recipe = _phase_recipe(program, composition, cut)
    assert recipe.physical_event_provenance is not None
    region = recipe.physical_event_provenance["schedule"]["regions"][0]
    assert region["predecessor_pairs"] == ((((0,), (1,)), ((1,), (2,))) if fold else (((0,), (1,)),))
    product_id = ordinals[binding.output]
    assert (product_id, 0, 1) in cut.cut_value_edges
    assert (
        next(slot for slot in recipe.slots if slot.name == f"value_{product_id}").element_type
        is CutePipelineElementType.BF16
    )

    # This is source-algebra CPU execution, not evidence of provider kernel numerics.
    values = {ordinal: value for value, ordinal in ordinals.items()}
    rng = np.random.default_rng(187)
    inputs = {
        value.name: np.asarray(jnp.asarray(rng.normal(size=value.shape), dtype=jnp.bfloat16)) for value in program.inputs
    }
    environment = dict(inputs)
    for phase in cut.phases:
        phase_program = TensorProgram(
            tuple(values[value_id] for value_id in phase.component.live_input_value_ids),
            tuple(program.operations[ordinal] for ordinal in phase.component.operation_ordinals),
            tuple(values[value_id] for value_id in phase.component.live_output_value_ids),
            program.numerical_policy,
        )
        environment.update(execute_tensor_program(phase_program, environment))
    bf16 = inputs["activation"].dtype
    expected_product = (
        inputs["activation"].astype(np.float32).reshape(3, 8) @ inputs["weight"].astype(np.float32).reshape(8, 16)
    ).astype(bf16)
    expected_added = (expected_product.astype(np.float32) + inputs["residual"].astype(np.float32)).astype(bf16)
    expected_squared = (expected_added.astype(np.float32) ** 2).astype(bf16)
    np.testing.assert_array_equal(environment["product"], expected_product)
    np.testing.assert_array_equal(environment["added"], expected_added)
    np.testing.assert_array_equal(environment["squared"], expected_squared)
    if fold:
        np.testing.assert_array_equal(environment["reduced"], expected_squared.astype(np.float32).sum(axis=1))


def test_materialized_contract_result_orders_every_reader_including_nonadjacent_phase():
    program = _producer_program(nonadjacent_reader=True)
    composition = _composition(
        program,
        (
            PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),
            PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
            PhysicalPhaseEdge(0, 3, RelationAccessClass.AFFINE),
        ),
    )
    assert len(composition.components) == 1
    cut = dependency_convex_contract_phase_schedule(program, composition, composition.components[0])
    assert cut is not None
    product_id = tensor_program_value_ordinals(program)[program.operations[0].output]
    assert cut.cut_value_edges == ((product_id, 0, 1), (product_id, 0, 2))
    recipe = _phase_recipe(program, composition, cut)
    assert recipe.physical_event_provenance is not None
    region = recipe.physical_event_provenance["schedule"]["regions"][0]
    assert region["predecessor_pairs"] == (((0,), (1,)), ((0,), (2,)))
    required = region["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
    assert required == ({"source": (0,), "target": (1,)}, {"source": (0,), "target": (2,)})


@pytest.mark.parametrize("fold", (False, True))
def test_initial_contract_cut_rejects_splitting_indivisible_physical_phase(fold):
    program = _producer_program(fold=fold)
    composition = _composition(
        program,
        (
            None
            if fold
            else (
                PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),
                PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
            )
        ),
    )
    component = composition.components[0]
    request = composition.scheduled_request
    task = request.task_plan.tasks[component.task_ids[0]]
    structure = task.structure
    assert structure is not None
    merged = replace(
        structure,
        phases=(
            replace(structure.phases[0], operation_ordinals=(0, 1)),
            *(replace(phase, id=phase.id - 1) for phase in structure.phases[2:]),
        ),
        phase_edges=tuple(
            replace(edge, source_phase=max(0, edge.source_phase - 1), target_phase=edge.target_phase - 1)
            for edge in structure.phase_edges
            if edge.target_phase > 1
        ),
    )
    task_plan = replace(
        request.task_plan,
        tasks=(
            replace(
                task,
                structure=merged,
                accesses=tuple(replace(access, phase_id=max(0, access.phase_id - 1)) for access in task.accesses),
            ),
        ),
    )
    request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_program=eventize_composed_task_plan(task_plan),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    composition = lower_canonical_kernel_composition(request)
    assert dependency_convex_contract_phase_schedule(program, composition, composition.components[0]) is None


@pytest.mark.parametrize(
    "operation_ordinals,output_ordinals,mkn",
    (
        ((28, 29, 30, 31), (39, 41), (2048, 4096, 4096)),
        ((47, 48, 49, 51), (58, 61), (2048, 14336, 4096)),
    ),
)
def test_reconstructed_full_shape_producer_topology_is_admitted_without_algebra_rewrite(
    operation_ordinals, output_ordinals, mkn
):
    fixture = Path(__file__).parent / "fixtures" / "dense_compile_boundary_program.json"
    full = deserialize_tensor_program(fixture.read_text())
    operations = tuple(full.operations[ordinal] for ordinal in operation_ordinals)
    ordinals = tensor_program_value_ordinals(full)
    values = {ordinal: value for value, ordinal in ordinals.items()}
    produced = {value for operation in operations for value in primitive_outputs(operation)}
    inputs = {value for operation in operations for value in primitive_inputs(operation) if value not in produced}
    program = TensorProgram(
        tuple(sorted(inputs, key=ordinals.__getitem__)),
        operations,
        tuple(values[ordinal] for ordinal in output_ordinals),
        full.numerical_policy,
    )
    composition = _composition(program)
    assert len(composition.components) == 1
    component = composition.components[0]
    original = dependency_convex_reduction_phase_schedule(program, composition, component)
    cut = dependency_convex_contract_phase_schedule(program, composition, component)
    assert original is not None and cut is not None
    assert tuple(phase.component.operation_ordinals for phase in original.phases) == ((0, 1, 2), (3,))
    assert tuple(phase.component.operation_ordinals for phase in cut.phases) == ((0,), (1, 2), (3,))
    assert sm90_contract_component_binding(program, original.phases[0].component) is None
    binding = sm90_contract_component_binding(program, cut.phases[0].component)
    assert binding is not None
    assert (binding.rows, binding.reduction, binding.features) == mkn
    assert binding.output.dtype is DType.BF16
    assert program.operations[0].accumulation_dtype is DType.BF16
    assert all(program.operations[index] is full.operations[ordinal] for index, ordinal in enumerate(operation_ordinals))
    assert set(original.cut_value_ids) < set(cut.cut_value_ids)
    assert all(
        any(edge is preserved for preserved in cut.crossing_phase_edges) for edge in original.crossing_phase_edges
    )
    assert tuple(cut.phases[1].component.live_output_value_ids) == original.phases[0].component.live_output_value_ids
    _phase_recipe(program, composition, cut)  # Shared cut checker proves exact launch/Event coverage.


@pytest.mark.parametrize(
    "case",
    (
        "permuted_left",
        "permuted_right",
        "strided",
        "offset",
        "modulo",
        "source_ordered",
        "ordered_contract",
        "fp64_accumulation",
        "fp32_output",
        "fp32_weight",
        "unaligned_k",
        "unaligned_n",
        "empty_reduction",
    ),
)
def test_sm90_contract_binding_rejects_incompatible_matrix_or_semantic_contract(case):
    row, k0, k1, column = TensorAxis(0, 3), TensorAxis(1, 2), TensorAxis(2, 4), TensorAxis(3, 16)
    if case == "unaligned_k":
        k1 = TensorAxis(2, 3)
    if case == "unaligned_n":
        column = TensorAxis(3, 15)
    reductions = () if case == "empty_reduction" else (k0, k1)
    left_axes = (row, k1, k0) if case == "permuted_left" else (row, *reductions)
    right_axes = (k1, k0, column) if case == "permuted_right" else (*reductions, column)
    left = ProgramValue("left", left_axes, DType.BF16)
    right = ProgramValue("right", right_axes, DType.FP32 if case == "fp32_weight" else DType.BF16)
    output = ProgramValue("output", (row, column), DType.FP32 if case == "fp32_output" else DType.BF16)
    maps = ()
    if case in ("strided", "offset", "modulo"):
        mapping = AxisIndexMap(
            k1,
            k1,
            divisor=2 if case != "modulo" else 1,
            offset=1 if case == "offset" else 0,
            modulus=2 if case == "modulo" else None,
        )
        maps = ((mapping,), ())
    contract = ContractPrimitive(
        "contract",
        (left, right),
        output,
        reductions,
        DType.FP64 if case == "fp64_accumulation" else DType.FP32,
        input_index_maps=maps,
        order=EvaluationOrder.SOURCE_ORDERED if case == "ordered_contract" else EvaluationOrder.ORDER_FREE,
    )
    program = TensorProgram(
        (left, right),
        (contract,),
        (output,),
        NumericalPolicy.SOURCE_ORDERED if case == "source_ordered" else NumericalPolicy.FAST,
    )
    composition = _composition(program)
    assert len(composition.components) == 1
    assert sm90_contract_component_binding(program, composition.components[0]) is None


def test_initial_contract_cut_preserves_source_order_when_reduction_groups_would_interleave():
    original = _producer_program()
    contract, added, squared, fold = original.operations
    late = replace(
        squared,
        inputs=(contract.output,),
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY, scalar_input(contract.output.name), scalar_input(contract.output.name)
        ),
    )
    fold = replace(fold, input=added.output)
    program = replace(
        original, operations=(contract, added, fold, late), outputs=(added.output, fold.output, late.output)
    )
    composition = _composition(
        program,
        (
            PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),
            PhysicalPhaseEdge(1, 2, RelationAccessClass.AFFINE),
            PhysicalPhaseEdge(0, 3, RelationAccessClass.IDENTITY),
        ),
    )
    component = composition.components[0]
    assert dependency_convex_contract_phase_schedule(program, composition, component) is None
    existing = dependency_convex_reduction_phase_schedule(program, composition, component)
    assert existing is not None
    assert tuple(phase.component.operation_ordinals for phase in existing.phases) == ((0, 1, 3), (2,))


@pytest.mark.parametrize("rows,admitted", ((32, False), (128, True)))
def test_sm90_contract_binding_preserves_auxiliary_tile_admission(rows, admitted):
    row, reduction, feature = TensorAxis(0, rows), TensorAxis(1, 64), TensorAxis(2, 256)
    activation = ProgramValue("activation", (row, reduction), DType.BF16)
    scale = ProgramValue("scale", (reduction,), DType.FP32)
    weight = ProgramValue("weight", (reduction, feature), DType.BF16)
    mapped = ProgramValue("mapped", activation.axes, DType.BF16)
    output = ProgramValue("output", (row, feature), DType.BF16)
    program = TensorProgram(
        (activation, scale, weight),
        (
            MapPrimitive(
                "scale",
                (activation, scale),
                mapped,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(activation.name), scalar_input(scale.name)),
            ),
            ContractPrimitive("contract", (mapped, weight), output, (reduction,), DType.FP32),
        ),
        (output,),
        NumericalPolicy.FAST,
    )
    composition = _composition(program)
    assert len(composition.components) == 1
    binding = sm90_contract_component_binding(program, composition.components[0])
    assert (binding is not None) is admitted
    if binding is not None:
        assert binding.scalar_leaves == (activation, scale)
        assert binding.generated.operands[0].value.dtype is DType.FP32


def _matrix_program(a_transposed=False, b_transposed=False, swapped=False, units=False):
    rows = (TensorAxis(0, 2), TensorAxis(1, 4))
    reductions = (TensorAxis(2, 2), TensorAxis(3, 4))
    features = (TensorAxis(4, 2), TensorAxis(5, 8))
    unit = TensorAxis(6, 1)
    a_axes = (*reductions, *rows) if a_transposed else (*rows, *reductions)
    b_axes = (*features, *reductions) if b_transposed else (*reductions, *features)
    if units:
        a_axes, b_axes = (unit, *a_axes), (*b_axes, unit)
    a = ProgramValue("a", a_axes, DType.BF16)
    b = ProgramValue("b", b_axes, DType.BF16)
    out = ProgramValue("out", (*rows, *features, *((unit,) if units else ())), DType.BF16)
    inputs = (b, a) if swapped else (a, b)
    program = TensorProgram(
        inputs, (ContractPrimitive("product", inputs, out, reductions, DType.FP32),), (out,), NumericalPolicy.FAST
    )
    return program, rows, reductions, features


def _assert_matrix_storage_correspondence(view, value, row_axes, column_axes):
    source = np.arange(np.prod(value.shape), dtype=np.int64).reshape(value.shape)
    view.validate_source_storage(source.shape, tuple(stride // source.itemsize for stride in source.strides), 0)
    matrix = np.ndarray(
        view.shape, dtype=source.dtype, buffer=source, strides=tuple(s * source.itemsize for s in view.strides)
    )
    assert np.shares_memory(source, matrix)
    assert matrix.ctypes.data == source.ctypes.data
    for row, column in np.ndindex(view.shape):
        coordinates = dict(zip(row_axes, np.unravel_index(row, tuple(axis.extent for axis in row_axes)), strict=True))
        coordinates.update(
            zip(column_axes, np.unravel_index(column, tuple(axis.extent for axis in column_axes)), strict=True)
        )
        assert matrix[row, column] == source[tuple(coordinates.get(axis, 0) for axis in value.axes)]
    # Writes through the same view must reach exactly the source element.
    matrix[-1, -1] = -17
    assert source.flat[-1] == -17


@pytest.mark.parametrize(
    "a_transposed,b_transposed,swapped", [(a, b, s) for a in (False, True) for b in (False, True) for s in (False, True)]
)
@pytest.mark.parametrize("units", (False, True))
def test_contract_matrix_views_preserve_logical_indices_and_original_allocation(
    a_transposed, b_transposed, swapped, units
):
    program, rows, reductions, features = _matrix_program(a_transposed, b_transposed, swapped, units)
    binding = sm90_contract_component_binding(program, _composition(program).components[0])
    assert binding is not None
    assert binding.activation.name == "a" and binding.weight.name == "b"
    assert binding.generated.operands == ()
    _assert_matrix_storage_correspondence(binding.activation_view, binding.activation, rows, reductions)
    _assert_matrix_storage_correspondence(binding.weight_view, binding.weight, reductions, features)
    _assert_matrix_storage_correspondence(binding.output_view, binding.output, rows, features)


@pytest.mark.parametrize("layout", ("wrong_stride", "offset", "wrong_shape"))
def test_contract_matrix_view_rejects_storage_that_would_require_a_copy_or_rebase(layout):
    program, _, _, _ = _matrix_program(True, True, True, True)
    binding = sm90_contract_component_binding(program, _composition(program).components[0])
    assert binding is not None
    source = np.empty(binding.activation.shape, dtype=np.int64)
    shape, strides, offset = source.shape, tuple(s // source.itemsize for s in source.strides), 0
    if layout == "wrong_stride":
        strides = (*strides[:-1], 2)
    elif layout == "offset":
        offset = 1
    else:
        shape = shape[::-1]
    with pytest.raises(ValueError):
        binding.activation_view.validate_source_storage(shape, strides, offset)


@pytest.mark.parametrize(
    "precision",
    [
        (a, b)
        for a in ContractOperandPrecision
        for b in ContractOperandPrecision
        if (a, b) != (ContractOperandPrecision.DEFAULT,) * 2
    ],
)
def test_contract_provider_rejects_unproved_source_operand_precision(precision):
    program, _, _, _ = _matrix_program()
    program = replace(program, operations=(replace(program.operations[0], operand_precision=precision),))
    assert sm90_contract_component_binding(program, _composition(program).components[0]) is None


@pytest.mark.parametrize("case", ("permuted_k", "interleaved_output", "mapped_transpose"))
def test_oriented_contract_views_reject_unproved_axis_or_transform_semantics(case):
    program, _, reductions, _ = _matrix_program(True, True)
    contract = program.operations[0]
    if case == "permuted_k":
        a = replace(contract.inputs[0], axes=(reductions[1], reductions[0], *contract.inputs[0].axes[2:]))
        contract = replace(contract, inputs=(a, contract.inputs[1]))
        program = replace(program, inputs=contract.inputs, operations=(contract,))
    elif case == "interleaved_output":
        axes = contract.output.axes
        output = replace(contract.output, axes=(axes[0], axes[2], axes[1], axes[3]))
        program = replace(program, outputs=(output,), operations=(replace(contract, output=output),))
    else:
        a = contract.inputs[0]
        mapped = replace(a, name="mapped")
        transform = MapPrimitive(
            "square",
            (a,),
            mapped,
            scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(a.name), scalar_input(a.name)),
        )
        program = replace(program, operations=(transform, replace(contract, inputs=(mapped, contract.inputs[1]))))
    composition = _composition(program)
    assert len(composition.components) == 1
    assert sm90_contract_component_binding(program, composition.components[0]) is None


def _captured_subprogram(operation_ordinals, output_ordinals):
    full = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    operations = tuple(full.operations[ordinal] for ordinal in operation_ordinals)
    ordinals = tensor_program_value_ordinals(full)
    values = {ordinal: value for value, ordinal in ordinals.items()}
    produced = {value for operation in operations for value in primitive_outputs(operation)}
    inputs = {value for operation in operations for value in primitive_inputs(operation) if value not in produced}
    return full, TensorProgram(
        tuple(sorted(inputs, key=ordinals.__getitem__)),
        operations,
        tuple(values[ordinal] for ordinal in output_ordinals),
        full.numerical_policy,
    )


@pytest.mark.parametrize(
    "operations,outputs,groups",
    (
        ((50, 53, 54, 55, 57), (65, 67), ((0, 1, 2, 3), (4,))),
        ((65, 67, 68, 72, 74, 75), (78, 82, 85), ((0,), (1,), (2, 3, 4), (5,))),
        ((117, 119, 120, 122, 123, 127, 129, 130), (133, 137, 140), ((0,), (1,), (2,), (3,), (4, 5, 6), (7,))),
    ),
)
def test_captured_backward_topologies_keep_interleaved_maps_and_exact_phase_event_provenance(
    operations, outputs, groups
):
    full, program = _captured_subprogram(operations, outputs)
    composition = _composition(program)
    assert len(composition.components) == 1
    component = composition.components[0]
    cut = dependency_convex_contract_phase_schedule(program, composition, component)
    assert cut is not None
    assert tuple(phase.component.operation_ordinals for phase in cut.phases) == groups
    assert all(program.operations[index] is full.operations[ordinal] for index, ordinal in enumerate(operations))
    task = composition.scheduled_request.task_plan.tasks[component.task_ids[0]]
    assert tuple(phase for group in cut.phases for phase in group.physical_phases) == task.structure.phases
    assert all(any(edge is original for original in task.structure.phase_edges) for edge in cut.crossing_phase_edges)
    assert set(component.live_output_value_ids) <= {
        value for phase in cut.phases for value in phase.component.live_output_value_ids
    }
    for phase in cut.phases:
        if isinstance(program.operations[phase.component.operation_ordinals[0]], ContractPrimitive):
            binding = sm90_contract_component_binding(program, phase.component)
            assert binding is not None
            assert binding.output.dtype is DType.BF16
            assert program.operations[phase.component.operation_ordinals[0]].accumulation_dtype is DType.BF16
        else:
            require_generic_codegen_admission(
                (ordinal, program.operations[ordinal]) for ordinal in phase.component.operation_ordinals
            )
    recipe = _phase_recipe(program, composition, cut)
    region = recipe.physical_event_provenance["schedule"]["regions"][0]
    predecessors = set(region["predecessor_pairs"])
    required = {
        (edge["source"], edge["target"])
        for edge in region["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
    }
    assert {((writer,), (reader,)) for _, writer, reader in cut.cut_value_edges} <= predecessors & required
    if len(groups) == 6:
        # The first two products reach group 2 across a nonadjacent producer.
        assert any(writer == 0 and reader == 2 for _, writer, reader in cut.cut_value_edges)
        assert any(writer == 2 and reader == 4 for _, writer, reader in cut.cut_value_edges)


@pytest.mark.parametrize(
    "ordinal,mkn,a_input",
    (
        (57, (2048, 4096, 14336), 0),
        (56, (14336, 2048, 4096), 1),
        (64, (4096, 2048, 14336), 1),
        (65, (2048, 14336, 4096), 0),
        (67, (2048, 14336, 4096), 0),
        (66, (4096, 2048, 14336), 1),
        (87, (4096, 2048, 4096), 1),
        (88, (4096, 4096, 2048), 1),
        (89, None, None),
        (95, None, None),
        (98, None, None),
        (101, None, None),
        (104, None, None),
        (116, (4096, 2048, 1024), 1),
        (117, (2048, 1024, 4096), 0),
        (119, (2048, 1024, 4096), 0),
        (122, (2048, 4096, 4096), 0),
        (118, (4096, 2048, 1024), 1),
        (121, (4096, 2048, 4096), 1),
    ),
)
def test_captured_backward_contract_inventory_admits_only_proven_matrix_views(ordinal, mkn, a_input):
    full = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    operation = full.operations[ordinal]
    program = TensorProgram(operation.inputs, (operation,), (operation.output,), full.numerical_policy)
    composition = _composition(program)
    assert len(composition.components) == 1
    binding = sm90_contract_component_binding(program, composition.components[0])
    if mkn is None:
        # Batched normalized-exp contractions retain their dedicated provider.
        assert binding is None
    else:
        assert binding is not None
        assert (binding.rows, binding.reduction, binding.features) == mkn
        assert binding.activation is operation.inputs[a_input]
        assert binding.weight is operation.inputs[1 - a_input]
        assert binding.output is operation.output
        assert binding.generated.operands == ()


@pytest.mark.parametrize("representation", ("predecessors", "required"))
@pytest.mark.parametrize("mutation", ("missing", "reversed"))
def test_interleaved_contract_cut_rejects_missing_or_reversed_nonadjacent_reader_proof(representation, mutation):
    _, program = _captured_subprogram((117, 119, 120, 122, 123, 127, 129, 130), (133, 137, 140))
    composition = _composition(program)
    cut = dependency_convex_contract_phase_schedule(program, composition, composition.components[0])
    assert cut is not None
    recipe = _phase_recipe(program, composition, cut)
    schedule = json.loads(json.dumps(recipe.physical_event_provenance["schedule"]["regions"][0]))
    if representation == "predecessors":
        schedule["predecessor_pairs"].remove([[0], [2]])
        if mutation == "reversed":
            schedule["predecessor_pairs"].append([[2], [0]])
    else:
        pairs = schedule["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
        pairs.remove({"source": [0], "target": [2]})
        if mutation == "reversed":
            pairs.append({"source": [2], "target": [0]})
    with pytest.raises(ValueError):
        verify_materialized_cut_launch_dependencies(
            recipe.launches,
            slots=recipe.slots,
            cut_slots=tuple(f"value_{value_id}" for value_id in cut.cut_value_ids),
            launch_schedule=schedule,
        )


def test_column_major_activation_requires_aligned_contiguous_dimension():
    row, k, column = TensorAxis(0, 7), TensorAxis(1, 8), TensorAxis(2, 16)
    a = ProgramValue("a", (k, row), DType.BF16)
    b = ProgramValue("b", (k, column), DType.BF16)
    output = ProgramValue("output", (row, column), DType.BF16)
    program = TensorProgram(
        (a, b), (ContractPrimitive("contract", (a, b), output, (k,), DType.FP32),), (output,), NumericalPolicy.FAST
    )
    assert sm90_contract_component_binding(program, _composition(program).components[0]) is None


def test_independent_map_and_contract_do_not_invent_a_materialized_cut():
    original = _producer_program(fold=False)
    contract, added, _ = original.operations
    independent = replace(
        added,
        inputs=(original.inputs[2],),
        expression=scalar_input(original.inputs[2].name),
    )
    program = replace(original, operations=(independent, contract), outputs=(independent.output, contract.output))
    composition = _composition(program, (PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),))
    assert len(composition.components) == 1
    assert dependency_convex_contract_phase_schedule(program, composition, composition.components[0]) is None


def test_interleaved_contract_cut_preserves_real_nested_event_allocations():
    _, program = _captured_subprogram((117, 119, 120, 122, 123, 127, 129, 130), (133, 137, 140))
    composition = _composition(program)
    cut = dependency_convex_contract_phase_schedule(program, composition, composition.components[0])
    assert cut is not None
    descriptor = StreamingContractFoldDescriptor(
        first_contract_name="first",
        fold_update_name="update",
        second_contract_name="second",
        finalize_name="finalize",
        fold_extent=512,
        resident_tile_size=128,
        streamed_tile_size=64,
        pipeline_depth=3,
        resident_reduction_dimension=128,
        streamed_reduction_dimension=128,
        output_dimension=128,
        element_bytes=2,
    )
    nested = streaming_physical_event_schedule_record(derive_streaming_physical_event_schedule(descriptor))
    assert nested["allocations"]
    recipe = _phase_recipe(program, composition, cut, (nested,))
    regions = recipe.physical_event_provenance["schedule"]["regions"]
    assert regions[1:] == (nested,)
    assert regions[1]["allocations"] == nested["allocations"]
    assert regions[1]["realizations"] == nested["realizations"]
