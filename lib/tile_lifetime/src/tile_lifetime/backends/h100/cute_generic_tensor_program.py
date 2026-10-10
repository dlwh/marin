# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first CuTe lowering for generic TensorProgram primitives."""

from __future__ import annotations

import hashlib
import logging
import math
import operator
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cuda import bindings as cuda_bindings
from cutlass.utils import SmemAllocator
from flash_attn.cute import utils

from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComponent,
    CanonicalKernelComposition,
    CuteKernelComponentFragment,
    FusedComponentEvaluation,
    bind_cute_pipeline_argument,
    compose_cute_pipeline_launch_recipe,
    composed_cute_compile_request,
    dependency_convex_component_fits_resource_envelope,
    dependency_convex_component_resource_record,
    dependency_convex_contract_phase_schedule,
    dependency_convex_reduction_phase_schedule,
    supports_same_coordinate_producer_reuse,
)
from tile_lifetime.cute_device_ir import CuteDeviceIrSnapshot, compile_cute_with_structured_snapshot
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CuteMultiImagePipelineArtifact,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelinePackedBytesArgument,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    elf64_symbol_bytes,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    RelationPrimitive,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)

from .bf16_tree_fold import Bf16TreeFoldPlan, require_bf16_tree_boundary, requires_bf16_tree_fold
from .cute_scalar_ast import evaluate_cute_scalar_expression
from .fp32_row_fold import ROW_MAXIMUM_LIVE_INPUTS, ROW_THREADS, Fp32RowFoldPlan, plan_fp32_row_fold
from .generic_codegen_admission import require_generic_codegen_admission

_THREADS = ROW_THREADS
_SCALAR_FOLD_ITEMS_PER_THREAD = 16
_SCALAR_FOLD_FINAL_ITEMS_PER_THREAD_LIMIT = 256
_SCALAR_FOLD_WARPS = _THREADS // 32
_SCALAR_FOLD_SHARED_BYTES = _SCALAR_FOLD_WARPS * 4
_MAXIMUM_FUSED_LIVE_INPUTS = ROW_MAXIMUM_LIVE_INPUTS
_MAXIMUM_FUSED_LIVE_OUTPUTS = 3
_KERNEL_POINTER_BYTES = 8
logger = logging.getLogger(__name__)


@dataclass
class CompiledGenericPrimitive:
    """One independently compiled primitive image and its typed boundary."""

    ordinal: int
    operation: TensorPrimitive
    executable: Any
    structured_device_ir: CuteDeviceIrSnapshot
    private_scratch: torch.Tensor | None = None
    bf16_tree_plan: Bf16TreeFoldPlan | None = None
    row_fold_plan: Fp32RowFoldPlan | None = None

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        boundary = tuple(_cute_tensor_boundary(tensor) for tensor in (*inputs, *outputs))
        if self.private_scratch is None:
            self.executable(*boundary)
            return
        self.executable(*boundary, self.private_scratch)

    @property
    def launch_grid(self) -> tuple[int, int, int]:
        if self.row_fold_plan is not None:
            return self.row_fold_plan.grid
        if self.bf16_tree_plan is not None:
            return self.bf16_tree_plan.grid
        elements = math.prod(primitive_outputs(self.operation)[0].shape)
        return ((elements + _THREADS - 1) // _THREADS, 1, 1)

    @property
    def shared_memory_bytes(self) -> int:
        if self.row_fold_plan is not None:
            return self.row_fold_plan.shared_bytes
        return self.bf16_tree_plan.shared_bytes if self.bf16_tree_plan is not None else 0

    @property
    def launch_block(self) -> tuple[int, int, int]:
        if self.row_fold_plan is not None:
            return self.row_fold_plan.block
        return self.bf16_tree_plan.block if self.bf16_tree_plan is not None else (_THREADS, 1, 1)

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(self.executable.kernel_info)


@dataclass
class _PaddedFusedExecutable:
    executable: Any
    input_count: int
    output_count: int
    evaluation: FusedComponentEvaluation
    launch_grid: tuple[int, int, int]
    launch_block: tuple[int, int, int]
    shared_memory_bytes: int
    row_fold_plan: Fp32RowFoldPlan | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.executable, name)

    def __call__(self, *tensors: torch.Tensor) -> Any:
        expected = self.input_count + self.output_count
        if len(tensors) != expected:
            raise ValueError(f"fused component expects {expected} boundary tensors, got {len(tensors)}")
        tensors = tuple(_cute_tensor_boundary(tensor) for tensor in tensors)
        inputs = tensors[: self.input_count]
        outputs = tensors[self.input_count :]
        dummy = inputs[0] if inputs else outputs[0]
        padded_inputs = (*inputs, *(dummy for _ in range(_MAXIMUM_FUSED_LIVE_INPUTS - self.input_count)))
        padded_outputs = (*outputs, *(dummy for _ in range(_MAXIMUM_FUSED_LIVE_OUTPUTS - self.output_count)))
        return self.executable(*padded_inputs, *padded_outputs)


class GenericPhysicalComponent(Protocol):
    """Emitter-selected implementation of one canonical schedule component."""

    @property
    def artifact(self) -> CutePipelineArtifact | CuteMultiImagePipelineArtifact: ...

    @property
    def kernel_names(self) -> tuple[str, ...]: ...

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None: ...

    def canonical_fragment(
        self,
        component: Any,
        value_ordinals: dict[ProgramValue, int],
    ) -> CuteKernelComponentFragment: ...


GenericPhysicalComponentCompiler = Callable[
    [TensorProgram, Any, tuple[ProgramValue, ...], tuple[ProgramValue, ...], tuple[torch.Tensor, ...]],
    GenericPhysicalComponent | None,
]


def _compile_physical_component(
    compilers: tuple[GenericPhysicalComponentCompiler, ...],
    program: TensorProgram,
    component: CanonicalKernelComponent,
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
    example_inputs: tuple[torch.Tensor, ...],
) -> GenericPhysicalComponent | None:
    for compiler in compilers:
        selected = compiler(program, component, live_inputs, live_outputs, example_inputs)
        if selected is not None:
            return selected
    return None


@dataclass
class CompiledGenericTensorProgram:
    """Program-order transport across generic independently compiled images."""

    program: TensorProgram
    primitives: tuple[CompiledGenericPrimitive, ...]

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.program.inputs):
            raise ValueError(f"generic TensorProgram expects {len(self.program.inputs)} inputs, got {len(inputs)}")
        environment: dict[ProgramValue, torch.Tensor] = {}
        for value, tensor in zip(self.program.inputs, inputs, strict=True):
            _validate_tensor(value, tensor)
            environment[value] = tensor
        for compiled in self.primitives:
            operation_inputs = tuple(environment[value] for value in primitive_inputs(compiled.operation))
            operation_outputs = tuple(
                _allocate(value, inputs[0].device) for value in primitive_outputs(compiled.operation)
            )
            compiled(operation_inputs, operation_outputs)
            environment.update(zip(primitive_outputs(compiled.operation), operation_outputs, strict=True))
        return tuple(environment[value] for value in self.program.outputs)


@dataclass
class CompiledGenericComponent:
    """One canonical schedule component compiled as one kernel or primitive residue."""

    component_id: int
    operation_ordinals: tuple[int, ...]
    live_inputs: tuple[ProgramValue, ...]
    live_outputs: tuple[ProgramValue, ...]
    primitives: tuple[CompiledGenericPrimitive, ...]
    fused_executable: Any | None = None
    structured_device_ir: CuteDeviceIrSnapshot | None = None
    fusion_rejection: str | None = None
    physical_component: GenericPhysicalComponent | None = None
    reduction_phases: tuple[CompiledGenericComponent, ...] = ()

    @property
    def is_fused(self) -> bool:
        return bool(self.reduction_phases) or self.fused_executable is not None or self.physical_component is not None

    @property
    def kernel_names(self) -> tuple[str, ...]:
        if self.physical_component is not None:
            return self.physical_component.kernel_names
        if self.reduction_phases:
            return tuple(name for phase in self.reduction_phases for name in phase.kernel_names)
        if self.fused_executable is not None:
            return tuple(self.fused_executable.kernel_info)
        return tuple(name for primitive in self.primitives for name in primitive.kernel_names)


@dataclass
class CompiledGenericKernelComposition:
    """Executable component cover with physical Contracts and generic non-Contract residues."""

    program: TensorProgram
    composition: CanonicalKernelComposition
    components: tuple[CompiledGenericComponent, ...]

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        environment = self.execute_environment(*inputs)
        return tuple(environment[value] for value in self.program.outputs)

    def execute_environment(self, *inputs: torch.Tensor) -> dict[ProgramValue, torch.Tensor]:
        """Execute the component cover and retain canonical intermediate value storage."""
        if len(inputs) != len(self.program.inputs):
            raise ValueError(f"generic TensorProgram expects {len(self.program.inputs)} inputs, got {len(inputs)}")
        environment: dict[ProgramValue, torch.Tensor] = {}
        for value, tensor in zip(self.program.inputs, inputs, strict=True):
            _validate_tensor(value, tensor)
            environment[value] = tensor
        device = inputs[0].device
        for component in self.components:
            if component.reduction_phases:
                phase_environment = {value: environment[value] for value in component.live_inputs}
                for phase in component.reduction_phases:
                    outputs = tuple(_allocate(value, device) for value in phase.live_outputs)
                    if phase.physical_component is not None:
                        phase.physical_component(
                            tuple(phase_environment[value] for value in phase.live_inputs),
                            outputs,
                        )
                    elif phase.fused_executable is not None:
                        phase.fused_executable(
                            *(phase_environment[value] for value in phase.live_inputs),
                            *outputs,
                        )
                    else:
                        if len(phase.primitives) != 1:
                            raise ValueError("a reduction-state residue phase requires one primitive image")
                        phase.primitives[0](
                            tuple(phase_environment[value] for value in phase.live_inputs),
                            outputs,
                        )
                    phase_environment.update(zip(phase.live_outputs, outputs, strict=True))
                environment.update((value, phase_environment[value]) for value in component.live_outputs)
                continue
            if component.physical_component is not None:
                outputs = tuple(_allocate(value, device) for value in component.live_outputs)
                component.physical_component(
                    tuple(environment[value] for value in component.live_inputs),
                    outputs,
                )
                environment.update(zip(component.live_outputs, outputs, strict=True))
                continue
            if component.fused_executable is not None:
                outputs = tuple(_allocate(value, device) for value in component.live_outputs)
                component.fused_executable(
                    *(environment[value] for value in component.live_inputs),
                    *outputs,
                )
                environment.update(zip(component.live_outputs, outputs, strict=True))
                continue
            for primitive in component.primitives:
                outputs = tuple(_allocate(value, device) for value in primitive_outputs(primitive.operation))
                primitive(
                    tuple(environment[value] for value in primitive_inputs(primitive.operation)),
                    outputs,
                )
                environment.update(zip(primitive_outputs(primitive.operation), outputs, strict=True))
        return environment


def compile_generic_tensor_program(
    program: TensorProgram, *, example_inputs: tuple[torch.Tensor, ...]
) -> CompiledGenericTensorProgram:
    """Compile non-Contract primitives with exact program-order transport."""
    require_generic_codegen_admission(enumerate(program.operations))
    if len(example_inputs) != len(program.inputs):
        raise ValueError("generic TensorProgram example inputs do not match the public algebra boundary")
    device = example_inputs[0].device
    environment: dict[ProgramValue, torch.Tensor] = {}
    for value, tensor in zip(program.inputs, example_inputs, strict=True):
        _validate_tensor(value, tensor)
        environment[value] = tensor
    compiled: list[CompiledGenericPrimitive] = []
    for ordinal, operation in enumerate(program.operations):
        outputs = tuple(_allocate(value, device) for value in primitive_outputs(operation))
        inputs = tuple(environment[value] for value in primitive_inputs(operation))
        compiled.append(
            _compile_primitive(ordinal, operation, inputs, outputs, numerical_policy=program.numerical_policy)
        )
        environment.update(zip(primitive_outputs(operation), outputs, strict=True))
    return CompiledGenericTensorProgram(program, tuple(compiled))


def compile_generic_kernel_composition(
    program: TensorProgram,
    composition: CanonicalKernelComposition,
    *,
    example_inputs: tuple[torch.Tensor, ...],
    maximum_fused_operations: int = 8,
    fused_evaluation: FusedComponentEvaluation = FusedComponentEvaluation.RECURSIVE,
    physical_component_compilers: tuple[GenericPhysicalComponentCompiler, ...] = (),
) -> CompiledGenericKernelComposition:
    """Compile components with physical Contracts and generic Map/Fold/Relation code.

    Contracts require an admitted physical provider, either for the whole
    component or a canonical phase. Unsupported Contracts fail before generic
    code generation. Non-Contract residues may use fused or primitive images
    within the declared resource envelope.
    """
    if not isinstance(fused_evaluation, FusedComponentEvaluation):
        raise ValueError("fused evaluation must be a FusedComponentEvaluation")
    if maximum_fused_operations < 2:
        raise ValueError("generic component fusion requires an operation bound of at least two")
    if not physical_component_compilers:
        require_generic_codegen_admission(enumerate(program.operations))
    if len(example_inputs) != len(program.inputs):
        raise ValueError("generic TensorProgram example inputs do not match the public algebra boundary")
    value_ordinals = tensor_program_value_ordinals(program)
    for value, tensor in zip(program.inputs, example_inputs, strict=True):
        _validate_tensor(value, tensor)
    values_by_ordinal = {ordinal: value for value, ordinal in value_ordinals.items()}
    environment = dict(zip(program.inputs, example_inputs, strict=True))
    components: list[CompiledGenericComponent] = []
    for component in composition.components:
        logger.info("Compiling component %d operations=%s", component.id, component.operation_ordinals)
        operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
        live_inputs = tuple(values_by_ordinal[value_id] for value_id in component.live_input_value_ids)
        live_outputs = tuple(values_by_ordinal[value_id] for value_id in component.live_output_value_ids)
        resource_record = dependency_convex_component_resource_record(program, component)
        reduction_schedule = dependency_convex_reduction_phase_schedule(program, composition, component)
        fused = None
        snapshot = None
        rejection = None
        primitives: tuple[CompiledGenericPrimitive, ...] = ()
        reduction_phases: tuple[CompiledGenericComponent, ...] = ()
        admitted_phase_components: dict[int, GenericPhysicalComponent] = {}
        physical_component = _compile_physical_component(
            physical_component_compilers,
            program,
            component,
            live_inputs,
            live_outputs,
            tuple(environment[value] for value in live_inputs),
        )
        if physical_component is None and physical_component_compilers:
            candidate_schedule = dependency_convex_contract_phase_schedule(program, composition, component)
            if candidate_schedule is not None and all(
                len(phase.component.operation_ordinals) == 1
                or dependency_convex_component_fits_resource_envelope(
                    phase.resource_record,
                    maximum_operations=maximum_fused_operations,
                    maximum_live_inputs=_MAXIMUM_FUSED_LIVE_INPUTS,
                    maximum_live_outputs=_MAXIMUM_FUSED_LIVE_OUTPUTS,
                    maximum_reduction_depth=1,
                )
                for phase in candidate_schedule.phases
            ):
                candidate_components: dict[int, GenericPhysicalComponent] = {}
                for contract_phase in candidate_schedule.phases:
                    if len(contract_phase.component.operation_ordinals) != 1 or not isinstance(
                        program.operations[contract_phase.component.operation_ordinals[0]], ContractPrimitive
                    ):
                        continue
                    contract_inputs = tuple(
                        values_by_ordinal[value_id] for value_id in contract_phase.component.live_input_value_ids
                    )
                    contract_outputs = tuple(
                        values_by_ordinal[value_id] for value_id in contract_phase.component.live_output_value_ids
                    )
                    logger.info(
                        "Trying physical component %d Contract phase operations=%s",
                        component.id,
                        contract_phase.component.operation_ordinals,
                    )
                    admitted = _compile_physical_component(
                        physical_component_compilers,
                        program,
                        contract_phase.component,
                        contract_inputs,
                        contract_outputs,
                        tuple(
                            environment[value] if value in environment else _allocate(value, example_inputs[0].device)
                            for value in contract_inputs
                        ),
                    )
                    logger.info(
                        "Physical component %d Contract phase provider=%s",
                        component.id,
                        type(admitted).__name__ if admitted is not None else None,
                    )
                    if admitted is None:
                        break
                    candidate_components[contract_phase.id] = admitted
                else:
                    reduction_schedule = candidate_schedule
                    admitted_phase_components = candidate_components
        if physical_component is None and reduction_schedule is None:
            require_bf16_tree_boundary(operations, program.numerical_policy)
            require_generic_codegen_admission(
                zip(component.operation_ordinals, operations, strict=True), component_id=component.id
            )
        if physical_component is not None:
            environment.update((value, _allocate(value, example_inputs[0].device)) for value in live_outputs)
        elif reduction_schedule is not None:
            phase_environment = dict(environment)
            compiled_phases = []
            for phase in reduction_schedule.phases:
                logger.info(
                    "Compiling component %d phase %d operations=%s",
                    component.id,
                    phase.id,
                    phase.component.operation_ordinals,
                )
                phase_live_inputs = tuple(
                    values_by_ordinal[value_id] for value_id in phase.component.live_input_value_ids
                )
                phase_live_outputs = tuple(
                    values_by_ordinal[value_id] for value_id in phase.component.live_output_value_ids
                )
                phase_physical_component = admitted_phase_components.get(phase.id)
                if phase_physical_component is None:
                    phase_physical_component = _compile_physical_component(
                        physical_component_compilers,
                        program,
                        phase.component,
                        phase_live_inputs,
                        phase_live_outputs,
                        tuple(phase_environment[value] for value in phase_live_inputs),
                    )
                if phase_physical_component is None:
                    require_bf16_tree_boundary(
                        tuple(program.operations[ordinal] for ordinal in phase.component.operation_ordinals),
                        program.numerical_policy,
                    )
                    require_generic_codegen_admission(
                        ((ordinal, program.operations[ordinal]) for ordinal in phase.component.operation_ordinals),
                        component_id=component.id,
                        phase_id=phase.id,
                    )
                output_tensors = tuple(_allocate(value, example_inputs[0].device) for value in phase_live_outputs)
                if phase_physical_component is not None:
                    phase_primitives = ()
                    phase_fused = None
                    phase_snapshot = None
                elif len(phase.component.operation_ordinals) == 1:
                    ordinal = phase.component.operation_ordinals[0]
                    operation = program.operations[ordinal]
                    phase_primitives = (
                        _compile_primitive(
                            ordinal,
                            operation,
                            tuple(phase_environment[value] for value in primitive_inputs(operation)),
                            output_tensors,
                            numerical_policy=program.numerical_policy,
                        ),
                    )
                    phase_fused = None
                    phase_snapshot = None
                else:
                    if not dependency_convex_component_fits_resource_envelope(
                        phase.resource_record,
                        maximum_operations=maximum_fused_operations,
                        maximum_live_inputs=_MAXIMUM_FUSED_LIVE_INPUTS,
                        maximum_live_outputs=_MAXIMUM_FUSED_LIVE_OUTPUTS,
                        maximum_reduction_depth=1,
                    ):
                        raise ValueError("a reduction-state producer phase exceeds the generic resource envelope")
                    phase_fused, phase_snapshot = _compile_fused_component(
                        tuple(program.operations[ordinal] for ordinal in phase.component.operation_ordinals),
                        phase_live_inputs,
                        phase_live_outputs,
                        tuple(phase_environment[value] for value in phase_live_inputs),
                        output_tensors,
                        evaluation=fused_evaluation,
                        numerical_policy=program.numerical_policy,
                    )
                    phase_primitives = ()
                compiled_phases.append(
                    CompiledGenericComponent(
                        component.id,
                        phase.component.operation_ordinals,
                        phase_live_inputs,
                        phase_live_outputs,
                        phase_primitives,
                        phase_fused,
                        phase_snapshot,
                        physical_component=phase_physical_component,
                    )
                )
                phase_environment.update(zip(phase_live_outputs, output_tensors, strict=True))
                logger.info("Compiled component %d phase %d", component.id, phase.id)
            reduction_phases = tuple(compiled_phases)
            environment.update((value, phase_environment[value]) for value in live_outputs)
        elif dependency_convex_component_fits_resource_envelope(
            resource_record,
            maximum_operations=maximum_fused_operations,
            maximum_live_inputs=_MAXIMUM_FUSED_LIVE_INPUTS,
            maximum_live_outputs=_MAXIMUM_FUSED_LIVE_OUTPUTS,
            maximum_reduction_depth=1,
        ):
            try:
                fused, snapshot = _compile_fused_component(
                    operations,
                    live_inputs,
                    live_outputs,
                    tuple(environment[value] for value in live_inputs),
                    tuple(_allocate(value, example_inputs[0].device) for value in live_outputs),
                    evaluation=fused_evaluation,
                    numerical_policy=program.numerical_policy,
                )
            except ValueError as error:
                fused = None
                snapshot = None
                rejection = str(error)
        if physical_component is None and fused is None and not reduction_phases:
            compiled_primitives = []
            for ordinal, operation in zip(component.operation_ordinals, operations, strict=True):
                outputs = tuple(_allocate(value, example_inputs[0].device) for value in primitive_outputs(operation))
                compiled_primitives.append(
                    _compile_primitive(
                        ordinal,
                        operation,
                        tuple(environment[value] for value in primitive_inputs(operation)),
                        outputs,
                        numerical_policy=program.numerical_policy,
                    )
                )
                environment.update(zip(primitive_outputs(operation), outputs, strict=True))
            primitives = tuple(compiled_primitives)
        elif physical_component is None:
            environment.update((value, _allocate(value, example_inputs[0].device)) for value in live_outputs)
        components.append(
            CompiledGenericComponent(
                component.id,
                component.operation_ordinals,
                live_inputs,
                live_outputs,
                primitives,
                fused,
                snapshot,
                rejection,
                physical_component,
                reduction_phases,
            )
        )
        logger.info("Compiled component %d operations=%s", component.id, component.operation_ordinals)
    return CompiledGenericKernelComposition(program, composition, tuple(components))


def build_generic_multi_image_artifact(
    compiled: CompiledGenericTensorProgram,
    composition: CanonicalKernelComposition,
    *,
    function_prefix: str,
) -> CuteMultiImagePipelineArtifact:
    """Export primitive images and bind them to one exact canonical recipe."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    if len(compiled.program.operations) != len(compiled.primitives):
        raise ValueError("generic multi-image artifact requires complete primitive code generation")
    value_ordinals = tensor_program_value_ordinals(compiled.program)
    compiled_by_ordinal = {primitive.ordinal: primitive for primitive in compiled.primitives}
    fragments: list[CuteKernelComponentFragment] = []
    image_by_launch: list[int] = []
    for component in composition.components:
        component_primitives = tuple(compiled_by_ordinal[ordinal] for ordinal in component.operation_ordinals)
        values = {
            value
            for primitive in component_primitives
            for value in (*primitive_inputs(primitive.operation), *primitive_outputs(primitive.operation))
        }
        slots = {
            value_ordinals[value]: CutePipelineSlot(
                name=f"local_value_{value_ordinals[value]}",
                element_type=_pipeline_element_type(value.dtype),
                shape=value.shape or (1,),
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for value in values
        }
        launches = []
        private_slots: list[CutePipelineSlot] = []
        for primitive in component_primitives:
            operation_inputs = primitive_inputs(primitive.operation)
            operation_outputs = primitive_outputs(primitive.operation)
            if primitive.private_scratch is not None:
                if len(operation_inputs) != 1 or len(operation_outputs) != 1 or len(primitive.kernel_names) != 2:
                    raise ValueError("hierarchical scalar Fold artifact requires one input, one output, and two kernels")
                partial_slot = CutePipelineSlot(
                    name=f"operation_{primitive.ordinal}_partials",
                    element_type=CutePipelineElementType.F32,
                    shape=tuple(primitive.private_scratch.shape),
                    kind=CutePipelineSlotKind.SCRATCH,
                    alignment=16,
                )
                private_slots.append(partial_slot)
                launches.extend(
                    (
                        CuteKernelLaunch(
                            name=f"operation_{primitive.ordinal}_partials",
                            kernel_symbol=primitive.kernel_names[0],
                            grid=(primitive.private_scratch.numel(), 1, 1),
                            block=(_THREADS, 1, 1),
                            dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                            function_attributes=(),
                            arguments=(
                                CutePipelineSlotArgument(
                                    "source",
                                    slots[value_ordinals[operation_inputs[0]]].name,
                                    CutePipelineSlotAccess.READ,
                                ),
                                CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.WRITE),
                            ),
                            source_tasks=component.task_ids,
                            source_phase="partial",
                        ),
                        CuteKernelLaunch(
                            name=f"operation_{primitive.ordinal}_final",
                            kernel_symbol=primitive.kernel_names[1],
                            grid=(1, 1, 1),
                            block=(_THREADS, 1, 1),
                            dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                            function_attributes=(),
                            arguments=(
                                CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.READ),
                                CutePipelineSlotArgument(
                                    "output",
                                    slots[value_ordinals[operation_outputs[0]]].name,
                                    CutePipelineSlotAccess.WRITE,
                                ),
                            ),
                            source_tasks=component.task_ids,
                            source_phase="merge",
                        ),
                    )
                )
                image_by_launch.extend((primitive.ordinal, primitive.ordinal))
                continue
            arguments = tuple(
                CutePipelineSlotArgument(
                    name=f"argument_{index}",
                    slot=slots[value_ordinals[value]].name,
                    access=(
                        CutePipelineSlotAccess.READ if index < len(operation_inputs) else CutePipelineSlotAccess.WRITE
                    ),
                )
                for index, value in enumerate((*operation_inputs, *operation_outputs))
            )
            launches.append(
                CuteKernelLaunch(
                    name=f"operation_{primitive.ordinal}",
                    kernel_symbol=primitive.kernel_names[0],
                    grid=primitive.launch_grid,
                    block=primitive.launch_block,
                    dynamic_shared_memory_bytes=primitive.shared_memory_bytes,
                    function_attributes=(),
                    arguments=arguments,
                    source_tasks=component.task_ids,
                )
            )
            image_by_launch.append(primitive.ordinal)
        fragments.append(
            CuteKernelComponentFragment(
                component_id=component.id,
                schedule_fingerprint=component.schedule_fingerprint,
                slots_by_value_id=slots,
                launches=tuple(launches),
                private_slots=tuple(private_slots),
            )
        )
    recipe = compose_cute_pipeline_launch_recipe(composition, tuple(fragments))
    images = tuple(
        export_generic_primitive_artifact(
            primitive,
            compiled.program,
            function_prefix=f"{function_prefix}_{primitive.ordinal}",
        )
        for primitive in compiled.primitives
    )
    return CuteMultiImagePipelineArtifact(
        request=composed_cute_compile_request(composition, recipe),
        images=images,
        launch_recipe=recipe,
        image_by_launch=tuple(image_by_launch),
    )


def build_generic_component_multi_image_artifact(
    compiled: CompiledGenericKernelComposition,
    *,
    function_prefix: str,
) -> CuteMultiImagePipelineArtifact:
    """Export a mixed fused-component/primitive-residue canonical artifact."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    value_ordinals = tensor_program_value_ordinals(compiled.program)
    fragments: list[CuteKernelComponentFragment] = []
    images: list[CutePipelineArtifact] = []
    image_by_launch: list[int] = []
    reduction_phase_schedules: list[dict[str, Any]] = []
    for component, canonical in zip(compiled.components, compiled.composition.components, strict=True):
        slots = {
            value_ordinals[value]: CutePipelineSlot(
                name=f"local_value_{value_ordinals[value]}",
                element_type=_pipeline_element_type(value.dtype),
                shape=value.shape or (1,),
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for value in (*component.live_inputs, *component.live_outputs)
        }
        launches: list[CuteKernelLaunch] = []
        private_slots: list[CutePipelineSlot] = []
        if component.physical_component is not None:
            image_index = len(images)
            artifact = component.physical_component.artifact
            fragment = component.physical_component.canonical_fragment(canonical, value_ordinals)
            fragments.append(fragment)
            if isinstance(artifact, CuteMultiImagePipelineArtifact):
                images.extend(artifact.images)
                image_by_launch.extend(image_index + index for index in artifact.image_by_launch)
            else:
                images.append(artifact)
                image_by_launch.extend((image_index,) * len(fragment.launches))
            continue
        if component.reduction_phases:
            compiled_groups = tuple(phase.operation_ordinals for phase in component.reduction_phases)
            reduction_schedule = next(
                (
                    schedule
                    for schedule in (
                        dependency_convex_reduction_phase_schedule(compiled.program, compiled.composition, canonical),
                        dependency_convex_contract_phase_schedule(compiled.program, compiled.composition, canonical),
                    )
                    if schedule is not None
                    and tuple(phase.component.operation_ordinals for phase in schedule.phases) == compiled_groups
                ),
                None,
            )
            if reduction_schedule is None:
                raise ValueError("compiled phases do not match their canonical component cut")
            physical_event_schedules = []
            materialized_cut_value_ids = set(reduction_schedule.cut_value_ids)
            for phase_index, phase in enumerate(component.reduction_phases):
                for value in (*phase.live_inputs, *phase.live_outputs):
                    slots.setdefault(
                        value_ordinals[value],
                        CutePipelineSlot(
                            name=f"local_value_{value_ordinals[value]}",
                            element_type=_pipeline_element_type(value.dtype),
                            shape=value.shape or (1,),
                            kind=CutePipelineSlotKind.SCRATCH,
                            alignment=16,
                        ),
                    )
                if phase.physical_component is not None:
                    physical = phase.physical_component
                    phase_component = reduction_schedule.phases[phase_index].component
                    fragment = physical.canonical_fragment(phase_component, value_ordinals)
                    required_values = set(phase_component.live_input_value_ids) | set(
                        phase_component.live_output_value_ids
                    )
                    if (
                        fragment.component_id != phase_component.id
                        or fragment.schedule_fingerprint != phase_component.schedule_fingerprint
                        or not required_values <= set(fragment.slots_by_value_id)
                        or any(not set(launch.source_tasks) <= set(canonical.task_ids) for launch in fragment.launches)
                    ):
                        raise ValueError("a physical phase fragment does not cover its canonical boundary")
                    local_to_component = {}
                    for value_id, slot in fragment.slots_by_value_id.items():
                        bound = slots.setdefault(value_id, slot)
                        if (bound.element_type, bound.shape) != (slot.element_type, slot.shape):
                            raise ValueError("a physical phase disagrees with canonical value storage")
                        slots[value_id] = replace(bound, alignment=max(bound.alignment, slot.alignment))
                        local_to_component[slot.name] = bound.name
                    reuse_offset = 1 + max(
                        (slot.reuse_group for slot in private_slots if slot.reuse_group is not None), default=-1
                    )
                    for slot in fragment.private_slots:
                        name = f"phase_{phase_index}_{slot.name}"
                        local_to_component[slot.name] = name
                        private_slots.append(
                            replace(
                                slot,
                                name=name,
                                reuse_group=None if slot.reuse_group is None else reuse_offset + slot.reuse_group,
                            )
                        )
                    launches.extend(
                        replace(
                            launch,
                            name=f"phase_{phase_index}_{launch.name}",
                            arguments=tuple(
                                bind_cute_pipeline_argument(argument, local_to_component)
                                for argument in launch.arguments
                            ),
                            source_tasks=canonical.task_ids,
                            source_phase=(
                                f"component_phase_{phase_index}_{launch.source_phase}"
                                if launch.source_phase is not None
                                else f"component_phase_{phase_index}"
                            ),
                        )
                        for launch in fragment.launches
                    )
                    physical_event_schedules.extend(fragment.physical_event_schedules)
                    materialized_cut_value_ids.update(fragment.materialized_cut_value_ids)
                    image_index = len(images)
                    if isinstance(physical.artifact, CuteMultiImagePipelineArtifact):
                        images.extend(physical.artifact.images)
                        image_by_launch.extend(image_index + index for index in physical.artifact.image_by_launch)
                    else:
                        images.append(physical.artifact)
                        image_by_launch.extend((image_index,) * len(fragment.launches))
                    continue
                if phase.fused_executable is not None:
                    image_index = len(images)
                    images.append(
                        _export_fused_component_artifact(
                            phase,
                            compiled.program,
                            function_prefix=(
                                f"{function_prefix}_component_{component.component_id}_phase_{phase_index}"
                            ),
                        )
                    )
                    arguments = (
                        *tuple(
                            CutePipelineSlotArgument(
                                f"input_{index}",
                                slots[value_ordinals[value]].name,
                                CutePipelineSlotAccess.READ,
                            )
                            for index, value in enumerate(phase.live_inputs)
                        ),
                        *tuple(
                            CutePipelinePackedBytesArgument(
                                f"padding_input_{index}", bytes(_KERNEL_POINTER_BYTES), _KERNEL_POINTER_BYTES
                            )
                            for index in range(_MAXIMUM_FUSED_LIVE_INPUTS - len(phase.live_inputs))
                        ),
                        *tuple(
                            CutePipelineSlotArgument(
                                f"output_{index}",
                                slots[value_ordinals[value]].name,
                                CutePipelineSlotAccess.WRITE,
                            )
                            for index, value in enumerate(phase.live_outputs)
                        ),
                        *tuple(
                            CutePipelinePackedBytesArgument(
                                f"padding_output_{index}", bytes(_KERNEL_POINTER_BYTES), _KERNEL_POINTER_BYTES
                            )
                            for index in range(_MAXIMUM_FUSED_LIVE_OUTPUTS - len(phase.live_outputs))
                        ),
                    )
                    launches.append(
                        CuteKernelLaunch(
                            name=f"reduction_depth_{phase_index + 1}",
                            kernel_symbol=phase.kernel_names[0],
                            grid=phase.fused_executable.launch_grid,
                            block=phase.fused_executable.launch_block,
                            dynamic_shared_memory_bytes=phase.fused_executable.shared_memory_bytes,
                            function_attributes=(),
                            arguments=arguments,
                            source_tasks=canonical.task_ids,
                            source_phase=f"reduction_depth_{phase_index + 1}",
                        )
                    )
                    image_by_launch.append(image_index)
                    continue
                if len(phase.primitives) != 1:
                    raise ValueError("a reduction-state residue phase must export one primitive image")
                primitive = phase.primitives[0]
                image_index = len(images)
                images.append(
                    export_generic_primitive_artifact(
                        primitive,
                        compiled.program,
                        function_prefix=f"{function_prefix}_{primitive.ordinal}",
                    )
                )
                operation_inputs = primitive_inputs(primitive.operation)
                operation_outputs = primitive_outputs(primitive.operation)
                if primitive.private_scratch is None:
                    values = (*operation_inputs, *operation_outputs)
                    launches.append(
                        CuteKernelLaunch(
                            name=f"reduction_depth_{phase_index + 1}",
                            kernel_symbol=primitive.kernel_names[0],
                            grid=primitive.launch_grid,
                            block=primitive.launch_block,
                            dynamic_shared_memory_bytes=primitive.shared_memory_bytes,
                            function_attributes=(),
                            arguments=tuple(
                                CutePipelineSlotArgument(
                                    f"argument_{index}",
                                    slots[value_ordinals[value]].name,
                                    (
                                        CutePipelineSlotAccess.READ
                                        if index < len(operation_inputs)
                                        else CutePipelineSlotAccess.WRITE
                                    ),
                                )
                                for index, value in enumerate(values)
                            ),
                            source_tasks=canonical.task_ids,
                            source_phase=f"reduction_depth_{phase_index + 1}",
                        )
                    )
                    image_by_launch.append(image_index)
                    continue
                partial_slot = CutePipelineSlot(
                    name=f"phase_{phase_index}_partials",
                    element_type=CutePipelineElementType.F32,
                    shape=tuple(primitive.private_scratch.shape),
                    kind=CutePipelineSlotKind.SCRATCH,
                    alignment=16,
                )
                private_slots.append(partial_slot)
                source_slot = slots[value_ordinals[operation_inputs[0]]].name
                output_slot = slots[value_ordinals[operation_outputs[0]]].name
                launches.extend(
                    (
                        CuteKernelLaunch(
                            name=f"reduction_depth_{phase_index + 1}_partials",
                            kernel_symbol=primitive.kernel_names[0],
                            grid=(primitive.private_scratch.numel(), 1, 1),
                            block=(_THREADS, 1, 1),
                            dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                            function_attributes=(),
                            arguments=(
                                CutePipelineSlotArgument("source", source_slot, CutePipelineSlotAccess.READ),
                                CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.WRITE),
                            ),
                            source_tasks=canonical.task_ids,
                            source_phase=f"reduction_depth_{phase_index + 1}_partials",
                        ),
                        CuteKernelLaunch(
                            name=f"reduction_depth_{phase_index + 1}_merge",
                            kernel_symbol=primitive.kernel_names[1],
                            grid=(1, 1, 1),
                            block=(_THREADS, 1, 1),
                            dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                            function_attributes=(),
                            arguments=(
                                CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.READ),
                                CutePipelineSlotArgument("output", output_slot, CutePipelineSlotAccess.WRITE),
                            ),
                            source_tasks=canonical.task_ids,
                            source_phase=f"reduction_depth_{phase_index + 1}_merge",
                        ),
                    )
                )
                image_by_launch.extend((image_index, image_index))
            phase_event_schedule = {
                "schema_version": 1,
                "kind": reduction_schedule.kind.value,
                "source_component_id": component.component_id,
                "phases": tuple(
                    {
                        "id": phase.id,
                        "physical_phase_ids": tuple(item.id for item in phase.physical_phases),
                        "operation_ordinals": phase.component.operation_ordinals,
                        "live_input_value_ids": phase.component.live_input_value_ids,
                        "live_output_value_ids": phase.component.live_output_value_ids,
                        "domain_shapes": tuple(item.domain_shape for item in phase.physical_phases),
                        "reduction_dimensions": tuple(item.reduction_dimensions for item in phase.physical_phases),
                    }
                    for phase in reduction_schedule.phases
                ),
                "crossing_phase_edges": tuple(
                    {
                        "source_phase": edge.source_phase,
                        "target_phase": edge.target_phase,
                        "relation_class": edge.relation_class.value,
                    }
                    for edge in reduction_schedule.crossing_phase_edges
                ),
                "events": tuple(
                    {
                        "cut_value_id": value_id,
                        "capacity": reduction_schedule.event_capacity,
                        "generation": 0,
                        "generation_count": reduction_schedule.generation_count,
                        "producer_phase": producer,
                        "consumer_phase": consumer,
                    }
                    for value_id, producer, consumer in reduction_schedule.cut_value_edges
                ),
            }
            reduction_phase_schedules.append(phase_event_schedule)
            fragments.append(
                CuteKernelComponentFragment(
                    component_id=component.component_id,
                    schedule_fingerprint=canonical.schedule_fingerprint,
                    slots_by_value_id=slots,
                    launches=tuple(launches),
                    private_slots=tuple(private_slots),
                    physical_event_schedules=tuple(physical_event_schedules),
                    materialized_cut_value_ids=tuple(sorted(materialized_cut_value_ids)),
                )
            )
            continue
        if component.fused_executable is not None:
            image_index = len(images)
            image = _export_fused_component_artifact(
                component,
                compiled.program,
                function_prefix=f"{function_prefix}_component_{component.component_id}",
            )
            images.append(image)
            boundary_arguments = tuple(
                CutePipelineSlotArgument(
                    name=f"argument_{index}",
                    slot=slots[value_ordinals[value]].name,
                    access=(
                        CutePipelineSlotAccess.READ
                        if index < len(component.live_inputs)
                        else CutePipelineSlotAccess.WRITE
                    ),
                )
                for index, value in enumerate((*component.live_inputs, *component.live_outputs))
            )
            # The kernel statically slices away padded tensors. Null pointer
            # payloads preserve its ABI without inventing scratch memory accesses.
            input_arguments = boundary_arguments[: len(component.live_inputs)]
            output_arguments = boundary_arguments[len(component.live_inputs) :]
            padded_inputs = (
                *input_arguments,
                *(
                    CutePipelinePackedBytesArgument(
                        name=f"padding_input_{index}",
                        data=bytes(_KERNEL_POINTER_BYTES),
                        alignment=_KERNEL_POINTER_BYTES,
                    )
                    for index in range(_MAXIMUM_FUSED_LIVE_INPUTS - len(component.live_inputs))
                ),
            )
            padded_outputs = (
                *output_arguments,
                *(
                    CutePipelinePackedBytesArgument(
                        name=f"padding_output_{index}",
                        data=bytes(_KERNEL_POINTER_BYTES),
                        alignment=_KERNEL_POINTER_BYTES,
                    )
                    for index in range(_MAXIMUM_FUSED_LIVE_OUTPUTS - len(component.live_outputs))
                ),
            )
            arguments = (*padded_inputs, *padded_outputs)
            launches.append(
                CuteKernelLaunch(
                    name="fused_component",
                    kernel_symbol=component.kernel_names[0],
                    grid=component.fused_executable.launch_grid,
                    block=component.fused_executable.launch_block,
                    dynamic_shared_memory_bytes=component.fused_executable.shared_memory_bytes,
                    function_attributes=(),
                    arguments=arguments,
                    source_tasks=canonical.task_ids,
                )
            )
            image_by_launch.append(image_index)
        else:
            for primitive in component.primitives:
                image_index = len(images)
                image = export_generic_primitive_artifact(
                    primitive,
                    compiled.program,
                    function_prefix=f"{function_prefix}_{primitive.ordinal}",
                )
                images.append(image)
                operation_inputs = primitive_inputs(primitive.operation)
                operation_outputs = primitive_outputs(primitive.operation)
                values = (*operation_inputs, *operation_outputs)
                for value in values:
                    slots.setdefault(
                        value_ordinals[value],
                        CutePipelineSlot(
                            name=f"local_value_{value_ordinals[value]}",
                            element_type=_pipeline_element_type(value.dtype),
                            shape=value.shape or (1,),
                            kind=CutePipelineSlotKind.SCRATCH,
                            alignment=16,
                        ),
                    )
                if primitive.private_scratch is not None:
                    if len(operation_inputs) != 1 or len(operation_outputs) != 1 or len(primitive.kernel_names) != 2:
                        raise ValueError(
                            "hierarchical scalar Fold artifact requires one input, one output, and two kernels"
                        )
                    partial_slot = CutePipelineSlot(
                        name=f"operation_{primitive.ordinal}_partials",
                        element_type=CutePipelineElementType.F32,
                        shape=tuple(primitive.private_scratch.shape),
                        kind=CutePipelineSlotKind.SCRATCH,
                        alignment=16,
                    )
                    private_slots.append(partial_slot)
                    source_slot = slots[value_ordinals[operation_inputs[0]]].name
                    output_slot = slots[value_ordinals[operation_outputs[0]]].name
                    partial_arguments = (
                        CutePipelineSlotArgument("source", source_slot, CutePipelineSlotAccess.READ),
                        CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.WRITE),
                    )
                    final_arguments = (
                        CutePipelineSlotArgument("partials", partial_slot.name, CutePipelineSlotAccess.READ),
                        CutePipelineSlotArgument("output", output_slot, CutePipelineSlotAccess.WRITE),
                    )
                    launches.extend(
                        (
                            CuteKernelLaunch(
                                name=f"operation_{primitive.ordinal}_partials",
                                kernel_symbol=primitive.kernel_names[0],
                                grid=(primitive.private_scratch.numel(), 1, 1),
                                block=(_THREADS, 1, 1),
                                dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                                function_attributes=(),
                                arguments=partial_arguments,
                                source_tasks=canonical.task_ids,
                                source_phase="partial",
                            ),
                            CuteKernelLaunch(
                                name=f"operation_{primitive.ordinal}_final",
                                kernel_symbol=primitive.kernel_names[1],
                                grid=(1, 1, 1),
                                block=(_THREADS, 1, 1),
                                dynamic_shared_memory_bytes=_SCALAR_FOLD_SHARED_BYTES,
                                function_attributes=(),
                                arguments=final_arguments,
                                source_tasks=canonical.task_ids,
                                source_phase="merge",
                            ),
                        )
                    )
                    image_by_launch.extend((image_index, image_index))
                    continue
                arguments = tuple(
                    CutePipelineSlotArgument(
                        name=f"argument_{index}",
                        slot=slots[value_ordinals[value]].name,
                        access=(
                            CutePipelineSlotAccess.READ
                            if index < len(operation_inputs)
                            else CutePipelineSlotAccess.WRITE
                        ),
                    )
                    for index, value in enumerate(values)
                )
                launches.append(
                    CuteKernelLaunch(
                        name=f"operation_{primitive.ordinal}",
                        kernel_symbol=primitive.kernel_names[0],
                        grid=primitive.launch_grid,
                        block=primitive.launch_block,
                        dynamic_shared_memory_bytes=primitive.shared_memory_bytes,
                        function_attributes=(),
                        arguments=arguments,
                        source_tasks=canonical.task_ids,
                    )
                )
                image_by_launch.append(image_index)
        fragments.append(
            CuteKernelComponentFragment(
                component_id=component.component_id,
                schedule_fingerprint=canonical.schedule_fingerprint,
                slots_by_value_id=slots,
                launches=tuple(launches),
                private_slots=tuple(private_slots),
            )
        )
    recipe = compose_cute_pipeline_launch_recipe(compiled.composition, tuple(fragments))
    request = composed_cute_compile_request(compiled.composition, recipe)
    request = replace(
        request, schedule={**request.schedule, "reduction_phase_schedules": tuple(reduction_phase_schedules)}
    )
    return CuteMultiImagePipelineArtifact(
        request=request,
        images=tuple(images),
        launch_recipe=recipe,
        image_by_launch=tuple(image_by_launch),
    )


def export_generic_primitive_artifact(
    primitive: CompiledGenericPrimitive,
    program: TensorProgram,
    *,
    function_prefix: str,
) -> CutePipelineArtifact:
    """Export one compiled primitive with its program identity and device image."""
    executable = primitive.executable
    with tempfile.TemporaryDirectory(prefix="shuttle-generic-primitive-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    request = CutePipelineCompileRequest(
        schema_version=1,
        target="sm90",
        numerical_policy=program.numerical_policy.value,
        physical_template="generic_tensor_primitive_simt_v1",
        tensors=tuple(
            {"shape": value.shape, "dtype": value.dtype.value}
            for value in (*primitive_inputs(primitive.operation), *primitive_outputs(primitive.operation))
        )
        + (
            ()
            if primitive.private_scratch is None
            else ({"shape": tuple(primitive.private_scratch.shape), "dtype": DType.FP32.value},)
        ),
        scalar_programs={},
        index_relations=(),
        contraction_dimensions={},
        schedule={
            "threads": _THREADS,
            "operation_ordinal": primitive.ordinal,
            **(primitive.bf16_tree_plan.canonical_record() if primitive.bf16_tree_plan is not None else {}),
            **(primitive.row_fold_plan.canonical_record() if primitive.row_fold_plan is not None else {}),
            **(
                {}
                if primitive.private_scratch is None
                else {
                    "kind": "hierarchical_all_axis_fold",
                    "partial_tile_elements": _THREADS * _SCALAR_FOLD_ITEMS_PER_THREAD,
                    "partial_count": primitive.private_scratch.numel(),
                    "final_grid": (1,),
                }
            ),
        },
        algebra_identity={"program_fingerprint": program.semantic_fingerprint},
    )
    return CutePipelineArtifact(
        request=request,
        host_object=host_object,
        device_image=elf64_symbol_bytes(host_object, "kernels_binary"),
        kernel_names=primitive.kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={"emitter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        structured_device_ir=(primitive.structured_device_ir,),
    )


def _export_fused_component_artifact(
    component: CompiledGenericComponent,
    program: TensorProgram,
    *,
    function_prefix: str,
) -> CutePipelineArtifact:
    executable = component.fused_executable
    snapshot = component.structured_device_ir
    if executable is None or snapshot is None:
        raise ValueError("cannot export an unfused canonical component as a fused image")
    operation_records = program.canonical_record["operations"]
    with tempfile.TemporaryDirectory(prefix="shuttle-generic-component-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    request = CutePipelineCompileRequest(
        schema_version=1,
        target="sm90",
        numerical_policy=program.numerical_policy.value,
        physical_template="generic_dependency_convex_component_simt_v1",
        tensors=tuple(
            {"shape": value.shape, "dtype": value.dtype.value}
            for value in (*component.live_inputs, *component.live_outputs)
        ),
        scalar_programs={
            f"operation_{ordinal}": operation_records[ordinal]["expression"]
            for ordinal, operation in zip(
                component.operation_ordinals,
                (program.operations[index] for index in component.operation_ordinals),
                strict=True,
            )
            if isinstance(operation, MapPrimitive)
        },
        index_relations=tuple(
            operation_records[ordinal]
            for ordinal in component.operation_ordinals
            for operation in (program.operations[ordinal],)
            if isinstance(operation, (ContractPrimitive, RelationPrimitive))
            or (isinstance(operation, MapPrimitive) and operation.predicate is not None)
        ),
        contraction_dimensions={},
        schedule={
            "threads": _THREADS,
            "component_id": component.component_id,
            "operation_ordinals": component.operation_ordinals,
            "live_input_count": len(component.live_inputs),
            "live_output_count": len(component.live_outputs),
            "maximum_live_inputs": _MAXIMUM_FUSED_LIVE_INPUTS,
            "maximum_live_outputs": _MAXIMUM_FUSED_LIVE_OUTPUTS,
            "abi_realization": "bounded_padded_private_slots",
            "evaluation": executable.evaluation.value,
            **(executable.row_fold_plan.canonical_record() if executable.row_fold_plan is not None else {}),
        },
        algebra_identity={"program_fingerprint": program.semantic_fingerprint},
    )
    return CutePipelineArtifact(
        request=request,
        host_object=host_object,
        device_image=elf64_symbol_bytes(host_object, "kernels_binary"),
        kernel_names=component.kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={"emitter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        structured_device_ir=(snapshot,),
    )


def _compile_primitive(
    ordinal: int,
    operation: TensorPrimitive,
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    *,
    numerical_policy: NumericalPolicy,
) -> CompiledGenericPrimitive:
    require_generic_codegen_admission(((ordinal, operation),))
    private_scratch = None
    tree_plan = None
    row_plan = plan_fp32_row_fold(
        (operation,), primitive_inputs(operation), primitive_outputs(operation), numerical_policy
    )
    emitter: Any
    if isinstance(operation, MapPrimitive):
        if len(outputs) != 1 or len(inputs) > 3:
            raise ValueError(f"generic CuTe Map {ordinal} currently requires zero to three tensor inputs")
        emitter = (_NullaryMapEmitter, _UnaryMapEmitter, _BinaryMapEmitter, _TernaryMapEmitter)[len(inputs)](operation)
    elif isinstance(operation, FoldPrimitive):
        if requires_bf16_tree_fold(operation, numerical_policy):
            tree_plan = Bf16TreeFoldPlan(operation, numerical_policy)
        if tree_plan is not None:
            emitter = _Bf16TreeFoldEmitter(tree_plan)
        elif row_plan is not None:
            emitter = _Fp32RowFoldEmitter(row_plan)
        elif (
            not operation.output.shape
            and operation.reducer is FoldReducer.SUM
            and operation.accumulation_dtype is DType.FP32
            and operation.order is EvaluationOrder.ORDER_FREE
            and numerical_policy is NumericalPolicy.FAST
        ):
            emitter = _HierarchicalScalarSumFoldEmitter(operation)
        else:
            emitter = _FoldEmitter(operation)
    elif isinstance(operation, RelationPrimitive):
        if len(outputs) != 1 or len(inputs) > 1:
            raise ValueError(f"generic CuTe Relation {ordinal} currently requires at most one input and one output")
        emitter = _SourceRelationEmitter(operation) if inputs else _GeneratedRelationEmitter(operation)
    else:
        raise ValueError(
            f"generic CuTe code generation does not support {type(operation).__name__} at ordinal {ordinal}"
        )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    boundary_tensors = (*inputs, *outputs)
    if isinstance(emitter, _HierarchicalScalarSumFoldEmitter):
        private_scratch = torch.empty(
            (emitter.partial_elements,),
            dtype=torch.float32,
            device=inputs[0].device,
        )
        boundary_tensors = (*boundary_tensors, private_scratch)
    # Shape and strides belong to the compiled TensorProgram. Keeping the CuTe
    # layout static makes each exported tensor argument one device pointer.
    arguments = tuple(
        cute.runtime.from_dlpack(_cute_tensor_boundary(tensor), assumed_align=16, enable_tvm_ffi=True)
        for tensor in boundary_tensors
    )
    executable, snapshot = compile_cute_with_structured_snapshot(
        cute,
        emitter,
        *arguments,
        fake_stream,
        options="--enable-tvm-ffi",
    )
    expected_kernels = 2 if isinstance(emitter, _HierarchicalScalarSumFoldEmitter) else 1
    if len(executable.kernel_info) != expected_kernels:
        raise ValueError(f"generic primitive {ordinal} must compile to exactly {expected_kernels} kernel(s)")
    return CompiledGenericPrimitive(ordinal, operation, executable, snapshot, private_scratch, tree_plan, row_plan)


def _cute_tensor_boundary(tensor: torch.Tensor) -> torch.Tensor:
    """Represent scalar storage as a one-element CuTe tensor at the ABI edge."""
    return tensor.reshape(1) if tensor.ndim == 0 else tensor


def _compile_fused_component(
    operations: tuple[TensorPrimitive, ...],
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    *,
    evaluation: FusedComponentEvaluation = FusedComponentEvaluation.RECURSIVE,
    numerical_policy: NumericalPolicy,
) -> tuple[Any, CuteDeviceIrSnapshot]:
    require_generic_codegen_admission(enumerate(operations))
    if any(requires_bf16_tree_fold(operation, numerical_policy) for operation in operations):
        raise ValueError("BF16 tree Fold requires a materialized primitive boundary")
    if not operations:
        raise ValueError("generic fused component requires algebra operations")
    produced = {value for operation in operations for value in primitive_outputs(operation)}
    if any(value not in produced for value in live_outputs):
        raise ValueError("generic fused component outputs must be produced inside the component")
    if any(
        not isinstance(
            operation,
            (MapPrimitive, FoldPrimitive, RelationPrimitive),
        )
        for operation in operations
    ):
        raise ValueError("generic fused component contains an unsupported algebra primitive")
    if any(
        isinstance(operation, FoldPrimitive) and operation.accumulation_dtype not in (DType.BF16, DType.FP32)
        for operation in operations
    ):
        raise ValueError("generic fused component requires BF16 or FP32 Fold accumulation")
    producer_by_value = {value: operation for operation in operations for value in primitive_outputs(operation)}
    if max(_reduction_depth(value, producer_by_value) for value in live_outputs) > 1:
        raise ValueError("generic fused component resource model rejects nested Fold reductions")
    if len(live_inputs) > _MAXIMUM_FUSED_LIVE_INPUTS:
        raise ValueError(
            f"generic fused component supports at most {_MAXIMUM_FUSED_LIVE_INPUTS} live inputs by resource proof"
        )
    if len(live_outputs) > _MAXIMUM_FUSED_LIVE_OUTPUTS:
        raise ValueError(
            f"generic fused component supports at most {_MAXIMUM_FUSED_LIVE_OUTPUTS} live outputs by resource proof"
        )
    realized_evaluation = (
        FusedComponentEvaluation.SAME_COORDINATE_SSA
        if evaluation is FusedComponentEvaluation.SAME_COORDINATE_SSA
        and supports_same_coordinate_producer_reuse(operations, live_inputs, live_outputs)
        else FusedComponentEvaluation.RECURSIVE
    )
    row_plan = plan_fp32_row_fold(operations, live_inputs, live_outputs, numerical_policy)
    emitter = (
        _FusedFp32RowFoldEmitter(row_plan, realized_evaluation)
        if row_plan is not None
        else _BoundedFusedComponentEmitter(operations, live_inputs, live_outputs, realized_evaluation)
    )
    dummy = inputs[0] if inputs else outputs[0]
    padded_inputs = (*inputs, *(dummy for _ in range(_MAXIMUM_FUSED_LIVE_INPUTS - len(inputs))))
    padded_outputs = (*outputs, *(dummy for _ in range(_MAXIMUM_FUSED_LIVE_OUTPUTS - len(outputs))))
    arguments = tuple(
        cute.runtime.from_dlpack(_cute_tensor_boundary(tensor), assumed_align=16, enable_tvm_ffi=True)
        for tensor in (*padded_inputs, *padded_outputs)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    executable, snapshot = compile_cute_with_structured_snapshot(
        cute,
        emitter,
        *arguments,
        fake_stream,
        options="--enable-tvm-ffi",
    )
    if len(executable.kernel_info) != 1:
        raise ValueError("generic fused component must compile to exactly one kernel")
    return (
        _PaddedFusedExecutable(
            executable,
            len(inputs),
            len(outputs),
            realized_evaluation,
            row_plan.grid if row_plan is not None else ((emitter.elements + _THREADS - 1) // _THREADS, 1, 1),
            row_plan.block if row_plan is not None else (_THREADS, 1, 1),
            row_plan.shared_bytes if row_plan is not None else 0,
            row_plan,
        ),
        snapshot,
    )


def _reduction_depth(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> int:
    operation = producer_by_value.get(value)
    if operation is None:
        return 0
    own = int(isinstance(operation, FoldPrimitive))
    return own + max(
        (_reduction_depth(operand, producer_by_value) for operand in primitive_inputs(operation)),
        default=0,
    )


class _FusedComponentEmitter:
    def __init__(
        self,
        operations: tuple[TensorPrimitive, ...],
        live_inputs: tuple[ProgramValue, ...],
        outputs: tuple[ProgramValue, ...],
    ):
        self.outputs = outputs
        self.output = outputs[0]
        self.elements = max(math.prod(output.shape) for output in outputs)
        self.operations = operations
        self.live_inputs = live_inputs
        self.producer_by_value = {value: operation for operation in operations for value in primitive_outputs(operation)}

    def value(self, linear: Any, tensors: tuple[cute.Tensor, ...]) -> Any:
        coordinates = _coordinates(linear, self.output.axes)
        bindings = dict(zip(self.live_inputs, tensors, strict=True))
        return self._value(self.output, coordinates, bindings)

    def output_value(self, output_index: int, linear: Any, tensors: tuple[cute.Tensor, ...]) -> Any:
        output = self.outputs[output_index]
        coordinates = _coordinates(linear, output.axes)
        bindings = dict(zip(self.live_inputs, tensors, strict=True))
        return self._value(output, coordinates, bindings)

    def _value(
        self,
        value: ProgramValue,
        coordinates: tuple[Any, ...],
        bindings: dict[ProgramValue, cute.Tensor],
        same_coordinate_values: dict[ProgramValue, Any] | None = None,
    ) -> Any:
        if same_coordinate_values is not None and value in same_coordinate_values:
            return same_coordinate_values[value]
        tensor = bindings.get(value)
        if tensor is not None:
            return _flatten(tensor)[_linear_offset(coordinates, value.axes)]
        operation = self.producer_by_value.get(value)
        if operation is None:
            raise ValueError(f"generic fused component has no binding for value {value.name!r}")
        if isinstance(operation, MapPrimitive):
            scalar_bindings = {
                operand.name: self._value(
                    operand,
                    tuple(coordinates[value.axes.index(axis)] for axis in operand.axes),
                    bindings,
                    same_coordinate_values,
                )
                for operand in operation.inputs
            }
            mapped = evaluate_cute_scalar_expression(operation.expression, scalar_bindings)
            if operation.predicate is None:
                return _store(mapped, value.dtype)
            coordinate_by_axis = dict(zip(operation.output.axes, coordinates, strict=True))
            allowed = _predicate(operation.predicate, coordinate_by_axis)
            assert operation.rejected_value is not None
            accepted = _store(mapped, value.dtype)
            rejected = _store(operation.rejected_value, value.dtype)
            return accepted if allowed else rejected
        if isinstance(operation, FoldPrimitive):
            initial = float("-inf") if operation.reducer is FoldReducer.MAXIMUM else 0.0
            total = (
                cutlass.BFloat16(initial)
                if cutlass.const_expr(operation.accumulation_dtype is DType.BF16)
                else cutlass.Float32(initial)
            )
            reduction_elements = math.prod(axis.extent for axis in operation.reduction_axes)
            for reduction in range(reduction_elements):
                reduction_coordinates = _coordinates(reduction, operation.reduction_axes)
                input_coordinates = tuple(
                    (
                        coordinates[operation.output.axes.index(axis)]
                        if axis in operation.output.axes
                        else reduction_coordinates[operation.reduction_axes.index(axis)]
                    )
                    for axis in operation.input.axes
                )
                item = (
                    cutlass.BFloat16(self._value(operation.input, input_coordinates, bindings))
                    if cutlass.const_expr(operation.accumulation_dtype is DType.BF16)
                    else cutlass.Float32(self._value(operation.input, input_coordinates, bindings))
                )
                if cutlass.const_expr(operation.reducer is FoldReducer.SUM):
                    total = (
                        cutlass.BFloat16(total + item)
                        if cutlass.const_expr(operation.accumulation_dtype is DType.BF16)
                        else total + item
                    )
                elif item > total:
                    total = item
            return _store(total, value.dtype)
        if isinstance(operation, RelationPrimitive):
            output_index = operation.outputs.index(value)
            expression = operation.resolved_result_expressions[output_index]
            coordinate_by_axis = dict(zip(operation.domain_axes, coordinates, strict=True))
            if expression is not None:
                generated = (
                    _predicate(expression, coordinate_by_axis)
                    if isinstance(expression, IndexPredicate)
                    else _index_expression(expression, coordinate_by_axis)
                )
                return _store(generated, value.dtype)
            source = operation.resolved_result_sources[output_index]
            if source is None:
                raise ValueError("generic fused Relation result has neither a source nor an expression")
            source_index = operation.inputs.index(source)
            indexing = operation.input_indexing[source_index]
            indices = tuple(_index_expression(index, coordinate_by_axis) for index in indexing.indices)
            if not _predicate(indexing.predicate, coordinate_by_axis):
                return _store(0, value.dtype)
            return _store(self._value(source, indices, bindings), value.dtype)
        raise AssertionError(f"unsupported fused operation {type(operation).__name__}")

    def launch(self, kernel: Any, arguments: tuple[Any, ...], stream: cuda.CUstream) -> None:
        kernel(*arguments).launch(
            grid=[cute.ceil_div(self.elements, _THREADS), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )


class _BoundedFusedComponentEmitter(_FusedComponentEmitter):
    def __init__(
        self,
        operations: tuple[TensorPrimitive, ...],
        live_inputs: tuple[ProgramValue, ...],
        live_outputs: tuple[ProgramValue, ...],
        evaluation: FusedComponentEvaluation,
    ):
        super().__init__(operations, live_inputs, live_outputs)
        self.evaluation = evaluation

    def shared_output_values(self, linear: Any, tensors: tuple[cute.Tensor, ...]) -> tuple[Any, ...]:
        coordinates = _coordinates(linear, self.outputs[0].axes)
        bindings = dict(zip(self.live_inputs, tensors, strict=True))
        scalar_values: dict[ProgramValue, Any] = {}
        for operation in self.operations:
            output = primitive_outputs(operation)[0]
            scalar_values[output] = self._value(output, coordinates, bindings, scalar_values)
        return tuple(scalar_values[output] for output in self.outputs)

    @cute.jit
    def __call__(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        third: cute.Tensor,
        fourth: cute.Tensor,
        fifth: cute.Tensor,
        sixth: cute.Tensor,
        seventh: cute.Tensor,
        eighth: cute.Tensor,
        first_output: cute.Tensor,
        second_output: cute.Tensor,
        third_output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.launch(
            self.kernel,
            (
                first,
                second,
                third,
                fourth,
                fifth,
                sixth,
                seventh,
                eighth,
                first_output,
                second_output,
                third_output,
            ),
            stream,
        )

    @cute.kernel
    def kernel(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        third: cute.Tensor,
        fourth: cute.Tensor,
        fifth: cute.Tensor,
        sixth: cute.Tensor,
        seventh: cute.Tensor,
        eighth: cute.Tensor,
        first_output: cute.Tensor,
        second_output: cute.Tensor,
        third_output: cute.Tensor,
    ):
        inputs = (first, second, third, fourth, fifth, sixth, seventh, eighth)[: len(self.live_inputs)]
        outputs = (first_output, second_output, third_output)[: len(self.outputs)]
        linear = _thread_linear()
        if cutlass.const_expr(self.evaluation is FusedComponentEvaluation.SAME_COORDINATE_SSA):
            if linear < math.prod(self.outputs[0].shape):
                values = self.shared_output_values(linear, inputs)
                for output_index in cutlass.range_constexpr(len(outputs)):
                    _flatten(outputs[output_index])[linear] = _store(
                        values[output_index], self.outputs[output_index].dtype
                    )
        else:
            for output_index in cutlass.range_constexpr(len(outputs)):
                output = _flatten(outputs[output_index])
                if linear < math.prod(self.outputs[output_index].shape):
                    output[linear] = _store(
                        self.output_value(output_index, linear, inputs),
                        self.outputs[output_index].dtype,
                    )


class _FusedFp32RowFoldEmitter(_BoundedFusedComponentEmitter):
    def __init__(self, plan: Fp32RowFoldPlan, evaluation: FusedComponentEvaluation):
        super().__init__(plan.operations, plan.live_inputs, plan.live_outputs, evaluation)
        self.plan = plan

    def launch(self, kernel: Any, arguments: tuple[Any, ...], stream: cuda.CUstream) -> None:
        kernel(*arguments).launch(
            grid=self.plan.grid,
            block=self.plan.block,
            smem=self.plan.shared_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        third: cute.Tensor,
        fourth: cute.Tensor,
        fifth: cute.Tensor,
        sixth: cute.Tensor,
        seventh: cute.Tensor,
        eighth: cute.Tensor,
        first_output: cute.Tensor,
        second_output: cute.Tensor,
        third_output: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        inputs = (first, second, third, fourth, fifth, sixth, seventh, eighth)[: len(self.live_inputs)]
        bindings = dict(zip(self.live_inputs, inputs, strict=True))
        total = cutlass.Float32(0.0)
        for iteration in cutlass.range(self.plan.items_per_thread, unroll=1):
            reduction = thread + iteration * _THREADS
            if reduction < self.plan.reduction_elements:
                linear = block * self.plan.reduction_elements + reduction
                coordinates = _coordinates(linear, self.plan.fold.input.axes)
                # _value applies each source Map conversion before accumulation.
                total += cutlass.Float32(self._value(self.plan.fold.input, coordinates, bindings))
        total = _scalar_fold_block_sum(total)
        if thread == 0:
            _flatten(first_output)[block] = total


class _Emitter:
    def __init__(self, output: ProgramValue):
        self.output = output
        self.elements = math.prod(output.shape)

    def launch(self, kernel: Any, arguments: tuple[Any, ...], stream: cuda.CUstream) -> None:
        kernel(*arguments).launch(
            grid=[cute.ceil_div(self.elements, _THREADS), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )


class _MapEmitter(_Emitter):
    def __init__(self, operation: MapPrimitive):
        super().__init__(operation.output)
        self.operation = operation

    def value(self, linear: Any, inputs: tuple[cute.Tensor, ...]) -> Any:
        coordinates = _coordinates(linear, self.operation.output.axes)
        bindings = {
            value.name: _flatten(tensor)[_broadcast_offset(coordinates, self.operation.output.axes, value.axes)]
            for value, tensor in zip(self.operation.inputs, inputs, strict=True)
        }
        mapped = evaluate_cute_scalar_expression(self.operation.expression, bindings)
        if self.operation.predicate is None:
            return mapped
        coordinate_by_axis = dict(zip(self.operation.output.axes, coordinates, strict=True))
        allowed = _predicate(self.operation.predicate, coordinate_by_axis)
        assert self.operation.rejected_value is not None
        return mapped if allowed else self.operation.rejected_value


class _NullaryMapEmitter(_MapEmitter):
    @cute.jit
    def __call__(self, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (output,), stream)

    @cute.kernel
    def kernel(self, output: cute.Tensor):
        linear = _thread_linear()
        flat_output = _flatten(output)
        if linear < self.elements:
            flat_output[linear] = _store(self.value(linear, ()), self.operation.output.dtype)


class _UnaryMapEmitter(_MapEmitter):
    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (source, output), stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        linear = _thread_linear()
        flat_output = _flatten(output)
        if linear < self.elements:
            flat_output[linear] = _store(self.value(linear, (source,)), self.operation.output.dtype)


class _BinaryMapEmitter(_MapEmitter):
    @cute.jit
    def __call__(self, first: cute.Tensor, second: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (first, second, output), stream)

    @cute.kernel
    def kernel(self, first: cute.Tensor, second: cute.Tensor, output: cute.Tensor):
        linear = _thread_linear()
        flat_output = _flatten(output)
        if linear < self.elements:
            flat_output[linear] = _store(self.value(linear, (first, second)), self.operation.output.dtype)


class _TernaryMapEmitter(_MapEmitter):
    @cute.jit
    def __call__(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        third: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.launch(self.kernel, (first, second, third, output), stream)

    @cute.kernel
    def kernel(self, first: cute.Tensor, second: cute.Tensor, third: cute.Tensor, output: cute.Tensor):
        linear = _thread_linear()
        flat_output = _flatten(output)
        if linear < self.elements:
            flat_output[linear] = _store(self.value(linear, (first, second, third)), self.operation.output.dtype)


class _FoldEmitter(_Emitter):
    def __init__(self, operation: FoldPrimitive):
        if operation.accumulation_dtype not in (DType.BF16, DType.FP32):
            raise ValueError("generic CuTe Fold currently requires BF16 or FP32 accumulation")
        super().__init__(operation.output)
        self.operation = operation
        self.reduction_elements = math.prod(axis.extent for axis in operation.reduction_axes)

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (source, output), stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        linear = _thread_linear()
        source = _flatten(source)
        output = _flatten(output)
        if linear < self.elements:
            output_coordinates = _coordinates(linear, self.operation.output.axes)
            initial = float("-inf") if self.operation.reducer is FoldReducer.MAXIMUM else 0.0
            total = (
                cutlass.BFloat16(initial)
                if cutlass.const_expr(self.operation.accumulation_dtype is DType.BF16)
                else cutlass.Float32(initial)
            )
            for reduction in cutlass.range_constexpr(self.reduction_elements):
                reduction_coordinates = _coordinates(reduction, self.operation.reduction_axes)
                input_coordinates = tuple(
                    (
                        output_coordinates[self.operation.output.axes.index(axis)]
                        if axis in self.operation.output.axes
                        else reduction_coordinates[self.operation.reduction_axes.index(axis)]
                    )
                    for axis in self.operation.input.axes
                )
                value = (
                    cutlass.BFloat16(source[_linear_offset(input_coordinates, self.operation.input.axes)])
                    if cutlass.const_expr(self.operation.accumulation_dtype is DType.BF16)
                    else cutlass.Float32(source[_linear_offset(input_coordinates, self.operation.input.axes)])
                )
                if cutlass.const_expr(self.operation.reducer is FoldReducer.SUM):
                    total = (
                        cutlass.BFloat16(total + value)
                        if cutlass.const_expr(self.operation.accumulation_dtype is DType.BF16)
                        else total + value
                    )
                else:
                    if value > total:
                        total = value
            output[linear] = _store(total, self.operation.output.dtype)


class _Bf16TreeFoldEmitter:
    def __init__(self, plan: Bf16TreeFoldPlan):
        self.plan = plan

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(source, output).launch(
            grid=self.plan.grid,
            block=self.plan.block,
            smem=self.plan.shared_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        source = _flatten(source)
        output = _flatten(output)
        shared = SmemAllocator().allocate_tensor(
            cutlass.BFloat16, cute.make_layout(self.plan.shared_elements), byte_alignment=2
        )
        tile = self.plan.outputs_per_cta
        count = self.plan.reduction_elements
        elements = count * tile
        for iteration in cutlass.range((elements + _THREADS - 1) // _THREADS, unroll=1):
            index = thread + iteration * _THREADS
            reduction = index // tile
            result = block * tile + index % tile
            if index < elements and result < self.plan.output_elements:
                output_coordinates = _coordinates(result, self.plan.operation.output.axes)
                reduction_coordinates = _coordinates(reduction, self.plan.operation.reduction_axes)
                offset = sum(
                    coordinate * stride
                    for coordinate, stride in zip(output_coordinates, self.plan.source_output_strides, strict=True)
                )
                offset += sum(
                    coordinate * stride
                    for coordinate, stride in zip(reduction_coordinates, self.plan.source_reduction_strides, strict=True)
                )
                shared[index] = source[offset]
        cute.arch.sync_threads()
        for level in cutlass.range_constexpr(self.plan.depth):
            distance = 1 << level
            pairs = (count + 2 * distance - 1) // (2 * distance)
            work = pairs * tile
            for iteration in cutlass.range((work + _THREADS - 1) // _THREADS, unroll=1):
                index = thread + iteration * _THREADS
                left = (index // tile) * (2 * distance)
                result = block * tile + index % tile
                if index < work and result < self.plan.output_elements and left + distance < count:
                    first = left * tile + index % tile
                    second = first + distance * tile
                    shared[first] = cutlass.BFloat16(cutlass.Float32(shared[first]) + cutlass.Float32(shared[second]))
            # Every CTA thread participates, including threads outside the final output tile.
            cute.arch.sync_threads()
        if thread < tile:
            result = block * tile + thread
            if result < self.plan.output_elements:
                output[result] = shared[thread]


@cute.jit
def _scalar_fold_block_sum(total):
    thread, _, _ = cute.arch.thread_idx()
    warp_totals = SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(_SCALAR_FOLD_WARPS), byte_alignment=4
    )
    total = utils.warp_reduce(total, operator.add)
    if cute.arch.lane_idx() == 0:
        warp_totals[thread // 32] = total
    cute.arch.sync_threads()
    result = cutlass.Float32(0.0)
    if thread == 0:
        # A single writer merges warp totals in fixed order, independent of warp scheduling.
        for warp in cutlass.range_constexpr(_SCALAR_FOLD_WARPS):
            result += warp_totals[warp]
    return result


class _Fp32RowFoldEmitter:
    def __init__(self, plan: Fp32RowFoldPlan):
        self.plan = plan

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(source, output).launch(
            grid=self.plan.grid,
            block=self.plan.block,
            smem=self.plan.shared_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        source = _flatten(source)
        total = cutlass.Float32(0.0)
        for iteration in cutlass.range(self.plan.items_per_thread, unroll=1):
            reduction = thread + iteration * _THREADS
            if reduction < self.plan.reduction_elements:
                total += cutlass.Float32(source[block * self.plan.reduction_elements + reduction])
        total = _scalar_fold_block_sum(total)
        if thread == 0:
            _flatten(output)[block] = total


class _HierarchicalScalarSumFoldEmitter(_FoldEmitter):
    """Multi-CTA partial reduction followed by one bounded final reduction."""

    def __init__(self, operation: FoldPrimitive):
        super().__init__(operation)
        if operation.order is not EvaluationOrder.ORDER_FREE:
            raise ValueError("hierarchical scalar SUM requires an order-free Fold")
        if tuple(operation.reduction_axes) != tuple(operation.input.axes):
            raise ValueError("hierarchical scalar SUM requires every input axis in physical storage order")
        self.tile_elements = _THREADS * _SCALAR_FOLD_ITEMS_PER_THREAD
        self.partial_elements = math.ceil(self.reduction_elements / self.tile_elements)
        final_items_per_thread = math.ceil(self.partial_elements / _THREADS)
        if final_items_per_thread > _SCALAR_FOLD_FINAL_ITEMS_PER_THREAD_LIMIT:
            raise ValueError(
                "hierarchical scalar SUM final level exceeds its per-thread resource bound: "
                f"{final_items_per_thread} > {_SCALAR_FOLD_FINAL_ITEMS_PER_THREAD_LIMIT}"
            )

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        output: cute.Tensor,
        partials: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.partial_kernel(source, partials).launch(
            grid=[self.partial_elements, 1, 1],
            block=[_THREADS, 1, 1],
            smem=_SCALAR_FOLD_SHARED_BYTES,
            stream=stream,
        )
        self.final_kernel(partials, output).launch(
            grid=[1, 1, 1],
            block=[_THREADS, 1, 1],
            smem=_SCALAR_FOLD_SHARED_BYTES,
            stream=stream,
        )

    @cute.kernel
    def partial_kernel(self, source: cute.Tensor, partials: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        source = _flatten(source)
        partials = _flatten(partials)
        total = cutlass.Float32(0.0)
        begin = block * self.tile_elements + thread
        for item in cutlass.range_constexpr(_SCALAR_FOLD_ITEMS_PER_THREAD):
            index = begin + item * _THREADS
            if index < self.reduction_elements:
                total += cutlass.Float32(source[index])
        total = _scalar_fold_block_sum(total)
        if thread == 0:
            partials[block] = total

    @cute.kernel
    def final_kernel(self, partials: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        partials = _flatten(partials)
        output = _flatten(output)
        total = cutlass.Float32(0.0)
        for index in cutlass.range(thread, self.partial_elements, _THREADS, unroll=1):
            total += cutlass.Float32(partials[index])
        total = _scalar_fold_block_sum(total)
        if thread == 0:
            output[0] = total


class _SourceRelationEmitter(_Emitter):
    def __init__(self, operation: RelationPrimitive):
        super().__init__(operation.outputs[0])
        self.operation = operation
        if operation.resolved_result_sources != (operation.inputs[0],):
            raise ValueError("generic source Relation requires its single indexed input as the result source")

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (source, output), stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor):
        linear = _thread_linear()
        source = _flatten(source)
        output = _flatten(output)
        if linear < self.elements:
            coordinates = _coordinates(linear, self.operation.domain_axes)
            coordinate_by_axis = dict(zip(self.operation.domain_axes, coordinates, strict=True))
            indexing = self.operation.input_indexing[0]
            indices = tuple(_index_expression(index, coordinate_by_axis) for index in indexing.indices)
            allowed = _predicate(indexing.predicate, coordinate_by_axis)
            value = source[_linear_offset(indices, indexing.operand.axes)]
            output[linear] = _store(value if allowed else 0, self.output.dtype)


class _GeneratedRelationEmitter(_Emitter):
    def __init__(self, operation: RelationPrimitive):
        super().__init__(operation.outputs[0])
        self.operation = operation
        if operation.resolved_result_expressions[0] is None:
            raise ValueError("generic source-free Relation requires a typed result expression")

    @cute.jit
    def __call__(self, output: cute.Tensor, stream: cuda.CUstream):
        self.launch(self.kernel, (output,), stream)

    @cute.kernel
    def kernel(self, output: cute.Tensor):
        linear = _thread_linear()
        output = _flatten(output)
        if linear < self.elements:
            coordinates = _coordinates(linear, self.operation.domain_axes)
            coordinate_by_axis = dict(zip(self.operation.domain_axes, coordinates, strict=True))
            expression = self.operation.resolved_result_expressions[0]
            assert expression is not None
            value = (
                _predicate(expression, coordinate_by_axis)
                if isinstance(expression, IndexPredicate)
                else _index_expression(expression, coordinate_by_axis)
            )
            output[linear] = _store(value, self.output.dtype)


def _thread_linear() -> Any:
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    return block * _THREADS + thread


def _flatten(tensor: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))


def _coordinates(linear: Any, axes: tuple[TensorAxis, ...]) -> tuple[Any, ...]:
    coordinates: list[Any] = []
    residual = linear
    for axis in reversed(axes):
        coordinates.append(residual % axis.extent)
        residual = residual // axis.extent
    return tuple(reversed(coordinates))


def _linear_offset(coordinates: tuple[Any, ...], axes: tuple[TensorAxis, ...]) -> Any:
    offset: Any = 0
    for coordinate, axis in zip(coordinates, axes, strict=True):
        offset = offset * axis.extent + coordinate
    return offset


def _broadcast_offset(
    output_coordinates: tuple[Any, ...],
    output_axes: tuple[TensorAxis, ...],
    input_axes: tuple[TensorAxis, ...],
) -> Any:
    by_axis = dict(zip(output_axes, output_coordinates, strict=True))
    return _linear_offset(tuple(by_axis[axis] for axis in input_axes), input_axes)


def _index_expression(expression: IndexExpression, coordinates: dict[TensorAxis, Any]) -> Any:
    kind = expression.kind
    if kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return coordinates[expression.axis]
    if kind is IndexExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    if kind is IndexExpressionKind.TENSOR_ELEMENT:
        raise ValueError("generic CuTe index expressions do not yet support tensor-element reads")
    if kind is IndexExpressionKind.SELECT:
        assert expression.predicate is not None
        return (
            _index_expression(expression.operands[0], coordinates)
            if _predicate(expression.predicate, coordinates)
            else _index_expression(expression.operands[1], coordinates)
        )
    left = _index_expression(expression.operands[0], coordinates)
    right = _index_expression(expression.operands[1], coordinates)
    if kind is IndexExpressionKind.ADD:
        return left + right
    if kind is IndexExpressionKind.SUBTRACT:
        return left - right
    if kind is IndexExpressionKind.MULTIPLY:
        return left * right
    if kind is IndexExpressionKind.FLOOR_DIVIDE:
        return left // right
    if kind is IndexExpressionKind.MODULO:
        return left % right
    if kind is IndexExpressionKind.MINIMUM:
        return left if left <= right else right
    if kind is IndexExpressionKind.MAXIMUM:
        return left if left >= right else right
    raise ValueError(f"unsupported generic CuTe index expression {kind.value}")


def _predicate(predicate: IndexPredicate | None, coordinates: dict[TensorAxis, Any]) -> Any:
    if predicate is None:
        return True
    kind = predicate.kind
    if kind is IndexPredicateKind.CONSTANT:
        assert predicate.constant is not None
        return predicate.constant
    if kind is IndexPredicateKind.IS_TRUE:
        return _index_expression(predicate.expressions[0], coordinates)
    if kind is IndexPredicateKind.NOT:
        return not _predicate(predicate.predicates[0], coordinates)
    if kind is IndexPredicateKind.AND:
        return _predicate(predicate.predicates[0], coordinates) and _predicate(predicate.predicates[1], coordinates)
    if kind is IndexPredicateKind.OR:
        return _predicate(predicate.predicates[0], coordinates) or _predicate(predicate.predicates[1], coordinates)
    left = _index_expression(predicate.expressions[0], coordinates)
    right = _index_expression(predicate.expressions[1], coordinates)
    if kind is IndexPredicateKind.EQUAL:
        return left == right
    if kind is IndexPredicateKind.NOT_EQUAL:
        return left != right
    if kind is IndexPredicateKind.LESS:
        return left < right
    if kind is IndexPredicateKind.LESS_EQUAL:
        return left <= right
    if kind is IndexPredicateKind.GREATER:
        return left > right
    if kind is IndexPredicateKind.GREATER_EQUAL:
        return left >= right
    raise ValueError(f"unsupported generic CuTe index predicate {kind.value}")


def _store(value: Any, dtype: DType) -> Any:
    if dtype is DType.BF16:
        return cutlass.BFloat16(value)
    if dtype is DType.FP32:
        return cutlass.Float32(value)
    if dtype is DType.INT32:
        return cutlass.Int32(value)
    if dtype is DType.BOOL:
        return cutlass.Boolean(value)
    raise ValueError(f"unsupported generic CuTe storage dtype {dtype.value}")


def _torch_dtype(dtype: DType) -> torch.dtype:
    return {
        DType.BF16: torch.bfloat16,
        DType.FP32: torch.float32,
        DType.INT32: torch.int32,
        DType.BOOL: torch.bool,
    }[dtype]


def _pipeline_element_type(dtype: DType) -> CutePipelineElementType:
    return {
        DType.BF16: CutePipelineElementType.BF16,
        DType.FP32: CutePipelineElementType.F32,
        DType.INT32: CutePipelineElementType.I32,
        DType.BOOL: CutePipelineElementType.PRED,
    }[dtype]


def _allocate(value: ProgramValue, device: torch.device) -> torch.Tensor:
    return torch.empty(value.shape or (), dtype=_torch_dtype(value.dtype), device=device)


def _validate_tensor(value: ProgramValue, tensor: torch.Tensor) -> None:
    if tuple(tensor.shape) != value.shape or tensor.dtype is not _torch_dtype(value.dtype):
        raise ValueError(f"tensor {value.name!r} does not match its typed algebra boundary")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"tensor {value.name!r} must be contiguous CUDA storage")
