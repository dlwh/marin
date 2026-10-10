# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable SM90 SIMT baseline for generic Relation-driven tensor programs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, cast

from shuttle.ir import DType
from tile_lifetime.backends.sm100.canonical_simt_emitter import render_canonical_simt_kernel
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.cuda_map_fold_codegen import render_cuda_scalar_expression
from tile_lifetime.physical_task_contract import (
    ExecutablePhysicalRequest,
    PhysicalTarget,
    TaskOperationKind,
    canonical_fingerprint,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    SelectionPrimitive,
    TensorProgram,
    deserialize_tensor_program,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)

_THREADS = 256


@dataclass(frozen=True)
class Sm90SimtLaunch:
    """One source-ordered primitive launch selected from generic structure."""

    operation_ordinal: int
    task_id: int
    symbol: str
    blocks: int
    threads: int
    argument_value_ordinals: tuple[int, ...]

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "operation_ordinal": self.operation_ordinal,
            "task_id": self.task_id,
            "symbol": self.symbol,
            "blocks": self.blocks,
            "threads": self.threads,
            "argument_value_ordinals": self.argument_value_ordinals,
        }


@dataclass(frozen=True)
class GeneratedSm90RelationSimt:
    """Generated baseline plus retained runtime-exact Event provenance."""

    request_fingerprint: str
    source: str
    launches: tuple[Sm90SimtLaunch, ...]
    staged_task_ids: tuple[int, ...]
    runtime_event_records: tuple[dict[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "request_fingerprint": self.request_fingerprint,
                "source_sha256": hashlib.sha256(self.source.encode()).hexdigest(),
                "launches": tuple(launch.canonical_record for launch in self.launches),
                "staged_task_ids": self.staged_task_ids,
                "runtime_events": self.runtime_event_records,
            }
        )


@dataclass(frozen=True)
class Sm90DependencyConvexLaunch:
    """One fused component launch or one explicitly retained primitive launch."""

    component_id: int
    task_ids: tuple[int, ...]
    operation_ordinals: tuple[int, ...]
    symbol: str
    blocks: int
    threads: int
    argument_value_ordinals: tuple[int, ...]
    fused: bool

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "task_ids": self.task_ids,
            "operation_ordinals": self.operation_ordinals,
            "symbol": self.symbol,
            "blocks": self.blocks,
            "threads": self.threads,
            "argument_value_ordinals": self.argument_value_ordinals,
            "fused": self.fused,
        }


@dataclass(frozen=True)
class GeneratedSm90DependencyConvexSimt:
    """Dependency-convex component code plus exact Event provenance."""

    request_fingerprint: str
    composition_fingerprint: str
    source: str
    launches: tuple[Sm90DependencyConvexLaunch, ...]
    staged_task_ids: tuple[int, ...]
    runtime_event_records: tuple[dict[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "request_fingerprint": self.request_fingerprint,
                "composition_fingerprint": self.composition_fingerprint,
                "source_sha256": hashlib.sha256(self.source.encode()).hexdigest(),
                "launches": tuple(launch.canonical_record for launch in self.launches),
                "staged_task_ids": self.staged_task_ids,
                "runtime_events": self.runtime_event_records,
            }
        )


def generate_sm90_relation_simt(request: ExecutablePhysicalRequest) -> GeneratedSm90RelationSimt:
    """Generate an ordered correctness baseline from task and index structure."""
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("generic SM90 SIMT generation requires an SM90 physical request")
    program = _tensor_program(request)
    supported = (MapPrimitive, RelationPrimitive, SelectionPrimitive, FoldPrimitive, ContractPrimitive)
    unsupported = tuple(
        (ordinal, type(operation).__name__)
        for ordinal, operation in enumerate(program.operations)
        if not isinstance(operation, supported)
    )
    if unsupported:
        raise ValueError(f"generic SM90 SIMT generation does not support operations {unsupported}")
    staged = tuple(
        task.id for task in request.task_plan.tasks if task.structure.principal_kind is TaskOperationKind.INPUT_STAGE
    )
    if not staged:
        raise ValueError("runtime-indexed external inputs require explicit staging tasks")
    staged_set = set(staged)
    if not any(
        dependence.source_task_id in staged_set
        and request.task_plan.tasks[dependence.target_task_id].structure.principal_kind is TaskOperationKind.RELATION
        for dependence in request.task_plan.dependence_programs
    ):
        raise ValueError("SM90 Relation baseline requires a runtime-exact staging-to-Relation edge")

    ordinals = tensor_program_value_ordinals(program)
    task_for_operation = {ordinal: task.id for task in request.task_plan.tasks for ordinal in task.operation_ordinals}
    identity = request.fingerprint[:16]
    headers = (
        "#include <cuda_bf16.h>",
        "#include <cuda_runtime.h>",
        "#include <stdint.h>",
        "#include <math.h>",
        "static __device__ __forceinline__ uint32_t shuttle_total_order_key(float value) {\n"
        "  const uint32_t bits = __float_as_uint(value);\n"
        "  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);\n"
        "}",
    )
    kernels: list[str] = []
    launches: list[Sm90SimtLaunch] = []
    host_launches: list[str] = []
    for operation_ordinal, operation in enumerate(program.operations):
        symbol = f"shuttle_sm90_relation_simt_{identity}_{operation_ordinal}"
        kernels.append(render_canonical_simt_kernel(symbol, operation, ordinals))
        values = (*primitive_inputs(operation), *primitive_outputs(operation))
        arguments = tuple(ordinals[value] for value in values)
        work_items = _work_items(operation)
        blocks = (work_items + _THREADS - 1) // _THREADS
        launches.append(
            Sm90SimtLaunch(
                operation_ordinal,
                task_for_operation[operation_ordinal],
                symbol,
                blocks,
                _THREADS,
                arguments,
            )
        )
        host_launches.append(
            f"  {symbol}<<<{blocks}, {_THREADS}, 0, stream>>>("
            + ", ".join(
                f"reinterpret_cast<{_cuda_type(value.dtype)}*>(arguments[{ordinals[value]}])" for value in values
            )
            + ");"
        )
    host_body = "\n".join(host_launches)
    host = f"""extern "C" int shuttle_sm90_relation_simt_run(void** arguments, cudaStream_t stream) {{
{host_body}
  return static_cast<int>(cudaGetLastError());
}}"""
    canonical_events = cast(tuple[dict[str, Any], ...], request.canonical_record["event_programs"])
    event_records = tuple(
        {
            "event": canonical_events[index],
            "dependence": (
                request.task_plan.dependence_programs[event.dependence_program_index].relation_program.canonical_record
            ),
            "realization": "single_stream_program_order_after_runtime_exact_instantiation",
        }
        for index, event in enumerate(request.event_programs)
    )
    return GeneratedSm90RelationSimt(
        request.fingerprint,
        "\n\n".join((*headers, *kernels, host)) + "\n",
        tuple(launches),
        staged,
        event_records,
    )


def generate_sm90_dependency_convex_simt(
    request: ExecutablePhysicalRequest,
    *,
    maximum_fused_operations: int = 16,
) -> GeneratedSm90DependencyConvexSimt:
    """Fuse structurally selected task components and retain primitive fallback."""
    if maximum_fused_operations < 2:
        raise ValueError("dependency-convex fusion requires at least two operations")
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("generic SM90 component generation requires an SM90 physical request")
    program = _tensor_program(request)
    composition = lower_canonical_kernel_composition(request)
    ordinals = tensor_program_value_ordinals(program)
    values_by_ordinal = {ordinal: value for value, ordinal in ordinals.items()}
    identity = canonical_fingerprint(
        {
            "request": request.fingerprint,
            "composition": composition.fingerprint,
            "maximum_fused_operations": maximum_fused_operations,
        }
    )[:16]
    headers = (
        "#include <cuda_bf16.h>",
        "#include <cuda_runtime.h>",
        "#include <stdint.h>",
        "#include <math.h>",
        "static __device__ __forceinline__ uint32_t shuttle_total_order_key(float value) {\n"
        "  const uint32_t bits = __float_as_uint(value);\n"
        "  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);\n"
        "}",
    )
    kernels: list[str] = []
    launches: list[Sm90DependencyConvexLaunch] = []
    host_launches: list[str] = []
    for component in composition.components:
        operations = tuple(program.operations[index] for index in component.operation_ordinals)
        live_inputs = tuple(values_by_ordinal[index] for index in component.live_input_value_ids)
        live_outputs = tuple(values_by_ordinal[index] for index in component.live_output_value_ids)
        can_fuse = (
            2 <= len(operations) <= maximum_fused_operations
            and bool(live_outputs)
            and all(isinstance(operation, (MapPrimitive, FoldPrimitive, ContractPrimitive)) for operation in operations)
            and all(_supports_dependency_convex_fusion(operation) for operation in operations)
        )
        if can_fuse:
            symbol = f"shuttle_sm90_dependency_convex_{identity}_{component.id}"
            kernels.append(
                _render_dependency_convex_kernel(
                    symbol,
                    operations,
                    live_inputs,
                    live_outputs,
                    ordinals,
                )
            )
            work_items = max(math.prod(value.shape) for value in live_outputs)
            blocks = (work_items + _THREADS - 1) // _THREADS
            arguments = (*live_inputs, *live_outputs)
            launches.append(
                Sm90DependencyConvexLaunch(
                    component.id,
                    component.task_ids,
                    component.operation_ordinals,
                    symbol,
                    blocks,
                    _THREADS,
                    tuple(ordinals[value] for value in arguments),
                    True,
                )
            )
            host_launches.append(_host_launch(symbol, blocks, arguments, ordinals))
            continue
        for operation_ordinal, operation in zip(component.operation_ordinals, operations, strict=True):
            symbol = f"shuttle_sm90_dependency_convex_{identity}_{component.id}_{operation_ordinal}"
            kernels.append(render_canonical_simt_kernel(symbol, operation, ordinals))
            arguments = (*primitive_inputs(operation), *primitive_outputs(operation))
            work_items = _work_items(operation)
            blocks = (work_items + _THREADS - 1) // _THREADS
            launches.append(
                Sm90DependencyConvexLaunch(
                    component.id,
                    component.task_ids,
                    (operation_ordinal,),
                    symbol,
                    blocks,
                    _THREADS,
                    tuple(ordinals[value] for value in arguments),
                    False,
                )
            )
            host_launches.append(_host_launch(symbol, blocks, arguments, ordinals))

    staged = tuple(
        task.id for task in request.task_plan.tasks if task.structure.principal_kind is TaskOperationKind.INPUT_STAGE
    )
    host_body = "\n".join(host_launches)
    host = f"""extern "C" int shuttle_sm90_dependency_convex_simt_run(void** arguments, cudaStream_t stream) {{
{host_body}
  return static_cast<int>(cudaGetLastError());
}}"""
    canonical_events = cast(tuple[dict[str, Any], ...], request.canonical_record["event_programs"])
    event_records = tuple(
        {
            "event": canonical_events[index],
            "dependence": (
                request.task_plan.dependence_programs[event.dependence_program_index].relation_program.canonical_record
            ),
            "realization": "component_launch_stream_order_after_runtime_exact_instantiation",
        }
        for index, event in enumerate(request.event_programs)
    )
    return GeneratedSm90DependencyConvexSimt(
        request.fingerprint,
        composition.fingerprint,
        "\n\n".join((*headers, *kernels, host)) + "\n",
        tuple(launches),
        staged,
        event_records,
    )


def _host_launch(
    symbol: str,
    blocks: int,
    values: tuple[ProgramValue, ...],
    ordinals: dict[ProgramValue, int],
) -> str:
    return (
        f"  {symbol}<<<{blocks}, {_THREADS}, 0, stream>>>("
        + ", ".join(f"reinterpret_cast<{_cuda_type(value.dtype)}*>(arguments[{ordinals[value]}])" for value in values)
        + ");"
    )


def _supports_dependency_convex_fusion(
    operation: MapPrimitive | FoldPrimitive | ContractPrimitive,
) -> bool:
    if isinstance(operation, ContractPrimitive):
        return operation.accumulation_dtype is DType.FP32
    if isinstance(operation, FoldPrimitive):
        return operation.accumulation_dtype in {DType.BF16, DType.FP32}
    return True


def _render_dependency_convex_kernel(
    symbol: str,
    operations: tuple[MapPrimitive | FoldPrimitive | ContractPrimitive, ...],
    live_inputs: tuple[ProgramValue, ...],
    live_outputs: tuple[ProgramValue, ...],
    ordinals: dict[ProgramValue, int],
) -> str:
    producer = {value: operation for operation in operations for value in primitive_outputs(operation)}
    renderer = _DependencyConvexValueRenderer(live_inputs, producer, ordinals)
    functions = tuple(renderer.function(value) for operation in operations for value in primitive_outputs(operation))
    parameters = [f"const {_cuda_type(value.dtype)}* argument_{ordinals[value]}" for value in live_inputs]
    parameters.extend(f"{_cuda_type(value.dtype)}* argument_{ordinals[value]}" for value in live_outputs)
    elements = max(math.prod(value.shape) for value in live_outputs)
    stores = []
    for value in live_outputs:
        coordinate_names = _linear_coordinate_names(value.axes, source="linear", indent="    ")
        call = renderer.call(value, {axis: f"coordinate_{axis.id}" for axis in value.axes})
        stores.append(
            f"  if (linear < {math.prod(value.shape)}) {{\n"
            f"{coordinate_names}\n"
            f"    argument_{ordinals[value]}[linear] = {call};\n"
            "  }"
        )
    return (
        "\n\n".join(functions) + f"\n\nextern \"C\" __global__ void {symbol}({', '.join(parameters)}) {{\n"
        "  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
        f"  if (linear >= {elements}) return;\n" + "\n".join(stores) + "\n}"
    )


class _DependencyConvexValueRenderer:
    def __init__(
        self,
        live_inputs: tuple[ProgramValue, ...],
        producer: dict[ProgramValue, MapPrimitive | FoldPrimitive | ContractPrimitive],
        ordinals: dict[ProgramValue, int],
    ):
        self.live_inputs = live_inputs
        self.producer = producer
        self.ordinals = ordinals

    def parameters(self) -> str:
        pointers = [f"const {_cuda_type(value.dtype)}* argument_{self.ordinals[value]}" for value in self.live_inputs]
        return ", ".join(pointers)

    def call(self, value: ProgramValue, coordinates: dict[Any, str]) -> str:
        arguments = [f"argument_{self.ordinals[item]}" for item in self.live_inputs]
        arguments.extend(coordinates[axis] for axis in value.axes)
        return f"shuttle_value_{self.ordinals[value]}({', '.join(arguments)})"

    def expression(self, value: ProgramValue, coordinates: dict[Any, str]) -> str:
        if value in self.live_inputs:
            return f"argument_{self.ordinals[value]}[{_axis_offset(value.axes, coordinates)}]"
        if value not in self.producer:
            raise ValueError(f"dependency-convex component has no producer or live input for {value.name!r}")
        return self.call(value, coordinates)

    def function(self, value: ProgramValue) -> str:
        operation = self.producer[value]
        coordinates = {axis: f"coordinate_{axis.id}" for axis in value.axes}
        coordinate_parameters = [f"int64_t {coordinates[axis]}" for axis in value.axes]
        parameters = ", ".join(filter(None, (self.parameters(), ", ".join(coordinate_parameters))))
        if isinstance(operation, MapPrimitive):
            scalar_inputs = {
                operand.name: self.expression(operand, {axis: coordinates[axis] for axis in operand.axes})
                for operand in operation.inputs
            }
            expression = render_cuda_scalar_expression(operation.expression, scalar_inputs)
            body = f"  return {_store(expression, value.dtype)};"
        elif isinstance(operation, ContractPrimitive):
            body = self._contract_body(operation, coordinates)
        else:
            assert isinstance(operation, FoldPrimitive)
            body = self._fold_body(operation, coordinates)
        return (
            f"static __device__ __forceinline__ {_cuda_type(value.dtype)} shuttle_value_{self.ordinals[value]}"
            f"({parameters}) {{\n{body}\n}}"
        )

    def _contract_body(self, operation: ContractPrimitive, coordinates: dict[Any, str]) -> str:
        reduction_size = math.prod(axis.extent for axis in operation.reduction_axes)
        reduction_coordinates = {axis: f"coordinate_{axis.id}" for axis in operation.reduction_axes}
        coordinate_lines = _linear_coordinate_names(
            operation.reduction_axes,
            source="reduction_linear",
            indent="    ",
        )
        loads = []
        for input_index, operand in enumerate(operation.inputs):
            operand_coordinates = {**coordinates, **reduction_coordinates}
            for mapping in operation.index_maps_for_input(input_index):
                mapped = f"(({operand_coordinates[mapping.domain_axis]}) / {mapping.divisor} + {mapping.offset})"
                if mapping.modulus is not None:
                    mapped = f"(({mapped}) % {mapping.modulus})"
                operand_coordinates[mapping.operand_axis] = mapped
            loads.append(self.expression(operand, {axis: operand_coordinates[axis] for axis in operand.axes}))
        return (
            "  float accumulator = 0.0f;\n"
            f"  for (int64_t reduction_linear = 0; reduction_linear < {reduction_size}; ++reduction_linear) {{\n"
            f"{coordinate_lines}\n"
            f"    accumulator = fmaf(static_cast<float>({loads[0]}), static_cast<float>({loads[1]}), accumulator);\n"
            "  }\n"
            f"  return {_store('accumulator', operation.output.dtype)};"
        )

    def _fold_body(self, operation: FoldPrimitive, coordinates: dict[Any, str]) -> str:
        reduction_size = math.prod(axis.extent for axis in operation.reduction_axes)
        reduction_coordinates = {axis: f"coordinate_{axis.id}" for axis in operation.reduction_axes}
        coordinate_lines = _linear_coordinate_names(
            operation.reduction_axes,
            source="reduction_linear",
            indent="    ",
        )
        input_coordinates = {**coordinates, **reduction_coordinates}
        loaded = self.expression(
            operation.input,
            {axis: input_coordinates[axis] for axis in operation.input.axes},
        )
        accumulator_type = _cuda_type(operation.accumulation_dtype)
        if operation.reducer is FoldReducer.SUM:
            initial = _store("0.0f", operation.accumulation_dtype)
            update = _store(
                f"static_cast<float>(accumulator) + static_cast<float>({loaded})",
                operation.accumulation_dtype,
            )
        elif operation.reducer is FoldReducer.MAXIMUM:
            initial = _store("-INFINITY", operation.accumulation_dtype)
            update = _store(
                f"(isnan(static_cast<float>(accumulator)) || isnan(static_cast<float>({loaded}))) "
                f"? NAN : fmaxf(static_cast<float>(accumulator), static_cast<float>({loaded}))",
                operation.accumulation_dtype,
            )
        else:
            raise ValueError(f"unsupported dependency-convex Fold reducer {operation.reducer.value}")
        return (
            f"  {accumulator_type} accumulator = {initial};\n"
            f"  for (int64_t reduction_linear = 0; reduction_linear < {reduction_size}; ++reduction_linear) {{\n"
            f"{coordinate_lines}\n"
            f"    accumulator = {update};\n"
            "  }\n"
            f"  return {_store('accumulator', operation.output.dtype)};"
        )


def _linear_coordinate_names(axes: tuple[Any, ...], *, source: str, indent: str) -> str:
    lines = []
    divisor = math.prod(axis.extent for axis in axes)
    for axis in axes:
        divisor //= axis.extent
        expression = f"({source} / {divisor}) % {axis.extent}" if divisor != 1 else f"{source} % {axis.extent}"
        lines.append(f"{indent}const int64_t coordinate_{axis.id} = {expression};")
    return "\n".join(lines)


def _axis_offset(axes: tuple[Any, ...], coordinates: dict[Any, str]) -> str:
    terms = []
    stride = 1
    for axis in reversed(axes):
        terms.append(f"({coordinates[axis]}) * {stride}")
        stride *= axis.extent
    return " + ".join(reversed(terms)) or "0"


def _store(expression: str, dtype: DType) -> str:
    if dtype is DType.BF16:
        return f"__float2bfloat16_rn(static_cast<float>({expression}))"
    return f"static_cast<{_cuda_type(dtype)}>({expression})"


def _tensor_program(request: ExecutablePhysicalRequest) -> TensorProgram:
    record = request.algebra_record
    if not isinstance(record, dict) or record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("generic SM90 Relation emission requires a TensorProgram algebra record")
    return deserialize_tensor_program(json.dumps(record, sort_keys=True, separators=(",", ":")))


def _work_items(operation: object) -> int:
    if isinstance(operation, SelectionPrimitive):
        return math.prod(axis.extent for axis in operation.score.axes if axis != operation.dimension)
    return math.prod(primitive_outputs(operation)[0].shape)


def _cuda_type(dtype: DType) -> str:
    return {
        DType.BOOL: "bool",
        DType.INT32: "int32_t",
        DType.BF16: "__nv_bfloat16",
        DType.FP32: "float",
        DType.FP64: "double",
    }[dtype]
