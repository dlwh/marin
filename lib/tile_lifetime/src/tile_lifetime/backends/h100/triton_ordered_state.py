# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a persistent SM90 CUDA kernel from a generic Scan body DAG."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from shuttle.ir import DType
from tile_lifetime.physical_task_contract import (
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalTarget,
    lower_executable_event_operations,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    MapPrimitive,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    ScanDirection,
    ScanPrimitive,
    TensorAxis,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
)

_SCHEDULE_FAMILY = "ordered_state_chunk_pipeline"
_MAXIMUM_LOCAL_STATE_BYTES = 128 * 1024
_MAXIMUM_LOCAL_CACHE_ENTRIES = 16 * 1024


@dataclass(frozen=True)
class GeneratedCudaOrderedState:
    """CUDA extension sources sealed to one canonical physical request."""

    binding_source: str
    cuda_source: str
    device_source: str
    request_fingerprint: str
    artifact_fingerprint: str
    source_sha256: str
    owner_count: int
    thread_count: int
    operand_names: tuple[str, ...]
    result_names: tuple[str, ...]

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return identities needed to reproduce and audit this artifact."""
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "owner_count": self.owner_count,
            "operand_names": self.operand_names,
            "request_fingerprint": self.request_fingerprint,
            "result_names": self.result_names,
            "source_sha256": self.source_sha256,
            "thread_count": self.thread_count,
        }


@dataclass(frozen=True)
class OrderedStateScanComponent:
    """One planner-selected generic Scan and its exact algebraic boundary."""

    program: TensorProgram
    source_program_fingerprint: str
    operation_ordinal: int

    @property
    def scan(self) -> ScanPrimitive:
        operation = self.program.operations[0]
        assert isinstance(operation, ScanPrimitive)
        return operation


@dataclass(frozen=True)
class _ScanKernelRegion:
    scan: ScanPrimitive
    state_row: TensorAxis
    state_column: TensorAxis
    producers: dict[ProgramValue, MapPrimitive | ContractPrimitive | FoldPrimitive | RelationPrimitive]
    input_bindings: dict[ProgramValue, tuple[ProgramValue, OperandIndexing | None]]
    state_row_axes: frozenset[TensorAxis]
    state_column_axes: frozenset[TensorAxis]
    warp_column_owner: bool
    matrix_owner: bool
    cached_values: dict[ProgramValue, tuple[int, tuple[TensorAxis, ...]]]
    cache_size: int


def generate_cuda_ordered_state_artifact(
    component: OrderedStateScanComponent,
    request: ExecutablePhysicalRequest,
) -> GeneratedCudaOrderedState:
    """Compile a supported generic Map/Contract Scan region to CUDA source.

    The selected schedule owns either one state column or a bounded state
    matrix, retaining that state across the traversal. The body is compiled
    recursively from its DAG: Maps use scalar ASTs and Contracts/Folds generate
    explicit source-ordered loops. Operation counts, names, and formulas are
    not selection keys.
    """
    program = component.program
    _validate_request(program, request)
    region = _scan_kernel_region(component.scan, program)
    cuda_source = _render_cuda_source(program, region)
    binding_source = _render_binding_source()
    device_source = _device_source(cuda_source)
    digest = hashlib.sha256((binding_source + "\n" + cuda_source).encode()).hexdigest()
    state = region.scan.state[0].initial
    thread_count = _matrix_thread_count(region) if region.matrix_owner else 128
    owner_count = state.axes[0].extent * state.axes[1].extent * (1 if region.matrix_owner else state.axes[-1].extent)
    if region.warp_column_owner:
        owner_count *= 32
    if region.matrix_owner:
        owner_count *= thread_count
    return GeneratedCudaOrderedState(
        binding_source,
        cuda_source,
        device_source,
        request.fingerprint,
        request.artifact_fingerprint,
        digest,
        owner_count,
        thread_count,
        tuple(value.name for value in program.inputs),
        tuple(value.name for value in program.outputs),
    )


def ordered_state_scan_component(program: TensorProgram, operation_ordinal: int) -> OrderedStateScanComponent:
    """Select a Scan component from a planner-provided operation ordinal."""
    if not 0 <= operation_ordinal < len(program.operations):
        raise ValueError("ordered-state component operation ordinal is outside the source program")
    scan = program.operations[operation_ordinal]
    if not isinstance(scan, ScanPrimitive):
        raise ValueError("ordered-state component selection requires a generic Scan operation")
    if program.operations == (scan,):
        component_program = program
    else:
        component_program = TensorProgram(
            primitive_inputs(scan),
            (scan,),
            primitive_outputs(scan),
            program.numerical_policy,
        )
    return OrderedStateScanComponent(component_program, program.semantic_fingerprint, operation_ordinal)


def _validate_request(program: TensorProgram, request: ExecutablePhysicalRequest) -> None:
    if request.algebra_fingerprint != program.semantic_fingerprint or request.algebra_record != program.canonical_record:
        raise ValueError("ordered-state artifact request does not describe the supplied algebra")
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the first ordered-state CUDA emitter requires an SM90 target")
    if request.numerical_policy is not PhysicalNumericalPolicy.SOURCE_ORDERED:
        raise ValueError("the first ordered-state CUDA emitter implements SOURCE_ORDERED traversal only")
    if len(request.schedules) != 1 or request.schedules[0].family.name != _SCHEDULE_FAMILY:
        raise ValueError("ordered-state CUDA emission requires one compatible generic Scan schedule")
    if any(event.retained_operations for event in lower_executable_event_operations(request)):
        raise ValueError("persistent ordered-state emission requires cross-chunk Events to erase by program order")


def _scan_kernel_region(scan: ScanPrimitive, program: TensorProgram) -> _ScanKernelRegion:
    if scan.direction not in (ScanDirection.FORWARD, ScanDirection.REVERSE):
        raise ValueError("ordered-state emission requires explicit forward or reverse Scan traversal")
    if len(scan.state) != 1:
        raise ValueError("ordered-state emission requires exactly one carried state")
    state = scan.state[0]
    if state.initial.dtype is not DType.FP32 or len(state.initial.axes) != 4:
        raise ValueError("ordered-state emission requires a rank-four FP32 carried state")
    state_row, state_column = state.body_argument.axes[-2:]
    state_row_axes = _axis_ownership_class(
        scan.body,
        (state.body_argument.axes[-2], state.body_result.axes[-2]),
    )
    state_column_axes = _axis_ownership_class(
        scan.body,
        (state.body_argument.axes[-1], state.body_result.axes[-1]),
    )
    if any(value.dtype not in {DType.BF16, DType.FP32} for value in program.inputs + program.outputs):
        raise ValueError("ordered-state emission supports BF16 and FP32 public tensors")
    if state.final not in program.outputs or not set(program.outputs) <= set(scan.outputs):
        raise ValueError("ordered-state emission requires the final state and only scheduled Scan outputs")

    # Keep one persistent owner per state column whenever the body has no
    # reduction across that axis. Relations and column-free tape outputs do not
    # couple owners: views are evaluated at the current coordinate and one
    # designated column can publish replicated/scalar tape values.
    matrix_owner = any(
        any(axis in state_column_axes for axis in operation.reduction_axes)
        for operation in scan.body.operations
        if isinstance(operation, (FoldPrimitive, ContractPrimitive))
    )
    public_emissions = tuple(binding for binding in scan.emitted_outputs if binding.destination in program.outputs)
    warp_column_owner = (
        not matrix_owner
        and state_row.extent >= 32
        and not any(set(binding.body_result.axes) & state_row_axes for binding in public_emissions)
        and not any(
            set(operation.reduction_axes) & state_row_axes
            for operation in scan.body.operations
            if isinstance(operation, FoldPrimitive)
        )
    )

    producers: dict[ProgramValue, MapPrimitive | ContractPrimitive | FoldPrimitive | RelationPrimitive] = {}
    for operation in scan.body.operations:
        if not isinstance(operation, (MapPrimitive, ContractPrimitive, FoldPrimitive, RelationPrimitive)):
            raise ValueError(
                "ordered-state Scan bodies currently support generic Map, Fold, Contract, and Relation operations"
            )
        if any(output.dtype not in {DType.BF16, DType.FP32} for output in primitive_outputs(operation)):
            raise ValueError("ordered-state body intermediates currently require BF16 or FP32 storage semantics")
        if isinstance(operation, ContractPrimitive):
            if not matrix_owner and any(axis in state_column_axes for axis in operation.reduction_axes):
                raise ValueError("state-column reductions cross persistent CUDA owners and are not supported")
            if operation.accumulation_dtype is not DType.FP32:
                raise ValueError("ordered-state Contracts currently require FP32 accumulation")
            serial_reduction_iterations = 1
            for axis in operation.reduction_axes:
                if axis not in state_row_axes:
                    serial_reduction_iterations *= axis.extent
            if serial_reduction_iterations > 4096:
                raise ValueError(
                    "ordered-state serial Contract reduction exceeds the schedule resource proof: "
                    f"required={serial_reduction_iterations} iterations, limit=4096 iterations"
                )
        if isinstance(operation, FoldPrimitive):
            if operation.reducer is not FoldReducer.SUM or operation.accumulation_dtype is not DType.FP32:
                raise ValueError("ordered-state Scan Folds currently require source-ordered FP32 summation")
        if isinstance(operation, RelationPrimitive):
            if (
                operation.predicate is not None
                or len(operation.inputs) != 1
                or len(operation.outputs) != 1
                or operation.resolved_result_sources != operation.inputs
            ):
                raise ValueError("ordered-state Scan Relations currently require one source-backed unpredicated view")
        for output in primitive_outputs(operation):
            producers[output] = operation

    input_bindings: dict[ProgramValue, tuple[ProgramValue, OperandIndexing | None]] = {
        state.body_argument: (state.initial, None),
        **{binding.body_argument: (binding.source, None) for binding in scan.invariants},
        **{binding.body_argument: (binding.source, binding.indexing) for binding in scan.scanned_inputs},
    }
    for body_input, (_, indexing) in input_bindings.items():
        if body_input != state.body_argument and indexing is not None:
            _validate_indexing(indexing)
    for binding in scan.emitted_outputs:
        _validate_indexing(binding.indexing)
    cached_values: dict[ProgramValue, tuple[int, tuple[TensorAxis, ...]]] = {}
    cache_size = 0
    fixed_seeds = (
        state.body_argument.axes[0],
        state.body_result.axes[0],
        state.body_argument.axes[1],
        state.body_result.axes[1],
    )
    if not matrix_owner:
        fixed_seeds += (state.body_argument.axes[-1], state.body_result.axes[-1])
    fixed_axes = _axis_ownership_class(scan.body, fixed_seeds)
    for operation in scan.body.operations:
        for output in primitive_outputs(operation):
            if not matrix_owner and set(output.axes) & state_row_axes:
                continue
            varying_axes = tuple(axis for axis in output.axes if axis not in fixed_axes)
            entries = 1
            for axis in varying_axes:
                entries *= axis.extent
            # Cache only bounded per-owner intermediates. Larger tensors remain
            # legal and recursively evaluated rather than becoming an implicit
            # materialization requirement.
            matrix_reuse = entries <= max(state_row.extent, state_column.extent)
            if (
                entries > 4096
                or cache_size + entries > _MAXIMUM_LOCAL_CACHE_ENTRIES
                or (matrix_owner and not matrix_reuse)
            ):
                continue
            cached_values[output] = (cache_size, varying_axes)
            cache_size += entries
    state_entries = state_row.extent * (state_column.extent if matrix_owner else 1)
    local_bytes = 2 * state_entries * 4 + cache_size * 5
    if local_bytes > _MAXIMUM_LOCAL_STATE_BYTES:
        raise ValueError(
            "ordered-state local state/cache storage exceeds the schedule resource proof: "
            f"required={local_bytes} bytes, limit={_MAXIMUM_LOCAL_STATE_BYTES} bytes"
        )
    return _ScanKernelRegion(
        scan,
        state_row,
        state_column,
        producers,
        input_bindings,
        frozenset(state_row_axes),
        frozenset(state_column_axes),
        warp_column_owner,
        matrix_owner,
        cached_values,
        cache_size,
    )


def _matrix_thread_count(region: _ScanKernelRegion) -> int:
    state_elements = region.state_row.extent * region.state_column.extent
    if region.cache_size:
        return 64
    if state_elements <= 32:
        return 32
    if state_elements <= 64:
        return 64
    return 128


def _axis_ownership_class(program: TensorProgram, seeds: tuple[TensorAxis, ...]) -> set[TensorAxis]:
    """Return axes connected to ``seeds`` by exact one-to-one index relations."""
    aliases: dict[TensorAxis, set[TensorAxis]] = {}

    def connect(left: TensorAxis, right: TensorAxis) -> None:
        if left.extent != right.extent:
            return
        aliases.setdefault(left, set()).add(right)
        aliases.setdefault(right, set()).add(left)

    for operation in program.operations:
        if isinstance(operation, ContractPrimitive):
            for input_index in range(len(operation.inputs)):
                for mapping in operation.index_maps_for_input(input_index):
                    if mapping.divisor == 1 and mapping.offset == 0 and mapping.modulus is None:
                        connect(mapping.domain_axis, mapping.operand_axis)
        elif isinstance(operation, RelationPrimitive):
            for indexing in operation.input_indexing:
                for operand_axis, expression in zip(indexing.operand.axes, indexing.indices, strict=True):
                    if expression.kind is IndexExpressionKind.AXIS and expression.axis is not None:
                        connect(expression.axis, operand_axis)

    ownership = set(seeds)
    frontier = list(seeds)
    while frontier:
        axis = frontier.pop()
        for alias in aliases.get(axis, ()):
            if alias in ownership:
                continue
            ownership.add(alias)
            frontier.append(alias)
    return ownership


def _validate_indexing(indexing: OperandIndexing) -> None:
    if indexing.predicate is not None:
        raise ValueError("predicated Scan bindings are not yet supported")
    for expression in indexing.indices:
        _render_index_expression(expression, {})


def _render_binding_source() -> str:
    return r"""#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> execute_cuda(std::vector<torch::Tensor> operands);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("execute", &execute_cuda, "generic ordered-state Scan (CUDA)");
}
"""


def _render_cuda_source(program: TensorProgram, region: _ScanKernelRegion) -> str:
    scan = region.scan
    state = scan.state[0]
    batch, parallel, state_row, state_column = state.initial.axes
    all_values = (
        *scan.body.inputs,
        *(value for operation in scan.body.operations for value in primitive_outputs(operation)),
    )
    value_ordinals = {value: ordinal for ordinal, value in enumerate(all_values)}
    body_axes = _body_axes(scan)
    coordinate_parameters = {axis: f"coordinate_{ordinal}" for ordinal, axis in enumerate(body_axes)}
    input_ordinals = {value: index for index, value in enumerate(program.inputs)}
    output_ordinals = {value: index for index, value in enumerate(program.outputs)}
    pointer_arguments = [
        f"const {_cuda_type(value.dtype)}* input_{index}" for index, value in enumerate(program.inputs)
    ] + [f"{_cuda_type(value.dtype)}* output_{index}" for index, value in enumerate(program.outputs)]
    function_arguments = ", ".join(
        [
            *pointer_arguments,
            "const float* state_values",
            "float* cache_values",
            "bool* cache_ready",
            "int t",
            "int lane",
            *(f"int {coordinate_parameters[axis]}" for axis in body_axes),
        ]
    )
    pointer_names = ", ".join(
        [
            *(f"input_{index}" for index in range(len(program.inputs))),
            *(f"output_{index}" for index in range(len(program.outputs))),
        ]
    )
    declarations = []
    definitions = []
    for value, ordinal in value_ordinals.items():
        declarations.append(f"__device__ __forceinline__ float value_{ordinal}({function_arguments});")
        definitions.append(
            _render_value_evaluator(
                value,
                ordinal,
                region,
                value_ordinals,
                input_ordinals,
                function_arguments,
                pointer_names,
                body_axes,
                coordinate_parameters,
            )
        )

    root_coordinates = _root_coordinates(region)
    emissions = []
    for emission_index, binding in enumerate(scan.emitted_outputs):
        if binding.destination not in output_ordinals:
            continue
        destination = f"output_{output_ordinals[binding.destination]}"
        coordinates = dict(root_coordinates)
        loop_lines, indentation, closing_lines = _coordinate_loops(binding.body_result.axes, region, coordinates, "    ")
        index_bindings = _scan_boundary_bindings(scan, binding.body_result, binding.indexing, coordinates)
        destination_indices = tuple(
            _render_index_expression(expression, index_bindings) for expression in binding.indexing.indices
        )
        offset = _flat_offset(binding.destination.axes, destination_indices)
        call = _value_call(
            f"value_{value_ordinals[binding.body_result]}",
            pointer_names,
            "state_values",
            body_axes,
            coordinates,
        )
        column_free = not set(binding.body_result.axes) & region.state_column_axes
        if region.warp_column_owner:
            store_condition = "lane == 0" + (" && v == 0" if column_free else "")
            statements = (
                f"{indentation}float emitted_{emission_index} = {call};",
                f"{indentation}if ({store_condition}) {destination}[{offset}] = "
                f"{_store_conversion(binding.destination.dtype, f'emitted_{emission_index}')};",
            )
        else:
            statements = (
                f"{indentation}{'if (v == 0) ' if not region.matrix_owner and column_free else ''}"
                f"{destination}[{offset}] = {_store_conversion(binding.destination.dtype, call)};",
            )
        emissions.append("\n".join((*loop_lines, *statements, *closing_lines)))

    state_input = f"input_{input_ordinals[state.initial]}"
    final_output = f"output_{output_ordinals[state.final]}"
    state_column_coordinate = "v_index" if region.matrix_owner else "v"
    initial_offset = _flat_offset(state.initial.axes, ("b", "h", "k_index", state_column_coordinate))
    final_offset = _flat_offset(state.final.axes, ("b", "h", "k_index", state_column_coordinate))
    next_coordinates = dict(root_coordinates)
    next_coordinates[state.body_result.axes[-2]] = "evaluation_k" if region.warp_column_owner else "k_index"
    next_coordinates[state.body_result.axes[-1]] = state_column_coordinate
    next_call = _value_call(
        f"value_{value_ordinals[state.body_result]}",
        pointer_names,
        "state_values",
        body_axes,
        next_coordinates,
    )
    launch_inputs = ", ".join(
        f"reinterpret_cast<const {_cuda_type(value.dtype)}*>(operands[{index}].data_ptr())"
        for index, value in enumerate(program.inputs)
    )
    launch_outputs = ", ".join(
        f"reinterpret_cast<{_cuda_type(value.dtype)}*>(outputs[{index}].data_ptr())"
        for index, value in enumerate(program.outputs)
    )
    output_allocations = "\n".join(_output_allocation(value) for value in program.outputs)
    checks = "\n".join(_shape_check(index, value) for index, value in enumerate(program.inputs))
    bf16_include = (
        "#include <cuda_bf16.h>" if any(value.dtype is DType.BF16 for value in program.inputs + program.outputs) else ""
    )
    contiguity_check = (
        "  for (const auto& tensor : operands) TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "
        '"ordered-state operands must be contiguous CUDA tensors");'
    )
    launch = (
        "  ordered_state_kernel<<<(owners + threads - 1) / threads, threads, 0, "
        f"at::cuda::getDefaultCUDAStream()>>>({launch_inputs}, {launch_outputs});"
    )
    launch_thread_count = _matrix_thread_count(region) if region.matrix_owner else 128
    launch_owner_count = (
        batch.extent
        * parallel.extent
        * (1 if region.matrix_owner else state_column.extent)
        * (32 if region.warp_column_owner else 1)
        * (launch_thread_count if region.matrix_owner else 1)
    )
    if region.warp_column_owner:
        owner_setup = """  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = thread & 31;
  int owner = thread >> 5;
  if (owner >= B * H * V) return;
  int v = owner % V;
  int h = (owner / V) % H;
  int b = owner / (H * V);"""
        state_storage = "(K + 31) / 32"
        initialize_state = (
            f"  for (int state_index = 0; state_index < {state_storage}; ++state_index) {{\n"
            "    int k_index = lane + state_index * 32;\n"
            f"    if (k_index < K) state_values[state_index] = {state_input}[{initial_offset}];\n"
            "  }"
        )
        advance_state = (
            f"    for (int state_index = 0; state_index < {state_storage}; ++state_index) {{\n"
            "      int k_index = lane + state_index * 32;\n"
            "      int evaluation_k = k_index < K ? k_index : K - 1;\n"
            f"      next_state[state_index] = {next_call};\n"
            "    }\n"
            f"    for (int state_index = 0; state_index < {state_storage}; ++state_index)\n"
            "      state_values[state_index] = next_state[state_index];"
        )
        store_final = (
            f"  for (int state_index = 0; state_index < {state_storage}; ++state_index) {{\n"
            "    int k_index = lane + state_index * 32;\n"
            f"    if (k_index < K) {final_output}[{final_offset}] = state_values[state_index];\n"
            "  }"
        )
        state_declarations = f"  float state_values[{state_storage}];\n  float next_state[{state_storage}];"
    elif region.matrix_owner:
        owner_setup = """  int lane = threadIdx.x;
  int owner = blockIdx.x;
  if (owner >= B * H) return;
  int v = 0;
  int h = owner % H;
  int b = owner / H;"""
        state_storage = "K * V"
        state_declarations = "  __shared__ float state_values[K * V];\n  __shared__ float next_state[K * V];"
        initialize_state = (
            "  for (int state_index = lane; state_index < K * V; state_index += blockDim.x) {\n"
            "    int k_index = state_index / V;\n"
            "    int v_index = state_index % V;\n"
            f"    state_values[state_index] = {state_input}[{initial_offset}];\n"
            "  }\n"
            "  __syncthreads();"
        )
        advance_state = (
            "    for (int state_index = lane; state_index < K * V; state_index += blockDim.x) {\n"
            "      int k_index = state_index / V;\n"
            "      int evaluation_k = k_index;\n"
            "      int v_index = state_index % V;\n"
            f"      next_state[state_index] = {next_call};\n"
            "    }\n"
            "    __syncthreads();\n"
            "    for (int state_index = lane; state_index < K * V; state_index += blockDim.x)\n"
            "      state_values[state_index] = next_state[state_index];\n"
            "    __syncthreads();"
        )
        store_final = (
            "  for (int state_index = lane; state_index < K * V; state_index += blockDim.x) {\n"
            "    int k_index = state_index / V;\n"
            "    int v_index = state_index % V;\n"
            f"    {final_output}[{final_offset}] = state_values[state_index];\n"
            "  }"
        )
    else:
        owner_setup = f"""  int owner = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = 0;
  if (owner >= B * H * {"1" if region.matrix_owner else "V"}) return;
  int v = {"0" if region.matrix_owner else "owner % V"};
  int h = {"owner % H" if region.matrix_owner else "(owner / V) % H"};
  int b = {"owner / H" if region.matrix_owner else "owner / (H * V)"};"""
        state_storage = f"K * {'V' if region.matrix_owner else '1'}"
        initialize_state = (
            f"  for (int v_index = 0; v_index < {'V' if region.matrix_owner else '1'}; ++v_index)\n"
            "    for (int k_index = 0; k_index < K; ++k_index)\n"
            f"      state_values[k_index * {'V' if region.matrix_owner else '1'} + v_index] = "
            f"{state_input}[{initial_offset}];"
        )
        advance_state = (
            f"    for (int v_index = 0; v_index < {'V' if region.matrix_owner else '1'}; ++v_index)\n"
            "      for (int k_index = 0; k_index < K; ++k_index) {\n"
            "        int evaluation_k = k_index;\n"
            f"        next_state[k_index * {'V' if region.matrix_owner else '1'} + v_index] = {next_call};\n"
            "      }\n"
            f"    for (int state_index = 0; state_index < K * {'V' if region.matrix_owner else '1'}; ++state_index)\n"
            "      state_values[state_index] = next_state[state_index];"
        )
        store_final = (
            f"  for (int v_index = 0; v_index < {'V' if region.matrix_owner else '1'}; ++v_index)\n"
            "    for (int k_index = 0; k_index < K; ++k_index)\n"
            f"      {final_output}[{final_offset}] = "
            f"state_values[k_index * {'V' if region.matrix_owner else '1'} + v_index];"
        )
        state_declarations = f"  float state_values[{state_storage}];\n  float next_state[{state_storage}];"
    return f'''// Generated from generic Shuttle Map/Contract/Scan algebra.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
{bf16_include}
#include <vector>

{"\n".join(declarations)}

{"\n\n".join(definitions)}

extern "C" __global__ void ordered_state_kernel({", ".join(pointer_arguments)}) {{
  constexpr int B = {batch.extent};
  constexpr int H = {parallel.extent};
  constexpr int K = {state_row.extent};
  constexpr int V = {state_column.extent};
  constexpr int T = {scan.axis.extent};
{owner_setup}
{state_declarations}
  float cache_values[{max(1, region.cache_size)}];
  bool cache_ready[{max(1, region.cache_size)}];
{initialize_state}
  for (int iteration = 0; iteration < T; ++iteration) {{
    int t = {"iteration" if scan.direction is ScanDirection.FORWARD else "T - 1 - iteration"};
    for (int cache_index = 0; cache_index < {region.cache_size}; ++cache_index)
      cache_ready[cache_index] = false;
{"\n".join(emissions)}
{advance_state}
  }}
{store_final}
}}

std::vector<torch::Tensor> execute_cuda(std::vector<torch::Tensor> operands) {{
  TORCH_CHECK(operands.size() == {len(program.inputs)}, "wrong ordered-state operand count");
{contiguity_check}
{checks}
  std::vector<torch::Tensor> outputs;
  outputs.reserve({len(program.outputs)});
{output_allocations}
  constexpr int owners = {launch_owner_count};
  constexpr int threads = {launch_thread_count};
  const c10::cuda::CUDAGuard device_guard(operands[0].device());
{launch}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}}
'''


def _device_source(cuda_source: str) -> str:
    """Remove the optional Torch host wrapper for direct NVRTC compilation."""
    device_start = cuda_source.index("__device__")
    host_start = cuda_source.index("\nstd::vector<torch::Tensor> execute_cuda")
    return "#include <cuda_bf16.h>\n\n" + cuda_source[device_start:host_start]


def _render_value_evaluator(
    value: ProgramValue,
    ordinal: int,
    region: _ScanKernelRegion,
    value_ordinals: dict[ProgramValue, int],
    input_ordinals: dict[ProgramValue, int],
    function_arguments: str,
    pointer_names: str,
    body_axes: tuple[TensorAxis, ...],
    coordinate_parameters: dict[TensorAxis, str],
) -> str:
    state = region.scan.state[0]
    cache = region.cached_values.get(value)
    cache_index = None
    cache_prelude: list[str] = []
    if cache is not None:
        base, varying_axes = cache
        offset = "0"
        for axis in varying_axes:
            offset = f"(({offset}) * {axis.extent} + ({coordinate_parameters[axis]}))"
        cache_index = f"({base} + {offset})"
        cache_prelude.append(f"  if (cache_ready[{cache_index}]) return cache_values[{cache_index}];")

    def finish(lines: list[str], result: str) -> str:
        if cache_index is not None:
            lines.extend(
                (
                    f"  cache_values[{cache_index}] = {result};",
                    f"  cache_ready[{cache_index}] = true;",
                    f"  return cache_values[{cache_index}];",
                )
            )
        else:
            lines.append(f"  return {result};")
        body = "\n".join((*cache_prelude, *lines))
        return f"__device__ __forceinline__ float value_{ordinal}({function_arguments}) {{\n{body}\n}}"

    if value == state.body_argument:
        row = coordinate_parameters[state.body_argument.axes[-2]]
        column = coordinate_parameters[state.body_argument.axes[-1]]
        local_column = column if region.matrix_owner else "0"
        local_row = f"(({row}) / 32)" if region.warp_column_owner else row
        expression = (
            f"state_values[({local_row}) * {region.state_column.extent if region.matrix_owner else 1} + "
            f"({local_column})]"
        )
    elif value in region.input_bindings:
        source, indexing = region.input_bindings[value]
        if indexing is None:
            coordinates = tuple(coordinate_parameters[axis] for axis in value.axes)
        else:
            bindings = _scan_boundary_bindings(region.scan, value, indexing, coordinate_parameters)
            coordinates = tuple(_render_index_expression(index, bindings) for index in indexing.indices)
        expression = _load_expression(source, input_ordinals[source], coordinates)
    else:
        operation = region.producers[value]
        if isinstance(operation, MapPrimitive):
            bindings = {
                operand.name: _value_call(
                    f"value_{value_ordinals[operand]}",
                    pointer_names,
                    "state_values",
                    body_axes,
                    coordinate_parameters,
                )
                for operand in operation.inputs
            }
            expression = _render_scalar(operation.expression, bindings)
        elif isinstance(operation, RelationPrimitive):
            source = operation.resolved_result_sources[operation.outputs.index(value)]
            assert source is not None
            indexing = operation.input_indexing[operation.inputs.index(source)]
            bindings = {axis.id: coordinate_parameters[axis] for axis in operation.domain_axes}
            source_coordinates = {
                axis: _render_index_expression(index, bindings)
                for axis, index in zip(source.axes, indexing.indices, strict=True)
            }
            expression = _value_call(
                f"value_{value_ordinals[source]}",
                pointer_names,
                "state_values",
                body_axes,
                source_coordinates,
            )
        elif isinstance(operation, FoldPrimitive):
            lines = ["  float accumulator = 0.0f;"]
            indentation = "  "
            overrides: dict[TensorAxis, str] = {}
            for axis in operation.reduction_axes:
                variable = f"reduction_{axis.id}"
                lines.append(f"{indentation}for (int {variable} = 0; {variable} < {axis.extent}; ++{variable}) {{")
                indentation += "  "
                overrides[axis] = variable
            operand_coordinates = dict(coordinate_parameters)
            operand_coordinates.update(overrides)
            contribution = _value_call(
                f"value_{value_ordinals[operation.input]}",
                pointer_names,
                "state_values",
                body_axes,
                operand_coordinates,
            )
            lines.append(f"{indentation}accumulator += {contribution};")
            for _ in operation.reduction_axes:
                indentation = indentation[:-2]
                lines.append(f"{indentation}}}")
            return finish(lines, "accumulator")
        else:
            lines = ["  float accumulator = 0.0f;"]
            indentation = "  "
            overrides: dict[TensorAxis, str] = {}
            warp_reduction = False
            for axis in operation.reduction_axes:
                variable = f"reduction_{axis.id}"
                if region.warp_column_owner and axis in region.state_row_axes:
                    warp_reduction = True
                    initialization = "lane"
                    increment = "+= 32"
                else:
                    initialization = "0"
                    increment = "++"
                lines.append(
                    f"{indentation}for (int {variable} = {initialization}; "
                    f"{variable} < {axis.extent}; {variable}{increment}) {{"
                )
                indentation += "  "
                overrides[axis] = variable
            products = []
            for input_index, operand in enumerate(operation.inputs):
                operand_coordinates = dict(coordinate_parameters)
                operand_coordinates.update(overrides)
                for mapping in operation.index_maps_for_input(input_index):
                    mapped = coordinate_parameters[mapping.domain_axis]
                    if mapping.domain_axis in overrides:
                        mapped = overrides[mapping.domain_axis]
                    if mapping.divisor != 1:
                        mapped = f"({mapped} / {mapping.divisor})"
                    if mapping.offset:
                        mapped = f"({mapped} + {mapping.offset})"
                    if mapping.modulus is not None:
                        mapped = f"({mapped} % {mapping.modulus})"
                    operand_coordinates[mapping.operand_axis] = mapped
                products.append(
                    _value_call(
                        f"value_{value_ordinals[operand]}",
                        pointer_names,
                        "state_values",
                        body_axes,
                        operand_coordinates,
                    )
                )
            lines.append(f"{indentation}accumulator += {' * '.join(products)};")
            for _ in operation.reduction_axes:
                indentation = indentation[:-2]
                lines.append(f"{indentation}}}")
            if warp_reduction:
                lines.extend(
                    f"  accumulator += __shfl_down_sync(0xffffffff, accumulator, {offset});"
                    for offset in (16, 8, 4, 2, 1)
                )
            return finish(lines, "accumulator")
    return finish([], expression)


def _value_call(
    function: str,
    pointers: str,
    state: str,
    body_axes: tuple[TensorAxis, ...],
    coordinates: dict[TensorAxis, str],
) -> str:
    arguments = ", ".join(coordinates.get(axis, "0") for axis in body_axes)
    return f"{function}({pointers}, {state}, cache_values, cache_ready, t, lane, {arguments})"


def _coordinate_loops(
    axes: tuple[TensorAxis, ...],
    region: _ScanKernelRegion,
    coordinates: dict[TensorAxis, str],
    indentation: str,
) -> tuple[list[str], str, list[str]]:
    loop_axes = tuple(
        axis
        for axis in axes
        if not (
            (axis in region.state_column_axes and not region.matrix_owner) or coordinates.get(axis) in {"b", "h", "v"}
        )
    )
    if region.matrix_owner:
        elements = 1
        for axis in loop_axes:
            elements *= axis.extent
        lines = [
            f"{indentation}for (int emit_linear = lane; emit_linear < {elements}; emit_linear += blockDim.x) {{",
            f"{indentation}  int emit_remaining = emit_linear;",
        ]
        for axis in reversed(loop_axes):
            variable = f"emit_{axis.id}"
            coordinates[axis] = variable
            lines.append(f"{indentation}  int {variable} = emit_remaining % {axis.extent};")
            lines.append(f"{indentation}  emit_remaining /= {axis.extent};")
        return lines, indentation + "  ", [f"{indentation}}}"]

    lines = []
    for axis in loop_axes:
        variable = f"emit_{axis.id}"
        coordinates[axis] = variable
        lines.append(f"{indentation}for (int {variable} = 0; {variable} < {axis.extent}; ++{variable}) {{")
        indentation += "  "
    closing_lines = []
    while len(closing_lines) < len(lines):
        indentation = indentation[:-2]
        closing_lines.append(f"{indentation}}}")
    return lines, indentation + "  " * len(lines), closing_lines


def _body_axes(scan: ScanPrimitive) -> tuple[TensorAxis, ...]:
    axes = {axis for value in scan.body.inputs + scan.body.outputs for axis in value.axes}
    for operation in scan.body.operations:
        for value in primitive_outputs(operation):
            axes.update(value.axes)
        if isinstance(operation, ContractPrimitive):
            axes.update(operation.reduction_axes)
    return tuple(sorted(axes, key=lambda axis: (axis.id, axis.extent, axis.label or "")))


def _root_coordinates(region: _ScanKernelRegion) -> dict[TensorAxis, str]:
    state = region.scan.state[0]
    coordinates: dict[TensorAxis, str] = {}
    for boundary in (state.body_argument, state.body_result):
        labels = ("b", "h", "0", "v")
        coordinates.update(
            {
                axis: label
                for axis, label in zip(boundary.axes, labels, strict=True)
                if not region.matrix_owner or label in {"b", "h"}
            }
        )
    if not region.matrix_owner:
        coordinates.update({axis: "v" for axis in region.state_column_axes})
    return coordinates


def _scan_boundary_bindings(
    scan: ScanPrimitive,
    body_value: ProgramValue,
    indexing: OperandIndexing,
    coordinates: dict[TensorAxis, str],
) -> dict[int, str]:
    non_scan_axes = tuple(axis for axis in indexing.domain_axes if axis != scan.axis)
    if len(non_scan_axes) != len(body_value.axes):
        raise ValueError("ordered-state Scan bindings must remove exactly the traversal axis")
    bindings = {scan.axis.id: "t"}
    bindings.update(
        {
            domain_axis.id: coordinates[body_axis]
            for domain_axis, body_axis in zip(non_scan_axes, body_value.axes, strict=True)
        }
    )
    return bindings


def _render_index_expression(expression: IndexExpression, bindings: dict[int, str]) -> str:
    kind = expression.kind
    if kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return bindings.get(expression.axis.id, f"axis_{expression.axis.id}")
    if kind is IndexExpressionKind.CONSTANT:
        assert expression.constant is not None
        return str(expression.constant)
    operands = tuple(_render_index_expression(operand, bindings) for operand in expression.operands)
    binary = {
        IndexExpressionKind.ADD: "+",
        IndexExpressionKind.SUBTRACT: "-",
        IndexExpressionKind.MULTIPLY: "*",
        IndexExpressionKind.FLOOR_DIVIDE: "/",
        IndexExpressionKind.MODULO: "%",
    }
    if kind in binary:
        return f"({operands[0]} {binary[kind]} {operands[1]})"
    raise ValueError(f"unsupported ordered-state index operation {kind.value!r}")


def _flat_offset(axes: tuple[TensorAxis, ...], coordinates: tuple[str, ...]) -> str:
    if len(axes) != len(coordinates):
        raise ValueError("tensor offset coordinates must cover every physical axis")
    expression = "0"
    for axis, coordinate in zip(axes, coordinates, strict=True):
        expression = f"(({expression}) * {axis.extent} + ({coordinate}))"
    return expression


def _load_expression(value: ProgramValue, ordinal: int, coordinates: tuple[str, ...]) -> str:
    load = f"input_{ordinal}[{_flat_offset(value.axes, coordinates)}]"
    return f"__bfloat162float({load})" if value.dtype is DType.BF16 else load


def _render_scalar(expression: ScalarExpression, bindings: dict[str, str]) -> str:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        if expression.input_name not in bindings:
            raise ValueError(f"unbound scalar input {expression.input_name!r} in ordered-state Map")
        return bindings[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return "true" if expression.constant else "false"
        return f"{expression.constant!r}f" if isinstance(expression.constant, float) else str(expression.constant)
    operands = tuple(_render_scalar(operand, bindings) for operand in expression.operands)
    binary = {
        ScalarExpressionKind.ADD: "+",
        ScalarExpressionKind.SUBTRACT: "-",
        ScalarExpressionKind.MULTIPLY: "*",
        ScalarExpressionKind.DIVIDE: "/",
        ScalarExpressionKind.LESS: "<",
        ScalarExpressionKind.LESS_EQUAL: "<=",
        ScalarExpressionKind.GREATER: ">",
        ScalarExpressionKind.GREATER_EQUAL: ">=",
        ScalarExpressionKind.EQUAL: "==",
        ScalarExpressionKind.NOT_EQUAL: "!=",
    }
    if kind in binary:
        return f"({operands[0]} {binary[kind]} {operands[1]})"
    if kind is ScalarExpressionKind.NEGATE:
        return f"(-{operands[0]})"
    unary = {
        ScalarExpressionKind.EXP: "expf",
        ScalarExpressionKind.LOG: "logf",
        ScalarExpressionKind.RSQRT: "rsqrtf",
        ScalarExpressionKind.TANH: "tanhf",
    }
    if kind in unary:
        return f"{unary[kind]}({operands[0]})"
    if kind is ScalarExpressionKind.MAXIMUM:
        return f"fmaxf({operands[0]}, {operands[1]})"
    if kind is ScalarExpressionKind.MINIMUM:
        return f"fminf({operands[0]}, {operands[1]})"
    if kind is ScalarExpressionKind.SELECT:
        return f"({operands[0]} ? {operands[1]} : {operands[2]})"
    if kind is ScalarExpressionKind.CONVERT:
        assert expression.result_dtype is not None and expression.conversion_semantics is not None
        if expression.result_dtype is DType.FP32 and expression.conversion_semantics in {
            ScalarConversionSemantics.EXACT,
            ScalarConversionSemantics.ROUND_NEAREST_EVEN,
        }:
            return f"float({operands[0]})"
        if (
            expression.result_dtype is DType.BF16
            and expression.conversion_semantics is ScalarConversionSemantics.ROUND_NEAREST_EVEN
        ):
            return f"__bfloat162float(__float2bfloat16_rn({operands[0]}))"
        conversion = expression.conversion_semantics.value
        result_dtype = expression.result_dtype.value
        raise ValueError(f"unsupported ordered-state scalar conversion {conversion} to {result_dtype}")
    raise ValueError(f"unsupported ordered-state scalar operation {kind.value!r}")


def _cuda_type(dtype: DType) -> str:
    return {DType.BF16: "__nv_bfloat16", DType.FP32: "float"}[dtype]


def _store_conversion(dtype: DType, expression: str) -> str:
    return f"__float2bfloat16_rn({expression})" if dtype is DType.BF16 else expression


def _torch_dtype(dtype: DType) -> str:
    return {DType.BF16: "torch::kBFloat16", DType.FP32: "torch::kFloat32"}[dtype]


def _output_allocation(value: ProgramValue) -> str:
    shape = ", ".join(str(axis.extent) for axis in value.axes)
    return f"  outputs.push_back(torch::empty({{{shape}}}, operands[0].options().dtype({_torch_dtype(value.dtype)})));"


def _shape_check(index: int, value: ProgramValue) -> str:
    shape = ", ".join(str(axis.extent) for axis in value.axes)
    return (
        f"  TORCH_CHECK(operands[{index}].sizes() == torch::IntArrayRef({{{shape}}}), "
        f'"operand {index} shape does not match compiled Scan algebra");'
    )
