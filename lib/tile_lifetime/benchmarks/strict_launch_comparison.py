# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU comparison of captured launch metadata and explicit buffer bindings."""

import hashlib
import json
import math
import struct
import traceback
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from phase_failure_evidence import write_report

from tile_lifetime.canonical_kernel_composition import CanonicalKernelComponent
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CutePipelineElementType,
    CutePipelineKernelArgument,
    CutePipelineLaunchRecipe,
    CutePipelinePackedBytesArgument,
    CutePipelineScalarArgument,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
    _launch_slot_accesses,
    cute_kernel_argument_size,
    launch_dependency_event_schedule,
)
from tile_lifetime.jax_normalized_exp_contract_training import (
    prepare_sm90_split_normalized_exp_value_and_grad,
    trace_jax_owned_normalized_exp_value_and_grad,
)
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import (
    NumericalPolicy,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)

MAX_NODES = 64
MAX_EDGES = 256
MAX_PARAMETERS = 64
MAX_PARAMETER_BYTES = 4096
MAX_CAPTURE_BYTES = 65536
MAX_SLOTS = 64
MAX_REPORT_BYTES = 1 << 20
_POINTER_LIMIT = 1 << 64


def argument_relocations(argument: CutePipelineKernelArgument) -> tuple[tuple[int, str, int], ...]:
    if isinstance(argument, CutePipelineSlotArgument):
        return ((0, argument.slot, argument.byte_offset),)
    if isinstance(argument, CutePipelineStructuredArgument):
        return tuple(
            (field.destination_byte_offset, field.slot, field.slot_byte_offset)
            for field in argument.fields
            if isinstance(field, CutePipelineStructuredSlotAddressField)
        )
    return ()


def expected_argument_bytes(argument: CutePipelineKernelArgument, addresses: dict[str, int]) -> bytes:
    if isinstance(argument, CutePipelineSlotArgument):
        return struct.pack("<Q", addresses[argument.slot] + argument.byte_offset)
    if isinstance(argument, CutePipelinePackedBytesArgument):
        return argument.data
    if isinstance(argument, CutePipelineStructuredArgument):
        result = bytearray(argument.zero_filled_template)
        for offset, slot, byte_offset in argument_relocations(argument):
            struct.pack_into("<Q", result, offset, addresses[slot] + byte_offset)
        return bytes(result)
    if isinstance(argument, CutePipelineScalarArgument):
        formats = {
            CutePipelineElementType.I32: "i",
            CutePipelineElementType.U32: "I",
            CutePipelineElementType.I64: "q",
            CutePipelineElementType.U64: "Q",
            CutePipelineElementType.F32: "f",
            CutePipelineElementType.F64: "d",
        }
        return struct.pack("<" + formats[argument.element_type], argument.value)
    raise ValueError("runtime-encoded TensorMap comparison requires an independent encoder")


class GraphNodeKind(StrEnum):
    EMPTY = "empty"
    KERNEL = "kernel"


@dataclass(frozen=True)
class CapturedParameter:
    offset: int
    size: int
    data: bytes


@dataclass(frozen=True)
class CapturedKernel:
    symbol: str
    parameters: tuple[CapturedParameter, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_bytes: int
    cluster: tuple[int, int, int]
    attributes: tuple[CuteKernelFunctionAttribute, ...]


@dataclass(frozen=True)
class CapturedNode:
    node_id: int
    kind: GraphNodeKind
    kernel: CapturedKernel | None = None


@dataclass(frozen=True)
class CapturedGraph:
    nodes: tuple[CapturedNode, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SlotBinding:
    name: str
    storage_base: int
    storage_bytes: int
    byte_offset: int
    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_type: CutePipelineElementType
    device: int


def component_boundary_values(program: TensorProgram, component: CanonicalKernelComponent):
    """Resolve a real component's live values and check its exact SSA boundary."""
    by_id = {ordinal: value for value, ordinal in tensor_program_value_ordinals(program).items()}
    if len(set(component.operation_ordinals)) != len(component.operation_ordinals) or any(
        not 0 <= ordinal < len(program.operations) for ordinal in component.operation_ordinals
    ):
        raise ValueError("component has duplicate or unknown operations")
    if tuple(sorted(component.operation_ordinals)) != component.operation_ordinals:
        raise ValueError("component operations do not preserve source order")
    if any(index not in by_id for index in (*component.live_input_value_ids, *component.live_output_value_ids)):
        raise ValueError("component references an unknown live value")
    inputs = tuple(by_id[index] for index in component.live_input_value_ids)
    outputs = tuple(by_id[index] for index in component.live_output_value_ids)
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("component has ambiguous live values")
    available = set(inputs)
    produced = set()
    for ordinal in component.operation_ordinals:
        operation = program.operations[ordinal]
        if not set(primitive_inputs(operation)) <= available:
            raise ValueError("component live inputs omit an actual source operand")
        produced.update(primitive_outputs(operation))
        available.update(primitive_outputs(operation))
    consumed = {
        value for ordinal in component.operation_ordinals for value in primitive_inputs(program.operations[ordinal])
    }
    outside = {
        value
        for ordinal, operation in enumerate(program.operations)
        if ordinal not in component.operation_ordinals
        for value in primitive_inputs(operation)
    } | set(program.outputs)
    # Match the canonical source boundary, including unconsumed terminal values.
    required_inputs = consumed - produced
    required_outputs = (produced - consumed) | (produced & outside)
    if set(inputs) != required_inputs:
        raise ValueError("component live inputs differ from the exact source boundary")
    if not outputs or set(outputs) != required_outputs:
        raise ValueError("component outputs differ from the exact external SSA boundary")
    return inputs, outputs


def argument_alignment(argument: CutePipelineKernelArgument) -> int:
    if isinstance(argument, (CutePipelinePackedBytesArgument, CutePipelineStructuredArgument)):
        return argument.alignment
    if isinstance(argument, (CutePipelineSlotArgument, CutePipelineScalarArgument)):
        return cute_kernel_argument_size(argument)
    raise ValueError("runtime-encoded TensorMap alignment/encoding is not admitted")


def bounded_report_bytes(report: dict) -> bytes:
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("launch evidence exceeds the 1 MiB report bound")
    return payload


class NumericalGateError(ValueError):
    def __init__(self, evidence):
        super().__init__("unchanged public numerical gate failed")
        self.evidence = evidence


def numerical_error_records(actual, expected, *, gate: float):
    """Compare the same FP32 conversion/subtraction boundary as the benchmark."""
    records = []
    for observed, reference in zip(actual, expected, strict=True):
        observed, reference = np.asarray(observed, dtype=np.float32), np.asarray(reference, dtype=np.float32)
        if observed.shape != reference.shape:
            raise ValueError("public numerical shape differs")
        with np.errstate(invalid="ignore", over="ignore"):
            difference = np.abs(observed - reference)
        finite = np.isfinite(difference)
        maximum = float(difference.max()) if finite.all() else None
        records.append(
            {
                "shape": list(observed.shape),
                "maximum_absolute_error": maximum,
                "maximum_absolute_finite_difference": float(difference[finite].max()) if finite.any() else None,
                "observed_nonfinite_elements": int(np.count_nonzero(~np.isfinite(observed))),
                "reference_nonfinite_elements": int(np.count_nonzero(~np.isfinite(reference))),
                "difference_nonfinite_elements": int(np.count_nonzero(~finite)),
                "passed": bool(finite.all() and maximum <= gate),
            }
        )
    if any(not record["passed"] for record in records):
        raise NumericalGateError(records)
    return records


def persist_precheck_failure(destination: Path, report: dict, error: Exception) -> None:
    """Persist a bounded failure record and retain exact exception text separately."""
    exact = "".join(traceback.format_exception(error)).encode()
    error_path = destination.with_suffix(destination.suffix + ".error.txt")
    error_path.write_bytes(exact)
    message = str(error)
    failure = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error_summary": message[:2048],
        "error_summary_truncated": len(message) > 2048,
        "exact_error_log": {"path": str(error_path), "bytes": len(exact), "sha256": hashlib.sha256(exact).hexdigest()},
        "exported_or_native_execution": False,
    }
    if isinstance(error, NumericalGateError):
        failure["numerical_evidence"] = error.evidence
    result = {**report, **failure}
    try:
        bounded_report_bytes(result)
    except (ValueError, TypeError):
        # The previous record may itself be malformed/oversized; never erase the failure.
        result = {**failure, "prior_report_omitted": "Previous report was not bounded finite JSON."}
        try:
            bounded_report_bytes(result)
        except (ValueError, TypeError):
            result.pop("numerical_evidence", None)
            result["numerical_evidence_omitted"] = "Evidence exceeded the bounded JSON representation."
    write_report(destination, result)


@dataclass(frozen=True)
class SlotLayout:
    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_type: CutePipelineElementType
    source: str


ORDERED_COMPILED_STRIDE_FIELDS = {
    **{
        name: f"{name}_stride"
        for name in (
            "query",
            "key",
            "value",
            "output",
            "output_cotangent",
            "normalized_output",
            "query_cotangent",
            "key_cotangent",
            "value_cotangent",
        )
    },
    "normalized_output_cotangent": "output_stride",
    "normalized_output_cotangent_residual": "output_stride",
}


def ordered_slot_layouts(recipe, compiled_strides):
    """Bind the ordered exporter's declared layout policy before graph capture."""
    compiled_roles = set(ORDERED_COMPILED_STRIDE_FIELDS)
    contiguous_roles = {
        "log_sum_exp",
        "output_dot",
        "log_sum_exp_log2",
        "query_accumulator",
        "key_accumulator",
        "value_accumulator",
        "query_accumulation_semaphore",
        "key_accumulation_semaphore",
        "value_accumulation_semaphore",
    }
    if set(compiled_strides) != {slot.name for slot in recipe.slots if slot.name in compiled_roles}:
        raise ValueError("compiled stride fields do not cover the exact declared provider roles")
    layouts = []
    for slot in recipe.slots:
        if slot.name in compiled_roles:
            strides = compiled_strides[slot.name]
            source = "compiled provider stride field"
        elif slot.name in contiguous_roles:
            strides = tuple(math.prod(slot.shape[index + 1 :]) for index in range(len(slot.shape)))
            source = "ordered low-level recipe encoder contiguous-layout requirement"
        else:
            raise ValueError("ordered layout evidence does not recognize a declared slot")
        layouts.append(SlotLayout(slot.name, slot.shape, tuple(strides), slot.element_type, source))
    return tuple(layouts)


@dataclass(frozen=True)
class LaunchScheduleEvidence:
    recipe_digest: str
    schedule: dict
    slot_layouts: tuple[SlotLayout, ...]


def validate_full_completion_edge(from_port: int, to_port: int, dependency_type: int, reserved: bytes) -> None:
    if any(type(value) is not int or value != 0 for value in (from_port, to_port, dependency_type)) or any(reserved):
        raise ValueError("capture requires full-completion dependency edges with zero reserved bytes")


def _saved_launch_edges(recipe: CutePipelineLaunchRecipe, evidence: LaunchScheduleEvidence):
    if evidence.recipe_digest != recipe.digest:
        raise ValueError("saved launch schedule belongs to a different recipe")
    schedule = evidence.schedule
    if recipe.physical_event_provenance is not None:
        authoritative = recipe.physical_event_provenance["schedule"]
        if json.dumps(authoritative, sort_keys=True) != json.dumps(schedule, sort_keys=True):
            raise ValueError("saved launch evidence contradicts authoritative recipe provenance")
    if schedule.get("schema_version") != 2 or "regions" in schedule:
        raise ValueError("capture cannot map absent or nested internal coordinates to launches")
    realization = schedule.get("realizations", ())
    if len(realization) != 1 or realization[0].get("kind") != "erased_stream_order":
        raise ValueError("capture requires one saved launch-domain stream-order realization")
    relation = realization[0]["event"]["required_dependence"]["relation"]
    expected_domain = {
        "kind": "task_family",
        "name": "pipeline_launch",
        "axes": [{"name": "launch", "extent": len(recipe.launches)}],
        "placement": "cuda_stream",
    }
    if any(
        json.dumps(relation[key], sort_keys=True) != json.dumps(expected_domain, sort_keys=True)
        for key in ("source_domain", "target_domain")
    ):
        raise ValueError("saved launch coordinates do not map to this recipe")
    predecessors = {(tuple(source), tuple(target)) for source, target in schedule["predecessor_pairs"]}
    required = {(tuple(pair["source"]), tuple(pair["target"])) for pair in relation["pairs"]}
    if predecessors != required:
        raise ValueError("saved typed predecessor representations disagree")
    if any(
        len(source) != 1 or len(target) != 1 or not 0 <= source[0] < target[0] < len(recipe.launches)
        for source, target in required
    ):
        raise ValueError("saved typed dependency has invalid or reversed launch coordinates")
    recomputed = launch_dependency_event_schedule(recipe.launches, slots=recipe.slots)
    if not set(recomputed["predecessor_pairs"]) <= required:
        raise ValueError("saved typed schedule omits an independently recomputed slot dependency")
    return required


def _graph_reachability(graph: CapturedGraph) -> dict[int, set[int]]:
    if not 0 < len(graph.nodes) <= MAX_NODES or len(graph.edges) > MAX_EDGES:
        raise ValueError("capture graph exceeds node/edge bounds")
    ids = [node.node_id for node in graph.nodes]
    if len(set(ids)) != len(ids) or len(set(graph.edges)) != len(graph.edges):
        raise ValueError("capture has duplicate node IDs or edges")
    if any(source not in ids or target not in ids for source, target in graph.edges):
        raise ValueError("capture edge references a missing node")
    successors = {node_id: set() for node_id in ids}
    for source, target in graph.edges:
        successors[source].add(target)
    reachable = {}
    for source in ids:
        found, pending = set(), list(successors[source])
        while pending:
            target = pending.pop()
            if target == source:
                raise ValueError("capture graph contains a cycle")
            if target not in found:
                found.add(target)
                pending.extend(successors[target])
        reachable[source] = found
    return reachable


def _slot_addresses(
    recipe: CutePipelineLaunchRecipe, bindings: tuple[SlotBinding, ...], layouts: tuple[SlotLayout, ...]
) -> dict[str, int]:
    if len(bindings) > MAX_SLOTS or len({item.name for item in bindings}) != len(bindings):
        raise ValueError("slot bindings exceed bounds or repeat names")
    by_name = {binding.name: binding for binding in bindings}
    if set(by_name) != {slot.name for slot in recipe.slots}:
        raise ValueError("explicit bindings must cover exactly every recipe slot")
    if len({binding.device for binding in bindings}) != 1:
        raise ValueError("slot bindings must name one actual device")
    expected = {layout.name: layout for layout in layouts}
    if len(expected) != len(layouts) or set(expected) != set(by_name):
        raise ValueError("saved layout evidence must cover every slot exactly once")
    intervals = []
    for slot in recipe.slots:
        binding = by_name[slot.name]
        layout = expected[slot.name]
        if not layout.source or (binding.shape, binding.strides, binding.element_type) != (
            layout.shape,
            layout.strides,
            layout.element_type,
        ):
            raise ValueError(f"slot {slot.name} differs from independently saved layout evidence")
        if binding.shape != slot.shape or binding.element_type is not slot.element_type:
            raise ValueError(f"slot {slot.name} shape or dtype differs")
        if len(binding.strides) != len(binding.shape) or any(stride <= 0 for stride in binding.strides):
            raise ValueError(f"slot {slot.name} has unsupported stride metadata")
        element_bytes = slot.required_bytes // math.prod(slot.shape)
        # Contiguous or dense axis permutations only; no holes, overlap or hidden copies.
        expected_stride = 1
        for extent, stride in sorted(zip(binding.shape, binding.strides, strict=True), key=lambda pair: pair[1]):
            if extent == 1:
                continue
            if stride != expected_stride:
                raise ValueError(f"slot {slot.name} is not a dense bijective storage view")
            expected_stride *= extent
        if (
            binding.storage_base <= 0
            or binding.storage_bytes <= 0
            or binding.byte_offset < 0
            or binding.pointer != binding.storage_base + binding.byte_offset
            or binding.storage_base + binding.storage_bytes >= _POINTER_LIMIT
            or binding.byte_offset + slot.required_bytes > binding.storage_bytes
            or binding.byte_offset % element_bytes
            or binding.pointer % slot.alignment
        ):
            raise ValueError(f"slot {slot.name} pointer, extent, offset or alignment differs")
        intervals.append((slot, binding.pointer, binding.pointer + slot.required_bytes))
    access = [_launch_slot_accesses(launch) for launch in recipe.launches]
    for index, (left, begin, end) in enumerate(intervals):
        for right, other_begin, other_end in intervals[index + 1 :]:
            if max(begin, other_begin) >= min(end, other_end):
                continue
            left_uses = [i for i, item in enumerate(access) if left.name in item]
            right_uses = [i for i, item in enumerate(access) if right.name in item]
            if (
                left.kind is not CutePipelineSlotKind.SCRATCH
                or right.kind is not CutePipelineSlotKind.SCRATCH
                or left.reuse_group is None
                or left.reuse_group != right.reuse_group
                or not left_uses
                or not right_uses
                or not (max(left_uses) < min(right_uses) or max(right_uses) < min(left_uses))
            ):
                raise ValueError("captured buffers alias without disjoint declared scratch lifetimes")
    return {name: item.pointer for name, item in by_name.items()}


def compare_captured_launches(
    recipe: CutePipelineLaunchRecipe,
    graph: CapturedGraph,
    bindings: tuple[SlotBinding, ...],
    evidence: LaunchScheduleEvidence,
) -> dict:
    """Compare every executing node against independent actual allocation identities.

    Driver offsets prove offset compatibility with recipe alignment. They do not
    report the compiler's declared alignment or the host argument-buffer base.
    """
    reachable = _graph_reachability(graph)
    addresses = _slot_addresses(recipe, bindings, evidence.slot_layouts)
    kernels = {}
    for node in graph.nodes:
        if node.kind is GraphNodeKind.EMPTY and node.kernel is None:
            continue
        if node.kind is not GraphNodeKind.KERNEL or node.kernel is None:
            raise ValueError("unsupported or malformed captured graph node")
        if node.kernel.symbol in kernels:
            raise ValueError("duplicate captured kernel symbol")
        kernels[node.kernel.symbol] = node
    symbols = [launch.kernel_symbol for launch in recipe.launches]
    if len(set(symbols)) != len(symbols) or set(kernels) != set(symbols):
        raise ValueError("captured roles have missing, extra or ambiguous kernel symbols")
    required = _saved_launch_edges(recipe, evidence)
    for (source,), (target,) in required:
        if kernels[symbols[target]].node_id not in reachable[kernels[symbols[source]].node_id]:
            raise ValueError("captured graph lacks a required directed dependency path")
    # The exported pipeline has source-order stream execution even for independent launches.
    for source, target in pairwise(symbols):
        if kernels[target].node_id not in reachable[kernels[source].node_id]:
            raise ValueError("captured launch order does not realize the exported stream order")
    used_bytes = 0
    reports = []
    for launch in recipe.launches:
        node = kernels[launch.kernel_symbol]
        actual = node.kernel
        if (actual.grid, actual.block, actual.shared_bytes) != (
            launch.grid,
            launch.block,
            launch.dynamic_shared_memory_bytes,
        ) or actual.cluster not in ((0, 0, 0), (1, 1, 1)):
            raise ValueError(f"launch {launch.name} grid/block/shared/cluster differs")
        attrs = {attr.kind: attr.value for attr in actual.attributes}
        if len(attrs) != len(actual.attributes) or any(
            attrs.get(attr.kind) != attr.value for attr in launch.function_attributes
        ):
            raise ValueError(f"launch {launch.name} function attributes differ")
        if len(actual.parameters) > MAX_PARAMETERS or len(actual.parameters) != len(launch.arguments):
            raise ValueError(f"launch {launch.name} parameter count differs")
        arguments = []
        previous_end = 0
        for index, (argument, parameter) in enumerate(zip(launch.arguments, actual.parameters, strict=True)):
            size = cute_kernel_argument_size(argument)
            alignment = argument_alignment(argument)
            if (
                size > MAX_PARAMETER_BYTES
                or parameter.size != size
                or len(parameter.data) != size
                or parameter.offset < previous_end
                or parameter.offset % alignment
                or parameter.offset + size > MAX_CAPTURE_BYTES
            ):
                raise ValueError(f"launch {launch.name} parameter {index} size/offset/alignment differs")
            previous_end = parameter.offset + size
            used_bytes += size
            if used_bytes > MAX_CAPTURE_BYTES:
                raise ValueError("capture exceeds aggregate parameter-byte bound")
            for _, slot_name, offset in argument_relocations(argument):
                slot = next(slot for slot in recipe.slots if slot.name == slot_name)
                if offset < 0 or offset >= slot.required_bytes or addresses[slot_name] + offset >= _POINTER_LIMIT:
                    raise ValueError("argument relocation exceeds explicit slot extent")
            expected = expected_argument_bytes(argument, addresses)
            meaningful = set(range(size))
            if isinstance(argument, CutePipelineStructuredArgument):
                meaningful = {
                    i
                    for field in argument.fields
                    for i in range(field.destination_byte_offset, field.destination_byte_offset + field.width)
                }
            differences = [i for i in sorted(meaningful) if parameter.data[i] != expected[i]]
            if differences:
                raise ValueError(
                    f"launch {launch.name} argument {argument.name} differs at declared bytes {differences[:16]}"
                )
            arguments.append(
                {
                    "index": index,
                    "name": argument.name,
                    "driver_offset": parameter.offset,
                    "driver_size": parameter.size,
                    "recipe_alignment": alignment,
                    "declared_bytes_compared": len(meaningful),
                    "sha256": hashlib.sha256(parameter.data).hexdigest(),
                }
            )
        reports.append(
            {
                "name": launch.name,
                "symbol": actual.symbol,
                "node": node.node_id,
                "parameters": arguments,
                "grid": actual.grid,
                "block": actual.block,
                "shared_bytes": actual.shared_bytes,
                "cluster": actual.cluster,
                "function_attributes": [attr.canonical_record for attr in actual.attributes],
            }
        )
    report = {
        "schema_version": 1,
        "launch_schedule_sha256": hashlib.sha256(json.dumps(evidence.schedule, sort_keys=True).encode()).hexdigest(),
        "metadata_matches": True,
        "native_execution_accepted": False,
        "nodes": len(graph.nodes),
        "edges": graph.edges,
        "launches": reports,
        "captured_parameter_bytes": used_bytes,
        "alignment_evidence": (
            "Driver offset/size compatibility with recipe alignment; tensor pointer alignment checked independently."
        ),
        "unresolved": [
            "cuFuncGetParamInfo does not expose declared compiler alignment or host argument-buffer base alignment."
        ],
        "saved_slot_layouts": [
            {
                "name": layout.name,
                "shape": layout.shape,
                "strides": layout.strides,
                "dtype": layout.element_type.value,
                "source": layout.source,
            }
            for layout in evidence.slot_layouts
        ],
        "slots": [
            {
                "name": b.name,
                "storage_base": b.storage_base,
                "storage_bytes": b.storage_bytes,
                "byte_offset": b.byte_offset,
                "pointer": b.pointer,
                "shape": b.shape,
                "strides": b.strides,
                "dtype": b.element_type.value,
                "device": b.device,
            }
            for b in bindings
        ],
    }
    bounded_report_bytes(report)
    return report


def attention_precheck_function(causal: bool, expanded: bool):
    def function(query, key, value):
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) * (128**-0.5)
        if causal:
            query_positions = jnp.arange(query.shape[1])[:, None]
            key_positions = jnp.arange(key.shape[1])[None, :]
            scores = jnp.where(key_positions <= query_positions, scores, -jnp.inf)
        maximum = jnp.max(scores, axis=-1, keepdims=True)
        exponential = jnp.exp(scores - maximum)
        probabilities = exponential / jnp.sum(exponential, axis=-1, keepdims=True)
        normalized = jnp.einsum("bhqk,bkhv->bqhv", probabilities, value)
        if expanded:
            return jnp.tanh(normalized) + 0.25 * normalized
        return normalized

    return function


def prepare_ordered_precheck_plan(examples, *, causal: bool, expanded: bool):
    function = attention_precheck_function(causal, expanded)
    trace = trace_jax_owned_normalized_exp_value_and_grad(
        function, *examples[:3], output_cotangent=examples[3], numerical_policy=NumericalPolicy.FAST
    )
    request = jax_tensor_program_executable_request(trace.value_and_grad, tile_size=128, target=PhysicalTarget.SM90)
    return prepare_sm90_split_normalized_exp_value_and_grad(request, trace, tile_size=128)
