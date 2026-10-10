# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded staging for generic injective Transport relations on one CUDA stream."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shuttle.ir import DType
from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineExpression,
    CuteAffineExpressionKind,
    CuteAffineMap,
    CuteEventCoverEndpoint,
    CuteEventCoverPiece,
    CuteEventRealization,
    CuteEventRealizationKind,
    CuteIntegerSet,
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineScalarArgument,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    affine_constant,
    affine_dimension,
)
from tile_lifetime.event_buffering import BoundedBufferPlan, derive_bounded_buffer_plan
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    EventTensorPlan,
    ImperativeEventOperation,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_plan_record,
    lower_event_tensor_plan,
    phased_event_storage_binding,
    verify_event_dataflow_program,
)
from tile_lifetime.tensor_program import (
    IndexExpression,
    IndexExpressionKind,
    TensorAxis,
    TensorProgram,
    TransportPrimitive,
    canonical_tensor_program_record,
)

_THREADS = 256


@dataclass(frozen=True)
class BoundedTransportPipeline:
    """Generated kernels, exact Event plans, and a finite staging assignment."""

    operation: TransportPrimitive
    tile_elements: int
    capacity: int
    event_program: EventDataflowProgram
    readiness: EventTensorPlan
    reuse: EventTensorPlan
    buffer: BoundedBufferPlan
    readiness_storage: EventStorageBinding
    reuse_storage: EventStorageBinding
    source: str
    source_sha256: str
    recipe: CutePipelineLaunchRecipe
    request: CutePipelineCompileRequest

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "request": self.request.digest,
                    "recipe": self.recipe.digest,
                    "source": self.source_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class BoundedTransportAotPlan:
    """CUDA commands and output paths for one bounded Transport pipeline."""

    pipeline: BoundedTransportPipeline
    source: str
    source_sha256: str
    source_path: Path
    host_object_path: Path
    device_image_path: Path
    host_compile_argv: tuple[str, ...]
    device_compile_argv: tuple[str, ...]
    host_entry: str
    architecture: str

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "request_digest": self.pipeline.request.digest,
            "recipe_digest": self.pipeline.recipe.digest,
            "host_entry": self.host_entry,
            "architecture": self.architecture,
        }


def lower_bounded_transport(
    operation: TransportPrimitive,
    *,
    tile_elements: int,
    capacity: int,
) -> BoundedTransportPipeline:
    """Lower an injective static Transport through bounded staged generations."""
    if operation.source.dtype is not DType.FP32:
        raise ValueError("bounded CUDA Transport currently supports FP32 payloads")
    if operation.routing_inputs or operation.predicate is not None:
        raise ValueError("bounded CUDA Transport currently requires a static unpredicated relation")
    if operation.source_indexing.predicate is not None or operation.destination_indexing.predicate is not None:
        raise ValueError("bounded CUDA Transport currently requires unpredicated operand index relations")
    if tile_elements <= 0 or capacity <= 0:
        raise ValueError("bounded CUDA Transport tile size and capacity must be positive")
    domain_elements = math.prod(axis.extent for axis in operation.domain_axes)
    if domain_elements != math.prod(operation.destination.shape):
        raise ValueError("bounded CUDA Transport requires a destination-covering relation")
    source_axes = _axis_indices(operation.source_indexing.indices, label="source")
    destination_axes = _axis_indices(operation.destination_indexing.indices, label="destination")
    if len(set(source_axes)) != len(operation.domain_axes) or set(source_axes) != set(operation.domain_axes):
        raise ValueError("bounded CUDA Transport source relation must be an axis permutation")
    for axis, source_axis in zip(source_axes, operation.source.axes, strict=True):
        if axis.extent != source_axis.extent:
            raise ValueError("bounded CUDA Transport source relation does not preserve axis extents")
    if len(set(destination_axes)) != len(operation.domain_axes) or set(destination_axes) != set(operation.domain_axes):
        raise ValueError("bounded CUDA Transport destination relation is not proven injective")
    for axis, destination_axis in zip(destination_axes, operation.destination.axes, strict=True):
        if axis.extent != destination_axis.extent:
            raise ValueError("bounded CUDA Transport destination relation does not preserve axis extents")
    generation_count = math.ceil(domain_elements / tile_elements)
    capacity = min(capacity, generation_count)
    if generation_count <= capacity:
        raise ValueError("bounded CUDA Transport requires more generations than staging slots")
    task_axis = TaskAxis("generation", generation_count)
    producers = TaskFamily("transport_producer", (task_axis,), "device")
    consumers = TaskFamily("transport_consumer", (task_axis,), "device")
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    ready_relation = TaskRelation.from_pairs(
        producers,
        consumers,
        tuple((((generation,), (generation,))) for generation in range(generation_count)),
    )
    ready_dependence = TaskDependence(ready_relation, visibility)
    readiness = derive_event_tensor_plan(
        ready_dependence,
        name="transport_tile_ready",
        generation_policy=EventGenerationPolicy.PHASED,
        scheduling_mode=EventSchedulingMode.STATIC,
    )
    initial_program = EventDataflowProgram((producers, consumers), (ready_dependence,), (readiness,))
    buffer = derive_bounded_buffer_plan(
        name="transport_staging",
        program=initial_program,
        producer=producers,
        uses=(ready_relation,),
        capacity=capacity,
        allocation_owner_for={coordinate: () for coordinate in producers.coordinates},
        slot_for={coordinate: coordinate[0] % capacity for coordinate in producers.coordinates},
        generation_for={coordinate: coordinate[0] // capacity for coordinate in producers.coordinates},
        visibility=visibility,
    )
    if len(buffer.reuse_dependences) != 1:
        raise ValueError("bounded CUDA Transport requires one exact staging-slot reuse relation")
    reuse_dependence = buffer.reuse_dependences[0]
    reuse = derive_event_tensor_plan(
        reuse_dependence,
        name="transport_slot_reuse",
        generation_policy=EventGenerationPolicy.PHASED,
        scheduling_mode=EventSchedulingMode.STATIC,
    )
    event_program = EventDataflowProgram(
        (producers, consumers),
        (ready_dependence, reuse_dependence),
        (readiness, reuse),
    )
    verify_event_dataflow_program(event_program)
    readiness_storage = phased_event_storage_binding(
        readiness,
        slot=lambda coordinate: coordinate[0] % capacity,
        generation=lambda coordinate: coordinate[0] // capacity,
    )
    reuse_storage = phased_event_storage_binding(
        reuse,
        slot=lambda coordinate: coordinate[0] % capacity,
        generation=lambda coordinate: coordinate[0] // capacity,
    )
    program_record = canonical_tensor_program_record(_transport_program(operation))
    identity = hashlib.sha256(json.dumps(program_record, sort_keys=True).encode()).hexdigest()[:16]
    producer_symbol = f"shuttle_bounded_transport_produce_{identity}"
    consumer_symbol = f"shuttle_bounded_transport_consume_{identity}"
    source = _render_source(
        operation,
        source_axes,
        destination_axes,
        tile_elements=tile_elements,
        producer_symbol=producer_symbol,
        consumer_symbol=consumer_symbol,
    )
    recipe = _launch_recipe(
        operation,
        generation_count=generation_count,
        tile_elements=tile_elements,
        capacity=capacity,
        producer_symbol=producer_symbol,
        consumer_symbol=consumer_symbol,
    )
    request = CutePipelineCompileRequest(
        schema_version=1,
        target="sm_90",
        numerical_policy=operation.order.value,
        physical_template="bounded_transport_stream_order_v1",
        tensors=tuple(slot.canonical_record for slot in recipe.slots),
        scalar_programs={},
        index_relations=(cast(tuple[dict[str, Any], ...], program_record["operations"])[0],),
        contraction_dimensions={},
        schedule={
            "tile_elements": tile_elements,
            "threads": _THREADS,
            "capacity": capacity,
            "generation_count": generation_count,
            "scope": EventMemoryScope.DEVICE.value,
            "buffer_slots": buffer.slots,
            "buffer_generations": buffer.generations,
            "readiness_event": event_tensor_plan_record(readiness),
            "reuse_event": event_tensor_plan_record(reuse),
            "readiness_operations": tuple(
                _imperative_record(item)
                for item in lower_event_tensor_plan(
                    readiness,
                    scheduling_mode=EventSchedulingMode.STATIC,
                    storage_binding=readiness_storage,
                )
            ),
            "reuse_operations": tuple(
                _imperative_record(item)
                for item in lower_event_tensor_plan(
                    reuse,
                    scheduling_mode=EventSchedulingMode.STATIC,
                    storage_binding=reuse_storage,
                )
            ),
            "event_legalization": "cuda_stream_order",
        },
        algebra_identity={"transport_fingerprint": identity},
    )
    return BoundedTransportPipeline(
        operation,
        tile_elements,
        capacity,
        event_program,
        readiness,
        reuse,
        buffer,
        readiness_storage,
        reuse_storage,
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        recipe,
        request,
    )


def bounded_transport_aot_plan(
    pipeline: BoundedTransportPipeline,
    *,
    output_directory: Path,
    nvcc: Path,
    architecture: str = "sm_90a",
) -> BoundedTransportAotPlan:
    """Create host-object and fatbin compilation commands."""
    if architecture not in {"sm_90", "sm_90a"}:
        raise ValueError("bounded CUDA Transport currently requires an SM90 architecture")
    host_entry = f"shuttle_bounded_transport_{pipeline.source_sha256[:16]}"
    source = pipeline.source + _render_host_wrapper(pipeline, host_entry)
    source_path = output_directory / f"{host_entry}.cu"
    host_object_path = output_directory / f"{host_entry}.o"
    device_image_path = output_directory / f"{host_entry}.fatbin"
    common = (str(nvcc), "--std=c++17", f"--gpu-architecture={architecture}")
    return BoundedTransportAotPlan(
        pipeline=pipeline,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_path=source_path,
        host_object_path=host_object_path,
        device_image_path=device_image_path,
        host_compile_argv=(
            *common,
            "--compiler-options=-fPIC",
            "--compile",
            str(source_path),
            "-o",
            str(host_object_path),
        ),
        device_compile_argv=(*common, "--fatbin", str(source_path), "-o", str(device_image_path)),
        host_entry=host_entry,
        architecture=architecture,
    )


def compile_bounded_transport_artifact(plan: BoundedTransportAotPlan) -> CutePipelineArtifact:
    """Compile and seal a bounded Transport artifact."""
    plan.source_path.parent.mkdir(parents=True, exist_ok=True)
    plan.source_path.write_text(plan.source)
    subprocess.run(plan.host_compile_argv, check=True)
    subprocess.run(plan.device_compile_argv, check=True)
    version = subprocess.run(
        (plan.host_compile_argv[0], "--version"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return seal_bounded_transport_artifact(
        plan,
        host_object=plan.host_object_path.read_bytes(),
        device_image=plan.device_image_path.read_bytes(),
        nvcc_version=version,
    )


def seal_bounded_transport_artifact(
    plan: BoundedTransportAotPlan,
    *,
    host_object: bytes,
    device_image: bytes,
    nvcc_version: str,
) -> CutePipelineArtifact:
    """Seal compiled payloads against the Transport, Event, and recipe identities."""
    if not host_object or not device_image:
        raise ValueError("bounded CUDA Transport compilation must produce a host object and CUDA fatbin")
    return CutePipelineArtifact(
        request=plan.pipeline.request,
        host_object=host_object,
        device_image=device_image,
        kernel_names=tuple(dict.fromkeys(launch.kernel_symbol for launch in plan.pipeline.recipe.launches)),
        host_entry=plan.host_entry,
        compiler_ir=plan.source,
        compiler_ir_format="cuda-c++",
        toolchain={"nvcc": nvcc_version, "architecture": plan.architecture},
        source_digests={
            "generated_source": plan.source_sha256,
            "launch_recipe": plan.pipeline.recipe.digest,
        },
        launch_recipe=plan.pipeline.recipe,
    )


def _transport_program(operation: TransportPrimitive) -> TensorProgram:
    return TensorProgram(operation.inputs, (operation,), operation.outputs)


def _axis_indices(expressions: tuple[IndexExpression, ...], *, label: str) -> tuple[TensorAxis, ...]:
    axes: list[TensorAxis] = []
    for expression in expressions:
        if expression.kind is not IndexExpressionKind.AXIS or expression.axis is None:
            raise ValueError(f"bounded CUDA Transport {label} indexing currently requires direct axis expressions")
        axes.append(expression.axis)
    return tuple(axes)


def _render_source(
    operation: TransportPrimitive,
    source_axes: tuple[TensorAxis, ...],
    destination_axes: tuple[TensorAxis, ...],
    *,
    tile_elements: int,
    producer_symbol: str,
    consumer_symbol: str,
) -> str:
    coordinates = _coordinate_lines(operation.domain_axes)
    source_offset = _offset(operation.source.axes, source_axes, operation.domain_axes)
    destination_offset = _offset(operation.destination.axes, destination_axes, operation.domain_axes)
    domain_elements = math.prod(axis.extent for axis in operation.domain_axes)
    return f"""#include <cuda_runtime.h>
#include <stdint.h>
extern "C" __global__ void {producer_symbol}(const float* source, float* staging,
                                              int64_t tile_begin, int32_t tile_count,
                                              int32_t slot) {{
  const int32_t local = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= tile_count) return;
  const int64_t linear = tile_begin + local;
{coordinates}
  staging[static_cast<int64_t>(slot) * {tile_elements} + local] = source[{source_offset}];
}}
extern "C" __global__ void {consumer_symbol}(const float* staging, float* destination,
                                              int64_t tile_begin, int32_t tile_count,
                                              int32_t slot) {{
  const int32_t local = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= tile_count) return;
  const int64_t linear = tile_begin + local;
{coordinates}
  destination[{destination_offset}] = staging[static_cast<int64_t>(slot) * {tile_elements} + local];
}}
static_assert({domain_elements} > 0);
"""


def _coordinate_lines(axes: tuple[TensorAxis, ...]) -> str:
    divisor = math.prod(axis.extent for axis in axes)
    lines = []
    for index, axis in enumerate(axes):
        divisor //= axis.extent
        expression = f"(linear / {divisor}) % {axis.extent}" if divisor != 1 else f"linear % {axis.extent}"
        lines.append(f"  const int64_t coordinate_{index} = {expression};")
    return "\n".join(lines)


def _offset(
    value_axes: tuple[TensorAxis, ...],
    indexing_axes: tuple[TensorAxis, ...],
    domain_axes: tuple[TensorAxis, ...],
) -> str:
    coordinates = {axis: f"coordinate_{index}" for index, axis in enumerate(domain_axes)}
    stride = 1
    terms = []
    for value_axis, indexing_axis in reversed(tuple(zip(value_axes, indexing_axes, strict=True))):
        terms.append(f"({coordinates[indexing_axis]}) * {stride}")
        stride *= value_axis.extent
    return " + ".join(reversed(terms))


def _launch_recipe(
    operation: TransportPrimitive,
    *,
    generation_count: int,
    tile_elements: int,
    capacity: int,
    producer_symbol: str,
    consumer_symbol: str,
) -> CutePipelineLaunchRecipe:
    source = CutePipelineSlot(
        "source", CutePipelineElementType.F32, operation.source.shape, CutePipelineSlotKind.PUBLIC_INPUT, 16, 0
    )
    destination = CutePipelineSlot(
        "destination",
        CutePipelineElementType.F32,
        operation.destination.shape,
        CutePipelineSlotKind.PUBLIC_OUTPUT,
        16,
        output_index=0,
    )
    staging = CutePipelineSlot(
        "staging", CutePipelineElementType.F32, (capacity, tile_elements), CutePipelineSlotKind.SCRATCH, 16
    )
    launches = []
    domain_elements = math.prod(axis.extent for axis in operation.domain_axes)
    for generation in range(generation_count):
        begin = generation * tile_elements
        count = min(tile_elements, domain_elements - begin)
        blocks = math.ceil(count / _THREADS)
        scalars = (
            CutePipelineScalarArgument("tile_begin", CutePipelineElementType.I64, begin),
            CutePipelineScalarArgument("tile_count", CutePipelineElementType.I32, count),
            CutePipelineScalarArgument("slot", CutePipelineElementType.I32, generation % capacity),
        )
        launches.extend(
            (
                CuteKernelLaunch(
                    f"produce_{generation}",
                    producer_symbol,
                    (blocks, 1, 1),
                    (_THREADS, 1, 1),
                    0,
                    (),
                    (
                        CutePipelineSlotArgument("source", "source", CutePipelineSlotAccess.READ),
                        CutePipelineSlotArgument("staging", "staging", CutePipelineSlotAccess.WRITE),
                        *scalars,
                    ),
                    source_tasks=(0,),
                ),
                CuteKernelLaunch(
                    f"consume_{generation}",
                    consumer_symbol,
                    (blocks, 1, 1),
                    (_THREADS, 1, 1),
                    0,
                    (),
                    (
                        CutePipelineSlotArgument("staging", "staging", CutePipelineSlotAccess.READ),
                        CutePipelineSlotArgument("destination", "destination", CutePipelineSlotAccess.WRITE),
                        *scalars,
                    ),
                    source_tasks=(1,),
                ),
            )
        )
    readiness = _stream_event(
        0, tuple((2 * generation, 2 * generation + 1, generation) for generation in range(generation_count))
    )
    reuse = _stream_event(
        1,
        tuple(
            (2 * (generation - capacity) + 1, 2 * generation, generation)
            for generation in range(capacity, generation_count)
        ),
    )
    return CutePipelineLaunchRecipe(
        2,
        (source, destination, staging),
        tuple(launches),
        event_realizations=(readiness, reuse),
        host_call_arguments=("source", "destination", "staging"),
    )


def _stream_event(event: int, edges: tuple[tuple[int, int, int], ...]) -> CuteEventRealization:
    if not edges:
        raise ValueError("bounded CUDA Transport stream Event must contain at least one dependence")
    pieces = []
    witnesses = []
    owner = CuteAffineMap(1, (affine_constant(0), affine_constant(0), affine_constant(0)))
    for producer, consumer, generation in edges:
        subset = CuteIntegerSet(
            1,
            (
                CuteAffineExpression(
                    CuteAffineExpressionKind.ADD,
                    operands=(affine_dimension(0), affine_constant(-generation)),
                ),
            ),
            (True,),
        )
        time = CuteAffineMap(1, (affine_constant(generation),))
        pieces.append(
            CuteEventCoverPiece(
                subset,
                CuteEventCoverEndpoint(producer, owner, time),
                CuteEventCoverEndpoint(consumer, owner, time),
            )
        )
        witnesses.extend((producer, generation, consumer, generation))
    return CuteEventRealization(event, CuteEventRealizationKind.STREAM_ORDER, tuple(witnesses), tuple(pieces))


def _imperative_record(operation: ImperativeEventOperation) -> dict[str, Any]:
    return {
        "kind": operation.kind.value,
        "event": operation.event,
        "task": operation.task,
        "count": operation.count,
        "scope": operation.memory_scope.value,
        "slot": operation.storage_slot,
        "generation": operation.generation,
        "release": operation.visibility.release_on_notify,
        "acquire": operation.visibility.acquire_before_consumer,
    }


def _render_host_wrapper(pipeline: BoundedTransportPipeline, host_entry: str) -> str:
    launches = []
    for launch in pipeline.recipe.launches:
        source, destination, begin, count, slot = launch.arguments
        assert isinstance(source, CutePipelineSlotArgument)
        assert isinstance(destination, CutePipelineSlotArgument)
        assert isinstance(begin, CutePipelineScalarArgument)
        assert isinstance(count, CutePipelineScalarArgument)
        assert isinstance(slot, CutePipelineScalarArgument)
        launches.append(
            f"  {launch.kernel_symbol}<<<{launch.grid[0]}, {_THREADS}, 0, stream>>>("
            f"reinterpret_cast<float*>(arguments[{0 if source.slot == 'source' else 2}]), "
            f"reinterpret_cast<float*>(arguments[{1 if destination.slot == 'destination' else 2}]), "
            f"{begin.value}, {count.value}, {slot.value});"
        )
    return "\n".join(
        (
            f'extern "C" int {host_entry}(void** arguments, cudaStream_t stream) {{',
            *launches,
            "  return static_cast<int>(cudaGetLastError());",
            "}",
            "",
        )
    )
