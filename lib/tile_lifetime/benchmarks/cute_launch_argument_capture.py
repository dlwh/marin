# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read node-owned CUDA graph parameters to diagnose exported launch recipes."""

import struct
from collections.abc import Callable
from typing import Any

import cuda.bindings.driver as cuda
import torch
from strict_launch_comparison import (
    MAX_CAPTURE_BYTES,
    MAX_EDGES,
    MAX_NODES,
    MAX_PARAMETER_BYTES,
    MAX_PARAMETERS,
    CapturedGraph,
    CapturedKernel,
    CapturedNode,
    CapturedParameter,
    GraphNodeKind,
    LaunchScheduleEvidence,
    SlotBinding,
    compare_captured_launches,
    validate_full_completion_edge,
)
from strict_launch_comparison import argument_relocations as _relocations
from strict_launch_comparison import expected_argument_bytes as _expected_bytes

from tile_lifetime.backends.h100.cute_launch_capture import checked_cuda as _cuda
from tile_lifetime.backends.h100.cute_launch_capture import kernel_parameter_bytes as _parameter_bytes
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineStructuredArgument,
    CutePipelineTensorMapArgument,
)


def capture_cute_launch_argument_report(
    call: Callable[[], None], recipe: CutePipelineLaunchRecipe, slot_addresses: dict[str, int]
) -> dict[str, Any]:
    """Capture a warmed call and compare actual parameter bytes to its recipe.

    Keep inputs and outputs alive through this call. The graph owns captured
    temporary allocations and argument storage until all bytes have been copied.
    Unknown slot addresses are inferred from declared relocations, with repeated
    occurrences checked for consistency. They do not establish slot identity.
    """
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        call()
    handle = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = _cuda(cuda.cuGraphGetNodes(handle))
    nodes, _ = _cuda(cuda.cuGraphGetNodes(handle, count))
    captured = {}
    ignored = []
    required_symbols = {launch.kernel_symbol for launch in recipe.launches}
    for node in nodes:
        (kind,) = _cuda(cuda.cuGraphNodeGetType(node))
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            ignored.append({"node": int(node), "kind": str(kind)})
            continue
        (parameters,) = _cuda(cuda.cuGraphKernelNodeGetParams(node))
        (name,) = _cuda(cuda.cuFuncGetName(parameters.func))
        symbol = name.decode()
        if symbol not in required_symbols:
            ignored.append({"node": int(node), "kind": str(kind), "symbol": symbol})
            continue
        if symbol in captured:
            raise ValueError(f"argument capture requires unique kernel symbols, repeated {symbol!r}")
        captured[symbol] = (
            _parameter_bytes(parameters),
            (parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ),
            (parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ),
            parameters.sharedMemBytes,
        )
    addresses = dict(slot_addresses)
    inferred = {}
    binding_conflicts = []
    for launch in recipe.launches:
        if launch.kernel_symbol not in captured:
            raise ValueError(f"kernel {launch.kernel_symbol!r} was not captured")
        actual = captured[launch.kernel_symbol][0]
        if len(actual) != len(launch.arguments):
            raise ValueError(f"captured parameter count differs for {launch.name!r}")
        for argument, data in zip(launch.arguments, actual, strict=True):
            for offset, slot, byte_offset in _relocations(argument):
                observed = struct.unpack_from("<Q", data, offset)[0] - byte_offset
                if slot not in addresses:
                    addresses[slot] = observed
                    inferred[slot] = observed
                elif addresses[slot] != observed:
                    binding_conflicts.append(
                        {
                            "launch": launch.name,
                            "argument": argument.name,
                            "slot": slot,
                            "expected": addresses[slot],
                            "actual": observed,
                        }
                    )
    reports = []
    for launch in recipe.launches:
        actual, grid, block, shared = captured[launch.kernel_symbol]
        arguments = []
        for index, (argument, data) in enumerate(zip(launch.arguments, actual, strict=True)):
            if isinstance(argument, CutePipelineTensorMapArgument):
                arguments.append(
                    {
                        "index": index,
                        "name": argument.name,
                        "actual_hex": data.hex(),
                        "matches": False,
                        "unsupported": "runtime-encoded TensorMap",
                    }
                )
                continue
            expected = _expected_bytes(argument, addresses)
            differences = [index for index in range(min(len(data), len(expected))) if data[index] != expected[index]]
            meaningful = set(range(len(expected)))
            if isinstance(argument, CutePipelineStructuredArgument):
                meaningful = {
                    index
                    for field in argument.fields
                    for index in range(field.destination_byte_offset, field.destination_byte_offset + field.width)
                }
            arguments.append(
                {
                    "index": index,
                    "name": argument.name,
                    "actual_hex": data.hex(),
                    "expected_hex": expected.hex(),
                    "differing_bytes": differences,
                    "differing_declared_bytes": sorted(set(differences) & meaningful),
                    "matches": len(data) == len(expected) and not (set(differences) & meaningful),
                }
            )
        reports.append(
            {
                "launch": launch.name,
                "kernel_symbol": launch.kernel_symbol,
                "arguments": arguments,
                "actual_grid": grid,
                "actual_block": block,
                "actual_shared_bytes": shared,
                "launch_dimensions_match": (
                    (grid, block, shared) == (launch.grid, launch.block, launch.dynamic_shared_memory_bytes)
                ),
            }
        )
    return {
        "schema_version": 1,
        "launches": reports,
        "ignored_nodes": ignored,
        "provided_slot_addresses": slot_addresses,
        "inferred_slot_addresses": inferred,
        "binding_conflicts": binding_conflicts,
        "matches": (
            not binding_conflicts
            and all(
                item["launch_dimensions_match"] and all(argument["matches"] for argument in item["arguments"])
                for item in reports
            )
        ),
    }


def preflight_cute_launch_argument_capture() -> dict[str, Any]:
    """Exercise retained graph metadata access before compiling the diagnostic workload."""
    source = torch.ones(1, device="cuda")
    destination = torch.empty_like(source)
    torch.add(source, 1, out=destination)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        torch.add(source, 1, out=destination)
    handle = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = _cuda(cuda.cuGraphGetNodes(handle))
    nodes, _ = _cuda(cuda.cuGraphGetNodes(handle, count))
    kernels = []
    for node in nodes:
        (kind,) = _cuda(cuda.cuGraphNodeGetType(node))
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        (parameters,) = _cuda(cuda.cuGraphKernelNodeGetParams(node))
        kernels.append({"parameter_bytes": [len(data) for data in _parameter_bytes(parameters)]})
    if not kernels:
        raise ValueError("retained graph preflight did not capture a CUDA kernel")
    return {"nodes": count, "kernels": kernels}


def explicit_tensor_slot_bindings(
    recipe: CutePipelineLaunchRecipe, tensors: dict[str, torch.Tensor]
) -> tuple[SlotBinding, ...]:
    """Read identities from caller/provider tensors before parameter inspection."""
    if set(tensors) != {slot.name for slot in recipe.slots}:
        raise ValueError("strict capture requires an explicit actual tensor for every recipe slot")
    element_types = {
        torch.bfloat16: CutePipelineElementType.BF16,
        torch.float32: CutePipelineElementType.F32,
        torch.int32: CutePipelineElementType.I32,
    }
    bindings = []
    for slot in recipe.slots:
        tensor = tensors[slot.name]
        if not tensor.is_cuda or tensor.dtype not in element_types:
            raise ValueError("strict ordered capture requires supported actual CUDA tensors")
        storage = tensor.untyped_storage()
        bindings.append(
            SlotBinding(
                slot.name,
                storage.data_ptr(),
                storage.nbytes(),
                tensor.storage_offset() * tensor.element_size(),
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                element_types[tensor.dtype],
                tensor.device.index,
            )
        )
    return tuple(bindings)


def capture_strict_cute_launch_report(
    call: Callable[[], None],
    recipe: CutePipelineLaunchRecipe,
    tensors: dict[str, torch.Tensor],
    evidence: LaunchScheduleEvidence,
) -> tuple[dict[str, Any], torch.cuda.CUDAGraph]:
    """Retain every node/edge and copy bounded graph-owned bytes before release.

    This captures compiler-owned execution only. It never executes the exported
    recipe. The returned graph is retained for separately gated replay checks.
    """
    bindings = explicit_tensor_slot_bindings(recipe, tensors)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        call()
    handle = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = _cuda(cuda.cuGraphGetNodes(handle))
    if not 0 < count <= MAX_NODES:
        raise ValueError("captured graph exceeds strict node bound")
    nodes, actual_count = _cuda(cuda.cuGraphGetNodes(handle, count))
    if actual_count != count:
        raise ValueError("captured node count changed during inspection")
    _, _, _, edge_count = _cuda(cuda.cuGraphGetEdges(handle))
    if edge_count > MAX_EDGES:
        raise ValueError("captured graph exceeds strict edge bound")
    sources, targets, edge_data, actual_edges = _cuda(cuda.cuGraphGetEdges(handle, edge_count))
    if actual_edges != edge_count:
        raise ValueError("captured edge count changed during inspection")
    for edge in edge_data:
        # Pinned cuda-bindings 13.3.1 exposes these as unsigned char scalars.
        validate_full_completion_edge(int(edge.from_port), int(edge.to_port), int(edge.type), bytes(edge.reserved))
    records = []
    total_bytes = 0
    for node in nodes:
        (kind,) = _cuda(cuda.cuGraphNodeGetType(node))
        if kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY:
            records.append(CapturedNode(int(node), GraphNodeKind.EMPTY))
            continue
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            raise ValueError(f"strict ordered capture rejects executing node {kind}")
        (params,) = _cuda(cuda.cuGraphKernelNodeGetParams(node))
        metadata = []
        for index in range(MAX_PARAMETERS + 1):
            status, offset, size = cuda.cuFuncGetParamInfo(params.func, index)
            if status == cuda.CUresult.CUDA_ERROR_INVALID_VALUE:
                break
            _cuda((status,))
            if index == MAX_PARAMETERS or not 0 < size <= MAX_PARAMETER_BYTES or offset + size > MAX_CAPTURE_BYTES:
                raise ValueError("captured parameter exceeds strict bounds before memory read")
            metadata.append((offset, size))
            total_bytes += size
            if total_bytes > MAX_CAPTURE_BYTES:
                raise ValueError("capture exceeds aggregate byte bound before memory read")
        copied = _parameter_bytes(params)
        if len(copied) != len(metadata):
            raise ValueError("driver parameter metadata changed during capture")
        parameters = tuple(
            CapturedParameter(offset, size, data) for (offset, size), data in zip(metadata, copied, strict=True)
        )
        (name,) = _cuda(cuda.cuFuncGetName(params.func))
        (cluster_attr,) = _cuda(
            cuda.cuGraphKernelNodeGetAttribute(
                node, cuda.CUkernelNodeAttrID(cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
            )
        )
        attributes = []
        for kind, driver_kind in (
            (
                CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            ),
            (
                CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            ),
            (
                CuteKernelFunctionAttributeKind.NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            ),
        ):
            (value,) = _cuda(cuda.cuFuncGetAttribute(driver_kind, params.func))
            if value >= 0:
                attributes.append(CuteKernelFunctionAttribute(kind, int(value)))
        cluster = cluster_attr.clusterDim
        records.append(
            CapturedNode(
                int(node),
                GraphNodeKind.KERNEL,
                CapturedKernel(
                    name.decode(),
                    parameters,
                    (params.gridDimX, params.gridDimY, params.gridDimZ),
                    (params.blockDimX, params.blockDimY, params.blockDimZ),
                    params.sharedMemBytes,
                    (cluster.x, cluster.y, cluster.z),
                    tuple(attributes),
                ),
            )
        )
    captured = CapturedGraph(
        tuple(records), tuple((int(source), int(target)) for source, target in zip(sources, targets, strict=True))
    )
    report = compare_captured_launches(recipe, captured, bindings, evidence)
    if explicit_tensor_slot_bindings(recipe, tensors) != bindings:
        raise ValueError("actual buffer identity changed during capture")
    return report, graph
