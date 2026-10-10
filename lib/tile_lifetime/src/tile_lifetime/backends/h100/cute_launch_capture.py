# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy actual launch arguments while their CUDA graph owns the storage."""

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import torch

from tile_lifetime.cute_pipeline_artifact import CuteKernelFunctionAttribute, CuteKernelFunctionAttributeKind


def checked_cuda(result: tuple[Any, ...]) -> tuple[Any, ...]:
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA graph argument inspection failed: {result[0]}")
    return result[1:]


def kernel_parameter_bytes(parameters: Any) -> tuple[bytes, ...]:
    offsets_and_sizes = []
    for index in range(1024):
        status, offset, size = cuda.cuFuncGetParamInfo(parameters.func, index)
        if status == cuda.CUresult.CUDA_ERROR_INVALID_VALUE:
            break
        checked_cuda((status,))
        offsets_and_sizes.append((offset, size))
    else:
        raise ValueError("captured kernel exceeds the diagnostic parameter bound")
    if parameters.kernelParams:
        addresses = ctypes.cast(int(parameters.kernelParams), ctypes.POINTER(ctypes.c_void_p))
        return tuple(ctypes.string_at(addresses[index], size) for index, (_, size) in enumerate(offsets_and_sizes))
    if not parameters.extra:
        if offsets_and_sizes:
            raise ValueError("captured kernel has parameters but no node-owned argument storage")
        return ()
    extra = ctypes.cast(int(parameters.extra), ctypes.POINTER(ctypes.c_void_p))
    buffer_address = None
    buffer_size = None
    for index in range(0, 8, 2):
        tag = extra[index]
        if tag is None:  # CU_LAUNCH_PARAM_END
            break
        if tag == 1:  # CU_LAUNCH_PARAM_BUFFER_POINTER
            buffer_address = extra[index + 1]
        elif tag == 2:  # CU_LAUNCH_PARAM_BUFFER_SIZE
            buffer_size = ctypes.c_size_t.from_address(extra[index + 1]).value
        else:
            raise ValueError(f"unknown CUDA extra-buffer launch tag {tag}")
    if buffer_address is None or buffer_size is None:
        raise ValueError("captured CUDA extra array omits its parameter buffer")
    if any(offset + size > buffer_size for offset, size in offsets_and_sizes):
        raise ValueError("captured CUDA parameter exceeds its node-owned buffer")
    return tuple(ctypes.string_at(buffer_address + offset, size) for offset, size in offsets_and_sizes)


@dataclass(frozen=True)
class CapturedCuteLaunch:
    symbol: str
    parameters: tuple[bytes, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory: int
    function_attributes: tuple[CuteKernelFunctionAttribute, ...]
    cluster_dimensions: tuple[int, int, int]


def capture_single_cute_launch(call: Callable[[], None]) -> CapturedCuteLaunch:
    """Capture exactly one kernel, rejecting auxiliary execution nodes."""
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        call()
    handle = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = checked_cuda(cuda.cuGraphGetNodes(handle))
    nodes, _ = checked_cuda(cuda.cuGraphGetNodes(handle, count))
    kernels = []
    for node in nodes:
        (kind,) = checked_cuda(cuda.cuGraphNodeGetType(node))
        if kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY:
            continue
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            raise ValueError(f"SM90 export captured an unsupported graph node: {kind}")
        (parameters,) = checked_cuda(cuda.cuGraphKernelNodeGetParams(node))
        (cluster_attribute,) = checked_cuda(
            cuda.cuGraphKernelNodeGetAttribute(
                node, cuda.CUkernelNodeAttrID(cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
            )
        )
        cluster = cluster_attribute.clusterDim
        if (cluster.x, cluster.y, cluster.z) not in {(0, 0, 0), (1, 1, 1)}:
            raise ValueError("SM90 export cannot discard nontrivial captured cluster dimensions")
        (name,) = checked_cuda(cuda.cuFuncGetName(parameters.func))
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
            (value,) = checked_cuda(cuda.cuFuncGetAttribute(driver_kind, parameters.func))
            if value >= 0:
                attributes.append(CuteKernelFunctionAttribute(kind, int(value)))
        kernels.append(
            CapturedCuteLaunch(
                name.decode(),
                kernel_parameter_bytes(parameters),
                (parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ),
                (parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ),
                parameters.sharedMemBytes,
                tuple(attributes),
                (cluster.x, cluster.y, cluster.z),
            )
        )
    if len(kernels) != 1:
        raise ValueError(f"SM90 export requires one captured kernel, found {len(kernels)}")
    return kernels[0]
