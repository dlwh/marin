# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check exported launch parameters against the CUDA driver's compiled ABI."""

from collections.abc import Iterator
from contextlib import contextmanager

import cuda.bindings.driver as cuda

from tile_lifetime.cute_pipeline_artifact import CuteMultiImagePipelineArtifact, cute_kernel_argument_size


@contextmanager
def _loaded_module(image: bytes) -> Iterator[cuda.CUmodule]:
    status, module = cuda.cuModuleLoadData(image)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA module load failed during launch ABI validation: {status}")
    try:
        yield module
    finally:
        (status,) = cuda.cuModuleUnload(module)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA module unload failed during launch ABI validation: {status}")


def validate_cute_launch_abi(artifact: CuteMultiImagePipelineArtifact) -> None:
    """Reject parameter-count or byte-width mismatches before native execution.

    The caller must have a current CUDA context on the target device. Images
    load independently so identical symbols in different images remain distinct.
    """
    for image_index, image in enumerate(artifact.images):
        with _loaded_module(image.device_image) as module:
            for launch, bound_image in zip(artifact.launch_recipe.launches, artifact.image_by_launch, strict=True):
                if bound_image != image_index:
                    continue
                status, function = cuda.cuModuleGetFunction(module, launch.kernel_symbol.encode())
                if status != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"CUDA function lookup failed for {launch.kernel_symbol!r}: {status}")
                expected = tuple(cute_kernel_argument_size(argument) for argument in launch.arguments)
                for index in range(len(expected) + 1):
                    status, _, size = cuda.cuFuncGetParamInfo(function, index)
                    if status == cuda.CUresult.CUDA_ERROR_INVALID_VALUE:
                        if index == len(expected):
                            break
                        raise ValueError(
                            f"CuTe launch {launch.name!r} image {image_index} declares {len(expected)} "
                            f"parameters, but CUDA kernel {launch.kernel_symbol!r} has {index}"
                        )
                    if status != cuda.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f"CUDA parameter query failed for {launch.kernel_symbol!r}: {status}")
                    if index == len(expected):
                        raise ValueError(
                            f"CuTe launch {launch.name!r} image {image_index} declares {len(expected)} "
                            f"parameters, but CUDA kernel {launch.kernel_symbol!r} has more"
                        )
                    if size != expected[index]:
                        raise ValueError(
                            f"CuTe launch {launch.name!r} image {image_index} parameter {index} "
                            f"({launch.arguments[index].name!r}) declares {expected[index]} bytes, "
                            f"but CUDA kernel {launch.kernel_symbol!r} requires {size}"
                        )
