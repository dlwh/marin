# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AOT-package a canonical SIMT launch recipe for CUDA SM100."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tile_lifetime.backends.sm100.canonical_simt_emitter import (
    CompleteSm100SimtPipeline,
    compose_sm100_canonical_simt_pipeline,
)
from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComposition,
    composed_cute_compile_request,
    lower_canonical_kernel_composition,
)
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttributeKind,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
)
from tile_lifetime.sm100_canonical_kernel_composition import SM100CanonicalKernelComposition

_ARCHITECTURE = re.compile(r"sm_100[a-z]?")


@dataclass(frozen=True)
class Sm100CanonicalAotCompilePlan:
    """Self-contained CUDA compilation inputs for one canonical recipe."""

    pipeline: CompleteSm100SimtPipeline
    source: str
    source_sha256: str
    source_path: Path
    host_object_path: Path
    device_image_path: Path
    host_compile_argv: tuple[str, ...]
    device_compile_argv: tuple[str, ...]
    request: CutePipelineCompileRequest
    host_entry: str
    kernel_names: tuple[str, ...]
    architecture: str
    cuda_include_directories: tuple[Path, ...]

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "request_digest": self.request.digest,
            "recipe_digest": self.pipeline.recipe.digest,
            "host_entry": self.host_entry,
            "kernel_names": self.kernel_names,
            "architecture": self.architecture,
            "cuda_include_directories": tuple(map(str, self.cuda_include_directories)),
            "runtime_parametric_event_erasures": tuple(
                erasure.canonical_record for erasure in self.pipeline.runtime_event_erasures
            ),
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def sm100_canonical_aot_compile_plan(
    composition: SM100CanonicalKernelComposition | CanonicalKernelComposition,
    *,
    output_directory: Path,
    nvcc: Path,
    architecture: str = "sm_100a",
    function_prefix: str = "shuttle_sm100_canonical_pipeline",
    cuda_include_directories: tuple[Path, ...] = (),
) -> Sm100CanonicalAotCompilePlan:
    """Create a workload-independent AOT plan from scheduled tasks and Events."""
    if not function_prefix.isidentifier():
        raise ValueError(f"SM100 AOT function prefix must be an identifier: {function_prefix!r}")
    if _ARCHITECTURE.fullmatch(architecture) is None:
        raise ValueError(f"SM100 AOT requires an SM100-family architecture, got {architecture!r}")
    if any(not path.is_dir() for path in cuda_include_directories):
        raise ValueError("SM100 AOT CUDA include directories must exist")
    pipeline = compose_sm100_canonical_simt_pipeline(composition)
    canonical = (
        composition
        if isinstance(composition, CanonicalKernelComposition)
        else lower_canonical_kernel_composition(composition.scheduled_request)
    )
    request = composed_cute_compile_request(canonical, pipeline.recipe)
    request = replace(
        request,
        schedule={
            **request.schedule,
            "runtime_parametric_event_erasures": tuple(
                erasure.canonical_record for erasure in pipeline.runtime_event_erasures
            ),
        },
    )
    host_entry = f"{function_prefix}_{pipeline.source_sha256[:16]}"
    source = pipeline.source + "\n" + _host_wrapper(pipeline, host_entry)
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    source_path = output_directory / f"{host_entry}.cu"
    host_object_path = output_directory / f"{host_entry}.o"
    device_image_path = output_directory / f"{host_entry}.fatbin"
    common = (
        str(nvcc),
        "--std=c++17",
        f"--gpu-architecture={architecture}",
        *(argument for path in cuda_include_directories for argument in ("-I", str(path))),
    )
    return Sm100CanonicalAotCompilePlan(
        pipeline=pipeline,
        source=source,
        source_sha256=source_sha256,
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
        request=request,
        host_entry=host_entry,
        kernel_names=tuple(launch.kernel_symbol for launch in pipeline.recipe.launches),
        architecture=architecture,
        cuda_include_directories=cuda_include_directories,
    )


def compile_sm100_canonical_aot_artifact(plan: Sm100CanonicalAotCompilePlan) -> CutePipelineArtifact:
    """Compile and seal one canonical SM100 source/recipe artifact."""
    plan.source_path.parent.mkdir(parents=True, exist_ok=True)
    plan.source_path.write_text(plan.source)
    subprocess.run(plan.host_compile_argv, check=True)
    subprocess.run(plan.device_compile_argv, check=True)
    version = subprocess.run(
        (plan.host_compile_argv[0], "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return seal_sm100_canonical_aot_artifact(
        plan,
        host_object=plan.host_object_path.read_bytes(),
        device_image=plan.device_image_path.read_bytes(),
        nvcc_version=version,
    )


def seal_sm100_canonical_aot_artifact(
    plan: Sm100CanonicalAotCompilePlan,
    *,
    host_object: bytes,
    device_image: bytes,
    nvcc_version: str,
) -> CutePipelineArtifact:
    """Seal already compiled SM100 payloads against their canonical identities."""
    if not host_object or not device_image:
        raise ValueError("SM100 AOT compilation must produce a host object and CUDA fatbin")
    return CutePipelineArtifact(
        request=plan.request,
        host_object=host_object,
        device_image=device_image,
        kernel_names=plan.kernel_names,
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


def _host_wrapper(pipeline: CompleteSm100SimtPipeline, host_entry: str) -> str:
    recipe = pipeline.recipe
    slot_by_name = {slot.name: slot for slot in recipe.slots}
    argument_index = {name: index for index, name in enumerate(recipe.host_call_arguments) if name is not None}
    launches = []
    for launch in recipe.launches:
        attributes = []
        for attribute in launch.function_attributes:
            try:
                cuda_attribute = {
                    CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES: (
                        "cudaFuncAttributeMaxDynamicSharedMemorySize"
                    ),
                    CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT: (
                        "cudaFuncAttributePreferredSharedMemoryCarveout"
                    ),
                }[attribute.kind]
            except KeyError as error:
                raise ValueError(
                    f"the SM100 SIMT AOT wrapper does not support {attribute.kind.value!r} function attributes"
                ) from error
            attributes.extend(
                (
                    f"  status = cudaFuncSetAttribute({launch.kernel_symbol}, {cuda_attribute}, {attribute.value});",
                    "  if (status != cudaSuccess) return static_cast<int>(status);",
                )
            )
        arguments = []
        for argument in launch.arguments:
            if not isinstance(argument, CutePipelineSlotArgument):
                raise ValueError("the SM100 SIMT AOT wrapper currently supports pointer slot arguments only")
            slot = slot_by_name[argument.slot]
            pointer = f"reinterpret_cast<unsigned char*>(arguments[{argument_index[argument.slot]}])"
            if argument.byte_offset:
                pointer = f"({pointer} + {argument.byte_offset})"
            qualifier = "const " if argument.access is CutePipelineSlotAccess.READ else ""
            arguments.append(f"reinterpret_cast<{qualifier}{_cuda_type(slot.element_type)}*>({pointer})")
        grid = ", ".join(map(str, launch.grid))
        block = ", ".join(map(str, launch.block))
        launches.append(
            "\n".join(
                (
                    *attributes,
                    f"  {launch.kernel_symbol}<<<dim3({grid}), dim3({block}), "
                    f"{launch.dynamic_shared_memory_bytes}, stream>>>({', '.join(arguments)});",
                )
            )
        )
    return "\n".join(
        (
            "#include <cuda_runtime.h>",
            f'extern "C" int {host_entry}(void** arguments, cudaStream_t stream) {{',
            "  cudaError_t status = cudaSuccess;",
            *launches,
            "  return static_cast<int>(cudaGetLastError());",
            "}",
            "",
        )
    )


def _cuda_type(element_type: CutePipelineElementType) -> str:
    try:
        return {
            CutePipelineElementType.PRED: "bool",
            CutePipelineElementType.I32: "int32_t",
            CutePipelineElementType.BF16: "__nv_bfloat16",
            CutePipelineElementType.F32: "float",
            CutePipelineElementType.F64: "double",
        }[element_type]
    except KeyError as error:
        raise ValueError(f"the SM100 SIMT AOT wrapper does not support {element_type.value} slots") from error
