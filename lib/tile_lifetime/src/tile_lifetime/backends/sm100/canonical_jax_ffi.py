# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose a canonical SM100 launch recipe through JAX typed FFI."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jaxlib

from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineArtifact,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotKind,
)


@dataclass(frozen=True)
class GeneratedSm100CanonicalJaxFfi:
    """One typed JAX handler over the canonical direct launch ABI."""

    target_name: str
    handler_symbol: str
    artifact: CutePipelineArtifact
    source: str


@dataclass(frozen=True)
class CompiledSm100CanonicalJaxFfi:
    """Loaded SM100 JAX handler and its reproducible build products."""

    generated: GeneratedSm100CanonicalJaxFfi
    library: ctypes.CDLL
    source_path: Path
    object_path: Path
    library_path: Path
    compile_argv: tuple[str, ...]


def generate_sm100_canonical_jax_ffi(
    artifact: CutePipelineArtifact,
    *,
    target_name: str,
) -> GeneratedSm100CanonicalJaxFfi:
    """Generate a typed handler from public and scratch slots in the recipe."""
    if not target_name or not target_name.replace(".", "_").isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    recipe = artifact.launch_recipe
    if recipe is None:
        raise ValueError("canonical SM100 JAX FFI requires an artifact launch recipe")
    if not recipe.host_call_arguments:
        raise ValueError("canonical SM100 JAX FFI requires an explicit host-call ABI")
    if any(slot.kind is CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT for slot in recipe.slots):
        raise ValueError("canonical SM100 JAX FFI does not support aliased input/output slots")
    _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_INPUT)
    _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_OUTPUT)
    handler_symbol = target_name.replace(".", "_")
    source = _handler_source(artifact, handler_symbol)
    return GeneratedSm100CanonicalJaxFfi(target_name, handler_symbol, artifact, source)


def compile_sm100_canonical_jax_ffi(
    generated: GeneratedSm100CanonicalJaxFfi,
    *,
    directory: Path,
    compiler: Path,
    cuda_include_directory: Path,
    cuda_runtime_library: Path,
) -> CompiledSm100CanonicalJaxFfi:
    """Link the generated handler and AOT CUDA host object into a shared library."""
    if not compiler.is_file():
        raise ValueError(f"host C++ compiler does not exist: {compiler}")
    if not (cuda_include_directory / "cuda_runtime_api.h").is_file() or not cuda_runtime_library.is_file():
        raise ValueError("CUDA host runtime headers or library do not exist")
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"{generated.handler_symbol}.cc"
    object_path = directory / f"{generated.handler_symbol}.o"
    library_path = directory / f"{generated.handler_symbol}.so"
    source_path.write_text(generated.source + "\n")
    object_path.write_bytes(generated.artifact.host_object)
    jax_include = Path(jaxlib.__file__).resolve().parent / "include"
    command = (
        str(compiler),
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        "-I",
        str(jax_include),
        "-I",
        str(cuda_include_directory),
        str(source_path),
        str(object_path),
        str(cuda_runtime_library),
        "-o",
        str(library_path),
        f"-Wl,-rpath,{cuda_runtime_library.parent}",
    )
    subprocess.run(command, check=True)
    return CompiledSm100CanonicalJaxFfi(
        generated,
        ctypes.CDLL(str(library_path)),
        source_path,
        object_path,
        library_path,
        command,
    )


def register_sm100_canonical_jax_ffi(compiled: CompiledSm100CanonicalJaxFfi) -> None:
    """Register a compiled canonical handler with JAX's CUDA backend."""
    handler = getattr(compiled.library, compiled.generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        compiled.generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_sm100_canonical_jax_ffi(
    generated: GeneratedSm100CanonicalJaxFfi,
    *operands: jax.Array,
) -> tuple[jax.Array, ...]:
    """Invoke a registered canonical handler using its public recipe boundary."""
    recipe = generated.artifact.launch_recipe
    assert recipe is not None
    inputs = _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_INPUT)
    outputs = _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_OUTPUT)
    if len(operands) != len(inputs):
        raise ValueError(f"canonical SM100 pipeline requires {len(inputs)} operands, found {len(operands)}")
    for slot, operand in zip(inputs, operands, strict=True):
        if operand.shape != slot.shape or operand.dtype != _jax_dtype(slot.element_type):
            raise ValueError(f"operand {slot.name!r} does not match its canonical pipeline slot")
    result = jax.ffi.ffi_call(
        generated.target_name,
        tuple(jax.ShapeDtypeStruct(slot.shape, _jax_dtype(slot.element_type)) for slot in outputs),
        vmap_method="broadcast_all",
        input_layouts=tuple(tuple(range(len(slot.shape))) for slot in inputs),
        output_layouts=tuple(tuple(range(len(slot.shape))) for slot in outputs),
    )(*operands)
    return tuple(result)


def _handler_source(artifact: CutePipelineArtifact, handler_symbol: str) -> str:
    recipe = artifact.launch_recipe
    assert recipe is not None
    inputs = _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_INPUT)
    outputs = _ordered_public_slots(recipe.slots, CutePipelineSlotKind.PUBLIC_OUTPUT)
    scratch = tuple(slot for slot in recipe.slots if slot.kind is CutePipelineSlotKind.SCRATCH)
    arguments = [
        *(f"ffi::Buffer<{_ffi_dtype(slot.element_type)}, {len(slot.shape)}> {slot.name}_buffer" for slot in inputs),
        *(
            f"ffi::Result<ffi::Buffer<{_ffi_dtype(slot.element_type)}, {len(slot.shape)}>> {slot.name}_result"
            for slot in outputs
        ),
    ]
    declarations = [
        *(f"  void* {slot.name}_pointer = {slot.name}_buffer.typed_data();" for slot in inputs),
        *(f"  void* {slot.name}_pointer = {slot.name}_result->typed_data();" for slot in outputs),
    ]
    for slot in scratch:
        declarations.extend(
            (
                f"  auto {slot.name}_storage = scratch.Allocate({_required_bytes(slot)}, {slot.alignment});",
                f'  if (!{slot.name}_storage) return ffi::Error::Internal("failed to allocate canonical temporary");',
                f"  void* {slot.name}_pointer = *{slot.name}_storage;",
            )
        )
    pointer_by_slot = {slot.name: f"{slot.name}_pointer" for slot in recipe.slots}
    pointer_arguments = tuple(
        pointer_by_slot[name] if name is not None else "nullptr" for name in recipe.host_call_arguments
    )
    bindings = [
        *(f"      .Arg<ffi::Buffer<{_ffi_dtype(slot.element_type)}, {len(slot.shape)}>>()" for slot in inputs),
        *(f"      .Ret<ffi::Buffer<{_ffi_dtype(slot.element_type)}, {len(slot.shape)}>>()" for slot in outputs),
    ]
    return f"""// Generated canonical launch-recipe bridge; contains no workload dispatch.
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" int {artifact.host_entry}(void** arguments, cudaStream_t stream);

namespace {{
ffi::Error Execute(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {",\n    ".join(arguments)}) {{
{chr(10).join(declarations)}
  void* arguments[] = {{{", ".join(pointer_arguments)}}};
  const int status = {artifact.host_entry}(arguments, stream);
  if (status != 0) return ffi::Error::Internal("canonical SM100 launch recipe failed");
  return ffi::Error::Success();
}}

auto Binding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{chr(10).join(bindings)};
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL({handler_symbol}, Execute, Binding());
"""


def _ordered_public_slots(
    slots: tuple[CutePipelineSlot, ...],
    kind: CutePipelineSlotKind,
) -> tuple[CutePipelineSlot, ...]:
    index_name = "input_index" if kind is CutePipelineSlotKind.PUBLIC_INPUT else "output_index"
    selected = tuple(slot for slot in slots if slot.kind is kind)
    indexes = tuple(sorted(getattr(slot, index_name) for slot in selected))
    if indexes != tuple(range(len(selected))):
        raise ValueError(f"canonical SM100 {kind.value} indexes must be contiguous")
    return tuple(sorted(selected, key=lambda slot: getattr(slot, index_name)))


def _required_bytes(slot: CutePipelineSlot) -> int:
    elements = 1
    for extent in slot.shape:
        elements *= extent
    return (
        elements
        * {
            CutePipelineElementType.PRED: 1,
            CutePipelineElementType.I32: 4,
            CutePipelineElementType.BF16: 2,
            CutePipelineElementType.F32: 4,
            CutePipelineElementType.F64: 8,
        }[slot.element_type]
    )


def _ffi_dtype(element_type: CutePipelineElementType) -> str:
    try:
        return {
            CutePipelineElementType.BF16: "ffi::DataType::BF16",
            CutePipelineElementType.F32: "ffi::DataType::F32",
            CutePipelineElementType.F64: "ffi::DataType::F64",
            CutePipelineElementType.I32: "ffi::DataType::S32",
            CutePipelineElementType.PRED: "ffi::DataType::PRED",
        }[element_type]
    except KeyError as error:
        raise ValueError(f"canonical SM100 JAX FFI does not support {element_type.value} public tensors") from error


def _jax_dtype(element_type: CutePipelineElementType):
    try:
        return {
            CutePipelineElementType.BF16: jax.numpy.bfloat16,
            CutePipelineElementType.F32: jax.numpy.float32,
            CutePipelineElementType.F64: jax.numpy.float64,
            CutePipelineElementType.I32: jax.numpy.int32,
            CutePipelineElementType.PRED: jax.numpy.bool_,
        }[element_type]
    except KeyError as error:
        raise ValueError(f"canonical SM100 JAX FFI does not support {element_type.value} public tensors") from error
