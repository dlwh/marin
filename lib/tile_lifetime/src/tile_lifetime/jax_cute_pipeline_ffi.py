# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose a graph-derived CuTe host artifact through JAX typed FFI."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import jax
import jaxlib

from shuttle.ir import DType
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineArtifact,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotKind,
)


class CutePipelineBindingKind(StrEnum):
    """Storage supplied to one generated CuTe tensor argument."""

    OPERAND = "operand"
    RESULT = "result"
    TEMPORARY = "temporary"


@dataclass(frozen=True)
class CutePipelineTensorBinding:
    """One contiguous tensor view in generated-host argument order."""

    name: str
    dtype: DType
    shape: tuple[int, ...]
    kind: CutePipelineBindingKind
    index: int | None = None
    alignment: int = 16

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not self.shape or any(extent <= 0 for extent in self.shape):
            raise ValueError(f"invalid CuTe pipeline tensor binding: {self!r}")
        if self.dtype not in (DType.BF16, DType.FP32):
            raise ValueError("CuTe pipeline FFI supports BF16 and FP32 tensors")
        if self.kind is CutePipelineBindingKind.TEMPORARY:
            if self.index is not None:
                raise ValueError("temporary CuTe pipeline bindings do not have public indexes")
        elif self.index is None or self.index < 0:
            raise ValueError("public CuTe pipeline bindings require nonnegative indexes")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("CuTe pipeline alignment must be a positive power of two")

    @property
    def strides(self) -> tuple[int, ...]:
        stride = 1
        strides = []
        for extent in reversed(self.shape):
            strides.append(stride)
            stride *= extent
        return tuple(reversed(strides))

    @property
    def required_bytes(self) -> int:
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements * (2 if self.dtype is DType.BF16 else 4)

    @property
    def jax_layout(self) -> tuple[int, ...]:
        """JAX FFI major-to-minor layout for this contiguous native tensor."""
        return tuple(range(len(self.shape)))


@dataclass(frozen=True)
class GeneratedCutePipelineFfi:
    """A fixed typed-FFI wrapper around one sealed native pipeline artifact."""

    target_name: str
    handler_symbol: str
    artifact: CutePipelineArtifact
    tensors: tuple[CutePipelineTensorBinding, ...]
    scalar: float | None
    source: str


@dataclass(frozen=True)
class CompiledCutePipelineFfi:
    """Loaded JAX FFI handler and its reproducible build products."""

    generated: GeneratedCutePipelineFfi
    library: ctypes.CDLL
    source_path: Path
    object_path: Path
    library_path: Path
    compile_argv: tuple[str, ...]


def generate_cute_pipeline_ffi(
    artifact: CutePipelineArtifact,
    *,
    target_name: str,
    tensors: tuple[CutePipelineTensorBinding, ...],
    scalar: float | None,
) -> GeneratedCutePipelineFfi:
    """Generate a typed native wrapper without interpreting workload semantics."""
    if not target_name or not target_name.replace(".", "_").isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    if not tensors:
        raise ValueError("CuTe pipeline FFI requires tensor arguments")
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise ValueError("CuTe pipeline tensor names must be unique")
    _validate_public_indexes(tensors, CutePipelineBindingKind.OPERAND)
    _validate_public_indexes(tensors, CutePipelineBindingKind.RESULT)
    host_call_arguments = () if artifact.launch_recipe is None else artifact.launch_recipe.host_call_arguments
    if host_call_arguments and scalar is not None:
        raise ValueError("an explicit CuTe host-call ABI cannot be combined with a legacy scalar argument")
    handler_symbol = target_name.replace(".", "_")
    source = _handler_source(
        handler_symbol=handler_symbol,
        host_entry=artifact.host_entry,
        tensors=tensors,
        scalar=scalar,
        host_call_arguments=host_call_arguments,
    )
    return GeneratedCutePipelineFfi(target_name, handler_symbol, artifact, tensors, scalar, source)


def cute_pipeline_bindings_from_artifact(
    artifact: CutePipelineArtifact,
) -> tuple[CutePipelineTensorBinding, ...]:
    """Derive the typed JAX boundary from indexed native launch-recipe slots."""
    recipe = artifact.launch_recipe
    if recipe is None:
        raise ValueError("CuTe JAX FFI requires an artifact launch recipe")
    if any(slot.kind is CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT for slot in recipe.slots):
        raise ValueError("CuTe JAX FFI does not yet support aliased public input/output slots")

    def binding(
        slot: CutePipelineSlot,
        kind: CutePipelineBindingKind,
        index: int | None,
    ) -> CutePipelineTensorBinding:
        dtype = {
            CutePipelineElementType.BF16: DType.BF16,
            CutePipelineElementType.F32: DType.FP32,
        }.get(slot.element_type)
        if dtype is None:
            raise ValueError(f"CuTe JAX FFI does not support slot element type {slot.element_type.value!r}")
        return CutePipelineTensorBinding(slot.name, dtype, slot.shape, kind, index, slot.alignment)

    inputs = tuple(
        binding(slot, CutePipelineBindingKind.OPERAND, slot.input_index)
        for slot in sorted(
            (slot for slot in recipe.slots if slot.kind is CutePipelineSlotKind.PUBLIC_INPUT),
            key=lambda slot: slot.input_index if slot.input_index is not None else -1,
        )
    )
    outputs = tuple(
        binding(slot, CutePipelineBindingKind.RESULT, slot.output_index)
        for slot in sorted(
            (slot for slot in recipe.slots if slot.kind is CutePipelineSlotKind.PUBLIC_OUTPUT),
            key=lambda slot: slot.output_index if slot.output_index is not None else -1,
        )
    )
    scratch = tuple(
        binding(slot, CutePipelineBindingKind.TEMPORARY, None)
        for slot in recipe.slots
        if slot.kind is CutePipelineSlotKind.SCRATCH
    )
    return (*inputs, *outputs, *scratch)


def compile_cute_pipeline_ffi(
    generated: GeneratedCutePipelineFfi,
    *,
    directory: Path,
    compiler: Path,
    cuda_include_directory: Path,
    cuda_runtime_library: Path,
    tvm_ffi_root: Path,
    cute_runtime_library: Path,
) -> CompiledCutePipelineFfi:
    """Link the generated host object into a JAX typed-FFI shared library."""
    if not compiler.is_file():
        raise ValueError(f"host C++ compiler does not exist: {compiler}")
    if not (cuda_include_directory / "cuda_runtime_api.h").is_file() or not cuda_runtime_library.is_file():
        raise ValueError("CUDA host runtime headers or library do not exist")
    tvm_include = tvm_ffi_root / "include"
    tvm_library = tvm_ffi_root / "lib" / "libtvm_ffi.so"
    if not (tvm_include / "tvm/ffi/c_api.h").is_file() or not tvm_library.is_file():
        raise ValueError(f"TVM FFI root is incomplete: {tvm_ffi_root}")
    if not cute_runtime_library.is_file():
        raise ValueError(f"CuTe runtime library does not exist: {cute_runtime_library}")
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
        str(tvm_include),
        "-I",
        str(cuda_include_directory),
        str(source_path),
        str(object_path),
        str(tvm_library),
        str(cute_runtime_library),
        str(cuda_runtime_library),
        "-o",
        str(library_path),
        f"-Wl,-rpath,{tvm_library.parent}",
        f"-Wl,-rpath,{cute_runtime_library.parent}",
        f"-Wl,-rpath,{cuda_runtime_library.parent}",
    )
    subprocess.run(command, check=True)
    return CompiledCutePipelineFfi(
        generated,
        ctypes.CDLL(str(library_path)),
        source_path,
        object_path,
        library_path,
        command,
    )


def register_cute_pipeline_ffi(compiled: CompiledCutePipelineFfi) -> None:
    """Register one compiled graph-derived pipeline with JAX's CUDA backend."""
    handler = getattr(compiled.library, compiled.generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        compiled.generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cute_pipeline_ffi(
    generated: GeneratedCutePipelineFfi,
    *operands: jax.Array,
) -> tuple[jax.Array, ...]:
    """Call a registered pipeline using its declared public tensor boundary."""
    input_bindings = _ordered_bindings(generated.tensors, CutePipelineBindingKind.OPERAND)
    output_bindings = _ordered_bindings(generated.tensors, CutePipelineBindingKind.RESULT)
    if len(operands) != len(input_bindings):
        raise ValueError(f"CuTe pipeline requires {len(input_bindings)} operands, found {len(operands)}")
    for binding, operand in zip(input_bindings, operands, strict=True):
        if operand.shape != binding.shape or operand.dtype != _jax_dtype(binding.dtype):
            raise ValueError(f"operand {binding.name!r} does not match its CuTe pipeline binding")
    output_shapes = tuple(jax.ShapeDtypeStruct(binding.shape, _jax_dtype(binding.dtype)) for binding in output_bindings)
    result = jax.ffi.ffi_call(
        generated.target_name,
        output_shapes,
        vmap_method="broadcast_all",
        input_layouts=tuple(binding.jax_layout for binding in input_bindings),
        output_layouts=tuple(binding.jax_layout for binding in output_bindings),
    )(*operands)
    return cast(tuple[jax.Array, ...], tuple(result))


def _handler_source(
    *,
    handler_symbol: str,
    host_entry: str,
    tensors: tuple[CutePipelineTensorBinding, ...],
    scalar: float | None,
    host_call_arguments: tuple[str | None, ...],
) -> str:
    inputs = _ordered_bindings(tensors, CutePipelineBindingKind.OPERAND)
    outputs = _ordered_bindings(tensors, CutePipelineBindingKind.RESULT)
    arguments = [
        *(f"ffi::Buffer<{_ffi_dtype(tensor.dtype)}, {len(tensor.shape)}> {tensor.name}_buffer" for tensor in inputs),
        *(
            f"ffi::Result<ffi::Buffer<{_ffi_dtype(tensor.dtype)}, {len(tensor.shape)}>> {tensor.name}_result"
            for tensor in outputs
        ),
    ]
    declarations = []
    pointers: dict[str, str] = {}
    for tensor in inputs:
        pointer = f"{tensor.name}_pointer"
        declarations.append(f"  void* {pointer} = {tensor.name}_buffer.typed_data();")
        pointers[tensor.name] = pointer
    for tensor in outputs:
        pointer = f"{tensor.name}_pointer"
        declarations.append(f"  void* {pointer} = {tensor.name}_result->typed_data();")
        pointers[tensor.name] = pointer
    for tensor in tensors:
        if tensor.kind is not CutePipelineBindingKind.TEMPORARY:
            continue
        storage = f"{tensor.name}_storage"
        declarations.extend(
            (
                f"  auto {storage} = scratch.Allocate({tensor.required_bytes}, {tensor.alignment});",
                f'  if (!{storage}) return ffi::Error::Internal("failed to allocate CuTe pipeline temporary");',
                f"  void* {tensor.name}_pointer = *{storage};",
            )
        )
        pointers[tensor.name] = f"{tensor.name}_pointer"
    tensor_views = []
    for tensor in tensors:
        shape = ", ".join(str(extent) for extent in tensor.shape)
        strides = ", ".join(str(stride) for stride in tensor.strides)
        tensor_views.append(
            f"""  int64_t {tensor.name}_shape[] = {{{shape}}};
  int64_t {tensor.name}_strides[] = {{{strides}}};
  DLTensor {tensor.name}_tensor{{
      {pointers[tensor.name]}, DLDevice{{kDLCUDA, device_ordinal}}, {len(tensor.shape)},
      DLDataType{{{_dl_dtype(tensor.dtype)}, {_dtype_bits(tensor.dtype)}, 1}},
      {tensor.name}_shape, {tensor.name}_strides, 0}};"""
        )
    tensor_by_name = {tensor.name: tensor for tensor in tensors}
    if host_call_arguments:
        unknown = set(name for name in host_call_arguments if name is not None) - set(tensor_by_name)
        if unknown:
            raise ValueError(f"CuTe host-call ABI references unbound tensors: {sorted(unknown)}")
        any_values = tuple(_host_argument_source(ordinal, name) for ordinal, name in enumerate(host_call_arguments))
        argument_count = len(host_call_arguments)
        scalar_source = ""
    else:
        any_values = tuple(_host_argument_source(ordinal, tensor.name) for ordinal, tensor in enumerate(tensors))
        argument_count = len(tensors) + int(scalar is not None)
        scalar_source = _scalar_argument_source(len(tensors), scalar)
    bindings = "\n".join(
        [
            *(f"      .Arg<ffi::Buffer<{_ffi_dtype(t.dtype)}, {len(t.shape)}>>()" for t in inputs),
            *(f"      .Ret<ffi::Buffer<{_ffi_dtype(t.dtype)}, {len(t.shape)}>>()" for t in outputs),
        ]
    )
    return f"""// Generated physical-artifact bridge; contains no workload dispatch.
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" int {host_entry}(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

namespace {{
ffi::Error Execute(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {",\n    ".join(arguments)}) {{
  int device_ordinal = -1;
  cudaError_t cuda_status = cudaGetDevice(&device_ordinal);
  if (cuda_status != cudaSuccess) return ffi::Error::Internal("cudaGetDevice failed");
{chr(10).join(declarations)}
{chr(10).join(tensor_views)}
  TVMFFIAny arguments[{argument_count}]{{}};
{chr(10).join(any_values)}
{scalar_source}
  TVMFFIAny result{{}};
  result.type_index = kTVMFFINone;
  TVMFFIStreamHandle old_stream = nullptr;
  if (TVMFFIEnvSetStream(kDLCUDA, device_ordinal, reinterpret_cast<void*>(stream), &old_stream) != 0)
    return ffi::Error::Internal("failed to set TVM FFI CUDA stream");
  int status = {host_entry}(nullptr, arguments, {argument_count}, &result);
  int restore_status = TVMFFIEnvSetStream(kDLCUDA, device_ordinal, old_stream, nullptr);
  if (status != 0) {{
    TVMFFIObjectHandle error = nullptr;
    TVMFFIErrorMoveFromRaised(&error);
    std::string message = "generated CuTe pipeline failed";
    if (error != nullptr) {{
      TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(error);
      message.append(": ");
      message.append(cell->message.data, cell->message.size);
      TVMFFIObjectDecRef(error);
    }}
    return ffi::Error::Internal(message);
  }}
  if (restore_status != 0) return ffi::Error::Internal("failed to restore TVM FFI CUDA stream");
  if (result.type_index != kTVMFFINone) return ffi::Error::Internal("CuTe pipeline returned an unexpected value");
  return ffi::Error::Success();
}}

auto Binding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{bindings};
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL({handler_symbol}, Execute, Binding());
"""


def _scalar_argument_source(ordinal: int, scalar: float | None) -> str:
    if scalar is None:
        return ""
    return f"""  arguments[{ordinal}].type_index = kTVMFFIFloat;
  arguments[{ordinal}].zero_padding = 0;
  arguments[{ordinal}].v_float64 = {scalar:.17g};"""


def _host_argument_source(ordinal: int, tensor_name: str | None) -> str:
    if tensor_name is None:
        return f"  arguments[{ordinal}].type_index = kTVMFFINone;"
    return f"""  arguments[{ordinal}].type_index = kTVMFFIDLTensorPtr;
  arguments[{ordinal}].zero_padding = 0;
  arguments[{ordinal}].v_ptr = &{tensor_name}_tensor;"""


def _validate_public_indexes(
    tensors: tuple[CutePipelineTensorBinding, ...],
    kind: CutePipelineBindingKind,
) -> None:
    indexes = sorted(tensor.index for tensor in tensors if tensor.kind is kind and tensor.index is not None)
    if indexes != list(range(len(indexes))):
        raise ValueError(f"CuTe pipeline {kind.value} indexes must be contiguous")


def _ordered_bindings(
    tensors: tuple[CutePipelineTensorBinding, ...],
    kind: CutePipelineBindingKind,
) -> tuple[CutePipelineTensorBinding, ...]:
    return tuple(sorted((tensor for tensor in tensors if tensor.kind is kind), key=lambda tensor: tensor.index or 0))


def _ffi_dtype(dtype: DType) -> str:
    return "ffi::DataType::BF16" if dtype is DType.BF16 else "ffi::DataType::F32"


def _dl_dtype(dtype: DType) -> str:
    return "kDLBfloat" if dtype is DType.BF16 else "kDLFloat"


def _dtype_bits(dtype: DType) -> int:
    return 16 if dtype is DType.BF16 else 32


def _jax_dtype(dtype: DType):
    return jax.numpy.bfloat16 if dtype is DType.BF16 else jax.numpy.float32
