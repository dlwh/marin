# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instantiate the SM90 normalized-exponential pullback physical template."""

from __future__ import annotations

import hashlib
import inspect
import math
import operator
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda import bindings as cuda_bindings
from cuda.bindings import driver as cuda
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess

from shuttle.ir import DType
from tile_lifetime.cute_device_ir import CuteDeviceIrSnapshot, compile_cute_with_structured_snapshot
from tile_lifetime.cute_fast_divmod import encode_cute_fast_divmod_divisor
from tile_lifetime.cute_pipeline_artifact import (
    CuteEventExecutionPoint,
    CuteEventRealizationKind,
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelinePackedBytesArgument,
    CutePipelineScalarArgument,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredImmediateField,
    CutePipelineStructuredSlotAddressField,
    CuteSourceTaskRole,
    CuteStructuralEventCover,
    elf64_symbol_bytes,
)
from tile_lifetime.cute_sm90_tensor_abi import (
    packed_log_sum_exp_argument,
    sm90_packed_head_tensor_map_argument,
    sm90_tensor_map_argument,
)
from tile_lifetime.h100_normalized_exp_backward_lowering import (
    H100NormalizedExpBackwardLowering,
    lower_h100_normalized_exp_backward,
)
from tile_lifetime.h100_scalar_ast import score_input_roles
from tile_lifetime.h100_streaming_lowering import LoweredScoreMap, Sm90BackwardAccumulatorSchedule
from tile_lifetime.jax_cute_pipeline_ffi import (
    CutePipelineBindingKind,
    CutePipelineTensorBinding,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractPullbackProgram,
    is_identity_post_normalization_pullback,
)
from tile_lifetime.sm90_backward_accumulation import (
    Sm90BackwardAccumulationPlan,
    ordered_backward_kernel_binding,
    validate_ordered_backward_recipe,
    validate_ordered_backward_variant,
)

from .cute_domain_restriction import cute_affine_index_predicate
from .cute_post_normalization_pullback import CutePostNormalizationPullback
from .cute_scalar_ast import (
    cute_keyword_score_map_adapter,
    cute_keyword_score_map_vjp_adapter,
    cute_post_normalization_vjp,
    cute_score_map,
    cute_score_map_vjp,
)
from .cute_streaming_emitter import (
    CompiledH100StreamingProgram,
    H100StreamingTemplate,
    build_h100_streaming_template,
)


def _order_source_slots(
    slots: tuple[CutePipelineSlot, ...],
    source_slot_count: int,
    source_slot_ordinals: Mapping[str, int] | None,
) -> tuple[CutePipelineSlot, ...]:
    if source_slot_ordinals is None:
        return slots
    source_slots = slots[:source_slot_count]
    ordinals = tuple(source_slot_ordinals.values())
    if (
        set(source_slot_ordinals) != {slot.name for slot in source_slots}
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
        or len(set(ordinals)) != len(source_slots)
    ):
        raise ValueError("source slot ordinals must uniquely identify nonnegative source buffers")
    return (
        *sorted(source_slots, key=lambda slot: source_slot_ordinals[slot.name]),
        *slots[source_slot_count:],
    )


def _value_and_grad_kernel_partition(
    kernel_names: tuple[str, ...],
    *,
    has_device_queues: bool,
    generic_finalizer: bool,
) -> tuple[str, str | None, tuple[str, ...]]:
    expected_kernels = (9 if generic_finalizer else 8) + int(has_device_queues)
    if len(kernel_names) != expected_kernels:
        raise ValueError(
            f"the SM90 value-and-pullback pipeline must contain {expected_kernels} kernels, got {len(kernel_names)}"
        )
    forward_kernel_index = int(has_device_queues)
    map_kernel_index = 2 + int(has_device_queues) if generic_finalizer else None
    excluded = {forward_kernel_index}
    if map_kernel_index is not None:
        excluded.add(map_kernel_index)
    reverse_kernel_names = tuple(kernel_name for index, kernel_name in enumerate(kernel_names) if index not in excluded)
    return (
        kernel_names[forward_kernel_index],
        None if map_kernel_index is None else kernel_names[map_kernel_index],
        reverse_kernel_names,
    )


@dataclass(frozen=True)
class H100NormalizedExponentialBackwardTemplate:
    """One structural reverse program bound to the Hopper physical skeleton."""

    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule]
    lowering: H100NormalizedExpBackwardLowering
    kernel: Any

    @property
    def physical_schedule(self) -> Sm90BackwardAccumulatorSchedule:
        return self.lowering.physical_schedule

    @property
    def score_map(self) -> LoweredScoreMap:
        return self.lowering.score_map


@dataclass
class CompiledH100NormalizedExponentialBackward:
    """Compiled preprocess/mainloop/postprocess pipeline and its work buffers."""

    template: H100NormalizedExponentialBackwardTemplate
    executable: Any
    query_cotangent: torch.Tensor
    key_cotangent: torch.Tensor
    value_cotangent: torch.Tensor
    output_dot: torch.Tensor
    log_sum_exp_log2: torch.Tensor
    query_accumulator: torch.Tensor
    key_accumulator: torch.Tensor
    value_accumulator: torch.Tensor
    normalized_output_cotangent: torch.Tensor | None
    normalized_output_cotangent_residual: torch.Tensor | None
    layout_transport_output: torch.Tensor | None
    layout_transport_output_cotangent: torch.Tensor | None
    query_accumulation_semaphore: torch.Tensor | None
    key_accumulation_semaphore: torch.Tensor | None
    value_accumulation_semaphore: torch.Tensor | None
    query_stride: tuple[int, ...]
    key_stride: tuple[int, ...]
    value_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    output_cotangent_stride: tuple[int, ...]
    output_cotangent_dtype: DType
    normalized_output_stride: tuple[int, ...] | None
    query_cotangent_stride: tuple[int, ...]
    key_cotangent_stride: tuple[int, ...]
    value_cotangent_stride: tuple[int, ...]
    streamed_gradients_are_direct: bool
    structured_device_ir: CuteDeviceIrSnapshot

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_cotangent: torch.Tensor,
        log_sum_exp: torch.Tensor,
        normalized_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.template.lowering.accumulation_plan is not None:
            _validate_runtime_tensors(query, key, value, output, output_cotangent, log_sum_exp)
            _validate_ordered_runtime_layout(self.template, query, key, value)
            for name, tensor in (
                ("query", query),
                ("key", key),
                ("value", value),
                ("output", output),
                ("output_cotangent", output_cotangent),
            ):
                if tuple(tensor.stride()) != getattr(self, f"{name}_stride"):
                    raise ValueError(f"ordered runtime {name} strides differ from the compiled layout")
        self.executable(
            query,
            key,
            value,
            output,
            output_cotangent,
            log_sum_exp,
            self.query_cotangent,
            self.key_cotangent,
            self.value_cotangent,
            self.output_dot,
            self.log_sum_exp_log2,
            self.query_accumulator,
            self.key_accumulator,
            self.value_accumulator,
            normalized_output,
            self.normalized_output_cotangent,
            self.normalized_output_cotangent_residual,
            self.layout_transport_output,
            self.layout_transport_output_cotangent,
            self.query_accumulation_semaphore,
            self.key_accumulation_semaphore,
            self.value_accumulation_semaphore,
        )
        return self.query_cotangent, self.key_cotangent, self.value_cotangent


class _AffineOutputLayoutTransport:
    """Materialize one declared affine output layout into feature-minor storage."""

    def __init__(self, shape: tuple[int, int, int, int]):
        self.batch, self.owner, self.relation, self.feature = shape

    @cute.jit
    def __call__(
        self,
        source_output: cute.Tensor,
        source_cotangent: cute.Tensor,
        output: cute.Tensor,
        cotangent: cute.Tensor,
        output_dot: cute.Tensor,
        log_sum_exp: cute.Tensor,
        log_sum_exp_log2: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            source_output,
            source_cotangent,
            output,
            cotangent,
            output_dot,
            log_sum_exp,
            log_sum_exp_log2,
        ).launch(
            grid=[self.owner, self.relation, self.batch],
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source_output: cute.Tensor,
        source_cotangent: cute.Tensor,
        output: cute.Tensor,
        cotangent: cute.Tensor,
        output_dot: cute.Tensor,
        log_sum_exp: cute.Tensor,
        log_sum_exp_log2: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        owner, relation, batch = cute.arch.block_idx()
        dot = cutlass.Float32(0.0)
        for feature_offset in cutlass.range_constexpr(_tile_count(self.feature, 32)):
            feature = lane + feature_offset * 32
            if feature < self.feature:
                primal = source_output[batch, owner, relation, feature]
                gradient = source_cotangent[batch, owner, relation, feature]
                output[batch, owner, relation, feature] = primal
                cotangent[batch, owner, relation, feature] = gradient
                dot += cutlass.Float32(primal) * cutlass.Float32(gradient)
        dot = utils.warp_reduce(dot, operator.add)
        if lane == 0:
            output_dot[batch, relation, owner] = dot
            log_sum_exp_log2[batch, relation, owner] = cutlass.Float32(
                log_sum_exp[batch, relation, owner]
            ) * cutlass.Float32(math.log2(math.e))


@dataclass
class CompiledH100NormalizedExponentialValueAndGrad:
    """One forward-state producer and its pullback in a single device image."""

    forward_template: H100StreamingTemplate
    backward_template: H100NormalizedExponentialBackwardTemplate
    executable: Any
    output: torch.Tensor
    query_cotangent: torch.Tensor
    key_cotangent: torch.Tensor
    value_cotangent: torch.Tensor
    log_sum_exp: torch.Tensor
    output_dot: torch.Tensor
    log_sum_exp_log2: torch.Tensor
    query_accumulator: torch.Tensor
    key_accumulator: torch.Tensor
    value_accumulator: torch.Tensor
    query_accumulation_semaphore: torch.Tensor | None
    key_accumulation_semaphore: torch.Tensor | None
    value_accumulation_semaphore: torch.Tensor | None
    normalized_output: torch.Tensor | None
    normalized_output_cotangent: torch.Tensor | None
    normalized_output_cotangent_residual: torch.Tensor | None
    query_stride: tuple[int, ...]
    key_stride: tuple[int, ...]
    value_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    output_cotangent_stride: tuple[int, ...]
    normalized_output_stride: tuple[int, ...] | None
    normalized_output_cotangent_stride: tuple[int, ...] | None
    normalized_output_cotangent_residual_stride: tuple[int, ...] | None
    query_cotangent_stride: tuple[int, ...]
    key_cotangent_stride: tuple[int, ...]
    value_cotangent_stride: tuple[int, ...]
    structured_device_ir: CuteDeviceIrSnapshot

    @property
    def template(self) -> H100NormalizedExponentialBackwardTemplate:
        """The pullback template consumed by the shared reverse recipe builder."""
        return self.backward_template

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_cotangent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.run_into(
            query,
            key,
            value,
            output_cotangent,
            self.output,
            self.query_cotangent,
            self.key_cotangent,
            self.value_cotangent,
        )
        return self.output, self.query_cotangent, self.key_cotangent, self.value_cotangent

    def run_into(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_cotangent: torch.Tensor,
        output: torch.Tensor,
        query_cotangent: torch.Tensor,
        key_cotangent: torch.Tensor,
        value_cotangent: torch.Tensor,
    ) -> None:
        """Execute into canonical component-owned result storage."""
        self.executable(
            query,
            key,
            value,
            output_cotangent,
            output,
            query_cotangent,
            key_cotangent,
            value_cotangent,
            self.log_sum_exp,
            self.output_dot,
            self.log_sum_exp_log2,
            self.query_accumulator,
            self.key_accumulator,
            self.value_accumulator,
            self.normalized_output,
            self.normalized_output_cotangent,
            self.normalized_output_cotangent_residual,
            self.query_accumulation_semaphore,
            self.key_accumulation_semaphore,
            self.value_accumulation_semaphore,
        )


def export_h100_normalized_exponential_backward_artifact(
    compiled: CompiledH100NormalizedExponentialBackward,
    *,
    function_prefix: str,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
    source_slot_reuse_groups: Mapping[str, int] | None = None,
) -> CutePipelineArtifact:
    """Export a sealed device artifact from one graph-derived CuTe pipeline."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    executable = compiled.executable
    with tempfile.TemporaryDirectory(prefix="shuttle-cute-artifact-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    device_image = elf64_symbol_bytes(host_object, "kernels_binary")
    kernel_names = tuple(executable.kernel_info)
    if not kernel_names:
        raise ValueError("compiled CuTe pipeline did not report any device kernels")
    return CutePipelineArtifact(
        request=_compiled_backward_request(compiled),
        host_object=host_object,
        device_image=device_image,
        kernel_names=kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={
            "emitter": _file_sha256(Path(__file__)),
            "algebra": _file_sha256(Path(inspect.getfile(type(compiled.template.program.forward.region)))),
            "physical_lowering": _file_sha256(Path(inspect.getfile(lower_h100_normalized_exp_backward))),
            **(
                {"ordered_accumulation_plan": _file_sha256(Path(inspect.getfile(Sm90BackwardAccumulationPlan)))}
                if compiled.template.lowering.accumulation_plan is not None
                else {}
            ),
            "schedule_model": _file_sha256(Path(inspect.getfile(Sm90BackwardAccumulatorSchedule))),
            "physical_template": _file_sha256(Path(inspect.getfile(type(compiled.template.kernel)))),
            "preprocess": _file_sha256(Path(inspect.getfile(FlashAttentionBackwardPreprocess))),
            "postprocess": _file_sha256(Path(inspect.getfile(FlashAttentionBackwardPostprocess))),
        },
        launch_recipe=_launch_recipe(
            compiled,
            kernel_names,
            source_slot_shapes=source_slot_shapes,
            source_slot_bindings=source_slot_bindings,
            source_slot_ordinals=source_slot_ordinals,
            source_slot_reuse_groups=source_slot_reuse_groups,
            reverse_output_cotangent_slot=(
                "layout_transport_output_cotangent"
                if compiled.layout_transport_output_cotangent is not None
                else (
                    "normalized_output_cotangent"
                    if compiled.normalized_output_cotangent is not None
                    else "output_cotangent"
                )
            ),
            reverse_output_cotangent_stride=(
                tuple(compiled.layout_transport_output_cotangent.stride())
                if compiled.layout_transport_output_cotangent is not None
                else None
            ),
            reverse_output_cotangent_residual_slot=(
                "normalized_output_cotangent_residual"
                if compiled.normalized_output_cotangent_residual is not None
                else None
            ),
        ),
        structured_device_ir=(compiled.structured_device_ir,),
    )


def export_h100_normalized_exponential_forward_artifact(
    compiled: CompiledH100StreamingProgram,
    *,
    compile_request: CutePipelineCompileRequest,
    function_prefix: str,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
) -> CutePipelineArtifact:
    """Export one source-independent normalized-exponential forward artifact."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    executable = compiled.executable
    with tempfile.TemporaryDirectory(prefix="shuttle-cute-artifact-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    device_image = elf64_symbol_bytes(host_object, "kernels_binary")
    kernel_names = tuple(executable.kernel_info)
    if len(kernel_names) != 1:
        raise ValueError(
            f"the SM90 normalized-exponential forward pipeline must contain one kernel, got {len(kernel_names)}"
        )
    template = build_h100_streaming_template(compiled.program)
    return CutePipelineArtifact(
        request=compile_request,
        host_object=host_object,
        device_image=device_image,
        kernel_names=kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={
            "emitter": _file_sha256(Path(__file__)),
            "algebra": _file_sha256(Path(inspect.getfile(type(compiled.program.region)))),
            "forward_lowering": _file_sha256(Path(inspect.getfile(build_h100_streaming_template))),
            "forward_template": _file_sha256(Path(inspect.getfile(type(template.kernel)))),
        },
        launch_recipe=_forward_launch_recipe(
            compiled,
            kernel_names[0],
            source_slot_shapes=source_slot_shapes,
            source_slot_bindings=source_slot_bindings,
            source_slot_ordinals=source_slot_ordinals,
        ),
        structured_device_ir=(compiled.structured_device_ir,),
    )


def export_h100_normalized_exponential_value_and_grad_artifact(
    compiled: CompiledH100NormalizedExponentialValueAndGrad,
    *,
    function_prefix: str,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
    source_slot_reuse_groups: Mapping[str, int] | None = None,
) -> CutePipelineArtifact:
    """Export the shared forward region and pullback as one sealed artifact."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    executable = compiled.executable
    with tempfile.TemporaryDirectory(prefix="shuttle-cute-artifact-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    device_image = elf64_symbol_bytes(host_object, "kernels_binary")
    kernel_names = tuple(executable.kernel_info)
    expected_kernels = (8 if compiled.normalized_output is None else 9) + int(
        compiled.query_accumulation_semaphore is not None
    )
    if len(kernel_names) != expected_kernels:
        raise ValueError(
            f"the SM90 value-and-pullback pipeline must contain {expected_kernels} kernels, got {len(kernel_names)}"
        )
    return CutePipelineArtifact(
        request=_compile_request(compiled.backward_template),
        host_object=host_object,
        device_image=device_image,
        kernel_names=kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={
            "emitter": _file_sha256(Path(__file__)),
            "algebra": _file_sha256(Path(inspect.getfile(type(compiled.backward_template.program.forward.region)))),
            "forward_lowering": _file_sha256(Path(inspect.getfile(build_h100_streaming_template))),
            "backward_lowering": _file_sha256(Path(inspect.getfile(lower_h100_normalized_exp_backward))),
            "forward_template": _file_sha256(Path(inspect.getfile(type(compiled.forward_template.kernel)))),
            "backward_template": _file_sha256(Path(inspect.getfile(type(compiled.backward_template.kernel)))),
            "preprocess": _file_sha256(Path(inspect.getfile(FlashAttentionBackwardPreprocess))),
            "post_normalization_pullback": _file_sha256(Path(inspect.getfile(CutePostNormalizationPullback))),
            "postprocess": _file_sha256(Path(inspect.getfile(FlashAttentionBackwardPostprocess))),
        },
        launch_recipe=_value_and_grad_launch_recipe(
            compiled,
            kernel_names,
            source_slot_shapes=source_slot_shapes,
            source_slot_bindings=source_slot_bindings,
            source_slot_ordinals=source_slot_ordinals,
            source_slot_reuse_groups=source_slot_reuse_groups,
        ),
        structured_device_ir=(compiled.structured_device_ir,),
    )


def h100_normalized_exponential_backward_ffi_bindings(
    compiled: CompiledH100NormalizedExponentialBackward,
) -> tuple[CutePipelineTensorBinding, ...]:
    """Describe the generated pipeline ABI in host-call argument order."""
    program = compiled.template.program
    region = program.forward.region
    query, key = region.producer.inputs
    value = region.consumer.inputs[1]
    output = region.finalizer.output
    log_sum_exp_shape = tuple(compiled.log_sum_exp_log2.shape)
    bindings = (
        CutePipelineTensorBinding("query", DType.BF16, query.shape, CutePipelineBindingKind.OPERAND, 0),
        CutePipelineTensorBinding("key", DType.BF16, key.shape, CutePipelineBindingKind.OPERAND, 1),
        CutePipelineTensorBinding("value", DType.BF16, value.shape, CutePipelineBindingKind.OPERAND, 2),
        CutePipelineTensorBinding("output", DType.BF16, output.shape, CutePipelineBindingKind.OPERAND, 3),
        CutePipelineTensorBinding(
            "output_cotangent", compiled.output_cotangent_dtype, output.shape, CutePipelineBindingKind.OPERAND, 4
        ),
        CutePipelineTensorBinding("log_sum_exp", DType.FP32, log_sum_exp_shape, CutePipelineBindingKind.OPERAND, 5),
        CutePipelineTensorBinding("query_cotangent", DType.BF16, query.shape, CutePipelineBindingKind.RESULT, 0),
        CutePipelineTensorBinding("key_cotangent", DType.BF16, key.shape, CutePipelineBindingKind.RESULT, 1),
        CutePipelineTensorBinding("value_cotangent", DType.BF16, value.shape, CutePipelineBindingKind.RESULT, 2),
        CutePipelineTensorBinding(
            "output_dot",
            DType.FP32,
            tuple(compiled.output_dot.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "log_sum_exp_log2",
            DType.FP32,
            log_sum_exp_shape,
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "query_accumulator",
            DType.FP32,
            tuple(compiled.query_accumulator.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "key_accumulator",
            DType.FP32,
            tuple(compiled.key_accumulator.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "value_accumulator",
            DType.FP32,
            tuple(compiled.value_accumulator.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
    )
    bindings = (
        *bindings,
        *_accumulation_semaphore_bindings(
            compiled.query_accumulation_semaphore,
            compiled.key_accumulation_semaphore,
            compiled.value_accumulation_semaphore,
        ),
    )
    if compiled.normalized_output_cotangent is None:
        return bindings
    assert compiled.normalized_output_cotangent_residual is not None
    return (
        *bindings,
        CutePipelineTensorBinding(
            "normalized_output",
            DType.FP32,
            output.shape,
            CutePipelineBindingKind.OPERAND,
            6,
        ),
        CutePipelineTensorBinding(
            "normalized_output_cotangent",
            DType.BF16,
            tuple(compiled.normalized_output_cotangent.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "normalized_output_cotangent_residual",
            DType.BF16,
            tuple(compiled.normalized_output_cotangent_residual.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
    )


def h100_normalized_exponential_value_and_grad_ffi_bindings(
    compiled: CompiledH100NormalizedExponentialValueAndGrad,
) -> tuple[CutePipelineTensorBinding, ...]:
    """Describe the four-input, value-plus-three-cotangent graph boundary."""
    program = compiled.backward_template.program
    region = program.forward.region
    query, key = region.producer.inputs
    value = region.consumer.inputs[1]
    output = region.finalizer.output
    bindings = (
        CutePipelineTensorBinding("query", DType.BF16, query.shape, CutePipelineBindingKind.OPERAND, 0),
        CutePipelineTensorBinding("key", DType.BF16, key.shape, CutePipelineBindingKind.OPERAND, 1),
        CutePipelineTensorBinding("value", DType.BF16, value.shape, CutePipelineBindingKind.OPERAND, 2),
        CutePipelineTensorBinding("output_cotangent", DType.BF16, output.shape, CutePipelineBindingKind.OPERAND, 3),
        CutePipelineTensorBinding("output", DType.BF16, output.shape, CutePipelineBindingKind.RESULT, 0),
        CutePipelineTensorBinding("query_cotangent", DType.BF16, query.shape, CutePipelineBindingKind.RESULT, 1),
        CutePipelineTensorBinding("key_cotangent", DType.BF16, key.shape, CutePipelineBindingKind.RESULT, 2),
        CutePipelineTensorBinding("value_cotangent", DType.BF16, value.shape, CutePipelineBindingKind.RESULT, 3),
        CutePipelineTensorBinding(
            "log_sum_exp", DType.FP32, tuple(compiled.log_sum_exp.shape), CutePipelineBindingKind.TEMPORARY
        ),
        CutePipelineTensorBinding(
            "output_dot", DType.FP32, tuple(compiled.output_dot.shape), CutePipelineBindingKind.TEMPORARY
        ),
        CutePipelineTensorBinding(
            "log_sum_exp_log2", DType.FP32, tuple(compiled.log_sum_exp_log2.shape), CutePipelineBindingKind.TEMPORARY
        ),
        CutePipelineTensorBinding(
            "query_accumulator",
            DType.FP32,
            tuple(compiled.query_accumulator.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "key_accumulator", DType.FP32, tuple(compiled.key_accumulator.shape), CutePipelineBindingKind.TEMPORARY
        ),
        CutePipelineTensorBinding(
            "value_accumulator",
            DType.FP32,
            tuple(compiled.value_accumulator.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
    )
    bindings = (
        *bindings,
        *_accumulation_semaphore_bindings(
            compiled.query_accumulation_semaphore,
            compiled.key_accumulation_semaphore,
            compiled.value_accumulation_semaphore,
        ),
    )
    if compiled.normalized_output is None:
        return bindings
    assert compiled.normalized_output_cotangent is not None
    assert compiled.normalized_output_cotangent_residual is not None
    return (
        *bindings,
        CutePipelineTensorBinding(
            "normalized_output",
            DType.FP32,
            tuple(compiled.normalized_output.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "normalized_output_cotangent",
            DType.BF16,
            tuple(compiled.normalized_output_cotangent.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
        CutePipelineTensorBinding(
            "normalized_output_cotangent_residual",
            DType.BF16,
            tuple(compiled.normalized_output_cotangent_residual.shape),
            CutePipelineBindingKind.TEMPORARY,
        ),
    )


def build_h100_normalized_exponential_backward_template(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    *,
    domain_traversal: ContractDomainTraversal | None = None,
    expand_output_cotangent: bool = False,
) -> H100NormalizedExponentialBackwardTemplate:
    """Bind generic Contract/Fold/Map pullback algebra to the SM90 mainloop."""
    lowering = lower_h100_normalized_exp_backward(program)
    region = program.forward.region
    dimensions = region.dimensions
    relation_group_size = lowering.relation_layout.relation_group_size
    physical_schedule = lowering.physical_schedule
    input_event_schedule = lowering.input_event_schedule
    score_map = lowering.score_map
    traversal = program.ownership.traversal if domain_traversal is None else domain_traversal
    if traversal is not program.ownership.traversal:
        raise ValueError("runtime traversal must match the relation-derived reverse ownership plan")
    forward_roles = score_input_roles(
        score_map.expression,
        score_input=score_map.score_input,
        owner_inputs=score_map.owner_inputs,
        streamed_inputs=score_map.streamed_inputs,
    )
    score_mod = cute_keyword_score_map_adapter(cute_score_map(score_map.expression, forward_roles))
    score_mod_bwd = cute_keyword_score_map_vjp_adapter(
        cute_score_map_vjp(
            lowering.score_map_vjp_codegen_expression,
            score_input_roles(
                lowering.score_map_vjp_codegen_expression,
                score_input=score_map.score_input,
                cotangent_input=program.pullback.normalized_exp_pullback.output.name,
                owner_inputs=score_map.owner_inputs,
                streamed_inputs=score_map.streamed_inputs,
            ),
        ),
    )
    domain_predicate = (
        None
        if region.domain_predicate is None
        else cute_affine_index_predicate(
            region.domain_predicate,
            owner_dimension=region.owner_axes[-1].id,
            streamed_dimension=dimensions.normalized_axes[0].id,
        )
    )
    # Keep the backward-only FlashAttention dependency out of the forward
    # artifact export path.  Some forward-only runtime environments carry a
    # narrower FlashAttention helper surface.
    from .cute_normalized_exp_backward_sm90 import ShuttleNormalizedExponentialBackwardSm90  # noqa: PLC0415

    kernel = ShuttleNormalizedExponentialBackwardSm90(
        cutlass.BFloat16,
        dimensions.producer_features[0].extent,
        dimensions.consumer_features[0].extent,
        relation_group_size=relation_group_size,
        lower_triangular_domain=traversal is ContractDomainTraversal.LOWER_TRIANGULAR,
        is_local=False,
        deterministic=physical_schedule.deterministic_accumulation,
        accumulation_plan=lowering.accumulation_plan,
        tile_m=physical_schedule.contract.tile_m,
        tile_n=physical_schedule.contract.tile_n,
        Q_stage=input_event_schedule.producer_lhs_pipeline.capacity,
        dO_stage=input_event_schedule.output_cotangent_pipeline.capacity,
        dO_expansion_terms=(
            physical_schedule.post_normalization_cotangent_expansion_terms
            if expand_output_cotangent or not is_identity_post_normalization_pullback(program)
            else 1
        ),
        PdS_stage=input_event_schedule.dataflow.score_cotangent_buffer.capacity,
        SdP_swapAB=physical_schedule.score_cotangent_swap_ab,
        dKV_swapAB=physical_schedule.right_gradient_swap_ab,
        dQ_swapAB=physical_schedule.left_gradient_swap_ab,
        AtomLayoutMSdP=physical_schedule.score_atom_layout_m,
        AtomLayoutNdKV=physical_schedule.right_gradient_atom_layout_n,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
        num_threads=physical_schedule.contract.threads,
        V_in_regs=physical_schedule.resident_consumer_operand_in_registers,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=domain_predicate,
        has_aux_tensors=False,
        q_subtile_factor=physical_schedule.owner_subtile_factor,
        dQ_single_wg=(input_event_schedule.dataflow.producer_lhs_gradient.axes[-1].extent == 1),
        terminal_contract_epilogue_scale=lowering.terminal_contract_epilogue_scale,
    )
    return H100NormalizedExponentialBackwardTemplate(program, lowering, kernel)


def compile_h100_normalized_exponential_backward(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_cotangent: torch.Tensor,
    log_sum_exp: torch.Tensor,
    normalized_output: torch.Tensor | None = None,
    domain_traversal: ContractDomainTraversal | None = None,
) -> CompiledH100NormalizedExponentialBackward:
    """Compile the owned SM90 pullback pipeline for concrete CUDA tensors."""
    identity_pullback = is_identity_post_normalization_pullback(program)
    expands_output_cotangent = not identity_pullback or output_cotangent.dtype is torch.float32
    requires_layout_transport = output.stride(-1) != 1 or output_cotangent.stride(-1) != 1
    if program.schedule.deterministic_accumulation:
        validate_ordered_backward_variant(
            relation_group_size=query.shape[2] // key.shape[2],
            layout_transport=requires_layout_transport,
            combined=False,
        )
    if requires_layout_transport and expands_output_cotangent:
        raise ValueError("affine output layout transport cannot currently be combined with a non-identity finalizer VJP")
    if expands_output_cotangent and normalized_output is None:
        raise ValueError("the post-normalization VJP requires the saved FP32 normalized Contract result")
    template = build_h100_normalized_exponential_backward_template(
        program,
        domain_traversal=domain_traversal,
        expand_output_cotangent=expands_output_cotangent,
    )
    _validate_runtime_tensors(query, key, value, output, output_cotangent, log_sum_exp)
    physical_schedule = template.physical_schedule
    if template.lowering.accumulation_plan is not None:
        _validate_ordered_runtime_layout(template, query, key, value)
    batch, sequence, query_heads, head_dimension = query.shape
    value_dimension = value.shape[-1]
    key_value_heads = key.shape[2]
    streamed_gradients_are_direct = template.lowering.relation_layout.relation_group_size == 1
    key_sequence = key.shape[1]
    query_tile = physical_schedule.contract.tile_m
    key_tile = physical_schedule.contract.tile_n
    query_rounded = _tile_count(sequence, query_tile) * query_tile
    key_rounded = _tile_count(key_sequence, key_tile) * key_tile

    query_cotangent = torch.empty_like(query)
    key_cotangent = torch.empty_like(key)
    value_cotangent = torch.empty_like(value)
    output_dot = torch.empty((batch, query_heads, query_rounded), dtype=torch.float32, device=query.device)
    log_sum_exp_log2 = torch.empty_like(output_dot)
    query_accumulator = torch.empty(
        (batch, query_heads, query_rounded * head_dimension), dtype=torch.float32, device=query.device
    )
    key_accumulator = torch.empty(
        (batch, key_value_heads, key_rounded * head_dimension), dtype=torch.float32, device=query.device
    )
    value_accumulator = torch.empty_like(key_accumulator)
    if value_dimension != head_dimension:
        value_accumulator = torch.empty(
            (batch, key_value_heads, key_rounded * value_dimension), dtype=torch.float32, device=query.device
        )
    if streamed_gradients_are_direct:
        key_accumulator = key_cotangent
        value_accumulator = value_cotangent
    query_semaphore, key_semaphore, value_semaphore = _accumulation_semaphore_tensors(
        template,
        batch=batch,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        query_blocks=query_rounded // query_tile,
        key_blocks=key_rounded // key_tile,
        device=query.device,
    )
    normalized_output_cotangent = None
    normalized_output_cotangent_residual = None
    layout_transport_output = None
    layout_transport_output_cotangent = None
    layout_transport = None
    post_normalization_pullback = None
    if expands_output_cotangent:
        assert normalized_output is not None
        if (
            tuple(normalized_output.shape) != tuple(output.shape)
            or normalized_output.dtype
            not in ((torch.bfloat16, torch.float32) if identity_pullback else (torch.float32,))
            or not normalized_output.is_cuda
        ):
            raise ValueError(
                "saved normalized output must have the physical output shape and a dtype supported by its scalar VJP"
            )
        normalized_output_cotangent = torch.empty_strided(
            tuple(output.shape), tuple(output.stride()), dtype=torch.bfloat16, device=query.device
        )
        normalized_output_cotangent_residual = torch.empty_strided(
            tuple(output.shape), tuple(output.stride()), dtype=torch.bfloat16, device=query.device
        )
        pullback = program.pullback
        vjp_inputs = {value.name for value in pullback.post_normalization_vjp.inputs}
        supported_inputs = {pullback.output_cotangent.name, pullback.normalized_output.output.name}
        if not vjp_inputs <= supported_inputs:
            raise ValueError(
                "the current SM90 post-normalization pullback cannot bind auxiliary scalar Map operands; "
                f"unbound inputs are {sorted(vjp_inputs - supported_inputs)}"
            )
        post_normalization_pullback = CutePostNormalizationPullback(
            value_dimension,
            cute_post_normalization_vjp(
                pullback.post_normalization_vjp.expression,
                normalized_output_input=pullback.normalized_output.output.name,
                cotangent_input=pullback.output_cotangent.name,
            ),
        )
    if requires_layout_transport:
        layout_transport_output = torch.empty(tuple(output.shape), dtype=torch.bfloat16, device=query.device)
        layout_transport_output_cotangent = torch.empty(tuple(output.shape), dtype=torch.bfloat16, device=query.device)
        layout_transport = _AffineOutputLayoutTransport(tuple(output.shape))

    preprocess = FlashAttentionBackwardPreprocess(
        cutlass.BFloat16,
        head_dimension,
        value_dimension,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        use_padded_offsets=physical_schedule.use_padded_epilogue_offsets,
    )
    query_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        head_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
    )
    key_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        head_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
        cluster_size=physical_schedule.epilogue_cluster_size,
    )
    value_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        value_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
        cluster_size=physical_schedule.epilogue_cluster_size,
    )
    zero_fill = _Float32ZeroFill(physical_schedule.epilogue_threads)
    semaphore_zero_fill = _Int32SemaphoreZeroFill(
        template.lowering.accumulation_plan.reset_block[0]
        if template.lowering.accumulation_plan is not None
        else physical_schedule.epilogue_threads
    )

    @cute.jit
    def launch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        dout: cute.Tensor,
        lse: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dpsum: cute.Tensor,
        lse_log2: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        normalized_out: cute.Tensor | None,
        normalized_dout: cute.Tensor | None,
        normalized_dout_residual: cute.Tensor | None,
        transported_out: cute.Tensor | None,
        transported_dout: cute.Tensor | None,
        dq_semaphore: cute.Tensor | None,
        dk_semaphore: cute.Tensor | None,
        dv_semaphore: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        unit_gradient_normalization = cutlass.Float32(1.0)
        terminal_contract_epilogue_scale = cutlass.Float32(template.lowering.terminal_contract_epilogue_scale)
        q, k, v, dq, dk, dv, dpsum, lse_log2, dq_accum, dk_accum, dv_accum = [
            assume_tensor_aligned(tensor)
            for tensor in (q, k, v, dq, dk, dv, dpsum, lse_log2, dq_accum, dk_accum, dv_accum)
        ]
        if cutlass.const_expr(not requires_layout_transport):
            out = assume_tensor_aligned(out)
            dout = assume_tensor_aligned(dout)
        if cutlass.const_expr(expands_output_cotangent):
            normalized_out, normalized_dout, normalized_dout_residual = [
                assume_tensor_aligned(tensor) for tensor in (normalized_out, normalized_dout, normalized_dout_residual)
            ]
        if cutlass.const_expr(requires_layout_transport):
            transported_out = assume_tensor_aligned(transported_out)
            transported_dout = assume_tensor_aligned(transported_dout)
            layout_transport(
                out,
                dout,
                transported_out,
                transported_dout,
                dpsum,
                lse,
                lse_log2,
                stream,
            )
            zero_fill(dq_accum, stream)
        if cutlass.const_expr(dq_semaphore is not None):
            semaphore_zero_fill(dq_semaphore, dk_semaphore, dv_semaphore, stream)
        if cutlass.const_expr(expands_output_cotangent):
            post_normalization_pullback(
                normalized_out,
                dout,
                normalized_dout,
                normalized_dout_residual,
                dpsum,
                stream,
            )
        preprocess_output = (
            normalized_dout
            if cutlass.const_expr(expands_output_cotangent)
            else transported_out if cutlass.const_expr(requires_layout_transport) else out
        )
        preprocess_cotangent = (
            normalized_dout
            if cutlass.const_expr(expands_output_cotangent)
            else transported_dout if cutlass.const_expr(requires_layout_transport) else dout
        )
        if cutlass.const_expr(not requires_layout_transport):
            preprocess(
                preprocess_output,
                preprocess_cotangent,
                dpsum,
                lse,
                lse_log2,
                dq_accum,
                None,
                None,
                None,
                None,
                None,
                unit_gradient_normalization,
                stream,
            )
        reverse_output_cotangent = dout
        if cutlass.const_expr(expands_output_cotangent):
            post_normalization_pullback(
                normalized_out,
                dout,
                normalized_dout,
                normalized_dout_residual,
                dpsum,
                stream,
            )
            reverse_output_cotangent = normalized_dout
        if cutlass.const_expr(requires_layout_transport):
            reverse_output_cotangent = transported_dout
        if cutlass.const_expr(not streamed_gradients_are_direct):
            zero_fill(dk_accum, stream)
            zero_fill(dv_accum, stream)
        template.kernel(
            q,
            k,
            v,
            reverse_output_cotangent,
            normalized_dout_residual,
            lse_log2,
            dpsum,
            dq_accum,
            dk_accum,
            dv_accum,
            mdQ_semaphore=dq_semaphore,
            mdK_semaphore=dk_semaphore,
            mdV_semaphore=dv_semaphore,
            stream=stream,
        )
        query_postprocess(dq_accum, dq, terminal_contract_epilogue_scale, None, None, stream)
        if cutlass.const_expr(not streamed_gradients_are_direct):
            key_postprocess(dk_accum, dk, terminal_contract_epilogue_scale, None, None, stream)
            value_postprocess(dv_accum, dv, cutlass.Float32(1.0), None, None, stream)

    tensors = (
        query,
        key,
        value,
        output,
        output_cotangent,
        log_sum_exp,
        query_cotangent,
        key_cotangent,
        value_cotangent,
        output_dot,
        log_sum_exp_log2,
        query_accumulator,
        key_accumulator,
        value_accumulator,
        normalized_output,
        normalized_output_cotangent,
        normalized_output_cotangent_residual,
        layout_transport_output,
        layout_transport_output_cotangent,
        query_semaphore,
        key_semaphore,
        value_semaphore,
    )
    cute_arguments = [
        (
            to_cute_tensor(tensor, assumed_align=16, leading_dim=_unique_unit_stride_dimension(tensor))
            if tensor is not None
            else None
        )
        for tensor in tensors
    ]
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    executable, structured_device_ir = compile_cute_with_structured_snapshot(
        cute,
        launch,
        *cute_arguments,
        fake_stream,
        options="--enable-tvm-ffi",
    )
    return CompiledH100NormalizedExponentialBackward(
        template,
        executable,
        query_cotangent,
        key_cotangent,
        value_cotangent,
        output_dot,
        log_sum_exp_log2,
        query_accumulator,
        key_accumulator,
        value_accumulator,
        normalized_output_cotangent,
        normalized_output_cotangent_residual,
        layout_transport_output,
        layout_transport_output_cotangent,
        query_semaphore,
        key_semaphore,
        value_semaphore,
        tuple(query.stride()),
        tuple(key.stride()),
        tuple(value.stride()),
        tuple(output.stride()),
        tuple(output_cotangent.stride()),
        DType.BF16 if output_cotangent.dtype is torch.bfloat16 else DType.FP32,
        tuple(normalized_output.stride()) if normalized_output is not None else None,
        tuple(query_cotangent.stride()),
        tuple(key_cotangent.stride()),
        tuple(value_cotangent.stride()),
        streamed_gradients_are_direct,
        structured_device_ir,
    )


def compile_h100_normalized_exponential_value_and_grad(
    program: NormalizedExpContractPullbackProgram[Any, Sm90BackwardAccumulatorSchedule],
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_cotangent: torch.Tensor,
    domain_traversal: ContractDomainTraversal | None = None,
) -> CompiledH100NormalizedExponentialValueAndGrad:
    """Compile the shared forward region and its pullback as one CuTe pipeline."""
    if program.schedule.deterministic_accumulation:
        validate_ordered_backward_variant(relation_group_size=1, layout_transport=False, combined=True)
    forward_template = build_h100_streaming_template(program.forward, domain_traversal=domain_traversal)
    backward_template = build_h100_normalized_exponential_backward_template(
        program,
        domain_traversal=domain_traversal,
    )
    if forward_template.score_map != backward_template.score_map:
        raise ValueError("forward and pullback lowered different score scalar Maps")

    relation_layout = backward_template.lowering.relation_layout
    output = torch.empty_strided(
        relation_layout.owner_physical_shape(relation_layout.consumer_feature_extent),
        relation_layout.owner_physical_strides(relation_layout.consumer_feature_extent),
        dtype=torch.bfloat16,
        device=query.device,
    )
    batch, sequence, query_heads, head_dimension = query.shape
    log_sum_exp = torch.empty((batch, query_heads, sequence), dtype=torch.float32, device=query.device)
    _validate_runtime_tensors(query, key, value, output, output_cotangent, log_sum_exp)

    physical_schedule = backward_template.physical_schedule
    value_dimension = value.shape[-1]
    key_value_heads = key.shape[2]
    key_sequence = key.shape[1]
    query_tile = physical_schedule.contract.tile_m
    key_tile = physical_schedule.contract.tile_n
    query_rounded = _tile_count(sequence, query_tile) * query_tile
    key_rounded = _tile_count(key_sequence, key_tile) * key_tile

    query_cotangent = torch.empty_like(query)
    key_cotangent = torch.empty_like(key)
    value_cotangent = torch.empty_like(value)
    output_dot = torch.empty((batch, query_heads, query_rounded), dtype=torch.float32, device=query.device)
    log_sum_exp_log2 = torch.empty_like(output_dot)
    query_accumulator = torch.empty(
        (batch, query_heads, query_rounded * head_dimension), dtype=torch.float32, device=query.device
    )
    key_accumulator = torch.empty(
        (batch, key_value_heads, key_rounded * head_dimension), dtype=torch.float32, device=query.device
    )
    value_accumulator = torch.empty(
        (batch, key_value_heads, key_rounded * value_dimension), dtype=torch.float32, device=query.device
    )
    query_semaphore, key_semaphore, value_semaphore = _accumulation_semaphore_tensors(
        backward_template,
        batch=batch,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        query_blocks=query_rounded // query_tile,
        key_blocks=key_rounded // key_tile,
        device=query.device,
    )

    identity_pullback = is_identity_post_normalization_pullback(program)
    normalized_output = None
    normalized_output_cotangent = None
    normalized_output_cotangent_residual = None
    post_normalization_pullback = None
    if not identity_pullback:
        normalized_output = torch.empty_strided(
            tuple(output.shape), tuple(output.stride()), dtype=torch.float32, device=query.device
        )
        normalized_output_cotangent = torch.empty_strided(
            tuple(output.shape), tuple(output.stride()), dtype=torch.bfloat16, device=query.device
        )
        normalized_output_cotangent_residual = torch.empty_strided(
            tuple(output.shape), tuple(output.stride()), dtype=torch.bfloat16, device=query.device
        )
        pullback = program.pullback
        vjp_inputs = {value.name for value in pullback.post_normalization_vjp.inputs}
        supported_inputs = {pullback.output_cotangent.name, pullback.normalized_output.output.name}
        if not vjp_inputs <= supported_inputs:
            raise ValueError(
                "the current SM90 post-normalization pullback cannot bind auxiliary scalar Map operands; "
                f"unbound inputs are {sorted(vjp_inputs - supported_inputs)}"
            )
        post_normalization_pullback = CutePostNormalizationPullback(
            value_dimension,
            cute_post_normalization_vjp(
                pullback.post_normalization_vjp.expression,
                normalized_output_input=pullback.normalized_output.output.name,
                cotangent_input=pullback.output_cotangent.name,
            ),
        )

    preprocess = FlashAttentionBackwardPreprocess(
        cutlass.BFloat16,
        head_dimension,
        value_dimension,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        use_padded_offsets=physical_schedule.use_padded_epilogue_offsets,
    )
    query_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        head_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
    )
    key_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        head_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
        cluster_size=physical_schedule.epilogue_cluster_size,
    )
    value_postprocess = FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        value_dimension,
        physical_schedule.compute_capability,
        physical_schedule.epilogue_tile_m,
        num_threads=physical_schedule.epilogue_threads,
        AtomLayoutMdQ=physical_schedule.left_atom_layout_m,
        cluster_size=physical_schedule.epilogue_cluster_size,
    )
    zero_fill = _Float32ZeroFill(physical_schedule.epilogue_threads)
    semaphore_zero_fill = _Int32SemaphoreZeroFill(physical_schedule.epilogue_threads)

    @cute.jit
    def launch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        dout: cute.Tensor,
        out: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        lse: cute.Tensor,
        dpsum: cute.Tensor,
        lse_log2: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        normalized_out: cute.Tensor | None,
        normalized_dout: cute.Tensor | None,
        normalized_dout_residual: cute.Tensor | None,
        dq_semaphore: cute.Tensor | None,
        dk_semaphore: cute.Tensor | None,
        dv_semaphore: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        unit_gradient_normalization = cutlass.Float32(1.0)
        terminal_contract_epilogue_scale = cutlass.Float32(backward_template.lowering.terminal_contract_epilogue_scale)
        tensors = (q, k, v, dout, out, dq, dk, dv, lse, dpsum, lse_log2, dq_accum, dk_accum, dv_accum)
        q, k, v, dout, out, dq, dk, dv, lse, dpsum, lse_log2, dq_accum, dk_accum, dv_accum = [
            assume_tensor_aligned(tensor) for tensor in tensors
        ]
        if cutlass.const_expr(dq_semaphore is not None):
            semaphore_zero_fill(dq_semaphore, dk_semaphore, dv_semaphore, stream)
        forward_template.kernel(
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            normalized_out,
            stream=stream,
        )
        preprocess(
            out,
            dout,
            dpsum,
            lse,
            lse_log2,
            dq_accum,
            None,
            None,
            None,
            None,
            None,
            unit_gradient_normalization,
            stream,
        )
        reverse_output_cotangent = dout
        if cutlass.const_expr(not identity_pullback):
            post_normalization_pullback(
                normalized_out,
                dout,
                normalized_dout,
                normalized_dout_residual,
                dpsum,
                stream,
            )
            reverse_output_cotangent = normalized_dout
        zero_fill(dk_accum, stream)
        zero_fill(dv_accum, stream)
        backward_template.kernel(
            q,
            k,
            v,
            reverse_output_cotangent,
            normalized_dout_residual,
            lse_log2,
            dpsum,
            dq_accum,
            dk_accum,
            dv_accum,
            mdQ_semaphore=dq_semaphore,
            mdK_semaphore=dk_semaphore,
            mdV_semaphore=dv_semaphore,
            stream=stream,
        )
        query_postprocess(dq_accum, dq, terminal_contract_epilogue_scale, None, None, stream)
        key_postprocess(dk_accum, dk, terminal_contract_epilogue_scale, None, None, stream)
        value_postprocess(dv_accum, dv, cutlass.Float32(1.0), None, None, stream)

    tensors = (
        query,
        key,
        value,
        output_cotangent,
        output,
        query_cotangent,
        key_cotangent,
        value_cotangent,
        log_sum_exp,
        output_dot,
        log_sum_exp_log2,
        query_accumulator,
        key_accumulator,
        value_accumulator,
        normalized_output,
        normalized_output_cotangent,
        normalized_output_cotangent_residual,
        query_semaphore,
        key_semaphore,
        value_semaphore,
    )
    cute_arguments = [to_cute_tensor(tensor, assumed_align=16) if tensor is not None else None for tensor in tensors]
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    executable, structured_device_ir = compile_cute_with_structured_snapshot(
        cute,
        launch,
        *cute_arguments,
        fake_stream,
        options="--enable-tvm-ffi",
    )
    return CompiledH100NormalizedExponentialValueAndGrad(
        forward_template=forward_template,
        backward_template=backward_template,
        executable=executable,
        output=output,
        query_cotangent=query_cotangent,
        key_cotangent=key_cotangent,
        value_cotangent=value_cotangent,
        log_sum_exp=log_sum_exp,
        output_dot=output_dot,
        log_sum_exp_log2=log_sum_exp_log2,
        query_accumulator=query_accumulator,
        key_accumulator=key_accumulator,
        value_accumulator=value_accumulator,
        normalized_output=normalized_output,
        normalized_output_cotangent=normalized_output_cotangent,
        normalized_output_cotangent_residual=normalized_output_cotangent_residual,
        query_accumulation_semaphore=query_semaphore,
        key_accumulation_semaphore=key_semaphore,
        value_accumulation_semaphore=value_semaphore,
        query_stride=tuple(query.stride()),
        key_stride=tuple(key.stride()),
        value_stride=tuple(value.stride()),
        output_stride=tuple(output.stride()),
        output_cotangent_stride=tuple(output_cotangent.stride()),
        normalized_output_stride=(tuple(normalized_output.stride()) if normalized_output is not None else None),
        normalized_output_cotangent_stride=(
            tuple(normalized_output_cotangent.stride()) if normalized_output_cotangent is not None else None
        ),
        normalized_output_cotangent_residual_stride=(
            tuple(normalized_output_cotangent_residual.stride())
            if normalized_output_cotangent_residual is not None
            else None
        ),
        query_cotangent_stride=tuple(query_cotangent.stride()),
        key_cotangent_stride=tuple(key_cotangent.stride()),
        value_cotangent_stride=tuple(value_cotangent.stride()),
        structured_device_ir=structured_device_ir,
    )


class _Float32ZeroFill:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads

    @cute.jit
    def __call__(self, tensor: cute.Tensor, stream: cuda.CUstream):
        self.kernel(tensor).launch(
            grid=[cute.ceil_div(cute.size(tensor), self.num_threads), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, tensor: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        flat = cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))
        index = block * self.num_threads + thread
        if index < cute.size(flat):
            flat[index] = cutlass.Float32(0.0)


class _Int32SemaphoreZeroFill:
    """Clear all derived deterministic-accumulation counters in one launch."""

    def __init__(self, num_threads: int):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key: cute.Tensor | None,
        value: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        element_count = cute.size(query)
        if cutlass.const_expr(key is not None):
            element_count += cute.size(key)
        if cutlass.const_expr(value is not None):
            element_count += cute.size(value)
        self.kernel(query, key, value).launch(
            grid=[cute.ceil_div(element_count, self.num_threads), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, query: cute.Tensor, key: cute.Tensor | None, value: cute.Tensor | None):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * self.num_threads + thread
        query_flat = cute.make_tensor(query.iterator, cute.make_layout(cute.size(query)))
        if index < cute.size(query_flat):
            query_flat[index] = cutlass.Int32(0)
        else:
            index -= cute.size(query_flat)
            if cutlass.const_expr(key is not None):
                key_flat = cute.make_tensor(key.iterator, cute.make_layout(cute.size(key)))
                if index < cute.size(key_flat):
                    key_flat[index] = cutlass.Int32(0)
                else:
                    index -= cute.size(key_flat)
                    if cutlass.const_expr(value is not None):
                        value_flat = cute.make_tensor(value.iterator, cute.make_layout(cute.size(value)))
                        if index < cute.size(value_flat):
                            value_flat[index] = cutlass.Int32(0)
            elif cutlass.const_expr(value is not None):
                value_flat = cute.make_tensor(value.iterator, cute.make_layout(cute.size(value)))
                if index < cute.size(value_flat):
                    value_flat[index] = cutlass.Int32(0)


def _validate_ordered_runtime_layout(
    template: H100NormalizedExponentialBackwardTemplate,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    plan = template.lowering.accumulation_plan
    if plan is None:
        raise ValueError("ordered runtime validation requires an accumulation plan")
    expected = (
        (plan.batch, plan.owner_extent, plan.heads, plan.producer_features),
        (plan.batch, plan.streamed_extent, plan.heads, plan.producer_features),
        (plan.batch, plan.streamed_extent, plan.heads, plan.consumer_features),
    )
    if tuple(tuple(tensor.shape) for tensor in (query, key, value)) != expected:
        raise ValueError("ordered runtime tensor shapes differ from the shared scheduler plan")


def _accumulation_semaphore_tensors(
    template: H100NormalizedExponentialBackwardTemplate,
    *,
    batch: int,
    query_heads: int,
    key_value_heads: int,
    query_blocks: int,
    key_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    queues = template.lowering.input_event_schedule.device_accumulation_queues
    deterministic = template.physical_schedule.deterministic_accumulation
    if not deterministic:
        if queues:
            raise ValueError("unordered accumulation must not allocate deterministic Event queues")
        return None, None, None

    relation_group_extent = template.lowering.relation_layout.relation_group_size
    expected_queue_count = 1 if relation_group_extent == 1 else 3
    if len(queues) != expected_queue_count:
        raise ValueError(
            "deterministic accumulation Event queues do not match the selected index relation: "
            f"expected {expected_queue_count}, found {len(queues)}"
        )

    query_owner_count = len(set(queues[0].allocation.allocation_owners))
    if query_owner_count != query_blocks:
        raise ValueError(
            "query-gradient Event queue owner domain does not match the emitted accumulator tiles: "
            f"expected {query_blocks}, found {query_owner_count}"
        )
    plan = template.lowering.accumulation_plan
    if plan is None or plan.counter_shape != (batch, query_heads, query_blocks, 1):
        raise ValueError("ordered semaphore replicas differ from the shared accumulation plan")
    query = torch.zeros(plan.counter_shape, dtype=torch.int32, device=device)
    if relation_group_extent == 1:
        return query, None, None

    rhs_owner_counts = tuple(len(set(queue.allocation.allocation_owners)) for queue in queues[1:])
    if rhs_owner_counts != (key_blocks, key_blocks):
        raise ValueError(
            "right-gradient Event queue owner domains do not match the emitted accumulator tiles: "
            f"expected {(key_blocks, key_blocks)}, found {rhs_owner_counts}"
        )
    key = torch.zeros((batch, key_value_heads, key_blocks, 1), dtype=torch.int32, device=device)
    value = torch.zeros_like(key)
    return query, key, value


def _accumulation_semaphore_bindings(
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
) -> tuple[CutePipelineTensorBinding, ...]:
    tensors = (
        ("query_accumulation_semaphore", query),
        ("key_accumulation_semaphore", key),
        ("value_accumulation_semaphore", value),
    )
    return tuple(
        CutePipelineTensorBinding(
            name,
            DType.INT32,
            tuple(tensor.shape),
            CutePipelineBindingKind.TEMPORARY,
        )
        for name, tensor in tensors
        if tensor is not None
    )


def _value_and_grad_launch_recipe(
    compiled: CompiledH100NormalizedExponentialValueAndGrad,
    kernel_names: tuple[str, ...],
    *,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
    source_slot_reuse_groups: Mapping[str, int] | None = None,
) -> CutePipelineLaunchRecipe:
    generic_finalizer = compiled.normalized_output is not None
    has_device_queues = compiled.query_accumulation_semaphore is not None
    forward_kernel_name, map_kernel_name, reverse_kernel_names = _value_and_grad_kernel_partition(
        kernel_names,
        has_device_queues=has_device_queues,
        generic_finalizer=generic_finalizer,
    )
    reverse_recipe = _launch_recipe(
        compiled,
        reverse_kernel_names,
        reverse_output_cotangent_slot=("normalized_output_cotangent" if generic_finalizer else "output_cotangent"),
        reverse_output_cotangent_stride=(
            compiled.normalized_output_cotangent_stride if generic_finalizer else compiled.output_cotangent_stride
        ),
        reverse_output_cotangent_residual_slot=("normalized_output_cotangent_residual" if generic_finalizer else None),
        reverse_output_cotangent_residual_stride=(
            compiled.normalized_output_cotangent_residual_stride if generic_finalizer else None
        ),
    )
    program = compiled.backward_template.program
    region = program.forward.region
    query, key = region.producer.inputs
    value = region.consumer.inputs[1]
    output = region.finalizer.output
    relation_layout = compiled.backward_template.lowering.relation_layout
    output_physical_shape = relation_layout.owner_physical_shape(relation_layout.consumer_feature_extent)
    source_shapes = source_slot_shapes or {}
    source_bindings = source_slot_bindings or {}
    source_reuse_groups = source_slot_reuse_groups or {}
    private_reuse_group = max(source_reuse_groups.values(), default=10) + 1

    slots = (
        _slot(
            "query",
            CutePipelineElementType.BF16,
            source_shapes.get("query", query.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("query", 0),
        ),
        _slot(
            "key",
            CutePipelineElementType.BF16,
            source_shapes.get("key", key.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("key", 1),
        ),
        _slot(
            "value",
            CutePipelineElementType.BF16,
            source_shapes.get("value", value.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("value", 2),
        ),
        _slot(
            "output_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("output_cotangent", output.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("output_cotangent", 3),
        ),
        _slot(
            "output",
            CutePipelineElementType.BF16,
            source_shapes.get("output", output.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("output", 0),
        ),
        _slot(
            "key_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("key_cotangent", key.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("key_cotangent", 2),
        ),
        _slot(
            "value_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("value_cotangent", value.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("value_cotangent", 3),
        ),
        _slot(
            "query_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("query_cotangent", query.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("query_cotangent", 1),
        ),
        _slot(
            "log_sum_exp",
            CutePipelineElementType.F32,
            source_shapes.get("log_sum_exp", tuple(compiled.log_sum_exp.shape)),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=source_reuse_groups.get("log_sum_exp"),
        ),
        _slot(
            "log_sum_exp_log2",
            CutePipelineElementType.F32,
            tuple(compiled.log_sum_exp_log2.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=source_reuse_groups.get("log_sum_exp_log2"),
        ),
        _slot(
            "output_dot",
            CutePipelineElementType.F32,
            tuple(compiled.output_dot.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=source_reuse_groups.get("output_dot"),
        ),
        _slot(
            "query_accumulator",
            CutePipelineElementType.F32,
            tuple(compiled.query_accumulator.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=private_reuse_group,
        ),
        _slot(
            "key_accumulator",
            CutePipelineElementType.F32,
            tuple(compiled.key_accumulator.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=private_reuse_group + 1,
        ),
        _slot(
            "value_accumulator",
            CutePipelineElementType.F32,
            tuple(compiled.value_accumulator.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=private_reuse_group + 2,
        ),
    )
    if generic_finalizer:
        assert compiled.normalized_output is not None and compiled.normalized_output_cotangent is not None
        assert compiled.normalized_output_cotangent_residual is not None
        assert compiled.normalized_output_stride is not None and compiled.normalized_output_cotangent_stride is not None
        assert compiled.normalized_output_cotangent_residual_stride is not None
        slots = (
            *slots[:11],
            _slot(
                "normalized_output",
                CutePipelineElementType.F32,
                tuple(compiled.normalized_output.shape),
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 3,
            ),
            _slot(
                "normalized_output_cotangent",
                CutePipelineElementType.BF16,
                tuple(compiled.normalized_output_cotangent.shape),
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 4,
            ),
            _slot(
                "normalized_output_cotangent_residual",
                CutePipelineElementType.BF16,
                tuple(compiled.normalized_output_cotangent_residual.shape),
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 5,
            ),
            *slots[11:],
        )
    for name, tensor in (
        ("query_accumulation_semaphore", compiled.query_accumulation_semaphore),
        ("key_accumulation_semaphore", compiled.key_accumulation_semaphore),
        ("value_accumulation_semaphore", compiled.value_accumulation_semaphore),
    ):
        if tensor is not None and all(slot.name != name for slot in slots):
            slots = (*slots, _slot(name, CutePipelineElementType.I32, tuple(tensor.shape), CutePipelineSlotKind.SCRATCH))
    slots = _order_source_slots(slots, 11, source_slot_ordinals)

    forward_launch = _forward_launch(
        compiled.forward_template,
        forward_kernel_name,
        writes_log_sum_exp=True,
        query_stride=compiled.query_stride,
        key_stride=compiled.key_stride,
        value_stride=compiled.value_stride,
        output_stride=compiled.output_stride,
        normalized_output_shape=(tuple(compiled.normalized_output.shape) if generic_finalizer else None),
        normalized_output_stride=compiled.normalized_output_stride,
    )
    reverse_launches = reverse_recipe.launches
    if generic_finalizer:
        assert map_kernel_name is not None
        map_fold_launch = _post_normalization_vjp_launch(
            compiled,
            map_kernel_name,
            output_physical_shape,
            relation_layout,
        )
        reverse_launches = (reverse_launches[0], map_fold_launch, *reverse_launches[1:])
    carry_state_producer = "post_normalization_vjp_and_output_dot_fold" if generic_finalizer else "preprocess"
    return CutePipelineLaunchRecipe(
        schema_version=2,
        slots=slots,
        launches=(forward_launch, *reverse_launches),
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_CARRY,
                CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
                CuteEventExecutionPoint("preprocess", 0),
            ),
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
                CuteEventExecutionPoint("normalized_exp_contract_pullback", 0),
            ),
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_CARRY,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint(carry_state_producer, 0),
                CuteEventExecutionPoint("normalized_exp_contract_pullback", 0),
            ),
        ),
    )


def _post_normalization_vjp_launch(
    compiled: CompiledH100NormalizedExponentialBackward | CompiledH100NormalizedExponentialValueAndGrad,
    kernel_name: str,
    output_physical_shape: tuple[int, ...],
    relation_layout: Any,
) -> CuteKernelLaunch:
    normalized_output = getattr(compiled, "normalized_output", None)
    normalized_output_shape = output_physical_shape if normalized_output is None else tuple(normalized_output.shape)
    normalized_output_stride = getattr(compiled, "normalized_output_stride", None)
    normalized_output_cotangent = compiled.normalized_output_cotangent
    normalized_output_cotangent_residual = compiled.normalized_output_cotangent_residual
    if normalized_output_cotangent is None or normalized_output_cotangent_residual is None:
        raise ValueError("post-normalization VJP launch requires cotangent expansion storage")
    normalized_output_cotangent_stride = getattr(compiled, "normalized_output_cotangent_stride", None)
    normalized_output_cotangent_residual_stride = getattr(
        compiled,
        "normalized_output_cotangent_residual_stride",
        None,
    )
    return CuteKernelLaunch(
        name="post_normalization_vjp_and_output_dot_fold",
        kernel_symbol=kernel_name,
        grid=(
            relation_layout.owner_extent,
            relation_layout.flattened_relation_axis.extent,
            relation_layout.batch_extent,
        ),
        block=(cute.arch.WARP_SIZE, 1, 1),
        dynamic_shared_memory_bytes=0,
        function_attributes=(),
        source_task_roles=(CuteSourceTaskRole.STATE_CARRY,),
        arguments=(
            _transposed_tensor_argument(
                "normalized_output",
                "normalized_output",
                CutePipelineSlotAccess.READ,
                normalized_output_shape,
                (1, 3, 2, 0),
                logical_strides=normalized_output_stride,
            ),
            _transposed_tensor_argument(
                "output_cotangent",
                "output_cotangent",
                CutePipelineSlotAccess.READ,
                output_physical_shape,
                (1, 3, 2, 0),
                logical_strides=compiled.output_cotangent_stride,
            ),
            _transposed_tensor_argument(
                "normalized_output_cotangent",
                "normalized_output_cotangent",
                CutePipelineSlotAccess.WRITE,
                tuple(normalized_output_cotangent.shape),
                (1, 3, 2, 0),
                logical_strides=normalized_output_cotangent_stride,
            ),
            _transposed_tensor_argument(
                "normalized_output_cotangent_residual",
                "normalized_output_cotangent_residual",
                CutePipelineSlotAccess.WRITE,
                tuple(normalized_output_cotangent_residual.shape),
                (1, 3, 2, 0),
                logical_strides=normalized_output_cotangent_residual_stride,
            ),
            _transposed_tensor_argument(
                "output_dot",
                "output_dot",
                CutePipelineSlotAccess.WRITE,
                tuple(compiled.output_dot.shape),
                (2, 1, 0),
            ),
        ),
    )


def _forward_launch_recipe(
    compiled: CompiledH100StreamingProgram,
    kernel_name: str,
    *,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
) -> CutePipelineLaunchRecipe:
    program = compiled.program
    query, key = program.region.producer.inputs
    value = program.region.consumer.inputs[1]
    output = program.region.finalizer.output
    relation_layout = compiled.relation_layout
    batch = relation_layout.batch_extent
    sequence = relation_layout.owner_extent
    query_heads = relation_layout.flattened_relation_axis.extent
    source_shapes = source_slot_shapes or {}
    source_bindings = source_slot_bindings or {}
    slots: tuple[CutePipelineSlot, ...] = (
        _slot(
            "query",
            CutePipelineElementType.BF16,
            source_shapes.get("query", query.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("query", 0),
        ),
        _slot(
            "key",
            CutePipelineElementType.BF16,
            source_shapes.get("key", key.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("key", 1),
        ),
        _slot(
            "value",
            CutePipelineElementType.BF16,
            source_shapes.get("value", value.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("value", 2),
        ),
        _slot(
            "output",
            CutePipelineElementType.BF16,
            source_shapes.get("output", output.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("output", 0),
        ),
    )
    if compiled.writes_log_sum_exp:
        slots = (
            *slots,
            _slot(
                "log_sum_exp",
                CutePipelineElementType.F32,
                (batch, query_heads, sequence),
                (
                    CutePipelineSlotKind.PUBLIC_OUTPUT
                    if "log_sum_exp" in source_bindings
                    else CutePipelineSlotKind.SCRATCH
                ),
                output_index=source_bindings.get("log_sum_exp"),
            ),
        )
    if compiled.writes_normalized_output:
        slots = (
            *slots,
            _slot(
                "normalized_output",
                CutePipelineElementType.F32,
                relation_layout.owner_physical_shape(relation_layout.consumer_feature_extent),
                CutePipelineSlotKind.PUBLIC_OUTPUT,
                output_index=source_bindings.get("normalized_output", 1),
            ),
        )
    source_slot_count = 4 + int(compiled.writes_log_sum_exp) + int(compiled.writes_normalized_output)
    slots = _order_source_slots(slots, source_slot_count, source_slot_ordinals)
    template = build_h100_streaming_template(program)
    return CutePipelineLaunchRecipe(
        schema_version=2,
        slots=slots,
        host_call_arguments=("query", "key", "value", "output", *(None for _ in range(12))),
        launches=(
            _forward_launch(
                template,
                kernel_name,
                writes_log_sum_exp=compiled.writes_log_sum_exp,
                query_stride=compiled.query_stride,
                key_stride=compiled.key_stride,
                value_stride=compiled.value_stride,
                output_stride=compiled.output_stride,
                normalized_output_shape=(
                    relation_layout.owner_physical_shape(relation_layout.consumer_feature_extent)
                    if compiled.writes_normalized_output
                    else None
                ),
                normalized_output_stride=compiled.normalized_output_stride,
                source_task_roles=(CuteSourceTaskRole.STATE_ROOT, CuteSourceTaskRole.STATE_TERMINAL),
            ),
        ),
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
                CuteEventExecutionPoint("normalized_exp_contract_forward", 0),
                CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
            ),
        ),
    )


def _forward_launch(
    template: H100StreamingTemplate,
    kernel_name: str,
    *,
    writes_log_sum_exp: bool,
    dynamic_shared_memory_bytes: int | None = None,
    row_state_access: CutePipelineSlotAccess | None = None,
    row_max_stride: tuple[int, ...] | None = None,
    row_sum_stride: tuple[int, ...] | None = None,
    query_stride: tuple[int, ...] | None = None,
    key_stride: tuple[int, ...] | None = None,
    value_stride: tuple[int, ...] | None = None,
    output_stride: tuple[int, ...] | None = None,
    normalized_output_shape: tuple[int, ...] | None = None,
    normalized_output_stride: tuple[int, ...] | None = None,
    source_task_roles: tuple[CuteSourceTaskRole, ...] = (CuteSourceTaskRole.STATE_ROOT,),
) -> CuteKernelLaunch:
    layout = template.relation_layout
    batch = layout.batch_extent
    sequence = layout.owner_extent
    key_sequence = layout.streamed_extent
    key_value_heads = layout.shared_relation_axis.extent
    query_dimension = layout.producer_feature_extent
    value_dimension = layout.consumer_feature_extent
    group_size = layout.relation_group_size
    flattened_relation_extent = layout.flattened_relation_axis.extent
    query_shape = layout.owner_physical_shape(query_dimension)
    key_shape = layout.streamed_physical_shape(query_dimension)
    value_shape = layout.streamed_physical_shape(value_dimension)
    output_shape = layout.owner_physical_shape(value_dimension)
    schedule = template.schedule
    owner_tile = schedule.tile_m
    streamed_tile = schedule.tile_n
    num_blocks = _tile_count(sequence * group_size, owner_tile)
    size_one_key_value_head = key_sequence * (query_dimension + value_dimension) * 2
    l2_minor = (
        1
        if size_one_key_value_head > 50 * 1024 * 1024
        else _floor_power_of_two((50 * 1024 * 1024) // size_one_key_value_head)
    )
    head_batches = key_value_heads * batch
    l2_remainder = head_batches % l2_minor
    total_blocks = num_blocks * head_batches
    shared_memory = dynamic_shared_memory_bytes
    if shared_memory is None:
        shared_memory = _forward_dynamic_shared_memory_bytes(
            owner_tile=owner_tile,
            streamed_tile=streamed_tile,
            stages=schedule.stages,
            query_dimension=query_dimension,
            value_dimension=value_dimension,
        )
    arguments: tuple[Any, ...] = (
        _coordinate_argument("query_coordinates", layout.owner_coordinate_shape(query_dimension)),
        _coordinate_argument("key_coordinates", layout.streamed_coordinate_shape(query_dimension)),
        _coordinate_argument("value_coordinates", layout.streamed_coordinate_shape(value_dimension)),
        _coordinate_argument("output_coordinates", layout.owner_coordinate_shape(value_dimension)),
    )
    if normalized_output_shape is not None:
        arguments = (
            *arguments,
            _packed_relation_tensor_argument(
                "normalized_output",
                "normalized_output",
                CutePipelineSlotAccess.WRITE,
                normalized_output_shape,
                group_size,
                logical_strides=normalized_output_stride,
            ),
        )
    if writes_log_sum_exp:
        arguments = (
            *arguments,
            packed_log_sum_exp_argument(
                "log_sum_exp",
                "log_sum_exp",
                batch=batch,
                sequence=sequence,
                shared_heads=key_value_heads,
                group_size=group_size,
            ),
        )
    if row_state_access is not None:
        state_shape = (batch, flattened_relation_extent, sequence)
        arguments = (
            *arguments,
            _transposed_tensor_argument(
                "row_max", "row_max", row_state_access, state_shape, (2, 1, 0), logical_strides=row_max_stride
            ),
            _transposed_tensor_argument(
                "row_sum", "row_sum", row_state_access, state_shape, (2, 1, 0), logical_strides=row_sum_stride
            ),
        )
    arguments = (
        *arguments,
        sm90_packed_head_tensor_map_argument(
            "query_tma",
            "query",
            query_shape,
            owner_tile,
            group_size,
            CutePipelineSlotAccess.READ,
            logical_strides=query_stride,
        ),
        sm90_tensor_map_argument(
            "key_tma",
            "key",
            key_shape,
            (owner_tile, streamed_tile, 1, 1),
            CutePipelineSlotAccess.READ,
            logical_strides=key_stride,
        ),
        sm90_tensor_map_argument(
            "value_tma",
            "value",
            value_shape,
            (owner_tile, streamed_tile, 1, 1),
            CutePipelineSlotAccess.READ,
            logical_strides=value_stride,
        ),
        sm90_packed_head_tensor_map_argument(
            "output_tma",
            "output",
            output_shape,
            owner_tile,
            group_size,
            CutePipelineSlotAccess.WRITE,
            logical_strides=output_stride,
        ),
        _scalar_argument("scale_log2", math.log2(math.e)),
        _predicate_argument("has_query_sequence_lengths", False),
        _predicate_argument("has_key_sequence_lengths", False),
    )
    if template.domain_traversal is ContractDomainTraversal.LOWER_TRIANGULAR:
        arguments = (
            *arguments,
            _i32_argument("total_blocks", total_blocks),
            _i32_argument("query_blocks", num_blocks),
            _i32_argument("key_value_heads", key_value_heads),
            _i32_argument("batch", batch),
            _i32_argument("l2_minor", l2_minor),
            _fast_divmod_argument("key_value_heads_divmod", key_value_heads),
            _fast_divmod_argument("l2_minor_divmod", l2_minor),
            _fast_divmod_argument("l2_major_divmod", l2_minor * num_blocks),
            _fast_divmod_argument("l2_residual_divmod", max(l2_remainder, 1)),
            _i32_argument("head_batch_quotient", head_batches // l2_minor),
            _fast_divmod_argument("num_splits_divmod", 1),
        )
        grid = (total_blocks, 1, 1)
    else:
        arguments = (
            *arguments,
            _i32_argument("query_blocks", num_blocks),
            _i32_argument("key_value_heads", key_value_heads),
            _i32_argument("batch", batch),
            _fast_divmod_argument("num_splits_divmod", 1),
        )
        grid = (num_blocks, key_value_heads, batch)
    return CuteKernelLaunch(
        name="normalized_exp_contract_forward",
        kernel_symbol=kernel_name,
        grid=grid,
        block=(schedule.threads, 1, 1),
        dynamic_shared_memory_bytes=shared_memory,
        function_attributes=(
            CuteKernelFunctionAttribute(
                CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
                shared_memory,
            ),
        ),
        source_task_roles=source_task_roles,
        arguments=arguments,
    )


def _launch_recipe(
    compiled: CompiledH100NormalizedExponentialBackward,
    kernel_names: tuple[str, ...],
    *,
    source_slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
    source_slot_bindings: Mapping[str, int] | None = None,
    source_slot_ordinals: Mapping[str, int] | None = None,
    source_slot_reuse_groups: Mapping[str, int] | None = None,
    reverse_output_cotangent_slot: str = "output_cotangent",
    reverse_output_cotangent_stride: tuple[int, ...] | None = None,
    reverse_output_cotangent_residual_slot: str | None = None,
    reverse_output_cotangent_residual_stride: tuple[int, ...] | None = None,
) -> CutePipelineLaunchRecipe:
    generic_finalizer = compiled.normalized_output_cotangent is not None
    layout_transport = compiled.layout_transport_output is not None
    streamed_gradients_are_direct = compiled.streamed_gradients_are_direct
    has_device_queues = compiled.query_accumulation_semaphore is not None
    accumulation_plan = compiled.template.lowering.accumulation_plan
    if accumulation_plan is not None:
        validate_ordered_backward_variant(
            relation_group_size=compiled.template.lowering.relation_layout.relation_group_size,
            layout_transport=layout_transport,
            combined=False,
        )
        ordered_binding = ordered_backward_kernel_binding(kernel_names, expanded_cotangent=generic_finalizer)
        if not has_device_queues or not streamed_gradients_are_direct:
            raise ValueError("ordered recipe requires its reset queue and direct streamed gradients")
    reverse_kernel_count = 3 if streamed_gradients_are_direct else 7
    expected_kernel_count = (
        reverse_kernel_count + int(has_device_queues) + int(layout_transport) + 2 * int(generic_finalizer)
    )
    if len(kernel_names) != expected_kernel_count:
        raise ValueError(
            f"the SM90 reverse pipeline must contain {expected_kernel_count} kernels, got {len(kernel_names)}"
        )
    if accumulation_plan is not None:
        semaphore_kernel_name = ordered_binding.reset
        layout_transport_kernel_name = None
        prime_map_kernel_name = ordered_binding.prime_vjp
        map_kernel_name = ordered_binding.vjp
        kernel_names = (ordered_binding.preprocess, ordered_binding.main, ordered_binding.postprocess)
    else:
        semaphore_kernel_name = kernel_names[0] if has_device_queues else None
        if has_device_queues:
            kernel_names = kernel_names[1:]
        layout_transport_kernel_name = kernel_names[0] if layout_transport else None
        if layout_transport:
            kernel_names = kernel_names[1:]
        prime_map_kernel_name = kernel_names[0] if generic_finalizer else None
        map_kernel_name = kernel_names[2] if generic_finalizer else None
        if map_kernel_name is not None:
            kernel_names = (kernel_names[1], *kernel_names[3:])

    program = compiled.template.program
    region = program.forward.region
    query, key = region.producer.inputs
    value = region.consumer.inputs[1]
    output = region.finalizer.output
    relation_layout = compiled.template.lowering.relation_layout
    batch = relation_layout.batch_extent
    query_sequence = relation_layout.owner_extent
    query_heads = relation_layout.flattened_relation_axis.extent
    query_dimension = relation_layout.producer_feature_extent
    key_sequence = relation_layout.streamed_extent
    key_value_heads = relation_layout.shared_relation_axis.extent
    value_dimension = relation_layout.consumer_feature_extent
    query_physical_shape = relation_layout.owner_physical_shape(query_dimension)
    key_physical_shape = relation_layout.streamed_physical_shape(query_dimension)
    value_physical_shape = relation_layout.streamed_physical_shape(value_dimension)
    output_physical_shape = relation_layout.owner_physical_shape(value_dimension)
    scale_log2 = math.log2(math.e)
    terminal_contract_epilogue_scale = compiled.template.lowering.terminal_contract_epilogue_scale
    physical_schedule = compiled.template.physical_schedule
    owner_tile = physical_schedule.contract.tile_m
    streamed_tile = physical_schedule.contract.tile_n
    epilogue_tile = physical_schedule.epilogue_tile_m
    threads = physical_schedule.epilogue_threads
    mainloop_shared_memory = physical_schedule.mainloop_dynamic_shared_memory_bytes
    epilogue_shared_memory = physical_schedule.epilogue_dynamic_shared_memory_bytes
    source_shapes = source_slot_shapes or {}
    source_bindings = source_slot_bindings or {}
    source_reuse_groups = source_slot_reuse_groups or {}
    private_reuse_group = max(source_reuse_groups.values(), default=10) + 1

    slots = (
        _slot(
            "query",
            CutePipelineElementType.BF16,
            source_shapes.get("query", query.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("query", 0),
        ),
        _slot(
            "key",
            CutePipelineElementType.BF16,
            source_shapes.get("key", key.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("key", 1),
        ),
        _slot(
            "value",
            CutePipelineElementType.BF16,
            source_shapes.get("value", value.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("value", 2),
        ),
        _slot(
            "output",
            CutePipelineElementType.BF16,
            source_shapes.get("output", output.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("output", 3),
        ),
        _slot(
            "output_cotangent",
            (
                CutePipelineElementType.BF16
                if getattr(compiled, "output_cotangent_dtype", DType.BF16) is DType.BF16
                else CutePipelineElementType.F32
            ),
            source_shapes.get("output_cotangent", output.shape),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("output_cotangent", 4),
        ),
        _slot(
            "log_sum_exp",
            CutePipelineElementType.F32,
            source_shapes.get("log_sum_exp", (*compiled.log_sum_exp_log2.shape[:-1], query_sequence)),
            CutePipelineSlotKind.PUBLIC_INPUT,
            input_index=source_bindings.get("log_sum_exp", 5),
        ),
        *(
            ()
            if not generic_finalizer
            else (
                _slot(
                    "normalized_output",
                    CutePipelineElementType.F32,
                    source_shapes.get("normalized_output", output.shape),
                    CutePipelineSlotKind.PUBLIC_INPUT,
                    input_index=source_bindings.get("normalized_output", 6),
                ),
            )
        ),
        _slot(
            "query_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("query_cotangent", query.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("query_cotangent", 0),
        ),
        _slot(
            "key_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("key_cotangent", key.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("key_cotangent", 1),
        ),
        _slot(
            "value_cotangent",
            CutePipelineElementType.BF16,
            source_shapes.get("value_cotangent", value.shape),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            output_index=source_bindings.get("value_cotangent", 2),
        ),
        _slot(
            "output_dot",
            CutePipelineElementType.F32,
            tuple(compiled.output_dot.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=source_reuse_groups.get("output_dot"),
        ),
        _slot(
            "log_sum_exp_log2",
            CutePipelineElementType.F32,
            tuple(compiled.log_sum_exp_log2.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=private_reuse_group,
        ),
        _slot(
            "query_accumulator",
            CutePipelineElementType.F32,
            tuple(compiled.query_accumulator.shape),
            CutePipelineSlotKind.SCRATCH,
            reuse_group=private_reuse_group + 1,
        ),
        *(
            ()
            if streamed_gradients_are_direct
            else (
                _slot(
                    "key_accumulator",
                    CutePipelineElementType.F32,
                    tuple(compiled.key_accumulator.shape),
                    CutePipelineSlotKind.SCRATCH,
                    reuse_group=private_reuse_group + 2,
                ),
                _slot(
                    "value_accumulator",
                    CutePipelineElementType.F32,
                    tuple(compiled.value_accumulator.shape),
                    CutePipelineSlotKind.SCRATCH,
                    reuse_group=private_reuse_group + 3,
                ),
            )
        ),
    )
    for name, tensor in (
        ("query_accumulation_semaphore", compiled.query_accumulation_semaphore),
        ("key_accumulation_semaphore", compiled.key_accumulation_semaphore),
        ("value_accumulation_semaphore", compiled.value_accumulation_semaphore),
    ):
        if tensor is not None:
            slots = (
                *slots,
                _slot(
                    name,
                    CutePipelineElementType.I32,
                    tuple(tensor.shape),
                    CutePipelineSlotKind.SCRATCH,
                ),
            )
    if reverse_output_cotangent_slot != "output_cotangent":
        slots = (
            *slots,
            _slot(
                reverse_output_cotangent_slot,
                CutePipelineElementType.BF16,
                output_physical_shape,
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 4,
            ),
        )
    if reverse_output_cotangent_residual_slot is not None:
        slots = (
            *slots,
            _slot(
                reverse_output_cotangent_residual_slot,
                CutePipelineElementType.BF16,
                output_physical_shape,
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 5,
            ),
        )
    if layout_transport:
        slots = (
            *slots,
            _slot(
                "layout_transport_output",
                CutePipelineElementType.BF16,
                output_physical_shape,
                CutePipelineSlotKind.SCRATCH,
                reuse_group=private_reuse_group + 6,
            ),
        )
    source_slot_count = 10 if generic_finalizer else 9
    slots = _order_source_slots(slots, source_slot_count, source_slot_ordinals)

    scheduler = _single_tile_scheduler_argument("scheduler")
    predicate = _predicate_argument("predicate", False)
    preprocess_grid = (_tile_count(query_sequence, epilogue_tile), query_heads, batch)
    main_grid = (
        accumulation_plan.grid
        if accumulation_plan is not None
        else (_tile_count(key_sequence, streamed_tile), query_heads, batch)
    )
    scheduler_arguments = (
        accumulation_plan.scheduler_arguments
        if accumulation_plan is not None
        else (
            _i32_argument("key_blocks", main_grid[0]),
            _i32_argument("query_heads", query_heads),
            _i32_argument("batch", batch),
            scheduler,
        )
    )
    query_postprocess_grid = (_tile_count(query_sequence, epilogue_tile), query_heads, batch)
    key_postprocess_grid = (_tile_count(key_sequence, epilogue_tile), key_value_heads, batch)

    output_transposed = _transposed_tensor_argument(
        "output",
        "output",
        CutePipelineSlotAccess.READ,
        output_physical_shape,
        (1, 3, 2, 0),
        logical_strides=compiled.output_stride,
    )
    output_cotangent_transposed = _transposed_tensor_argument(
        "output_cotangent",
        "output_cotangent",
        CutePipelineSlotAccess.READ,
        output_physical_shape,
        (1, 3, 2, 0),
        logical_strides=compiled.output_cotangent_stride,
    )
    output_dot_transposed = _transposed_tensor_argument(
        "output_dot",
        "output_dot",
        CutePipelineSlotAccess.WRITE,
        tuple(compiled.output_dot.shape),
        (2, 1, 0),
    )
    log_sum_exp_transposed = _transposed_tensor_argument(
        "log_sum_exp",
        "log_sum_exp",
        CutePipelineSlotAccess.READ,
        (batch, query_heads, query_sequence),
        (2, 1, 0),
    )
    log_sum_exp_log2_transposed = _transposed_tensor_argument(
        "log_sum_exp_log2",
        "log_sum_exp_log2",
        CutePipelineSlotAccess.WRITE,
        tuple(compiled.log_sum_exp_log2.shape),
        (2, 1, 0),
    )
    query_accumulator_transposed = _transposed_tensor_argument(
        "query_accumulator",
        "query_accumulator",
        CutePipelineSlotAccess.WRITE,
        tuple(compiled.query_accumulator.shape),
        (2, 1, 0),
    )
    preprocess_output = output_transposed
    preprocess_output_cotangent = output_cotangent_transposed
    if layout_transport:
        preprocess_output = _transposed_tensor_argument(
            "layout_transport_output",
            "layout_transport_output",
            CutePipelineSlotAccess.READ,
            output_physical_shape,
            (1, 3, 2, 0),
        )
        preprocess_output_cotangent = _transposed_tensor_argument(
            "layout_transport_output_cotangent",
            "layout_transport_output_cotangent",
            CutePipelineSlotAccess.READ,
            output_physical_shape,
            (1, 3, 2, 0),
        )

    launches = (
        *(
            ()
            if semaphore_kernel_name is None
            else (
                _semaphore_zero_fill_launch(
                    semaphore_kernel_name,
                    compiled.query_accumulation_semaphore,
                    compiled.key_accumulation_semaphore,
                    compiled.value_accumulation_semaphore,
                    accumulation_plan.reset_block[0] if accumulation_plan is not None else threads,
                ),
            )
        ),
        *(
            ()
            if layout_transport_kernel_name is None
            else (
                CuteKernelLaunch(
                    name="affine_output_layout_transport",
                    kernel_symbol=layout_transport_kernel_name,
                    grid=(query_sequence, query_heads, batch),
                    block=(32, 1, 1),
                    dynamic_shared_memory_bytes=0,
                    function_attributes=(),
                    source_task_roles=(CuteSourceTaskRole.STATE_CARRY,),
                    arguments=(
                        _tensor_argument(
                            "source_output",
                            "output",
                            CutePipelineSlotAccess.READ,
                            output_physical_shape,
                            compiled.output_stride,
                        ),
                        _tensor_argument(
                            "source_output_cotangent",
                            "output_cotangent",
                            CutePipelineSlotAccess.READ,
                            output_physical_shape,
                            compiled.output_cotangent_stride,
                        ),
                        _tensor_argument(
                            "output",
                            "layout_transport_output",
                            CutePipelineSlotAccess.WRITE,
                            output_physical_shape,
                            _contiguous_strides(output_physical_shape),
                        ),
                        _tensor_argument(
                            "output_cotangent",
                            "layout_transport_output_cotangent",
                            CutePipelineSlotAccess.WRITE,
                            output_physical_shape,
                            _contiguous_strides(output_physical_shape),
                        ),
                        _tensor_argument(
                            "output_dot",
                            "output_dot",
                            CutePipelineSlotAccess.WRITE,
                            tuple(compiled.output_dot.shape),
                            _contiguous_strides(tuple(compiled.output_dot.shape)),
                        ),
                        _tensor_argument(
                            "log_sum_exp",
                            "log_sum_exp",
                            CutePipelineSlotAccess.READ,
                            (batch, query_heads, query_sequence),
                            _contiguous_strides((batch, query_heads, query_sequence)),
                        ),
                        _tensor_argument(
                            "log_sum_exp_log2",
                            "log_sum_exp_log2",
                            CutePipelineSlotAccess.WRITE,
                            tuple(compiled.log_sum_exp_log2.shape),
                            _contiguous_strides(tuple(compiled.log_sum_exp_log2.shape)),
                        ),
                    ),
                ),
            )
        ),
        *(
            (
                _zero_fill_launch(
                    "zero_query_accumulator",
                    kernel_names[0],
                    "query_accumulator",
                    tuple(compiled.query_accumulator.shape),
                    threads,
                    source_task_roles=(CuteSourceTaskRole.STATE_CARRY,),
                ),
            )
            if layout_transport
            else (
                CuteKernelLaunch(
                    name="preprocess",
                    kernel_symbol=kernel_names[0],
                    grid=preprocess_grid,
                    block=(threads, 1, 1),
                    dynamic_shared_memory_bytes=0,
                    function_attributes=(),
                    source_task_roles=(CuteSourceTaskRole.STATE_CARRY,),
                    arguments=(
                        preprocess_output,
                        preprocess_output_cotangent,
                        output_dot_transposed,
                        log_sum_exp_transposed,
                        log_sum_exp_log2_transposed,
                        query_accumulator_transposed,
                        _scalar_argument("scale_log2", scale_log2),
                        _i32_argument("query_blocks", preprocess_grid[0]),
                        _i32_argument("query_heads", query_heads),
                        _i32_argument("batch", batch),
                        scheduler,
                    ),
                ),
            )
        ),
        *(
            ()
            if map_kernel_name is None
            else (
                _post_normalization_vjp_launch(
                    compiled,
                    map_kernel_name,
                    output_physical_shape,
                    relation_layout,
                ),
            )
        ),
        *(
            ()
            if streamed_gradients_are_direct
            else (
                _zero_fill_launch(
                    "zero_key_accumulator",
                    kernel_names[1],
                    "key_accumulator",
                    tuple(compiled.key_accumulator.shape),
                    threads,
                    source_task_roles=(CuteSourceTaskRole.STATE_TERMINAL,),
                ),
                _zero_fill_launch(
                    "zero_value_accumulator",
                    kernel_names[2],
                    "value_accumulator",
                    tuple(compiled.value_accumulator.shape),
                    threads,
                    source_task_roles=(CuteSourceTaskRole.STATE_TERMINAL,),
                ),
            )
        ),
        CuteKernelLaunch(
            name="normalized_exp_contract_pullback",
            kernel_symbol=kernel_names[1] if streamed_gradients_are_direct else kernel_names[3],
            grid=main_grid,
            block=(compiled.template.physical_schedule.contract.threads, 1, 1),
            dynamic_shared_memory_bytes=mainloop_shared_memory,
            function_attributes=(
                CuteKernelFunctionAttribute(
                    CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
                    mainloop_shared_memory,
                ),
            ),
            source_task_roles=(CuteSourceTaskRole.STATE_CARRY, CuteSourceTaskRole.STATE_TERMINAL),
            arguments=(
                _coordinate_argument("query_coordinates", (query_sequence, query_dimension, query_heads, batch)),
                _coordinate_argument("key_coordinates", (key_sequence, query_dimension, key_value_heads, batch)),
                _coordinate_argument("value_coordinates", (key_sequence, value_dimension, key_value_heads, batch)),
                _coordinate_argument(
                    "output_cotangent_coordinates", (query_sequence, value_dimension, query_heads, batch)
                ),
                *(
                    ()
                    if reverse_output_cotangent_residual_slot is None
                    else (
                        _coordinate_argument(
                            "output_cotangent_residual_coordinates",
                            (query_sequence, value_dimension, query_heads, batch),
                        ),
                    )
                ),
                *(
                    (
                        _coordinate_argument(
                            "key_cotangent_coordinates", (key_sequence, query_dimension, key_value_heads, batch)
                        ),
                        _coordinate_argument(
                            "value_cotangent_coordinates", (key_sequence, value_dimension, key_value_heads, batch)
                        ),
                    )
                    if streamed_gradients_are_direct
                    else (
                        _transposed_tensor_argument(
                            "key_accumulator",
                            "key_accumulator",
                            CutePipelineSlotAccess.READ_WRITE,
                            tuple(compiled.key_accumulator.shape),
                            (2, 1, 0),
                        ),
                        _transposed_tensor_argument(
                            "value_accumulator",
                            "value_accumulator",
                            CutePipelineSlotAccess.READ_WRITE,
                            tuple(compiled.value_accumulator.shape),
                            (2, 1, 0),
                        ),
                    )
                ),
                sm90_tensor_map_argument(
                    "query_tma",
                    "query",
                    query_physical_shape,
                    (owner_tile, owner_tile, 1, 1),
                    CutePipelineSlotAccess.READ,
                    logical_strides=compiled.query_stride,
                ),
                sm90_tensor_map_argument(
                    "key_tma",
                    "key",
                    key_physical_shape,
                    (owner_tile, streamed_tile, 1, 1),
                    CutePipelineSlotAccess.READ,
                    logical_strides=compiled.key_stride,
                ),
                sm90_tensor_map_argument(
                    "value_tma",
                    "value",
                    value_physical_shape,
                    (owner_tile, streamed_tile, 1, 1),
                    CutePipelineSlotAccess.READ,
                    logical_strides=compiled.value_stride,
                ),
                sm90_tensor_map_argument(
                    "output_cotangent_tma",
                    reverse_output_cotangent_slot,
                    output_physical_shape,
                    (owner_tile, owner_tile, 1, 1),
                    CutePipelineSlotAccess.READ,
                    logical_strides=(
                        compiled.output_cotangent_stride
                        if reverse_output_cotangent_stride is None
                        else reverse_output_cotangent_stride
                    ),
                ),
                *(
                    ()
                    if reverse_output_cotangent_residual_slot is None
                    else (
                        sm90_tensor_map_argument(
                            "output_cotangent_residual_tma",
                            reverse_output_cotangent_residual_slot,
                            output_physical_shape,
                            (owner_tile, owner_tile, 1, 1),
                            CutePipelineSlotAccess.READ,
                            logical_strides=reverse_output_cotangent_residual_stride,
                        ),
                    )
                ),
                *(
                    (
                        sm90_tensor_map_argument(
                            "key_cotangent_tma",
                            "key_cotangent",
                            key_physical_shape,
                            (owner_tile, streamed_tile, 1, 1),
                            CutePipelineSlotAccess.WRITE,
                            logical_strides=compiled.key_cotangent_stride,
                        ),
                        sm90_tensor_map_argument(
                            "value_cotangent_tma",
                            "value_cotangent",
                            value_physical_shape,
                            (owner_tile, streamed_tile, 1, 1),
                            CutePipelineSlotAccess.WRITE,
                            logical_strides=compiled.value_cotangent_stride,
                        ),
                    )
                    if streamed_gradients_are_direct
                    else ()
                ),
                _transposed_tensor_argument(
                    "log_sum_exp",
                    "log_sum_exp_log2",
                    CutePipelineSlotAccess.READ,
                    tuple(compiled.log_sum_exp_log2.shape),
                    (2, 1, 0),
                ),
                _transposed_tensor_argument(
                    "output_dot",
                    "output_dot",
                    CutePipelineSlotAccess.READ,
                    tuple(compiled.output_dot.shape),
                    (2, 1, 0),
                ),
                _transposed_tensor_argument(
                    "query_accumulator",
                    "query_accumulator",
                    CutePipelineSlotAccess.READ_WRITE,
                    tuple(compiled.query_accumulator.shape),
                    (2, 1, 0),
                ),
                *(
                    ()
                    if compiled.query_accumulation_semaphore is None
                    else (
                        _transposed_tensor_argument(
                            "query_accumulation_semaphore",
                            "query_accumulation_semaphore",
                            CutePipelineSlotAccess.READ_WRITE,
                            tuple(compiled.query_accumulation_semaphore.shape),
                            (2, 3, 1, 0),
                        ),
                        *(
                            ()
                            if compiled.key_accumulation_semaphore is None
                            else (
                                _transposed_tensor_argument(
                                    "key_accumulation_semaphore",
                                    "key_accumulation_semaphore",
                                    CutePipelineSlotAccess.READ_WRITE,
                                    tuple(compiled.key_accumulation_semaphore.shape),
                                    (2, 3, 1, 0),
                                ),
                                _transposed_tensor_argument(
                                    "value_accumulation_semaphore",
                                    "value_accumulation_semaphore",
                                    CutePipelineSlotAccess.READ_WRITE,
                                    tuple(compiled.value_accumulation_semaphore.shape),
                                    (2, 3, 1, 0),
                                ),
                            )
                        ),
                    )
                ),
                predicate,
                _predicate_argument("has_query_sequence_lengths", False),
                _predicate_argument("has_key_sequence_lengths", False),
                _predicate_argument("has_auxiliary_tensors", False),
                _scalar_argument("scale_log2", scale_log2),
                *scheduler_arguments,
                *(
                    (
                        _fast_divmod_argument(
                            "query_head_to_key_value_head",
                            relation_layout.relation_group_size,
                        ),
                    )
                    if relation_layout.relation_group_size > 1
                    else ()
                ),
            ),
        ),
        _postprocess_launch(
            "query_postprocess",
            kernel_names[2] if streamed_gradients_are_direct else kernel_names[4],
            "query_accumulator",
            tuple(compiled.query_accumulator.shape),
            "query_cotangent",
            query_physical_shape,
            terminal_contract_epilogue_scale,
            query_postprocess_grid,
            threads,
            epilogue_shared_memory,
            source_task_roles=(CuteSourceTaskRole.STATE_CARRY,),
            output_strides=compiled.query_cotangent_stride,
        ),
        *(
            ()
            if streamed_gradients_are_direct
            else (
                _postprocess_launch(
                    "key_postprocess",
                    kernel_names[5],
                    "key_accumulator",
                    tuple(compiled.key_accumulator.shape),
                    "key_cotangent",
                    key_physical_shape,
                    terminal_contract_epilogue_scale,
                    key_postprocess_grid,
                    threads,
                    epilogue_shared_memory,
                    source_task_roles=(CuteSourceTaskRole.STATE_TERMINAL,),
                    output_strides=compiled.key_cotangent_stride,
                ),
                _postprocess_launch(
                    "value_postprocess",
                    kernel_names[6],
                    "value_accumulator",
                    tuple(compiled.value_accumulator.shape),
                    "value_cotangent",
                    value_physical_shape,
                    1.0,
                    key_postprocess_grid,
                    threads,
                    epilogue_shared_memory,
                    source_task_roles=(CuteSourceTaskRole.STATE_TERMINAL,),
                    output_strides=compiled.value_cotangent_stride,
                ),
            )
        ),
    )
    if map_kernel_name is not None:
        preprocess_index = int(semaphore_kernel_name is not None)
        assert prime_map_kernel_name is not None
        map_launch = _post_normalization_vjp_launch(
            compiled,
            prime_map_kernel_name,
            output_physical_shape,
            relation_layout,
        )
        launches = (
            *launches[:preprocess_index],
            replace(map_launch, name="prime_post_normalization_vjp"),
            *launches[preprocess_index:],
        )
    recipe = CutePipelineLaunchRecipe(
        schema_version=2,
        slots=slots,
        launches=launches,
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_CARRY,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint(
                    "affine_output_layout_transport" if layout_transport else "preprocess",
                    0,
                ),
                CuteEventExecutionPoint("normalized_exp_contract_pullback", 0),
            ),
        ),
    )

    if accumulation_plan is not None:
        validate_ordered_backward_recipe(accumulation_plan, recipe, expanded_cotangent=generic_finalizer)
    return recipe


def _slot(
    name: str,
    element_type: CutePipelineElementType,
    shape: tuple[int, ...],
    kind: CutePipelineSlotKind,
    *,
    input_index: int | None = None,
    output_index: int | None = None,
    reuse_group: int | None = None,
) -> CutePipelineSlot:
    return CutePipelineSlot(name, element_type, tuple(shape), kind, 16, input_index, output_index, reuse_group)


def _tile_count(extent: int, tile: int) -> int:
    return (extent + tile - 1) // tile


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _unique_unit_stride_dimension(tensor: torch.Tensor) -> int:
    if tensor.stride(-1) == 1:
        return tensor.ndim - 1
    dimensions = tuple(
        dimension
        for dimension, (extent, stride) in enumerate(zip(tensor.shape, tensor.stride(), strict=True))
        if extent > 1 and stride == 1
    )
    if len(dimensions) != 1:
        raise ValueError(
            "affine layout transport requires one nontrivial unit-stride dimension; "
            f"found shape {tuple(tensor.shape)} and stride {tensor.stride()}"
        )
    return dimensions[0]


def _transposed_tensor_argument(
    name: str,
    slot: str,
    access: CutePipelineSlotAccess,
    shape: tuple[int, ...],
    order: tuple[int, ...],
    *,
    logical_strides: tuple[int, ...] | None = None,
) -> CutePipelineStructuredArgument:
    strides = logical_strides or _contiguous_strides(shape)
    if len(strides) != len(shape):
        raise ValueError(f"CuTe tensor argument {name!r} stride rank does not match its shape")
    transposed_shape = tuple(shape[index] for index in order)
    transposed_strides = tuple(strides[index] for index in order)
    return _tensor_argument(name, slot, access, transposed_shape, transposed_strides)


def _tensor_argument(
    name: str,
    slot: str,
    access: CutePipelineSlotAccess,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> CutePipelineStructuredArgument:
    leading_dimensions = tuple(
        index for index, (extent, stride) in enumerate(zip(shape, strides, strict=True)) if extent > 1 and stride == 1
    )
    if len(leading_dimensions) != 1:
        raise ValueError(f"CuTe tensor argument {name!r} must have exactly one nontrivial unit stride")
    dynamic_strides = tuple(stride for index, stride in enumerate(strides) if index != leading_dimensions[0])
    stride_offset = (8 + 4 * len(shape) + 7) // 8 * 8
    size = stride_offset + 8 * len(dynamic_strides)
    return CutePipelineStructuredArgument(
        name=name,
        size=size,
        alignment=8,
        fields=(
            CutePipelineStructuredSlotAddressField("data", slot, access, slot_byte_offset=0, destination_byte_offset=0),
            CutePipelineStructuredImmediateField("shape", 8, struct.pack(f"<{len(shape)}i", *shape)),
            CutePipelineStructuredImmediateField(
                "strides", stride_offset, struct.pack(f"<{len(dynamic_strides)}q", *dynamic_strides)
            ),
        ),
    )


def _packed_relation_tensor_argument(
    name: str,
    slot: str,
    access: CutePipelineSlotAccess,
    original_shape: tuple[int, ...],
    group_size: int,
    *,
    logical_strides: tuple[int, ...] | None = None,
) -> CutePipelineStructuredArgument:
    batch, sequence, heads, dimension = original_shape
    if heads % group_size:
        raise ValueError(f"packed relation tensor argument {name!r} requires a divisible head extent")
    strides = logical_strides or _contiguous_strides(original_shape)
    if len(strides) != 4 or strides[3] != 1:
        raise ValueError(f"packed relation tensor argument {name!r} requires a rank-four unit-minor layout")
    batch_stride, sequence_stride, head_stride, _ = strides
    shape_fields = (sequence, dimension, heads // group_size, batch)
    stride_fields = (head_stride, sequence_stride, head_stride * group_size, batch_stride)
    return CutePipelineStructuredArgument(
        name=name,
        size=56,
        alignment=8,
        fields=(
            CutePipelineStructuredSlotAddressField("data", slot, access, slot_byte_offset=0, destination_byte_offset=0),
            CutePipelineStructuredImmediateField("shape", 8, struct.pack("<4i", *shape_fields)),
            CutePipelineStructuredImmediateField("strides", 24, struct.pack("<4q", *stride_fields)),
        ),
    )


def _coordinate_argument(name: str, shape: tuple[int, int, int, int]) -> CutePipelinePackedBytesArgument:
    return CutePipelinePackedBytesArgument(name, struct.pack("<4i", *shape), 4)


def _scalar_argument(name: str, value: float) -> CutePipelineScalarArgument:
    return CutePipelineScalarArgument(name, CutePipelineElementType.F32, value)


def _i32_argument(name: str, value: int) -> CutePipelineScalarArgument:
    return CutePipelineScalarArgument(name, CutePipelineElementType.I32, value)


def _predicate_argument(name: str, value: bool) -> CutePipelinePackedBytesArgument:
    return CutePipelinePackedBytesArgument(name, struct.pack("<?", value), 1)


def _single_tile_scheduler_argument(name: str) -> CutePipelinePackedBytesArgument:
    return CutePipelinePackedBytesArgument(name, struct.pack("<iiBBxx", 1, 1, 0, 0), 4)


def _fast_divmod_argument(name: str, divisor: int) -> CutePipelinePackedBytesArgument:
    return CutePipelinePackedBytesArgument(
        name,
        encode_cute_fast_divmod_divisor(divisor),
        4,
    )


def _floor_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two floor requires a positive value")
    return 1 << (value.bit_length() - 1)


def _forward_dynamic_shared_memory_bytes(
    *,
    owner_tile: int,
    streamed_tile: int,
    stages: int,
    query_dimension: int,
    value_dimension: int,
) -> int:
    supported = {
        (64, 128, 2, 128, 128): 148_480,
        (128, 128, 2, 128, 128): 164_864,
    }
    key = (owner_tile, streamed_tile, stages, query_dimension, value_dimension)
    if key not in supported:
        raise ValueError(f"the combined SM90 artifact has no verified shared-memory boundary for {key}")
    return supported[key]


def _zero_fill_launch(
    name: str,
    kernel_symbol: str,
    slot: str,
    shape: tuple[int, ...],
    threads: int,
    *,
    source_task_roles: tuple[CuteSourceTaskRole, ...],
) -> CuteKernelLaunch:
    elements = math.prod(shape)
    return CuteKernelLaunch(
        name=name,
        kernel_symbol=kernel_symbol,
        grid=((elements + threads - 1) // threads, 1, 1),
        block=(threads, 1, 1),
        dynamic_shared_memory_bytes=0,
        function_attributes=(),
        source_task_roles=source_task_roles,
        arguments=(
            _tensor_argument(
                name="tensor",
                slot=slot,
                access=CutePipelineSlotAccess.WRITE,
                shape=shape,
                strides=_contiguous_strides(shape),
            ),
        ),
    )


def _semaphore_zero_fill_launch(
    kernel_symbol: str,
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    threads: int,
) -> CuteKernelLaunch:
    if query is None:
        raise ValueError("deterministic semaphore zero-fill requires the query-gradient Event queue")
    tensors = (
        ("query", "query_accumulation_semaphore", query),
        ("key", "key_accumulation_semaphore", key),
        ("value", "value_accumulation_semaphore", value),
    )
    element_count = sum(tensor.numel() for _, _, tensor in tensors if tensor is not None)
    return CuteKernelLaunch(
        name="initialize_deterministic_accumulation_queues",
        kernel_symbol=kernel_symbol,
        grid=((element_count + threads - 1) // threads, 1, 1),
        block=(threads, 1, 1),
        dynamic_shared_memory_bytes=0,
        function_attributes=(),
        source_task_roles=(),
        arguments=tuple(
            _tensor_argument(
                name=name,
                slot=slot,
                access=CutePipelineSlotAccess.WRITE,
                shape=tuple(tensor.shape),
                strides=_contiguous_strides(tuple(tensor.shape)),
            )
            for name, slot, tensor in tensors
            if tensor is not None
        ),
    )


def _postprocess_launch(
    name: str,
    kernel_symbol: str,
    accumulator_slot: str,
    accumulator_shape: tuple[int, ...],
    output_slot: str,
    output_shape: tuple[int, ...],
    scale: float,
    grid: tuple[int, int, int],
    threads: int,
    dynamic_shared_memory_bytes: int,
    *,
    source_task_roles: tuple[CuteSourceTaskRole, ...],
    output_strides: tuple[int, ...] | None = None,
) -> CuteKernelLaunch:
    return CuteKernelLaunch(
        name=name,
        kernel_symbol=kernel_symbol,
        grid=grid,
        block=(threads, 1, 1),
        dynamic_shared_memory_bytes=dynamic_shared_memory_bytes,
        function_attributes=(),
        source_task_roles=source_task_roles,
        arguments=(
            _tensor_argument(
                "accumulator",
                accumulator_slot,
                CutePipelineSlotAccess.READ,
                accumulator_shape,
                _contiguous_strides(accumulator_shape),
            ),
            _tensor_argument(
                "output",
                output_slot,
                CutePipelineSlotAccess.WRITE,
                output_shape,
                output_strides or _contiguous_strides(output_shape),
            ),
            _scalar_argument("scale", scale),
            _predicate_argument("has_sequence_lengths", False),
            _i32_argument("blocks", grid[0]),
            _i32_argument("heads", grid[1]),
            _i32_argument("batch", grid[2]),
            _single_tile_scheduler_argument("scheduler"),
        ),
    )


def _validate_runtime_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_cotangent: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> None:
    if any(tensor.dtype is not torch.bfloat16 for tensor in (query, key, value, output)):
        raise ValueError("the SM90 reverse template requires BF16 Q/K/V/O")
    if output_cotangent.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("the SM90 reverse template requires BF16 or FP32 output cotangent")
    if any(not tensor.is_cuda for tensor in (query, key, value, output, output_cotangent, log_sum_exp)):
        raise ValueError("the SM90 reverse template requires CUDA-resident tensors")
    if output.shape != output_cotangent.shape:
        raise ValueError("final output and its cotangent must have the same shape")
    if query.shape[:3] != output.shape[:3]:
        raise ValueError("producer-left and final-output owner dimensions must match")
    if key.shape[:3] != value.shape[:3] or query.shape[0] != key.shape[0]:
        raise ValueError("producer-right and consumer-operand relation dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("producer Contract feature dimensions must match")
    if output.shape[-1] != value.shape[-1]:
        raise ValueError("consumer Contract feature dimensions must match")
    if query.shape[-1] != 128:
        raise ValueError("the current SM90 reverse template requires a producer feature dimension of 128")
    if value.shape[-1] != 128:
        raise ValueError("the current SM90 reverse template requires a consumer feature dimension of 128")
    if log_sum_exp.shape != (query.shape[0], query.shape[2], query.shape[1]):
        raise ValueError("log_sum_exp must have shape [batch, query_heads, query_sequence]")
    if log_sum_exp.dtype is not torch.float32:
        raise ValueError("log_sum_exp must be FP32")


def _compile_request(template: H100NormalizedExponentialBackwardTemplate) -> CutePipelineCompileRequest:
    return template.lowering.compile_request


def _compiled_backward_request(
    compiled: CompiledH100NormalizedExponentialBackward,
) -> CutePipelineCompileRequest:
    request = _compile_request(compiled.template)
    if compiled.layout_transport_output is None:
        return request
    assert compiled.layout_transport_output_cotangent is not None
    transport = {
        "realization": "declared_affine_to_feature_minor_private_slot",
        "source_output_strides": compiled.output_stride,
        "source_output_cotangent_strides": compiled.output_cotangent_stride,
        "destination_output_strides": tuple(compiled.layout_transport_output.stride()),
        "destination_output_cotangent_strides": tuple(compiled.layout_transport_output_cotangent.stride()),
        "fold": "output_dot",
        "event_edge": ("affine_output_layout_transport", "normalized_exp_contract_pullback"),
    }
    return replace(request, schedule={**request.schedule, "output_layout_transport": transport})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
