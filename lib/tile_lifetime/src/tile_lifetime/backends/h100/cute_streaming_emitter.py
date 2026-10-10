# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instantiate the SM90 streaming contraction/fold skeleton from tensor semantics.

This optional backend module deliberately lives outside the typed core package.
It depends on CUDA, CuTe DSL, QuACK, and a small set of physical helper
primitives extracted from the pinned FlashAttention CuTe implementation.  It
does not call a FlashAttention interface or select a kernel by attention name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from tile_lifetime.cute_device_ir import CuteDeviceIrSnapshot, compile_cute_with_structured_snapshot
from tile_lifetime.event_dataflow_adapters import streaming_contract_fold_event_descriptor
from tile_lifetime.h100_scalar_ast import (
    FinalizerStorageConversion,
    finalizer_storage_conversion,
    normalized_finalizer_expression,
    score_input_roles,
)
from tile_lifetime.h100_streaming_lowering import (
    H100StreamingSchedule,
    LoweredScoreMap,
    Sm90ForwardPipelineSchedule,
    Sm90RelationPackedLayout,
    lower_h100_streaming_program,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractProgram,
)
from tile_lifetime.streaming_event_schedule import (
    StreamingPhysicalEventSchedule,
    derive_streaming_physical_event_schedule,
    verify_streaming_event_backend_parameters,
)

from .cute_domain_restriction import cute_affine_index_predicate
from .cute_scalar_ast import cute_finalizer_map, cute_score_map
from .cute_streaming_sm90 import ShuttleNormalizedExponentialContractSm90


@dataclass
class CompiledH100StreamingProgram:
    """One CuTe executable and the semantic/runtime contract it implements."""

    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule]
    score_map: LoweredScoreMap
    schedule: H100StreamingSchedule
    event_schedule: StreamingPhysicalEventSchedule
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    query_stride: tuple[int, ...]
    key_stride: tuple[int, ...]
    value_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    normalized_output_stride: tuple[int, ...] | None
    relation_layout: Sm90RelationPackedLayout
    writes_log_sum_exp: bool
    writes_normalized_output: bool
    executable: Any
    structured_device_ir: CuteDeviceIrSnapshot
    finalizer_storage_conversion: FinalizerStorageConversion | None

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        log_sum_exp: torch.Tensor | None = None,
        normalized_output: torch.Tensor | None = None,
    ) -> None:
        _validate_runtime_tensor("query", query, self.query_shape, self.query_stride)
        _validate_runtime_tensor("key", key, self.key_shape, self.key_stride)
        _validate_runtime_tensor("value", value, self.value_shape, self.value_stride)
        _validate_runtime_tensor("output", output, (*self.query_shape[:-1], self.value_shape[-1]), self.output_stride)
        if self.writes_log_sum_exp:
            expected_lse = (self.query_shape[0], self.query_shape[2], self.query_shape[1])
            if log_sum_exp is None or tuple(log_sum_exp.shape) != expected_lse or log_sum_exp.dtype is not torch.float32:
                raise ValueError(f"log_sum_exp must be FP32 with shape {expected_lse}")
        elif log_sum_exp is not None:
            raise ValueError("this compiled forward does not materialize log_sum_exp")
        if self.writes_normalized_output:
            if (
                normalized_output is None
                or tuple(normalized_output.shape) != tuple(output.shape)
                or normalized_output.dtype is not torch.float32
                or not normalized_output.is_cuda
            ):
                raise ValueError("normalized_output must be CUDA FP32 with the physical output shape")
        elif normalized_output is not None:
            raise ValueError("this compiled forward does not materialize the normalized output")
        self.executable(
            query,
            key,
            value,
            output,
            log_sum_exp,
            *([None] * 10),
            normalized_output,
            None,
            None,
        )


@dataclass(frozen=True)
class H100StreamingTemplate:
    """Graph-derived forward kernel before concrete tensor compilation."""

    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule]
    score_map: LoweredScoreMap
    schedule: H100StreamingSchedule
    event_schedule: StreamingPhysicalEventSchedule
    relation_layout: Sm90RelationPackedLayout
    domain_traversal: ContractDomainTraversal
    kernel: Any
    finalizer_storage_conversion: FinalizerStorageConversion | None


def build_h100_streaming_template(
    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule],
    *,
    domain_traversal: ContractDomainTraversal | None = None,
    event_schedule: StreamingPhysicalEventSchedule | None = None,
    conversion_faithful_normalized_map: Any | None = None,
    conversion_faithful_score_map: Any | None = None,
    conversion_faithful_statistics_score_map: Any | None = None,
    conversion_faithful_statistics_sum_map: Any | None = None,
    conversion_faithful_statistics_lse_map: Any | None = None,
    finalizer_map_override: Any | None = None,
) -> H100StreamingTemplate:
    """Bind one generic Contract/Map/Fold region to the SM90 forward skeleton."""
    lowering = lower_h100_streaming_program(program)
    score_map = lowering.score_map
    schedule = lowering.schedule
    selected_event_schedule = event_schedule or derive_streaming_physical_event_schedule(
        streaming_contract_fold_event_descriptor(program)
    )
    resident_pipeline = selected_event_schedule.resident_input_pipeline
    first_streamed_pipeline = selected_event_schedule.first_streamed_input_pipeline
    second_streamed_pipeline = selected_event_schedule.second_streamed_input_pipeline
    if first_streamed_pipeline.capacity != schedule.stages or second_streamed_pipeline.capacity != schedule.stages:
        raise ValueError("Event Tensor buffer depth disagrees with the legalized streaming schedule")
    verify_streaming_event_backend_parameters(
        selected_event_schedule,
        resident_input_stages=resident_pipeline.capacity,
        first_streamed_input_stages=first_streamed_pipeline.capacity,
        second_streamed_input_stages=second_streamed_pipeline.capacity,
        barriers_per_stage=first_streamed_pipeline.physical_barriers_per_stage,
        transfer_warps=selected_event_schedule.workers.transfer_warps,
        matrix_warpgroups=selected_event_schedule.workers.matrix_warpgroups,
        scheduler_arrival_threads=selected_event_schedule.workers.scheduler_arrival_threads,
    )
    traversal = score_map.domain_traversal if domain_traversal is None else domain_traversal
    score_mod = cute_score_map(
        score_map.expression,
        score_input_roles(
            score_map.expression,
            score_input=score_map.score_input,
            owner_inputs=score_map.owner_inputs,
            streamed_inputs=score_map.streamed_inputs,
        ),
    )
    region = program.region
    domain_predicate = (
        None
        if region.domain_predicate is None
        else cute_affine_index_predicate(
            region.domain_predicate,
            owner_dimension=region.owner_axes[-1].id,
            streamed_dimension=region.dimensions.normalized_axes[0].id,
        )
    )
    finalizer = normalized_finalizer_expression(
        lowering.finalizer_expression,
        accumulator_input=region.consumer.output.name,
        normalizer_input=region.fold_state.normalizer.output.name,
    )
    storage_conversion = finalizer_storage_conversion(
        finalizer,
        destination_dtype=region.finalizer.output.dtype,
    )
    emitted_finalizer = finalizer if storage_conversion is None else storage_conversion.residual_expression
    emitted_finalizer_map = (
        cute_finalizer_map(emitted_finalizer) if finalizer_map_override is None else finalizer_map_override
    )
    query_shape = region.producer.inputs[0].shape
    value_shape = region.consumer.inputs[1].shape
    kernel = ShuttleNormalizedExponentialContractSm90(
        cutlass.BFloat16,
        query_shape[-1],
        value_shape[-1],
        lowering.relation_layout.relation_group_size,
        lower_triangular_domain=traversal is ContractDomainTraversal.LOWER_TRIANGULAR,
        is_local=False,
        pack_index_relation=True,
        tile_m=schedule.tile_m,
        tile_n=schedule.tile_n,
        num_stages=first_streamed_pipeline.capacity,
        num_threads=schedule.threads,
        Q_in_regs=schedule.resident_left_in_registers,
        score_mod=score_mod,
        mask_mod=domain_predicate,
        has_aux_tensors=False,
        output_scale=1.0,
        finalizer_map=emitted_finalizer_map,
        conversion_faithful_normalized_map=conversion_faithful_normalized_map,
        conversion_faithful_score_map=conversion_faithful_score_map,
        conversion_faithful_statistics_score_map=conversion_faithful_statistics_score_map,
        conversion_faithful_statistics_sum_map=conversion_faithful_statistics_sum_map,
        conversion_faithful_statistics_lse_map=conversion_faithful_statistics_lse_map,
        intra_wg_overlap=(
            schedule.intra_warpgroup_overlap
            if conversion_faithful_normalized_map is None and conversion_faithful_statistics_sum_map is None
            else False
        ),
        mma_pv_is_rs=schedule.consumer_register_source,
        query_pipeline_stages=resident_pipeline.capacity,
        key_pipeline_stages=first_streamed_pipeline.capacity,
        value_pipeline_stages=second_streamed_pipeline.capacity,
        pipeline_barriers_per_stage=first_streamed_pipeline.physical_barriers_per_stage,
        transfer_warps=selected_event_schedule.workers.transfer_warps,
        matrix_warpgroups=selected_event_schedule.workers.matrix_warpgroups,
        scheduler_arrival_threads=selected_event_schedule.workers.scheduler_arrival_threads,
        query_transaction_bytes=resident_pipeline.transaction_bytes,
        key_transaction_bytes=first_streamed_pipeline.transaction_bytes,
        value_transaction_bytes=second_streamed_pipeline.transaction_bytes,
        qk_reduction_traversal=schedule.producer_reduction_traversal.value,
        qk_operand_orientation=schedule.producer_operand_orientation.value,
    )
    return H100StreamingTemplate(
        program=program,
        score_map=score_map,
        schedule=schedule,
        event_schedule=selected_event_schedule,
        relation_layout=lowering.relation_layout,
        domain_traversal=traversal,
        kernel=kernel,
        finalizer_storage_conversion=storage_conversion,
    )


def compile_h100_streaming_program(
    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule],
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    log_sum_exp: torch.Tensor | None,
    normalized_output: torch.Tensor | None = None,
    domain_traversal: ContractDomainTraversal | None = None,
    event_schedule: StreamingPhysicalEventSchedule | None = None,
) -> CompiledH100StreamingProgram:
    """Compile generic QK/Map/Fold/PV semantics into the SM90 skeleton."""
    region = program.region
    query_value, key_value = region.producer.inputs
    value_value = region.consumer.inputs[1]
    template = build_h100_streaming_template(
        program,
        domain_traversal=domain_traversal,
        event_schedule=event_schedule,
    )
    relation_layout = template.relation_layout
    query_shape = relation_layout.owner_physical_shape(query_value.shape[-1])
    key_shape = relation_layout.streamed_physical_shape(key_value.shape[-1])
    value_shape = relation_layout.streamed_physical_shape(value_value.shape[-1])
    for name, tensor, shape in (
        ("query", query, query_shape),
        ("key", key, key_shape),
        ("value", value, value_shape),
        ("output", output, relation_layout.owner_physical_shape(value_value.shape[-1])),
    ):
        _validate_runtime_tensor(name, tensor, shape)

    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cute_arguments = [to_cute_tensor(tensor) for tensor in (query, key, value)]
    unit_stride_dimensions = tuple(index for index, stride in enumerate(output.stride()) if stride == 1)
    if len(unit_stride_dimensions) != 1:
        raise ValueError("normalized-exp output must have one unit-stride physical dimension")
    cute_arguments.append(to_cute_tensor(output, leading_dim=unit_stride_dimensions[0]))
    if log_sum_exp is not None:
        cute_arguments.append(to_cute_tensor(log_sum_exp, assumed_align=4))
    else:
        cute_arguments.append(None)
    cute_arguments.extend([None] * 10)
    if normalized_output is not None:
        if (
            tuple(normalized_output.shape) != tuple(output.shape)
            or normalized_output.dtype is not torch.float32
            or not normalized_output.is_cuda
        ):
            raise ValueError("normalized_output must be CUDA FP32 with the physical output shape")
        cute_arguments.append(to_cute_tensor(normalized_output, leading_dim=unit_stride_dimensions[0]))
    else:
        cute_arguments.append(None)
    cute_arguments.extend([None, None])
    cute_arguments.append(fake_stream)
    executable, structured_device_ir = compile_cute_with_structured_snapshot(
        cute,
        template.kernel,
        *cute_arguments,
        options="--enable-tvm-ffi",
    )
    return CompiledH100StreamingProgram(
        program=program,
        score_map=template.score_map,
        schedule=template.schedule,
        event_schedule=template.event_schedule,
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        query_stride=tuple(query.stride()),
        key_stride=tuple(key.stride()),
        value_stride=tuple(value.stride()),
        output_stride=tuple(output.stride()),
        normalized_output_stride=(tuple(normalized_output.stride()) if normalized_output is not None else None),
        relation_layout=template.relation_layout,
        writes_log_sum_exp=log_sum_exp is not None,
        writes_normalized_output=normalized_output is not None,
        executable=executable,
        structured_device_ir=structured_device_ir,
        finalizer_storage_conversion=template.finalizer_storage_conversion,
    )


def _validate_runtime_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError(f"{name} must be BF16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA-resident")
    if stride is not None and tuple(tensor.stride()) != stride:
        raise ValueError(f"{name} has strides {tuple(tensor.stride())}, expected compiled strides {stride}")
