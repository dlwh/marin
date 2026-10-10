# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach the generic SM90 normalized-exp schedule as a canonical component."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComponent,
    CuteKernelComponentFragment,
)
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineArtifact,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
    CutePipelineTensorMapArgument,
)
from tile_lifetime.executable_normalized_exp_contract import prepare_sm90_normalized_exp_contract
from tile_lifetime.physical_task_contract import ExecutablePhysicalRequest
from tile_lifetime.tensor_program import ContractPrimitive, DType, ProgramValue, TensorAxis, TensorProgram

from .cute_generic_tensor_program import GenericPhysicalComponentCompiler
from .cute_streaming_backward_emitter import export_h100_normalized_exponential_forward_artifact
from .cute_streaming_emitter import CompiledH100StreamingProgram, compile_h100_streaming_program


@dataclass
class CompiledSm90NormalizedExpComponent:
    """One schedule-selected SM90 kernel bound to canonical live values."""

    request: ExecutablePhysicalRequest
    compiled: CompiledH100StreamingProgram
    artifact: CutePipelineArtifact
    query: ProgramValue
    key: ProgramValue
    value: ProgramValue
    output: ProgramValue
    output_physical_axes: tuple[TensorAxis, ...]

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return self.artifact.kernel_names

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 3 or len(outputs) != 1:
            raise ValueError("SM90 normalized-exp component requires three inputs and one output")
        by_value = dict(zip((self.query, self.key, self.value), inputs, strict=True))
        output_view = _axis_permutation_view(outputs[0], self.output, self.output_physical_axes)
        self.compiled(by_value[self.query], by_value[self.key], by_value[self.value], output_view)

    def canonical_fragment(
        self,
        component: CanonicalKernelComponent,
        value_ordinals: dict[ProgramValue, int],
    ) -> CuteKernelComponentFragment:
        values_by_name = {
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "output": self.output,
        }
        slots = {
            value_ordinals[item]: CutePipelineSlot(
                name=f"local_value_{value_ordinals[item]}",
                element_type=CutePipelineElementType.BF16,
                shape=item.shape,
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for item in values_by_name.values()
        }
        source_launch = self.artifact.launch_recipe
        if source_launch is None or len(source_launch.launches) != 1:
            raise ValueError("SM90 normalized-exp artifact must contain one launch recipe entry")
        local_slot_by_source = {name: slots[value_ordinals[item]].name for name, item in values_by_name.items()}
        launch = source_launch.launches[0]
        arguments = tuple(_rebind_argument(argument, local_slot_by_source) for argument in launch.arguments)
        return CuteKernelComponentFragment(
            component_id=component.id,
            schedule_fingerprint=component.schedule_fingerprint,
            slots_by_value_id=slots,
            launches=(
                replace(
                    launch,
                    arguments=arguments,
                    source_task_roles=(),
                    source_tasks=component.task_ids,
                ),
            ),
        )


def sm90_normalized_exp_component_compiler(
    request: ExecutablePhysicalRequest,
) -> GenericPhysicalComponentCompiler:
    """Return a component compiler driven by the attached generic schedule."""
    plan = prepare_sm90_normalized_exp_contract(request)
    region = plan.program.region
    layout = plan.program.schedule.contract.relation_layout
    if layout is None:
        raise ValueError("SM90 normalized-exp component requires a relation-packed schedule layout")
    if len(layout.batch_shape) != 1:
        raise ValueError("SM90 normalized-exp component currently requires one explicit batch axis")
    if any(axis.extent != 1 for axis in region.dimensions.relation_group_axes):
        raise ValueError("SM90 normalized-exp component currently requires singleton relation-group axes")
    query_plan, key_plan = region.producer.inputs
    value_plan = region.consumer.inputs[1]
    batch_axes = query_plan.axes[: len(layout.batch_shape)]
    if len(region.dimensions.shared_relation_axes) != 1:
        raise ValueError("SM90 normalized-exp component currently requires one shared relation axis")
    if len(region.dimensions.producer_features) != 1 or len(region.dimensions.consumer_features) != 1:
        raise ValueError("SM90 normalized-exp component currently requires one producer and consumer feature axis")
    shared_axis = region.dimensions.shared_relation_axes[0]
    group_axes = region.dimensions.relation_group_axes
    owner_axes = tuple(axis for axis in region.owner_axes if axis not in (*batch_axes, shared_axis, *group_axes))
    if len(owner_axes) != 1 or len(region.dimensions.normalized_axes) != 1:
        raise ValueError("SM90 normalized-exp component currently requires one owner and streamed axis")
    owner_axis = owner_axes[0]
    streamed_axis = region.dimensions.normalized_axes[0]
    producer_feature = region.dimensions.producer_features[0]
    consumer_feature = region.dimensions.consumer_features[0]
    query_physical_axes = _operand_axes(
        region.producer,
        region.producer.inputs.index(query_plan),
        (*batch_axes, owner_axis, shared_axis, producer_feature),
    )
    key_physical_axes = _operand_axes(
        region.producer,
        region.producer.inputs.index(key_plan),
        (*batch_axes, streamed_axis, shared_axis, producer_feature),
    )
    value_physical_axes = _operand_axes(
        region.consumer,
        region.consumer.inputs.index(value_plan),
        (*batch_axes, streamed_axis, shared_axis, consumer_feature),
    )
    output_physical_axes = (*batch_axes, owner_axis, shared_axis, consumer_feature)

    def compile_component(
        program: TensorProgram,
        component: CanonicalKernelComponent,
        live_inputs: tuple[ProgramValue, ...],
        live_outputs: tuple[ProgramValue, ...],
        example_inputs: tuple[torch.Tensor, ...],
    ) -> CompiledSm90NormalizedExpComponent | None:
        if component.task_ids != plan.owned_task_ids:
            return None
        if len(live_outputs) != 1:
            # A forward region embedded in an AD program can expose saved
            # values to a later pullback.  The forward-only kernel does not
            # implement that boundary; leave the exact component to the
            # generic fallback or a declared pullback provider.
            return None
        value_by_name = {item.name: item for item in (*live_inputs, *live_outputs)}
        try:
            query = value_by_name[query_plan.name]
            key = value_by_name[key_plan.name]
            value = value_by_name[value_plan.name]
        except KeyError as error:
            raise ValueError("SM90 normalized-exp component live boundary disagrees with its algebra") from error
        output = live_outputs[0]
        if any(item.dtype is not DType.BF16 for item in (query, key, value, output)):
            return None
        input_by_value = dict(zip(live_inputs, example_inputs, strict=True))
        query_tensor = _axis_permutation_view(input_by_value[query], query, query_physical_axes)
        key_tensor = _axis_permutation_view(input_by_value[key], key, key_physical_axes)
        value_tensor = _axis_permutation_view(input_by_value[value], value, value_physical_axes)
        output_storage = torch.empty(output.shape, dtype=torch.bfloat16, device=query_tensor.device)
        output_tensor = _axis_permutation_view(output_storage, output, output_physical_axes)
        compiled = compile_h100_streaming_program(
            plan.program,
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            output=output_tensor,
            log_sum_exp=None,
            event_schedule=plan.schedule_local_refinement.event_schedule,
        )
        artifact = export_h100_normalized_exponential_forward_artifact(
            compiled,
            compile_request=plan.compile_request,
            function_prefix=f"shuttle_component_{request.fingerprint[:16]}",
            source_slot_shapes={
                "query": query.shape,
                "key": key.shape,
                "value": value.shape,
                "output": output.shape,
            },
        )
        artifact = replace(
            artifact,
            internal_event_legalizations=(plan.internal_event_legalization,),
        )
        return CompiledSm90NormalizedExpComponent(
            request=request,
            compiled=compiled,
            artifact=artifact,
            query=query,
            key=key,
            value=value,
            output=output,
            output_physical_axes=output_physical_axes,
        )

    return compile_component


def _operand_axes(
    operation: ContractPrimitive,
    input_index: int,
    iteration_axes: tuple[TensorAxis, ...],
) -> tuple[TensorAxis, ...]:
    mappings = {mapping.domain_axis: mapping.operand_axis for mapping in operation.index_maps_for_input(input_index)}
    return tuple(mappings.get(axis, axis) for axis in iteration_axes)


def _axis_permutation_view(
    tensor: torch.Tensor,
    value: ProgramValue,
    physical_axes: tuple[TensorAxis, ...],
) -> torch.Tensor:
    if len(value.axes) != len(physical_axes):
        raise ValueError(
            f"SM90 normalized-exp physical axes {tuple(axis.id for axis in physical_axes)} cannot view "
            f"canonical value {value.name!r} axes {tuple(axis.id for axis in value.axes)}"
        )
    if tuple(tensor.shape) == tuple(axis.extent for axis in physical_axes):
        return tensor
    if set(value.axes) != set(physical_axes):
        raise ValueError(
            f"SM90 normalized-exp physical axes {tuple(axis.id for axis in physical_axes)} cannot view "
            f"canonical value {value.name!r} axes {tuple(axis.id for axis in value.axes)}"
        )
    permutation = tuple(value.axes.index(axis) for axis in physical_axes)
    return tensor.permute(permutation)


def _rebind_argument(argument: object, local_slot_by_source: dict[str, str]) -> object:
    if isinstance(argument, CutePipelineStructuredArgument):
        return replace(
            argument,
            fields=tuple(
                (
                    replace(field, slot=local_slot_by_source[field.slot])
                    if isinstance(field, CutePipelineStructuredSlotAddressField)
                    else field
                )
                for field in argument.fields
            ),
        )
    if not isinstance(argument, (CutePipelineSlotArgument, CutePipelineTensorMapArgument)):
        return argument
    try:
        slot = local_slot_by_source[argument.slot]
    except KeyError as error:
        raise ValueError(f"SM90 normalized-exp launch references unknown source slot {argument.slot!r}") from error
    return replace(argument, slot=slot)
