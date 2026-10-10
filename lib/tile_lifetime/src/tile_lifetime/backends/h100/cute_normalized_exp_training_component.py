# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the shared SM90 value-and-pullback pipeline to a canonical component."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from tile_lifetime.canonical_kernel_composition import CanonicalKernelComponent, CuteKernelComponentFragment
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CuteMultiImagePipelineArtifact,
    CutePipelineArtifact,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
    CutePipelineTensorMapArgument,
    bind_physical_event_provenance,
    combine_physical_event_schedules,
    launch_dependency_event_schedule,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90BackwardAccumulatorSchedule,
    Sm90RelationPackedLayout,
    select_sm90_contract_schedule,
)
from tile_lifetime.index_domain_canonicalization import normalized_exp_task_algebra_from_record
from tile_lifetime.jax_normalized_exp_contract_training import (
    ExecutableNormalizedExpValueAndGradPlan,
    ExecutableSplitNormalizedExpValueAndGradPlan,
)
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    normalized_exp_contract_pullback_program,
)
from tile_lifetime.sm90_backward_accumulation import Sm90AccumulationMode
from tile_lifetime.tensor_program import (
    DType,
    ProgramValue,
    TensorAxis,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)
from tile_lifetime.terminal_gradient_layout import (
    DenseOutputView,
    bind_dense_output_view,
    terminal_gradient_layout,
)

from .cute_generic_tensor_program import (
    CompiledGenericPrimitive,
    GenericPhysicalComponentCompiler,
    compile_generic_tensor_program,
    export_generic_primitive_artifact,
)
from .cute_streaming_backward_emitter import (
    CompiledH100NormalizedExponentialBackward,
    CompiledH100NormalizedExponentialValueAndGrad,
    compile_h100_normalized_exponential_backward,
    compile_h100_normalized_exponential_value_and_grad,
    export_h100_normalized_exponential_backward_artifact,
    export_h100_normalized_exponential_forward_artifact,
    export_h100_normalized_exponential_value_and_grad_artifact,
)
from .cute_streaming_emitter import CompiledH100StreamingProgram, compile_h100_streaming_program
from .cute_two_pass_normalized_exp import (
    CompiledH100ConversionFaithfulNormalizedExp,
    compile_h100_conversion_faithful_normalized_exp,
    conversion_faithful_boundary,
    export_h100_conversion_faithful_normalized_exp_artifact,
)


@dataclass
class CompiledSm90NormalizedExpValueAndGradComponent:
    """Generated forward/pullback launches with canonical live-value storage."""

    compiled: CompiledH100NormalizedExponentialValueAndGrad
    artifact: CutePipelineArtifact
    query: ProgramValue
    key: ProgramValue
    value: ProgramValue
    output_cotangent: ProgramValue
    output: ProgramValue
    query_cotangent: ProgramValue
    key_cotangent: ProgramValue
    value_cotangent: ProgramValue
    live_inputs: tuple[ProgramValue, ...]
    live_outputs: tuple[ProgramValue, ...]
    physical_axes: dict[ProgramValue, tuple[TensorAxis, ...]]
    physical_shapes: dict[ProgramValue, tuple[int, ...]]

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return self.artifact.kernel_names

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        by_value = dict(zip(self.live_inputs, inputs, strict=True))
        output_by_value = dict(zip(self.live_outputs, outputs, strict=True))
        self.compiled.run_into(
            *(
                _packed_view(by_value[item], item, self.physical_axes[item], self.physical_shapes[item])
                for item in (self.query, self.key, self.value, self.output_cotangent)
            ),
            *(
                _packed_view(output_by_value[item], item, self.physical_axes[item], self.physical_shapes[item])
                for item in (self.output, self.query_cotangent, self.key_cotangent, self.value_cotangent)
            ),
        )

    def canonical_fragment(
        self,
        component: CanonicalKernelComponent,
        value_ordinals: dict[ProgramValue, int],
    ) -> CuteKernelComponentFragment:
        values_by_source_name = {
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "output_cotangent": self.output_cotangent,
            "output": self.output,
            "query_cotangent": self.query_cotangent,
            "key_cotangent": self.key_cotangent,
            "value_cotangent": self.value_cotangent,
        }
        slots = {
            value_ordinals[item]: CutePipelineSlot(
                name=f"local_value_{value_ordinals[item]}",
                element_type=CutePipelineElementType.BF16,
                shape=item.shape,
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for item in values_by_source_name.values()
        }
        recipe = self.artifact.launch_recipe
        if recipe is None:
            raise ValueError("SM90 value-and-gradient artifact requires a launch recipe")
        mapped_names = {name: slots[value_ordinals[item]].name for name, item in values_by_source_name.items()}
        private_slots = tuple(slot for slot in recipe.slots if slot.name not in mapped_names)
        launches = tuple(
            replace(
                launch,
                arguments=tuple(_rebind_argument(argument, mapped_names) for argument in launch.arguments),
                source_task_roles=(),
                source_tasks=component.task_ids,
            )
            for launch in recipe.launches
        )
        return CuteKernelComponentFragment(
            component_id=component.id,
            schedule_fingerprint=component.schedule_fingerprint,
            slots_by_value_id=slots,
            launches=launches,
            private_slots=private_slots,
        )


@dataclass
class CompiledSm90NormalizedExpDirectionalComponent:
    """One independently emitted direction bound to its canonical live values."""

    compiled: (
        CompiledH100StreamingProgram
        | CompiledH100ConversionFaithfulNormalizedExp
        | CompiledH100NormalizedExponentialBackward
    )
    artifact: CuteMultiImagePipelineArtifact
    role_values: dict[str, ProgramValue]
    live_inputs: tuple[ProgramValue, ...]
    live_outputs: tuple[ProgramValue, ...]
    physical_axes: dict[ProgramValue, tuple[TensorAxis, ...]]
    physical_shapes: dict[ProgramValue, tuple[int, ...]]
    output_transports: dict[ProgramValue, CompiledGenericPrimitive]
    output_views: dict[ProgramValue, DenseOutputView]

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(launch.kernel_symbol for launch in self.artifact.launch_recipe.launches)

    def copy_output(self, destination: torch.Tensor, value: ProgramValue, source: torch.Tensor) -> None:
        canonical = _bound_output_view(source, self.output_views[value])
        if value not in self.output_transports:
            destination.copy_(canonical)
            return
        storage = torch.as_strided(source, size=(source.numel(),), stride=(1,))
        self.output_transports[value]((storage,), (destination,))

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        input_by_value = dict(zip(self.live_inputs, inputs, strict=True))
        output_by_value = dict(zip(self.live_outputs, outputs, strict=True))

        def view(value: ProgramValue) -> torch.Tensor:
            return _packed_view(
                input_by_value[value],
                value,
                self.physical_axes[value],
                self.physical_shapes[value],
            )

        def storage(value: ProgramValue) -> torch.Tensor:
            return torch.empty(
                self.physical_shapes[value],
                dtype=output_by_value[value].dtype,
                device=output_by_value[value].device,
            )

        if isinstance(self.compiled, (CompiledH100StreamingProgram, CompiledH100ConversionFaithfulNormalizedExp)):
            output = storage(self.role_values["output"])
            log_sum_exp = storage(self.role_values["log_sum_exp"])
            normalized_output = (
                storage(self.role_values["normalized_output"]) if "normalized_output" in self.role_values else None
            )
            if isinstance(self.compiled, CompiledH100ConversionFaithfulNormalizedExp):
                if normalized_output is not None:
                    raise ValueError("conversion-faithful two-pass forward does not materialize normalized output")
                layout = self.compiled.boundary.relation_layout
                state_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
                row_max = torch.empty(state_shape, dtype=torch.bfloat16, device=output.device)
                row_sum = torch.empty(state_shape, dtype=torch.float32, device=output.device)
                self.compiled(
                    view(self.role_values["query"]),
                    view(self.role_values["key"]),
                    view(self.role_values["value"]),
                    output,
                    row_max,
                    row_sum,
                    log_sum_exp,
                )
            else:
                self.compiled(
                    view(self.role_values["query"]),
                    view(self.role_values["key"]),
                    view(self.role_values["value"]),
                    output,
                    log_sum_exp,
                    normalized_output,
                )
            output_value = self.role_values["output"]
            self.copy_output(output_by_value[output_value], output_value, output)
            self.copy_output(
                output_by_value[self.role_values["log_sum_exp"]],
                self.role_values["log_sum_exp"],
                log_sum_exp,
            )
            if normalized_output is not None:
                normalized_value = self.role_values["normalized_output"]
                self.copy_output(
                    output_by_value[normalized_value],
                    normalized_value,
                    normalized_output,
                )
            return
        normalized_output = self.role_values.get("normalized_output", self.role_values["output"])
        self.compiled(
            view(self.role_values["query"]),
            view(self.role_values["key"]),
            view(self.role_values["value"]),
            view(self.role_values["output"]),
            view(self.role_values["output_cotangent"]),
            view(self.role_values["log_sum_exp"]),
            view(normalized_output),
        )
        for value, tensor in zip(
            (
                self.role_values["query_cotangent"],
                self.role_values["key_cotangent"],
                self.role_values["value_cotangent"],
            ),
            (self.compiled.query_cotangent, self.compiled.key_cotangent, self.compiled.value_cotangent),
            strict=True,
        ):
            self.copy_output(output_by_value[value], value, tensor)

    def canonical_fragment(
        self,
        component: CanonicalKernelComponent,
        value_ordinals: dict[ProgramValue, int],
    ) -> CuteKernelComponentFragment:
        slots = {
            value_ordinals[value]: CutePipelineSlot(
                name=f"local_value_{value_ordinals[value]}",
                element_type=(
                    CutePipelineElementType.BF16 if value.dtype is DType.BF16 else CutePipelineElementType.F32
                ),
                shape=value.shape,
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for value in (*self.live_inputs, *self.live_outputs)
        }
        mapped_names = {role: slots[value_ordinals[value]].name for role, value in self.role_values.items()}
        recipe = self.artifact.launch_recipe
        if recipe is None:
            raise ValueError("SM90 directional artifact requires a launch recipe")
        private_slots = tuple(slot for slot in recipe.slots if slot.name not in mapped_names)
        launches = tuple(
            replace(
                launch,
                arguments=tuple(_rebind_argument(argument, mapped_names) for argument in launch.arguments),
                source_task_roles=(),
                source_tasks=component.task_ids,
            )
            for launch in recipe.launches
        )
        return CuteKernelComponentFragment(
            component_id=component.id,
            schedule_fingerprint=component.schedule_fingerprint,
            slots_by_value_id=slots,
            launches=launches,
            private_slots=private_slots,
            physical_event_schedules=(
                (recipe.physical_event_provenance["schedule"],) if recipe.physical_event_provenance is not None else ()
            ),
        )


def sm90_normalized_exp_value_and_grad_component_compiler(
    plan: ExecutableNormalizedExpValueAndGradPlan,
) -> GenericPhysicalComponentCompiler:
    """Compile the schedule-selected pullback component without graph-name dispatch."""
    pullback_schedule = next(
        (schedule for schedule in plan.physical_request.schedules if schedule.parameters.get("direction") == "pullback"),
        None,
    )
    if pullback_schedule is None:
        raise ValueError("value-and-gradient request has no selected SM90 pullback schedule")
    layout = plan.pullback_plan.lowering.relation_layout

    def compile_component(
        program: TensorProgram,
        component: CanonicalKernelComponent,
        live_inputs: tuple[ProgramValue, ...],
        live_outputs: tuple[ProgramValue, ...],
        example_inputs: tuple[torch.Tensor, ...],
    ) -> CompiledSm90NormalizedExpValueAndGradComponent | None:
        if component.task_ids != pullback_schedule.task_ids:
            return None
        if len(plan.physical_request.normalized_exp_contract_regions) != 1:
            raise ValueError("SM90 value-and-gradient component requires one declared forward region")
        if len(plan.physical_request.normalized_exp_contract_pullbacks) != 1:
            raise ValueError("SM90 value-and-gradient component requires one declared pullback region")
        forward_capability = plan.physical_request.normalized_exp_contract_regions[0]
        pullback_capability = plan.physical_request.normalized_exp_contract_pullbacks[0]
        values_by_ordinal = {ordinal: item for item, ordinal in tensor_program_value_ordinals(program).items()}
        producer_inputs = primitive_inputs(program.operations[forward_capability.producer_operation])
        consumer_inputs = primitive_inputs(program.operations[forward_capability.consumer_operation])
        if len(producer_inputs) != 2 or len(consumer_inputs) != 2:
            raise ValueError("SM90 value-and-gradient component requires binary producer and consumer Contracts")
        query = producer_inputs[forward_capability.producer_owner_operand]
        key = producer_inputs[1 - forward_capability.producer_owner_operand]
        value = consumer_inputs[1 - forward_capability.normalized_operand]
        axes_by_id = {axis.id: axis for item in values_by_ordinal.values() for axis in item.axes}

        def capability_axes(references: tuple[str, ...]) -> tuple[TensorAxis, ...]:
            try:
                return tuple(axes_by_id[int(reference.removeprefix("a"))] for reference in references)
            except (KeyError, ValueError) as error:
                raise ValueError("normalized-exp capability references an unavailable canonical axis") from error

        shared_axes = capability_axes(forward_capability.shared_relation_axes)
        group_axes = capability_axes(forward_capability.relation_group_axes)
        normalized_axes = capability_axes(forward_capability.normalized_axes)
        producer_features = capability_axes(forward_capability.producer_features)
        consumer_features = capability_axes(forward_capability.consumer_features)
        if not all(len(axes) == 1 for axes in (shared_axes, normalized_axes, producer_features, consumer_features)):
            raise ValueError("current SM90 value-and-gradient component requires one axis per physical dimension")
        producer_output = primitive_outputs(program.operations[forward_capability.producer_operation])
        if len(producer_output) != 1:
            raise ValueError("declared producer Contract must produce one score tensor")
        relation_axes = {*shared_axes, *group_axes, *normalized_axes}
        owner_domain_axes = tuple(axis for axis in producer_output[0].axes if axis not in relation_axes)
        batch_rank = len(layout.batch_shape)
        if len(owner_domain_axes) != batch_rank + 1:
            raise ValueError("declared producer Contract does not expose one owner axis after batch axes")
        batch_axes = owner_domain_axes[:batch_rank]
        owner_axis = owner_domain_axes[-1]
        shared_axis = shared_axes[0]
        streamed_axis = normalized_axes[0]
        producer_feature = producer_features[0]
        consumer_feature = consumer_features[0]
        query_axes = (*batch_axes, owner_axis, shared_axis, *group_axes, producer_feature)
        key_axes = (*batch_axes, streamed_axis, shared_axis, producer_feature)
        value_axes = (*batch_axes, streamed_axis, shared_axis, consumer_feature)
        output_axes = (*batch_axes, owner_axis, shared_axis, *group_axes, consumer_feature)
        output_cotangent_inputs = primitive_inputs(
            program.operations[pullback_capability.post_normalization_vjp_operation]
        )
        if len(output_cotangent_inputs) != 1:
            raise ValueError("declared post-normalization pullback does not bind one output cotangent")
        output_cotangent = output_cotangent_inputs[0]
        output = values_by_ordinal[forward_capability.finalizer_value]
        terminal_outputs = tuple(
            primitive_outputs(program.operations[ordinal])
            for ordinal in pullback_capability.terminal_contract_operations
        )
        if any(len(outputs) != 1 for outputs in terminal_outputs):
            raise ValueError("declared terminal gradient Contracts must each produce one value")
        query_cotangent, key_cotangent, value_cotangent = tuple(outputs[0] for outputs in terminal_outputs)
        if not set((query, key, value, output_cotangent)) <= set(live_inputs) or not set(
            (output, query_cotangent, key_cotangent, value_cotangent)
        ) <= set(live_outputs):
            raise ValueError("SM90 value-and-gradient component boundary disagrees with its declared algebra roles")
        boundary = (
            query,
            key,
            value,
            output_cotangent,
            output,
            query_cotangent,
            key_cotangent,
            value_cotangent,
        )
        if any(item.dtype is not DType.BF16 for item in boundary):
            return None
        physical_axes = {
            query: query_axes,
            key: key_axes,
            value: value_axes,
            output_cotangent: output_axes,
            output: output_axes,
            query_cotangent: query_axes,
            key_cotangent: key_axes,
            value_cotangent: value_axes,
        }
        physical_shapes = {
            query: layout.owner_physical_shape(layout.producer_feature_extent),
            key: layout.streamed_physical_shape(layout.producer_feature_extent),
            value: layout.streamed_physical_shape(layout.consumer_feature_extent),
            output_cotangent: layout.owner_physical_shape(layout.consumer_feature_extent),
            output: layout.owner_physical_shape(layout.consumer_feature_extent),
            query_cotangent: layout.owner_physical_shape(layout.producer_feature_extent),
            key_cotangent: layout.streamed_physical_shape(layout.producer_feature_extent),
            value_cotangent: layout.streamed_physical_shape(layout.consumer_feature_extent),
        }
        input_by_value = dict(zip(live_inputs, example_inputs, strict=True))
        packed_inputs = tuple(
            _packed_view(input_by_value[item], item, physical_axes[item], physical_shapes[item])
            for item in (query, key, value, output_cotangent)
        )
        compiled = compile_h100_normalized_exponential_value_and_grad(
            plan.pullback_plan.program,
            query=packed_inputs[0],
            key=packed_inputs[1],
            value=packed_inputs[2],
            output_cotangent=packed_inputs[3],
        )
        artifact = export_h100_normalized_exponential_value_and_grad_artifact(
            compiled,
            function_prefix=f"shuttle_training_component_{plan.fingerprint[:16]}",
            source_slot_shapes={name: item.shape for name, item in values_by_source_name(boundary).items()},
        )
        return CompiledSm90NormalizedExpValueAndGradComponent(
            compiled,
            artifact,
            query,
            key,
            value,
            output_cotangent,
            output,
            query_cotangent,
            key_cotangent,
            value_cotangent,
            live_inputs,
            live_outputs,
            physical_axes,
            physical_shapes,
        )

    return compile_component


def sm90_split_normalized_exp_component_compiler(
    plan: ExecutableSplitNormalizedExpValueAndGradPlan,
    *,
    accumulation: Sm90AccumulationMode = Sm90AccumulationMode.UNORDERED,
) -> GenericPhysicalComponentCompiler:
    """Compile disjoint forward and reverse refinements against their live values."""
    forward_program = plan.forward_plan.program
    forward_source_program = normalized_exp_task_algebra_from_record(
        plan.forward_plan.source_request.algebra_record
    ).task_program
    layout = forward_program.schedule.contract.relation_layout
    if layout is None:
        raise ValueError("split normalized-exp forward schedule requires a relation layout")
    reverse_contract = select_sm90_contract_schedule(
        owner_extent=layout.owner_extent,
        streamed_extent=layout.streamed_extent,
        producer_feature_extent=layout.producer_feature_extent,
        consumer_feature_extent=layout.consumer_feature_extent,
        threads=384,
        owner_tile=64,
        streamed_tile=128,
        relation_layout=layout,
    )
    if not isinstance(accumulation, Sm90AccumulationMode):
        raise ValueError("SM90 accumulation requires an explicit physical mode")
    reverse_schedule = Sm90BackwardAccumulatorSchedule(
        contract=reverse_contract, deterministic_accumulation=accumulation is Sm90AccumulationMode.ORDERED
    )
    if reverse_schedule.post_normalization_cotangent_expansion_terms > 1:
        reverse_schedule = replace(reverse_schedule, consumer_output_cotangent_stages=1)
    reverse_program = normalized_exp_contract_pullback_program(
        forward_program,
        schedule=reverse_schedule,
        traversal=(
            ContractDomainTraversal.FULL
            if forward_program.region.domain_predicate is None
            else ContractDomainTraversal.LOWER_TRIANGULAR
        ),
    )
    direction_by_tasks = {
        schedule.task_ids: schedule.parameters.get("direction") for schedule in plan.directional_schedules
    }
    schedule_by_tasks = {schedule.task_ids: schedule for schedule in plan.directional_schedules}
    values_by_ordinal = {ordinal: value for value, ordinal in tensor_program_value_ordinals(plan.program).items()}
    forward_component = next(
        component for component in plan.composition.components if direction_by_tasks.get(component.task_ids) == "forward"
    )
    forward_live_outputs = {values_by_ordinal[ordinal] for ordinal in forward_component.live_output_value_ids}

    def component_boundary_ancestor(
        program: TensorProgram,
        component: CanonicalKernelComponent,
        value: ProgramValue,
        live_inputs: tuple[ProgramValue, ...],
    ) -> ProgramValue:
        live = set(live_inputs)
        producers = {
            output: (ordinal, operation)
            for ordinal, operation in enumerate(program.operations)
            for output in primitive_outputs(operation)
        }
        owned_ordinals = set(component.operation_ordinals)
        pending = [value]
        ancestors: set[ProgramValue] = set()
        visited: set[ProgramValue] = set()
        while pending:
            candidate = pending.pop()
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate in live:
                ancestors.add(candidate)
                continue
            producer = producers.get(candidate)
            if producer is None or producer[0] not in owned_ordinals:
                continue
            pending.extend(primitive_inputs(producer[1]))
        if len(ancestors) != 1:
            external_ancestors = ancestors - forward_live_outputs
            if len(external_ancestors) == 1:
                return next(iter(external_ancestors))
            names = sorted(item.name for item in ancestors)
            raise ValueError(f"split normalized-exp boundary ancestry must resolve one value, found {names}")
        return next(iter(ancestors))

    def compile_component(
        program: TensorProgram,
        component: CanonicalKernelComponent,
        live_inputs: tuple[ProgramValue, ...],
        live_outputs: tuple[ProgramValue, ...],
        example_inputs: tuple[torch.Tensor, ...],
    ) -> CompiledSm90NormalizedExpDirectionalComponent | None:
        direction = direction_by_tasks.get(component.task_ids)
        if direction is None:
            return None
        if len(plan.physical_request.normalized_exp_contract_regions) != 1:
            raise ValueError("split normalized-exp component requires one declared forward region")
        forward_capability = plan.physical_request.normalized_exp_contract_regions[0]
        producer_inputs = primitive_inputs(program.operations[forward_capability.producer_operation])
        consumer_inputs = primitive_inputs(program.operations[forward_capability.consumer_operation])
        if len(producer_inputs) != 2 or len(consumer_inputs) != 2:
            raise ValueError("split normalized-exp component requires binary producer and consumer Contracts")
        query = producer_inputs[forward_capability.producer_owner_operand]
        key = producer_inputs[1 - forward_capability.producer_owner_operand]
        value = consumer_inputs[1 - forward_capability.normalized_operand]
        layout_roles = _directional_layout_roles(program, layout, query, key, value)
        input_by_value = dict(zip(live_inputs, example_inputs, strict=True))
        physical_axes = dict(layout_roles[0])
        physical_shapes = dict(layout_roles[1])
        output_views = {}
        gradient_layouts = ()
        if direction == "forward":
            output = next(item for item in live_outputs if item.dtype is DType.BF16)
            log_sum_exp = next(
                item for item in live_outputs if item.dtype is DType.FP32 and len(item.shape) < len(output.shape)
            )
            normalized_outputs = tuple(
                item for item in live_outputs if item.dtype is DType.FP32 and set(item.axes) == set(output.axes)
            )
            role_values = {"query": query, "key": key, "value": value, "output": output, "log_sum_exp": log_sum_exp}
            if normalized_outputs:
                role_values["normalized_output"] = normalized_outputs[0]
            _add_output_layouts(physical_axes, physical_shapes, layout, role_values)
            packed = tuple(
                _packed_view(input_by_value[item], item, physical_axes[item], physical_shapes[item])
                for item in (query, key, value)
            )
            output_storage = torch.empty(physical_shapes[output], dtype=torch.bfloat16, device=packed[0].device)
            lse_storage = torch.empty(physical_shapes[log_sum_exp], dtype=torch.float32, device=packed[0].device)
            normalized_storage = (
                torch.empty(physical_shapes[normalized_outputs[0]], dtype=torch.float32, device=packed[0].device)
                if normalized_outputs
                else None
            )
            forward_schedule = schedule_by_tasks[component.task_ids]
            if forward_schedule.parameters.get("execution_graph") == "contract_fold_then_recomputed_map_contract":
                if normalized_storage is not None:
                    raise ValueError("conversion-faithful two-pass schedule cannot export a normalized-output tile")
                boundary = conversion_faithful_boundary(
                    forward_source_program,
                    layout,
                    forward_program.region.domain_predicate,
                    score_map=forward_program.region.score_map,
                    owner_dimension=forward_program.region.owner_axes[-1].id,
                    producer_contract=forward_program.region.producer,
                    consumer_contract=forward_program.region.consumer,
                )
                state_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
                row_max = torch.empty(state_shape, dtype=torch.bfloat16, device=packed[0].device)
                row_sum = torch.empty(state_shape, dtype=torch.float32, device=packed[0].device)
                compiled = compile_h100_conversion_faithful_normalized_exp(
                    boundary,
                    forward_program,
                    query=packed[0],
                    key=packed[1],
                    value=packed[2],
                    output=output_storage,
                    row_max=row_max,
                    row_sum=row_sum,
                    log_sum_exp=lse_storage,
                )

            else:
                compiled = compile_h100_streaming_program(
                    forward_program,
                    query=packed[0],
                    key=packed[1],
                    value=packed[2],
                    output=output_storage,
                    log_sum_exp=lse_storage,
                    normalized_output=normalized_storage,
                )

        else:
            capability = plan.physical_request.normalized_exp_contract_pullbacks[0]
            output = primitive_outputs(program.operations[capability.normalized_output_operation])[0]
            post_normalization_inputs = primitive_inputs(program.operations[capability.post_normalization_vjp_operation])
            cotangent_inputs = tuple(value for value in post_normalization_inputs if value != output)
            if len(cotangent_inputs) != 1:
                raise ValueError("split normalized-exp post-normalization VJP must expose one cotangent input")
            output_cotangent = component_boundary_ancestor(
                program,
                component,
                cotangent_inputs[0],
                live_inputs,
            )
            log_sum_exp = plan.saved_values[0]
            required_boundary = {output, output_cotangent, log_sum_exp}
            if not required_boundary <= set(live_inputs):
                missing = sorted(value.name for value in required_boundary - set(live_inputs))
                raise ValueError(f"split normalized-exp reverse component is missing boundary values {missing}")
            saved_normalized = tuple(
                item
                for item in plan.saved_values
                if item in live_inputs and item.dtype is DType.FP32 and item.shape == output.shape
            )
            gradient_layouts = tuple(
                terminal_gradient_layout(program, component.operation_ordinals, ordinal, primal, live_outputs)
                for ordinal, primal in zip(capability.terminal_contract_operations, (query, key, value), strict=True)
            )
            gradients = tuple(binding.output for binding in gradient_layouts)
            role_values = {
                "query": query,
                "key": key,
                "value": value,
                "output": output,
                "output_cotangent": output_cotangent,
                "log_sum_exp": log_sum_exp,
                "query_cotangent": gradients[0],
                "key_cotangent": gradients[1],
                "value_cotangent": gradients[2],
                "normalized_output": saved_normalized[0] if saved_normalized else output,
            }
            _add_output_layouts(physical_axes, physical_shapes, layout, role_values)
            packed_inputs = tuple(
                _packed_view(input_by_value[item], item, physical_axes[item], physical_shapes[item])
                for item in (query, key, value, output, output_cotangent, log_sum_exp)
            )
            normalized_tensor = (
                _packed_view(
                    input_by_value[saved_normalized[0]],
                    saved_normalized[0],
                    physical_axes[saved_normalized[0]],
                    physical_shapes[saved_normalized[0]],
                )
                if saved_normalized
                else packed_inputs[3]
            )
            for role, tensor in zip(
                ("query", "key", "value", "log_sum_exp"),
                (*packed_inputs[:3], packed_inputs[5]),
                strict=True,
            ):
                if tensor.stride(-1) != 1:
                    raise ValueError(
                        f"split normalized-exp reverse {role} boundary requires a contiguous minor dimension; "
                        f"found shape {tuple(tensor.shape)} and stride {tensor.stride()}"
                    )
            for role, tensor in zip(
                ("output", "output_cotangent", "normalized_output"),
                (*packed_inputs[3:5], normalized_tensor),
                strict=True,
            ):
                unit_stride_dimensions = tuple(
                    dimension
                    for dimension, (extent, stride) in enumerate(zip(tensor.shape, tensor.stride(), strict=True))
                    if extent > 1 and stride == 1
                )
                if tensor.stride(-1) != 1 and len(unit_stride_dimensions) != 1:
                    raise ValueError(
                        f"split normalized-exp reverse {role} boundary has no unique affine leading axis; "
                        f"found shape {tuple(tensor.shape)} and stride {tensor.stride()}"
                    )
            compiled = compile_h100_normalized_exponential_backward(
                reverse_program,
                query=packed_inputs[0],
                key=packed_inputs[1],
                value=packed_inputs[2],
                output=packed_inputs[3],
                output_cotangent=packed_inputs[4],
                log_sum_exp=packed_inputs[5],
                normalized_output=normalized_tensor,
            )

        output_tensors = (
            {
                "output": output_storage,
                "log_sum_exp": lse_storage,
                **({"normalized_output": normalized_storage} if normalized_storage is not None else {}),
            }
            if direction == "forward"
            else {
                "query_cotangent": compiled.query_cotangent,
                "key_cotangent": compiled.key_cotangent,
                "value_cotangent": compiled.value_cotangent,
            }
        )
        if direction == "forward":
            output_views = {
                role_values[role]: bind_dense_output_view(
                    role_values[role], physical_axes[role_values[role]], tuple(tensor.shape), tuple(tensor.stride())
                )
                for role, tensor in output_tensors.items()
            }
        else:
            for binding, primal, tensor in zip(
                gradient_layouts, (query, key, value), output_tensors.values(), strict=True
            ):
                if tuple(tensor.shape) != physical_shapes[primal]:
                    raise ValueError("compiled gradient allocation must preserve the admitted physical shape")
                output_views[binding.output] = binding.bind_storage(
                    physical_axes[primal], tuple(tensor.shape), tuple(tensor.stride())
                )
        for role, tensor in output_tensors.items():
            _bound_output_view(tensor, output_views[role_values[role]])
        # Capture the same allocations whose complete canonical views were just
        # proved. Invocation validates these exact shape/stride bindings again.
        if isinstance(compiled, CompiledH100ConversionFaithfulNormalizedExp):
            artifact = export_h100_conversion_faithful_normalized_exp_artifact(
                compiled,
                compile_request=plan.forward_plan.compile_request,
                function_prefix=f"shuttle_split_forward_{plan.fingerprint[:16]}",
            )
        elif isinstance(compiled, CompiledH100StreamingProgram):
            artifact = export_h100_normalized_exponential_forward_artifact(
                compiled,
                compile_request=plan.forward_plan.compile_request,
                function_prefix=f"shuttle_split_forward_{plan.fingerprint[:16]}",
            )
        else:
            artifact = export_h100_normalized_exponential_backward_artifact(
                compiled,
                function_prefix=f"shuttle_split_reverse_{plan.fingerprint[:16]}",
            )
        artifact = replace(
            artifact,
            request=replace(
                artifact.request,
                schedule={
                    **artifact.request.schedule,
                    "terminal_gradient_layouts": tuple(binding.canonical_record for binding in gradient_layouts),
                    "canonical_output_views": tuple(binding.canonical_record for binding in output_views.values()),
                },
            ),
        )
        transported_artifact, output_transports = _export_output_transports(
            artifact, role_values, output_views, output_tensors
        )
        return CompiledSm90NormalizedExpDirectionalComponent(
            compiled,
            transported_artifact,
            role_values,
            live_inputs,
            live_outputs,
            physical_axes,
            physical_shapes,
            output_transports,
            output_views,
        )

    return compile_component


def values_by_source_name(values: tuple[ProgramValue, ...]) -> dict[str, ProgramValue]:
    """Return source artifact role names for a complete component boundary."""
    names = (
        "query",
        "key",
        "value",
        "output_cotangent",
        "output",
        "query_cotangent",
        "key_cotangent",
        "value_cotangent",
    )
    return dict(zip(names, values, strict=True))


def _directional_layout_roles(
    program: TensorProgram,
    layout: Sm90RelationPackedLayout,
    query: ProgramValue,
    key: ProgramValue,
    value: ProgramValue,
) -> tuple[dict[ProgramValue, tuple[TensorAxis, ...]], dict[ProgramValue, tuple[int, ...]]]:
    return (
        {query: query.axes, key: key.axes, value: value.axes},
        {
            query: layout.owner_physical_shape(layout.producer_feature_extent),
            key: layout.streamed_physical_shape(layout.producer_feature_extent),
            value: layout.streamed_physical_shape(layout.consumer_feature_extent),
        },
    )


def _add_output_layouts(
    physical_axes: dict[ProgramValue, tuple[TensorAxis, ...]],
    physical_shapes: dict[ProgramValue, tuple[int, ...]],
    layout: Sm90RelationPackedLayout,
    roles: dict[str, ProgramValue],
) -> None:
    query = roles["query"]
    value = roles["value"]
    batch_rank = len(layout.batch_shape)
    batch_axes = query.axes[:batch_rank]
    owner_axis = query.axes[batch_rank]
    relation_axes = query.axes[batch_rank + 1 : -1]
    output_axes = (*batch_axes, owner_axis, *relation_axes, value.axes[-1])
    output_shape = layout.owner_physical_shape(layout.consumer_feature_extent)
    row_axes = (*batch_axes, *relation_axes, owner_axis)
    row_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
    for role in ("output", "output_cotangent", "normalized_output"):
        item = roles.get(role)
        if item is not None:
            physical_axes[item] = output_axes
            physical_shapes[item] = output_shape
    log_sum_exp = roles.get("log_sum_exp")
    if log_sum_exp is not None:
        physical_axes[log_sum_exp] = row_axes
        physical_shapes[log_sum_exp] = row_shape


def _packed_view(
    tensor: torch.Tensor,
    value: ProgramValue,
    logical_physical_axes: tuple[TensorAxis, ...],
    physical_shape: tuple[int, ...],
) -> torch.Tensor:
    if set(value.axes) != set(logical_physical_axes) or len(value.axes) != len(logical_physical_axes):
        raise ValueError(f"cannot pack canonical value {value.name!r} into the selected relation layout")
    permutation = tuple(value.axes.index(axis) for axis in logical_physical_axes)
    ordered = tensor if permutation == tuple(range(len(permutation))) else tensor.permute(permutation)
    if len(ordered.shape) == len(physical_shape):
        return torch.as_strided(ordered, size=physical_shape, stride=ordered.stride())
    if len(physical_shape) == 4 and len(ordered.shape) >= 5:
        relation_start = 2
    elif len(physical_shape) == 3 and len(ordered.shape) >= 4:
        relation_start = 1
    else:
        raise ValueError(f"canonical value {value.name!r} requires an unsupported physical packing")
    relation_shape = ordered.shape[relation_start:-1]
    relation_strides = ordered.stride()[relation_start:-1]
    if any(
        relation_strides[index] != relation_shape[index + 1] * relation_strides[index + 1]
        for index in range(len(relation_shape) - 1)
    ):
        raise ValueError(f"canonical value {value.name!r} requires an explicit input layout transport")
    return torch.as_strided(
        ordered,
        size=physical_shape,
        stride=(
            *(ordered.stride(index) for index in range(relation_start)),
            relation_strides[-1],
            ordered.stride(-1),
        ),
    )


def _bound_output_view(tensor: torch.Tensor, binding: DenseOutputView) -> torch.Tensor:
    expected_dtype = {DType.BF16: torch.bfloat16, DType.FP32: torch.float32}[binding.output.dtype]
    if tensor.dtype != expected_dtype:
        raise ValueError("physical output dtype differs from the bound layout")
    binding.validate_storage(tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), binding.output.dtype)
    if tensor.data_ptr() != tensor.untyped_storage().data_ptr():
        raise ValueError("physical output view must preserve its allocation base")
    if tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
        raise ValueError("physical output view must cover its exact allocation")
    result = torch.as_strided(tensor, size=binding.output.shape, stride=binding.result_strides)
    if result.data_ptr() != tensor.data_ptr() or result.is_contiguous() != binding.contiguous:
        raise ValueError("physical output does not realize its proved canonical view")
    return result


def _export_output_transports(
    artifact: CutePipelineArtifact,
    role_values: dict[str, ProgramValue],
    output_views: dict[ProgramValue, DenseOutputView],
    output_tensors: dict[str, torch.Tensor],
) -> tuple[CuteMultiImagePipelineArtifact, dict[ProgramValue, CompiledGenericPrimitive]]:
    recipe = artifact.launch_recipe
    if recipe is None:
        raise ValueError("physical output transports require a launch recipe")
    transported_tensors = {}
    original_slots = {slot.name: slot for slot in recipe.slots}
    for role, tensor in output_tensors.items():
        value = role_values[role]
        slot = original_slots[role]
        expected_element_type = {DType.BF16: CutePipelineElementType.BF16, DType.FP32: CutePipelineElementType.F32}[
            value.dtype
        ]
        if slot.element_type is not expected_element_type or math.prod(slot.shape) != math.prod(value.shape):
            raise ValueError("canonical output slot must preserve the physical dtype and byte extent")
        canonical = _bound_output_view(tensor, output_views[value])
        if not canonical.is_contiguous():
            transported_tensors[role] = tensor
    physical_names = {role: f"physical_{role}" for role in transported_tensors}
    slots = tuple(
        (
            replace(slot, name=physical_names[slot.name], kind=CutePipelineSlotKind.SCRATCH, output_index=None)
            if slot.name in physical_names
            else replace(slot, shape=role_values[slot.name].shape) if slot.name in output_tensors else slot
        )
        for slot in recipe.slots
    )
    launches = tuple(
        replace(launch, arguments=tuple(_rebind_argument(argument, physical_names) for argument in launch.arguments))
        for launch in recipe.launches
    )
    images = [artifact]
    image_by_launch = [0] * len(launches)
    transports = {}
    transport_records = []
    for role, tensor in transported_tensors.items():
        value = role_values[role]
        program = output_views[value].transport_program()
        storage = torch.as_strided(tensor, size=(tensor.numel(),), stride=(1,))
        compiled = compile_generic_tensor_program(program, example_inputs=(storage,))
        primitive = compiled.primitives[0]
        image = export_generic_primitive_artifact(
            primitive, program, function_prefix=f"canonical_copy_{role}_{artifact.request.digest[:16]}"
        )
        transports[value] = primitive
        transport_records.append({"role": role, "program_fingerprint": program.semantic_fingerprint})
        image_by_launch.append(len(images))
        images.append(image)
        slots = (*slots, replace(original_slots[role], shape=value.shape))
        launches = (
            *launches,
            CuteKernelLaunch(
                name=f"canonical_copy_{role}",
                kernel_symbol=primitive.kernel_names[0],
                grid=(math.ceil(tensor.numel() / 256), 1, 1),
                block=(256, 1, 1),
                dynamic_shared_memory_bytes=0,
                function_attributes=(),
                arguments=(
                    CutePipelineSlotArgument("source", physical_names[role], CutePipelineSlotAccess.READ),
                    CutePipelineSlotArgument("destination", role, CutePipelineSlotAccess.WRITE),
                ),
            ),
        )
    original_schedules = (
        () if recipe.physical_event_provenance is None else (recipe.physical_event_provenance["schedule"],)
    )
    transported_recipe = replace(
        recipe, slots=slots, launches=launches, physical_event_provenance=None, host_call_arguments=()
    )
    transported_recipe = bind_physical_event_provenance(
        transported_recipe,
        combine_physical_event_schedules(launch_dependency_event_schedule(launches, slots=slots), *original_schedules),
    )
    return (
        CuteMultiImagePipelineArtifact(
            request=replace(
                artifact.request,
                schedule={**artifact.request.schedule, "canonical_output_transports": tuple(transport_records)},
            ),
            images=tuple(images),
            launch_recipe=transported_recipe,
            image_by_launch=tuple(image_by_launch),
        ),
        transports,
    )


def _rebind_argument(argument: object, mapped_names: dict[str, str]) -> object:
    if isinstance(argument, CutePipelineStructuredArgument):
        return replace(
            argument,
            fields=tuple(
                (
                    replace(field, slot=mapped_names.get(field.slot, field.slot))
                    if isinstance(field, CutePipelineStructuredSlotAddressField)
                    else field
                )
                for field in argument.fields
            ),
        )
    if not isinstance(argument, (CutePipelineSlotArgument, CutePipelineTensorMapArgument)):
        return argument
    return replace(argument, slot=mapped_names.get(argument.slot, argument.slot))
