# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural SM90 WGMMA attachment for canonical Contract components."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import cutlass
import quack.cache
import torch
from cuda import bindings as cuda_bindings
from quack.gemm_config import GemmConfig
from quack.gemm_runtime.host import _compile_gemm_epi

from tile_lifetime import sm90_contract_binding
from tile_lifetime.backends.h100.cute_launch_capture import capture_single_cute_launch
from tile_lifetime.backends.h100.cute_map_load_contract import (
    run_sm90_map_load_contract,
)
from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComponent,
    CuteKernelComponentFragment,
    bind_cute_pipeline_argument,
)
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotKind,
    elf64_symbol_bytes,
)
from tile_lifetime.cute_sm90_launch_abi import (
    relocated_sm90_arguments,
    sm90_launch_function_attributes,
    sm90_launch_parameter_schema,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    DType,
    ProgramValue,
    TensorProgram,
    serialize_scalar_expression,
)

from .cute_generic_tensor_program import GenericPhysicalComponentCompiler

_QUACK_EXPORTABLE_PLAN_LOCK = threading.Lock()


@dataclass
class CompiledSm90ContractComponent:
    """A compiled QuACK WGMMA plan bound to one canonical component."""

    binding: sm90_contract_binding.Sm90ContractComponentBinding
    live_inputs: tuple[ProgramValue, ...]
    transform: Any
    plan: Any
    artifact: CutePipelineArtifact
    device: torch.device
    captured_launch: CuteKernelLaunch | None = None

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return self.artifact.kernel_names

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        if len(outputs) != 1:
            raise ValueError("SM90 WGMMA Contract component requires one output")
        by_value = dict(zip(self.live_inputs, inputs, strict=True))
        activation = _matrix_view(by_value[self.binding.activation], self.binding.activation_view)
        weight = _matrix_view(by_value[self.binding.weight], self.binding.weight_view)
        output = _matrix_view(outputs[0], self.binding.output_view)
        transform_operands = _transform_operands(self.binding, by_value)
        run_sm90_map_load_contract(
            self.plan,
            self.transform,
            self.binding.generated.schedule,
            activation,
            weight,
            output,
            transform_operands,
        )

    def canonical_fragment(
        self,
        component: CanonicalKernelComponent,
        value_ordinals: dict[ProgramValue, int],
    ) -> CuteKernelComponentFragment:
        values = (*self.live_inputs, self.binding.output)
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
            for value in values
        }
        if self.captured_launch is None:
            self.captured_launch = _captured_launch(self)
        bindings = {f"tensor_{index}": slots[value_ordinals[value]].name for index, value in enumerate(values)}
        launch = replace(
            self.captured_launch,
            arguments=tuple(
                bind_cute_pipeline_argument(argument, bindings) for argument in self.captured_launch.arguments
            ),
            source_tasks=component.task_ids,
        )
        return CuteKernelComponentFragment(component.id, component.schedule_fingerprint, slots, (launch,))


def sm90_contract_component_compiler() -> GenericPhysicalComponentCompiler:
    """Return a provider selected only from canonical Contract structure."""

    def compile_component(
        program: TensorProgram,
        component: CanonicalKernelComponent,
        live_inputs: tuple[ProgramValue, ...],
        live_outputs: tuple[ProgramValue, ...],
        example_inputs: tuple[torch.Tensor, ...],
    ) -> CompiledSm90ContractComponent | None:
        binding = sm90_contract_binding.sm90_contract_component_binding(program, component)
        if binding is None:
            return None
        if live_outputs != (binding.output,) or set(live_inputs) != {*binding.scalar_leaves, binding.weight}:
            return None
        tensors = dict(zip(live_inputs, example_inputs, strict=True))
        activation = _matrix_view(tensors[binding.activation], binding.activation_view)
        weight = _matrix_view(tensors[binding.weight], binding.weight_view)
        output = _matrix_view(
            torch.empty(binding.output.shape, dtype=torch.bfloat16, device=activation.device), binding.output_view
        )
        namespace: dict[str, object] = {}
        exec(binding.generated.source, namespace)
        epilogue = namespace["generated_epilogue"]
        transform = namespace["generated_transform"]
        config = GemmConfig(
            tile_m=binding.generated.schedule.tile_m,
            tile_n=binding.generated.schedule.tile_n,
            tile_k=binding.generated.schedule.tile_k,
            pingpong=binding.generated.schedule.pingpong,
            is_dynamic_persistent=False,
            cluster_m=binding.generated.schedule.cluster_m,
            cluster_n=binding.generated.schedule.cluster_n,
            device_capacity=9,
        )
        transform_operands = _transform_operands(binding, tensors)
        plan = _exportable_plan(epilogue, transform, transform_operands, activation, weight, output, config)
        artifact = _artifact(program, component, binding, plan)
        return CompiledSm90ContractComponent(binding, live_inputs, transform, plan, artifact, activation.device)

    return compile_component


def _matrix_view(tensor: torch.Tensor, view: sm90_contract_binding.Sm90MatrixView) -> torch.Tensor:
    view.validate_source_storage(tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
    if tensor.dtype is not torch.bfloat16:
        raise ValueError("SM90 matrix views require BF16 storage")
    matrix = tensor.view(view.storage_shape)
    if view.order is sm90_contract_binding.MatrixStorageOrder.COLUMN_MAJOR:
        matrix = matrix.transpose(0, 1)
    if (
        tuple(matrix.shape) != view.shape
        or tuple(matrix.stride()) != view.strides
        or matrix.data_ptr() != tensor.data_ptr()
        or matrix.storage_offset() != 0
    ):
        raise ValueError("SM90 matrix view must preserve the proved shape, strides, and base storage")
    return matrix


def _captured_launch(component: CompiledSm90ContractComponent) -> CuteKernelLaunch:
    if (component.binding.generated.schedule.cluster_m, component.binding.generated.schedule.cluster_n) != (1, 1):
        raise ValueError("SM90 Contract export does not support dynamic cluster launch attributes")
    schema = sm90_launch_parameter_schema(component.plan.gemm_plan.compiled_fn.ir_module)
    values = (*component.live_inputs, component.binding.output)
    # Independent allocations make slot identity observable even for equal
    # shapes or aliased caller examples. Capture does not execute these graphs.
    tensors = tuple(
        torch.zeros(
            value.shape, dtype=torch.bfloat16 if value.dtype is DType.BF16 else torch.float32, device=component.device
        )
        for value in values
    )
    component(tensors[:-1], tensors[-1:])
    torch.cuda.synchronize(component.device)
    baseline = capture_single_cute_launch(partial(component, tensors[:-1], tensors[-1:]))
    if baseline.symbol not in component.kernel_names:
        raise ValueError("SM90 captured launch is absent from the exported device image")
    addresses = {f"tensor_{index}": tensor.data_ptr() for index, tensor in enumerate(tensors)}
    relocations = {}
    for index, tensor in enumerate(tensors):
        alternate = torch.empty_like(tensor)
        relocated = (*tensors[:index], alternate, *tensors[index + 1 :])
        captured = capture_single_cute_launch(partial(component, relocated[:-1], relocated[-1:]))
        if replace(captured, parameters=baseline.parameters) != baseline:
            raise ValueError("SM90 launch geometry changes when relocating a tensor")
        relocations[f"tensor_{index}"] = alternate.data_ptr(), captured.parameters
    arguments = relocated_sm90_arguments(
        schema,
        baseline.parameters,
        addresses,
        relocations,
        {
            f"tensor_{index}": (
                CutePipelineSlotAccess.WRITE if value is component.binding.output else CutePipelineSlotAccess.READ
            )
            for index, value in enumerate(values)
        },
    )
    return CuteKernelLaunch(
        name="wgmma_contract",
        kernel_symbol=baseline.symbol,
        grid=baseline.grid,
        block=baseline.block,
        dynamic_shared_memory_bytes=baseline.shared_memory,
        function_attributes=sm90_launch_function_attributes(baseline.cluster_dimensions, baseline.function_attributes),
        arguments=arguments,
    )


def _exportable_plan(
    epilogue: Any,
    transform: Any,
    transform_operands: dict[str, torch.Tensor],
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: GemmConfig,
) -> Any:
    """Compile a QuACK plan while retaining its exportable CuTe representation.

    QuACK's persistent cache reloads a TVM-FFI callable, which is sufficient for
    execution but deliberately omits the host object, fatbin, and compiler IR
    required by a Shuttle physical artifact. Serialize this small compile-only
    section and bypass that lossy cache; the resulting sealed Shuttle artifact
    is itself content-addressed and reusable.
    """
    with _QUACK_EXPORTABLE_PLAN_LOCK:
        cache_enabled = quack.cache.CACHE_ENABLED
        quack.cache.CACHE_ENABLED = False
        _compile_gemm_epi.cache_clear()
        epilogue._plan_cache.clear()
        try:
            return epilogue.plan(
                activation,
                weight,
                out={"D": output},
                config=config,
                transform_a=transform,
                transform_operands=transform_operands,
            )
        finally:
            quack.cache.CACHE_ENABLED = cache_enabled


def _transform_operands(
    binding: sm90_contract_binding.Sm90ContractComponentBinding,
    tensors: dict[ProgramValue, torch.Tensor],
) -> dict[str, torch.Tensor]:
    values = tuple(value for value in binding.scalar_leaves if value is not binding.activation)
    operands: dict[str, torch.Tensor] = {}
    for generated, value in zip(binding.generated.operands, values, strict=True):
        tensor = tensors[value]
        if generated.delivery.startswith("mvec_"):
            tensor = tensor.reshape(binding.rows)
        else:
            tensor = tensor.reshape(binding.reduction)
        operands[generated.parameter] = tensor
    return operands


def _artifact(
    program: TensorProgram,
    component: CanonicalKernelComponent,
    binding: sm90_contract_binding.Sm90ContractComponentBinding,
    plan: Any,
) -> CutePipelineArtifact:
    compiled = plan.gemm_plan.compiled_fn
    function_prefix = f"shuttle_wgmma_contract_{binding.generated.source_sha256[:16]}"
    with tempfile.TemporaryDirectory(prefix="shuttle-wgmma-contract-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        compiled.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    expression = json.loads(serialize_scalar_expression(binding.expression))
    request = CutePipelineCompileRequest(
        schema_version=1,
        target="sm90",
        numerical_policy=program.numerical_policy.value,
        physical_template="sm90_wgmma_map_load_contract_v1",
        tensors=tuple(
            {"shape": value.shape, "dtype": value.dtype.value}
            for value in (*binding.scalar_leaves, binding.weight, binding.output)
        ),
        scalar_programs={"map_load": expression},
        index_relations=tuple(component.operations),
        contraction_dimensions={
            "rows": binding.rows,
            "reduction": binding.reduction,
            "features": binding.features,
        },
        schedule={
            "family": binding.generated.schedule.family,
            "fingerprint": binding.generated.schedule.fingerprint,
            "tile_m": binding.generated.schedule.tile_m,
            "tile_n": binding.generated.schedule.tile_n,
            "tile_k": binding.generated.schedule.tile_k,
            "source_accumulation_dtype": next(
                operation.accumulation_dtype.value
                for operation in (program.operations[ordinal] for ordinal in binding.operation_ordinals)
                if isinstance(operation, ContractPrimitive)
            ),
            "physical_accumulation_dtype": "fp32",
            "matrix_views": {
                "activation": binding.activation_view.canonical_record,
                "weight": binding.weight_view.canonical_record,
                "output": binding.output_view.canonical_record,
            },
        },
        algebra_identity={
            "program_fingerprint": program.semantic_fingerprint,
            "component_schedule_fingerprint": component.schedule_fingerprint,
        },
    )
    kernel_names = tuple(compiled.kernel_info)
    return CutePipelineArtifact(
        request=request,
        host_object=host_object,
        device_image=elf64_symbol_bytes(host_object, "kernels_binary"),
        kernel_names=kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(compiled.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={
            "generator": binding.generated.source_sha256,
            "provider": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
