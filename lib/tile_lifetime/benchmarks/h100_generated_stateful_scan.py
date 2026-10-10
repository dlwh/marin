# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and check JAX-owned forward and reverse generic Scan regions on H100."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from cuda.bindings import driver, nvrtc

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_generic_tensor_program import compile_generic_kernel_composition
from tile_lifetime.backends.h100.triton_ordered_state import (
    generate_cuda_ordered_state_artifact,
    ordered_state_scan_component,
)
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition
from tile_lifetime.dense_flow_planner import derive_composed_task_plan
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import PhysicalNumericalPolicy, PhysicalTarget
from tile_lifetime.stateful_scan_reference import (
    NaturalAffineScanConfig,
    natural_affine_scan_input_specifications,
    natural_affine_scan_region,
    natural_affine_scan_training_step,
)
from tile_lifetime.stateful_scan_task_pipeline import (
    attach_chunked_scan_physical_request,
    attach_whole_program_scan_physical_request,
    derive_chunked_scan_reference_plan,
    refine_whole_program_scan_tasks,
)
from tile_lifetime.tensor_program import (
    ScanPrimitive,
    TensorProgram,
    execute_tensor_program,
    primitive_inputs,
    primitive_outputs,
)


def _region(scan: ScanPrimitive, source: TensorProgram) -> TensorProgram:
    return TensorProgram(primitive_inputs(scan), (scan,), primitive_outputs(scan), source.numerical_policy)


class _NvrtcOrderedState:
    def __init__(self, source: str, region: TensorProgram, *, owner_count: int, thread_count: int):
        _check_cuda(driver.cuInit(0), "initialize CUDA")
        _, device = driver.cuDeviceGet(0)
        _, self.context = driver.cuDevicePrimaryCtxRetain(device)
        _check_cuda(driver.cuCtxSetCurrent(self.context), "select CUDA context")
        _, program = nvrtc.nvrtcCreateProgram(source.encode(), b"ordered_state.cu", 0, [], [])
        options = [
            b"--gpu-architecture=compute_90a",
            b"--use_fast_math",
            b"--std=c++17",
            b"--include-path=/app/.venv/lib/python3.12/site-packages/nvidia/cu13/include",
        ]
        result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if result[0].value != 0:
            _, size = nvrtc.nvrtcGetProgramLogSize(program)
            log = bytearray(size)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.decode())
        _, size = nvrtc.nvrtcGetPTXSize(program)
        image = bytearray(size)
        nvrtc.nvrtcGetPTX(program, image)
        result, self.module = driver.cuModuleLoadData(image)
        _check_cuda((result,), "load ordered-state CUDA image")
        result, self.function = driver.cuModuleGetFunction(self.module, b"ordered_state_kernel")
        _check_cuda((result,), "resolve ordered-state kernel")
        self.region = region
        self.owners = owner_count
        self.threads = thread_count

    def execute(
        self,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        _check_cuda(driver.cuCtxSetCurrent(self.context), "select ordered-state CUDA context")
        if outputs is None:
            outputs = [
                torch.empty(
                    value.shape,
                    dtype=torch.bfloat16 if value.dtype is DType.BF16 else torch.float32,
                    device=inputs[0].device,
                )
                for value in self.region.outputs
            ]
        arguments = [ctypes.c_void_p(tensor.data_ptr()) for tensor in (*inputs, *outputs)]
        parameters = (ctypes.c_void_p * len(arguments))(*(ctypes.addressof(argument) for argument in arguments))
        torch.cuda.synchronize()
        stream = driver.CUstream(0)
        result = driver.cuLaunchKernel(
            self.function,
            (self.owners + self.threads - 1) // self.threads,
            1,
            1,
            self.threads,
            1,
            1,
            0,
            stream,
            parameters,
            0,
        )
        if result[0].value != 0:
            raise RuntimeError(f"ordered-state launch failed: {result[0].name}")
        return outputs


def _check_cuda(result, action: str) -> None:
    if result[0].value != 0:
        raise RuntimeError(f"failed to {action}: {result[0].name}")


def _compile(region: TensorProgram, *, chunk_extent: int, state_column_tile: int):
    plan = derive_chunked_scan_reference_plan(
        region,
        chunk_extent=chunk_extent,
        state_column_tile=state_column_tile,
    )
    request = attach_chunked_scan_physical_request(
        plan,
        algebra_record=region.canonical_record,
        algebra_fingerprint=region.semantic_fingerprint,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "ordinary_jax", "automatic_differentiation": "jax"},
    )
    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(region, 0), request)
    extension = _NvrtcOrderedState(
        artifact.device_source,
        region,
        owner_count=artifact.owner_count,
        thread_count=artifact.thread_count,
    )
    return plan, request, artifact, extension


class _CompiledScanComponent:
    def __init__(
        self,
        runner: _NvrtcOrderedState,
        source_sha256: str,
        live_inputs: tuple,
        live_outputs: tuple,
    ):
        self.runner = runner
        self.source_sha256 = source_sha256
        self.live_inputs = live_inputs
        self.live_outputs = live_outputs

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return ("ordered_state_kernel",)

    def __call__(self, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
        supplied_inputs = dict(zip(self.live_inputs, inputs, strict=True))
        ordered_inputs = [supplied_inputs[value] for value in self.runner.region.inputs]
        supplied = dict(zip(self.live_outputs, outputs, strict=True))
        all_outputs = [
            supplied.get(
                value,
                torch.empty(
                    value.shape,
                    dtype=torch.bfloat16 if value.dtype is DType.BF16 else torch.float32,
                    device=inputs[0].device,
                ),
            )
            for value in self.runner.region.outputs
        ]
        self.runner.execute(ordered_inputs, all_outputs)


def _scan_component_compiler(*, chunk_extent: int, state_column_tile: int):
    def compile_component(program, component, live_inputs, live_outputs, _example_inputs):
        ordinals = set(component.operation_ordinals)
        if len(ordinals) != 1:
            return None
        operation = program.operations[next(iter(ordinals))]
        if not isinstance(operation, ScanPrimitive):
            return None
        if set(live_inputs) != set(primitive_inputs(operation)) or not set(live_outputs) <= set(
            primitive_outputs(operation)
        ):
            raise ValueError(
                "persistent Scan component must expose the exact generic Scan boundary: "
                f"inputs={[value.name for value in live_inputs]} expected_inputs="
                f"{[value.name for value in primitive_inputs(operation)]}; "
                f"outputs={[value.name for value in live_outputs]} expected_outputs="
                f"{[value.name for value in primitive_outputs(operation)]}"
            )
        region = TensorProgram(live_inputs, (operation,), primitive_outputs(operation), program.numerical_policy)
        _, _, artifact, runner = _compile(
            region,
            chunk_extent=chunk_extent,
            state_column_tile=state_column_tile,
        )
        return _CompiledScanComponent(runner, artifact.source_sha256, live_inputs, live_outputs)

    return compile_component


def _random_region_inputs(region: TensorProgram, rng: np.random.Generator) -> tuple[jnp.ndarray, ...]:
    values = []
    for value in region.inputs:
        array = (0.08 * rng.normal(size=value.shape)).astype(np.float32)
        values.append(jnp.asarray(array, dtype=jnp.bfloat16 if value.dtype is DType.BF16 else jnp.float32))
    return tuple(values)


def _torch_inputs(region: TensorProgram, values: tuple[jnp.ndarray, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.asarray(np.asarray(value, dtype=np.float32), device="cuda").to(
            torch.bfloat16 if specification.dtype is DType.BF16 else torch.float32
        )
        for specification, value in zip(region.inputs, values, strict=True)
    )


def _latency(call, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))


def _jax_latency(call, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(call())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(1_000 * (time.perf_counter() - start))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--key-dimension", type=int, default=16)
    parser.add_argument("--value-dimension", type=int, default=16)
    parser.add_argument("--update-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    config = NaturalAffineScanConfig(
        batch=1,
        sequence=args.sequence,
        heads=args.heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        update_rank=args.update_rank,
    )
    rng = np.random.default_rng(args.seed)
    source_inputs = tuple(
        jnp.asarray(
            (0.08 * rng.normal(size=specification.shape)).astype(np.float32),
            dtype=specification.dtype,
        )
        for specification in natural_affine_scan_input_specifications(config)
    )
    output, final_state = natural_affine_scan_region(config)(*source_inputs)
    public_arguments = (
        *source_inputs,
        jnp.asarray(0.08 * rng.normal(size=output.shape), dtype=output.dtype),
        jnp.asarray(0.08 * rng.normal(size=final_state.shape), dtype=final_state.dtype),
    )
    imported = jax_function_to_tensor_program(natural_affine_scan_training_step(config), *public_arguments)

    results = []
    for ordinal, scan in enumerate(imported.operations):
        if not isinstance(scan, ScanPrimitive):
            continue
        region = _region(scan, imported)
        inputs = _random_region_inputs(region, rng)
        expected = execute_tensor_program(
            region,
            {value.name: argument for value, argument in zip(region.inputs, inputs, strict=True)},
        )
        plan, request, artifact, extension = _compile(
            region,
            chunk_extent=min(32, config.sequence),
            state_column_tile=config.value_dimension,
        )
        torch_inputs = _torch_inputs(region, inputs)
        outputs = extension.execute(list(torch_inputs))
        errors = []
        for output_value, actual in zip(region.outputs, outputs, strict=True):
            reference = np.asarray(expected[output_value.name], dtype=np.float32)
            errors.append(float(np.max(np.abs(actual.float().cpu().numpy() - reference))))
        results.append(
            {
                "operation_ordinal": ordinal,
                "direction": scan.direction.value,
                "body_operation_count": len(scan.body.operations),
                "public_input_count": len(region.inputs),
                "public_output_count": len(region.outputs),
                "task_plan_fingerprint": plan.task_plan.fingerprint,
                "request_fingerprint": request.fingerprint,
                "source_sha256": artifact.source_sha256,
                "maximum_absolute_error": max(errors),
                "latency_ms": _latency(
                    lambda extension=extension, torch_inputs=torch_inputs: extension.execute(list(torch_inputs)),
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
            }
        )

    tile_extents = {axis.id: min(axis.extent, 32) for value in imported.inputs for axis in value.axes}
    coarse = derive_composed_task_plan(
        imported,
        tile_extent_by_axis=tile_extents,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    refined = refine_whole_program_scan_tasks(
        imported,
        coarse,
        chunk_extent=min(32, config.sequence),
        state_column_tile=config.value_dimension,
    )
    whole_request = attach_whole_program_scan_physical_request(
        imported,
        refined,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        provenance={"source_boundary": "ordinary_jax", "automatic_differentiation": "jax"},
    )
    composition = lower_canonical_kernel_composition(whole_request)
    torch_public_inputs = _torch_inputs(imported, public_arguments)
    executable = compile_generic_kernel_composition(
        imported,
        composition,
        example_inputs=torch_public_inputs,
        physical_component_compilers=(
            _scan_component_compiler(
                chunk_extent=min(32, config.sequence),
                state_column_tile=config.value_dimension,
            ),
        ),
    )
    generated_outputs = executable(*torch_public_inputs)
    expected_outputs = natural_affine_scan_training_step(config)(*public_arguments)
    whole_errors = tuple(
        float(np.max(np.abs(actual.float().cpu().numpy() - np.asarray(expected, dtype=np.float32))))
        for actual, expected in zip(generated_outputs, expected_outputs, strict=True)
    )
    physical_scan_components = tuple(
        component
        for component in executable.components
        if isinstance(component.physical_component, _CompiledScanComponent)
    )
    whole_latency = _latency(
        lambda: executable(*torch_public_inputs),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    jax_training_step = jax.jit(natural_affine_scan_training_step(config))
    jax.block_until_ready(jax_training_step(*public_arguments))
    jax_whole_latency = _jax_latency(
        lambda: jax_training_step(*public_arguments),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(
        json.dumps(
            {
                "shape": {
                    "batch": config.batch,
                    "sequence": config.sequence,
                    "heads": config.heads,
                    "key_dimension": config.key_dimension,
                    "value_dimension": config.value_dimension,
                    "update_rank": config.update_rank,
                },
                "source_program_fingerprint": imported.semantic_fingerprint,
                "regions": results,
                "whole_program": {
                    "task_plan_fingerprint": refined.task_plan.fingerprint,
                    "request_fingerprint": whole_request.fingerprint,
                    "composition_fingerprint": composition.fingerprint,
                    "task_count": len(refined.task_plan.tasks),
                    "event_count": len(refined.event_plans),
                    "cross_component_event_count": len(composition.concrete_events),
                    "component_count": len(composition.components),
                    "launch_count": sum(len(component.kernel_names) for component in executable.components),
                    "scan_component_source_sha256": tuple(
                        component.physical_component.source_sha256 for component in physical_scan_components
                    ),
                    "maximum_absolute_error": max(whole_errors),
                    "output_maximum_absolute_errors": whole_errors,
                    "latency_ms": whole_latency,
                    "jax_latency_ms": jax_whole_latency,
                    "generated_to_jax_latency_ratio": whole_latency / jax_whole_latency,
                },
                "toolchain": {
                    "jax": __import__("jax").__version__,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
