# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile supported ordinary JAX functions through Shuttle's SM90 pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import jax

from tile_lifetime.cute_pipeline_artifact import CutePipelineArtifact
from tile_lifetime.executable_normalized_exp_contract import (
    ExecutableNormalizedExpContractPlan,
    prepare_sm90_normalized_exp_contract,
)
from tile_lifetime.jax_cute_pipeline_ffi import (
    CompiledCutePipelineFfi,
    GeneratedCutePipelineFfi,
    call_cute_pipeline_ffi,
    compile_cute_pipeline_ffi,
    cute_pipeline_bindings_from_artifact,
    generate_cute_pipeline_ffi,
    register_cute_pipeline_ffi,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import (
    ExecutableEventLowering,
    ExecutablePhysicalRequest,
    PhysicalTarget,
    lower_executable_event_operations,
)
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import NumericalPolicy, TensorProgram


@dataclass(frozen=True)
class JaxSm90Toolchain:
    """Native paths required to link a CuTe artifact into JAX typed FFI."""

    compiler: Path
    cuda_include_directory: Path
    cuda_runtime_library: Path
    tvm_ffi_root: Path
    cute_runtime_library: Path


class CompleteRequestArtifact(Protocol):
    """A physical artifact whose source-task and Event cover was verified."""

    artifact: CutePipelineArtifact

    def require_whole_request_runtime(self) -> None: ...


class CompleteRequestArtifactCompiler(Protocol):
    """Compile one fully scheduled physical request to a native artifact."""

    def __call__(self, request: ExecutablePhysicalRequest) -> CompleteRequestArtifact: ...


@dataclass(frozen=True)
class PreparedJaxSm90Function:
    """An ordinary JAX function lowered through exact algebra, tasks, and Events."""

    source_program: TensorProgram
    physical_request: ExecutablePhysicalRequest
    plan: ExecutableNormalizedExpContractPlan
    preselection_event_lowerings: tuple[ExecutableEventLowering, ...]
    target_name: str

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "source_algebra_fingerprint": self.source_program.semantic_fingerprint,
            "transformed_algebra_fingerprint": self.physical_request.algebra_fingerprint,
            "physical_request_fingerprint": self.physical_request.fingerprint,
            "physical_request": self.physical_request.canonical_record,
            "compile_request_digest": self.plan.compile_request.digest,
            "covered_task_ids": tuple(sorted(self.plan.covered_task_ids)),
            "event_erasure_witnesses": self.plan.event_erasure_witnesses,
            "preselection_event_lowerings": tuple(
                lowering.canonical_record for lowering in self.preselection_event_lowerings
            ),
            "schedule_local_refinement": self.plan.schedule_local_refinement.canonical_record,
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class CompiledJaxSm90Function:
    """Registered JAX callable backed by one sealed CuTe physical artifact."""

    prepared: PreparedJaxSm90Function
    generated_ffi: GeneratedCutePipelineFfi
    compiled_ffi: CompiledCutePipelineFfi

    def __call__(self, *operands: jax.Array) -> jax.Array:
        outputs = call_cute_pipeline_ffi(self.generated_ffi, *operands)
        if len(outputs) != 1:
            raise ValueError("ordinary JAX function wrapper requires exactly one physical result")
        return outputs[0]

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "prepared": self.prepared.canonical_record,
            "artifact_manifest": self.generated_ffi.artifact.manifest,
            "ffi_target": self.generated_ffi.target_name,
            "ffi_handler": self.generated_ffi.handler_symbol,
            "ffi_compile_argv": self.compiled_ffi.compile_argv,
        }


def prepare_jax_sm90_function(
    function: Callable[..., Any],
    *example_arguments: Any,
    tile_size: int,
    target_name: str,
    numerical_policy: NumericalPolicy,
) -> PreparedJaxSm90Function:
    """Lower an ordinary JAX callable to an SM90 request with explicit component boundaries."""
    source = replace(
        jax_function_to_tensor_program(function, *example_arguments),
        numerical_policy=numerical_policy,
    )
    request = jax_tensor_program_executable_request(source, tile_size=tile_size, target=PhysicalTarget.SM90)
    preselection_event_lowerings = lower_executable_event_operations(request)
    attached = attach_sm90_contract_fold_schedule(request)
    plan = prepare_sm90_normalized_exp_contract(attached.request)
    return PreparedJaxSm90Function(source, attached.request, plan, preselection_event_lowerings, target_name)


def compile_prepared_jax_sm90_function(
    prepared: PreparedJaxSm90Function,
    *,
    directory: Path,
    toolchain: JaxSm90Toolchain,
    artifact_compiler: CompleteRequestArtifactCompiler,
) -> CompiledJaxSm90Function:
    """Compile, link, and register a prepared ordinary JAX function."""
    compiled_artifact = artifact_compiler(prepared.physical_request)
    compiled_artifact.require_whole_request_runtime()
    bindings = cute_pipeline_bindings_from_artifact(compiled_artifact.artifact)
    generated = generate_cute_pipeline_ffi(
        compiled_artifact.artifact,
        target_name=prepared.target_name,
        tensors=bindings,
        scalar=None,
    )
    compiled = compile_cute_pipeline_ffi(
        generated,
        directory=directory,
        compiler=toolchain.compiler,
        cuda_include_directory=toolchain.cuda_include_directory,
        cuda_runtime_library=toolchain.cuda_runtime_library,
        tvm_ffi_root=toolchain.tvm_ffi_root,
        cute_runtime_library=toolchain.cute_runtime_library,
    )
    register_cute_pipeline_ffi(compiled)
    return CompiledJaxSm90Function(prepared, generated, compiled)
