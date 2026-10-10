# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile an emitter-covered generic executable task region with CuTe."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from tile_lifetime.cute_pipeline_artifact import CutePipelineArtifact
from tile_lifetime.executable_normalized_exp_contract import (
    ExecutableNormalizedExpContractPlan,
    prepare_sm90_normalized_exp_contract,
)
from tile_lifetime.physical_task_contract import ExecutablePhysicalRequest

from .cute_streaming_backward_emitter import export_h100_normalized_exponential_forward_artifact
from .cute_streaming_emitter import compile_h100_streaming_program


@dataclass(frozen=True)
class ExecutableNormalizedExpContractArtifact:
    """Compiled normalized-exp kernel and the exact boundary it does not own."""

    plan: ExecutableNormalizedExpContractPlan
    artifact: CutePipelineArtifact

    def require_whole_request_runtime(self) -> None:
        """Require a runtime implementation for every task in the source request."""
        self.plan.require_whole_request_runtime()


def compile_h100_executable_normalized_exp_contract(
    request: ExecutablePhysicalRequest,
) -> ExecutableNormalizedExpContractArtifact:
    """Compile the normalized-exp task without hiding residual Relation tasks."""
    plan = prepare_sm90_normalized_exp_contract(request)
    if not torch.cuda.is_available():
        raise RuntimeError("SM90 CuTe device compilation requires a CUDA compiler device")
    layout = plan.program.schedule.contract.relation_layout
    assert layout is not None
    device = torch.device("cuda", torch.cuda.current_device())
    query = torch.empty(layout.owner_physical_shape(layout.producer_feature_extent), dtype=torch.bfloat16, device=device)
    key = torch.empty(
        layout.streamed_physical_shape(layout.producer_feature_extent), dtype=torch.bfloat16, device=device
    )
    value = torch.empty(
        layout.streamed_physical_shape(layout.consumer_feature_extent), dtype=torch.bfloat16, device=device
    )
    output = torch.empty(
        layout.owner_physical_shape(layout.consumer_feature_extent), dtype=torch.bfloat16, device=device
    )
    compiled = compile_h100_streaming_program(
        plan.program,
        query=query,
        key=key,
        value=value,
        output=output,
        log_sum_exp=None,
        event_schedule=plan.schedule_local_refinement.event_schedule,
    )
    artifact = export_h100_normalized_exponential_forward_artifact(
        compiled,
        compile_request=plan.compile_request,
        function_prefix=f"shuttle_executable_{request.fingerprint[:16]}",
        source_slot_shapes={
            "query": tuple(query.shape),
            "key": tuple(key.shape),
            "value": tuple(value.shape),
            "output": tuple(output.shape),
        },
    )
    artifact = replace(
        artifact,
        internal_event_legalizations=(plan.internal_event_legalization,),
    )
    return ExecutableNormalizedExpContractArtifact(plan, artifact)
