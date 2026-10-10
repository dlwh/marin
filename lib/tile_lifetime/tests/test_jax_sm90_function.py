# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
)
from tile_lifetime.event_dataflow import ImperativeEventOpKind
from tile_lifetime.jax_cute_pipeline_ffi import (
    CutePipelineBindingKind,
    cute_pipeline_bindings_from_artifact,
    generate_cute_pipeline_ffi,
)
from tile_lifetime.jax_sm90_function import prepare_jax_sm90_function
from tile_lifetime.streaming_attention_backward_reference import (
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention,
)
from tile_lifetime.tensor_program import NumericalPolicy


def _artifact() -> CutePipelineArtifact:
    source = CutePipelineSlot(
        "source",
        CutePipelineElementType.BF16,
        (2, 4),
        CutePipelineSlotKind.PUBLIC_INPUT,
        16,
        input_index=0,
    )
    result = CutePipelineSlot(
        "result",
        CutePipelineElementType.BF16,
        (2, 4),
        CutePipelineSlotKind.PUBLIC_OUTPUT,
        16,
        output_index=0,
    )
    launch = CuteKernelLaunch(
        name="kernel",
        kernel_symbol="kernel_symbol",
        grid=(1, 1, 1),
        block=(32, 1, 1),
        dynamic_shared_memory_bytes=0,
        function_attributes=(),
        arguments=(
            CutePipelineSlotArgument("source_pointer", "source", CutePipelineSlotAccess.READ),
            CutePipelineSlotArgument("result_pointer", "result", CutePipelineSlotAccess.WRITE),
        ),
    )
    request = CutePipelineCompileRequest(
        schema_version=4,
        target="sm_90a",
        numerical_policy="fast",
        physical_template="contract_fold_contract",
        tensors=(),
        scalar_programs={},
        index_relations=(),
        contraction_dimensions={},
        schedule={},
        algebra_identity={"source": "test"},
    )
    return CutePipelineArtifact(
        request=request,
        host_object=b"host",
        device_image=b"device",
        kernel_names=("kernel_symbol",),
        host_entry="host_entry",
        compiler_ir="module {}",
        toolchain={"compiler": "test"},
        source_digests={"emitter": "0" * 64},
        launch_recipe=CutePipelineLaunchRecipe(
            schema_version=2,
            slots=(result, source),
            launches=(launch,),
            host_call_arguments=("source", "result"),
        ),
    )


def test_artifact_launch_recipe_defines_contiguous_jax_ffi_boundary() -> None:
    artifact = _artifact()
    bindings = cute_pipeline_bindings_from_artifact(artifact)
    generated = generate_cute_pipeline_ffi(
        artifact,
        target_name="shuttle.test",
        tensors=bindings,
        scalar=None,
    )

    assert tuple((binding.name, binding.kind, binding.index) for binding in bindings) == (
        ("source", CutePipelineBindingKind.OPERAND, 0),
        ("result", CutePipelineBindingKind.RESULT, 0),
    )
    assert tuple(binding.jax_layout for binding in bindings) == ((0, 1), (0, 1))
    assert "TVMFFIAny arguments[2]" in generated.source
    assert "arguments, 2, &result" in generated.source
    assert "kTVMFFIFloat" not in generated.source


def test_ordinary_jax_callable_preserves_events_until_exact_sm90_schedule_cover() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    function = causal_gqa_attention(config)

    prepared = prepare_jax_sm90_function(
        function,
        query,
        key,
        key,
        tile_size=4,
        target_name="shuttle.ordinary_jax_test",
        numerical_policy=NumericalPolicy.FAST,
    )
    record = prepared.canonical_record

    assert not prepared.plan.whole_request_runtime_supported
    assert record["source_algebra_fingerprint"] == prepared.source_program.semantic_fingerprint
    assert record["transformed_algebra_fingerprint"] == prepared.physical_request.algebra_fingerprint
    assert record["physical_request_fingerprint"] == prepared.physical_request.fingerprint
    assert record["compile_request_digest"] == prepared.plan.compile_request.digest
    assert record["covered_task_ids"] == prepared.plan.owned_task_ids
    witnesses = cast(tuple[dict[str, Any], ...], record["event_erasure_witnesses"])
    assert tuple(witness["event"] for witness in witnesses) == ("event.1", "event.2", "event.3")
    assert all(lowering.retained_operations for lowering in prepared.preselection_event_lowerings)
    assert {
        operation.kind
        for lowering in prepared.preselection_event_lowerings
        for operation in lowering.retained_operations
    } == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }
    assert prepared.plan.boundary_event_names == ("event.0", "event.4")
    with pytest.raises(ValueError, match=r"boundary Events \[event.0, event.4\]"):
        prepared.plan.require_whole_request_runtime()


def test_ordinary_jax_callable_rejects_source_ordered_at_sm90_schedule_selection() -> None:
    config = StreamingAttentionBackwardDebugConfig(head_dimension=128)
    query = np.ones(
        (config.batch, config.query_length, config.query_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )
    key = np.ones(
        (config.batch, config.key_length, config.key_value_heads, config.head_dimension),
        dtype=jnp.bfloat16,
    )

    with pytest.raises(
        ValueError,
        match="SM90 normalized-exponential Contract/Fold schedule does not implement SOURCE_ORDERED Fold evaluation",
    ):
        prepare_jax_sm90_function(
            causal_gqa_attention(config),
            query,
            key,
            key,
            tile_size=4,
            target_name="shuttle.ordinary_jax_source_ordered_test",
            numerical_policy=NumericalPolicy.SOURCE_ORDERED,
        )
