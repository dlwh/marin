# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from test_sm90_backward_accumulation import _recipe

from lib.tile_lifetime.benchmarks.strict_launch_comparison import (
    MAX_CAPTURE_BYTES,
    CapturedGraph,
    CapturedKernel,
    CapturedNode,
    CapturedParameter,
    GraphNodeKind,
    LaunchScheduleEvidence,
    NumericalGateError,
    SlotBinding,
    SlotLayout,
    bounded_report_bytes,
    component_boundary_values,
    numerical_error_records,
    persist_precheck_failure,
    prepare_ordered_precheck_plan,
    validate_full_completion_edge,
)
from lib.tile_lifetime.benchmarks.strict_launch_comparison import (
    compare_captured_launches as _compare_captured_launches,
)
from tile_lifetime.canonical_kernel_composition import CanonicalKernelComponent
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CuteKernelLaunch,
    CutePipelineLaunchRecipe,
    CutePipelineScalarArgument,
    CutePipelineSlot,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredImmediateField,
    CutePipelineStructuredSlotAddressField,
    launch_dependency_event_schedule,
)
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineElementType as Type,
)
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineSlotAccess as Access,
)
from tile_lifetime.h100_streaming_lowering import Sm90BackwardAccumulatorSchedule
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    is_identity_post_normalization_pullback,
    normalized_exp_contract_pullback_program,
)
from tile_lifetime.sm90_backward_accumulation import Sm90BackwardAccumulationPlan
from tile_lifetime.tensor_program import deserialize_tensor_program


def _layouts(recipe):
    return tuple(
        SlotLayout(
            slot.name,
            slot.shape,
            tuple(int(np.prod(slot.shape[index + 1 :])) for index in range(len(slot.shape))),
            slot.element_type,
            "independent test declaration",
        )
        for slot in recipe.slots
    )


def compare_captured_launches(recipe, graph, bindings):
    evidence = LaunchScheduleEvidence(
        recipe.digest, launch_dependency_event_schedule(recipe.launches, slots=recipe.slots), _layouts(recipe)
    )
    return _compare_captured_launches(recipe, graph, bindings, evidence)


def _case(base=0x1000):
    slot = CutePipelineSlot("counter", Type.I32, (4,), CutePipelineSlotKind.SCRATCH, 16)
    reset = CuteKernelLaunch(
        "reset",
        "reset_symbol",
        (1, 1, 1),
        (128, 1, 1),
        0,
        (),
        (CutePipelineSlotArgument("counter", "counter", Access.WRITE),),
    )
    structured = CutePipelineStructuredArgument(
        "counter_view",
        24,
        8,
        (
            CutePipelineStructuredSlotAddressField("pointer", "counter", Access.READ_WRITE, 0, 0),
            CutePipelineStructuredImmediateField("shape", 8, np.asarray([4, 1], dtype="<i8").tobytes()),
        ),
    )
    main = CuteKernelLaunch("main", "main_symbol", (1, 1, 1), (128, 1, 1), 0, (), (structured,))
    recipe = CutePipelineLaunchRecipe(2, (slot,), (reset, main))
    binding = SlotBinding("counter", base, 32, 16, base + 16, (4,), (1,), Type.I32, 0)
    # Independently construct captured storage bytes, including an offset view.
    reset_bytes = struct.pack("<Q", base + 16)
    main_bytes = struct.pack("<Qqq", base + 16, 4, 1)
    nodes = (
        CapturedNode(
            30,
            GraphNodeKind.KERNEL,
            CapturedKernel(
                "main_symbol", (CapturedParameter(0, 24, main_bytes),), (1, 1, 1), (128, 1, 1), 0, (0, 0, 0), ()
            ),
        ),
        CapturedNode(
            10,
            GraphNodeKind.KERNEL,
            CapturedKernel(
                "reset_symbol", (CapturedParameter(0, 8, reset_bytes),), (1, 1, 1), (128, 1, 1), 0, (1, 1, 1), ()
            ),
        ),
        CapturedNode(20, GraphNodeKind.EMPTY),
    )
    return recipe, CapturedGraph(nodes, ((10, 20), (20, 30))), (binding,)


def test_real_declared_fields_and_relocation_match_despite_node_enumeration_order():
    recipe, graph, bindings = _case()
    result = compare_captured_launches(recipe, graph, bindings)
    assert result["metadata_matches"] and not result["native_execution_accepted"]
    assert result["launches"][0]["node"] == 10 and result["launches"][1]["node"] == 30
    assert result["unresolved"]
    assert len(bounded_report_bytes(result)) < 1 << 20
    moved_recipe, moved_graph, moved_bindings = _case(0x4000)
    assert compare_captured_launches(moved_recipe, moved_graph, moved_bindings)["metadata_matches"]
    with pytest.raises(ValueError, match="declared bytes"):
        compare_captured_launches(recipe, graph, moved_bindings)


@pytest.mark.parametrize(
    "change",
    [
        {"pointer": 0x1020},
        {"byte_offset": 0},
        {"storage_bytes": 31},
        {"pointer": 0x1011, "byte_offset": 17},
        {"shape": (2, 2), "strides": (2, 1)},
        {"strides": (2,)},
        {"element_type": Type.F32},
        {"storage_base": (1 << 64) - 32, "pointer": (1 << 64) - 16},
    ],
)
def test_wrong_independent_storage_identity_rejects(change):
    recipe, graph, bindings = _case()
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, graph, (replace(bindings[0], **change),))


@pytest.mark.parametrize("bindings", [(), ("duplicate",)])
def test_missing_and_ambiguous_binding_rejects(bindings):
    recipe, graph, original = _case()
    values = () if not bindings else original * 2
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, graph, values)


@pytest.mark.parametrize("edges", [(), ((20, 10), (30, 20)), ((10, 20), (20, 30), (30, 10)), ((10, 99),)])
def test_missing_reversed_cyclic_or_unknown_dependency_rejects(edges):
    recipe, graph, bindings = _case()
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, replace(graph, edges=edges), bindings)


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "unknown"},
        {"grid": (2, 1, 1)},
        {"block": (64, 1, 1)},
        {"shared_bytes": 16},
        {"cluster": (2, 1, 1)},
        {"parameters": ()},
        {"parameters": (CapturedParameter(0, 8, b"\0" * 8),)},
        {"parameters": (CapturedParameter(1, 24, b"\0" * 24),)},
        {"parameters": (CapturedParameter(MAX_CAPTURE_BYTES, 24, b"\0" * 24),)},
        {"parameters": (CapturedParameter(0, 24, struct.pack("<Qqq", 0x1010, 5, 1)),)},
    ],
)
def test_parameter_geometry_and_declared_immediate_mismatch_rejects(change):
    recipe, graph, bindings = _case()
    node = graph.nodes[0]
    altered = replace(node, kernel=replace(node.kernel, **change))
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, replace(graph, nodes=(altered, *graph.nodes[1:])), bindings)


@pytest.mark.parametrize("node", [CapturedNode(40, "memset"), CapturedNode(40, GraphNodeKind.EMPTY, object())])
def test_unknown_executing_or_malformed_node_rejects(node):
    recipe, graph, bindings = _case()
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, replace(graph, nodes=(*graph.nodes, node)), bindings)


def test_duplicate_or_extra_kernel_and_graph_bounds_reject():
    recipe, graph, bindings = _case()
    duplicate = replace(graph.nodes[0], node_id=40)
    for nodes in ((*graph.nodes, duplicate), graph.nodes * 30):
        with pytest.raises(ValueError):
            compare_captured_launches(recipe, replace(graph, nodes=nodes), bindings)
    with pytest.raises(ValueError, match="report bound"):
        bounded_report_bytes({"data": "x" * (1 << 20)})


def test_aliasing_unrelated_buffers_rejects():
    recipe, graph, bindings = _case()
    alias = replace(recipe.slots[0], name="other")
    reset = replace(
        recipe.launches[0],
        arguments=(*recipe.launches[0].arguments, CutePipelineSlotArgument("other", "other", Access.WRITE)),
    )
    recipe = replace(recipe, slots=(*recipe.slots, alias), launches=(reset, recipe.launches[1]))
    with pytest.raises(ValueError, match="alias"):
        compare_captured_launches(recipe, graph, (*bindings, replace(bindings[0], name="other")))


@pytest.mark.parametrize("fields", [(1, 0, 0, b""), (0, 1, 0, b""), (0, 0, 1, b""), (0, 0, 0, b"\1")])
def test_nondefault_dependency_annotations_reject(fields):
    with pytest.raises(ValueError, match="full-completion"):
        validate_full_completion_edge(*fields)


@pytest.mark.parametrize("mutation", ["missing", "contradictory", "unmapped", "identity"])
def test_saved_typed_schedule_is_checked_independently(mutation):
    recipe, graph, bindings = _case()
    schedule = launch_dependency_event_schedule(recipe.launches, slots=recipe.slots)
    digest = recipe.digest
    if mutation == "missing":
        schedule["predecessor_pairs"] = ()
        schedule["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"] = ()
    elif mutation == "contradictory":
        schedule["predecessor_pairs"] = ()
    elif mutation == "unmapped":
        schedule["realizations"][0]["event"]["required_dependence"]["relation"]["source_domain"]["name"] = "internal"
    else:
        digest = "stale"
    with pytest.raises(ValueError):
        _compare_captured_launches(recipe, graph, bindings, LaunchScheduleEvidence(digest, schedule, _layouts(recipe)))


def test_exact_dense_c33_boundary_uses_typed_source_values_and_preserves_external_group_folds():
    fixture = Path(__file__).parent / "fixtures"
    program = deserialize_tensor_program((fixture / "dense_compile_boundary_program.json").read_text())
    record = json.loads((fixture / "ordered_precheck_dense_boundary.json").read_text())
    operations = tuple(record["operations"])
    component = CanonicalKernelComponent(
        33,
        33,
        "pinned-cpu-boundary",
        (33,),
        operations,
        tuple(program.canonical_record["operations"][i] for i in operations),
        tuple(record["inputs"]),
        tuple(record["outputs"]),
    )
    inputs, outputs = component_boundary_values(program, component)
    assert len(inputs) == 6 and [value.shape for value in outputs] == [
        (1, 2048, 32, 128),
        (1, 2048, 8, 4, 128),
        (1, 2048, 8, 4, 128),
    ]
    assert 111 not in operations and 114 not in operations
    with pytest.raises(ValueError, match="omit"):
        component_boundary_values(program, replace(component, live_input_value_ids=component.live_input_value_ids[1:]))
    with pytest.raises(ValueError, match="external SSA"):
        component_boundary_values(
            program, replace(component, live_output_value_ids=component.live_output_value_ids[:-1])
        )
    with pytest.raises(ValueError, match="exact source"):
        component_boundary_values(program, replace(component, live_input_value_ids=(*component.live_input_value_ids, 0)))
    with pytest.raises(ValueError, match="ambiguous"):
        component_boundary_values(program, replace(component, live_output_value_ids=component.live_output_value_ids * 2))


@pytest.mark.parametrize("expanded", [False, True])
def test_prepared_hardware_cases_select_the_real_split_typed_backward_boundary(expanded):
    shape = (1, 256, 2, 128)
    examples = tuple(np.ones(item, dtype=jnp.bfloat16) for item in (shape, shape, shape, shape))
    plan = prepare_ordered_precheck_plan(examples, causal=True, expanded=expanded)
    schedules = [
        schedule for schedule in plan.directional_schedules if schedule.parameters.get("direction") == "pullback"
    ]
    assert len(schedules) == 1
    owners = [component for component in plan.composition.components if component.task_ids == schedules[0].task_ids]
    assert len(owners) == 1
    inputs, outputs = component_boundary_values(plan.program, owners[0])
    assert plan.saved_values[0] in inputs and len(outputs) == 3
    forward = plan.forward_plan.program
    layout = forward.schedule.contract.relation_layout
    accumulation = Sm90BackwardAccumulationPlan(
        layout.owner_extent,
        layout.streamed_extent,
        64,
        128,
        1,
        2,
        128,
        128,
        ContractDomainTraversal.LOWER_TRIANGULAR,
    )
    assert max(len(accumulation.contributors(owner)) for owner in range(accumulation.owner_blocks)) == 2
    assert accumulation.contributor_ranks[(3, 1)] == 0 and accumulation.contributor_ranks[(3, 0)] == 1
    backward = normalized_exp_contract_pullback_program(
        forward,
        schedule=Sm90BackwardAccumulatorSchedule(contract=forward.schedule.contract),
        traversal=ContractDomainTraversal.LOWER_TRIANGULAR,
    )
    # Nonlinear source maps stay outside this physical region; this is not VJP-role coverage.
    assert is_identity_post_normalization_pullback(backward)


@pytest.mark.parametrize("expanded", [False, True])
def test_plain_full_public_pipeline_has_an_explicit_unresolved_admission_boundary(expanded):
    examples = tuple(np.ones((1, 256, 2, 128), dtype=jnp.bfloat16) for _ in range(4))
    with pytest.raises(ValueError, match="one explicit owner coordinate"):
        prepare_ordered_precheck_plan(examples, causal=False, expanded=expanded)


def test_persisted_numerical_failure_preserves_nonfinite_and_magnitude_evidence(tmp_path):
    destination = tmp_path / "failure.json"
    with pytest.raises(NumericalGateError) as caught:
        numerical_error_records(
            (np.array([0.0, 2.0, np.nan], dtype=np.float32),),
            (np.array([-0.0, 1.0, 0.0], dtype=np.float32),),
            gate=6.103515625e-5,
        )
    persist_precheck_failure(destination, {"cases": []}, caught.value)
    report = json.loads(destination.read_text())
    assert report["status"] == "failed" and not report["exported_or_native_execution"]
    (record,) = report["numerical_evidence"]
    assert record["observed_nonfinite_elements"] == 1
    assert record["maximum_absolute_finite_difference"] == 1.0 and not record["passed"]
    assert destination.stat().st_size <= 1 << 20
    assert Path(report["exact_error_log"]["path"]).read_text().endswith("unchanged public numerical gate failed\n")


def test_oversized_failure_text_cannot_erase_the_bounded_failure_record(tmp_path):
    destination = tmp_path / "failure.json"
    exact = "X" * ((1 << 20) + 17)
    persist_precheck_failure(destination, {"oversized": exact}, ValueError(exact))
    report = json.loads(destination.read_text())
    assert report["status"] == "failed" and report["error_summary_truncated"]
    assert "prior_report_omitted" in report and destination.stat().st_size < 4096
    assert Path(report["exact_error_log"]["path"]).read_text() == f"ValueError: {exact}\n"


def test_fp32_numerical_gate_uses_original_subtraction_and_boundary(tmp_path):
    gate = 6.103515625e-5
    records = numerical_error_records((np.array([gate], dtype=np.float32),), (np.zeros(1),), gate=gate)
    assert records[0]["passed"] and records[0]["maximum_absolute_error"] == gate
    with pytest.raises(NumericalGateError):
        numerical_error_records(
            (np.array([np.nextafter(np.float32(gate), np.float32(np.inf))]),), (np.zeros(1),), gate=gate
        )


@pytest.mark.parametrize("expanded", [False, True])
def test_ordered_role_sequence_graph_matches_the_saved_static_launch_schedule(expanded):
    plan, recipe = _recipe(expanded)
    bindings = tuple(
        SlotBinding(
            slot.name,
            0x10000000 + index * 0x10000000,
            slot.required_bytes,
            0,
            0x10000000 + index * 0x10000000,
            slot.shape,
            tuple(int(np.prod(slot.shape[axis + 1 :])) for axis in range(len(slot.shape))),
            slot.element_type,
            0,
        )
        for index, slot in enumerate(recipe.slots)
    )
    addresses = {binding.name: binding.pointer for binding in bindings}
    nodes = []
    for index, launch in enumerate(recipe.launches):
        parameters, offset = [], 0
        for argument in launch.arguments:
            if isinstance(argument, CutePipelineSlotArgument):
                payload, alignment = struct.pack("<Q", addresses[argument.slot]), 8
            elif isinstance(argument, CutePipelineScalarArgument):
                payload, alignment = struct.pack("<i", argument.value), 4
            else:
                # Scheduler payload semantics are covered by the existing independent tests.
                payload, alignment = argument.data, argument.alignment
            offset = (offset + alignment - 1) // alignment * alignment
            parameters.append(CapturedParameter(offset, len(payload), payload))
            offset += len(payload)
        nodes.append(
            CapturedNode(
                index,
                GraphNodeKind.KERNEL,
                CapturedKernel(
                    launch.kernel_symbol,
                    tuple(parameters),
                    launch.grid,
                    launch.block,
                    launch.dynamic_shared_memory_bytes,
                    (1, 1, 1),
                    launch.function_attributes,
                ),
            )
        )
    graph = CapturedGraph(tuple(reversed(nodes)), tuple((i, i + 1) for i in range(len(nodes) - 1)))
    report = compare_captured_launches(recipe, graph, bindings)
    assert report["launches"][0]["name"].startswith("reset")
    assert report["launches"][-2]["grid"] == plan.grid
    assert len(report["launches"]) == (6 if expanded else 4)
    with pytest.raises(ValueError, match="captured"):
        compare_captured_launches(recipe, replace(graph, edges=graph.edges[1:]), bindings)


def test_missing_kernel_and_function_attribute_mismatch_reject():
    recipe, graph, bindings = _case()
    with pytest.raises(ValueError):
        compare_captured_launches(recipe, replace(graph, nodes=graph.nodes[1:], edges=()), bindings)
    attribute = CuteKernelFunctionAttribute(CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES, 1024)
    launches = (recipe.launches[0], replace(recipe.launches[1], function_attributes=(attribute,)))
    with pytest.raises(ValueError, match="attribute"):
        compare_captured_launches(replace(recipe, launches=launches), graph, bindings)


def test_dense_permutation_cannot_disagree_with_the_saved_compiled_slot_layout():
    recipe, graph, bindings = _case()
    slot = replace(recipe.slots[0], shape=(2, 3))
    recipe = replace(recipe, slots=(slot,))
    binding = replace(bindings[0], shape=(2, 3), strides=(3, 1), storage_bytes=64)
    assert compare_captured_launches(recipe, graph, (binding,))["metadata_matches"]
    with pytest.raises(ValueError, match="saved layout"):
        compare_captured_launches(recipe, graph, (replace(binding, strides=(1, 2)),))
    evidence = LaunchScheduleEvidence(
        recipe.digest, launch_dependency_event_schedule(recipe.launches, slots=recipe.slots), ()
    )
    with pytest.raises(ValueError, match="layout evidence"):
        _compare_captured_launches(recipe, graph, (binding,), evidence)
