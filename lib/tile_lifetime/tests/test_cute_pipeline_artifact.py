# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tile_lifetime.cute_pipeline_artifact import (
    CuteAffineMap,
    CuteEventCoverEndpoint,
    CuteEventCoverPiece,
    CuteEventExecutionPoint,
    CuteEventMechanismAllocation,
    CuteEventPhysicalScope,
    CuteEventRealization,
    CuteEventRealizationKind,
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CuteKernelLaunch,
    CuteMultiImagePipelineArtifact,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelinePackedBytesArgument,
    CutePipelineScalarArgument,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    CutePipelineStructuredArgument,
    CutePipelineStructuredImmediateField,
    CutePipelineStructuredSlotAddressField,
    CutePipelineTensorMapArgument,
    CuteSourceTaskRole,
    CuteStructuralEventCover,
    CuteTensorMapInterleave,
    CuteTensorMapL2Promotion,
    CuteTensorMapOobFill,
    CuteTensorMapSwizzle,
    affine_constant,
    bind_event_realizations,
    bind_launch_source_tasks,
    bind_physical_event_provenance,
    combine_physical_event_schedules,
    launch_dependency_event_schedule,
    verify_materialized_cut_launch_dependencies,
)
from tile_lifetime.physical_artifact_compiler import decode_gpu_physical_artifact_request
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    derive_streaming_physical_event_schedule,
    streaming_physical_event_schedule_record,
)

_NATIVE_SCALAR_VALUE_AND_GRAD_FIXTURE = (
    Path(__file__).parent / "fixtures/native_schema_v3/jax-bf16-causal-gqa-scalar-value-and-grad-s128-d128-sm90.json"
)


def _captured_phase_cut_recipe():
    record = json.loads((Path(__file__).parent / "fixtures/phase_cut_event_provenance_v5.json").read_text())
    slots = tuple(
        CutePipelineSlot(
            **{
                **slot,
                "element_type": CutePipelineElementType(slot["element_type"]),
                "kind": CutePipelineSlotKind(slot["kind"]),
                "shape": tuple(slot["shape"]),
            }
        )
        for slot in record["slots"]
    )
    launches = []
    for launch in record["launches"]:
        arguments = []
        for argument in launch["arguments"]:
            fields = {name: value for name, value in argument.items() if name != "kind"}
            match argument["kind"]:
                case "slot":
                    arguments.append(
                        CutePipelineSlotArgument(**{**fields, "access": CutePipelineSlotAccess(fields["access"])})
                    )
                case "packed_bytes":
                    arguments.append(
                        CutePipelinePackedBytesArgument(
                            fields["name"], bytes.fromhex(fields["data_hex"]), fields["alignment"]
                        )
                    )
                case "scalar":
                    arguments.append(
                        CutePipelineScalarArgument(
                            **{**fields, "element_type": CutePipelineElementType(fields["element_type"])}
                        )
                    )
                case "structured_bytes":
                    members = []
                    for field in argument["fields"]:
                        if field["kind"] == "slot_address":
                            members.append(
                                CutePipelineStructuredSlotAddressField(
                                    **{key: value for key, value in field.items() if key not in {"kind", "access"}},
                                    access=CutePipelineSlotAccess(field["access"]),
                                )
                            )
                        else:
                            members.append(
                                CutePipelineStructuredImmediateField(
                                    field["name"], field["destination_byte_offset"], bytes.fromhex(field["data_hex"])
                                )
                            )
                    arguments.append(
                        CutePipelineStructuredArgument(
                            fields["name"], fields["size"], fields["alignment"], tuple(members)
                        )
                    )
                case _:
                    raise ValueError(f"unhandled captured argument {argument['kind']}")
        launches.append(
            CuteKernelLaunch(
                **{
                    key: value
                    for key, value in launch.items()
                    if key
                    not in {"arguments", "function_attributes", "source_task_roles", "grid", "block", "source_tasks"}
                },
                arguments=tuple(arguments),
                function_attributes=tuple(
                    CuteKernelFunctionAttribute(CuteKernelFunctionAttributeKind(a["kind"]), a["value"])
                    for a in launch["function_attributes"]
                ),
                source_task_roles=tuple(CuteSourceTaskRole(role) for role in launch["source_task_roles"]),
                grid=tuple(launch["grid"]),
                block=tuple(launch["block"]),
                source_tasks=tuple(launch["source_tasks"]),
            )
        )
    cuts = tuple(f"value_{event['cut_value_id']}" for phase in record["phase_summaries"] for event in phase["events"])
    return record, slots, tuple(launches), cuts


def test_captured_phase_cuts_are_covered_by_typed_launch_events_and_nested_regions_survive() -> None:
    record, slots, launches, cuts = _captured_phase_cut_recipe()
    assert json.loads(json.dumps([launch.canonical_record for launch in launches])) == record["launches"]
    assert json.loads(json.dumps([slot.canonical_record for slot in slots])) == record["slots"]
    typed = launch_dependency_event_schedule(launches, slots=slots)
    assert json.loads(json.dumps(typed)) == record["launch_event_schedule"]
    verify_materialized_cut_launch_dependencies(launches, slots=slots, cut_slots=cuts, launch_schedule=typed)
    nested = record["nested_event_schedules"]
    combined = combine_physical_event_schedules(typed, *nested)
    assert combined["regions"][1:] == tuple(nested)
    assert len(cuts) == 6
    assert len(combined["regions"]) == 3
    assert all(summary not in combined["regions"] for summary in record["phase_summaries"])


@pytest.mark.parametrize("representation", ["predecessors", "required"])
@pytest.mark.parametrize("mutation", ["missing", "reversed"])
def test_captured_phase_cut_rejects_missing_or_reversed_typed_edge(representation, mutation) -> None:
    record, slots, launches, cuts = _captured_phase_cut_recipe()
    schedule = record["launch_event_schedule"]
    # Captured component 3 writes value_19 in launch 3; launch 4 consumes it
    # through an actual TMA structured relocation rather than a bare pointer.
    if representation == "predecessors":
        schedule["predecessor_pairs"].remove([[3], [4]])
        if mutation == "reversed":
            schedule["predecessor_pairs"].append([[4], [3]])
    else:
        pairs = schedule["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
        pairs.remove({"source": [3], "target": [4]})
        if mutation == "reversed":
            pairs.append({"source": [4], "target": [3]})
    with pytest.raises(ValueError, match="lacks typed launch dependence 3->4"):
        verify_materialized_cut_launch_dependencies(launches, slots=slots, cut_slots=cuts, launch_schedule=schedule)


@pytest.mark.parametrize("mutation", ["binding", "writer", "reader", "multiple_writers", "reader_before_writer"])
def test_captured_phase_cut_requires_unambiguous_materialized_storage(mutation) -> None:
    record, slots, launches, cuts = _captured_phase_cut_recipe()
    cut = cuts[0]
    if mutation == "binding":
        slots = tuple(slot for slot in slots if slot.name != cut)
    elif mutation == "writer":
        launches = tuple(
            replace(launch, arguments=()) if index == 3 else launch for index, launch in enumerate(launches)
        )
    elif mutation == "reader":
        launches = tuple(
            launch if index == 3 else replace(launch, arguments=()) for index, launch in enumerate(launches)
        )
    elif mutation == "multiple_writers":
        launches = (
            *launches[:-1],
            replace(
                launches[-1], arguments=(CutePipelineSlotArgument("extra_writer", cut, CutePipelineSlotAccess.WRITE),)
            ),
        )
    else:
        launches = (*launches[:3], launches[4], launches[3], *launches[5:])
    with pytest.raises(ValueError, match="materialized phase cut"):
        verify_materialized_cut_launch_dependencies(
            launches, slots=slots, cut_slots=(cut,), launch_schedule=record["launch_event_schedule"]
        )


def _slots() -> tuple[CutePipelineSlot, ...]:
    return (
        CutePipelineSlot(
            "source",
            CutePipelineElementType.BF16,
            (2, 16),
            CutePipelineSlotKind.PUBLIC_INPUT,
            16,
            input_index=0,
        ),
        CutePipelineSlot(
            "state",
            CutePipelineElementType.F32,
            (2,),
            CutePipelineSlotKind.PUBLIC_INPUT_OUTPUT,
            16,
            input_index=1,
            output_index=1,
        ),
        CutePipelineSlot(
            "result",
            CutePipelineElementType.BF16,
            (2, 16),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            16,
            output_index=0,
        ),
        CutePipelineSlot(
            "workspace",
            CutePipelineElementType.F32,
            (2, 16),
            CutePipelineSlotKind.SCRATCH,
            128,
        ),
    )


def _launches() -> tuple[CuteKernelLaunch, ...]:
    return (
        CuteKernelLaunch(
            name="prepare",
            kernel_symbol="_Z7preparePKvPv",
            grid=(2, 1, 1),
            block=(128, 1, 1),
            dynamic_shared_memory_bytes=32768,
            function_attributes=(
                CuteKernelFunctionAttribute(CuteKernelFunctionAttributeKind.PREFERRED_SHARED_MEMORY_CARVEOUT, 100),
                CuteKernelFunctionAttribute(
                    CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
                    65536,
                ),
            ),
            arguments=(
                CutePipelineSlotArgument("source_pointer", "source", CutePipelineSlotAccess.READ),
                CutePipelineSlotArgument("workspace_pointer", "workspace", CutePipelineSlotAccess.WRITE),
                CutePipelineScalarArgument("scale", CutePipelineElementType.F32, 0.125),
                CutePipelineTensorMapArgument(
                    "source_tensor_map",
                    slot="source",
                    access=CutePipelineSlotAccess.READ,
                    element_type=CutePipelineElementType.BF16,
                    global_shape=(16, 2),
                    global_byte_strides=(32,),
                    box_shape=(16, 2),
                    element_strides=(1, 1),
                    interleave=CuteTensorMapInterleave.NONE,
                    swizzle=CuteTensorMapSwizzle.BYTES_128,
                    l2_promotion=CuteTensorMapL2Promotion.BYTES_128,
                    oob_fill=CuteTensorMapOobFill.NONE,
                ),
            ),
            source_tasks=(0,),
        ),
        CuteKernelLaunch(
            name="finish",
            kernel_symbol="_Z6finishPKvPvS0_",
            grid=(1, 2, 1),
            block=(384, 1, 1),
            dynamic_shared_memory_bytes=0,
            function_attributes=(),
            arguments=(
                CutePipelineSlotArgument("workspace_pointer", "workspace", CutePipelineSlotAccess.READ),
                CutePipelineSlotArgument("state_pointer", "state", CutePipelineSlotAccess.READ_WRITE),
                CutePipelineSlotArgument("result_pointer", "result", CutePipelineSlotAccess.WRITE),
                CutePipelineScalarArgument("count", CutePipelineElementType.I32, 8),
                CutePipelineStructuredArgument(
                    "epilogue_parameters",
                    size=32,
                    alignment=8,
                    fields=(
                        CutePipelineStructuredSlotAddressField(
                            "source_base",
                            slot="source",
                            access=CutePipelineSlotAccess.READ,
                            slot_byte_offset=2,
                            destination_byte_offset=8,
                        ),
                        CutePipelineStructuredImmediateField(
                            "flags",
                            destination_byte_offset=0,
                            data=b"\x01\x02\x03\x04",
                        ),
                        CutePipelineStructuredImmediateField(
                            "limit",
                            destination_byte_offset=24,
                            data=b"\x08\x00\x00\x00",
                        ),
                    ),
                ),
                CutePipelinePackedBytesArgument(
                    "epilogue",
                    b"\x02\x01",
                    2,
                ),
            ),
            source_tasks=(1,),
        ),
    )


def _recipe() -> CutePipelineLaunchRecipe:
    return CutePipelineLaunchRecipe(schema_version=2, slots=_slots(), launches=_launches())


def _request() -> CutePipelineCompileRequest:
    return CutePipelineCompileRequest(
        schema_version=2,
        target="sm_90a",
        numerical_policy="fast",
        physical_template="contract_fold_contract",
        tensors=(),
        scalar_programs={},
        index_relations=(),
        contraction_dimensions={},
        schedule={},
    )


def _artifact(recipe: CutePipelineLaunchRecipe, *, device_image: bytes = b"fatbin") -> CutePipelineArtifact:
    return CutePipelineArtifact(
        request=_request(),
        host_object=b"host-object",
        device_image=device_image,
        kernel_names=("_Z7preparePKvPv", "_Z6finishPKvPvS0_"),
        host_entry="pipeline_entry",
        compiler_ir="module {}",
        toolchain={"compiler": "test"},
        source_digests={"emitter": "0" * 64},
        launch_recipe=recipe,
    )


def test_multi_image_artifact_scopes_symbols_and_retains_recipe_identity() -> None:
    recipe = _recipe()
    images = tuple(
        CutePipelineArtifact(
            request=_request(),
            host_object=f"host-{index}".encode(),
            device_image=f"image-{index}".encode(),
            kernel_names=(launch.kernel_symbol,),
            host_entry=f"entry_{index}",
            compiler_ir=f"module {index}",
            toolchain={"compiler": "test"},
            source_digests={"emitter": "0" * 64},
        )
        for index, launch in enumerate(recipe.launches)
    )

    artifact = CuteMultiImagePipelineArtifact(_request(), images, recipe, (0, 1))

    assert artifact.canonical_record["launch_recipe_digest"] == recipe.digest
    assert len(artifact.digest) == 64
    with pytest.raises(ValueError, match="symbol is absent"):
        CuteMultiImagePipelineArtifact(_request(), images, recipe, (1, 0))


@pytest.mark.parametrize("image_by_launch", ((0, 1), (1, 0)))
def test_native_transport_encodes_image_table_and_scopes_duplicate_symbols(image_by_launch) -> None:
    recipe = replace(_recipe(), launches=tuple(replace(launch, kernel_symbol="shared_kernel") for launch in _launches()))
    images = tuple(
        replace(_artifact(_recipe(), device_image=image), kernel_names=("shared_kernel",), launch_recipe=recipe)
        for image in (b"first-fatbin", b"second-fatbin")
    )
    artifact = CuteMultiImagePipelineArtifact(_request(), images, recipe, image_by_launch)
    transport = artifact.native_transport

    assert transport[:21] == b"SHUTGPU\0" + struct.pack("<IqB", 12, 5, 4)
    offset = 21
    for digest in (recipe.digest, _request().digest):
        assert struct.unpack_from("<Q", transport, offset) == (64,)
        assert transport[offset + 8 : offset + 72] == digest.encode()
        offset += 72
    assert struct.unpack_from("<Q", transport, offset) == (2,)
    offset += 8
    for image in images:
        assert struct.unpack_from("<Q", transport, offset) == (len(image.device_image),)
        offset += 8
        assert transport[offset : offset + len(image.device_image)] == image.device_image
        offset += len(image.device_image)
        assert struct.unpack_from("<Q", transport, offset) == (64,)
        assert transport[offset + 8 : offset + 72] == hashlib.sha256(image.device_image).hexdigest().encode()
        offset += 72
    assert struct.unpack_from("<QqQ", transport, offset) == (2, 0, image_by_launch[0])
    assert transport[-64:] == hashlib.sha256(transport[:-72]).hexdigest().encode()


def test_native_transport_rejects_images_for_another_device_target() -> None:
    recipe = _recipe()
    foreign = replace(_artifact(recipe), request=replace(_request(), target="sm100a"))
    artifact = CuteMultiImagePipelineArtifact(_request(), (_artifact(recipe), foreign), recipe, (0, 1))
    with pytest.raises(ValueError, match="match the pipeline device target"):
        _ = artifact.native_transport


def test_launch_recipe_canonicalizes_named_storage_and_attributes_but_preserves_launch_abi() -> None:
    recipe = _recipe()
    prepare, finish = recipe.launches
    equivalent = CutePipelineLaunchRecipe(
        schema_version=2,
        slots=tuple(reversed(recipe.slots)),
        launches=(
            replace(prepare, function_attributes=tuple(reversed(prepare.function_attributes))),
            replace(
                finish,
                arguments=(
                    *finish.arguments[:4],
                    replace(
                        cast(CutePipelineStructuredArgument, finish.arguments[4]),
                        fields=tuple(reversed(cast(CutePipelineStructuredArgument, finish.arguments[4]).fields)),
                    ),
                    finish.arguments[5],
                ),
            ),
        ),
    )

    assert equivalent.canonical_json == recipe.canonical_json
    assert equivalent.digest == recipe.digest

    reordered = replace(recipe, launches=tuple(reversed(recipe.launches)))
    assert reordered.digest != recipe.digest

    changed_source_coverage = replace(
        recipe,
        launches=(replace(prepare, source_tasks=(0, 1)), finish),
    )
    assert changed_source_coverage.digest != recipe.digest

    changed_source_role = replace(
        recipe,
        launches=(replace(prepare, source_task_roles=(CuteSourceTaskRole.STATE_ROOT,)), finish),
    )
    assert changed_source_role.digest != recipe.digest

    changed_arguments = replace(
        recipe,
        launches=(
            replace(
                prepare,
                arguments=(
                    *prepare.arguments[:2],
                    replace(cast(CutePipelineScalarArgument, prepare.arguments[2]), value=0.25),
                    prepare.arguments[3],
                ),
            ),
            finish,
        ),
    )
    assert changed_arguments.digest != recipe.digest


def test_artifact_manifest_seals_one_fatbin_and_native_launch_cache_identity(tmp_path) -> None:
    recipe = _recipe()
    artifact = _artifact(recipe)
    manifest_path, object_path, fatbin_path, mlir_path = artifact.write(tmp_path, stem="pipeline")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 3
    assert manifest["device_image"]["format"] == "cuda-fatbin"
    assert manifest["device_image"]["kernels"] == ["_Z7preparePKvPv", "_Z6finishPKvPvS0_"]
    assert [launch["name"] for launch in manifest["launch_recipe"]["launches"]] == ["prepare", "finish"]
    assert [launch["source_tasks"] for launch in manifest["launch_recipe"]["launches"]] == [[0], [1]]
    assert manifest["launch_recipe"]["launches"][0]["arguments"][3] == {
        "access": "read",
        "alignment": 64,
        "box_shape": [16, 2],
        "byte_offset": 0,
        "element_strides": [1, 1],
        "element_type": "bf16",
        "global_byte_strides": [32],
        "global_shape": [16, 2],
        "interleave": "none",
        "kind": "tensor_map",
        "l2_promotion": "bytes_128",
        "name": "source_tensor_map",
        "oob_fill": "none",
        "slot": "source",
        "swizzle": "bytes_128",
    }
    structured_record = manifest["launch_recipe"]["launches"][1]["arguments"][4]
    assert structured_record["kind"] == "structured_bytes"
    assert structured_record["gap_fill"] == "zero"
    assert [field["destination_byte_offset"] for field in structured_record["fields"]] == [0, 8, 24]
    structured = cast(CutePipelineStructuredArgument, recipe.launches[1].arguments[4])
    expected_template = bytearray(32)
    expected_template[0:4] = b"\x01\x02\x03\x04"
    expected_template[24:28] = b"\x08\x00\x00\x00"
    assert structured.zero_filled_template == bytes(expected_template)
    assert manifest["launch_recipe"]["launches"][1]["arguments"][5] == {
        "alignment": 2,
        "data_hex": "0201",
        "kind": "packed_bytes",
        "name": "epilogue",
    }
    assert "tma_descriptor" not in artifact.canonical_manifest_json
    assert manifest["native_cache_identity"] == artifact.native_cache_identity
    assert object_path.read_bytes() == b"host-object"
    assert fatbin_path.read_bytes() == b"fatbin"
    assert mlir_path.read_text() == "module {}"

    assert _artifact(recipe, device_image=b"different-fatbin").native_cache_identity != artifact.native_cache_identity


def test_structural_launch_roles_bind_source_tasks_without_launch_position_knowledge() -> None:
    prepare, finish = _launches()
    unbound = replace(
        _recipe(),
        launches=(
            replace(
                prepare,
                source_task_roles=(CuteSourceTaskRole.STATE_ROOT,),
                source_tasks=(),
            ),
            replace(
                finish,
                source_task_roles=(CuteSourceTaskRole.STATE_CARRY, CuteSourceTaskRole.STATE_TERMINAL),
                source_tasks=(),
            ),
        ),
    )

    bound = bind_launch_source_tasks(
        unbound,
        {
            CuteSourceTaskRole.STATE_TERMINAL: (8,),
            CuteSourceTaskRole.STATE_ROOT: (3,),
            CuteSourceTaskRole.STATE_CARRY: (5,),
        },
    )

    assert tuple(launch.source_tasks for launch in bound.launches) == ((3,), (5, 8))
    assert tuple(launch.source_task_roles for launch in bound.launches) == (
        (CuteSourceTaskRole.STATE_ROOT,),
        (CuteSourceTaskRole.STATE_CARRY, CuteSourceTaskRole.STATE_TERMINAL),
    )


def test_structural_event_covers_bind_exact_source_events_without_using_task_list_order() -> None:
    prepare, finish = _launches()
    owner = CuteAffineMap(1, (affine_constant(0), affine_constant(0), affine_constant(0)))
    phase_zero = CuteAffineMap(1, (affine_constant(0),))
    phase_one = CuteAffineMap(1, (affine_constant(1),))
    recipe = replace(
        _recipe(),
        launches=(
            replace(prepare, source_task_roles=(CuteSourceTaskRole.STATE_ROOT,), source_tasks=()),
            replace(
                finish,
                source_task_roles=(CuteSourceTaskRole.STATE_CARRY, CuteSourceTaskRole.STATE_TERMINAL),
                source_tasks=(),
            ),
        ),
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_CARRY,
                CuteEventExecutionPoint("prepare", 0),
                CuteEventExecutionPoint("finish", 0),
            ),
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_CARRY,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint("finish", 0),
                CuteEventExecutionPoint("finish", 1),
                producer_owner=owner,
                consumer_owner=owner,
                producer_time=phase_zero,
                consumer_time=phase_one,
            ),
        ),
    )
    recipe = bind_launch_source_tasks(
        recipe,
        {
            CuteSourceTaskRole.STATE_ROOT: (3,),
            CuteSourceTaskRole.STATE_CARRY: (5,),
            CuteSourceTaskRole.STATE_TERMINAL: (8,),
        },
    )

    bound = bind_event_realizations(
        recipe,
        source_events=((0, 3, 5, 1, None, None), (1, 5, 8, 1, None, None)),
        source_task_roles={
            3: CuteSourceTaskRole.STATE_ROOT,
            5: CuteSourceTaskRole.STATE_CARRY,
            8: CuteSourceTaskRole.STATE_TERMINAL,
        },
    )

    assert tuple(
        (realization.event, realization.kind, realization.covering_launch_order)
        for realization in bound.event_realizations
    ) == (
        (0, CuteEventRealizationKind.STREAM_ORDER, (0, 0, 1, 0)),
        (1, CuteEventRealizationKind.PROGRAM_ORDER, (1, 0, 1, 1)),
    )
    assert len(bound.event_realizations[0].cover_pieces) == 1
    assert len(bound.event_realizations[1].cover_pieces) == 1
    assert all(realization.mechanism is None for realization in bound.event_realizations)


def test_algebraic_refinement_erases_a_source_event_without_claiming_phase_order() -> None:
    prepare, finish = _launches()
    owner = CuteAffineMap(1, (affine_constant(0), affine_constant(0), affine_constant(0)))
    phase = CuteAffineMap(1, (affine_constant(0),))
    recipe = replace(
        _recipe(),
        launches=(
            replace(prepare, source_task_roles=(CuteSourceTaskRole.STATE_ROOT,), source_tasks=()),
            replace(
                finish,
                source_task_roles=(CuteSourceTaskRole.STATE_ROOT, CuteSourceTaskRole.STATE_TERMINAL),
                source_tasks=(),
            ),
        ),
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_TERMINAL,
                CuteEventExecutionPoint("finish", 0),
                CuteEventExecutionPoint("finish", 0),
                CuteEventRealizationKind.ALGEBRAIC_REFINEMENT,
                producer_owner=owner,
                consumer_owner=owner,
                producer_time=phase,
                consumer_time=phase,
            ),
        ),
    )
    recipe = bind_launch_source_tasks(
        recipe,
        {
            CuteSourceTaskRole.STATE_ROOT: (3,),
            CuteSourceTaskRole.STATE_TERMINAL: (8,),
        },
    )

    bound = bind_event_realizations(
        recipe,
        source_events=((0, 3, 8, 1, None, None),),
        source_task_roles={
            3: CuteSourceTaskRole.STATE_ROOT,
            8: CuteSourceTaskRole.STATE_TERMINAL,
        },
    )

    assert len(bound.event_realizations) == 1
    assert bound.event_realizations[0].kind is CuteEventRealizationKind.ALGEBRAIC_REFINEMENT
    assert len(bound.event_realizations[0].cover_pieces) == 1
    transport = _artifact(bound).native_transport
    assert transport.startswith(b"SHUTGPU\0")
    assert bound.event_realizations[0].mechanism is None
    assert int.from_bytes(transport[8:12], "little") == 12


def test_named_source_task_phases_erase_to_strict_stream_order() -> None:
    prepare, finish = _launches()
    recipe = replace(
        _recipe(),
        launches=(
            replace(prepare, source_tasks=(5,), source_phase="partial"),
            replace(finish, source_tasks=(5,), source_phase="merge"),
        ),
        structural_event_covers=(),
    )

    bound = bind_event_realizations(
        recipe,
        source_events=((0, 5, 5, 1, "partial", "merge"),),
        source_task_roles={5: CuteSourceTaskRole.STATE_CARRY},
    )

    assert len(bound.event_realizations) == 1
    assert bound.event_realizations[0].kind is CuteEventRealizationKind.STREAM_ORDER
    assert bound.event_realizations[0].covering_launch_order == (0, 0, 1, 0)


def test_named_source_task_phase_event_rejects_reverse_launch_order() -> None:
    prepare, finish = _launches()
    recipe = replace(
        _recipe(),
        launches=(
            replace(prepare, source_tasks=(5,), source_phase="partial"),
            replace(finish, source_tasks=(5,), source_phase="merge"),
        ),
        structural_event_covers=(),
    )

    with pytest.raises(ValueError, match="does not follow physical launch order"):
        bind_event_realizations(
            recipe,
            source_events=((0, 5, 5, 1, "merge", "partial"),),
            source_task_roles={5: CuteSourceTaskRole.STATE_CARRY},
        )


def test_real_native_phase_event_rejects_a_reversed_physical_phase_binding() -> None:
    request = decode_gpu_physical_artifact_request(_NATIVE_SCALAR_VALUE_AND_GRAD_FIXTURE.read_text())
    phase_event = request.event_relations[-1]
    prepare, finish = _launches()
    recipe = replace(
        _recipe(),
        launches=(
            replace(prepare, source_tasks=(phase_event.producer_task,), source_phase="partial"),
            replace(finish, source_tasks=(phase_event.consumer_task,), source_phase="merge"),
        ),
        structural_event_covers=(),
    )

    with pytest.raises(ValueError, match="does not follow physical launch order"):
        bind_event_realizations(
            recipe,
            source_events=(
                (
                    0,
                    phase_event.producer_task,
                    phase_event.consumer_task,
                    len(phase_event.relation_domain_shape),
                    phase_event.consumer_phase,
                    phase_event.producer_phase,
                ),
            ),
            source_task_roles={phase_event.producer_task: CuteSourceTaskRole.STATE_CARRY},
        )


@pytest.mark.parametrize(
    "kind",
    (
        CuteEventRealizationKind.MBARRIER,
        CuteEventRealizationKind.PIPELINE,
        CuteEventRealizationKind.QUEUE,
    ),
)
def test_native_v11_carries_bounded_physical_event_mechanisms(kind: CuteEventRealizationKind) -> None:
    relation_owner = CuteAffineMap(1, (affine_constant(0), affine_constant(0), affine_constant(0)))
    relation_time = CuteAffineMap(1, (affine_constant(0),))
    piece = CuteEventCoverPiece(
        None,
        CuteEventCoverEndpoint(0, relation_owner, relation_time),
        CuteEventCoverEndpoint(1, relation_owner, relation_time),
    )
    mechanism = CuteEventMechanismAllocation(
        allocation=0,
        scope=CuteEventPhysicalScope.CLUSTER,
        capacity=4,
        slot=CuteAffineMap(1, (affine_constant(0),)),
        generation_phase=CuteAffineMap(1, (affine_constant(0), affine_constant(1))),
        generation_count=2,
        arrival_count=8,
        transaction_bytes=128,
    )
    realization = CuteEventRealization(0, kind, (), (piece,), mechanism)
    recipe = replace(_recipe(), event_realizations=(realization,))

    assert recipe.canonical_record["event_realizations"][0]["mechanism"] == {
        "allocation": 0,
        "arrival_count": 8,
        "capacity": 4,
        "generation_count": 2,
        "generation_phase": mechanism.generation_phase.canonical_record,
        "scope": "cluster",
        "slot": mechanism.slot.canonical_record,
        "transaction_bytes": 128,
    }
    transport = _artifact(recipe).native_transport
    assert int.from_bytes(transport[8:12], "little") == 12


def test_physical_event_provenance_is_distinct_from_source_event_realizations() -> None:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "realizations": (
            {
                "plan": "internal_ready",
                "event": {
                    "required_dependence": {
                        "relation": {"pairs": ({"source": (0,), "target": (0,)},)},
                    },
                },
                "kind": "physical",
                "mechanism": "bounded pipeline",
            },
        ),
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    recipe = bind_physical_event_provenance(_recipe(), payload)

    assert recipe.event_realizations == ()
    provenance = recipe.canonical_record["physical_event_provenance"]
    assert provenance["physical_schedule_fingerprint"] == payload["fingerprint"]
    assert provenance["schedule"] == payload
    transport = _artifact(recipe).native_transport
    assert int.from_bytes(transport[8:12], "little") == 12


def test_derived_physical_event_schedule_binds_its_schedule_identity() -> None:
    schedule = derive_streaming_physical_event_schedule(
        StreamingContractFoldDescriptor(
            first_contract_name="first_contract",
            fold_update_name="state_update",
            second_contract_name="second_contract",
            finalize_name="finalize",
            fold_extent=512,
            resident_tile_size=128,
            streamed_tile_size=64,
            pipeline_depth=3,
            resident_reduction_dimension=128,
            streamed_reduction_dimension=128,
            output_dimension=128,
            element_bytes=2,
        )
    )
    record = streaming_physical_event_schedule_record(schedule)
    recipe = bind_physical_event_provenance(_recipe(), record)

    assert recipe.physical_event_provenance is not None
    assert recipe.physical_event_provenance["physical_schedule_fingerprint"] == schedule.fingerprint
    assert recipe.physical_event_provenance["schedule"] == record


def test_combined_internal_event_schedules_bind_through_the_same_envelope() -> None:
    descriptor = StreamingContractFoldDescriptor(
        first_contract_name="first_contract",
        fold_update_name="state_update",
        second_contract_name="second_contract",
        finalize_name="finalize",
        fold_extent=512,
        resident_tile_size=128,
        streamed_tile_size=64,
        pipeline_depth=3,
        resident_reduction_dimension=128,
        streamed_reduction_dimension=128,
        output_dimension=128,
        element_bytes=2,
    )
    first = streaming_physical_event_schedule_record(derive_streaming_physical_event_schedule(descriptor))
    second = streaming_physical_event_schedule_record(
        derive_streaming_physical_event_schedule(replace(descriptor, pipeline_depth=2))
    )
    combined = combine_physical_event_schedules(first, second)
    recipe = bind_physical_event_provenance(_recipe(), combined)

    assert recipe.physical_event_provenance is not None
    assert recipe.physical_event_provenance["physical_schedule_fingerprint"] == combined["fingerprint"]
    assert recipe.physical_event_provenance["schedule"] == combined


def test_event_realizations_enforce_mechanism_and_erasure_boundaries() -> None:
    relation_owner = CuteAffineMap(1, (affine_constant(0), affine_constant(0), affine_constant(0)))
    relation_time = CuteAffineMap(1, (affine_constant(0),))
    piece = CuteEventCoverPiece(
        None,
        CuteEventCoverEndpoint(0, relation_owner, relation_time),
        CuteEventCoverEndpoint(1, relation_owner, relation_time),
    )

    with pytest.raises(ValueError, match="require a physical mechanism allocation"):
        CuteEventRealization(0, CuteEventRealizationKind.MBARRIER, (), (piece,))

    mechanism = CuteEventMechanismAllocation(
        0,
        CuteEventPhysicalScope.DEVICE,
        1,
        CuteAffineMap(1, (affine_constant(0),)),
        CuteAffineMap(1, (affine_constant(0),)),
        1,
        1,
    )
    with pytest.raises(ValueError, match="cannot allocate a physical mechanism"):
        CuteEventRealization(
            0,
            CuteEventRealizationKind.STREAM_ORDER,
            (0, 0, 1, 0),
            (piece,),
            mechanism,
        )


def test_structural_launch_roles_reject_uncovered_source_tasks() -> None:
    prepare, finish = _launches()
    unbound = replace(
        _recipe(),
        launches=(
            replace(prepare, source_task_roles=(CuteSourceTaskRole.STATE_ROOT,), source_tasks=()),
            replace(finish, source_task_roles=(CuteSourceTaskRole.STATE_CARRY,), source_tasks=()),
        ),
    )

    with pytest.raises(ValueError, match=r"do not cover source tasks \(8,\)"):
        bind_launch_source_tasks(
            unbound,
            {
                CuteSourceTaskRole.STATE_ROOT: (3,),
                CuteSourceTaskRole.STATE_CARRY: (5,),
                CuteSourceTaskRole.STATE_TERMINAL: (8,),
            },
        )


def test_launch_recipe_rejects_storage_access_that_cannot_implement_the_public_boundary() -> None:
    prepare, finish = _launches()
    invalid_prepare = replace(
        prepare,
        arguments=(
            replace(
                cast(CutePipelineSlotArgument, prepare.arguments[0]),
                access=CutePipelineSlotAccess.WRITE,
            ),
            *prepare.arguments[1:],
        ),
    )

    with pytest.raises(ValueError, match="public input slot 'source' may only be read"):
        CutePipelineLaunchRecipe(schema_version=2, slots=_slots(), launches=(invalid_prepare, finish))


def test_launch_recipe_rejects_unknown_or_out_of_bounds_storage_references() -> None:
    prepare, finish = _launches()
    with pytest.raises(ValueError, match="references unknown slot 'missing'"):
        CutePipelineLaunchRecipe(
            schema_version=2,
            slots=_slots(),
            launches=(
                replace(
                    prepare,
                    arguments=(
                        replace(cast(CutePipelineSlotArgument, prepare.arguments[0]), slot="missing"),
                        *prepare.arguments[1:],
                    ),
                ),
                finish,
            ),
        )

    with pytest.raises(ValueError, match="address range exceeds slot 'source'"):
        CutePipelineLaunchRecipe(
            schema_version=2,
            slots=_slots(),
            launches=(
                replace(
                    prepare,
                    arguments=(
                        replace(cast(CutePipelineSlotArgument, prepare.arguments[0]), byte_offset=64),
                        *prepare.arguments[1:],
                    ),
                ),
                finish,
            ),
        )


def test_launch_recipe_rejects_malformed_physical_launch_contracts() -> None:
    tensor_map = cast(CutePipelineTensorMapArgument, _launches()[0].arguments[3])
    with pytest.raises(ValueError, match="requires one global byte stride"):
        replace(tensor_map, global_byte_strides=())

    with pytest.raises(ValueError, match="block exceeds CUDA thread limits"):
        replace(_launches()[0], block=(1024, 2, 1))

    with pytest.raises(ValueError, match="requests more dynamic shared memory"):
        replace(
            _launches()[0],
            dynamic_shared_memory_bytes=65537,
        )


def test_dynamic_tensor_map_binding_and_geometry_participate_in_recipe_identity() -> None:
    recipe = _recipe()
    prepare, finish = recipe.launches
    tensor_map = cast(CutePipelineTensorMapArgument, prepare.arguments[3])
    result_bound = replace(tensor_map, slot="result", access=CutePipelineSlotAccess.WRITE)
    different_shape = replace(
        tensor_map,
        global_shape=(8, 4),
        global_byte_strides=(16,),
        box_shape=(8, 4),
    )

    result_bound_recipe = replace(
        recipe,
        launches=(replace(prepare, arguments=(*prepare.arguments[:3], result_bound)), finish),
    )
    different_shape_recipe = replace(
        recipe,
        launches=(replace(prepare, arguments=(*prepare.arguments[:3], different_shape)), finish),
    )

    assert result_bound_recipe.digest != recipe.digest
    assert different_shape_recipe.digest != recipe.digest
    assert json.loads(recipe.canonical_json)["launches"][0]["arguments"][3]["slot"] == "source"


def test_structured_argument_relocations_participate_in_identity_and_obey_slot_ranges() -> None:
    recipe = _recipe()
    prepare, finish = recipe.launches
    structure = cast(CutePipelineStructuredArgument, finish.arguments[4])
    source_address = cast(CutePipelineStructuredSlotAddressField, structure.fields[0])
    changed_structure = replace(
        structure,
        fields=(replace(source_address, slot_byte_offset=4), *structure.fields[1:]),
    )
    changed_recipe = replace(
        recipe,
        launches=(
            prepare,
            replace(finish, arguments=(*finish.arguments[:4], changed_structure, finish.arguments[5])),
        ),
    )
    assert changed_recipe.digest != recipe.digest

    invalid_structure = replace(
        structure,
        fields=(replace(source_address, slot_byte_offset=64), *structure.fields[1:]),
    )
    with pytest.raises(ValueError, match="address range exceeds slot 'source'"):
        replace(
            recipe,
            launches=(
                prepare,
                replace(finish, arguments=(*finish.arguments[:4], invalid_structure, finish.arguments[5])),
            ),
        )

    write_structure = replace(
        structure,
        fields=(replace(source_address, access=CutePipelineSlotAccess.WRITE), *structure.fields[1:]),
    )
    with pytest.raises(ValueError, match="public input slot 'source' may only be read"):
        replace(
            recipe,
            launches=(
                prepare,
                replace(finish, arguments=(*finish.arguments[:4], write_structure, finish.arguments[5])),
            ),
        )


def test_structured_argument_rejects_misaligned_or_overlapping_pointer_fields() -> None:
    with pytest.raises(ValueError, match="destination must be pointer-aligned"):
        CutePipelineStructuredSlotAddressField(
            "pointer",
            slot="source",
            access=CutePipelineSlotAccess.READ,
            slot_byte_offset=0,
            destination_byte_offset=4,
        )

    pointer = CutePipelineStructuredSlotAddressField(
        "pointer",
        slot="source",
        access=CutePipelineSlotAccess.READ,
        slot_byte_offset=0,
        destination_byte_offset=8,
    )
    with pytest.raises(ValueError, match="fields 'bytes' and 'pointer' overlap"):
        CutePipelineStructuredArgument(
            "parameters",
            size=24,
            alignment=8,
            fields=(
                CutePipelineStructuredImmediateField("bytes", 4, b"\x00" * 8),
                pointer,
            ),
        )


def test_artifact_rejects_launch_symbols_absent_from_its_fatbin_manifest() -> None:
    recipe = _recipe()
    with pytest.raises(ValueError, match="references symbols absent from its device image"):
        replace(_artifact(recipe), kernel_names=("_Z7preparePKvPv",))


def test_native_transport_requires_source_task_coverage_for_every_launch() -> None:
    prepare, finish = _launches()
    artifact = _artifact(replace(_recipe(), launches=(replace(prepare, source_tasks=()), finish)))

    with pytest.raises(ValueError, match="source-task coverage for every physical launch"):
        _ = artifact.native_transport


def test_physical_request_decodes_typed_shapes_and_affine_index_relations() -> None:
    tensor_type = {"kind": "ranked_tensor", "shape": [1, 8, 4, 2048, 128], "element_type": "bf16"}
    scalar_type = {"kind": "scalar", "element_type": "f32"}
    affine_map = {
        "kind": "affine_map",
        "dimensions": 3,
        "symbols": 0,
        "results": [
            {"kind": "dimension", "position": 0},
            {
                "kind": "floor_divide",
                "left": {"kind": "dimension", "position": 1},
                "right": {"kind": "constant", "value": 4},
            },
        ],
    }

    def operation(
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        operands: list[int] | None = None,
        results: list[dict[str, object]] | None = None,
        operations: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        regions: list[object] = []
        if operations is not None:
            regions = [[{"arguments": [], "operations": operations}]]
        return {
            "name": name,
            "attributes": attributes or {},
            "operands": operands or [],
            "results": results or [],
            "regions": regions,
        }

    contract = operation(
        "shuttle.contract",
        attributes={"indexing_maps": [affine_map], "reduction_dimensions": [2]},
        operands=[0],
        results=[{"id": 1, "type": scalar_type}],
    )
    event_metadata = {
        "event": 0,
        "producer_task": 0,
        "consumer_task": 1,
        "relation_domain_shape": [8, 4],
        "relation_predicate": {
            "kind": "integer_set",
            "dimensions": 2,
            "symbols": 0,
            "constraints": [
                {
                    "kind": "add",
                    "left": {"kind": "dimension", "position": 0},
                    "right": {"kind": "constant", "value": -3},
                }
            ],
            "equalities": [False],
        },
        "producer_indices": {
            **affine_map,
            "dimensions": 2,
            "results": [
                {"kind": "dimension", "position": 0},
                {"kind": "dimension", "position": 1},
            ],
        },
        "consumer_indices": {
            **affine_map,
            "dimensions": 2,
            "results": [{"kind": "dimension", "position": 1}],
        },
        "storage_capacity": 4,
        "storage_slot": {
            "kind": "affine_map",
            "dimensions": 2,
            "symbols": 0,
            "results": [
                {
                    "kind": "modulo",
                    "left": {"kind": "dimension", "position": 0},
                    "right": {"kind": "constant", "value": 4},
                }
            ],
        },
        "generation_index": {
            "kind": "affine_map",
            "dimensions": 2,
            "symbols": 0,
            "results": [
                {
                    "kind": "floor_divide",
                    "left": {"kind": "dimension", "position": 0},
                    "right": {"kind": "constant", "value": 4},
                }
            ],
        },
        "generations": 2,
        "notify_count": 1,
        "scope": {"kind": "mlir_attribute", "value": "#shuttle.event_scope<device>"},
    }
    event_realization = operation(
        "shuttle.event_realization",
        attributes={
            "event": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.event_realization_kind<imperative_wait_notify>"},
            "covering_launch_order": [],
        },
    )
    event_notify = operation(
        "shuttle.imperative_event_notify",
        attributes={**event_metadata, "release": True},
    )
    event_wait = operation(
        "shuttle.imperative_event_wait",
        attributes={**event_metadata, "acquire": True},
    )
    event_plan_attributes = {
        "schema_version": 1,
        "source_schedule_fingerprint": "b" * 64,
        "physical_launch_recipe_fingerprint": "c" * 64,
        "fingerprint": "d" * 64,
    }
    request = {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "a" * 64,
        "binding_indices": [0],
        "region_result_indices": [None],
        "reuse_groups": [0],
        "region": {
            **operation("shuttle.region", operands=[0], results=[{"id": 2, "type": tensor_type}]),
            "regions": [[{"arguments": [{"id": 0, "type": tensor_type}], "operations": [contract]}]],
        },
        "materialization": operation("shuttle.materialization_plan", operations=[]),
        "schedule": operation("shuttle.schedule_plan", operations=[]),
        "event_lowering_plan": operation(
            "shuttle.event_lowering_plan",
            attributes=event_plan_attributes,
            operations=[
                event_realization,
                event_notify,
                event_wait,
                operation("shuttle.event_lowering_plan_yield"),
            ],
        ),
    }

    decoded = decode_gpu_physical_artifact_request(json.dumps(request))
    decoded_contract = tuple(decoded.region.walk())[1]

    assert decoded.region.body().arguments[0].type.shape == (1, 8, 4, 2048, 128)
    assert decoded.binding_indices == (0,)
    assert decoded.region_result_indices == (None,)
    assert decoded.reuse_groups == (0,)
    assert decoded.fingerprint == "a" * 64
    assert decoded_contract.results[0].type.element_type == "f32"
    assert decoded_contract.attributes["indexing_maps"] == [affine_map]
    assert decoded.event_lowering_plan.attributes == event_plan_attributes
    decoded_event_operations = decoded.event_lowering_plan.body().operations
    assert tuple(operation.name for operation in decoded_event_operations) == (
        "shuttle.event_realization",
        "shuttle.imperative_event_notify",
        "shuttle.imperative_event_wait",
        "shuttle.event_lowering_plan_yield",
    )
    assert decoded_event_operations[1].attributes == {**event_metadata, "release": True}
    assert decoded_event_operations[2].attributes == {**event_metadata, "acquire": True}
    assert len(decoded.event_requirements) == 1
    requirement = decoded.event_requirements[0]
    assert requirement.ordinal == 0
    assert requirement.relation_domain_shape == (8, 4)
    assert requirement.relation_predicate is not None
    assert requirement.producer_indices.dimensions == 2
    assert requirement.consumer_indices.dimensions == 2
    assert requirement.storage_slot.dimensions == 2
    assert requirement.generation_index.dimensions == 2
    assert requirement.storage_capacity == 4
    assert requirement.generations == 2
    assert requirement.notify_count == 1
    assert requirement.scope == "device"
    assert requirement.release is True
    assert requirement.acquire is True

    mismatched_pair = json.loads(json.dumps(request))
    mismatched_pair["event_lowering_plan"]["regions"][0][0]["operations"][2]["attributes"]["notify_count"] = 2
    with pytest.raises(ValueError, match="notify and wait metadata must match exactly"):
        decode_gpu_physical_artifact_request(json.dumps(mismatched_pair))

    mismatched_realization = json.loads(json.dumps(request))
    mismatched_realization["event_lowering_plan"]["regions"][0][0]["operations"][0]["attributes"]["kind"] = {
        "kind": "mlir_attribute",
        "value": "#shuttle.event_realization_kind<stream_order_erased>",
    }
    mismatched_realization["event_lowering_plan"]["regions"][0][0]["operations"][0]["attributes"][
        "covering_launch_order"
    ] = [0, 0, 1, 0]
    with pytest.raises(ValueError, match="does not match an imperative realization"):
        decode_gpu_physical_artifact_request(json.dumps(mismatched_realization))

    erased = json.loads(json.dumps(mismatched_realization))
    del erased["event_lowering_plan"]["regions"][0][0]["operations"][1:3]
    assert decode_gpu_physical_artifact_request(json.dumps(erased)).event_requirements == ()


def test_physical_request_requires_schema_v3_event_lowering_plan() -> None:
    operation = {
        "name": "shuttle.region",
        "attributes": {},
        "operands": [],
        "results": [],
        "regions": [],
    }
    request = {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "a" * 64,
        "binding_indices": [],
        "region_result_indices": [],
        "reuse_groups": [],
        "region": operation,
        "materialization": {**operation, "name": "shuttle.materialization_plan"},
        "schedule": {**operation, "name": "shuttle.schedule_plan"},
        "event_lowering_plan": {**operation, "name": "shuttle.event_lowering_plan"},
    }

    old_schema = {**request, "schema_version": 2}
    with pytest.raises(ValueError, match="unsupported Shuttle physical artifact request"):
        decode_gpu_physical_artifact_request(json.dumps(old_schema))

    missing_event_plan = dict(request)
    del missing_event_plan["event_lowering_plan"]
    with pytest.raises(ValueError, match="unsupported Shuttle physical artifact request"):
        decode_gpu_physical_artifact_request(json.dumps(missing_event_plan))

    wrong_event_plan = {
        **request,
        "event_lowering_plan": {**operation, "name": "shuttle.schedule_plan"},
    }
    with pytest.raises(ValueError, match=r"expected shuttle\.event_lowering_plan"):
        decode_gpu_physical_artifact_request(json.dumps(wrong_event_plan))


def test_launch_event_schedule_covers_multiphase_and_reused_slot_hazards() -> None:
    def launch(name: str, accesses: tuple[tuple[str, CutePipelineSlotAccess], ...]) -> CuteKernelLaunch:
        return CuteKernelLaunch(
            name=name,
            kernel_symbol="shared",
            grid=(1, 1, 1),
            block=(32, 1, 1),
            dynamic_shared_memory_bytes=0,
            function_attributes=(),
            arguments=tuple(
                CutePipelineSlotArgument(f"pointer_{i}", slot, access) for i, (slot, access) in enumerate(accesses)
            ),
        )

    read, write = CutePipelineSlotAccess.READ, CutePipelineSlotAccess.WRITE
    launches = (
        launch("partial", (("partial", write),)),
        launch("unrelated", (("other", write),)),
        launch("merge", (("partial", read), ("state", write))),
        launch("consumer", (("state", read),)),
        launch("reuse", (("partial", write), ("state", write))),
        launch("alias_reader", (("partial_alias", read),)),
    )
    slots = tuple(
        CutePipelineSlot(
            name,
            CutePipelineElementType.F32,
            (1,),
            CutePipelineSlotKind.SCRATCH,
            16,
            reuse_group=0 if name in {"partial", "partial_alias"} else None,
        )
        for name in ("partial", "other", "state", "partial_alias")
    )
    schedule = launch_dependency_event_schedule(launches, slots=slots)
    assert schedule["predecessor_pairs"] == (
        ((0,), (2,)),
        ((0,), (4,)),
        ((2,), (3,)),
        ((2,), (4,)),
        ((3,), (4,)),
        ((4,), (5,)),
    )
    assert schedule["realizations"][0]["kind"] == "erased_stream_order"
    assert schedule["realizations"][0]["allocation_ids"] == ()
    assert schedule["allocations"] == ()


def test_native_transport_separates_explicit_reuse_from_slot_ordinals() -> None:
    source, state, result, workspace = _slots()
    slots = (
        source,
        state,
        result,
        replace(workspace, reuse_group=4),
        replace(workspace, name="independent"),
        replace(workspace, name="later_alias", reuse_group=4),
    )
    prepare, finish = _launches()
    launches = (
        replace(
            prepare,
            arguments=(
                *prepare.arguments,
                CutePipelineSlotArgument("independent", "independent", CutePipelineSlotAccess.WRITE),
            ),
        ),
        replace(
            finish,
            arguments=(
                *finish.arguments,
                CutePipelineSlotArgument("independent", "independent", CutePipelineSlotAccess.READ),
            ),
        ),
        replace(
            prepare,
            name="reuse",
            source_tasks=(2,),
            arguments=(CutePipelineSlotArgument("later_alias", "later_alias", CutePipelineSlotAccess.WRITE),),
        ),
    )
    transport = _artifact(replace(_recipe(), slots=slots, launches=launches)).native_transport
    # Version 12 ends with fixed-layout slot records and the length-prefixed SHA256.
    cursor = len(transport) - 72
    reuse_groups = []
    for slot in reversed(slots):
        reuse_groups.append(struct.unpack_from("<q", transport, cursor - 17)[0])
        cursor -= 77 + 16 * len(slot.shape)
    assert tuple(reversed(reuse_groups)) == (0, 1, 2, 3, 4, 3)
