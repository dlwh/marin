# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture a corrected physical forward component for native boundary diagnosis."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from cute_launch_argument_capture import capture_cute_launch_argument_report

from tile_lifetime.backends.h100.cute_generic_tensor_program import CompiledGenericKernelComposition
from tile_lifetime.backends.h100.cute_launch_abi import validate_cute_launch_abi


def export_first_forward_fixture(
    executable: CompiledGenericKernelComposition,
    inputs: tuple[torch.Tensor, ...],
    directory: Path,
    maximum_absolute_error: float,
) -> dict:
    """Export actual canonical inputs and outputs, including the physical layout copy."""
    environment = executable.execute_environment(*inputs)
    component = next(component for component in executable.components if component.physical_component is not None)
    physical = component.physical_component
    assert physical is not None
    artifact = physical.artifact
    physical_inputs = tuple(environment[value] for value in physical.live_inputs)
    physical_outputs = tuple(torch.empty_like(environment[value]) for value in physical.live_outputs)
    known_addresses = {
        role: environment[value].data_ptr()
        for role, value in physical.role_values.items()
        if value in physical.live_inputs
    }
    output_by_value = dict(zip(physical.live_outputs, physical_outputs, strict=True))
    known_addresses.update(
        (role, output_by_value[value].data_ptr())
        for role, value in physical.role_values.items()
        if value in physical.output_transports
    )
    physical(physical_inputs, physical_outputs)
    torch.cuda.synchronize()
    argument_report = capture_cute_launch_argument_report(
        lambda: physical(physical_inputs, physical_outputs), artifact.launch_recipe, known_addresses
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    (directory.parent / "first-forward-arguments.json").write_text(json.dumps(argument_report, indent=2) + "\n")
    validate_cute_launch_abi(artifact)
    directory.mkdir(parents=True, exist_ok=True)
    canonical_component = next(item for item in executable.composition.components if item.id == component.component_id)
    artifact = replace(
        artifact,
        launch_recipe=replace(
            artifact.launch_recipe,
            launches=tuple(
                replace(launch, source_task_roles=(), source_tasks=canonical_component.task_ids)
                for launch in artifact.launch_recipe.launches
            ),
        ),
    )
    transport = artifact.native_transport
    (directory / "bundle.bin").write_bytes(transport)
    manifest = {
        "bundle": "bundle.bin",
        "inputs": [],
        "expected_outputs": [],
        "max_absolute_error": maximum_absolute_error,
    }
    metadata = {}
    for key, attribute in (("inputs", "input_index"), ("expected_outputs", "output_index")):
        slots = sorted(
            (slot for slot in artifact.launch_recipe.slots if getattr(slot, attribute) is not None),
            key=lambda slot: getattr(slot, attribute),
        )
        for index, slot in enumerate(slots):
            tensor = environment[physical.role_values[slot.name]]
            if tuple(tensor.shape) != slot.shape or not tensor.is_contiguous():
                raise ValueError(f"Diagnostic tensor {slot.name!r} does not match its canonical dense slot")
            filename = f"{key}-{index}.bin"
            payload = tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            (directory / filename).write_bytes(payload)
            manifest[key].append(filename)
            metadata[slot.name] = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "component_id": component.component_id,
        "argument_report": argument_report,
        "total_device_image_bytes": sum(len(image.device_image) for image in artifact.images),
        "native_transport_bytes": len(transport),
        "cuda_parameter_abi_validated": True,
        "launch_recipe": artifact.launch_recipe.canonical_record,
        "tensor_metadata": metadata,
        "fixture_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())
        },
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "Python canonical environment at the first physical forward boundary, including its layout copy",
    }
