# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tile_lifetime.backends.h100.bounded_transport import (
    bounded_transport_aot_plan,
    lower_bounded_transport,
    seal_bounded_transport_artifact,
)
from tile_lifetime.cute_pipeline_artifact import CuteEventRealizationKind
from tile_lifetime.tensor_program import (
    DType,
    OperandIndexing,
    ProgramValue,
    TensorAxis,
    TransportPrimitive,
    index_axis,
)


def _transpose() -> TransportPrimitive:
    row = TensorAxis(0, 3, "row")
    column = TensorAxis(1, 4, "column")
    source = ProgramValue("source", (row, column), DType.FP32)
    destination = ProgramValue("destination", (column, row), DType.FP32)
    return TransportPrimitive(
        "transpose",
        source,
        destination,
        (),
        (row, column),
        OperandIndexing(source, (row, column), (index_axis(row), index_axis(column))),
        OperandIndexing(destination, (row, column), (index_axis(column), index_axis(row))),
    )


def test_bounded_transport_preserves_exact_ready_and_reuse_generations(tmp_path: Path) -> None:
    pipeline = lower_bounded_transport(_transpose(), tile_elements=4, capacity=2)

    assert pipeline.buffer.slots == (0, 1, 0)
    assert pipeline.buffer.generations == (0, 0, 1)
    assert tuple((pair.source, pair.target) for pair in pipeline.readiness.scheduled_dependence.pairs) == (
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
    )
    assert tuple((pair.source, pair.target) for pair in pipeline.reuse.scheduled_dependence.pairs) == (((0,), (2,)),)
    assert tuple(realization.kind for realization in pipeline.recipe.event_realizations) == (
        CuteEventRealizationKind.STREAM_ORDER,
        CuteEventRealizationKind.STREAM_ORDER,
    )
    assert len(pipeline.recipe.launches) == 6
    assert "cuda_stream_order" == pipeline.request.schedule["event_legalization"]

    plan = bounded_transport_aot_plan(pipeline, output_directory=tmp_path, nvcc=Path("nvcc"))
    artifact = seal_bounded_transport_artifact(
        plan,
        host_object=b"host-object",
        device_image=b"fatbin-image",
        nvcc_version="Cuda compilation tools, release 13.0",
    )
    assert artifact.launch_recipe == pipeline.recipe
    assert artifact.manifest["compiler_ir"]["format"] == "cuda-c++"
    assert artifact.native_transport.startswith(b"SHUTGPU\0")


def test_bounded_transport_requires_real_reuse_and_axis_permutations() -> None:
    operation = _transpose()
    with pytest.raises(ValueError, match="more generations than staging slots"):
        lower_bounded_transport(operation, tile_elements=4, capacity=3)

    row, column = operation.domain_axes
    colliding = TransportPrimitive(
        "colliding",
        operation.source,
        operation.destination,
        (),
        operation.domain_axes,
        operation.source_indexing,
        OperandIndexing(operation.destination, operation.domain_axes, (index_axis(column), index_axis(column))),
    )
    with pytest.raises(ValueError, match="not proven injective"):
        lower_bounded_transport(colliding, tile_elements=4, capacity=2)

    projected_source = TransportPrimitive(
        "projected source",
        operation.source,
        operation.destination,
        (),
        operation.domain_axes,
        OperandIndexing(operation.source, operation.domain_axes, (index_axis(row), index_axis(row))),
        operation.destination_indexing,
    )
    with pytest.raises(ValueError, match="source relation must be an axis permutation"):
        lower_bounded_transport(projected_source, tile_elements=4, capacity=2)
