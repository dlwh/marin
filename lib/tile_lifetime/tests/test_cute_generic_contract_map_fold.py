# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from tile_lifetime.backends.h100.cute_generic_tensor_program import (  # noqa: E402
    build_generic_component_multi_image_artifact,
    build_generic_multi_image_artifact,
    compile_generic_kernel_composition,
    compile_generic_tensor_program,
)
from tile_lifetime.backends.h100.cute_launch_abi import validate_cute_launch_abi  # noqa: E402
from tile_lifetime.backends.h100.generic_codegen_admission import UnsupportedContractLoweringError  # noqa: E402
from tile_lifetime.canonical_kernel_composition import lower_canonical_kernel_composition  # noqa: E402
from tile_lifetime.cute_pipeline_artifact import (  # noqa: E402
    CutePipelineSlotArgument,
    CutePipelineStructuredArgument,
    CutePipelineStructuredSlotAddressField,
)
from tile_lifetime.physical_task_contract import PhysicalTarget  # noqa: E402
from tile_lifetime.streaming_attention_executable import (  # noqa: E402
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import (  # noqa: E402
    ContractPrimitive,
    DType,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    execute_tensor_program,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


def _program() -> TensorProgram:
    row = TensorAxis(3, 5, "row")
    column = TensorAxis(7, 7, "column")
    contracted = TensorAxis(11, 11, "contracted")
    left = ProgramValue("left", (row, contracted), DType.FP32)
    right = ProgramValue("right", (contracted, column), DType.FP32)
    product = ProgramValue("product", (row, column), DType.FP32)
    transformed = ProgramValue("transformed", (row, column), DType.FP32)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    contract = ContractPrimitive("contract", (left, right), product, (contracted,), DType.FP32)
    scalar_map = MapPrimitive(
        "map",
        (product,),
        transformed,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_unary(ScalarExpressionKind.TANH, scalar_input(product.name)),
            scalar_constant(0.125),
        ),
    )
    fold = FoldPrimitive("fold", transformed, reduced, (column,), FoldReducer.SUM, DType.FP32)
    return TensorProgram((left, right), (contract, scalar_map, fold), (reduced,), NumericalPolicy.FAST)


@pytest.mark.slow
def test_generic_map_fold_images_preserve_numerics_and_export_abi() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    chain = _program()
    task_program = TensorProgram(
        (chain.operations[0].output,), chain.operations[1:], chain.outputs, chain.numerical_policy
    )
    request = jax_tensor_program_executable_request(task_program, tile_size=128, target=PhysicalTarget.SM90)
    source = np.arange(35, dtype=np.float32).reshape(task_program.inputs[0].shape) / 35
    device_inputs = (torch.asarray(source, device="cuda"),)
    expected = execute_tensor_program(task_program, {task_program.inputs[0].name: source})["reduced"]

    compiled = compile_generic_tensor_program(task_program, example_inputs=device_inputs)
    (actual,) = compiled(*device_inputs)
    torch.cuda.synchronize()

    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-4, atol=2e-4)
    assert len(compiled.primitives) == 2
    assert sum(len(primitive.kernel_names) for primitive in compiled.primitives) == 2
    composition = lower_canonical_kernel_composition(request)
    artifact = build_generic_multi_image_artifact(
        compiled,
        composition,
        function_prefix="generic_map_fold",
    )
    validate_cute_launch_abi(artifact)
    assert len(artifact.images) == 2
    assert len(artifact.launch_recipe.launches) == 2
    assert len(artifact.launch_recipe.event_realizations) == len(request.event_plans)
    assert all(
        realization.kind.value in {"program_order", "stream_order"}
        for realization in artifact.launch_recipe.event_realizations
    )
    assert artifact.request.algebra_identity["algebra_fingerprint"] == request.algebra_fingerprint

    fused = compile_generic_kernel_composition(
        task_program,
        composition,
        example_inputs=device_inputs,
    )
    (fused_actual,) = fused(*device_inputs)
    torch.cuda.synchronize()
    np.testing.assert_allclose(fused_actual.cpu().numpy(), expected, rtol=2e-4, atol=2e-4)
    fused_artifact = build_generic_component_multi_image_artifact(
        fused,
        function_prefix="generic_map_fold_components",
    )
    validate_cute_launch_abi(fused_artifact)
    first_launch = fused_artifact.launch_recipe.launches[0]
    first_argument = first_launch.arguments[0]
    assert isinstance(first_argument, CutePipelineSlotArgument)
    oversized_argument = CutePipelineStructuredArgument(
        first_argument.name,
        size=16,
        alignment=8,
        fields=(
            CutePipelineStructuredSlotAddressField(
                "pointer", first_argument.slot, first_argument.access, first_argument.byte_offset, 0
            ),
        ),
    )
    malformed = replace(
        fused_artifact,
        launch_recipe=replace(
            fused_artifact.launch_recipe,
            launches=(
                replace(first_launch, arguments=(oversized_argument, *first_launch.arguments[1:])),
                *fused_artifact.launch_recipe.launches[1:],
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"declares 16 bytes.*requires 8"):
        validate_cute_launch_abi(malformed)
    fused_operations = tuple(
        tuple(task_program.operations[ordinal] for ordinal in component.operation_ordinals)
        for component in fused.components
        if component.is_fused
    )
    assert any(isinstance(operations[-1], FoldPrimitive) for operations in fused_operations)
    assert len(fused_artifact.images) == len(fused_artifact.launch_recipe.launches)
    assert len(fused_artifact.launch_recipe.launches) < len(artifact.launch_recipe.launches)
    assert len(fused_artifact.launch_recipe.event_realizations) == len(request.event_plans)
    assert all(
        realization.kind.value in {"program_order", "stream_order"}
        for realization in fused_artifact.launch_recipe.event_realizations
    )
    assert fused_artifact.request.algebra_identity["algebra_fingerprint"] == request.algebra_fingerprint


@pytest.mark.slow
@pytest.mark.parametrize("fused", (False, True))
def test_generic_compiler_rejects_contract_without_physical_provider(fused) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    program = _program()
    inputs = tuple(torch.zeros(value.shape, device="cuda", dtype=torch.float32) for value in program.inputs)
    request = jax_tensor_program_executable_request(program, tile_size=128, target=PhysicalTarget.SM90)
    composition = lower_canonical_kernel_composition(request)
    with pytest.raises(UnsupportedContractLoweringError) as raised:
        if fused:
            compile_generic_kernel_composition(program, composition, example_inputs=inputs)
        else:
            compile_generic_tensor_program(program, example_inputs=inputs)
    assert raised.value.contracts == ((0, program.operations[0]),)
