# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a reusable SM90 WGMMA Map-load/Contract component."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from shuttle.ir import DType
from tile_lifetime.canonical_kernel_composition import CuteKernelComponentFragment
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineElementType,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
)
from tile_lifetime.tensor_program import ProgramValue, ScalarExpression, ScalarExpressionKind, TensorAxis


@dataclass(frozen=True)
class Sm90MapLoadContractSchedule:
    """Physical WGMMA geometry, independent of the scalar Map program."""

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    cluster_m: int = 1
    cluster_n: int = 1
    pingpong: bool = True

    def __post_init__(self) -> None:
        if self.tile_m not in {64, 128} or self.tile_n not in {64, 128, 192, 256} or self.tile_k != 64:
            raise ValueError("SM90 WGMMA Map-load/Contract requires FA4-derived M/N tiles and a 64-wide K tile")
        if self.cluster_m <= 0 or self.cluster_n <= 0:
            raise ValueError("SM90 WGMMA Map-load/Contract cluster dimensions must be positive")

    @property
    def family(self) -> str:
        return "sm90_wgmma_map_load_contract_v1"

    @property
    def threads_per_cta(self) -> int:
        """Threads in QuACK's producer plus resident math warpgroups."""
        return 128 * (3 if self.pingpong else 2)

    @property
    def fingerprint(self) -> str:
        record = {
            "family": self.family,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "cluster_m": self.cluster_m,
            "cluster_n": self.cluster_n,
            "pingpong": self.pingpong,
        }
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Sm90MapLoadOperand:
    """One typed scalar-Map leaf delivered to each resident A fragment."""

    value: ProgramValue
    parameter: str
    delivery: str


def run_sm90_map_load_contract(
    plan: Any,
    transform: Any,
    schedule: Sm90MapLoadContractSchedule,
    activation: Any,
    weight: Any,
    output: Any,
    transform_operands: dict[str, Any],
) -> None:
    """Pack mainloop operands at QuACK's prepared-plan invocation boundary."""
    if transform_operands:
        activation = transform.bundle(activation, transform_operands, schedule.tile_m, schedule.tile_k)
    plan.run(activation, weight, out={"D": output})


@dataclass(frozen=True)
class GeneratedSm90MapLoadContract:
    """Importable QuACK source for one algebra-derived WGMMA component."""

    source: str
    source_sha256: str
    schedule: Sm90MapLoadContractSchedule
    activation: ProgramValue
    operands: tuple[Sm90MapLoadOperand, ...]
    weight: ProgramValue
    output: ProgramValue

    @property
    def executable_with_current_quack_adapter(self) -> bool:
        """Whether the recorded typed multi-auxiliary QuACK adapter can deliver every leaf."""
        if not self.operands:
            return True
        rows, reduction = self.activation.shape
        return rows % self.schedule.tile_m == 0 and all(
            not operand.delivery.startswith("kvec_") or reduction % self.schedule.tile_k == 0
            for operand in self.operands
        )

    def canonical_fragment(
        self,
        *,
        component_id: int,
        schedule_fingerprint: str,
        value_ids: dict[ProgramValue, int],
        kernel_symbol: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        dynamic_shared_memory_bytes: int,
        source_tasks: tuple[int, ...],
    ) -> CuteKernelComponentFragment:
        """Bind a compiled QuACK kernel symbol to the canonical component ABI."""
        values = (self.activation, *(operand.value for operand in self.operands), self.weight, self.output)
        slots = {
            value_ids[value]: CutePipelineSlot(
                name=f"local_value_{value_ids[value]}",
                element_type=_element_type(value.dtype),
                shape=value.shape,
                kind=CutePipelineSlotKind.SCRATCH,
                alignment=16,
            )
            for value in values
        }
        arguments = tuple(
            CutePipelineSlotArgument(
                name=f"argument_{value_ids[value]}",
                slot=slots[value_ids[value]].name,
                access=CutePipelineSlotAccess.WRITE if value is self.output else CutePipelineSlotAccess.READ,
            )
            for value in values
        )
        launch = CuteKernelLaunch(
            name="map_load_contract",
            kernel_symbol=kernel_symbol,
            grid=grid,
            block=block,
            dynamic_shared_memory_bytes=dynamic_shared_memory_bytes,
            function_attributes=(),
            arguments=arguments,
            source_tasks=source_tasks,
        )
        return CuteKernelComponentFragment(component_id, schedule_fingerprint, slots, (launch,))


def generate_sm90_map_load_contract(
    *,
    expression: ScalarExpression,
    activation: ProgramValue,
    leaves: tuple[ProgramValue, ...],
    row_axis: TensorAxis,
    reduction_axis: TensorAxis,
    weight_feature_axis: TensorAxis,
    output_row_axis: TensorAxis,
    output_feature_axis: TensorAxis,
    weight: ProgramValue,
    output: ProgramValue,
    schedule: Sm90MapLoadContractSchedule | None = None,
) -> GeneratedSm90MapLoadContract:
    """Compile a scalar Map AST into QuACK's resident A-fragment hook."""
    if activation.dtype is not DType.BF16 or weight.dtype is not DType.BF16:
        raise ValueError("SM90 WGMMA Map-load/Contract requires BF16 activation and weight storage")
    if len(activation.axes) != 2 or activation.axes != (row_axis, reduction_axis):
        raise ValueError("SM90 WGMMA Map-load activation must use logical (row, reduction) axes")
    if weight.axes != (reduction_axis, weight_feature_axis):
        raise ValueError("SM90 WGMMA Contract weight must use logical (reduction, feature) axes")
    if output.axes != (output_row_axis, output_feature_axis):
        raise ValueError("SM90 WGMMA Contract output axes disagree with the explicit Contract domain")
    if output_row_axis.extent != row_axis.extent or output_feature_axis.extent != weight_feature_axis.extent:
        raise ValueError("SM90 WGMMA Contract affine axis relations must preserve row and feature extents")
    if len(leaves) != len(set(leaves)):
        raise ValueError("SM90 WGMMA Map-load scalar leaves must be unique")
    if len({value.name for value in leaves}) != len(leaves):
        raise ValueError("SM90 WGMMA Map-load scalar leaf names must be unique")
    operands = tuple(
        _operand(value, index, row_axis=row_axis, reduction_axis=reduction_axis)
        for index, value in enumerate(value for value in leaves if value is not activation)
    )
    selected = schedule or _schedule_for_operand_resources(operands)
    aliases = {activation.name: "activation"}
    aliases.update((operand.value.name, operand.parameter) for operand in operands)
    mapped = _expression_source(expression, aliases)
    arguments = {operand.parameter: operand.delivery for operand in operands}
    parameter_suffix = "" if not operands else ", " + ", ".join(operand.parameter for operand in operands)
    source = f"""# Generated from a generic scalar Map AST and explicit Contract axes; do not edit.
import cutlass
import cutlass.cute as cute
from quack.epilogue import gemm_epilogue
from quack.operand_transform import a_transform


@a_transform(vec_size=8, args={arguments!r})
def generated_transform(activation{parameter_suffix}):
    return {mapped}


@gemm_epilogue()
def generated_epilogue(acc):
    return {{"D": acc}}
"""
    return GeneratedSm90MapLoadContract(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        schedule=selected,
        activation=activation,
        operands=operands,
        weight=weight,
        output=output,
    )


def _schedule_for_operand_resources(operands: tuple[Sm90MapLoadOperand, ...]) -> Sm90MapLoadContractSchedule:
    row_operands = sum(operand.delivery.startswith("mvec_") for operand in operands)
    if row_operands >= 2:
        return Sm90MapLoadContractSchedule(tile_m=64, tile_n=256, pingpong=False)
    return Sm90MapLoadContractSchedule()


def _operand(
    value: ProgramValue,
    index: int,
    *,
    row_axis: TensorAxis,
    reduction_axis: TensorAxis,
) -> Sm90MapLoadOperand:
    if row_axis in value.axes and all(axis == row_axis or axis.extent == 1 for axis in value.axes):
        delivery = {
            DType.BF16: "mvec_bf16",
            DType.FP32: "mvec_fp32",
        }.get(value.dtype)
    elif reduction_axis in value.axes and all(axis == reduction_axis or axis.extent == 1 for axis in value.axes):
        delivery = {
            DType.BF16: "kvec_bf16",
            DType.FP32: "kvec_fp32",
        }.get(value.dtype)
    else:
        raise ValueError(
            f"SM90 WGMMA Map-load leaf {value.name!r} requires a row or reduction vector, got rank {len(value.axes)}"
        )
    if delivery is None:
        raise ValueError(f"SM90 WGMMA Map-load leaf {value.name!r} requires BF16 or FP32 storage")
    return Sm90MapLoadOperand(value, f"operand_{index}", delivery)


def _expression_source(expression: ScalarExpression, aliases: dict[str, str]) -> str:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return aliases[expression.input_name]
        except KeyError as error:
            raise ValueError(f"scalar Map input {expression.input_name!r} has no fragment-load binding") from error
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return repr(expression.constant)
    operands = tuple(_expression_source(operand, aliases) for operand in expression.operands)
    infix = {
        ScalarExpressionKind.ADD: "+",
        ScalarExpressionKind.SUBTRACT: "-",
        ScalarExpressionKind.MULTIPLY: "*",
        ScalarExpressionKind.DIVIDE: "/",
        ScalarExpressionKind.LESS: "<",
        ScalarExpressionKind.LESS_EQUAL: "<=",
        ScalarExpressionKind.GREATER: ">",
        ScalarExpressionKind.GREATER_EQUAL: ">=",
        ScalarExpressionKind.EQUAL: "==",
        ScalarExpressionKind.NOT_EQUAL: "!=",
    }
    if expression.kind in infix:
        return f"({operands[0]} {infix[expression.kind]} {operands[1]})"
    if expression.kind is ScalarExpressionKind.CONVERT:
        if expression.result_dtype is DType.BF16:
            return f"({operands[0]}).to(cutlass.BFloat16).to(cutlass.Float32)"
        if expression.result_dtype is DType.FP32:
            return f"({operands[0]}).to(cutlass.Float32)"
    if expression.kind is ScalarExpressionKind.NEGATE:
        return f"(-{operands[0]})"
    unary = {
        ScalarExpressionKind.EXP: "exp",
        ScalarExpressionKind.LOG: "log",
        ScalarExpressionKind.RSQRT: "rsqrt",
        ScalarExpressionKind.TANH: "tanh",
    }
    if expression.kind in unary:
        return f"cute.math.{unary[expression.kind]}({operands[0]}, fastmath=True)"
    if expression.kind is ScalarExpressionKind.MAXIMUM:
        return f"cute.where({operands[0]} >= {operands[1]}, {operands[0]}, {operands[1]})"
    if expression.kind is ScalarExpressionKind.MINIMUM:
        return f"cute.where({operands[0]} <= {operands[1]}, {operands[0]}, {operands[1]})"
    if expression.kind is ScalarExpressionKind.SELECT:
        return f"cute.where({operands[0]}, {operands[1]}, {operands[2]})"
    raise ValueError(f"SM90 WGMMA Map-load scalar code generation does not support {expression.kind.value!r}")


def _element_type(dtype: DType) -> CutePipelineElementType:
    if dtype is DType.BF16:
        return CutePipelineElementType.BF16
    if dtype is DType.FP32:
        return CutePipelineElementType.F32
    raise ValueError(f"unsupported canonical Map-load/Contract storage dtype {dtype.value!r}")
