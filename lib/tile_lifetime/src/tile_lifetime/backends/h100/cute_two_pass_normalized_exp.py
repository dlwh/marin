# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed two-pass lowering for normalized-exp Contracts.

This is the explicit-boundary alternative to the online one-pass SM90 schedule.
The first launch computes the two row Fold states.  The second launch
recomputes the producer Contract and scalar Maps, materializes the typed
normalized weight in registers, and performs the weighted Contract.  Schedule
selection lives above this module; this emitter has no attention or score-map
mode dispatch.  FAST programs may apply a separately fingerprinted scalar-AST
peephole across the producer store when its conversion contract permits it.
"""

import hashlib
import inspect
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda import bindings as cuda_bindings
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from tile_lifetime.cute_device_ir import CuteDeviceIrSnapshot, compile_cute_with_structured_snapshot
from tile_lifetime.cute_pipeline_artifact import (
    CuteEventExecutionPoint,
    CuteEventRealizationKind,
    CutePipelineArtifact,
    CutePipelineCompileRequest,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotKind,
    CuteSourceTaskRole,
    CuteStructuralEventCover,
    elf64_symbol_bytes,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90PackedRowStateWriteProof,
    Sm90ProducerConsumerHandoffProof,
    Sm90RelationPackedLayout,
    prove_sm90_packed_row_state_writes,
    prove_sm90_producer_consumer_handoff,
)
from tile_lifetime.layout_relation_canonicalization import (
    LayoutRelationKind,
    LayoutRelationProof,
    canonicalize_contract_layout_relations,
)
from tile_lifetime.normalized_exp_contract_region import AffineIndexPredicate, NormalizedExpContractProgram
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    DType,
    FoldPrimitive,
    FoldReducer,
    IndexExpressionKind,
    MapPrimitive,
    NumericalPolicy,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorProgram,
    canonical_tensor_program_record,
    primitive_inputs,
    scalar_binary,
    scalar_expression_inputs,
    scalar_input,
    scalar_unary,
)

from .cute_domain_restriction import cute_affine_index_predicate
from .cute_scalar_ast import evaluate_cute_scalar_expression
from .cute_streaming_backward_emitter import _forward_launch
from .cute_streaming_emitter import H100StreamingTemplate, build_h100_streaming_template

_THREADS = 128


@dataclass(frozen=True)
class ConversionFaithfulNormalizedExpBoundary:
    """Explicit typed algebra assigned to the two physical passes."""

    source: TensorProgram
    producer: ContractPrimitive
    score_maps: tuple[MapPrimitive, ...]
    maximum: FoldPrimitive
    center: MapPrimitive
    exponential: MapPrimitive
    sum_input: MapPrimitive
    normalizer: FoldPrimitive
    log_normalizer: MapPrimitive
    denominator: MapPrimitive
    normalize: MapPrimitive
    consumer: ContractPrimitive
    finalizer: MapPrimitive | None
    producer_owner_operand: int
    normalized_operand: int
    relation_layout: Sm90RelationPackedLayout
    role_operation_ordinals: tuple[tuple[str, int], ...]
    relation_proofs: tuple[LayoutRelationProof, ...]
    domain_predicate: AffineIndexPredicate | None = None
    owner_dimension: int | None = None
    sum_input_is_fold_widening: bool = False
    normalization_was_sunk_through_contract: bool = False
    denominator_is_fold_state: bool = False

    @property
    def score_map(self) -> MapPrimitive:
        return self.score_maps[-1]

    def __post_init__(self) -> None:
        producer_output = self.producer.output
        score = self.score_map.output
        row_max = self.maximum.output
        centered = self.center.output
        exponential = self.exponential.output
        sum_input = self.sum_input.output
        row_sum = self.normalizer.output
        denominator = self.denominator.output
        normalized = self.normalize.output
        consumer_output = self.consumer.output
        if not self.score_maps:
            raise ValueError("two-pass lowering requires at least one scalar score Map")
        expected_score_input = producer_output
        for operation in self.score_maps:
            if operation.inputs != (expected_score_input,):
                raise ValueError("two-pass score Maps must form a unary SSA chain from the producer Contract")
            expected_score_input = operation.output
        if self.maximum.input != score or self.maximum.reducer is not FoldReducer.MAXIMUM:
            raise ValueError("two-pass maximum Fold must consume the score Map")
        if set(self.center.inputs) != {score, row_max}:
            raise ValueError("two-pass center Map must consume mapped score and row maximum")
        if self.exponential.inputs != (centered,):
            raise ValueError("two-pass exponential Map must consume the centered score")
        if self.sum_input.inputs != (exponential,):
            raise ValueError("two-pass sum-input conversion must consume the exponential Map")
        if self.normalizer.input != sum_input or self.normalizer.reducer is not FoldReducer.SUM:
            raise ValueError("two-pass sum Fold must consume the converted exponential")
        if self.denominator.inputs != (row_sum,):
            raise ValueError("two-pass denominator conversion must consume the sum state")
        if set(self.log_normalizer.inputs) != {row_max, row_sum} or self.log_normalizer.output.dtype is not DType.FP32:
            raise ValueError("two-pass saved log-normalizer must be an FP32 Map of maximum and sum state")
        if len(self.normalize.inputs) != 2 or exponential not in self.normalize.inputs:
            raise ValueError("two-pass normalization Map must consume exponential and converted denominator")
        denominator_inputs = tuple(value for value in self.normalize.inputs if value != exponential)
        if len(denominator_inputs) != 1 or denominator_inputs[0].dtype is not denominator.dtype:
            raise ValueError("two-pass normalization denominator role has incompatible storage dtype")
        if self.normalized_operand not in (0, 1) or self.consumer.inputs[self.normalized_operand] != normalized:
            raise ValueError("two-pass weighted Contract must consume the typed normalized weight")
        if self.producer_owner_operand not in (0, 1):
            raise ValueError("two-pass producer owner operand must identify one binary Contract input")
        if self.finalizer is not None and self.finalizer.inputs != (consumer_output,):
            raise ValueError("two-pass finalizer Map must consume only the weighted Contract")
        if self.producer.inputs[0].dtype is not DType.BF16 or self.producer.inputs[1].dtype is not DType.BF16:
            raise ValueError("current two-pass SM90 producer requires BF16 inputs")
        typed_values = (
            producer_output,
            score,
            row_max,
            centered,
            exponential,
            normalized,
            consumer_output,
            consumer_output if self.finalizer is None else self.finalizer.output,
        )
        if any(value.dtype is not DType.BF16 for value in typed_values):
            raise ValueError("conversion-faithful two-pass boundaries must be explicit BF16 values")
        expected_denominator_dtype = DType.FP32 if self.denominator_is_fold_state else DType.BF16
        if denominator.dtype is not expected_denominator_dtype:
            raise ValueError("two-pass denominator dtype disagrees with its Fold-state boundary")
        typed_maps = (
            *self.score_maps,
            self.center,
            self.exponential,
            self.sum_input,
            self.log_normalizer,
            self.denominator,
            self.normalize,
            *((self.finalizer,) if self.finalizer is not None else ()),
        )
        rne_maps = tuple(operation for operation in typed_maps if operation is not self.sum_input)
        if any(
            operation.output_conversion is not ScalarConversionSemantics.ROUND_NEAREST_EVEN for operation in rne_maps
        ):
            raise ValueError("current two-pass BF16 Map stores require round-to-nearest-even conversion semantics")
        expected_sum_conversion = (
            ScalarConversionSemantics.EXACT
            if self.sum_input_is_fold_widening
            else ScalarConversionSemantics.ROUND_NEAREST_EVEN
        )
        if self.sum_input.output_conversion is not expected_sum_conversion:
            raise ValueError("two-pass sum-input conversion disagrees with its declared Fold boundary")
        if self.sum_input.output.dtype is not DType.FP32 or row_sum.dtype is not DType.FP32:
            raise ValueError("conversion-faithful two-pass sum input and state must be FP32")
        if self.producer.accumulation_dtype not in {DType.BF16, DType.FP32} or self.consumer.accumulation_dtype not in {
            DType.BF16,
            DType.FP32,
        }:
            raise ValueError("current two-pass SM90 Contracts require BF16 or FP32 declared accumulation")
        if self.maximum.reduction_axes != self.normalizer.reduction_axes:
            raise ValueError("two-pass normalized-exp Folds must reduce the same axes")
        if self.producer.reduction_axes != (self.producer.inputs[0].axes[-1],):
            raise ValueError("current two-pass producer requires a trailing feature reduction")
        if self.consumer.reduction_axes != self.maximum.reduction_axes:
            raise ValueError("two-pass consumer and normalized-exp Folds must share the streamed reduction")
        if self.relation_layout.consumer_feature_extent > _THREADS:
            raise ValueError(f"current two-pass emitter supports at most {_THREADS} consumer features")
        if self.domain_predicate is not None and self.owner_dimension is None:
            raise ValueError("predicated two-pass lowering requires its logical owner dimension")

    @property
    def source_program(self) -> TensorProgram:
        return self.source

    @property
    def boundary_digest(self) -> str:
        record = canonical_tensor_program_record(self.source_program)
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def scalar_realization_record(self) -> dict[str, object]:
        prefix = _fast_contract_scalar_prefix(self)
        return {
            "kind": "fast_contract_scalar_prefix" if prefix is not None else "typed_map_boundary",
            "producer_alpha": None if prefix is None else prefix.scale,
            "rewritten_expression_path": None if prefix is None else prefix.expression_path,
        }

    @property
    def scalar_realization_fingerprint(self) -> str:
        encoded = json.dumps(self.scalar_realization_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def schedule_record(self) -> dict[str, object]:
        layout = self.relation_layout

        def index_map_record(mapping: AxisIndexMap) -> tuple[int, int, int | None]:
            return mapping.divisor, mapping.offset, mapping.modulus

        return {
            "family": "sm90_contract_fold_then_map_contract",
            "passes": (
                (
                    "producer_contract_maximum_traversal",
                    "score_map_chain_maximum_traversal",
                    "maximum_fold",
                    "producer_contract_sum_traversal",
                    "score_map_chain_sum_traversal",
                    "center_map",
                    "exponential_map",
                    "sum_input_conversion",
                    "sum_fold",
                    "saved_log_normalizer_map",
                ),
                (
                    "producer_contract",
                    "score_map_chain",
                    "center_map",
                    "exponential_map",
                    "denominator_conversion",
                    "normalize_map",
                    "consumer_contract",
                    "finalizer_map",
                ),
            ),
            "relation_layout": {
                "batch_shape": layout.batch_shape,
                "shared_relation_extent": layout.shared_relation_axis.extent,
                "relation_group_shape": layout.relation_group_shape,
                "owner_extent": layout.owner_extent,
                "streamed_extent": layout.streamed_extent,
                "producer_feature_extent": layout.producer_feature_extent,
                "consumer_feature_extent": layout.consumer_feature_extent,
                "shared_relation_map": index_map_record(layout.shared_relation_map),
                "group_relation_maps": tuple(index_map_record(mapping) for mapping in layout.group_relation_maps),
            },
            "state_dtypes": (self.maximum.output.dtype.value, self.normalizer.output.dtype.value),
            "sum_input_realization": "fold_accumulator_widening" if self.sum_input_is_fold_widening else "explicit_map",
            "normalization_realization": (
                "fold_invariant_division_sunk_through_contract"
                if self.normalization_was_sunk_through_contract
                else "normalized_map_before_contract"
            ),
            "denominator_realization": "fp32_fold_state" if self.denominator_is_fold_state else "explicit_map",
            "typed_boundary_dtypes": tuple(
                value.dtype.value
                for value in (
                    self.producer.output,
                    self.score_map.output,
                    self.center.output,
                    self.exponential.output,
                    self.sum_input.output,
                    self.denominator.output,
                    self.normalize.output,
                    self.consumer.output,
                    self.consumer.output if self.finalizer is None else self.finalizer.output,
                )
            ),
            "contract_accumulation_dtypes": (
                self.producer.accumulation_dtype.value,
                self.consumer.accumulation_dtype.value,
            ),
            "contract_operand_roles": {
                "producer_owner_operand": self.producer_owner_operand,
                "consumer_normalized_operand": self.normalized_operand,
            },
        }

    @property
    def schedule_fingerprint(self) -> str:
        encoded = json.dumps(self.schedule_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class CompiledH100ConversionFaithfulNormalizedExp:
    """One two-launch CuTe executable plus its exact typed boundary."""

    boundary: ConversionFaithfulNormalizedExpBoundary
    executable: Any
    statistics_template: H100StreamingTemplate
    weighted_contract_template: H100StreamingTemplate
    structured_device_ir: CuteDeviceIrSnapshot
    row_state_write_proof: Sm90PackedRowStateWriteProof
    producer_consumer_handoff_proof: Sm90ProducerConsumerHandoffProof
    physical_schedule_record: dict[str, object]
    query_stride: tuple[int, ...]
    key_stride: tuple[int, ...]
    value_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    row_max_stride: tuple[int, ...]
    row_sum_stride: tuple[int, ...]
    log_sum_exp_stride: tuple[int, ...]

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(self.executable.kernel_info)

    @property
    def schedule_fingerprint(self) -> str:
        record = {
            "boundary_schedule": self.boundary.schedule_record,
            "physical_schedule": self.physical_schedule_record,
            "row_state_write_proof": self.row_state_write_proof.canonical_record,
            "producer_consumer_handoff_proof": self.producer_consumer_handoff_proof.canonical_record,
        }
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        row_max: torch.Tensor,
        row_sum: torch.Tensor,
        log_sum_exp: torch.Tensor,
    ) -> None:
        layout = self.boundary.relation_layout
        _validate_tensor(
            "query",
            query,
            layout.owner_physical_shape(layout.producer_feature_extent),
            torch.bfloat16,
            self.query_stride,
        )
        _validate_tensor(
            "key", key, layout.streamed_physical_shape(layout.producer_feature_extent), torch.bfloat16, self.key_stride
        )
        _validate_tensor(
            "value",
            value,
            layout.streamed_physical_shape(layout.consumer_feature_extent),
            torch.bfloat16,
            self.value_stride,
        )
        _validate_tensor(
            "output",
            output,
            layout.owner_physical_shape(layout.consumer_feature_extent),
            torch.bfloat16,
            self.output_stride,
        )
        state_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
        _validate_tensor("row_max", row_max, state_shape, torch.bfloat16, self.row_max_stride)
        _validate_tensor("row_sum", row_sum, state_shape, torch.float32, self.row_sum_stride)
        _validate_tensor("log_sum_exp", log_sum_exp, state_shape, torch.float32, self.log_sum_exp_stride)
        self.executable(query, key, value, output, row_max, row_sum, log_sum_exp)


def conversion_faithful_boundary(
    source_program: TensorProgram,
    relation_layout: Sm90RelationPackedLayout,
    domain_predicate: AffineIndexPredicate | None,
    *,
    score_map: MapPrimitive | None = None,
    owner_dimension: int | None = None,
    producer_contract: ContractPrimitive | None = None,
    consumer_contract: ContractPrimitive | None = None,
) -> ConversionFaithfulNormalizedExpBoundary:
    """Recover typed pass roles after proving layout-only Relations."""
    canonical = canonicalize_contract_layout_relations(source_program)
    program = canonical.program
    operations = program.operations
    consumers: dict[ProgramValue, list[object]] = {}
    for operation in operations:
        for value in primitive_inputs(operation):
            consumers.setdefault(value, []).append(operation)

    contracts = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))
    if producer_contract is not None or consumer_contract is not None:
        if producer_contract is None or consumer_contract is None:
            raise ValueError("two-pass explicit Contract bindings must include both producer and consumer")

        def bound_contract(reference: ContractPrimitive, role: str) -> ContractPrimitive:
            matches = tuple(operation for operation in contracts if operation.output.name == reference.output.name)
            if len(matches) != 1:
                raise ValueError(f"two-pass {role} Contract binding resolves {len(matches)} canonical operations")
            return matches[0]

        producer = bound_contract(producer_contract, "producer")
        consumer = bound_contract(consumer_contract, "consumer")
        if producer is consumer:
            raise ValueError("two-pass producer and consumer Contract bindings must be distinct")
    else:
        if len(contracts) != 2:
            raise ValueError(f"conversion-faithful two-pass lowering requires two Contracts, found {len(contracts)}")
        producer = next(
            (
                operation
                for operation in contracts
                if any(isinstance(use, MapPrimitive) for use in consumers[operation.output])
            ),
            None,
        )
        if producer is None:
            raise ValueError(
                "two-pass producer Contract result does not reach a scalar Map after layout canonicalization"
            )
        consumer = next(operation for operation in contracts if operation is not producer)

    def unique_use(value: ProgramValue, expected: type, label: str):
        matches = tuple(operation for operation in consumers.get(value, ()) if isinstance(operation, expected))
        if len(matches) != 1:
            raise ValueError(f"two-pass {label} requires one {expected.__name__} consumer, found {len(matches)}")
        return matches[0]

    if score_map is None:
        score_maps: list[MapPrimitive] = []
        score_value = producer.output
        while True:
            fold_uses = tuple(use for use in consumers.get(score_value, ()) if isinstance(use, FoldPrimitive))
            if fold_uses:
                if len(fold_uses) != 1 or not score_maps:
                    raise ValueError("two-pass score chain must terminate at one maximum Fold")
                maximum = fold_uses[0]
                break
            map_uses = tuple(use for use in consumers.get(score_value, ()) if isinstance(use, MapPrimitive))
            if len(map_uses) != 1 or map_uses[0].inputs != (score_value,):
                raise ValueError("two-pass producer must reach the maximum Fold through one unary score-Map chain")
            score_maps.append(map_uses[0])
            score_value = map_uses[0].output
    else:
        maximum_matches = tuple(
            operation
            for operation in operations
            if isinstance(operation, FoldPrimitive) and operation.reducer is FoldReducer.MAXIMUM
        )
        if len(maximum_matches) != 1:
            raise ValueError(f"two-pass lowering requires one maximum Fold, found {len(maximum_matches)}")
        maximum = maximum_matches[0]
        score_inputs = scalar_expression_inputs(score_map.expression)
        template_producer_inputs = tuple(value for value in score_map.inputs if value.name in score_inputs)
        if len(template_producer_inputs) != 1 or template_producer_inputs[0] != score_map.inputs[0]:
            raise ValueError("current two-pass score Map template requires one producer scalar input")
        source_score_operation = next(
            (
                operation
                for operation in operations
                if isinstance(operation, MapPrimitive) and operation.output == maximum.input
            ),
            None,
        )
        if source_score_operation is None:
            raise ValueError("two-pass maximum Fold input lacks a source score Map")
        synthesized_score = replace(
            score_map,
            name=source_score_operation.name,
            inputs=(producer.output,),
            output=maximum.input,
            expression=_rename_scalar_input(
                score_map.expression,
                template_producer_inputs[0].name,
                producer.output.name,
            ),
            predicate=None,
            rejected_value=None,
        )
        score_maps = [synthesized_score]
    score_map = score_maps[-1]
    normalized_axis = maximum.reduction_axes[0]

    def operand_domain_axes(contract: ContractPrimitive, operand: int) -> set[object]:
        mappings = {mapping.operand_axis: mapping.domain_axis for mapping in contract.index_maps_for_input(operand)}
        return {mappings.get(axis, axis) for axis in contract.inputs[operand].axes}

    producer_owner_candidates = tuple(
        operand for operand in range(2) if normalized_axis not in operand_domain_axes(producer, operand)
    )
    if len(producer_owner_candidates) != 1:
        raise ValueError("two-pass producer index relations do not identify one owner-resident operand")
    producer_owner_operand = producer_owner_candidates[0]
    center = next(
        (
            operation
            for operation in consumers.get(score_map.output, ())
            if isinstance(operation, MapPrimitive) and maximum.output in operation.inputs
        ),
        None,
    )
    if center is None:
        raise ValueError("two-pass center Map does not consume mapped score and maximum state")
    exponential = unique_use(center.output, MapPrimitive, "centered score")
    sum_candidates = tuple(
        operation
        for operation in consumers.get(exponential.output, ())
        if isinstance(operation, MapPrimitive)
        and operation.output.dtype is DType.FP32
        and any(isinstance(use, FoldPrimitive) for use in consumers.get(operation.output, ()))
    )
    sum_input_is_fold_widening = False
    if len(sum_candidates) == 1:
        sum_input = sum_candidates[0]
        normalizer = unique_use(sum_input.output, FoldPrimitive, "converted exponential")
    elif not sum_candidates:
        direct_folds = tuple(
            operation
            for operation in consumers.get(exponential.output, ())
            if isinstance(operation, FoldPrimitive)
            and operation.reducer is FoldReducer.SUM
            and operation.accumulation_dtype is DType.FP32
            and operation.output.dtype is DType.FP32
        )
        if len(direct_folds) != 1:
            raise ValueError(
                "two-pass exponential requires one explicit FP32 conversion or one FP32-accumulating sum Fold, "
                f"found {len(sum_candidates)} conversions and {len(direct_folds)} direct Folds"
            )
        source_normalizer = direct_folds[0]
        sum_input = MapPrimitive(
            "realize sum-Fold input widening",
            (exponential.output,),
            ProgramValue(
                f"{exponential.output.name}.sum_fold_widened",
                exponential.output.axes,
                DType.FP32,
            ),
            scalar_input(exponential.output.name),
            output_conversion=ScalarConversionSemantics.EXACT,
        )
        normalizer = replace(source_normalizer, input=sum_input.output)
        sum_input_is_fold_widening = True
    else:
        raise ValueError(f"two-pass exponential requires one FP32 sum-input conversion, found {len(sum_candidates)}")
    denominator_candidates = tuple(
        operation
        for operation in consumers.get(normalizer.output, ())
        if isinstance(operation, MapPrimitive) and operation.inputs == (normalizer.output,)
    )
    denominator_is_fold_state = False
    if len(denominator_candidates) == 1:
        denominator = denominator_candidates[0]
        denominator_value = denominator.output
    elif not denominator_candidates:
        denominator = MapPrimitive(
            "bind FP32 sum-Fold state as normalization denominator",
            (normalizer.output,),
            ProgramValue(f"{normalizer.output.name}.normalization_denominator", normalizer.output.axes, DType.FP32),
            scalar_input(normalizer.output.name),
        )
        denominator_value = normalizer.output
        denominator_is_fold_state = True
    else:
        raise ValueError(
            "two-pass sum state requires at most one unary denominator conversion, "
            f"found {len(denominator_candidates)}"
        )

    relation_proofs = list(canonical.proofs)
    denominator_uses = consumers.get(denominator_value, ())
    if (
        not denominator_is_fold_state
        and len(denominator_uses) == 1
        and isinstance(denominator_uses[0], RelationPrimitive)
    ):
        relation = denominator_uses[0]
        if relation.predicate is not None or len(relation.inputs) != 1 or len(relation.outputs) != 1:
            raise ValueError("two-pass denominator Relation must be one unpredicated tensor view")
        if relation.resolved_result_expressions[0] is not None:
            raise ValueError("two-pass denominator Relation cannot synthesize values")
        indexing = relation.input_indexing[0]
        source_to_result_dimensions: list[int | None] = []
        for source_axis, expression in zip(relation.inputs[0].axes, indexing.indices, strict=True):
            if expression.kind is IndexExpressionKind.AXIS and expression.axis in relation.domain_axes:
                source_to_result_dimensions.append(relation.domain_axes.index(expression.axis))
                continue
            if expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and source_axis.extent == 1:
                source_to_result_dimensions.append(None)
                continue
            raise ValueError("two-pass denominator Relation must be an identity-or-broadcast index projection")
        original_ordinals = tuple(
            index for index, operation in enumerate(source_program.operations) if operation.name == relation.name
        )
        if len(original_ordinals) != 1:
            raise ValueError("two-pass denominator Relation lacks a unique source operation ordinal")
        relation_proofs.append(
            LayoutRelationProof(
                LayoutRelationKind.BROADCAST_PROJECTION,
                original_ordinals,
                relation.inputs[0].shape,
                relation.outputs[0].shape,
                tuple(source_to_result_dimensions),
                (),
                "normalized_into_two_pass_scalar_role_binding",
            )
        )
        denominator_value = relation.outputs[0]
    normalize = next(
        (
            operation
            for operation in consumers.get(exponential.output, ())
            if isinstance(operation, MapPrimitive) and denominator_value in operation.inputs
        ),
        None,
    )
    normalization_was_sunk = False
    if normalize is None:
        try:
            unnormalized_operand = consumer.inputs.index(exponential.output)
        except ValueError as error:
            raise ValueError(
                "two-pass normalized weight must be explicit or represented by a Fold-invariant consumer finalizer"
            ) from error
        sunk_finalizers = tuple(
            operation
            for operation in consumers.get(consumer.output, ())
            if isinstance(operation, MapPrimitive) and denominator_value in operation.inputs
        )
        if len(sunk_finalizers) != 1:
            raise ValueError(
                "two-pass Fold-invariant normalization requires one consumer finalizer, " f"found {len(sunk_finalizers)}"
            )
        source_finalizer = sunk_finalizers[0]
        normalized_value = ProgramValue(
            f"{exponential.output.name}.normalized_before_contract",
            exponential.output.axes,
            exponential.output.dtype,
        )
        normalize = replace(
            source_finalizer,
            name="realize Fold-invariant normalization before Contract",
            inputs=tuple(
                (
                    exponential.output
                    if value == consumer.output
                    else denominator.output if value == denominator_value else value
                )
                for value in source_finalizer.inputs
            ),
            output=normalized_value,
            expression=_rename_scalar_input(
                _rename_scalar_input(
                    source_finalizer.expression,
                    consumer.output.name,
                    exponential.output.name,
                ),
                denominator_value.name,
                denominator.output.name,
            ),
            predicate=None,
            rejected_value=None,
        )
        consumer = replace(
            consumer,
            inputs=tuple(
                normalized_value if index == unnormalized_operand else value
                for index, value in enumerate(consumer.inputs)
            ),
        )
        finalizer = MapPrimitive(
            source_finalizer.name,
            (consumer.output,),
            source_finalizer.output,
            scalar_input(consumer.output.name),
            output_conversion=source_finalizer.output_conversion,
        )
        normalization_was_sunk = True
    try:
        normalized_operand = consumer.inputs.index(normalize.output)
    except ValueError as error:
        raise ValueError("two-pass consumer Contract does not consume the normalized-weight Map") from error
    if not normalization_was_sunk:
        finalizer_matches = tuple(
            operation for operation in consumers.get(consumer.output, ()) if isinstance(operation, MapPrimitive)
        )
        # A branched result has no unique terminal Map in this region.  Close the
        # physical boundary at the Contract and leave every semantic branch to its
        # declared downstream task instead of selecting one by source order.
        finalizer = finalizer_matches[0] if len(finalizer_matches) == 1 else None

    log_normalizer = MapPrimitive(
        "materialize conversion-faithful normalized-exp row state",
        (normalizer.output, maximum.output),
        ProgramValue("normalized_exp_contract.two_pass_log_normalizer", normalizer.output.axes, DType.FP32),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_unary(ScalarExpressionKind.LOG, scalar_input(normalizer.output.name)),
            scalar_input(maximum.output.name),
        ),
    )
    source_ordinals_by_name = {operation.name: index for index, operation in enumerate(source_program.operations)}
    role_operations = {
        "producer_contract": producer,
        "maximum_fold": maximum,
        "center_map": center,
        "exponential_map": exponential,
        "sum_fold": normalizer,
        "consumer_contract": consumer,
    }
    if not sum_input_is_fold_widening:
        role_operations["sum_input_conversion"] = sum_input
    if not denominator_is_fold_state:
        role_operations["denominator_conversion"] = denominator
    if not normalization_was_sunk:
        role_operations["normalize_map"] = normalize
    else:
        role_operations["sunk_normalization_finalizer"] = finalizer
    role_operations.update({f"score_map.{index}": operation for index, operation in enumerate(score_maps)})
    if finalizer is not None:
        role_operations["finalizer_map"] = finalizer
    try:
        role_ordinals = tuple(
            (role, source_ordinals_by_name[operation.name]) for role, operation in role_operations.items()
        )
    except KeyError as error:
        raise ValueError("layout-canonicalized two-pass role lacks a source operation ordinal") from error
    return ConversionFaithfulNormalizedExpBoundary(
        source_program,
        producer,
        tuple(score_maps),
        maximum,
        center,
        exponential,
        sum_input,
        normalizer,
        log_normalizer,
        denominator,
        normalize,
        consumer,
        finalizer,
        producer_owner_operand,
        normalized_operand,
        relation_layout,
        role_ordinals,
        tuple(relation_proofs),
        domain_predicate,
        owner_dimension,
        sum_input_is_fold_widening,
        normalization_was_sunk,
        denominator_is_fold_state,
    )


def compile_h100_conversion_faithful_normalized_exp(
    boundary: ConversionFaithfulNormalizedExpBoundary,
    streaming_program: NormalizedExpContractProgram,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> CompiledH100ConversionFaithfulNormalizedExp:
    """Compile the exact-boundary stats/recompute pair selected by the caller."""
    layout = boundary.relation_layout
    state_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
    for name, tensor, shape, dtype in (
        ("query", query, layout.owner_physical_shape(layout.producer_feature_extent), torch.bfloat16),
        ("key", key, layout.streamed_physical_shape(layout.producer_feature_extent), torch.bfloat16),
        ("value", value, layout.streamed_physical_shape(layout.consumer_feature_extent), torch.bfloat16),
        ("output", output, layout.owner_physical_shape(layout.consumer_feature_extent), torch.bfloat16),
        ("row_max", row_max, state_shape, torch.bfloat16),
        ("row_sum", row_sum, state_shape, torch.float32),
        ("log_sum_exp", log_sum_exp, state_shape, torch.float32),
    ):
        _validate_tensor(name, tensor, shape, dtype)
    emitter = _CuteTiledTwoPassNormalizedExp(boundary, streaming_program)
    row_state_write_proof = prove_sm90_packed_row_state_writes(
        boundary.relation_layout,
        owner_tile=streaming_program.schedule.tile_m,
    )
    producer_consumer_handoff_proof = prove_sm90_producer_consumer_handoff(
        streaming_program.schedule.contract,
        streaming_program.schedule.producer_operand_orientation,
        stages=streaming_program.schedule.stages,
        resident_left_in_registers=streaming_program.schedule.resident_left_in_registers,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    executable, snapshot = compile_cute_with_structured_snapshot(
        cute,
        emitter,
        *(to_cute_tensor(tensor) for tensor in (query, key, value, output, row_max, row_sum, log_sum_exp)),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    for template in (emitter.statistics_template, emitter.weighted_contract_template):
        if template.kernel.dynamic_shared_memory_bytes <= 0 or template.kernel.num_threads != template.schedule.threads:
            raise ValueError("two-pass compiled launch metadata disagrees with its selected streaming schedule")
    if len(tuple(executable.kernel_info)) != 2:
        raise ValueError("conversion-faithful normalized-exp lowering must emit exactly two kernels")
    return CompiledH100ConversionFaithfulNormalizedExp(
        boundary,
        executable,
        emitter.statistics_template,
        emitter.weighted_contract_template,
        snapshot,
        row_state_write_proof,
        producer_consumer_handoff_proof,
        {
            "tile_m": streaming_program.schedule.tile_m,
            "tile_n": streaming_program.schedule.tile_n,
            "threads": streaming_program.schedule.threads,
            "stages": streaming_program.schedule.stages,
            "producer_reduction_traversal": streaming_program.schedule.producer_reduction_traversal.value,
            "producer_operand_orientation": streaming_program.schedule.producer_operand_orientation.value,
            "producer_consumer_handoff": producer_consumer_handoff_proof.canonical_record,
            "producer_consumer_handoff_event": (
                None
                if not producer_consumer_handoff_proof.requires_register_to_shared_transpose
                else {
                    "buffer": "producer_consumer_fragment_scratch",
                    "capacity": 1,
                    "bytes": producer_consumer_handoff_proof.shared_memory_bytes,
                    "producer_ownership": producer_consumer_handoff_proof.producer_fragment_axes,
                    "scratch_ownership": producer_consumer_handoff_proof.scratch_axes,
                    "fold_ownership": producer_consumer_handoff_proof.fold_fragment_axes,
                    "barrier_participants": producer_consumer_handoff_proof.barrier_participants,
                    "generations": producer_consumer_handoff_proof.barrier_generations,
                    "transfers": (
                        {
                            "phase": "maximum_fold_sweep",
                            "path": "register_to_shared_to_register",
                        },
                        {
                            "phase": "sum_fold_sweep",
                            "path": "register_to_shared_to_register",
                        },
                        {
                            "phase": "weighted_contract",
                            "path": "register_to_shared",
                        },
                    ),
                    "realization": "cta_shared_store_fence_named_barrier",
                }
            ),
        },
        tuple(query.stride()),
        tuple(key.stride()),
        tuple(value.stride()),
        tuple(output.stride()),
        tuple(row_max.stride()),
        tuple(row_sum.stride()),
        tuple(log_sum_exp.stride()),
    )


def export_h100_conversion_faithful_normalized_exp_artifact(
    compiled: CompiledH100ConversionFaithfulNormalizedExp,
    *,
    compile_request: CutePipelineCompileRequest,
    function_prefix: str,
) -> CutePipelineArtifact:
    """Export the two launches and their private saved-state Event edge."""
    if not function_prefix.isidentifier():
        raise ValueError(f"CuTe artifact prefix must be an identifier: {function_prefix!r}")
    with tempfile.TemporaryDirectory(prefix="shuttle-two-pass-cute-artifact-") as temporary_directory:
        object_path = Path(temporary_directory) / f"{function_prefix}.o"
        compiled.executable.export_to_c(
            str(object_path),
            function_name=function_prefix,
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        host_object = object_path.read_bytes()
    kernel_names = compiled.kernel_names
    if len(kernel_names) != 2:
        raise ValueError(f"two-pass normalized-exp artifact requires two kernels, got {len(kernel_names)}")
    layout = compiled.boundary.relation_layout
    state_shape = (layout.batch_extent, layout.flattened_relation_axis.extent, layout.owner_extent)
    slots = (
        CutePipelineSlot(
            "query",
            CutePipelineElementType.BF16,
            layout.owner_physical_shape(layout.producer_feature_extent),
            CutePipelineSlotKind.PUBLIC_INPUT,
            16,
            input_index=0,
        ),
        CutePipelineSlot(
            "key",
            CutePipelineElementType.BF16,
            layout.streamed_physical_shape(layout.producer_feature_extent),
            CutePipelineSlotKind.PUBLIC_INPUT,
            16,
            input_index=1,
        ),
        CutePipelineSlot(
            "value",
            CutePipelineElementType.BF16,
            layout.streamed_physical_shape(layout.consumer_feature_extent),
            CutePipelineSlotKind.PUBLIC_INPUT,
            16,
            input_index=2,
        ),
        CutePipelineSlot(
            "output",
            CutePipelineElementType.BF16,
            layout.owner_physical_shape(layout.consumer_feature_extent),
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            16,
            output_index=0,
        ),
        CutePipelineSlot("row_max", CutePipelineElementType.BF16, state_shape, CutePipelineSlotKind.SCRATCH, 16),
        CutePipelineSlot("row_sum", CutePipelineElementType.F32, state_shape, CutePipelineSlotKind.SCRATCH, 16),
        CutePipelineSlot(
            "log_sum_exp",
            CutePipelineElementType.F32,
            state_shape,
            CutePipelineSlotKind.PUBLIC_OUTPUT,
            16,
            output_index=1,
        ),
    )

    launches = tuple(
        replace(
            _forward_launch(
                template,
                kernel_name,
                writes_log_sum_exp=access is CutePipelineSlotAccess.WRITE,
                dynamic_shared_memory_bytes=template.kernel.dynamic_shared_memory_bytes,
                row_state_access=access,
                query_stride=compiled.query_stride,
                key_stride=compiled.key_stride,
                value_stride=compiled.value_stride,
                output_stride=compiled.output_stride,
                row_max_stride=compiled.row_max_stride,
                row_sum_stride=compiled.row_sum_stride,
                source_task_roles=roles,
            ),
            name=name,
            source_phase=phase,
        )
        for template, kernel_name, access, name, phase, roles in (
            (
                compiled.statistics_template,
                kernel_names[0],
                CutePipelineSlotAccess.WRITE,
                "normalized_exp_statistics",
                "contract_fold",
                (CuteSourceTaskRole.STATE_ROOT,),
            ),
            (
                compiled.weighted_contract_template,
                kernel_names[1],
                CutePipelineSlotAccess.READ,
                "normalized_exp_weighted_contract",
                "map_contract",
                (CuteSourceTaskRole.STATE_CARRY, CuteSourceTaskRole.STATE_TERMINAL),
            ),
        )
    )
    recipe = CutePipelineLaunchRecipe(
        schema_version=2,
        slots=slots,
        launches=launches,
        structural_event_covers=(
            CuteStructuralEventCover(
                CuteSourceTaskRole.STATE_ROOT,
                CuteSourceTaskRole.STATE_CARRY,
                CuteEventExecutionPoint("normalized_exp_statistics", 0),
                CuteEventExecutionPoint("normalized_exp_weighted_contract", 0),
                CuteEventRealizationKind.STREAM_ORDER,
            ),
        ),
        host_call_arguments=("query", "key", "value", "output", "row_max", "row_sum", "log_sum_exp"),
    )
    return CutePipelineArtifact(
        request=compile_request,
        host_object=host_object,
        device_image=elf64_symbol_bytes(host_object, "kernels_binary"),
        kernel_names=kernel_names,
        host_entry=f"__tvm_ffi_{function_prefix}",
        compiler_ir=str(compiled.executable.ir_module),
        toolchain={
            "cuda-python": cuda_bindings.__version__,
            "nvidia-cutlass-dsl": cutlass.__version__,
            "torch": torch.__version__,
        },
        source_digests={
            "emitter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scalar_ast": (
                hashlib.sha256(Path(inspect.getfile(evaluate_cute_scalar_expression)).read_bytes()).hexdigest()
            ),
            "boundary": compiled.boundary.boundary_digest,
            "scalar_realization": compiled.boundary.scalar_realization_fingerprint,
        },
        launch_recipe=recipe,
        structured_device_ir=(compiled.structured_device_ir,),
    )


class _CuteTiledTwoPassNormalizedExp:
    def __init__(
        self,
        boundary: ConversionFaithfulNormalizedExpBoundary,
        streaming_program: NormalizedExpContractProgram,
    ):
        self.boundary = boundary
        normalized_map = _cute_conversion_faithful_normalized_map(boundary)
        finalizer_map = _cute_conversion_faithful_finalizer(boundary)
        statistics_score_map = _cute_conversion_faithful_score_map(boundary)
        statistics_sum_map = _cute_conversion_faithful_sum_map(boundary)
        statistics_lse_map = _cute_conversion_faithful_lse_map(boundary)
        self.statistics_template = build_h100_streaming_template(
            streaming_program,
            conversion_faithful_score_map=statistics_score_map,
            conversion_faithful_statistics_score_map=statistics_score_map,
            conversion_faithful_statistics_sum_map=statistics_sum_map,
            conversion_faithful_statistics_lse_map=statistics_lse_map,
        )
        self.weighted_contract_template = build_h100_streaming_template(
            streaming_program,
            conversion_faithful_score_map=statistics_score_map,
            conversion_faithful_normalized_map=normalized_map,
            finalizer_map_override=finalizer_map,
        )

        self.statistics = self.statistics_template.kernel
        self.weighted_contract = self.weighted_contract_template.kernel

    @cute.jit
    def __call__(self, query, key, value, output, row_max, row_sum, log_sum_exp, stream):
        self.statistics(
            query,
            key,
            value,
            output,
            log_sum_exp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            row_max,
            row_sum,
            stream,
        )
        self.weighted_contract(
            query,
            key,
            value,
            output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            row_max,
            row_sum,
            stream,
        )


@dataclass(frozen=True)
class _FastContractScalarPrefix:
    scale: float
    expression_path: tuple[int, ...]
    residual_expression: ScalarExpression


def _constant_scalar_value(expression: ScalarExpression) -> float | None:
    if expression.kind is ScalarExpressionKind.CONSTANT and isinstance(expression.constant, (int, float)):
        return float(expression.constant)
    if expression.kind is ScalarExpressionKind.CONVERT:
        return _constant_scalar_value(expression.operands[0])
    if expression.kind is ScalarExpressionKind.NEGATE:
        value = _constant_scalar_value(expression.operands[0])
        return None if value is None else -value
    return None


def _replace_scalar_subexpression(
    expression: ScalarExpression,
    path: tuple[int, ...],
    replacement: ScalarExpression,
) -> ScalarExpression:
    if not path:
        return replacement
    operand = path[0]
    operands = list(expression.operands)
    operands[operand] = _replace_scalar_subexpression(operands[operand], path[1:], replacement)
    return replace(expression, operands=tuple(operands))


def _innermost_uniform_input_scale(
    expression: ScalarExpression,
    input_name: str,
    path: tuple[int, ...] = (),
) -> tuple[float, tuple[int, ...]] | None:
    for index, operand in enumerate(expression.operands):
        nested = _innermost_uniform_input_scale(operand, input_name, (*path, index))
        if nested is not None:
            return nested
    if expression.kind is ScalarExpressionKind.MULTIPLY:
        left, right = expression.operands
        if left.kind is ScalarExpressionKind.INPUT and left.input_name == input_name:
            constant = _constant_scalar_value(right)
            if constant is not None:
                return constant, path
        if right.kind is ScalarExpressionKind.INPUT and right.input_name == input_name:
            constant = _constant_scalar_value(left)
            if constant is not None:
                return constant, path
    if expression.kind is ScalarExpressionKind.DIVIDE:
        numerator, denominator = expression.operands
        if numerator.kind is ScalarExpressionKind.INPUT and numerator.input_name == input_name:
            constant = _constant_scalar_value(denominator)
            if constant not in (None, 0.0):
                return 1.0 / constant, path
    return None


def _fast_contract_scalar_prefix(
    boundary: ConversionFaithfulNormalizedExpBoundary,
) -> _FastContractScalarPrefix | None:
    if boundary.source.numerical_policy is not NumericalPolicy.FAST:
        return None
    first_map = boundary.score_maps[0]
    if (
        first_map.inputs != (boundary.producer.output,)
        or boundary.producer.output.dtype is not DType.BF16
        or boundary.producer.accumulation_dtype is not DType.FP32
        or first_map.output_conversion is not ScalarConversionSemantics.ROUND_NEAREST_EVEN
    ):
        return None
    match = _innermost_uniform_input_scale(first_map.expression, boundary.producer.output.name)
    if match is None:
        return None
    scale, path = match
    if not math.isfinite(scale) or scale == 0.0:
        return None
    residual = _replace_scalar_subexpression(
        first_map.expression,
        path,
        scalar_input(boundary.producer.output.name),
    )
    return _FastContractScalarPrefix(scale=scale, expression_path=path, residual_expression=residual)


def _cute_conversion_faithful_score_map(boundary: ConversionFaithfulNormalizedExpBoundary):
    prefix = _fast_contract_scalar_prefix(boundary)

    @cute.jit
    def score_map(score):
        value = _cute_store(score if prefix is None else score * prefix.scale, boundary.producer.output)
        for index, operation in enumerate(boundary.score_maps):
            expression = prefix.residual_expression if prefix is not None and index == 0 else operation.expression
            value = evaluate_cute_scalar_expression(expression, {operation.inputs[0].name: value})
            value = _cute_store(value, operation.output)
        return value

    return score_map


def _cute_conversion_faithful_sum_map(boundary: ConversionFaithfulNormalizedExpBoundary):
    @cute.jit
    def sum_map(score, maximum):
        centered = evaluate_cute_scalar_expression(
            boundary.center.expression,
            {boundary.score_map.output.name: score, boundary.maximum.output.name: maximum},
        )
        centered = _cute_store(centered, boundary.center.output)
        exponential = evaluate_cute_scalar_expression(
            boundary.exponential.expression,
            {boundary.center.output.name: centered},
        )
        exponential = _cute_store(exponential, boundary.exponential.output)
        converted = evaluate_cute_scalar_expression(
            boundary.sum_input.expression,
            {boundary.exponential.output.name: exponential},
        )
        return _cute_store(converted, boundary.sum_input.output)

    return sum_map


def _cute_conversion_faithful_lse_map(boundary: ConversionFaithfulNormalizedExpBoundary):
    @cute.jit
    def lse_map(maximum, total):
        value = evaluate_cute_scalar_expression(
            boundary.log_normalizer.expression,
            {boundary.maximum.output.name: maximum, boundary.normalizer.output.name: total},
        )
        return _cute_store(value, boundary.log_normalizer.output)

    return lse_map


def _cute_conversion_faithful_normalized_map(boundary: ConversionFaithfulNormalizedExpBoundary):
    denominator_input = next(value.name for value in boundary.normalize.inputs if value != boundary.exponential.output)

    @cute.jit
    def normalized_map(score, maximum, total):
        centered = evaluate_cute_scalar_expression(
            boundary.center.expression,
            {boundary.score_map.output.name: score, boundary.maximum.output.name: maximum},
        )
        centered = _cute_store(centered, boundary.center.output)
        exponential = evaluate_cute_scalar_expression(
            boundary.exponential.expression,
            {boundary.center.output.name: centered},
        )
        exponential = _cute_store(exponential, boundary.exponential.output)
        denominator = evaluate_cute_scalar_expression(
            boundary.denominator.expression,
            {boundary.normalizer.output.name: total},
        )
        denominator = _cute_store(denominator, boundary.denominator.output)
        normalized = evaluate_cute_scalar_expression(
            boundary.normalize.expression,
            {boundary.exponential.output.name: exponential, denominator_input: denominator},
        )
        return _cute_store(normalized, boundary.normalize.output)

    return normalized_map


def _cute_conversion_faithful_finalizer(boundary: ConversionFaithfulNormalizedExpBoundary):
    @cute.jit
    def finalizer(accumulator):
        contracted = _cute_store(accumulator, boundary.consumer.output)
        if cutlass.const_expr(boundary.finalizer is not None):
            finalized = evaluate_cute_scalar_expression(
                boundary.finalizer.expression,
                {boundary.consumer.output.name: contracted},
            )
            return _cute_store(finalized, boundary.finalizer.output)
        return contracted

    return finalizer


@cute.jit
def _cute_store(value, destination: cutlass.Constexpr[ProgramValue]):
    if cutlass.const_expr(destination.dtype is DType.BF16):
        return cutlass.BFloat16(value)
    return cutlass.Float32(value)


class _DiagnosticSimtTwoPassNormalizedExp:
    def __init__(self, boundary: ConversionFaithfulNormalizedExpBoundary):
        self.boundary = boundary
        layout = boundary.relation_layout
        self.batch = layout.batch_extent
        self.owner = layout.owner_extent
        self.streamed = layout.streamed_extent
        self.relations = layout.flattened_relation_axis.extent
        self.shared_relations = layout.shared_relation_axis.extent
        self.relation_group = layout.relation_group_size
        self.producer_feature = layout.producer_feature_extent
        self.consumer_feature = layout.consumer_feature_extent
        self.normalize_denominator_input = next(
            value.name for value in boundary.normalize.inputs if value != boundary.exponential.output
        )
        self.rejected_score = (
            boundary.score_map.rejected_value if boundary.score_map.rejected_value is not None else -math.inf
        )

        partials_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _THREADS], 16]

        @cute.struct
        class StatisticsScratch:
            partials: partials_struct

        self.StatisticsScratch = StatisticsScratch
        self.predicate = (
            None
            if boundary.domain_predicate is None
            else cute_affine_index_predicate(
                boundary.domain_predicate,
                owner_dimension=boundary.owner_dimension,
                streamed_dimension=boundary.maximum.reduction_axes[0].id,
            )
        )

    @cute.jit
    def __call__(self, query, key, value, output, row_max, row_sum, log_sum_exp, stream):
        rows = self.batch * self.owner * self.relations
        self.statistics_kernel(query, key, row_max, row_sum, log_sum_exp).launch(
            grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )
        self.weighted_contract_kernel(query, key, value, row_max, row_sum, output).launch(
            grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def statistics_kernel(self, query, key, row_max, row_sum, log_sum_exp):
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        batch, owner, relation = self._row_coordinates(row)
        shared_relation = relation // self.relation_group
        storage = cutlass.utils.SmemAllocator().allocate(self.StatisticsScratch)
        partials = storage.partials.get_tensor(cute.make_layout((_THREADS,)))
        local_maximum = cutlass.Float32(-math.inf)
        for iteration in cutlass.range_constexpr(cute.ceil_div(self.streamed, _THREADS)):
            streamed = iteration * _THREADS + thread
            if streamed < self.streamed:
                mapped = self._mapped_score(query, key, batch, owner, relation, shared_relation, streamed)
                if cutlass.Float32(mapped) > local_maximum:
                    local_maximum = cutlass.Float32(mapped)
        partials[thread] = local_maximum
        cute.arch.barrier()
        if thread == 0:
            maximum = cutlass.Float32(-math.inf)
            for lane in cutlass.range_constexpr(_THREADS):
                if partials[lane] > maximum:
                    maximum = partials[lane]
            row_max[batch, relation, owner] = cutlass.BFloat16(maximum)
        cute.arch.barrier()
        maximum = row_max[batch, relation, owner]
        local_total = cutlass.Float32(0.0)
        for iteration in cutlass.range_constexpr(cute.ceil_div(self.streamed, _THREADS)):
            streamed = iteration * _THREADS + thread
            if streamed < self.streamed:
                mapped = self._mapped_score(query, key, batch, owner, relation, shared_relation, streamed)
                centered = self._map(
                    self.boundary.center,
                    {self.boundary.score_map.output.name: mapped, self.boundary.maximum.output.name: maximum},
                )
                exponential = self._map(
                    self.boundary.exponential,
                    {self.boundary.center.output.name: centered},
                )
                sum_input = self._map(
                    self.boundary.sum_input,
                    {self.boundary.exponential.output.name: exponential},
                )
                local_total += cutlass.Float32(sum_input)
        partials[thread] = local_total
        cute.arch.barrier()
        if thread == 0:
            total = cutlass.Float32(0.0)
            for lane in cutlass.range_constexpr(_THREADS):
                total += partials[lane]
            row_sum[batch, relation, owner] = total
            log_sum_exp[batch, relation, owner] = self._map(
                self.boundary.log_normalizer,
                {
                    self.boundary.maximum.output.name: maximum,
                    self.boundary.normalizer.output.name: total,
                },
            )

    @cute.kernel
    def weighted_contract_kernel(self, query, key, value, row_max, row_sum, output):
        feature, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if feature < self.consumer_feature:
            batch, owner, relation = self._row_coordinates(row)
            shared_relation = relation // self.relation_group
            maximum = row_max[batch, relation, owner]
            total_state = cutlass.Float32(row_sum[batch, relation, owner])
            denominator = self._map(
                self.boundary.denominator,
                {self.boundary.normalizer.output.name: total_state},
            )
            accumulator = cutlass.Float32(0.0)
            for streamed in cutlass.range_constexpr(self.streamed):
                mapped = self._mapped_score(query, key, batch, owner, relation, shared_relation, streamed)
                centered = self._map(
                    self.boundary.center,
                    {self.boundary.score_map.output.name: mapped, self.boundary.maximum.output.name: maximum},
                )
                exponential = self._map(
                    self.boundary.exponential,
                    {self.boundary.center.output.name: centered},
                )
                normalized = self._map(
                    self.boundary.normalize,
                    {
                        self.boundary.exponential.output.name: exponential,
                        self.normalize_denominator_input: denominator,
                    },
                )
                accumulator += cutlass.Float32(normalized) * cutlass.Float32(
                    value[batch, streamed, shared_relation, feature]
                )
            contracted = self._store(accumulator, self.boundary.consumer.output)
            finalized = contracted
            if cutlass.const_expr(self.boundary.finalizer is not None):
                finalized = self._map(
                    self.boundary.finalizer,
                    {self.boundary.consumer.output.name: contracted},
                )
            output[batch, owner, relation, feature] = finalized

    @cute.jit
    def _mapped_score(self, query, key, batch, owner, relation, shared_relation, streamed):
        total = cutlass.Float32(0.0)
        for feature in cutlass.range_constexpr(self.producer_feature):
            total += cutlass.Float32(query[batch, owner, relation, feature]) * cutlass.Float32(
                key[batch, streamed, shared_relation, feature]
            )
        raw = self._store(total, self.boundary.producer.output)
        accepted = True
        if cutlass.const_expr(self.predicate is not None):
            accepted = self.predicate(batch, relation, owner, streamed, None, None)
        mapped = raw
        for operation in self.boundary.score_maps:
            mapped = self._map(operation, {operation.inputs[0].name: mapped})
        if cutlass.const_expr(self.predicate is not None):
            selected = mapped
            if accepted:
                selected = mapped
            else:
                selected = cutlass.BFloat16(self.rejected_score)
            mapped = selected
        return mapped

    @cute.jit
    def _map(self, operation: cutlass.Constexpr[MapPrimitive], bindings):
        value = evaluate_cute_scalar_expression(operation.expression, bindings)
        return self._store(value, operation.output)

    @cute.jit
    def _store(self, value, destination: cutlass.Constexpr[ProgramValue]):
        if cutlass.const_expr(destination.dtype is DType.BF16):
            return cutlass.BFloat16(value)
        return cutlass.Float32(value)

    @cute.jit
    def _row_coordinates(self, row):
        relation = row % self.relations
        residual = row // self.relations
        owner = residual % self.owner
        batch = residual // self.owner
        return batch, owner, relation


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...] | None = None,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype is not dtype:
        raise ValueError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA-resident")
    if stride is not None and tuple(tensor.stride()) != stride:
        raise ValueError(f"{name} has strides {tuple(tensor.stride())}, expected {stride}")


def _rename_scalar_input(expression: ScalarExpression, source_name: str, result_name: str) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT and expression.input_name == source_name:
        return replace(expression, input_name=result_name)
    return replace(
        expression,
        operands=tuple(_rename_scalar_input(operand, source_name, result_name) for operand in expression.operands),
    )
