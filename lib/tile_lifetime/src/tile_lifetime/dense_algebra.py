# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build generic dense algebra and retain historical semantic erasure."""

from shuttle.ir import DType
from tile_lifetime.ir import LayerNormOp, LinearOp, ResidualAddOp, RMSNormOp, TensorGraph, TensorValue
from tile_lifetime.plan import SemanticLoweringStep
from tile_lifetime.semantic_erasure import ErasedTensorProgram, build_tensor_erasure_report
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_select,
    scalar_unary,
)
from tile_lifetime.tensor_program import (
    NumericalPolicy as TensorNumericalPolicy,
)


class DenseSemanticErasureError(ValueError):
    """The recovered dense graph is outside the generic prototype vocabulary."""


def dense_composition_tensor_program(
    *,
    batch_size: int,
    sequence_length: int,
    model_dimension: int,
    heads: int,
    query_key_dimension: int,
    value_dimension: int,
    feed_forward_dimension: int,
    epsilon: float = 1e-6,
    causal: bool = True,
    dtype: DType = DType.FP32,
    accumulation_dtype: DType = DType.FP32,
    numerical_policy: TensorNumericalPolicy = TensorNumericalPolicy.SOURCE_ORDERED,
) -> TensorProgram:
    """Build one transformer-like composition from generic tensor algebra.

    The adapter name describes the test program; every operation in the
    returned program is a generic Contract, Map, or Fold. Contraction feature
    axes and the normalized-exponential reduction axis remain distinct.
    """
    extents = (
        batch_size,
        sequence_length,
        model_dimension,
        heads,
        query_key_dimension,
        value_dimension,
        feed_forward_dimension,
    )
    if any(extent <= 0 for extent in extents) or epsilon <= 0:
        raise ValueError("dense composition extents and epsilon must be positive")

    batch = TensorAxis(0, batch_size, "batch")
    query_token = TensorAxis(1, sequence_length, "query_token")
    key_token = TensorAxis(2, sequence_length, "key_token")
    model = TensorAxis(3, model_dimension, "model_feature")
    head = TensorAxis(4, heads, "head")
    query_key = TensorAxis(5, query_key_dimension, "query_key_feature")
    value_feature = TensorAxis(6, value_dimension, "value_feature")
    feed_forward = TensorAxis(7, feed_forward_dimension, "feed_forward_feature")

    source = ProgramValue("source", (batch, query_token, model), dtype)
    input_scale = ProgramValue("input_scale", (model,), dtype)
    query_weight = ProgramValue("query_weight", (model, head, query_key), dtype)
    key_weight = ProgramValue("key_weight", (model, head, query_key), dtype)
    value_weight = ProgramValue("value_weight", (model, head, value_feature), dtype)
    output_weight = ProgramValue("output_weight", (head, value_feature, model), dtype)
    mlp_scale = ProgramValue("mlp_scale", (model,), dtype)
    gate_weight = ProgramValue("gate_weight", (model, feed_forward), dtype)
    up_weight = ProgramValue("up_weight", (model, feed_forward), dtype)
    down_weight = ProgramValue("down_weight", (feed_forward, model), dtype)
    query_position = ProgramValue("query_position", (query_token,), DType.INT32)
    key_position = ProgramValue("key_position", (key_token,), DType.INT32)

    operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive] = []

    def mapped(
        name: str,
        inputs: tuple[ProgramValue, ...],
        output: ProgramValue,
        expression,
    ) -> ProgramValue:
        operations.append(MapPrimitive(name, inputs, output, expression))
        return output

    square = mapped(
        "input.square",
        (source,),
        ProgramValue("input.square", source.axes, accumulation_dtype),
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(source.name), scalar_input(source.name)),
    )
    sum_squares = ProgramValue("input.sum_squares", (batch, query_token), accumulation_dtype)
    operations.append(
        FoldPrimitive(
            "input.fold",
            square,
            sum_squares,
            (model,),
            FoldReducer.SUM,
            accumulation_dtype,
        )
    )
    inverse_scale = mapped(
        "input.finalize_fold",
        (sum_squares,),
        ProgramValue("input.inverse_scale", sum_squares.axes, accumulation_dtype),
        scalar_unary(
            ScalarExpressionKind.RSQRT,
            scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_binary(
                    ScalarExpressionKind.DIVIDE,
                    scalar_input(sum_squares.name),
                    scalar_constant(float(model_dimension)),
                ),
                scalar_constant(epsilon),
            ),
        ),
    )
    feature_scaled = mapped(
        "input.feature_scale",
        (source, input_scale),
        ProgramValue("input.feature_scaled", source.axes, dtype),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(source.name),
            scalar_input(input_scale.name),
        ),
    )
    normalized = mapped(
        "input.apply_fold",
        (feature_scaled, inverse_scale),
        ProgramValue("input.normalized", source.axes, dtype),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(feature_scaled.name),
            scalar_input(inverse_scale.name),
        ),
    )

    query = ProgramValue("query", (batch, query_token, head, query_key), accumulation_dtype)
    key = ProgramValue("key", (batch, key_token, head, query_key), accumulation_dtype)
    value = ProgramValue("value", (batch, key_token, head, value_feature), accumulation_dtype)
    operations.append(
        ContractPrimitive("project.query", (normalized, query_weight), query, (model,), accumulation_dtype)
    )
    token_view = AxisIndexMap(domain_axis=key_token, operand_axis=query_token)
    operations.extend(
        (
            ContractPrimitive(
                "project.key",
                (normalized, key_weight),
                key,
                (model,),
                accumulation_dtype,
                input_index_maps=((token_view,), ()),
            ),
            ContractPrimitive(
                "project.value",
                (normalized, value_weight),
                value,
                (model,),
                accumulation_dtype,
                input_index_maps=((token_view,), ()),
            ),
        )
    )

    raw_score = ProgramValue("score.raw", (batch, head, query_token, key_token), accumulation_dtype)
    operations.append(ContractPrimitive("score.contract", (query, key), raw_score, (query_key,), accumulation_dtype))
    scaled_score_expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input(raw_score.name),
        scalar_constant(query_key_dimension**-0.5),
    )
    score_inputs: tuple[ProgramValue, ...] = (raw_score,)
    if causal:
        scaled_score_expression = scalar_select(
            scalar_binary(
                ScalarExpressionKind.LESS_EQUAL,
                scalar_input(key_position.name),
                scalar_input(query_position.name),
            ),
            scaled_score_expression,
            scalar_constant(float("-inf")),
        )
        score_inputs = (raw_score, query_position, key_position)
    mapped_score = mapped(
        "score.map",
        score_inputs,
        ProgramValue("score.mapped", raw_score.axes, accumulation_dtype),
        scaled_score_expression,
    )
    row_axes = (batch, head, query_token)
    row_max = ProgramValue("score.row_max", row_axes, accumulation_dtype)
    operations.append(
        FoldPrimitive("score.maximum", mapped_score, row_max, (key_token,), FoldReducer.MAXIMUM, accumulation_dtype)
    )
    centered = mapped(
        "score.center",
        (mapped_score, row_max),
        ProgramValue("score.centered", raw_score.axes, accumulation_dtype),
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_input(mapped_score.name),
            scalar_input(row_max.name),
        ),
    )
    exponential = mapped(
        "score.exponential",
        (centered,),
        ProgramValue("score.exponential", raw_score.axes, accumulation_dtype),
        scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered.name)),
    )
    row_sum = ProgramValue("score.row_sum", row_axes, accumulation_dtype)
    operations.append(
        FoldPrimitive("score.sum", exponential, row_sum, (key_token,), FoldReducer.SUM, accumulation_dtype)
    )
    weighted = ProgramValue("weighted", (batch, query_token, head, value_feature), accumulation_dtype)
    operations.append(
        ContractPrimitive("value.contract", (exponential, value), weighted, (key_token,), accumulation_dtype)
    )
    attended = mapped(
        "value.normalize",
        (weighted, row_sum),
        ProgramValue("attended", weighted.axes, dtype),
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            scalar_input(weighted.name),
            scalar_input(row_sum.name),
        ),
    )
    projected = ProgramValue("projected", source.axes, accumulation_dtype)
    operations.append(
        ContractPrimitive(
            "output.contract", (attended, output_weight), projected, (head, value_feature), accumulation_dtype
        )
    )
    residual = mapped(
        "residual.add",
        (source, projected),
        ProgramValue("residual", source.axes, dtype),
        scalar_binary(ScalarExpressionKind.ADD, scalar_input(source.name), scalar_input(projected.name)),
    )

    mlp_square = mapped(
        "mlp.square",
        (residual,),
        ProgramValue("mlp.square", residual.axes, accumulation_dtype),
        scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input(residual.name), scalar_input(residual.name)),
    )
    mlp_sum_squares = ProgramValue("mlp.sum_squares", (batch, query_token), accumulation_dtype)
    operations.append(
        FoldPrimitive(
            "mlp.fold",
            mlp_square,
            mlp_sum_squares,
            (model,),
            FoldReducer.SUM,
            accumulation_dtype,
        )
    )
    mlp_inverse_scale = mapped(
        "mlp.finalize_fold",
        (mlp_sum_squares,),
        ProgramValue("mlp.inverse_scale", mlp_sum_squares.axes, accumulation_dtype),
        scalar_unary(
            ScalarExpressionKind.RSQRT,
            scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_binary(
                    ScalarExpressionKind.DIVIDE,
                    scalar_input(mlp_sum_squares.name),
                    scalar_constant(float(model_dimension)),
                ),
                scalar_constant(epsilon),
            ),
        ),
    )
    mlp_feature_scaled = mapped(
        "mlp.feature_scale",
        (residual, mlp_scale),
        ProgramValue("mlp.feature_scaled", residual.axes, dtype),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(residual.name),
            scalar_input(mlp_scale.name),
        ),
    )
    mlp_input = mapped(
        "mlp.apply_fold",
        (mlp_feature_scaled, mlp_inverse_scale),
        ProgramValue("mlp.input", residual.axes, dtype),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(mlp_feature_scaled.name),
            scalar_input(mlp_inverse_scale.name),
        ),
    )
    gate = ProgramValue("gate", (batch, query_token, feed_forward), accumulation_dtype)
    up = ProgramValue("up", gate.axes, accumulation_dtype)
    operations.extend(
        (
            ContractPrimitive("gate.contract", (mlp_input, gate_weight), gate, (model,), accumulation_dtype),
            ContractPrimitive("up.contract", (mlp_input, up_weight), up, (model,), accumulation_dtype),
        )
    )
    sigmoid_gate = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input(gate.name),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            scalar_unary(
                ScalarExpressionKind.EXP,
                scalar_binary(
                    ScalarExpressionKind.SUBTRACT,
                    scalar_constant(0.0),
                    scalar_input(gate.name),
                ),
            ),
        ),
    )
    hidden = mapped(
        "activation.map",
        (gate, up),
        ProgramValue("hidden", gate.axes, dtype),
        scalar_binary(ScalarExpressionKind.MULTIPLY, sigmoid_gate, scalar_input(up.name)),
    )
    down = ProgramValue("down", source.axes, accumulation_dtype)
    operations.append(
        ContractPrimitive("down.contract", (hidden, down_weight), down, (feed_forward,), accumulation_dtype)
    )
    output = mapped(
        "output.add",
        (residual, down),
        ProgramValue("output", source.axes, dtype),
        scalar_binary(ScalarExpressionKind.ADD, scalar_input(residual.name), scalar_input(down.name)),
    )
    inputs = (
        source,
        input_scale,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        mlp_scale,
        gate_weight,
        up_weight,
        down_weight,
        *((query_position, key_position) if causal else ()),
    )
    return TensorProgram(
        inputs=inputs,
        operations=tuple(operations),
        outputs=(output,),
        numerical_policy=numerical_policy,
    )


def erase_dense_semantics(graph: TensorGraph) -> ErasedTensorProgram:
    """Lower named dense frontend operations to a closed generic tensor program."""
    values: dict[int, ProgramValue] = {}
    inputs: list[ProgramValue] = []
    operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive] = []
    lowering_steps: list[SemanticLoweringStep] = []
    next_axis_id = 0

    def new_axis(extent: int, label: str | None = None) -> TensorAxis:
        nonlocal next_axis_id
        axis = TensorAxis(next_axis_id, extent, label)
        next_axis_id += 1
        return axis

    def bind(value: TensorValue, axes: tuple[TensorAxis, ...], *, external: bool) -> ProgramValue:
        existing = values.get(value.id)
        if existing is not None:
            if existing.axes != axes:
                raise DenseSemanticErasureError(
                    f"value {value.name!r} has inconsistent logical axes {existing.axes} and {axes}"
                )
            return existing
        if tuple(axis.extent for axis in axes) != value.shape:
            raise DenseSemanticErasureError(
                f"logical axes {tuple(axis.extent for axis in axes)} do not match {value.name!r} shape {value.shape}"
            )
        program_value = ProgramValue(value.name, axes, value.dtype)
        values[value.id] = program_value
        if external:
            inputs.append(program_value)
        return program_value

    def external(value: TensorValue, axes: tuple[TensorAxis, ...] | None = None) -> ProgramValue:
        existing = values.get(value.id)
        if existing is not None:
            if axes is not None and existing.axes != axes:
                raise DenseSemanticErasureError(f"value {value.name!r} requires an unsupported logical-axis unification")
            return existing
        if axes is None:
            axes = tuple(new_axis(extent) for extent in value.shape)
        return bind(value, axes, external=True)

    def produced(value: TensorValue, axes: tuple[TensorAxis, ...]) -> ProgramValue:
        return bind(value, axes, external=False)

    for operation in graph.operations:
        if isinstance(operation, LinearOp):
            input_value = external(operation.input)
            if len(input_value.axes) != 2:
                raise DenseSemanticErasureError("the generic H100 contraction prototype requires rank-two inputs")
            row_axis, reduction_axis = input_value.axes
            output_axis = new_axis(operation.output.shape[1])
            weight = external(operation.weight, (reduction_axis, output_axis))
            output = produced(operation.output, (row_axis, output_axis))
            operations.append(
                ContractPrimitive(
                    name=f"contract.{len(operations)}",
                    inputs=(input_value, weight),
                    output=output,
                    reduction_axes=(reduction_axis,),
                    accumulation_dtype=operation.accumulation_dtype,
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Contract",)))
            continue

        if isinstance(operation, ResidualAddOp):
            left = external(operation.left)
            right = external(operation.right, left.axes)
            output = produced(operation.output, left.axes)
            operations.append(
                MapPrimitive(
                    name=f"map.{len(operations)}",
                    inputs=(left, right),
                    output=output,
                    expression=scalar_binary(
                        ScalarExpressionKind.ADD,
                        scalar_input(left.name),
                        scalar_input(right.name),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Map",)))
            continue

        if isinstance(operation, RMSNormOp):
            source = external(operation.input)
            reduction_axis = source.axes[operation.axis]
            gamma = external(operation.gamma, (reduction_axis,))
            row_axes = tuple(axis for axis in source.axes if axis != reduction_axis)
            square = ProgramValue(f"value.{operation.id}.square", source.axes, operation.reduction_dtype)
            summed = ProgramValue(f"value.{operation.id}.fold", row_axes, operation.reduction_dtype)
            inverse_scale = ProgramValue(f"value.{operation.id}.row_scalar", row_axes, operation.reduction_dtype)
            feature_scaled = ProgramValue(f"value.{operation.id}.feature_scaled", source.axes, source.dtype)
            output = produced(operation.output, source.axes)
            operations.extend(
                (
                    MapPrimitive(
                        name=f"map.{len(operations)}",
                        inputs=(source,),
                        output=square,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(source.name),
                            scalar_input(source.name),
                        ),
                    ),
                    FoldPrimitive(
                        name=f"fold.{len(operations) + 1}",
                        input=square,
                        output=summed,
                        reduction_axes=(reduction_axis,),
                        reducer=FoldReducer.SUM,
                        accumulation_dtype=operation.reduction_dtype,
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 2}",
                        inputs=(summed,),
                        output=inverse_scale,
                        expression=scalar_unary(
                            ScalarExpressionKind.RSQRT,
                            scalar_binary(
                                ScalarExpressionKind.ADD,
                                scalar_binary(
                                    ScalarExpressionKind.DIVIDE,
                                    scalar_input(summed.name),
                                    scalar_constant(float(reduction_axis.extent)),
                                ),
                                scalar_constant(operation.epsilon),
                            ),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 3}",
                        inputs=(source, gamma),
                        output=feature_scaled,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(source.name),
                            scalar_input(gamma.name),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 4}",
                        inputs=(feature_scaled, inverse_scale),
                        output=output,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(feature_scaled.name),
                            scalar_input(inverse_scale.name),
                        ),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Map", "Fold", "Map")))
            continue

        if isinstance(operation, LayerNormOp):
            source = external(operation.input)
            reduction_axis = source.axes[operation.axis]
            gamma = external(operation.gamma, (reduction_axis,))
            beta = external(operation.beta, (reduction_axis,))
            row_axes = tuple(axis for axis in source.axes if axis != reduction_axis)
            square = ProgramValue(f"value.{operation.id}.square", source.axes, operation.reduction_dtype)
            summed = ProgramValue(f"value.{operation.id}.sum", row_axes, operation.reduction_dtype)
            sum_squares = ProgramValue(f"value.{operation.id}.sum_squares", row_axes, operation.reduction_dtype)
            mean = ProgramValue(f"value.{operation.id}.mean", row_axes, operation.reduction_dtype)
            inverse_scale = ProgramValue(f"value.{operation.id}.row_scalar", row_axes, operation.reduction_dtype)
            centered = ProgramValue(f"value.{operation.id}.centered", source.axes, operation.reduction_dtype)
            row_scaled = ProgramValue(f"value.{operation.id}.row_scaled", source.axes, operation.reduction_dtype)
            feature_scaled = ProgramValue(f"value.{operation.id}.feature_scaled", source.axes, source.dtype)
            output = produced(operation.output, source.axes)
            operations.extend(
                (
                    MapPrimitive(
                        name=f"map.{len(operations)}",
                        inputs=(source,),
                        output=square,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(source.name),
                            scalar_input(source.name),
                        ),
                    ),
                    FoldPrimitive(
                        name=f"fold.{len(operations) + 1}",
                        input=source,
                        output=summed,
                        reduction_axes=(reduction_axis,),
                        reducer=FoldReducer.SUM,
                        accumulation_dtype=operation.reduction_dtype,
                    ),
                    FoldPrimitive(
                        name=f"fold.{len(operations) + 2}",
                        input=square,
                        output=sum_squares,
                        reduction_axes=(reduction_axis,),
                        reducer=FoldReducer.SUM,
                        accumulation_dtype=operation.reduction_dtype,
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 3}",
                        inputs=(summed,),
                        output=mean,
                        expression=scalar_binary(
                            ScalarExpressionKind.DIVIDE,
                            scalar_input(summed.name),
                            scalar_constant(float(reduction_axis.extent)),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 4}",
                        inputs=(sum_squares, mean),
                        output=inverse_scale,
                        expression=scalar_unary(
                            ScalarExpressionKind.RSQRT,
                            scalar_binary(
                                ScalarExpressionKind.ADD,
                                scalar_binary(
                                    ScalarExpressionKind.SUBTRACT,
                                    scalar_binary(
                                        ScalarExpressionKind.DIVIDE,
                                        scalar_input(sum_squares.name),
                                        scalar_constant(float(reduction_axis.extent)),
                                    ),
                                    scalar_binary(
                                        ScalarExpressionKind.MULTIPLY,
                                        scalar_input(mean.name),
                                        scalar_input(mean.name),
                                    ),
                                ),
                                scalar_constant(operation.epsilon),
                            ),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 5}",
                        inputs=(source, mean),
                        output=centered,
                        expression=scalar_binary(
                            ScalarExpressionKind.SUBTRACT,
                            scalar_input(source.name),
                            scalar_input(mean.name),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 6}",
                        inputs=(centered, inverse_scale),
                        output=row_scaled,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(centered.name),
                            scalar_input(inverse_scale.name),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 7}",
                        inputs=(row_scaled, gamma),
                        output=feature_scaled,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(row_scaled.name),
                            scalar_input(gamma.name),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 8}",
                        inputs=(feature_scaled, beta),
                        output=output,
                        expression=scalar_binary(
                            ScalarExpressionKind.ADD,
                            scalar_input(feature_scaled.name),
                            scalar_input(beta.name),
                        ),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Map", "Fold", "Map")))
            continue

        raise DenseSemanticErasureError(
            f"dense semantic erasure does not support frontend operation {type(operation).__name__}"
        )

    outputs = tuple(
        values[value.id]
        for value in graph.values
        if value.id in values and graph.producer(value) is not None and not graph.consumers(value)
    )
    program = TensorProgram(inputs=tuple(inputs), operations=tuple(operations), outputs=outputs)
    source_semantics = tuple(dict.fromkeys(type(operation).__name__ for operation in graph.operations))
    report = build_tensor_erasure_report(
        program,
        source_semantics=source_semantics,
        lowering_steps=tuple(lowering_steps),
    )
    return ErasedTensorProgram(program=program, report=report)
