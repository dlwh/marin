# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic schedule-derived RMSNorm reference for the ABI21 preimage."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from abi15_h100_numerics import (
    bf16_bits_to_fraction,
    f32_bits_to_bf16_bits,
    f32_bits_to_fraction,
    fraction_to_bf16_bits,
    fraction_to_f32_bits,
)

ROW_COUNT = 14
FEATURE_COUNT = 19
EPSILON = Fraction(1, 1_000_000)


@dataclass(frozen=True)
class Inputs:
    """Raw BF16 public inputs for the architecture execution coordinate."""

    x: tuple[int, ...]
    weight: tuple[int, ...]
    output_cotangent: tuple[int, ...]


@dataclass(frozen=True)
class Outputs:
    """Raw BF16 public outputs in composed-boundary order."""

    result: tuple[int, ...]
    input_cotangent: tuple[int, ...]
    weight_cotangent: tuple[int, ...]


@dataclass(frozen=True)
class AnalyticOutputs:
    """Binary64 analytic RMSNorm outputs before public BF16 rounding."""

    result: tuple[float, ...]
    input_cotangent: tuple[float, ...]
    weight_cotangent: tuple[float, ...]


@dataclass(frozen=True)
class ErrorMetrics:
    """Per-role deviation from the independent binary64 reference."""

    maximum_absolute_error: float
    mean_absolute_error: float
    relative_linf_error: float
    maximum_bf16_ulp_error: int


def _f32(value: Fraction) -> int:
    return fraction_to_f32_bits(value)


def _binary(lhs: int, rhs: int, operation: Callable[[Fraction, Fraction], Fraction]) -> int:
    return _f32(operation(f32_bits_to_fraction(lhs), f32_bits_to_fraction(rhs)))


def _add(lhs: int, rhs: int) -> int:
    return _binary(lhs, rhs, lambda left, right: left + right)


def _multiply(lhs: int, rhs: int) -> int:
    return _binary(lhs, rhs, lambda left, right: left * right)


def _divide(lhs: int, rhs: int) -> int:
    return _binary(lhs, rhs, lambda left, right: left / right)


def _sqrt(value: int) -> int:
    source = f32_bits_to_fraction(value)
    if source < 0:
        raise ValueError("RMSNorm variance must be nonnegative")
    if source == 0:
        return 0
    low = 0
    high = 0x7F7FFFFF
    while low < high:
        candidate = (low + high + 1) // 2
        if f32_bits_to_fraction(candidate) ** 2 <= source:
            low = candidate
        else:
            high = candidate - 1
    lower = f32_bits_to_fraction(low)
    upper_bits = low + 1
    upper = f32_bits_to_fraction(upper_bits)
    midpoint = (lower + upper) / 2
    if source < midpoint**2 or (source == midpoint**2 and low % 2 == 0):
        return low
    return upper_bits


def schedule_tree_sum(values: tuple[int, ...]) -> int:
    """Reduce one f32 leaf tile with ABI21's emitted adjacent-pair tree."""
    if not values or len(values) > 32:
        raise ValueError("the architecture oracle requires one nonempty SIMT32 leaf tile")
    width = 1 << (len(values) - 1).bit_length()
    level = [*values, *([0] * (width - len(values)))]
    while len(level) > 1:
        level = [_add(level[index], level[index + 1]) for index in range(0, len(level), 2)]
    return _add(level[0], 0)


def _raw_bf16(numerator: int, denominator: int) -> int:
    return fraction_to_bf16_bits(Fraction(numerator, denominator))


def architecture_inputs() -> Inputs:
    """Return the predeclared rank-three architecture input cell."""
    x = tuple(
        _raw_bf16(((17 * row + 11 * feature + 3) % 31) - 15, 8)
        for row in range(ROW_COUNT)
        for feature in range(FEATURE_COUNT)
    )
    weight = tuple(_raw_bf16(24 + ((7 * feature + 5) % 17), 32) for feature in range(FEATURE_COUNT))
    output_cotangent = tuple(
        _raw_bf16(((13 * row + 5 * feature + 1) % 29) - 14, 8)
        for row in range(ROW_COUNT)
        for feature in range(FEATURE_COUNT)
    )
    return Inputs(x, weight, output_cotangent)


def _validate_inputs(inputs: Inputs) -> None:
    if len(inputs.x) != ROW_COUNT * FEATURE_COUNT:
        raise ValueError("x shape differs from [2,7,19]")
    if len(inputs.weight) != FEATURE_COUNT:
        raise ValueError("weight shape differs from [19]")
    if len(inputs.output_cotangent) != ROW_COUNT * FEATURE_COUNT:
        raise ValueError("output cotangent shape differs from [2,7,19]")
    for values in (inputs.x, inputs.weight, inputs.output_cotangent):
        if any(type(value) is not int or not 0 <= value <= 0xFFFF for value in values):
            raise ValueError("public input is not raw BF16 storage")
        if any(value & 0x7F80 == 0x7F80 for value in values):
            raise ValueError("public input must be finite")


def schedule_reference(inputs: Inputs) -> Outputs:
    """Evaluate the fixed-leaf SIMT32 recurrence and explicit FP32 boundaries."""
    _validate_inputs(inputs)
    x = tuple(value << 16 for value in inputs.x)
    weight = tuple(value << 16 for value in inputs.weight)
    output_cotangent = tuple(value << 16 for value in inputs.output_cotangent)
    feature_count = _f32(Fraction(FEATURE_COUNT))
    epsilon = _f32(EPSILON)
    minus_half = _f32(Fraction(-1, 2))
    one = _f32(Fraction(1))

    results: list[int] = []
    input_cotangents: list[int] = []
    normalized_rows: list[tuple[int, ...]] = []
    for row in range(ROW_COUNT):
        start = row * FEATURE_COUNT
        row_x = x[start : start + FEATURE_COUNT]
        row_dy = output_cotangent[start : start + FEATURE_COUNT]
        squares = tuple(_multiply(value, value) for value in row_x)
        mean_square = _divide(schedule_tree_sum(squares), feature_count)
        variance_epsilon = _add(mean_square, epsilon)
        inverse_rms = _divide(one, _sqrt(variance_epsilon))
        inverse_over_variance = _divide(inverse_rms, variance_epsilon)
        variance_scale = _multiply(minus_half, inverse_over_variance)

        normalized = tuple(_multiply(value, inverse_rms) for value in row_x)
        normalized_rows.append(normalized)
        weighted_cotangent = tuple(_multiply(dy, gamma) for dy, gamma in zip(row_dy, weight, strict=True))
        correlation = schedule_tree_sum(
            tuple(_multiply(value, cotangent) for value, cotangent in zip(row_x, weighted_cotangent, strict=True))
        )
        derivative_scale = schedule_tree_sum((_divide(_multiply(correlation, variance_scale), feature_count),))

        for value, normalized_value, gamma, cotangent in zip(row_x, normalized, weight, weighted_cotangent, strict=True):
            results.append(f32_bits_to_bf16_bits(_multiply(normalized_value, gamma)))
            direct = _multiply(cotangent, inverse_rms)
            first_variance_path = _multiply(value, derivative_scale)
            second_variance_path = _multiply(derivative_scale, value)
            input_cotangents.append(f32_bits_to_bf16_bits(_add(_add(direct, first_variance_path), second_variance_path)))

    weight_cotangents = []
    for feature in range(FEATURE_COUNT):
        products = tuple(
            _multiply(normalized_rows[row][feature], output_cotangent[row * FEATURE_COUNT + feature])
            for row in range(ROW_COUNT)
        )
        weight_cotangents.append(f32_bits_to_bf16_bits(schedule_tree_sum((schedule_tree_sum(products),))))
    return Outputs(tuple(results), tuple(input_cotangents), tuple(weight_cotangents))


def binary64_reference(inputs: Inputs) -> AnalyticOutputs:
    """Evaluate the closed RMSNorm and VJP equations in independent binary64."""
    x_exact, weight_exact, output_cotangent_exact = decoded_inputs(inputs)
    x = tuple(float(value) for value in x_exact)
    weight = tuple(float(value) for value in weight_exact)
    output_cotangent = tuple(float(value) for value in output_cotangent_exact)
    result: list[float] = []
    input_cotangent: list[float] = []
    normalized_rows: list[tuple[float, ...]] = []
    for row in range(ROW_COUNT):
        start = row * FEATURE_COUNT
        row_x = x[start : start + FEATURE_COUNT]
        row_dy = output_cotangent[start : start + FEATURE_COUNT]
        inverse_rms = 1.0 / math.sqrt(sum(value * value for value in row_x) / FEATURE_COUNT + float(EPSILON))
        normalized = tuple(value * inverse_rms for value in row_x)
        normalized_rows.append(normalized)
        correlation = sum(value * gamma * dy for value, gamma, dy in zip(row_x, weight, row_dy, strict=True))
        for value, normalized_value, gamma, dy in zip(row_x, normalized, weight, row_dy, strict=True):
            result.append(normalized_value * gamma)
            input_cotangent.append(gamma * dy * inverse_rms - value * correlation * inverse_rms**3 / FEATURE_COUNT)
    weight_cotangent = tuple(
        sum(normalized_rows[row][feature] * output_cotangent[row * FEATURE_COUNT + feature] for row in range(ROW_COUNT))
        for feature in range(FEATURE_COUNT)
    )
    return AnalyticOutputs(tuple(result), tuple(input_cotangent), weight_cotangent)


def error_metrics(actual: tuple[int, ...], expected: tuple[float, ...]) -> ErrorMetrics:
    """Return aggregate BF16-versus-binary64 metrics for one public role."""
    if not actual or len(actual) != len(expected):
        raise ValueError("actual and expected results must have the same nonzero size")
    observed = tuple(float(bf16_bits_to_fraction(value)) for value in actual)
    absolute = tuple(abs(value - reference) for value, reference in zip(observed, expected, strict=True))
    maximum = max(absolute)
    scale = max(abs(value) for value in expected)

    def ordered_bf16(bits: int) -> int:
        return bits ^ (0xFFFF if bits & 0x8000 else 0x8000)

    ulp = max(
        abs(ordered_bf16(value) - ordered_bf16(fraction_to_bf16_bits(Fraction.from_float(reference))))
        for value, reference in zip(actual, expected, strict=True)
    )
    return ErrorMetrics(maximum, sum(absolute) / len(absolute), maximum / scale if scale else 0.0, ulp)


def decoded_inputs(inputs: Inputs) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return exact rational values for independent analytic comparisons."""
    _validate_inputs(inputs)
    return tuple(
        tuple(bf16_bits_to_fraction(value) for value in values)
        for values in (inputs.x, inputs.weight, inputs.output_cotangent)
    )
