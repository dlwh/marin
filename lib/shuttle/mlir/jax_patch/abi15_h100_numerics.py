# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent exact-bit numerical oracle for the ABI15 H100 evidence cell."""

import hashlib
import struct
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

ROWS = 33
REDUCTION = 2
COLUMNS = 34


class ReductionPolicy(StrEnum):
    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True)
class Inputs:
    lhs: tuple[int, ...]
    rhs: tuple[int, ...]
    output_cotangent: tuple[int, ...]


@dataclass(frozen=True)
class Outputs:
    primal: tuple[int, ...]
    lhs_cotangent: tuple[int, ...]
    rhs_cotangent: tuple[int, ...]


def bf16_sha256(values: tuple[int, ...]) -> str:
    """Return the SHA-256 of little-endian raw BF16 storage."""
    return hashlib.sha256(struct.pack(f"<{len(values)}H", *values)).hexdigest()


def _round_ratio_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        return quotient + 1
    return quotient


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if exponent >= 0:
        if value.numerator < value.denominator << exponent:
            exponent -= 1
    elif value.numerator << -exponent < value.denominator:
        exponent -= 1
    return exponent


def fraction_to_f32_bits(value: Fraction) -> int:
    """Round a finite rational to IEEE binary32 using round-to-nearest-even."""
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    if exponent < -126:
        significand = _round_ratio_ties_to_even(magnitude.numerator << 149, magnitude.denominator)
        if significand == 0:
            return sign
        if significand >= 1 << 23:
            return sign | 0x00800000
        return sign | significand
    shift = 23 - exponent
    numerator = magnitude.numerator << shift if shift >= 0 else magnitude.numerator
    denominator = magnitude.denominator if shift >= 0 else magnitude.denominator << -shift
    significand = _round_ratio_ties_to_even(numerator, denominator)
    if significand == 1 << 24:
        exponent += 1
        significand = 1 << 23
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def f32_bits_to_fraction(bits: int) -> Fraction:
    """Decode a finite IEEE binary32 bit pattern exactly."""
    sign = -1 if bits & 0x80000000 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("non-finite binary32 value")
    if exponent == 0:
        return sign * Fraction(fraction, 1 << 149)
    significand = (1 << 23) | fraction
    power = exponent - 127 - 23
    if power >= 0:
        return sign * Fraction(significand << power)
    return sign * Fraction(significand, 1 << -power)


def bf16_bits_to_fraction(bits: int) -> Fraction:
    """Decode a finite BF16 bit pattern exactly."""
    return f32_bits_to_fraction(bits << 16)


def f32_bits_to_bf16_bits(bits: int) -> int:
    """Round finite binary32 bits to BF16 using round-to-nearest-even."""
    upper = bits >> 16
    lower = bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper % 2 == 1):
        upper += 1
    return upper & 0xFFFF


def fraction_to_bf16_bits(value: Fraction) -> int:
    """Round a finite rational to BF16 through the explicit binary32 boundary."""
    return f32_bits_to_bf16_bits(fraction_to_f32_bits(value))


def _fma(accumulator: int, lhs: int, rhs: int) -> int:
    exact = bf16_bits_to_fraction(lhs) * bf16_bits_to_fraction(rhs) + f32_bits_to_fraction(accumulator)
    return fraction_to_f32_bits(exact)


def _add(lhs: int, rhs: int) -> int:
    return fraction_to_f32_bits(f32_bits_to_fraction(lhs) + f32_bits_to_fraction(rhs))


def _reduce(products: tuple[tuple[int, int], ...], policy: ReductionPolicy) -> int:
    if policy == ReductionPolicy.SOURCE_ORDERED:
        accumulator = 0
        for lhs, rhs in products:
            accumulator = _fma(accumulator, lhs, rhs)
        return accumulator
    lanes = [0] * 32
    for lane in range(32):
        for index in range(lane, len(products), 32):
            lhs, rhs = products[index]
            lanes[lane] = _fma(lanes[lane], lhs, rhs)
    for offset in (16, 8, 4, 2, 1):
        previous = lanes.copy()
        for lane in range(32 - offset):
            lanes[lane] = _add(previous[lane], previous[lane + offset])
    return lanes[0]


def _dense_value(numerator: int) -> int:
    return fraction_to_bf16_bits(Fraction(numerator, 8))


def dense_inputs() -> Inputs:
    """Return exact dyadic inputs with independent row, column, and operand tags."""
    lhs = tuple(
        _dense_value(((17 * row + 29 * reduction + 3) % 31) - 15)
        for row in range(ROWS)
        for reduction in range(REDUCTION)
    )
    rhs = tuple(
        _dense_value(((23 * reduction + 11 * column + 5) % 29) - 14)
        for reduction in range(REDUCTION)
        for column in range(COLUMNS)
    )
    output_cotangent = tuple(
        _dense_value(((13 * row + 7 * column + 1) % 27) - 13) for row in range(ROWS) for column in range(COLUMNS)
    )
    return Inputs(lhs=lhs, rhs=rhs, output_cotangent=output_cotangent)


def sparse_policy_inputs() -> Inputs:
    """Return the +32-lane policy discriminator as raw BF16 values."""
    lhs = [0] * (ROWS * REDUCTION)
    rhs = [0] * (REDUCTION * COLUMNS)
    output_cotangent = [0] * (ROWS * COLUMNS)
    for row in (0, 1, 32):
        lhs[row * REDUCTION] = 0x3F80
    for column in (0, 1, 32):
        rhs[column] = 0x3F80
    output_cotangent[0] = 0x4C00
    output_cotangent[COLUMNS] = 0x3F80
    output_cotangent[32 * COLUMNS] = 0xCC00
    output_cotangent[1] = 0x3F80
    output_cotangent[32] = 0xCC00
    return Inputs(tuple(lhs), tuple(rhs), tuple(output_cotangent))


def reference(inputs: Inputs, policy: ReductionPolicy) -> Outputs:
    """Evaluate the composed matmul and VJP with the selected closed reduction."""
    if len(inputs.lhs) != ROWS * REDUCTION:
        raise ValueError("lhs shape mismatch")
    if len(inputs.rhs) != REDUCTION * COLUMNS:
        raise ValueError("rhs shape mismatch")
    if len(inputs.output_cotangent) != ROWS * COLUMNS:
        raise ValueError("output cotangent shape mismatch")

    primal = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            products = tuple(
                (
                    inputs.lhs[row * REDUCTION + reduction],
                    inputs.rhs[reduction * COLUMNS + column],
                )
                for reduction in range(REDUCTION)
            )
            primal.append(f32_bits_to_bf16_bits(_reduce(products, policy)))

    lhs_cotangent = []
    for row in range(ROWS):
        for reduction in range(REDUCTION):
            products = tuple(
                (
                    inputs.output_cotangent[row * COLUMNS + column],
                    inputs.rhs[reduction * COLUMNS + column],
                )
                for column in range(COLUMNS)
            )
            lhs_cotangent.append(f32_bits_to_bf16_bits(_reduce(products, policy)))

    rhs_cotangent = []
    for reduction in range(REDUCTION):
        for column in range(COLUMNS):
            products = tuple(
                (
                    inputs.lhs[row * REDUCTION + reduction],
                    inputs.output_cotangent[row * COLUMNS + column],
                )
                for row in range(ROWS)
            )
            rhs_cotangent.append(f32_bits_to_bf16_bits(_reduce(products, policy)))
    return Outputs(tuple(primal), tuple(lhs_cotangent), tuple(rhs_cotangent))
