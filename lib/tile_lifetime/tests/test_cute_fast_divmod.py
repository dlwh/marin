# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct

from tile_lifetime.cute_fast_divmod import encode_cute_fast_divmod_divisor


def _divide_with_cute_fast_divmod(dividend: int, encoded_divisor: bytes) -> int:
    _, multiplier, shift_one, shift_two = struct.unpack("<IIBBxx", encoded_divisor)
    high_product = (dividend * multiplier) >> 32
    return (high_product + ((dividend - high_product) >> shift_one)) >> shift_two


def test_cute_fast_divmod_encodes_non_power_of_two_divisor() -> None:
    encoded_divisor = encode_cute_fast_divmod_divisor(3)

    assert encoded_divisor == bytes.fromhex("030000005655555501010000")
    dividends = (*range(256), 2**16 - 1, 2**31 - 1, 2**32 - 1)
    assert tuple(_divide_with_cute_fast_divmod(value, encoded_divisor) for value in dividends) == tuple(
        value // 3 for value in dividends
    )
