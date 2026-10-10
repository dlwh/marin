# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host encoding for CuTe's 32-bit fast-divmod divisor ABI."""

from __future__ import annotations

import struct

_UINT32_BITS = 32
_UINT32_MAX = 2**_UINT32_BITS - 1


def encode_cute_fast_divmod_divisor(divisor: int) -> bytes:
    """Encode a positive 32-bit divisor for a CuTe kernel parameter."""
    if divisor <= 0 or divisor > _UINT32_MAX:
        raise ValueError("CuTe fast-divmod divisor must be a positive 32-bit integer")

    exponent = (divisor - 1).bit_length()
    multiplier = ((1 << _UINT32_BITS) * ((1 << exponent) - divisor)) // divisor + 1
    shift_one = min(exponent, 1)
    shift_two = max(exponent - 1, 0)
    return struct.pack("<IIBBxx", divisor, multiplier, shift_one, shift_two)
