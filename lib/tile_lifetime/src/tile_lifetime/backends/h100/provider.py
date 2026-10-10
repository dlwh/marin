# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy entry point for the optional SM90 physical compiler dependencies."""


def compile_normalized_exp_contract_artifact(serialized: str) -> bytes | None:
    """Compile a supported SM90 request after its CUDA dependencies are needed."""
    from tile_lifetime.backends.h100.physical_normalized_exp_contract_compiler import (  # noqa: PLC0415
        compile_h100_normalized_exp_contract_artifact,
    )

    return compile_h100_normalized_exp_contract_artifact(serialized)
