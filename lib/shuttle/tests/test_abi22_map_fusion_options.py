# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from shuttle.options import (
    PIPELINE_ABI_VERSION,
    ExecutionMode,
    Materialization,
    Numerics,
    Options,
    Tuning,
    canonical_options_json,
    options_digest,
)


def _tuning(materialization: Materialization) -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=materialization,
        maximum_candidates=1,
    )


@pytest.mark.parametrize(
    ("numerics", "materialization", "expected_json", "expected_digest"),
    [
        (
            Numerics.SOURCE_ORDERED,
            Materialization.PREFER_FUSION,
            '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
            '"pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],'
            '"materialization":"prefer_fusion","maximum_candidates":1,'
            '"pipeline_stages":1,"tile_sizes":[]}}',
            "f496b036c3ab59ed67c3c993b95fb7d4a94bb18bd96145d023daf112e9138cde",
        ),
        (
            Numerics.FAST,
            Materialization.PREFER_FUSION,
            '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
            '"pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],'
            '"materialization":"prefer_fusion","maximum_candidates":1,'
            '"pipeline_stages":1,"tile_sizes":[]}}',
            "10de636a98b8aef9f56270c33f288459fd3bf64d9206daf809ccbf8f04425bec",
        ),
        (
            Numerics.SOURCE_ORDERED,
            Materialization.PREFER_MATERIALIZATION,
            '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
            '"pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],'
            '"materialization":"prefer_materialization","maximum_candidates":1,'
            '"pipeline_stages":1,"tile_sizes":[]}}',
            "be51372b29a43ff74fc3b5757029f1bd8403367c00f77cc2ecde6b97c6cf14dc",
        ),
        (
            Numerics.FAST,
            Materialization.PREFER_MATERIALIZATION,
            '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
            '"pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],'
            '"materialization":"prefer_materialization","maximum_candidates":1,'
            '"pipeline_stages":1,"tile_sizes":[]}}',
            "a9c500984b900263032afc82ad4814606c274ba23627f8db7367e36a4093d739",
        ),
    ],
)
def test_abi22_map_fusion_rotates_full_cache_identity(
    numerics: Numerics,
    materialization: Materialization,
    expected_json: str,
    expected_digest: str,
) -> None:
    tuning = _tuning(materialization)
    options = Options(
        numerics=numerics,
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        tuning=tuning,
    )

    assert PIPELINE_ABI_VERSION == 22
    assert canonical_options_json(options) == expected_json
    assert (
        options_digest(
            numerics=numerics,
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            tuning=tuning,
        )
        == expected_digest
    )
