# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options, options_digest

ABI21_SOURCE_ORDERED = (
    '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
    '"pipeline_abi_version":21,"schema_version":1,'
    '"tuning":{"cluster_shape":[],"materialization":"automatic",'
    '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
)
ABI21_FAST = (
    '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
    '"pipeline_abi_version":21,"schema_version":1,'
    '"tuning":{"cluster_shape":[],"materialization":"automatic",'
    '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
)
ABI21_SOURCE_ORDERED_DIGEST = "731a06b7bd067960293e47448e483e07ca7373a6ff09bfa94fb3dc804d7de280"
ABI21_FAST_DIGEST = "c658672602d79f23894da60c8300ecdaf40905acefe9f33ad58216571f48d2da"


def _default_tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def test_abi21_full_prefix_fold_historical_identities_remain_pinned() -> None:
    tuning = _default_tuning()
    source_ordered = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )
    fast = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.FAST,
        tuning=tuning,
    )
    assert hashlib.sha256(ABI21_SOURCE_ORDERED.encode()).hexdigest() == ABI21_SOURCE_ORDERED_DIGEST
    assert hashlib.sha256(ABI21_FAST.encode()).hexdigest() == ABI21_FAST_DIGEST
    assert source_ordered["xla_shuttle_options"] != ABI21_SOURCE_ORDERED
    assert fast["xla_shuttle_options"] != ABI21_FAST
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=tuning,
        )
        != ABI21_SOURCE_ORDERED_DIGEST
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.FAST,
            tuning=tuning,
        )
        != ABI21_FAST_DIGEST
    )
