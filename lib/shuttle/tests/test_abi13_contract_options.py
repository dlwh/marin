# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options, options_digest


def test_abi13_gpu_contract_options_have_a_distinct_default_cache_identity() -> None:
    tuning = Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )
    options = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )

    assert options["xla_shuttle_options"] == (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":13,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=tuning,
        )
        == "432ecf5407e7c81d32a8b3515c050681e02f2f7f90d24988eee35ea449ccb5bd"
    )
