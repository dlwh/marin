# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options, options_digest


def _default_tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def test_abi20_projected_parameter_rotates_both_gpu_policy_identities() -> None:
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
    assert source_ordered["xla_shuttle_options"] == (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":20,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert fast["xla_shuttle_options"] == (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":20,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=tuning,
        )
        == "e88a2d0d73e15ed5fbfb98e0d30a5a2c0ca581b8369543225d53a1fa29bfdde2"
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.FAST,
            tuning=tuning,
        )
        == "dc9a9609b9ca7ae14c9797345780c2b9c128ccf8d87ae4cf8a47088ee2d0e5c5"
    )
