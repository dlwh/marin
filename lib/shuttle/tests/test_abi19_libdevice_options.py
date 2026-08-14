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


def test_abi19_libdevice_exp_rotates_both_gpu_policy_identities() -> None:
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
        '"pipeline_abi_version":19,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert fast["xla_shuttle_options"] == (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":19,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=tuning,
        )
        == "0cd8037af7d6c4ca898af022b4ed862c784c5deb3adbe471677190bba6a4056a"
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.FAST,
            tuning=tuning,
        )
        == "fb364386414b3b32717ddf77e5cfd6da10c11a3f98069361e67b2ee9751a64e2"
    )
