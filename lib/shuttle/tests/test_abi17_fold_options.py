# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib


def test_abi17_fold_option_payloads_remain_frozen_historical_goldens() -> None:
    source_ordered = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":17,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    fast = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":17,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )

    assert hashlib.sha256(source_ordered.encode()).hexdigest() == (
        "a64697dcfa7afc90d236a964473841d8e24e135d6c17ab24d188eb06a1a0c8fe"
    )
    assert hashlib.sha256(fast.encode()).hexdigest() == (
        "09e3f11c7d49292d92ab432f3917d285dcdc80feeb79e33a5919a485897c1278"
    )
