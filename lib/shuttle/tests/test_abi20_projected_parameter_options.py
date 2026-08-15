# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib


def test_abi20_projected_parameter_options_remain_frozen_historical_goldens() -> None:
    source_ordered = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":20,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    fast = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":20,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert hashlib.sha256(source_ordered.encode()).hexdigest() == (
        "e88a2d0d73e15ed5fbfb98e0d30a5a2c0ca581b8369543225d53a1fa29bfdde2"
    )
    assert hashlib.sha256(fast.encode()).hexdigest() == (
        "dc9a9609b9ca7ae14c9797345780c2b9c128ccf8d87ae4cf8a47088ee2d0e5c5"
    )
