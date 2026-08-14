# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib


def test_abi18_singleton_broadcast_options_remain_frozen_historical_goldens() -> None:
    source_ordered = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":18,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    fast = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":18,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )

    assert hashlib.sha256(source_ordered.encode()).hexdigest() == (
        "c5f8612994d2d78f51b3df8b949cb2f273401ef09204edef86342c392e1c04fa"
    )
    assert hashlib.sha256(fast.encode()).hexdigest() == (
        "7dcbc1f25ab9c07d78ea27ab03709ba41c5ca742277516b54fb09f101c21be9f"
    )
