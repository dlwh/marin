# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib


def test_abi19_libdevice_exp_historical_cache_identities_are_frozen() -> None:
    source_ordered = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":19,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    fast = (
        '{"execution_mode":"gpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":19,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )
    assert hashlib.sha256(source_ordered.encode()).hexdigest() == (
        "0cd8037af7d6c4ca898af022b4ed862c784c5deb3adbe471677190bba6a4056a"
    )
    assert hashlib.sha256(fast.encode()).hexdigest() == (
        "fb364386414b3b32717ddf77e5cfd6da10c11a3f98069361e67b2ee9751a64e2"
    )
