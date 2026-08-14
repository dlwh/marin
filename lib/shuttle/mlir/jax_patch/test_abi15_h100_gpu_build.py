# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unsealed ABI15 pre-hardware build boundary."""

from copy import deepcopy

import pytest
from abi15_h100_gpu_build import parse_ptxas_resources, validate_build_candidate

SHA = "a" * 64


def _candidate() -> dict[str, object]:
    transports = []
    for policy_index, policy in enumerate(("source_ordered", "fast")):
        for call_index, (call, launches) in enumerate((("forward", 2), ("backward", 10))):
            seed = 1 + policy_index * 2 + call_index
            transports.append(
                {
                    "bundle_root": f"{seed:x}" * 64,
                    "call": call,
                    "code_sha256": f"{(seed + 4) % 15:x}" * 64,
                    "device_root": f"{(seed + 8) % 15:x}" * 64,
                    "invocation_root": f"{(seed + 12) % 15:x}" * 64,
                    "launches": launches,
                    "policy": policy,
                    "ptx_slices": [
                        {
                            "cubin_sha256": f"{(ordinal % 8) + 1:x}" * 64,
                            "ordinal": ordinal,
                            "ptx_sha256": f"{(ordinal % 7) + 1:x}" * 64,
                            "registers": 12,
                            "spill_bytes": 0,
                        }
                        for ordinal in range(launches)
                    ],
                    "transport_sha256": f"{(seed + 16) % 15:x}" * 64,
                }
            )
    return {
        "commands": [
            ["jax-build", "--cuda_compute_capabilities=9.0"],
            ["ptxas", "-arch=sm_90"],
        ],
        "registration_transcript_sha256": "f" * 64,
        "schema_version": 1,
        "source": {
            "harness_commit": "1" * 40,
            "harness_tree": "2" * 40,
            "jax_commit": "3" * 40,
            "jax_patch_sha256": ["4" * 64, "5" * 64, "6" * 64],
            "jax_tree": "7" * 40,
            "xla_commit": "8" * 40,
        },
        "status": "pre_hardware_build_candidate_not_authority",
        "toolchain": {
            "bazel_sha256": "9" * 64,
            "bazel_version": "candidate",
            "ptxas_sha256": "a" * 64,
            "ptxas_version": "candidate",
            "target": "sm_90",
        },
        "transports": transports,
        "wheels": {
            name: {"sha256": digest * 64, "size": 1}
            for name, digest in (
                ("cuda_plugin", "b"),
                ("cuda_pjrt", "c"),
                ("jaxlib", "d"),
            )
        },
    }


def test_parse_ptxas_resources_requires_one_register_and_zero_or_reported_spills() -> None:
    assert parse_ptxas_resources("Used 32 registers\n0 bytes spill stores, 0 bytes spill loads") == (32, 0)

    with pytest.raises(ValueError):
        parse_ptxas_resources("Used 32 registers\nUsed 33 registers\n0 bytes spill stores, 0 bytes spill loads")


def test_build_candidate_closes_four_transports_and_twenty_four_assemblies() -> None:
    candidate = _candidate()

    validated = validate_build_candidate(candidate)

    assert sum(len(item["ptx_slices"]) for item in validated["transports"]) == 24
    assert validated["status"] == "pre_hardware_build_candidate_not_authority"


@pytest.mark.parametrize("mutation", ["status", "policy", "slice", "spill", "toolchain"])
def test_build_candidate_rejects_self_consistent_semantic_drift(mutation: str) -> None:
    candidate = deepcopy(_candidate())
    if mutation == "status":
        candidate["status"] = "sealed"
    elif mutation == "policy":
        candidate["transports"][2]["policy"] = "source_ordered"
    elif mutation == "slice":
        candidate["transports"][1]["ptx_slices"].pop()
        candidate["transports"][1]["launches"] = 9
    elif mutation == "spill":
        candidate["transports"][3]["ptx_slices"][8]["spill_bytes"] = 4
    else:
        candidate["toolchain"]["target"] = "sm_89"

    with pytest.raises(ValueError):
        validate_build_candidate(candidate)
