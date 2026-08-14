# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

import pytest
from abi15_h100_composed_contract import load_contract
from abi15_h100_evidence import expected_role_paths, load_role_manifest
from abi15_h100_evidence_verify import (
    OUTPUT_ROLES,
    RUNS,
    validate_numerical_files,
    validate_observer_and_process_files,
)
from abi15_h100_numerics import (
    ReductionPolicy,
    dense_inputs,
    reference,
    sparse_policy_inputs,
)


def _write_bits(path: Path, values: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(values)}H", *values))


def _numerical_fixture(root: Path) -> dict[str, str]:
    roles = expected_role_paths(load_role_manifest())
    for dataset, inputs in (
        ("dense", dense_inputs()),
        ("sparse_policy_discriminator", sparse_policy_inputs()),
    ):
        for role, values in (
            ("lhs", inputs.lhs),
            ("rhs", inputs.rhs),
            ("output_cotangent", inputs.output_cotangent),
        ):
            _write_bits(root / roles[f"input_{dataset}_{role}"], values)
        for policy in ReductionPolicy:
            outputs = reference(inputs, policy)
            for output_role, values in zip(
                OUTPUT_ROLES,
                (outputs.primal, outputs.lhs_cotangent, outputs.rhs_cotangent),
                strict=True,
            ):
                _write_bits(
                    root / roles[f"reference_{policy.value}_{dataset}_{output_role}"],
                    values,
                )
                for run in RUNS:
                    _write_bits(
                        root / roles[f"output_{policy.value}_{dataset}_{run}_{output_role}"],
                        values,
                    )
        baseline = reference(inputs, ReductionPolicy.SOURCE_ORDERED)
        for output_role, values in zip(
            OUTPUT_ROLES,
            (baseline.primal, baseline.lhs_cotangent, baseline.rhs_cotangent),
            strict=True,
        ):
            _write_bits(root / roles[f"baseline_{dataset}_{output_role}"], values)
    return roles


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _observer_fixture(root: Path) -> dict[str, str]:
    roles = expected_role_paths(load_role_manifest())
    cache = [("fast", "a" * 64), ("source_ordered", "b" * 64)]
    for process, pid, hits, event_count, before, after in (
        ("baseline", 100, 0, 0, [], []),
        ("populate", 101, 0, 6, [], cache),
        ("reuse", 102, 2, 0, cache, cache),
    ):
        _write_json(
            root / roles[f"{process}_report"],
            {
                "cache_after": after,
                "cache_before": before,
                "cache_hits": hits,
                "observer_events": event_count,
                "pid": pid,
                "process": process,
            },
        )
    contract = load_contract()
    for policy in ("source_ordered", "fast"):
        identity = contract.source_ordered if policy == "source_ordered" else contract.fast
        common = [
            "invocation",
            None,
            policy,
            identity.options_sha256,
            "tuning",
            "region",
            "coverage",
            "unsupported",
        ]
        records = [
            [*common[:1], "algebra_coverage", *common[2:], "", False, ""],
            [
                *common[:1],
                "generated_bundle",
                *common[2:],
                identity.generated_fingerprint,
                False,
                "",
            ],
            [
                *common[:1],
                "final_erasure",
                *common[2:],
                identity.final_fingerprint,
                True,
                "",
            ],
        ]
        populate = root / roles[f"observer_populate_{policy}"]
        populate.parent.mkdir(parents=True, exist_ok=True)
        populate.write_text("".join(json.dumps(record) + "\n" for record in records))
        reuse = root / roles[f"observer_reuse_{policy}"]
        reuse.parent.mkdir(parents=True, exist_ok=True)
        reuse.write_bytes(b"")
    return roles


def test_numerical_verifier_recomputes_both_policies_and_all_repeats(
    tmp_path: Path,
) -> None:
    roles = _numerical_fixture(tmp_path)

    report = validate_numerical_files(tmp_path, roles)

    assert report["bitwise_repeatability"] is True
    assert len(report["independent_reference_sha256"]) == 12


@pytest.mark.parametrize(
    "role",
    [
        "input_dense_lhs",
        "reference_fast_sparse_policy_discriminator_rhs_cotangent",
        "output_source_ordered_dense_thread_1_lhs_cotangent",
        "output_fast_sparse_policy_discriminator_reuse_primal",
    ],
)
def test_numerical_verifier_rejects_input_reference_policy_or_repeat_mutation(tmp_path: Path, role: str) -> None:
    roles = _numerical_fixture(tmp_path)
    path = tmp_path / roles[role]
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 1
    path.write_bytes(mutated)

    with pytest.raises(ValueError):
        validate_numerical_files(tmp_path, roles)


def test_observer_verifier_binds_processes_policy_fingerprints_and_empty_reuse(
    tmp_path: Path,
) -> None:
    roles = _observer_fixture(tmp_path)

    report = validate_observer_and_process_files(tmp_path, roles)

    assert report == {"observer_events": 6, "populate_hits": 0, "reuse_hits": 2}


@pytest.mark.parametrize("mutation", ["generated", "policy", "reuse"])
def test_observer_verifier_rejects_fingerprint_policy_or_reuse_mutation(tmp_path: Path, mutation: str) -> None:
    roles = _observer_fixture(tmp_path)
    populate = tmp_path / roles["observer_populate_fast"]
    if mutation == "reuse":
        (tmp_path / roles["observer_reuse_fast"]).write_text("[]\n")
    else:
        records = [json.loads(line) for line in populate.read_text().splitlines()]
        if mutation == "generated":
            records[1][8] = "f" * 64
        else:
            records[0][2] = "source_ordered"
        populate.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError):
        validate_observer_and_process_files(tmp_path, roles)
