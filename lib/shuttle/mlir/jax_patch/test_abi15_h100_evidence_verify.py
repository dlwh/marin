# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import struct
from pathlib import Path

import pytest
from abi15_h100_composed_contract import load_contract
from abi15_h100_evidence import expected_role_paths, load_role_manifest
from abi15_h100_evidence_verify import (
    OUTPUT_ROLES,
    RUNS,
    _observer_contract,
    _validate_record,
    _validate_registration_pointers,
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
    cache = []
    snapshots = []
    for index, policy in enumerate(("fast", "source_ordered")):
        contents = f"{policy}-cache".encode()
        path = root / roles[f"cache_entry_{policy}"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        filename = f"jit_composed-{index:064x}-cache"
        cache.append((policy, filename, len(contents), digest))
        snapshots.append(
            {
                "filename": filename,
                "sha256": digest,
                "size": len(contents),
            }
        )
    _write_json(root / roles["cache_populate_before"], [])
    _write_json(root / roles["cache_populate_after"], snapshots)
    _write_json(root / roles["cache_reuse_before"], snapshots)
    _write_json(root / roles["cache_reuse_after"], snapshots)
    runtime_roles = {
        "cuda_pjrt_plugin_sha256": "cuda_pjrt_plugin",
        "cuda_pjrt_record_sha256": "cuda_pjrt_record",
        "cuda_plugin_extension_sha256": "cuda_plugin_extension",
        "cuda_plugin_record_sha256": "cuda_plugin_record",
        "jax_common_sha256": "jax_common",
        "jax_extension_sha256": "jax_extension",
    }
    runtime = {}
    for field, role in runtime_roles.items():
        path = root / roles[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        runtime[field] = hashlib.sha256(role.encode()).hexdigest()
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
                "runtime_binaries": runtime,
                "runtime_mappings": [
                    {
                        "device": "00:1a",
                        "end": pid * 0x10000 + (index + 1) * 0x1000,
                        "inode": 1000 + index,
                        "path": f"/runtime/{filename}",
                        "role": role,
                        "sha256": runtime[digest_field],
                        "start": pid * 0x10000 + index * 0x1000,
                    }
                    for index, (role, filename, digest_field) in enumerate(
                        (
                            ("cuda_pjrt_plugin", "xla_cuda_plugin.so", "cuda_pjrt_plugin_sha256"),
                            (
                                "cuda_plugin_extension",
                                "cuda_plugin_extension.so",
                                "cuda_plugin_extension_sha256",
                            ),
                            ("jax_common", "libjax_common.so", "jax_common_sha256"),
                            ("jax_extension", "_jax.so", "jax_extension_sha256"),
                        ),
                        1,
                    )
                ],
            },
        )
    for policy in ("source_ordered", "fast"):
        identity, fixture = _observer_contract(policy)
        contract = load_contract()
        policy_contract = contract.source_ordered if policy == "source_ordered" else contract.fast
        common = [
            17,
            None,
            policy,
            identity.policy_digest,
            identity.tuning_digest,
            fixture.region_membership,
            fixture.coverage_manifest(identity),
            fixture.unsupported_fingerprint,
        ]
        records = [
            [*common[:1], "algebra_coverage", *common[2:], "", False, ""],
            [
                *common[:1],
                "generated_bundle",
                *common[2:],
                policy_contract.generated_fingerprint,
                False,
                "",
            ],
            [
                17,
                "final_erasure",
                policy,
                identity.policy_digest,
                identity.tuning_digest,
                "",
                "",
                "",
                policy_contract.final_fingerprint,
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


@pytest.mark.parametrize(
    "mutation",
    ["pointer", "duplicate", "mapping", "extra", "device", "inode", "path", "sha256"],
)
def test_registration_pointer_ranges_reject_semantic_mutation(mutation: str) -> None:
    registered = {
        "api_version": 1,
        "stage_pointers": {"execute": 0x1300, "initialize": 0x1200, "instantiate": 0x1000, "prepare": 0x1100},
        "target": "shuttle.gpu.executable_bundle.v2",
        "traits": 0,
    }
    expected_sha256 = "a" * 64
    mappings = [
        {
            "device": "00:1a",
            "end": 0x2000,
            "inode": 123,
            "path": "/runtime/cuda_plugin_extension.so",
            "sha256": expected_sha256,
            "start": 0x1000,
        }
    ]
    _validate_registration_pointers(registered, mappings, expected_sha256)
    if mutation == "pointer":
        registered["stage_pointers"]["execute"] = 0x3000
    elif mutation == "duplicate":
        registered["stage_pointers"]["execute"] = 0x1200
    elif mutation == "mapping":
        mappings[0]["start"] = 0x2000
    elif mutation == "device":
        mappings[0]["device"] = "device"
    elif mutation == "inode":
        mappings[0]["inode"] = 0
    elif mutation == "path":
        mappings[0]["path"] = "/runtime/same-basename/cuda_plugin_extension.so/../other.so"
    elif mutation == "sha256":
        mappings[0]["sha256"] = "b" * 64
    else:
        mappings[0]["extra"] = None

    with pytest.raises(ValueError):
        _validate_registration_pointers(registered, mappings, expected_sha256)


def test_wheel_record_binds_exact_runtime_member_and_rejects_digest_mutation(tmp_path: Path) -> None:
    extension = tmp_path / "cuda_plugin_extension.so"
    extension.write_bytes(b"extension")
    digest = base64.urlsafe_b64encode(hashlib.sha256(extension.read_bytes()).digest()).rstrip(b"=").decode()
    record = tmp_path / "RECORD"
    record.write_text(f"jax_plugins/cuda/cuda_plugin_extension.so,sha256={digest},{extension.stat().st_size}\n")

    _validate_record(record, "/cuda_plugin_extension.so", extension, "CUDA plugin")

    record.write_text("jax_plugins/cuda/cuda_plugin_extension.so,sha256=wrong,9\n")
    with pytest.raises(ValueError):
        _validate_record(record, "/cuda_plugin_extension.so", extension, "CUDA plugin")


@pytest.mark.parametrize(
    "mutation",
    [
        "algebra",
        "bool_type",
        "cache_duplicate",
        "cache_name",
        "cache_size",
        "cache_swap",
        "coverage",
        "failure",
        "final_structural",
        "generated",
        "invocation_type",
        "mapping",
        "phase",
        "policy",
        "region",
        "reuse",
        "tuning",
        "unsupported",
    ],
)
def test_observer_verifier_rejects_fingerprint_policy_or_reuse_mutation(tmp_path: Path, mutation: str) -> None:
    roles = _observer_fixture(tmp_path)
    populate = tmp_path / roles["observer_populate_fast"]
    if mutation == "reuse":
        (tmp_path / roles["observer_reuse_fast"]).write_text("[]\n")
    elif mutation == "cache_swap":
        fast = tmp_path / roles["cache_entry_fast"]
        source_ordered = tmp_path / roles["cache_entry_source_ordered"]
        fast_bytes, source_ordered_bytes = fast.read_bytes(), source_ordered.read_bytes()
        fast.write_bytes(source_ordered_bytes)
        source_ordered.write_bytes(fast_bytes)
    elif mutation in {"cache_duplicate", "cache_name", "cache_size"}:
        snapshot_roles = ("cache_populate_after", "cache_reuse_before", "cache_reuse_after")
        snapshots = json.loads((tmp_path / roles[snapshot_roles[0]]).read_text())
        if mutation == "cache_duplicate":
            snapshots[1]["filename"] = snapshots[0]["filename"]
        elif mutation == "cache_name":
            snapshots[0]["filename"] = "private-cache-key"
        else:
            snapshots[0]["size"] += 1
        for role in snapshot_roles:
            _write_json(tmp_path / roles[role], snapshots)
        for role in ("populate_report", "reuse_report"):
            report_path = tmp_path / roles[role]
            report = json.loads(report_path.read_text())
            if report["cache_after"]:
                report["cache_after"] = [
                    [policy, snapshot["filename"], snapshot["size"], digest]
                    for (policy, _, _, digest), snapshot in zip(report["cache_after"], snapshots, strict=True)
                ]
            if report["cache_before"]:
                report["cache_before"] = [
                    [policy, snapshot["filename"], snapshot["size"], digest]
                    for (policy, _, _, digest), snapshot in zip(report["cache_before"], snapshots, strict=True)
                ]
            _write_json(report_path, report)
    elif mutation == "mapping":
        report = tmp_path / roles["reuse_report"]
        value = json.loads(report.read_text())
        value["runtime_mappings"][0]["path"] = "/runtime/different.so"
        _write_json(report, value)
    else:
        records = [json.loads(line) for line in populate.read_text().splitlines()]
        if mutation == "generated":
            records[1][8] = "f" * 64
        elif mutation == "policy":
            records[0][2] = "source_ordered"
        elif mutation == "algebra":
            records[0][8] = "f" * 64
        elif mutation == "bool_type":
            records[0][9] = 0
        elif mutation == "coverage":
            records[0][6] += "drift"
        elif mutation == "failure":
            records[1][10] = "pass"
        elif mutation == "final_structural":
            records[2][5] = "retained"
        elif mutation == "invocation_type":
            records[0][0] = "17"
        elif mutation == "phase":
            records[0][1] = "generated_bundle"
        elif mutation == "region":
            records[1][5] += "drift"
        elif mutation == "tuning":
            records[0][4] = "f" * 64
        else:
            records[1][7] = "f" * 64
        populate.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError):
        validate_observer_and_process_files(tmp_path, roles)
