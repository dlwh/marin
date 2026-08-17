# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mutation-sensitive tests for the ABI15 raw evidence boundary."""

import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import abi15_h100_evidence
import pytest
from abi15_h100_composed_contract import CONTRACT_SHA256, FIXTURE_SHA256
from abi15_h100_evidence import (
    ProcessReport,
    expected_role_paths,
    load_raw_artifact,
    load_role_manifest,
    require_sealed_hardware_authority,
    seal_raw_artifact,
    validate_process_reports,
    validate_unsealed_prerun_contract,
)

SHA = "a" * 64
COMMIT = "b" * 40
TREE = "c" * 40
RUNTIME = tuple(
    (name, f"{index:x}" * 64)
    for index, name in enumerate(
        (
            "cuda_pjrt_plugin_sha256",
            "cuda_pjrt_record_sha256",
            "cuda_plugin_extension_sha256",
            "cuda_plugin_record_sha256",
            "jax_common_sha256",
            "jax_extension_sha256",
        ),
        1,
    )
)
RUNTIME_BY_ROLE = {
    "cuda_pjrt_plugin": dict(RUNTIME)["cuda_pjrt_plugin_sha256"],
    "cuda_plugin_extension": dict(RUNTIME)["cuda_plugin_extension_sha256"],
    "jax_common": dict(RUNTIME)["jax_common_sha256"],
    "jax_extension": dict(RUNTIME)["jax_extension_sha256"],
}
MAPPINGS = tuple(
    (
        role,
        f"/runtime/{filename}",
        "00:1a",
        1000 + index,
        0x1000 * index,
        0x1000 * (index + 1),
        RUNTIME_BY_ROLE[role],
    )
    for index, (role, filename) in enumerate(
        (
            ("cuda_pjrt_plugin", "xla_cuda_plugin.so"),
            ("cuda_plugin_extension", "cuda_plugin_extension.so"),
            ("jax_common", "libjax_common.so"),
            ("jax_extension", "_jax.so"),
        ),
        1,
    )
)
CACHE = (
    ("fast", "jit_composed-" + "b" * 64 + "-cache", 17, "b" * 64),
    ("source_ordered", "jit_composed-" + "a" * 64 + "-cache", 19, SHA),
)


def _materialize_raw_artifact(root: Path) -> dict[str, str]:
    roles = expected_role_paths(load_role_manifest())
    for role, relative in roles.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "architecture_build_contract":
            source = Path(__file__).with_name("abi15-h100-architecture-build-contract-v1.json")
            path.write_bytes(source.read_bytes())
        elif role == "prerun_contract":
            source = (
                Path(__file__).parents[4]
                / ".agents/projects/tile_lifetime_compiler/shuttle_abi15_h100_composed_prerun_v1.json"
            )
            path.write_bytes(source.read_bytes())
        elif role == "fixture":
            source = Path(__file__).parents[1] / "test/Inputs/jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir"
            path.write_bytes(source.read_bytes())
        else:
            path.write_bytes(f"{role}\n".encode())
    return roles


def test_role_manifest_expands_exact_policy_call_and_run_coordinates() -> None:
    roles = expected_role_paths(load_role_manifest())

    assert roles["transport_source_ordered_forward"] == "compiler/source_ordered/forward.bin"
    assert roles["ptx_fast_backward_09"] == "assembly/fast/backward/entry-09.ptx"
    assert roles["cubin_source_ordered_forward_01"] == "assembly/source_ordered/forward/entry-01.cubin"
    assert (
        roles["output_fast_sparse_policy_discriminator_thread_1_rhs_cotangent"]
        == "outputs/fast/sparse_policy_discriminator/thread_1/rhs_cotangent.bf16"
    )
    assert "accepted_bundle" not in roles
    assert len(set(roles.values())) == len(roles)


def test_raw_artifact_round_trip_covers_exact_roles_and_immutable_contract(
    tmp_path: Path,
) -> None:
    roles = _materialize_raw_artifact(tmp_path)
    manifest = seal_raw_artifact(tmp_path, roles, harness_commit=COMMIT, harness_tree=TREE)

    artifact = load_raw_artifact(tmp_path, expected_harness_commit=COMMIT, expected_harness_tree=TREE)

    assert artifact.manifest_sha256 == manifest
    assert set(artifact.files) == set(roles)
    assert artifact.files["prerun_contract"].sha256 == CONTRACT_SHA256
    assert artifact.files["fixture"].sha256 == FIXTURE_SHA256


@pytest.mark.parametrize("mutation", ["substitute", "extra", "role"])
def test_raw_artifact_rejects_byte_file_or_self_consistent_role_mutation(tmp_path: Path, mutation: str) -> None:
    roles = _materialize_raw_artifact(tmp_path)
    seal_raw_artifact(tmp_path, roles, harness_commit=COMMIT, harness_tree=TREE)
    if mutation == "substitute":
        (tmp_path / roles["baseline_report"]).write_text("changed\n")
    elif mutation == "extra":
        (tmp_path / "extra").write_text("extra\n")
    else:
        manifest_path = tmp_path / "raw-artifact-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = next(item for item in manifest["files"] if item["role"] == "baseline_report")
        record["role"] = "invented_report"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError):
        load_raw_artifact(tmp_path, expected_harness_commit=COMMIT, expected_harness_tree=TREE)


@pytest.mark.parametrize("entry_kind", ["fifo", "socket"])
def test_raw_artifact_lstats_and_rejects_every_non_regular_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_kind: str
) -> None:
    roles = _materialize_raw_artifact(tmp_path)
    seal_raw_artifact(tmp_path, roles, harness_commit=COMMIT, harness_tree=TREE)
    extra = tmp_path / "unmanifested-entry"
    if entry_kind == "fifo":
        os.mkfifo(extra)
    else:
        extra.write_bytes(b"socket-placeholder")
        real_lstat = os.lstat
        monkeypatch.setattr(
            abi15_h100_evidence.os,
            "lstat",
            lambda path: (SimpleNamespace(st_mode=stat.S_IFSOCK | 0o600) if Path(path) == extra else real_lstat(path)),
        )
    with pytest.raises(ValueError, match="non-regular"):
        load_raw_artifact(tmp_path, expected_harness_commit=COMMIT, expected_harness_tree=TREE)


def test_prerun_contract_remains_explicitly_unsealed() -> None:
    document = validate_unsealed_prerun_contract()

    assert document["hardware_authority"] == {
        "image_digest": None,
        "launch_authorization_sha256": None,
        "source_capsule_sha256": None,
        "status": "not_authorized",
    }
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        require_sealed_hardware_authority()


def test_process_reports_bind_distinct_processes_two_entry_cache_and_observer_state() -> None:
    reports = (
        ProcessReport("baseline", 100, 0, 0, (), (), RUNTIME, MAPPINGS),
        ProcessReport("populate", 101, 0, 6, (), CACHE, RUNTIME, MAPPINGS),
        ProcessReport(
            "reuse",
            102,
            2,
            0,
            CACHE,
            CACHE,
            RUNTIME,
            MAPPINGS,
        ),
    )

    validate_process_reports(reports)

    with pytest.raises(ValueError):
        validate_process_reports((reports[0], reports[1], replace(reports[2], pid=101)))
    with pytest.raises(ValueError):
        validate_process_reports((reports[0], reports[1], replace(reports[2], observer_events=1)))
    with pytest.raises(ValueError):
        validate_process_reports((reports[0], reports[1], replace(reports[2], runtime_mappings=MAPPINGS[:-1])))


@pytest.mark.parametrize("field", ["device", "inode", "path", "sha256", "role"])
def test_process_reports_reject_self_consistent_mapping_identity_drift(field: str) -> None:
    reports = (
        ProcessReport("baseline", 100, 0, 0, (), (), RUNTIME, MAPPINGS),
        ProcessReport("populate", 101, 0, 6, (), CACHE, RUNTIME, MAPPINGS),
        ProcessReport("reuse", 102, 2, 0, CACHE, CACHE, RUNTIME, MAPPINGS),
    )
    mappings = [list(record) for record in MAPPINGS]
    index = {"role": 0, "path": 1, "device": 2, "inode": 3, "sha256": 6}[field]
    if field == "device":
        mappings[0][index] = "00:2b"
    elif field == "inode":
        mappings[0][index] += 1
    elif field == "path":
        mappings[0][index] = "/runtime/other.so"
    elif field == "sha256":
        mappings[0][index] = "f" * 64
    else:
        mappings[0][index], mappings[1][index] = mappings[1][index], mappings[0][index]
    mutated = tuple(tuple(record) for record in mappings)

    if field in {"device", "inode"}:
        mutated_reports = (*reports[:2], replace(reports[2], runtime_mappings=mutated))
    else:
        mutated_reports = tuple(replace(report, runtime_mappings=mutated) for report in reports)
    with pytest.raises(ValueError):
        validate_process_reports(mutated_reports)
    with pytest.raises(ValueError):
        validate_process_reports(
            (
                reports[0],
                reports[1],
                replace(reports[2], cache_after=CACHE[:-1]),
            )
        )
