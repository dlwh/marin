# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mutation-sensitive tests for the ABI15 raw evidence boundary."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from abi15_h100_composed_contract import CONTRACT_SHA256, FIXTURE_SHA256
from abi15_h100_evidence import (
    IrisTerminalReceipt,
    ProcessReport,
    controller_accepted_envelope,
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


def _materialize_raw_artifact(root: Path) -> dict[str, str]:
    roles = expected_role_paths(load_role_manifest())
    for role, relative in roles.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "prerun_contract":
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
        ProcessReport("baseline", 100, 0, 0, (), ()),
        ProcessReport("populate", 101, 0, 6, (), (("fast", "b" * 64), ("source_ordered", SHA))),
        ProcessReport(
            "reuse",
            102,
            2,
            0,
            (("fast", "b" * 64), ("source_ordered", SHA)),
            (("fast", "b" * 64), ("source_ordered", SHA)),
        ),
    )

    validate_process_reports(reports)

    with pytest.raises(ValueError):
        validate_process_reports((reports[0], reports[1], replace(reports[2], pid=101)))
    with pytest.raises(ValueError):
        validate_process_reports((reports[0], reports[1], replace(reports[2], observer_events=1)))
    with pytest.raises(ValueError):
        validate_process_reports(
            (
                reports[0],
                reports[1],
                replace(reports[2], cache_after=(("source_ordered", SHA),)),
            )
        )


def test_controller_accepted_envelope_requires_verified_archive_and_terminal_receipt() -> None:
    receipt = IrisTerminalReceipt(
        cluster="cw-us-east-02a",
        job_id="/dlwh/abi15-h100-aabbccddeeff",
        bundle_id=SHA,
        authorization_sha256="b" * 64,
        task_image="ghcr.io/marin-community/iris-task-h100-evidence:a@sha256:" + "c" * 64,
        gpu_count=1,
        gpu_variant="H100",
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        attempt_uid="d" * 32,
        task_state="SUCCEEDED",
        ready_log_seq=10,
        ack_log_seq=11,
        terminal_log_seq=12,
    )

    envelope = controller_accepted_envelope(
        archive_sha256=SHA,
        export_manifest_sha256="e" * 64,
        verification_report_sha256="f" * 64,
        verifier_sha256="1" * 64,
        receipt=receipt,
    )

    assert envelope["status"] == "controller_verified_terminal_untimed"
    assert envelope["performance_status"] == "not_started"
    with pytest.raises(ValueError):
        controller_accepted_envelope(
            archive_sha256=SHA,
            export_manifest_sha256="e" * 64,
            verification_report_sha256="f" * 64,
            verifier_sha256="1" * 64,
            receipt=replace(receipt, task_state="RUNNING"),
        )
