# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Red contract for controller-signed ABI15 evidence receipts."""

import base64
import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.codec import reconstruct_launch_job_request
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from rigging.token_authority import signing_key_from_private_pem
from sqlalchemy import update as sa_update

CONTRACT_FILENAME = "abi15-h100-controller-receipt-red-v1.json"
DOMAIN = "marin.shuttle.abi15.evidence-receipt.v1"
PRIVATE_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
OTHER_PRIVATE_SEED = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
TEST_KEY_ID = "kPrK_qmxVWaYVA9wwBF6Iuo3vVzz7TxHCTwXBygrS4k"
OTHER_KEY_ID = "abi15-receipt-rfc8032-test-key-2"
VERIFIER_BYTES = b"abi15 receipt red verifier fixture\n"
# RFC 8032 test vector 1. This key is public test data, never an authority key.
RFC8032_TEST_PRIVATE_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIJ1hsZ3v/VpguoRK9JLsLMREScVpezJpGXA7rAMcrn9g
-----END PRIVATE KEY-----
"""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _contract() -> dict[str, Any]:
    raw = Path(__file__).with_name(CONTRACT_FILENAME).read_bytes()
    document = json.loads(raw)
    assert raw == _canonical(document)
    return document


def _receipt_module():
    return importlib.import_module("abi15_h100_receipt_verify")


def _controller_module():
    return importlib.import_module("iris.cluster.controller.shuttle_receipts")


def _controller_db(path: Path):
    database = importlib.import_module("iris.cluster.controller.db").ControllerDB(path)
    RunTemplatesProjection(database)
    return database


def _launch_request(contract: dict[str, Any]) -> controller_pb2.Controller.LaunchJobRequest:
    launch = contract["launch"]
    raw = base64.b64decode(launch["launch_request_base64"], validate=True)
    request = controller_pb2.Controller.LaunchJobRequest.FromString(raw)
    assert request.SerializeToString(deterministic=True) == raw
    assert hashlib.sha256(raw).hexdigest() == launch["launch_request_sha256"]
    return request


def _ack_payload(contract: dict[str, Any], *, attempt_uid: str = "0123456789abcdef") -> dict[str, object]:
    launch = contract["launch"]
    authority = contract["authority"]
    return {
        "archive_sha256": "2" * 64,
        "attempt_id": launch["attempt_id"],
        "attempt_uid": attempt_uid,
        "authorization_sha256": launch["authorization_sha256"],
        "bundle_sha256": launch["bundle_sha256"],
        "challenge": launch["challenge"],
        "cluster": authority["cluster"],
        "controller_id": authority["controller_id"],
        "domain": contract["domain"],
        "export_manifest_sha256": "3" * 64,
        "image_digest": launch["image_digest"],
        "issued_at_ms": 1_800_000_000_000,
        "job_id": launch["job_id"],
        "job_state": "RUNNING",
        "launch_request_sha256": launch["launch_request_sha256"],
        "ready_log_seq": 100,
        "ready_sha256": "4" * 64,
        "source_capsule_sha256": launch["source_capsule_sha256"],
        "source_manifest_sha256": launch["source_manifest_sha256"],
        "stage": "ACK",
        "task_id": launch["task_id"],
        "task_state": "RUNNING",
        "verdict": "accepted",
        "verification_report_sha256": "5" * 64,
        "verifier_sha256": launch["verifier_sha256"],
    }


def _signed(
    payload: dict[str, object],
    *,
    seed: bytes = PRIVATE_SEED,
    key_id: str = TEST_KEY_ID,
    signing_domain: str = DOMAIN,
) -> bytes:
    key = Ed25519PrivateKey.from_private_bytes(seed)
    signature = key.sign(signing_domain.encode() + b"\0" + _canonical(payload))
    return _canonical(
        {
            "payload": payload,
            "schema_version": 1,
            "signature": {
                "algorithm": "Ed25519",
                "key_id": key_id,
                "value_base64": base64.b64encode(signature).decode("ascii"),
            },
        }
    )


def _terminal_payload(contract: dict[str, Any], ack: bytes) -> dict[str, object]:
    launch = contract["launch"]
    authority = contract["authority"]
    ack_payload = json.loads(ack)["payload"]
    return {
        "ack_log_seq": 101,
        "ack_receipt_sha256": hashlib.sha256(ack).hexdigest(),
        "attempt_id": ack_payload["attempt_id"],
        "attempt_uid": ack_payload["attempt_uid"],
        "authorization_sha256": launch["authorization_sha256"],
        "challenge": launch["challenge"],
        "cluster": authority["cluster"],
        "controller_id": authority["controller_id"],
        "domain": contract["domain"],
        "final_job_snapshot_sha256": "6" * 64,
        "final_task_snapshot_sha256": "7" * 64,
        "issued_at_ms": 1_800_000_001_000,
        "job_id": launch["job_id"],
        "job_state": "SUCCEEDED",
        "launch_request_sha256": launch["launch_request_sha256"],
        "ready_log_seq": 100,
        "runner_exit_code": 0,
        "stage": "TERMINAL",
        "task_id": launch["task_id"],
        "task_state": "SUCCEEDED",
        "terminal_log_seq": 102,
    }


def _other_public_key_pem() -> str:
    return (
        Ed25519PrivateKey.from_private_bytes(OTHER_PRIVATE_SEED)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def _seed_running_attempt(
    database: Any,
    contract: dict[str, Any],
    *,
    request: controller_pb2.Controller.LaunchJobRequest | None = None,
) -> str:
    if request is None:
        request = _launch_request(contract)
    job_id = JobName.from_string(contract["launch"]["job_id"])
    task_id = job_id.task(0)
    with database.transaction() as tx:
        ops.job.submit(
            tx,
            job_id=job_id,
            request=request,
            ts=Timestamp.from_ms(1_800_000_000_000),
            submitting_user="dlwh",
        )
        writes.promote_for_dispatch(tx, task_id, 0, 1_800_000_000_100)
        writes.mark_jobs_running(tx, [job_id], 1_800_000_000_100)
        tx.execute(
            sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(state=job_pb2.TASK_STATE_RUNNING)
        )
        tx.execute(
            sa_update(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id, task_attempts_table.c.attempt_id == 0)
            .values(state=job_pb2.TASK_STATE_RUNNING, started_at_ms=1_800_000_000_100)
        )
    with database.read_snapshot() as tx:
        attempt = reads.bulk_get_attempts(tx, [(task_id, 0)])[(task_id, 0)]
    return str(attempt.attempt_uid)


def _finish_attempt(database: Any, contract: dict[str, Any]) -> None:
    job_id = JobName.from_string(contract["launch"]["job_id"])
    task_id = job_id.task(0)
    finished_at_ms = 1_800_000_001_000
    with database.transaction() as tx:
        tx.execute(
            sa_update(jobs_table)
            .where(jobs_table.c.job_id == job_id)
            .values(
                state=job_pb2.JOB_STATE_SUCCEEDED,
                exit_code=0,
                finished_at_ms=finished_at_ms,
            )
        )
        tx.execute(
            sa_update(tasks_table)
            .where(tasks_table.c.task_id == task_id)
            .values(
                state=job_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                finished_at_ms=finished_at_ms,
            )
        )
        tx.execute(
            sa_update(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id, task_attempts_table.c.attempt_id == 0)
            .values(
                state=job_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                finished_at_ms=finished_at_ms,
            )
        )


def _evidence_paths(tmp_path: Path, contract: dict[str, Any], attempt_uid: str) -> dict[str, Path]:
    archive = tmp_path / "raw-evidence.tar"
    export_manifest = tmp_path / "export-manifest.json"
    ready_log_record = tmp_path / "ready-log-record.json"
    terminal_log_record = tmp_path / "terminal-log-record.json"
    verifier = tmp_path / "verifier.bin"
    archive.write_bytes(b"abi15 raw evidence archive fixture\n")
    export_manifest.write_bytes(_canonical({"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    ready_log_record.write_bytes(
        _canonical(
            {
                "attempt_id": 0,
                "attempt_uid": attempt_uid,
                "job_id": contract["launch"]["job_id"],
                "log_seq": 100,
                "record": "READY",
                "task_id": contract["launch"]["task_id"],
            }
        )
    )
    terminal_log_record.write_bytes(
        _canonical(
            {
                "attempt_id": 0,
                "attempt_uid": attempt_uid,
                "job_id": contract["launch"]["job_id"],
                "log_seq": 102,
                "record": "TERMINAL",
                "runner_exit_code": 0,
                "task_id": contract["launch"]["task_id"],
            }
        )
    )
    verifier.write_bytes(VERIFIER_BYTES)
    return {
        "archive": archive,
        "export_manifest": export_manifest,
        "ready_log_record": ready_log_record,
        "terminal_log_record": terminal_log_record,
        "verifier": verifier,
    }


def _install_verifier(
    monkeypatch: pytest.MonkeyPatch,
    controller: Any,
    calls: list[tuple[Path, Path]],
    reports: list[bytes] | None = None,
    report_mutation: dict[str, object] | None = None,
) -> None:
    def run_verifier(*, archive_path: Path, export_manifest_path: Path) -> bytes:
        calls.append((archive_path, export_manifest_path))
        report: dict[str, object] = {
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "export_manifest_sha256": hashlib.sha256(export_manifest_path.read_bytes()).hexdigest(),
            "verdict": "accepted",
        }
        if report_mutation is not None:
            report.update(report_mutation)
        raw = _canonical(report)
        if reports is not None:
            reports.append(raw)
        return raw

    monkeypatch.setattr(controller, "run_abi15_evidence_verifier", run_verifier)


def _issue_ack(store: Any, paths: dict[str, Path]) -> bytes:
    return store.issue_ack(
        archive_path=paths["archive"],
        export_manifest_path=paths["export_manifest"],
        ready_log_record_path=paths["ready_log_record"],
        verifier_path=paths["verifier"],
    )


def test_receipt_contract_is_canonical_and_binds_exact_zero_retry_launch() -> None:
    contract = _contract()
    request = _launch_request(contract)

    assert contract["status"] == "test_only_red_controller_receipt_contract_not_launch_authority"
    assert contract["design_stops"] == {
        "controller_signer_integration": None,
        "hardware_launch": None,
        "image_transfer": None,
        "receipt_authority_approval": None,
        "secret_projection": None,
    }
    assert contract["authority"]["previous_key_fallback_allowed"] is False
    assert "attempt_uid" not in contract["launch"]
    assert request.max_retries_failure == contract["launch"]["max_retries_failure"] == 0
    assert request.max_retries_preemption == contract["launch"]["max_retries_preemption"] == 0
    assert request.max_task_failures == contract["launch"]["max_task_failures"] == 0
    assert request.existing_job_policy == job_pb2.EXISTING_JOB_POLICY_ERROR
    assert request.fail_if_exists is True


def test_controller_fixture_uses_reconstructed_launch_and_controller_minted_attempt(
    tmp_path: Path,
) -> None:
    contract = _contract()
    database = _controller_db(tmp_path)
    attempt_uid = _seed_running_attempt(database, contract)
    job_id = JobName.from_string(contract["launch"]["job_id"])
    task_id = job_id.task(0)

    try:
        with database.read_snapshot() as tx:
            job = reads.get_job_detail(tx, job_id)
            reconstructed = reconstruct_launch_job_request(job, workdir_files=reads.get_workdir_files(tx, job_id))
            attempt = reads.bulk_get_attempts(tx, [(task_id, 0)])[(task_id, 0)]
        assert reconstructed == _launch_request(contract)
        assert attempt.attempt_uid == attempt_uid
        assert attempt.state == job_pb2.TASK_STATE_RUNNING
        assert attempt_uid != "0123456789abcdef"
    finally:
        database.close()


def test_task_verifier_accepts_exact_ack_and_terminal_chain() -> None:
    receipts = _receipt_module()
    contract = _contract()
    ack = _signed(_ack_payload(contract))
    terminal = _signed(_terminal_payload(contract, ack))

    verified_ack = receipts.verify_ack_receipt(ack, contract)
    verified_terminal = receipts.verify_terminal_receipt(terminal, ack, contract)

    assert verified_ack["stage"] == "ACK"
    assert verified_terminal["stage"] == "TERMINAL"
    assert verified_terminal["ack_receipt_sha256"] == hashlib.sha256(ack).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "attacker.receipt.v1"),
        ("cluster", "attacker-cluster"),
        ("controller_id", "attacker-controller"),
        ("authorization_sha256", "8" * 64),
        ("challenge", "8" * 64),
        ("job_id", "/attacker/job"),
        ("task_id", "/attacker/job/0"),
        ("attempt_id", 1),
        ("attempt_uid", "fedcba9876543210"),
        ("launch_request_sha256", "8" * 64),
        ("image_digest", "sha256:" + "8" * 64),
        ("bundle_sha256", "8" * 64),
        ("source_capsule_sha256", "8" * 64),
        ("source_manifest_sha256", "8" * 64),
        ("verifier_sha256", "8" * 64),
        ("job_state", "SUCCEEDED"),
        ("task_state", "SUCCEEDED"),
        ("verdict", "rejected"),
    ],
)
def test_task_verifier_rejects_resigned_ack_identity_or_state_substitution(field: str, value: object) -> None:
    receipts = _receipt_module()
    contract = _contract()
    payload = _ack_payload(contract)
    payload[field] = value

    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(_signed(payload), contract)


def test_task_verifier_rejects_wrong_signature_domain_prefix() -> None:
    receipts = _receipt_module()
    contract = _contract()
    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(_signed(_ack_payload(contract), signing_domain="attacker.receipt.v1"), contract)


def test_task_verifier_uses_only_the_current_pinned_key_during_real_rotation() -> None:
    receipts = _receipt_module()
    contract = _contract()
    rotated = deepcopy(contract)
    rotated["authority"]["key_id"] = OTHER_KEY_ID
    rotated["authority"]["public_key_spki_pem"] = _other_public_key_pem()
    payload = _ack_payload(rotated)

    current_receipt = _signed(payload, seed=OTHER_PRIVATE_SEED, key_id=OTHER_KEY_ID)
    old_key_receipt = _signed(payload)
    assert receipts.verify_ack_receipt(current_receipt, rotated)["stage"] == "ACK"
    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(old_key_receipt, rotated)


@pytest.mark.parametrize(
    ("stage", "mutation"),
    [
        ("ACK", "top_field"),
        ("ACK", "payload_field"),
        ("ACK", "missing_payload_field"),
        ("ACK", "algorithm"),
        ("ACK", "key_id"),
        ("ACK", "signature"),
        ("ACK", "noncanonical"),
        ("ACK", "bool_attempt"),
        ("ACK", "bool_time"),
        ("TERMINAL", "top_field"),
        ("TERMINAL", "payload_field"),
        ("TERMINAL", "missing_payload_field"),
        ("TERMINAL", "bool_attempt"),
        ("TERMINAL", "bool_time"),
    ],
)
def test_task_verifier_rejects_open_missing_or_weakly_typed_receipt(stage: str, mutation: str) -> None:
    receipts = _receipt_module()
    contract = _contract()
    ack = _signed(_ack_payload(contract))
    payload = _ack_payload(contract) if stage == "ACK" else _terminal_payload(contract, ack)
    document = json.loads(_signed(payload))
    if mutation == "top_field":
        document["extra"] = None
    elif mutation == "payload_field":
        document["payload"]["extra"] = None
    elif mutation == "missing_payload_field":
        document["payload"].pop("challenge")
    elif mutation == "algorithm":
        document["signature"]["algorithm"] = "EdDSA"
    elif mutation == "key_id":
        document["signature"]["key_id"] = "attacker"
    elif mutation == "signature":
        document["signature"]["value_base64"] = "not-base64"
    elif mutation == "bool_attempt":
        document["payload"]["attempt_id"] = False
    elif mutation == "bool_time":
        document["payload"]["issued_at_ms"] = True
    if mutation in {"payload_field", "missing_payload_field", "bool_attempt", "bool_time"}:
        value = _signed(document["payload"])
    else:
        value = _canonical(document)
    if mutation == "noncanonical":
        value = json.dumps(document).encode()

    with pytest.raises(ValueError):
        if stage == "ACK":
            receipts.verify_ack_receipt(value, contract)
        else:
            receipts.verify_terminal_receipt(value, ack, contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "attacker.receipt.v1"),
        ("cluster", "attacker-cluster"),
        ("controller_id", "attacker-controller"),
        ("authorization_sha256", "9" * 64),
        ("challenge", "9" * 64),
        ("job_id", "/attacker/job"),
        ("task_id", "/attacker/job/0"),
        ("attempt_id", 1),
        ("attempt_uid", "fedcba9876543210"),
        ("launch_request_sha256", "9" * 64),
        ("ack_receipt_sha256", "9" * 64),
        ("runner_exit_code", 1),
        ("job_state", "RUNNING"),
        ("task_state", "FAILED"),
        ("ready_log_seq", 101),
        ("ack_log_seq", 100),
        ("terminal_log_seq", 101),
        ("issued_at_ms", 1_799_999_999_999),
    ],
)
def test_terminal_verifier_rejects_cross_chain_replay_or_state_substitution(field: str, value: object) -> None:
    receipts = _receipt_module()
    contract = _contract()
    ack = _signed(_ack_payload(contract))
    terminal = _terminal_payload(contract, ack)
    terminal[field] = value

    with pytest.raises(ValueError):
        receipts.verify_terminal_receipt(_signed(terminal), ack, contract)


def test_task_success_requires_valid_ack_and_does_not_accept_terminal_as_ack() -> None:
    receipts = _receipt_module()
    contract = _contract()
    ack = _signed(_ack_payload(contract))
    terminal = _signed(_terminal_payload(contract, ack))

    assert receipts.task_success_allowed(ack, contract) is True
    with pytest.raises(ValueError):
        receipts.task_success_allowed(terminal, contract)


def test_controller_derives_ack_from_database_and_actual_verifier_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    receipts = _receipt_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    calls: list[tuple[Path, Path]] = []
    reports: list[bytes] = []
    _install_verifier(monkeypatch, controller, calls, reports)
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )

    try:
        ack = _issue_ack(store, paths)
        payload = json.loads(ack)["payload"]
        assert receipts.verify_ack_receipt(ack, contract) == payload
        assert payload["attempt_uid"] == attempt_uid
        assert payload["job_state"] == "RUNNING"
        assert payload["task_state"] == "RUNNING"
        with database.read_snapshot() as tx:
            job_id = JobName.from_string(contract["launch"]["job_id"])
            reconstructed = reconstruct_launch_job_request(
                reads.get_job_detail(tx, job_id),
                workdir_files=reads.get_workdir_files(tx, job_id),
            ).SerializeToString(deterministic=True)
        assert payload["launch_request_sha256"] == hashlib.sha256(reconstructed).hexdigest()
        assert payload["archive_sha256"] == hashlib.sha256(paths["archive"].read_bytes()).hexdigest()
        assert payload["export_manifest_sha256"] == hashlib.sha256(paths["export_manifest"].read_bytes()).hexdigest()
        assert payload["ready_sha256"] == hashlib.sha256(paths["ready_log_record"].read_bytes()).hexdigest()
        assert payload["ready_log_seq"] == 100
        assert payload["verification_report_sha256"] == hashlib.sha256(reports[0]).hexdigest()
        assert payload["verifier_sha256"] == hashlib.sha256(paths["verifier"].read_bytes()).hexdigest()
        assert calls == [(paths["archive"], paths["export_manifest"])]
    finally:
        database.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "image",
        "bundle",
        "argv",
        "max_retries_failure",
        "max_retries_preemption",
        "max_task_failures",
        "fail_if_exists",
        "existing_job_policy",
    ],
)
def test_controller_rejects_persisted_launch_request_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    controller = _controller_module()
    contract = _contract()
    request = controller_pb2.Controller.LaunchJobRequest()
    request.CopyFrom(_launch_request(contract))
    if mutation == "image":
        request.task_image = "ghcr.io/attacker/image@sha256:" + "9" * 64
    elif mutation == "bundle":
        request.bundle_id = "9" * 64
    elif mutation == "argv":
        request.entrypoint.run_command.argv[-1] = "9" * 64
    elif mutation == "fail_if_exists":
        request.fail_if_exists = False
    elif mutation == "existing_job_policy":
        request.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_RECREATE
    else:
        setattr(request, mutation, 1)
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract, request=request)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") is None
    finally:
        database.close()


def test_controller_receipts_are_persistently_idempotent_per_authorization_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    contract = _contract()
    database_path = tmp_path / "db"
    database = _controller_db(database_path)
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        first = _issue_ack(store, paths)
    finally:
        database.close()

    reopened_database = _controller_db(database_path)
    reopened = controller.ControllerReceiptStore(
        reopened_database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        assert _issue_ack(reopened, paths) == first
        paths["archive"].write_bytes(b"substituted archive\n")
        with pytest.raises(ValueError):
            _issue_ack(reopened, paths)
        assert reopened.stored_receipt("ACK") == first
    finally:
        reopened_database.close()


@pytest.mark.parametrize("input_name", ["export_manifest", "ready_log_record"])
def test_controller_rejects_changed_ack_input_bytes_and_preserves_stored_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        first = _issue_ack(store, paths)
        paths[input_name].write_bytes(paths[input_name].read_bytes() + b"changed\n")
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") == first
    finally:
        database.close()


def test_controller_rejects_changed_verifier_report_and_preserves_stored_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        first = _issue_ack(store, paths)
        _install_verifier(
            monkeypatch,
            controller,
            [],
            report_mutation={"controller_test_nonce": "changed"},
        )
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") == first
    finally:
        database.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "/attacker/job"),
        ("task_id", "/attacker/job/0"),
        ("attempt_id", 1),
        ("attempt_uid", "fedcba9876543210"),
        ("record", "TERMINAL"),
        ("log_seq", 101),
    ],
)
def test_controller_rejects_canonical_ready_identity_substitution_without_partial_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    record = json.loads(paths["ready_log_record"].read_bytes())
    record[field] = value
    paths["ready_log_record"].write_bytes(_canonical(record))
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") is None
    finally:
        database.close()


def test_controller_rejects_nonrunning_database_state_or_changed_verifier_report_without_partial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    calls: list[tuple[Path, Path]] = []

    def rejected_verifier(*, archive_path: Path, export_manifest_path: Path) -> bytes:
        calls.append((archive_path, export_manifest_path))
        return _canonical({"verdict": "rejected"})

    monkeypatch.setattr(controller, "run_abi15_evidence_verifier", rejected_verifier)
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") is None

        _install_verifier(monkeypatch, controller, calls)
        with database.transaction() as tx:
            tx.execute(
                sa_update(tasks_table)
                .where(tasks_table.c.task_id == JobName.from_string(contract["launch"]["task_id"]))
                .values(state=job_pb2.TASK_STATE_FAILED)
            )
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") is None
    finally:
        database.close()


def test_controller_rejects_controller_database_cross_attempt_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    task_id = JobName.from_string(contract["launch"]["task_id"])
    try:
        with database.transaction() as tx:
            writes.promote_for_dispatch(tx, task_id, 1, 1_800_000_000_200)
            tx.execute(
                sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(state=job_pb2.TASK_STATE_RUNNING)
            )
            tx.execute(
                sa_update(task_attempts_table)
                .where(task_attempts_table.c.task_id == task_id, task_attempts_table.c.attempt_id == 1)
                .values(state=job_pb2.TASK_STATE_RUNNING, started_at_ms=1_800_000_000_200)
            )
        with pytest.raises(ValueError):
            _issue_ack(store, paths)
        assert store.stored_receipt("ACK") is None
    finally:
        database.close()


def test_controller_derives_terminal_from_same_successful_attempt_and_database_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    receipts = _receipt_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        ack = _issue_ack(store, paths)
        with pytest.raises(ValueError):
            store.issue_terminal(terminal_log_record_path=paths["terminal_log_record"])

        _finish_attempt(database, contract)
        terminal = store.issue_terminal(terminal_log_record_path=paths["terminal_log_record"])
        payload = json.loads(terminal)["payload"]
        assert receipts.verify_terminal_receipt(terminal, ack, contract) == payload
        assert payload["ack_receipt_sha256"] == hashlib.sha256(ack).hexdigest()
        assert payload["attempt_uid"] == attempt_uid
        assert payload["job_state"] == "SUCCEEDED"
        assert payload["task_state"] == "SUCCEEDED"
        assert payload["runner_exit_code"] == 0
    finally:
        database.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "/attacker/job"),
        ("task_id", "/attacker/job/0"),
        ("attempt_id", 1),
        ("attempt_uid", "fedcba9876543210"),
        ("record", "READY"),
        ("log_seq", 103),
        ("runner_exit_code", 1),
    ],
)
def test_controller_rejects_canonical_terminal_identity_substitution_without_partial_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path / "db")
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        _issue_ack(store, paths)
        _finish_attempt(database, contract)
        record = json.loads(paths["terminal_log_record"].read_bytes())
        record[field] = value
        paths["terminal_log_record"].write_bytes(_canonical(record))
        with pytest.raises(ValueError):
            store.issue_terminal(terminal_log_record_path=paths["terminal_log_record"])
        assert store.stored_receipt("TERMINAL") is None
    finally:
        database.close()


def test_controller_terminal_receipt_is_persistently_idempotent_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller_module()
    contract = _contract()
    database_path = tmp_path / "db"
    database = _controller_db(database_path)
    attempt_uid = _seed_running_attempt(database, contract)
    paths = _evidence_paths(tmp_path, contract, attempt_uid)
    _install_verifier(monkeypatch, controller, [])
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        _issue_ack(store, paths)
        _finish_attempt(database, contract)
        first = store.issue_terminal(terminal_log_record_path=paths["terminal_log_record"])
    finally:
        database.close()

    reopened_database = _controller_db(database_path)
    reopened = controller.ControllerReceiptStore(
        reopened_database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    try:
        assert reopened.issue_terminal(terminal_log_record_path=paths["terminal_log_record"]) == first
        record = json.loads(paths["terminal_log_record"].read_bytes())
        record["log_seq"] = 103
        paths["terminal_log_record"].write_bytes(_canonical(record))
        with pytest.raises(ValueError):
            reopened.issue_terminal(terminal_log_record_path=paths["terminal_log_record"])
        assert reopened.stored_receipt("TERMINAL") == first
    finally:
        reopened_database.close()


@pytest.mark.parametrize("retry_field", ["max_retries_failure", "max_retries_preemption", "max_task_failures"])
def test_controller_rejects_any_nonzero_retry_policy(tmp_path: Path, retry_field: str) -> None:
    controller = _controller_module()
    contract = _contract()
    request = _launch_request(contract)
    setattr(request, retry_field, 1)
    raw = request.SerializeToString(deterministic=True)
    retried = deepcopy(contract)
    retried["launch"][retry_field] = 1
    retried["launch"]["launch_request_base64"] = base64.b64encode(raw).decode()
    retried["launch"]["launch_request_sha256"] = hashlib.sha256(raw).hexdigest()
    database = _controller_db(tmp_path)
    try:
        with pytest.raises(ValueError):
            controller.ControllerReceiptStore(
                database,
                signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
                retried,
            )
    finally:
        database.close()
