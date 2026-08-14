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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from rigging.token_authority import signing_key_from_private_pem

CONTRACT_FILENAME = "abi15-h100-controller-receipt-red-v1.json"
DOMAIN = "marin.shuttle.abi15.evidence-receipt.v1"
PRIVATE_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
OTHER_PRIVATE_SEED = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
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
    return importlib.import_module("iris.cluster.controller.db").ControllerDB(path)


def _ack_payload(contract: dict[str, Any]) -> dict[str, object]:
    launch = contract["launch"]
    authority = contract["authority"]
    return {
        "archive_sha256": "2" * 64,
        "attempt_id": launch["attempt_id"],
        "attempt_uid": launch["attempt_uid"],
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


def _signed(payload: dict[str, object], *, seed: bytes = PRIVATE_SEED) -> bytes:
    key = Ed25519PrivateKey.from_private_bytes(seed)
    signature = key.sign(DOMAIN.encode() + b"\0" + _canonical(payload))
    return _canonical(
        {
            "payload": payload,
            "schema_version": 1,
            "signature": {
                "algorithm": "Ed25519",
                "key_id": "kPrK_qmxVWaYVA9wwBF6Iuo3vVzz7TxHCTwXBygrS4k",
                "value_base64": base64.b64encode(signature).decode("ascii"),
            },
        }
    )


def _terminal_payload(contract: dict[str, Any], ack: bytes) -> dict[str, object]:
    launch = contract["launch"]
    authority = contract["authority"]
    return {
        "ack_log_seq": 101,
        "ack_receipt_sha256": hashlib.sha256(ack).hexdigest(),
        "attempt_id": launch["attempt_id"],
        "attempt_uid": launch["attempt_uid"],
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


def test_receipt_contract_is_canonical_and_keeps_external_authority_unapproved() -> None:
    contract = _contract()

    assert contract["status"] == "test_only_red_controller_receipt_contract_not_launch_authority"
    assert contract["design_stops"] == {
        "controller_signer_integration": None,
        "hardware_launch": None,
        "image_transfer": None,
        "receipt_authority_approval": None,
        "secret_projection": None,
    }
    assert contract["authority"]["previous_key_fallback_allowed"] is False


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


def test_task_verifier_rejects_task_forged_key_or_previous_key_fallback() -> None:
    receipts = _receipt_module()
    contract = _contract()
    forged = _signed(_ack_payload(contract), seed=OTHER_PRIVATE_SEED)
    rotated = deepcopy(contract)
    rotated["authority"]["previous_key_fallback_allowed"] = True

    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(forged, contract)
    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(_signed(_ack_payload(rotated)), rotated)


@pytest.mark.parametrize(
    "mutation",
    ["top_field", "payload_field", "algorithm", "key_id", "signature", "noncanonical", "bool_attempt", "bool_time"],
)
def test_task_verifier_rejects_open_malformed_or_weakly_typed_receipt(mutation: str) -> None:
    receipts = _receipt_module()
    contract = _contract()
    document = json.loads(_signed(_ack_payload(contract)))
    if mutation == "top_field":
        document["extra"] = None
    elif mutation == "payload_field":
        document["payload"]["extra"] = None
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
    if mutation in {"payload_field", "bool_attempt", "bool_time"}:
        value = _signed(document["payload"])
    else:
        value = _canonical(document)
    if mutation == "noncanonical":
        value = json.dumps(document).encode()

    with pytest.raises(ValueError):
        receipts.verify_ack_receipt(value, contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
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
def test_terminal_verifier_rejects_replay_failure_or_nonmonotonic_sequence(field: str, value: object) -> None:
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


def test_controller_store_is_idempotent_per_authorization_and_stage(tmp_path: Path) -> None:
    controller = _controller_module()
    contract = _contract()
    signing_key = signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM)
    database = _controller_db(tmp_path)
    store = controller.ControllerReceiptStore(database, signing_key, contract)
    ack_payload = _ack_payload(contract)

    try:
        first = store.issue_ack(ack_payload)
    finally:
        database.close()
    reopened_database = _controller_db(tmp_path)
    reopened = controller.ControllerReceiptStore(reopened_database, signing_key, contract)
    try:
        second = reopened.issue_ack(deepcopy(ack_payload))

        assert first == second
        changed = deepcopy(ack_payload)
        changed["archive_sha256"] = "9" * 64
        with pytest.raises(ValueError):
            reopened.issue_ack(changed)
    finally:
        reopened_database.close()


def test_controller_store_requires_ack_running_then_terminal_success(tmp_path: Path) -> None:
    controller = _controller_module()
    contract = _contract()
    database = _controller_db(tmp_path)
    store = controller.ControllerReceiptStore(
        database,
        signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM),
        contract,
    )
    ack_payload = _ack_payload(contract)
    unissued_ack = _signed(ack_payload)
    terminal_payload = _terminal_payload(contract, unissued_ack)

    try:
        with pytest.raises(ValueError):
            store.issue_terminal(terminal_payload)

        ack = store.issue_ack(ack_payload)
        terminal_payload = _terminal_payload(contract, ack)
        terminal = store.issue_terminal(terminal_payload)
        assert json.loads(terminal)["payload"]["ack_receipt_sha256"] == hashlib.sha256(ack).hexdigest()
    finally:
        database.close()


def test_controller_store_rejects_retry_or_cross_attempt_receipt(tmp_path: Path) -> None:
    controller = _controller_module()
    contract = _contract()
    signing_key = signing_key_from_private_pem(RFC8032_TEST_PRIVATE_KEY_PEM)
    retried = deepcopy(contract)
    retried["launch"]["max_retries"] = 1
    database = _controller_db(tmp_path)

    try:
        with pytest.raises(ValueError):
            controller.ControllerReceiptStore(database, signing_key, retried)
        store = controller.ControllerReceiptStore(database, signing_key, contract)
        other_attempt = _ack_payload(contract)
        other_attempt["attempt_uid"] = "fedcba9876543210"
        with pytest.raises(ValueError):
            store.issue_ack(other_attempt)
    finally:
        database.close()
