# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify controller-signed ABI15 H100 evidence receipts."""

import base64
import hashlib
import json
import re
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_HEX_16 = re.compile(r"[0-9a-f]{16}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_FIELDS = frozenset({"payload", "schema_version", "signature"})
_SIGNATURE_FIELDS = frozenset({"algorithm", "key_id", "value_base64"})
_ACK_INTEGER_FIELDS = frozenset({"attempt_id", "issued_at_ms", "ready_log_seq"})
_TERMINAL_INTEGER_FIELDS = frozenset(
    {
        "ack_log_seq",
        "attempt_id",
        "issued_at_ms",
        "ready_log_seq",
        "runner_exit_code",
        "terminal_log_seq",
    }
)
_ACK_DIGEST_FIELDS = frozenset(
    {
        "archive_sha256",
        "authorization_sha256",
        "bundle_sha256",
        "export_manifest_sha256",
        "launch_request_sha256",
        "ready_sha256",
        "source_capsule_sha256",
        "source_manifest_sha256",
        "verification_report_sha256",
        "verifier_sha256",
    }
)
_TERMINAL_DIGEST_FIELDS = frozenset(
    {
        "ack_receipt_sha256",
        "authorization_sha256",
        "final_job_snapshot_sha256",
        "final_task_snapshot_sha256",
        "launch_request_sha256",
    }
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _document(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ValueError("receipt must be bytes")
    try:
        document = json.loads(raw, object_pairs_hook=_closed_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("receipt must be canonical UTF-8 JSON") from exc
    if type(document) is not dict or raw != _canonical(document):
        raise ValueError("receipt must be canonical JSON")
    if set(document) != _RECEIPT_FIELDS:
        raise ValueError("receipt schema is not closed")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("receipt schema_version must equal 1")
    if type(document["payload"]) is not dict or type(document["signature"]) is not dict:
        raise ValueError("receipt payload and signature must be objects")
    return document


def _contract_object(contract: dict[str, Any], key: str) -> dict[str, Any]:
    value = contract.get(key)
    if type(value) is not dict:
        raise ValueError(f"contract {key} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _digest(value: object, field: str) -> str:
    text = _string(value, field)
    if _HEX_64.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_field_types(payload: dict[str, Any], integer_fields: frozenset[str]) -> None:
    for field, value in payload.items():
        if field in integer_fields:
            _integer(value, field, minimum=0)
        else:
            _string(value, field)


def _verify_signature(document: dict[str, Any], contract: dict[str, Any]) -> None:
    signature = document["signature"]
    if set(signature) != _SIGNATURE_FIELDS:
        raise ValueError("signature schema is not closed")
    if signature["algorithm"] != "Ed25519":
        raise ValueError("signature algorithm must be Ed25519")
    authority = _contract_object(contract, "authority")
    if authority.get("previous_key_fallback_allowed") is not False:
        raise ValueError("previous-key fallback must remain disabled")
    if signature["key_id"] != authority.get("key_id"):
        raise ValueError("receipt key_id does not match the pinned authority")
    try:
        encoded = _string(signature["value_base64"], "signature.value_base64")
        signature_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("signature must use canonical base64") from exc
    if base64.b64encode(signature_bytes).decode("ascii") != encoded:
        raise ValueError("signature must use canonical base64")
    try:
        public_key = serialization.load_pem_public_key(
            _string(authority.get("public_key_spki_pem"), "authority.public_key_spki_pem").encode()
        )
    except ValueError as exc:
        raise ValueError("authority public key is not valid SPKI PEM") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("authority public key must be Ed25519")
    domain = _string(contract.get("domain"), "contract.domain")
    try:
        public_key.verify(signature_bytes, domain.encode() + b"\0" + _canonical(document["payload"]))
    except InvalidSignature as exc:
        raise ValueError("receipt signature is invalid") from exc


def _parse_receipt(raw: bytes, contract: dict[str, Any], field_list: str) -> dict[str, Any]:
    document = _document(raw)
    fields = contract.get(field_list)
    if type(fields) is not list or not fields or any(type(field) is not str for field in fields):
        raise ValueError(f"contract {field_list} must be a nonempty string list")
    payload = document["payload"]
    if set(payload) != set(fields) or len(payload) != len(fields):
        raise ValueError(f"{field_list} does not match the receipt payload")
    _verify_signature(document, contract)
    return payload


def _require_equal(payload: dict[str, Any], field: str, expected: object) -> None:
    if payload[field] != expected:
        raise ValueError(f"receipt {field} does not match the contract or receipt chain")


def verify_ack_receipt(raw: bytes, contract: dict[str, Any]) -> dict[str, Any]:
    """Verify an ACK receipt and return its closed payload."""
    payload = _parse_receipt(raw, contract, "ack_payload_fields")
    _validate_field_types(payload, _ACK_INTEGER_FIELDS)
    for field in _ACK_DIGEST_FIELDS:
        _digest(payload[field], field)
    if _HEX_16.fullmatch(payload["attempt_uid"]) is None:
        raise ValueError("attempt_uid must be 16 lowercase hexadecimal characters")
    launch = _contract_object(contract, "launch")
    authority = _contract_object(contract, "authority")
    expected = {
        "attempt_id": launch["attempt_id"],
        "authorization_sha256": launch["authorization_sha256"],
        "bundle_sha256": launch["bundle_sha256"],
        "challenge": launch["challenge"],
        "cluster": authority["cluster"],
        "controller_id": authority["controller_id"],
        "domain": contract["domain"],
        "image_digest": launch["image_digest"],
        "job_id": launch["job_id"],
        "job_state": "RUNNING",
        "launch_request_sha256": launch["launch_request_sha256"],
        "source_capsule_sha256": launch["source_capsule_sha256"],
        "source_manifest_sha256": launch["source_manifest_sha256"],
        "stage": "ACK",
        "task_id": launch["task_id"],
        "task_state": "RUNNING",
        "verdict": "accepted",
        "verifier_sha256": launch["verifier_sha256"],
    }
    for field, value in expected.items():
        _require_equal(payload, field, value)
    _integer(payload["issued_at_ms"], "issued_at_ms", minimum=1)
    _integer(payload["ready_log_seq"], "ready_log_seq", minimum=1)
    return payload


def verify_terminal_receipt(raw: bytes, ack_raw: bytes, contract: dict[str, Any]) -> dict[str, Any]:
    """Verify a TERMINAL receipt chained to an exact ACK receipt."""
    ack = verify_ack_receipt(ack_raw, contract)
    payload = _parse_receipt(raw, contract, "terminal_payload_fields")
    _validate_field_types(payload, _TERMINAL_INTEGER_FIELDS)
    for field in _TERMINAL_DIGEST_FIELDS:
        _digest(payload[field], field)
    if _HEX_16.fullmatch(payload["attempt_uid"]) is None:
        raise ValueError("attempt_uid must be 16 lowercase hexadecimal characters")
    expected = {
        "ack_receipt_sha256": hashlib.sha256(ack_raw).hexdigest(),
        "attempt_id": ack["attempt_id"],
        "attempt_uid": ack["attempt_uid"],
        "authorization_sha256": ack["authorization_sha256"],
        "challenge": ack["challenge"],
        "cluster": ack["cluster"],
        "controller_id": ack["controller_id"],
        "domain": ack["domain"],
        "job_id": ack["job_id"],
        "job_state": "SUCCEEDED",
        "launch_request_sha256": ack["launch_request_sha256"],
        "ready_log_seq": ack["ready_log_seq"],
        "runner_exit_code": 0,
        "stage": "TERMINAL",
        "task_id": ack["task_id"],
        "task_state": "SUCCEEDED",
    }
    for field, value in expected.items():
        _require_equal(payload, field, value)
    issued_at_ms = _integer(payload["issued_at_ms"], "issued_at_ms", minimum=1)
    if issued_at_ms < ack["issued_at_ms"]:
        raise ValueError("TERMINAL must not predate ACK")
    ready_seq = _integer(payload["ready_log_seq"], "ready_log_seq", minimum=1)
    ack_seq = _integer(payload["ack_log_seq"], "ack_log_seq", minimum=1)
    terminal_seq = _integer(payload["terminal_log_seq"], "terminal_log_seq", minimum=1)
    if not ready_seq < ack_seq < terminal_seq:
        raise ValueError("receipt log sequence must satisfy READY < ACK < TERMINAL")
    return payload


def task_success_allowed(ack_raw: bytes, contract: dict[str, Any]) -> bool:
    """Return true only for a valid ACK receipt for the pinned running attempt."""
    verify_ack_receipt(ack_raw, contract)
    return True
