# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from abi15_h100_composed_contract import (
    CONTRACT_SHA256,
    STATUS,
    contract_path,
    load_contract,
    validate_contract_document,
)


def test_contract_recomputes_fixture_oracles_and_policy_identities() -> None:
    contract = load_contract()
    assert contract.sha256 == CONTRACT_SHA256
    assert contract.status == STATUS
    assert contract.source_ordered.options_sha256 != contract.fast.options_sha256
    assert contract.source_ordered.generated_fingerprint != contract.fast.generated_fingerprint
    assert contract.source_ordered.final_fingerprint != contract.fast.final_fingerprint


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("fixture", "rows"), 1),
        (("policies", "fast", "generated_fingerprint"), "f" * 64),
        (("hardware_authority", "status"), "authorized"),
    ],
)
def test_contract_rejects_caller_rewritten_authority(tmp_path, path, value) -> None:
    document = json.loads(contract_path().read_text())
    selected = document
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = value
    rewritten = tmp_path / "contract.json"
    rewritten.write_text(json.dumps(document, sort_keys=True))
    with pytest.raises(ValueError, match="byte drift"):
        load_contract(rewritten)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schemas", "transport"), 3, "schema"),
        (("topology", "calls_per_policy"), 1, "topology"),
        (("process_contract", "reuse_hits"), 1, "process contract"),
        (
            ("datasets", "sparse_policy_discriminator", "selected_bits", "fast", "lhs_cotangent_0_0"),
            "0x0000",
            "selected-bit",
        ),
        (("policies", "fast", "options_sha256"), "f" * 64, "options"),
        (
            ("policies", "source_ordered", "generated_fingerprint"),
            "f" * 64,
            "generated fingerprint",
        ),
        (
            ("policies", "source_ordered", "final_fingerprint"),
            "f" * 64,
            "final fingerprint",
        ),
        (
            ("policies", "fast", "generated_fingerprint"),
            "f" * 64,
            "generated fingerprint",
        ),
        (
            ("policies", "fast", "final_fingerprint"),
            "f" * 64,
            "final fingerprint",
        ),
    ],
)
def test_semantic_validator_rejects_self_consistent_field_substitution(path, value, message) -> None:
    document = json.loads(contract_path().read_text())
    selected = document
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_contract_document(document)
