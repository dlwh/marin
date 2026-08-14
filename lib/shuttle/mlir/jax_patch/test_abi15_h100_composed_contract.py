# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from abi15_h100_composed_contract import (
    CONTRACT_SHA256,
    STATUS,
    contract_path,
    load_contract,
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
