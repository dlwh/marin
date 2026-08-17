# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from abi21_rmsnorm_prehardware_contract import (
    ARCHITECTURE_FILENAME,
    CONTRACT_FILENAME,
    ROLE_MANIFEST_FILENAME,
    contract_path,
    expected_role_paths,
    load_architecture_contract,
    load_contract,
    load_role_manifest,
    validate_architecture_document,
    validate_contract_document,
    validate_role_manifest_document,
)


def _document(filename: str) -> dict[str, object]:
    return json.loads(Path(__file__).with_name(filename).read_text())


def test_contract_closes_architecture_metamorphic_and_acceptance_tiers() -> None:
    contract = load_contract()

    assert contract.contract_id == "shuttle_abi21_rmsnorm_prehardware_v1"
    assert contract.status == "local_unsealed_prehardware_design"
    assert contract.pipeline_abi_version == 21
    assert contract.architecture_shape == (2, 7, 19)
    assert contract.metamorphic_shape == (3, 2, 5, 11, 263)
    assert contract.acceptance_shape == (2048, 4096)
    assert contract.boundaries == ("forward", "backward", "composed")
    assert contract.policies == ("source_ordered", "fast")
    assert contract.acceptance_hardware == ("h100", "b200")
    assert contract.acceptance_cells == 12
    assert contract.representative_live_temporary_bytes == {
        "backward": 167_804_936,
        "composed": 167_804_936,
    }
    assert contract.numerical_policy_status == "pending_independent_review"
    assert "numerical_policy_contract_pending" in contract.readiness_blockers
    assert all("resource" not in blocker for blocker in contract.readiness_blockers)


def test_architecture_contract_binds_raw_and_live_hook_topologies() -> None:
    architecture = load_architecture_contract()

    assert architecture.raw_fixture_counts == {
        "forward": (19, 21),
        "backward": (48, 51),
        "composed": (51, 54),
    }
    assert architecture.xla_hook_counts == {
        "forward": (19, 21),
        "backward": (43, 46),
        "composed": (46, 49),
    }
    assert architecture.external_bindings == {"forward": 3, "backward": 5, "composed": 6}
    assert architecture.task_census == {
        "forward": (18, 1, 0),
        "backward": (38, 3, 2),
        "composed": (41, 3, 2),
    }
    assert architecture.public_result_projections == {
        "forward": (0,),
        "backward": (1, 0),
        "composed": (0, 2, 1),
    }
    assert architecture.executed_ptx_slices == 216


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("acceptance_shape", "acceptance tier"),
        ("live_bytes", "live temporary"),
        ("drop_boundary", "boundary matrix"),
        ("authorize", "authority"),
        ("policy_identity", "policy identity"),
        ("bitwise_gate", "ordinary JAX comparator"),
        ("old_contract", "Target-1 v1"),
        ("architecture_digest", "architecture contract"),
    ],
)
def test_contract_rejects_self_consistent_scope_policy_or_authority_mutation(mutation: str, message: str) -> None:
    document = deepcopy(_document(CONTRACT_FILENAME))
    if mutation == "acceptance_shape":
        document["shape_tiers"]["acceptance"]["dimensions"] = [7, 13]
    elif mutation == "live_bytes":
        document["shape_tiers"]["acceptance"]["live_temporary_bytes"]["backward"] += 1
    elif mutation == "drop_boundary":
        document["matrix"]["boundaries"].pop()
        document["matrix"]["acceptance_cells"] = 8
    elif mutation == "authorize":
        document["authority"]["status"] = "authorized"
        document["authority"]["image_digest"] = "sha256:" + "a" * 64
    elif mutation == "policy_identity":
        document["numerical_policy"]["normalized_code_equality_required"] = False
    elif mutation == "bitwise_gate":
        document["numerical_policy"]["ordinary_jax_role"] = "bitwise_acceptance_gate"
    elif mutation == "old_contract":
        document["numerical_policy"]["imported_target1_v1"] = True
    else:
        document["dependencies"]["architecture_contract_sha256"] = "a" * 64

    with pytest.raises(ValueError, match=message):
        validate_contract_document(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hook_count", "XLA hook"),
        ("projection", "projection"),
        ("task_census", "task census"),
        ("ptx_total", "PTX"),
        ("column_fold", "ColumnFold"),
    ],
)
def test_architecture_contract_rejects_topology_or_projection_mutation(mutation: str, message: str) -> None:
    document = deepcopy(_document(ARCHITECTURE_FILENAME))
    if mutation == "hook_count":
        document["xla_hook_topology"]["backward"]["launches"] = 48
    elif mutation == "projection":
        document["xla_hook_topology"]["backward"]["public_result_projection"] = [0, 1]
    elif mutation == "task_census":
        document["xla_hook_topology"]["composed"]["task_census"]["map"] -= 1
    elif mutation == "ptx_total":
        document["executed_matrix"]["ptx_slices"] = 236
    else:
        document["metamorphic_column_folds"]["main"]["reduction_extent"] = 329

    with pytest.raises(ValueError, match=message):
        validate_architecture_document(document)


def test_role_manifest_expands_only_the_bounded_runtime_and_compile_witness() -> None:
    roles = expected_role_paths(load_role_manifest())

    assert roles["ptx_fast_backward_42"] == "assembly/fast/backward/entry-42.ptx"
    assert "ptx_fast_backward_43" not in roles
    assert roles["cubin_source_ordered_forward_18"] == "assembly/source_ordered/forward/entry-18.cubin"
    assert roles["cache_entry_fast_composed"] == "cache/entries/fast/composed.bin"
    assert roles["output_source_ordered_backward_thread_1_input_cotangent"] == (
        "outputs/source_ordered/backward/thread_1/input-cotangent.bf16"
    )
    assert roles["metamorphic_fast_composed_ptx_witness"] == ("metamorphic/fast/composed/ptx-witness.json")
    assert "accepted_bundle" not in roles
    assert len(roles) == len(set(roles.values()))


@pytest.mark.parametrize("mutation", ["drop_policy", "extra_launch", "alias_path", "acceptance_role"])
def test_role_manifest_rejects_open_or_collapsed_role_roster(mutation: str) -> None:
    document = deepcopy(_document(ROLE_MANIFEST_FILENAME))
    if mutation == "drop_policy":
        document["policies"].pop()
    elif mutation == "extra_launch":
        document["calls"]["backward"].append(43)
    elif mutation == "alias_path":
        document["static_roles"]["architecture_contract"] = document["static_roles"]["contract"]
    else:
        document["static_roles"]["accepted_bundle"] = "accepted/bundle.json"

    with pytest.raises(ValueError):
        validate_role_manifest_document(document)


def test_exact_loader_rejects_rewritten_canonical_contract(tmp_path: Path) -> None:
    document = _document(CONTRACT_FILENAME)
    document["authority"]["source_capsule_sha256"] = "a" * 64
    rewritten = tmp_path / CONTRACT_FILENAME
    rewritten.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="byte drift"):
        load_contract(rewritten)

    assert contract_path().name == CONTRACT_FILENAME
