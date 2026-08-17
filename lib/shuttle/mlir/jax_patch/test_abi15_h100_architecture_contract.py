# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from abi15_h100_architecture_contract import FILENAME, load_architecture_build_contract


def _document() -> dict[str, object]:
    return json.loads(Path(__file__).with_name(FILENAME).read_text())


def _write(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_architecture_design_derives_exact_four_policy_coordinates_without_authority() -> None:
    document = load_architecture_build_contract()

    assert [len(document["coordinates"][call]["launches"]) for call in ("forward", "backward")] == [2, 10]
    assert document["build_policy"]["future_outputs"] == {
        "ack_trust_root": None,
        "authorization_sha256": None,
        "bazel_path": None,
        "bazel_sha256": None,
        "cubin_roots": None,
        "image_digest": None,
        "ptxas_path": None,
        "ptxas_sha256": None,
        "ptx_roots": None,
        "wheel_sha256": None,
    }


@pytest.mark.parametrize(
    "mutation",
    ["read_result", "write_operand", "uninitialized", "dependencies", "kind", "dimensions", "output", "design_stop"],
)
def test_architecture_design_rejects_use_def_access_dependency_or_semantic_mutation(
    tmp_path: Path, mutation: str
) -> None:
    document = deepcopy(_document())
    backward = document["coordinates"]["backward"]
    if mutation == "read_result":
        backward["launches"][9]["input_slots"] = [12]
    elif mutation == "write_operand":
        backward["launches"][0]["output_slots"] = [2]
    elif mutation == "uninitialized":
        backward["launches"][0]["input_slots"] = [3]
    elif mutation == "dependencies":
        backward["launches"][3]["dependencies"] = [2, 1]
    elif mutation == "kind":
        backward["launches"][4]["kind"] = "map_f32_identity"
    elif mutation == "dimensions":
        backward["slots"][6]["dimensions"] = [2, 34]
    elif mutation == "output":
        document["build_policy"]["future_outputs"]["image_digest"] = "sha256:" + "a" * 64
    else:
        document["design_stops"]["task_success_exit"] = 0
    path = tmp_path / FILENAME
    _write(path, document)

    with pytest.raises(ValueError):
        load_architecture_build_contract(path)
