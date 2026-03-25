# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool_module():
    root = Path(__file__).resolve().parents[2]
    tool_path = root / "lib" / "marin" / "tools" / "generate_synthetic_reasoning_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_reasoning_dataset", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool_module()


def test_domain_registry_size() -> None:
    domains = tool.list_domains()
    assert len(domains) >= 20
    assert "bfs_shortest_path" in domains
    assert "knapsack_01_dp" in domains
    assert "propositional_entailment" in domains
    assert "n_queens_backtracking" in domains
    assert "floyd_warshall_apsp" in domains


def test_clrs_registry() -> None:
    clrs_domains = tool.list_clrs_domains()
    assert len(clrs_domains) >= 8
    assert "clrs_bfs" in clrs_domains
    assert "clrs_dijkstra" in clrs_domains
    assert "clrs_mst_prim" in clrs_domains


def test_small_generation_and_manifest(tmp_path: Path) -> None:
    selected = ["euclid_gcd", "binary_search", "propositional_entailment", "n_queens_backtracking"]
    manifest = tool.generate_dataset(
        output_dir=tmp_path,
        domains=selected,
        examples_per_domain=5,
        seed=123,
        shard_size=3,
        include_operation_names=False,
        assistant_style="symbolic",
        max_attempt_multiplier=10,
    )

    assert manifest["totals"]["examples_emitted"] == 20
    assert manifest["totals"]["domains"] == 4

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["totals"]["examples_emitted"] == 20

    for domain_entry in loaded_manifest["domains"]:
        assert domain_entry["emitted_examples"] == 5
        assert domain_entry["verification_pass_rate"] > 0.0
        for file_path in domain_entry["canonical_files"]:
            assert Path(file_path).exists()
        for file_path in domain_entry["oai_files"]:
            assert Path(file_path).exists()


def test_small_clrs_profile_generation(tmp_path: Path) -> None:
    selected = ["clrs_bfs", "clrs_dijkstra", "clrs_lis"]
    manifest = tool.generate_dataset(
        output_dir=tmp_path,
        domains=selected,
        profile="clrs_style",
        examples_per_domain=4,
        seed=99,
        shard_size=2,
        include_operation_names=False,
        assistant_style="symbolic",
        max_attempt_multiplier=10,
    )

    assert manifest["profile"] == "clrs_style"
    assert manifest["totals"]["examples_emitted"] == 12
    for domain_entry in manifest["domains"]:
        assert domain_entry["domain"].startswith("clrs_")
        assert domain_entry["solver_domain"] in tool.list_domains()
        assert domain_entry["profile"] == "clrs_style"


def test_nl_template_adapter_output(tmp_path: Path) -> None:
    manifest = tool.generate_dataset(
        output_dir=tmp_path,
        domains=["binary_search"],
        profile="native",
        examples_per_domain=2,
        seed=7,
        shard_size=10,
        include_operation_names=False,
        assistant_style="nl_template",
        max_attempt_multiplier=10,
    )

    oai_file = Path(manifest["domains"][0]["oai_files"][0])
    import gzip

    with gzip.open(oai_file, "rt", encoding="utf-8") as f:
        row = json.loads(next(f))
    assert row["metadata"]["assistant_style"] == "nl_template"
    assistant_text = row["messages"][1]["content"]
    assert "Reasoning" in assistant_text or "reasoning" in assistant_text
    assert (
        "Final answer" in assistant_text or "final result" in assistant_text or "Therefore, the answer" in assistant_text
    )
