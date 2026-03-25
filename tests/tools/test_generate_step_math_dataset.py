# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool_module():
    root = Path(__file__).resolve().parents[2]
    tool_path = root / "lib" / "marin" / "tools" / "generate_step_math_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_step_math_dataset", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool_module()


def test_verify_expression_transition() -> None:
    valid, reason = tool.verify_expression_transition("2 + 3 * (4 - 1)", "11")
    assert valid, reason

    valid, reason = tool.verify_expression_transition("2 + 3", "6")
    assert not valid, reason


def test_verify_equation_transition() -> None:
    valid, reason = tool.verify_equation_transition("2x + 3 = 11", "(2x + 3) - 3 = 11 - 3")
    assert valid, reason

    valid, reason = tool.verify_equation_transition("2x + 3 = 11", "2x = 7")
    assert not valid, reason


def test_build_dataset_with_mock_solver(monkeypatch) -> None:
    def fake_solver(specs, bridge_path, mathsteps_home):
        del bridge_path, mathsteps_home
        return [
            {
                "ok": True,
                "steps": [
                    {
                        "change_type": "NO_OP",
                        "before": spec.solver_input,
                        "after": spec.solver_input,
                    }
                ],
                "final_state": spec.solver_input,
            }
            for spec in specs
        ]

    monkeypatch.setattr(tool, "run_mathsteps_batch", fake_solver)

    records = tool.build_dataset(
        num_examples=10,
        seed=7,
        algebra_ratio=0.5,
        batch_size=4,
        max_attempts=40,
        bridge_path=Path("/tmp/bridge.js"),
        mathsteps_home=Path("/tmp/mathsteps"),
        allow_unverified=False,
    )

    assert len(records) == 10
    assert all(record["verification"]["all_steps_valid"] for record in records)


def test_build_dataset_rejects_unverified_solver_steps(monkeypatch) -> None:
    def fake_bad_solver(specs, bridge_path, mathsteps_home):
        del bridge_path, mathsteps_home
        return [
            {
                "ok": True,
                "steps": [
                    {
                        "change_type": "BROKEN",
                        "before": spec.solver_input,
                        "after": "x = 99999",
                    }
                ],
                "final_state": "x = 99999",
            }
            for spec in specs
        ]

    monkeypatch.setattr(tool, "run_mathsteps_batch", fake_bad_solver)

    with pytest.raises(RuntimeError, match="Failed to generate requested examples"):
        tool.build_dataset(
            num_examples=3,
            seed=11,
            algebra_ratio=1.0,
            batch_size=2,
            max_attempts=3,
            bridge_path=Path("/tmp/bridge.js"),
            mathsteps_home=Path("/tmp/mathsteps"),
            allow_unverified=False,
        )
