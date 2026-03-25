# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_tool_module():
    root = Path(__file__).resolve().parents[2]
    tool_path = root / "lib" / "marin" / "tools" / "linearize_step_math_to_chat.py"
    spec = importlib.util.spec_from_file_location("linearize_step_math_to_chat", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool_module()


def test_render_assistant_response_includes_steps_and_final() -> None:
    record = {
        "solver_final_state": "x = 4",
        "canonical_final_answer": "{4}",
        "steps": [
            {"change_type": "SUBTRACT_FROM_BOTH_SIDES", "before": "2x + 3 = 11", "after": "2x = 8"},
            {"change_type": "DIVIDE_FROM_BOTH_SIDES", "before": "2x = 8", "after": "x = 4"},
        ],
    }

    text = tool.render_assistant_response(record, include_change_type=True)
    assert "Step-by-step solution:" in text
    assert "[SUBTRACT_FROM_BOTH_SIDES] 2x + 3 = 11 -> 2x = 8" in text
    assert "Final answer: x = 4" in text
    assert "Canonical answer: {4}" in text


def test_convert_records_drops_unverified() -> None:
    verified = {
        "id": "a",
        "question": "Solve for x: x+1=2",
        "domain": "algebra_linear_equation",
        "solver_kind": "solve_equation",
        "solver_input": "x+1=2",
        "seed": 1,
        "verification": {"all_steps_valid": True},
        "steps": [{"change_type": "SUBTRACT_FROM_BOTH_SIDES", "before": "x + 1 = 2", "after": "x = 1"}],
        "solver_final_state": "x = 1",
    }
    unverified = {
        "id": "b",
        "question": "Solve for x: x+1=2",
        "domain": "algebra_linear_equation",
        "solver_kind": "solve_equation",
        "solver_input": "x+1=2",
        "seed": 2,
        "verification": {"all_steps_valid": False},
        "steps": [{"change_type": "BAD", "before": "x + 1 = 2", "after": "x = 999"}],
        "solver_final_state": "x = 999",
    }

    out = tool.convert_records(
        [verified, unverified],
        include_change_type=True,
        drop_unverified=True,
        source="synthetic/stepmath-v0",
    )

    assert len(out) == 1
    assert out[0]["id"] == "a"
    assert out[0]["messages"][0]["role"] == "user"
    assert out[0]["messages"][1]["role"] == "assistant"
