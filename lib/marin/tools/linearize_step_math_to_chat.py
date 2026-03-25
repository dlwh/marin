# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert step-math JSONL records into OpenAI chat-format examples."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as in_file:  # type: ignore[arg-type]
        return [json.loads(line) for line in in_file if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as out_file:  # type: ignore[arg-type]
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _render_step(step: dict[str, Any], include_change_type: bool) -> str:
    before = str(step["before"])
    after = str(step["after"])
    if include_change_type:
        change_type = str(step.get("change_type", "UNKNOWN"))
        return f"[{change_type}] {before} -> {after}"
    return f"{before} -> {after}"


def render_assistant_response(record: dict[str, Any], include_change_type: bool) -> str:
    """Render deterministic step-by-step response text."""
    steps = record.get("steps", [])
    if not steps:
        return f"Final answer: {record.get('canonical_final_answer', record.get('solver_final_state', ''))}".strip()

    lines = ["Step-by-step solution:"]
    for i, step in enumerate(steps, start=1):
        lines.append(f"{i}. {_render_step(step, include_change_type=include_change_type)}")

    final_answer = record.get("solver_final_state", record.get("canonical_final_answer", ""))
    canonical = record.get("canonical_final_answer")
    lines.append(f"Final answer: {final_answer}")
    if canonical and canonical != final_answer:
        lines.append(f"Canonical answer: {canonical}")
    return "\n".join(lines)


def to_chat_record(
    record: dict[str, Any],
    include_change_type: bool,
    source: str,
) -> dict[str, Any]:
    """Convert one step-math record to an OpenAI chat-format record."""
    question = str(record["question"])
    assistant_response = render_assistant_response(record, include_change_type=include_change_type)
    return {
        "id": record.get("id"),
        "source": source,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": {
            "domain": record.get("domain"),
            "solver_kind": record.get("solver_kind"),
            "solver_input": record.get("solver_input"),
            "verification": record.get("verification", {}),
            "original_seed": record.get("seed"),
        },
    }


def convert_records(
    records: list[dict[str, Any]],
    include_change_type: bool,
    drop_unverified: bool,
    source: str,
) -> list[dict[str, Any]]:
    """Convert many records to chat-format training examples."""
    out: list[dict[str, Any]] = []
    for record in records:
        verified = bool(record.get("verification", {}).get("all_steps_valid", False))
        if drop_unverified and not verified:
            continue
        out.append(
            to_chat_record(
                record,
                include_change_type=include_change_type,
                source=source,
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linearize step-math records into OpenAI chat-format JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL/JSONL.GZ from generate_step_math_dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL/JSONL.GZ in OpenAI chat format")
    parser.add_argument(
        "--source",
        default="synthetic/stepmath-v0",
        help="Source tag to include in each output row",
    )
    parser.add_argument(
        "--drop-unverified",
        action="store_true",
        help="Drop any example without verification.all_steps_valid=true",
    )
    parser.add_argument(
        "--omit-change-type",
        action="store_true",
        help="Do not include mathsteps rule names in rendered steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.input.expanduser().resolve())
    converted = convert_records(
        records=records,
        include_change_type=not args.omit_change_type,
        drop_unverified=args.drop_unverified,
        source=args.source,
    )
    _write_jsonl(args.output.expanduser().resolve(), converted)


if __name__ == "__main__":
    main()
