# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for the Track 1 RMSNorm-to-Contract H100 canary."""

import argparse
import os
import sys
from pathlib import Path

from track1_map_h100_prototype import canonical_json, validate_rmsnorm_contract_result
from track1_rmsnorm_contract_h100_subject import (
    MAX_CELL_REPORT_BYTES,
    MAX_PRIMAL_CHECKPOINT_BYTES,
    result_document,
    run_matrix,
)

CELL_REPORT_PREFIX = "RMSNORM_CONTRACT_CELL "


def emit_cell_report(document: dict[str, object]) -> None:
    """Flush one bounded canonical diagnostic into the durable phase log."""
    payload = canonical_json(document)
    if len(payload) >= MAX_CELL_REPORT_BYTES:
        raise RuntimeError("RMSNorm Contract cell report exceeds its bounded diagnostic size")
    sys.stdout.write(CELL_REPORT_PREFIX + payload.decode())
    sys.stdout.flush()


def write_primal_checkpoint(path: Path, document: dict[str, object]) -> None:
    """Persist one bounded create-only checkpoint before composed execution."""
    payload = canonical_json(document)
    if len(payload) >= MAX_PRIMAL_CHECKPOINT_BYTES:
        raise RuntimeError("RMSNorm Contract primal checkpoint exceeds its bounded size")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primal-checkpoint", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for path in (arguments.output, arguments.primal_checkpoint):
        if path.exists() or path.is_symlink():
            raise RuntimeError("RMSNorm Contract worker output paths must not exist")
    document = result_document(
        run_matrix(
            report_cell=emit_cell_report,
            report_primal_checkpoint=lambda checkpoint: write_primal_checkpoint(arguments.primal_checkpoint, checkpoint),
        )
    )
    validate_rmsnorm_contract_result(document)
    arguments.output.write_bytes(canonical_json(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
