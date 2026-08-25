# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for the anonymous generic composition H100 canary."""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from track1_generic_composition_h100_subject import result_document, run_matrix
from track1_generic_composition_numerical_policy import CELL_REPORT_PREFIX
from track1_map_h100_prototype import (
    canonical_json,
    validate_generic_composition_cell_report,
    validate_generic_composition_result,
)


def emit_cell_report(
    stream: BinaryIO,
    report: dict[str, object],
    *,
    validator: Callable[[object], dict[str, object]] = validate_generic_composition_cell_report,
) -> None:
    """Persist and surface one closed coordinate record before threshold gating."""
    validated = validator(report)
    payload = canonical_json(validated)
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    sys.stdout.write(CELL_REPORT_PREFIX + payload.decode())
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise RuntimeError("generic composition worker output must not exist")
    if arguments.diagnostics.exists() or arguments.diagnostics.is_symlink():
        raise RuntimeError("generic composition worker diagnostics must not exist")
    with arguments.diagnostics.open("xb") as stream:
        results = run_matrix(report_cell=lambda report: emit_cell_report(stream, report))
    document = result_document(results)
    validate_generic_composition_result(document)
    with arguments.output.open("xb") as stream:
        stream.write(canonical_json(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
