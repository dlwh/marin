# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import track1_generic_composition_h100_worker as worker
from track1_map_h100_prototype import canonical_json


def test_cell_report_is_flushed_to_stdout_and_create_only_diagnostics(tmp_path: Path, capsys) -> None:
    report = {"case_id": "rank2-a", "schema": "test-only-record"}
    diagnostics = tmp_path / "coordinates.ndjson"

    with diagnostics.open("xb") as stream:
        worker.emit_cell_report(stream, report, validator=lambda value: value)

    assert diagnostics.read_bytes() == canonical_json(report)
    line = capsys.readouterr().out
    assert line.startswith(worker.CELL_REPORT_PREFIX)
    assert json.loads(line.removeprefix(worker.CELL_REPORT_PREFIX)) == report
