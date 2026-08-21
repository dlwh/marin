# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys

import pytest
import track1_map_h100_prototype as prototype
import track1_rmsnorm_contract_h100_worker as worker


class _RecordingOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_worker_flushes_canonical_cell_evidence_before_subject_failure(monkeypatch, tmp_path) -> None:
    cell = {
        "boundary": "composed",
        "mode": "disabled",
        "role": "input_cotangent",
        "schema": "rmsnorm_contract_cell_v2",
        "schema_version": 2,
        "shape": [5, 19, 7],
    }

    def fail_after_report(*, report_cell):
        report_cell(cell)
        raise RuntimeError("RMSNorm Contract input_cotangent exceeds its independent binary64 threshold")

    output = _RecordingOutput()
    monkeypatch.setattr(worker, "run_matrix", fail_after_report)
    monkeypatch.setattr(sys, "stdout", output)

    with pytest.raises(RuntimeError, match="input_cotangent exceeds"):
        worker.main(["--output", str(tmp_path / "result.json")])

    line = output.getvalue()
    assert line.startswith(worker.CELL_REPORT_PREFIX)
    assert json.loads(line.removeprefix(worker.CELL_REPORT_PREFIX)) == cell
    assert output.flush_count >= 1
    phase_log = tmp_path / "rmsnorm-contract.log"
    phase_log.write_text(line)

    assert prototype._sanitized_log_tail(phase_log) == line.rstrip("\n")
