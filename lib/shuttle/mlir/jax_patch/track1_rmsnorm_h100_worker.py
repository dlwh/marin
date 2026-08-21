# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for the Track 1 RMSNorm H100 canary."""

import argparse
from pathlib import Path

from track1_map_h100_prototype import canonical_json, validate_rmsnorm_result
from track1_rmsnorm_h100_subject import result_document, run_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise RuntimeError("RMSNorm worker output path must not exist")
    document = result_document(run_matrix())
    validate_rmsnorm_result(document)
    arguments.output.write_bytes(canonical_json(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
