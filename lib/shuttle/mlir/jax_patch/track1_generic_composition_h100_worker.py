# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for the anonymous generic composition H100 canary."""

import argparse
from pathlib import Path

from track1_generic_composition_h100_subject import result_document, run_matrix
from track1_map_h100_prototype import canonical_json, validate_generic_composition_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise RuntimeError("generic composition worker output must not exist")
    document = result_document(run_matrix())
    validate_generic_composition_result(document)
    with arguments.output.open("xb") as stream:
        stream.write(canonical_json(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
