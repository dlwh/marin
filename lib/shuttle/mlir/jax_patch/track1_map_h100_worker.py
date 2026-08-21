# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for Track 1 Map populate and reuse roles."""

import argparse
import json
from pathlib import Path

from track1_map_h100_prototype import canonical_json, validate_worker_result
from track1_map_h100_subject import Coordinate, Mode, result_document, run_matrix


def _populate_attribution(path: Path) -> dict[Coordinate, tuple[str, int, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("populate result must be one regular file")
    raw = path.read_bytes()
    document = json.loads(raw)
    if type(document) is not dict or raw != canonical_json(document) or document.get("worker") != "populate":
        raise ValueError("populate result must be one canonical populate document")
    validated = validate_worker_result(document, "populate")
    coordinates = validated["coordinates"]
    assert type(coordinates) is list
    attribution = {}
    for item in coordinates:
        if type(item) is not dict or set(item) != {
            "cache_entry",
            "gradient_sha256",
            "mode",
            "observer",
            "primal_sha256",
            "rank",
        }:
            raise ValueError("populate result coordinate has the wrong schema")
        entry = item["cache_entry"]
        if type(entry) is not list or len(entry) != 3:
            raise ValueError("populate result cache entry has the wrong schema")
        attribution[Coordinate(int(item["rank"]), Mode(item["mode"]))] = (str(entry[0]), int(entry[1]), str(entry[2]))
    if len(attribution) != 6:
        raise ValueError("populate result coordinates are not unique")
    return attribution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("populate", "reuse"), required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--populate-result", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise RuntimeError("worker output path must not exist")
    if arguments.role == "populate":
        if arguments.populate_result is not None:
            raise ValueError("populate worker cannot consume a populate result")
        expected = None
    else:
        if arguments.populate_result is None:
            raise ValueError("reuse worker requires the populate result")
        expected = _populate_attribution(arguments.populate_result)
    results = run_matrix(
        arguments.cache_directory,
        reuse=arguments.role == "reuse",
        expected_attribution=expected,
    )
    document = result_document(results, reuse=arguments.role == "reuse")
    arguments.output.write_bytes(canonical_json(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
