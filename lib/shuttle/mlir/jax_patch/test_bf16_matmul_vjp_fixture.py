# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


def test_bf16_matmul_vjp_fixture_is_exact_ordinary_jax() -> None:
    fixture_directory = Path(__file__).parents[1] / "test" / "Inputs"
    generator = fixture_directory / "regenerate-jax-bf16-matmul-vjp-fixture.py"
    subprocess.run(
        [sys.executable, str(generator), "--output-dir", str(fixture_directory)],
        check=True,
    )
