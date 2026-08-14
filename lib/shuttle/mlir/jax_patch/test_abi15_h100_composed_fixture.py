# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


def test_composed_fixture_is_exact_ordinary_jax() -> None:
    fixture_directory = Path(__file__).parents[1] / "test" / "Inputs"
    generator = Path(__file__).with_name("abi15_h100_composed_fixture.py")
    subprocess.run(
        [sys.executable, str(generator), "--output-dir", str(fixture_directory)],
        check=True,
    )


def test_composed_fixture_closes_noncanonical_affine_shape() -> None:
    fixture = (
        Path(__file__).parents[1] / "test" / "Inputs" / "jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir"
    ).read_text()
    assert "tensor<33x2xbf16>" in fixture
    assert "tensor<2x34xbf16>" in fixture
    assert "tensor<33x34xbf16>" in fixture
    assert fixture.count("stablehlo.dot_general") == 3
    assert "contracting_dims = [0] x [0]" in fixture
    assert "contracting_dims = [1] x [1]" in fixture
