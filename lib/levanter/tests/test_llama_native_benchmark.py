# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "bench_llama_native.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_llama_native", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = BENCHMARK
SCRIPT_SPEC.loader.exec_module(BENCHMARK)


def test_matched_training_benchmark_preserves_parameters_updates_and_layouts(tmp_path):
    config = BENCHMARK.BenchmarkConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        batch_size=2,
        seq_len=8,
        warmup_steps=1,
        measured_steps=3,
    )

    report = BENCHMARK.run_benchmark(config, output_dir=tmp_path)

    assert report["comparison_valid"]
    assert report["initial_parameter_parity"]["allclose"]
    assert report["provenance"]["parameter_sha256"]
    assert report["provenance"]["batch_sha256"]
    assert all(result["allclose"] for result in report["parity"].values())
    assert report["parity"]["measured_step_losses"]["allclose"]
    legacy_losses = report["variants"]["legacy"]["measured_step_losses"]
    native_losses = report["variants"]["native"]["measured_step_losses"]
    np.testing.assert_allclose(legacy_losses, native_losses, atol=config.parity_atol, rtol=config.parity_rtol)
    assert legacy_losses[-1] < legacy_losses[0]
    for variant in report["variants"].values():
        assert len(variant["timing"]["step_seconds"]) == config.measured_steps
        assert len(variant["measured_step_losses"]) == config.measured_steps
        assert variant["final_loss"] == variant["measured_step_losses"][-1]
        assert variant["timing"]["tokens_per_step"] == 16
        assert variant["timing"]["elapsed_seconds"] > 0
        assert (
            variant["timing"]["tokens_per_second"] == config.measured_steps * 16 / variant["timing"]["elapsed_seconds"]
        )
        assert variant["layouts"]["parameters"]
        assert variant["layouts"]["updates"]
        assert variant["layouts"]["adam_mu"]
        assert variant["layouts"]["optimizer_state_after_one_step"]
        assert variant["layouts"]["activations"]["final_hidden"]["shape"] == [2, 8, 16]
        assert variant["compiled_memory_analysis"]["argument_size_in_bytes"] > 0
