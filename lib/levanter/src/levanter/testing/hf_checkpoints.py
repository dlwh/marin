# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local HF Llama sources for training and checkpoint tests."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import safetensors.numpy
from haliax import Axis
from haliax.state_dict import to_torch_compatible_state_dict
from jaxtyping import PRNGKeyArray
from transformers import LlamaConfig as HfLlamaConfig

from levanter.models.llama_config import LlamaConfig
from levanter.testing.helpers import use_test_mesh


@dataclass(frozen=True)
class LocalLlamaCheckpoint:
    path: Path
    config: LlamaConfig
    weights: dict[str, np.ndarray]


def write_llama_checkpoint(
    source: Path, config: LlamaConfig, hf_config: HfLlamaConfig, *, key: PRNGKeyArray
) -> LocalLlamaCheckpoint:
    """Write a real local HF checkpoint and retain its exact source weights."""
    source.mkdir()
    with use_test_mesh():
        model = config.build(Axis("vocab", hf_config.vocab_size), key=key)
        weights = to_torch_compatible_state_dict(model)
    hf_config.save_pretrained(source)
    safetensors.numpy.save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    return LocalLlamaCheckpoint(source, config, weights)
