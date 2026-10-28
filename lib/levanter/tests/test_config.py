# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import io
import os
import shutil
import subprocess
import sys
import textwrap

import draccus
import pytest
from jax.sharding import PartitionSpec
from rigging.filesystem.storage_path import StoragePath

import levanter.config
from levanter.data.text.datasets import LmDataConfig, UrlDatasetSourceConfig


def test_main_wrapper_loads_from_fsspec():
    StoragePath("memory://test.yaml").write_text(
        """
        project: test
        """
    )

    args = ["--config_path", "memory://test.yaml", "--x", "2"]

    @dataclasses.dataclass
    class Config:
        project: str
        x: int = 1

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.project == "test"
        assert config.x == 2

    main()


def test_remote_config_temp_file_handle_is_closed():
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        pytest.skip("/proc/self/fd is required to inspect open file descriptors")

    StoragePath("memory://test_fd.yaml").write_text("project: test\n")

    config_path, remaining_args = levanter.config._maybe_get_config_path_and_cmdline_args(
        ["--config_path", "memory://test_fd.yaml"]
    )

    assert remaining_args == []
    assert os.path.exists(config_path)
    assert _open_file_descriptors_for_path(config_path) == []


def test_lm_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LmDataConfig = dataclasses.field(default_factory=LmDataConfig)

    yaml_config = """
    data:
        tokenizer: gpt2
        cache_dir: "gs://levanter-data/tokenized"
        components:
          wikitext:
            source:
              type: url
              validation_urls:
                - "gs://example"
              train_urls:
                - "gs://example"
            cache_dir: "gs://levanter-data/tokenized/wikitext"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert "wikitext" in config.data.components
        comp = config.data.components["wikitext"]
        assert isinstance(comp.source, UrlDatasetSourceConfig)
        assert comp.cache_dir == "gs://levanter-data/tokenized/wikitext"

    main()


def test_lm_mixture_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LmDataConfig = dataclasses.field(default_factory=LmDataConfig)

    yaml_config = """
    data:
        components:
            owt:
                source:
                    type: url
                    train_urls:
                        - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
                    validation_urls:
                        - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
            wikitext:
                source:
                    type: hf
                    id: dlwh/wikitext_103_detokenized
        train_weights:
            owt: 0.6
            wikitext: 0.4
        tokenizer: gpt2
        cache_dir: "gs://levanter-data/tokenized/mixture"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data is not None
        # TODO: assert more things

    main()


@pytest.mark.parametrize(
    "spec",
    [
        PartitionSpec(("replica_dcn", "data")),  # Grug's batch spec
        PartitionSpec("model", ("replica_dcn", "data")),
        PartitionSpec(None, "model"),
        PartitionSpec(),
        PartitionSpec("data", PartitionSpec.UNCONSTRAINED),
    ],
)
def test_partition_spec_codec_roundtrips(spec):
    assert draccus.decode(PartitionSpec, draccus.encode(spec)) == spec


def test_config_with_partition_spec_dumps_and_reloads():
    """log_configuration dumps configs to YAML via draccus; a PartitionSpec field
    used to raise "No parser for object P(...)" and drop the whole config artifact.
    The registered codec makes it dump and round-trip cleanly."""

    @dataclasses.dataclass
    class Config:
        spec: PartitionSpec = dataclasses.field(default_factory=lambda: PartitionSpec(("replica_dcn", "data")))
        other: PartitionSpec = dataclasses.field(default_factory=lambda: PartitionSpec("model", None))

    cfg = Config()
    reloaded = draccus.load(Config, io.StringIO(draccus.dump(cfg)))
    assert reloaded.spec == cfg.spec
    assert reloaded.other == cfg.other


def _write_yaml_to_memory(yaml: str, path: str = "memory://test.yaml"):
    StoragePath(path).write_text(yaml)
    return path


def _open_file_descriptors_for_path(path: str):
    open_fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            fd_path = os.readlink(os.path.join("/proc/self/fd", fd))
        except FileNotFoundError:
            continue
        if fd_path == path:
            open_fds.append(fd)
    return open_fds


@pytest.mark.parametrize("first_import", ["native-first", "config-first"])
def test_llama_local_hf_selection_and_backend_roundtrip(
    first_import, tiny_hf_tokenizer_path, tmp_path, record_property
):
    """Local HF inference stays named while serialized selectors retain their backends."""
    source = tmp_path / "local-hf"
    shutil.copytree(tiny_hf_tokenizer_path, source)
    io_report = tmp_path / "tokenizer-io.json"
    script = textwrap.dedent(
        r"""
        import json
        import sys
        from pathlib import Path
        from unittest.mock import patch

        import draccus
        from transformers import LlamaConfig as HfLlamaConfig

        first_import, directory, io_destination = sys.argv[1:]
        if first_import == "native-first":
            from levanter.models.llama_native_lm import NativeLlamaLmConfig, NativeLlamaLmHeadModel
            from levanter.models.llama_config import LlamaConfig
        else:
            from levanter.models.llama_config import LlamaConfig
            from levanter.models.llama_native_lm import NativeLlamaLmConfig, NativeLlamaLmHeadModel
        from levanter.compat import hf_checkpoints
        from levanter.models.lm_model import LmConfig

        source = Path(directory)
        HfLlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        ).save_pretrained(source)
        real_load_tokenizer = hf_checkpoints.load_tokenizer
        requests = []
        unexpected_refusals = []
        def local_tokenizer_io(model_name_or_path, *args, **kwargs):
            reference = str(model_name_or_path)
            revision = kwargs.get("revision")
            if Path(reference).is_dir():
                requests.append({"reference": reference, "revision": revision, "status": "LOCAL"})
                return real_load_tokenizer(model_name_or_path, *args, **kwargs)
            if reference == "NousResearch/Llama-2-7b-hf" and revision is None:
                requests.append(
                    {"reference": reference, "revision": revision, "status": "REDIRECT", "local": str(source)}
                )
                return real_load_tokenizer(str(source), *args, **kwargs)
            requests.append({"reference": reference, "revision": revision, "status": "REFUSED"})
            unexpected_refusals.append({"reference": reference, "revision": revision})
            raise OSError("Nonlocal tokenizer I/O refused by the local HF selection test")

        try:
            with patch.object(hf_checkpoints, "load_tokenizer", local_tokenizer_io):
                converter = hf_checkpoints.HFCheckpointConverter.from_hf(str(source))
            if unexpected_refusals:
                raise AssertionError(f"Unexpected tokenizer requests were refused: {unexpected_refusals}")
            assert converter.LevConfigClass is LlamaConfig
            named = draccus.decode(LmConfig, draccus.encode(converter.default_config, LmConfig))
            native = draccus.decode(
                LmConfig,
                draccus.encode(NativeLlamaLmConfig.from_hf_config(HfLlamaConfig.from_pretrained(source)), LmConfig),
            )
            from levanter.models.llama import LlamaLMHeadModel

            assert named.model_type is LlamaLMHeadModel
            assert native.model_type is NativeLlamaLmHeadModel
        finally:
            Path(io_destination).write_text(
                json.dumps({"requests": requests, "unexpected_refusals": unexpected_refusals}, indent=2) + "\n"
            )
        """
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, first_import, str(source), str(io_report)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=45,
    )
    if io_report.exists():
        record_property("tokenizer_io", io_report.read_text())
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
