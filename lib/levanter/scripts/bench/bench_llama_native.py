# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched legacy/native Llama training benchmark on one fixed device mesh.

Both variants execute the existing Trainer's full AdamW step. Parameters are
initialized once by legacy Llama and imported through their HF tensor names;
no model or tokenizer download is performed. Run ``--mode compare`` to enforce
numerical and placement parity before publishing a speed ratio.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib
import jmp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import haliax as hax
from haliax._src.state_dict import flatten_modules_for_export, to_state_dict

from levanter.callbacks import JitCallback
from levanter.data.text.examples import GrugLmExample
from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding
from levanter.models.lm_model import LmExample
from levanter.tracker import NoopTracker, current_tracker
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.trainer_state import InsideJitInfo, TrainerState
from levanter.utils.mesh import MeshConfig


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    vocab_size: int = 128
    hidden_dim: int = 64
    intermediate_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    batch_size: int = 2
    seq_len: int = 32
    data_size: int = 1
    model_size: int = 1
    compute_dtype: str = "float32"
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    seed: int = 42
    warmup_steps: int = 5
    measured_steps: int = 20
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-4


class _CaptureStep(JitCallback):
    def inside_step(self, state: TrainerState, inside_info: InsideJitInfo):
        return inside_info.grads, inside_info.updates

    def on_step(self, step_info, cb_info):
        return None


def _legacy_config(config: BenchmarkConfig) -> LlamaConfig:
    return LlamaConfig(
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_seq_len=config.seq_len,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,
        use_bias=False,
        attn_backend=AttentionBackend.VANILLA,
        upcast_attn=False,
        scan_layers=True,
        gradient_checkpointing=config.gradient_checkpointing,
    )


def _native_config(config: BenchmarkConfig) -> NativeLlamaConfig:
    return NativeLlamaConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_seq_len=config.seq_len,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        use_bias=False,
        upcast_attn=False,
        rope_theta=10000.0,
        attention_implementation="reference",
        gradient_checkpointing=config.gradient_checkpointing,
    )


def _parameter_mapping() -> dict[str, str]:
    return {"embed": "data", "mlp": "model", "kv_head": "model", "heads": "model", "vocab": "model"}


def _compute_mapping() -> dict[str, str]:
    return {"batch": "data", "mlp": "model", "kv_head": "model", "heads": "model", "vocab": "model"}


@eqx.filter_jit
def _canonical_parameters(model) -> dict[str, jax.Array]:
    if isinstance(model, NativeLlamaModel):
        return model.to_state_dict()
    return to_state_dict(flatten_modules_for_export(model))


def _array_manifest(arrays: dict[str, jax.Array]) -> dict[str, dict[str, Any]]:
    """Record physical placement, including axis types and exact device assignment."""
    result = {}
    for name, array in sorted(arrays.items()):
        sharding = array.sharding
        if not isinstance(sharding, NamedSharding):
            result[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sharding_type": type(sharding).__name__,
                "devices": sorted(
                    f"{device.platform}:{device.process_index}:{device.id}" for device in sharding.device_set
                ),
                "memory_kind": sharding.memory_kind,
            }
            continue
        spec = tuple(sharding.spec) + (None,) * (array.ndim - len(sharding.spec))
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "spec": [
                (
                    list(entry)
                    if isinstance(entry, tuple) and len(entry) != 1
                    else entry[0] if isinstance(entry, tuple) else entry
                )
                for entry in spec
            ],
            "mesh": dict(sharding.mesh.shape),
            "axis_types": [axis_type.name for axis_type in sharding.mesh.axis_types],
            "devices": [
                f"{device.platform}:{device.process_index}:{device.id}" for device in sharding.mesh.devices.flat
            ],
            "memory_kind": sharding.memory_kind,
        }
    return result


def _tree_manifest(tree) -> dict[str, dict[str, Any]]:
    return _array_manifest(
        {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
            if eqx.is_array(leaf)
        }
    )


def _physical_layout(manifest: dict) -> dict:
    result = {key: value for key, value in manifest.items() if key != "axis_types"}
    if "spec" in manifest:
        # Auto-mode compilation may erase size-one axes; they cannot partition
        # physical data. Preserve exact declared specs in the report itself.
        def nontrivial(entry):
            axes = entry if isinstance(entry, list) else [entry]
            axes = [axis for axis in axes if axis is not None and manifest["mesh"][axis] > 1]
            return axes[0] if len(axes) == 1 else axes or None

        result["spec"] = [nontrivial(entry) for entry in manifest["spec"]]
    return result


def _assert_matching_manifests(legacy: dict, native: dict, *, label: str) -> None:
    if legacy.keys() != native.keys():
        raise ValueError(f"{label} tensor names differ: {sorted(legacy.keys() ^ native.keys())}")
    for name in legacy:
        # Auto -> Explicit is the intended control-mode change. Physical tensor
        # partitions, topology, dtype and memory kind must still be identical.
        if _physical_layout(legacy[name]) != _physical_layout(native[name]):
            raise ValueError(f"{label} placement differs for {name}: {legacy[name]} != {native[name]}")


def _numerical_comparison(legacy: dict[str, jax.Array], native: dict[str, jax.Array], config: BenchmarkConfig) -> dict:
    if legacy.keys() != native.keys():
        raise ValueError("Numerical comparison requires identical tensor names")
    tensors = {}
    for name in sorted(legacy):
        left = np.asarray(legacy[name], dtype=np.float32)
        right = np.asarray(native[name], dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError(f"Tensor shape mismatch for {name}: {left.shape} != {right.shape}")
        difference = np.abs(left - right)
        tensors[name] = {
            "max_abs": float(difference.max(initial=0)),
            "mean_abs": float(difference.mean()),
            "relative_l2": float(np.linalg.norm(left - right) / max(float(np.linalg.norm(left)), 1e-30)),
            "allclose": bool(np.allclose(left, right, atol=config.parity_atol, rtol=config.parity_rtol)),
        }
    return {
        "allclose": all(tensor["allclose"] for tensor in tensors.values()),
        "max_abs": max((tensor["max_abs"] for tensor in tensors.values()), default=0),
        "tensors": tensors,
    }


def _tensor_digest(tensors: dict[str, jax.Array]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        array = np.asarray(value)
        digest.update(json.dumps([name, list(array.shape), str(array.dtype)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _timing_summary(samples: list[float], tokens_per_step: int) -> dict:
    if not samples or any(sample <= 0 for sample in samples):
        raise ValueError("Step durations must be positive, synchronized measurements")
    elapsed = sum(samples)
    return {
        "step_seconds": samples,
        "mean_step_seconds": statistics.mean(samples),
        "median_step_seconds": statistics.median(samples),
        "minimum_step_seconds": min(samples),
        "p90_step_seconds": float(np.percentile(samples, 90)),
        "elapsed_seconds": elapsed,
        "steps_per_second": len(samples) / elapsed,
        "tokens_per_second": len(samples) * tokens_per_step / elapsed,
        "tokens_per_step": tokens_per_step,
    }


def _shared_loss(hidden, head, tokens, loss_weight):
    labels = jnp.roll(tokens, -1, axis=-1)
    mesh = jax.sharding.get_abstract_mesh()
    specs = (P("data", None, None), P(None, None), P("data", None), P("data", None))
    place = (
        jax.sharding.reshard
        if all(axis_type is AxisType.Explicit for axis_type in mesh.axis_types)
        else jax.lax.with_sharding_constraint
    )
    arguments = tuple(
        place(array, spec) for array, spec in zip((hidden, head, labels, loss_weight), specs, strict=True)
    )

    def shard_loss(local_hidden, local_head, local_labels, local_weights):
        local_sum = fused_cross_entropy_loss_and_logsumexp_penalty(
            local_hidden.reshape((-1, local_hidden.shape[-1])),
            local_labels.reshape(-1),
            local_head.T,
            weight=local_weights.reshape(-1),
            reduction="sum",
            dtype=jnp.float32,
            implementation="xla",
        )
        total_sum = jax.lax.psum(local_sum, "data")
        total_weight = jax.lax.psum(jnp.sum(local_weights), "data")
        return total_sum / total_weight

    return jax.shard_map(shard_loss, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False)(*arguments)


def _legacy_loss(model, batch: LmExample, *, key):
    hidden = model.activations(batch.tokens, batch.attn_mask, key=key)
    hidden = hidden.rearrange(("batch", "position", "embed")).array
    head = model.get_lm_head().rearrange(("vocab", "embed")).array
    return _shared_loss(hidden, head, batch.tokens.array, batch.loss_weight.array)


def _native_loss(model, batch: GrugLmExample, *, key):
    hidden = model.activations(batch.tokens, key=key)
    return _shared_loss(hidden, model.get_lm_head(), batch.tokens, batch.loss_weight)


def _copy_arrays(tree):
    return jax.tree.map(lambda leaf: leaf.copy() if eqx.is_array(leaf) else leaf, tree)


def _compiled_memory(compiled) -> dict[str, int] | None:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    return {
        name: int(getattr(analysis, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
            "host_argument_size_in_bytes",
            "host_output_size_in_bytes",
            "host_alias_size_in_bytes",
            "host_temp_size_in_bytes",
        )
    }


def _activation_manifest(kind: str, trainer: Trainer, state: TrainerState, batch) -> dict:
    @eqx.filter_jit
    def activations(model, batch):
        model = trainer.mp.cast_to_compute(model)
        if kind == "legacy":
            hidden = model.activations(batch.tokens, batch.attn_mask)
            hidden = hidden.rearrange(("batch", "position", "embed")).array
            head = model.get_lm_head().rearrange(("vocab", "embed")).array
            tokens = batch.tokens.array
        else:
            hidden = model.activations(batch.tokens)
            head = model.get_lm_head()
            tokens = batch.tokens
        return {"final_hidden": hidden, "lm_head_before_common_ce": head, "tokens": tokens}

    with hax.axis_mapping(_compute_mapping()):
        arrays = activations(state.model, batch)
    jax.block_until_ready(arrays)
    return _array_manifest(arrays)


def _run_variant(
    kind: str, config: BenchmarkConfig, mesh: Mesh, model, batch, output_dir: Path | None, dump_hlo: bool
) -> tuple[dict, dict]:
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.dtype(config.compute_dtype), output_dtype=jnp.float32)
    trainer_config = TrainerConfig(
        id=f"llama-native-bench-{kind}",
        mp=mp,
        train_batch_size=config.batch_size,
        num_train_steps=1 + config.warmup_steps + config.measured_steps,
        mesh=MeshConfig(
            axes={"data": config.data_size, "model": config.model_size},
            compute_mapping=_compute_mapping(),
            param_mapping=_parameter_mapping(),
        ),
        use_explicit_mesh_axes=kind == "native",
        load_checkpoint=False,
        log_jaxprs=False,
        log_xla_hlo=False,
        tracker=(),
    )
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    trainer = Trainer(
        trainer_config,
        optimizer,
        _legacy_loss if kind == "legacy" else _native_loss,
        add_default_hooks=False,
        explicit_layout=ExplicitTrainingLayout(mesh, P("data", None)) if kind == "native" else None,
    )
    start = time.perf_counter()
    state = trainer.initial_state(jax.random.PRNGKey(config.seed + 1), model=_copy_arrays(model))
    jax.block_until_ready(state)
    state_init_seconds = time.perf_counter() - start
    model_manifest = _array_manifest(_canonical_parameters(state.model))
    raw_model_manifest = _tree_manifest(state.model)
    optimizer_manifest = _tree_manifest(state.opt_state)
    activations_manifest = _activation_manifest(kind, trainer, state, batch)

    # Capture real gradients and updates through Trainer's existing JIT hook. This
    # diagnostic executable and its extra outputs are never used for timing.
    trainer.add_hook(_CaptureStep(), every=1)
    diagnostic_state = _copy_arrays(state)
    diagnostic = trainer._jit_train_step_fn(diagnostic_state, (batch,), {})
    jax.block_until_ready(diagnostic)
    gradients, updates = diagnostic.hook_infos[0]
    diagnostic_data = {
        "loss": {"loss": diagnostic.loss},
        "gradients": _canonical_parameters(gradients),
        "updates": _canonical_parameters(updates),
        "parameters_after_one_step": _canonical_parameters(diagnostic.new_state.model),
        "adam_mu": _canonical_parameters(diagnostic.new_state.opt_state[0].mu),
        "adam_nu": _canonical_parameters(diagnostic.new_state.opt_state[0].nu),
        "adam_count": {"count": diagnostic.new_state.opt_state[0].count},
    }
    # Materialize parity outputs on the host so they do not pollute live device
    # memory while measuring subsequent full training steps.
    diagnostics = {label: jax.tree.map(np.asarray, tensors) for label, tensors in diagnostic_data.items()}
    diagnostic_manifests = {
        "gradients": _array_manifest(diagnostic_data["gradients"]),
        "gradients_raw": _tree_manifest(gradients),
        "updates_raw": _tree_manifest(updates),
        "updates": _array_manifest(diagnostic_data["updates"]),
        "parameters_after_one_step": _array_manifest(diagnostic_data["parameters_after_one_step"]),
        "adam_mu": _array_manifest(diagnostic_data["adam_mu"]),
        "adam_nu": _array_manifest(diagnostic_data["adam_nu"]),
        "adam_count": _array_manifest(diagnostic_data["adam_count"]),
        "optimizer_state_after_one_step": _tree_manifest(diagnostic.new_state.opt_state),
    }
    del diagnostic_state, diagnostic, diagnostic_data, gradients, updates

    step_fn = trainer._jit_train_step_fn_no_hook
    start = time.perf_counter()
    lowered = step_fn.lower(state, (batch,), {})
    lowering_seconds = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    memory = _compiled_memory(compiled)
    if dump_hlo:
        assert output_dir is not None
        (output_dir / f"{kind}.stablehlo.txt").write_text(lowered.as_text())
        (output_dir / f"{kind}.optimized_hlo.txt").write_text(compiled.as_text())

    # Calling the production wrapper preserves its argument/static processing.
    # Record the first dispatch separately even if JAX reuses lowering's cache.
    start = time.perf_counter()
    result = step_fn(state, (batch,), {})
    jax.block_until_ready(result)
    first_execution_seconds = time.perf_counter() - start
    state = result.new_state
    del result
    for _ in range(config.warmup_steps):
        result = step_fn(state, (batch,), {})
        jax.block_until_ready(result)
        state = result.new_state
    samples = []
    for _ in range(config.measured_steps):
        jax.block_until_ready(state)
        start = time.perf_counter()
        result = step_fn(state, (batch,), {})
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
        state = result.new_state
    result_report = {
        "kind": kind,
        "state_init_seconds": state_init_seconds,
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "compiled_memory_analysis": memory,
        "timing": _timing_summary(samples, config.batch_size * config.seq_len),
        "final_loss": float(result.loss),
        "final_step": int(state.step),
        "layouts": {
            "parameters": model_manifest,
            "parameters_raw": raw_model_manifest,
            "optimizer_state_initial": optimizer_manifest,
            "activations": activations_manifest,
            **diagnostic_manifests,
        },
    }
    return result_report, diagnostics


def _provenance(config: BenchmarkConfig, mesh: Mesh, source_commit: str | None, source_tree_state: str) -> dict:
    if source_commit is None:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        source_tree_state = "dirty" if dirty else "clean"
    return {
        "config": dataclasses.asdict(config),
        "git_commit": source_commit,
        "source_tree_state": source_tree_state,
        "architecture": dataclasses.asdict(_native_config(config)),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "equinox": eqx.__version__,
        "optax": optax.__version__,
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "devices": [str(device) for device in mesh.devices.flat],
        "device_kinds": [device.device_kind for device in mesh.devices.flat],
        "mesh": dict(mesh.shape),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "attention_backend": "vanilla",
        "attention_entrypoints": {
            "legacy": "AttentionBackend.VANILLA",
            "native": "reference_attention(logits_dtype=None)",
        },
        "numpy": np.__version__,
        "loss_backend": "shared_raw_fused_cross_entropy_loss_and_logsumexp_penalty:xla",
        "loss_head_placement": "replicated in common loss adapter before the raw CE kernel",
        "parameter_dtype": "float32",
        "optimizer": "optax.adamw",
        "optimizer_moment_dtype": "float32",
        "scan_layers": True,
        "donation": "Trainer production step: state donated, batch retained",
        "timed_scope": "Trainer full compiled step, including gradients and AdamW; synchronized wall time; no hooks/data/I/O",
    }


def run_benchmark(
    config: BenchmarkConfig,
    mode: str = "compare",
    *,
    output_dir: Path | None = None,
    dump_hlo: bool = False,
    source_bundle_marker: str | None = None,
    source_commit: str | None = None,
    source_tree_state: str = "unknown",
) -> dict:
    if dump_hlo and output_dir is None:
        raise ValueError("HLO capture requires an output directory")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if jax.process_count() != 1:
        raise ValueError("This benchmark requires one host so both variants use identical local hardware")
    if config.warmup_steps < 1 or config.measured_steps < 1:
        raise ValueError("At least one warmup and measured step are required")
    if config.compute_dtype not in ("float32", "bfloat16"):
        raise ValueError("Use float32 or bfloat16 compute; parameters and AdamW moments remain float32")
    device_count = config.data_size * config.model_size
    if device_count > jax.local_device_count():
        raise ValueError(f"Requested {device_count} devices but only {jax.local_device_count()} are local")
    if config.batch_size % config.data_size or config.num_kv_heads % config.model_size:
        raise ValueError("Batch and KV heads must be divisible by their data/model mesh axes")
    devices = np.array(jax.local_devices()[:device_count]).reshape(config.data_size, config.model_size)
    legacy_mesh = Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
    native_mesh = Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    report = {
        "schema": "llama-native-training-benchmark.v1",
        "mode": mode,
        "provenance": _provenance(config, legacy_mesh, source_commit, source_tree_state),
    }
    report["provenance"]["source_bundle_marker"] = source_bundle_marker
    report["provenance"]["mesh_axis_types"] = {"legacy": ["Auto", "Auto"], "native": ["Explicit", "Explicit"]}
    report["provenance"][
        "layout_comparison"
    ] = "Identical physical partitions and devices; Auto -> Explicit is the intended control-mode difference"
    variants = {}
    diagnostics = {}
    tokens_np = np.random.default_rng(config.seed + 2).integers(
        0, config.vocab_size, (config.batch_size, config.seq_len), dtype=np.int32
    )
    weights_np = np.ones_like(tokens_np, dtype=np.float32)
    weights_np[:, -1] = 0
    report["provenance"]["batch_sha256"] = _tensor_digest({"tokens": tokens_np, "loss_weight": weights_np})
    with current_tracker(NoopTracker()), jax.set_mesh(legacy_mesh), hax.axis_mapping(_parameter_mapping()):
        start = time.perf_counter()
        legacy = hax.named_jit(_legacy_config(config).build, axis_resources=_parameter_mapping())(
            hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(config.seed)
        )
        jax.block_until_ready(legacy)
        parameters = _canonical_parameters(legacy)
        jax.block_until_ready(parameters)
        report["legacy_parameter_init_seconds"] = time.perf_counter() - start
        report["provenance"]["parameter_sha256"] = _tensor_digest(parameters)
        report["provenance"]["parameter_count"] = sum(array.size for array in parameters.values())
        source_manifest = _array_manifest(parameters)
        source_state = jax.tree.map(np.asarray, parameters)
        sharding = NamedSharding(legacy_mesh, P("data", None))
        tokens = jax.device_put(tokens_np, sharding)
        weights = jax.device_put(weights_np, sharding)
        named_batch = LmExample(
            tokens=hax.named(tokens, (hax.Axis("batch", config.batch_size), hax.Axis("position", config.seq_len))),
            loss_weight=hax.named(
                weights, (hax.Axis("batch", config.batch_size), hax.Axis("position", config.seq_len))
            ),
            attn_mask=AttentionMask.causal(),
        )
        if mode in ("legacy", "compare"):
            variants["legacy"], diagnostics["legacy"] = _run_variant(
                "legacy", config, legacy_mesh, legacy, named_batch, output_dir, dump_hlo
            )
    del legacy, parameters, named_batch, tokens, weights
    if mode in ("native", "compare"):
        with current_tracker(NoopTracker()), jax.set_mesh(native_mesh):
            start = time.perf_counter()
            native = NativeLlamaModel.init(
                _native_config(config),
                key=jax.random.PRNGKey(config.seed),
                sharding=NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model"),
            )
            native = native.from_state_dict(source_state)
            jax.block_until_ready(native)
            report["native_parameter_import_seconds"] = time.perf_counter() - start
            _assert_matching_manifests(
                source_manifest, _array_manifest(_canonical_parameters(native)), label="initial parameters"
            )
            native_parameters = _canonical_parameters(native)
            report["provenance"]["native_parameter_sha256"] = _tensor_digest(native_parameters)
            if report["provenance"]["parameter_sha256"] != report["provenance"]["native_parameter_sha256"]:
                raise ValueError("Native parameter import changed the exact fixed initial tensors")
            initial_parity = _numerical_comparison(source_state, native_parameters, config)
            report["initial_parameter_parity"] = initial_parity
            sharding = NamedSharding(native_mesh, P("data", None))
            raw_batch = GrugLmExample(
                tokens=jax.device_put(tokens_np, sharding), loss_weight=jax.device_put(weights_np, sharding)
            )
            variants["native"], diagnostics["native"] = _run_variant(
                "native", config, native_mesh, native, raw_batch, output_dir, dump_hlo
            )
    report["variants"] = variants
    if mode == "compare":
        parity = {}
        for label in diagnostics["legacy"]:
            parity[label] = _numerical_comparison(diagnostics["legacy"][label], diagnostics["native"][label], config)
        for label in (
            "parameters",
            "gradients",
            "updates",
            "parameters_after_one_step",
            "adam_mu",
            "adam_nu",
            "adam_count",
            "activations",
        ):
            _assert_matching_manifests(
                variants["legacy"]["layouts"][label], variants["native"]["layouts"][label], label=label
            )
        report["parity"] = parity
        report["comparison_valid"] = all(comparison["allclose"] for comparison in parity.values())
        report["native_speedup"] = (
            variants["legacy"]["timing"]["mean_step_seconds"] / variants["native"]["timing"]["mean_step_seconds"]
            if report["comparison_valid"]
            else None
        )
        report["native_median_speedup"] = (
            variants["legacy"]["timing"]["median_step_seconds"] / variants["native"]["timing"]["median_step_seconds"]
            if report["comparison_valid"]
            else None
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "native", "compare"), default="compare")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/llama-native-benchmark"))
    )
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--source-bundle-marker", default=os.environ.get("SOURCE_BUNDLE_MARKER"))
    parser.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT"))
    parser.add_argument(
        "--source-tree-state",
        choices=("clean", "dirty", "unknown"),
        default=os.environ.get("SOURCE_TREE_STATE", "unknown"),
    )
    for field in dataclasses.fields(BenchmarkConfig):
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(
                f"--{field.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(default), default=default)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    output_dir = args.pop("output_dir")
    dump_hlo = args.pop("dump_hlo")
    source_bundle_marker = args.pop("source_bundle_marker")
    source_commit = args.pop("source_commit")
    source_tree_state = args.pop("source_tree_state")
    report = run_benchmark(
        BenchmarkConfig(**args),
        mode,
        output_dir=output_dir,
        dump_hlo=dump_hlo,
        source_bundle_marker=source_bundle_marker,
        source_commit=source_commit,
        source_tree_state=source_tree_state,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    output = output_dir / f"{mode}.json"
    output.write_text(serialized + "\n")
    summary = {
        "mode": mode,
        "report": str(output),
        "comparison_valid": report.get("comparison_valid"),
        "native_speedup": report.get("native_speedup"),
        "native_median_speedup": report.get("native_median_speedup"),
        "mean_step_seconds": {
            name: variant["timing"]["mean_step_seconds"] for name, variant in report["variants"].items()
        },
        "median_step_seconds": {
            name: variant["timing"]["median_step_seconds"] for name, variant in report["variants"].items()
        },
    }
    print("BENCH_SUMMARY " + json.dumps(summary, sort_keys=True))
    if mode == "compare" and not report["comparison_valid"]:
        raise SystemExit("Numerical parity failed; no speed ratio is valid")


if __name__ == "__main__":
    main()
