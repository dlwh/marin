# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from jax.sharding import PartitionSpec as P

import levanter
import levanter.analysis
import levanter.config
import levanter.tracker
import levanter.callbacks
import levanter.eval
import levanter.eval_harness
from levanter import callbacks
from levanter.callbacks._iris_status import iris_status_reporter
from levanter.callbacks.labeled_eval import LabeledLmEvalConfig, add_labeled_lm_eval_callbacks
from levanter.adaptor import AdaptorConfig, AdaptorExportConfig, LoraAdaptorConfig, NoAdaptorConfig
from levanter.callbacks.tensorstore_callbacks import install_tensorstore_metrics_hook_if_enabled
from levanter.checkpoint import discover_checkpoint_candidates, latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, ModelWithHfSerializationMixin, build_generation_config
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.model_loading import native_evaluation_axis_mapping
from levanter.main.model_init import ModelInitContext, build_model, prepare_model_init_context
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native import NativeLlamaSharding
from levanter.models.llama_native_lm import NativeLlamaLmConfig, NativeLlamaLmHeadModel, native_llama_sharding
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, split_activations
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.trainer_state import trainables_only
from levanter.training_control import TrainingDashboard
from levanter.utils.jax_utils import parameter_count

logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: LmDataConfig = field(default_factory=LmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: bool | str = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint
    pad_tokenizer_to_match_model: bool = False
    """If True, pad the tokenizer's vocab to match the model's vocab size by adding dummy tokens.
    Useful when the model checkpoint has a larger vocab than the tokenizer (e.g., Qwen models
    pad their vocab to be divisible by 4 for TPU efficiency)."""

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    z_loss_weight: float = 0.0

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None
    hf_generation_eos_token_ids: Optional[list[int]] = None

    adapter: AdaptorConfig = field(default_factory=NoAdaptorConfig)
    peft_save_path: Optional[str] = None
    peft_hf_upload: Optional[str] = None
    merged_hf_save_path: Optional[str] = None
    merged_hf_upload: Optional[str] = None

    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer
    initialize_from_checkpoint_path: Optional[str] = None
    """
    If provided, will initialize from this checkpoint, used for llama style ablation. This resets the data loader.
    Note that this differs from --trainer.initialize_from, which does not reset the data loader.
    """
    initialize_model_from_checkpoint_path: Optional[str] = None
    """
    If provided, initialize only the model weights from the ``model`` subtree of this native Levanter
    checkpoint, leaving a fresh optimizer state and step 0 (the same init as ``initialize_from_hf``, from a
    native checkpoint). Unlike ``initialize_from_checkpoint_path`` (a full-state restore) it never reads the
    checkpoint's optimizer state or step. The load is strict: every model leaf must be present. ``config.model``
    must be the architecture the checkpoint was saved with.
    """
    eval_harness: Optional[LmEvalHarnessConfig] = None
    eval_harness_steps: int = 10000
    labeled_eval: LabeledLmEvalConfig | None = None

    # TODO: really need to add callback framework
    log_entropy: bool = False


def _restore_lm_model_from_partial_checkpoint(
    checkpointed_model: LmHeadModel,
    source_model: LmHeadModel,
    trainable_filter,
) -> LmHeadModel:
    checkpointed_trainables = trainables_only(checkpointed_model, trainable_filter)
    return eqx.combine(checkpointed_trainables, source_model)


def _native_llama_sharding(config: TrainerConfig) -> NativeLlamaSharding:
    return native_llama_sharding(config.parameter_axis_mapping, config.compute_axis_mapping, config.batch_axis_name)


def _load_lm_model_from_configured_source(
    *,
    config: TrainLmConfig,
    converter,
    Vocab: Axis,
    model_key,
    adapter_key,
    parameter_axis_mapping,
    trainer: Trainer,
    source_checkpoint_path: str | None = None,
) -> LmHeadModel:
    if config.initialize_from_hf:
        assert converter is not None
        model_init = None
        load_dtype = trainer.mp.compute_dtype
        if isinstance(config.model, NativeLlamaLmConfig):

            def initialize_native_model(vocab: Axis) -> ModelWithHfSerializationMixin:
                return cast(
                    ModelWithHfSerializationMixin,
                    build_model(
                        config.model,
                        vocab,
                        key=model_key,
                        parameter_axis_mapping=trainer.parameter_axis_mapping,
                        compute_axis_mapping=trainer.compute_axis_mapping,
                        batch_axis_name=trainer.config.batch_axis_name,
                    ),
                )

            model_init = initialize_native_model
            # Compute precision must not round pretrained master weights before initialization.
            load_dtype = trainer.mp.param_dtype
        model = converter.load_pretrained(
            config.model.model_type,
            config=(
                config.model
                if isinstance(config.model, NativeLlamaLmConfig) or not config.use_hf_model_config
                else None
            ),
            axis_mapping=parameter_axis_mapping,
            dtype=load_dtype,
            model_init=model_init,
        )
        model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
    elif (
        config.initialize_from_checkpoint_path is not None or config.initialize_model_from_checkpoint_path is not None
    ):
        # Both load the checkpoint's `model` subtree into an abstract template. They differ
        # only in how main() uses the loaded weights.
        source = config.initialize_from_checkpoint_path or config.initialize_model_from_checkpoint_path
        checkpoint_path = source_checkpoint_path or latest_checkpoint_path(source)
        model = eqx.filter_eval_shape(
            build_model,
            config.model,
            Vocab,
            key=model_key,
            parameter_axis_mapping=trainer.parameter_axis_mapping,
            compute_axis_mapping=trainer.compute_axis_mapping,
            batch_axis_name=trainer.config.batch_axis_name,
        )
        model = load_checkpoint(
            model, checkpoint_path, subpath="model", axis_mapping=parameter_axis_mapping, mesh=trainer.device_mesh
        )
        model = hax.shard(model, parameter_axis_mapping)
        model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
    else:
        model = build_model(
            config.model,
            Vocab,
            key=model_key,
            parameter_axis_mapping=trainer.parameter_axis_mapping,
            compute_axis_mapping=trainer.compute_axis_mapping,
            batch_axis_name=trainer.config.batch_axis_name,
        )

    if not isinstance(config.adapter, NoAdaptorConfig):
        model = config.adapter.apply(model, key=adapter_key, axis_mapping=parameter_axis_mapping)

    return model


def main(config: TrainLmConfig) -> callbacks.StepInfo:
    """Train a language model and return its final ordinary Trainer step."""
    native = isinstance(config.model, NativeLlamaLmConfig)
    native_model_context: ModelInitContext | None = None
    explicit_layout = None
    export_config = AdaptorExportConfig(
        hf_save_path=config.hf_save_path,
        hf_upload=config.hf_upload,
        hf_save_steps=config.hf_save_steps,
        hf_save_dtype=config.hf_save_dtype,
        peft_save_path=config.peft_save_path,
        peft_hf_upload=config.peft_hf_upload,
        merged_hf_save_path=config.merged_hf_save_path,
        merged_hf_upload=config.merged_hf_upload,
    )
    if native:
        if config.pad_tokenizer_to_match_model:
            raise ValueError("llama_native HF initialization requires an unchanged tokenizer")
        if config.initialize_from_hf and config.use_hf_model_config:
            native_model_context = prepare_model_init_context(
                config.model,
                tokenizer=config.data.the_tokenizer,
                initialize_from_hf=config.initialize_from_hf,
                use_hf_model_config=True,
            )
            config = dataclasses.replace(config, model=native_model_context.model)
        export_config.validate_native(config.model)
        if config.trainer.quantization is not None:
            raise ValueError("llama_native training does not support Haliax quantization")
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, use_explicit_mesh_axes=True))
        _native_llama_sharding(config.trainer)
    tokenizer = config.data.the_tokenizer
    source_vocab_size = None

    # The three weight-init sources are mutually exclusive: HF conversion, native full-state restore, and
    # native weights-only init.
    _init_sources = [
        config.initialize_from_hf,
        config.initialize_from_checkpoint_path is not None,
        config.initialize_model_from_checkpoint_path is not None,
    ]
    if sum(bool(s) for s in _init_sources) > 1:
        raise ValueError(
            "Specify at most one of initialize_from_hf, initialize_from_checkpoint_path, "
            "initialize_model_from_checkpoint_path."
        )
    # trainer.initialize_from is a full-state resume that restores step > 0, which would skip the
    # step-0 weights-only init below and silently use the resumed checkpoint instead. Reject the combo,
    # matching how initialize_from_hf rejects it.
    if config.initialize_model_from_checkpoint_path is not None and config.trainer.initialize_from is not None:
        raise ValueError("Cannot specify both initialize_model_from_checkpoint_path and trainer.initialize_from")

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf and not native:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        converter.warn_if_tokenizer_mismatch(tokenizer)
        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.pad_tokenizer_to_match_model:
            converter = converter.with_tokenizer_padded_to_match_model()

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig) and not native:
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
        if config.pad_tokenizer_to_match_model:
            converter = converter.with_tokenizer_padded_to_match_model()
    else:
        converter = native_model_context.converter if native_model_context is not None else None

    levanter.trainer.initialize(config)
    if native:
        sharding = _native_llama_sharding(config.trainer)
        mesh = config.trainer.device_mesh
        config.model.validate_sharding(sharding, mesh)
        explicit_layout = ExplicitTrainingLayout(mesh, P(sharding.batch, None))
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def loss_function(model: LmHeadModel, example: LmExample | GrugLmExample, *, key=None):
        if isinstance(example, GrugLmExample):
            example = named_lm_example_from_grug(
                example, model.Pos.resize(example.tokens.shape[-1]), batch_axis=config.trainer.batch_axis_name
            )
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=config.z_loss_weight)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with (
        Trainer(
            config.trainer,
            optimizer,
            loss_function,
            explicit_layout=explicit_layout,
            checkpoint_metadata=(
                (lambda model: cast(NativeLlamaLmHeadModel, model).checkpoint_metadata()) if native else None
            ),
        ) as trainer,
        TrainingDashboard(config, trainer.request_checkpoint, config.trainer.id or "unknown"),
    ):
        checkpoint = trainer.checkpoint_to_load
        resuming = checkpoint is not None
        source_checkpoint_path = None
        if isinstance(config.model, NativeLlamaLmConfig):
            if checkpoint is not None:
                source_vocab_size = config.model.checkpoint_vocab_size(checkpoint.metadata)
            else:
                source = config.initialize_from_checkpoint_path or config.initialize_model_from_checkpoint_path
                if source is not None:
                    candidates = discover_checkpoint_candidates(source)
                    if not candidates:
                        raise FileNotFoundError(f"No complete checkpoint found in {source}")
                    source_checkpoint = candidates[-1]
                    source_checkpoint_path = source_checkpoint.path
                    source_vocab_size = config.model.checkpoint_vocab_size(source_checkpoint.metadata)
            if config.initialize_from_hf and (
                source_vocab_size is None or isinstance(config.adapter, LoraAdaptorConfig)
            ):
                if config.trainer.initialize_from is not None:
                    raise ValueError("Cannot specify both initialize_from_hf and initialize_from")
                if converter is None:
                    converter = config.model.hf_checkpoint_converter(
                        config.initialize_from_hf if isinstance(config.initialize_from_hf, str) else None,
                        tokenizer=tokenizer,
                    )
                hf_config = converter.default_hf_config
                config.model.validate_hf_config(hf_config)
                if source_vocab_size is not None and source_vocab_size != hf_config.vocab_size:
                    raise ValueError("llama_native adapter source vocabulary does not match the checkpoint")
                source_vocab_size = hf_config.vocab_size
            if source_vocab_size is not None:
                if len(tokenizer) > source_vocab_size:
                    raise ValueError("llama_native tokenizer vocabulary exceeds the checkpoint vocabulary")
                config.model.validate_sharding(
                    _native_llama_sharding(config.trainer), trainer.device_mesh, source_vocab_size
                )

        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping
        if native:
            compute_axis_mapping = native_evaluation_axis_mapping(
                config.model,
                parameter_axis_mapping,
                compute_axis_mapping,
                trainer.device_mesh,
                config.trainer.batch_axis_name,
            )

        # some axes we need
        EvalBatch = config.trainer.EvalBatch
        model_max_seq_len = config.model.max_seq_len
        train_length = config.train_seq_len
        if train_length is None:
            train_length = model_max_seq_len

        if train_length <= 0:
            raise ValueError(f"train_length must be positive, got {train_length}")

        if train_length > model_max_seq_len:
            raise ValueError(f"train_length ({train_length}) cannot exceed model max_seq_len ({model_max_seq_len}).")

        if train_length != model_max_seq_len:
            logger.info(f"Training with sequence length {train_length} (model supports {model_max_seq_len}).")

        Pos = config.model.max_Pos.resize(train_length)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        vocab_mapping = parameter_axis_mapping
        if native:
            vocab_mapping = {**parameter_axis_mapping, "vocab": _native_llama_sharding(config.trainer).model_axis}
        Vocab = (
            Axis("vocab", source_vocab_size)
            if source_vocab_size is not None
            else round_axis_for_partitioning(Axis("vocab", vocab_size), vocab_mapping)
        )
        if vocab_size != Vocab.size:
            logger.info(f"Using model vocabulary {Vocab.size} for tokenizer vocabulary {vocab_size}")

        # Get the training dataset
        make_train_dataset = config.data.train_mixture if native else config.data.train_set
        train_dataset = make_train_dataset(
            Pos,
            config.trainer.batch_schedule,
            key=data_key,
        )
        install_tensorstore_metrics_hook_if_enabled(trainer)

        # Get the tagged evaluation datasets
        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)

        adapter_key = jrandom.fold_in(model_key, ord("a"))
        load_source_model = functools.partial(
            _load_lm_model_from_configured_source,
            config=config,
            converter=converter,
            Vocab=Vocab,
            model_key=model_key,
            adapter_key=adapter_key,
            parameter_axis_mapping=parameter_axis_mapping,
            trainer=trainer,
            source_checkpoint_path=source_checkpoint_path,
        )
        weight_source = config.initialize_from_hf or config.initialize_model_from_checkpoint_path is not None
        if weight_source and not resuming:
            logger.info("Initializing trainer state directly from pretrained weights")
            initial_model = load_source_model()
            state = trainer.initial_state(
                training_key,
                model=initial_model,
                is_trainable=config.adapter.trainable_filter(initial_model),
            )
        elif isinstance(config.adapter, NoAdaptorConfig):
            state = trainer.initial_state(
                training_key,
                model_init=functools.partial(
                    build_model,
                    config.model,
                    Vocab,
                    key=model_key,
                    parameter_axis_mapping=trainer.parameter_axis_mapping,
                    compute_axis_mapping=trainer.compute_axis_mapping,
                    batch_axis_name=trainer.config.batch_axis_name,
                ),
            )
        else:
            initial_model = config.adapter.apply(
                build_model(
                    config.model,
                    Vocab,
                    key=model_key,
                    parameter_axis_mapping=trainer.parameter_axis_mapping,
                    compute_axis_mapping=trainer.compute_axis_mapping,
                    batch_axis_name=trainer.config.batch_axis_name,
                ),
                key=adapter_key,
                axis_mapping=parameter_axis_mapping,
            )
            state = trainer.initial_state(
                training_key,
                model=initial_model,
                is_trainable=config.adapter.trainable_filter(initial_model),
            )

        if not resuming and config.initialize_from_checkpoint_path is not None:
            checkpoint_path = source_checkpoint_path or latest_checkpoint_path(config.initialize_from_checkpoint_path)
            state = load_checkpoint(state, checkpoint_path)
            # reset to step 0, we're just initializing weights here
            state = dataclasses.replace(state, step=jnp.array(0))

        if resuming and not isinstance(config.adapter, NoAdaptorConfig):
            logger.info(
                "Adapter checkpoints only store trainable weights. Reconstructing the base LM model from the "
                "configured source before overlaying resumed adapter parameters."
            )
            source_model = load_source_model()
            state = dataclasses.replace(
                state,
                model=_restore_lm_model_from_partial_checkpoint(
                    state.model,
                    source_model,
                    config.adapter.trainable_filter(source_model),
                ),
            )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            # Write eval metrics to the same directory as checkpoints
            checkpoint_path = None
            if config.trainer.checkpointer is not None:
                checkpoint_path = config.trainer.checkpointer.expanded_path(trainer.run_id)

            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                tagged_eval_datasets,
                tokenizer,
                trainer.device_mesh,
                compute_axis_mapping,
                max_eval_examples_per_ds,
                mp=config.trainer.mp,
                checkpoint_path=checkpoint_path,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        if config.labeled_eval is not None:
            add_labeled_lm_eval_callbacks(
                trainer,
                labeled_eval_config=config.labeled_eval,
                data_config=config.data,
                trainer_config=config.trainer,
                EvalBatch=EvalBatch,
                Pos=Pos,
                tokenizer=tokenizer,
                device_mesh=trainer.device_mesh,
                axis_mapping=compute_axis_mapping,
                max_eval_examples_per_dataset=max_eval_examples_per_ds,
            )

        flops_per_token = config.model.flops_per_token(Vocab.size if native else vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )
        trainer.add_hook(
            iris_status_reporter(
                Pos.size, trainer.config.batch_schedule, trainer.config.num_train_steps, flops_per_example
            ),
            every=10,
        )

        if isinstance(train_dataset, MixtureDataset):
            trainer.add_hook(
                callbacks.mixture_weight_logging_hook(trainer.config.batch_schedule, train_dataset), every=1
            )

        export_converter = converter
        if native and any(
            path is not None for path in (config.hf_save_path, config.peft_save_path, config.merged_hf_save_path)
        ):
            export_converter = config.model.hf_export_converter(tokenizer)
            if converter is not None:
                export_converter = export_converter.replaced(reference_checkpoint=converter.reference_checkpoint)
        config.adapter.install_export_hooks(
            trainer=trainer,
            converter=export_converter,
            tokenizer=tokenizer,
            export=dataclasses.replace(
                export_config,
                generation_config=build_generation_config(tokenizer, config.hf_generation_eos_token_ids),
            ),
        )

        if config.eval_harness is not None:
            eval_harness = config.eval_harness
            trainer.add_hook(
                levanter.eval_harness.lm_eval_harness(
                    eval_harness, tokenizer, EvalBatch, compute_axis_mapping, trainer.mp
                ),
                every=config.eval_harness_steps,
            )

        @named_jit(axis_resources=compute_axis_mapping)
        def compute_logits(model: LmHeadModel, example: LmExample):
            model = trainer.mp.cast_to_compute(model)
            activations, _ = split_activations(
                model.activations(example.tokens, key=None, attn_mask=example.attn_mask)
            )
            head = model.get_lm_head()
            logits = hax.dot(activations, head, axis=model.Embed)
            return logits

        if config.log_entropy:
            for name, dataset in config.data.validation_sets(Pos).items():
                trainer.add_hook(
                    levanter.analysis.cb_compute_entropies(
                        compute_logits,
                        Vocab,
                        dataset,
                        prefix=os.path.join("analysis", name) if name else "analysis",
                        batch_size=EvalBatch.size,
                        mapping=compute_axis_mapping,
                    ),
                    every=config.trainer.steps_per_eval,
                )

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        ## OK, actually run training!
        final_info = trainer.train(state, train_loader)

    # This isn't necessary except when Levanter is run in a subprocess (as happens under Iris/Fray)
    trainer.tracker.finish()
    return final_info


if __name__ == "__main__":
    levanter.config.main(main)()
