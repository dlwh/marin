# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Normal Levanter LM boundaries around the array-only Llama implementation."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.state_dict import ModuleWithStateDictSerialization
from jax.sharding import PartitionSpec as P, get_abstract_mesh, reshard
from jaxtyping import PRNGKeyArray
from transformers import PretrainedConfig as HfConfig
from transformers import LlamaConfig as HfLlamaConfig

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.compat.hf_config import hf_rope_config
from levanter.data.text.examples import grug_attention_mask_from_named
from levanter.grug.attention import AttentionMask as RawAttentionMask
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.tokenizers import MarinTokenizer
from levanter.utils.activation import ActivationFunctionEnum


@LmConfig.register_subclass("llama_native")
@dataclass(frozen=True)
class NativeLlamaLmConfig(LlamaConfig):
    """Llama training configuration with raw-array parameters and explicit placement.

    The default attention backend is the portable reference implementation.
    Select ``splash`` explicitly to use the native TPU Splash implementation.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.activation_function != ActivationFunctionEnum.silu:
            raise ValueError("llama_native supports only the silu activation")
        if not self.scan_layers:
            raise ValueError("llama_native requires scan_layers: true")
        if not isinstance(self.rope, DefaultRotaryEmbeddingsConfig):
            raise ValueError("llama_native supports only default or linear RoPE")
        if self.attn_backend not in (
            None,
            AttentionBackend.DEFAULT,
            AttentionBackend.VANILLA,
            AttentionBackend.SPLASH,
        ):
            raise ValueError("llama_native supports vanilla/reference or splash attention")
        if self.flash_attention_block_size is not None:
            raise ValueError("llama_native does not support flash_attention_block_size")
        self.native_config(1)

    @property
    def requires_explicit_mesh_axes(self) -> bool:
        return True

    @property
    def model_type(self) -> type["NativeLlamaLmHeadModel"]:  # pyrefly: ignore[bad-override]
        return NativeLlamaLmHeadModel

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "NativeLlamaLmConfig":
        named_config = LlamaConfig.from_hf_config(hf_config)
        return cls(**{field.name: getattr(named_config, field.name) for field in dataclasses.fields(named_config)})

    def hf_checkpoint_converter(
        self, ref_checkpoint: str | None = None, *, tokenizer: MarinTokenizer | None = None
    ) -> HFCheckpointConverter:
        return HFCheckpointConverter(
            type(self),
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            HfConfigClass=HfLlamaConfig,
            tokenizer=tokenizer if tokenizer is not None else self.tokenizer,
        )

    def validate_hf_config(self, hf_config: HfConfig) -> None:
        """Reject HF architectures whose weights would change meaning under this configuration."""
        if hf_config.model_type != "llama":
            raise ValueError("llama_native HF initialization requires a Llama checkpoint")
        if (
            self.hybrid_norm
            or self.input_embedding_norm
            or self.use_qk_norm
            or not self.use_layer_norm_weight
            or any(getattr(hf_config, name, False) for name in ("hybrid_norm", "input_embedding_norm", "use_qk_norm"))
            or not getattr(hf_config, "use_layer_norm_weight", True)
        ):
            raise ValueError("llama_native HF initialization requires standard weighted RMSNorm")
        # The native use_bias also enables norm biases, which standard HF Llama
        # does not serialize even when attention_bias or mlp_bias is enabled.
        if self.use_bias or hf_config.attention_bias or hf_config.mlp_bias:
            raise ValueError("llama_native HF initialization supports only bias-free Llama")
        if hf_config.attention_dropout != 0:
            raise ValueError("llama_native HF initialization does not support attention dropout")
        expected = {
            "hidden_size": self.hidden_dim,
            "intermediate_size": self.intermediate_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.actual_head_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_act": self.activation_function.value,
            "rms_norm_eps": self.layer_norm_epsilon,
        }
        for name, value in expected.items():
            actual = getattr(hf_config, name)
            if actual != value:
                raise ValueError(f"llama_native HF {name} mismatch: checkpoint={actual}, configured={value}")
        theta, scaling = hf_rope_config(hf_config)
        if getattr(hf_config, "partial_rotary_factor", 1.0) != 1.0 or (
            scaling is not None and scaling.get("partial_rotary_factor", 1.0) != 1.0
        ):
            raise ValueError("llama_native HF initialization requires full-head RoPE")
        rope = RotaryEmbeddingsConfig.from_hf_config(theta, scaling)
        if not isinstance(rope, DefaultRotaryEmbeddingsConfig) or rope != self.rope:
            raise ValueError("llama_native HF RoPE parameters must match the configured default/linear RoPE")

    def native_config(self, vocab_size: int) -> NativeLlamaConfig:
        """Resolve tokenizer vocabulary and the supported architecture fields."""
        assert isinstance(self.rope, DefaultRotaryEmbeddingsConfig)
        fields = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(NativeLlamaConfig)
            if field.name not in {"vocab_size", "rope_theta", "rope_factor", "attention_implementation"}
        }
        return NativeLlamaConfig(
            vocab_size=vocab_size,
            rope_theta=self.rope.theta,
            rope_factor=self.rope.factor,
            attention_implementation="tpu_splash" if self.attn_backend == AttentionBackend.SPLASH else "reference",
            **fields,
        )

    def build(  # pyrefly: ignore[bad-override]  # This config selects a different LmHeadModel implementation.
        self,
        Vocab: Axis,
        *,
        key: PRNGKeyArray,
        sharding: NativeLlamaSharding = NativeLlamaSharding(),
    ) -> "NativeLlamaLmHeadModel":
        return NativeLlamaLmHeadModel.init(Vocab, self, key=key, sharding=sharding)


class NativeLlamaLmHeadModel(ModuleWithStateDictSerialization, LmHeadModel[NativeLlamaLmConfig]):
    """Named input/loss/evaluation boundary; every model parameter is a raw array."""

    native: NativeLlamaModel
    _config: NativeLlamaLmConfig = eqx.field(static=True)

    @property
    def config(self) -> NativeLlamaLmConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.native.config.vocab_size)

    @property
    def _batch_partition(self) -> tuple[str, ...] | None:
        mesh = get_abstract_mesh()
        return tuple(axis for axis in self.native.sharding.batch_axes if mesh.shape[axis] > 1) or None

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: NativeLlamaLmConfig,
        *,
        key: PRNGKeyArray,
        sharding: NativeLlamaSharding = NativeLlamaSharding(),
    ) -> "NativeLlamaLmHeadModel":
        return cls(config.native_config(Vocab.size).build(key=key, sharding=sharding), config)

    def _attention_mask(
        self, mask: AttentionMask | NamedArray | None, token_axes: tuple[Axis, ...]
    ) -> RawAttentionMask | jax.Array:
        if mask is None:
            # The normal LM API treats None as unmasked. The raw model's default
            # is causal, so represent the unmasked case explicitly.
            return RawAttentionMask()
        Pos = token_axes[-1]
        KeyPos = Pos.alias(self.KeyPos.name)
        batch_axes = token_axes[:-1]
        dense = mask if isinstance(mask, NamedArray) else None
        if isinstance(mask, AttentionMask):
            if (
                mask.causal_offset is not None
                or mask.explicit_mask is not None
                or mask.bidirectional_window is not None
            ):
                dense = mask.materialize(Pos, KeyPos)
            else:
                if mask.segment_ids is not None:
                    query, key = mask.segment_ids
                    query = query.broadcast_to((*batch_axes, Pos))
                    if key.has_axis(Pos.name):
                        key = key.rename({Pos.name: KeyPos.name})
                    key = key.broadcast_to((*batch_axes, KeyPos))
                    mask = dataclasses.replace(mask, segment_ids=(query, key))
                return grug_attention_mask_from_named(mask)
        if self.native.config.attention_implementation != "reference":
            raise ValueError(
                "Dense, offset and bidirectional attention masks require llama_native reference attention"
            )
        assert dense is not None
        return dense.broadcast_to((*batch_axes, Pos, KeyPos)).array

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        Pos = input_ids.resolve_axis(self.Pos.name)
        ordered = input_ids.rearrange((..., Pos))
        if ordered.ndim not in (1, 2):
            raise ValueError("llama_native expects tokens with position and at most one batch axis")
        tokens = ordered.array
        unbatched = tokens.ndim == 1
        if unbatched:
            # Keep the unbatched LM API usable on a data-sharded model. Identical
            # examples occupy each batch partition; only one output is returned.
            batch_size = math.prod(get_abstract_mesh().shape[axis] for axis in self.native.sharding.batch_axes)
            tokens = jnp.broadcast_to(tokens, (batch_size, tokens.shape[0]))
        positions = None
        if pos_ids is not None:
            positions = pos_ids.broadcast_to(ordered.axes).array
            if unbatched:
                positions = jnp.broadcast_to(positions, tokens.shape)
        mask = self._attention_mask(attn_mask, ordered.axes)
        hidden = self.native.activations(tokens, mask, pos_ids=positions, key=key)
        if unbatched:
            hidden = hidden.at[0].get(out_sharding=P(None, None))
        else:
            hidden = reshard(hidden, P(self._batch_partition, None, None))
        return hax.named(hidden, (*ordered.axes, self.Embed))

    def get_lm_head(self) -> NamedArray:
        # The shared CE kernel partitions examples and requires a complete head
        # on each participant. Storage remains FSDP/tensor-parallel.
        return hax.named(reshard(self.native.get_lm_head(), P(None, None)), (self.Vocab, self.Embed))

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(hax.ReductionFunction, hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ):
        mapping = {}
        tokens = example.tokens.rearrange((..., self.Pos.name))
        weights = example.loss_weight.broadcast_to(tokens.axes)
        batch_axes = tuple(axis for axis in tokens.axes if axis.name != self.Pos.name)
        batch_partition = self._batch_partition
        if batch_axes and batch_partition is not None:
            mapping[batch_axes[0].name] = batch_partition
        token_spec = P(batch_partition, None) if batch_axes else P(None)
        # Unit-size mesh axes are physically replicated. Canonicalize them before
        # jnp.roll: its empty slice can otherwise lose those axes and make the
        # explicit-sharding concatenate reject a single-example microbatch.
        example = dataclasses.replace(
            example,
            tokens=hax.named(reshard(tokens.array, token_spec), tokens.axes),
            loss_weight=hax.named(reshard(weights.array, token_spec), weights.axes),
        )
        # The loss's shard_map must agree with explicit array types, independently
        # of a caller's named-axis compute mapping.
        with hax.axis_mapping(mapping):
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "NativeLlamaLmHeadModel":
        if new_size == self.Vocab.size:
            return self
        embed_key, head_key = maybe_rng_split(key, 2)
        # The existing resize helper concatenates old rows with padding. Gather
        # its resize axis so both pieces share a layout, then restore storage.
        embeddings = hax.named(
            reshard(self.native.token_embeddings, P(None, self.native.sharding.fsdp_axis)), (self.Vocab, self.Embed)
        )
        embeddings = hax.tree_util.resize_axis(embeddings, self.Vocab, new_size, key=embed_key)
        head = self.native.lm_head
        if head is not None:
            weight = hax.tree_util.resize_axis(self.get_lm_head(), self.Vocab, new_size, key=head_key)
            head = dataclasses.replace(head, weight=reshard(weight.array, self.native.sharding.column_storage))
        native = dataclasses.replace(
            self.native,
            token_embeddings=reshard(embeddings.array, self.native.sharding.column_storage),
            lm_head=head,
            config=dataclasses.replace(self.native.config, vocab_size=new_size),
        )
        return dataclasses.replace(self, native=native)

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return self.native.to_state_dict(prefix)

    def from_state_dict(self, state_dict, prefix: str | None = None) -> "NativeLlamaLmHeadModel":
        return dataclasses.replace(self, native=self.native.from_state_dict(state_dict, prefix))
