# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Dict, Optional, Union, cast

import equinox as eqx
import jax
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockFoldable, BlockSeq, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.attention import Attention, AttentionMask
from levanter.layers.gated_mlp import LlamaMlp
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.models.llama_config import LlamaConfig
from levanter.models.llama_embedding import LlamaEmbedding
from levanter.models.lm_model import LmHeadModel, resize_embeddings_and_lm_head


class LlamaDecoderLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    post_attn_layernorm: Optional[hnn.RmsNorm] = None
    post_mlp_layernorm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)
        post_attn_ln = None
        post_mlp_ln = None
        if config.hybrid_norm:
            post_attn_ln = config.mk_LayerNorm(config.Embed)
            post_mlp_ln = config.mk_LayerNorm(config.Embed)
        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2, post_attn_ln, post_mlp_ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output

    @named_call
    def decode(
        self,
        x: NamedArray,
        kv_cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output, kv_cache = self.self_attn.paged_decode(x, kv_cache, batch_info, pos_ids=pos_ids, key=k_attn)

        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output, kv_cache

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> KvPageCache:
        """
        Creates an empty page cache for this layer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return self.self_attn.empty_page_cache(spec, dtype=dtype)


class LlamaTransformer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    layers: BlockFoldable[LlamaDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaTransformer":
        S = Stacked
        if not config.scan_layers:
            S = BlockSeq

        layers = S.init(config.Layers, LlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return LlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = cast(NamedArray, self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        x = self.norm(x)

        return x

    @named_call
    def decode(
        self,
        kv_cache: ListCache[KvPageCache],
        x: NamedArray,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        # Unfortunately, JAX does not seem to want to intelligently reuse memory here, so we manually unroll the loop
        # x, kv_cache = self.layers.scan_via(LlamaDecoderLayer.decode)(
        #     x,
        #     kv_cache,
        #     batch_info,
        #     pos_ids=pos_ids,
        #     key=keys,
        # )
        # x = self.norm(x)
        # return x, kv_cache
        caches = list(kv_cache)
        updated_caches: list[KvPageCache] = []

        for i in range(self.config.num_layers):
            with jax.named_scope("slice layer"):
                layer = self.layers.get_layer(i)
            with jax.named_scope("slice cache"):
                this_cache = caches[i]
            x, this_cache = layer.decode(
                x,
                this_cache,
                batch_info,
                pos_ids=pos_ids,
                key=keys[i] if keys is not None else None,
            )
            with jax.named_scope("update cache"):
                updated_caches.append(this_cache)

        x = self.norm(x)

        return x, ListCache(updated_caches)

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        """
        Creates an empty page cache for this transformer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        # sadly this is too cute/smart for XLA to handle aliasing correctly
        # return self.layers.vmap_via(LlamaDecoderLayer.initial_cache)(spec, dtype=dtype)
        caches = [layer.initial_cache(spec, dtype=dtype) for layer in self.layers.unstacked()]
        return ListCache(caches)


class LlamaLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[LlamaConfig]):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return LlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
            pos_ids: NamedArray | None = None,

        Returns:
            NamedArray: logits with shape {Batch, Pos, Vocab}
        """
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKeyArray for random number generation
            pos_ids: position IDs with shape {Pos}

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
        new_embeddings, new_lm_head = resize_embeddings_and_lm_head(
            self.Vocab, self.embeddings, self.lm_head, new_size, key
        )
        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        """
        Creates an initial cache for this model. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return hax.auto_sharded(self.transformer.initial_cache(spec, dtype=dtype))

    @named_call
    def decode(
        self,
        input_ids: NamedArray,  # token IDs for *this* step (shape {Pos} or {Batch, Pos})
        kv_cache: ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        """Run one decode / pre-fill step with an existing paged-KV *state*.

        Parameters
        ----------
        input_ids : NamedArray
            Token IDs for the positions being decoded **this call**.
        kv_cache : ListCache[KvPageCache]
            Current paged-KV cache (one per layer). Obtain the initial value via
            ``self.initial_cache`` and update with the returned *new_state* each step.
        pos_ids : NamedArray
            Absolute position IDs matching *input_ids* (negative IDs can mark padding as
            in the lower-level API).
        key : jax.random.PRNGKey | None
            RNG key for dropout etc.  Can be omitted during inference.

        Returns
        -------
        logits : NamedArray
            Logits for the provided tokens (axes match *input_ids* + ``Vocab``).
        new_state : ListCache[KvPageCache]
            Updated cache to pass into the next decode call.
        """

        # Embed the incoming token IDs
        x = self.embeddings.embed(input_ids)

        # Propagate through the transformer with paged-KV caching
        k_t = maybe_rng_split(key, 1)[0] if key is not None else None
        x, new_state = self.transformer.decode(kv_cache, x, batch_info, pos_ids, key=k_t)

        # Project to logits
        if self.lm_head is not None:
            logits = self.lm_head(x, key=None)
        else:
            logits = self.embeddings.unembed(x)

        return logits, new_state
