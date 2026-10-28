# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Dict, Optional, cast

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.models.llama_config import LlamaConfig


class LlamaEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    token_embeddings: hnn.Embedding
    norm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        norm = None
        if config.input_embedding_norm:
            norm = config.mk_LayerNorm(config.Embed)
        return LlamaEmbedding(token_embeddings, norm)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return cast(Axis, self.token_embeddings.Embed)

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings(input_ids)
        if self.norm is not None:
            input_embeds = self.norm(input_embeds)
        return input_embeds

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)
