# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.random as jrandom

import haliax.nn as hnn
from haliax import Axis

from levanter.models.llama_config import LlamaConfig
from levanter.models.llama_embedding import LlamaEmbedding
from levanter.models.llama_head import LlamaHeadModel
from levanter.models.llama_transformer import LlamaTransformer


class LlamaLMHeadModel(LlamaHeadModel):
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
