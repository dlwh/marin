# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Llama configuration with the existing named-runtime default."""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type

from haliax import Axis, AxisSpec
from haliax.nn.scan import ScanCheckpointPolicy

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.compat.hf_config import hf_config_from_kwargs, hf_rope_config
from levanter.layers.attention import AttentionBackend, AttentionConfig
from levanter.layers.normalization import LayerNormConfigBase, RmsNormConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("llama")
@dataclass(frozen=True)
class LlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        max_seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        hybrid_norm (bool, optional): whether to use hybrid normalization with additional layer norms after attention and MLP. Defaults to False.
        input_embedding_norm (bool, optional): whether to use layer normalization after input embeddings. Defaults to False.
    """

    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int | None = None  # if set, will use this as the head dimension instead of hidden_dim // num_heads
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    hybrid_norm: bool = (
        False  # if set to True, this will add an additional layer norm after the attention and MLP layers
    )
    use_qk_norm: bool = (
        False  # if set to True, this will add an additional layer norm after the key and query projections
    )
    input_embedding_norm: bool = False

    # Attention-related config
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: Optional[str] = None

    # Axis
    @property
    def KeyPos(self) -> Axis:
        return self.max_Pos.alias("key_position")

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    Layers = property(lambda self: Axis(name="layer", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["LlamaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta, rope_scaling = hf_rope_config(hf_config)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)
        return LlamaConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's LlamaConfig

        Raises:
            ValueError: If hybrid_norm or input_embedding_norm are enabled, as these features
                are not supported in the HuggingFace config format.
        """
        if self.hybrid_norm or self.input_embedding_norm or self.use_qk_norm:
            raise ValueError(
                "Cannot export to HuggingFace format with hybrid_norm, input_embedding_norm, or use_qk_norm enabled. "
                "These features are not supported in the HuggingFace config format. "
                "Please disable these features before exporting."
            )

        if config_overrides is None:
            config_overrides = {}

        if self.rope:
            rope_theta, rope_scaling = self.rope.to_hf_config()
        else:
            rope_theta = None
            rope_scaling = None

        return hf_config_from_kwargs(
            HfLlamaConfig,
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=self.use_bias,
            mlp_bias=self.use_bias,
            _attn_implementation="eager",
            **config_overrides,
        )

    @property
    def model_type(self) -> Type[LmHeadModel]:  # pyrefly: ignore[bad-override]
        from levanter.models.llama import LlamaLMHeadModel  # noqa: PLC0415  # circular schema/runtime import

        return LlamaLMHeadModel

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(
            use_weight=self.use_layer_norm_weight,
            use_bias=self.use_bias,
            eps=self.layer_norm_epsilon,
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.actual_head_size
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        mlp = 3 * self.hidden_dim * self.intermediate_dim

        transformer_layer = attn + mlp + 2 * self.hidden_dim  # plus 2 rmsnorm
        if self.hybrid_norm:
            transformer_layer += 2 * self.hidden_dim

        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm
        if self.input_embedding_norm:
            transformer += self.hidden_dim

        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head

    def attention_config(self) -> AttentionConfig:
        """Convert this LlamaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config if self.use_qk_norm else None,
        )

    @property
    def actual_head_size(self):
        """Returns the actual head size based on the head_dim or calculated from hidden_dim and num_heads."""
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_dim // self.num_heads
