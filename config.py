# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""My version of Gemma-2 model config without sliding window."""

import dataclasses

import torch


# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


@dataclasses.dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'bfloat16'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: float = 30.0
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: float = 50.0
    # Query vector is normalized using the inverse square root of this value.
    query_pre_attn_scalar: int = 256

    def get_dtype(self) -> torch.dtype:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE[self.dtype]


def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=2304,
        intermediate_size=9216,
        head_dim=256,
        query_pre_attn_scalar=256,
    )


def get_config_for_9b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=3584,
        intermediate_size=14336,
        head_dim=256,
        query_pre_attn_scalar=256,
    )


def get_config_for_27b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=46,
        num_attention_heads=32,
        num_key_value_heads=16,
        hidden_size=4608,
        intermediate_size=36864,
        head_dim=128,
        query_pre_attn_scalar=144,  # hidden_size / num_attention_heads
    )


def get_model_config(variant: str) -> GemmaConfig:
    if variant == '2b':
        return get_config_for_2b()
    elif variant == '9b':
        return get_config_for_9b()
    elif variant == '27b':
        return get_config_for_27b()
    else:
        raise ValueError(
                f'Invalid variant {variant}. Supported variants are "2b", "9b" and "27b".')
