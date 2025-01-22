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
"""My implementation of Gemma-2 model without sliding window. Top-p calculation added."""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

import config as gemma_config


def get_top_token_ids(
        log_probas: torch.Tensor,  # [B, L, V]
        vocab_ids: torch.Tensor,  # [B, Vc]
        pad_id: torch.Tensor,  # [1]
        top_p: torch.Tensor,  # [1]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, L, Vc], [B, L, Vc]
    batch_size, seq_len, _ = log_probas.shape
    # Get log probas for current vocab
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=log_probas.device)  # [B]
    seq_indices = torch.arange(seq_len, dtype=torch.long, device=log_probas.device)  # [L]
    log_probas = log_probas[batch_indices[:, None, None], seq_indices[None, :, None], vocab_ids[:, None, :]]  # [B, L, Vc]
    log_probas = log_probas.where((vocab_ids != pad_id)[:, None, :], float('-inf'))  # [B, L, Vc]
    # Convert log probas to probas
    probas = torch.exp(log_probas.float())  # [B, L, Vc]
    # Normalize probas
    probas = probas / probas.sum(dim=-1, keepdim=True)  # [B, L, Vc]
    # Sort probas
    sorted_probas, sorted_indices = torch.sort(probas, descending=True)  # [B, L, Vc], [B, L, Vc]
    # Get sorted losses
    sorted_losses = torch.gather(-log_probas, 2, sorted_indices)  # [B, L, Vc]
    # Get sorted token ids
    sorted_ids = torch.gather(vocab_ids[:, None, :].expand(-1, seq_len, -1), 2, sorted_indices)  # [B, L, Vc]
    # Calc cumulative probas
    cumulative_probas = torch.cumsum(sorted_probas, dim=-1)  # [B, L, Vc]
    # Tokens with cumulative probability above the threshold should be removed
    indices_to_remove = cumulative_probas > top_p  # [B, L, Vc]
    # Shift the indices to the right to keep also the first token above the threshold
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()  # [B, L, Vc]
    # First token always must be presented
    indices_to_remove[..., 0] = 0  # [B, L, Vc]
    # Zero probas that should be removed
    sorted_probas[indices_to_remove] = 0  # [B, L, Vc]
    # Inf losses that should be removed
    sorted_losses[indices_to_remove] = float('+inf')  # [B, L, Vc]
    # Leave only top ids (mask others by pad_id)
    sorted_ids[indices_to_remove] = pad_id  # [B, L, Vc]

    return sorted_ids, sorted_losses, sorted_probas  # [B, L, Vc], [B, L, Vc], [B, L, Vc]


def get_losses(
        log_probas: torch.Tensor,  # [B, L, V]
        token_ids: torch.Tensor,  # [B, L]
        pad_id: torch.Tensor,  # [1]
) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, L], [L]
    batch_size, seq_len = token_ids.shape
    # Create tensor for cumulated losses
    cumulative_losses = torch.zeros_like(token_ids, dtype=log_probas.dtype)  # [B, L]
    # Shift tokens by one position to calc loss (drop BOS token) and
    # pick up log probas by token ids and apply minus operation to get losses
    cumulative_losses[:, 1:] = -log_probas[:, :-1, :].gather(2, token_ids[:, 1:].unsqueeze(2)).squeeze(2)  # [B, L]
    # Cumulate losses
    cumulative_losses = torch.cumsum(cumulative_losses, dim=-1)  # [B, L]
    # Mask losses with pad tokens by '+inf'
    cumulative_losses = cumulative_losses.where(token_ids != pad_id, float('+inf'))  # [B, L]
    # Create tensor for lengths (except BOS token)
    lengths = torch.arange(seq_len, dtype=token_ids.dtype, device=token_ids.device)  # [L]

    return cumulative_losses, lengths  # [B, L], [L]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: float,
        query_pre_attn_scalar: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = attn_logit_softcapping

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, input_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)  # [batch_size, max_seq_len, n_local_heads, head_dim]
        xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)  # [batch_size, max_seq_len, n_local_heads, head_dim]

        q = xq.transpose(1, 2)  # [batch_size, n_local_heads, input_len, head_dim]
        k = xk.transpose(1, 2)  # [batch_size, n_local_heads, max_seq_len, head_dim]
        v = xv.transpose(1, 2)  # [batch_size, n_local_heads, max_seq_len, head_dim]

        q.mul_(self.scaling)  # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores / self.attn_logit_softcapping
        scores = torch.tanh(scores)
        scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        output = torch.matmul(scores, v)  # [batch_size, n_local_heads, input_len, head_dim]

        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)  # [batch_size, input_len, hidden_dim]
        output = self.o_proj(output)
        return output


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(Gemma2DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        max_position_embeddings = config.max_position_embeddings
        max_seq_len = 256
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.embedder = nn.Embedding(vocab_size, config.hidden_size)
        self.model = GemmaModel(config)

        # Pre-compute rotary embedding table.
        freqs_cis = precompute_freqs_cis(head_dim, max_position_embeddings * 2, theta=10000)
        self.register_buffer('freqs_cis', freqs_cis)

        causal_mask = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, dtype=torch.float)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

        normalizer = torch.tensor(config.hidden_size**0.5, dtype=self.embedder.weight.dtype)
        self.register_buffer('normalizer', normalizer)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # long: [B, L]
        vocab_ids: torch.Tensor,  # long: [B, Vc]
        pad_id: torch.Tensor,  # long: [1]
        top_p: torch.Tensor,  # float32: [1]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        freqs_cis = self.freqs_cis[:seq_len]
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        hidden_states = self.embedder(input_ids)  # [B, L, hidden_size]
        hidden_states = hidden_states * self.normalizer
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )

        # Unembedding (lm-head)
        logits = torch.matmul(hidden_states, self.embedder.weight.t())

        # Softcapping
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping

        log_probas = self.log_softmax(logits)

        top_token_ids, top_token_losses, _ = get_top_token_ids(
            log_probas,
            vocab_ids,
            pad_id,
            top_p,
            )  # [B, L, Vc], [B, L, Vc], [B, L, Vc]

        # Cumulative losses along sequence length
        cumulative_losses, lengths = get_losses(log_probas, input_ids, pad_id)  # [B, L], [L]

        # Average losses along sequence length for top tokens
        top_token_mean_losses = (top_token_losses + cumulative_losses[:, :, None]) / (lengths + 1)[:, None]  # [B, L, Vc]

        # Calc mean (by length) losses
        corr_lengths = lengths.clone()
        corr_lengths[0] = 1  # [L]
        mean_losses = cumulative_losses / corr_lengths  # [B, L]

        return top_token_ids, top_token_mean_losses, mean_losses  # [B, L, Vc], [B, L, Vc], [B, L]
