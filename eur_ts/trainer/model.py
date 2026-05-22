from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eur_ts.config import ModelConfig
from .fixed_meaning import (
    DIGIT_TOKENS,
    FIXED_MEANING_DIGIT_PLACE_DIMENSION,
    FIXED_MEANING_MAX_DIGIT_PLACE,
    build_fixed_meaning_token_table,
)
from .tokenizer import ArithmeticTokenizer, POSITION_ENCODING_FIXED_MEANING


class FixedMeaningEmbedding(nn.Module):
    def __init__(self, tokens: Sequence[str], d_model: int) -> None:
        super().__init__()
        table = build_fixed_meaning_token_table(tokens, d_model)
        self.register_buffer("_weight", table)
        digit_token_mask = torch.zeros(len(tokens), dtype=torch.bool)
        for token_id, token in enumerate(tokens):
            if token in DIGIT_TOKENS:
                digit_token_mask[token_id] = True
        self.register_buffer("_digit_token_mask", digit_token_mask)

    @property
    def weight(self) -> Tensor:
        return cast(Tensor, self._weight)

    @property
    def digit_token_mask(self) -> Tensor:
        return cast(Tensor, self._digit_token_mask)

    def forward(
        self,
        input_ids: Tensor,
        digit_place_values: Tensor | None = None,
    ) -> Tensor:
        embeddings = F.embedding(input_ids, self.weight)
        is_digit = self.digit_token_mask[input_ids]
        if not torch.any(is_digit):
            return embeddings

        if digit_place_values is None:
            place_values = _local_digit_place_values(is_digit, dtype=embeddings.dtype)
        else:
            if digit_place_values.shape != input_ids.shape:
                raise ValueError("digit_place_values shape must match input_ids")
            place_values = digit_place_values.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        embeddings = embeddings.clone()
        embeddings[..., FIXED_MEANING_DIGIT_PLACE_DIMENSION] = torch.where(
            is_digit,
            place_values,
            embeddings[..., FIXED_MEANING_DIGIT_PLACE_DIMENSION],
        )
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_model = _required_d_model(config)
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        sequence_length = hidden_states.size(1)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        normalized = self.norm_1(hidden_states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


class SmallCausalTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tokenizer: ArithmeticTokenizer,
    ) -> None:
        super().__init__()
        if config.position_encoding != POSITION_ENCODING_FIXED_MEANING:
            raise ValueError(
                f"unsupported position encoding: {config.position_encoding!r}"
            )
        d_model = _required_d_model(config)
        self.config = config
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                "tokenizer vocabulary size must match model_config.vocab_size"
            )
        self.token_embedding = FixedMeaningEmbedding(tokenizer.id_to_token, d_model)
        self.position_embedding = None
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        digit_place_values: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds model limit {self.config.sequence_length}"
            )
        hidden_states = self.token_embedding(input_ids, digit_place_values)
        hidden_states = self.dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if logits.shape[:2] != (batch_size, sequence_length):
            raise RuntimeError("unexpected logits shape")
        return logits


def _local_digit_place_values(is_digit: Tensor, *, dtype: torch.dtype) -> Tensor:
    if is_digit.ndim != 2:
        raise ValueError(
            f"fixed_meaning digit-place encoding expects rank-2 input, got {tuple(is_digit.shape)}"
        )
    batch_size, sequence_length = is_digit.shape
    current_places = torch.zeros(batch_size, device=is_digit.device, dtype=torch.long)
    place_indices = torch.zeros(
        (batch_size, sequence_length),
        device=is_digit.device,
        dtype=torch.long,
    )
    for position in range(sequence_length):
        current_places = torch.where(
            is_digit[:, position],
            current_places + 1,
            torch.zeros_like(current_places),
        )
        place_indices[:, position] = current_places.clamp_max(
            FIXED_MEANING_MAX_DIGIT_PLACE
        )
    return place_indices.to(dtype=dtype) / 10.0


def _required_d_model(config: ModelConfig) -> int:
    if config.d_model is None:
        raise ValueError("model_config.d_model is required")
    return config.d_model
