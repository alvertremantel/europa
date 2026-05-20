from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from eur_ts.config import ModelConfig
from .tokenizer import (
    ArithmeticTokenizer,
    PLACE_NONE,
    POSITION_ENCODING_FIXED_MEANING,
    POSITION_ENCODING_TYPE_PLACE,
)

_CONTROL_TOKENS = [
    "<pad>",
    "<do>",
    "<eos>",
    "<sep>",
    "<calc>",
    "<work>",
    "<step>",
    "<final>",
    "undefined",
    "remainder",
]
_OPERATOR_TOKENS = ["+", "-", "*", "/", "=", "(", ")"]


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.d_model),
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
        tokenizer: ArithmeticTokenizer | None = None,
    ) -> None:
        super().__init__()
        if config.position_encoding not in {
            POSITION_ENCODING_TYPE_PLACE,
            POSITION_ENCODING_FIXED_MEANING,
        }:
            raise ValueError(
                f"unsupported position encoding: {config.position_encoding!r}"
            )
        self.config = config
        if config.position_encoding == POSITION_ENCODING_TYPE_PLACE:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.type_embedding: nn.Embedding | None = nn.Embedding(
                config.token_type_vocab_size,
                config.d_model,
            )
            self.place_embedding: nn.Embedding | None = nn.Embedding(
                config.place_vocab_size,
                config.d_model,
                padding_idx=PLACE_NONE,
            )
            self.position_embedding: nn.Embedding | None = None
        else:
            if tokenizer is None:
                raise ValueError(
                    "fixed_meaning models require a tokenizer to build frozen input embeddings"
                )
            if tokenizer.vocab_size != config.vocab_size:
                raise ValueError(
                    "tokenizer vocabulary size must match model_config.vocab_size"
                )
            token_table = _build_fixed_token_embedding(
                tokenizer.id_to_token,
                config.d_model,
            )
            position_table = _build_sinusoidal_position_embedding(
                config.sequence_length,
                config.d_model,
            )
            self.token_embedding = nn.Embedding.from_pretrained(
                token_table, freeze=True
            )
            self.type_embedding = None
            self.place_embedding = None
            self.position_embedding = nn.Embedding.from_pretrained(
                position_table,
                freeze=True,
            )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.position_encoding == POSITION_ENCODING_TYPE_PLACE:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        type_ids: Tensor | None = None,
        place_ids: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds model limit {self.config.sequence_length}"
            )
        if self.config.position_encoding == POSITION_ENCODING_FIXED_MEANING:
            positions = torch.arange(sequence_length, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
            assert self.position_embedding is not None
            hidden_states = self.token_embedding(input_ids) + self.position_embedding(
                positions
            )
        else:
            if type_ids is None or place_ids is None:
                raise ValueError(
                    "type_place encoding requires explicit type_ids and place_ids"
                )
            if type_ids.shape != input_ids.shape or place_ids.shape != input_ids.shape:
                raise ValueError("type_ids and place_ids shapes must match input_ids")
            assert self.type_embedding is not None and self.place_embedding is not None
            hidden_states = (
                self.token_embedding(input_ids)
                + self.type_embedding(type_ids)
                + self.place_embedding(place_ids)
            )
        hidden_states = self.dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if logits.shape[:2] != (batch_size, sequence_length):
            raise RuntimeError("unexpected logits shape")
        return logits


def _build_fixed_token_embedding(tokens: list[str], d_model: int) -> Tensor:
    table = torch.zeros((len(tokens), d_model), dtype=torch.float32)
    control_count = max(len(_CONTROL_TOKENS) - 1, 1)
    operator_count = max(len(_OPERATOR_TOKENS) - 1, 1)

    for token_id, token in enumerate(tokens):
        values: list[float] = []
        if token.isdigit():
            values = [int(token) / 9.0, 1.0, 0.0, 0.0, 0.0]
        elif token in _OPERATOR_TOKENS:
            values = [
                0.0,
                0.0,
                1.0,
                0.0,
                _OPERATOR_TOKENS.index(token) / operator_count,
            ]
        elif token in _CONTROL_TOKENS:
            values = [
                0.0,
                0.0,
                0.0,
                1.0,
                _CONTROL_TOKENS.index(token) / control_count,
            ]
        _write_prefix(table[token_id], values)
    return table


def _write_prefix(target: Tensor, values: list[float]) -> None:
    for index, value in enumerate(values[: target.numel()]):
        target[index] = value


def _build_sinusoidal_position_embedding(sequence_length: int, d_model: int) -> Tensor:
    table = torch.zeros((sequence_length, d_model), dtype=torch.float32)
    if d_model == 0 or sequence_length == 0:
        return table
    positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(d_model, 1))
    )
    table[:, 0::2] = torch.sin(positions * div_term)
    if d_model > 1:
        table[:, 1::2] = torch.cos(positions * div_term[: table[:, 1::2].shape[1]])
    return table
