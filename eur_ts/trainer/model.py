from __future__ import annotations

import torch
from torch import Tensor, nn

from eur_ts.config import ModelConfig
from .tokenizer import PLACE_NONE, POSITION_ENCODING_TYPE_PLACE


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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.position_encoding != POSITION_ENCODING_TYPE_PLACE:
            raise ValueError(
                f"unsupported position encoding: {config.position_encoding!r}"
            )
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.type_embedding = nn.Embedding(config.token_type_vocab_size, config.d_model)
        self.place_embedding = nn.Embedding(
            config.place_vocab_size,
            config.d_model,
            padding_idx=PLACE_NONE,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
        if type_ids is None or place_ids is None:
            raise ValueError(
                "type_place encoding requires explicit type_ids and place_ids"
            )
        if type_ids.shape != input_ids.shape or place_ids.shape != input_ids.shape:
            raise ValueError("type_ids and place_ids shapes must match input_ids")

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
