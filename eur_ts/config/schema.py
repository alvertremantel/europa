from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int = 64
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 1024
    dropout: float = 0.1
    position_encoding: str = "type_place"
    token_type_vocab_size: int = 3
    place_vocab_size: int = 9


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data-1m"
    output_dir: str = "runs/arithmetic-small"
    resume_from: str | None = None
    additional_epochs: int | None = None
    sequence_length: int = 64
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 100
    max_new_tokens: int = 24
    seed: int = 42
    device: str = "cuda"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 1024
    dropout: float = 0.1
    position_encoding: str = "type_place"
    training_mode: str = "token_stream"
    training_format: str = "final_only"
    skip_overlong_examples: bool = False
    curriculum_name: str | None = None
