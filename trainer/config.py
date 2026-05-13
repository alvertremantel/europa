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


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data-1m"
    output_dir: str = "runs/arithmetic-small"
    sequence_length: int = 64
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_batches: int = 50
    exact_match_samples: int = 256
    max_new_tokens: int = 24
    seed: int = 42
    device: str = "cuda"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 1024
    dropout: float = 0.1
