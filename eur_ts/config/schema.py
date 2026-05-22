from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int = 64
    d_model: int | None = None
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 1024
    dropout: float = 0.1
    position_encoding: str = "type_place"
    token_type_vocab_size: int = 3
    place_vocab_size: int = 9

    def __post_init__(self) -> None:
        _require_positive_int("ModelConfig.vocab_size", self.vocab_size)
        _require_positive_int("ModelConfig.sequence_length", self.sequence_length)
        _require_positive_int("ModelConfig.d_model", self.d_model)
        _require_positive_int("ModelConfig.n_heads", self.n_heads)
        _require_positive_int("ModelConfig.n_layers", self.n_layers)
        _require_positive_int("ModelConfig.mlp_hidden", self.mlp_hidden)
        assert self.d_model is not None
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "ModelConfig.d_model must be divisible by ModelConfig.n_heads"
            )


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
    d_model: int | None = None
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 1024
    dropout: float = 0.1
    position_encoding: str = "type_place"
    training_mode: str = "token_stream"
    training_format: str = "final_only"
    skip_overlong_examples: bool = False
    curriculum_name: str | None = None

    def __post_init__(self) -> None:
        _require_positive_int("TrainConfig.sequence_length", self.sequence_length)
        _require_positive_int("TrainConfig.batch_size", self.batch_size)
        _require_positive_int("TrainConfig.epochs", self.epochs)
        _require_positive_int("TrainConfig.log_interval", self.log_interval)
        _require_positive_int("TrainConfig.max_new_tokens", self.max_new_tokens)
        _require_positive_int("TrainConfig.seed", self.seed)
        _require_positive_int("TrainConfig.d_model", self.d_model)
        _require_positive_int("TrainConfig.n_heads", self.n_heads)
        _require_positive_int("TrainConfig.n_layers", self.n_layers)
        _require_positive_int("TrainConfig.mlp_hidden", self.mlp_hidden)
        assert self.d_model is not None
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "TrainConfig.d_model must be divisible by TrainConfig.n_heads"
            )


def _require_positive_int(name: str, value: int | None) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
