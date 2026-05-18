from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from eur_ts.config import ModelConfig
from eur_ts.trainer.data import (
    ArithmeticTokenizer,
    PLACE_VOCAB_SIZE,
    POSITION_ENCODING_TYPE_PLACE,
    TOKEN_TYPE_VOCAB_SIZE,
)
from eur_ts.trainer.model import SmallCausalTransformer
from eur_ts.trainer.training.checkpointing import load_checkpoint_payload


@dataclass(frozen=True)
class CheckpointArtifacts:
    tokenizer: ArithmeticTokenizer
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    metadata: dict[str, Any]
    position_encoding: str


def get_hooked_model(checkpoint_path: Path, device: str = "cpu") -> None:
    load_hooked_resources(checkpoint_path, device=device)
    return None


def load_checkpoint_artifacts(
    checkpoint_path: Path,
    device: str = "cpu",
) -> CheckpointArtifacts:
    payload = load_checkpoint_payload(checkpoint_path, torch.device(device))
    tokenizer_state = payload.get("tokenizer")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("checkpoint is missing tokenizer state")
    tokenizer = ArithmeticTokenizer.from_state(
        cast(dict[str, list[str]], tokenizer_state)
    )

    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        legacy_model = payload.get("model")
        if isinstance(legacy_model, SmallCausalTransformer):
            model_state = cast(dict[str, torch.Tensor], legacy_model.state_dict())
        else:
            raise ValueError("checkpoint is missing model_state")

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        legacy_config = payload.get("config")
        if isinstance(legacy_config, ModelConfig):
            model_config = cast(dict[str, Any], legacy_config.__dict__)
        else:
            raise ValueError("checkpoint is missing model_config")

    normalized_model_config = _normalize_model_config(
        cast(dict[str, Any], model_config),
        vocab_size=tokenizer.vocab_size,
    )
    position_encoding = cast(str, normalized_model_config.get("position_encoding"))
    metadata: dict[str, Any] = {
        "epoch": payload.get("epoch"),
        "exact_match": payload.get("exact_match"),
        "val_loss": payload.get("val_loss"),
        "train_loss": payload.get("train_loss"),
        "model_config": normalized_model_config,
        "train_config": payload.get("train_config"),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
    }
    return CheckpointArtifacts(
        tokenizer=tokenizer,
        model_state=cast(dict[str, torch.Tensor], model_state),
        model_config=normalized_model_config,
        metadata=metadata,
        position_encoding=position_encoding,
    )


def load_hooked_resources(
    checkpoint_path: Path,
    device: str = "cpu",
) -> None:
    del checkpoint_path, device
    raise ValueError(
        "TransformerLens backend support is unavailable for type_place checkpoints"
    )


def load_native_resources(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer, dict[str, Any]]:
    artifacts = load_checkpoint_artifacts(checkpoint_path, device=device)
    model = SmallCausalTransformer(_build_model_config(artifacts.model_config))
    model.load_state_dict(artifacts.model_state)
    model.to(device)
    model.eval()

    return model, artifacts.tokenizer, artifacts.metadata


def _build_model_config(config_dict: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        vocab_size=int(config_dict["vocab_size"]),
        sequence_length=int(config_dict.get("sequence_length", 64)),
        d_model=int(config_dict["d_model"]),
        n_heads=int(config_dict["n_heads"]),
        n_layers=int(config_dict["n_layers"]),
        mlp_hidden=int(config_dict["mlp_hidden"]),
        dropout=float(config_dict["dropout"]),
        position_encoding=cast(str, config_dict.get("position_encoding")),
        token_type_vocab_size=int(
            config_dict.get("token_type_vocab_size", TOKEN_TYPE_VOCAB_SIZE)
        ),
        place_vocab_size=int(config_dict.get("place_vocab_size", PLACE_VOCAB_SIZE)),
    )


def _normalize_model_config(
    config_dict: dict[str, Any], *, vocab_size: int
) -> dict[str, Any]:
    return {
        **config_dict,
        "vocab_size": int(config_dict.get("vocab_size", vocab_size)),
        "sequence_length": int(config_dict.get("sequence_length", 64)),
        "d_model": int(config_dict["d_model"]),
        "n_heads": int(config_dict["n_heads"]),
        "n_layers": int(config_dict["n_layers"]),
        "mlp_hidden": int(config_dict["mlp_hidden"]),
        "dropout": float(config_dict["dropout"]),
        "position_encoding": _required_position_encoding(config_dict),
        "token_type_vocab_size": int(
            config_dict.get("token_type_vocab_size", TOKEN_TYPE_VOCAB_SIZE)
        ),
        "place_vocab_size": int(config_dict.get("place_vocab_size", PLACE_VOCAB_SIZE)),
    }


def _required_position_encoding(config_dict: dict[str, Any]) -> str:
    value = config_dict.get("position_encoding")
    if value != POSITION_ENCODING_TYPE_PLACE:
        raise ValueError(f"unsupported checkpoint position encoding: {value!r}")
    return cast(str, value)
