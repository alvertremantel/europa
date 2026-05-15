from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainConfig
from .data import ArithmeticTokenizer
from .model import SmallCausalTransformer
from .training.checkpointing import (
    load_model_checkpoint,
    save_checkpoint_payload_for_compat,
)
from .training.loop import train_model


def save_checkpoint(
    output_dir: Path,
    file_name: str,
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    train_config: TrainConfig,
    epoch: int,
    val_loss: float,
    exact_match: float,
) -> None:
    save_checkpoint_payload_for_compat(
        output_dir=output_dir,
        file_name=file_name,
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        epoch=epoch,
        val_loss=val_loss,
        exact_match=exact_match,
    )


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer]:
    return load_model_checkpoint(checkpoint_path, device)


__all__ = ["load_checkpoint", "save_checkpoint", "train_model"]
