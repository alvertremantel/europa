from __future__ import annotations

from pathlib import Path

import torch

from .data import ArithmeticTokenizer
from .model import SmallCausalTransformer
from .training.checkpointing import load_model_checkpoint
from .training.loop import train_model


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer]:
    return load_model_checkpoint(checkpoint_path, device)


__all__ = ["load_checkpoint", "train_model"]
