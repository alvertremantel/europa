"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.training import (
    CheckpointManager,
    build_checkpoint_payload,
    load_checkpoint_payload,
    load_model_checkpoint,
    save_checkpoint_payload,
    train_model,
)

__all__ = [
    "CheckpointManager",
    "build_checkpoint_payload",
    "load_checkpoint_payload",
    "load_model_checkpoint",
    "save_checkpoint_payload",
    "train_model",
]
