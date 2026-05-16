"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.core import load_checkpoint, save_checkpoint, train_model

__all__ = ["load_checkpoint", "save_checkpoint", "train_model"]
