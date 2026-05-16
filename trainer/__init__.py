"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer import load_checkpoint, train_model

__all__ = ["load_checkpoint", "train_model"]
