"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.model import SmallCausalTransformer, TransformerBlock

__all__ = ["SmallCausalTransformer", "TransformerBlock"]
