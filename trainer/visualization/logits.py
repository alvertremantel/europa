"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.visualization.logits import logit_trajectory, token_embeddings

__all__ = ["logit_trajectory", "token_embeddings"]
