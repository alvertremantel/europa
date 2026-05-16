"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.training.state import capture_rng_state, restore_rng_state

__all__ = ["capture_rng_state", "restore_rng_state"]
