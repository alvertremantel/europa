"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.hooks import ActivationCapture, HookRegistry

__all__ = ["ActivationCapture", "HookRegistry"]
