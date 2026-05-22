"""Compatibility facade for interpretability hooks."""

from .interp.hooks import ActivationCapture, HookRegistry

__all__ = ["ActivationCapture", "HookRegistry"]
