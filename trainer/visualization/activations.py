"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.visualization.activations import (
    activation_heatmap,
    layer_transition,
    mlp_contribution,
    token_position_influence,
)

__all__ = [
    "activation_heatmap",
    "layer_transition",
    "mlp_contribution",
    "token_position_influence",
]
