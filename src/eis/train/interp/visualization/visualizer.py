from __future__ import annotations

from ..hooks import ActivationCapture

from .activations import (
    activation_heatmap,
    layer_transition,
    mlp_contribution,
    token_position_influence,
)
from .attention import attention_patterns
from .base import VisualizerBase
from .logits import logit_trajectory, token_embeddings
from .summary import interactive_explore, summary_report


class InterpreterVisualizer(VisualizerBase):
    def attention_patterns(
        self,
        capture: ActivationCapture,
        layer_idx: int | None = None,
        head_idx: int | None = None,
    ) -> None:
        attention_patterns(self, capture, layer_idx, head_idx)

    def activation_heatmap(
        self,
        capture: ActivationCapture,
        which: str = "layer_outputs",
        layer_idx: int | None = None,
    ) -> None:
        activation_heatmap(capture, which, layer_idx)

    def token_embeddings(self, capture: ActivationCapture) -> None:
        token_embeddings(capture)

    def logit_trajectory(self, capture: ActivationCapture) -> None:
        logit_trajectory(capture)

    def mlp_contribution(self, capture: ActivationCapture) -> None:
        mlp_contribution(capture)

    def layer_transition(self, capture: ActivationCapture, layer_idx: int) -> None:
        layer_transition(capture, layer_idx)

    def token_position_influence(self, capture: ActivationCapture, target_pos: int) -> None:
        token_position_influence(capture, target_pos)

    def summary_report(self, capture: ActivationCapture) -> None:
        summary_report(self._token_str, capture)

    def interactive_explore(self, capture: ActivationCapture) -> None:
        interactive_explore(capture)
