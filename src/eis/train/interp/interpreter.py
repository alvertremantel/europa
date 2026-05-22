"""
Easy-to-use wrapper for mechanistic interpretation.
Loads a model and enables interactive visualization of its computation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ..core import load_checkpoint
from ..data import ArithmeticTokenizer
from .hooks import HookRegistry
from .visualizer import InterpreterVisualizer


class MechanisticInterpreter:
    """Load and interpret a trained model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        self.model, self.tokenizer = load_checkpoint(self.checkpoint_path, self.device)
        self.model = self.model.to(self.device)
        self.config = self.model.config
        self.tokenizer: ArithmeticTokenizer

        self.model.eval()

        self.hook_registry = HookRegistry(self.model)
        self.visualizer = InterpreterVisualizer(
            tokenizer_vocab={
                index: token for index, token in enumerate(self.tokenizer.id_to_token)
            }
        )

    def forward_with_capture(self, input_ids: Tensor | list[int]) -> tuple:
        """
        Run the model and capture all internal activations.

        Args:
            input_ids: Token IDs [batch, seq] or list of ints

        Returns:
            (output_logits, capture) tuple
        """
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device=self.device)
        elif input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        # Reset capture
        self.hook_registry.capture.clear()
        self.hook_registry.capture.input_ids = input_ids
        self.hook_registry.capture.batch_size = input_ids.shape[0]
        self.hook_registry.capture.sequence_length = input_ids.shape[1]
        digit_place_values = torch.tensor(
            [
                self.tokenizer.fixed_meaning_digit_place_values_for_token_ids(row)
                for row in input_ids.detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(input_ids, digit_place_values)

        return logits, self.hook_registry.capture

    def visualize_summary(self) -> None:
        """Print a summary of the last forward pass."""
        self.visualizer.summary_report(self.hook_registry.capture)

    def visualize_activations(self, layer_idx: int | None = None) -> None:
        """
        Visualize neuron activations through layers.

        Args:
            layer_idx: Show specific layer, or None for all
        """
        if layer_idx is None:
            self.visualizer.activation_heatmap(
                self.hook_registry.capture, which="layer_outputs"
            )
        else:
            self.visualizer.activation_heatmap(
                self.hook_registry.capture, which="layer_outputs", layer_idx=layer_idx
            )

    def visualize_layer_transition(self, layer_idx: int) -> None:
        """Show how a layer transforms its input."""
        self.visualizer.layer_transition(self.hook_registry.capture, layer_idx)

    def visualize_attention(self) -> None:
        """Visualize attention patterns."""
        self.visualizer.attention_patterns(self.hook_registry.capture)

    def visualize_embeddings(self) -> None:
        """Show token embeddings."""
        self.visualizer.token_embeddings(self.hook_registry.capture)

    def visualize_logits(self) -> None:
        """Show how logits evolve."""
        self.visualizer.logit_trajectory(self.hook_registry.capture)

    def visualize_mlp(self) -> None:
        """Show MLP transformations."""
        self.visualizer.mlp_contribution(self.hook_registry.capture)

    def visualize_position_influence(self, pos: int) -> None:
        """Show what influences a specific position."""
        self.visualizer.token_position_influence(self.hook_registry.capture, pos)

    def explore_step_by_step(self, input_ids: Tensor | list[int]) -> None:
        """
        Interactive step-by-step exploration mode.
        Run a forward pass and launch exploration interface.
        """
        _, capture = self.forward_with_capture(input_ids)
        self.visualizer.interactive_explore(capture)

    def cleanup(self) -> None:
        """Remove all hooks."""
        self.hook_registry.remove_hooks()

    def __enter__(self) -> MechanisticInterpreter:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()
