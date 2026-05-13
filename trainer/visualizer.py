"""
Visualization utilities for mechanistic interpretability.
Provides intuitive visual interfaces for understanding model computation.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch import Tensor

from .hooks import ActivationCapture


class InterpreterVisualizer:
    """Visualize model internals during inference."""

    def __init__(self, tokenizer_vocab: dict[int, str] | None = None) -> None:
        self.tokenizer_vocab = tokenizer_vocab or {}
        self.fig_count = 0

    def _token_str(self, token_id: int) -> str:
        """Convert token ID to readable string."""
        if token_id in self.tokenizer_vocab:
            return self.tokenizer_vocab[token_id]
        return f"[{token_id}]"

    def attention_patterns(
        self,
        capture: ActivationCapture,
        layer_idx: int | None = None,
        head_idx: int | None = None,
    ) -> None:
        """
        Visualize attention patterns.

        Args:
            capture: Captured activations
            layer_idx: Specific layer to show, or None for all
            head_idx: Specific head to show, or None for all
        """
        if not capture.attention_outputs:
            print("No attention outputs captured")
            return

        if layer_idx is None:
            # Show overview of all layers
            self._attention_overview(capture)
        else:
            # Show detailed view of single layer
            self._attention_detailed(capture, layer_idx, head_idx)

    def _attention_overview(self, capture: ActivationCapture) -> None:
        """Show attention patterns from all layers at once."""
        n_layers = len(capture.attention_outputs)
        n_heads = 4  # Known from config
        seq_len = capture.sequence_length

        fig, axes = plt.subplots(
            n_layers, n_heads, figsize=(12, 2 * n_layers), squeeze=False
        )
        fig.suptitle("Attention Patterns (All Layers × All Heads)", fontsize=14)

        for layer_idx in range(n_layers):
            attn_output = capture.attention_outputs[layer_idx]  # [batch, seq, d_model]
            # Approximate attention pattern from output (real weights not captured)
            # We can compute the attention pattern from the queries/keys post-hoc
            for head_idx in range(n_heads):
                ax = axes[layer_idx, head_idx]
                # Placeholder: show activation magnitude instead
                ax.set_title(f"L{layer_idx}H{head_idx}", fontsize=10)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _attention_detailed(
        self, capture: ActivationCapture, layer_idx: int, head_idx: int | None = None
    ) -> None:
        """Show detailed attention heatmap for a single layer."""
        print(f"Detailed attention visualization for layer {layer_idx}")
        print("(Note: Full attention weights require model modification to capture)")

    def activation_heatmap(
        self,
        capture: ActivationCapture,
        which: str = "layer_outputs",
        layer_idx: int | None = None,
    ) -> None:
        """
        Show neuron activation heatmaps.

        Args:
            capture: Captured activations
            which: "layer_outputs", "attention_outputs", "mlp_outputs", "norm_1_outputs", or "norm_2_outputs"
            layer_idx: Specific layer to highlight, or None for all
        """
        activations = getattr(capture, which, None)
        if activations is None:
            print(f"No {which} captured")
            return

        n_layers = len(activations)

        if layer_idx is None:
            # Overview: show max activation across sequence for each layer
            self._activation_overview(activations, which, n_layers)
        else:
            # Detailed: show full activation matrix for one layer
            self._activation_detailed(activations, which, layer_idx)

    def _activation_overview(
        self, activations: list[Tensor], which: str, n_layers: int
    ) -> None:
        """Show max activation per neuron across all layers."""
        fig, axes = plt.subplots(1, n_layers, figsize=(15, 3), squeeze=False)
        fig.suptitle(f"{which} - Max Activation per Neuron", fontsize=14)

        for layer_idx, act in enumerate(activations):
            ax = axes[0, layer_idx]
            # [batch, seq, d_model] -> take max across batch and seq
            max_acts = act[0].max(dim=0).values  # [d_model]
            ax.imshow(max_acts.cpu().numpy().reshape(1, -1), cmap="hot", aspect="auto")
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("Neuron dimension")
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def _activation_detailed(
        self, activations: list[Tensor], which: str, layer_idx: int
    ) -> None:
        """Show full activation heatmap for one layer."""
        act = activations[layer_idx]  # [batch, seq, d_model]
        act_np = act[0].cpu().numpy()  # [seq, d_model]

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(act_np.T, cmap="RdBu_r", aspect="auto")
        ax.set_xlabel("Token position")
        ax.set_ylabel("Neuron dimension")
        ax.set_title(f"{which} - Layer {layer_idx}\n(Rows=neurons, Columns=sequence)")
        plt.colorbar(im, ax=ax, label="Activation magnitude")
        plt.tight_layout()
        plt.show()

    def token_embeddings(self, capture: ActivationCapture) -> None:
        """
        Show how tokens are embedded (first layer).
        """
        if capture.token_embeds is None:
            print("No token embeddings captured")
            return

        embeds = capture.token_embeds[0]  # [seq_len, d_model]
        embeds_np = embeds.cpu().numpy()

        # Show as heatmap
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(embeds_np.T, cmap="RdBu_r", aspect="auto")
        ax.set_xlabel("Token position")
        ax.set_ylabel("Embedding dimension")
        ax.set_title("Token Embeddings (Before Transformer)")
        plt.colorbar(im, ax=ax, label="Embedding value")
        plt.tight_layout()
        plt.show()

    def logit_trajectory(self, capture: ActivationCapture) -> None:
        """
        Show how logits evolve through the model.
        Traces the most likely token at each position.
        """
        if capture.logits is None:
            print("No logits captured")
            return

        logits = capture.logits[0].cpu().detach()  # [seq_len, vocab_size]
        seq_len = logits.shape[0]

        # Get top-k predictions at each position
        topk_vals, topk_ids = torch.topk(logits, k=3, dim=-1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Plot 1: Confidence in top prediction over sequence
        top1_conf = torch.softmax(logits, dim=-1).max(dim=-1).values
        ax1.plot(top1_conf.numpy(), marker="o", markersize=4, label="Top-1 confidence")
        ax1.set_ylabel("Softmax probability")
        ax1.set_xlabel("Token position")
        ax1.set_title("Model Confidence Trajectory")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Entropy over sequence
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        ax2.plot(entropy.numpy(), marker="s", markersize=4, color="orange")
        ax2.set_ylabel("Entropy (nats)")
        ax2.set_xlabel("Token position")
        ax2.set_title("Prediction Uncertainty Trajectory")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def mlp_contribution(self, capture: ActivationCapture) -> None:
        """Show how MLP layers transform data."""
        if not capture.mlp_outputs:
            print("No MLP outputs captured")
            return

        fig, axes = plt.subplots(
            1, len(capture.mlp_outputs), figsize=(15, 3), squeeze=False
        )
        fig.suptitle("MLP Output Activations per Layer", fontsize=14)

        for layer_idx, mlp_out in enumerate(capture.mlp_outputs):
            ax = axes[0, layer_idx]
            # [batch, seq, d_model]
            mlp_np = mlp_out[0].cpu().numpy()  # [seq, d_model]
            im = ax.imshow(mlp_np.T, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("Token position")
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def layer_transition(self, capture: ActivationCapture, layer_idx: int) -> None:
        """
        Show how a single layer transforms its input.
        Side-by-side comparison of input vs output.
        """
        if layer_idx >= len(capture.layer_inputs) or layer_idx >= len(
            capture.layer_outputs
        ):
            print(f"Layer {layer_idx} not captured")
            return

        inp = capture.layer_inputs[layer_idx][0].cpu().numpy()  # [seq, d_model]
        out = capture.layer_outputs[layer_idx][0].cpu().numpy()  # [seq, d_model]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        vmin, vmax = -2, 2
        im1 = ax1.imshow(inp.T, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Layer {layer_idx} INPUT")
        ax1.set_ylabel("Neuron")
        ax1.set_xlabel("Position")

        im2 = ax2.imshow(out.T, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Layer {layer_idx} OUTPUT")
        ax2.set_ylabel("Neuron")
        ax2.set_xlabel("Position")

        delta = out - inp
        im3 = ax3.imshow(delta.T, cmap="RdBu_r", aspect="auto")
        ax3.set_title(f"Layer {layer_idx} CHANGE (Δ)")
        ax3.set_ylabel("Neuron")
        ax3.set_xlabel("Position")

        plt.colorbar(im3, ax=ax3, label="Change magnitude")
        plt.tight_layout()
        plt.show()

    def token_position_influence(
        self, capture: ActivationCapture, target_pos: int
    ) -> None:
        """
        Show which positions influence the output at target_pos.
        Useful for understanding attention flow.
        """
        n_layers = len(capture.layer_outputs)
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2 * n_layers))
        if n_layers == 1:
            axes = [axes]

        fig.suptitle(
            f"Layer-wise activation at position {target_pos}", fontsize=14
        )

        for layer_idx, layer_out in enumerate(capture.layer_outputs):
            acts = layer_out[0, target_pos, :].cpu().numpy()  # [d_model]
            ax = axes[layer_idx]
            ax.bar(range(len(acts)), acts)
            ax.set_ylabel("Activation")
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlim([0, len(acts)])

        plt.tight_layout()
        plt.show()

    def summary_report(self, capture: ActivationCapture) -> None:
        """Print a text summary of the captured computation."""
        print("\n" + "=" * 60)
        print("MECHANISTIC INTERPRETATION SUMMARY")
        print("=" * 60)

        if capture.input_ids is not None:
            print(f"\nInput shape: {capture.input_ids.shape}")
            print(f"Sequence length: {capture.sequence_length}")

        print(f"\n{len(capture.layer_outputs)} transformer layers captured")
        print(f"  - Token embeddings: {capture.token_embeds.shape if capture.token_embeds is not None else 'Not captured'}")
        print(
            f"  - Pos embeddings: {capture.pos_embeds.shape if capture.pos_embeds is not None else 'Not captured'}"
        )

        for i, out in enumerate(capture.layer_outputs):
            print(f"  - Layer {i} output: {out.shape}")

        if capture.logits is not None:
            print(f"\nFinal logits: {capture.logits.shape}")
            probs = torch.softmax(capture.logits[0], dim=-1)
            top_k = torch.topk(probs, k=3)
            print(f"  Top 3 tokens at last position:")
            for val, idx in zip(top_k.values, top_k.indices):
                print(f"    - Token {idx.item()}: {val.item():.3f}")

        print("\n" + "=" * 60)

    def interactive_explore(self, capture: ActivationCapture) -> None:
        """
        Launch an interactive exploration mode.
        This is a placeholder for a more sophisticated UI.
        """
        print(
            "\n"
            + "=" * 60
        )
        print("MECHANISTIC INTERPRETER - Interactive Mode")
        print("=" * 60)
        print(
            "\nUse these methods to explore:"
        )
        print("  visualizer.activation_heatmap(capture, which='layer_outputs')")
        print("  visualizer.activation_heatmap(capture, which='layer_outputs', layer_idx=0)")
        print("  visualizer.attention_patterns(capture)")
        print(
            "  visualizer.attention_patterns(capture, layer_idx=0)"
        )
        print("  visualizer.layer_transition(capture, layer_idx=0)")
        print("  visualizer.token_embeddings(capture)")
        print("  visualizer.logit_trajectory(capture)")
        print("  visualizer.mlp_contribution(capture)")
        print("  visualizer.token_position_influence(capture, target_pos=5)")
        print("  visualizer.summary_report(capture)")
        print(
            "=" * 60 + "\n"
        )
