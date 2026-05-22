from __future__ import annotations

import matplotlib.pyplot as plt

from ..hooks import ActivationCapture


def attention_patterns(
    visualizer: object,
    capture: ActivationCapture,
    layer_idx: int | None = None,
    head_idx: int | None = None,
) -> None:
    if not capture.attention_outputs:
        print("No attention outputs captured")
        return
    if layer_idx is None:
        _attention_overview(capture)
        return
    _attention_detailed(layer_idx, head_idx)


def _attention_overview(capture: ActivationCapture) -> None:
    n_layers = len(capture.attention_outputs)
    n_heads = 4
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(12, 2 * n_layers), squeeze=False)
    fig.suptitle("Attention Patterns (All Layers × All Heads)", fontsize=14)
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            ax.set_title(f"L{layer_idx}H{head_idx}", fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def _attention_detailed(layer_idx: int, head_idx: int | None = None) -> None:
    print(f"Detailed attention visualization for layer {layer_idx}")
    if head_idx is not None:
        print(f"Requested head {head_idx}")
    print("(Note: Full attention weights require model modification to capture)")
