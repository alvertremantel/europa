from __future__ import annotations

import matplotlib.pyplot as plt
from torch import Tensor

from ..hooks import ActivationCapture


def activation_heatmap(
    capture: ActivationCapture,
    which: str = "layer_outputs",
    layer_idx: int | None = None,
) -> None:
    activations = getattr(capture, which, None)
    if activations is None:
        print(f"No {which} captured")
        return
    n_layers = len(activations)
    if layer_idx is None:
        _activation_overview(activations, which, n_layers)
        return
    _activation_detailed(activations, which, layer_idx)


def _activation_overview(activations: list[Tensor], which: str, n_layers: int) -> None:
    fig, axes = plt.subplots(1, n_layers, figsize=(15, 3), squeeze=False)
    fig.suptitle(f"{which} - Max Activation per Neuron", fontsize=14)
    for layer_idx, act in enumerate(activations):
        ax = axes[0, layer_idx]
        max_acts = act[0].max(dim=0).values
        ax.imshow(max_acts.cpu().numpy().reshape(1, -1), cmap="hot", aspect="auto")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Neuron dimension")
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def _activation_detailed(activations: list[Tensor], which: str, layer_idx: int) -> None:
    act = activations[layer_idx]
    act_np = act[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(act_np.T, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neuron dimension")
    ax.set_title(f"{which} - Layer {layer_idx}\n(Rows=neurons, Columns=sequence)")
    plt.colorbar(im, ax=ax, label="Activation magnitude")
    plt.tight_layout()
    plt.show()


def mlp_contribution(capture: ActivationCapture) -> None:
    if not capture.mlp_outputs:
        print("No MLP outputs captured")
        return
    fig, axes = plt.subplots(1, len(capture.mlp_outputs), figsize=(15, 3), squeeze=False)
    fig.suptitle("MLP Output Activations per Layer", fontsize=14)
    for layer_idx, mlp_out in enumerate(capture.mlp_outputs):
        ax = axes[0, layer_idx]
        mlp_np = mlp_out[0].cpu().numpy()
        ax.imshow(mlp_np.T, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Token position")
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def layer_transition(capture: ActivationCapture, layer_idx: int) -> None:
    if layer_idx >= len(capture.layer_inputs) or layer_idx >= len(capture.layer_outputs):
        print(f"Layer {layer_idx} not captured")
        return
    inp = capture.layer_inputs[layer_idx][0].cpu().numpy()
    out = capture.layer_outputs[layer_idx][0].cpu().numpy()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = -2, 2
    ax1.imshow(inp.T, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Layer {layer_idx} INPUT")
    ax1.set_ylabel("Neuron")
    ax1.set_xlabel("Position")
    ax2.imshow(out.T, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
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


def token_position_influence(capture: ActivationCapture, target_pos: int) -> None:
    n_layers = len(capture.layer_outputs)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2 * n_layers))
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(f"Layer-wise activation at position {target_pos}", fontsize=14)
    for layer_idx, layer_out in enumerate(capture.layer_outputs):
        acts = layer_out[0, target_pos, :].cpu().numpy()
        ax = axes[layer_idx]
        ax.bar(range(len(acts)), acts)
        ax.set_ylabel("Activation")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlim([0, len(acts)])
    plt.tight_layout()
    plt.show()
