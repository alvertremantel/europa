from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from ..hooks import ActivationCapture


def token_embeddings(capture: ActivationCapture) -> None:
    if capture.token_embeds is None:
        print("No token embeddings captured")
        return
    embeds_np = capture.token_embeds[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(embeds_np.T, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Embedding dimension")
    ax.set_title("Token Embeddings (Before Transformer)")
    plt.colorbar(im, ax=ax, label="Embedding value")
    plt.tight_layout()
    plt.show()


def logit_trajectory(capture: ActivationCapture) -> None:
    if capture.logits is None:
        print("No logits captured")
        return
    logits = capture.logits[0].cpu().detach()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    top1_conf = torch.softmax(logits, dim=-1).max(dim=-1).values
    ax1.plot(top1_conf.numpy(), marker="o", markersize=4, label="Top-1 confidence")
    ax1.set_ylabel("Softmax probability")
    ax1.set_xlabel("Token position")
    ax1.set_title("Model Confidence Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    ax2.plot(entropy.numpy(), marker="s", markersize=4, color="orange")
    ax2.set_ylabel("Entropy (nats)")
    ax2.set_xlabel("Token position")
    ax2.set_title("Prediction Uncertainty Trajectory")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
