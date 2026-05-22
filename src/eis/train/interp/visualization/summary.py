from __future__ import annotations

from collections.abc import Callable

import torch

from ..hooks import ActivationCapture


def summary_report(token_lookup: Callable[[int], str], capture: ActivationCapture) -> None:
    print("\n" + "=" * 60)
    print("MECHANISTIC INTERPRETATION SUMMARY")
    print("=" * 60)
    if capture.input_ids is not None:
        print(f"\nInput shape: {capture.input_ids.shape}")
        print(f"Sequence length: {capture.sequence_length}")
        decoded = [token_lookup(token.item()) for token in capture.input_ids[0]]
        print(f"Input tokens: {' '.join(decoded)}")
    print(f"\n{len(capture.layer_outputs)} transformer layers captured")
    print(
        f"  - Token embeddings: {capture.token_embeds.shape if capture.token_embeds is not None else 'Not captured'}"
    )
    print(
        f"  - Pos embeddings: {capture.pos_embeds.shape if capture.pos_embeds is not None else 'Not captured'}"
    )
    for i, out in enumerate(capture.layer_outputs):
        print(f"  - Layer {i} output: {out.shape}")
    if capture.logits is not None:
        print(f"\nFinal logits: {capture.logits.shape}")
        probs = torch.softmax(capture.logits[0, -1], dim=-1)
        top_k = torch.topk(probs, k=3)
        print("  Top 3 tokens at last position:")
        for val, idx in zip(top_k.values, top_k.indices, strict=False):
            print(f"    - Token {token_lookup(idx.item())}: {val.item():.3f}")
    print("\n" + "=" * 60)


def interactive_explore(_capture: ActivationCapture) -> None:
    print("\n" + "=" * 60)
    print("MECHANISTIC INTERPRETER - Interactive Mode")
    print("=" * 60)
    print("\nUse these methods to explore:")
    print("  visualizer.activation_heatmap(capture, which='layer_outputs')")
    print("  visualizer.activation_heatmap(capture, which='layer_outputs', layer_idx=0)")
    print("  visualizer.attention_patterns(capture)")
    print("  visualizer.attention_patterns(capture, layer_idx=0)")
    print("  visualizer.layer_transition(capture, layer_idx=0)")
    print("  visualizer.token_embeddings(capture)")
    print("  visualizer.logit_trajectory(capture)")
    print("  visualizer.mlp_contribution(capture)")
    print("  visualizer.token_position_influence(capture, target_pos=5)")
    print("  visualizer.summary_report(capture)")
    print("=" * 60 + "\n")
