from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np


class PredictionSummary(TypedDict):
    token: str
    confidence: float
    logit: float


def build_top_prediction_summaries(
    *,
    probs: np.ndarray,
    logits: np.ndarray,
    tokens_by_id: list[str],
    top_k: int = 5,
) -> tuple[list[PredictionSummary], list[list[PredictionSummary]]]:
    top_predictions: list[PredictionSummary] = []
    top_k_predictions: list[list[PredictionSummary]] = []

    top_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
    for position, indices in enumerate(top_indices):
        ranked: list[PredictionSummary] = []
        for token_index in indices:
            ranked.append(
                {
                    "token": tokens_by_id[int(token_index)],
                    "confidence": float(probs[position, token_index]),
                    "logit": float(logits[position, token_index]),
                }
            )
        top_k_predictions.append(ranked)
        top_predictions.append(ranked[0])

    return top_predictions, top_k_predictions


def build_attention_summary(
    *, attention_by_layer: list[np.ndarray], tokens: list[str]
) -> dict[str, Any]:
    head_summaries: list[list[dict[str, Any]]] = []

    for layer_attention in attention_by_layer:
        layer_heads: list[dict[str, Any]] = []
        for head_attention in layer_attention:
            query_count, key_count = head_attention.shape
            clipped = np.clip(head_attention, 1e-12, 1.0)
            entropy = float((-clipped * np.log2(clipped)).sum(axis=-1).mean())
            max_index = int(np.argmax(head_attention))
            query_index, key_index = divmod(max_index, key_count)
            diagonal_extent = min(query_count, key_count)
            diagonal_mean = float(
                np.diag(head_attention[:diagonal_extent, :diagonal_extent]).mean()
            )

            layer_heads.append(
                {
                    "entropy": entropy,
                    "max_weight": float(head_attention.max()),
                    "mean_diagonal": diagonal_mean,
                    "strongest_pair": {
                        "query_index": query_index,
                        "key_index": key_index,
                        "query_token": tokens[query_index],
                        "key_token": tokens[key_index],
                        "weight": float(head_attention[query_index, key_index]),
                    },
                }
            )
        head_summaries.append(layer_heads)

    return {"heads": head_summaries}


def build_activation_summary(*, stacked_activations: np.ndarray) -> dict[str, Any]:
    token_layer_norms = np.linalg.norm(stacked_activations, axis=-1)
    token_layer_max_abs = np.max(np.abs(stacked_activations), axis=-1)

    return {
        "token_layer_l2": token_layer_norms.tolist(),
        "token_layer_max_abs": token_layer_max_abs.tolist(),
        "layer_mean_l2": token_layer_norms.mean(axis=0).tolist(),
        "layer_peak_l2": token_layer_norms.max(axis=0).tolist(),
        "token_peak_l2": token_layer_norms.max(axis=1).tolist(),
        "global_max_abs": float(token_layer_max_abs.max())
        if token_layer_max_abs.size
        else 0.0,
    }


def summarize_checkpoint(
    *, checkpoint_path: str, device: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "path": checkpoint_path,
        "device": device,
        "epoch": metadata.get("epoch"),
        "exact_match": _optional_float(metadata.get("exact_match")),
        "val_loss": _optional_float(metadata.get("val_loss")),
        "train_loss": _optional_float(metadata.get("train_loss")),
        "checkpoint_schema_version": metadata.get("checkpoint_schema_version"),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
