"""Deterministic PNG rendering for ITS export bundles."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eur_is.backend.schemas import AnalyzeResponse
from eur_is.export import layout


def render_png_files(
    analysis: AnalyzeResponse,
) -> tuple[dict[str, bytes], list[dict[str, str]]]:
    files: dict[str, bytes] = {}
    unavailable: list[dict[str, str]] = []

    files[layout.OVERVIEW_METRICS_ASSET_PATH] = _bar_png(
        "Overview metrics",
        ["answer_tokens", "layers", "heads"],
        [
            float(analysis.generated_answer.token_count),
            float(analysis.config.n_layers),
            float(analysis.config.n_heads),
        ],
    )
    files[layout.PROMPT_PREDICTIONS_ASSET_PATH] = _line_png(
        "Prompt prediction confidence",
        list(range(len(analysis.top_predictions))),
        [entry.confidence for entry in analysis.top_predictions],
        xlabel="token index",
        ylabel="confidence",
    )
    answer_confidences = [
        token.top_predictions[0].confidence if token.top_predictions else 0.0
        for token in analysis.generated_answer_top_k
    ]
    files[layout.GENERATED_ANSWER_TOPK_ASSET_PATH] = _bar_png(
        "Generated answer top-1 confidence",
        [str(index) for index in range(len(answer_confidences))],
        answer_confidences,
        xlabel="generated token index",
        ylabel="confidence",
    )
    files[layout.ACTIVATION_L2_ASSET_PATH] = _heatmap_png(
        np.array(analysis.activation_summary.token_layer_l2, dtype=float),
        title="Activation L2 heatmap",
        xlabel="layer",
        ylabel="token index",
        xlabels=[str(index) for index in range(analysis.config.n_layers)],
        ylabels=[str(index) for index in range(len(analysis.tokens))],
    )
    files[layout.ACTIVATION_MAX_ABS_ASSET_PATH] = _heatmap_png(
        np.array(analysis.activation_summary.token_layer_max_abs, dtype=float),
        title="Activation max-abs heatmap",
        xlabel="layer",
        ylabel="token index",
        xlabels=[str(index) for index in range(analysis.config.n_layers)],
        ylabels=[str(index) for index in range(len(analysis.tokens))],
    )
    files[layout.LOGIT_TRAJECTORY_ASSET_PATH] = _logit_trajectory_png(analysis)

    if analysis.attention_summary is not None:
        files[layout.ATTENTION_HEAD_SUMMARY_ASSET_PATH] = _attention_head_summary_png(
            analysis
        )
    else:
        files[layout.ATTENTION_UNAVAILABLE_ASSET_PATH] = _placeholder_png(
            "Attention summary unavailable"
        )
        unavailable.append(
            {
                "section": "attention_summary",
                "reason": "Attention summary is unavailable for this runtime.",
                "placeholder_asset": layout.ATTENTION_UNAVAILABLE_ASSET_PATH,
            }
        )

    if analysis.attention is not None:
        files[layout.ATTENTION_SELECTED_MAPS_ASSET_PATH] = _attention_map_png(analysis)
    else:
        files[layout.ATTENTION_MAPS_UNAVAILABLE_ASSET_PATH] = _placeholder_png(
            "Attention maps unavailable"
        )
        unavailable.append(
            {
                "section": "attention_maps",
                "reason": "Raw attention tensors are unavailable for this runtime.",
                "placeholder_asset": layout.ATTENTION_MAPS_UNAVAILABLE_ASSET_PATH,
            }
        )

    network = analysis.network
    if network is not None:
        files[layout.NETWORK_MLP_ASSET_PATH] = _network_mlp_png(network)
        files[layout.NETWORK_ATTENTION_ASSET_PATH] = _network_attention_png(network)
        files[layout.NETWORK_RESIDUAL_ASSET_PATH] = _network_residual_png(network)
    else:
        files[layout.NETWORK_UNAVAILABLE_ASSET_PATH] = _placeholder_png(
            "Network MLP analysis unavailable"
        )
        files[layout.NETWORK_ATTENTION_UNAVAILABLE_ASSET_PATH] = _placeholder_png(
            "Network attention analysis unavailable"
        )
        files[layout.NETWORK_RESIDUAL_UNAVAILABLE_ASSET_PATH] = _placeholder_png(
            "Network residual analysis unavailable"
        )
        unavailable.extend(
            [
                {
                    "section": "network_mlp",
                    "reason": "Network MLP analysis is unavailable for this runtime.",
                    "placeholder_asset": layout.NETWORK_UNAVAILABLE_ASSET_PATH,
                },
                {
                    "section": "network_attention",
                    "reason": "Network attention analysis is unavailable for this runtime.",
                    "placeholder_asset": layout.NETWORK_ATTENTION_UNAVAILABLE_ASSET_PATH,
                },
                {
                    "section": "network_residual",
                    "reason": "Network residual analysis is unavailable for this runtime.",
                    "placeholder_asset": layout.NETWORK_RESIDUAL_UNAVAILABLE_ASSET_PATH,
                },
            ]
        )
    return files, unavailable


def _save_figure(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def _bar_png(
    title: str,
    labels: list[str],
    values: list[float],
    xlabel: str = "",
    ylabel: str = "",
) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#7b68ee")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    return _save_figure(fig)


def _line_png(
    title: str, xs: list[int], ys: list[float], xlabel: str, ylabel: str
) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o", color="#4ecdc4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    return _save_figure(fig)


def _heatmap_png(
    data: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: list[str],
    ylabels: list[str],
) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(image, ax=ax)
    return _save_figure(fig)


def _placeholder_png(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=13)
    return _save_figure(fig)


def _logit_trajectory_png(analysis: AnalyzeResponse) -> bytes:
    values = [entry.logit or 0.0 for entry in analysis.top_predictions]
    return _line_png(
        "Top-prediction logit trajectory",
        list(range(len(values))),
        values,
        xlabel="token index",
        ylabel="logit",
    )


def _attention_head_summary_png(analysis: AnalyzeResponse) -> bytes:
    assert analysis.attention_summary is not None
    labels: list[str] = []
    values: list[float] = []
    for layer_index, heads in enumerate(analysis.attention_summary.heads):
        for head_index, head in enumerate(heads):
            labels.append(f"L{layer_index}H{head_index}")
            values.append(head.max_weight)
    return _bar_png("Attention head max weight", labels, values, ylabel="weight")


def _attention_map_png(analysis: AnalyzeResponse) -> bytes:
    assert analysis.attention is not None
    first = np.array(analysis.attention[0][0], dtype=float)
    return _heatmap_png(
        first,
        title="Attention map (layer 0 head 0)",
        xlabel="key index",
        ylabel="query index",
        xlabels=[str(index) for index in range(first.shape[1])],
        ylabels=[str(index) for index in range(first.shape[0])],
    )


def _network_mlp_png(network: dict[str, Any]) -> bytes:
    layers = network.get("mlp", {}).get("layers", [])
    if not layers:
        return _placeholder_png("Network MLP analysis contains no layers")
    rows = []
    for layer in layers:
        token_values = []
        for token in layer.get("tokens", []):
            token_values.append(
                float(
                    token.get("max_abs_activation") or token.get("output_norm") or 0.0
                )
            )
        rows.append(token_values or [0.0])
    max_width = max(len(row) for row in rows)
    padded = np.array(
        [row + [0.0] * (max_width - len(row)) for row in rows], dtype=float
    )
    return _heatmap_png(
        padded,
        title="Network MLP token activity",
        xlabel="token index",
        ylabel="layer",
        xlabels=[str(index) for index in range(padded.shape[1])],
        ylabels=[str(index) for index in range(padded.shape[0])],
    )


def _network_attention_png(network: dict[str, Any]) -> bytes:
    layers = network.get("attention", {}).get("layers", [])
    if not layers:
        return _placeholder_png("Network attention analysis contains no layers")
    labels: list[str] = []
    values: list[float] = []
    for layer in layers:
        for head in layer.get("heads", []):
            labels.append(
                f"L{head.get('layer', layer.get('layer'))}H{head.get('head')}"
            )
            values.append(float(head.get("max_weight") or 0.0))
    return _bar_png("Network attention activity", labels, values, ylabel="max weight")


def _network_residual_png(network: dict[str, Any]) -> bytes:
    layers = network.get("residual", {}).get("layers", [])
    if not layers:
        return _placeholder_png("Network residual analysis contains no layers")
    rows = []
    for layer in layers:
        rows.append(
            [float(token.get("norm") or 0.0) for token in layer.get("tokens", [])]
            or [0.0]
        )
    max_width = max(len(row) for row in rows)
    padded = np.array(
        [row + [0.0] * (max_width - len(row)) for row in rows], dtype=float
    )
    return _heatmap_png(
        padded,
        title="Network residual token norms",
        xlabel="token index",
        ylabel="layer",
        xlabels=[str(index) for index in range(padded.shape[1])],
        ylabels=[str(index) for index in range(padded.shape[0])],
    )
