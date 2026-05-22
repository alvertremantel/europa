"""CSV and JSONL export serializers."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from eis.app.backend.schemas import AnalyzeResponse
from eis.app.export import layout


def _csv_bytes(fieldnames: list[str], rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )


def build_table_files(analysis: AnalyzeResponse) -> dict[str, bytes]:
    files: dict[str, bytes] = {}

    token_rows = [
        {
            "token_index": index,
            "token": token,
            "prompt_index": int(index <= analysis.answer_position),
            "is_answer_position": int(index == analysis.answer_position),
        }
        for index, token in enumerate(analysis.tokens)
    ]
    files[layout.TOKENS_TABLE_PATH] = _csv_bytes(
        ["token_index", "token", "prompt_index", "is_answer_position"], token_rows
    )

    prompt_prediction_rows = []
    for token_index, (actual_token, prediction, top_k_predictions) in enumerate(
        zip(
            analysis.tokens,
            analysis.top_predictions,
            analysis.top_k_predictions,
            strict=False,
        )
    ):
        prompt_prediction_rows.append(
            {
                "token_index": token_index,
                "token": actual_token,
                "predicted_token": prediction.token,
                "confidence": prediction.confidence,
                "logit": prediction.logit,
                "top_k_tokens": " | ".join(entry.token for entry in top_k_predictions),
            }
        )
    files[layout.PROMPT_PREDICTIONS_TABLE_PATH] = _csv_bytes(
        [
            "token_index",
            "token",
            "predicted_token",
            "confidence",
            "logit",
            "top_k_tokens",
        ],
        prompt_prediction_rows,
    )

    generated_answer_rows = []
    for generated_token_index, token_info in enumerate(analysis.generated_answer_top_k):
        for rank, prediction in enumerate(token_info.top_predictions, start=1):
            generated_answer_rows.append(
                {
                    "generated_token_index": generated_token_index,
                    "token": token_info.token,
                    "rank": rank,
                    "predicted_token": prediction.token,
                    "confidence": prediction.confidence,
                    "logit": prediction.logit,
                }
            )
    files[layout.GENERATED_ANSWER_TOPK_TABLE_PATH] = _csv_bytes(
        [
            "generated_token_index",
            "token",
            "rank",
            "predicted_token",
            "confidence",
            "logit",
        ],
        generated_answer_rows,
    )

    activation_rows = []
    for token_index, (l2_row, max_abs_row) in enumerate(
        zip(
            analysis.activation_summary.token_layer_l2,
            analysis.activation_summary.token_layer_max_abs,
            strict=False,
        )
    ):
        for layer, (l2_value, max_abs_value) in enumerate(
            zip(l2_row, max_abs_row, strict=False)
        ):
            activation_rows.append(
                {
                    "token_index": token_index,
                    "token": analysis.tokens[token_index],
                    "layer": layer,
                    "l2": l2_value,
                    "max_abs": max_abs_value,
                }
            )
    files[layout.ACTIVATION_SUMMARY_TABLE_PATH] = _csv_bytes(
        ["token_index", "token", "layer", "l2", "max_abs"], activation_rows
    )

    logits_rows = [
        {
            "token_index": token_index,
            "token": analysis.tokens[token_index],
            "values": values,
        }
        for token_index, values in enumerate(analysis.logits)
    ]
    files[layout.LOGITS_TENSOR_PATH] = _jsonl_bytes(logits_rows)

    activation_tensor_rows = [
        {
            "token_index": token_index,
            "token": analysis.tokens[token_index],
            "layers": layers,
        }
        for token_index, layers in enumerate(analysis.activations)
    ]
    files[layout.ACTIVATIONS_TENSOR_PATH] = _jsonl_bytes(activation_tensor_rows)

    if analysis.attention_summary is not None:
        attention_head_rows = []
        for layer_index, layer_heads in enumerate(analysis.attention_summary.heads):
            for head_index, head in enumerate(layer_heads):
                attention_head_rows.append(
                    {
                        "layer": layer_index,
                        "head": head_index,
                        "entropy": head.entropy,
                        "max_weight": head.max_weight,
                        "mean_diagonal": head.mean_diagonal,
                        "query_index": head.strongest_pair.query_index,
                        "key_index": head.strongest_pair.key_index,
                        "query_token": head.strongest_pair.query_token,
                        "key_token": head.strongest_pair.key_token,
                        "weight": head.strongest_pair.weight,
                    }
                )
        files[layout.ATTENTION_HEAD_SUMMARY_TABLE_PATH] = _csv_bytes(
            [
                "layer",
                "head",
                "entropy",
                "max_weight",
                "mean_diagonal",
                "query_index",
                "key_index",
                "query_token",
                "key_token",
                "weight",
            ],
            attention_head_rows,
        )
    if analysis.attention is not None:
        attention_rows = []
        for layer_index, layer in enumerate(analysis.attention):
            for head_index, weights in enumerate(layer):
                attention_rows.append(
                    {"layer": layer_index, "head": head_index, "weights": weights}
                )
        files[layout.ATTENTION_TENSOR_PATH] = _jsonl_bytes(attention_rows)

    if analysis.network is not None:
        files.update(_build_network_tables(analysis))

    return files


def _build_network_tables(analysis: AnalyzeResponse) -> dict[str, bytes]:
    assert analysis.network is not None
    files: dict[str, bytes] = {}

    mlp_rows = []
    for layer_data in analysis.network.get("mlp", {}).get("layers", []):
        for token_data in layer_data.get("tokens", []):
            row = {
                "layer": layer_data.get("layer"),
                "availability": layer_data.get("availability"),
                "token_index": token_data.get("token_index"),
                "token": token_data.get("token"),
                "active_fraction_abs": token_data.get("active_fraction_abs"),
                "max_abs_activation": token_data.get("max_abs_activation"),
                "output_norm": token_data.get("output_norm"),
            }
            top_neurons = token_data.get("top_neurons", [])
            if top_neurons:
                for neuron in top_neurons:
                    mlp_rows.append(
                        {
                            **row,
                            "neuron_index": neuron.get("neuron_index"),
                            "value": neuron.get("value"),
                            "abs_value": neuron.get("abs_value"),
                        }
                    )
            else:
                mlp_rows.append(
                    {
                        **row,
                        "neuron_index": None,
                        "value": None,
                        "abs_value": None,
                    }
                )
    files[layout.NETWORK_MLP_TABLE_PATH] = _csv_bytes(
        [
            "layer",
            "availability",
            "token_index",
            "token",
            "active_fraction_abs",
            "max_abs_activation",
            "output_norm",
            "neuron_index",
            "value",
            "abs_value",
        ],
        mlp_rows,
    )

    attention_rows = []
    for layer_data in analysis.network.get("attention", {}).get("layers", []):
        for head_data in layer_data.get("heads", []):
            attention_rows.append(
                {
                    "layer": head_data.get("layer", layer_data.get("layer")),
                    "head": head_data.get("head"),
                    "availability": layer_data.get("availability"),
                    "mean_entropy": head_data.get("mean_entropy"),
                    "max_weight": head_data.get("max_weight"),
                    "self_attention_mass": head_data.get("self_attention_mass"),
                    "previous_token_mass": head_data.get("previous_token_mass"),
                    "query_index": head_data.get("strongest_pair", {}).get(
                        "query_index"
                    ),
                    "key_index": head_data.get("strongest_pair", {}).get("key_index"),
                }
            )
    files[layout.NETWORK_ATTENTION_TABLE_PATH] = _csv_bytes(
        [
            "layer",
            "head",
            "availability",
            "mean_entropy",
            "max_weight",
            "self_attention_mass",
            "previous_token_mass",
            "query_index",
            "key_index",
        ],
        attention_rows,
    )

    residual_rows = []
    for layer_data in analysis.network.get("residual", {}).get("layers", []):
        for token_data in layer_data.get("tokens", []):
            dimensions = token_data.get("top_dimensions", [])
            if dimensions:
                for dimension in dimensions:
                    residual_rows.append(
                        {
                            "layer": layer_data.get("layer"),
                            "availability": layer_data.get("availability"),
                            "token_index": token_data.get("token_index"),
                            "token": token_data.get("token"),
                            "norm": token_data.get("norm"),
                            "dimension": dimension.get("dimension"),
                            "value": dimension.get("value"),
                            "abs_value": dimension.get("abs_value"),
                        }
                    )
            else:
                residual_rows.append(
                    {
                        "layer": layer_data.get("layer"),
                        "availability": layer_data.get("availability"),
                        "token_index": token_data.get("token_index"),
                        "token": token_data.get("token"),
                        "norm": token_data.get("norm"),
                        "dimension": None,
                        "value": None,
                        "abs_value": None,
                    }
                )
    files[layout.NETWORK_RESIDUAL_TABLE_PATH] = _csv_bytes(
        [
            "layer",
            "availability",
            "token_index",
            "token",
            "norm",
            "dimension",
            "value",
            "abs_value",
        ],
        residual_rows,
    )
    return files
