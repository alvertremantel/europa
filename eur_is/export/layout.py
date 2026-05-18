"""Canonical export bundle paths."""

from __future__ import annotations

MANIFEST_PATH = "manifest.json"
SUMMARY_PATH = "summary.md"
RAW_ANALYZE_RESPONSE_PATH = "raw/analyze-response.json"
RAW_HEALTH_PATH = "raw/health.json"
TABLES_DIR = "tables"
TENSORS_DIR = "tensors"
ASSETS_DIR = "assets"

TOKENS_TABLE_PATH = f"{TABLES_DIR}/tokens.csv"
PROMPT_PREDICTIONS_TABLE_PATH = f"{TABLES_DIR}/prompt_predictions.csv"
GENERATED_ANSWER_TOPK_TABLE_PATH = f"{TABLES_DIR}/generated_answer_topk.csv"
ATTENTION_HEAD_SUMMARY_TABLE_PATH = f"{TABLES_DIR}/attention_head_summary.csv"
ACTIVATION_SUMMARY_TABLE_PATH = f"{TABLES_DIR}/activation_summary_token_layer.csv"
NETWORK_MLP_TABLE_PATH = f"{TABLES_DIR}/network_mlp_tokens.csv"
NETWORK_ATTENTION_TABLE_PATH = f"{TABLES_DIR}/network_attention_heads.csv"
NETWORK_RESIDUAL_TABLE_PATH = f"{TABLES_DIR}/network_residual_tokens.csv"

LOGITS_TENSOR_PATH = f"{TENSORS_DIR}/logits.jsonl"
ACTIVATIONS_TENSOR_PATH = f"{TENSORS_DIR}/activations.jsonl"
ATTENTION_TENSOR_PATH = f"{TENSORS_DIR}/attention.jsonl"

OVERVIEW_METRICS_ASSET_PATH = f"{ASSETS_DIR}/overview_metrics.png"
PROMPT_PREDICTIONS_ASSET_PATH = f"{ASSETS_DIR}/prompt_predictions_confidence.png"
GENERATED_ANSWER_TOPK_ASSET_PATH = f"{ASSETS_DIR}/generated_answer_topk.png"
ACTIVATION_L2_ASSET_PATH = f"{ASSETS_DIR}/activation_l2_heatmap.png"
ACTIVATION_MAX_ABS_ASSET_PATH = f"{ASSETS_DIR}/activation_max_abs_heatmap.png"
LOGIT_TRAJECTORY_ASSET_PATH = f"{ASSETS_DIR}/logit_trajectory_topk.png"
ATTENTION_HEAD_SUMMARY_ASSET_PATH = f"{ASSETS_DIR}/attention_head_summary.png"
ATTENTION_UNAVAILABLE_ASSET_PATH = f"{ASSETS_DIR}/attention_unavailable.png"
ATTENTION_SELECTED_MAPS_ASSET_PATH = f"{ASSETS_DIR}/attention_selected_maps.png"
ATTENTION_MAPS_UNAVAILABLE_ASSET_PATH = f"{ASSETS_DIR}/attention_maps_unavailable.png"
NETWORK_MLP_ASSET_PATH = f"{ASSETS_DIR}/network_mlp_heatmap.png"
NETWORK_UNAVAILABLE_ASSET_PATH = f"{ASSETS_DIR}/network_unavailable.png"
NETWORK_ATTENTION_ASSET_PATH = f"{ASSETS_DIR}/network_attention_activity.png"
NETWORK_ATTENTION_UNAVAILABLE_ASSET_PATH = (
    f"{ASSETS_DIR}/network_attention_unavailable.png"
)
NETWORK_RESIDUAL_ASSET_PATH = f"{ASSETS_DIR}/network_residual_heatmap.png"
NETWORK_RESIDUAL_UNAVAILABLE_ASSET_PATH = (
    f"{ASSETS_DIR}/network_residual_unavailable.png"
)

REQUIRED_BASE_PATHS = [
    MANIFEST_PATH,
    SUMMARY_PATH,
    RAW_ANALYZE_RESPONSE_PATH,
    RAW_HEALTH_PATH,
    TOKENS_TABLE_PATH,
    PROMPT_PREDICTIONS_TABLE_PATH,
    GENERATED_ANSWER_TOPK_TABLE_PATH,
    ACTIVATION_SUMMARY_TABLE_PATH,
    LOGITS_TENSOR_PATH,
    ACTIVATIONS_TENSOR_PATH,
    OVERVIEW_METRICS_ASSET_PATH,
    PROMPT_PREDICTIONS_ASSET_PATH,
    GENERATED_ANSWER_TOPK_ASSET_PATH,
    ACTIVATION_L2_ASSET_PATH,
    ACTIVATION_MAX_ABS_ASSET_PATH,
    LOGIT_TRAJECTORY_ASSET_PATH,
]
