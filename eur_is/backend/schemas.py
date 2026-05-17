"""Pydantic request / response models for the Europa ALM-IS web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    include_network: bool = False
    mlp_threshold: float = 0.0
    top_k: int = Field(default=5)
    top_neurons: int = Field(default=8)
    selected_token_index: int | None = None


class ModelConfigResponse(BaseModel):
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    mlp_hidden: int | None = None
    sequence_length: int
    vocab_size: int
    dropout: float | None = None


class ProblemMetadataResponse(BaseModel):
    category: str
    kind: str
    curriculum_group: str


class TopPrediction(BaseModel):
    token: str
    confidence: float
    logit: float | None = None


class GeneratedAnswerToken(BaseModel):
    token: str
    top_predictions: list[TopPrediction]


class GeneratedAnswerResponse(BaseModel):
    text: str
    tokens: list[str]
    token_count: int
    is_correct: bool
    is_valid_canonical: bool
    validation_error: str | None = None


class StrongestAttentionPair(BaseModel):
    query_index: int
    key_index: int
    query_token: str
    key_token: str
    weight: float


class AttentionHeadSummary(BaseModel):
    entropy: float
    max_weight: float
    mean_diagonal: float
    strongest_pair: StrongestAttentionPair


class AttentionSummaryResponse(BaseModel):
    heads: list[list[AttentionHeadSummary]]


class ActivationSummaryResponse(BaseModel):
    token_layer_l2: list[list[float]]
    token_layer_max_abs: list[list[float]]
    layer_mean_l2: list[float]
    layer_peak_l2: list[float]
    token_peak_l2: list[float]
    global_max_abs: float


class CheckpointResponse(BaseModel):
    path: str
    device: str
    epoch: int | None = None
    exact_match: float | None = None
    val_loss: float | None = None
    train_loss: float | None = None
    checkpoint_schema_version: int | None = None


class AnalyzeResponse(BaseModel):
    tokens: list[str]
    attention: list[list[list[list[float]]]]
    activations: list[list[list[float]]]
    logits: list[list[float]]
    top_predictions: list[TopPrediction]
    top_k_predictions: list[list[TopPrediction]]
    attention_summary: AttentionSummaryResponse
    activation_summary: ActivationSummaryResponse
    answer_position: int
    generated_answer: GeneratedAnswerResponse
    generated_answer_top_k: list[GeneratedAnswerToken]
    config: ModelConfigResponse
    problem: ProblemMetadataResponse | None = None
    checkpoint: CheckpointResponse
    network: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    checkpoint: CheckpointResponse
    detail: str | None = None
