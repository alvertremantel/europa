from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from web_app.backend.analysis import (
    build_activation_summary,
    build_attention_summary,
    build_top_prediction_summaries,
    summarize_checkpoint,
)
from web_app.backend.model_utils import load_hooked_resources
from web_app.backend.network_analysis import clamp_network_options, extract_network_analysis

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Europa ALM-IS Web API",
    description=(
        "Analyze arithmetic prompts against a checkpoint-backed TransformerLens model. "
        "Returns raw attention/residual tensors for CircuitsVis plus compact summaries "
        "for dashboard rendering."
    ),
)

# Global state to keep model and tokenizer in memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("runs/test-extended-plus/checkpoint-best.pt")

model = None
tokenizer = None
checkpoint_metadata: dict[str, Any] = {}


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


class TopPrediction(BaseModel):
    token: str
    confidence: float
    logit: float | None = None


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
    config: ModelConfigResponse
    checkpoint: CheckpointResponse
    network: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    checkpoint: CheckpointResponse
    detail: str | None = None


def load_resources() -> None:
    global model, tokenizer, checkpoint_metadata
    if model is None or tokenizer is None:
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        model, tokenizer, checkpoint_metadata = load_hooked_resources(CHECKPOINT_PATH, device=DEVICE)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        load_resources()
    except RuntimeError as error:
        logger.error("Failed to load checkpoint: %s", error)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    cleaned_prompt = request.prompt.strip()
    if not cleaned_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        load_resources()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    assert tokenizer is not None
    assert model is not None

    try:
        token_ids = tokenizer.encode_prompt(cleaned_prompt)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if len(token_ids) > model.cfg.n_ctx:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt is too long for the loaded checkpoint context window "
                f"({len(token_ids)} tokens > {model.cfg.n_ctx})."
            ),
        )

    tokens = [tokenizer.id_to_token[token_id] for token_id in token_ids]
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    network_options = clamp_network_options(
        mlp_threshold=request.mlp_threshold,
        top_k=request.top_k,
        top_neurons=request.top_neurons,
        selected_token_index=request.selected_token_index,
        token_count=len(tokens),
    )

    try:
        with torch.no_grad():
            logits, cache = model.run_with_cache(input_tensor)

        attention_by_layer: list[np.ndarray] = []
        residual_layers: list[np.ndarray] = []
        for layer_idx in range(model.cfg.n_layers):
            attention_by_layer.append(
                cache[f"blocks.{layer_idx}.attn.hook_pattern"][0].detach().cpu().numpy()
            )
            residual_layers.append(
                cache[f"blocks.{layer_idx}.hook_resid_post"][0].detach().cpu().numpy()
            )

        stacked_activations = np.stack(residual_layers, axis=1)
        logits_np = logits[0].detach().cpu().numpy()
        probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

        top_predictions, top_k_predictions = build_top_prediction_summaries(
            probs=probs,
            logits=logits_np,
            tokens_by_id=tokenizer.id_to_token,
            top_k=request.top_k,
        )
        attention_summary = build_attention_summary(
            attention_by_layer=attention_by_layer,
            tokens=tokens,
        )
        activation_summary = build_activation_summary(
            stacked_activations=stacked_activations,
        )
        network_analysis = None
        if request.include_network:
            network_analysis = extract_network_analysis(
                model=model,
                tokenizer=tokenizer,
                tokens=tokens,
                cache=cache,
                mlp_threshold=float(network_options["mlp_threshold"]),
                top_k=int(network_options["top_k"]),
                top_neurons=int(network_options["top_neurons"]),
                selected_token_index=network_options["selected_token_index"],
            )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return AnalyzeResponse(
        tokens=tokens,
        attention=[layer.tolist() for layer in attention_by_layer],
        activations=stacked_activations.tolist(),
        logits=logits_np.tolist(),
        top_predictions=[TopPrediction(**prediction) for prediction in top_predictions],
        top_k_predictions=[
            [TopPrediction(**prediction) for prediction in predictions]
            for predictions in top_k_predictions
        ],
        attention_summary=AttentionSummaryResponse(**attention_summary),
        activation_summary=ActivationSummaryResponse(**activation_summary),
        answer_position=len(tokens) - 1,
        config=ModelConfigResponse(
            n_layers=model.cfg.n_layers,
            n_heads=model.cfg.n_heads,
            d_model=model.cfg.d_model,
        ),
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(CHECKPOINT_PATH),
                device=DEVICE,
                metadata=checkpoint_metadata,
            )
        ),
        network=network_analysis,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok"
    detail = None
    try:
        load_resources()
    except RuntimeError as error:
        status = "error"
        detail = str(error)

    return HealthResponse(
        status=status,
        device=DEVICE,
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(CHECKPOINT_PATH),
                device=DEVICE,
                metadata=checkpoint_metadata,
            )
        ),
        detail=detail,
    )
