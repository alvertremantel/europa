"""FastAPI application for the Europa ALM-IS web API."""

from __future__ import annotations

import logging

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from eur_is.backend.analysis import (
    build_activation_summary,
    build_attention_summary,
    build_top_prediction_summaries,
    summarize_checkpoint,
)
from eur_is.backend.network_analysis import clamp_network_options, extract_network_analysis
from eur_is.backend.schemas import (
    ActivationSummaryResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AttentionSummaryResponse,
    CheckpointResponse,
    HealthResponse,
    ModelConfigResponse,
    TopPrediction,
)
from eur_is.backend import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Europa ALM-IS Web API",
    description=(
        "Analyze arithmetic prompts against a checkpoint-backed TransformerLens model. "
        "Returns raw attention/residual tensors for CircuitsVis plus compact summaries "
        "for dashboard rendering."
    ),
)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        settings.load_resources()
    except RuntimeError as error:
        logger.error("Failed to load checkpoint: %s", error)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    cleaned_prompt = request.prompt.strip()
    if not cleaned_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        settings.load_resources()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    model = settings.model
    tokenizer = settings.tokenizer
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
    input_tensor = torch.tensor(
        token_ids, dtype=torch.long, device=settings.DEVICE
    ).unsqueeze(0)
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
                checkpoint_path=str(settings.CHECKPOINT_PATH),
                device=settings.DEVICE,
                metadata=settings.checkpoint_metadata,
            )
        ),
        network=network_analysis,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok"
    detail = None
    try:
        settings.load_resources()
    except RuntimeError as error:
        status = "error"
        detail = str(error)

    return HealthResponse(
        status=status,
        device=settings.DEVICE,
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(settings.CHECKPOINT_PATH),
                device=settings.DEVICE,
                metadata=settings.checkpoint_metadata,
            )
        ),
        detail=detail,
    )
