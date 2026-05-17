"""FastAPI application for the Europa ALM-IS web API."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from eur_is.backend.analysis import (
    summarize_checkpoint,
    summarize_runtime_metadata,
)
from eur_is.backend.network_analysis import clamp_network_options
from eur_is.backend.schemas import (
    ActivationSummaryResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AttentionSummaryResponse,
    CheckpointResponse,
    GeneratedAnswerResponse,
    GeneratedAnswerToken,
    HealthResponse,
    ModelConfigResponse,
    TopPrediction,
)
from eur_is.backend import settings

logger = logging.getLogger(__name__)
MAX_GENERATED_ANSWER_TOKENS = 32

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

    runtime = settings.get_runtime()
    tokenizer = runtime.tokenizer

    try:
        token_ids = tokenizer.encode_prompt(cleaned_prompt)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        runtime.ensure_prompt_fits(len(token_ids))
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    prompt_text = tokenizer.decode(token_ids)
    expression_text = prompt_text.split(" <ans>", maxsplit=1)[0].strip()
    network_options = clamp_network_options(
        mlp_threshold=request.mlp_threshold,
        top_k=request.top_k,
        top_neurons=request.top_neurons,
        selected_token_index=request.selected_token_index,
        token_count=len(token_ids),
    )

    try:
        analysis = runtime.analyze_prompt(
            prompt_token_ids=token_ids,
            top_k=request.top_k,
            expression_text=expression_text,
            max_generated_answer_tokens=MAX_GENERATED_ANSWER_TOKENS,
        )
        network_analysis = None
        if request.include_network and runtime.capabilities.network_analysis:
            network_analysis = runtime.build_network_analysis(
                analysis=analysis,
                network_options=network_options,
            )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    runtime_metadata = summarize_runtime_metadata(
        position_encoding=runtime.position_encoding,
        analysis_runtime=runtime.analysis_runtime,
        capabilities=runtime.capabilities.to_dict(),
    )

    return AnalyzeResponse(
        **runtime_metadata,
        tokens=analysis.tokens,
        attention=(
            [layer.tolist() for layer in analysis.attention_by_layer]
            if analysis.attention_by_layer is not None
            else None
        ),
        activations=analysis.stacked_activations.tolist(),
        logits=analysis.logits.tolist(),
        top_predictions=[
            TopPrediction(**prediction) for prediction in analysis.top_predictions
        ],
        top_k_predictions=[
            [TopPrediction(**prediction) for prediction in predictions]
            for predictions in analysis.top_k_predictions
        ],
        attention_summary=(
            AttentionSummaryResponse(**analysis.attention_summary)
            if analysis.attention_summary is not None
            else None
        ),
        activation_summary=ActivationSummaryResponse(**analysis.activation_summary),
        answer_position=analysis.answer_position,
        generated_answer=GeneratedAnswerResponse(
            text=analysis.generated_answer["text"],
            tokens=analysis.generated_answer["tokens"],
            token_count=analysis.generated_answer["token_count"],
            is_correct=analysis.generated_answer["is_correct"],
            is_valid_canonical=analysis.generated_answer["is_valid_canonical"],
            validation_error=analysis.generated_answer["validation_error"],
        ),
        generated_answer_top_k=[
            GeneratedAnswerToken(
                token=entry["token"],
                top_predictions=[
                    TopPrediction(**prediction)
                    for prediction in entry["top_predictions"]
                ],
            )
            for entry in analysis.generated_answer_top_k
        ],
        config=ModelConfigResponse(
            n_layers=runtime.n_layers,
            n_heads=runtime.n_heads,
            d_model=runtime.d_model,
        ),
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(settings.CHECKPOINT_PATH),
                device=settings.DEVICE,
                metadata=runtime.checkpoint_metadata,
            )
        ),
        network=network_analysis,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok"
    detail = None
    runtime_metadata = summarize_runtime_metadata(
        position_encoding=None,
        analysis_runtime=None,
        capabilities=None,
    )
    metadata = settings.checkpoint_metadata
    try:
        settings.load_resources()
        if settings.runtime is not None:
            metadata = settings.runtime.checkpoint_metadata
            runtime_metadata = summarize_runtime_metadata(
                position_encoding=settings.runtime.position_encoding,
                analysis_runtime=settings.runtime.analysis_runtime,
                capabilities=settings.runtime.capabilities.to_dict(),
            )
    except RuntimeError as error:
        status = "error"
        detail = str(error)

    return HealthResponse(
        **runtime_metadata,
        status=status,
        device=settings.DEVICE,
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(settings.CHECKPOINT_PATH),
                device=settings.DEVICE,
                metadata=metadata,
            )
        ),
        detail=detail,
    )
