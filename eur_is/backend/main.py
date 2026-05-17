"""FastAPI application for the Europa ALM-IS web API."""

from __future__ import annotations

import logging

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from eur_is.backend.analysis import (
    GeneratedAnswerSummary,
    GeneratedAnswerTokenSummary,
    build_ranked_predictions_for_distribution,
    build_activation_summary,
    build_attention_summary,
    build_top_prediction_summaries,
    evaluate_generated_answer,
    summarize_checkpoint,
)
from eur_is.backend.network_analysis import (
    clamp_network_options,
    extract_network_analysis,
)
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

    model = settings.model
    tokenizer = settings.tokenizer
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model resources are unavailable.")

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
    prompt_text = tokenizer.decode(token_ids)
    expression_text = prompt_text.split(" <ans>", maxsplit=1)[0].strip()
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
        generated_answer, generated_answer_top_k = _generate_answer_details(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=token_ids,
            top_k=request.top_k,
            expression_text=expression_text,
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
        generated_answer=GeneratedAnswerResponse(
            text=generated_answer["text"],
            tokens=generated_answer["tokens"],
            token_count=generated_answer["token_count"],
            is_correct=generated_answer["is_correct"],
            is_valid_canonical=generated_answer["is_valid_canonical"],
            validation_error=generated_answer["validation_error"],
        ),
        generated_answer_top_k=[
            GeneratedAnswerToken(
                token=entry["token"],
                top_predictions=[
                    TopPrediction(**prediction)
                    for prediction in entry["top_predictions"]
                ],
            )
            for entry in generated_answer_top_k
        ],
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


def _generate_answer_details(
    *,
    model,
    tokenizer,
    prompt_token_ids: list[int],
    top_k: int,
    expression_text: str,
) -> tuple[GeneratedAnswerSummary, list[GeneratedAnswerTokenSummary]]:
    generated = torch.tensor(
        prompt_token_ids, dtype=torch.long, device=settings.DEVICE
    ).unsqueeze(0)
    answer_token_ids: list[int] = []
    answer_top_k: list[GeneratedAnswerTokenSummary] = []

    for _ in range(MAX_GENERATED_ANSWER_TOKENS):
        window = generated[:, -model.cfg.n_ctx :]
        step_logits = model(window)[0, -1].detach().cpu()
        step_probs = torch.softmax(step_logits, dim=-1)
        ranked = build_ranked_predictions_for_distribution(
            probs=step_probs.numpy(),
            logits=step_logits.numpy(),
            tokens_by_id=tokenizer.id_to_token,
            top_k=top_k,
        )
        next_token_id = int(step_logits.argmax().item())
        if next_token_id == tokenizer.eos_id:
            break
        answer_token_ids.append(next_token_id)
        answer_top_k.append(
            {
                "token": tokenizer.id_to_token[next_token_id],
                "top_predictions": ranked,
            }
        )
        next_token = torch.tensor(
            [[next_token_id]], dtype=torch.long, device=settings.DEVICE
        )
        generated = torch.cat((generated, next_token), dim=1)

    generated_text = tokenizer.decode(generated.squeeze(0).tolist())
    if " <ans> " in generated_text:
        generated_text = generated_text.split(" <ans> ", maxsplit=1)[1]
    generated_answer = evaluate_generated_answer(
        expression_text=expression_text,
        generated_text=generated_text,
    )
    generated_answer["tokens"] = [
        tokenizer.id_to_token[token_id] for token_id in answer_token_ids
    ]
    generated_answer["token_count"] = len(answer_token_ids)
    return generated_answer, answer_top_k


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
