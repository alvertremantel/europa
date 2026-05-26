"""Shared dashboard analysis construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eis.app.backend.analysis import (
    summarize_checkpoint,
    summarize_problem,
    summarize_runtime_metadata,
)
from eis.app.backend.network_analysis import clamp_network_options
from eis.app.backend.runtime import BaseCheckpointRuntime
from eis.app.backend.schemas import (
    ActivationSummaryResponse,
    AnalyzeResponse,
    AttentionSummaryResponse,
    CheckpointResponse,
    GeneratedAnswerResponse,
    GeneratedAnswerToken,
    ModelConfigResponse,
    ProblemMetadataResponse,
    TopPrediction,
)

MAX_GENERATED_ANSWER_TOKENS = 32


class AnalysisServiceError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _first_config_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    raise RuntimeError(
        "Loaded runtime is missing required model configuration metadata."
    )


def _expression_from_prompt(prompt: str) -> str:
    fields = prompt.strip().split()
    if fields[:2] == ["<do>", "<calc>"]:
        fields = fields[2:]
    if "=" in fields:
        fields = fields[: fields.index("=")]
    if "<ans>" in fields:
        fields = fields[: fields.index("<ans>")]
    return " ".join(fields)


def _classification_prompt(prompt: str) -> str:
    expression = _expression_from_prompt(prompt)
    return f"{expression} = <ans>" if expression else "= <ans>"


def build_analyze_response(
    *,
    runtime: BaseCheckpointRuntime,
    checkpoint_path: Path,
    device: str,
    prompt: str,
    include_network: bool = False,
    mlp_threshold: float = 0.0,
    top_k: int = 5,
    top_neurons: int = 8,
    selected_token_index: int | None = None,
) -> AnalyzeResponse:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise AnalysisServiceError("Prompt cannot be empty.", status_code=400)

    tokenizer = runtime.tokenizer

    problem_metadata = None
    try:
        problem_metadata = summarize_problem(_classification_prompt(cleaned_prompt))
    except ValueError:
        problem_metadata = None

    try:
        token_ids = tokenizer.encode_prompt(cleaned_prompt)
    except ValueError as error:
        raise AnalysisServiceError(str(error), status_code=400) from error

    try:
        runtime.ensure_prompt_fits(len(token_ids))
    except ValueError as error:
        raise AnalysisServiceError(str(error), status_code=400) from error

    prompt_text = tokenizer.decode(token_ids)
    expression_text = _expression_from_prompt(prompt_text)
    network_options = clamp_network_options(
        mlp_threshold=mlp_threshold,
        top_k=top_k,
        top_neurons=top_neurons,
        selected_token_index=selected_token_index,
        token_count=len(token_ids),
    )

    try:
        analysis = runtime.analyze_prompt(
            prompt_token_ids=token_ids,
            top_k=top_k,
            expression_text=expression_text,
            max_generated_answer_tokens=MAX_GENERATED_ANSWER_TOKENS,
        )
        network_analysis = None
        if include_network and runtime.capabilities.network_analysis:
            network_analysis = runtime.build_network_analysis(
                analysis=analysis,
                network_options=network_options,
            )
    except AnalysisServiceError:
        raise
    except Exception as error:
        raise AnalysisServiceError(str(error), status_code=500) from error

    checkpoint_model_config = runtime.checkpoint_metadata.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        checkpoint_model_config = {}
    runtime_metadata = summarize_runtime_metadata(
        position_encoding=runtime.position_encoding,
        analysis_runtime=runtime.analysis_runtime,
        capabilities=runtime.capabilities.to_dict(),
    )
    runtime_model_config = getattr(runtime.model, "config", None)
    runtime_tl_config = getattr(runtime.model, "cfg", None)
    d_head = _first_config_value(
        getattr(runtime_tl_config, "d_head", None),
        checkpoint_model_config.get("d_head"),
        runtime.d_model // max(runtime.n_heads, 1),
    )
    mlp_hidden = _first_config_value(
        getattr(runtime_tl_config, "d_mlp", None),
        getattr(runtime_model_config, "mlp_hidden", None),
        checkpoint_model_config.get("mlp_hidden"),
    )
    sequence_length = _first_config_value(
        getattr(runtime_tl_config, "n_ctx", None),
        getattr(runtime_model_config, "sequence_length", None),
        checkpoint_model_config.get("sequence_length"),
        getattr(runtime, "context_window", None),
    )
    vocab_size = _first_config_value(
        getattr(runtime_tl_config, "d_vocab", None),
        getattr(runtime_model_config, "vocab_size", None),
        checkpoint_model_config.get("vocab_size"),
        tokenizer.vocab_size,
    )
    dropout = _first_config_value(
        getattr(runtime_tl_config, "attn_dropout", None),
        getattr(runtime_model_config, "dropout", None),
        checkpoint_model_config.get("dropout"),
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
            d_head=int(d_head),
            mlp_hidden=int(mlp_hidden) if mlp_hidden is not None else None,
            sequence_length=int(sequence_length),
            vocab_size=int(vocab_size),
            dropout=float(dropout) if dropout is not None else None,
        ),
        problem=ProblemMetadataResponse(**problem_metadata)
        if problem_metadata
        else None,
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(checkpoint_path),
                device=device,
                metadata=runtime.checkpoint_metadata,
            )
        ),
        network=network_analysis,
    )
