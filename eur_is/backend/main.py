"""FastAPI application for the Europa ALM-IS web API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from eur_is.backend.analysis import summarize_checkpoint, summarize_runtime_metadata
from eur_is.backend.analysis_service import AnalysisServiceError, build_analyze_response
from eur_is.backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CheckpointResponse,
    ExportRequest,
    HealthResponse,
)
from eur_is.backend import settings
from eur_is.export.models import ExportOptions
from eur_is.export.writer import build_zip_bytes

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
    try:
        settings.load_resources()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        return build_analyze_response(
            runtime=settings.get_runtime(),
            checkpoint_path=settings.CHECKPOINT_PATH,
            device=settings.DEVICE,
            prompt=request.prompt,
            include_network=request.include_network,
            mlp_threshold=request.mlp_threshold,
            top_k=request.top_k,
            top_neurons=request.top_neurons,
            selected_token_index=request.selected_token_index,
        )
    except AnalysisServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@app.post("/api/export")
async def export_analysis(request: ExportRequest) -> StreamingResponse:
    try:
        settings.load_resources()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    # Export bundles are comprehensive and always include network analysis.
    try:
        analysis = build_analyze_response(
            runtime=settings.get_runtime(),
            checkpoint_path=settings.CHECKPOINT_PATH,
            device=settings.DEVICE,
            prompt=request.prompt,
            include_network=True,
            mlp_threshold=request.mlp_threshold,
            top_k=request.top_k,
            top_neurons=request.top_neurons,
            selected_token_index=request.selected_token_index,
        )
    except AnalysisServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    health_payload = await health()
    options = ExportOptions.model_validate(
        {
            "sections": request.sections,
            "output_mode": "zip",
            "prompt_source": "api",
            "checkpoint_path": str(settings.CHECKPOINT_PATH),
            "device": settings.DEVICE,
            "include_network": True,
            "mlp_threshold": request.mlp_threshold,
            "top_k": request.top_k,
            "top_neurons": request.top_neurons,
            "selected_token_index": request.selected_token_index,
        }
    )
    payload = build_zip_bytes(
        prompt=request.prompt,
        analysis=analysis,
        health=health_payload,
        options=options,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"its-export-{timestamp}.zip"
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
