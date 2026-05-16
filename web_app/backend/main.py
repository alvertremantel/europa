"""Compatibility shim: re-exports from canonical package."""

from eur_is.backend.main import analyze, app, health, startup_event
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

__all__ = [
    "ActivationSummaryResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "AttentionSummaryResponse",
    "CheckpointResponse",
    "HealthResponse",
    "ModelConfigResponse",
    "TopPrediction",
    "analyze",
    "app",
    "health",
    "startup_event",
]
