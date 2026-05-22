"""CLI-friendly prompt analysis runner for ITS exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eis.app.backend.analysis import summarize_checkpoint, summarize_runtime_metadata
from eis.app.backend.analysis_service import build_analyze_response
from eis.app.backend.runtime import load_checkpoint_runtime
from eis.app.backend.schemas import AnalyzeResponse, CheckpointResponse, HealthResponse


@dataclass
class ExportRunResult:
    analysis: AnalyzeResponse
    health: HealthResponse


def run_export_analysis(
    *,
    checkpoint_path: Path,
    device: str,
    prompt: str,
    include_network: bool = True,
    mlp_threshold: float = 0.0,
    top_k: int = 5,
    top_neurons: int = 8,
    selected_token_index: int | None = None,
) -> ExportRunResult:
    runtime = load_checkpoint_runtime(checkpoint_path, device=device)
    analysis = build_analyze_response(
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        device=device,
        prompt=prompt,
        include_network=include_network,
        mlp_threshold=mlp_threshold,
        top_k=top_k,
        top_neurons=top_neurons,
        selected_token_index=selected_token_index,
    )
    runtime_metadata = summarize_runtime_metadata(
        position_encoding=runtime.position_encoding,
        analysis_runtime=runtime.analysis_runtime,
        capabilities=runtime.capabilities.to_dict(),
    )
    health = HealthResponse(
        **runtime_metadata,
        status="ok",
        device=device,
        checkpoint=CheckpointResponse(
            **summarize_checkpoint(
                checkpoint_path=str(checkpoint_path),
                device=device,
                metadata=runtime.checkpoint_metadata,
            )
        ),
        detail=None,
    )
    return ExportRunResult(analysis=analysis, health=health)
