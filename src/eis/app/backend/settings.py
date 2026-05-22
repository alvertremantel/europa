"""Global settings and resource loading for the Europa ALM-IS web API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from eis.app.backend.runtime import BaseCheckpointRuntime, load_checkpoint_runtime

logger = logging.getLogger(__name__)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "runs/test-extended-plus/checkpoint-best.pt"
CHECKPOINT_ENV_VAR = "EUR_IS_CHECKPOINT_PATH"
CHECKPOINT_PATH: Path = Path(
    os.environ.get(CHECKPOINT_ENV_VAR) or str(DEFAULT_CHECKPOINT_PATH)
).resolve()

model: Any = None
tokenizer: Any = None
checkpoint_metadata: dict[str, Any] = {}
runtime: BaseCheckpointRuntime | None = None


def load_resources() -> None:
    """Load model, tokenizer, metadata, and runtime from the checkpoint."""
    global runtime, model, tokenizer, checkpoint_metadata
    if runtime is None:
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        try:
            runtime = load_checkpoint_runtime(
                CHECKPOINT_PATH,
                device=DEVICE,
            )
            model = runtime.model
            tokenizer = runtime.tokenizer
            checkpoint_metadata = runtime.checkpoint_metadata
        except Exception as error:  # pragma: no cover - exercised via FastAPI smoke
            raise RuntimeError(
                f"Failed to load checkpoint at {CHECKPOINT_PATH}: {error}"
            ) from error


def get_runtime() -> BaseCheckpointRuntime:
    load_resources()
    if runtime is None:
        raise RuntimeError("Model resources are unavailable.")
    return runtime
