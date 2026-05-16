"""Global settings and resource loading for the Europa ALM-IS web API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from eur_is.backend.model_utils import load_hooked_resources

logger = logging.getLogger(__name__)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH: Path = Path("runs/test-extended-plus/checkpoint-best.pt")

model: Any = None
tokenizer: Any = None
checkpoint_metadata: dict[str, Any] = {}


def load_resources() -> None:
    """Load model, tokenizer, and metadata from the fixed checkpoint path."""
    global model, tokenizer, checkpoint_metadata
    if model is None or tokenizer is None:
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        try:
            model, tokenizer, checkpoint_metadata = load_hooked_resources(
                CHECKPOINT_PATH,
                device=DEVICE,
            )
        except Exception as error:  # pragma: no cover - exercised via FastAPI smoke
            raise RuntimeError(
                f"Failed to load checkpoint at {CHECKPOINT_PATH}: {error}"
            ) from error
