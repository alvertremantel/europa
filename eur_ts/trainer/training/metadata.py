"""Training run metadata and history writers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from eur_ts.artifacts import write_toml
from eur_ts.config import ModelConfig, TrainConfig
from ..data import ArithmeticExample
from ..model import SmallCausalTransformer
from ..utils import device_metadata, parameter_count


def resolve_target_epoch(config: TrainConfig, checkpoint_epoch: int) -> int:
    if config.additional_epochs is not None:
        return checkpoint_epoch + config.additional_epochs
    return config.epochs


def scratchpad_fraction(examples: list[ArithmeticExample]) -> float:
    if not examples:
        return 0.0
    scratchpad_count = sum(
        1
        for example in examples
        if example.training_format
        in {"parentheses_intermediate", "multiply_intermediate"}
    )
    return scratchpad_count / len(examples)


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    write_toml(path, {"history": history})


def write_run_metadata(
    path: Path,
    *,
    config: TrainConfig,
    model_config: ModelConfig,
    model: SmallCausalTransformer,
    device: torch.device,
    resume_source: str | None,
    run_started_at: float,
    run_completed_at: float | None,
    history: list[dict[str, object]],
) -> None:
    metadata = {
        "train_config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count": parameter_count(model),
        "device": device_metadata(device),
        "resume_source": resume_source,
        "checkpoint_policy": {
            "checkpoint_dir_name": config.checkpoint_dir_name,
            "checkpoint_keep_last": config.checkpoint_keep_last,
            "checkpoint_max_kept": config.checkpoint_max_kept,
            "checkpoint_keep_best": config.checkpoint_keep_best,
            "checkpoint_jump_threshold": config.checkpoint_jump_threshold,
        },
        "run_started_at_unix": run_started_at,
        "run_completed_at_unix": run_completed_at,
        "history_length": len(history),
    }
    write_toml(path, metadata)
