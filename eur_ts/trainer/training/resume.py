"""Training state initialization and resume helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import torch

from eur_ts.config import ModelConfig, TrainConfig
from ..data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_ABSOLUTE,
    POSITION_ROLE_VOCAB_SIZE,
    vocab_for_training_format,
)
from ..model import SmallCausalTransformer

from .checkpointing import best_exact_match_from_history, load_checkpoint_payload
from .state import restore_rng_state


def resolve_resume_path(config: TrainConfig, output_dir: Path) -> Path | None:
    if config.resume_from:
        return Path(config.resume_from)
    if config.auto_resume:
        return output_dir / "checkpoint-last.pt"
    return None


def initialize_training_state(
    *,
    config: TrainConfig,
    device: torch.device,
    resume_path: Path | None,
) -> tuple[
    ArithmeticTokenizer,
    SmallCausalTransformer,
    ModelConfig,
    torch.optim.Optimizer,
    list[dict[str, object]],
    int,
    float,
    int,
    str | None,
    int | None,
]:
    if resume_path is None:
        tokenizer = ArithmeticTokenizer(
            vocab_for_training_format(config.training_format)
        )
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            sequence_length=config.sequence_length,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
            position_encoding=config.position_encoding,
            position_vocab_size=POSITION_ROLE_VOCAB_SIZE,
        )
        model = SmallCausalTransformer(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return (
            tokenizer,
            model,
            model_config,
            optimizer,
            [],
            1,
            -math.inf,
            0,
            None,
            None,
        )

    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    payload = load_checkpoint_payload(resume_path, device)
    tokenizer_state = payload.get("tokenizer")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("resume checkpoint is missing tokenizer")
    tokenizer = ArithmeticTokenizer.from_state(
        cast(dict[str, list[str]], tokenizer_state)
    )

    model_config_state = payload.get("model_config")
    if not isinstance(model_config_state, dict):
        raise ValueError("resume checkpoint is missing model_config")
    state = cast(dict[str, object], model_config_state)
    model_config = ModelConfig(
        vocab_size=_as_int(state["vocab_size"], "model_config.vocab_size"),
        sequence_length=_as_int(
            state["sequence_length"], "model_config.sequence_length"
        ),
        d_model=_as_int(state["d_model"], "model_config.d_model"),
        n_heads=_as_int(state["n_heads"], "model_config.n_heads"),
        n_layers=_as_int(state["n_layers"], "model_config.n_layers"),
        mlp_hidden=_as_int(state["mlp_hidden"], "model_config.mlp_hidden"),
        dropout=_as_float(state["dropout"], "model_config.dropout"),
        position_encoding=_as_optional_str(
            state.get("position_encoding"),
            "model_config.position_encoding",
            default=POSITION_ENCODING_ABSOLUTE,
        ),
        position_vocab_size=_as_optional_int(
            state.get("position_vocab_size"),
            "model_config.position_vocab_size",
            default=POSITION_ROLE_VOCAB_SIZE,
        ),
    )

    for field_name in (
        "sequence_length",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_hidden",
        "dropout",
        "position_encoding",
    ):
        requested = getattr(config, field_name)
        actual = getattr(model_config, field_name)
        if requested != actual:
            print(
                f"Warning: ignoring config value {field_name}={requested!r} and using checkpoint value {actual!r}."
            )

    model = SmallCausalTransformer(model_config)
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("resume checkpoint is missing model_state")
    model.load_state_dict(cast(dict[str, torch.Tensor], model_state))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            optimizer_state = cast(dict[str, object], training_state).get(
                "optimizer_state"
            )
    if not isinstance(optimizer_state, dict):
        raise ValueError(
            "resume checkpoint does not contain optimizer_state; weights-only resume is not supported"
        )
    optimizer.load_state_dict(cast(dict[str, object], optimizer_state))

    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            rng_state = cast(dict[str, object], training_state).get("rng_state")
    if not isinstance(rng_state, dict) or not restore_rng_state(
        cast(dict[str, object], rng_state)
    ):
        print("Warning: resume checkpoint is missing a restorable RNG state.")

    history = history_from_payload(payload, resume_path)
    checkpoint_epoch = int(cast(int, payload.get("epoch", 0)))
    resume_source = str(resume_path)
    best_exact_match = payload.get("best_exact_match")
    if not isinstance(best_exact_match, (float, int)):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            best_exact_match = cast(dict[str, object], training_state).get(
                "best_exact_match"
            )
    best_value = (
        float(best_exact_match)
        if isinstance(best_exact_match, (float, int))
        else best_exact_match_from_history(history)
    )
    global_step_value = payload.get("global_step")
    if not isinstance(global_step_value, int):
        training_state = payload.get("training_state")
        training_state_dict = (
            cast(dict[str, object], training_state)
            if isinstance(training_state, dict)
            else None
        )
        if training_state_dict is not None and isinstance(
            training_state_dict.get("global_step"),
            int,
        ):
            global_step_value = cast(int, training_state_dict["global_step"])
        else:
            global_step_value = 0
    return (
        tokenizer,
        model,
        model_config,
        optimizer,
        history,
        checkpoint_epoch + 1,
        best_value,
        global_step_value,
        resume_source,
        checkpoint_epoch,
    )


def history_from_payload(
    payload: dict[str, object], resume_path: Path
) -> list[dict[str, object]]:
    history = payload.get("history")
    if isinstance(history, list):
        payload_history = [
            cast(dict[str, object], entry)
            for entry in history
            if isinstance(entry, dict)
        ]
    else:
        payload_history = []
    training_state = payload.get("training_state")
    training_state_dict = (
        cast(dict[str, object], training_state)
        if isinstance(training_state, dict)
        else None
    )
    if training_state_dict is not None and isinstance(
        training_state_dict.get("history"), list
    ):
        payload_history = [
            cast(dict[str, object], entry)
            for entry in cast(list[object], training_state_dict["history"])
            if isinstance(entry, dict)
        ]
    history_paths = [resume_path.parent / "history.json"]
    if resume_path.parent.name == "checkpoints":
        history_paths.append(resume_path.parent.parent / "history.json")
    for history_path in history_paths:
        if history_path.exists():
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                file_history = [
                    cast(dict[str, object], entry)
                    for entry in loaded
                    if isinstance(entry, dict)
                ]
                if len(file_history) >= len(payload_history):
                    return file_history
    return payload_history


def _as_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _as_optional_str(value: object, field_name: str, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _as_optional_int(value: object, field_name: str, *, default: int) -> int:
    if value is None:
        return default
    return _as_int(value, field_name)
