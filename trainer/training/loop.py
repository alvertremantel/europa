from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.config import ModelConfig, TrainConfig
from trainer.data import ArithmeticTokenizer, TokenBlockDataset, load_token_stream
from trainer.inference import evaluate_exact_match, evaluate_loss, loss_for_batch
from trainer.model import SmallCausalTransformer
from trainer.utils import (
    configure_runtime,
    device_metadata,
    parameter_count,
    resolve_device,
    set_seed,
)

from .checkpointing import (
    CheckpointManager,
    best_exact_match_from_history,
    build_checkpoint_payload,
    load_checkpoint_payload,
)
from .state import capture_rng_state, restore_rng_state


def train_model(config: TrainConfig) -> None:
    device = resolve_device(config.device)
    configure_runtime(device)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = _resolve_resume_path(config, output_dir)
    if resume_path is None:
        set_seed(config.seed)

    (
        tokenizer,
        model,
        effective_model_config,
        optimizer,
        history,
        start_epoch,
        best_exact_match,
        global_step,
        resume_source,
        resumed_from_epoch,
    ) = _initialize_training_state(
        config=config,
        device=device,
        resume_path=resume_path,
    )

    print(json.dumps(asdict(config), indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "parameters": parameter_count(model),
                "vocab_size": tokenizer.vocab_size,
                **device_metadata(device),
            },
            indent=2,
            sort_keys=True,
        )
    )

    train_tokens = load_token_stream(data_dir / "train.txt", tokenizer)
    val_tokens = load_token_stream(data_dir / "val.txt", tokenizer)

    train_dataset = TokenBlockDataset(
        train_tokens,
        effective_model_config.sequence_length,
    )
    val_dataset = TokenBlockDataset(
        val_tokens,
        effective_model_config.sequence_length,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("dataset is too small for the configured sequence length")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    checkpoint_manager = CheckpointManager(output_dir, config)
    run_metadata_path = output_dir / "run-metadata.json"
    run_started_at = time.time()
    target_epoch = _resolve_target_epoch(config, start_epoch - 1)
    if target_epoch < start_epoch - 1:
        raise ValueError(
            f"target epoch {target_epoch} is before checkpoint epoch {start_epoch - 1}"
        )
    if target_epoch == start_epoch - 1:
        print(f"No epochs remain to train; checkpoint is already at epoch {target_epoch}.")
        _write_run_metadata(
            run_metadata_path,
            config=config,
            model_config=effective_model_config,
            model=model,
            device=device,
            resume_source=resume_source,
            run_started_at=run_started_at,
            run_completed_at=time.time(),
            history=history,
        )
        return

    for epoch in range(start_epoch, target_epoch + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss_total = 0.0
        epoch_batches = 0
        log_loss_total = 0.0
        log_batches = 0

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, inputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss_total += step_loss
            log_loss_total += step_loss
            epoch_batches += 1
            log_batches += 1
            global_step += 1

            if step % config.log_interval == 0:
                average_loss = log_loss_total / max(log_batches, 1)
                print(
                    f"epoch={epoch} step={step} train_loss={average_loss:.4f} "
                    f"elapsed={time.perf_counter() - epoch_start:.1f}s"
                )
                log_loss_total = 0.0
                log_batches = 0

        train_loss = epoch_loss_total / max(epoch_batches, 1)
        val_loss = evaluate_loss(model, val_loader, device, config.eval_batches)
        exact_match = evaluate_exact_match(
            model=model,
            tokenizer=tokenizer,
            file_path=data_dir / "val.txt",
            sample_count=config.exact_match_samples,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        best_exact_match = max(best_exact_match, exact_match)

        epoch_duration_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        metrics: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "exact_match": exact_match,
            "epoch_duration_seconds": epoch_duration_seconds,
            "learning_rate": lr,
            "checkpoint_path": None,
            "checkpoint_roles": [],
        }
        if resumed_from_epoch is not None:
            metrics["resumed_from"] = resumed_from_epoch

        payload = build_checkpoint_payload(
            model=model,
            tokenizer=tokenizer,
            train_config=config,
            epoch=epoch,
            val_loss=val_loss,
            exact_match=exact_match,
            optimizer_state=cast(dict[str, object], optimizer.state_dict()),
            rng_state=capture_rng_state(),
            history=[*history, dict(metrics)],
            best_exact_match=best_exact_match,
            global_step=global_step,
            checkpoint_roles=[],
            resume_source=resume_source,
            train_loss=train_loss,
        )
        checkpoint_path, checkpoint_roles = checkpoint_manager.save_epoch(
            payload=payload,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            exact_match=exact_match,
            global_step=global_step,
        )

        metrics["checkpoint_path"] = str(checkpoint_path.relative_to(output_dir))
        metrics["checkpoint_roles"] = checkpoint_roles
        history.append(metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

        _write_history(output_dir / "history.json", history)
        _write_run_metadata(
            run_metadata_path,
            config=config,
            model_config=effective_model_config,
            model=model,
            device=device,
            resume_source=resume_source,
            run_started_at=run_started_at,
            run_completed_at=None,
            history=history,
        )

    _write_run_metadata(
        run_metadata_path,
        config=config,
        model_config=effective_model_config,
        model=model,
        device=device,
        resume_source=resume_source,
        run_started_at=run_started_at,
        run_completed_at=time.time(),
        history=history,
    )


def _resolve_resume_path(config: TrainConfig, output_dir: Path) -> Path | None:
    if config.resume_from:
        return Path(config.resume_from)
    if config.auto_resume:
        return output_dir / "checkpoint-last.pt"
    return None


def _initialize_training_state(
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
        tokenizer = ArithmeticTokenizer()
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            sequence_length=config.sequence_length,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
        )
        model = SmallCausalTransformer(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        return tokenizer, model, model_config, optimizer, [], 1, -math.inf, 0, None, None

    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    payload = load_checkpoint_payload(resume_path, device)
    tokenizer_state = payload.get("tokenizer")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("resume checkpoint is missing tokenizer")
    tokenizer = ArithmeticTokenizer.from_state(cast(dict[str, list[str]], tokenizer_state))

    model_config_state = payload.get("model_config")
    if not isinstance(model_config_state, dict):
        raise ValueError("resume checkpoint is missing model_config")
    model_config = ModelConfig(**cast(dict[str, object], model_config_state))

    for field_name in (
        "sequence_length",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_hidden",
        "dropout",
    ):
        requested = getattr(config, field_name)
        actual = getattr(model_config, field_name)
        if requested != actual:
            print(
                f"Warning: ignoring CLI {field_name}={requested!r} and using checkpoint value {actual!r}."
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
            optimizer_state = training_state.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError(
            "resume checkpoint does not contain optimizer_state; weights-only resume is not supported"
        )
    optimizer.load_state_dict(cast(dict[str, object], optimizer_state))

    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            rng_state = training_state.get("rng_state")
    if not isinstance(rng_state, dict) or not restore_rng_state(
        cast(dict[str, object], rng_state)
    ):
        print("Warning: resume checkpoint is missing a restorable RNG state.")

    history = _history_from_payload(payload, resume_path)
    checkpoint_epoch = int(cast(int, payload.get("epoch", 0)))
    resume_source = str(resume_path)
    best_exact_match = payload.get("best_exact_match")
    if not isinstance(best_exact_match, (float, int)):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            best_exact_match = training_state.get("best_exact_match")
    best_value = (
        float(best_exact_match)
        if isinstance(best_exact_match, (float, int))
        else best_exact_match_from_history(history)
    )
    global_step_value = payload.get("global_step")
    if not isinstance(global_step_value, int):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict) and isinstance(
            training_state.get("global_step"),
            int,
        ):
            global_step_value = cast(int, training_state["global_step"])
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


def _history_from_payload(
    payload: dict[str, object], resume_path: Path
) -> list[dict[str, object]]:
    history = payload.get("history")
    if isinstance(history, list):
        payload_history = [
            cast(dict[str, object], entry) for entry in history if isinstance(entry, dict)
        ]
    else:
        payload_history = []
    training_state = payload.get("training_state")
    if isinstance(training_state, dict) and isinstance(training_state.get("history"), list):
        payload_history = [
            cast(dict[str, object], entry)
            for entry in cast(list[object], training_state["history"])
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


def _resolve_target_epoch(config: TrainConfig, checkpoint_epoch: int) -> int:
    if config.additional_epochs is not None:
        return checkpoint_epoch + config.additional_epochs
    return config.epochs


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def _write_run_metadata(
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
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
