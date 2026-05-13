from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import ArithmeticTokenizer, TokenBlockDataset, load_token_stream
from .inference import evaluate_exact_match, evaluate_loss, loss_for_batch
from .model import SmallCausalTransformer
from .utils import (
    configure_runtime,
    device_metadata,
    parameter_count,
    resolve_device,
    set_seed,
)


def save_checkpoint(
    output_dir: Path,
    file_name: str,
    model: SmallCausalTransformer,
    tokenizer: ArithmeticTokenizer,
    train_config: TrainConfig,
    epoch: int,
    val_loss: float,
    exact_match: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.to_state(),
        "train_config": asdict(train_config),
        "epoch": epoch,
        "val_loss": val_loss,
        "exact_match": exact_match,
    }
    torch.save(payload, output_dir / file_name)


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[SmallCausalTransformer, ArithmeticTokenizer]:
    payload = torch.load(checkpoint_path, map_location=device)
    tokenizer = ArithmeticTokenizer.from_state(payload["tokenizer"])
    model_config = ModelConfig(**payload["model_config"])
    model = SmallCausalTransformer(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def train_model(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_runtime(device)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_dataset = TokenBlockDataset(train_tokens, config.sequence_length)
    val_dataset = TokenBlockDataset(val_tokens, config.sequence_length)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_exact_match = -math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.perf_counter()

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, inputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            if step % config.log_interval == 0:
                average_loss = running_loss / config.log_interval
                print(
                    f"epoch={epoch} step={step} train_loss={average_loss:.4f} "
                    f"elapsed={time.perf_counter() - start_time:.1f}s"
                )
                running_loss = 0.0

        val_loss = evaluate_loss(model, val_loader, device, config.eval_batches)
        exact_match = evaluate_exact_match(
            model=model,
            tokenizer=tokenizer,
            file_path=data_dir / "val.txt",
            sample_count=config.exact_match_samples,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )

        metrics = {
            "epoch": epoch,
            "val_loss": val_loss,
            "exact_match": exact_match,
        }
        history.append(metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

        save_checkpoint(
            output_dir=output_dir,
            file_name="checkpoint-last.pt",
            model=model,
            tokenizer=tokenizer,
            train_config=config,
            epoch=epoch,
            val_loss=val_loss,
            exact_match=exact_match,
        )
        if exact_match >= best_exact_match:
            best_exact_match = exact_match
            save_checkpoint(
                output_dir=output_dir,
                file_name="checkpoint-best.pt",
                model=model,
                tokenizer=tokenizer,
                train_config=config,
                epoch=epoch,
                val_loss=val_loss,
                exact_match=exact_match,
            )

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
