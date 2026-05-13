from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig
from .core import load_checkpoint, train_model
from .inference import generate_completion
from .utils import configure_runtime, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or query a small arithmetic language model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from scratch")
    train_parser.add_argument("--data-dir", type=str, default="data-1m")
    train_parser.add_argument("--output-dir", type=str, default="runs/arithmetic-small")
    train_parser.add_argument("--sequence-length", type=int, default=64)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--log-interval", type=int, default=100)
    train_parser.add_argument("--eval-batches", type=int, default=50)
    train_parser.add_argument("--exact-match-samples", type=int, default=256)
    train_parser.add_argument("--max-new-tokens", type=int, default=24)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=6)
    train_parser.add_argument("--mlp-hidden", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.1)

    predict_parser = subparsers.add_parser(
        "predict", help="Generate an answer from a saved checkpoint"
    )
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--prompt", type=str, required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=24)
    predict_parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def namespace_to_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_batches=args.eval_batches,
        exact_match_samples=args.exact_match_samples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device=args.device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_model(namespace_to_train_config(args))
        return

    device = resolve_device(args.device)
    configure_runtime(device)
    model, tokenizer = load_checkpoint(Path(args.checkpoint), device)
    prediction = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(prediction)


if __name__ == "__main__":
    main()
