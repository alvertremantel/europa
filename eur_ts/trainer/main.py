from __future__ import annotations

import argparse
from pathlib import Path

from eur_ts.config.toml_io import load_train_config

from .core import load_checkpoint, train_model
from .inference import generate_completion
from .utils import configure_runtime, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or query a small arithmetic language model (Europa ALM-IS)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from a TOML config file",
    )
    train_parser.add_argument("config", type=str, help="Path to a training TOML file")

    predict_parser = subparsers.add_parser(
        "predict", help="Generate an answer from a saved checkpoint"
    )
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--prompt", type=str, required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=24)
    predict_parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train_model(load_train_config(Path(args.config)))
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
