from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig
from .core import load_checkpoint, train_model
from .inference import generate_completion
from .utils import configure_runtime, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or query a small arithmetic language model (Europa ALM-IS)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from scratch or resume from an epoch checkpoint",
    )
    train_parser.add_argument("--data-dir", type=str, default="data-1m")
    train_parser.add_argument("--output-dir", type=str, default="runs/arithmetic-small")
    train_parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from an explicit checkpoint path.",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output-dir>/checkpoint-last.pt.",
    )
    train_parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="On resume, train this many more epochs beyond the checkpoint epoch.",
    )
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
    train_parser.add_argument("--checkpoint-keep-last", type=int, default=5)
    train_parser.add_argument("--checkpoint-max-kept", type=int, default=10)
    train_parser.add_argument("--checkpoint-keep-best", type=int, default=1)
    train_parser.add_argument("--checkpoint-jump-threshold", type=float, default=0.05)
    train_parser.add_argument(
        "--training-mode",
        choices=("token_stream", "examples"),
        default="token_stream",
        help="Use the legacy flat token stream or line-aware per-example training.",
    )
    train_parser.add_argument(
        "--training-format",
        choices=(
            "final_only",
            "light_scratchpad",
            "parentheses_intermediate",
            "multiply_intermediate",
        ),
        default="final_only",
        help="Opt-in target format for example-mode training.",
    )
    train_parser.add_argument(
        "--skip-overlong-examples",
        action="store_true",
        help="Skip rather than fail on per-example sequences that exceed --sequence-length.",
    )
    train_parser.add_argument(
        "--curriculum-name",
        choices=("baseline_mixed_v1", "mul_focus_v1"),
        default=None,
        help="Opt-in mixed-curriculum preset for --training-mode examples.",
    )
    train_parser.add_argument(
        "--balanced-val",
        action="store_true",
        help="Log balanced validation loss from a deterministic example sample.",
    )
    train_parser.add_argument(
        "--balanced-val-group-by",
        choices=("kind", "category", "curriculum_group"),
        default="kind",
    )
    train_parser.add_argument("--balanced-val-sample-size-per-group", type=int, default=8)
    train_parser.add_argument("--balanced-val-seed", type=int, default=42)
    train_parser.add_argument("--balanced-val-batch-size", type=int, default=None)

    predict_parser = subparsers.add_parser(
        "predict", help="Generate an answer from a saved checkpoint"
    )
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--prompt", type=str, required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=24)
    predict_parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def namespace_to_train_config(args: argparse.Namespace) -> TrainConfig:
    if args.additional_epochs is not None and args.additional_epochs <= 0:
        raise SystemExit("--additional-epochs must be positive when provided")
    resume_from = args.resume_from
    if args.resume and resume_from is not None:
        print("Both --resume and --resume-from were provided; using --resume-from.")
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from=resume_from,
        auto_resume=args.resume,
        additional_epochs=args.additional_epochs,
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
        checkpoint_keep_last=args.checkpoint_keep_last,
        checkpoint_max_kept=args.checkpoint_max_kept,
        checkpoint_keep_best=args.checkpoint_keep_best,
        checkpoint_jump_threshold=args.checkpoint_jump_threshold,
        training_mode=args.training_mode,
        training_format=args.training_format,
        skip_overlong_examples=args.skip_overlong_examples,
        curriculum_name=args.curriculum_name,
        balanced_val_enabled=args.balanced_val,
        balanced_val_group_by=args.balanced_val_group_by,
        balanced_val_sample_size_per_group=args.balanced_val_sample_size_per_group,
        balanced_val_seed=args.balanced_val_seed,
        balanced_val_batch_size=args.balanced_val_batch_size,
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
