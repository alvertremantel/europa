"""CLI entrypoint for the evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from eis.artifacts import toml_text
from eis.train.runtime.utils import configure_runtime, resolve_device

from .args import parse_args
from .metadata import (
    checkpoint_payload,
    expected_available_kind_counts,
    kind_definitions_from_metadata,
    load_metadata,
    ordered_categories,
    resolve_data_dir,
    resolve_max_new_tokens,
    resolve_output_prefix,
    skipped_kinds_from_metadata,
)
from .runner import run_evaluation
from .sampling import (
    collect_selected_examples,
    print_selection_summary,
    validate_available_counts,
)


def print_console_summary(summary: dict[str, object]) -> None:
    print(
        toml_text(
            {
                "summary": {
                    "checkpoint": summary["checkpoint"],
                    "data_dir": summary["data_dir"],
                    "pool_splits": summary["pool_splits"],
                    "sample_size_per_kind": summary["sample_size_per_kind"],
                    "sample_seed": summary["sample_seed"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "examples_per_second": summary["examples_per_second"],
                    "overall": summary["overall"],
                    "device": summary["device"],
                }
            }
        ).rstrip()
    )
    print("category accuracy:")
    for row in cast(list[dict[str, object]], summary["categories"]):
        row_accuracy = row["accuracy"]
        formatted = "n/a" if row_accuracy is None else f"{row_accuracy:.4f}"
        print(
            f"  {row['name']}: {formatted} "
            f"({row['perfect_count']} perfect / {row['missed_count']} missed / {row['evaluated_count']} total)"
        )

    print("worst kinds:")
    for row in cast(list[dict[str, object]], summary["bottom_kinds"]):
        row_accuracy = row["accuracy"]
        formatted = "n/a" if row_accuracy is None else f"{row_accuracy:.4f}"
        print(
            f"  {row['name']}: {formatted} "
            f"({row['perfect_count']} perfect / {row['missed_count']} missed / {row['evaluated_count']} total)"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.sample_size_per_kind <= 0:
        raise SystemExit("--sample-size-per-kind must be positive")
    if args.failures_per_kind < 0:
        raise SystemExit("--failures-per-kind must be non-negative")
    if args.progress_interval_kinds < 0:
        raise SystemExit("--progress-interval-kinds must be non-negative")

    checkpoint_path = Path(args.checkpoint)
    device = resolve_device(args.device)
    configure_runtime(device)
    payload = checkpoint_payload(checkpoint_path, device)
    data_dir = resolve_data_dir(args, payload)
    max_new_tokens = resolve_max_new_tokens(args, payload)
    output_prefix = resolve_output_prefix(args, checkpoint_path)
    metadata = load_metadata(data_dir)
    categories = ordered_categories(metadata)
    kind_definitions = kind_definitions_from_metadata(metadata)
    skipped_kinds = skipped_kinds_from_metadata(metadata)
    expected_counts = expected_available_kind_counts(metadata, args.splits)

    selected_examples, available_counts = collect_selected_examples(
        data_dir=data_dir,
        splits=args.splits,
        sample_size_per_kind=args.sample_size_per_kind,
        sample_seed=args.sample_seed,
    )
    validate_available_counts(
        actual_available_counts=available_counts,
        expected_available_counts_map=expected_counts,
    )
    print_selection_summary(
        splits=args.splits,
        sample_size_per_kind=args.sample_size_per_kind,
        selected_examples=selected_examples,
        available_counts=available_counts,
    )

    summary = run_evaluation(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        splits=args.splits,
        device=device,
        max_new_tokens=max_new_tokens,
        sample_size_per_kind=args.sample_size_per_kind,
        sample_seed=args.sample_seed,
        output_prefix=output_prefix,
        failures_per_kind=args.failures_per_kind,
        progress_interval_kinds=args.progress_interval_kinds,
        metadata=metadata,
        categories=categories,
        kind_definitions=kind_definitions,
        skipped_kinds=skipped_kinds,
        expected_counts=expected_counts,
        selected_examples=selected_examples,
        available_counts=available_counts,
    )
    print_console_summary(summary)


if __name__ == "__main__":
    main()
