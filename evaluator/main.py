from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "PyTorch is required to evaluate the model. Install it first, then rerun this script."
    ) from error

from generator.core import (
    KindSpec,
    format_signed_number,
    iter_kind_specs,
    parse_signed_number,
)
from generator.core import stable_hash, validate_line
from trainer.utils import (
    answer_from_line,
    configure_runtime,
    device_metadata,
    prompt_from_line,
    resolve_device,
)
from trainer.inference import (
    generate_completion,
)
from trainer.core import (
    load_checkpoint,
)

from .core import BucketStats, SelectedExample, bucket_row, sort_kind_rows


CATEGORY_ORDER = ("binary", "three_input", "parentheses", "negative_input")
ALL_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Europa ALM-IS model across sampled problem strata."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
        help="Dataset files to draw the per-kind sample pool from.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sample-size-per-kind", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--failures-per-kind", type=int, default=3)
    parser.add_argument("--progress-interval-kinds", type=int, default=0)
    return parser.parse_args()


def checkpoint_payload(
    checkpoint_path: Path, device: torch.device
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)!r}")
    return cast(dict[str, object], payload)


def train_config_from_payload(payload: dict[str, object]) -> dict[str, object]:
    config = payload.get("train_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing train_config")
    return cast(dict[str, object], config)


def resolve_data_dir(args: argparse.Namespace, payload: dict[str, object]) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    train_config = train_config_from_payload(payload)
    data_dir = train_config.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("--data-dir was not provided and checkpoint has no data_dir")
    return Path(data_dir)


def resolve_max_new_tokens(args: argparse.Namespace, payload: dict[str, object]) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    train_config = train_config_from_payload(payload)
    value = train_config.get("max_new_tokens", 24)
    if not isinstance(value, int):
        raise ValueError("checkpoint max_new_tokens is not an integer")
    return value


def resolve_output_prefix(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output_prefix is not None:
        return Path(args.output_prefix)
    return checkpoint_path.parent / f"{checkpoint_path.stem}-strata-eval"


def load_metadata(data_dir: Path) -> dict[str, object] | None:
    metadata_path = data_dir / "meta.json"
    if not metadata_path.exists():
        return None
    return cast(
        dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8"))
    )


def spec_to_definition(spec: KindSpec) -> dict[str, object]:
    return {
        "category": spec.category,
        "band_pattern": list(spec.band_pattern),
        "wildcard": spec.wildcard,
        "strategy": spec.strategy,
        "op": spec.op,
        "inner_op": spec.inner_op,
        "outer_op": spec.outer_op,
        "shape": spec.shape,
        "sign_side": spec.sign_side,
    }


def fallback_kind_definitions() -> dict[str, dict[str, object]]:
    return {spec.name: spec_to_definition(spec) for spec in iter_kind_specs()}


def kind_definitions_from_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if metadata is None:
        return fallback_kind_definitions()
    definitions = metadata.get("kind_definitions")
    if not isinstance(definitions, dict):
        raise ValueError("metadata is missing kind_definitions")
    return cast(dict[str, dict[str, object]], definitions)


def skipped_kinds_from_metadata(metadata: dict[str, object] | None) -> dict[str, str]:
    if metadata is None:
        return {}
    skipped = metadata.get("skipped_kinds", {})
    if not isinstance(skipped, dict):
        raise ValueError("metadata skipped_kinds has unexpected type")
    return {str(kind): str(reason) for kind, reason in skipped.items()}


def ordered_categories(metadata: dict[str, object] | None) -> list[str]:
    if metadata is None:
        return list(CATEGORY_ORDER)
    categories = metadata.get("categories")
    if not isinstance(categories, list) or not all(
        isinstance(category, str) for category in categories
    ):
        return list(CATEGORY_ORDER)
    return cast(list[str], categories)


def expected_available_kind_counts(
    metadata: dict[str, object] | None, splits: list[str]
) -> dict[str, int] | None:
    if metadata is None:
        return None
    split_counts = metadata.get("split_kind_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("metadata is missing split_kind_counts")

    totals: dict[str, int] = defaultdict(int)
    split_count_map = cast(dict[str, object], split_counts)
    for split in splits:
        raw_counts = split_count_map.get(split)
        if not isinstance(raw_counts, dict):
            raise ValueError(f"metadata is missing split_kind_counts[{split!r}]")
        for kind, count in cast(dict[str, object], raw_counts).items():
            if not isinstance(count, int):
                raise ValueError(
                    f"metadata split count for {kind!r} in {split!r} is not an integer"
                )
            totals[str(kind)] += count
    return dict(totals)


def is_canonical_prediction(text: str) -> bool:
    try:
        return format_signed_number(parse_signed_number(text)) == text
    except ValueError:
        return False


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_kind_csv(path: Path, kind_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "status",
        "category",
        "available_count",
        "evaluated_count",
        "perfect_count",
        "missed_count",
        "accuracy",
        "canonical_prediction_rate",
        "wildcard",
        "strategy",
        "band_pattern",
        "op",
        "inner_op",
        "outer_op",
        "shape",
        "sign_side",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in kind_rows:
            band_pattern = row.get("band_pattern")
            writer.writerow(
                {
                    "name": row["name"],
                    "status": row["status"],
                    "category": row.get("category"),
                    "available_count": row["available_count"],
                    "evaluated_count": row["evaluated_count"],
                    "perfect_count": row["perfect_count"],
                    "missed_count": row["missed_count"],
                    "accuracy": row["accuracy"],
                    "canonical_prediction_rate": row["canonical_prediction_rate"],
                    "wildcard": row.get("wildcard"),
                    "strategy": row.get("strategy"),
                    "band_pattern": "-".join(
                        cast(list[str], band_pattern)
                        if isinstance(band_pattern, list)
                        else []
                    ),
                    "op": row.get("op"),
                    "inner_op": row.get("inner_op"),
                    "outer_op": row.get("outer_op"),
                    "shape": row.get("shape"),
                    "sign_side": row.get("sign_side"),
                }
            )


def write_errors_jsonl(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in errors:
            handle.write(json.dumps(error, sort_keys=True) + "\n")


def top_or_bottom_kinds(
    kind_rows: list[dict[str, object]], *, descending: bool, count: int
) -> list[dict[str, object]]:
    evaluated = [
        row
        for row in kind_rows
        if row["status"] == "evaluated" and row["evaluated_count"]
    ]
    return sorted(
        evaluated,
        key=lambda row: (
            cast(float, row["accuracy"]),
            cast(int, row["evaluated_count"]),
            str(row["name"]),
        ),
        reverse=descending,
    )[:count]


def selection_sort_key(entry: tuple[int, int, SelectedExample]) -> tuple[int, int]:
    neg_key, neg_ordinal, _ = entry
    return (-neg_key, -neg_ordinal)


def collect_selected_examples(
    *,
    data_dir: Path,
    splits: list[str],
    sample_size_per_kind: int,
    sample_seed: int,
) -> tuple[dict[str, list[SelectedExample]], dict[str, int]]:
    heaps: dict[str, list[tuple[int, int, SelectedExample]]] = defaultdict(list)
    available_counts: dict[str, int] = defaultdict(int)
    ordinal = 0

    for split in splits:
        split_path = data_dir / f"{split}.txt"
        if not split_path.exists():
            raise SystemExit(f"split file does not exist: {split_path}")

        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                sample = validate_line(line)
                available_counts[sample.kind] += 1
                key = stable_hash(f"{sample_seed}\t{sample.kind}\t{line}")
                example = SelectedExample(
                    split=split,
                    line_number=line_number,
                    line=line,
                    category=sample.category,
                    kind=sample.kind,
                    key=key,
                )
                heap = heaps[sample.kind]
                heapq.heappush(heap, (-key, -ordinal, example))
                if len(heap) > sample_size_per_kind:
                    heapq.heappop(heap)
                ordinal += 1

    selected_examples: dict[str, list[SelectedExample]] = {}
    for kind, heap in heaps.items():
        selected_examples[kind] = [
            entry[2] for entry in sorted(heap, key=selection_sort_key)
        ]

    return selected_examples, dict(available_counts)


def validate_available_counts(
    *,
    actual_available_counts: dict[str, int],
    expected_available_counts_map: dict[str, int] | None,
) -> None:
    if expected_available_counts_map is None:
        return
    if actual_available_counts != expected_available_counts_map:
        raise ValueError(
            "available kind counts derived from the selected splits do not match metadata"
        )


def ordered_selected_kinds(
    selected_examples: dict[str, list[SelectedExample]], categories: list[str]
) -> list[str]:
    category_index = {category: index for index, category in enumerate(categories)}
    return sorted(
        selected_examples,
        key=lambda kind: (
            category_index.get(
                selected_examples[kind][0].category, len(category_index)
            ),
            kind,
        ),
    )


def print_selection_summary(
    *,
    splits: list[str],
    sample_size_per_kind: int,
    selected_examples: dict[str, list[SelectedExample]],
    available_counts: dict[str, int],
) -> None:
    evaluated_kinds = len(selected_examples)
    selected_total = sum(len(examples) for examples in selected_examples.values())
    available_total = sum(available_counts.values())
    print(
        json.dumps(
            {
                "pool_splits": splits,
                "sample_size_per_kind": sample_size_per_kind,
                "evaluated_kinds": evaluated_kinds,
                "selected_examples": selected_total,
                "available_examples_in_pool": available_total,
            },
            indent=2,
            sort_keys=True,
        )
    )


def print_console_summary(summary: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "checkpoint": summary["checkpoint"],
                "data_dir": summary["data_dir"],
                "pool_splits": summary["pool_splits"],
                "sample_size_per_kind": summary["sample_size_per_kind"],
                "sample_seed": summary["sample_seed"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "examples_per_second": summary["examples_per_second"],
                "overall": summary["overall"],
                "device": summary["device"],
            },
            indent=2,
            sort_keys=True,
        )
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


def main() -> None:
    args = parse_args()
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

    model, tokenizer = load_checkpoint(checkpoint_path, device)
    category_stats = {category: BucketStats() for category in categories}
    kind_stats: dict[str, BucketStats] = defaultdict(BucketStats)
    errors: list[dict[str, object]] = []
    ordered_kinds = ordered_selected_kinds(selected_examples, categories)
    total_examples = sum(len(examples) for examples in selected_examples.values())
    evaluated_examples = 0
    start_time = time.perf_counter()

    for kind_index, kind in enumerate(ordered_kinds, start=1):
        for example in selected_examples[kind]:
            prompt = prompt_from_line(example.line)
            expected_answer = answer_from_line(example.line)
            prediction = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            perfect = prediction == expected_answer
            canonical_prediction = is_canonical_prediction(prediction)

            for stats in (category_stats[example.category], kind_stats[example.kind]):
                stats.evaluated_count += 1
                if perfect:
                    stats.perfect_count += 1
                if canonical_prediction:
                    stats.canonical_prediction_count += 1

            if not perfect:
                error_row: dict[str, object] = {
                    "split": example.split,
                    "line_number": example.line_number,
                    "category": example.category,
                    "kind": example.kind,
                    "prompt": prompt,
                    "expected": expected_answer,
                    "prediction": prediction,
                    "prediction_is_canonical": canonical_prediction,
                }
                errors.append(error_row)
                failure_examples = kind_stats[example.kind].failure_examples
                if len(failure_examples) < args.failures_per_kind:
                    failure_examples.append(error_row)

            evaluated_examples += 1

        if (
            args.progress_interval_kinds > 0
            and kind_index % args.progress_interval_kinds == 0
        ):
            print(
                f"evaluated {kind_index}/{len(ordered_kinds)} kinds "
                f"({evaluated_examples}/{total_examples} examples)"
            )

    elapsed_seconds = time.perf_counter() - start_time
    overall_stats = BucketStats(
        evaluated_count=sum(stats.evaluated_count for stats in category_stats.values()),
        perfect_count=sum(stats.perfect_count for stats in category_stats.values()),
        canonical_prediction_count=sum(
            stats.canonical_prediction_count for stats in category_stats.values()
        ),
    )

    category_rows = [
        bucket_row(
            name=category,
            stats=category_stats[category],
            available_count=sum(
                count
                for kind, count in available_counts.items()
                if kind_definitions.get(kind, {}).get("category") == category
            ),
        )
        for category in categories
    ]

    all_kind_names = set(kind_definitions) | set(available_counts)
    kind_rows: list[dict[str, object]] = []
    for kind in all_kind_names:
        definition = dict(kind_definitions.get(kind, {}))
        status = "evaluated"
        if kind in skipped_kinds:
            status = "skipped_by_generator"
        elif available_counts.get(kind, 0) == 0:
            status = "missing_in_pool"

        row = bucket_row(
            name=kind,
            stats=kind_stats.get(kind, BucketStats()),
            available_count=available_counts.get(kind, 0),
            extra=definition,
            status=status,
        )
        if kind in skipped_kinds:
            row["skip_reason"] = skipped_kinds[kind]
        kind_rows.append(row)

    kind_rows = sort_kind_rows(kind_rows, categories)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "pool_splits": args.splits,
        "sample_size_per_kind": args.sample_size_per_kind,
        "sample_seed": args.sample_seed,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "examples_per_second": (
            None
            if overall_stats.evaluated_count == 0 or elapsed_seconds <= 0
            else round(overall_stats.evaluated_count / elapsed_seconds, 3)
        ),
        "device": device_metadata(device),
        "overall": bucket_row(
            name="overall",
            stats=overall_stats,
            available_count=sum(available_counts.values()),
        ),
        "categories": category_rows,
        "kinds": kind_rows,
        "skipped_kinds": skipped_kinds,
        "bottom_kinds": top_or_bottom_kinds(kind_rows, descending=False, count=10),
        "top_kinds": top_or_bottom_kinds(kind_rows, descending=True, count=10),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_summary_json(output_prefix.with_suffix(".summary.json"), summary)
    write_kind_csv(output_prefix.with_suffix(".kinds.csv"), kind_rows)
    write_errors_jsonl(output_prefix.with_suffix(".errors.jsonl"), errors)
    print_console_summary(summary)


if __name__ == "__main__":
    main()
