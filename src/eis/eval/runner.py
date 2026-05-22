"""Core evaluation runner logic."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import torch

from eis.train.runtime.utils import (
    answer_from_line,
    device_metadata,
    prompt_from_line,
)
from eis.train.inference import generate_completion
from eis.train.core import load_checkpoint

from .core import BucketStats, SelectedExample, bucket_row, sort_kind_rows
from .sampling import (
    is_canonical_prediction,
    ordered_selected_kinds,
    top_or_bottom_kinds,
)
from .writers import write_errors_toml, write_kind_csv, write_summary_toml


def run_evaluation(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    splits: list[str],
    device: torch.device,
    max_new_tokens: int,
    sample_size_per_kind: int,
    sample_seed: int,
    output_prefix: Path,
    failures_per_kind: int,
    progress_interval_kinds: int,
    metadata: dict[str, object] | None,
    categories: list[str],
    kind_definitions: dict[str, dict[str, object]],
    skipped_kinds: dict[str, str],
    expected_counts: dict[str, int] | None,
    selected_examples: dict[str, list[SelectedExample]],
    available_counts: dict[str, int],
) -> dict[str, object]:
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
                if len(failure_examples) < failures_per_kind:
                    failure_examples.append(error_row)

            evaluated_examples += 1

        if progress_interval_kinds > 0 and kind_index % progress_interval_kinds == 0:
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
        "pool_splits": splits,
        "sample_size_per_kind": sample_size_per_kind,
        "sample_seed": sample_seed,
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
    write_summary_toml(output_prefix.with_suffix(".summary.toml"), summary)
    write_kind_csv(output_prefix.with_suffix(".kinds.csv"), kind_rows)
    write_errors_toml(output_prefix.with_suffix(".errors.toml"), errors)

    return summary
