"""Example sampling for strata evaluation."""

from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import cast

from eis.artifacts import toml_text
from eis.data.core import (
    stable_hash,
    validate_line,
)
from eis.data.answers import is_canonical_answer

from .core import SelectedExample


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
        toml_text(
            {
                "selection": {
                    "pool_splits": splits,
                    "sample_size_per_kind": sample_size_per_kind,
                    "evaluated_kinds": evaluated_kinds,
                    "selected_examples": selected_total,
                    "available_examples_in_pool": available_total,
                }
            }
        ).rstrip()
    )


def is_canonical_prediction(text: str) -> bool:
    return is_canonical_answer(text)


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
