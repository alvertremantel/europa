from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectedExample:
    split: str
    line_number: int
    line: str
    category: str
    kind: str
    key: int


@dataclass
class BucketStats:
    evaluated_count: int = 0
    perfect_count: int = 0
    canonical_prediction_count: int = 0
    failure_examples: list[dict[str, object]] = field(default_factory=list)


def accuracy(perfect_count: int, evaluated_count: int) -> float | None:
    if evaluated_count == 0:
        return None
    return perfect_count / evaluated_count


def missed_count(stats: BucketStats) -> int:
    return stats.evaluated_count - stats.perfect_count


def bucket_row(
    *,
    name: str,
    stats: BucketStats,
    available_count: int | None = None,
    extra: dict[str, object] | None = None,
    status: str = "evaluated",
) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "status": status,
        "available_count": available_count,
        "evaluated_count": stats.evaluated_count,
        "perfect_count": stats.perfect_count,
        "missed_count": missed_count(stats),
        "accuracy": accuracy(stats.perfect_count, stats.evaluated_count),
        "canonical_prediction_rate": accuracy(
            stats.canonical_prediction_count, stats.evaluated_count
        ),
        "failure_examples": stats.failure_examples,
    }
    if extra is not None:
        row.update(extra)
    return row


def sort_kind_rows(
    rows: list[dict[str, object]], categories: list[str]
) -> list[dict[str, object]]:
    category_index = {category: index for index, category in enumerate(categories)}
    return sorted(
        rows,
        key=lambda row: (
            category_index.get(str(row.get("category", "")), len(category_index)),
            str(row["name"]),
        ),
    )
