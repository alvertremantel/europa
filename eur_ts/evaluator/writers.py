"""Output writers for evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast


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
