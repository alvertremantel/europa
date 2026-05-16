"""Compatibility shim: re-exports from canonical package."""

from eur_ts.evaluator.core import (
    BucketStats,
    SelectedExample,
    accuracy,
    bucket_row,
    missed_count,
    sort_kind_rows,
)

__all__ = [
    "SelectedExample",
    "BucketStats",
    "accuracy",
    "bucket_row",
    "missed_count",
    "sort_kind_rows",
]
