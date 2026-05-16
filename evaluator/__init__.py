"""Compatibility shim: re-exports from canonical package."""

from eur_ts.evaluator.core import BucketStats, SelectedExample, bucket_row, accuracy, missed_count, sort_kind_rows

__all__ = ["SelectedExample", "BucketStats", "bucket_row", "accuracy", "missed_count", "sort_kind_rows"]
