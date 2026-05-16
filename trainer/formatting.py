"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.formatting import (
    TrainingFormat,
    extract_final_answer,
    final_answer_from_line,
    format_training_line,
)

__all__ = [
    "TrainingFormat",
    "extract_final_answer",
    "final_answer_from_line",
    "format_training_line",
]
