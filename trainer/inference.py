"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.inference import (
    evaluate_balanced_loss,
    evaluate_exact_match,
    evaluate_exact_match_examples,
    evaluate_loss,
    generate_completion,
    loss_for_batch,
    loss_for_example_batch,
)

__all__ = [
    "evaluate_balanced_loss",
    "evaluate_exact_match",
    "evaluate_exact_match_examples",
    "evaluate_loss",
    "generate_completion",
    "loss_for_batch",
    "loss_for_example_batch",
]
