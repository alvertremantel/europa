"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.utils import (
    answer_from_line,
    configure_runtime,
    device_metadata,
    parameter_count,
    prompt_from_line,
    read_examples,
    resolve_device,
    set_seed,
)

__all__ = [
    "answer_from_line",
    "configure_runtime",
    "device_metadata",
    "parameter_count",
    "prompt_from_line",
    "read_examples",
    "resolve_device",
    "set_seed",
]
