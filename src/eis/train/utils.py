"""Compatibility facade for runtime utilities."""

from .runtime.utils import (
    answer_from_line,
    configure_runtime,
    device_metadata,
    parameter_count,
    prompt_from_line,
    read_examples,
    resolve_device,
    sample_examples,
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
    "sample_examples",
    "set_seed",
]
