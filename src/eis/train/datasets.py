"""Compatibility facade for dataset helpers."""

from .data.datasets import (
    ExampleSequenceDataset,
    TokenBlockDataset,
    load_token_stream,
    load_token_stream_with_digit_places,
)

__all__ = [
    "ExampleSequenceDataset",
    "TokenBlockDataset",
    "load_token_stream",
    "load_token_stream_with_digit_places",
]
