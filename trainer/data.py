"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.data import (
    ArithmeticExample,
    ArithmeticTokenizer,
    BASE_VOCAB,
    ExampleSequenceDataset,
    LEGACY_BASE_VOCAB,
    SCRATCHPAD_TOKENS,
    SPECIAL_FIELD_TOKENS,
    TokenBlockDataset,
    load_examples,
    load_token_stream,
    transform_examples,
    vocab_for_training_format,
)

__all__ = [
    "ArithmeticExample",
    "ArithmeticTokenizer",
    "BASE_VOCAB",
    "ExampleSequenceDataset",
    "LEGACY_BASE_VOCAB",
    "SCRATCHPAD_TOKENS",
    "SPECIAL_FIELD_TOKENS",
    "TokenBlockDataset",
    "load_examples",
    "load_token_stream",
    "transform_examples",
    "vocab_for_training_format",
]
