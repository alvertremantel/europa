"""Data facade: re-exports from tokenizer, examples, and datasets modules."""

from .tokenizer import (
    ArithmeticTokenizer,
    BASE_VOCAB,
    LEGACY_BASE_VOCAB,
    POSITION_ENCODING_FIXED_MEANING,
    SCRATCHPAD_TOKENS,
    SPECIAL_FIELD_TOKENS,
    SUPPORTED_POSITION_ENCODINGS,
    vocab_for_training_format,
)
from .examples import ArithmeticExample, load_examples, transform_examples
from .datasets import (
    ExampleSequenceDataset,
    TokenBlockDataset,
    load_token_stream,
)

__all__ = [
    "ArithmeticExample",
    "ArithmeticTokenizer",
    "BASE_VOCAB",
    "ExampleSequenceDataset",
    "LEGACY_BASE_VOCAB",
    "POSITION_ENCODING_FIXED_MEANING",
    "SCRATCHPAD_TOKENS",
    "SPECIAL_FIELD_TOKENS",
    "SUPPORTED_POSITION_ENCODINGS",
    "TokenBlockDataset",
    "load_examples",
    "load_token_stream",
    "transform_examples",
    "vocab_for_training_format",
]
