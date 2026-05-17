"""Data facade: re-exports from tokenizer, examples, and datasets modules."""

from .tokenizer import (
    ArithmeticTokenizer,
    BASE_VOCAB,
    LEGACY_BASE_VOCAB,
    POSITION_ENCODING_ABSOLUTE,
    POSITION_ENCODING_DIGIT_ROLES,
    POSITION_ROLE_VOCAB_SIZE,
    SCRATCHPAD_TOKENS,
    SPECIAL_FIELD_TOKENS,
    vocab_for_training_format,
)
from .examples import ArithmeticExample, load_examples, transform_examples
from .datasets import (
    ExampleSequenceDataset,
    TokenBlockDataset,
    load_token_stream,
    load_token_stream_with_roles,
)

__all__ = [
    "ArithmeticExample",
    "ArithmeticTokenizer",
    "BASE_VOCAB",
    "ExampleSequenceDataset",
    "LEGACY_BASE_VOCAB",
    "POSITION_ENCODING_ABSOLUTE",
    "POSITION_ENCODING_DIGIT_ROLES",
    "POSITION_ROLE_VOCAB_SIZE",
    "SCRATCHPAD_TOKENS",
    "SPECIAL_FIELD_TOKENS",
    "TokenBlockDataset",
    "load_examples",
    "load_token_stream",
    "load_token_stream_with_roles",
    "transform_examples",
    "vocab_for_training_format",
]
