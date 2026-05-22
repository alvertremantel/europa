from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

DIGIT_TOKENS = tuple(str(value) for value in range(10))
OPERATOR_TOKENS = ("+", "-", "*", "/", "=", "(", ")")
CONTROL_TOKENS = (
    "<pad>",
    "<do>",
    "<eos>",
    "<sep>",
    "<calc>",
    "<work>",
    "<step>",
    "<final>",
    "undefined",
    "remainder",
)
SPECIAL_FIELD_TOKENS = frozenset(
    {"<do>", "<calc>", "<work>", "<step>", "<final>", "undefined", "remainder"}
)
INFO_TOKENS = frozenset(CONTROL_TOKENS)
SEPARATOR_TOKENS = frozenset({"<pad>", "<do>", "<eos>", "<sep>", "<calc>"})

FIXED_MEANING_DIMENSIONS = (
    "act1",
    "act2",
    "how1",
    "how2",
    "math1",
    "math2",
    "math3",
    "math4",
    "math5",
    "math6",
    "form1",
    "form2",
)
FIXED_MEANING_WIDTH = len(FIXED_MEANING_DIMENSIONS)
FIXED_MEANING_DIGIT_VALUE_DIMENSION = FIXED_MEANING_DIMENSIONS.index("math5")
FIXED_MEANING_DIGIT_PLACE_DIMENSION = FIXED_MEANING_DIMENSIONS.index("math6")
FIXED_MEANING_MAX_DIGIT_PLACE = 9


def _vector(*values: float) -> tuple[float, ...]:
    if len(values) != FIXED_MEANING_WIDTH:
        raise ValueError(
            f"fixed_meaning vectors must have width {FIXED_MEANING_WIDTH}, got {len(values)}"
        )
    return tuple(float(value) for value in values)


def _digit_vector(value: float) -> tuple[float, ...]:
    return _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0)


# Aligned with docs/fixed_meaning_plan.csv. For digit tokens, the authored math6 value is
# intentionally dynamic and is filled at runtime from the digit's place within each
# full reversed 8-digit numeral.
FIXED_MEANING_TOKEN_VECTORS: dict[str, tuple[float, ...]] = {
    "<pad>": _vector(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.5, 0.0),
    "<do>": _vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.25),
    "<eos>": _vector(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.25),
    "<sep>": _vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.5, 0.0),
    "<calc>": _vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "<work>": _vector(0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5),
    "<step>": _vector(0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5),
    "<final>": _vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5),
    "undefined": _vector(0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "remainder": _vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "+": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "-": _vector(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "*": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "/": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "=": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
    "(": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
    ")": _vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
    "0": _digit_vector(0.0),
    "1": _digit_vector(0.1),
    "2": _digit_vector(0.2),
    "3": _digit_vector(0.3),
    "4": _digit_vector(0.4),
    "5": _digit_vector(0.5),
    "6": _digit_vector(0.6),
    "7": _digit_vector(0.7),
    "8": _digit_vector(0.8),
    "9": _digit_vector(0.9),
}


def fixed_meaning_width() -> int:
    return FIXED_MEANING_WIDTH


def build_fixed_meaning_token_table(tokens: Sequence[str], d_model: int) -> Tensor:
    _validate_fixed_meaning_vectors(d_model)
    table = torch.zeros((len(tokens), d_model), dtype=torch.float32)
    missing_tokens = [
        token for token in tokens if token not in FIXED_MEANING_TOKEN_VECTORS
    ]
    if missing_tokens:
        raise ValueError(
            "fixed_meaning token vectors are missing definitions for: "
            + ", ".join(sorted(set(missing_tokens)))
        )
    for token_id, token in enumerate(tokens):
        table[token_id] = torch.tensor(
            FIXED_MEANING_TOKEN_VECTORS[token], dtype=torch.float32
        )
    return table


def _validate_fixed_meaning_vectors(d_model: int) -> None:
    if d_model != FIXED_MEANING_WIDTH:
        raise ValueError(
            "fixed_meaning d_model must match the width of "
            f"eur_ts.trainer.fixed_meaning.FIXED_MEANING_TOKEN_VECTORS "
            f"({FIXED_MEANING_WIDTH}), got {d_model}"
        )
    invalid = [
        token
        for token, vector in FIXED_MEANING_TOKEN_VECTORS.items()
        if len(vector) != FIXED_MEANING_WIDTH
    ]
    if invalid:
        raise ValueError(
            "fixed_meaning token vectors must all share one width; invalid tokens: "
            + ", ".join(sorted(invalid))
        )


__all__ = [
    "CONTROL_TOKENS",
    "DIGIT_TOKENS",
    "FIXED_MEANING_DIGIT_PLACE_DIMENSION",
    "FIXED_MEANING_DIGIT_VALUE_DIMENSION",
    "FIXED_MEANING_DIMENSIONS",
    "FIXED_MEANING_MAX_DIGIT_PLACE",
    "FIXED_MEANING_TOKEN_VECTORS",
    "FIXED_MEANING_WIDTH",
    "INFO_TOKENS",
    "OPERATOR_TOKENS",
    "SEPARATOR_TOKENS",
    "SPECIAL_FIELD_TOKENS",
    "build_fixed_meaning_token_table",
    "fixed_meaning_width",
]
