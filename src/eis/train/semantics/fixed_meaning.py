from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

DIGIT_TOKENS = tuple(str(value) for value in range(10))
OPERATOR_TOKENS = ("+", "-", "*", "/", "<", ">", "=", "(", ")", "{", "}")
CONTROL_TOKENS = ("<pad>", "<do>", "<eos>", "<calc>", "<ans>", "true", "false")
SPECIAL_FIELD_TOKENS = frozenset({"<do>", "<calc>", "<ans>", "true", "false"})
INFO_TOKENS = frozenset(CONTROL_TOKENS)
SEPARATOR_TOKENS = frozenset({"<pad>", "<do>", "<eos>", "<calc>", "<ans>"})

FIXED_MEANING_DIMENSIONS = (
    "act1",
    "act2",
    "how1",
    "how2",
    "add_sub",
    "mul_div",
    "compare_lt_gt",
    "equals",
    "digit_value",
    "digit_place",
    "wrapper_round",
    "wrapper_curly",
    "truth",
    "answer_phase",
    "form1",
    "form2",
)
FIXED_MEANING_WIDTH = len(FIXED_MEANING_DIMENSIONS)
_DIM = {name: index for index, name in enumerate(FIXED_MEANING_DIMENSIONS)}
FIXED_MEANING_DIGIT_VALUE_DIMENSION = _DIM["digit_value"]
FIXED_MEANING_DIGIT_PLACE_DIMENSION = _DIM["digit_place"]
FIXED_MEANING_MAX_DIGIT_PLACE = 9


def _vector(**values: float) -> tuple[float, ...]:
    vector = [0.0] * FIXED_MEANING_WIDTH
    for name, value in values.items():
        vector[_DIM[name]] = float(value)
    return tuple(vector)


def _digit_vector(value: float) -> tuple[float, ...]:
    return _vector(digit_value=value)


# Digit place is filled dynamically by the tokenizer/runtime. REDUX extends the
# fixed space with explicit comparison, wrapper, truth, and answer-phase axes.
FIXED_MEANING_TOKEN_VECTORS: dict[str, tuple[float, ...]] = {
    "<pad>": _vector(how1=-1.0, digit_place=-1.0, form1=0.5),
    "<do>": _vector(act1=1.0, digit_place=-1.0, form2=0.25),
    "<eos>": _vector(act1=-1.0, digit_place=-1.0, form2=-0.25),
    "<calc>": _vector(act2=1.0, digit_place=-1.0),
    "<ans>": _vector(act2=-1.0, answer_phase=1.0, digit_place=-1.0),
    "true": _vector(truth=1.0, answer_phase=1.0, digit_place=-1.0),
    "false": _vector(truth=-1.0, answer_phase=1.0, digit_place=-1.0),
    "+": _vector(add_sub=1.0, digit_place=-1.0),
    "-": _vector(add_sub=-1.0, digit_place=-1.0),
    "*": _vector(mul_div=1.0, digit_place=-1.0),
    "/": _vector(mul_div=-1.0, digit_place=-1.0),
    "<": _vector(compare_lt_gt=1.0, digit_place=-1.0),
    ">": _vector(compare_lt_gt=-1.0, digit_place=-1.0),
    "=": _vector(equals=1.0, digit_place=-1.0),
    "(": _vector(wrapper_round=1.0, digit_place=-1.0),
    ")": _vector(wrapper_round=-1.0, digit_place=-1.0),
    "{": _vector(wrapper_curly=1.0, digit_place=-1.0),
    "}": _vector(wrapper_curly=-1.0, digit_place=-1.0),
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
            f"eis.train.semantics.fixed_meaning.FIXED_MEANING_TOKEN_VECTORS "
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
