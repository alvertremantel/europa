from __future__ import annotations

from typing import Sequence

from ..semantics.fixed_meaning import (
    FIXED_MEANING_MAX_DIGIT_PLACE,
    SPECIAL_FIELD_TOKENS,
)

BASE_VOCAB_TOKENS = [
    "<pad>",
    "<do>",
    "<eos>",
    "<calc>",
    "<ans>",
    "+",
    "-",
    "*",
    "/",
    "<",
    ">",
    "=",
    "(",
    ")",
    "{",
    "}",
    "true",
    "false",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
LEGACY_BASE_VOCAB = BASE_VOCAB_TOKENS
SCRATCHPAD_TOKENS: list[str] = []
BASE_VOCAB = list(BASE_VOCAB_TOKENS)

POSITION_ENCODING_FIXED_MEANING = "fixed_meaning"
SUPPORTED_POSITION_ENCODINGS = {POSITION_ENCODING_FIXED_MEANING}


def vocab_for_training_format(training_format: str) -> list[str]:
    if training_format == "final_only":
        return list(BASE_VOCAB_TOKENS)
    raise ValueError("training_format must be final_only")


class ArithmeticTokenizer:
    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        tokens = list(vocab) if vocab is not None else list(BASE_VOCAB)
        if "<bos>" in tokens or "<sep>" in tokens:
            raise ValueError(
                "legacy tokenizer vocabulary with <bos>/<sep> is unsupported"
            )
        if "<do>" not in tokens or "<calc>" not in tokens or "<ans>" not in tokens:
            raise ValueError(
                "tokenizer vocabulary must contain <do>, <calc>, and <ans>"
            )
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

        self.pad_id = self.token_to_id["<pad>"]
        self.do_id = self.token_to_id["<do>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.calc_token = "<calc>"
        self.calc_id = self.token_to_id[self.calc_token]
        self.ans_token = "<ans>"
        self.ans_id = self.token_to_id[self.ans_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode_field(self, field: str) -> list[int]:
        if field in SPECIAL_FIELD_TOKENS:
            return [self.token_to_id[field]]
        return [self.token_to_id[character] for character in field]

    def encode_fields(
        self,
        fields: Sequence[str],
        *,
        include_eos: bool,
        append_trailing_separator: bool,
    ) -> list[int]:
        if not fields:
            raise ValueError("cannot encode an empty field sequence")

        token_ids: list[int] = []
        for index, field in enumerate(fields):
            token_ids.extend(self.encode_field(field))
            if append_trailing_separator and index == len(fields) - 1:
                raise ValueError("REDUX tokenization does not use trailing separators")
        if include_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def encode_line(self, line: str) -> list[int]:
        fields = _normalize_line_fields(line)
        return self._encode_canonical_fields(fields, include_eos=True)

    def encode_prompt(self, prompt: str) -> list[int]:
        fields = _normalize_prompt_fields(prompt)
        return self._encode_canonical_fields(fields, include_eos=False)

    def fixed_meaning_digit_place_values_for_token_ids(
        self,
        token_ids: Sequence[int],
    ) -> list[float]:
        values = [0.0] * len(token_ids)
        digit_run: list[int] = []

        def flush_digit_run() -> None:
            if not digit_run:
                return
            for place, token_index in enumerate(digit_run, start=1):
                values[token_index] = min(place, FIXED_MEANING_MAX_DIGIT_PLACE) / float(
                    FIXED_MEANING_MAX_DIGIT_PLACE
                )
            digit_run.clear()

        for index, token_id in enumerate(token_ids):
            token = self.id_to_token[token_id]
            if token.isdigit():
                digit_run.append(index)
                continue
            flush_digit_run()
        flush_digit_run()
        return values

    def fixed_meaning_digit_place_value_after_prefix(
        self,
        prefix_token_ids: Sequence[int],
        token_id: int,
    ) -> float:
        token = self.id_to_token[token_id]
        if not token.isdigit():
            return 0.0

        digit_place = 1
        for previous_token_id in reversed(prefix_token_ids):
            if not self.id_to_token[previous_token_id].isdigit():
                break
            digit_place += 1
            if digit_place >= FIXED_MEANING_MAX_DIGIT_PLACE:
                break
        return min(digit_place, FIXED_MEANING_MAX_DIGIT_PLACE) / float(
            FIXED_MEANING_MAX_DIGIT_PLACE
        )

    def _encode_canonical_fields(
        self, fields: Sequence[str], *, include_eos: bool
    ) -> list[int]:
        if len(fields) < 4 or fields[0] != "<do>" or fields[1] != "<calc>":
            raise ValueError("canonical fields must begin with <do> <calc>")
        token_ids = [self.do_id, self.calc_id]
        for field in fields[2:]:
            token_ids.extend(self.encode_field(field))
        if include_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        fields: list[str] = []
        current_field: list[str] = []

        def flush_current() -> None:
            if current_field:
                fields.append("".join(current_field))
                current_field.clear()

        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token == "<pad>":
                continue
            if token == "<eos>":
                break
            if token in SPECIAL_FIELD_TOKENS:
                flush_current()
                fields.append(token)
                continue
            if token in {"+", "-", "*", "/", "<", ">", "="}:
                flush_current()
                fields.append(token)
                continue
            if token in {"(", "{"}:
                flush_current()
                current_field.append(token)
                continue
            if token in {
                ")",
                "}",
            }:
                current_field.append(token)
                flush_current()
                continue
            current_field.append(token)

        flush_current()
        return " ".join(fields)

    def decode_answer_tokens(self, token_ids: Sequence[int]) -> str:
        return self.decode(token_ids)

    def to_state(self) -> dict[str, list[str]]:
        return {"vocab": self.id_to_token}

    @classmethod
    def from_state(cls, state: dict[str, list[str]]) -> "ArithmeticTokenizer":
        return cls(vocab=state["vocab"])


def _normalize_prompt_fields(prompt: str) -> list[str]:
    fields = prompt.strip().split()
    if not fields:
        raise ValueError("prompt cannot be empty")
    if fields[:2] != ["<do>", "<calc>"]:
        fields = ["<do>", "<calc>", *fields]
    if "<bos>" in fields or "<sep>" in fields:
        raise ValueError("legacy <bos>/<sep> prompt tokens are unsupported")
    if "<eos>" in fields:
        fields = fields[: fields.index("<eos>")]
    if "=" not in fields:
        fields.append("=")
    else:
        fields = fields[: fields.index("=") + 1]
    if len(fields) <= fields.index("=") + 1 or fields[fields.index("=") + 1] != "<ans>":
        fields.append("<ans>")
    return fields


def _normalize_line_fields(line: str) -> list[str]:
    fields = line.strip().split()
    if not fields:
        raise ValueError("line cannot be empty")
    if fields[:2] != ["<do>", "<calc>"]:
        raise ValueError("sample lines must begin with <do> <calc>")
    if "<bos>" in fields or "<sep>" in fields:
        raise ValueError("legacy <bos>/<sep> sample tokens are unsupported")
    if "=" not in fields:
        raise ValueError("sample line is missing '='")
    equals_index = fields.index("=")
    if equals_index >= len(fields) - 2 or fields[equals_index + 1] != "<ans>":
        raise ValueError("sample line must contain '= <ans> <answer>'")
    return fields
