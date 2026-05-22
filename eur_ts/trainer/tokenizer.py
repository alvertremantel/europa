from __future__ import annotations

from typing import Sequence

from .fixed_meaning import SPECIAL_FIELD_TOKENS

BASE_VOCAB_TOKENS = [
    "<pad>",
    "<do>",
    "<eos>",
    "<sep>",
    "<calc>",
    "undefined",
    "remainder",
    "+",
    "-",
    "*",
    "/",
    "=",
    "(",
    ")",
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
SCRATCHPAD_TOKENS = ["<work>", "<step>", "<final>"]
BASE_VOCAB = list(BASE_VOCAB_TOKENS)

POSITION_ENCODING_FIXED_MEANING = "fixed_meaning"
SUPPORTED_POSITION_ENCODINGS = {POSITION_ENCODING_FIXED_MEANING}


def vocab_for_training_format(training_format: str) -> list[str]:
    if training_format == "final_only":
        return list(BASE_VOCAB_TOKENS)
    return [
        *BASE_VOCAB_TOKENS[:5],
        *SCRATCHPAD_TOKENS,
        *BASE_VOCAB_TOKENS[5:],
    ]


class ArithmeticTokenizer:
    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        tokens = list(vocab) if vocab is not None else list(BASE_VOCAB)
        if "<bos>" in tokens or "<ans>" in tokens:
            raise ValueError(
                "legacy tokenizer vocabulary with <bos>/<ans> is unsupported"
            )
        if len(tokens) <= 4 or tokens[1] != "<do>" or tokens[4] != "<calc>":
            raise ValueError(
                "tokenizer vocabulary must keep <do> at id 1 and <calc> at id 4"
            )
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

        self.pad_id = self.token_to_id["<pad>"]
        self.do_id = self.token_to_id["<do>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.sep_id = self.token_to_id["<sep>"]
        self.calc_token = "<calc>"
        self.calc_id = self.token_to_id[self.calc_token]

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
            if index < len(fields) - 1 or append_trailing_separator:
                token_ids.append(self.sep_id)
        if include_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def encode_line(self, line: str) -> list[int]:
        fields = _normalize_line_fields(line)
        return self._encode_canonical_fields(fields, include_eos=True)

    def encode_prompt(self, prompt: str) -> list[int]:
        fields = _normalize_prompt_fields(prompt)
        return self._encode_canonical_fields(fields, include_eos=False)

    def _encode_canonical_fields(
        self, fields: Sequence[str], *, include_eos: bool
    ) -> list[int]:
        if len(fields) < 4 or fields[0] != "<do>" or fields[1] != "<calc>":
            raise ValueError("canonical fields must begin with <do> <calc>")
        token_ids = [self.do_id, self.calc_id]
        for index, field in enumerate(fields[2:], start=2):
            token_ids.extend(self.encode_field(field))
            if field == "=" or index < len(fields) - 1:
                token_ids.append(self.sep_id)
        if include_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        fields: list[str] = []
        current_field: list[str] = []

        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token == "<pad>":
                continue
            if token == "<eos>":
                break
            if token == "<sep>":
                if current_field:
                    fields.append("".join(current_field))
                    current_field = []
                continue
            if token in SPECIAL_FIELD_TOKENS:
                if current_field:
                    fields.append("".join(current_field))
                    current_field = []
                fields.append(token)
                continue
            current_field.append(token)

        if current_field:
            fields.append("".join(current_field))
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
    if "<ans>" in fields or "<bos>" in fields:
        raise ValueError("legacy <bos>/<ans> prompt tokens are unsupported")
    if "<eos>" in fields:
        fields = fields[: fields.index("<eos>")]
    if "=" not in fields:
        fields.append("=")
    else:
        fields = fields[: fields.index("=") + 1]
    return fields


def _normalize_line_fields(line: str) -> list[str]:
    fields = line.strip().split()
    if not fields:
        raise ValueError("line cannot be empty")
    if fields[:2] != ["<do>", "<calc>"]:
        raise ValueError("sample lines must begin with <do> <calc>")
    if "<ans>" in fields or "<bos>" in fields:
        raise ValueError("legacy <bos>/<ans> sample tokens are unsupported")
    if "=" not in fields:
        raise ValueError("sample line is missing '='")
    if fields.index("=") >= len(fields) - 1:
        raise ValueError("sample line is missing answer after '='")
    return fields
