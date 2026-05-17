from __future__ import annotations

from typing import Sequence

LEGACY_BASE_VOCAB = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<ans>",
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
SCRATCHPAD_TOKENS = ["<work>", "<step>", "<final>"]
BASE_VOCAB = list(LEGACY_BASE_VOCAB)

SPECIAL_FIELD_TOKENS = {
    "<ans>",
    "<work>",
    "<step>",
    "<final>",
    "undefined",
    "remainder",
}

POSITION_ENCODING_ABSOLUTE = "absolute"
POSITION_ENCODING_DIGIT_ROLES = "digit_roles"
POSITION_ROLE_NONE = 0
NUMBER_DIGIT_COUNT = 8
POSITION_ROLE_VOCAB_SIZE = NUMBER_DIGIT_COUNT + 1
SEPARATOR_TOKENS = {"<pad>", "<bos>", "<eos>", "<sep>"}


def vocab_for_training_format(training_format: str) -> list[str]:
    if training_format == "final_only":
        return list(LEGACY_BASE_VOCAB)
    return [
        *LEGACY_BASE_VOCAB[:5],
        *SCRATCHPAD_TOKENS,
        *LEGACY_BASE_VOCAB[5:],
    ]


class ArithmeticTokenizer:
    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        tokens = list(vocab) if vocab is not None else list(BASE_VOCAB)
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.sep_id = self.token_to_id["<sep>"]
        self.answer_token = "<ans>"
        self.answer_id = self.token_to_id[self.answer_token]

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

        token_ids = [self.bos_id]
        for index, field in enumerate(fields):
            token_ids.extend(self.encode_field(field))
            if index < len(fields) - 1 or append_trailing_separator:
                token_ids.append(self.sep_id)
        if include_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def encode_fields_with_roles(
        self,
        fields: Sequence[str],
        *,
        include_eos: bool,
        append_trailing_separator: bool,
    ) -> tuple[list[int], list[int]]:
        if not fields:
            raise ValueError("cannot encode an empty field sequence")

        token_ids = [self.bos_id]
        position_role_ids = [POSITION_ROLE_NONE]
        for index, field in enumerate(fields):
            encoded_field = self.encode_field(field)
            field_roles = self._field_position_roles(field)
            if len(encoded_field) != len(field_roles):
                raise RuntimeError(
                    f"field role count did not match encoded token count for {field!r}"
                )
            token_ids.extend(encoded_field)
            position_role_ids.extend(field_roles)
            if index < len(fields) - 1 or append_trailing_separator:
                token_ids.append(self.sep_id)
                position_role_ids.append(POSITION_ROLE_NONE)
        if include_eos:
            token_ids.append(self.eos_id)
            position_role_ids.append(POSITION_ROLE_NONE)
        return token_ids, position_role_ids

    def encode_line(self, line: str) -> list[int]:
        token_ids, _ = self.encode_fields_with_roles(
            line.strip().split(),
            include_eos=True,
            append_trailing_separator=False,
        )
        return token_ids

    def encode_line_with_roles(self, line: str) -> tuple[list[int], list[int]]:
        return self.encode_fields_with_roles(
            line.strip().split(),
            include_eos=True,
            append_trailing_separator=False,
        )

    def encode_prompt(self, prompt: str) -> list[int]:
        fields = prompt.strip().split()
        if not fields:
            raise ValueError("prompt cannot be empty")
        if self.answer_token in fields:
            fields = fields[: fields.index(self.answer_token) + 1]
        else:
            fields.append(self.answer_token)
        token_ids, _ = self.encode_fields_with_roles(
            fields,
            include_eos=False,
            append_trailing_separator=True,
        )
        return token_ids

    def encode_prompt_with_roles(self, prompt: str) -> tuple[list[int], list[int]]:
        fields = prompt.strip().split()
        if not fields:
            raise ValueError("prompt cannot be empty")
        if self.answer_token in fields:
            fields = fields[: fields.index(self.answer_token) + 1]
        else:
            fields.append(self.answer_token)
        return self.encode_fields_with_roles(
            fields,
            include_eos=False,
            append_trailing_separator=True,
        )

    def position_role_ids_for_token_ids(self, token_ids: Sequence[int]) -> list[int]:
        role_ids: list[int] = []
        current_field: list[str] = []

        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token in SEPARATOR_TOKENS:
                if current_field:
                    role_ids.extend(_field_token_position_roles(current_field))
                    current_field = []
                role_ids.append(POSITION_ROLE_NONE)
                continue
            current_field.append(token)

        if current_field:
            role_ids.extend(_field_token_position_roles(current_field))

        if len(role_ids) != len(token_ids):
            raise RuntimeError("position role IDs must align with token IDs")
        return role_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        fields: list[str] = []
        current_field: list[str] = []

        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token in {"<pad>", "<bos>"}:
                continue
            if token == "<eos>":
                break
            if token == "<sep>":
                if current_field:
                    fields.append("".join(current_field))
                    current_field = []
                continue
            current_field.append(token)

        if current_field:
            fields.append("".join(current_field))
        return " ".join(fields)

    def to_state(self) -> dict[str, list[str]]:
        return {"vocab": self.id_to_token}

    @classmethod
    def from_state(cls, state: dict[str, list[str]]) -> "ArithmeticTokenizer":
        return cls(vocab=state["vocab"])

    def _field_position_roles(self, field: str) -> list[int]:
        if field in SPECIAL_FIELD_TOKENS:
            return _field_token_position_roles([field])
        return _field_token_position_roles(list(field))


def _field_token_position_roles(tokens: Sequence[str]) -> list[int]:
    if len(tokens) == 1 and tokens[0] in SPECIAL_FIELD_TOKENS:
        return [POSITION_ROLE_NONE]
    if 1 <= len(tokens) <= NUMBER_DIGIT_COUNT and all(
        token.isdigit() for token in tokens
    ):
        return list(range(1, len(tokens) + 1))
    if len(tokens) >= 2 and tokens[0] == "(" and tokens[1] == "-":
        roles = [POSITION_ROLE_NONE, POSITION_ROLE_NONE]
        next_digit_role = 1
        for token in tokens[2:]:
            if token.isdigit() and next_digit_role <= NUMBER_DIGIT_COUNT:
                roles.append(next_digit_role)
                next_digit_role += 1
            else:
                roles.append(POSITION_ROLE_NONE)
        return roles
    return [POSITION_ROLE_NONE] * len(tokens)
