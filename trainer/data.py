from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

BASE_VOCAB = [
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
        if field in {"<ans>", "undefined", "remainder"}:
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

    def encode_line(self, line: str) -> list[int]:
        return self.encode_fields(
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
        return self.encode_fields(
            fields,
            include_eos=False,
            append_trailing_separator=True,
        )

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


class TokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: list[int], sequence_length: int) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, (self.data.numel() - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.sequence_length
        stop = start + self.sequence_length + 1
        chunk = self.data[start:stop]
        return chunk[:-1], chunk[1:]


def load_token_stream(file_path: Path, tokenizer: ArithmeticTokenizer) -> list[int]:
    token_ids: list[int] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                token_ids.extend(tokenizer.encode_line(line))
    return token_ids
