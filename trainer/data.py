from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from generator.core import validate_line

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

SPECIAL_FIELD_TOKENS = {"<ans>", "<work>", "<step>", "<final>", "undefined", "remainder"}


@dataclass(frozen=True)
class ArithmeticExample:
    line: str
    prompt: str
    answer: str
    kind: str | None = None
    category: str | None = None
    band_pattern: tuple[str, ...] | None = None
    training_format: str | None = None


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


class ExampleSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        examples: Sequence[ArithmeticExample],
        tokenizer: ArithmeticTokenizer,
        sequence_length: int,
        *,
        skip_overlong: bool = False,
    ) -> None:
        self.examples: list[ArithmeticExample] = []
        self.items: list[tuple[Tensor, Tensor, Tensor]] = []
        self.sequence_length = sequence_length
        self.skipped_by_format: dict[str, int] = {}

        for example in examples:
            token_ids = tokenizer.encode_line(example.line)
            if len(token_ids) > sequence_length + 1:
                training_format = example.training_format or "unknown"
                self.skipped_by_format[training_format] = (
                    self.skipped_by_format.get(training_format, 0) + 1
                )
                if skip_overlong:
                    continue
                raise ValueError(
                    f"example exceeds sequence length {sequence_length}: "
                    f"{len(token_ids) - 1} input tokens needed for {example.line!r}"
                )

            padded = list(token_ids)
            padded.extend([tokenizer.pad_id] * (sequence_length + 1 - len(padded)))
            data = torch.tensor(padded, dtype=torch.long)
            inputs = data[:-1]
            targets = data[1:]
            loss_mask = targets.ne(tokenizer.pad_id)
            self.examples.append(example)
            self.items.append((inputs, targets, loss_mask))

        if not self.items:
            raise ValueError("no examples remain after sequence-length filtering")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.items[index]


def load_token_stream(file_path: Path, tokenizer: ArithmeticTokenizer) -> list[int]:
    token_ids: list[int] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                token_ids.extend(tokenizer.encode_line(line))
    return token_ids


def load_examples(
    file_path: Path,
    *,
    include_metadata: bool = False,
    training_format: str | None = None,
) -> list[ArithmeticExample]:
    examples: list[ArithmeticExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            prompt, answer = _split_line(line)
            kind = None
            category = None
            band_pattern: tuple[str, ...] | None = None
            if include_metadata:
                parsed = validate_line(line)
                kind = parsed.kind
                category = parsed.category
                band_pattern = _band_pattern_from_kind(kind)
            examples.append(
                ArithmeticExample(
                    line=line,
                    prompt=prompt,
                    answer=answer,
                    kind=kind,
                    category=category,
                    band_pattern=band_pattern,
                    training_format=training_format,
                )
            )
    return examples


def transform_examples(
    examples: Sequence[ArithmeticExample],
    *,
    training_format: str,
) -> list[ArithmeticExample]:
    from .formatting import format_training_line, final_answer_from_line

    transformed: list[ArithmeticExample] = []
    for example in examples:
        line, applied_format = format_training_line(example.line, training_format)
        transformed.append(
            ArithmeticExample(
                line=line,
                prompt=example.prompt,
                answer=final_answer_from_line(line),
                kind=example.kind,
                category=example.category,
                band_pattern=example.band_pattern,
                training_format=applied_format,
            )
        )
    return transformed


def _split_line(line: str) -> tuple[str, str]:
    parts = line.split(" <ans> ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid sample line: {line!r}")
    return f"{parts[0]} <ans>", parts[1]


def _band_pattern_from_kind(kind: str) -> tuple[str, ...] | None:
    parts = kind.split("::")
    if len(parts) < 2:
        return None
    pattern = parts[2] if parts[0] == "parentheses" and len(parts) >= 3 else parts[1]
    return tuple(pattern.split("-")) if pattern else None
