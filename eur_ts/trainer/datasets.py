from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .examples import ArithmeticExample
from .tokenizer import ArithmeticTokenizer, POSITION_ENCODING_TYPE_PLACE


class TokenBlockDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
        *,
        type_ids: list[int] | None = None,
        place_ids: list[int] | None = None,
    ) -> None:
        if (type_ids is None) != (place_ids is None):
            raise ValueError(
                "type IDs and place IDs must either both be set or both be omitted"
            )
        if type_ids is not None and (
            len(token_ids) != len(type_ids) or len(token_ids) != len(place_ids or [])
        ):
            raise ValueError(
                "token IDs, type IDs, and place IDs must have equal length"
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.type_data = (
            torch.tensor(type_ids, dtype=torch.long) if type_ids is not None else None
        )
        self.place_data = (
            torch.tensor(place_ids, dtype=torch.long) if place_ids is not None else None
        )
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, (self.data.numel() - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        start = index * self.sequence_length
        stop = start + self.sequence_length + 1
        chunk = self.data[start:stop]
        if self.type_data is None or self.place_data is None:
            return chunk[:-1], chunk[1:]
        type_chunk = self.type_data[start:stop]
        place_chunk = self.place_data[start:stop]
        return chunk[:-1], type_chunk[:-1], place_chunk[:-1], chunk[1:]


class ExampleSequenceDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        examples: Sequence[ArithmeticExample],
        tokenizer: ArithmeticTokenizer,
        sequence_length: int,
        *,
        position_encoding: str = POSITION_ENCODING_TYPE_PLACE,
        skip_overlong: bool = False,
    ) -> None:
        self.examples: list[ArithmeticExample] = []
        self.items: list[tuple[Tensor, ...]] = []
        self.sequence_length = sequence_length
        self.skipped_by_format: dict[str, int] = {}
        include_type_place = position_encoding == POSITION_ENCODING_TYPE_PLACE

        for example in examples:
            if include_type_place:
                token_ids, type_ids, place_ids = tokenizer.encode_line_with_type_place(
                    example.line
                )
            else:
                token_ids = tokenizer.encode_line(example.line)
                type_ids = None
                place_ids = None
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
            if include_type_place:
                assert type_ids is not None and place_ids is not None
                padded_types = list(type_ids)
                padded_types.extend([0] * (sequence_length + 1 - len(padded_types)))
                padded_places = list(place_ids)
                padded_places.extend([0] * (sequence_length + 1 - len(padded_places)))
                type_data = torch.tensor(padded_types, dtype=torch.long)
                place_data = torch.tensor(padded_places, dtype=torch.long)
                self.items.append(
                    (inputs, type_data[:-1], place_data[:-1], targets, loss_mask)
                )
            else:
                self.items.append((inputs, targets, loss_mask))

        if not self.items:
            raise ValueError("no examples remain after sequence-length filtering")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return self.items[index]


def load_token_stream(file_path: Path, tokenizer: ArithmeticTokenizer) -> list[int]:
    token_ids: list[int] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                token_ids.extend(tokenizer.encode_line(line))
    return token_ids


def load_token_stream_with_type_place(
    file_path: Path,
    tokenizer: ArithmeticTokenizer,
) -> tuple[list[int], list[int], list[int]]:
    token_ids = load_token_stream(file_path, tokenizer)
    type_ids, place_ids = tokenizer.type_place_ids_for_token_ids(token_ids)
    return token_ids, type_ids, place_ids
