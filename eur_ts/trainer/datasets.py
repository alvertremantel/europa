from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .examples import ArithmeticExample
from .tokenizer import ArithmeticTokenizer


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
