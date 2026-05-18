from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eur_ts.generator.core import validate_line


@dataclass(frozen=True)
class ArithmeticExample:
    line: str
    prompt: str
    answer: str
    kind: str | None = None
    category: str | None = None
    band_pattern: tuple[str, ...] | None = None
    training_format: str | None = None


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
    parts = line.split(" = ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid sample line: {line!r}")
    return f"{parts[0]} =", parts[1]


def _band_pattern_from_kind(kind: str) -> tuple[str, ...] | None:
    parts = kind.split("::")
    if len(parts) < 2:
        return None
    pattern = parts[2] if parts[0] == "parentheses" and len(parts) >= 3 else parts[1]
    return tuple(pattern.split("-")) if pattern else None
