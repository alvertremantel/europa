from __future__ import annotations

from typing import Literal

TrainingFormat = Literal["final_only"]


def format_training_line(line: str, training_format: str) -> tuple[str, str]:
    """Return a REDUX final-only training line and the format applied."""
    if training_format != "final_only":
        raise ValueError("training_format must be final_only")
    return line.strip(), "final_only"


def final_answer_from_line(line: str) -> str:
    parts = line.strip().split(" = <ans> ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid REDUX sample line: {line!r}")
    return parts[1].strip()


def extract_final_answer(text: str) -> str:
    return text.strip()
