from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .numbers import format_signed_number, parse_signed_number


AnswerKind = Literal["numeric", "boolean"]


@dataclass(frozen=True)
class Answer:
    kind: AnswerKind
    value: int | bool

    def format(self) -> str:
        if self.kind == "numeric":
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise TypeError("numeric answers require an integer value")
            return format_signed_number(self.value)
        if not isinstance(self.value, bool):
            raise TypeError("boolean answers require a bool value")
        return "true" if self.value else "false"


def numeric_answer(value: int) -> Answer:
    return Answer(kind="numeric", value=value)


def boolean_answer(value: bool) -> Answer:
    return Answer(kind="boolean", value=value)


def parse_answer(text: str) -> Answer:
    if text == "true":
        return boolean_answer(True)
    if text == "false":
        return boolean_answer(False)
    return numeric_answer(parse_signed_number(text))


def format_answer(value: Answer | int | bool) -> str:
    if isinstance(value, Answer):
        return value.format()
    if isinstance(value, bool):
        return boolean_answer(value).format()
    return numeric_answer(value).format()


def is_canonical_answer(text: str) -> bool:
    try:
        return parse_answer(text).format() == text
    except ValueError:
        return False
