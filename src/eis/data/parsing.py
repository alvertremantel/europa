from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .answers import Answer, boolean_answer, format_answer, numeric_answer, parse_answer
from .config import BINARY_OPERATIONS, COMPARISON_OPERATIONS
from .kinds import binary_kind_name, comparison_kind_name, negative_kind_name
from .numbers import canonical_band_pattern, fits_number_width, parse_signed_number
from .sampling import Sample, apply_comparison, apply_operation, format_sample


@dataclass(frozen=True)
class ParsedSample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int | bool


def parse_line(line: str) -> ParsedSample:
    parts = line.strip().split()
    if (
        len(parts) != 8
        or parts[0] != "<do>"
        or parts[1] != "<calc>"
        or parts[-3] != "="
        or parts[-2] != "<ans>"
    ):
        raise ValueError(f"invalid sample format: {line!r}")

    expression_fields = tuple(parts[2:-3])
    answer = parse_answer(parts[-1])

    if len(expression_fields) == 3:
        return parse_binary_expression(expression_fields, answer)
    raise ValueError(f"unsupported expression shape: {line!r}")


def parse_binary_expression(
    expression_fields: tuple[str, ...], answer: Answer | int | bool
) -> ParsedSample:
    if len(expression_fields) != 3:
        raise ValueError(f"unsupported binary expression: {expression_fields!r}")
    parsed_answer = _coerce_answer(answer)
    left = parse_signed_number(expression_fields[0])
    op = expression_fields[1]
    right = parse_signed_number(expression_fields[2])

    if op in BINARY_OPERATIONS:
        expected = apply_operation(op, left, right)
        if parsed_answer != numeric_answer(expected):
            raise ValueError(
                f"arithmetic mismatch: {expression_fields!r} -> {format_answer(parsed_answer)} != {format_answer(expected)}"
            )
        if not fits_number_width(expected):
            raise ValueError(f"arithmetic answer exceeds width: {expression_fields!r}")
        if left < 0 and right < 0:
            raise ValueError(
                f"two-negative inputs are not supported: {expression_fields!r}"
            )
        if left < 0 or right < 0:
            if op == "/":
                raise ValueError(
                    f"negative division is not generated: {expression_fields!r}"
                )
            pattern = canonical_band_pattern((left, right), absolute=True)
            sign_side = "left" if left < 0 else "right"
            return ParsedSample(
                category="negative_input",
                kind=negative_kind_name(cast(tuple[str, str], pattern), op, sign_side),
                expression_fields=expression_fields,
                answer=expected,
            )
        pattern = canonical_band_pattern((left, right))
        return ParsedSample(
            category="arithmetic",
            kind=binary_kind_name(cast(tuple[str, str], pattern), op),
            expression_fields=expression_fields,
            answer=expected,
        )

    if op in COMPARISON_OPERATIONS:
        expected_bool = apply_comparison(op, left, right)
        if parsed_answer != boolean_answer(expected_bool):
            raise ValueError(
                f"comparison mismatch: {expression_fields!r} -> {format_answer(parsed_answer)} != {format_answer(expected_bool)}"
            )
        if left < 0 or right < 0:
            raise ValueError(
                f"comparison inputs must be non-negative: {expression_fields!r}"
            )
        pattern = canonical_band_pattern((left, right))
        return ParsedSample(
            category="comparison",
            kind=comparison_kind_name(cast(tuple[str, str], pattern), op),
            expression_fields=expression_fields,
            answer=expected_bool,
        )

    raise ValueError(f"unsupported binary operator: {expression_fields!r}")


def _coerce_answer(answer: Answer | int | bool) -> Answer:
    if isinstance(answer, Answer):
        return answer
    if isinstance(answer, bool):
        return boolean_answer(answer)
    return numeric_answer(answer)


def validate_line(line: str) -> ParsedSample:
    sample = parse_line(line)
    expected = format_sample(
        Sample(
            category=sample.category,
            kind=sample.kind,
            expression_fields=sample.expression_fields,
            answer=sample.answer,
        )
    )
    if line.strip() != expected:
        raise ValueError(f"non-canonical sample line: {line!r}")
    return sample
