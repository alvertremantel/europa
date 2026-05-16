from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .config import BINARY_OPERATIONS, COMPOSITE_OPERATIONS
from .kinds import (
    binary_kind_name,
    negative_kind_name,
    parentheses_kind_name,
    three_input_kind_name,
)
from .numbers import (
    canonical_band_pattern,
    fits_number_width,
    parse_signed_number,
    parse_unsigned_number,
)
from .sampling import Sample, apply_operation, format_sample


@dataclass(frozen=True)
class ParsedSample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int


def parse_line(line: str) -> ParsedSample:
    parts = line.strip().split()
    if len(parts) < 6 or parts[-3] != "=" or parts[-2] != "<ans>":
        raise ValueError(f"invalid sample format: {line!r}")

    expression_fields = tuple(parts[:-3])
    answer = parse_signed_number(parts[-1])

    if len(expression_fields) == 3:
        return parse_binary_expression(expression_fields, answer)
    if len(expression_fields) == 5:
        return parse_three_input_expression(expression_fields, answer)
    if len(expression_fields) == 7:
        return parse_parentheses_expression(expression_fields, answer)
    raise ValueError(f"unsupported expression shape: {line!r}")


def parse_binary_expression(
    expression_fields: tuple[str, ...], answer: int
) -> ParsedSample:
    left = parse_signed_number(expression_fields[0])
    op = expression_fields[1]
    right = parse_signed_number(expression_fields[2])
    if op not in BINARY_OPERATIONS:
        raise ValueError(f"unsupported binary operator: {expression_fields!r}")

    expected = apply_operation(op, left, right)
    if answer != expected:
        raise ValueError(
            f"arithmetic mismatch: {expression_fields!r} -> {answer} != {expected}"
        )

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
            answer=answer,
        )

    pattern = canonical_band_pattern((left, right))
    return ParsedSample(
        category="binary",
        kind=binary_kind_name(cast(tuple[str, str], pattern), op),
        expression_fields=expression_fields,
        answer=answer,
    )


def parse_three_input_expression(
    expression_fields: tuple[str, ...], answer: int
) -> ParsedSample:
    left = parse_unsigned_number(expression_fields[0])
    first_op = expression_fields[1]
    middle = parse_unsigned_number(expression_fields[2])
    second_op = expression_fields[3]
    right = parse_unsigned_number(expression_fields[4])

    if first_op != second_op or first_op not in COMPOSITE_OPERATIONS:
        raise ValueError(f"unsupported three-input expression: {expression_fields!r}")

    first = apply_operation(first_op, left, middle)
    expected = apply_operation(first_op, first, right)
    if first < 0 or expected < 0:
        raise ValueError(
            f"three-input subtraction must stay non-negative: {expression_fields!r}"
        )
    if not fits_number_width(first) or not fits_number_width(expected):
        raise ValueError(f"three-input expression exceeds width: {expression_fields!r}")
    if answer != expected:
        raise ValueError(
            f"arithmetic mismatch: {expression_fields!r} -> {answer} != {expected}"
        )

    pattern = canonical_band_pattern((left, middle, right))
    return ParsedSample(
        category="three_input",
        kind=three_input_kind_name(cast(tuple[str, str, str], pattern), first_op),
        expression_fields=expression_fields,
        answer=answer,
    )


def parse_parentheses_expression(
    expression_fields: tuple[str, ...], answer: int
) -> ParsedSample:
    if expression_fields[0] == "(" and expression_fields[4] == ")":
        left = parse_unsigned_number(expression_fields[1])
        inner_op = expression_fields[2]
        middle = parse_unsigned_number(expression_fields[3])
        outer_op = expression_fields[5]
        right = parse_unsigned_number(expression_fields[6])
        shape = "left"
        first = apply_operation(inner_op, left, middle)
        expected = apply_operation(outer_op, first, right)
    elif expression_fields[2] == "(" and expression_fields[6] == ")":
        left = parse_unsigned_number(expression_fields[0])
        outer_op = expression_fields[1]
        middle = parse_unsigned_number(expression_fields[3])
        inner_op = expression_fields[4]
        right = parse_unsigned_number(expression_fields[5])
        shape = "right"
        first = apply_operation(inner_op, middle, right)
        expected = apply_operation(outer_op, left, first)
    else:
        raise ValueError(f"unsupported parentheses expression: {expression_fields!r}")

    if inner_op not in COMPOSITE_OPERATIONS or outer_op not in COMPOSITE_OPERATIONS:
        raise ValueError(f"unsupported parentheses operators: {expression_fields!r}")
    if first < 0 or expected < 0:
        raise ValueError(
            f"parentheses expression must stay non-negative: {expression_fields!r}"
        )
    if not fits_number_width(first) or not fits_number_width(expected):
        raise ValueError(f"parentheses expression exceeds width: {expression_fields!r}")
    if answer != expected:
        raise ValueError(
            f"arithmetic mismatch: {expression_fields!r} -> {answer} != {expected}"
        )

    pattern = canonical_band_pattern((left, middle, right))
    return ParsedSample(
        category="parentheses",
        kind=parentheses_kind_name(
            cast(tuple[str, str, str], pattern),
            shape,
            inner_op,
            outer_op,
        ),
        expression_fields=expression_fields,
        answer=answer,
    )


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
