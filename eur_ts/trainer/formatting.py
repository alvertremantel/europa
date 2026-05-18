from __future__ import annotations

from typing import Literal

from eur_ts.generator.core import (
    apply_operation,
    format_signed_number,
    parse_signed_number,
    parse_unsigned_number,
    validate_line,
)

TrainingFormat = Literal[
    "final_only",
    "light_scratchpad",
    "parentheses_intermediate",
    "multiply_intermediate",
]


def format_training_line(line: str, training_format: str) -> tuple[str, str]:
    """Return a training line and the concrete format that was applied."""
    if training_format == "final_only":
        return line.strip(), "final_only"
    if training_format == "parentheses_intermediate":
        transformed = _format_parentheses_intermediate(line)
        return (
            (transformed, "parentheses_intermediate")
            if transformed
            else (line.strip(), "final_only")
        )
    if training_format == "multiply_intermediate":
        transformed = _format_multiply_intermediate(line)
        return (
            (transformed, "multiply_intermediate")
            if transformed
            else (line.strip(), "final_only")
        )
    if training_format == "light_scratchpad":
        transformed = _format_parentheses_intermediate(line)
        if transformed is not None:
            return transformed, "parentheses_intermediate"
        transformed = _format_multiply_intermediate(line)
        if transformed is not None:
            return transformed, "multiply_intermediate"
        return line.strip(), "final_only"
    raise ValueError(
        "training_format must be one of final_only, light_scratchpad, "
        "parentheses_intermediate, multiply_intermediate"
    )


def final_answer_from_line(line: str) -> str:
    answer_part = line.strip().split(" = ", maxsplit=1)[1]
    if " <final> " in answer_part:
        return answer_part.rsplit(" <final> ", maxsplit=1)[1].strip()
    return answer_part.strip()


def extract_final_answer(text: str) -> str:
    stripped = text.strip()
    if " <final> " in stripped:
        stripped = stripped.rsplit(" <final> ", maxsplit=1)[1].strip()
        return stripped.split(maxsplit=1)[0] if stripped else ""
    if "<final>" in stripped:
        stripped = stripped.rsplit("<final>", maxsplit=1)[1].strip()
        return stripped.split(maxsplit=1)[0] if stripped else ""
    return stripped


def _format_parentheses_intermediate(line: str) -> str | None:
    parsed = validate_line(line)
    if parsed.category != "parentheses":
        return None

    fields = parsed.expression_fields
    if fields[0] == "(" and fields[4] == ")":
        left = parse_unsigned_number(fields[1])
        inner_op = fields[2]
        middle = parse_unsigned_number(fields[3])
        intermediate = apply_operation(inner_op, left, middle)
    elif fields[2] == "(" and fields[6] == ")":
        middle = parse_unsigned_number(fields[3])
        inner_op = fields[4]
        right = parse_unsigned_number(fields[5])
        intermediate = apply_operation(inner_op, middle, right)
    else:  # validate_line should prevent this.
        return None

    return _with_work(line, format_signed_number(intermediate))


def _format_multiply_intermediate(line: str) -> str | None:
    parsed = validate_line(line)
    fields = parsed.expression_fields
    if parsed.category == "binary" and len(fields) == 3 and fields[1] == "*":
        left = parse_signed_number(fields[0])
        right = parse_signed_number(fields[2])
        intermediate = apply_operation("*", left, right)
        return _with_work(line, format_signed_number(intermediate))
    if parsed.category == "three_input" and len(fields) == 5 and fields[1] == "*":
        left = parse_unsigned_number(fields[0])
        middle = parse_unsigned_number(fields[2])
        intermediate = apply_operation("*", left, middle)
        return _with_work(line, format_signed_number(intermediate))
    return None


def _with_work(line: str, intermediate: str) -> str:
    prompt, answer = line.strip().split(" = ", maxsplit=1)
    return f"{prompt} = <work> <step> {intermediate} <final> {answer}"
