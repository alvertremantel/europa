#!/usr/bin/env python3
"""
Convert a regular math problem (e.g., "2 + 2") into the REDUX reversed-digit format.
Outputs: "<do> <calc> {200000} + {200000} = <ans> {400000}"
"""

import sys
import re


def reverse_and_pad(num_str: str, width: int = 6) -> str:
    """Reverse digits of a number and pad with zeros."""
    digits = str(int(num_str))  # Remove leading zeros, validate it's a number
    reversed_digits = digits[::-1]
    return reversed_digits.ljust(width, "0")


def promptize(math_expr: str) -> str:
    """
    Convert a math expression like "2 + 2" into reversed-digit format.

    Args:
        math_expr: A math expression like "2 + 2" or "123 + 456"

    Returns:
        Formatted prompt: "<do> <calc> {200000} + {200000} = <ans> {400000}"
    """
    # Parse the expression
    # Support: +, -, *, /, <, >
    # Pattern: number operator number
    match = re.match(r"^\s*(-?\d+)\s*([\+\-\*/<>])\s*(-?\d+)\s*$", math_expr.strip())

    if not match:
        raise ValueError(f"Invalid math expression: {math_expr}")

    operand1, operator, operand2 = match.groups()

    # Calculate the result
    try:
        a = int(operand1)
        b = int(operand2)

        if operator == "+":
            result = a + b
        elif operator == "-":
            result = a - b
        elif operator == "*":
            result = a * b
        elif operator == "/":
            if b == 0:
                raise ValueError("Division by zero")
            result = int(a / b)  # Integer division
        elif operator == "<":
            result = a < b
        elif operator == ">":
            result = a > b
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

    # Format operands and result in REDUX reversed-digit format
    operand1_fmt = format_signed(int(operand1))
    operand2_fmt = format_signed(int(operand2))
    result_fmt = (
        "true"
        if result is True
        else "false"
        if result is False
        else format_signed(int(result))
    )

    # Build the prompt
    prompt = (
        f"<do> <calc> {operand1_fmt} {operator} {operand2_fmt} = <ans> {result_fmt}"
    )
    return prompt


def format_signed(value: int) -> str:
    if value >= 0:
        return "{" + reverse_and_pad(str(value)) + "}"
    return "(" + reverse_and_pad(str(abs(value))) + ")"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: promptize_math.py '<math_expr>'")
        print("Example: promptize_math.py '2 + 2'")
        print("Example: promptize_math.py '123 + 456'")
        sys.exit(1)

    expr = " ".join(sys.argv[1:])  # Join args in case they're separated

    try:
        prompt = promptize(expr)
        print(prompt)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
