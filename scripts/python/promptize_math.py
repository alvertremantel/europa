#!/usr/bin/env python3
"""
Convert a regular math problem (e.g., "2 + 2") into the reversed-digit prompt format.
Outputs: "<do> <calc> 02000000 + 02000000 = 04000000"
"""

import sys
import re


def reverse_and_pad(num_str: str, width: int = 8) -> str:
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
        Formatted prompt: "<do> <calc> 02000000 + 02000000 = 04000000"
    """
    # Parse the expression
    # Support: +, -, *, /
    # Pattern: number operator number
    match = re.match(r"^\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*$", math_expr.strip())

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
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

    # Format operands and result in reversed-digit format
    operand1_fmt = reverse_and_pad(operand1)
    operand2_fmt = reverse_and_pad(operand2)
    result_fmt = reverse_and_pad(str(result))

    # Build the prompt
    prompt = f"<do> <calc> {operand1_fmt} {operator} {operand2_fmt} = {result_fmt}"
    return prompt


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
