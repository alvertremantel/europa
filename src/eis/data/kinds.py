from __future__ import annotations

from dataclasses import dataclass

from .config import COMPOSITE_OPERATIONS, BINARY_OPERATIONS, _THREE_BAND_PATTERNS, _TWO_BAND_PATTERNS
from .numbers import pattern_label, wildcard_for_pattern


@dataclass(frozen=True)
class KindSpec:
    name: str
    category: str
    band_pattern: tuple[str, ...]
    wildcard: bool
    strategy: str
    op: str | None = None
    inner_op: str | None = None
    outer_op: str | None = None
    shape: str | None = None
    sign_side: str | None = None


def binary_kind_name(pattern: tuple[str, str], op: str) -> str:
    return f"binary::{pattern_label(pattern)}::{op}"


def three_input_kind_name(pattern: tuple[str, str, str], op: str) -> str:
    return f"three_input::{pattern_label(pattern)}::{op}"


def parentheses_kind_name(
    pattern: tuple[str, str, str], shape: str, inner_op: str, outer_op: str
) -> str:
    return f"parentheses::{shape}::{pattern_label(pattern)}::{inner_op}{outer_op}"


def negative_kind_name(pattern: tuple[str, str], op: str, sign_side: str) -> str:
    return f"negative_input::{pattern_label(pattern)}::{op}::neg_{sign_side}"


def iter_kind_specs() -> list[KindSpec]:
    specs: list[KindSpec] = []

    for pattern in _TWO_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in BINARY_OPERATIONS:
            specs.append(
                KindSpec(
                    name=binary_kind_name(pattern, op),
                    category="binary",
                    band_pattern=pattern,
                    wildcard=wildcard,
                    strategy="exhaustive",
                    op=op,
                )
            )

    for pattern in _THREE_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in COMPOSITE_OPERATIONS:
            specs.append(
                KindSpec(
                    name=three_input_kind_name(pattern, op),
                    category="three_input",
                    band_pattern=pattern,
                    wildcard=wildcard,
                    strategy="sampled",
                    op=op,
                )
            )

    for pattern in _THREE_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for shape in ("left", "right"):
            for inner_op in COMPOSITE_OPERATIONS:
                for outer_op in COMPOSITE_OPERATIONS:
                    specs.append(
                        KindSpec(
                            name=parentheses_kind_name(
                                pattern,
                                shape,
                                inner_op,
                                outer_op,
                            ),
                            category="parentheses",
                            band_pattern=pattern,
                            wildcard=wildcard,
                            strategy="sampled",
                            inner_op=inner_op,
                            outer_op=outer_op,
                            shape=shape,
                        )
                    )

    for pattern in _TWO_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in COMPOSITE_OPERATIONS:
            for sign_side in ("left", "right"):
                specs.append(
                    KindSpec(
                        name=negative_kind_name(pattern, op, sign_side),
                        category="negative_input",
                        band_pattern=pattern,
                        wildcard=wildcard,
                        strategy="sampled",
                        op=op,
                        sign_side=sign_side,
                    )
                )

    return specs
