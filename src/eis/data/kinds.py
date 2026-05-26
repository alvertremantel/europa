from __future__ import annotations

from dataclasses import dataclass

from .config import BINARY_OPERATIONS, COMPARISON_OPERATIONS, _TWO_BAND_PATTERNS
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
    return f"arithmetic::{pattern_label(pattern)}::{op}"


def negative_kind_name(pattern: tuple[str, str], op: str, sign_side: str) -> str:
    return f"negative_input::{pattern_label(pattern)}::{op}::neg_{sign_side}"


def comparison_kind_name(pattern: tuple[str, str], op: str) -> str:
    return f"comparison::{pattern_label(pattern)}::{op}"


def iter_kind_specs() -> list[KindSpec]:
    specs: list[KindSpec] = []

    for pattern in _TWO_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in BINARY_OPERATIONS:
            specs.append(
                KindSpec(
                    name=binary_kind_name(pattern, op),
                    category="arithmetic",
                    band_pattern=pattern,
                    wildcard=wildcard,
                    strategy="exhaustive",
                    op=op,
                )
            )

    for pattern in _TWO_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in ("+", "-", "*"):
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

    for pattern in _TWO_BAND_PATTERNS:
        wildcard = wildcard_for_pattern(pattern)
        for op in COMPARISON_OPERATIONS:
            specs.append(
                KindSpec(
                    name=comparison_kind_name(pattern, op),
                    category="comparison",
                    band_pattern=pattern,
                    wildcard=wildcard,
                    strategy="exhaustive",
                    op=op,
                )
            )

    return specs
