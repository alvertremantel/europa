from __future__ import annotations

import hashlib
from itertools import permutations

from .config import BANDS, NUMBER_WIDTH, _BAND_ORDER


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def format_unsigned_number(value: int) -> str:
    if value < 0:
        raise ValueError(f"numbers must be non-negative, got {value}")
    if value >= 10**NUMBER_WIDTH:
        raise ValueError(f"value {value} exceeds {NUMBER_WIDTH}-digit width")
    return f"{value:0{NUMBER_WIDTH}d}"[::-1]


def parse_unsigned_number(text: str) -> int:
    if len(text) != NUMBER_WIDTH or not text.isdigit():
        raise ValueError(f"invalid formatted number: {text!r}")
    value = int(text[::-1])
    if format_unsigned_number(value) != text:
        raise ValueError(f"non-canonical formatted number: {text!r}")
    return value


def format_signed_number(value: int) -> str:
    if value >= 0:
        return format_unsigned_number(value)
    return f"(-{format_unsigned_number(-value)})"


def parse_signed_number(text: str) -> int:
    if text.startswith("(-") and text.endswith(")"):
        magnitude = parse_unsigned_number(text[2:-1])
        if magnitude == 0:
            raise ValueError("negative zero is not allowed")
        return -magnitude
    return parse_unsigned_number(text)


def fits_number_width(value: int) -> bool:
    return abs(value) < 10**NUMBER_WIDTH


def classify_band(value: int) -> str:
    for band in BANDS:
        if band.start <= value <= band.end:
            return band.name
    raise ValueError(f"value {value} is outside configured bands")


def canonical_band_pattern(
    values: tuple[int, ...], *, absolute: bool = False
) -> tuple[str, ...]:
    band_names = [classify_band(abs(value) if absolute else value) for value in values]
    return tuple(sorted(band_names, key=_BAND_ORDER.__getitem__))


def pattern_label(pattern: tuple[str, ...]) -> str:
    return "-".join(pattern)


def wildcard_for_pattern(pattern: tuple[str, ...]) -> bool:
    return len(set(pattern)) > 1


def ordered_band_patterns(pattern: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    unique = {ordering for ordering in permutations(pattern)}
    return tuple(sorted(unique, key=lambda ordering: [*_band_sort_key(ordering)]))


def _band_sort_key(ordering: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(_BAND_ORDER[name] for name in ordering)
