from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TextIO

from .config import (
    SAMPLED_KIND_MAX_ATTEMPTS,
    SAMPLED_TRAIN_SAMPLES_PER_KIND,
    TEST_SAMPLES_PER_KIND,
    VAL_SAMPLES_PER_KIND,
    _BANDS_BY_NAME,
)
from .kinds import KindSpec
from .numbers import (
    fits_number_width,
    format_signed_number,
    format_unsigned_number,
    ordered_band_patterns,
    stable_hash,
)


@dataclass(frozen=True)
class Sample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int


def apply_operation(op: str, left: int, right: int) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0 or left % right != 0:
        raise ValueError(
            f"division must be exact with non-zero divisor: {left} / {right}"
        )
    return left // right


def format_sample(sample: Sample) -> str:
    return f"{' '.join(sample.expression_fields)} = <ans> {format_signed_number(sample.answer)}"


def shuffled_samples(samples: list[Sample], *, seed: int, kind: str) -> list[Sample]:
    return sorted(
        samples,
        key=lambda sample: stable_hash(f"{seed}\t{kind}\t{format_sample(sample)}"),
    )


def write_sample(handle: TextIO, sample: Sample) -> None:
    handle.write(format_sample(sample) + "\n")


def build_exhaustive_binary_samples(spec: KindSpec) -> list[Sample]:
    if spec.op is None:
        raise ValueError(f"binary kind is missing its operator: {spec}")

    samples: list[Sample] = []
    for band_names in ordered_band_patterns(spec.band_pattern):
        left_band, right_band = (_BANDS_BY_NAME[name] for name in band_names)
        for left in left_band.values():
            for right in right_band.values():
                if spec.op == "-" and left < right:
                    continue
                if spec.op == "/" and (right == 0 or left % right != 0):
                    continue

                answer = apply_operation(spec.op, left, right)
                sample = Sample(
                    category=spec.category,
                    kind=spec.name,
                    expression_fields=(
                        format_signed_number(left),
                        spec.op,
                        format_signed_number(right),
                    ),
                    answer=answer,
                )
                samples.append(sample)

    return samples


def random_band_value(
    rng: random.Random, band_name: str, *, positive_only: bool
) -> int:
    band = _BANDS_BY_NAME[band_name]
    start = max(band.start, 1) if positive_only else band.start
    if start > band.end:
        raise ValueError(f"band {band_name} cannot provide requested values")
    return rng.randint(start, band.end)


def random_three_input_candidate(spec: KindSpec, rng: random.Random) -> Sample | None:
    if spec.op is None:
        raise ValueError(f"three-input kind is missing its operator: {spec}")

    band_names = rng.choice(ordered_band_patterns(spec.band_pattern))
    values = tuple(
        random_band_value(rng, band_name, positive_only=False)
        for band_name in band_names
    )
    left, middle, right = values

    if spec.op == "+":
        first = left + middle
        answer = first + right
    elif spec.op == "-":
        first = left - middle
        answer = first - right
        if first < 0 or answer < 0:
            return None
    else:
        first = left * middle
        answer = first * right

    if not fits_number_width(first) or not fits_number_width(answer):
        return None

    return Sample(
        category=spec.category,
        kind=spec.name,
        expression_fields=(
            format_unsigned_number(left),
            spec.op,
            format_unsigned_number(middle),
            spec.op,
            format_unsigned_number(right),
        ),
        answer=answer,
    )


def random_parentheses_candidate(spec: KindSpec, rng: random.Random) -> Sample | None:
    if spec.inner_op is None or spec.outer_op is None or spec.shape is None:
        raise ValueError(f"parentheses kind is incomplete: {spec}")

    band_names = rng.choice(ordered_band_patterns(spec.band_pattern))
    left, middle, right = tuple(
        random_band_value(rng, band_name, positive_only=False)
        for band_name in band_names
    )

    if spec.shape == "left":
        first = apply_operation(spec.inner_op, left, middle)
        answer = apply_operation(spec.outer_op, first, right)
        expression_fields = (
            "(",
            format_unsigned_number(left),
            spec.inner_op,
            format_unsigned_number(middle),
            ")",
            spec.outer_op,
            format_unsigned_number(right),
        )
    else:
        first = apply_operation(spec.inner_op, middle, right)
        answer = apply_operation(spec.outer_op, left, first)
        expression_fields = (
            format_unsigned_number(left),
            spec.outer_op,
            "(",
            format_unsigned_number(middle),
            spec.inner_op,
            format_unsigned_number(right),
            ")",
        )

    if first < 0 or answer < 0:
        return None
    if not fits_number_width(first) or not fits_number_width(answer):
        return None

    return Sample(
        category=spec.category,
        kind=spec.name,
        expression_fields=expression_fields,
        answer=answer,
    )


def random_negative_candidate(spec: KindSpec, rng: random.Random) -> Sample | None:
    if spec.op is None or spec.sign_side is None:
        raise ValueError(f"negative-input kind is incomplete: {spec}")

    band_names = rng.choice(ordered_band_patterns(spec.band_pattern))
    left_magnitude = random_band_value(
        rng,
        band_names[0],
        positive_only=spec.sign_side == "left",
    )
    right_magnitude = random_band_value(
        rng,
        band_names[1],
        positive_only=spec.sign_side == "right",
    )

    left = -left_magnitude if spec.sign_side == "left" else left_magnitude
    right = -right_magnitude if spec.sign_side == "right" else right_magnitude
    answer = apply_operation(spec.op, left, right)
    if not fits_number_width(answer):
        return None

    return Sample(
        category=spec.category,
        kind=spec.name,
        expression_fields=(
            format_signed_number(left),
            spec.op,
            format_signed_number(right),
        ),
        answer=answer,
    )


def build_sampled_kind_samples(spec: KindSpec, seed: int) -> list[Sample]:
    required_holdout = VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND
    target_total = required_holdout + SAMPLED_TRAIN_SAMPLES_PER_KIND
    rng = random.Random(stable_hash(f"{seed}\t{spec.name}"))
    samples: list[Sample] = []
    local_lines: set[str] = set()

    for _ in range(SAMPLED_KIND_MAX_ATTEMPTS):
        if len(samples) >= target_total:
            break

        if spec.category == "three_input":
            candidate = random_three_input_candidate(spec, rng)
        elif spec.category == "parentheses":
            candidate = random_parentheses_candidate(spec, rng)
        elif spec.category == "negative_input":
            candidate = random_negative_candidate(spec, rng)
        else:
            raise ValueError(f"unexpected sampled category: {spec.category}")

        if candidate is None:
            continue

        line = format_sample(candidate)
        if line in local_lines:
            continue
        local_lines.add(line)
        samples.append(candidate)

    if len(samples) < required_holdout:
        raise ValueError(
            f"kind {spec.name} only yielded {len(samples)} unique samples after sampling"
        )

    return samples
