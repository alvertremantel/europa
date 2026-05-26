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
from .answers import format_answer
from .kinds import KindSpec
from .numbers import (
    fits_number_width,
    format_signed_number,
    ordered_band_patterns,
    stable_hash,
)


@dataclass(frozen=True)
class Sample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int | bool


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


def apply_comparison(op: str, left: int, right: int) -> bool:
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    raise ValueError(f"unsupported comparison operator: {op}")


def format_sample(sample: Sample) -> str:
    return f"<do> <calc> {' '.join(sample.expression_fields)} = <ans> {format_answer(sample.answer)}"


def shuffled_samples(samples: list[Sample], *, seed: int, kind: str) -> list[Sample]:
    return sorted(
        samples,
        key=lambda sample: stable_hash(f"{seed}\t{kind}\t{format_sample(sample)}"),
    )


def write_sample(handle: TextIO, sample: Sample) -> None:
    handle.write(format_sample(sample) + "\n")


def build_exhaustive_binary_samples(spec: KindSpec) -> list[Sample]:
    if spec.op is None:
        raise ValueError(f"two-input kind is missing its operator: {spec}")

    samples: list[Sample] = []
    for band_names in ordered_band_patterns(spec.band_pattern):
        left_band, right_band = (_BANDS_BY_NAME[name] for name in band_names)
        for left in left_band.values():
            for right in right_band.values():
                if spec.category == "arithmetic" and spec.op == "-" and left < right:
                    continue
                if (
                    spec.category == "arithmetic"
                    and spec.op == "/"
                    and (right == 0 or left % right != 0)
                ):
                    continue

                answer = (
                    apply_comparison(spec.op, left, right)
                    if spec.category == "comparison"
                    else apply_operation(spec.op, left, right)
                )
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

    if spec.category == "comparison":
        true_samples = [sample for sample in samples if sample.answer is True]
        false_samples = [sample for sample in samples if sample.answer is False]
        balanced_count = min(len(true_samples), len(false_samples))
        if balanced_count == 0:
            raise ValueError(f"comparison kind cannot be balanced: {spec.name}")
        return [
            *sorted(true_samples, key=format_sample)[:balanced_count],
            *sorted(false_samples, key=format_sample)[:balanced_count],
        ]

    return samples


def random_band_value(
    rng: random.Random, band_name: str, *, positive_only: bool
) -> int:
    band = _BANDS_BY_NAME[band_name]
    start = max(band.start, 1) if positive_only else band.start
    if start > band.end:
        raise ValueError(f"band {band_name} cannot provide requested values")
    return rng.randint(start, band.end)


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

        if spec.category == "negative_input":
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
