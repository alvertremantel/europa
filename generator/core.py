from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations
from pathlib import Path
from typing import Mapping, TextIO, cast


SPLITS = ("train", "val", "test")
BINARY_OPERATIONS = ("+", "-", "*", "/")
COMPOSITE_OPERATIONS = ("+", "-", "*")
NUMBER_WIDTH = 8
VAL_SAMPLES_PER_KIND = 16
TEST_SAMPLES_PER_KIND = 16
SAMPLED_TRAIN_SAMPLES_PER_KIND = 128
SAMPLED_KIND_MAX_ATTEMPTS = 200_000


@dataclass(frozen=True)
class Band:
    name: str
    start: int
    end: int

    def values(self) -> range:
        return range(self.start, self.end + 1)


BANDS = (
    Band("small", 0, 20),
    Band("medium", 21, 100),
    Band("large", 101, 500),
)
_BANDS_BY_NAME = {band.name: band for band in BANDS}
_BAND_NAMES = tuple(band.name for band in BANDS)
_BAND_ORDER = {band.name: index for index, band in enumerate(BANDS)}
_TWO_BAND_PATTERNS = tuple(combinations_with_replacement(_BAND_NAMES, 2))
_THREE_BAND_PATTERNS = tuple(combinations_with_replacement(_BAND_NAMES, 3))


@dataclass(frozen=True)
class Sample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class ParsedSample:
    category: str
    kind: str
    expression_fields: tuple[str, ...]
    answer: int


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


@dataclass(frozen=True)
class Config:
    seed: int = 42
    output_dir: str = "data"
    validate: bool = True


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def generate_dataset(config: Config) -> None:
    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)

    kind_specs = iter_kind_specs()
    candidate_samples: dict[str, list[Sample]] = {}
    candidate_counts: dict[str, int] = {}
    kind_definitions: dict[str, dict[str, object]] = {}
    skipped_kinds: dict[str, str] = {}
    category_counts = Counter()
    global_lines: set[str] = set()

    for spec in kind_specs:
        try:
            samples = (
                build_exhaustive_binary_samples(spec)
                if spec.strategy == "exhaustive"
                else build_sampled_kind_samples(spec, config.seed)
            )
        except ValueError as error:
            if spec.strategy == "sampled":
                skipped_kinds[spec.name] = str(error)
                print(f"[{spec.name}] skipped: {error}")
                continue
            raise

        samples = shuffled_samples(samples, seed=config.seed, kind=spec.name)
        for sample in samples:
            line = format_sample(sample)
            if line in global_lines:
                raise ValueError(f"sample leaked across kinds: {line!r}")
            global_lines.add(line)

        candidate_samples[spec.name] = samples
        candidate_counts[spec.name] = len(samples)
        category_counts[spec.category] += len(samples)
        kind_definitions[spec.name] = {
            "category": spec.category,
            "band_pattern": list(spec.band_pattern),
            "wildcard": spec.wildcard,
            "strategy": spec.strategy,
            "op": spec.op,
            "inner_op": spec.inner_op,
            "outer_op": spec.outer_op,
            "shape": spec.shape,
            "sign_side": spec.sign_side,
            "selected_count": len(samples),
        }
        print(f"[{spec.name}] unique candidates: {len(samples):,}")

    split_counts = {split: 0 for split in SPLITS}
    split_category_counts = {split: Counter() for split in SPLITS}
    split_kind_counts = {split: Counter() for split in SPLITS}
    split_files: dict[str, TextIO] = {}

    try:
        for split in SPLITS:
            split_files[split] = (output_dir / f"{split}.txt").open(
                "w", encoding="utf-8", newline="\n"
            )

        for spec in kind_specs:
            if spec.name not in candidate_samples:
                continue

            samples = candidate_samples[spec.name]
            val_samples = samples[:VAL_SAMPLES_PER_KIND]
            test_samples = samples[
                VAL_SAMPLES_PER_KIND : VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND
            ]
            train_samples = samples[VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND :]

            for split, split_samples in (
                ("train", train_samples),
                ("val", val_samples),
                ("test", test_samples),
            ):
                for sample in split_samples:
                    write_sample(split_files[split], sample)
                    split_counts[split] += 1
                    split_category_counts[split][sample.category] += 1
                    split_kind_counts[split][sample.kind] += 1

            print(
                f"[{spec.name}] train={len(train_samples):,} val={len(val_samples):,} test={len(test_samples):,}"
            )
    finally:
        for handle in split_files.values():
            handle.close()

    metadata = {
        "format": "fixed-width-reversed-infix",
        "number_width": NUMBER_WIDTH,
        "number_encoding": "zero-padded decimal reversed",
        "categories": ["binary", "three_input", "parentheses", "negative_input"],
        "binary_operations": list(BINARY_OPERATIONS),
        "composite_operations": list(COMPOSITE_OPERATIONS),
        "special_tokens": ["<ans>"],
        "operator_tokens": ["+", "-", "*", "/", "=", "(", ")"],
        "bands": {band.name: [band.start, band.end] for band in BANDS},
        "kind_definitions": kind_definitions,
        "candidate_counts": candidate_counts,
        "category_candidate_counts": dict(category_counts),
        "skipped_kinds": skipped_kinds,
        "split_counts": split_counts,
        "split_category_counts": {
            split: dict(split_category_counts[split]) for split in SPLITS
        },
        "split_kind_counts": {
            split: dict(split_kind_counts[split]) for split in SPLITS
        },
        "total_unique_candidates": sum(candidate_counts.values()),
        "seed": config.seed,
        "val_samples_per_kind": VAL_SAMPLES_PER_KIND,
        "test_samples_per_kind": TEST_SAMPLES_PER_KIND,
        "sampled_train_samples_per_kind": SAMPLED_TRAIN_SAMPLES_PER_KIND,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if config.validate:
        validate_output(output_dir=output_dir, metadata=metadata)


def validate_output(*, output_dir: Path, metadata: Mapping[str, object]) -> None:
    seen_lines: set[str] = set()
    split_counts = Counter()
    split_category_counts = {split: Counter() for split in SPLITS}
    split_kind_counts = {split: Counter() for split in SPLITS}
    kind_definitions = cast(dict[str, dict[str, object]], metadata["kind_definitions"])

    for split in SPLITS:
        path = output_dir / f"{split}.txt"
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line in seen_lines:
                    raise ValueError(f"duplicate sample found across splits: {line!r}")
                seen_lines.add(line)

                sample = validate_line(line)
                if sample.kind not in kind_definitions:
                    raise ValueError(f"sample mapped to unknown kind: {sample.kind}")

                split_counts[split] += 1
                split_category_counts[split][sample.category] += 1
                split_kind_counts[split][sample.kind] += 1

    expected_split_counts = metadata["split_counts"]
    if dict(split_counts) != expected_split_counts:
        raise ValueError(
            f"split counts mismatch: {dict(split_counts)} != {expected_split_counts}"
        )

    expected_category_counts = metadata["split_category_counts"]
    actual_category_counts = {
        split: dict(split_category_counts[split]) for split in SPLITS
    }
    if actual_category_counts != expected_category_counts:
        raise ValueError(
            f"category counts mismatch: {actual_category_counts} != {expected_category_counts}"
        )

    expected_kind_counts = metadata["split_kind_counts"]
    actual_kind_counts = {split: dict(split_kind_counts[split]) for split in SPLITS}
    if actual_kind_counts != expected_kind_counts:
        raise ValueError(
            f"kind counts mismatch: {actual_kind_counts} != {expected_kind_counts}"
        )

    summary = {
        "split_counts": dict(split_counts),
        "split_category_counts": actual_category_counts,
        "total_unique_rows": sum(split_counts.values()),
        "wildcard_eval_rows": sum(
            count
            for split in ("val", "test")
            for kind, count in split_kind_counts[split].items()
            if bool(kind_definitions[kind]["wildcard"])
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
