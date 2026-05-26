from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, TextIO, cast

from eis.artifacts import toml_text, write_toml

from .config import (
    BANDS,
    BINARY_OPERATIONS,
    COMPARISON_OPERATIONS,
    COMPOSITE_OPERATIONS,
    NUMBER_WIDTH,
    SPLITS,
    SAMPLED_TRAIN_SAMPLES_PER_KIND,
    TEST_SAMPLES_PER_KIND,
    VAL_SAMPLES_PER_KIND,
    Config,
)
from .kinds import KindSpec
from .kinds import iter_kind_specs
from .parsing import validate_line
from .sampling import (
    Sample,
    build_exhaustive_binary_samples,
    build_sampled_kind_samples,
    format_sample,
    shuffled_samples,
    write_sample,
)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _split_kind_samples(
    spec: KindSpec, samples: list[Sample]
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if spec.category != "comparison":
        val_samples = samples[:VAL_SAMPLES_PER_KIND]
        test_samples = samples[
            VAL_SAMPLES_PER_KIND : VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND
        ]
        train_samples = samples[VAL_SAMPLES_PER_KIND + TEST_SAMPLES_PER_KIND :]
        return train_samples, val_samples, test_samples

    if VAL_SAMPLES_PER_KIND % 2 or TEST_SAMPLES_PER_KIND % 2:
        raise ValueError("comparison holdout sizes must be even")

    by_answer: dict[bool, list[Sample]] = {True: [], False: []}
    for sample in samples:
        if not isinstance(sample.answer, bool):
            raise ValueError(
                f"comparison sample has non-boolean answer: {format_sample(sample)}"
            )
        by_answer[sample.answer].append(sample)

    val_half = VAL_SAMPLES_PER_KIND // 2
    test_half = TEST_SAMPLES_PER_KIND // 2
    holdout = val_half + test_half
    if min(len(bucket) for bucket in by_answer.values()) < holdout:
        raise ValueError(
            f"comparison kind cannot supply balanced holdouts: {spec.name}"
        )

    val_samples = by_answer[True][:val_half] + by_answer[False][:val_half]
    test_samples = (
        by_answer[True][val_half:holdout] + by_answer[False][val_half:holdout]
    )
    train_samples = by_answer[True][holdout:] + by_answer[False][holdout:]
    return train_samples, val_samples, test_samples


def _build_data_mix_metadata(
    split_category_counts: Mapping[str, Counter[str]],
) -> dict[str, object]:
    train_counts = dict(split_category_counts["train"])
    arithmetic_count = train_counts.get("arithmetic", 0)
    negative_input_count = train_counts.get("negative_input", 0)
    comparison_count = train_counts.get("comparison", 0)
    computation_count = arithmetic_count + negative_input_count
    comparison_ratio = (
        comparison_count / computation_count if computation_count else None
    )

    return {
        "train_category_counts": train_counts,
        "computation_train_count": computation_count,
        "comparison_train_count": comparison_count,
        "comparison_to_computation_ratio": comparison_ratio,
        "generation_policy": {
            "arithmetic": "exhaustive per kind after fixed validation/test holdouts",
            "negative_input": (
                "sampled per kind with fixed train quota plus validation/test holdouts"
            ),
            "comparison": (
                "exhaustive per kind, balanced overall by answer class, and split with equal true/false counts in train/val/test"
            ),
        },
        "note": (
            "This REDUX baseline records the actual generated category mix instead of claiming a fixed comparison/computation target."
        ),
    }


def _validate_comparison_balance(
    split_comparison_answer_counts: Mapping[str, Mapping[str, Counter[bool]]],
    kind_definitions: Mapping[str, Mapping[str, object]],
) -> None:
    comparison_kinds = [
        kind
        for kind, definition in kind_definitions.items()
        if definition["category"] == "comparison"
    ]
    for split in SPLITS:
        for kind in comparison_kinds:
            counts = split_comparison_answer_counts[split].get(kind, Counter())
            if counts[True] != counts[False]:
                raise ValueError(
                    f"comparison balance mismatch for {split}/{kind}: true={counts[True]} false={counts[False]}"
                )


def _validate_data_mix_metadata(
    *, metadata: Mapping[str, object], split_category_counts: Mapping[str, Counter[str]]
) -> None:
    data_mix = cast(dict[str, object], metadata["data_mix"])
    expected_train_counts = cast(dict[str, int], data_mix["train_category_counts"])
    actual_train_counts = dict(split_category_counts["train"])
    if actual_train_counts != expected_train_counts:
        raise ValueError(
            f"data mix train counts mismatch: {actual_train_counts} != {expected_train_counts}"
        )

    computation_count = actual_train_counts.get(
        "arithmetic", 0
    ) + actual_train_counts.get("negative_input", 0)
    expected_ratio = cast(float | None, data_mix["comparison_to_computation_ratio"])
    actual_ratio = (
        actual_train_counts.get("comparison", 0) / computation_count
        if computation_count
        else None
    )
    if actual_ratio != expected_ratio:
        raise ValueError(
            f"data mix comparison ratio mismatch: {actual_ratio} != {expected_ratio}"
        )


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
            train_samples, val_samples, test_samples = _split_kind_samples(
                spec, samples
            )

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
        "format": "redux-fixed-width-reversed-infix-v1",
        "number_width": NUMBER_WIDTH,
        "number_encoding": "six-digit zero-padded decimal reversed, wrapped as {digits} or (digits)",
        "categories": ["arithmetic", "negative_input", "comparison"],
        "binary_operations": list(BINARY_OPERATIONS),
        "comparison_operations": list(COMPARISON_OPERATIONS),
        "composite_operations": list(COMPOSITE_OPERATIONS),
        "special_tokens": ["<do>", "<calc>", "<ans>", "<eos>"],
        "reserved_tokens": [
            "<pad> (batching only; excluded from generated examples and ignored in loss)"
        ],
        "operator_tokens": ["+", "-", "*", "/", "<", ">", "=", "(", ")", "{", "}"],
        "data_mix": _build_data_mix_metadata(split_category_counts),
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
    write_toml(output_dir / "meta.toml", metadata)

    if config.validate:
        validate_output(output_dir=output_dir, metadata=metadata)


def validate_output(*, output_dir: Path, metadata: Mapping[str, object]) -> None:
    seen_lines: set[str] = set()
    split_counts = Counter()
    split_category_counts = {split: Counter() for split in SPLITS}
    split_kind_counts = {split: Counter() for split in SPLITS}
    split_comparison_answer_counts: dict[str, dict[str, Counter[bool]]] = {
        split: {} for split in SPLITS
    }
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
                if sample.category == "comparison":
                    split_comparison_answer_counts[split].setdefault(
                        sample.kind, Counter()
                    )[bool(sample.answer)] += 1

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

    _validate_comparison_balance(split_comparison_answer_counts, kind_definitions)
    _validate_data_mix_metadata(
        metadata=metadata, split_category_counts=split_category_counts
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
    print(toml_text({"validation": summary}).rstrip())
