from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class KindLengthStats:
    row_count: int = 0
    max_prompt_tokens: int = 0
    max_line_tokens: int = 0
    max_answer_tokens: int = 0
    max_generation_steps: int = 0
    max_context_for_last_answer_token: int = 0
    max_context_for_eos: int = 0
    prompt_exceeds_sequence_count: int = 0
    line_exceeds_sequence_count: int = 0
    answer_context_exceeds_sequence_count: int = 0
    eos_context_exceeds_sequence_count: int = 0
    generation_steps_exceed_max_new_tokens_count: int = 0
    max_abs_answer: int = 0
    max_abs_intermediate: int = 0
    answer_width_exceeds_count: int = 0
    intermediate_width_exceeds_count: int = 0
    answer_width_threshold_count: int = 0
    intermediate_width_threshold_count: int = 0
    max_abs_answer_line: str = ""
    max_abs_intermediate_line: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check dataset token lengths against model limits."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--kinds-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> tuple[object, int, int]:
    from eur_ts.trainer.data import ArithmeticTokenizer

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)!r}")

    tokenizer_state = payload.get("tokenizer")
    train_config = payload.get("train_config")
    if not isinstance(tokenizer_state, dict) or not isinstance(train_config, dict):
        raise ValueError("checkpoint is missing tokenizer or train_config")

    tokenizer = ArithmeticTokenizer.from_state(
        cast(dict[str, list[str]], tokenizer_state)
    )
    sequence_length = train_config.get("sequence_length")
    max_new_tokens = train_config.get("max_new_tokens")
    if not isinstance(sequence_length, int) or not isinstance(max_new_tokens, int):
        raise ValueError(
            "checkpoint train_config is missing sequence_length or max_new_tokens"
        )
    return tokenizer, sequence_length, max_new_tokens


def intermediate_abs_value(sample: object) -> int:
    from eur_ts.generator.core import (
        apply_operation,
        parse_signed_number,
        parse_unsigned_number,
    )

    expression_fields = cast(tuple[str, ...], getattr(sample, "expression_fields"))
    if len(expression_fields) == 5:
        left = parse_unsigned_number(expression_fields[0])
        middle = parse_unsigned_number(expression_fields[2])
        op = expression_fields[1]
        return abs(apply_operation(op, left, middle))

    if len(expression_fields) == 7:
        if expression_fields[0] == "(" and expression_fields[4] == ")":
            left = parse_unsigned_number(expression_fields[1])
            middle = parse_unsigned_number(expression_fields[3])
            inner_op = expression_fields[2]
            return abs(apply_operation(inner_op, left, middle))
        if expression_fields[2] == "(" and expression_fields[6] == ")":
            middle = parse_unsigned_number(expression_fields[3])
            right = parse_unsigned_number(expression_fields[5])
            inner_op = expression_fields[4]
            return abs(apply_operation(inner_op, middle, right))
        raise ValueError(f"unsupported parentheses expression: {expression_fields!r}")

    if len(expression_fields) == 3:
        left = parse_signed_number(expression_fields[0])
        right = parse_signed_number(expression_fields[2])
        _ = (left, right)
        return 0

    raise ValueError(f"unsupported expression shape: {expression_fields!r}")


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0 or variance_y == 0:
        return None
    return covariance / (variance_x * variance_y) ** 0.5


def output_path_for(kinds_csv_path: Path, requested: str | None) -> Path:
    if requested is not None:
        return Path(requested)
    return kinds_csv_path.with_suffix(".length-safety.json")


def load_eval_rows(kinds_csv_path: Path) -> list[dict[str, str]]:
    with kinds_csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [cast(dict[str, str], row) for row in csv.DictReader(handle)]


def analyze_dataset(
    *,
    data_dir: Path,
    tokenizer: object,
    sequence_length: int,
    max_new_tokens: int,
) -> tuple[dict[str, KindLengthStats], dict[str, int]]:
    from eur_ts.generator.core import validate_line
    from eur_ts.trainer.data import ArithmeticTokenizer
    from eur_ts.trainer.utils import answer_from_line, prompt_from_line

    typed_tokenizer = cast(ArithmeticTokenizer, tokenizer)
    kind_stats: dict[str, KindLengthStats] = {}
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_path = data_dir / f"{split}.txt"
        with split_path.open("r", encoding="utf-8") as handle:
            line_count = 0
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                sample = validate_line(line)
                prompt = prompt_from_line(line)
                answer = answer_from_line(line)
                answer_value = abs(cast(int, getattr(sample, "answer")))
                intermediate_value = intermediate_abs_value(sample)

                prompt_tokens = len(typed_tokenizer.encode_prompt(prompt))
                line_tokens = len(typed_tokenizer.encode_line(line))
                answer_tokens = len(typed_tokenizer.encode_field(answer))
                generation_steps = answer_tokens + 1
                context_for_last_answer_token = prompt_tokens + max(
                    answer_tokens - 1, 0
                )
                context_for_eos = prompt_tokens + answer_tokens

                stats = kind_stats.setdefault(sample.kind, KindLengthStats())
                stats.row_count += 1
                stats.max_prompt_tokens = max(stats.max_prompt_tokens, prompt_tokens)
                stats.max_line_tokens = max(stats.max_line_tokens, line_tokens)
                stats.max_answer_tokens = max(stats.max_answer_tokens, answer_tokens)
                stats.max_generation_steps = max(
                    stats.max_generation_steps, generation_steps
                )
                stats.max_context_for_last_answer_token = max(
                    stats.max_context_for_last_answer_token,
                    context_for_last_answer_token,
                )
                stats.max_context_for_eos = max(
                    stats.max_context_for_eos, context_for_eos
                )
                if answer_value > stats.max_abs_answer:
                    stats.max_abs_answer = answer_value
                    stats.max_abs_answer_line = line
                if intermediate_value > stats.max_abs_intermediate:
                    stats.max_abs_intermediate = intermediate_value
                    stats.max_abs_intermediate_line = line
                if prompt_tokens > sequence_length:
                    stats.prompt_exceeds_sequence_count += 1
                if line_tokens > sequence_length:
                    stats.line_exceeds_sequence_count += 1
                if context_for_last_answer_token > sequence_length:
                    stats.answer_context_exceeds_sequence_count += 1
                if context_for_eos > sequence_length:
                    stats.eos_context_exceeds_sequence_count += 1
                if generation_steps > max_new_tokens:
                    stats.generation_steps_exceed_max_new_tokens_count += 1
                if answer_value >= 100_000_000:
                    stats.answer_width_exceeds_count += 1
                if intermediate_value >= 100_000_000:
                    stats.intermediate_width_exceeds_count += 1
                if answer_value >= 99_000_000:
                    stats.answer_width_threshold_count += 1
                if intermediate_value >= 99_000_000:
                    stats.intermediate_width_threshold_count += 1
                line_count += 1
            split_counts[split] = line_count

    return kind_stats, split_counts


def grouped_length_summary(
    merged_rows: list[dict[str, object]], key: str
) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = {}
    for row in merged_rows:
        groups.setdefault(cast(int, row[key]), []).append(row)

    summary: list[dict[str, object]] = []
    for group_key, rows in sorted(groups.items()):
        accuracies = [cast(float, row["accuracy"]) for row in rows]
        summary.append(
            {
                key: group_key,
                "kind_count": len(rows),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "mean_accuracy": sum(accuracies) / len(accuracies),
            }
        )
    return summary


def build_report(
    *,
    eval_rows: list[dict[str, str]],
    kind_stats: dict[str, KindLengthStats],
    split_counts: dict[str, int],
    sequence_length: int,
    max_new_tokens: int,
    checkpoint_path: Path,
    data_dir: Path,
) -> dict[str, object]:
    prompt_lengths: list[float] = []
    line_lengths: list[float] = []
    answer_lengths: list[float] = []
    eos_context_lengths: list[float] = []
    max_abs_answer_values: list[float] = []
    max_abs_intermediate_values: list[float] = []
    accuracies: list[float] = []
    merged_rows: list[dict[str, object]] = []

    violation_totals = {
        "prompt_exceeds_sequence_count": 0,
        "line_exceeds_sequence_count": 0,
        "answer_context_exceeds_sequence_count": 0,
        "eos_context_exceeds_sequence_count": 0,
        "generation_steps_exceed_max_new_tokens_count": 0,
        "answer_width_exceeds_count": 0,
        "intermediate_width_exceeds_count": 0,
        "answer_width_threshold_count": 0,
        "intermediate_width_threshold_count": 0,
    }

    for row in eval_rows:
        kind = row["name"]
        if row["status"] != "evaluated":
            continue
        if kind not in kind_stats:
            raise ValueError(
                f"evaluated kind {kind!r} was not found in dataset analysis"
            )

        stats = kind_stats[kind]
        prompt_lengths.append(float(stats.max_prompt_tokens))
        line_lengths.append(float(stats.max_line_tokens))
        answer_lengths.append(float(stats.max_answer_tokens))
        eos_context_lengths.append(float(stats.max_context_for_eos))
        max_abs_answer_values.append(float(stats.max_abs_answer))
        max_abs_intermediate_values.append(float(stats.max_abs_intermediate))
        accuracy = float(row["accuracy"])
        accuracies.append(accuracy)
        for key in violation_totals:
            violation_totals[key] += getattr(stats, key)

        merged_rows.append(
            {
                "name": kind,
                "category": row["category"],
                "accuracy": accuracy,
                **asdict(stats),
            }
        )

    merged_rows.sort(key=lambda entry: (entry["max_prompt_tokens"], entry["name"]))

    longest_prompt_rows = sorted(
        merged_rows,
        key=lambda entry: (
            cast(int, entry["max_prompt_tokens"]),
            cast(int, entry["max_context_for_eos"]),
            cast(str, entry["name"]),
        ),
        reverse=True,
    )[:12]
    lowest_accuracy_rows = sorted(
        merged_rows,
        key=lambda entry: (cast(float, entry["accuracy"]), cast(str, entry["name"])),
    )[:12]

    return {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "sequence_length": sequence_length,
        "max_new_tokens": max_new_tokens,
        "split_counts": split_counts,
        "violations": violation_totals,
        "overall_maxima": {
            "max_prompt_tokens": max(
                stats.max_prompt_tokens for stats in kind_stats.values()
            ),
            "max_line_tokens": max(
                stats.max_line_tokens for stats in kind_stats.values()
            ),
            "max_answer_tokens": max(
                stats.max_answer_tokens for stats in kind_stats.values()
            ),
            "max_generation_steps": max(
                stats.max_generation_steps for stats in kind_stats.values()
            ),
            "max_context_for_last_answer_token": max(
                stats.max_context_for_last_answer_token for stats in kind_stats.values()
            ),
            "max_context_for_eos": max(
                stats.max_context_for_eos for stats in kind_stats.values()
            ),
            "max_abs_answer": max(
                stats.max_abs_answer for stats in kind_stats.values()
            ),
            "max_abs_intermediate": max(
                stats.max_abs_intermediate for stats in kind_stats.values()
            ),
        },
        "accuracy_correlations": {
            "accuracy_vs_max_prompt_tokens": pearson_correlation(
                prompt_lengths, accuracies
            ),
            "accuracy_vs_max_line_tokens": pearson_correlation(
                line_lengths, accuracies
            ),
            "accuracy_vs_max_answer_tokens": pearson_correlation(
                answer_lengths, accuracies
            ),
            "accuracy_vs_max_context_for_eos": pearson_correlation(
                eos_context_lengths, accuracies
            ),
            "accuracy_vs_max_abs_answer": pearson_correlation(
                max_abs_answer_values, accuracies
            ),
            "accuracy_vs_max_abs_intermediate": pearson_correlation(
                max_abs_intermediate_values, accuracies
            ),
        },
        "prompt_length_summary": grouped_length_summary(
            merged_rows, key="max_prompt_tokens"
        ),
        "answer_length_summary": grouped_length_summary(
            merged_rows, key="max_answer_tokens"
        ),
        "width_summary": {
            "kinds_with_answer_ge_99m": sum(
                1
                for stats in kind_stats.values()
                if stats.answer_width_threshold_count > 0
            ),
            "kinds_with_answer_ge_100m": sum(
                1
                for stats in kind_stats.values()
                if stats.answer_width_exceeds_count > 0
            ),
            "kinds_with_intermediate_ge_99m": sum(
                1
                for stats in kind_stats.values()
                if stats.intermediate_width_threshold_count > 0
            ),
            "kinds_with_intermediate_ge_100m": sum(
                1
                for stats in kind_stats.values()
                if stats.intermediate_width_exceeds_count > 0
            ),
            "max_abs_answer_kind": max(
                merged_rows,
                key=lambda entry: (
                    cast(int, entry["max_abs_answer"]),
                    cast(str, entry["name"]),
                ),
            ),
            "max_abs_intermediate_kind": max(
                merged_rows,
                key=lambda entry: (
                    cast(int, entry["max_abs_intermediate"]),
                    cast(str, entry["name"]),
                ),
            ),
        },
        "longest_prompt_kinds": longest_prompt_rows,
        "lowest_accuracy_kinds": lowest_accuracy_rows,
    }


def print_report(report: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "data_dir": report["data_dir"],
                "sequence_length": report["sequence_length"],
                "max_new_tokens": report["max_new_tokens"],
                "split_counts": report["split_counts"],
                "overall_maxima": report["overall_maxima"],
                "violations": report["violations"],
                "accuracy_correlations": report["accuracy_correlations"],
                "prompt_length_summary": report["prompt_length_summary"],
                "answer_length_summary": report["answer_length_summary"],
                "width_summary": report["width_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("longest prompt kinds:")
    for row in cast(list[dict[str, object]], report["longest_prompt_kinds"]):
        print(
            f"  {row['name']}: prompt={row['max_prompt_tokens']} line={row['max_line_tokens']} "
            f"answer={row['max_answer_tokens']} eos_context={row['max_context_for_eos']} acc={row['accuracy']:.4f}"
        )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    kinds_csv_path = Path(args.kinds_csv)

    tokenizer, sequence_length, max_new_tokens = load_checkpoint_metadata(
        checkpoint_path
    )
    eval_rows = load_eval_rows(kinds_csv_path)
    kind_stats, split_counts = analyze_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_new_tokens=max_new_tokens,
    )
    report = build_report(
        eval_rows=eval_rows,
        kind_stats=kind_stats,
        split_counts=split_counts,
        sequence_length=sequence_length,
        max_new_tokens=max_new_tokens,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
    )
    output_path = output_path_for(kinds_csv_path, args.output)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"wrote length safety report to {output_path}")


if __name__ == "__main__":
    main()
