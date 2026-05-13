from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, cast

import numpy as np


BandName = Literal["small", "medium", "large"]
FeatureKind = Literal["numeric", "categorical"]

_BAND_LEVEL = {"small": 0, "medium": 1, "large": 2}


@dataclass(frozen=True)
class EvalKindRow:
    name: str
    category: str
    available_count: int
    evaluated_count: int
    perfect_count: int
    missed_count: int
    accuracy: float
    canonical_prediction_rate: float
    wildcard: bool
    strategy: str
    band_pattern: tuple[BandName, ...]
    op: str | None
    inner_op: str | None
    outer_op: str | None
    shape: str | None
    sign_side: str | None
    operations: tuple[str, ...]
    op_family: str
    max_band: BandName
    max_band_level: int
    large_count: int
    medium_count: int
    small_count: int
    contains_add: bool
    contains_subtract: bool
    contains_multiply: bool
    contains_divide: bool


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    getter: Callable[[EvalKindRow], float | str]


@dataclass(frozen=True)
class FitResult:
    sse: float
    fitted: np.ndarray
    residuals: np.ndarray
    rank: int


@dataclass(frozen=True)
class TestResult:
    subset: str
    tested_feature: str
    control_features: tuple[str, ...]
    row_count: int
    levels: tuple[str, ...]
    observed_sse_improvement: float
    partial_r2: float
    permutation_p_value: float
    permutations: int


FEATURES: dict[str, FeatureSpec] = {
    "log_available_count": FeatureSpec(
        name="log_available_count",
        kind="numeric",
        getter=lambda row: math.log10(row.available_count),
    ),
    "category": FeatureSpec(
        name="category",
        kind="categorical",
        getter=lambda row: row.category,
    ),
    "wildcard": FeatureSpec(
        name="wildcard",
        kind="categorical",
        getter=lambda row: str(row.wildcard),
    ),
    "max_band": FeatureSpec(
        name="max_band",
        kind="categorical",
        getter=lambda row: row.max_band,
    ),
    "max_band_level": FeatureSpec(
        name="max_band_level",
        kind="numeric",
        getter=lambda row: float(row.max_band_level),
    ),
    "contains_multiply": FeatureSpec(
        name="contains_multiply",
        kind="categorical",
        getter=lambda row: str(row.contains_multiply),
    ),
    "contains_subtract": FeatureSpec(
        name="contains_subtract",
        kind="categorical",
        getter=lambda row: str(row.contains_subtract),
    ),
    "contains_divide": FeatureSpec(
        name="contains_divide",
        kind="categorical",
        getter=lambda row: str(row.contains_divide),
    ),
    "op_family": FeatureSpec(
        name="op_family",
        kind="categorical",
        getter=lambda row: row.op_family,
    ),
    "shape": FeatureSpec(
        name="shape",
        kind="categorical",
        getter=lambda row: row.shape or "n/a",
    ),
    "sign_side": FeatureSpec(
        name="sign_side",
        kind="categorical",
        getter=lambda row: row.sign_side or "n/a",
    ),
    "large_count": FeatureSpec(
        name="large_count",
        kind="numeric",
        getter=lambda row: float(row.large_count),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-stratum evaluation CSVs for structural performance predictors."
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def stable_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def parse_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"invalid boolean field: {text!r}")


def optional_text(text: str) -> str | None:
    return text or None


def parse_band_pattern(text: str) -> tuple[BandName, ...]:
    parts = tuple(cast(BandName, part) for part in text.split("-"))
    if not parts or any(part not in _BAND_LEVEL for part in parts):
        raise ValueError(f"invalid band pattern: {text!r}")
    return parts


def derived_operations(
    *, op: str | None, inner_op: str | None, outer_op: str | None
) -> tuple[str, ...]:
    if op is not None:
        return (op,)
    operations = tuple(
        operator for operator in (inner_op, outer_op) if operator is not None
    )
    if not operations:
        raise ValueError("row is missing both single-op and inner/outer ops")
    return operations


def operation_family(
    *,
    category: str,
    op: str | None,
    inner_op: str | None,
    outer_op: str | None,
) -> str:
    if category == "parentheses":
        if inner_op is None or outer_op is None:
            raise ValueError("parentheses row is missing inner or outer operator")
        return f"{inner_op}{outer_op}"
    if op is None:
        raise ValueError(f"{category} row is missing op")
    return op


def parse_row(raw: dict[str, str]) -> EvalKindRow:
    status = raw["status"]
    if status != "evaluated":
        raise ValueError(f"expected only evaluated rows, found status={status!r}")

    category = raw["category"]
    op = optional_text(raw["op"])
    inner_op = optional_text(raw["inner_op"])
    outer_op = optional_text(raw["outer_op"])
    shape = optional_text(raw["shape"])
    sign_side = optional_text(raw["sign_side"])
    band_pattern = parse_band_pattern(raw["band_pattern"])
    operations = derived_operations(op=op, inner_op=inner_op, outer_op=outer_op)

    large_count = sum(1 for band in band_pattern if band == "large")
    medium_count = sum(1 for band in band_pattern if band == "medium")
    small_count = sum(1 for band in band_pattern if band == "small")
    max_band = cast(BandName, max(band_pattern, key=_BAND_LEVEL.__getitem__))

    return EvalKindRow(
        name=raw["name"],
        category=category,
        available_count=int(raw["available_count"]),
        evaluated_count=int(raw["evaluated_count"]),
        perfect_count=int(raw["perfect_count"]),
        missed_count=int(raw["missed_count"]),
        accuracy=float(raw["accuracy"]),
        canonical_prediction_rate=float(raw["canonical_prediction_rate"]),
        wildcard=parse_bool(raw["wildcard"]),
        strategy=raw["strategy"],
        band_pattern=band_pattern,
        op=op,
        inner_op=inner_op,
        outer_op=outer_op,
        shape=shape,
        sign_side=sign_side,
        operations=operations,
        op_family=operation_family(
            category=category,
            op=op,
            inner_op=inner_op,
            outer_op=outer_op,
        ),
        max_band=max_band,
        max_band_level=_BAND_LEVEL[max_band],
        large_count=large_count,
        medium_count=medium_count,
        small_count=small_count,
        contains_add="+" in operations,
        contains_subtract="-" in operations,
        contains_multiply="*" in operations,
        contains_divide="/" in operations,
    )


def load_rows(csv_path: Path) -> list[EvalKindRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_row(cast(dict[str, str], raw_row)) for raw_row in reader]


def empirical_logit(row: EvalKindRow) -> float:
    return math.log((row.perfect_count + 0.5) / (row.missed_count + 0.5))


def summarize_rows(rows: list[EvalKindRow]) -> dict[str, object]:
    total_evaluated = sum(row.evaluated_count for row in rows)
    total_perfect = sum(row.perfect_count for row in rows)
    return {
        "kind_count": len(rows),
        "evaluated_count": total_evaluated,
        "perfect_count": total_perfect,
        "missed_count": total_evaluated - total_perfect,
        "accuracy": total_perfect / total_evaluated,
    }


def group_summary(
    rows: list[EvalKindRow],
    *,
    name: str,
    getter: Callable[[EvalKindRow], str],
) -> dict[str, object]:
    groups: dict[str, list[EvalKindRow]] = {}
    for row in rows:
        groups.setdefault(getter(row), []).append(row)

    summaries: list[dict[str, object]] = []
    for label, group_rows in groups.items():
        summary = summarize_rows(group_rows)
        summary["label"] = label
        summaries.append(summary)

    summaries.sort(key=lambda entry: (-cast(float, entry["accuracy"]), entry["label"]))
    return {"name": name, "groups": summaries}


def build_design_matrix(
    rows: list[EvalKindRow], feature_names: list[str]
) -> tuple[np.ndarray, dict[str, list[str]]]:
    columns: list[np.ndarray] = [np.ones(len(rows), dtype=np.float64)]
    levels_used: dict[str, list[str]] = {}

    for feature_name in feature_names:
        spec = FEATURES[feature_name]
        if spec.kind == "numeric":
            values = np.array(
                [float(cast(float, spec.getter(row))) for row in rows], dtype=np.float64
            )
            columns.append(values)
            levels_used[feature_name] = []
            continue

        raw_levels = [str(cast(str, spec.getter(row))) for row in rows]
        levels = sorted(set(raw_levels))
        levels_used[feature_name] = levels
        if len(levels) <= 1:
            continue
        for level in levels[1:]:
            columns.append(
                np.array([1.0 if value == level else 0.0 for value in raw_levels])
            )

    return np.column_stack(columns), levels_used


def fit_linear_model(X: np.ndarray, y: np.ndarray) -> FitResult:
    coefficients, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefficients
    model_residuals = y - fitted
    if residuals.size:
        sse = float(residuals[0])
    else:
        sse = float(model_residuals.T @ model_residuals)
    return FitResult(sse=sse, fitted=fitted, residuals=model_residuals, rank=int(rank))


def permutation_partial_test(
    *,
    rows: list[EvalKindRow],
    subset: str,
    tested_feature: str,
    control_features: list[str],
    permutations: int,
    seed: int,
) -> TestResult | None:
    if len(rows) < 3:
        return None

    reduced_feature_names = list(control_features)
    full_feature_names = [*control_features, tested_feature]
    X_reduced, reduced_levels = build_design_matrix(rows, reduced_feature_names)
    X_full, full_levels = build_design_matrix(rows, full_feature_names)
    if X_full.shape[1] == X_reduced.shape[1]:
        return None

    y = np.array([empirical_logit(row) for row in rows], dtype=np.float64)
    reduced_fit = fit_linear_model(X_reduced, y)
    full_fit = fit_linear_model(X_full, y)
    observed = reduced_fit.sse - full_fit.sse
    if observed < 0:
        observed = 0.0

    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(reduced_fit.residuals)
        y_star = reduced_fit.fitted + permuted
        reduced_star = fit_linear_model(X_reduced, y_star)
        full_star = fit_linear_model(X_full, y_star)
        improvement = reduced_star.sse - full_star.sse
        if improvement >= observed - 1e-12:
            exceedances += 1

    partial_r2 = 0.0 if reduced_fit.sse <= 0 else observed / reduced_fit.sse
    return TestResult(
        subset=subset,
        tested_feature=tested_feature,
        control_features=tuple(control_features),
        row_count=len(rows),
        levels=tuple(full_levels.get(tested_feature, [])),
        observed_sse_improvement=observed,
        partial_r2=partial_r2,
        permutation_p_value=(exceedances + 1) / (permutations + 1),
        permutations=permutations,
    )


def subset_rows(
    rows: list[EvalKindRow], category: str | None = None
) -> list[EvalKindRow]:
    if category is None:
        return rows
    return [row for row in rows if row.category == category]


def output_path_for(csv_path: Path, requested: str | None) -> Path:
    if requested is not None:
        return Path(requested)
    return csv_path.with_suffix(".analysis.json")


def run_tests(
    rows: list[EvalKindRow], permutations: int, seed: int
) -> list[TestResult]:
    tests: list[tuple[str, list[EvalKindRow], str, list[str]]] = [
        ("all", rows, "log_available_count", []),
        ("all", rows, "category", ["log_available_count"]),
        ("all", rows, "wildcard", ["log_available_count", "category"]),
        (
            "sampled_only",
            [row for row in rows if row.strategy == "sampled"],
            "category",
            [],
        ),
        (
            "sampled_only",
            [row for row in rows if row.strategy == "sampled"],
            "max_band",
            ["category"],
        ),
        (
            "sampled_only",
            [row for row in rows if row.strategy == "sampled"],
            "contains_multiply",
            ["category", "max_band"],
        ),
        (
            "sampled_only",
            [row for row in rows if row.strategy == "sampled"],
            "contains_subtract",
            ["category", "max_band", "contains_multiply"],
        ),
        ("binary", subset_rows(rows, "binary"), "op_family", ["log_available_count"]),
        ("binary", subset_rows(rows, "binary"), "max_band", ["op_family"]),
        ("three_input", subset_rows(rows, "three_input"), "op_family", ["max_band"]),
        (
            "three_input",
            subset_rows(rows, "three_input"),
            "max_band",
            ["op_family"],
        ),
        (
            "negative_input",
            subset_rows(rows, "negative_input"),
            "op_family",
            ["max_band", "sign_side"],
        ),
        (
            "negative_input",
            subset_rows(rows, "negative_input"),
            "sign_side",
            ["op_family", "max_band"],
        ),
        (
            "negative_input",
            subset_rows(rows, "negative_input"),
            "max_band",
            ["op_family", "sign_side"],
        ),
        (
            "parentheses",
            subset_rows(rows, "parentheses"),
            "shape",
            ["op_family", "max_band"],
        ),
        (
            "parentheses",
            subset_rows(rows, "parentheses"),
            "op_family",
            ["shape", "max_band"],
        ),
        (
            "parentheses",
            subset_rows(rows, "parentheses"),
            "max_band",
            ["shape", "op_family"],
        ),
        (
            "parentheses",
            subset_rows(rows, "parentheses"),
            "wildcard",
            ["shape", "op_family", "max_band"],
        ),
    ]

    results: list[TestResult] = []
    for subset_name, subset, tested_feature, controls in tests:
        if not subset:
            continue
        test_seed = seed + stable_int(f"{subset_name}|{tested_feature}") % 1_000_000
        result = permutation_partial_test(
            rows=subset,
            subset=subset_name,
            tested_feature=tested_feature,
            control_features=controls,
            permutations=permutations,
            seed=test_seed,
        )
        if result is not None:
            results.append(result)
    return results


def top_bottom(
    rows: list[EvalKindRow], count: int
) -> dict[str, list[dict[str, object]]]:
    sorted_rows = sorted(rows, key=lambda row: (row.accuracy, row.name))
    worst = [
        {
            "name": row.name,
            "category": row.category,
            "op_family": row.op_family,
            "shape": row.shape,
            "band_pattern": list(row.band_pattern),
            "accuracy": row.accuracy,
            "perfect_count": row.perfect_count,
            "missed_count": row.missed_count,
        }
        for row in sorted_rows[:count]
    ]
    best = [
        {
            "name": row.name,
            "category": row.category,
            "op_family": row.op_family,
            "shape": row.shape,
            "band_pattern": list(row.band_pattern),
            "accuracy": row.accuracy,
            "perfect_count": row.perfect_count,
            "missed_count": row.missed_count,
        }
        for row in reversed(sorted_rows[-count:])
    ]
    return {"best": best, "worst": worst}


def build_report(
    rows: list[EvalKindRow], permutations: int, seed: int
) -> dict[str, object]:
    category_summaries = group_summary(
        rows, name="category", getter=lambda row: row.category
    )
    operation_summaries = group_summary(
        rows, name="operation_family", getter=lambda row: row.op_family
    )
    sampled_rows = [row for row in rows if row.strategy == "sampled"]
    sampled_operation_summaries = group_summary(
        sampled_rows,
        name="sampled_operation_family",
        getter=lambda row: f"{row.category}:{row.op_family}",
    )
    max_band_summaries = group_summary(
        rows, name="max_band", getter=lambda row: row.max_band
    )
    multiply_summaries = group_summary(
        rows,
        name="contains_multiply",
        getter=lambda row: str(row.contains_multiply),
    )
    tests = run_tests(rows, permutations=permutations, seed=seed)

    return {
        "overall": summarize_rows(rows),
        "category_summary": category_summaries,
        "operation_summary": operation_summaries,
        "sampled_operation_summary": sampled_operation_summaries,
        "max_band_summary": max_band_summaries,
        "contains_multiply_summary": multiply_summaries,
        "top_bottom": top_bottom(rows, count=12),
        "tests": [asdict(result) for result in tests],
    }


def print_report(report: dict[str, object]) -> None:
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print("category summary:")
    category_summary = cast(dict[str, object], report["category_summary"])
    for group in cast(list[dict[str, object]], category_summary["groups"]):
        print(
            f"  {group['label']}: accuracy={group['accuracy']:.4f} "
            f"perfect={group['perfect_count']} missed={group['missed_count']} kinds={group['kind_count']}"
        )

    print("statistical tests:")
    for test in cast(list[dict[str, object]], report["tests"]):
        print(
            f"  {test['subset']} | {test['tested_feature']} | "
            f"partial_r2={test['partial_r2']:.4f} p={test['permutation_p_value']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.permutations <= 0:
        raise SystemExit("--permutations must be positive")

    csv_path = Path(args.csv_path)
    rows = load_rows(csv_path)
    report = build_report(rows, permutations=args.permutations, seed=args.seed)
    output_path = output_path_for(csv_path, args.output)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
