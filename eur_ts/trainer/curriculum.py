from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .data import ArithmeticExample


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    end_progress: float
    weights: dict[str, float]


PRESETS: dict[str, tuple[CurriculumStage, ...]] = {
    "baseline_mixed_v1": (
        CurriculumStage(
            name="foundations",
            end_progress=1 / 3,
            weights={"easy_binary_add_sub": 0.75, "binary_mul_div": 0.25},
        ),
        CurriculumStage(
            name="mul_div_focus",
            end_progress=2 / 3,
            weights={"easy_binary_add_sub": 0.35, "binary_mul_div": 0.65},
        ),
        CurriculumStage(
            name="compositional_mix",
            end_progress=1.0,
            weights={
                "easy_binary_add_sub": 0.20,
                "binary_mul_div": 0.35,
                "compositional_parentheses_three_input": 0.30,
                "negative_input": 0.15,
            },
        ),
    ),
    "mul_focus_v1": (
        CurriculumStage(
            name="mul_warmup",
            end_progress=0.5,
            weights={"easy_binary_add_sub": 0.40, "binary_mul_div": 0.60},
        ),
        CurriculumStage(
            name="mul_composition",
            end_progress=1.0,
            weights={
                "easy_binary_add_sub": 0.15,
                "binary_mul_div": 0.55,
                "compositional_parentheses_three_input": 0.20,
                "negative_input": 0.10,
            },
        ),
    ),
}


def curriculum_group(example: ArithmeticExample) -> str:
    category = example.category or "unknown"
    kind = example.kind or ""
    op = _kind_operator(kind)
    if category == "negative_input":
        return "negative_input"
    if category in {"parentheses", "three_input"}:
        return "compositional_parentheses_three_input"
    if category == "binary" and op in {"*", "/"}:
        return "binary_mul_div"
    if category == "binary" and op in {"+", "-"}:
        return "easy_binary_add_sub"
    return "other"


def select_curriculum_stage(
    name: str,
    epoch: int,
    *,
    target_epoch: int,
) -> tuple[int, CurriculumStage]:
    stages = PRESETS.get(name)
    if stages is None:
        raise ValueError(
            f"unknown curriculum preset {name!r}; choose from {sorted(PRESETS)}"
        )
    for index, stage in enumerate(stages):
        if _epoch_progress(epoch, target_epoch) <= stage.end_progress:
            return index + 1, stage
    return len(stages), stages[-1]


def resample_for_curriculum(
    examples: Sequence[ArithmeticExample],
    *,
    curriculum_name: str,
    epoch: int,
    target_epoch: int,
    seed: int,
    sample_count: int | None = None,
) -> tuple[list[ArithmeticExample], dict[str, int], dict[str, float], str, int]:
    stage_index, stage = select_curriculum_stage(
        curriculum_name,
        epoch,
        target_epoch=target_epoch,
    )
    by_group: dict[str, list[ArithmeticExample]] = defaultdict(list)
    for example in examples:
        by_group[curriculum_group(example)].append(example)

    available_weights = {
        group: weight
        for group, weight in stage.weights.items()
        if weight > 0 and by_group.get(group)
    }
    if not available_weights:
        raise ValueError(
            f"no examples are available for curriculum stage {stage.name!r}"
        )
    total_weight = sum(available_weights.values())
    normalized = {
        group: weight / total_weight for group, weight in available_weights.items()
    }

    rng = random.Random(seed + epoch * 1_000_003)
    groups = list(normalized)
    weights = [normalized[group] for group in groups]
    count = sample_count or len(examples)
    sampled: list[ArithmeticExample] = []
    counts: Counter[str] = Counter()
    for _ in range(count):
        group = rng.choices(groups, weights=weights, k=1)[0]
        example = rng.choice(by_group[group])
        sampled.append(example)
        counts[group] += 1
    rng.shuffle(sampled)
    return sampled, dict(counts), normalized, stage.name, stage_index


def build_balanced_example_sample(
    examples: Sequence[ArithmeticExample],
    *,
    group_by: str = "kind",
    sample_size_per_group: int = 8,
    seed: int = 42,
) -> list[ArithmeticExample]:
    if sample_size_per_group <= 0:
        raise ValueError("sample_size_per_group must be positive")
    grouped: dict[str, list[ArithmeticExample]] = defaultdict(list)
    for example in examples:
        key = _group_key(example, group_by)
        grouped[key].append(example)
    rng = random.Random(seed)
    balanced: list[ArithmeticExample] = []
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        if len(bucket) >= sample_size_per_group:
            balanced.extend(bucket[:sample_size_per_group])
        else:
            balanced.extend(rng.choice(bucket) for _ in range(sample_size_per_group))
    return balanced


def count_curriculum_groups(examples: Sequence[ArithmeticExample]) -> dict[str, int]:
    return dict(Counter(curriculum_group(example) for example in examples))


def _group_key(example: ArithmeticExample, group_by: str) -> str:
    if group_by == "kind":
        return example.kind or "unknown"
    if group_by == "category":
        return example.category or "unknown"
    if group_by == "curriculum_group":
        return curriculum_group(example)
    raise ValueError(
        "balanced validation group_by must be kind, category, or curriculum_group"
    )


def _kind_operator(kind: str) -> str | None:
    parts = kind.split("::")
    if not parts:
        return None
    if parts[0] in {"binary", "three_input"} and len(parts) >= 3:
        return parts[2]
    if parts[0] == "negative_input" and len(parts) >= 3:
        return parts[2]
    return None


def _epoch_progress(epoch: int, target_epoch: int) -> float:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if target_epoch <= 0:
        raise ValueError("target_epoch must be positive")
    return (epoch - 1) / max(target_epoch - 1, 1)
