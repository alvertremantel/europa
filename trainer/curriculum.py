"""Compatibility shim: re-exports from canonical package."""

from eur_ts.trainer.curriculum import (
    PRESETS,
    CurriculumStage,
    build_balanced_example_sample,
    count_curriculum_groups,
    curriculum_group,
    resample_for_curriculum,
    select_curriculum_stage,
)

__all__ = [
    "PRESETS",
    "CurriculumStage",
    "build_balanced_example_sample",
    "count_curriculum_groups",
    "curriculum_group",
    "resample_for_curriculum",
    "select_curriculum_stage",
]
