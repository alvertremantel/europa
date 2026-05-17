from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tomllib
from typing import cast

from eur_ts.trainer.curriculum import PRESETS

from .schema import TrainConfig

ALLOWED_TRAINING_MODES = {"token_stream", "examples"}
ALLOWED_TRAINING_FORMATS = {
    "final_only",
    "light_scratchpad",
    "parentheses_intermediate",
    "multiply_intermediate",
}
ALLOWED_BALANCED_GROUP_BY = {"kind", "category", "curriculum_group"}

SECTION_SPECS: dict[str, dict[str, tuple[str, type, bool]]] = {
    "paths": {
        "data_dir": ("data_dir", str, True),
        "output_dir": ("output_dir", str, True),
    },
    "runtime": {
        "device": ("device", str, True),
        "seed": ("seed", int, True),
    },
    "resume": {
        "resume_from": ("resume_from", str, False),
        "auto_resume": ("auto_resume", bool, True),
        "additional_epochs": ("additional_epochs", int, False),
    },
    "model": {
        "sequence_length": ("sequence_length", int, True),
        "d_model": ("d_model", int, True),
        "n_heads": ("n_heads", int, True),
        "n_layers": ("n_layers", int, True),
        "mlp_hidden": ("mlp_hidden", int, True),
        "dropout": ("dropout", float, True),
    },
    "optimization": {
        "batch_size": ("batch_size", int, True),
        "epochs": ("epochs", int, True),
        "learning_rate": ("learning_rate", float, True),
        "weight_decay": ("weight_decay", float, True),
        "grad_clip": ("grad_clip", float, True),
    },
    "logging": {
        "log_interval": ("log_interval", int, True),
        "eval_batches": ("eval_batches", int, True),
        "exact_match_samples": ("exact_match_samples", int, True),
        "max_new_tokens": ("max_new_tokens", int, True),
    },
    "checkpoint": {
        "checkpoint_keep_last": ("checkpoint_keep_last", int, True),
        "checkpoint_max_kept": ("checkpoint_max_kept", int, True),
        "checkpoint_keep_best": ("checkpoint_keep_best", int, True),
        "checkpoint_jump_threshold": ("checkpoint_jump_threshold", float, True),
        "checkpoint_dir_name": ("checkpoint_dir_name", str, True),
    },
    "training": {
        "training_mode": ("training_mode", str, True),
        "training_format": ("training_format", str, True),
        "skip_overlong_examples": ("skip_overlong_examples", bool, True),
        "curriculum_name": ("curriculum_name", str, False),
    },
    "balanced_validation": {
        "enabled": ("balanced_val_enabled", bool, True),
        "group_by": ("balanced_val_group_by", str, True),
        "sample_size_per_group": ("balanced_val_sample_size_per_group", int, True),
        "seed": ("balanced_val_seed", int, True),
        "batch_size": ("balanced_val_batch_size", int, False),
    },
}


def load_train_config(path: Path) -> TrainConfig:
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} did not contain a TOML table")
    values = _collect_values(loaded)
    _validate_semantics(values)
    return TrainConfig(
        data_dir=_required_str(values, "data_dir"),
        output_dir=_required_str(values, "output_dir"),
        resume_from=_optional_str(values, "resume_from"),
        auto_resume=_required_bool(values, "auto_resume"),
        additional_epochs=_optional_int(values, "additional_epochs"),
        sequence_length=_required_int(values, "sequence_length"),
        batch_size=_required_int(values, "batch_size"),
        epochs=_required_int(values, "epochs"),
        learning_rate=_required_float(values, "learning_rate"),
        weight_decay=_required_float(values, "weight_decay"),
        grad_clip=_required_float(values, "grad_clip"),
        log_interval=_required_int(values, "log_interval"),
        eval_batches=_required_int(values, "eval_batches"),
        exact_match_samples=_required_int(values, "exact_match_samples"),
        max_new_tokens=_required_int(values, "max_new_tokens"),
        seed=_required_int(values, "seed"),
        device=_required_str(values, "device"),
        d_model=_required_int(values, "d_model"),
        n_heads=_required_int(values, "n_heads"),
        n_layers=_required_int(values, "n_layers"),
        mlp_hidden=_required_int(values, "mlp_hidden"),
        dropout=_required_float(values, "dropout"),
        checkpoint_keep_last=_required_int(values, "checkpoint_keep_last"),
        checkpoint_max_kept=_required_int(values, "checkpoint_max_kept"),
        checkpoint_keep_best=_required_int(values, "checkpoint_keep_best"),
        checkpoint_jump_threshold=_required_float(values, "checkpoint_jump_threshold"),
        checkpoint_dir_name=_required_str(values, "checkpoint_dir_name"),
        training_mode=_required_str(values, "training_mode"),
        training_format=_required_str(values, "training_format"),
        skip_overlong_examples=_required_bool(values, "skip_overlong_examples"),
        curriculum_name=_optional_str(values, "curriculum_name"),
        balanced_val_enabled=_required_bool(values, "balanced_val_enabled"),
        balanced_val_group_by=_required_str(values, "balanced_val_group_by"),
        balanced_val_sample_size_per_group=_required_int(
            values, "balanced_val_sample_size_per_group"
        ),
        balanced_val_seed=_required_int(values, "balanced_val_seed"),
        balanced_val_batch_size=_optional_int(values, "balanced_val_batch_size"),
    )


def _collect_values(loaded: Mapping[str, object]) -> dict[str, object]:
    errors: list[str] = []
    values: dict[str, object] = {}

    for section_name in loaded:
        if section_name not in SECTION_SPECS:
            errors.append(f"unknown section [{section_name}]")

    for section_name, key_specs in SECTION_SPECS.items():
        raw_section = loaded.get(section_name)
        if raw_section is None:
            errors.append(f"missing required section [{section_name}]")
            continue
        if not isinstance(raw_section, dict):
            errors.append(f"section [{section_name}] must be a table")
            continue
        section_map = cast(dict[str, object], raw_section)
        for key in section_map:
            if key not in key_specs:
                errors.append(f"unknown key [{section_name}].{key}")
        for key, (field_name, target_type, required) in key_specs.items():
            if key not in section_map:
                if required:
                    errors.append(f"missing required key [{section_name}].{key}")
                else:
                    values[field_name] = None
                continue
            raw_value = section_map[key]
            coerced = _coerce_value(
                section_name, key, raw_value, target_type, required, errors
            )
            values[field_name] = coerced

    if errors:
        raise ValueError("Invalid training config:\n- " + "\n- ".join(errors))
    return values


def _coerce_value(
    section_name: str,
    key: str,
    raw_value: object,
    target_type: type,
    required: bool,
    errors: list[str],
) -> object:
    field_ref = f"[{section_name}].{key}"
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if raw_value == "":
            if required:
                errors.append(f"{field_ref} must not be empty")
            return None
        if target_type is str:
            return raw_value
        if target_type is bool:
            lowered = raw_value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            errors.append(f"{field_ref} must be a boolean")
            return None
        try:
            if target_type is int:
                return int(raw_value)
            if target_type is float:
                return float(raw_value)
        except ValueError:
            errors.append(f"{field_ref} must be a {target_type.__name__}")
            return None
    if raw_value is None:
        if required:
            errors.append(f"{field_ref} must not be empty")
        return None
    if target_type is float and isinstance(raw_value, int):
        return float(raw_value)
    if target_type is bool and isinstance(raw_value, bool):
        return raw_value
    if (
        target_type is int
        and isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
    ):
        return raw_value
    if target_type is float and isinstance(raw_value, float):
        return raw_value
    if target_type is str and isinstance(raw_value, str):
        return raw_value
    errors.append(f"{field_ref} must be a {target_type.__name__}")
    return None


def _validate_semantics(values: Mapping[str, object]) -> None:
    errors: list[str] = []

    def require_positive(name: str) -> None:
        value = values[name]
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{name} must be a positive integer")

    for name in (
        "sequence_length",
        "batch_size",
        "epochs",
        "log_interval",
        "eval_batches",
        "exact_match_samples",
        "max_new_tokens",
        "balanced_val_sample_size_per_group",
        "seed",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_hidden",
    ):
        require_positive(name)

    for name in ("checkpoint_keep_last", "checkpoint_keep_best"):
        value = values[name]
        if not isinstance(value, int) or value < 0:
            errors.append(f"{name} must be a non-negative integer")

    checkpoint_max_kept = values["checkpoint_max_kept"]
    if not isinstance(checkpoint_max_kept, int):
        errors.append("checkpoint_max_kept must be an integer")
    additional_epochs = values["additional_epochs"]
    if additional_epochs is not None and (
        not isinstance(additional_epochs, int) or additional_epochs <= 0
    ):
        errors.append("additional_epochs must be a positive integer when provided")
    balanced_val_batch_size = values["balanced_val_batch_size"]
    if balanced_val_batch_size is not None and (
        not isinstance(balanced_val_batch_size, int) or balanced_val_batch_size <= 0
    ):
        errors.append(
            "balanced_val_batch_size must be a positive integer when provided"
        )

    for name in (
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "checkpoint_jump_threshold",
    ):
        value = values[name]
        if not isinstance(value, (int, float)):
            errors.append(f"{name} must be numeric")
    dropout = values["dropout"]
    if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) <= 1.0:
        errors.append("dropout must be between 0.0 and 1.0")

    d_model = values["d_model"]
    n_heads = values["n_heads"]
    if isinstance(d_model, int) and isinstance(n_heads, int) and d_model % n_heads != 0:
        errors.append("d_model must be divisible by n_heads")

    training_mode = values["training_mode"]
    if training_mode not in ALLOWED_TRAINING_MODES:
        errors.append(f"training_mode must be one of {sorted(ALLOWED_TRAINING_MODES)}")
    training_format = values["training_format"]
    if training_format not in ALLOWED_TRAINING_FORMATS:
        errors.append(
            f"training_format must be one of {sorted(ALLOWED_TRAINING_FORMATS)}"
        )
    balanced_group = values["balanced_val_group_by"]
    if balanced_group not in ALLOWED_BALANCED_GROUP_BY:
        errors.append(
            f"balanced_val_group_by must be one of {sorted(ALLOWED_BALANCED_GROUP_BY)}"
        )
    curriculum_name = values["curriculum_name"]
    if curriculum_name is not None and curriculum_name not in PRESETS:
        errors.append(f"curriculum_name must be one of {sorted(PRESETS)} or empty")

    auto_resume = values["auto_resume"]
    resume_from = values["resume_from"]
    if auto_resume is True and resume_from is not None:
        errors.append("resume.auto_resume and resume.resume_from cannot both be set")

    if errors:
        raise ValueError("Invalid training config:\n- " + "\n- ".join(errors))


def _required_str(values: Mapping[str, object], key: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise TypeError(f"expected string for {key}")
    return value


def _optional_str(values: Mapping[str, object], key: str) -> str | None:
    value = values[key]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"expected optional string for {key}")


def _required_bool(values: Mapping[str, object], key: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise TypeError(f"expected bool for {key}")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected int for {key}")
    return value


def _optional_int(values: Mapping[str, object], key: str) -> int | None:
    value = values[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected optional int for {key}")
    return value


def _required_float(values: Mapping[str, object], key: str) -> float:
    value = values[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected float for {key}")
    return float(value)
