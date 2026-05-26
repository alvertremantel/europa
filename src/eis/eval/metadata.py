"""Metadata loading and resolution for the evaluator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import cast

import torch

from eis.artifacts import read_legacy_json, read_toml
from eis.data.core import KindSpec, iter_kind_specs

CATEGORY_ORDER = ("arithmetic", "negative_input", "comparison")


def spec_to_definition(spec: KindSpec) -> dict[str, object]:
    return {
        "category": spec.category,
        "band_pattern": list(spec.band_pattern),
        "wildcard": spec.wildcard,
        "strategy": spec.strategy,
        "op": spec.op,
        "inner_op": spec.inner_op,
        "outer_op": spec.outer_op,
        "shape": spec.shape,
        "sign_side": spec.sign_side,
    }


def fallback_kind_definitions() -> dict[str, dict[str, object]]:
    return {spec.name: spec_to_definition(spec) for spec in iter_kind_specs()}


def checkpoint_payload(
    checkpoint_path: Path, device: torch.device
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)!r}")
    return cast(dict[str, object], payload)


def train_config_from_payload(payload: dict[str, object]) -> dict[str, object]:
    config = payload.get("train_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing train_config")
    return cast(dict[str, object], config)


def resolve_data_dir(args: argparse.Namespace, payload: dict[str, object]) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    train_config = train_config_from_payload(payload)
    data_dir = train_config.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("--data-dir was not provided and checkpoint has no data_dir")
    return Path(data_dir)


def resolve_max_new_tokens(args: argparse.Namespace, payload: dict[str, object]) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    train_config = train_config_from_payload(payload)
    value = train_config.get("max_new_tokens", 24)
    if not isinstance(value, int):
        raise ValueError("checkpoint max_new_tokens is not an integer")
    return value


def resolve_output_prefix(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output_prefix is not None:
        return Path(args.output_prefix)
    return checkpoint_path.parent / f"{checkpoint_path.stem}-strata-eval"


def load_metadata(data_dir: Path) -> dict[str, object] | None:
    metadata_path = data_dir / "meta.toml"
    if metadata_path.exists():
        return read_toml(metadata_path)
    legacy_metadata_path = data_dir / "meta.json"
    if legacy_metadata_path.exists():
        return cast(dict[str, object], read_legacy_json(legacy_metadata_path))
    return None


def kind_definitions_from_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if metadata is None:
        return fallback_kind_definitions()
    definitions = metadata.get("kind_definitions")
    if not isinstance(definitions, dict):
        raise ValueError("metadata is missing kind_definitions")
    return cast(dict[str, dict[str, object]], definitions)


def skipped_kinds_from_metadata(metadata: dict[str, object] | None) -> dict[str, str]:
    if metadata is None:
        return {}
    skipped = metadata.get("skipped_kinds", {})
    if not isinstance(skipped, dict):
        raise ValueError("metadata skipped_kinds has unexpected type")
    return {str(kind): str(reason) for kind, reason in skipped.items()}


def ordered_categories(metadata: dict[str, object] | None) -> list[str]:
    if metadata is None:
        return list(CATEGORY_ORDER)
    categories = metadata.get("categories")
    if not isinstance(categories, list) or not all(
        isinstance(category, str) for category in categories
    ):
        return list(CATEGORY_ORDER)
    return cast(list[str], categories)


def expected_available_kind_counts(
    metadata: dict[str, object] | None, splits: list[str]
) -> dict[str, int] | None:
    if metadata is None:
        return None
    split_counts = metadata.get("split_kind_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("metadata is missing split_kind_counts")

    totals: dict[str, int] = defaultdict(int)
    split_count_map = cast(dict[str, object], split_counts)
    for split in splits:
        raw_counts = split_count_map.get(split)
        if not isinstance(raw_counts, dict):
            raise ValueError(f"metadata is missing split_kind_counts[{split!r}]")
        for kind, count in cast(dict[str, object], raw_counts).items():
            if not isinstance(count, int):
                raise ValueError(
                    f"metadata split count for {kind!r} in {split!r} is not an integer"
                )
            totals[str(kind)] += count
    return dict(totals)
