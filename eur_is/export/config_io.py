"""Export config loading with isolated file-format dispatch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eur_is.export.models import ExportOptions


def export_options_from_mapping(mapping: Mapping[str, Any]) -> ExportOptions:
    return ExportOptions.model_validate(dict(mapping))


def load_export_options(path: Path) -> ExportOptions:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Export config files must contain a JSON object mapping.")
        return export_options_from_mapping(data)
    raise ValueError(
        f"Unsupported export config format for {path}. Add new suffix handling here."
    )
