from __future__ import annotations

import json
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import tomli_w


TomlMapping = Mapping[str, object]


def to_toml_data(value: object) -> object:
    """Convert project artifact data into TOML-compatible Python values.

    TOML has no null value, so mapping entries whose value is ``None`` are
    omitted. ``None`` inside arrays is unsupported and raises an error because
    dropping array elements would silently change list semantics.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return to_toml_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        data: dict[str, object] = {}
        for key, item in value.items():
            if item is None:
                continue
            data[str(key)] = to_toml_data(item)
        return data
    if isinstance(value, tuple):
        return [to_toml_array_item(item) for item in value]
    if isinstance(value, list):
        return [to_toml_array_item(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported TOML artifact value type: {type(value)!r}")


def to_toml_array_item(value: object) -> object:
    if value is None:
        raise TypeError("TOML arrays cannot contain None values")
    return to_toml_data(value)


def toml_text(payload: TomlMapping) -> str:
    data = to_toml_data(payload)
    if not isinstance(data, dict):
        raise TypeError("TOML document payload must be a mapping")
    return tomli_w.dumps(cast(dict[str, Any], data))


def write_toml(path: Path, payload: TomlMapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(toml_text(payload))
    try:
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"TOML artifact {path} did not contain a table")
    return cast(dict[str, object], loaded)


def read_legacy_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
