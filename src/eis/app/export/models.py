"""Typed export bundle models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from eis.app.export import layout

EXPORT_SCHEMA_VERSION = 1
ExportSection = Literal["raw", "tables", "tensors", "markdown", "png"]
OutputMode = Literal["directory", "zip"]
DEFAULT_SECTIONS: tuple[ExportSection, ...] = (
    "raw",
    "tables",
    "tensors",
    "markdown",
    "png",
)


class ExportOptions(BaseModel):
    schema_version: int = EXPORT_SCHEMA_VERSION
    sections: list[ExportSection] = Field(
        default_factory=lambda: list(DEFAULT_SECTIONS)
    )
    output_mode: OutputMode = "zip"
    prompt_source: str = "direct"
    checkpoint_path: str | None = None
    device: str = "auto"
    include_network: bool = True
    mlp_threshold: float = 0.0
    top_k: int = 5
    top_neurons: int = 8
    selected_token_index: int | None = None
    png_assets: bool = True

    @field_validator("sections", mode="before")
    @classmethod
    def _normalize_sections(cls, value: Any) -> list[ExportSection]:
        if value is None:
            return list(DEFAULT_SECTIONS)
        seen: list[ExportSection] = []
        for entry in value:
            normalized = str(entry).strip().lower()
            if normalized not in DEFAULT_SECTIONS:
                raise ValueError(f"Unsupported export section: {entry}")
            typed = cast(ExportSection, normalized)
            if typed not in seen:
                seen.append(typed)
        return seen or list(DEFAULT_SECTIONS)

    @field_validator("png_assets")
    @classmethod
    def _require_png_assets(cls, value: bool) -> bool:
        if value is False:
            raise ValueError(
                "png_assets=false is unsupported; PNG generation is required."
            )
        return True

    @model_validator(mode="after")
    def _ensure_png_section(self) -> "ExportOptions":
        if "png" not in self.sections:
            self.sections.append("png")
        return self


class ExportFileEntry(BaseModel):
    path: str
    media_type: str
    section: str
    size_bytes: int | None = None


class UnavailableSection(BaseModel):
    section: str
    reason: str
    placeholder_asset: str | None = None


class ExportBundleManifest(BaseModel):
    schema_version: int = EXPORT_SCHEMA_VERSION
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    prompt: str
    checkpoint_path: str
    runtime: str | None = None
    position_encoding: str | None = None
    capabilities: dict[str, bool] | None = None
    options: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    files: list[ExportFileEntry] = Field(default_factory=list)
    unavailable_sections: list[UnavailableSection] = Field(default_factory=list)


def canonical_minimal_paths() -> list[str]:
    return list(layout.REQUIRED_BASE_PATHS)
