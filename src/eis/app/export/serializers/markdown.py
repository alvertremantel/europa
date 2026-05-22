"""Markdown summary serializer."""

from __future__ import annotations

from eis.app.backend.schemas import AnalyzeResponse, HealthResponse
from eis.app.export import layout
from eis.app.export.models import ExportBundleManifest


def build_summary_markdown(
    *,
    prompt: str,
    analysis: AnalyzeResponse,
    health: HealthResponse,
    manifest: ExportBundleManifest,
) -> dict[str, bytes]:
    lines = [
        "# ITS export summary",
        "",
        "## Prompt",
        "",
        f"- Prompt: `{prompt}`",
        f"- Generated answer: `{analysis.generated_answer.text}`",
        f"- Correct: `{analysis.generated_answer.is_correct}`",
        f"- Checkpoint: `{analysis.checkpoint.path}`",
        f"- Runtime: `{analysis.analysis_runtime}`",
        f"- Position encoding: `{analysis.position_encoding}`",
        "",
        "## Runtime status",
        "",
        f"- Health: `{health.status}`",
        f"- Device: `{health.device}`",
        f"- Network capability: `{analysis.capabilities.network_analysis if analysis.capabilities else None}`",
        "",
        "## Model config",
        "",
        f"- Layers: `{analysis.config.n_layers}`",
        f"- Heads: `{analysis.config.n_heads}`",
        f"- d_model: `{analysis.config.d_model}`",
        f"- Sequence length: `{analysis.config.sequence_length}`",
        "",
        "## Problem metadata",
        "",
        f"- Category: `{analysis.problem.category if analysis.problem else 'unknown'}`",
        f"- Kind: `{analysis.problem.kind if analysis.problem else 'unknown'}`",
        f"- Curriculum group: `{analysis.problem.curriculum_group if analysis.problem else 'unknown'}`",
        "",
        "## Included sections",
        "",
    ]
    lines.extend(f"- `{section}`" for section in manifest.options.get("sections", []))
    lines.extend(["", "## Graph assets", ""])
    lines.extend(
        f"- `{entry.path}`"
        for entry in manifest.files
        if entry.path.startswith("assets/")
    )
    if manifest.unavailable_sections:
        lines.extend(["", "## Unavailable sections", ""])
        lines.extend(
            f"- `{entry.section}`: {entry.reason}"
            for entry in manifest.unavailable_sections
        )
    return {layout.SUMMARY_PATH: ("\n".join(lines) + "\n").encode("utf-8")}
