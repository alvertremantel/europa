"""Bundle writers for ITS exports."""

from __future__ import annotations

import json
import mimetypes
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from eis.app.backend.schemas import AnalyzeResponse, HealthResponse
from eis.app.export import layout
from eis.app.export.models import (
    ExportBundleManifest,
    ExportFileEntry,
    ExportOptions,
    UnavailableSection,
)
from eis.app.export.png import render_png_files
from eis.app.export.serializers.markdown import build_summary_markdown
from eis.app.export.serializers.raw import build_raw_files
from eis.app.export.serializers.tables import build_table_files


def build_bundle_file_map(
    *,
    prompt: str,
    analysis: AnalyzeResponse,
    health: HealthResponse,
    options: ExportOptions,
) -> tuple[dict[str, bytes], ExportBundleManifest]:
    files: dict[str, bytes] = {}

    # Always include raw files (required bundle contract).
    files.update(build_raw_files(analysis=analysis, health=health))

    # Generate all table/tensor content once, then include based on sections.
    all_table_files = build_table_files(analysis)

    # Always include required-bundle paths.
    _REQUIRED_TABLE_PATHS = {
        layout.TOKENS_TABLE_PATH,
        layout.PROMPT_PREDICTIONS_TABLE_PATH,
        layout.GENERATED_ANSWER_TOPK_TABLE_PATH,
        layout.ACTIVATION_SUMMARY_TABLE_PATH,
        layout.LOGITS_TENSOR_PATH,
        layout.ACTIVATIONS_TENSOR_PATH,
    }
    for path, content in all_table_files.items():
        if path in _REQUIRED_TABLE_PATHS:
            files[path] = content

    # Include optional tables gated by "tables" section.
    if "tables" in options.sections:
        for path in (
            layout.ATTENTION_HEAD_SUMMARY_TABLE_PATH,
            layout.NETWORK_MLP_TABLE_PATH,
            layout.NETWORK_ATTENTION_TABLE_PATH,
            layout.NETWORK_RESIDUAL_TABLE_PATH,
        ):
            if path in all_table_files:
                files[path] = all_table_files[path]

    # Include optional tensors gated by "tensors" section.
    if "tensors" in options.sections:
        for path in (layout.ATTENTION_TENSOR_PATH,):
            if path in all_table_files:
                files[path] = all_table_files[path]

    png_files, unavailable_items = render_png_files(analysis)
    files.update(png_files)

    manifest = ExportBundleManifest(
        prompt=prompt,
        checkpoint_path=analysis.checkpoint.path,
        runtime=analysis.analysis_runtime,
        position_encoding=analysis.position_encoding,
        capabilities=(
            analysis.capabilities.model_dump() if analysis.capabilities else None
        ),
        options=options.model_dump(mode="json"),
        warnings=list(
            (analysis.network or {}).get("availability", {}).get("warnings", [])
        ),
        unavailable_sections=[UnavailableSection(**item) for item in unavailable_items],
    )

    # Populate the file listing *before* building summary so graph assets appear.
    manifest.files = _build_file_entries(files)

    files.update(
        build_summary_markdown(
            prompt=prompt, analysis=analysis, health=health, manifest=manifest
        )
    )
    files[layout.MANIFEST_PATH] = json.dumps(
        manifest.model_dump(mode="json"), indent=2, sort_keys=True
    ).encode("utf-8")
    manifest.files = _build_file_entries(files)
    files[layout.MANIFEST_PATH] = json.dumps(
        manifest.model_dump(mode="json"), indent=2, sort_keys=True
    ).encode("utf-8")
    return files, manifest


def write_bundle(
    *,
    prompt: str,
    analysis: AnalyzeResponse,
    health: HealthResponse,
    options: ExportOptions,
    output_path: Path,
) -> ExportBundleManifest:
    files, manifest = build_bundle_file_map(
        prompt=prompt, analysis=analysis, health=health, options=options
    )
    if options.output_mode == "directory":
        output_path.mkdir(parents=True, exist_ok=True)
        for relative_path, content in files.items():
            target = output_path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path, content in files.items():
            archive.writestr(relative_path, content)
    return manifest


def build_zip_bytes(
    *,
    prompt: str,
    analysis: AnalyzeResponse,
    health: HealthResponse,
    options: ExportOptions,
) -> bytes:
    files, _ = build_bundle_file_map(
        prompt=prompt, analysis=analysis, health=health, options=options
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path, content in files.items():
            archive.writestr(relative_path, content)
    return buffer.getvalue()


def _guess_media_type(path: str) -> str:
    if path.endswith(".jsonl"):
        return "application/jsonl"
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _build_file_entries(files: dict[str, bytes]) -> list[ExportFileEntry]:
    return [
        ExportFileEntry(
            path=path,
            media_type=_guess_media_type(path),
            section=path.split("/", maxsplit=1)[0] if "/" in path else path,
            size_bytes=len(content),
        )
        for path, content in sorted(files.items())
    ]
