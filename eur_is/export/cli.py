"""CLI entrypoint for ITS export bundles."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from eur_is.export.config_io import load_export_options
from eur_is.export.models import ExportOptions
from eur_is.export.runner import run_export_analysis
from eur_is.export.writer import build_bundle_file_map, write_bundle


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_mapping: dict[str, object] = {}
    if args.config:
        config_mapping = load_export_options(Path(args.config)).model_dump(mode="json")

    cli_mapping = _build_cli_mapping(args)
    options = ExportOptions.model_validate({**config_mapping, **cli_mapping})

    checkpoint_raw = options.checkpoint_path or ""
    if not checkpoint_raw.strip():
        parser.error("--checkpoint is required")
    checkpoint_path = Path(checkpoint_raw)
    device = _resolve_device(options.device)

    if args.prompts_file:
        prompts = _load_prompts(Path(args.prompts_file))
        if not prompts:
            parser.error("--prompts-file contained no prompts")
        output_path = Path(args.output)
        if options.output_mode == "directory":
            output_path.mkdir(parents=True, exist_ok=True)
            batch_manifest: list[dict[str, str]] = []
            for index, prompt in enumerate(prompts):
                prompt_id = f"{index:03d}-{_sanitize_prompt_id(prompt)}"
                result = run_export_analysis(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    prompt=prompt,
                    include_network=options.include_network,
                    mlp_threshold=options.mlp_threshold,
                    top_k=options.top_k,
                    top_neurons=options.top_neurons,
                    selected_token_index=options.selected_token_index,
                )
                write_bundle(
                    prompt=prompt,
                    analysis=result.analysis,
                    health=result.health,
                    options=options.model_copy(update={"output_mode": "directory"}),
                    output_path=output_path / prompt_id,
                )
                batch_manifest.append(
                    {"prompt_id": prompt_id, "prompt": prompt, "status": "ok"}
                )
            (output_path / "batch-manifest.json").write_text(
                json.dumps({"prompts": batch_manifest}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        batch_manifest: list[dict[str, str]] = []
        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
            for index, prompt in enumerate(prompts):
                prompt_id = f"{index:03d}-{_sanitize_prompt_id(prompt)}"
                result = run_export_analysis(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    prompt=prompt,
                    include_network=options.include_network,
                    mlp_threshold=options.mlp_threshold,
                    top_k=options.top_k,
                    top_neurons=options.top_neurons,
                    selected_token_index=options.selected_token_index,
                )
                files, _ = build_bundle_file_map(
                    prompt=prompt,
                    analysis=result.analysis,
                    health=result.health,
                    options=options,
                )
                for relative_path, content in files.items():
                    archive.writestr(f"{prompt_id}/{relative_path}", content)
                batch_manifest.append(
                    {"prompt_id": prompt_id, "prompt": prompt, "status": "ok"}
                )
            archive.writestr(
                "batch-manifest.json",
                json.dumps({"prompts": batch_manifest}, indent=2, sort_keys=True),
            )
        return 0

    if not args.prompt:
        parser.error("Provide either --prompt or --prompts-file")
    result = run_export_analysis(
        checkpoint_path=checkpoint_path,
        device=device,
        prompt=args.prompt,
        include_network=options.include_network,
        mlp_threshold=options.mlp_threshold,
        top_k=options.top_k,
        top_neurons=options.top_neurons,
        selected_token_index=options.selected_token_index,
    )
    write_bundle(
        prompt=args.prompt,
        analysis=result.analysis,
        health=result.health,
        options=options,
        output_path=Path(args.output),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="its-export")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-neurons", type=int)
    parser.add_argument("--mlp-threshold", type=float)
    parser.add_argument("--selected-token-index", type=int)
    parser.add_argument("--config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--zip", action="store_true")
    mode.add_argument("--directory", action="store_true")
    return parser


def _build_cli_mapping(args: argparse.Namespace) -> dict[str, object]:
    """Build a mapping of only explicitly provided CLI values."""
    mapping: dict[str, object] = {}
    if args.checkpoint is not None:
        mapping["checkpoint_path"] = args.checkpoint
    if args.device is not None:
        mapping["device"] = args.device
    if args.mlp_threshold is not None:
        mapping["mlp_threshold"] = args.mlp_threshold
    if args.top_k is not None:
        mapping["top_k"] = args.top_k
    if args.top_neurons is not None:
        mapping["top_neurons"] = args.top_neurons
    if args.selected_token_index is not None:
        mapping["selected_token_index"] = args.selected_token_index
    if args.directory or args.zip:
        mapping["output_mode"] = "directory" if args.directory else "zip"
    if args.prompts_file:
        mapping["prompt_source"] = "file"
    return mapping


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_prompts(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sanitize_prompt_id(prompt: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", prompt).strip("-").lower()
    return (normalized or "prompt")[:48]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
