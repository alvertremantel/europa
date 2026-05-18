from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from eur_is.backend.schemas import (
    ActivationSummaryResponse,
    AnalyzeResponse,
    AttentionHeadSummary,
    AttentionSummaryResponse,
    CheckpointResponse,
    GeneratedAnswerResponse,
    GeneratedAnswerToken,
    HealthResponse,
    ModelConfigResponse,
    ProblemMetadataResponse,
    RuntimeCapabilitiesResponse,
    TopPrediction,
)
from eur_is.export.config_io import export_options_from_mapping
from eur_is.export.layout import (
    ACTIVATION_SUMMARY_TABLE_PATH,
    ATTENTION_HEAD_SUMMARY_ASSET_PATH,
    ATTENTION_MAPS_UNAVAILABLE_ASSET_PATH,
    GENERATED_ANSWER_TOPK_TABLE_PATH,
    MANIFEST_PATH,
    NETWORK_UNAVAILABLE_ASSET_PATH,
    RAW_ANALYZE_RESPONSE_PATH,
    SUMMARY_PATH,
    TOKENS_TABLE_PATH,
)
from eur_is.export.models import ExportOptions, canonical_minimal_paths
from eur_is.export.png import render_png_files
from eur_is.export.serializers.raw import build_raw_files
from eur_is.export.serializers.tables import build_table_files
from eur_is.export.writer import build_bundle_file_map, write_bundle


def build_fake_analysis(
    *, attention: bool = True, network: bool = True
) -> AnalyzeResponse:
    capabilities = RuntimeCapabilitiesResponse(
        prompt_analysis=True,
        generated_answer=True,
        attention_view=attention,
        network_analysis=network,
        circuitsvis_attention=attention,
    )
    attention_summary = None
    attention_tensor = None
    if attention:
        attention_summary = AttentionSummaryResponse(
            heads=[
                [
                    AttentionHeadSummary(
                        entropy=1.0,
                        max_weight=0.75,
                        mean_diagonal=0.5,
                        strongest_pair={
                            "query_index": 0,
                            "key_index": 1,
                            "query_token": "3",
                            "key_token": "+",
                            "weight": 0.75,
                        },
                    )
                ]
            ]
        )
        attention_tensor = [[[[0.25, 0.75], [0.6, 0.4]]]]
    network_payload = None
    if network:
        network_payload = {
            "availability": {"warnings": ["example warning"]},
            "controls": {
                "mlp_threshold": 0.0,
                "top_k": 5,
                "top_neurons": 8,
                "selected_token_index": 1,
            },
            "mlp": {
                "availability": "available",
                "threshold": 0.0,
                "layers": [
                    {
                        "layer": 0,
                        "availability": "available",
                        "tokens": [
                            {
                                "token_index": 0,
                                "token": "3",
                                "active_fraction_abs": 0.5,
                                "max_abs_activation": 1.25,
                                "top_neurons": [
                                    {
                                        "neuron_index": 2,
                                        "value": 1.25,
                                        "abs_value": 1.25,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            "attention": {
                "availability": "available",
                "layers": [
                    {
                        "layer": 0,
                        "availability": "available",
                        "heads": [
                            {
                                "layer": 0,
                                "head": 0,
                                "mean_entropy": 0.5,
                                "max_weight": 0.75,
                                "self_attention_mass": 0.25,
                                "previous_token_mass": 0.25,
                                "strongest_pair": {"query_index": 0, "key_index": 1},
                            }
                        ],
                    }
                ],
            },
            "residual": {
                "availability": "available",
                "layers": [
                    {
                        "layer": 0,
                        "availability": "available",
                        "tokens": [
                            {
                                "token_index": 0,
                                "token": "3",
                                "norm": 2.5,
                                "top_dimensions": [
                                    {"dimension": 1, "value": 0.5, "abs_value": 0.5}
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    return AnalyzeResponse(
        position_encoding="type_place",
        analysis_runtime="native_pytorch",
        capabilities=capabilities,
        tokens=["3", "+"],
        attention=attention_tensor,
        activations=[[[1.0, 0.0]], [[0.5, 0.5]]],
        logits=[[2.0, 0.0, -1.0], [1.5, 0.5, -0.5]],
        top_predictions=[
            TopPrediction(token="3", confidence=0.8, logit=2.0),
            TopPrediction(token="0", confidence=0.7, logit=1.5),
        ],
        top_k_predictions=[
            [TopPrediction(token="3", confidence=0.8, logit=2.0)],
            [TopPrediction(token="0", confidence=0.7, logit=1.5)],
        ],
        attention_summary=attention_summary,
        activation_summary=ActivationSummaryResponse(
            token_layer_l2=[[1.0], [0.75]],
            token_layer_max_abs=[[1.0], [0.5]],
            layer_mean_l2=[0.875],
            layer_peak_l2=[1.0],
            token_peak_l2=[1.0, 0.75],
            global_max_abs=1.0,
        ),
        answer_position=1,
        generated_answer=GeneratedAnswerResponse(
            text="03000000",
            tokens=list("03000000"),
            token_count=8,
            is_correct=True,
            is_valid_canonical=True,
            validation_error=None,
        ),
        generated_answer_top_k=[
            GeneratedAnswerToken(
                token="0",
                top_predictions=[TopPrediction(token="0", confidence=0.9, logit=3.0)],
            )
        ],
        config=ModelConfigResponse(
            n_layers=1,
            n_heads=1,
            d_model=2,
            d_head=2,
            mlp_hidden=4,
            sequence_length=64,
            vocab_size=3,
            dropout=0.0,
        ),
        problem=ProblemMetadataResponse(
            category="binary",
            kind="binary::small-small::+",
            curriculum_group="easy_binary_add_sub",
        ),
        checkpoint=CheckpointResponse(
            path="runs/fake.pt",
            device="cpu",
            epoch=1,
            exact_match=1.0,
            val_loss=0.1,
            train_loss=0.1,
            checkpoint_schema_version=1,
        ),
        network=network_payload,
    )


def build_fake_health() -> HealthResponse:
    return HealthResponse(
        position_encoding="type_place",
        analysis_runtime="native_pytorch",
        capabilities=RuntimeCapabilitiesResponse(
            prompt_analysis=True,
            generated_answer=True,
            attention_view=True,
            network_analysis=True,
            circuitsvis_attention=True,
        ),
        status="ok",
        device="cpu",
        checkpoint=CheckpointResponse(
            path="runs/fake.pt",
            device="cpu",
            epoch=1,
            exact_match=1.0,
            val_loss=0.1,
            train_loss=0.1,
            checkpoint_schema_version=1,
        ),
        detail=None,
    )


def test_export_options_defaults_normalization_and_png_requirement() -> None:
    options = export_options_from_mapping({"sections": ["RAW", "tables", "png"]})
    assert options.sections == ["raw", "tables", "png"]
    assert options.output_mode == "zip"
    with pytest.raises(ValueError):
        ExportOptions(png_assets=False)


def test_canonical_minimal_paths_include_core_bundle_files() -> None:
    paths = canonical_minimal_paths()
    assert MANIFEST_PATH in paths
    assert SUMMARY_PATH in paths
    assert RAW_ANALYZE_RESPONSE_PATH in paths
    assert TOKENS_TABLE_PATH in paths


def test_raw_serializer_contains_analysis_payload_keys() -> None:
    files = build_raw_files(analysis=build_fake_analysis(), health=build_fake_health())
    payload = json.loads(files[RAW_ANALYZE_RESPONSE_PATH].decode("utf-8"))
    assert {
        "tokens",
        "logits",
        "activations",
        "generated_answer",
        "config",
        "checkpoint",
    } <= set(payload)
    assert "network" in payload


def test_table_serializers_emit_expected_rows_and_columns() -> None:
    files = build_table_files(build_fake_analysis())
    token_rows = list(
        csv.DictReader(StringIO(files[TOKENS_TABLE_PATH].decode("utf-8")))
    )
    activation_rows = list(
        csv.DictReader(StringIO(files[ACTIVATION_SUMMARY_TABLE_PATH].decode("utf-8")))
    )
    answer_rows = list(
        csv.DictReader(
            StringIO(files[GENERATED_ANSWER_TOPK_TABLE_PATH].decode("utf-8"))
        )
    )
    assert len(token_rows) == 2
    assert {"token_index", "token", "prompt_index"} <= set(token_rows[0])
    assert len(activation_rows) == 2
    assert {"token_index", "layer", "l2", "max_abs"} <= set(activation_rows[0])
    assert len(answer_rows) == 1
    assert {"generated_token_index", "predicted_token", "confidence", "logit"} <= set(
        answer_rows[0]
    )


def test_markdown_summary_includes_unavailable_notes() -> None:
    analysis = build_fake_analysis(attention=False, network=False)
    health = build_fake_health()
    files, manifest = build_bundle_file_map(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=ExportOptions(output_mode="zip"),
    )
    summary = files[SUMMARY_PATH].decode("utf-8")
    assert "<do> <calc> 03000000 + 03000000 =" in summary
    assert "runs/fake.pt" in summary
    assert "03000000" in summary
    assert "native_pytorch" in summary
    assert manifest.unavailable_sections
    assert "attention_summary" in summary


def test_png_renderer_emits_data_pngs_and_placeholders() -> None:
    files, unavailable = render_png_files(build_fake_analysis())
    assert ATTENTION_HEAD_SUMMARY_ASSET_PATH in files
    assert files[ATTENTION_HEAD_SUMMARY_ASSET_PATH]
    native_files, native_unavailable = render_png_files(
        build_fake_analysis(attention=False, network=False)
    )
    assert ATTENTION_MAPS_UNAVAILABLE_ASSET_PATH in native_files
    assert NETWORK_UNAVAILABLE_ASSET_PATH in native_files
    assert native_unavailable
    assert unavailable == []


def test_writer_supports_directory_and_zip_outputs(tmp_path: Path) -> None:
    analysis = build_fake_analysis()
    health = build_fake_health()
    directory_output = tmp_path / "bundle-dir"
    write_bundle(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=ExportOptions(output_mode="directory"),
        output_path=directory_output,
    )
    assert (directory_output / MANIFEST_PATH).exists()
    assert (directory_output / SUMMARY_PATH).exists()

    zip_output = tmp_path / "bundle.zip"
    write_bundle(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=ExportOptions(output_mode="zip"),
        output_path=zip_output,
    )
    with ZipFile(zip_output) as archive:
        names = set(archive.namelist())
        assert MANIFEST_PATH in names
        assert SUMMARY_PATH in names
        manifest = json.loads(archive.read(MANIFEST_PATH).decode("utf-8"))
    assert manifest["files"]
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert MANIFEST_PATH in manifest_paths
    assert SUMMARY_PATH in manifest_paths
    assert manifest["options"]["png_assets"] is True


def test_cli_single_prompt_and_batch_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eur_is.export import cli
    from eur_is.export.runner import ExportRunResult

    def fake_run_export_analysis(**_: object) -> ExportRunResult:
        return ExportRunResult(
            analysis=build_fake_analysis(), health=build_fake_health()
        )

    monkeypatch.setattr(cli, "run_export_analysis", fake_run_export_analysis)
    single_output = tmp_path / "single.zip"
    assert (
        cli.main(
            [
                "--checkpoint",
                "runs/fake.pt",
                "--prompt",
                "<do> <calc> 03000000 + 03000000 =",
                "--output",
                str(single_output),
                "--zip",
            ]
        )
        == 0
    )
    assert single_output.exists()

    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(
        "<do> <calc> 03000000 + 03000000 =\n<do> <calc> 04000000 + 01000000 =\n",
        encoding="utf-8",
    )
    batch_output = tmp_path / "batch"
    assert (
        cli.main(
            [
                "--checkpoint",
                "runs/fake.pt",
                "--prompts-file",
                str(prompts_file),
                "--output",
                str(batch_output),
                "--directory",
            ]
        )
        == 0
    )
    assert (batch_output / "batch-manifest.json").exists()
    prompt_dirs = [path for path in batch_output.iterdir() if path.is_dir()]
    assert len(prompt_dirs) == 2
    for prompt_dir in prompt_dirs:
        assert (prompt_dir / MANIFEST_PATH).exists()
        assert (prompt_dir / SUMMARY_PATH).exists()


# --- New tests for the ITS export follow-up work ---


def test_cli_missing_checkpoint_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """--checkpoint must be provided; the CLI should reject empty/missing paths."""
    from eur_is.export import cli

    output = tmp_path / "out.zip"
    # No --checkpoint at all.
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--prompt",
                "<do> <calc> 03000000 + 03000000 =",
                "--output",
                str(output),
            ]
        )
    _, stderr = capsys.readouterr()
    assert "checkpoint is required" in stderr


def test_cli_config_precedence_device_not_overridden_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config values must not be clobbered by argparse defaults."""
    import json as _json

    from eur_is.export import cli
    from eur_is.export.runner import ExportRunResult

    def fake_run_export_analysis(**_: object) -> ExportRunResult:
        return ExportRunResult(
            analysis=build_fake_analysis(), health=build_fake_health()
        )

    monkeypatch.setattr(cli, "run_export_analysis", fake_run_export_analysis)

    # Config overrides device; CLI does not pass --device.
    config_path = tmp_path / "export.json"
    config_path.write_text(
        _json.dumps({"checkpoint_path": "runs/fake.pt", "device": "cpu"}),
        encoding="utf-8",
    )
    output = tmp_path / "single.zip"

    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--prompt",
                "<do> <calc> 03000000 + 03000000 =",
                "--output",
                str(output),
                "--zip",
            ]
        )
        == 0
    )
    assert output.exists()


def test_cli_config_precedence_output_mode_not_overridden_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config output_mode must survive when the user does not pass --zip/--directory."""
    import json as _json

    from eur_is.export import cli
    from eur_is.export.runner import ExportRunResult

    def fake_run_export_analysis(**_: object) -> ExportRunResult:
        return ExportRunResult(
            analysis=build_fake_analysis(), health=build_fake_health()
        )

    monkeypatch.setattr(cli, "run_export_analysis", fake_run_export_analysis)

    config_path = tmp_path / "export.json"
    config_path.write_text(
        _json.dumps(
            {
                "checkpoint_path": "runs/fake.pt",
                "device": "cpu",
                "output_mode": "directory",
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bundle-dir"

    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--prompt",
                "<do> <calc> 03000000 + 03000000 =",
                "--output",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / MANIFEST_PATH).exists()
    assert (out_dir / SUMMARY_PATH).exists()


def test_cli_explicit_flag_overrides_config_output_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit --zip flag must override config output_mode."""
    import json as _json

    from eur_is.export import cli
    from eur_is.export.runner import ExportRunResult

    def fake_run_export_analysis(**_: object) -> ExportRunResult:
        return ExportRunResult(
            analysis=build_fake_analysis(), health=build_fake_health()
        )

    monkeypatch.setattr(cli, "run_export_analysis", fake_run_export_analysis)

    config_path = tmp_path / "export.json"
    config_path.write_text(
        _json.dumps(
            {
                "checkpoint_path": "runs/fake.pt",
                "device": "cpu",
                "output_mode": "directory",
            }
        ),
        encoding="utf-8",
    )
    out_zip = tmp_path / "bundle.zip"

    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--prompt",
                "<do> <calc> 03000000 + 03000000 =",
                "--output",
                str(out_zip),
                "--zip",
            ]
        )
        == 0
    )
    assert out_zip.exists()
    from zipfile import ZipFile

    with ZipFile(out_zip) as archive:
        names = set(archive.namelist())
        assert MANIFEST_PATH in names
        assert SUMMARY_PATH in names


def test_summary_includes_graph_assets_when_available() -> None:
    """summary.md must list graph assets from the populated manifest."""
    analysis = build_fake_analysis(attention=True, network=True)
    health = build_fake_health()
    files, manifest = build_bundle_file_map(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=ExportOptions(output_mode="zip"),
    )
    summary = files[SUMMARY_PATH].decode("utf-8")
    # Graph assets should list real PNG files, not be empty.
    graph_section = summary.split("## Graph assets")[1].split("##")[0]
    asset_paths = [
        line.strip("- `").rstrip("`").strip()
        for line in graph_section.splitlines()
        if line.startswith("- `")
    ]
    assert len(asset_paths) > 0, "summary.md should list graph asset entries"
    assert any("assets/" in path for path in asset_paths)
    # Manifest should also have the files populated.
    assert manifest.files
    assert any(entry.path.startswith("assets/") for entry in manifest.files)


def test_section_behavior_required_paths_always_generated(
    tmp_path: Path,
) -> None:
    """Required-bundle paths are always generated regardless of sections."""
    analysis = build_fake_analysis(attention=True, network=True)
    health = build_fake_health()
    # Ask for no optional sections.
    options = ExportOptions(output_mode="directory", sections=["markdown", "png"])
    manifest = write_bundle(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=options,
        output_path=tmp_path / "bundle",
    )
    # Required paths must exist on disk and in the manifest.
    assert (tmp_path / "bundle" / RAW_ANALYZE_RESPONSE_PATH).exists()
    assert (tmp_path / "bundle" / TOKENS_TABLE_PATH).exists()
    manifest_paths = {entry.path for entry in manifest.files}
    assert RAW_ANALYZE_RESPONSE_PATH in manifest_paths
    assert TOKENS_TABLE_PATH in manifest_paths

    # Unless the runtime supports it, attention/network tables are optional;
    # they should be absent when "tables" is not in sections.
    assert ATTENTION_HEAD_SUMMARY_ASSET_PATH in manifest_paths  # PNGs always
    attention_csv = "tables/attention_head_summary.csv"
    assert attention_csv not in manifest_paths, (
        "attention_head_summary table should be omitted when tables section is unselected"
    )


def test_section_behavior_optional_tables_gated(tmp_path: Path) -> None:
    """When 'tables' section is requested, optional tables appear."""
    analysis = build_fake_analysis(attention=True, network=True)
    health = build_fake_health()
    options = ExportOptions(
        output_mode="directory", sections=["tables", "markdown", "png"]
    )
    manifest = write_bundle(
        prompt="<do> <calc> 03000000 + 03000000 =",
        analysis=analysis,
        health=health,
        options=options,
        output_path=tmp_path / "bundle",
    )
    manifest_paths = {entry.path for entry in manifest.files}
    attention_csv = "tables/attention_head_summary.csv"
    assert attention_csv in manifest_paths, (
        "attention_head_summary table must be included when tables section is selected"
    )
