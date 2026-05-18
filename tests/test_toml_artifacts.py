from __future__ import annotations

import tomllib
from typing import cast

from eur_ts.artifacts import read_toml, toml_text, write_toml
from eur_ts.config import TrainConfig
from eur_ts.evaluator.metadata import load_metadata
from eur_ts.trainer.training.checkpointing import CheckpointManager
from eur_ts.trainer.training.resume import history_from_payload


def test_toml_artifact_writer_omits_none_and_round_trips(tmp_path):
    path = tmp_path / "artifact.toml"

    write_toml(
        path,
        {
            "name": "demo",
            "optional": None,
            "nested": {"value": 3, "missing": None},
            "records": [
                {"kind": "binary::small-small::+", "score": 1.0, "note": None},
                {"kind": "parentheses::small-medium-large::(+)*", "score": 0.5},
            ],
        },
    )

    loaded = read_toml(path)

    assert loaded["name"] == "demo"
    assert "optional" not in loaded
    assert loaded["nested"] == {"value": 3}
    records = cast(list[dict[str, object]], loaded["records"])
    assert records[0]["kind"] == "binary::small-small::+"
    assert "note" not in records[0]


def test_toml_text_emits_parseable_document():
    text = toml_text({"summary": {"accuracy": 0.75, "roles": ["best", "last"]}})

    loaded = tomllib.loads(text)

    assert loaded["summary"]["accuracy"] == 0.75
    assert loaded["summary"]["roles"] == ["best", "last"]


def test_evaluator_metadata_prefers_toml_and_falls_back_to_legacy_json(tmp_path):
    write_toml(tmp_path / "meta.toml", {"format": "toml"})
    (tmp_path / "meta.json").write_text('{"format":"json"}\n', encoding="utf-8")

    assert load_metadata(tmp_path) == {"format": "toml"}

    (tmp_path / "meta.toml").unlink()

    assert load_metadata(tmp_path) == {"format": "json"}


def test_resume_history_prefers_toml_and_falls_back_to_legacy_json(tmp_path):
    checkpoint_path = tmp_path / "checkpoint-last.pt"
    checkpoint_path.write_bytes(b"placeholder")
    payload: dict[str, object] = {"history": [{"epoch": 1}]}
    write_toml(tmp_path / "history.toml", {"history": [{"epoch": 1}, {"epoch": 2}]})
    (tmp_path / "history.json").write_text(
        '[{"epoch":1},{"epoch":3}]\n', encoding="utf-8"
    )

    assert history_from_payload(payload, checkpoint_path) == [
        {"epoch": 1},
        {"epoch": 2},
    ]

    (tmp_path / "history.toml").unlink()

    assert history_from_payload(payload, checkpoint_path) == [
        {"epoch": 1},
        {"epoch": 3},
    ]


def test_checkpoint_manifest_reads_legacy_json_and_writes_toml(tmp_path):
    manager = CheckpointManager(
        tmp_path,
        TrainConfig(output_dir=str(tmp_path), checkpoint_dir_name="checkpoints"),
    )
    legacy_manifest = tmp_path / "checkpoints" / "manifest.json"
    legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest.write_text(
        '{"schema_version":1,"records":[{"epoch":1,"path":"epoch-0001.pt","available":true,"val_loss":1.0,"exact_match":0.5,"train_loss":1.2,"roles":["last"],"global_step":10}]}\n',
        encoding="utf-8",
    )

    manifest = manager.load_manifest()
    records = cast(list[dict[str, object]], manifest["records"])
    assert records[0]["epoch"] == 1

    manager._write_manifest(manifest)

    assert (tmp_path / "checkpoints" / "manifest.toml").exists()
    assert read_toml(tmp_path / "checkpoints" / "manifest.toml")["schema_version"] == 1
