from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from eur_ts.config.cli import create_new_template, main
from eur_ts.config.templates import TEMPLATE_FILENAME

VALID_TOML = """
[paths]
data_dir = "data/my-dataset"
output_dir = "runs/my-run"

[runtime]
device = "auto"
seed = 42

[resume]
resume_from = ""
additional_epochs = ""

[model]
sequence_length = 32
d_model = 16
n_heads = 4
n_layers = 1
mlp_hidden = 64
dropout = 0.1
position_encoding = "type_place"

[optimization]
batch_size = 8
epochs = 2
learning_rate = 0.001
weight_decay = 0.1
grad_clip = 1.0

[logging]
log_interval = 5
max_new_tokens = 24

[training]
training_mode = "token_stream"
training_format = "final_only"
skip_overlong_examples = false
curriculum_name = ""
"""


def write_config(tmp_path: Path, content: str = VALID_TOML) -> Path:
    path = tmp_path / "train-config.toml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_create_new_template_writes_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    main(["--new"])

    created = tmp_path / TEMPLATE_FILENAME
    assert created.exists()
    assert "[paths]" in created.read_text(encoding="utf-8")


def test_create_new_template_refuses_overwrite(tmp_path: Path) -> None:
    create_new_template(tmp_path)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        create_new_template(tmp_path)


def test_config_guide_prints_variables(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--guide"])
    output = capsys.readouterr().out

    assert "training_mode" in output
    assert "50 validation problems" in output


def test_config_size_prints_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(
        tmp_path, VALID_TOML.replace("mlp_hidden = 64", "mlp_hidden = 32")
    )

    main(["--size", str(path)])
    payload = tomllib.loads(capsys.readouterr().out)["model_size"]

    assert payload["total_parameters"] > 0
    assert payload["total_virtual_neurons"] == 1 * 32 * 32
    assert payload["n_layers"] == 1


def test_config_cli_rejects_multiple_modes() -> None:
    with pytest.raises(SystemExit):
        main(["--new", "--guide"])
