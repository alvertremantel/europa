from __future__ import annotations

from pathlib import Path

import pytest

from eur_ts.config.templates import TRAIN_CONFIG_GUIDE, TRAIN_CONFIG_TEMPLATE
from eur_ts.config.toml_io import load_train_config


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


def test_load_train_config_parses_valid_toml(tmp_path: Path) -> None:
    config = load_train_config(write_config(tmp_path))

    assert config.data_dir == "data/my-dataset"
    assert config.output_dir == "runs/my-run"
    assert config.device == "auto"
    assert config.sequence_length == 32
    assert config.n_heads == 4
    assert config.position_encoding == "type_place"
    assert config.additional_epochs is None
    assert config.curriculum_name is None


def test_load_train_config_accepts_fixed_meaning_position_encoding(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            'position_encoding = "type_place"', 'position_encoding = "fixed_meaning"'
        ),
    )

    config = load_train_config(path)

    assert config.position_encoding == "fixed_meaning"


def test_load_train_config_rejects_blank_required_value(tmp_path: Path) -> None:
    path = write_config(
        tmp_path, VALID_TOML.replace('data_dir = "data/my-dataset"', 'data_dir = ""')
    )

    with pytest.raises(ValueError, match=r"\[paths\]\.data_dir must not be empty"):
        load_train_config(path)


def test_load_train_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_TOML + "\n[extra]\nvalue = 1\n")

    with pytest.raises(ValueError, match=r"unknown section \[extra\]"):
        load_train_config(path)


def test_load_train_config_rejects_invalid_enum(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace('training_mode = "token_stream"', 'training_mode = "bogus"'),
    )

    with pytest.raises(ValueError, match="training_mode must be one of"):
        load_train_config(path)


def test_load_train_config_rejects_removed_auto_resume(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            '[resume]\nresume_from = ""\nadditional_epochs = ""',
            '[resume]\nresume_from = ""\nauto_resume = false\nadditional_epochs = ""',
        ),
    )

    with pytest.raises(ValueError, match=r"\[resume\]\.auto_resume has been removed"):
        load_train_config(path)


def test_load_train_config_rejects_removed_checkpoint_section(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML
        + "\n[checkpoint]\ncheckpoint_keep_last = 2\ncheckpoint_max_kept = 3\n",
    )

    with pytest.raises(ValueError, match=r"\[checkpoint\] has been removed"):
        load_train_config(path)


def test_load_train_config_rejects_removed_eval_batches(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            "[logging]\nlog_interval = 5\nmax_new_tokens = 24",
            "[logging]\nlog_interval = 5\neval_batches = 1\nmax_new_tokens = 24",
        ),
    )

    with pytest.raises(ValueError, match=r"\[logging\]\.eval_batches has been removed"):
        load_train_config(path)


def test_load_train_config_rejects_removed_exact_match_samples(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            "[logging]\nlog_interval = 5\nmax_new_tokens = 24",
            "[logging]\nlog_interval = 5\nexact_match_samples = 64\nmax_new_tokens = 24",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"\[logging\]\.exact_match_samples has been removed",
    ):
        load_train_config(path)


def test_load_train_config_allows_omitted_optional_keys(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace('resume_from = ""\n', "")
        .replace('additional_epochs = ""\n', "")
        .replace('curriculum_name = ""\n', ""),
    )

    config = load_train_config(path)

    assert config.resume_from is None
    assert config.additional_epochs is None
    assert config.curriculum_name is None


def test_template_and_guide_cover_key_variables() -> None:
    for key in (
        "data_dir",
        "output_dir",
        "device",
        "sequence_length",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_hidden",
        "position_encoding",
        "training_mode",
        "training_format",
        "max_new_tokens",
    ):
        assert key in TRAIN_CONFIG_TEMPLATE
        assert key in TRAIN_CONFIG_GUIDE
    assert "fixed_meaning" in TRAIN_CONFIG_TEMPLATE
    assert "fixed_meaning" in TRAIN_CONFIG_GUIDE
