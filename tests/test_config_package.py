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
auto_resume = false
additional_epochs = ""

[model]
sequence_length = 32
d_model = 16
n_heads = 4
n_layers = 1
mlp_hidden = 64
dropout = 0.1

[optimization]
batch_size = 8
epochs = 2
learning_rate = 0.001
weight_decay = 0.1
grad_clip = 1.0

[logging]
log_interval = 5
eval_batches = 1
exact_match_samples = 2
max_new_tokens = 24

[checkpoint]
checkpoint_keep_last = 2
checkpoint_max_kept = 3
checkpoint_keep_best = 1
checkpoint_jump_threshold = 0.05
checkpoint_dir_name = "checkpoints"

[training]
training_mode = "token_stream"
training_format = "final_only"
skip_overlong_examples = false
curriculum_name = ""

[balanced_validation]
enabled = false
group_by = "kind"
sample_size_per_group = 2
seed = 42
batch_size = ""
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
    assert config.additional_epochs is None
    assert config.curriculum_name is None
    assert config.balanced_val_batch_size is None


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


def test_load_train_config_allows_zero_checkpoint_keep_values(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            "checkpoint_keep_last = 2", "checkpoint_keep_last = 0"
        ).replace("checkpoint_keep_best = 1", "checkpoint_keep_best = 0"),
    )

    config = load_train_config(path)

    assert config.checkpoint_keep_last == 0
    assert config.checkpoint_keep_best == 0


def test_load_train_config_allows_omitted_optional_keys(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace('resume_from = ""\n', "")
        .replace('additional_epochs = ""\n', "")
        .replace('curriculum_name = ""\n', "")
        .replace('batch_size = ""\n', ""),
    )

    config = load_train_config(path)

    assert config.resume_from is None
    assert config.additional_epochs is None
    assert config.curriculum_name is None
    assert config.balanced_val_batch_size is None


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
        "training_mode",
        "training_format",
        "sample_size_per_group",
    ):
        assert key in TRAIN_CONFIG_TEMPLATE
        assert key in TRAIN_CONFIG_GUIDE
