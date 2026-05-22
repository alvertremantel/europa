from __future__ import annotations

import pytest

from eis.train.cli import parse_args


def test_train_subcommand_accepts_config_path() -> None:
    args = parse_args(["train", "train-config.toml"])

    assert args.command == "train"
    assert args.config == "train-config.toml"


def test_train_subcommand_rejects_legacy_training_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(["train", "--data-dir", "data/my-dataset"])
