from __future__ import annotations

import argparse
from pathlib import Path
import sys

from eis.artifacts import toml_text

from .sizing import model_size_from_config
from .templates import TEMPLATE_FILENAME, TRAIN_CONFIG_GUIDE, TRAIN_CONFIG_TEMPLATE
from .toml_io import load_train_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create, inspect, and explain Europa ALM-IS TOML training configs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-n", "--new", action="store_true", dest="new")
    group.add_argument("-g", "--guide", action="store_true", dest="guide")
    group.add_argument("-s", "--size", type=str, dest="size")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.new:
        create_new_template(Path.cwd())
        return
    if args.guide:
        print(TRAIN_CONFIG_GUIDE)
        return
    if args.size:
        config = load_train_config(Path(args.size))
        print(toml_text({"model_size": model_size_from_config(config)}).rstrip())
        return
    raise SystemExit("one of --new, --guide, or --size is required")


def create_new_template(root: Path) -> Path:
    path = root / TEMPLATE_FILENAME
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing config template: {path}")
    path.write_text(TRAIN_CONFIG_TEMPLATE, encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
