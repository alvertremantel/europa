from __future__ import annotations

import argparse
from dataclasses import asdict

from eis.artifacts import toml_text

from .core import Config, generate_dataset


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Generate a stratified arithmetic dataset for Europa ALM-IS."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    return Config(
        seed=args.seed,
        output_dir=args.output_dir,
        validate=not args.no_validate,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    print(toml_text({"generator": asdict(config)}).rstrip())
    generate_dataset(config)


if __name__ == "__main__":
    main()
