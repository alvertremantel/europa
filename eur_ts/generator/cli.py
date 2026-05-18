from __future__ import annotations

import argparse
from dataclasses import asdict

from eur_ts.artifacts import toml_text

from .core import Config, generate_dataset


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Generate a stratified arithmetic dataset for Europa ALM-IS."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    return Config(
        seed=args.seed,
        output_dir=args.output_dir,
        validate=not args.no_validate,
    )


def main() -> None:
    config = parse_args()
    print(toml_text({"generator": asdict(config)}).rstrip())
    generate_dataset(config)


if __name__ == "__main__":
    main()
