from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .core import Config, generate_dataset


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Generate a stratified arithmetic dataset."
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
    print(json.dumps(asdict(config), indent=2, sort_keys=True))
    generate_dataset(config)


if __name__ == "__main__":
    main()
