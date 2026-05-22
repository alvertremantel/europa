"""Argument parsing for the evaluator CLI."""

from __future__ import annotations

import argparse

ALL_SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Europa ALM-IS model across sampled problem strata."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
        help="Dataset files to draw the per-kind sample pool from.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sample-size-per-kind", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--failures-per-kind", type=int, default=3)
    parser.add_argument("--progress-interval-kinds", type=int, default=0)
    return parser.parse_args(argv)
