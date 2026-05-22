"""Compatibility facade for training examples."""

from .data.examples import ArithmeticExample, load_examples, transform_examples

__all__ = ["ArithmeticExample", "load_examples", "transform_examples"]
