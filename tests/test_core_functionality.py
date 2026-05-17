from __future__ import annotations

import torch

from eur_ts.evaluator.core import BucketStats, bucket_row
from eur_ts.generator.core import (
    format_signed_number,
    format_unsigned_number,
    parse_signed_number,
    validate_line,
)
from eur_ts.trainer.config import ModelConfig
from eur_ts.trainer.data import ArithmeticTokenizer, vocab_for_training_format
from eur_ts.trainer.formatting import final_answer_from_line, format_training_line
from eur_ts.trainer.model import SmallCausalTransformer


def test_generator_validates_canonical_lines() -> None:
    line = "30000000 + 40000000 = <ans> 70000000"

    parsed = validate_line(line)

    assert parsed.category == "binary"
    assert parsed.kind == "binary::small-small::+"
    assert parsed.answer == 7
    assert format_unsigned_number(6) == "60000000"
    assert format_signed_number(-6) == "(-60000000)"
    assert parse_signed_number("(-60000000)") == -6


def test_training_tokenizer_round_trips_and_formats_scratchpads() -> None:
    line = "( 30000000 + 40000000 ) * 20000000 = <ans> 41000000"
    transformed, applied_format = format_training_line(line, "light_scratchpad")
    tokenizer = ArithmeticTokenizer(vocab_for_training_format("light_scratchpad"))

    assert applied_format == "parentheses_intermediate"
    assert transformed == (
        "( 30000000 + 40000000 ) * 20000000 = <ans> "
        "<work> <step> 70000000 <final> 41000000"
    )
    assert final_answer_from_line(transformed) == "41000000"
    assert tokenizer.decode(tokenizer.encode_line(transformed)) == transformed
    assert tokenizer.decode(tokenizer.encode_prompt(line)) == (
        "( 30000000 + 40000000 ) * 20000000 = <ans>"
    )


def test_small_transformer_forward_shape() -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
    )
    model = SmallCausalTransformer(config).eval()
    input_ids = torch.tensor(
        [tokenizer.encode_prompt("30000000 + 40000000 = <ans>")],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], tokenizer.vocab_size)


def test_evaluator_bucket_row_keeps_summary_math() -> None:
    stats = BucketStats(
        evaluated_count=4, perfect_count=3, canonical_prediction_count=2
    )

    row = bucket_row(name="binary::small-small::+", stats=stats, available_count=10)

    assert row["missed_count"] == 1
    assert row["accuracy"] == 0.75
    assert row["canonical_prediction_rate"] == 0.5
    assert row["available_count"] == 10
