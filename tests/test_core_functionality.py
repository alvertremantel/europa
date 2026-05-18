from __future__ import annotations

import pytest
import torch

from eur_ts.evaluator.core import BucketStats, bucket_row
from eur_ts.generator.core import (
    format_signed_number,
    format_unsigned_number,
    parse_signed_number,
    validate_line,
)
from eur_ts.config import ModelConfig
from eur_ts.trainer.data import ArithmeticTokenizer, vocab_for_training_format
from eur_ts.trainer.formatting import final_answer_from_line, format_training_line
from eur_ts.trainer.inference import _forward_model
from eur_ts.trainer.model import SmallCausalTransformer
from eur_ts.trainer.training.checkpointing import _model_config_from_payload


def test_generator_validates_canonical_lines() -> None:
    line = "<do> <calc> 30000000 + 40000000 = 70000000"
    parsed = validate_line(line)
    assert parsed.category == "binary"
    assert parsed.kind == "binary::small-small::+"
    assert parsed.answer == 7
    assert format_unsigned_number(6) == "60000000"
    assert format_signed_number(-6) == "(-60000000)"
    assert parse_signed_number("(-60000000)") == -6


def test_training_tokenizer_round_trips_and_formats_scratchpads() -> None:
    line = "<do> <calc> ( 30000000 + 40000000 ) * 20000000 = 41000000"
    transformed, applied_format = format_training_line(line, "light_scratchpad")
    tokenizer = ArithmeticTokenizer(vocab_for_training_format("light_scratchpad"))
    assert applied_format == "parentheses_intermediate"
    assert transformed == (
        "<do> <calc> ( 30000000 + 40000000 ) * 20000000 = "
        "<work> <step> 70000000 <final> 41000000"
    )
    assert final_answer_from_line(transformed) == "41000000"
    assert tokenizer.decode(tokenizer.encode_line(transformed)) == transformed
    assert tokenizer.decode(tokenizer.encode_prompt(line)) == (
        "<do> <calc> ( 30000000 + 40000000 ) * 20000000 ="
    )


def test_training_tokenizer_assigns_type_and_place_ids() -> None:
    tokenizer = ArithmeticTokenizer()
    token_ids, type_ids, place_ids = tokenizer.encode_prompt_with_type_place(
        "<do> <calc> (-60000000) + 30000000 ="
    )
    tokens = [tokenizer.id_to_token[token_id] for token_id in token_ids]
    assert tokens[:2] == ["<do>", "<calc>"]
    assert tokens[2:12] == ["(", "-", "6", "0", "0", "0", "0", "0", "0", "0"]
    assert type_ids[0] == 0
    assert type_ids[tokens.index("+")] == 1
    digit_indexes = [index for index, token in enumerate(tokens) if token.isdigit()]
    assert [type_ids[index] for index in digit_indexes] == [2] * 16
    assert place_ids[4:12] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert place_ids[16:24] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_training_tokenizer_assigns_places_to_generated_prefixes() -> None:
    tokenizer = ArithmeticTokenizer()
    prompt_ids = tokenizer.encode_prompt("<do> <calc> 30000000 + 40000000 =")
    for prefix, expected_places in [
        ("7", [1]),
        ("70", [1, 2]),
        ("70000000", [1, 2, 3, 4, 5, 6, 7, 8]),
    ]:
        token_ids = prompt_ids + [tokenizer.token_to_id[token] for token in prefix]
        _, place_ids = tokenizer.type_place_ids_for_token_ids(token_ids)
        assert place_ids[-len(expected_places) :] == expected_places


def test_small_transformer_type_place_forward_shape() -> None:
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
    input_token_ids, input_type_ids, input_place_ids = (
        tokenizer.encode_prompt_with_type_place("<do> <calc> 30000000 + 40000000 =")
    )
    input_ids = torch.tensor([input_token_ids], dtype=torch.long)
    type_ids = torch.tensor([input_type_ids], dtype=torch.long)
    place_ids = torch.tensor([input_place_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, type_ids, place_ids)
        helper_logits = _forward_model(
            model, input_ids, type_ids=type_ids, place_ids=place_ids
        )

    assert logits.shape == (1, input_ids.shape[1], tokenizer.vocab_size)
    assert torch.allclose(helper_logits, logits)
    with pytest.raises(ValueError, match="type_place"):
        model(input_ids)


def test_legacy_checkpoint_model_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="position_encoding"):
        _model_config_from_payload(
            {
                "model_config": {
                    "vocab_size": 32,
                    "sequence_length": 64,
                    "d_model": 16,
                    "n_heads": 4,
                    "n_layers": 1,
                    "mlp_hidden": 32,
                    "dropout": 0.0,
                }
            }
        )


def test_evaluator_bucket_row_keeps_summary_math() -> None:
    stats = BucketStats(
        evaluated_count=4, perfect_count=3, canonical_prediction_count=2
    )
    row = bucket_row(name="binary::small-small::+", stats=stats, available_count=10)
    assert row["missed_count"] == 1
    assert row["accuracy"] == 0.75
    assert row["canonical_prediction_rate"] == 0.5
    assert row["available_count"] == 10
