from __future__ import annotations

import torch

from eur_ts.evaluator.core import BucketStats, bucket_row
from eur_ts.generator.core import (
    format_signed_number,
    format_unsigned_number,
    parse_signed_number,
    validate_line,
)
from eur_ts.config import ModelConfig
from eur_ts.trainer.data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_ABSOLUTE,
    POSITION_ENCODING_DIGIT_ROLES,
    vocab_for_training_format,
)
from eur_ts.trainer.formatting import final_answer_from_line, format_training_line
from eur_ts.trainer.inference import _forward_model
from eur_ts.trainer.model import SmallCausalTransformer
from eur_ts.trainer.training.checkpointing import _model_config_from_payload


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


def test_training_tokenizer_assigns_digit_roles_only_within_numbers() -> None:
    tokenizer = ArithmeticTokenizer()

    _, position_roles = tokenizer.encode_prompt_with_roles(
        "(-60000000) + 30000000 = <ans>"
    )

    assert position_roles == [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        0,
        0,
        0,
        0,
    ]


def test_training_tokenizer_assigns_digit_roles_to_generated_prefixes() -> None:
    tokenizer = ArithmeticTokenizer()
    prompt_ids = tokenizer.encode_prompt("30000000 + 40000000 = <ans>")

    for prefix, expected_roles in [
        ("7", [1]),
        ("70", [1, 2]),
        ("70000000", [1, 2, 3, 4, 5, 6, 7, 8]),
    ]:
        token_ids = prompt_ids + [tokenizer.token_to_id[token] for token in prefix]

        position_roles = tokenizer.position_role_ids_for_token_ids(token_ids)

        assert position_roles[-len(expected_roles) :] == expected_roles


def test_training_tokenizer_role_reconstruction_tolerates_malformed_fields() -> None:
    tokenizer = ArithmeticTokenizer()
    prompt_ids = tokenizer.encode_prompt("30000000 + 40000000 = <ans>")
    token_ids = prompt_ids + [tokenizer.token_to_id["7"], tokenizer.answer_id]

    position_roles = tokenizer.position_role_ids_for_token_ids(token_ids)

    assert len(position_roles) == len(token_ids)
    assert position_roles[-2:] == [0, 0]


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


def test_absolute_forward_helper_ignores_digit_role_ids() -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
        position_encoding=POSITION_ENCODING_ABSOLUTE,
    )
    model = SmallCausalTransformer(config).eval()
    input_token_ids, input_position_ids = tokenizer.encode_prompt_with_roles(
        "30000000 + 40000000 = <ans>"
    )
    input_ids = torch.tensor([input_token_ids], dtype=torch.long)
    digit_role_ids = torch.tensor([input_position_ids], dtype=torch.long)

    with torch.no_grad():
        direct_logits = model(input_ids)
        helper_logits = _forward_model(
            model,
            input_ids,
            position_ids=digit_role_ids,
        )

    assert torch.allclose(helper_logits, direct_logits)


def test_small_transformer_digit_roles_forward_shape() -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
        position_encoding=POSITION_ENCODING_DIGIT_ROLES,
    )
    model = SmallCausalTransformer(config).eval()
    input_token_ids, input_position_ids = tokenizer.encode_prompt_with_roles(
        "30000000 + 40000000 = <ans>"
    )
    input_ids = torch.tensor([input_token_ids], dtype=torch.long)
    position_ids = torch.tensor([input_position_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, position_ids)

    assert logits.shape == (1, input_ids.shape[1], tokenizer.vocab_size)


def test_legacy_checkpoint_model_config_defaults_to_absolute_positions() -> None:
    config = _model_config_from_payload(
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

    assert config.position_encoding == "absolute"
    assert config.position_vocab_size == 9


def test_evaluator_bucket_row_keeps_summary_math() -> None:
    stats = BucketStats(
        evaluated_count=4, perfect_count=3, canonical_prediction_count=2
    )

    row = bucket_row(name="binary::small-small::+", stats=stats, available_count=10)

    assert row["missed_count"] == 1
    assert row["accuracy"] == 0.75
    assert row["canonical_prediction_rate"] == 0.5
    assert row["available_count"] == 10
