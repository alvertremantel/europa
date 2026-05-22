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
from eur_ts.trainer.curriculum import select_curriculum_stage
from eur_ts.trainer.data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_FIXED_MEANING,
    vocab_for_training_format,
)
from eur_ts.trainer.datasets import ExampleSequenceDataset, TokenBlockDataset
from eur_ts.trainer.examples import ArithmeticExample
from eur_ts.trainer.fixed_meaning import (
    FIXED_MEANING_DIGIT_PLACE_DIMENSION,
    build_fixed_meaning_token_table,
    fixed_meaning_width,
)
from eur_ts.trainer.formatting import final_answer_from_line, format_training_line
from eur_ts.trainer.inference import sample_exact_match_probe
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


def test_training_tokenizer_preserves_negative_digit_structure() -> None:
    tokenizer = ArithmeticTokenizer()
    token_ids = tokenizer.encode_prompt("<do> <calc> (-60000000) + 30000000 =")
    tokens = [tokenizer.id_to_token[token_id] for token_id in token_ids]

    assert tokens[:2] == ["<do>", "<calc>"]
    assert tokens[2:12] == ["(", "-", "6", "0", "0", "0", "0", "0", "0", "0"]


def test_small_transformer_fixed_meaning_forward_shape() -> None:
    tokenizer = ArithmeticTokenizer()
    d_model = fixed_meaning_width()
    assert d_model == 12
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=d_model,
        n_heads=1,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
        position_encoding=POSITION_ENCODING_FIXED_MEANING,
    )
    model = SmallCausalTransformer(config, tokenizer=tokenizer).eval()
    input_token_ids = tokenizer.encode_prompt("<do> <calc> 30000000 + 40000000 =")
    input_ids = torch.tensor([input_token_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], tokenizer.vocab_size)
    assert model.token_embedding.weight.requires_grad is False
    expected_table = build_fixed_meaning_token_table(tokenizer.id_to_token, d_model)
    assert torch.allclose(model.token_embedding.weight, expected_table)
    assert model.position_embedding is None
    assert model.lm_head.weight.requires_grad is True


def test_fixed_meaning_digit_place_encoding_restarts_for_each_number() -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=fixed_meaning_width(),
        n_heads=1,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
        position_encoding=POSITION_ENCODING_FIXED_MEANING,
    )
    model = SmallCausalTransformer(config, tokenizer=tokenizer).eval()
    token_ids = tokenizer.encode_prompt("<do> <calc> 30000000 + 40000000 =")
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        embeddings = model.token_embedding(input_ids)

    digit_place_values = [
        embeddings[0, index, FIXED_MEANING_DIGIT_PLACE_DIMENSION].item()
        for index, token_id in enumerate(token_ids)
        if tokenizer.id_to_token[token_id] in set("0123456789")
    ]
    assert digit_place_values == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 2
    )


def test_fixed_meaning_datasets_emit_token_only_batches() -> None:
    tokenizer = ArithmeticTokenizer()
    line = "<do> <calc> 30000000 + 40000000 = 70000000"
    token_ids = tokenizer.encode_line(line)
    block_dataset = TokenBlockDataset(token_ids * 2, sequence_length=8)
    block_item = block_dataset[0]
    assert len(block_item) == 2

    example = ArithmeticExample(
        line=line,
        prompt="<do> <calc> 30000000 + 40000000 =",
        answer="70000000",
    )
    example_dataset = ExampleSequenceDataset(
        [example],
        tokenizer,
        sequence_length=32,
        position_encoding=POSITION_ENCODING_FIXED_MEANING,
    )
    example_item = example_dataset[0]
    assert len(example_item) == 3


def test_fixed_meaning_rejects_d_model_mismatch() -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=32,
        d_model=fixed_meaning_width() + 1,
        n_heads=1,
        n_layers=1,
        mlp_hidden=32,
        dropout=0.0,
        position_encoding=POSITION_ENCODING_FIXED_MEANING,
    )

    with pytest.raises(ValueError, match="fixed_meaning d_model"):
        SmallCausalTransformer(config, tokenizer=tokenizer)


def test_type_place_position_encoding_is_rejected() -> None:
    tokenizer = ArithmeticTokenizer()

    with pytest.raises(ValueError, match="fixed_meaning"):
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            sequence_length=32,
            d_model=fixed_meaning_width(),
            n_heads=1,
            n_layers=1,
            mlp_hidden=32,
            dropout=0.0,
            position_encoding="type_place",
        )


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


def test_sample_exact_match_probe_is_deterministic_and_unique(tmp_path) -> None:
    path = tmp_path / "val.txt"
    lines = [
        f"<do> <calc> {index:08d} + 00000000 = {index:08d}" for index in range(100)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample_a = sample_exact_match_probe(path, seed=7)
    sample_b = sample_exact_match_probe(path, seed=7)

    assert len(sample_a) == 50
    assert sample_a == sample_b
    assert len(set(sample_a)) == 50
    assert set(sample_a).issubset(set(lines))


def test_sample_exact_match_probe_uses_all_examples_when_split_is_small(
    tmp_path,
) -> None:
    path = tmp_path / "val.txt"
    lines = [f"<do> <calc> {index:08d} + 00000000 = {index:08d}" for index in range(12)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample = sample_exact_match_probe(path, seed=11)

    assert sample == lines


def test_curriculum_stage_selection_scales_across_total_epochs() -> None:
    assert (
        select_curriculum_stage("baseline_mixed_v1", 1, target_epoch=6)[1].name
        == "foundations"
    )
    assert (
        select_curriculum_stage("baseline_mixed_v1", 3, target_epoch=6)[1].name
        == "mul_div_focus"
    )
    assert (
        select_curriculum_stage("baseline_mixed_v1", 5, target_epoch=6)[1].name
        == "compositional_mix"
    )

    assert (
        select_curriculum_stage("baseline_mixed_v1", 1, target_epoch=100)[1].name
        == "foundations"
    )
    assert (
        select_curriculum_stage("baseline_mixed_v1", 35, target_epoch=100)[1].name
        == "mul_div_focus"
    )
    assert (
        select_curriculum_stage("baseline_mixed_v1", 68, target_epoch=100)[1].name
        == "compositional_mix"
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
