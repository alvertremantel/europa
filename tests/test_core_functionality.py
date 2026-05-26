from __future__ import annotations

from collections import Counter
import tomllib

import pytest
import torch

from eis.data.config import Config, SPLITS
from eis.eval.core import BucketStats, bucket_row
from eis.data.core import (
    format_answer,
    format_signed_number,
    format_unsigned_number,
    generate_dataset,
    is_canonical_answer,
    parse_answer,
    parse_signed_number,
    validate_line,
)
from eis.config import ModelConfig
from eis.train.curriculum import select_curriculum_stage
from eis.train.data import (
    ArithmeticTokenizer,
    POSITION_ENCODING_FIXED_MEANING,
    vocab_for_training_format,
)
from eis.train.datasets import ExampleSequenceDataset, TokenBlockDataset
from eis.train.examples import ArithmeticExample
from eis.train.semantics.fixed_meaning import (
    FIXED_MEANING_DIGIT_PLACE_DIMENSION,
    build_fixed_meaning_token_table,
    fixed_meaning_width,
)
from eis.train.formatting import final_answer_from_line, format_training_line
from eis.train.inference import (
    generate_completion,
    generate_completions,
    sample_exact_match_probe,
)
from eis.train.model import SmallCausalTransformer
from eis.train.training.checkpointing import _model_config_from_payload


def test_generator_validates_canonical_lines() -> None:
    line = "<do> <calc> {300000} + {400000} = <ans> {700000}"
    parsed = validate_line(line)
    assert parsed.category == "arithmetic"
    assert parsed.kind == "arithmetic::small-small::+"
    assert parsed.answer == 7
    assert format_unsigned_number(6) == "600000"
    assert format_signed_number(6) == "{600000}"
    assert format_signed_number(-6) == "(600000)"
    assert format_signed_number(0) == "{000000}"
    assert parse_signed_number("{600000}") == 6
    assert parse_signed_number("(600000)") == -6
    with pytest.raises(ValueError):
        parse_signed_number("600000")
    with pytest.raises(ValueError):
        parse_signed_number("(000000)")


def test_generated_redux_dataset_balances_comparison_splits_and_records_data_mix(
    tmp_path,
) -> None:
    output_dir = tmp_path / "dataset"

    generate_dataset(Config(output_dir=str(output_dir), validate=True))

    comparison_counts_by_split_and_kind: dict[str, dict[str, Counter[bool]]] = {
        split: {} for split in SPLITS
    }
    for split in SPLITS:
        lines = (output_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for line in lines:
            sample = validate_line(line)
            if sample.category != "comparison":
                continue
            comparison_counts_by_split_and_kind[split].setdefault(
                sample.kind, Counter()
            )[bool(sample.answer)] += 1

    for split, by_kind in comparison_counts_by_split_and_kind.items():
        assert by_kind
        for counts in by_kind.values():
            assert counts[True] == counts[False], split

    metadata = tomllib.loads((output_dir / "meta.toml").read_text(encoding="utf-8"))
    train_category_counts = metadata["data_mix"]["train_category_counts"]
    assert train_category_counts == metadata["split_category_counts"]["train"]

    computation_count = (
        train_category_counts["arithmetic"] + train_category_counts["negative_input"]
    )
    assert metadata["data_mix"]["computation_train_count"] == computation_count
    assert (
        metadata["data_mix"]["comparison_train_count"]
        == train_category_counts["comparison"]
    )
    assert metadata["data_mix"]["comparison_to_computation_ratio"] == pytest.approx(
        train_category_counts["comparison"] / computation_count
    )


def test_redux_answers_parse_and_validate_canonicality() -> None:
    assert format_answer(parse_answer("{000000}")) == "{000000}"
    assert format_answer(parse_answer("(100000)")) == "(100000)"
    assert format_answer(parse_answer("true")) == "true"
    assert format_answer(parse_answer("false")) == "false"
    assert is_canonical_answer("{600000}") is True
    assert is_canonical_answer("True") is False
    assert is_canonical_answer("600000") is False


def test_redux_parser_rejects_legacy_protocol_forms() -> None:
    with pytest.raises(ValueError):
        validate_line("<do> <calc> 30000000 + 40000000 = 70000000")
    with pytest.raises(ValueError):
        validate_line("<do> <calc> {300000} + {400000} = {700000}")


def test_training_tokenizer_round_trips_redux_lines_and_prompts() -> None:
    line = "<do> <calc> {300000} + {400000} = <ans> {700000}"
    transformed, applied_format = format_training_line(line, "final_only")
    tokenizer = ArithmeticTokenizer(vocab_for_training_format("final_only"))
    assert applied_format == "final_only"
    assert transformed == line
    assert final_answer_from_line(transformed) == "{700000}"
    assert tokenizer.decode(tokenizer.encode_line(transformed)) == transformed
    assert tokenizer.decode(tokenizer.encode_prompt("{300000} + {400000} =")) == (
        "<do> <calc> {300000} + {400000} = <ans>"
    )
    assert "<sep>" not in [
        tokenizer.id_to_token[token_id] for token_id in tokenizer.encode_line(line)
    ]
    with pytest.raises(ValueError, match="final_only"):
        vocab_for_training_format("light_scratchpad")


def test_training_tokenizer_preserves_negative_digit_structure() -> None:
    tokenizer = ArithmeticTokenizer()
    token_ids = tokenizer.encode_prompt("<do> <calc> (600000) + {300000} =")
    tokens = [tokenizer.id_to_token[token_id] for token_id in token_ids]

    assert tokens[:2] == ["<do>", "<calc>"]
    assert tokens[2:10] == ["(", "6", "0", "0", "0", "0", "0", ")"]


def test_small_transformer_fixed_meaning_forward_shape() -> None:
    tokenizer = ArithmeticTokenizer()
    d_model = fixed_meaning_width()
    assert d_model == 16
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
    input_token_ids = tokenizer.encode_prompt("<do> <calc> {300000} + {400000} =")
    input_ids = torch.tensor([input_token_ids], dtype=torch.long)
    input_digit_places = torch.tensor(
        [tokenizer.fixed_meaning_digit_place_values_for_token_ids(input_token_ids)],
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(input_ids, input_digit_places)

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
    token_ids = tokenizer.encode_prompt("<do> <calc> {300000} + {400000} =")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    input_digit_places = torch.tensor(
        [tokenizer.fixed_meaning_digit_place_values_for_token_ids(token_ids)],
        dtype=torch.float32,
    )

    with torch.no_grad():
        embeddings = model.token_embedding(input_ids, input_digit_places)

    digit_place_values = [
        embeddings[0, index, FIXED_MEANING_DIGIT_PLACE_DIMENSION].item()
        for index, token_id in enumerate(token_ids)
        if tokenizer.id_to_token[token_id] in set("0123456789")
    ]
    assert digit_place_values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2)


def test_incremental_digit_place_matches_full_recompute() -> None:
    tokenizer = ArithmeticTokenizer()
    prefix = [tokenizer.token_to_id[token] for token in "12"]
    next_ids = [tokenizer.token_to_id[token] for token in "345"]

    token_ids = list(prefix)
    incremental_values: list[float] = []
    for next_id in next_ids:
        incremental_values.append(
            tokenizer.fixed_meaning_digit_place_value_after_prefix(token_ids, next_id)
        )
        token_ids.append(next_id)

    full_values = tokenizer.fixed_meaning_digit_place_values_for_token_ids(token_ids)
    assert incremental_values == full_values[-len(next_ids) :]


def test_batched_generation_matches_single_prompt_generation() -> None:
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
    prompts = [
        "<do> <calc> {300000} + {400000} = <ans>",
        "<do> <calc> {500000} - {200000} = <ans>",
    ]

    single = [
        generate_completion(model, tokenizer, prompt, 3, torch.device("cpu"))
        for prompt in prompts
    ]
    batched = generate_completions(model, tokenizer, prompts, 3, torch.device("cpu"))

    assert batched == single


def test_fixed_meaning_datasets_emit_digit_place_batches() -> None:
    tokenizer = ArithmeticTokenizer()
    line = "<do> <calc> {300000} + {400000} = <ans> {700000}"
    token_ids = tokenizer.encode_line(line)
    stream_token_ids = token_ids * 2
    stream_digit_places = tokenizer.fixed_meaning_digit_place_values_for_token_ids(
        stream_token_ids
    )
    block_dataset = TokenBlockDataset(
        stream_token_ids,
        stream_digit_places,
        sequence_length=8,
    )
    block_item = block_dataset[0]
    assert len(block_item) == 3

    example = ArithmeticExample(
        line=line,
        prompt="<do> <calc> {300000} + {400000} = <ans>",
        answer="{700000}",
    )
    example_dataset = ExampleSequenceDataset(
        [example],
        tokenizer,
        sequence_length=32,
        position_encoding=POSITION_ENCODING_FIXED_MEANING,
    )
    example_item = example_dataset[0]
    assert len(example_item) == 4


def test_token_stream_digit_places_are_computed_before_block_slicing() -> None:
    tokenizer = ArithmeticTokenizer()
    line = "<do> <calc> {123450} + {876540} = <ans> {999990}"
    token_ids = tokenizer.encode_line(line)
    digit_place_values = tokenizer.fixed_meaning_digit_place_values_for_token_ids(
        token_ids
    )
    dataset = TokenBlockDataset(token_ids, digit_place_values, sequence_length=16)

    input_ids, input_digit_places, _ = dataset[1]
    input_tokens = [tokenizer.id_to_token[token_id] for token_id in input_ids.tolist()]

    assert input_tokens[:5] == ["4", "0", "}", "=", "<ans>"]
    assert input_digit_places[:5].tolist() == pytest.approx([0.5, 0.6, 0.0, 0.0, 0.0])


def test_fixed_meaning_rejects_d_model_mismatch() -> None:
    tokenizer = ArithmeticTokenizer()

    with pytest.raises(ValueError, match="fixed_meaning position_encoding"):
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            sequence_length=32,
            d_model=fixed_meaning_width() + 1,
            n_heads=1,
            n_layers=1,
            mlp_hidden=32,
            dropout=0.0,
            position_encoding=POSITION_ENCODING_FIXED_MEANING,
        )


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
        f"<do> <calc> {{{index:06d}}} + {{000000}} = <ans> {{{index:06d}}}"
        for index in range(100)
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
    lines = [
        f"<do> <calc> {{{index:06d}}} + {{000000}} = <ans> {{{index:06d}}}"
        for index in range(12)
    ]
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
    row = bucket_row(name="arithmetic::small-small::+", stats=stats, available_count=10)
    assert row["missed_count"] == 1
    assert row["accuracy"] == 0.75
    assert row["canonical_prediction_rate"] == 0.5
    assert row["available_count"] == 10
