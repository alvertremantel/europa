"""Compatibility shim: delegates to canonical evaluator modules."""

from eur_ts.evaluator.args import ALL_SPLITS, parse_args
from eur_ts.evaluator.cli import main, print_console_summary
from eur_ts.evaluator.metadata import (
    CATEGORY_ORDER,
    checkpoint_payload,
    expected_available_kind_counts,
    fallback_kind_definitions,
    kind_definitions_from_metadata,
    load_metadata,
    ordered_categories,
    resolve_data_dir,
    resolve_max_new_tokens,
    resolve_output_prefix,
    skipped_kinds_from_metadata,
)
from eur_ts.evaluator.runner import run_evaluation
from eur_ts.evaluator.sampling import (
    collect_selected_examples,
    is_canonical_prediction,
    ordered_selected_kinds,
    print_selection_summary,
    selection_sort_key,
    top_or_bottom_kinds,
    validate_available_counts,
)
from eur_ts.evaluator.writers import (
    write_errors_jsonl,
    write_kind_csv,
    write_summary_json,
)

if __name__ == "__main__":
    main()

__all__ = [
    "ALL_SPLITS",
    "CATEGORY_ORDER",
    "checkpoint_payload",
    "collect_selected_examples",
    "expected_available_kind_counts",
    "fallback_kind_definitions",
    "is_canonical_prediction",
    "kind_definitions_from_metadata",
    "load_metadata",
    "main",
    "ordered_categories",
    "ordered_selected_kinds",
    "parse_args",
    "print_console_summary",
    "print_selection_summary",
    "resolve_data_dir",
    "resolve_max_new_tokens",
    "resolve_output_prefix",
    "run_evaluation",
    "selection_sort_key",
    "skipped_kinds_from_metadata",
    "top_or_bottom_kinds",
    "validate_available_counts",
    "write_errors_jsonl",
    "write_kind_csv",
    "write_summary_json",
]
