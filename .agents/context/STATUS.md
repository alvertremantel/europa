# Status

## Summary

- `generate.py` now targets a stratified dataset format spanning binary, three-input, parenthesized, and one-negative-input arithmetic.
- `train.py` still supports `train` and `predict`, and its tokenizer now includes `(` and `)` in addition to the arithmetic operators and `=`.
- `evaluate.py` now provides sampled per-stratum exact-match evaluation with per-category and per-kind breakdowns plus report files.
- `scripts/analyze_strata_eval.py` now analyzes per-kind evaluation CSVs and runs permutation-based tests for structural predictors of model accuracy.
- `scripts/check_length_safety.py` now verifies that dataset prompt/answer lengths stay within checkpoint context and generation limits, and that final/intermediate values stay within the fixed numeric width.
- The project remains configured under `uv` with a local `.venv` targeting Python 3.12.
- PyTorch is installed locally with CUDA support and detects the RTX 4060.
- The revised generator implementation is present, but existing checked-in dataset directories may still reflect older generated corpora until regenerated.
- A long training run has now been tested on the extended dataset and completed successfully.

## Current Environment

- Python target: `3.12`
- Environment manager: `uv`
- Virtual environment path: `.venv/`
- Package lock file: `uv.lock`
- Installed Torch build: CUDA-enabled `torch 2.11.0+cu128`

## Dataset Status

- Current generator output format: `<expression> = <ans> <result>`
- Number encoding: 8-digit zero-padded decimal strings, reversed
- Negative number encoding: `(-AAAAAAAA)`
- Bands: `small=0..20`, `medium=21..100`, `large=101..500`
- Binary category: exhaustive `+`, `-`, `*`, `/` with exact integer division only
- Three-input category: sampled `+`, `-`, `*` only
- Parentheses category: sampled `(A op B) op C` and `A op (B op C)` using `+`, `-`, `*`
- Negative-input category: sampled two-input `+`, `-`, `*` with exactly one negative operand
- Default sampled target: `128` train rows plus `16` val and `16` test rows per included sampled kind
- Current verified output size: `670163` rows total (`661651` train, `4256` val, `4256` test)
- Four parenthesized `a - (b * c)` kind/band combinations are skipped because they cannot satisfy the fixed holdout minimum under the non-negative constraint.

## Recent Training Result

- The user ran an extended training run on the extended dataset for `100` epochs.
- Final reported metrics for that run:
  - train loss: `0.4887`
  - val loss: `0.576`
  - exact match: `0.925`
- This confirms the current extended dataset and training script are at least workable together for a substantial training run.

## Training Script Status

- `train.py` supports:
  - `train`
  - `predict`
- `evaluate.py` supports sampled evaluation from a saved checkpoint across chosen data-file pools, with fixed-size per-kind selection.
- `scripts/analyze_strata_eval.py` supports post-hoc category/op/band summaries and permutation tests over evaluation kinds.
- `scripts/check_length_safety.py` supports post-hoc verification that no evaluated kind requires more prompt or answer tokens than the model configuration allows, or more numeric digits than the format permits.
- The model is a small decoder-only transformer trained with next-token prediction.
- The tokenizer is arithmetic-specific and fixed, not learned from corpus statistics.
- `<ans>` prompt splitting behavior is unchanged.

## Default Training Configuration

- Sequence length: `64`
- Batch size: `128`
- Epochs: `5`
- Learning rate: `3e-4`
- Weight decay: `0.1`
- Model width: `256`
- Attention heads: `4`
- Transformer layers: `6`
- MLP hidden size: `1024`
- Dropout: `0.1`
- Parameter count: about `4.76M`

## Current Training Behavior

- Training reads symbolic lines from `train.txt` and `val.txt`.
- The script tokenizes fixed-width digits plus separators and parentheses into a single token stream.
- Validation uses:
  - language-model loss on a limited number of validation batches
  - sampled exact-match generation on validation prompts
- `checkpoint-best.pt` is chosen by sampled validation exact-match.

## Current Artifacts Produced By Training

When training is run with `--output-dir runs/arithmetic-small`, the main outputs are:

- `runs/arithmetic-small/checkpoint-best.pt`
- `runs/arithmetic-small/checkpoint-last.pt`
- `runs/arithmetic-small/history.json`
- `runs/arithmetic-small/checkpoint-best-strata-eval.summary.json`
- `runs/arithmetic-small/checkpoint-best-strata-eval.kinds.csv`
- `runs/arithmetic-small/checkpoint-best-strata-eval.errors.jsonl`
- `runs/arithmetic-small/checkpoint-best-strata-eval.kinds.analysis.json`
- `runs/arithmetic-small/checkpoint-best-strata-eval.kinds.length-safety.json`

## Important Limitations

- Resume training is not implemented yet.
- Optimizer state is not saved for restart.
- Evaluation is not broken out per operation yet.
- Validation exact-match is still sampled, not full-split.
- Checkpoints remain tightly coupled to the tokenizer and model architecture.

## Immediate Next Useful Changes

1. Regenerate canonical dataset directories with the revised generator.
2. Refactor `train.py` into modules/packages, matching the new generator structure.
3. Add richer aggregated reporting, such as per-operation and wildcard-vs-non-wildcard summaries.
4. Decide whether sampled per-stratum evaluation should remain a separate top-level script or move under a refactored training/evaluation package.
