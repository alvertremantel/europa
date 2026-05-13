# Europa ALM-IS

Europa Arithmetic Language Model Interpretability Suite — synthetic arithmetic dataset generation and small language-model training.

The project trains a narrow decoder-only language model on symbolic arithmetic text. The current dataset format is fixed-width, reversed-digit, and infix.

Example prompts:

```text
03000000 + 03000000 = <ans>
( 03000000 + 04000000 ) * 05000000 = <ans>
(-03000000) + 05000000 = <ans>
03000000 * 04000000 * 02000000 = <ans>
```

Example answers:

```text
06000000
30000000
01000000
(-08000000)
```

## Project Layout

- `generate.py`: generator CLI entrypoint
- `generator/`: stratified generator package
- `.agents/plans/DATA_PLAN.md`: dataset specification
- `train.py`: training and inference entrypoint
- `evaluate.py`: per-stratum sampler and exact-match evaluator with per-kind reports
- `scripts/analyze_strata_eval.py`: statistical analysis for per-kind evaluation reports
- `scripts/check_length_safety.py`: verifies dataset token lengths and answer widths stay within model limits
- `.agents/plans/TRAINING_PLAN.md`: compact training hand-off notes
- `data-1m/`, `data-2m/`, `data-5m/`: existing generated datasets
- `runs/`: training outputs, checkpoints, and metrics

## Dataset Format

Each line is symbolic arithmetic text:

```text
<expression> = <ans> <result>
```

Supported expression forms:

```text
03000000 / 51000000 = <ans> 20000000
00000001 / 02000000 = <ans> 50000000
03000000 * 04000000 * 02000000 = <ans> 42000000
( 03000000 + 04000000 ) * 05000000 = <ans> 53000000
05000000 - ( 03000000 * 02000000 ) = <ans> 04000000
(-03000000) + 05000000 = <ans> 02000000
```

Rules:

1. Every space is a single token separator.
2. Every number is encoded as an 8-digit zero-padded decimal string and then reversed.
3. Negative numbers are written as `(-AAAAAAAA)`.
4. Binary division only includes exact integer quotients with a non-zero divisor.
5. Non-negative subtraction categories only include cases whose intermediate and final values stay non-negative.
6. Three-input problems currently use `+`, `-`, and `*`, but not `/`.
7. Parenthesized problems currently use `+`, `-`, and `*`, with exactly two terms inside parentheses and one outside.
8. `<ans>` marks the boundary between prompt and answer.

## Generator Behavior

`generate.py` now emits a stratified, deduplicated corpus across four categories:

1. binary non-negative problems
2. three-input problems
3. parenthesized three-term problems
4. two-input problems with exactly one negative operand

Default command:

```bash
uv run python generate.py --output-dir data
```

Important details:

1. Non-overlapping bands are `small=0..20`, `medium=21..100`, and `large=101..500`.
2. Binary kinds are generated exhaustively.
3. Three-input, parenthesized, and negative-input kinds are sampled to `128` train rows plus `16` validation and `16` test rows per included kind.
4. Samples are deduplicated globally by final serialized sample text.
5. The cross-band patterns are marked as wildcard kinds in `meta.json`.
6. The generator now lives under `generator/`, while `generate.py` remains the top-level CLI entrypoint.
7. The current generator writes `670163` unique rows: `661651` train, `4256` val, `4256` test.
8. Four parenthesized kinds of the form `a - (b * c)` at larger band combinations are skipped because they do not supply enough valid non-negative examples for the fixed holdout sizes.

## Current Model And Tokenization

The training script uses a tiny arithmetic-specific tokenizer rather than a general-purpose text tokenizer.

Vocabulary categories:

- control tokens: `<bos>`, `<eos>`, `<sep>`, `<ans>`
- word tokens: `undefined`, `remainder`
- character tokens: `+`, `-`, `*`, `/`, `=`, `(`, `)`, `0`-`9`

Default model shape:

- decoder-only transformer
- sequence length: `64`
- width: `256`
- heads: `4`
- layers: `6`
- MLP hidden size: `1024`
- parameter count: about `4.76M`

## Environment Setup

This repo is configured for `uv` and Python `3.12`.

Create or refresh the environment:

```bash
uv venv --python 3.12 .venv
uv sync --all-groups
```

Run commands through `uv run` so they use the local project environment.

## Training Workflow

### 1. Generate Or Regenerate Data

```bash
uv run python generate.py --output-dir data-revised
```

### 2. Train A Model

```bash
uv run python train.py train --data-dir data-revised --output-dir runs/arithmetic-small --epochs 5
```

### 3. Query The Trained Model

```bash
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
```

### 4. Evaluate By Problem Stratum

```bash
uv run python evaluate.py --checkpoint runs/arithmetic-small/checkpoint-best.pt --data-dir data-revised --sample-size-per-kind 50
```

The evaluator:

1. Draws a fixed-size deterministic sample from each realized problem kind across the chosen pool of dataset files.
2. Reconstructs each sample's canonical category and kind from the dataset line itself.
3. Writes report files next to the checkpoint:
   - `checkpoint-best-strata-eval.summary.json`
   - `checkpoint-best-strata-eval.kinds.csv`
   - `checkpoint-best-strata-eval.errors.jsonl`
4. Reports `perfect_count`, `missed_count`, and `accuracy` for each problem kind, plus generator-skipped kinds from `meta.json` when available.

### 5. Analyze Which Problem Characteristics Predict Errors

```bash
uv run python scripts/analyze_strata_eval.py runs/arithmetic-small/checkpoint-best-strata-eval.kinds.csv
```

The analysis script:

1. Reads the per-kind evaluation CSV produced by `evaluate.py`.
2. Aggregates accuracy by category, operator family, and band size.
3. Runs permutation-based partial tests on kind-level performance to estimate which structural features predict accuracy.
4. Writes a JSON report next to the input CSV by default.

### 6. Verify Length And Width Safety

```bash
uv run python scripts/check_length_safety.py \
  --checkpoint runs/arithmetic-small/checkpoint-best.pt \
  --data-dir data-revised \
  --kinds-csv runs/arithmetic-small/checkpoint-best-strata-eval.kinds.csv
```

The length-safety script:

1. Scans the actual dataset lines used for evaluation.
2. Computes prompt, answer, and full-line token lengths by kind.
3. Checks them against the checkpoint's `sequence_length` and `max_new_tokens`.
4. Verifies that final answers and sampled intermediates never exceed the fixed 8-digit numeric width.
5. Writes a JSON report next to the kinds CSV by default.

More examples:

```bash
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "03000000 / 51000000 = <ans>"
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "00000001 / 02000000 = <ans>"
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "42010000 * 03000000 = <ans>"
```

## How Inference Works

Training teaches the model to continue a full arithmetic line.

Example training line:

```text
03000000 / 51000000 = <ans> 20000000
```

At inference time you provide the prompt prefix:

```text
03000000 / 51000000 = <ans>
```

The model generates the answer tokens after `<ans>`.

## Existing Datasets

The repo already contains `data-1m`, `data-2m`, and `data-5m`. If those directories were generated before the dataset revision, regenerate them before training with the revised tokenizer assumptions.

## Current Limitations

- resume training is not implemented yet
- optimizer state is not saved for restart
- exact-match evaluation uses a validation sample slice, not the entire validation split
- there is no built-in per-operation report yet
- `train.py` still does not include built-in end-of-run reporting; `evaluate.py` now handles per-stratum sampled evaluation as a separate command
- checkpoints depend on the current tokenizer and model architecture

## Useful Commands

Generate dataset:

```bash
uv run python generate.py --output-dir data
```

Train:

```bash
uv run python train.py train --data-dir data --output-dir runs/arithmetic-small
```

Predict:

```bash
uv run python train.py predict --checkpoint runs/arithmetic-small/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
```

Evaluate:

```bash
uv run python evaluate.py --checkpoint runs/arithmetic-small/checkpoint-best.pt --data-dir data --sample-size-per-kind 50
```
