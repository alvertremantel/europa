# First New-Scheme Training Operation: miller-2.0 and urey-2.0

This document is written for an autonomous coding/training agent running on a normal desktop checkout of Europa ALM-IS. It defines the first two models to train with the new opt-in example/curriculum/scratchpad training scheme.

## Goals

Train two matched experimental checkpoints on the same freshly generated stratified arithmetic dataset:

| Model | Scale target | Required structure | Proposed config | Expected params |
|---|---:|---|---|---:|
| `miller-2.0` | 25K-100K | 2 heads / 4 layers | `d_model=32`, `mlp_hidden=64`, `seq_len=64` | ~37,056 |
| `urey-2.0` | 500K-2M | 4 heads / 8 layers | `d_model=128`, `mlp_hidden=512`, `seq_len=64` | ~1,598,080 |

Interpretation intent:

- `miller-2.0` is the tiny curriculum-only control. It should test whether line-aware curriculum training improves a very small model without changing the answer format.
- `urey-2.0` is the larger curriculum-plus-light-scratchpad run. It should test whether extra capacity can exploit compact intermediate supervision on multiplication and parentheses examples.

## Environment setup

From a clean checkout:

```bash
uv sync
uv run ruff check .
```

Use the project CLI entrypoints only. Do not run nonexistent top-level scripts such as `python generate.py`.

## Dataset generation and validation

Use one shared dataset so model differences are attributable to training configuration rather than data changes.

```bash
mkdir -p data/training
uv run generate --seed 20260515 --output-dir data/training/europa-2.0-curriculum
```

The generator validates by default. Do **not** pass `--no-validate` for this operation.

After generation, confirm the expected files exist:

```bash
ls data/training/europa-2.0-curriculum
```

Expected files:

- `train.txt`
- `val.txt`
- `test.txt`
- `meta.json`

Run a metadata/scratchpad smoke check before training:

```bash
uv run python - <<'PY'
from pathlib import Path
from trainer.data import ArithmeticTokenizer, ExampleSequenceDataset, load_examples, transform_examples, vocab_for_training_format
from trainer.curriculum import build_balanced_example_sample, count_curriculum_groups

root = Path("data/training/europa-2.0-curriculum")
train = load_examples(root / "train.txt", include_metadata=True)
val = load_examples(root / "val.txt", include_metadata=True)
print("train_examples", len(train))
print("val_examples", len(val))
print("train_curriculum_groups", count_curriculum_groups(train))

scratch = transform_examples(train[:256], training_format="light_scratchpad")
tok = ArithmeticTokenizer(vocab_for_training_format("light_scratchpad"))
ds = ExampleSequenceDataset(scratch, tok, 64, skip_overlong=False)
balanced = build_balanced_example_sample(val, group_by="kind", sample_size_per_group=1, seed=20260515)
print("scratchpad_smoke_examples", len(ds))
print("balanced_val_smoke_examples", len(balanced))
PY
```

If this fails because examples exceed sequence length, stop and report the failure rather than silently changing the run design.

## Run 1: miller-2.0

Purpose: very small example-mode curriculum-only baseline with the legacy final-answer target format.

```bash
mkdir -p artifacts/models/miller-2.0
cat > artifacts/models/miller-2.0/train.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

uv run train train \
  --data-dir data/training/europa-2.0-curriculum \
  --output-dir runs/miller-2.0 \
  --training-mode examples \
  --training-format final_only \
  --curriculum-name baseline_mixed_v1 \
  --balanced-val \
  --balanced-val-group-by kind \
  --balanced-val-sample-size-per-group 4 \
  --balanced-val-seed 20260515 \
  --sequence-length 64 \
  --batch-size 128 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-batches 50 \
  --exact-match-samples 256 \
  --max-new-tokens 24 \
  --seed 20260515 \
  --device cuda \
  --d-model 32 \
  --n-heads 2 \
  --n-layers 4 \
  --mlp-hidden 64 \
  --dropout 0.1
SH
chmod +x artifacts/models/miller-2.0/train.sh
artifacts/models/miller-2.0/train.sh 2>&1 | tee artifacts/models/miller-2.0/train.log
```

If CUDA is unavailable, stop and mention it in the report. Do not switch to CPU for the full run unless explicitly instructed by the human operator.

## Run 2: urey-2.0

Purpose: larger example-mode model with the same curriculum plus compact scratchpad supervision for multiplication and parentheses examples.

```bash
mkdir -p artifacts/models/urey-2.0
cat > artifacts/models/urey-2.0/train.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

uv run train train \
  --data-dir data/training/europa-2.0-curriculum \
  --output-dir runs/urey-2.0 \
  --training-mode examples \
  --training-format light_scratchpad \
  --curriculum-name baseline_mixed_v1 \
  --balanced-val \
  --balanced-val-group-by kind \
  --balanced-val-sample-size-per-group 4 \
  --balanced-val-seed 20260515 \
  --sequence-length 64 \
  --batch-size 128 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-batches 50 \
  --exact-match-samples 256 \
  --max-new-tokens 48 \
  --seed 20260515 \
  --device cuda \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 8 \
  --mlp-hidden 512 \
  --dropout 0.1
SH
chmod +x artifacts/models/urey-2.0/train.sh
artifacts/models/urey-2.0/train.sh 2>&1 | tee artifacts/models/urey-2.0/train.log
```

## Evaluation

Evaluate both best checkpoints against the shared dataset:

```bash
uv run evaluate \
  --checkpoint runs/miller-2.0/checkpoint-best.pt \
  --data-dir data/training/europa-2.0-curriculum \
  2>&1 | tee artifacts/models/miller-2.0/evaluate.log

uv run evaluate \
  --checkpoint runs/urey-2.0/checkpoint-best.pt \
  --data-dir data/training/europa-2.0-curriculum \
  --max-new-tokens 48 \
  2>&1 | tee artifacts/models/urey-2.0/evaluate.log
```

Run a few manual prediction probes and save them:

```bash
cat > artifacts/models/miller-2.0/prediction-probes.txt <<'EOF'
03000000 + 04000000 = <ans>
06000000 * 07000000 = <ans>
( 02000000 + 03000000 ) * 04000000 = <ans>
05000000 + (-02000000) = <ans>
EOF

cat artifacts/models/miller-2.0/prediction-probes.txt | while read -r prompt; do
  printf '\nPROMPT: %s\n' "$prompt"
  uv run train predict --checkpoint runs/miller-2.0/checkpoint-best.pt --prompt "$prompt" --device cuda --max-new-tokens 24
done | tee artifacts/models/miller-2.0/prediction-probes.log

cp artifacts/models/miller-2.0/prediction-probes.txt artifacts/models/urey-2.0/prediction-probes.txt
cat artifacts/models/urey-2.0/prediction-probes.txt | while read -r prompt; do
  printf '\nPROMPT: %s\n' "$prompt"
  uv run train predict --checkpoint runs/urey-2.0/checkpoint-best.pt --prompt "$prompt" --device cuda --max-new-tokens 48
done | tee artifacts/models/urey-2.0/prediction-probes.log
```

## Required after-training report

After both runs and evaluations finish, create:

```text
artifacts/trainings/after-training-report-miller-2.0-urey-2.0.md
```

The report should be factual and self-contained. Include:

1. **Environment**
   - Git commit hash.
   - GPU name and CUDA availability.
   - `uv run ruff check .` result.
2. **Dataset**
   - Exact dataset command and seed.
   - Counts from `meta.json` and the smoke-check output.
   - Any validation or sequence-length problems.
3. **Model configs**
   - Full CLI command for each model.
   - Parameter count printed by training.
   - Confirm that `miller-2.0` is between 25K and 100K parameters.
   - Confirm that `urey-2.0` is between 500K and 2M parameters.
4. **Training curves**
   - Final and best `train_loss`, `val_loss`, `exact_match`, `balanced_val_loss`, and `balanced_exact_match` from `history.json`.
   - Note curriculum stages and per-stage sample counts if visible in logs/history.
   - Note `scratchpad_fraction` for `urey-2.0`.
5. **Evaluation summary**
   - Overall evaluation metrics for both models.
   - Per-category and weakest-kind findings from evaluator outputs.
   - Compare final-answer behavior on addition/subtraction, multiplication/division, parentheses/three-input, and negative-input strata.
6. **Qualitative probes**
   - Paste prediction probe outputs.
   - For `urey-2.0`, explicitly say whether generated outputs contain scratchpad markers and whether the final extracted answer is clean.
7. **Operational notes**
   - Wall-clock training time.
   - Any interruptions, resumes, CUDA memory issues, or manual deviations from this document.
8. **Next-run recommendations**
   - Suggest concrete changes only from observed evidence: epochs, width/depth, batch size, curriculum preset, scratchpad scope, or validation sample sizing.

Do not include speculative mechanistic-interpretability claims in the report. If there are interesting behavioral asymmetries, present them as observations with file paths and metric evidence.
