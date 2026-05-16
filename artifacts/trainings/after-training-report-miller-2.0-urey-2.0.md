# After-Training Report: `miller-2.0` and `urey-2.0`

## 1. Environment

- Git commit: `3247d4a36f5f4548e685b27a6d14a5703777ac63`
- GPU: `NVIDIA GeForce RTX 4060` (`8188 MiB` reported by `nvidia-smi`)
- CUDA availability: `True` (`torch.cuda.device_count() == 1`)
- `uv run ruff check .`: `All checks passed!`

## 2. Dataset

Exact dataset command:

```bash
mkdir -p data/training
uv run generate --seed 20260515 --output-dir data/training/europa-2.0-curriculum
```

Produced files:

- `data/training/europa-2.0-curriculum/train.txt`
- `data/training/europa-2.0-curriculum/val.txt`
- `data/training/europa-2.0-curriculum/test.txt`
- `data/training/europa-2.0-curriculum/meta.json`

`meta.json` / generation counts:

- `total_unique_rows`: `670163`
- `wildcard_eval_rows`: `5600`
- Split counts:
  - train: `661651`
  - val: `4256`
  - test: `4256`
- Split category counts:
  - train: `binary 630675`, `three_input 3840`, `parentheses 22528`, `negative_input 4608`
  - val: `binary 384`, `three_input 480`, `parentheses 2816`, `negative_input 576`
  - test: `binary 384`, `three_input 480`, `parentheses 2816`, `negative_input 576`

Smoke-check output:

- `train_examples 661651`
- `val_examples 4256`
- `train_curriculum_groups {'easy_binary_add_sub': 376368, 'binary_mul_div': 254307, 'compositional_parentheses_three_input': 26368, 'negative_input': 4608}`
- `scratchpad_smoke_examples 256`
- `balanced_val_smoke_examples 266`

Validation / sequence-length notes:

- Generator validation stayed enabled.
- Scratchpad smoke check passed at `seq_len=64`; no overlength failure occurred.
- During generation, several `parentheses::right::*::*-` strata reported insufficient unique candidates before split assignment (`0` or `3` unique samples for some medium/large cases), but dataset generation completed successfully and the expected output files were written.

## 3. Model configs

### `miller-2.0`

```bash
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
```

- Parameter count: `37056`
- Range check: **within** the required `25K-100K` range.

### `urey-2.0`

```bash
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
```

- Parameter count: `1598080`
- Range check: **within** the required `500K-2M` range.

## 4. Training curves

### `miller-2.0` (`runs/miller-2.0/history.json`)

| Metric | Final | Best |
|---|---:|---:|
| train_loss | `0.582942` (epoch 20) | `0.582942` (epoch 20) |
| val_loss | `3.663539` (epoch 20) | `3.491710` (epoch 15) |
| exact_match | `0.207031` (epoch 20) | `0.207031` (epoch 20) |
| balanced_val_loss | `0.571363` (epoch 20) | `0.570725` (epoch 19) |
| balanced_exact_match | `0.052632` (epoch 20) | `0.052632` (epoch 20) |

### `urey-2.0` (`runs/urey-2.0/history.json`)

| Metric | Final | Best |
|---|---:|---:|
| train_loss | `0.347451` (epoch 20) | `0.347451` (epoch 20) |
| val_loss | `4.749334` (epoch 20) | `4.353463` (epoch 8) |
| exact_match | `0.898438` (epoch 20) | `0.921875` (epoch 19) |
| balanced_val_loss | `0.388430` (epoch 20) | `0.379028` (epoch 8) |
| balanced_exact_match | `0.842105` (epoch 20) | `0.842105` (epoch 20) |

Curriculum stages observed in both histories:

1. `foundations`
   - epoch 1 sample counts: `easy_binary_add_sub 495941`, `binary_mul_div 165710`
2. `mul_div_focus`
   - epoch 2 sample counts: `easy_binary_add_sub 232008`, `binary_mul_div 429643`
3. `compositional_mix`
   - first visible epoch 3 sample counts: `easy_binary_add_sub 132035`, `binary_mul_div 231766`, `compositional_parentheses_three_input 198284`, `negative_input 99566`
   - later epochs stayed close to the configured `0.2 / 0.35 / 0.3 / 0.15` weighting with small epoch-to-epoch variation.

Scratchpad fraction:

- `miller-2.0`: `0.0` throughout
- `urey-2.0`: `0.7218045112781954` throughout recorded history

## 5. Evaluation summary

Evaluation artifacts:

- `runs/miller-2.0/checkpoint-best-strata-eval.summary.json`
- `runs/miller-2.0/checkpoint-best-strata-eval.kinds.csv`
- `runs/miller-2.0/checkpoint-best-strata-eval.errors.jsonl`
- `runs/urey-2.0/checkpoint-best-strata-eval.summary.json`
- `runs/urey-2.0/checkpoint-best-strata-eval.kinds.csv`
- `runs/urey-2.0/checkpoint-best-strata-eval.errors.jsonl`

### Overall

| Model | Overall accuracy | Canonical prediction rate | Evaluated examples |
|---|---:|---:|---:|
| `miller-2.0` | `0.046541` | `0.997368` | `13300` |
| `urey-2.0` | `0.868346` | `1.0` | `13300` |

### Per-category

| Model | Binary | Three-input | Parentheses | Negative-input |
|---|---:|---:|---:|---:|
| `miller-2.0` | `0.1650` | `0.0507` | `0.0288` | `0.0511` |
| `urey-2.0` | `0.9067` | `0.7593` | `0.8600` | `0.9744` |

### Weakest kinds

`miller-2.0` bottom kinds (all `0.0` accuracy in sampled evaluation):

- `binary::large-large::*`
- `binary::medium-large::*`
- `negative_input::large-large::*::neg_left`
- `negative_input::large-large::*::neg_right`
- `negative_input::large-large::+::neg_left`
- `negative_input::large-large::-::neg_right`
- `negative_input::medium-large::*::neg_left`
- `negative_input::medium-large::*::neg_right`
- `negative_input::medium-large::-::neg_right`
- `negative_input::medium-medium::*::neg_left`

`urey-2.0` bottom kinds:

- `parentheses::left::large-large-large::**` — `0.0`
- `parentheses::right::large-large-large::**` — `0.0`
- `three_input::large-large-large::*` — `0.0`
- `three_input::medium-large-large::*` — `0.0`
- `three_input::medium-medium-large::*` — `0.0`
- `parentheses::left::medium-large-large::**` — `0.02`
- `parentheses::right::medium-large-large::**` — `0.02`
- `parentheses::left::medium-medium-large::**` — `0.04`
- `parentheses::right::medium-medium-large::**` — `0.10`
- `parentheses::right::small-large-large::**` — `0.14`

### Comparison by requested strata

Aggregated from evaluator kind rows:

| Model | Binary add/sub | Binary mul/div | Three-input | Parentheses | Negative-input |
|---|---:|---:|---:|---:|---:|
| `miller-2.0` | `0.1517` | `0.1783` | `0.0507` | `0.0288` | `0.0511` |
| `urey-2.0` | `1.0000` | `0.8133` | `0.7593` | `0.8600` | `0.9744` |

Observations:

- `miller-2.0` stayed weak everywhere; even its strongest major slice (`binary_mul_div`) was only `0.1783`.
- `urey-2.0` solved sampled binary addition/subtraction perfectly (`1.0000`) and was also very strong on negative-input examples (`0.9744`).
- `urey-2.0` remained materially weaker on multiplicative compositional structure than on additive structure: `binary_mul_div 0.8133`, `three_input 0.7593`, and several `**` parentheses kinds at or near zero.
- The biggest qualitative gap between the two runs is on compositional strata: `parentheses` improved from `0.0288` to `0.8600`, and `three_input` improved from `0.0507` to `0.7593`.

## 6. Qualitative probes

Probe file:

```text
03000000 + 04000000 = <ans>
06000000 * 07000000 = <ans>
( 02000000 + 03000000 ) * 04000000 = <ans>
05000000 + (-02000000) = <ans>
```

### `miller-2.0` raw outputs

```text
PROMPT: 03000000 + 04000000 = <ans>
02000000

PROMPT: 06000000 * 07000000 = <ans>
00050000

PROMPT: ( 02000000 + 03000000 ) * 04000000 = <ans>
00000000

PROMPT: 05000000 + (-02000000) = <ans>
(-01000000)
```

### `urey-2.0` raw outputs

```text
PROMPT: 03000000 + 04000000 = <ans>
07000000

PROMPT: 06000000 * 07000000 = <ans>
00240000

PROMPT: ( 02000000 + 03000000 ) * 04000000 = <ans>
00020000

PROMPT: 05000000 + (-02000000) = <ans>
03000000
```

Scratchpad-marker note for `urey-2.0`:

- The saved probe outputs did **not** contain visible scratchpad markers.
- The emitted strings were clean single answers rather than mixed scratchpad-plus-answer text.

## 7. Operational notes

- `miller-2.0` wall-clock training time: `2131.922 s` (`00:35:31.922`)
- `urey-2.0` wall-clock training time: `4813.273 s` (`01:20:13.273`)
- `resume_source` was `null` for both runs; no resume was recorded.
- No CUDA-memory failure was recorded in run metadata.
- No manual deviation from the prescribed training commands was made during the training runs themselves.

## 8. Next-run recommendations

1. **Increase `miller-2.0` width/depth rather than rerunning the same tiny config.** Evidence: after 20 epochs it still reached only `0.0465` overall sampled evaluator accuracy and `0.0526` balanced exact match.
2. **For the `urey` line, keep the larger scale but review epoch selection.** Evidence: `val_loss` and `balanced_val_loss` were best at epoch 8, while `exact_match` peaked later at epoch 19 and final epoch 20 gave the best `balanced_exact_match`. A rerun should either reduce epochs or make checkpoint selection explicitly match the metric that matters most.
3. **Increase validation sample sizing.** Evidence: `urey-2.0` showed materially different conclusions depending on metric (`val_loss` best at epoch 8, `exact_match` best at epoch 19, `balanced_exact_match` best at epoch 20), suggesting that the current sampled validation view is noisy for late-stage comparisons.
4. **If the scratchpad variant is continued, keep scratchpad targeted rather than expanding it immediately.** Evidence: `urey-2.0` already gained large improvements on compositional categories, but its weakest remaining kinds were concentrated in multiplication-heavy `three_input::*::*` and parentheses `**` strata, so the current bottleneck is specific rather than broad.
