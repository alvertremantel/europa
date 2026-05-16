# After-Training Report: `miller-2.0` and `urey-2.0`

## Executive analysis

This first new-scheme operation produced one useful lower-bound control and one
substantive success:

- **`miller-2.0` is a useful tiny-model lower bound, but not a successful task
  model.** Its `37,056` parameters are inside the requested tiny range, but the
  final evaluator accuracy was only `0.046541`. Even simple qualitative probes
  failed. The run is still valuable because it shows that the curriculum alone
  does not make this extremely small 2-head / 4-layer configuration robust on
  the full stratified dataset.
- **`urey-2.0` is a strong first success for the new example-mode scheme.** At
  `1,598,080` parameters, it reached `0.868346` sampled evaluator accuracy with
  perfect sampled binary add/sub performance, very strong negative-input
  performance, and good parentheses performance.
- **The main remaining weakness is sharply localized rather than broad.**
  `urey-2.0` fails or nearly fails on large multiplicative composition,
  especially `three_input::*` and parentheses `**` kinds involving medium/large
  bands. This localization is the most important result of the run because it
  suggests targeted next experiments and concrete mechanistic-interpretability
  probes.
- **This pair does not isolate scratchpad benefit.** `miller-2.0` and
  `urey-2.0` differ in both capacity and target format. The strongest immediate
  control is therefore a same-architecture `urey-2.0-final-only` run with
  `--training-format final_only` and the same curriculum.

Important metric caveat: for scratchpad-trained checkpoints, the legacy
token-stream `val_loss` is not directly comparable to final-only runs because it
is still computed on final-only validation text. For `urey-2.0`, prioritize
`exact_match`, `balanced_exact_match`, evaluator accuracy, and category/kind
breakdowns over raw `val_loss` when judging task behavior.

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

Analysis note:

- The dataset is strongly binary-heavy in raw train counts (`630675` binary rows
  versus `26368` compositional parentheses/three-input rows and `4608`
  negative-input rows). The curriculum sampler therefore matters: it is not just
  shuffling examples, it is substantially reweighting rare compositional and
  negative categories during training. This makes the observed `urey-2.0`
  category profile a meaningful test of mixed-curriculum resampling.

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

Interpretation:

- The tiny model continued improving in training loss through epoch 20, but the
  evaluator remained near failure. This is consistent with learning local format
  and token-distribution regularities without learning robust arithmetic.
- The gap between low training loss and poor evaluator accuracy argues against
  spending much effort on exact reruns of this configuration. If a sub-100K model
  is still desired, increase width while preserving the required 2-head / 4-layer
  structure.
- A plausible next tiny configuration is `d_model=48`, `mlp_hidden=96`, 2 heads,
  4 layers, sequence length 64, which is approximately `80K` parameters with the
  final-only vocabulary. This stays inside the requested small-model band while
  more than doubling the representational budget.

### `urey-2.0` (`runs/urey-2.0/history.json`)

| Metric | Final | Best |
|---|---:|---:|
| train_loss | `0.347451` (epoch 20) | `0.347451` (epoch 20) |
| val_loss | `4.749334` (epoch 20) | `4.353463` (epoch 8) |
| exact_match | `0.898438` (epoch 20) | `0.921875` (epoch 19) |
| balanced_val_loss | `0.388430` (epoch 20) | `0.379028` (epoch 8) |
| balanced_exact_match | `0.842105` (epoch 20) | `0.842105` (epoch 20) |

Interpretation:

- `urey-2.0` is clearly capacity-sufficient for many strata under the new scheme.
  The high exact-match and evaluator accuracy show that example-mode curriculum
  training can work well at roughly 1.6M parameters.
- The best raw `val_loss` and `balanced_val_loss` occurring around epoch 8 while
  exact-match metrics peak later means checkpoint selection is metric-sensitive.
  For these arithmetic models, exact-match and per-kind evaluator accuracy are
  more behaviorally meaningful than raw next-token loss, especially when
  scratchpad-format training is involved.
- The recorded `scratchpad_fraction` of about `0.7218` means most balanced
  validation examples were transformed into scratchpad-bearing targets. This is
  useful for stress-testing the format, but it also reinforces the need for a
  same-size final-only control before attributing gains to scratchpad
  supervision.

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

Analysis:

- `miller-2.0` should be interpreted as a lower-bound control, not as evidence
  that small models cannot benefit from curriculum. At `37K` parameters it is
  likely too constrained for the full operation/band/category mixture.
- `urey-2.0` shows a structured competence profile: ordinary binary add/sub is
  solved, negative-input behavior is unexpectedly strong, and many parentheses
  cases work, while multiplicative scale/composition remains the bottleneck.
- The weakest `urey-2.0` rows are not random failures. They cluster around large
  chained multiplication: `three_input::*` with large bands and parentheses
  `**` kinds. Future evaluation should preserve this per-kind breakdown because
  overall accuracy hides the remaining hard cases.
- The negative-input result is notable. Negative examples were rare in the raw
  dataset but curriculum replay made them learnable for `urey-2.0`; this is a
  strong sign that reweighting rare strata can work.

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

Analysis caveat:

- These probe logs do **not** prove that `urey-2.0` failed to generate
  scratchpad markers. The public `train predict` command returns the extracted
  final-answer field, so a raw generation such as
  `<work> <step> ... <final> 07000000` would still be printed as just
  `07000000`. To inspect scratchpad behavior, either download the checkpoint and
  run a raw decoding helper, or add a CLI/debug option that disables final-answer
  extraction.
- The qualitative probes are nevertheless consistent with the evaluator summary:
  `urey-2.0` answers simple add/sub and negative-input examples correctly, but
  misses multiplication and parenthesized multiplication probes.

## 7. Operational notes

- `miller-2.0` wall-clock training time: `2131.922 s` (`00:35:31.922`)
- `urey-2.0` wall-clock training time: `4813.273 s` (`01:20:13.273`)
- `resume_source` was `null` for both runs; no resume was recorded.
- No CUDA-memory failure was recorded in run metadata.
- No manual deviation from the prescribed training commands was made during the training runs themselves.

## 8. Next-run recommendations

1. **Train `urey-2.0-final-only` as the highest-priority control.** Use the same
   architecture, seed, dataset, curriculum, epochs, and validation settings as
   `urey-2.0`, but set `--training-format final_only` and `--max-new-tokens 24`.
   This isolates whether the `urey-2.0` gains came from capacity/curriculum alone
   or from compact scratchpad supervision.
2. **Train a stronger tiny model rather than rerunning exact `miller-2.0`.** Keep
   the required 2 heads / 4 layers, but increase to approximately `80K`
   parameters, e.g. `d_model=48`, `mlp_hidden=96`, `training-format final_only`,
   and the same `baseline_mixed_v1` curriculum. Evidence: `miller-2.0` reached
   only `0.0465` overall sampled evaluator accuracy and `0.0526` balanced exact
   match.
3. **Add raw scratchpad decoding before making claims about scratchpad use.** The
   current prediction probes are post-processed final answers. Downloading the
   checkpoint or adding a raw-generation debug path is necessary to determine
   whether the model emits `<work>`, `<step>`, and `<final>` internally.
4. **Focus future curriculum changes on multiplicative composition.** The
   remaining `urey-2.0` failures are concentrated in large `three_input::*` and
   parentheses `**` kinds. A later `mul_focus_v1` run or a refined
   multiplication-composition stage is justified, but should come after the
   same-size final-only control.
5. **Increase balanced validation sample size for future comparison runs.** The
   late-epoch metric disagreement (`val_loss` best at epoch 8, exact-match best
   around epochs 19-20) suggests the current balanced sample is useful but still
   noisy for checkpoint-selection decisions.
6. **Use exact-match and per-kind evaluator accuracy as primary behavioral
   metrics.** Keep raw `val_loss` for continuity, but do not rely on it alone for
   scratchpad-trained models because it is computed on legacy final-only
   validation text.

## 9. Mechanistic-interpretability leads

These are proposed study targets, not mechanistic conclusions.

1. **Multiplication/composition boundary.** Compare successful binary
   multiplication cases against failed large `three_input::*` and parentheses
   `**` cases. The key question is whether intermediate products are represented
   and then lost, or never represented cleanly.
2. **Parentheses `**` failure mode.** `urey-2.0` handles many parentheses cases
   but collapses on large chained multiplication. This is a good candidate for
   layer-by-layer activation comparison across operator pairs such as `+*`, `*+`,
   and `**`.
3. **Negative-input success.** Negative-input performance is much stronger than
   its raw-data frequency would suggest. Compare sign-handling examples with
   ordinary binary add/sub examples to see whether sign handling is localized to
   specific heads or MLP features.
4. **Scratchpad token role.** If raw decoding confirms scratchpad-marker usage,
   inspect whether `<work>`, `<step>`, and `<final>` correspond to separable
   computation/copying phases. If raw decoding shows the model bypasses visible
   scratchpads, that is also useful: it would suggest final-answer computation
   can emerge despite scratchpad-supervised targets.
5. **Checkpoint-transition analysis.** Because `urey-2.0` metrics differ by
   epoch and exact-match peaks late, checkpoints around epochs 8, 19, and 20 are
   valuable comparison points if available. Look for changes in multiplicative
   strata rather than only overall accuracy.
