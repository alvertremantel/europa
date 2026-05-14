# europa-atm-1.1

**Europa Arithmetic Transformer Model 1.1** -- a compact causal transformer trained on the legacy `europa-1-basic-large-95` arithmetic corpus.

| Property | Value |
|---|---|
| Checkpoint | `data/models/europa-atm-1.1/checkpoint-best.pt` |
| Parameters | 177,792 (~0.178M) |
| Training epochs | 200 |
| Best epoch | 51 (selected by highest validation exact-match) |
| Best-checkpoint val loss | 0.6096 |
| Best-checkpoint val exact-match | 98.44% |
| Final val loss | 0.5990 |
| Final val exact-match | 97.27% |
| Compatibility strata accuracy | 55.00% (600 examples, 12 kinds) |

---

## 1. Model Architecture

`SmallCausalTransformer` -- a decoder-only transformer with pre-norm residual blocks and tied input/output embeddings.

| Component | Configuration |
|---|---|
| Embedding dimension (`d_model`) | 128 |
| Attention heads (`n_heads`) | 2 |
| Transformer layers (`n_layers`) | 2 |
| MLP hidden dimension | 64 |
| MLP activation | GELU |
| Sequence length | 64 tokens |
| Vocabulary size | 24 tokens |
| Dropout | 0.1 |
| Normalization | LayerNorm, pre-norm |
| Embedding tie | `lm_head.weight` tied to `token_embedding.weight` |
| Positional encoding | Learned position embeddings |
| Attention mask | Causal |

### Parameter Breakdown

| Component | Parameters |
|---|---|
| Token embedding (24 x 128) | 3,072 |
| Position embedding (64 x 128) | 8,192 |
| Per-layer attention | 66,048 |
| Per-layer MLP (128 -> 64 -> 128) | 16,576 |
| Per-layer LayerNorm x2 | 512 |
| Per-layer total | 83,136 |
| 2 layers total | 166,272 |
| Final LayerNorm | 256 |
| **Total** | **177,792** |

### Vocabulary (24 tokens)

| Token | Purpose |
|---|---|
| `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<ans>` | Sequence control |
| `undefined` | Division-by-zero sentinel |
| `remainder` | Quotient/remainder separator |
| `+`, `-`, `*`, `/`, `=` | Operator tokens |
| `(`, `)` | Reserved parentheses tokens |
| `0`-`9` | Digit tokens |

---

## 2. Data Format

### Number Encoding

All numbers are **8-digit zero-padded decimals, reversed**.

| Value | Encoded |
|---|---|
| 6 | `60000000` |
| 123 | `32100000` |

### Line Format

Standard arithmetic answers are emitted as:

```text
<lhs> <op> <rhs> = <ans> <result>
```

Division may also emit sentinel-style outputs:

```text
<lhs> / <rhs> = <ans> undefined
<lhs> / <rhs> = <ans> <quotient> remainder <remainder>
```

### Legacy Family Layout

This dataset predates the current 266-kind generator. It contains only **binary** expressions, organized into three nested operand families:

| Family | `+` / `-` operands | `*` operands | `/` operands |
|---|---|---|---|
| small | 0-9, 0-9 | 0-9, 0-9 | 0-9, 0-9 |
| medium | 0-99, 0-99 | 0-49, 0-49 | 0-999, 0-99 |
| large | 0-999, 0-999 | 0-99, 0-99 | 0-9999, 0-99 |

The `large` family is subsampled to **95%** of its full size, producing the dataset name `europa-1-basic-large-95`.

### Dataset Size

| Split | Count |
|---|---|
| Train | 2,253,466 |
| Val | 124,578 |
| Test | 124,836 |
| **Total** | **2,502,880** |

Raw operation counts from `meta.json`:

| Operation | Count |
|---|---|
| `+` | 960,100 |
| `-` | 480,579 |
| `*` | 12,101 |
| `/` | 1,050,100 |

Because the family bands are nested, many easy examples appear in multiple source families. Under the compatibility stratification used below, the union of train/val/test contains **2,390,856 unique lines**.

---

## 3. Training Procedure

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch size | 128 |
| Sequence length | 64 |
| Seed | 42 |
| Device | CUDA |

Training command (`eis-info/models/ATM-1.1/train.sh`):

```bash
uv run train train \
    --data-dir data/training/europa-1-basic-large-95 \
    --output-dir data/models/europa-atm-1.1-rep \
    --epochs 200 \
    --seed 42 \
    --d-model 128 \
    --n-heads 2 \
    --n-layers 2 \
    --mlp-hidden 64
```

### Training History

The model learned very quickly and then plateaued.

| Epoch | Val Loss | Exact-Match | Notes |
|---|---|---|---|
| 1 | 0.6128 | 2.34% | Initial learning |
| 5 | 0.6032 | 56.64% | Passes 50% exact-match |
| 9 | 0.6089 | 90.63% | Passes 90% exact-match |
| 14 | 0.6085 | 97.66% | Near-saturation |
| 51 | 0.6096 | **98.44%** | Best checkpoint saved |
| 178 | **0.5926** | 97.66% | Lowest validation loss |
| 200 | 0.5990 | 97.27% | Final epoch |

Unlike `ATM-1`, this run does **not** show steady late-epoch gains. Exact-match saturates early, then oscillates in a narrow 96-98% band for most of training.

---

## 4. Evaluation Methodology

### Compatibility Strata Evaluation

The current `evaluator/main.py` cannot be run directly on this legacy dataset because:

1. `meta.json` does not contain `kind_definitions`, and
2. division answers may be `undefined` or multi-field `quotient remainder remainder` strings, which are not handled by `generator.core.validate_line`.

To preserve the **same evaluation paradigm**, I ran a compatibility evaluation with the evaluator's core methodology unchanged:

1. Build a reproducible per-kind sample pool from the union of `train/val/test`.
2. Use deterministic BLAKE2b hash sampling with seed 42.
3. Prompt the model up to `<ans>` and generate greedily for up to 24 tokens.
4. Score exact string match against the ground truth answer.
5. Record a canonical-prediction flag for valid numbers, `undefined`, or `quotient remainder remainder` outputs.

### Compatibility Stratification

Since this corpus has only binary problems, I evaluated **12 intrinsic kinds**:

```text
binary_basic::{small|medium|large}::{+|-|*|/}
```

Each line is assigned to the **smallest** family whose operand constraints can generate it, duplicates are removed, and **50 examples per kind** are evaluated.

### Evaluation Artifacts

Saved next to the checkpoint:

- `data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.summary.json`
- `data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.kinds.csv`
- `data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.errors.jsonl`
- `data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.kinds.analysis.json`
- `data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.length-safety.json`

### Evaluation Environment

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (7.62 GB, CUDA 8.9) |
| Throughput | 67.7 examples/second |
| Total time | ~8.9 seconds |
| Examples evaluated | 600 (50 x 12 kinds) |
| Unique pool size | 2,390,856 |

---

## 5. Evaluation Results

### Overall Performance

| Metric | Value |
|---|---|
| **Overall accuracy** | **55.00%** (330 / 600) |
| Canonical prediction rate | 100.00% |
| Errors | 270 |

The model always emits a structurally valid answer string. Its failures are arithmetic/algorithmic, not formatting failures.

### By Operation

| Operation | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| `+` | 100.00% | 3 | 150 |
| `-` | 100.00% | 3 | 150 |
| `/` | 16.67% | 3 | 150 |
| `*` | 3.33% | 3 | 150 |

### By Intrinsic Band

| Band | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| small | 62.50% | 4 | 200 |
| medium | 51.50% | 4 | 200 |
| large | 51.00% | 4 | 200 |

### Multiplication Effect

| Contains Multiply | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| No | 72.22% | 9 | 450 |
| Yes | 3.33% | 3 | 150 |

Multiplication is the dominant failure mode. Division is also very weak, especially once quotient/remainder behavior is required.

### Best Kinds

All six addition and subtraction kinds scored **100%**:

- `binary_basic::small::+`
- `binary_basic::small::-`
- `binary_basic::medium::+`
- `binary_basic::medium::-`
- `binary_basic::large::+`
- `binary_basic::large::-`

### Worst Kinds

| Kind | Accuracy |
|---|---|
| `binary_basic::large::*` | 0.00% |
| `binary_basic::medium::*` | 2.00% |
| `binary_basic::large::/` | 4.00% |
| `binary_basic::medium::/` | 4.00% |
| `binary_basic::small::*` | 8.00% |

### Representative Error Patterns

1. **Multiplication collapse to small memorized outputs**
   - `27000000 * 19000000` -> predicted `20000000`, expected `25560000`
2. **Division quotient/remainder failures**
   - `42770000 / 71000000` -> predicted `09500000 remainder 40000000`, expected `45400000 remainder 60000000`
3. **Exact division vs remainder confusion**
   - `60000000 / 20000000` -> predicted `00000000 remainder 60000000`, expected `30000000`

---

## 6. Length & Safety Analysis

| Metric | Value |
|---|---|
| Max prompt tokens | 25 |
| Max full line tokens | 45 |
| Max answer tokens | 19 |
| Max generation steps | 20 |
| Max context at last answer token | 43 |
| Max context at EOS | 44 |

No context-window or generation-budget violations were found:

- Prompt overflow count: 0
- Full-line overflow count: 0
- Answer-context overflow count: 0
- EOS-context overflow count: 0
- Generation-step overflow count: 0

Answer-type prevalence across the full dataset:

| Answer type | Count |
|---|---|
| Plain number | 1,507,278 |
| `undefined` | 10,507 |
| `quotient remainder remainder` | 985,095 |

The long-tail sequence burden comes almost entirely from division outputs with explicit remainder fields, but even those remain well inside the 64-token training limit and 24-token generation budget.

---

## 7. Observations

1. **This model has effectively solved addition and subtraction.** All six intrinsic `+` and `-` kinds reached 100% compatibility-strata accuracy.
2. **Multiplication is almost absent.** 5/150 multiplication samples were correct overall, with `large::*` at 0%.
3. **Division is better than multiplication, but still poor.** The model especially struggles with exact quotient/remainder structure selection.
4. **Output formatting is fully learned.** Canonical prediction rate is 100%, even on wrong answers.
5. **Training saturated early.** The best checkpoint appears at epoch 51, while the last 149 epochs mostly trade tiny validation-loss fluctuations against nearly flat exact-match.

---

## 8. Reproduction

Artifacts already produced for this checkpoint:

```text
data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.summary.json
data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.kinds.csv
data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.errors.jsonl
data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.kinds.analysis.json
data/models/europa-atm-1.1/checkpoint-best-compat-strata-eval.length-safety.json
```

Quick inference example:

```bash
uv run train predict \
  --checkpoint data/models/europa-atm-1.1/checkpoint-best.pt \
  --prompt "03000000 + 03000000 = <ans>"
```
