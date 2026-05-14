# europa-alm-1

**Europa Arithmetic Language Model 1** -- a small causal transformer trained on synthetic reversed-digit arithmetic.

| Property | Value |
|---|---|
| Checkpoint | `runs/europa-alm-1/checkpoint-best.pt` |
| Parameters | 4,761,856 (~4.76M) |
| Training epochs | 100 |
| Best epoch | 100 (selected by highest validation exact-match) |
| Final val loss | 0.5766 |
| Final val exact-match | 92.58% |
| Strata evaluation accuracy | 45.08% (13,300 examples, 266 kinds) |

---

## 1. Model Architecture

`SmallCausalTransformer` -- a standard decoder-only transformer with pre-norm residuals and tied input/output embeddings.

| Component | Configuration |
|---|---|
| Embedding dimension (`d_model`) | 256 |
| Attention heads (`n_heads`) | 4 |
| Transformer layers (`n_layers`) | 6 |
| MLP hidden dimension | 1,024 |
| MLP activation | GELU |
| Sequence length | 64 tokens |
| Vocabulary size | 25 tokens |
| Dropout | 0.1 (applied to embeddings + MLP output) |
| Normalization | LayerNorm, pre-norm (before attention and MLP) |
| Embedding tie | `lm_head.weight` tied to `token_embedding.weight` |
| Positional encoding | Learned (non-shared) position embeddings |
| Attention mask | Causal (upper-triangular, built per forward pass) |

### Parameter Breakdown

| Component | Parameters |
|---|---|
| Token embedding (25 x 256) | 6,400 |
| Position embedding (64 x 256) | 16,384 |
| Per-layer attention (in_proj + out_proj) | 263,168 |
| Per-layer MLP (256 -> 1024 -> 256) | 525,568 |
| Per-layer LayerNorm x2 | 1,024 |
| Per-layer total | 789,760 |
| 6 layers total | 4,738,560 |
| Final LayerNorm | 512 |
| **Total** | **4,761,856** |

### Vocabulary (25 tokens)

| Token | Purpose |
|---|---|
| `<pad>` | Padding |
| `<bos>` | Beginning-of-sequence |
| `<eos>` | End-of-sequence |
| `<sep>` | Field separator (between tokens in the expression) |
| `<ans>` | Answer boundary marker |
| `undefined` | Division-by-zero sentinel |
| `remainder` | Non-exact division sentinel |
| `+`, `-`, `*`, `/`, `=` | Operator tokens |
| `(`, `)` | Parentheses tokens |
| `0`-`9` | Digit tokens |

---

## 2. Data Format

### Number Encoding

All numbers are **8-digit zero-padded decimals, reversed** (least-significant digit first):

| Value | Encoded |
|---|---|
| 6 | `60000000` |
| 123 | `32100000` |
| -6 | `(-60000000)` |

This encoding forces the model to process arithmetic digit-by-digit from least to most significant, making carry/borrow propagation a local operation in token space.

### Line Format

```
<expression tokens separated by spaces> = <ans> <reversed result>
```

Example:
```
03000000 + 03000000 = <ans> 60000000
```

### Magnitude Bands

| Band | Range | Description |
|---|---|---|
| small | 0-20 | Single-digit and low two-digit operands |
| medium | 21-100 | Two-digit operands |
| large | 101-500 | Three-digit operands |

### Problem Categories

| Category | Form | Strategy | Operations | # Kinds |
|---|---|---|---|---|
| `binary` | `A op B` | Exhaustive | `+`, `-`, `*`, `/` | 24 |
| `three_input` | `A op B op C` (same op) | Sampled (128 train + 16 val + 16 test per kind) | `+`, `-`, `*` | 30 |
| `parentheses` | `(A op B) op C` or `A op (B op C)` | Sampled | `+`, `-`, `*` (inner x outer) | 176 |
| `negative_input` | `(-A) op B` or `A op (-B)` | Sampled | `+`, `-`, `*` | 36 |

**Total kinds: 266** | **Total unique samples: 670,163**

### Dataset Splits

| Split | Count | Binary | Three-input | Parentheses | Negative |
|---|---|---|---|---|---|
| Train | 661,651 | 630,675 | 3,840 | 22,528 | 4,608 |
| Val | 4,256 | 384 | 480 | 2,816 | 576 |
| Test | 4,256 | 384 | 480 | 2,816 | 576 |

Binary kinds are generated exhaustively (all valid operand combinations). Sampled kinds use rejection sampling with up to 200,000 attempts per kind, constrained to non-negative intermediates and answers that fit within 8 digits.

---

## 3. Training Procedure

### Optimizer & Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (L2 norm) |
| Batch size | 128 |
| Sequence length | 64 |
| Seed | 42 |
| Device | CUDA (TF32 matmul enabled) |

### Training Loop

1. **Data loading**: All lines from `train.txt` and `val.txt` are tokenized into a single flat token stream, then chunked into non-overlapping blocks of 64 tokens (`TokenBlockDataset`). Each block produces `(input_ids, target_ids)` where `target_ids` is `input_ids` shifted by one position.

2. **Per-epoch**: The model trains on shuffled mini-batches from the training set. Loss is cross-entropy over the full sequence (all positions contribute).

3. **Per-epoch evaluation**:
   - **Validation loss**: Mean cross-entropy over 50 batches of the validation set.
   - **Exact-match**: 256 samples drawn from `val.txt`; the model generates up to 24 tokens per prompt and the full generated string is compared to the ground truth.

4. **Checkpointing**: `checkpoint-last.pt` is saved every epoch. `checkpoint-best.pt` is saved when the exact-match score improves over the previous best. **No optimizer state is saved** -- training cannot be resumed.

### Training History

The model was trained for **100 epochs**. Key milestones:

| Epoch | Val Loss | Exact-Match | Notes |
|---|---|---|---|
| 1 | 0.7075 | 5.47% | Initial learning |
| 8 | 0.6370 | 59.38% | Rapid improvement phase begins |
| 10 | 0.6296 | 66.80% | |
| 20 | 0.6084 | 73.44% | |
| 40 | 0.5931 | 79.69% | |
| 60 | 0.5849 | 84.77% | |
| 80 | 0.5806 | 89.06% | |
| 100 | **0.5766** | **92.58%** | Best checkpoint selected |

The exact-match score shows a steep learning curve between epochs 7-10 (jumping from ~22% to ~67%), then a steady climb with diminishing returns through epoch 100. The best checkpoint is from the final epoch, indicating the model had not yet saturated.

---

## 4. Evaluation Methodology

### Strata Evaluation (`evaluator/` package)

The evaluation uses the `evaluator` package to perform **stratified kind-level assessment** across all 266 problem kinds.

#### Procedure

1. **Sample selection**: For each of the 266 kinds, exactly **50 examples** are deterministically selected from the union of train/val/test splits using a stable hash-based sampling scheme (BLAKE2b, seed=42). This ensures reproducibility and balanced coverage.

2. **Generation**: For each selected example, the model is given the prompt up to and including `<ans>` and generates up to **24 new tokens** autoregressively (greedy decoding via `generate_completion`).

3. **Scoring**:
   - **Perfect**: The generated string exactly matches the ground-truth answer string.
   - **Canonical prediction**: The generated string is a valid formatted number (8-digit reversed decimal, or negative form) -- i.e., `format_signed_number(parse_signed_number(text)) == text`. This measures whether the model produces well-formed output even when wrong.

4. **Aggregation**: Results are aggregated at three levels:
   - **Overall**: Across all 13,300 examples
   - **Category**: Per problem category (binary, three_input, parentheses, negative_input)
   - **Kind**: Per individual kind (266 rows)

5. **Output artifacts**:
   - `*.summary.json` -- Full results with overall, category, and kind-level stats
   - `*.kinds.csv` -- Per-kind accuracy table
   - `*.errors.jsonl` -- All individual error cases
   - `*.kinds.analysis.json` -- Statistical analysis (ANOVA-style permutation tests)
   - `*.kinds.length-safety.json` -- Token-length and numeric-width safety analysis

#### Evaluation Environment

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (7.62 GB, CUDA 8.9) |
| Throughput | 32.6 examples/second |
| Total time | ~6.8 minutes (408s) |
| Examples evaluated | 13,300 (50 x 266 kinds) |
| Pool size | 670,163 total available examples |

---

## 5. Evaluation Results

### Overall Performance

| Metric | Value |
|---|---|
| **Overall accuracy** | **45.08%** (5,996 / 13,300) |
| Canonical prediction rate | 99.86% |
| Errors | 7,304 |

The near-perfect canonical prediction rate (99.86%) indicates the model almost always produces well-formed reversed-digit numbers, even when the arithmetic is wrong.

### By Category

| Category | Accuracy | Perfect | Missed | Evaluated | Available |
|---|---|---|---|---|---|
| **binary** | **92.50%** | 1,110 | 90 | 1,200 | 631,443 |
| **negative_input** | **72.44%** | 1,304 | 496 | 1,800 | 5,760 |
| **three_input** | **49.87%** | 748 | 752 | 1,500 | 4,800 |
| **parentheses** | **32.20%** | 2,834 | 5,966 | 8,800 | 28,160 |

Binary operations are nearly mastered. Negative inputs show moderate competence. Three-input chains and parenthesized expressions are significant weak points.

### By Operation Family

| Operation | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| `-` (subtraction) | 82.43% | 28 | 1,400 |
| `++` (double addition) | 82.40% | 20 | 1,000 |
| `/` (division) | 72.00% | 6 | 300 |
| `+` (addition) | 69.57% | 28 | 1,400 |
| `-+` (sub-then-add) | 63.70% | 20 | 1,000 |
| `+-` (add-then-sub) | 58.43% | 20 | 1,000 |
| `*` (multiplication) | 35.20% | 20 | 1,000 |
| `--` (double subtraction) | 16.75% | 16 | 800 |
| `*-` (mult-then-sub) | 7.10% | 20 | 1,000 |
| `*+` (mult-then-add) | 5.90% | 20 | 1,000 |
| `**` (double multiplication) | 3.40% | 20 | 1,000 |
| `-*` (sub-then-mult) | 3.10% | 20 | 1,000 |
| `+*` (add-then-mult) | 0.03% | 20 | 1,000 |

**Key insight**: Multiplication involving a second operation (especially as the inner operation) is the model's primary failure mode. Any kind with `*` as the inner operation in parentheses has near-zero accuracy.

### By Magnitude Band (max operand band)

| Max Band | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| small | 60.00% | 31 | 1,550 |
| medium | 49.29% | 82 | 4,100 |
| large | 39.80% | 153 | 7,650 |

Performance degrades monotonically with operand magnitude, consistent with the model needing to propagate carries/borrows across more digit positions.

### Multiplication Effect

| Contains Multiply | Accuracy | Kinds | Evaluated |
|---|---|---|---|
| No | 68.30% | 142 | 7,100 |
| Yes | 18.50% | 124 | 6,200 |

The presence of multiplication is the single strongest predictor of failure (partial R-squared = 0.54 in the sampled-only regression analysis, p < 0.001).

### Best-Performing Kinds (100% accuracy)

All 12 perfect kinds are binary `+` and `-` operations:

| Kind | Band Pattern | Operation |
|---|---|---|
| `binary::small-small::-` | small-small | Subtraction |
| `binary::small-small::+` | small-small | Addition |
| `binary::small-small::*` | small-small | Multiplication |
| `binary::small-medium::-` | small-medium | Subtraction |
| `binary::small-medium::+` | small-medium | Addition |
| `binary::small-large::-` | small-large | Subtraction |
| `binary::small-large::+` | small-large | Addition |
| `binary::small-large::*` | small-large | Multiplication |
| `binary::medium-medium::-` | medium-medium | Subtraction |
| `binary::medium-medium::+` | medium-medium | Addition |
| `binary::medium-large::-` | medium-large | Subtraction |
| `binary::medium-large::+` | medium-large | Addition |

### Worst-Performing Kinds (0% accuracy)

All 12 worst kinds are parenthesized expressions with multiplication as the inner operation:

| Kind | Shape | Band Pattern | Operations |
|---|---|---|---|
| `parentheses::left::large-large-large::**` | left | large-large-large | `*` then `*` |
| `parentheses::left::large-large-large::*+` | left | large-large-large | `*` then `+` |
| `parentheses::left::large-large-large::*-` | left | large-large-large | `*` then `-` |
| `parentheses::left::large-large-large::+*` | left | large-large-large | `+` then `*` |
| `parentheses::left::large-large-large::-*` | left | large-large-large | `-` then `*` |
| `parentheses::left::medium-large-large::**` | left | medium-large-large | `*` then `*` |
| `parentheses::left::medium-large-large::*-` | left | medium-large-large | `*` then `-` |
| `parentheses::left::medium-large-large::+*` | left | medium-large-large | `+` then `*` |
| `parentheses::left::medium-large-large::-*` | left | medium-large-large | `-` then `*` |
| `parentheses::left::medium-medium-large::**` | left | medium-medium-large | `*` then `*` |
| `parentheses::left::medium-medium-large::+*` | left | medium-medium-large | `+` then `*` |
| `parentheses::left::medium-medium-large::-*` | left | medium-medium-large | `-` then `*` |

Notably, all zero-accuracy kinds are **left-associative** parentheses with large operands. Right-associative variants of the same patterns perform slightly better but still poorly.

### Statistical Analysis

Permutation tests (4,000 permutations) on kind-level accuracy reveal:

**Binary category**: Operation type is the dominant predictor (partial R-squared = 0.85, p < 0.001). Division is the weakest binary operation. Operand band has no significant effect after controlling for operation.

**Three-input category**: Operation type dominates (partial R-squared = 0.89, p < 0.001). Multiplication is catastrophically bad (2-5% accuracy), addition is moderate (54-86%), subtraction is intermediate (40-70%). Max band has a modest but significant effect (partial R-squared = 0.30, p = 0.009).

**Parentheses category**: Operation family is the strongest predictor (partial R-squared = 0.80, p < 0.001). Max band also matters (partial R-squared = 0.24, p < 0.001). Wildcard status has a small but significant effect (partial R-squared = 0.15, p < 0.001). Parentheses shape (left vs right) has no significant effect.

**Negative-input category**: Operation type dominates (partial R-squared = 0.72, p < 0.001). Max band and sign side (left vs right) are not significant after controlling for operation.

---

## 6. Length & Safety Analysis

| Metric | Value |
|---|---|
| Max prompt tokens | 40 (limit: 64) |
| Max full line tokens | 49 (limit: 64) |
| Max answer tokens | 11 (limit: 24 generation) |
| Max generation steps | 12 |
| Max context at last answer token | 47 |
| Max context at EOS | 48 |
| Max absolute answer | 99,210,888 |
| Max absolute intermediate | 247,005 |

**No sequence-length violations** were found. All prompts fit within the 64-token context window, and all answers fit within the 24-token generation budget. One kind (`three_input::large-large-large::*`) has answers approaching the 8-digit width limit (99,210,888), but no answers exceed it.

### Accuracy vs. Length Correlations

| Correlation | Value |
|---|---|
| Accuracy vs. max prompt tokens | -0.55 |
| Accuracy vs. max line tokens | -0.56 |
| Accuracy vs. max answer tokens | +0.22 |
| Accuracy vs. max absolute answer | -0.17 |
| Accuracy vs. max absolute intermediate | -0.37 |

Longer prompts correlate with lower accuracy (r = -0.55), primarily because longer prompts indicate more complex expressions (parentheses, three-input). The weak positive correlation with answer length (+0.22) suggests that kinds requiring more answer digits tend to be from categories the model handles better (e.g., negative-input subtraction producing large borrow results).

---

## 7. Limitations & Observations

1. **Multiplication is the bottleneck**: The model handles addition and subtraction well but struggles with multiplication, especially when it appears as an inner operation in parenthesized expressions. This suggests the model has not learned a reliable digit-by-digit multiplication algorithm.

2. **No saturation**: The training curve shows continued improvement through epoch 100 with no plateau. More training epochs or a larger model would likely yield gains.

3. **Division weakness**: Even in the binary category, division is the weakest operation (72% overall, dropping to 50-54% for cross-band divisions). This is expected given the small number of valid division examples (641 large-large, 629 medium-large, etc.).

4. **Canonical output**: The 99.86% canonical prediction rate shows the model has learned the output format robustly -- errors are arithmetic, not structural.

5. **Left vs. right parentheses**: No significant difference in accuracy between left- and right-associative parenthesized expressions (p = 0.37), suggesting the model does not have a strong positional bias for evaluation order.

---

## 8. Reproduction

```bash
# Generate evaluation dataset
uv run generate --output-dir data/europa-alm-1-eval --seed 42

# Run strata evaluation
uv run evaluate \
  --checkpoint runs/europa-alm-1/checkpoint-best.pt \
  --data-dir data/europa-alm-1-eval \
  --sample-size-per-kind 50 \
  --sample-seed 42

# Quick inference
uv run train predict \
  --checkpoint runs/europa-alm-1/checkpoint-best.pt \
  --prompt "03000000 + 03000000 = <ans>"
```
