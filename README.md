# Europa Interpretability Suite

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![TransformerLens](https://img.shields.io/badge/TransformerLens-7B68EE?style=for-the-badge&logo=pytorch&logoColor=white)

**Synthetic arithmetic data generation + small causal transformer training + mechanistic interpretability tooling**

</div>

---

Europa ALM-IS (europa-is) is a research-grade toolkit for training and interpreting small language models on synthetic arithmetic tasks. It generates stratified datasets of reversed-digit math problems, trains a configurable causal transformer, evaluates performance across fine-grained problem strata, and provides deep mechanistic interpretability tools — all backed by a web UI for interactive analysis.

## Quick Start

```bash
uv sync          # Python 3.12, requires CUDA for PyTorch (cu128)
```

Three CLI entrypoints are defined — **do not** run `python generate.py` etc. (those scripts don't exist):

```bash
uv run generate --output-dir data/my-dataset          # generate dataset
uv run train train --data-dir data/my-dataset --output-dir runs/my-run  # train
uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

Lint: `uv run ruff check .`

---

## Workflow

### 1. Generate Data

Generate a stratified arithmetic dataset with four problem categories across magnitude bands (small: 0–20, medium: 21–100, large: 101–500).

```bash
uv run generate --output-dir data/my-dataset --seed 42
```

| Option | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--output-dir` | `data` | Directory to save `train.txt`, `val.txt`, `test.txt`, `meta.json` |
| `--no-validate` | — | Skip post-generation validation pass |

**Problem categories:**

| Category | Description | Strategy | Operations |
|---|---|---|---|
| `binary` | `A op B` with 2 operands | Exhaustive | `+`, `-`, `*`, `/` |
| `three_input` | `A op B op C` (same op) | Sampled | `+`, `-`, `*` |
| `parentheses` | `(A op B) op C` or `A op (B op C)` | Sampled | `+`, `-`, `*` (inner × outer) |
| `negative_input` | `(-A) op B` or `A op (-B)` | Sampled | `+`, `-`, `*` |

Numbers are **8-digit zero-padded decimals, reversed** (e.g. 6 → `60000000`). Negatives: `(-60000000)`. The `<ans>` token marks the prompt/answer boundary.

### 2. Train a Model

Train a small causal transformer on the generated data.

```bash
uv run train train \
  --data-dir data/my-dataset \
  --output-dir runs/my-run \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --checkpoint-max-kept 10 \
  --checkpoint-keep-last 5
```

Resume examples:

```bash
uv run train train --output-dir runs/my-run --resume --additional-epochs 20
uv run train train --resume-from runs/my-run/checkpoint-last.pt --epochs 120
```

**Training options:**

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data-1m` | Path to generated dataset |
| `--output-dir` | `runs/arithmetic-small` | Checkpoint & log output directory |
| `--resume-from` | — | Resume from an explicit checkpoint |
| `--resume` | — | Resume from `<output-dir>/checkpoint-last.pt` |
| `--additional-epochs` | — | Continue for N more epochs beyond the resumed epoch |
| `--sequence-length` | `64` | Max context window |
| `--batch-size` | `128` | Training batch size |
| `--epochs` | `5` | Number of training epochs |
| `--learning-rate` | `3e-4` | AdamW learning rate |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient clipping norm |
| `--log-interval` | `100` | Steps between log prints |
| `--eval-batches` | `50` | Validation batches per eval |
| `--exact-match-samples` | `256` | Samples for exact-match eval |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Device (`cuda`, `cpu`, `auto`) |
| `--checkpoint-keep-last` | `5` | Always retain this many latest physical epoch checkpoints |
| `--checkpoint-max-kept` | `10` | Max retained physical epoch checkpoints (`<=0` keeps all) |
| `--checkpoint-keep-best` | `1` | Extra best-performing checkpoints to retain |
| `--checkpoint-jump-threshold` | `0.05` | Exact-match jump size that tags before/after comparison checkpoints |
| `--training-mode` | `token_stream` | `token_stream` keeps the legacy flat-token baseline; `examples` preserves line boundaries |
| `--training-format` | `final_only` | Example-mode target format: `final_only`, `light_scratchpad`, `parentheses_intermediate`, or `multiply_intermediate` |
| `--curriculum-name` | — | Example-mode mixed curriculum preset: `baseline_mixed_v1` or `mul_focus_v1` |
| `--balanced-val` | — | Also log deterministic balanced validation loss/exact-match from examples |
| `--balanced-val-group-by` | `kind` | Balance validation by `kind`, `category`, or `curriculum_group` |

**Model architecture options:**

| Option | Default | Description |
|---|---|---|
| `--d-model` | `256` | Embedding dimension |
| `--n-heads` | `4` | Number of attention heads |
| `--n-layers` | `6` | Number of transformer blocks |
| `--mlp-hidden` | `1024` | MLP hidden dimension |
| `--dropout` | `0.1` | Dropout rate |

~4.76M parameters at default config.

Experimental curriculum and scratchpad options are opt-in. The default remains the
original final-answer-only token-stream training path. In example mode, short
structured scratchpads use compact fields after `<ans>` such as
`<work> <step> ... <final> ...`; prediction and evaluation still compare the
final answer field.

Tiny matched-run recipe:

```bash
# baseline control
uv run train train --data-dir data/my-dataset --output-dir runs/tiny-baseline --epochs 1 --device auto

# same model/data with a conservative mixed curriculum
uv run train train --data-dir data/my-dataset --output-dir runs/tiny-curriculum \
  --epochs 1 --device auto --training-mode examples --curriculum-name baseline_mixed_v1 \
  --balanced-val --balanced-val-sample-size-per-group 2

# curriculum plus scoped light scratchpads for multiplication/parentheses examples
uv run train train --data-dir data/my-dataset --output-dir runs/tiny-scratchpad \
  --epochs 1 --device auto --training-mode examples --training-format light_scratchpad \
  --curriculum-name baseline_mixed_v1 --balanced-val --balanced-val-sample-size-per-group 2 \
  --max-new-tokens 48
```

Training writes:

- `checkpoint-last.pt` and `checkpoint-best.pt` root aliases for compatibility
- per-epoch snapshots under `checkpoints/epoch-XXXX.pt`
- `checkpoints/manifest.json` with retention and pruning history
- `history.json` with rich epoch metrics
- `run-metadata.json` with config, device, retention, and resume provenance

### 3. Predict / Inference

Query a saved checkpoint with a custom prompt:

```bash
uv run train predict \
  --checkpoint runs/my-run/checkpoint-best.pt \
  --prompt "03000000 + 03000000 = <ans>" \
  --max-new-tokens 24
```

| Option | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint file |
| `--prompt` | *(required)* | Input prompt string |
| `--max-new-tokens` | `24` | Max tokens to generate |
| `--device` | `auto` | Device override |

### 4. Evaluate a Model

Run stratified evaluation across problem kinds with detailed per-category and per-kind reporting.

```bash
uv run evaluate \
  --checkpoint runs/my-run/checkpoint-best.pt \
  --data-dir data/my-dataset
```

**Evaluation options:**

| Option | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint |
| `--data-dir` | *(from checkpoint)* | Dataset directory (optional if embedded) |
| `--splits` | `train val test` | Splits to draw sample pool from |
| `--device` | `auto` | Device override |
| `--max-new-tokens` | *(from checkpoint)* | Generation limit |
| `--sample-size-per-kind` | `50` | Examples sampled per kind |
| `--sample-seed` | `42` | Sampling seed |
| `--output-prefix` | *(auto)* | Output file prefix (defaults to `<checkpoint-stem>-strata-eval`) |
| `--failures-per-kind` | `3` | Max failure examples saved per kind |
| `--progress-interval-kinds` | `0` | Print progress every N kinds (0 = silent) |

**Outputs** (saved next to the checkpoint):

| File | Format | Contents |
|---|---|---|
| `*.summary.json` | JSON | Overall stats, category accuracy, top/bottom 10 kinds, device info |
| `*.kinds.csv` | CSV | Per-kind rows with accuracy, canonical prediction rate, available counts |
| `*.errors.jsonl` | JSONL | Individual error cases with prompt, expected, and prediction |

---

## Mechanistic Interpretability

Europa ALM-IS ships with built-in tools for understanding *how* the model solves arithmetic:

### `MechanisticInterpreter`

High-level wrapper in `trainer/interpreter.py` for loading checkpoints and running interpretability analyses:

```python
from trainer.interpreter import MechanisticInterpreter

with MechanisticInterpreter("runs/my-run/checkpoint-best.pt") as interp:
    logits, capture = interp.forward_with_capture(token_ids)
    interp.visualize_summary()
    interp.visualize_activations()
    interp.visualize_attention()
    interp.visualize_logits()
    interp.visualize_mlp()
    interp.visualize_layer_transition(layer_idx=2)
    interp.visualize_position_influence(pos=5)
    interp.explore_step_by_step(token_ids)  # interactive mode
```

### `HookRegistry` & `ActivationCapture`

Forward-hook system (`trainer/hooks.py`) that captures all intermediate states:

- **Embeddings**: token, positional, combined
- **Per-layer**: inputs, outputs, attention outputs, MLP outputs, norm outputs
- **Final**: hidden state, norm output, logits

### `InterpreterVisualizer`

Matplotlib-based visualizations (`trainer/visualization/` with `trainer/visualizer.py` as a compatibility shim):

- Activation heatmaps (per-layer and overview)
- Attention pattern grids
- Logit trajectory & prediction confidence plots
- MLP contribution heatmaps
- Layer transition (input vs output vs delta)
- Token position influence bar charts
- Interactive exploration mode

### Web UI

A FastAPI + React/Vite application for interactive model analysis:

```bash
# Backend (serves /api/analyze and /api/health)
uv run uvicorn web_app.backend.main:app --reload

# Frontend (cd into web_app/frontend/)
cd web_app/frontend && npm install && npm run dev
```

The backend hardcodes the checkpoint at `runs/test-extended-plus/checkpoint-best.pt`. The `/api/analyze` endpoint returns attention patterns, layer activations, logits, and top predictions for use with `circuitsvis` components.

---

## Project Structure

```
├── generator/          # Stratified arithmetic data generation
│   ├── core.py         # KindSpec, sampling, formatting, validation
│   └── main.py         # CLI entrypoint
├── trainer/            # Model, training, inference, interpretability
│   ├── config.py       # ModelConfig + TrainConfig dataclasses
│   ├── model.py        # SmallCausalTransformer (pre-norm, tied embeddings)
│   ├── core.py         # Compatibility shim for training/checkpoint APIs
│   ├── data.py         # ArithmeticTokenizer, dataset loading
│   ├── inference.py    # generate_completion, evaluate_loss, evaluate_exact_match
│   ├── hooks.py        # HookRegistry + ActivationCapture
│   ├── interpreter.py  # MechanisticInterpreter high-level API
│   ├── training/       # Training loop, checkpointing, resume state
│   ├── visualization/  # Split matplotlib visualization helpers
│   ├── visualizer.py   # Compatibility shim exporting InterpreterVisualizer
│   └── main.py         # CLI entrypoint (train / predict subcommands)
├── evaluator/          # Stratified evaluation & error analysis
│   ├── core.py         # BucketStats, row builders
│   └── main.py         # CLI entrypoint
├── web_app/            # Interactive analysis web interface
│   ├── backend/        # FastAPI API server
│   └── frontend/       # React + Vite + circuitsvis SPA
└── info/               # Researcher-facing documentation
```

---

## Key Constraints

- **Resume is epoch-boundary only** — optimizer and RNG state are saved, but training resumes from `checkpoint_epoch + 1`, not mid-epoch.
- **Checkpoints are self-contained** — they embed the tokenizer and model architecture; incompatible across code changes.
- **GPU expected** — PyTorch is pinned to `pytorch-cu128` (CUDA 12.8). CPU fallback works but is slow.
- **No test suite** — the project has no automated tests.
- **Project venv** is `.venv/` (gitignored). Use `uv run` for all commands.
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories.
