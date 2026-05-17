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

Four CLI entrypoints are defined — **do not** run `python generate.py` etc. (those scripts don't exist):

```bash
uv run generate --output-dir data/my-dataset          # generate dataset
uv run config --new                                   # create train-config.toml in CWD
uv run train train train-config.toml                  # train from TOML config
uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "03000000 + 03000000 = <ans>"
uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

Lint: `uv run ruff check .`
Tests: `uv run pytest`

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

Training conditions now come from a structured TOML file.

```bash
uv run config --new
uv run config --guide
# fill in train-config.toml first
uv run config --size train-config.toml
uv run train train train-config.toml
```

The generated `train-config.toml` includes every training variable with comments.
Required values begin as empty-string placeholders and must be filled before training.

Key derived size metrics:

- `total_parameters`
- `total_virtual_neurons = n_layers * sequence_length * mlp_hidden`

Experimental curriculum and scratchpad settings remain available through TOML under `[training]` and `[balanced_validation]`.

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

High-level wrapper in `eur_ts/trainer/interpreter.py` for loading checkpoints and running interpretability analyses:

```python
from eur_ts.trainer.interpreter import MechanisticInterpreter

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

Forward-hook system (`eur_ts/trainer/hooks.py`) that captures all intermediate states:

- **Embeddings**: token, positional, combined
- **Per-layer**: inputs, outputs, attention outputs, MLP outputs, norm outputs
- **Final**: hidden state, norm output, logits

### `InterpreterVisualizer`

Matplotlib-based visualizations (`eur_ts/trainer/visualization/`):

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
uv run uvicorn eur_is.backend.main:app --reload

# Frontend (cd into eur_is/frontend/)
cd eur_is/frontend && npm install && npm run dev
```

The backend hardcodes the checkpoint at `runs/test-extended-plus/checkpoint-best.pt`. The `/api/analyze` endpoint returns attention patterns, layer activations, logits, top-k next-token predictions, compact attention/activation summaries, and checkpoint metadata for the dashboard's `circuitsvis` and overview panels. Passing `include_network=true` adds the Network panel payload: bounded MLP firing summaries, attention-head activity metrics, and residual-after-attention summaries from the TransformerLens cache.

---

## Project Structure

```
├── eur_ts/             # Europa Training Suite canonical Python package
│   ├── generator/      # Stratified arithmetic data generation
│   ├── trainer/        # Model, training, inference, interpretability
│   └── evaluator/      # Stratified evaluation & error analysis
├── eur_is/             # Interactive analysis web interface
│   ├── backend/        # FastAPI API server
│   └── frontend/       # React + Vite + circuitsvis SPA
├── tests/              # Pytest smoke tests for canonical core behavior
└── info/               # Researcher-facing documentation
```

---

## Key Constraints

- **Resume is epoch-boundary only** — optimizer and RNG state are saved, but training resumes from `checkpoint_epoch + 1`, not mid-epoch.
- **Checkpoints are self-contained** — they embed the tokenizer and model architecture; incompatible across code changes.
- **GPU expected** — PyTorch is pinned to `pytorch-cu128` (CUDA 12.8). CPU fallback works but is slow.
- **Tests are smoke-focused** — the pytest suite covers core canonical generator, trainer, model, and evaluator behavior.
- **Project venv** is `.venv/` (gitignored). Use `uv run` for all commands.
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories.
