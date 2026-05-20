# AGENTS.md — Europa ALM-IS

## Quick setup

```bash
uv sync          # Python 3.12, requires CUDA for PyTorch
```

PyTorch is pulled from the `pytorch-cu128` index (CUDA 12.8). A GPU is expected for training/evaluation.

## Developer commands (via `uv run`)

Four CLI entrypoints are defined in `pyproject.toml` — **do not** run `python generate.py` etc. (those top-level scripts do not exist):

```bash
uv run generate --output-dir data/my-dataset          # generate dataset
uv run config --new                                   # create train-config.toml in CWD
uv run train train train-config.toml                  # train from TOML config
uv run train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> 03000000 + 03000000 ="
uv run evaluate --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

Lint: `uv run ruff check .`

Tests: `uv run pytest`

## Architecture

Canonical training code lives under one `eur_ts/` package root:

| Package | Purpose |
|---|---|
| `eur_ts/generator/` | Stratified arithmetic data generation (binary, three_input, parentheses, negative_input categories) |
| `eur_ts/trainer/` | Causal transformer training + inference. Checkpoints stay self-contained for downstream native/runtime analysis |
| `eur_ts/evaluator/` | Per-stratum evaluation, writes summary TOML, kinds CSV, and errors TOML next to the checkpoint |

Supporting:
- `eur_is/` — FastAPI backend (`eur_is/backend/main.py`) + React/Vite frontend (`eur_is/frontend/`). Backend hardcodes checkpoint path at `runs/test-extended-plus/checkpoint-best.pt`.
- `tests/` — pytest smoke tests for core canonical generator/trainer/evaluator behavior.
- `info/` — Researcher-facing repository notes and workflow documentation.

## Dataset format

Each line: `<do> <calc> <expression> = <result>`

Numbers are **8-digit zero-padded decimals, reversed** (e.g. 6 → `60000000`). Negatives: `(-60000000)`. `<do> <calc>` starts calculation prompts; `=` remains followed by a separator in tokenized prompts.

Output files: `train.txt`, `val.txt`, `test.txt`, `meta.toml`.

## Key constraints

- Resume support is available at epoch boundaries via TOML fields `resume.resume_from` and `resume.additional_epochs`; optimizer and RNG state are checkpointed.
- Checkpoints embed tokenizer + model architecture; incompatible across changes.
- Root aliases `checkpoint-last.pt` and `checkpoint-best.pt` remain compatibility paths, while physical epoch checkpoints live under `checkpoints/` and are all retained.
- The project venv is `.venv/` (gitignored). Use uv for all work. 
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories.

## Training defaults (TrainConfig)

`d_model=256, n_heads=4, n_layers=6, mlp_hidden=1024, seq_len=64, batch=128, lr=3e-4, epochs=5, dropout=0.1` — ~4.76M params.
