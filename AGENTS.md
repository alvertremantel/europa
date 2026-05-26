# AGENTS.md — Europa ALM-IS

## Quick setup

```bash
uv sync          # Python 3.12, requires CUDA for PyTorch
```

PyTorch is pulled from the `pytorch-cu128` index (CUDA 12.8). A GPU is expected for training/evaluation.

## Developer commands (via `uv run`)

Use the unified `eis` CLI by default — **do not** run `python generate.py` etc. (those top-level scripts do not exist):

```bash
uv run eis data generate --output-dir data/my-dataset # generate dataset
uv run eis config new                                # create train-config.toml in CWD
uv run eis train run train-config.toml               # train from TOML config
uv run eis train predict --checkpoint runs/my-run/checkpoint-best.pt --prompt "<do> <calc> {300000} + {300000} = <ans>"
uv run eis eval run --checkpoint runs/my-run/checkpoint-best.pt --data-dir data/my-dataset
```

Legacy aliases (`generate`, `config`, `train`, `evaluate`, `its-export`) still exist for compatibility.

Lint: `uv run ruff check .`

Tests: `uv run --group dev python -m pytest`

## Architecture

Canonical training code lives under one `src/eis/` package root:

| Package | Purpose |
|---|---|
| `src/eis/data/` | REDUX arithmetic data generation (arithmetic, negative_input, comparison categories) |
| `src/eis/train/` | Causal transformer training + inference. Checkpoints stay self-contained for downstream native/runtime analysis |
| `src/eis/eval/` | Per-stratum evaluation, writes summary TOML, kinds CSV, and errors TOML next to the checkpoint |

Supporting:
- `src/eis/app/` — FastAPI backend (`src/eis/app/backend/main.py`) + React/Vite frontend (`src/eis/app/frontend/`). Backend hardcodes checkpoint path at `runs/test-extended-plus/checkpoint-best.pt`.
- `tests/` — pytest smoke tests for core canonical generator/trainer/evaluator behavior.
- `info/` — Researcher-facing repository notes and workflow documentation.

## Dataset format

Each line: `<do> <calc> <expression> = <ans> <result>`

Numbers are **6-digit zero-padded decimals, reversed** and explicitly wrapped (e.g. 6 → `{600000}`, -6 → `(600000)`). Comparison answers are `true` / `false`. `<do> <calc>` starts calculation prompts; prompts end at `= <ans>`. Whitespace is only for text readability; there is no internal `<sep>` token.

Output files: `train.txt`, `val.txt`, `test.txt`, `meta.toml`.

## Key constraints

- Resume support is available at epoch boundaries via TOML fields `resume.resume_from` and `resume.additional_epochs`; optimizer and RNG state are checkpointed.
- Checkpoints embed tokenizer + model architecture; incompatible across changes.
- Root aliases `checkpoint-last.pt` and `checkpoint-best.pt` remain compatibility paths, while physical epoch checkpoints live under `checkpoints/` and are all retained.
- The project venv is `.venv/` (gitignored). Use uv for all work. 
- `data/old/`, `runs/old/`, `.agents/*/old` are gitignored scratch directories.

## Training defaults (TrainConfig)

There is no canonical default `d_model`; set it explicitly in TOML. Other TrainConfig defaults remain `n_heads=4, n_layers=6, mlp_hidden=1024, seq_len=64, batch=128, lr=3e-4, epochs=5, dropout=0.1`.
